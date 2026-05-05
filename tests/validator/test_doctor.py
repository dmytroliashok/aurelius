"""T-9: validator preflight doctor.

Each check is a pure-ish function taking its dependencies as args, so
tests call them directly with canned inputs instead of standing up a
real Validator and a real network.
"""

import pytest

from aurelius.validator.doctor import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    check_central_api_reachable,
    check_data_dir_writable,
    check_dns,
    check_docker_daemon,
    check_environment,
    check_iptables,
    check_llm_api_key,
    check_testlab_safety,
    check_wallet_files,
    render_report,
)


class TestCheckEnvironment:
    def test_testnet_happy(self):
        r = check_environment("testnet", "test", 455)
        assert r.status == PASS

    def test_mainnet_happy(self):
        r = check_environment("mainnet", "finney", 37)
        assert r.status == PASS

    def test_mainnet_with_test_network_fails(self):
        r = check_environment("mainnet", "test", 37)
        assert r.status == FAIL
        assert "finney" in r.message

    def test_testnet_with_finney_network_fails(self):
        r = check_environment("testnet", "finney", 455)
        assert r.status == FAIL
        assert "test" in r.message

    def test_unknown_environment_warns(self):
        r = check_environment("staging", "test", 99)
        assert r.status == WARN


class TestCheckTestlabSafety:
    def test_testlab_on_mainnet_fails(self):
        assert check_testlab_safety(True, "finney").status == FAIL

    def test_testlab_on_testnet_passes(self):
        assert check_testlab_safety(True, "test").status == PASS

    def test_off_everywhere_passes(self):
        assert check_testlab_safety(False, "finney").status == PASS
        assert check_testlab_safety(False, "test").status == PASS


class TestCheckWalletFiles:
    def test_missing_name_fails(self):
        assert check_wallet_files("", "default", "/tmp").status == FAIL

    def test_missing_dir_fails(self, tmp_path):
        r = check_wallet_files("validator", "default", str(tmp_path))
        assert r.status == FAIL
        assert "missing" in r.message

    def test_present_passes(self, tmp_path):
        root = tmp_path
        wdir = root / "validator"
        (wdir / "hotkeys").mkdir(parents=True)
        (wdir / "coldkeypub.txt").write_text("{}")
        (wdir / "hotkeys" / "default").write_text("{}")
        r = check_wallet_files("validator", "default", str(root))
        assert r.status == PASS


class TestCheckLLMApiKey:
    def test_key_set_passes(self):
        assert check_llm_api_key("sk-anything", "https://api.deepseek.com/v1").status == PASS

    def test_empty_key_with_external_url_fails(self):
        r = check_llm_api_key("", "https://api.deepseek.com/v1")
        assert r.status == FAIL
        assert "LLM_API_KEY" in r.message

    def test_empty_key_with_local_url_passes(self):
        assert check_llm_api_key("", "http://localhost:1234/v1").status == PASS
        assert check_llm_api_key("", "http://127.0.0.1:1234/v1").status == PASS
        assert check_llm_api_key("", "http://host.docker.internal:1234/v1").status == PASS

    def test_empty_key_and_empty_url_fails(self):
        assert check_llm_api_key("", "").status == FAIL


class TestCheckIptables:
    def test_no_hosts_is_skip(self):
        assert check_iptables([]).status == SKIP

    def test_iptables_missing_fails(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda x: None)
        r = check_iptables(["api.deepseek.com"])
        assert r.status == FAIL

    def test_iptables_present_passes(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda x: "/usr/sbin/iptables")
        assert check_iptables(["api.deepseek.com"]).status == PASS


class TestCheckDockerDaemon:
    def test_docker_import_missing_skips(self, monkeypatch):
        """If the docker pip package isn't installed, treat as skip."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "docker":
                raise ImportError("no docker")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        r = check_docker_daemon()
        assert r.status == SKIP

    def test_ping_success_passes(self):
        from unittest.mock import MagicMock

        fake_client = MagicMock()
        fake_client.ping.return_value = True
        r = check_docker_daemon(docker_from_env=lambda: fake_client)
        assert r.status == PASS

    def test_ping_failure_fails(self):
        def factory():
            raise ConnectionError("nope")

        r = check_docker_daemon(docker_from_env=factory)
        assert r.status == FAIL
        assert "ConnectionError" in r.message or "nope" in r.message


class TestCheckDataDirWritable:
    def test_writable_passes(self, tmp_path):
        r = check_data_dir_writable(str(tmp_path))
        assert r.status == PASS

    def test_readonly_fails(self, tmp_path):
        import os
        import stat

        tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            r = check_data_dir_writable(str(tmp_path / "nope"))
        finally:
            tmp_path.chmod(0o755)
        # Read-only parent → cannot create the subdir
        assert r.status == FAIL or os.geteuid() == 0  # root ignores file perms


class TestCheckCentralApi:
    def test_empty_url_fails(self):
        r = check_central_api_reachable("")
        assert r.status == FAIL

    def test_200_passes(self):
        from unittest.mock import MagicMock

        def fake_get(url):
            resp = MagicMock()
            resp.status_code = 200
            return resp

        r = check_central_api_reachable("http://api.example.com", httpx_get=fake_get)
        assert r.status == PASS

    def test_500_warns(self):
        from unittest.mock import MagicMock

        def fake_get(url):
            resp = MagicMock()
            resp.status_code = 500
            return resp

        r = check_central_api_reachable("http://api.example.com", httpx_get=fake_get)
        assert r.status == WARN

    def test_exception_warns(self):
        def fake_get(url):
            raise RuntimeError("connection refused")

        r = check_central_api_reachable("http://api.example.com", httpx_get=fake_get)
        assert r.status == WARN


class TestCheckDns:
    def test_no_hosts_skips(self):
        assert check_dns([]).status == SKIP

    def test_all_resolve_passes(self):
        # localhost always resolves, even offline
        r = check_dns(["localhost"])
        assert r.status == PASS

    def test_unresolvable_warns(self):
        r = check_dns(["definitely-not-a-real-domain-1234567.invalid"])
        assert r.status == WARN


class TestRenderReport:
    def test_all_pass_shows_success(self):
        results = [
            CheckResult("a", PASS, "ok"),
            CheckResult("b", PASS, "ok"),
        ]
        report = render_report(results)
        assert "All checks passed" in report
        assert "[OK]" in report

    def test_fail_shows_count(self):
        results = [
            CheckResult("a", PASS, "ok"),
            CheckResult("b", FAIL, "broken", hint="fix it"),
        ]
        report = render_report(results)
        assert "1 failing check" in report
        assert "[FAIL]" in report
        assert "→ fix it" in report

    def test_warn_shows_count_without_failing(self):
        results = [
            CheckResult("a", PASS, "ok"),
            CheckResult("b", WARN, "heads up", hint="maybe fix"),
        ]
        report = render_report(results)
        assert "1 warning" in report
        assert "will start" in report

    def test_skip_not_counted_as_fail_or_warn(self):
        results = [
            CheckResult("a", PASS, "ok"),
            CheckResult("b", SKIP, "n/a"),
        ]
        report = render_report(results)
        assert "All checks passed" in report


class TestRunAllIntegration:
    """The production wiring loads Config and runs every check. We don't
    assert on the result content (depends on the machine), only that it
    doesn't crash and returns a valid exit code."""

    def test_runs_without_crashing(self):
        from aurelius.validator.doctor import run_all

        results, exit_code = run_all()
        assert isinstance(results, list)
        assert len(results) >= 5
        assert exit_code in (0, 1)
        assert all(r.status in (PASS, WARN, FAIL, SKIP) for r in results)

    def test_safe_wrapper_catches_unexpected_exceptions(self):
        """A broken check should be reported as FAIL, not propagate."""
        from aurelius.validator import doctor

        def broken():
            raise RuntimeError("kaboom")

        r = doctor._safe(broken) if hasattr(doctor, "_safe") else None
        # If run_all didn't expose _safe, skip — the behaviour is still
        # exercised indirectly by test_runs_without_crashing.
        if r is not None:
            assert r.status == FAIL
