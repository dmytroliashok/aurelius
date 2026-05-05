"""Validator preflight doctor (T-9).

Runs the same checks the validator itself performs at startup, but in
isolation, and prints a pass/warn/fail table. Intended for operators
to run before `aurelius-validator` so they see the configuration
problem as a labelled check — not as a cryptic traceback 30 seconds
into boot.

Invoke via `aurelius-validator doctor` (see `validator.main`).

Each check is a pure-ish function: its dependencies are passed in
(config values, env vars, paths), not pulled from globals, so tests
can call it directly with canned inputs. `run_all()` wires the
production checks up to `Config` and returns `(results, exit_code)`.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Status names. "warn" is advisory (don't block startup); "fail" is
# blocking (the validator would also refuse to proceed). "skip" means
# the check isn't applicable to the current configuration.
PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    hint: str | None = None

    @property
    def is_failing(self) -> bool:
        return self.status == FAIL


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_environment(environment: str, network: str, netuid: int) -> CheckResult:
    """Basic env sanity: valid triple, not something obviously wrong."""
    known = {"local", "testnet", "mainnet"}
    if environment not in known:
        return CheckResult(
            "environment",
            WARN,
            f"ENVIRONMENT={environment!r} not in {sorted(known)}",
            "Set ENVIRONMENT to testnet or mainnet (or local for dev).",
        )
    if environment == "mainnet" and network != "finney":
        return CheckResult(
            "environment",
            FAIL,
            f"mainnet requires BT_SUBTENSOR_NETWORK=finney, got {network!r}",
            "Set BT_SUBTENSOR_NETWORK=finney (mainnet is SN37 on finney).",
        )
    if environment == "testnet" and network != "test":
        return CheckResult(
            "environment",
            FAIL,
            f"testnet requires BT_SUBTENSOR_NETWORK=test, got {network!r}",
            "Set BT_SUBTENSOR_NETWORK=test (testnet is SN455 on test).",
        )
    return CheckResult("environment", PASS, f"env={environment} network={network} netuid={netuid}")


def check_testlab_safety(testlab_mode: bool, network: str) -> CheckResult:
    """Mirrors Validator._check_testlab_safety."""
    if testlab_mode and network == "finney":
        return CheckResult(
            "testlab_safety",
            FAIL,
            "TESTLAB_MODE=1 on mainnet (finney) is never safe",
            "Unset TESTLAB_MODE, or switch to testnet.",
        )
    return CheckResult(
        "testlab_safety",
        PASS,
        f"TESTLAB_MODE={'on' if testlab_mode else 'off'} network={network}",
    )


def check_wallet_files(wallet_name: str, wallet_hotkey: str, wallet_root: str) -> CheckResult:
    """Coldkey + hotkey files exist and are readable."""
    if not wallet_name or not wallet_hotkey:
        return CheckResult(
            "wallet_files",
            FAIL,
            "WALLET_NAME or WALLET_HOTKEY is empty",
            "Set WALLET_NAME and WALLET_HOTKEY in validator.env.",
        )
    root = Path(wallet_root).expanduser()
    wallet_dir = root / wallet_name
    coldkey = wallet_dir / "coldkeypub.txt"
    hotkey = wallet_dir / "hotkeys" / wallet_hotkey
    missing = [p for p in (coldkey, hotkey) if not p.exists()]
    if missing:
        return CheckResult(
            "wallet_files",
            FAIL,
            f"missing file(s): {', '.join(str(p) for p in missing)}",
            f"Create the wallet with `btcli wallet create --wallet-name {wallet_name} "
            f"--wallet-path {root} --hotkey {wallet_hotkey}`, or set WALLET_PATH if "
            f"your wallets live elsewhere.",
        )
    return CheckResult(
        "wallet_files",
        PASS,
        f"coldkey + hotkey readable under {wallet_dir}",
    )


def check_llm_api_key(api_key: str, base_url: str) -> CheckResult:
    """Either we have a key or the base URL is a local endpoint."""
    if api_key:
        return CheckResult("llm_api_key", PASS, "LLM_API_KEY is set")
    # Local endpoints don't need a key (dev / LM-Studio).
    if not base_url:
        return CheckResult(
            "llm_api_key",
            FAIL,
            "LLM_API_KEY is empty and LLM_BASE_URL is unset",
            "Set LLM_API_KEY to your DeepSeek key, or set LLM_BASE_URL to a local LLM.",
        )
    try:
        from urllib.parse import urlparse

        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        host = ""
    local_hosts = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
    if host in local_hosts or host.startswith("10.") or host.startswith("192.168."):
        return CheckResult(
            "llm_api_key",
            PASS,
            f"no key needed for local LLM at {base_url}",
        )
    return CheckResult(
        "llm_api_key",
        FAIL,
        f"LLM_API_KEY is empty but LLM_BASE_URL={base_url} points to an external API",
        "Set LLM_API_KEY or change LLM_BASE_URL to a local endpoint.",
    )


def check_iptables(sim_allowed_hosts: list[str]) -> CheckResult:
    """Needed only when the operator configured an egress allowlist."""
    if not sim_allowed_hosts:
        return CheckResult(
            "iptables",
            SKIP,
            "SIM_ALLOWED_LLM_HOSTS is empty — no network isolation required",
        )
    if shutil.which("iptables") is None:
        return CheckResult(
            "iptables",
            FAIL,
            "iptables binary not on PATH but SIM_ALLOWED_LLM_HOSTS is set",
            "Install iptables in the validator container (apt install iptables) and "
            "grant NET_ADMIN, or set SIM_ALLOWED_LLM_HOSTS='' to explicitly accept "
            "unrestricted sim egress.",
        )
    return CheckResult("iptables", PASS, "iptables available; egress isolation can be applied")


def check_docker_daemon(docker_from_env=None) -> CheckResult:
    """Fail-fast version of what the sim runner does on its first call.

    `docker_from_env` can be injected for testing. In production we fall
    back to the real `docker.from_env`.
    """
    if docker_from_env is None:
        try:
            import docker

            docker_from_env = docker.from_env
        except ImportError:
            return CheckResult(
                "docker_daemon",
                SKIP,
                "`docker` Python package not installed — simulations will be disabled",
                "pip install -e '.[simulation]' if you want Concordia sims.",
            )
    try:
        client = docker_from_env()
        client.ping()
    except Exception as e:
        return CheckResult(
            "docker_daemon",
            FAIL,
            f"Docker daemon not reachable: {e}",
            "Ensure Docker is running. For containerised validators, make sure "
            "DOCKER_HOST points at the docker-socket-proxy.",
        )
    return CheckResult("docker_daemon", PASS, "Docker daemon responded to ping")


def check_data_dir_writable(data_dir: str) -> CheckResult:
    """Validator persists state here — a read-only mount makes every
    restart lose rate-limiter + validation-count progress."""
    path = Path(data_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".aurelius_doctor_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        return CheckResult(
            "data_dir_writable",
            FAIL,
            f"cannot write to DATA_DIR={data_dir}: {e}",
            "Make the data volume writable by the container user (uid 1000 by default).",
        )
    return CheckResult("data_dir_writable", PASS, f"DATA_DIR={data_dir} is writable")


def check_central_api_reachable(url: str, httpx_get=None, timeout: float = 5.0) -> CheckResult:
    """Central API /health returns 200."""
    if not url:
        return CheckResult(
            "central_api",
            FAIL,
            "CENTRAL_API_URL is empty",
            "Set CENTRAL_API_URL in validator.env.",
        )
    if httpx_get is None:
        import httpx

        def httpx_get(u):
            return httpx.get(u, timeout=timeout)

    try:
        resp = httpx_get(url.rstrip("/") + "/health")
    except Exception as e:
        return CheckResult(
            "central_api",
            WARN,
            f"could not reach {url}/health: {e}",
            "Verify CENTRAL_API_URL is correct and reachable from this host.",
        )
    if getattr(resp, "status_code", 0) != 200:
        return CheckResult(
            "central_api",
            WARN,
            f"{url}/health returned HTTP {resp.status_code}",
            "Central API may be in degraded mode; the validator can still start but "
            "will fail-closed on every submission until it recovers.",
        )
    return CheckResult("central_api", PASS, f"{url}/health 200 OK")


def check_dns(hosts: list[str]) -> CheckResult:
    """Every configured LLM/Central host resolves."""
    hosts = [h for h in hosts if h]
    if not hosts:
        return CheckResult("dns", SKIP, "no hosts configured to check")
    unresolved = []
    for host in hosts:
        try:
            socket.getaddrinfo(host, None)
        except socket.gaierror:
            unresolved.append(host)
    if unresolved:
        return CheckResult(
            "dns",
            WARN,
            f"DNS lookup failed for: {', '.join(unresolved)}",
            "Fix DNS or remove unreachable hostnames.",
        )
    return CheckResult("dns", PASS, f"{len(hosts)} host(s) resolve")


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_STATUS_GLYPH = {PASS: "[OK]   ", WARN: "[WARN] ", FAIL: "[FAIL] ", SKIP: "[SKIP] "}


def render_report(results: list[CheckResult]) -> str:
    """Pass/warn/fail table for humans."""
    name_w = max((len(r.name) for r in results), default=10)
    lines = ["Aurelius validator preflight:"]
    for r in results:
        lines.append(f"  {_STATUS_GLYPH[r.status]}{r.name.ljust(name_w)}  {r.message}")
        if r.hint and r.status in (WARN, FAIL):
            lines.append(f"  {' ' * len(_STATUS_GLYPH[r.status])}{' ' * name_w}  → {r.hint}")
    fail_count = sum(1 for r in results if r.status == FAIL)
    warn_count = sum(1 for r in results if r.status == WARN)
    if fail_count:
        lines.append(f"\n  {fail_count} failing check(s) — validator will not start.")
    elif warn_count:
        lines.append(f"\n  {warn_count} warning(s) — validator will start but review above.")
    else:
        lines.append("\n  All checks passed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all() -> tuple[list[CheckResult], int]:
    """Wire the production checks up to ``Config`` and run them all.

    Returns (results, exit_code). exit_code is 0 iff no FAIL result.
    """
    from aurelius.config import ENVIRONMENT, Config

    checks: list[CheckResult] = []

    def _safe(fn, *args, **kwargs):
        """Run a check, capturing unexpected exceptions as FAIL results."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return CheckResult(
                getattr(fn, "__name__", "unknown").replace("check_", ""),
                FAIL,
                f"check raised {type(e).__name__}: {e}",
                "Report this to the Aurelius team — a check should never crash.",
            )

    checks.append(_safe(check_environment, ENVIRONMENT, Config.NETWORK, Config.NETUID))
    checks.append(_safe(check_testlab_safety, Config.TESTLAB_MODE, Config.NETWORK))
    checks.append(
        _safe(
            check_wallet_files,
            Config.WALLET_NAME,
            Config.WALLET_HOTKEY,
            os.environ.get("WALLET_PATH") or "~/.bittensor/wallets",
        )
    )
    checks.append(_safe(check_llm_api_key, Config.LLM_API_KEY, Config.LLM_BASE_URL))
    checks.append(_safe(check_iptables, list(getattr(Config, "SIM_ALLOWED_LLM_HOSTS", []) or [])))
    checks.append(_safe(check_docker_daemon))
    from aurelius.config import _DATA_DIR

    checks.append(_safe(check_data_dir_writable, _DATA_DIR))
    checks.append(_safe(check_central_api_reachable, Config.CENTRAL_API_URL))
    hosts = [Config.LLM_BASE_URL, Config.CENTRAL_API_URL]
    hostnames = []
    for u in hosts:
        try:
            from urllib.parse import urlparse

            h = urlparse(u).hostname if u else ""
            if h:
                hostnames.append(h)
        except Exception:
            pass
    checks.append(_safe(check_dns, hostnames))

    exit_code = 1 if any(r.is_failing for r in checks) else 0
    return checks, exit_code


def main() -> int:
    """CLI entry point. Use `aurelius-validator doctor`."""
    results, exit_code = run_all()
    print(render_report(results))
    return exit_code
