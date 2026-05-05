"""Tests for DockerSimulationRunner.

Note: These tests don't actually run Docker containers. They test the
runner's logic (translation, limits computation, result handling) using
mocks. Real Docker integration tests require @pytest.mark.slow.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aurelius.config import Config
from aurelius.simulation.docker_runner import DEFAULT_IMAGE_TAG, DockerSimulationRunner, SimulationResult


def _valid_config() -> dict:
    return {
        "name": "hospital_dilemma_one",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A doctor faces a difficult decision in a hospital.",
        "agents": [
            {
                "name": "Dr. Chen",
                "identity": "I am a surgeon with experience.",
                "goal": "I want to save lives.",
                "philosophy": "deontology",
            },
            {
                "name": "Nurse Patel",
                "identity": "I am a senior nurse.",
                "goal": "I want patient safety.",
                "philosophy": "care_ethics",
            },
        ],
        "scenes": [
            {"steps": 3, "mode": "decision"},
            {"steps": 2, "mode": "reflection"},
        ],
    }


class TestDockerSimulationRunner:
    def test_compute_limits_two_agents(self):
        runner = DockerSimulationRunner()
        limits = runner._compute_limits(2)
        assert limits["timeout"] == 600  # base * (2/2)
        assert limits["mem_limit"] == 4096 * 1024 * 1024  # 4GB base
        assert limits["nano_cpus"] == 2_000_000_000

    def test_compute_limits_three_agents(self):
        runner = DockerSimulationRunner()
        limits = runner._compute_limits(3)
        assert limits["timeout"] == 900  # 600 * (3/2)
        assert limits["mem_limit"] == int(4096 * 1.25 * 1024 * 1024)  # 5120 MB

    def test_compute_limits_four_agents(self):
        runner = DockerSimulationRunner()
        limits = runner._compute_limits(4)
        assert limits["timeout"] == 1200  # 600 * (4/2)
        assert limits["mem_limit"] == int(4096 * 1.5 * 1024 * 1024)  # 6144 MB

    def test_invalid_config_returns_error(self):
        runner = DockerSimulationRunner()
        # Config with no agents — translator should still work but produce empty agents
        result = runner.run_simulation({"not_a_valid": "config"})
        # Will fail at Docker level, not translation
        assert isinstance(result, SimulationResult)

    def test_update_image(self):
        runner = DockerSimulationRunner(image_tag="v1.0.0")
        assert runner.image_tag == "v1.0.0"
        # update with same tag is a no-op
        runner.update_image("v1.0.0")
        assert runner.image_tag == "v1.0.0"

    def test_default_configuration(self):
        """Defaults fall through from Config when no remote_config is injected."""
        runner = DockerSimulationRunner()
        assert runner.image_name == Config.CONCORDIA_IMAGE_NAME
        assert runner.image_tag == DEFAULT_IMAGE_TAG
        assert runner.llm_model == Config.LLM_MODEL
        assert runner.base_timeout == Config.SIM_BASE_TIMEOUT
        assert runner.base_ram_mb == Config.SIM_BASE_RAM_MB
        assert runner.cpu_count == Config.SIM_CPU_COUNT


class TestDockerHealthChecks:
    """CS-11: Explicit Docker daemon startup + periodic health check.

    The runner's constructor eagerly talks to Docker (builds images,
    warms container pools). These tests instantiate a runner and then
    clear the cached client so the behaviour under test runs against a
    freshly-mocked daemon.
    """

    def test_preflight_raises_when_daemon_unreachable(self):
        runner = DockerSimulationRunner()
        runner._docker_client = None
        fake_client = MagicMock()
        fake_client.ping.side_effect = ConnectionError("daemon down")
        with patch("docker.from_env", return_value=fake_client):
            with pytest.raises(RuntimeError, match="Docker daemon is not reachable"):
                runner.preflight_check()
        assert runner._docker_client is None  # cached client cleared on failure

    def test_preflight_passes_when_daemon_up(self):
        runner = DockerSimulationRunner()
        runner._docker_client = None
        fake_client = MagicMock()
        fake_client.ping.return_value = True
        with patch("docker.from_env", return_value=fake_client):
            runner.preflight_check()
        assert runner._docker_client is fake_client

    def test_health_check_returns_false_without_raising(self):
        runner = DockerSimulationRunner()
        fake_client = MagicMock()
        fake_client.ping.side_effect = ConnectionError("nope")
        runner._docker_client = fake_client
        assert runner.health_check() is False
        # The cached client was cleared so the next _get_client() forces a re-init.
        assert runner._docker_client is None

    def test_health_check_returns_true_when_daemon_up(self):
        runner = DockerSimulationRunner()
        fake_client = MagicMock()
        fake_client.ping.return_value = True
        runner._docker_client = fake_client
        assert runner.health_check() is True
        fake_client.ping.assert_called_once()


class TestRestrictedNetworkIptablesGuard:
    """B-1: RestrictedNetwork.ensure_network must fail closed when egress
    restriction is configured but iptables is unavailable, instead of
    silently logging an error and letting simulations run unrestricted
    (ASSERTIONS.md §Concordia violation).
    """

    def test_ensure_network_raises_when_iptables_missing_and_hosts_set(self):
        from aurelius.simulation.docker_runner import NetworkIsolationUnavailableError, RestrictedNetwork

        net = RestrictedNetwork(allowed_hosts=["api.deepseek.com"])
        with patch("shutil.which", return_value=None):
            with pytest.raises(NetworkIsolationUnavailableError, match="iptables"):
                net.ensure_network()

    def test_ensure_network_skips_check_when_allowlist_empty(self):
        """Empty allowlist is the documented opt-out; don't block it.
        We swap in a stub docker client so the network-creation path
        doesn't reach a real daemon in this unit test.
        """
        from aurelius.simulation.docker_runner import RestrictedNetwork

        net = RestrictedNetwork(allowed_hosts=[])
        fake_client = MagicMock()
        fake_client.networks.list.return_value = []
        fake_net = MagicMock()
        fake_net.attrs = {"IPAM": {"Config": [{"Subnet": "172.20.0.0/16"}]}}
        fake_client.networks.create.return_value = fake_net
        net._docker_client = fake_client
        with patch("shutil.which", return_value=None):
            net.ensure_network()  # no raise

    def test_ensure_network_passes_iptables_check_when_present(self):
        from aurelius.simulation.docker_runner import RestrictedNetwork

        net = RestrictedNetwork(allowed_hosts=["api.deepseek.com"])
        fake_client = MagicMock()
        fake_client.networks.list.return_value = []
        fake_net = MagicMock()
        fake_net.attrs = {"IPAM": {"Config": [{"Subnet": "172.20.0.0/16"}]}}
        fake_client.networks.create.return_value = fake_net
        net._docker_client = fake_client
        with patch("shutil.which", return_value="/usr/sbin/iptables"):
            with patch.object(net, "_apply_iptables_rules"):
                net.ensure_network()

    def test_network_isolation_unavailable_is_runtime_error(self):
        """Subclass of RuntimeError so existing ``except RuntimeError``
        handlers (e.g. validator startup preflight) catch it."""
        from aurelius.simulation.docker_runner import NetworkIsolationUnavailableError

        assert issubclass(NetworkIsolationUnavailableError, RuntimeError)

    def test_runner_init_propagates_isolation_failure(self):
        """DockerSimulationRunner.__init__ must surface NetworkIsolationUnavailableError
        from RestrictedNetwork.ensure_network() instead of swallowing it into
        the catch-all branch. Requires an explicit non-empty allowlist —
        an empty allowlist is an opt-out that skips RestrictedNetwork entirely."""
        from aurelius.simulation.docker_runner import NetworkIsolationUnavailableError

        with patch("shutil.which", return_value=None):
            with patch.object(
                DockerSimulationRunner,
                "_resolve_allowed_hosts",
                return_value=["api.deepseek.com"],
            ):
                with pytest.raises(NetworkIsolationUnavailableError):
                    DockerSimulationRunner(
                        llm_api_key="sk-test",
                        llm_base_url="https://api.deepseek.com/v1",
                    )
