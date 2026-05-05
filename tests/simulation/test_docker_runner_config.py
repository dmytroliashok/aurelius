"""Tests that DockerSimulationRunner resolves config at call time, not import time.

Regression protection: previously `DEFAULT_BASE_TIMEOUT`, `DEFAULT_BASE_RAM_MB`,
and `DEFAULT_CPU_COUNT` were captured from `Config` at module import, and
`RestrictedNetwork.NETWORK_NAME` was a class attribute frozen at class-definition
time. Those captures prevented remote config changes from taking effect.
"""

from types import SimpleNamespace

from aurelius.simulation.docker_runner import (
    DEFAULT_IMAGE_TAG,
    DockerSimulationRunner,
    RestrictedNetwork,
)


def _fake_remote(**overrides):
    base = {
        "concordia_image_name": "fake/image",
        "concordia_image_digest": "",
        "llm_model": "fake-model",
        "llm_base_url": "https://fake.example.com",
        "sim_base_timeout": 900,
        "sim_base_ram_mb": 8192,
        "sim_cpu_count": 4,
        "container_pool_size": 0,
        "sim_network_name": "fake-net",
        "sim_allowed_llm_hosts": ["fake.example.com"],
        "sim_data_dir": "",
        "sim_data_host_dir": "",
        "require_image_digest": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRemoteConfigWiring:
    def test_runner_reads_all_values_from_remote_config(self):
        rc = _fake_remote()
        runner = DockerSimulationRunner(remote_config=rc)
        assert runner.image_name == "fake/image"
        assert runner.image_tag == DEFAULT_IMAGE_TAG
        assert runner.llm_model == "fake-model"
        assert runner.llm_base_url == "https://fake.example.com"
        assert runner.base_timeout == 900
        assert runner.base_ram_mb == 8192
        assert runner.cpu_count == 4
        assert runner._pool_size == 0

    def test_explicit_kwargs_override_remote_config(self):
        rc = _fake_remote()
        runner = DockerSimulationRunner(
            remote_config=rc,
            image_name="explicit/image",
            base_timeout=120,
            base_ram_mb=1024,
            cpu_count=1,
        )
        assert runner.image_name == "explicit/image"
        assert runner.base_timeout == 120
        assert runner.base_ram_mb == 1024
        assert runner.cpu_count == 1


class TestNoModuleLevelCapture:
    def test_restricted_network_name_is_instance_attribute(self):
        """RestrictedNetwork used to freeze NETWORK_NAME as a class attribute at
        class-definition time. Now it must be per-instance."""
        net_a = RestrictedNetwork(network_name="net-a")
        net_b = RestrictedNetwork(network_name="net-b")
        assert net_a.network_name == "net-a"
        assert net_b.network_name == "net-b"
        # Different instances → different names (not class-global)
        assert net_a.network_name != net_b.network_name

    def test_runner_timeout_tracks_remote_config_changes_across_instances(self):
        """Creating a new runner with different remote_config values must use
        those new values. This guards against the previous module-level capture."""
        rc_1 = _fake_remote(sim_base_timeout=600)
        runner_1 = DockerSimulationRunner(remote_config=rc_1)
        assert runner_1.base_timeout == 600

        rc_2 = _fake_remote(sim_base_timeout=1200)
        runner_2 = DockerSimulationRunner(remote_config=rc_2)
        assert runner_2.base_timeout == 1200

    def test_no_module_level_default_constants(self):
        """The old DEFAULT_BASE_TIMEOUT / DEFAULT_BASE_RAM_MB / DEFAULT_CPU_COUNT
        module-level names are gone — if anything imports them, it'll fail here."""
        from aurelius.simulation import docker_runner

        assert not hasattr(docker_runner, "DEFAULT_BASE_TIMEOUT")
        assert not hasattr(docker_runner, "DEFAULT_BASE_RAM_MB")
        assert not hasattr(docker_runner, "DEFAULT_CPU_COUNT")


class TestFallbackWhenNoRemoteConfig:
    def test_runner_without_remote_config_falls_back_to_config(self):
        """Passing remote_config=None reads from aurelius.config.Config at call time
        (preserving backward compat for tests that don't inject a remote_config)."""
        from aurelius.config import Config

        runner = DockerSimulationRunner()
        assert runner.base_timeout == Config.SIM_BASE_TIMEOUT
        assert runner.base_ram_mb == Config.SIM_BASE_RAM_MB
        assert runner.cpu_count == Config.SIM_CPU_COUNT
