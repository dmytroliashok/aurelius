"""Docker-based Concordia simulation runner.

Manages ephemeral containers for isolated simulation execution with
resource limits, network restrictions, and pre-warmed pool.
"""

import json
import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aurelius.config import Config
from aurelius.simulation.coherence import CoherenceResult, validate_coherence
from aurelius.simulation.transcript import Transcript, extract_transcript
from aurelius.simulation.translator import translate_config

if TYPE_CHECKING:
    from aurelius.validator.remote_config import RemoteConfig

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_TAG = "v3.0.0"
DEFAULT_POOL_SIZE = 2


class NetworkIsolationUnavailableError(RuntimeError):
    """Raised when sim egress restriction is configured but cannot be enforced.

    ASSERTIONS.md §Concordia requires simulation containers to have no network
    access except to configured LLM API endpoints. If `SIM_ALLOWED_LLM_HOSTS`
    is non-empty but the iptables binary is missing (or equivalently the
    container lacks NET_ADMIN), running simulations would silently violate
    this invariant. Callers must fail closed — either install iptables or
    set `SIM_ALLOWED_LLM_HOSTS=''` to explicitly acknowledge unrestricted
    egress.
    """


@dataclass
class SimulationResult:
    success: bool
    transcript: Transcript | None = None
    coherence: CoherenceResult | None = None
    error: str | None = None
    wall_clock_seconds: float = 0.0


class RestrictedNetwork:
    """Docker network with iptables rules allowing only specific LLM API endpoints.

    Creates a custom bridge network and adds iptables rules to restrict
    outbound traffic to the resolved IPs of configured LLM API hostnames.
    """

    def __init__(self, allowed_hosts: list[str] | None = None, network_name: str | None = None):
        self._allowed_hosts = allowed_hosts or []
        self._network_name = network_name if network_name is not None else Config.SIM_NETWORK_NAME
        self._docker_client = None
        self._network = None

    @property
    def network_name(self) -> str:
        return self._network_name

    @property
    def NETWORK_NAME(self) -> str:  # noqa: N802 — preserved for callers that still use the old name
        return self._network_name

    def _get_client(self):
        if self._docker_client is None:
            import docker

            self._docker_client = docker.from_env()
        return self._docker_client

    def _resolve_hosts(self) -> list[str]:
        """Resolve allowed hostnames to IP addresses."""
        import socket

        ips = set()
        for host in self._allowed_hosts:
            try:
                for info in socket.getaddrinfo(host, None):
                    ips.add(info[4][0])
            except socket.gaierror:
                logger.warning("Could not resolve LLM API host: %s", host)
        return sorted(ips)

    def _check_iptables_available(self) -> None:
        """Fail closed if iptables is required but unavailable.

        B-1: ``_apply_iptables_rules`` previously logged an ERROR and
        continued when ``iptables`` was missing, leaving containers with
        unrestricted egress. That silently violates ASSERTIONS.md §Concordia.
        """
        if not self._allowed_hosts:
            # Operator explicitly opted out of egress restriction via
            # empty allowlist — honor it instead of failing closed.
            return
        import shutil

        if shutil.which("iptables") is None:
            raise NetworkIsolationUnavailableError(
                "SIM_ALLOWED_LLM_HOSTS is non-empty but the iptables binary "
                "is not on PATH. Without it, simulation containers would "
                "have unrestricted network egress (ASSERTIONS.md §Concordia). "
                "Either install iptables and grant the validator container "
                "NET_ADMIN, or set SIM_ALLOWED_LLM_HOSTS='' to explicitly "
                "accept unrestricted egress."
            )

    def ensure_network(self) -> str:
        """Create or reuse the restricted network. Returns network name.

        Raises:
            NetworkIsolationUnavailableError: the allowlist is non-empty but
                iptables is unavailable to enforce it.
        """
        self._check_iptables_available()
        client = self._get_client()

        # Check if network already exists (may be stale from a crashed process)
        try:
            existing = client.networks.list(names=[self.NETWORK_NAME])
            if existing:
                self._network = existing[0]
                # Clean up stale iptables rules from previous runs before re-applying
                try:
                    self._network.reload()
                    ipam = self._network.attrs.get("IPAM", {})
                    configs = ipam.get("Config", [])
                    subnet = configs[0].get("Subnet", "") if configs else ""
                    if subnet:
                        self._remove_iptables_rules(subnet)
                        logger.info("Cleaned up stale iptables rules for existing network %s", self.NETWORK_NAME)
                except Exception:
                    pass
                # Re-apply fresh rules
                allowed_ips = self._resolve_hosts()
                if allowed_ips:
                    self._apply_iptables_rules(allowed_ips)
                return self.NETWORK_NAME
        except Exception:
            pass

        # Create with internal=False so we can apply iptables rules
        try:
            ipam_config = None  # Let Docker assign
            self._network = client.networks.create(
                self.NETWORK_NAME,
                driver="bridge",
                internal=False,
                ipam=ipam_config,
            )

            # Apply iptables rules to restrict outbound
            allowed_ips = self._resolve_hosts()
            if allowed_ips:
                self._apply_iptables_rules(allowed_ips)
            else:
                logger.warning("No LLM API IPs resolved — simulation containers will have no outbound access")

            return self.NETWORK_NAME
        except Exception as e:
            logger.warning("Failed to create restricted network, falling back to none: %s", e)
            return "none"

    def _apply_iptables_rules(self, allowed_ips: list[str]) -> None:
        """Apply iptables rules to restrict the Docker network to allowed IPs."""
        import subprocess

        chain = "DOCKER-USER"
        # Get network subnet
        if not self._network:
            return

        self._network.reload()
        ipam = self._network.attrs.get("IPAM", {})
        configs = ipam.get("Config", [])
        subnet = configs[0].get("Subnet", "") if configs else ""
        if not subnet:
            logger.warning("Could not determine subnet for restricted network")
            return

        # Validate subnet is a well-formed CIDR to prevent injection into iptables
        import ipaddress

        try:
            ipaddress.IPv4Network(subnet, strict=False)
        except (ipaddress.AddressValueError, ipaddress.NetmaskValueError, ValueError) as e:
            logger.error("Invalid subnet from Docker IPAM: %r — %s", subnet, e)
            return

        try:
            # Allow established connections
            subprocess.run(
                [
                    "iptables",
                    "-I",
                    chain,
                    "-s",
                    subnet,
                    "-m",
                    "conntrack",
                    "--ctstate",
                    "ESTABLISHED,RELATED",
                    "-j",
                    "ACCEPT",
                ],
                check=True,
                capture_output=True,
            )
            # Allow DNS (needed for initial resolution)
            subprocess.run(
                ["iptables", "-I", chain, "-s", subnet, "-p", "udp", "--dport", "53", "-j", "ACCEPT"],
                check=True,
                capture_output=True,
            )
            # Allow specific LLM API IPs on HTTPS
            for ip in allowed_ips:
                subprocess.run(
                    ["iptables", "-I", chain, "-s", subnet, "-d", ip, "-p", "tcp", "--dport", "443", "-j", "ACCEPT"],
                    check=True,
                    capture_output=True,
                )
            # Drop everything else from this subnet
            subprocess.run(
                ["iptables", "-A", chain, "-s", subnet, "-j", "DROP"],
                check=True,
                capture_output=True,
            )
            logger.info("iptables rules applied: %d allowed IPs for subnet %s", len(allowed_ips), subnet)
        except FileNotFoundError:
            logger.error(
                "SECURITY: iptables not found — simulation containers will have UNRESTRICTED network access. "
                "This is expected on macOS/Windows but is a security risk on Linux. "
                "Install iptables or set SIM_ALLOWED_LLM_HOSTS='' to acknowledge."
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                "SECURITY: Failed to apply iptables rules — simulation containers have unrestricted access: %s",
                e.stderr.decode() if e.stderr else e,
            )

    def cleanup(self) -> None:
        """Remove the restricted network and associated iptables rules."""
        if self._network:
            try:
                self._network.reload()
                ipam = self._network.attrs.get("IPAM", {})
                configs = ipam.get("Config", [])
                subnet = configs[0].get("Subnet", "") if configs else ""
                if subnet:
                    self._remove_iptables_rules(subnet)
            except Exception as e:
                logger.debug("Could not clean up iptables rules: %s", e)
            try:
                self._network.remove()
            except Exception as e:
                logger.warning("Failed to remove restricted network: %s", e)
            self._network = None

    def _remove_iptables_rules(self, subnet: str) -> None:
        """Remove iptables rules for the given subnet (best-effort)."""
        import subprocess

        chain = "DOCKER-USER"
        # List all rules in the chain and remove ones matching our subnet
        try:
            result = subprocess.run(
                ["iptables", "-L", chain, "-n", "--line-numbers"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return
            # Parse rules matching our subnet, remove in reverse order to preserve line numbers
            lines_to_delete = []
            for line in result.stdout.splitlines():
                if subnet.split("/")[0] in line:
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        lines_to_delete.append(int(parts[0]))
            for line_num in reversed(lines_to_delete):
                subprocess.run(
                    ["iptables", "-D", chain, str(line_num)],
                    capture_output=True,
                )
            if lines_to_delete:
                logger.info("Cleaned up %d iptables rules for subnet %s", len(lines_to_delete), subnet)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("iptables cleanup error: %s", e)


class ContainerPool:
    """Pre-warmed Docker container pool.

    Keeps N containers created and ready from the current image tag.
    Containers are leased out for simulations and replaced after use.
    Pool is drained and recreated when the image tag changes.
    """

    def __init__(self, image: str, pool_size: int = DEFAULT_POOL_SIZE):
        self._image = image
        self._pool_size = pool_size
        self._available: list = []
        self._lock = threading.Lock()
        self._docker_client = None

    def _get_client(self):
        if self._docker_client is None:
            import docker

            self._docker_client = docker.from_env()
        return self._docker_client

    def warm(self) -> None:
        """Fill the pool to target size with idle containers."""
        with self._lock:
            needed = self._pool_size - len(self._available)
            if needed <= 0:
                return
        for _ in range(needed):
            try:
                client = self._get_client()
                container = client.containers.create(
                    image=self._image,
                    command=["sleep", "infinity"],
                    detach=True,
                    network_mode="none",
                    user="65534:65534",
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges:true"],
                    pids_limit=256,
                )
                with self._lock:
                    if len(self._available) >= self._pool_size:
                        # Another thread filled the pool; discard this container
                        try:
                            container.remove(force=True)
                        except Exception:
                            pass
                        break
                    self._available.append(container)
            except Exception as e:
                logger.warning("Failed to pre-warm container: %s", e)
                break
        logger.debug("Container pool: %d/%d ready", len(self._available), self._pool_size)

    def lease(self):
        """Get a pre-warmed container, or None if pool is empty."""
        with self._lock:
            if self._available:
                return self._available.pop(0)
        return None

    def drain(self) -> None:
        """Remove and destroy all pooled containers."""
        with self._lock:
            containers = list(self._available)
            self._available.clear()
        for c in containers:
            try:
                c.remove(force=True)
            except Exception as e:
                logger.warning("Failed to remove pooled container during drain: %s", e)
        logger.debug("Container pool drained")

    def update_image(self, new_image: str) -> None:
        """Switch to a new image, draining existing containers."""
        if new_image == self._image:
            return
        logger.info("Container pool: image changed %s → %s, draining", self._image, new_image)
        self.drain()
        self._image = new_image
        self.warm()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._available)

    def close(self) -> None:
        self.drain()
        if self._docker_client:
            self._docker_client.close()
            self._docker_client = None


class DockerSimulationRunner:
    """Manages Docker containers for Concordia simulation."""

    def __init__(
        self,
        remote_config: "RemoteConfig | None" = None,
        llm_api_key: str | None = None,
        image_name: str | None = None,
        image_tag: str = DEFAULT_IMAGE_TAG,
        image_digest: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        base_timeout: int | None = None,
        base_ram_mb: int | None = None,
        cpu_count: int | None = None,
    ):
        self._remote_config = remote_config

        def _cfg(remote_attr: str, config_attr: str):
            if remote_config is not None:
                return getattr(remote_config, remote_attr)
            return getattr(Config, config_attr)

        self.image_name = image_name if image_name is not None else _cfg("concordia_image_name", "CONCORDIA_IMAGE_NAME")
        self.image_tag = image_tag
        self.image_digest = (
            image_digest if image_digest is not None else _cfg("concordia_image_digest", "CONCORDIA_IMAGE_DIGEST")
        )
        self.llm_model = llm_model if llm_model is not None else _cfg("llm_model", "LLM_MODEL")
        self.llm_api_key = llm_api_key or ""
        self.llm_base_url = llm_base_url if llm_base_url is not None else _cfg("llm_base_url", "LLM_BASE_URL")
        self.base_timeout = base_timeout if base_timeout is not None else _cfg("sim_base_timeout", "SIM_BASE_TIMEOUT")
        self.base_ram_mb = base_ram_mb if base_ram_mb is not None else _cfg("sim_base_ram_mb", "SIM_BASE_RAM_MB")
        self.cpu_count = cpu_count if cpu_count is not None else _cfg("sim_cpu_count", "SIM_CPU_COUNT")
        self._docker_client = None
        self._active_containers: list = []  # Track running containers for cleanup

        # Restricted network for LLM API access
        self._restricted_network: RestrictedNetwork | None = None
        self._is_local_llm = self._check_local_base_url(self.llm_base_url)
        if self.llm_api_key and not self._is_local_llm:
            # External LLM API (including base_url pointing to public APIs like DeepSeek)
            # — restrict container to known API hosts only when the operator
            # has asked for enforcement. An empty resolved allowlist is an
            # explicit opt-out: skip RestrictedNetwork entirely rather than
            # resurrecting enforcement by auto-appending the LLM base host.
            allowed_hosts = self._resolve_allowed_hosts()
            if allowed_hosts:
                if self.llm_base_url:
                    # Ensure the base_url host is reachable through the
                    # restricted network (convenience — only applied when
                    # enforcement is already desired).
                    from urllib.parse import urlparse

                    base_host = urlparse(self.llm_base_url).hostname
                    if base_host and base_host not in allowed_hosts:
                        allowed_hosts.append(base_host)
                network_name = _cfg("sim_network_name", "SIM_NETWORK_NAME")
                self._restricted_network = RestrictedNetwork(
                    allowed_hosts=allowed_hosts,
                    network_name=network_name,
                )
                try:
                    self._restricted_network.ensure_network()
                except NetworkIsolationUnavailableError:
                    # Fail closed — never silently run sims with unrestricted
                    # egress when the operator asked for an allowlist.
                    raise
                except Exception as e:
                    logger.warning("Failed to set up restricted network: %s", e)
                    self._restricted_network = None

        # Ensure image exists (auto-build on first run)
        self._ensure_image()

        # Container pool
        self._pool_size = _cfg("container_pool_size", "CONTAINER_POOL_SIZE")
        full_image = f"{self.image_name}:{self.image_tag}"
        self._pool = ContainerPool(full_image, pool_size=self._pool_size)
        if self._pool_size > 0:
            try:
                self._pool.warm()
            except Exception as e:
                logger.warning("Failed to warm container pool: %s", e)

    def _resolve_allowed_hosts(self) -> list[str]:
        """Return the LLM API hostnames allowed through the restricted network."""
        if self._remote_config is not None:
            return list(self._remote_config.sim_allowed_llm_hosts)
        return list(Config.SIM_ALLOWED_LLM_HOSTS)

    def _ensure_image(self):
        """Ensure the Concordia simulation image is available.

        Resolution order:
        1. Check for image locally
        2. Try ``docker pull`` from GHCR
        3. Fall back to local build from package data
        """
        import importlib.resources
        import shutil

        import docker as docker_lib

        tag = f"{self.image_name}:{self.image_tag}"
        client = self._get_client()

        # 1. Already present locally
        try:
            client.images.get(tag)
            logger.debug("Concordia image %s found locally", tag)
            return
        except docker_lib.errors.ImageNotFound:
            pass

        # 2. Try pulling from registry (e.g. GHCR)
        logger.info("Concordia image %s not found locally — attempting pull...", tag)
        try:
            client.images.pull(self.image_name, tag=self.image_tag)
            logger.info("Concordia image %s pulled successfully", tag)
            return
        except Exception as pull_err:
            logger.info("Pull failed (%s) — falling back to local build", pull_err)

        # 3. Build locally from package data
        logger.info("Building Concordia image %s (first run, may take a minute)...", tag)

        try:
            # Build context: temp directory with Dockerfile + entrypoint.py
            with tempfile.TemporaryDirectory() as build_ctx:
                # Copy Dockerfile from package data
                dockerfile_ref = importlib.resources.files("aurelius.simulation").joinpath("Dockerfile.concordia")
                with importlib.resources.as_file(dockerfile_ref) as dockerfile_path:
                    shutil.copy2(dockerfile_path, Path(build_ctx) / "Dockerfile")

                # Copy entrypoint.py from package
                entrypoint_ref = importlib.resources.files("aurelius.simulation").joinpath("entrypoint.py")
                with importlib.resources.as_file(entrypoint_ref) as entrypoint_path:
                    shutil.copy2(entrypoint_path, Path(build_ctx) / "entrypoint.py")

                # Build image
                _image, build_logs = client.images.build(
                    path=build_ctx,
                    tag=tag,
                    rm=True,
                )
                for chunk in build_logs:
                    if "stream" in chunk:
                        line = chunk["stream"].strip()
                        if line:
                            logger.debug("docker build: %s", line)

            logger.info("Concordia image %s built successfully", tag)
        except Exception as e:
            logger.warning(
                "Failed to build Concordia image %s: %s. "
                "Simulations will fail until the image is available. "
                "Build manually with: docker build -f Dockerfile.concordia -t %s .",
                tag,
                e,
                tag,
            )

    def _get_client(self):
        if self._docker_client is None:
            import docker

            self._docker_client = docker.from_env()
            # Fail-fast: verify Docker daemon is reachable
            try:
                self._docker_client.ping()
            except PermissionError as e:
                self._docker_client = None
                raise RuntimeError(
                    f"Docker socket permission denied: {e}. "
                    "The container user cannot access /var/run/docker.sock. "
                    "Add '--group-add 0' to your docker run command, or use a docker-proxy container."
                ) from e
            except Exception as e:
                self._docker_client = None
                raise RuntimeError(
                    f"Docker daemon is not reachable: {e}. "
                    "Ensure Docker is running and accessible (check DOCKER_HOST env var for remote daemons)."
                ) from e
        return self._docker_client

    # CS-11: explicit startup + periodic health-check API. `_get_client`
    # only pings on first lazy use, which masks daemon outages until the
    # first simulation — these helpers make the check deliberate.
    def preflight_check(self) -> None:
        """Raise RuntimeError if the Docker daemon is unreachable. Called once at startup."""
        self._get_client()

    def health_check(self) -> bool:
        """Return True iff the Docker daemon currently responds to ping.

        On failure, clear the cached client so the next `_get_client()`
        call forces a fresh connection and surfaces a clear error.
        """
        try:
            if self._docker_client is None:
                self._get_client()
            else:
                self._docker_client.ping()
            return True
        except Exception as e:
            logger.error("docker_daemon_unreachable | %s", e)
            self._docker_client = None
            return False

    @staticmethod
    def _check_local_base_url(base_url: str) -> bool:
        """Return True if base_url points to a local/private network address."""
        if not base_url:
            return False
        from urllib.parse import urlparse

        hostname = urlparse(base_url).hostname or ""
        if hostname in ("localhost", "127.0.0.1", "::1", "host.docker.internal"):
            return True
        # Private IP ranges (10.x, 172.16-31.x, 192.168.x)
        parts = hostname.split(".")
        if len(parts) == 4:
            try:
                first = int(parts[0])
                second = int(parts[1])
                if first == 10:
                    return True
                if first == 172 and 16 <= second <= 31:
                    return True
                if first == 192 and second == 168:
                    return True
            except ValueError:
                pass
        return False

    @staticmethod
    def _detect_host_path(container_path: str) -> str | None:
        """Auto-detect the host-side path for a container mount.

        Reads /proc/self/mountinfo to find the host source path for a given
        mount point. Works for both bind mounts and Docker named volumes on
        Linux. Returns None on macOS/bare metal or if detection fails.
        """
        try:
            with open("/proc/self/mountinfo") as f:
                for line in f:
                    fields = line.split()
                    mount_point = fields[4]
                    if mount_point == container_path:
                        # mountinfo format: id parent major:minor root mount_point options ... - fs_type source super_opts
                        # root = path within the source filesystem that is mounted
                        # For bind mounts: root = /host/path (the host-side directory)
                        # For named volumes: root = /var/lib/docker/volumes/<name>/_data
                        root = fields[3]
                        if root != "/":
                            return root
                        # root == "/" means the entire device is mounted — can't resolve host path
                        return None
        except (FileNotFoundError, ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _resolve_sim_host_dir(sim_data_dir: str | None, sim_data_host_dir: str) -> str:
        """Resolve the host-side path for simulation data.

        Resolution order:
        1. Explicit SIM_DATA_HOST_DIR (always wins)
        2. Auto-detect from /proc/self/mountinfo (Linux Docker)
        3. Same as sim_data_dir (bare metal — no translation needed)
        """
        if sim_data_host_dir:
            return sim_data_host_dir

        if not sim_data_dir:
            return ""

        # Check if we're inside a Docker container
        in_docker = Path("/.dockerenv").exists()
        if not in_docker:
            # Bare metal: container path == host path, no translation needed
            return sim_data_dir

        # Docker: try auto-detection from mount info
        detected = DockerSimulationRunner._detect_host_path(sim_data_dir)
        if detected:
            logger.info("Auto-detected SIM_DATA_HOST_DIR=%s from mount info", detected)
            return detected

        logger.warning(
            "Running in Docker but cannot auto-detect host path for %s. "
            "Set SIM_DATA_HOST_DIR to the host-side absolute path of your %s volume mount. "
            "Example: if you used '-v /home/user/simdata:%s', set SIM_DATA_HOST_DIR=/home/user/simdata",
            sim_data_dir,
            sim_data_dir,
            sim_data_dir,
        )
        return ""

    def _get_network_mode(self) -> str:
        """Determine the appropriate network mode for simulation containers."""
        if self._is_local_llm:
            # Local LLM server — use bridge for full access
            return "bridge"
        if self._restricted_network:
            # External LLM API — use restricted network
            return self._restricted_network.network_name
        if self.llm_api_key:
            # External LLM configured but operator opted out of restricted
            # egress (empty SIM_ALLOWED_LLM_HOSTS). Use bridge so the sim
            # container can reach the LLM API on the public internet;
            # isolation is explicitly waived.
            return "bridge"
        # No LLM configured — fully isolated (no reason to grant network).
        return "none"

    def _compute_limits(self, agent_count: int) -> dict:
        """Compute resource limits scaled by agent count. Max timeout capped at 30 min."""
        timeout = min(int(self.base_timeout * (agent_count / 2)), 1800)  # Cap at 30 min
        ram_bytes = int(self.base_ram_mb * (1 + 0.25 * (agent_count - 2)) * 1024 * 1024)
        return {
            "timeout": timeout,
            "mem_limit": ram_bytes,
            "nano_cpus": self.cpu_count * 1_000_000_000,
        }

    def run_simulation(self, config: dict) -> SimulationResult:
        """Run a Concordia simulation in an isolated Docker container.

        Args:
            config: Validated scenario config dict.

        Returns:
            SimulationResult with transcript and coherence check.
        """
        start_time = time.monotonic()

        # Translate config
        try:
            setup = translate_config(config)
        except Exception as e:
            return SimulationResult(success=False, error=f"Translation failed: {e}")

        agent_count = len(setup.agents)
        limits = self._compute_limits(agent_count)
        agent_names = [a.name for a in setup.agents]

        # Write setup to temp file for volume mount.
        # SIM_DATA_DIR: path inside this container where temp files are created
        # SIM_DATA_HOST_DIR: corresponding path on the Docker host (for volume mounts)
        # In Docker-in-Docker, the Docker daemon runs on the host and needs host paths.
        # _resolve_sim_host_dir auto-detects the host path on Linux Docker.
        if self._remote_config is not None:
            sim_data_dir = self._remote_config.sim_data_dir
            sim_data_host_dir_raw = self._remote_config.sim_data_host_dir
        else:
            sim_data_dir = Config.SIM_DATA_DIR
            sim_data_host_dir_raw = Config.SIM_DATA_HOST_DIR
        sim_data_host_dir = self._resolve_sim_host_dir(sim_data_dir, sim_data_host_dir_raw)
        try:
            with tempfile.TemporaryDirectory(dir=sim_data_dir) as tmpdir:
                input_path = Path(tmpdir) / "setup.json"
                output_path = Path(tmpdir) / "output.json"

                input_path.write_text(setup.model_dump_json(indent=2))
                # World-readable/writable: container runs as nobody (65534) for security.
                # The temp dir needs write access for output.json.
                Path(tmpdir).chmod(0o777)
                input_path.chmod(0o444)

                # For DinD: translate container path to host path for volume mount
                if sim_data_host_dir and sim_data_dir and tmpdir.startswith(sim_data_dir):
                    mount_path = sim_data_host_dir + tmpdir[len(sim_data_dir) :]
                else:
                    mount_path = tmpdir

                # Run container — use digest for tamper-proof image reference (CS-06)
                client = self._get_client()
                require_digest = (
                    self._remote_config.require_image_digest
                    if self._remote_config is not None
                    else Config.REQUIRE_IMAGE_DIGEST
                )
                if self.image_digest:
                    full_image = f"{self.image_name}@{self.image_digest}"
                elif require_digest:
                    # Strict mode (mainnet/testnet): fail closed on missing digest
                    return SimulationResult(
                        success=False,
                        error=(
                            "CONCORDIA_IMAGE_DIGEST is required when REQUIRE_IMAGE_DIGEST=1. "
                            "Refusing to run a tag-only image in production."
                        ),
                        wall_clock_seconds=time.monotonic() - start_time,
                    )
                else:
                    logger.warning(
                        "No image digest configured for %s:%s — running tag-only image. "
                        "Set CONCORDIA_IMAGE_DIGEST for tamper-evident integrity verification.",
                        self.image_name,
                        self.image_tag,
                    )
                    full_image = f"{self.image_name}:{self.image_tag}"

                env = {
                    "LLM_MODEL": self.llm_model,
                }

                # Pass API key via mounted secrets file (not env var) to avoid
                # exposure in `docker inspect` output.
                if self.llm_api_key:
                    secrets_path = Path(tmpdir) / ".llm_secrets"
                    secrets_path.write_text(self.llm_api_key)
                    secrets_path.chmod(0o444)  # World-readable: container runs as nobody (65534)
                    env["LLM_API_KEY_FILE"] = "/data/.llm_secrets"
                if self.llm_base_url:
                    env["LLM_BASE_URL"] = self.llm_base_url
                    # Also set OPENAI_BASE_URL so the OpenAI SDK (used by Concordia
                    # library internals) picks up the custom endpoint automatically.
                    env["OPENAI_BASE_URL"] = self.llm_base_url

                network = self._get_network_mode()

                timeout = limits["timeout"]
                container = client.containers.run(
                    image=full_image,
                    command=["/data/setup.json", "/data/output.json"],
                    volumes={
                        mount_path: {"bind": "/data", "mode": "rw"},
                    },
                    environment=env,
                    mem_limit=limits["mem_limit"],
                    nano_cpus=limits["nano_cpus"],
                    network_mode=network,
                    # Security hardening
                    user="65534:65534",  # nobody — don't run as root
                    cap_drop=["ALL"],  # Drop all Linux capabilities
                    security_opt=["no-new-privileges:true"],
                    pids_limit=256,  # Prevent fork bombs
                    ipc_mode="private",  # Isolate IPC namespace
                    # Execution control
                    detach=True,  # Don't block — use wait() with timeout
                    remove=False,  # We remove manually after reading output
                )
                self._active_containers.append(container)

                # Enforce timeout: wait then kill if exceeded
                try:
                    container.wait(timeout=timeout)
                except (TimeoutError, ConnectionError, Exception) as e:
                    logger.warning("Container exceeded timeout (%ds), killing: %s", timeout, type(e).__name__)
                    timeout_logs = ""
                    try:
                        timeout_logs = container.logs(tail=30).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    try:
                        container.kill()
                    except Exception:
                        logger.debug("Container already stopped during timeout cleanup")
                    try:
                        container.remove(force=True)
                    except Exception as cleanup_err:
                        logger.warning("Failed to remove timed-out container: %s", cleanup_err)
                    if container in self._active_containers:
                        self._active_containers.remove(container)
                    error_msg = f"Simulation timed out after {timeout}s"
                    if timeout_logs:
                        logger.error("Timed-out container logs:\n%s", timeout_logs)
                        error_msg += f" | logs: {timeout_logs[:500]}"
                    return SimulationResult(
                        success=False,
                        error=error_msg,
                        wall_clock_seconds=time.monotonic() - start_time,
                    )

                # Capture logs before cleanup (for diagnostics on failure)
                container_logs = ""
                try:
                    container_logs = container.logs(tail=50).decode("utf-8", errors="replace")
                except Exception:
                    pass

                # Clean up container
                try:
                    container.remove()
                except Exception as cleanup_err:
                    logger.warning("Failed to remove simulation container: %s", cleanup_err)
                if container in self._active_containers:
                    self._active_containers.remove(container)

                elapsed = time.monotonic() - start_time

                # Parse output
                if not output_path.exists():
                    error_msg = "No output produced by simulation container"
                    if container_logs:
                        logger.error("Simulation container logs:\n%s", container_logs)
                        error_msg += f" | logs: {container_logs[:500]}"
                    return SimulationResult(
                        success=False,
                        error=error_msg,
                        wall_clock_seconds=elapsed,
                    )

                # Guard against oversized output (DoS via disk/memory)
                output_size = output_path.stat().st_size
                if output_size > 10 * 1024 * 1024:  # 10 MB limit
                    return SimulationResult(
                        success=False,
                        error=f"Output too large ({output_size} bytes)",
                        wall_clock_seconds=elapsed,
                    )
                raw_output = json.loads(output_path.read_text())
                transcript = extract_transcript(raw_output)
                transcript.metadata.wall_clock_seconds = elapsed
                transcript.metadata.docker_image_tag = self.image_tag
                transcript.metadata.llm_model = self.llm_model

                # Diagnostic surface: when the entrypoint catches an exception
                # it still writes a 1-event fallback ``output.json`` with
                # ``completed: false``, so the earlier ``output_path.exists()``
                # branch (line ~891) never fires and the captured stderr
                # would otherwise be silently dropped. Emit it here so the
                # validator log shows the underlying traceback.
                if not transcript.completed and container_logs:
                    logger.error(
                        "Simulation reported completed=false; container logs:\n%s",
                        container_logs,
                    )

                coherence = validate_coherence(transcript, expected_agents=agent_names)

                # Replenish pool in background after use
                if self._pool_size > 0:
                    threading.Thread(target=self._pool.warm, daemon=True).start()

                return SimulationResult(
                    success=coherence.passed,
                    transcript=transcript,
                    coherence=coherence,
                    wall_clock_seconds=elapsed,
                )

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error("Simulation failed: %s", e)
            return SimulationResult(
                success=False,
                error=str(e),
                wall_clock_seconds=elapsed,
            )

    def update_image(self, new_tag: str, expected_digest: str | None = None) -> None:
        """Update the Docker image tag with optional digest verification.

        If expected_digest is provided, the pulled image must match it.
        This prevents a compromised remote config from pushing malicious images.
        """
        if new_tag == self.image_tag and not expected_digest:
            return

        logger.info("Updating Concordia image tag: %s → %s", self.image_tag, new_tag)
        try:
            client = self._get_client()
            image = client.images.pull(self.image_name, tag=new_tag)
            repo_digests = image.attrs.get("RepoDigests", [])
            pulled_digest = repo_digests[0].split("@")[-1] if repo_digests else ""

            if expected_digest:
                if not pulled_digest:
                    logger.error(
                        "Image has no digest but expected=%s — rejecting image update",
                        expected_digest,
                    )
                    return
                if pulled_digest != expected_digest:
                    logger.error(
                        "Image digest mismatch! expected=%s got=%s — rejecting image update",
                        expected_digest,
                        pulled_digest,
                    )
                    return

            if not pulled_digest:
                logger.error(
                    "Pulled image %s:%s has no digest — rejecting image update (CS6: digest required)",
                    self.image_name,
                    new_tag,
                )
                return

            self.image_tag = new_tag
            self.image_digest = pulled_digest
            logger.info("Image verified: %s:%s digest=%s", self.image_name, new_tag, pulled_digest[:24])

            # Update pre-warmed pool with new image
            if self._pool_size > 0:
                new_full = f"{self.image_name}:{new_tag}"
                self._pool.update_image(new_full)
        except Exception:
            logger.warning("Failed to pull image %s:%s", self.image_name, new_tag)

    def close(self) -> None:
        """Clean up Docker client, container pool, restricted network, and active containers."""
        # Kill any containers still running (e.g. from cancelled async tasks)
        for container in list(self._active_containers):
            try:
                container.kill()
                container.remove(force=True)
                logger.info("Cleaned up active simulation container on shutdown")
            except Exception:
                pass
        self._active_containers.clear()

        self._pool.close()
        if self._restricted_network:
            self._restricted_network.cleanup()
        if self._docker_client:
            self._docker_client.close()
            self._docker_client = None
