import concurrent.futures
import json
import logging
import signal
import socket
import subprocess
import threading
import sys
import time
from pathlib import Path
from typing import Tuple  # noqa: UP035 — bittensor SDK requires typing.Tuple

import bittensor as bt

import aurelius
from aurelius.common.version import PROTOCOL_VERSION
from aurelius.config import Config
from aurelius.miner.config_store import ConfigStore
from aurelius.miner.work_token import generate_work_id
from aurelius.protocol import ScenarioConfigSynapse

logger = logging.getLogger(__name__)


class Miner:
    def __init__(self):
        self.config = Config
        self.should_exit = False

        # Warn if wallet is still on defaults (easy identity collision)
        if self.config.WALLET_NAME == "default" and self.config.WALLET_HOTKEY == "default":
            logger.warning(
                "WALLET_NAME and WALLET_HOTKEY are both 'default'. "
                "Set explicit wallet names to avoid identity collisions between operators."
            )

        # Guard against TESTLAB_MODE on mainnet — disables validator-permit
        # checks, allowing any registered hotkey to query the miner (CS-H2).
        if self.config.TESTLAB_MODE and self.config.NETWORK == "finney":
            raise RuntimeError(
                "TESTLAB_MODE=1 is not allowed on mainnet (finney). "
                "This disables validator-permit checks and exposes the miner to "
                "unauthorized queries. Remove TESTLAB_MODE or set ENVIRONMENT=testnet."
            )

        self.wallet = bt.Wallet(name=self.config.WALLET_NAME, hotkey=self.config.WALLET_HOTKEY)
        self.subtensor = bt.Subtensor(network=self.config.NETWORK)
        self.metagraph = bt.Metagraph(
            netuid=self.config.NETUID,
            network=self.config.NETWORK,
            subtensor=self.subtensor,
        )

        config_dir = self.config.MINER_CONFIG_DIR
        if not Path(config_dir).is_dir():
            raise ValueError(
                f"MINER_CONFIG_DIR={config_dir!r} does not exist. "
                "Create the directory and add scenario JSON files, or set MINER_CONFIG_DIR to an existing path."
            )
        self.config_store = ConfigStore(config_dir)
        logger.info("Config store: %d configs loaded from %s", self.config_store.count, config_dir)
        self._generator_script = Path(__file__).resolve().parents[2] / "scripts" / "generate_miner_scenario_openrouter.py"
        self._generator_model = self.config.OPENROUTER_MODEL
        self._generator_attempts = self.config.GENERATOR_ATTEMPTS
        self._generator_timeout = self.config.GENERATOR_TIMEOUT

        external_ip = self.config.AXON_EXTERNAL_IP
        if external_ip == "auto":
            external_ip = self._detect_external_ip()
            logger.info("Auto-detected external IP: %s", external_ip)

        self.axon = bt.Axon(
            wallet=self.wallet,
            port=self.config.AXON_PORT,
            external_ip=external_ip,
            external_port=self.config.AXON_EXTERNAL_PORT,
        )
        self.axon.attach(forward_fn=self.forward, blacklist_fn=self.blacklist)

        logger.info("Serving axon on netuid %d, port %d", self.config.NETUID, self.config.AXON_PORT)
        self.axon.serve(netuid=self.config.NETUID, subtensor=self.subtensor)
        self.axon.start()

        logger.info("Miner started | wallet=%s hotkey=%s", self.wallet.name, self.wallet.hotkey_str)

        # Fetch and display deposit address for operator convenience.
        # Best-effort: a flaky API must never block miner startup; fall back
        # to the `aurelius-deposit` CLI if this banner can't be printed.
        try:
            from aurelius.common.central_api import CentralAPIClient, CentralAPIError

            with CentralAPIClient(self.config.CENTRAL_API_URL, timeout=5) as client:
                addr = client.get_designated_address().address
            if addr:
                logger.info("Work-token deposit address: %s", addr)
                logger.info(
                    "To deposit: btcli stake transfer --origin-netuid %d --dest-netuid %d"
                    " --dest %s --amount <AMOUNT> --network %s",
                    self.config.NETUID,
                    self.config.NETUID,
                    addr,
                    self.config.NETWORK,
                )
        except CentralAPIError as e:
            logger.debug("Could not fetch deposit address banner: %s", e)

    def forward(self, synapse: ScenarioConfigSynapse) -> ScenarioConfigSynapse:
        nxt = self.config_store.next_with_path()
        if nxt is None:
            logger.warning("No configs available to serve")
            threading.Thread(
                target=self._retire_and_replenish,
                args=(None,),
                daemon=True,
                name="retire-replenish-empty",
            ).start()
            return synapse
        scenario_config, config_path = nxt

        result = generate_work_id(scenario_config, self.wallet.hotkey.ss58_address, wallet=self.wallet)

        synapse.scenario_config = scenario_config
        synapse.work_id = result.work_id
        synapse.work_id_nonce = result.nonce
        synapse.work_id_time_ns = result.time_ns
        synapse.work_id_signature = result.signature
        synapse.miner_version = aurelius.__version__
        synapse.miner_protocol_version = PROTOCOL_VERSION

        logger.info("Serving config '%s' with work_id %s", scenario_config.get("name", "?"), result.work_id[:16])

        # Return immediately so the axon can send the response; retire/replenish runs after (non-blocking).
        threading.Thread(
            target=self._retire_and_replenish,
            args=(config_path,),
            daemon=True,
            name="retire-replenish",
        ).start()

        return synapse

    def _retire_and_replenish(self, used_path: Path | None) -> None:
        """Delete the served config file (if any) and generate/store a replacement."""
        if used_path is not None:
            try:
                used_path.unlink(missing_ok=True)
                logger.info("Retired served config: %s", used_path.name)
            except OSError as e:
                logger.warning("Failed to delete served config %s: %s", used_path.name, e)
            finally:
                self.config_store.reload()
        else:
            logger.info("Replenishing config store (no served file to retire)")
            self.config_store.reload()

        generated = self._generate_new_config_file()
        if generated is not None:
            self.config_store.reload()

    def _generate_new_config_file(self) -> Path | None:
        """Call generator script to create one new config JSON file."""
        if not self._generator_script.exists():
            logger.warning("Generator script missing, cannot replenish: %s", self._generator_script)
            return None

        out_file = Path(self.config.MINER_CONFIG_DIR) / f"autogen_{int(time.time())}.json"
        cmd = [
            sys.executable,
            str(self._generator_script),
            "--model",
            self._generator_model,
            "--attempts",
            str(self._generator_attempts),
            "--timeout",
            str(self._generator_timeout),
            "--config-dir",
            str(self.config.MINER_CONFIG_DIR),
            "--output",
            str(out_file),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(10.0, self._generator_timeout + 15.0),
                check=False,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Config generation failed (rc=%d): %s",
                    proc.returncode,
                    (proc.stderr or proc.stdout).strip()[:300],
                )
                return None
            # Validate JSON readability for early visibility.
            json.loads(out_file.read_text(encoding="utf-8"))
            logger.info("Generated replacement config: %s", out_file.name)
            return out_file
        except subprocess.TimeoutExpired:
            logger.warning("Config generation timed out")
            return None
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Generated config invalid or unreadable: %s", e)
            return None

    def blacklist(self, synapse: ScenarioConfigSynapse) -> Tuple[bool, str]:  # noqa: UP006
        caller = synapse.dendrite.hotkey
        if caller not in self.metagraph.hotkeys:
            return True, f"Hotkey {caller} not in metagraph"

        uid = self.metagraph.hotkeys.index(caller)
        # On testnet, validator_permit may not be set for low-stake validators.
        # Allow any registered hotkey to query in testlab mode.
        if not Config.TESTLAB_MODE and not self.metagraph.validator_permit[uid]:
            return True, f"UID {uid} lacks validator permit"

        return False, ""

    @staticmethod
    def _detect_external_ip() -> str:
        """Detect external IP using UDP socket trick (no actual traffic sent).

        Falls back to gethostbyname if that fails. Raises RuntimeError if
        all methods return a loopback address.
        """
        # Method 1: UDP connect to public DNS — reveals the default route IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except OSError:
            pass

        # Method 2: hostname resolution
        try:
            ip = socket.gethostbyname(socket.gethostname())
            if ip and not ip.startswith("127."):
                return ip
        except socket.gaierror:
            pass

        raise RuntimeError(
            "Could not detect a non-loopback external IP. Set AXON_EXTERNAL_IP explicitly in your environment."
        )

    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Miner running. Press Ctrl+C to exit.")
        sync_timeout = max(self.config.METAGRAPH_SYNC_INTERVAL, 60)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        while not self.should_exit:
            try:
                time.sleep(self.config.METAGRAPH_SYNC_INTERVAL)
                future = executor.submit(self.metagraph.sync, subtensor=self.subtensor)
                try:
                    future.result(timeout=sync_timeout)
                    logger.debug("Metagraph synced: %d neurons", self.metagraph.n)
                except concurrent.futures.TimeoutError:
                    logger.warning("Metagraph sync timed out after %ds — skipping this cycle", sync_timeout)
            except KeyboardInterrupt:
                break
        executor.shutdown(wait=False)

        self.stop()

    def stop(self):
        logger.info("Stopping miner...")
        self.axon.stop()

    def _signal_handler(self, signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        self.should_exit = True


def _configure_logging():
    log_format = Config.LOG_FORMAT
    if log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger

            handler = logging.StreamHandler()
            handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            logging.root.handlers = [handler]
            logging.root.setLevel(logging.INFO)
            return
        except ImportError:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main():
    _configure_logging()
    miner = Miner()
    miner.run()


if __name__ == "__main__":
    main()
