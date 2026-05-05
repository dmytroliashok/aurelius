"""C-5: forward-compat guard.

Captured synapse + scenario fixtures from every past release live under
`fixtures/vX.Y.Z/`. Each PR's tests exercise the current validator's
schema + version-gate against every one of them, so a change that
would reject an older miner (or a new required scenario field) lights
up in CI before it ships.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurelius.common.schema import validate_scenario_config
from aurelius.common.version import PROTOCOL_VERSION, VersionResult, check_compatibility

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _discover_fixtures() -> list[tuple[str, Path, Path]]:
    """Return (version_name, synapse_path, meta_path) for every fixture
    directory under fixtures/."""
    out = []
    for version_dir in sorted(_FIXTURES_DIR.iterdir()):
        if not version_dir.is_dir():
            continue
        synapse = version_dir / "synapse.json"
        meta = version_dir / "meta.json"
        if synapse.exists() and meta.exists():
            out.append((version_dir.name, synapse, meta))
    return out


_FIXTURES = _discover_fixtures()


def test_at_least_one_fixture_exists():
    """Something under fixtures/ must exist, or the whole compat
    guarantee is a no-op. This test exists so a PR that accidentally
    deletes all fixtures doesn't produce a green suite."""
    assert len(_FIXTURES) >= 1, (
        f"No fixtures under {_FIXTURES_DIR}. Capture at least the current release."
    )


@pytest.mark.parametrize(
    "version,synapse_path,meta_path",
    _FIXTURES,
    ids=[v for v, _, _ in _FIXTURES],
)
class TestForwardCompat:
    """Per-fixture checks. pytest parameterization gives each past
    release its own line in the report so a regression points you
    straight at which version stopped loading."""

    def _load(self, synapse_path, meta_path):
        synapse = json.loads(synapse_path.read_text())
        meta = json.loads(meta_path.read_text())
        return synapse, meta

    def test_scenario_config_still_passes_schema(self, version, synapse_path, meta_path):
        """Adding a new required scenario field would break every prior-
        version submission. Forbidden — new fields must stay optional."""
        synapse, _ = self._load(synapse_path, meta_path)
        result = validate_scenario_config(synapse["scenario_config"])
        assert result.valid, (
            f"v{version} scenario_config no longer passes schema. "
            f"Error: {getattr(result, 'errors', result)}. "
            f"Likely cause: a new required field was added to the schema. "
            f"All new fields must be optional to preserve compatibility."
        )

    def test_version_gate_matches_expected(self, version, synapse_path, meta_path):
        """The (local PROTOCOL_VERSION, fixture protocol_version) pair
        must produce the documented outcome."""
        _, meta = self._load(synapse_path, meta_path)
        remote = meta["protocol_version"]
        expected = meta["expected"]
        outcome = check_compatibility(PROTOCOL_VERSION, remote)
        expected_to_enum = {
            "accept": VersionResult.ACCEPT,
            "warn": VersionResult.WARN,
            "reject": VersionResult.REJECT,
        }
        assert outcome == expected_to_enum[expected], (
            f"v{version}: check_compatibility(local={PROTOCOL_VERSION}, "
            f"remote={remote}) returned {outcome} but meta says {expected}. "
            f"Either the version gate changed or the fixture's expected "
            f"outcome needs updating for the new local version."
        )

    def test_synapse_has_required_fields(self, version, synapse_path, meta_path):
        """Every fixture must carry the fields the current Synapse marks
        as required or has default values for; missing a mandatory field
        means the capture is broken."""
        synapse, _ = self._load(synapse_path, meta_path)
        for required in ("scenario_config", "work_id", "miner_protocol_version"):
            assert required in synapse, (
                f"v{version}/synapse.json missing required field {required!r}"
            )
