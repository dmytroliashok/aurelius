import pytest

from aurelius.common.version import PROTOCOL_VERSION, SemanticVersion, VersionResult, check_compatibility


class TestSemanticVersion:
    def test_parse_valid(self):
        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_zero(self):
        v = SemanticVersion.parse("0.0.0")
        assert v == SemanticVersion(0, 0, 0)

    def test_str_roundtrip(self):
        assert str(SemanticVersion.parse("1.2.3")) == "1.2.3"

    def test_parse_invalid_two_parts(self):
        with pytest.raises(ValueError, match="Invalid semantic version"):
            SemanticVersion.parse("1.2")

    def test_parse_invalid_non_numeric(self):
        with pytest.raises(ValueError):
            SemanticVersion.parse("1.2.x")

    def test_frozen(self):
        v = SemanticVersion(1, 0, 0)
        with pytest.raises(AttributeError):
            v.major = 2


class TestCheckCompatibility:
    def test_exact_match(self):
        assert check_compatibility("1.0.0", "1.0.0") == VersionResult.ACCEPT

    def test_patch_mismatch_accepts(self):
        assert check_compatibility("1.0.0", "1.0.1") == VersionResult.ACCEPT
        assert check_compatibility("1.0.3", "1.0.0") == VersionResult.ACCEPT

    def test_minor_mismatch_warns(self):
        assert check_compatibility("1.0.0", "1.1.0") == VersionResult.WARN
        assert check_compatibility("1.3.0", "1.2.0") == VersionResult.WARN

    def test_major_mismatch_rejects(self):
        assert check_compatibility("1.0.0", "2.0.0") == VersionResult.REJECT
        assert check_compatibility("2.0.0", "1.0.0") == VersionResult.REJECT

    def test_accepts_semantic_version_objects(self):
        local = SemanticVersion(1, 0, 0)
        remote = SemanticVersion(1, 1, 0)
        assert check_compatibility(local, remote) == VersionResult.WARN

    def test_mixed_string_and_object(self):
        assert check_compatibility("1.0.0", SemanticVersion(1, 0, 1)) == VersionResult.ACCEPT


class TestProtocolVersion:
    def test_protocol_version_is_valid(self):
        v = SemanticVersion.parse(PROTOCOL_VERSION)
        assert v.major >= 1
