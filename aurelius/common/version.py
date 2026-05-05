from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

PROTOCOL_VERSION = "1.1.0"


class VersionResult(str, Enum):
    REJECT = "reject"
    WARN = "warn"
    ACCEPT = "accept"


@dataclass(frozen=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        parts = version_str.strip().split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid semantic version: {version_str!r}")
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def check_compatibility(local: str | SemanticVersion, remote: str | SemanticVersion) -> VersionResult:
    """Check version compatibility between local and remote versions.

    Returns REJECT on major mismatch, WARN on minor mismatch, ACCEPT on patch-only or exact match.
    """
    if isinstance(local, str):
        local = SemanticVersion.parse(local)
    if isinstance(remote, str):
        remote = SemanticVersion.parse(remote)

    if local.major != remote.major:
        return VersionResult.REJECT
    if local.minor != remote.minor:
        return VersionResult.WARN
    return VersionResult.ACCEPT
