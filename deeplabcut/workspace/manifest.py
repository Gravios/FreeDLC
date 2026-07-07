#
# FreeDLC workspace layer
#
"""Reading and writing TOML manifests.

Every generated artifact set in a workspace carries a small TOML manifest
(``project.toml``, ``model.toml``, ``run.toml``, ``video.toml``). This module is
the single choke point for manifest I/O so the rest of the code never touches a
TOML library directly.

Reading uses the standard library (:mod:`tomllib`); writing uses ``tomli_w``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli_w
import tomllib

__all__ = ["read_manifest", "write_manifest", "update_manifest"]


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Load a TOML manifest into a plain dict.

    Raises:
        FileNotFoundError: if the manifest does not exist.
        tomllib.TOMLDecodeError: if the file is not valid TOML.
    """
    path = Path(path)
    with path.open("rb") as fh:
        return tomllib.load(fh)


def write_manifest(path: str | Path, data: dict[str, Any]) -> Path:
    """Serialize ``data`` to ``path`` as TOML, creating parent directories.

    ``None`` values are dropped recursively: TOML has no null, and an absent key
    is the correct representation of "not set" for our manifests.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        tomli_w.dump(_drop_none(data), fh)
    return path


def update_manifest(path: str | Path, **changes: Any) -> dict[str, Any]:
    """Read a manifest, apply top-level ``changes``, write it back, and return it."""
    data = read_manifest(path)
    data.update(changes)
    write_manifest(path, data)
    return data


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_drop_none(v) for v in value]
    return value
