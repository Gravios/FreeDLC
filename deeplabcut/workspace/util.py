#
# FreeDLC workspace layer
#
"""Small shared utilities for the workspace layer (no heavy dependencies)."""
from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["code_version", "sha256_file"]


def code_version() -> str | None:
    """Best-effort version string for provenance.

    Tries the installed package metadata, then a ``deeplabcut.__version__`` /
    ``VERSION`` attribute, and returns ``None`` if neither is available (e.g. an
    editable checkout that is not installed). Kept dependency-free and lazy so
    importing the workspace layer never pulls in the full ``deeplabcut`` package.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("deeplabcut")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    return None


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, hex-encoded (constant memory)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
