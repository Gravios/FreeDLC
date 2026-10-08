#
# FreeDLC workspace layer
#
"""Small shared utilities for the workspace layer (no heavy dependencies)."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

__all__ = ["FILES_LOGGER", "files_log", "shown", "code_version", "sha256_file", "same_file", "materialize"]

#: Name of the logger that reports which files an operation reads and writes.
FILES_LOGGER = "deeplabcut.workspace.files"

#: Silent unless asked for: ``fdlc <command> --verbose`` raises it to INFO and gives
#: it a console handler. It stays at WARNING otherwise so the lines do not ride along
#: whenever something else (training) installs a root handler at INFO.
files_log = logging.getLogger(FILES_LOGGER)
files_log.setLevel(logging.WARNING)


def shown(path: str | Path) -> str:
    """``path`` as the file report prints it: absolute, and for a symlink with the
    file it resolves to (``link -> target``), so a line always names a real location."""
    path = Path(path)
    absolute = os.path.abspath(path)
    return f"{absolute} -> {path.resolve()}" if path.is_symlink() else absolute


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


def same_file(a: str | Path, b: str | Path) -> bool:
    """True if ``a`` and ``b`` both exist and are the same file (symlinks followed).

    Missing paths, dangling links and link loops compare unequal rather than raising.
    """
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def materialize(src: str | Path, dst: str | Path, link: str = "symlink") -> Path:
    """Place ``src`` at ``dst`` as a symlink or a copy, without ever damaging ``src``.

    This is the one place the workspace creates a link or copy of a source file, so
    the hazards of replacing ``dst`` are handled once:

    * ``src`` is resolved to its real file *before* ``dst`` is touched, and a link
      always points at that real file. If ``dst`` already is that file (reached
      under another name, e.g. through a staging symlink) nothing is done --
      unlinking it first would delete the only copy and leave a link to itself.
    * an existing ``dst`` is removed, never written through: copying onto a symlink
      would otherwise overwrite whatever that link points at.

    Idempotent. Returns ``dst``.

    Raises:
        ValueError: if ``link`` is not ``"symlink"`` or ``"copy"``.
        FileNotFoundError: if ``src`` is missing, a dangling link, or a link loop.
    """
    if link not in ("symlink", "copy"):
        raise ValueError(f"link must be symlink|copy, got {link!r}")
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"source file not found (or a broken link): {src}")
    target = src.resolve()

    if same_file(target, dst) and (link == "symlink" or not dst.is_symlink()):
        return dst  # dst is the source itself, or already a link to it

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if link == "symlink":
        dst.symlink_to(target)
    else:
        shutil.copy2(target, dst)
    return dst
