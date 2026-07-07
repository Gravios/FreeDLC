#
# FreeDLC workspace layer
#
"""Stable identifiers for workspace entities.

Runs and models are identified by an opaque, sortable, collision-resistant id of
the form ``YYYYMMDD-HHMMSS-<6hex>`` (e.g. ``20260707-141530-a1b9c2``). The
timestamp makes ids sortable and human-legible; the 24-bit random suffix makes
two ids generated in the same second distinct.

Design rule: an id encodes *when it was minted*, nothing else. It never encodes
the shuffle, training fraction, network, experimenter, or any other coordinate --
those live in the entity's manifest. This is the deliberate inverse of the legacy
``<Task><date>-trainset95shuffle1`` naming, where coordinates were smeared across
directory and file names and had to be parsed back out.
"""
from __future__ import annotations

import re
import secrets
from datetime import datetime

__all__ = [
    "ID_RE",
    "new_id",
    "new_run_id",
    "new_model_id",
    "is_id",
    "slugify",
    "video_id_from_path",
]

#: Canonical id pattern: 8-digit date, 6-digit time, 6 lowercase hex.
ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def new_id(now: datetime | None = None) -> str:
    """Mint a new ``YYYYMMDD-HHMMSS-<6hex>`` id.

    Args:
        now: timestamp to use; defaults to the current local time. Passing an
            explicit value makes id generation deterministic in tests.
    """
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


#: Runs and models share the id format; separate names document intent at call sites.
new_run_id = new_id
new_model_id = new_id


def is_id(value: str) -> bool:
    """True if ``value`` is a well-formed workspace id."""
    return bool(ID_RE.match(value))


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_SLUG_EDGES = re.compile(r"^[-_]+|[-_]+$")


def slugify(name: str) -> str:
    """Reduce an arbitrary name to a filesystem-safe slug.

    Lowercases, replaces any run of non-alphanumeric characters with a single
    hyphen, and trims leading/trailing separators. Used to derive human-readable
    (but still stable) ids for sources such as videos.

    Raises:
        ValueError: if the name reduces to the empty string.
    """
    slug = _SLUG_EDGES.sub("", _SLUG_STRIP.sub("-", name.strip().lower()))
    if not slug:
        raise ValueError(f"name {name!r} does not contain any slug-able characters")
    return slug


def video_id_from_path(path) -> str:
    """Derive a stable video id from a video file path (its slugified stem)."""
    from pathlib import Path

    return slugify(Path(path).stem)
