#
# FreeDLC workspace layer
#
"""Helpers for reading a PyTorch training directory.

Both migration (harvesting a legacy ``dlc-models-pytorch/.../train`` dir) and
training (harvesting the dir a training run just produced) need to interpret the
same artifacts: the ``pytorch_config.yaml`` and the ``snapshot-*.pt`` /
``snapshot-detector-*.pt`` checkpoints. That shared logic lives here so it exists
in exactly one place.

Snapshot filename convention (from the pose-estimation runners):
``snapshot-<epoch:03>.pt``, ``snapshot-best-<epoch:03>.pt``, and the detector
equivalents prefixed ``snapshot-detector-``.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "POSE_PREFIX",
    "DETECTOR_PREFIX",
    "list_snapshots",
    "pick_default_snapshot",
    "is_best_snapshot",
    "read_net_type",
    "read_bodyparts",
]

POSE_PREFIX = "snapshot"
DETECTOR_PREFIX = "snapshot-detector"

# "snapshot-050" / "snapshot-best-050" / "snapshot-detector-best-020" -> (best?, epoch)
_SNAPSHOT_UID_RE = re.compile(r"-(?:(best)-)?(\d+)$")


def _epoch(path: Path) -> int:
    m = _SNAPSHOT_UID_RE.search(path.stem)
    return int(m.group(2)) if m else -1


def is_best_snapshot(path: Path) -> bool:
    m = _SNAPSHOT_UID_RE.search(path.stem)
    return bool(m and m.group(1))


def list_snapshots(train_dir: str | Path, *, detector: bool) -> list[Path]:
    """List pose (or detector) snapshots in ``train_dir``, sorted by epoch."""
    train_dir = Path(train_dir)
    files = []
    for p in train_dir.glob("snapshot-*.pt"):
        is_detector = p.name.startswith(DETECTOR_PREFIX + "-")
        if is_detector == detector:
            files.append(p)
    return sorted(files, key=_epoch)


def pick_default_snapshot(snapshots: list[Path]) -> Path:
    """The default snapshot: the best-performing one if any, else the highest epoch."""
    bests = [s for s in snapshots if is_best_snapshot(s)]
    return (bests or snapshots)[-1]


def read_net_type(pose_config: str | Path) -> str | None:
    """Read the network architecture from a ``pytorch_config.yaml``."""
    import yaml

    with Path(pose_config).open() as fh:
        cfg = yaml.safe_load(fh) or {}
    for key in ("net_type", "default_net_type"):
        if cfg.get(key):
            return str(cfg[key])
    backbone = (cfg.get("model") or {}).get("backbone") or {}
    return backbone.get("type")


def read_bodyparts(pose_config: str | Path) -> list[str]:
    """Read the ordered bodypart names from a ``pytorch_config.yaml`` metadata block."""
    import yaml

    with Path(pose_config).open() as fh:
        cfg = yaml.safe_load(fh) or {}
    return list((cfg.get("metadata") or {}).get("bodyparts") or [])
