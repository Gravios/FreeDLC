#
# FreeDLC workspace layer -- portable bundle tests
#
"""Tests that model bundles are path-free and relocatable.

A trained model's pytorch_config.yaml embeds absolute paths (source project,
config location, pretrained weight_init checkpoints). Bundling must strip these
so the bundle -- and any project containing it -- can be moved anywhere on any
machine. Also covers link=symlink and snapshots="best". No torch.

Standalone: ``python tests/workspace/test_portable.py`` -> ``portable: N/N checks passed``.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import yaml

from deeplabcut import workspace as ws


def _train_dir(root: Path) -> Path:
    td = root / "train"
    td.mkdir(parents=True)
    cfg = {
        "net_type": "hrnet_w32",
        "method": "td",
        "metadata": {
            "project_path": "C:\\Users\\juenglin\\Desktop\\Deeplabcut\\Cacna1Train",
            "pose_config_path": (
                "C:\\Users\\juenglin\\Desktop\\Deeplabcut\\Cacna1Train"
                "\\dlc-models-pytorch\\pytorch_config.yaml"
            ),
            "bodyparts": ["nose", "tailbase"],
            "unique_bodyparts": [],
        },
        "train_settings": {
            "batch_size": 8,
            "weight_init": {
                "snapshot_path": "C:\\ProgramData\\anaconda3\\...\\superanimal_topviewmouse_hrnet_w32.pt",
                "detector_snapshot_path": "C:\\ProgramData\\anaconda3\\...\\superanimal_fasterrcnn.pt",
            },
        },
    }
    with (td / "pytorch_config.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh)
    for name in ("snapshot-1700.pt", "snapshot-best-1760.pt",
                 "snapshot-detector-500.pt", "snapshot-detector-best-480.pt"):
        (td / name).write_bytes(b"w")
    return td


def _all_strings(obj):
    out = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            out.append(o)
    walk(obj)
    return out


def test_pose_yaml_has_no_absolute_paths():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        b = ws.ModelBundle.from_train_dir(d / "bundle", _train_dir(d), model_id="m")
        cfg = yaml.safe_load((b.path / "pose.yaml").read_text())
        strings = _all_strings(cfg)
        assert not any("C:\\" in s or s.startswith("/") for s in strings), strings
        assert cfg["metadata"]["project_path"] == ""
        assert cfg["metadata"]["pose_config_path"] == "pose.yaml"
        assert "weight_init" not in cfg.get("train_settings", {})


def test_model_toml_has_no_absolute_paths():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        b = ws.ModelBundle.from_train_dir(d / "bundle", _train_dir(d), model_id="m",
                                          legacy={"shuffle": 21, "source": "trainset85shuffle21"})
        raw = (b.path / "model.toml").read_text()
        assert "C:\\" not in raw and "/tmp/" not in raw and "/media/" not in raw


def test_snapshots_best_only():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        b = ws.ModelBundle.from_train_dir(d / "bundle", _train_dir(d), model_id="m", snapshots="best")
        names = sorted(p.name for p in b.snapshots_dir.iterdir())
        assert names == ["detector-snapshot-detector-best-480.pt", "pose-snapshot-best-1760.pt"]
        assert b.card.default_snapshot == "pose-snapshot-best-1760.pt"
        assert b.card.default_detector_snapshot == "detector-snapshot-detector-best-480.pt"


def test_link_symlink_does_not_copy():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        b = ws.ModelBundle.from_train_dir(d / "bundle", _train_dir(d), model_id="m",
                                          link="symlink", snapshots="best")
        assert (b.snapshots_dir / "pose-snapshot-best-1760.pt").is_symlink()


def test_bundle_opens_after_relocation():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.ModelBundle.from_train_dir(d / "bundle", _train_dir(d), model_id="m", snapshots="best")
        moved = d / "relocated"
        shutil.move(str(d / "bundle"), str(moved))
        b = ws.ModelBundle.open(moved)                     # opens at the new location
        assert b.card.model_id == "m"
        assert b.pose_config_path == moved / "pose.yaml"   # path resolves to where it now lives
        assert b.snapshot_path("default").exists()


def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"portable: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
