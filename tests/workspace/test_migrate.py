#
# FreeDLC workspace layer -- migration tests
#
"""Tests for deeplabcut.workspace.migrate.

Builds a synthetic legacy DeepLabCut project on disk (config.yaml, videos/,
dlc-models-pytorch/) and migrates it, asserting the resulting workspace. No
torch/pyarrow/h5py needed; PyYAML is used to write the fixtures.

Standalone: ``python tests/workspace/test_migrate.py`` -> ``migrate: N/N checks passed``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from deeplabcut import workspace as ws
from deeplabcut.workspace import migrate


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh)


def _make_train_dir(root: Path, folder: str, *, net="resnet_50", bodyparts=("snout", "paw"),
                    pose_snaps=("snapshot-050.pt", "snapshot-best-100.pt"),
                    detector_snaps=()) -> Path:
    train = root / "dlc-models-pytorch" / "iteration-0" / folder / "train"
    train.mkdir(parents=True, exist_ok=True)
    _write_yaml(train / "pytorch_config.yaml",
                {"net_type": net, "metadata": {"bodyparts": list(bodyparts),
                                               "unique_bodyparts": [], "individuals": ["single"]}})
    for name in list(pose_snaps) + list(detector_snaps):
        (train / name).write_bytes(b"\x00weights")
    return train


def _make_legacy_project(root: Path, *, multi=False) -> dict:
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    clip = videos / "clip1.mp4"
    clip.write_bytes(b"not a real video")
    cfg = {
        "Task": "reaching",
        "scorer": "gravio",
        "date": "Jul7",
        "multianimalproject": multi,
        "bodyparts": "MULTI!" if multi else ["snout", "paw", "tail"],
        "multianimalbodyparts": ["snout", "paw"] if multi else None,
        "individuals": ["m1", "m2"] if multi else None,
        "uniquebodyparts": ["corner"] if multi else [],
        "skeleton": [["snout", "paw"]],
        "video_sets": {str(clip.resolve()): {"crop": "0, 640, 0, 480"}},
        "TrainingFraction": [0.95],
        "iteration": 0,
        "snapshotindex": -1,
    }
    _write_yaml(root / "config.yaml", cfg)
    return cfg


# --------------------------------------------------------------- config mapping
def test_config_mapping_single_animal():
    cfg = {"Task": "reach", "scorer": "gravio", "multianimalproject": False,
           "bodyparts": ["a", "b"], "uniquebodyparts": [], "skeleton": [["a", "b"]]}
    pc = migrate.legacy_config_to_project_config(cfg)
    assert pc.task == "reach" and pc.bodyparts == ["a", "b"]
    assert pc.experimenters == ["gravio"] and not pc.multi_animal and pc.individuals == []
    assert pc.skeleton == [["a", "b"]]


def test_config_mapping_multi_animal():
    cfg = {"Task": "social", "scorer": "lab", "multianimalproject": True,
           "bodyparts": "MULTI!", "multianimalbodyparts": ["snout", "tail"],
           "individuals": ["m1", "m2"], "uniquebodyparts": ["nest"]}
    pc = migrate.legacy_config_to_project_config(cfg)
    assert pc.multi_animal and pc.bodyparts == ["snout", "tail"]
    assert pc.individuals == ["m1", "m2"] and pc.unique_bodyparts == ["nest"]


# ------------------------------------------------------------- model discovery
def test_discover_models_and_snapshot_ordering():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_train_dir(root, "reachingJul7-trainset95shuffle1")
        _make_train_dir(root, "reachingJul7-trainset80shuffle3",
                        pose_snaps=("snapshot-010.pt",),
                        detector_snaps=("snapshot-detector-005.pt", "snapshot-detector-best-020.pt"))
        models = migrate.discover_legacy_models(root)
        by_shuffle = {m.shuffle: m for m in models}
        assert set(by_shuffle) == {1, 3}
        m1 = by_shuffle[1]
        assert m1.train_fraction == 0.95 and m1.iteration == 0 and m1.net_type == "resnet_50"
        assert not m1.top_down and len(m1.pose_snapshots) == 2
        # best is preferred as default even though 100 > 50 here it's also highest
        assert migrate._pick_default(m1.pose_snapshots).name == "snapshot-best-100.pt"
        m3 = by_shuffle[3]
        assert m3.top_down and len(m3.detector_snapshots) == 2
        assert migrate._pick_default(m3.detector_snapshots).name == "snapshot-detector-best-020.pt"


def test_pick_default_prefers_best_over_higher_epoch():
    snaps = [Path("snapshot-best-050.pt"), Path("snapshot-200.pt")]
    assert migrate._pick_default(snaps).name == "snapshot-best-050.pt"


# ----------------------------------------------------------------- video paths
def test_legacy_video_paths_resolution():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        cfg = _make_legacy_project(root)
        paths = migrate.legacy_video_paths(cfg, root)
        assert len(paths) == 1 and paths[0].name == "clip1.mp4"


# --------------------------------------------------------- full migration (E2E)
def test_migrate_project_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        legacy = Path(d) / "legacy"
        _make_legacy_project(legacy)
        _make_train_dir(legacy, "reachingJul7-trainset95shuffle1")
        _make_train_dir(legacy, "reachingJul7-trainset95shuffle2",
                        pose_snaps=("snapshot-050.pt", "snapshot-best-100.pt"),
                        detector_snaps=("snapshot-detector-best-030.pt",))

        proj = migrate.migrate_project(legacy, Path(d) / "ws")

        # project.toml
        assert proj.config.task == "reaching"
        assert proj.config.experimenters == ["gravio"]  # scorer became metadata, not a path
        assert proj.config.bodyparts == ["snout", "paw", "tail"]
        assert "gravio" not in str(proj.root)

        # video registered from video_sets
        assert proj.videos() == ["clip1"]
        assert proj.video_record("clip1").source_path.endswith("clip1.mp4")

        # two portable model bundles, each with legacy provenance in its card
        model_ids = proj.models()
        assert len(model_ids) == 2
        shuffles = set()
        top_down_seen = False
        for mid in model_ids:
            b = ws.ModelBundle.from_project(proj, mid)
            assert b.card.architecture == "resnet_50" and b.card.bodyparts == ["snout", "paw"]
            assert b.card.legacy["train_fraction"] == 0.95
            shuffles.add(b.card.legacy["shuffle"])
            # default snapshot is the 'best' one; the other is preserved too
            assert "best" in b.card.default_snapshot
            assert (b.snapshots_dir / "pose-snapshot-050.pt").exists()
            if b.card.top_down:
                top_down_seen = True
                assert b.detector_snapshot_path().exists()
        assert shuffles == {1, 2} and top_down_seen


# ------------------------------------------------------------------ smoke runner
def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"migrate: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
