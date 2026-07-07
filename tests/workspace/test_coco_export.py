#
# FreeDLC workspace layer -- COCO export tests
#
"""Tests for deeplabcut.workspace.coco_export (the pure native-training pieces).

Covers the tidy-long -> COCO conversion, the train/test split, JSON writing, and
the workspace-project -> DeepLabCut-project-dict mapping. The torch training
driver (native_train) is only checked for lazy imports, elsewhere.

Standalone: ``python tests/workspace/test_coco_export.py`` -> ``coco: N/N checks passed``.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from deeplabcut import workspace as ws
from deeplabcut.workspace import coco_export
from deeplabcut.workspace.schema import ProjectConfig


def _labels(images, individuals, bodyparts, fill=1.0):
    rows = []
    for img in images:
        for ind in individuals:
            for bpt in bodyparts:
                rows.append({"image": img, "individual": ind, "bodypart": bpt, "x": fill, "y": fill})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- project mapping
def test_project_dict_single_animal():
    c = ProjectConfig(task="reach", bodyparts=["snout", "paw"], experimenters=["gravio"])
    d = coco_export.workspace_to_dlc_project_dict(c)
    assert d["bodyparts"] == ["snout", "paw"] and not d["multianimalproject"]
    assert d["scorer"] == "gravio" and "multianimalbodyparts" not in d


def test_project_dict_multi_animal():
    c = ProjectConfig(task="social", bodyparts=["snout", "tail"], multi_animal=True,
                      individuals=["m1", "m2"], unique_bodyparts=["nest"])
    d = coco_export.workspace_to_dlc_project_dict(c)
    assert d["multianimalproject"] and d["bodyparts"] == "MULTI!"
    assert d["multianimalbodyparts"] == ["snout", "tail"] and d["individuals"] == ["m1", "m2"]


# --------------------------------------------------------------- labels->coco
def test_labels_to_coco_shape_and_keypoints():
    df = _labels(["i1", "i2"], ["single"], ["snout", "paw"])
    coco = coco_export.labels_to_coco({"v1": df}, ["snout", "paw"],
                                      image_dims={"v1/i1": (640, 480), "v1/i2": (640, 480)})
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 2
    assert coco["images"][0]["file_name"] == "v1/i1" and coco["images"][0]["width"] == 640
    assert coco["categories"][0]["keypoints"] == ["snout", "paw"]
    a = coco["annotations"][0]
    assert a["keypoints"] == [1.0, 1.0, 2, 1.0, 1.0, 2] and a["num_keypoints"] == 2


def test_labels_to_coco_unlabeled_visibility():
    df = pd.DataFrame([
        {"image": "i1", "individual": "single", "bodypart": "snout", "x": 5.0, "y": 6.0},
        {"image": "i1", "individual": "single", "bodypart": "paw", "x": float("nan"), "y": float("nan")},
    ])
    coco = coco_export.labels_to_coco({"v1": df}, ["snout", "paw"])
    a = coco["annotations"][0]
    assert a["keypoints"] == [5.0, 6.0, 2, 0.0, 0.0, 0] and a["num_keypoints"] == 1


def test_labels_to_coco_multi_individual():
    df = _labels(["i1"], ["m1", "m2"], ["snout"])
    coco = coco_export.labels_to_coco({"v1": df}, ["snout"])
    assert len(coco["images"]) == 1 and len(coco["annotations"]) == 2


# ---------------------------------------------------------------------- split
def test_split_coco_is_deterministic_and_partitions():
    df = _labels([f"i{i}" for i in range(10)], ["single"], ["snout"])
    coco = coco_export.labels_to_coco({"v1": df}, ["snout"])
    train, test = coco_export.split_coco(coco, train_fraction=0.8, seed=0)
    assert len(train["images"]) == 8 and len(test["images"]) == 2
    train_ids = {im["id"] for im in train["images"]}
    test_ids = {im["id"] for im in test["images"]}
    assert train_ids.isdisjoint(test_ids)                       # partition
    # annotations follow their image
    assert all(a["image_id"] in train_ids for a in train["annotations"])
    # deterministic
    train2, _ = coco_export.split_coco(coco, train_fraction=0.8, seed=0)
    assert [im["id"] for im in train2["images"]] == [im["id"] for im in train["images"]]


def test_write_coco_json_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        coco = coco_export.labels_to_coco({"v1": _labels(["i1"], ["single"], ["snout"])}, ["snout"])
        p = coco_export.write_coco_json(coco, Path(d) / "train.json")
        assert json.loads(p.read_text())["categories"][0]["keypoints"] == ["snout"]


# ------------------------------------------------------- dataset staging (E2E)
def test_export_coco_dataset_stages_json_and_frames():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        # simulate ingested annotations for one video: frames + a labels provider
        frames = proj.layout.frames_dir("v1")
        frames.mkdir(parents=True)
        (frames / "i1.png").write_bytes(b"px")
        df = _labels(["i1.png"], ["single"], ["snout", "paw"])

        train_json, test_json = coco_export.export_coco_dataset(
            proj, Path(d) / "dataset", video_ids=["v1"],
            train_fraction=1.0, seed=0, labels_provider=lambda p, v: df,
        )
        assert train_json.exists() and test_json.exists()
        # frame materialized under dataset/images/<video_id>/
        assert (Path(d) / "dataset" / "images" / "v1" / "i1.png").is_symlink()
        assert json.loads(train_json.read_text())["images"][0]["file_name"] == "v1/i1.png"


# --------------------------------------------------------- native driver (lazy)
def test_native_train_imports_lazily():
    import ast

    src = (Path(ws.__file__).parent / "native_train.py").read_text()
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    modules = [n.module for n in top if isinstance(n, ast.ImportFrom) and n.module]
    modules += [a.name for n in top if isinstance(n, ast.Import) for a in n.names]
    assert not any(m and m.split(".")[0] in {"torch", "deeplabcut", "PIL"} for m in modules), modules


# ------------------------------------------------------------------ smoke runner
def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"coco: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
