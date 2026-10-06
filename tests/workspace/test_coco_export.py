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
from deeplabcut.workspace.manifest import write_manifest
from deeplabcut.workspace.schema import ProjectConfig, VideoRecord


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
        # the JSONs sit where COCOLoader looks: <dataset>/annotations/<name>
        assert train_json == Path(d) / "dataset" / "annotations" / "train.json"
        assert test_json == Path(d) / "dataset" / "annotations" / "test.json"
        # frame materialized under dataset/images/<video_id>/
        assert (Path(d) / "dataset" / "images" / "v1" / "i1.png").is_symlink()
        assert json.loads(train_json.read_text())["images"][0]["file_name"] == "v1/i1.png"


def _register_dims(proj, vid, kind, w, h):
    """Write a video.toml with known dimensions, so annotation_scale can be derived."""
    rec = VideoRecord(video_id=vid, source_path=f"{kind}.mp4", width=w, height=h, link="reference")
    write_manifest(proj.layout.video_toml(vid, kind), rec.to_dict())


def test_export_uses_processed_frames_when_scaled():
    # labels.parquet is in processed space (annotate scales into it), so export must
    # materialize the processed frames -- not the original ones they were drawn on.
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        _register_dims(proj, "v1", "original", 1920, 1080)
        _register_dims(proj, "v1", "processed", 192, 108)
        assert proj.annotation_scale("v1") != (1.0, 1.0)   # a real scale exists

        for kind, tag in (("original", b"ORIG-1920x1080"), ("processed", b"PROC-192x108")):
            fdir = proj.layout.frames_dir("v1", kind)
            fdir.mkdir(parents=True)
            (fdir / "img0004.png").write_bytes(tag)

        df = _labels(["img0004.png"], ["single"], ["snout", "paw"])
        coco_export.export_coco_dataset(
            proj, Path(d) / "dataset", video_ids=["v1"],
            train_fraction=1.0, seed=0, link="symlink", labels_provider=lambda p, v: df,
        )
        staged = Path(d) / "dataset" / "images" / "v1" / "img0004.png"
        # provenance: the staged frame resolves into frames/processed, the low-res set
        assert staged.resolve() == proj.layout.frames_dir("v1", "processed").resolve() / "img0004.png"
        assert staged.read_bytes() == b"PROC-192x108"


def test_export_uses_original_frames_without_processed():
    # no processed counterpart -> labels stay in original space -> original frames.
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        _register_dims(proj, "v1", "original", 1920, 1080)
        assert proj.annotation_scale("v1") == (1.0, 1.0)

        fdir = proj.layout.frames_dir("v1", "original")
        fdir.mkdir(parents=True)
        (fdir / "img0004.png").write_bytes(b"ORIG")

        df = _labels(["img0004.png"], ["single"], ["snout", "paw"])
        coco_export.export_coco_dataset(
            proj, Path(d) / "dataset", video_ids=["v1"],
            train_fraction=1.0, seed=0, link="symlink", labels_provider=lambda p, v: df,
        )
        staged = Path(d) / "dataset" / "images" / "v1" / "img0004.png"
        assert staged.resolve() == proj.layout.frames_dir("v1", "original").resolve() / "img0004.png"


def test_image_sizes_and_evaluation_read_the_label_space_frames():
    # the same rule as the export: anything that reads frames by label must take the
    # set the labels are in, or sizes/predictions describe images of another resolution.
    from PIL import Image

    from deeplabcut.workspace import evaluate, native_train

    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        _register_dims(proj, "v1", "original", 1920, 1080)
        _register_dims(proj, "v1", "processed", 192, 108)
        for kind, size in (("original", (1920, 1080)), ("processed", (192, 108))):
            fdir = proj.layout.frames_dir("v1", kind)
            fdir.mkdir(parents=True)
            Image.new("RGB", size).save(fdir / "img0004.png")

        assert native_train.probe_image_dims(proj, ["v1"]) == {"v1/img0004.png": (192, 108)}

        df = _labels(["img0004.png"], ["single"], ["snout", "paw"])
        seen = []

        class _Card:
            model_id = "m"

        class _Bundle:
            card = _Card()

        real = evaluate.infer_on_frames
        evaluate.infer_on_frames = lambda bundle, frames_dir, images: seen.append(Path(frames_dir)) or df
        try:
            evaluate.evaluate_model(proj, _Bundle(), videos=["v1"], labels_provider=lambda p, v: df,
                                    write=False)
        finally:
            evaluate.infer_on_frames = real
        assert seen == [proj.layout.frames_dir("v1", "processed")]


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
