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
from deeplabcut.workspace.schema import LabelsRecord, ProjectConfig, VideoRecord


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


def test_labels_to_coco_gives_every_annotation_a_usable_bbox():
    """Regression: DeepLabCut's COCOLoader throws away annotations with an empty bbox.

    The export used to write ``"bbox": []``, so every label was discarded and the
    trainer learned from blank targets. The loader's own filter is reproduced here
    (``np.all(keypoints <= 0) or len(bbox) == 0``) so the test fails if it would bite.
    """
    df = pd.DataFrame([   # both keypoints on one row of pixels: bare extents would be 0 high
        {"image": "i1", "individual": "single", "bodypart": "snout", "x": 100.0, "y": 50.0},
        {"image": "i1", "individual": "single", "bodypart": "paw", "x": 140.0, "y": 50.0},
    ])
    coco = coco_export.labels_to_coco({"v1": df}, ["snout", "paw"], image_dims={"v1/i1": (192, 108)})
    (a,) = coco["annotations"]
    assert not (all(v <= 0 for v in a["keypoints"]) or len(a["bbox"]) == 0)     # survives the loader
    x, y, w, h = a["bbox"]
    assert w > 0 and h > 0 and a["area"] == w * h
    assert x <= 100.0 and x + w >= 140.0 and y <= 50.0 <= y + h                 # encloses the keypoints
    assert x >= 0 and y >= 0 and x + w <= 192 and y + h <= 108                   # clipped to the image
    # without known image size nothing is clipped, but the box is still valid
    (b,) = coco_export.labels_to_coco({"v1": df}, ["snout", "paw"])["annotations"]
    assert b["bbox"] == [80.0, 30.0, 80.0, 40.0]


def test_labels_to_coco_drops_what_was_never_labeled():
    """An extracted-but-unlabeled frame must not become a 'nothing here' training example."""
    nan = float("nan")
    df = pd.DataFrame([
        {"image": "labeled", "individual": "m1", "bodypart": "snout", "x": 5.0, "y": 6.0},
        {"image": "labeled", "individual": "m2", "bodypart": "snout", "x": nan, "y": nan},
        {"image": "blank", "individual": "m1", "bodypart": "snout", "x": nan, "y": nan},
        {"image": "blank", "individual": "m2", "bodypart": "snout", "x": nan, "y": nan},
    ])
    coco = coco_export.labels_to_coco({"v1": df}, ["snout"])
    assert [im["file_name"] for im in coco["images"]] == ["v1/labeled"]
    assert len(coco["annotations"]) == 1 and coco["annotations"][0]["num_keypoints"] == 1
    assert coco["annotations"][0]["image_id"] == coco["images"][0]["id"]


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


def test_export_leaves_out_labels_without_a_readable_frame():
    # a labeled frame that is missing or a dead link must not reach the dataset: the
    # loader would drop it after the split, shrinking train/test behind our back.
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        frames = proj.layout.frames_dir("v1")
        frames.mkdir(parents=True)
        (frames / "i1.png").write_bytes(b"px")
        (frames / "i2.png").symlink_to(frames / "i2.png")      # a link to itself
        (frames / "unlabeled.png").write_bytes(b"px")
        df = _labels(["i1.png", "i2.png", "i3.png"], ["single"], ["snout", "paw"])

        train_json, _ = coco_export.export_coco_dataset(
            proj, Path(d) / "dataset", video_ids=["v1"],
            train_fraction=1.0, seed=0, labels_provider=lambda p, v: df,
        )
        coco = json.loads(train_json.read_text())
        assert [im["file_name"] for im in coco["images"]] == ["v1/i1.png"]
        assert len(coco["annotations"]) == 1
        staged = sorted(p.name for p in (Path(d) / "dataset" / "images" / "v1").iterdir())
        assert staged == ["i1.png"]                              # only labeled, readable frames

        (frames / "i1.png").unlink()
        try:
            coco_export.export_coco_dataset(
                proj, Path(d) / "dataset2", video_ids=["v1"], labels_provider=lambda p, v: df,
            )
        except ValueError as err:
            assert "nothing to train on" in str(err)
        else:
            raise AssertionError("expected ValueError")


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


# ------------------------------------------------------ choosing the frame set
def _paired_project(root: Path, *, space: str | None):
    """A 1920x1080 / 192x108 pair with one frame per set and, optionally, a labels record."""
    proj = ws.Project.create(root / "ws", task="reach", bodyparts=["snout", "paw"])
    _register_dims(proj, "v1", "original", 1920, 1080)
    _register_dims(proj, "v1", "processed", 192, 108)
    for kind, tag in (("original", b"ORIG"), ("processed", b"PROC")):
        fdir = proj.layout.frames_dir("v1", kind)
        fdir.mkdir(parents=True)
        (fdir / "img0004.png").write_bytes(tag)
    if space is not None:
        scale = (0.1, 0.1) if space == "processed" else (1.0, 1.0)
        rec = LabelsRecord(video_id="v1", space=space, scale_x=scale[0], scale_y=scale[1])
        write_manifest(proj.layout.labels_toml("v1"), rec.to_dict())
    return proj


def _export(proj, dest: Path, df, frames):
    train_json, _ = coco_export.export_coco_dataset(
        proj, dest, video_ids=["v1"], train_fraction=1.0, seed=0,
        labels_provider=lambda p, v: df, frames=frames,
    )
    coco = json.loads(train_json.read_text())
    staged = dest / "images" / "v1" / "img0004.png"
    return staged.read_bytes(), coco["annotations"][0]["keypoints"][:2]


def test_export_frames_converts_labels_into_the_chosen_set():
    with tempfile.TemporaryDirectory() as d:
        # labels stored in processed space (annotate scaled them): 96, 54 on a 192x108 frame
        proj = _paired_project(Path(d), space="processed")
        df = _labels(["img0004.png"], ["single"], ["snout", "paw"]).assign(x=96.0, y=54.0)
        assert _export(proj, Path(d) / "a", df, "processed") == (b"PROC", [96.0, 54.0])
        image, (x, y) = _export(proj, Path(d) / "b", df, "original")
        assert image == b"ORIG" and abs(x - 960.0) < 1e-6 and abs(y - 540.0) < 1e-6
        assert _export(proj, Path(d) / "c", df, None) == (b"PROC", [96.0, 54.0])   # as stored


def test_export_frames_trusts_the_record_not_the_current_pairing():
    # labels ingested in ORIGINAL space, the processed video registered afterwards:
    # the record says original, so "processed" scales them down instead of assuming
    # they already are -- and the stored-space default stays on the original frames.
    with tempfile.TemporaryDirectory() as d:
        proj = _paired_project(Path(d), space="original")
        assert proj.label_frames_kind("v1") == "original"
        df = _labels(["img0004.png"], ["single"], ["snout", "paw"]).assign(x=960.0, y=540.0)
        image, (x, y) = _export(proj, Path(d) / "a", df, "processed")
        assert image == b"PROC" and abs(x - 96.0) < 1e-6 and abs(y - 54.0) < 1e-6
        assert _export(proj, Path(d) / "b", df, None) == (b"ORIG", [960.0, 540.0])


def test_export_file_report_names_labels_frames_and_dataset():
    import contextlib
    import io

    from deeplabcut.workspace import cli

    with tempfile.TemporaryDirectory() as d:
        proj = _paired_project(Path(d), space="processed")
        df = _labels(["img0004.png"], ["single"], ["snout", "paw"]).assign(x=96.0, y=54.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), cli._file_report(True):
            _export(proj, Path(d) / "ds", df, "original")
        out = buf.getvalue()
        lay = proj.layout
        for needed in (
            "video v1: 1 labeled frame(s)",
            f"{lay.labels_parquet('v1')} (stored in processed pixels, coordinates x10 y10)",
            str(lay.frames_dir("v1", "original")),
            f"{Path(d) / 'ds' / 'annotations' / 'train.json'} (1 image(s))",
            str(Path(d) / "ds" / "images"),
        ):
            assert needed in out, f"{needed!r} missing from:\n{out}"


def test_processed_space_labels_follow_a_replaced_processed_video():
    # labels stored for a 192x108 processed video (scale 0.1); the processed video is
    # then re-registered at 640x360. "processed" means the video as it is now, so the
    # stored coordinates are rescaled through original pixels instead of reused as-is.
    with tempfile.TemporaryDirectory() as d:
        proj = _paired_project(Path(d), space="processed")
        assert proj.labels_scale_to("v1", "processed") == (1.0, 1.0)      # same size: exact identity
        _register_dims(proj, "v1", "processed", 640, 360)
        fx, fy = proj.labels_scale_to("v1", "processed")
        assert abs(fx - 640 / 192) < 1e-9 and abs(fy - 360 / 108) < 1e-9
        fx, fy = proj.labels_scale_to("v1", "original")                   # unaffected by the swap
        assert abs(fx - 10.0) < 1e-9 and abs(fy - 10.0) < 1e-9


def test_labels_without_a_record_fall_back_to_the_pairing():
    with tempfile.TemporaryDirectory() as d:
        proj = _paired_project(Path(d), space=None)
        rec = proj.labels_record("v1")
        assert rec.space == "processed" and abs(rec.scale_x - 0.1) < 1e-9
        fx, fy = proj.labels_scale_to("v1", "original")
        assert abs(fx - 10.0) < 1e-9 and abs(fy - 10.0) < 1e-9


def test_check_frames_names_every_video_that_cannot_be_used():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        for vid in ("a", "b"):
            _register_dims(proj, vid, "original", 1920, 1080)
        proj.check_frames(["a", "b"], "original")
        proj.check_frames(["a", "b"], None)
        try:
            proj.check_frames(["a", "b"], "processed")
        except ValueError as err:
            msg = str(err)
            assert "2 video(s)" in msg and "a: no processed" in msg and "b: no processed" in msg
            assert "--frames original" in msg
            assert "add-video <project> <video> --processed --video-id <id>" in msg   # real syntax
        else:
            raise AssertionError("expected ValueError")
        try:
            proj.labels_scale_to("a", "thumbnails")
        except ValueError as err:
            assert "frames must be one of" in str(err)
        else:
            raise AssertionError("expected ValueError")


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
