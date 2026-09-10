"""Frame extraction and the annotate orchestration (napari launch stubbed).

The napari viewer needs Qt, which is not available here, so ``annotate_video`` takes an
injectable ``_launch``; these tests substitute a fake that writes a CollectedData file
the way napari would, exercising the whole extract -> stage -> ingest path around the GUI.
Frame extraction itself is real (cv2), run on a tiny synthetic video.

Run via ``tests/workspace/run_all.sh`` or directly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.workspace import Project
from deeplabcut.workspace import annotate as ann
from deeplabcut.workspace import frames as frames_mod

BODYPARTS = ["snout", "paw"]


def _make_video(path: Path, *, n_frames: int = 50, size=32) -> Path:
    """Write a tiny mp4 whose frames change over time (so kmeans has something to cluster).

    ``size`` is an int (square) or a ``(width, height)`` tuple.
    """
    import cv2

    w, h = (size, size) if isinstance(size, int) else size
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), (i * 5) % 255, dtype=np.uint8)
        frame[: h // 2, : w // 2] = (255 - (i * 5) % 255)
        writer.write(frame)
    writer.release()
    return path


def _project_with_video(root: Path, *, n_frames: int = 50):
    """Create a workspace project with one registered (copied) synthetic video."""
    proj = Project.create(root / "ws", task="reach", bodyparts=BODYPARTS)
    src = _make_video(root / "raw" / "Clip 01.mp4", n_frames=n_frames)
    vid = proj.add_video(src, link="copy")
    return proj, vid


# ------------------------------------------------------------- resolve_media
def test_resolve_media_finds_registered_video():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        media = frames_mod.resolve_media(proj, vid)
        assert media.is_file() and media.name.startswith("video")


def test_resolve_media_rejects_unregistered():
    with tempfile.TemporaryDirectory() as d:
        proj = Project.create(Path(d) / "ws", task="reach", bodyparts=BODYPARTS)
        try:
            frames_mod.resolve_media(proj, "ghost")
        except FileNotFoundError as err:
            assert "not registered" in str(err)
        else:
            raise AssertionError("expected FileNotFoundError")


# ------------------------------------------------------------ extract_frames
def test_extract_uniform_writes_n_frames():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        written = frames_mod.extract_frames(proj, vid, n=10, mode="uniform")
        assert len(written) == 10
        assert all(p.suffix == ".png" and p.name.startswith("img") for p in written)
        assert written == sorted(written)                        # names sort in frame order
        assert all(p.parent == proj.layout.frames_dir(vid) for p in written)


def test_extract_is_idempotent_without_overwrite():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        first = frames_mod.extract_frames(proj, vid, n=8, mode="uniform")
        again = frames_mod.extract_frames(proj, vid, n=999, mode="uniform")   # n ignored: frames exist
        assert again == first


def test_extract_overwrite_replaces():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        frames_mod.extract_frames(proj, vid, n=8, mode="uniform")
        redone = frames_mod.extract_frames(proj, vid, n=5, mode="uniform", overwrite=True)
        assert len(redone) == 5
        assert len(list(proj.layout.frames_dir(vid).glob("*.png"))) == 5   # stale ones gone


def test_extract_kmeans_selects_frames():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d), n_frames=60)
        written = frames_mod.extract_frames(proj, vid, n=6, mode="kmeans")
        assert 1 <= len(written) <= 6                            # clusters may merge; never more than n
        assert all(p.parent == proj.layout.frames_dir(vid) for p in written)


def test_extract_rejects_bad_mode():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        try:
            frames_mod.extract_frames(proj, vid, mode="magic")
        except ValueError as err:
            assert "mode must be" in str(err)
        else:
            raise AssertionError("expected ValueError")


def test_uniform_indices_are_spread():
    idx = frames_mod._frame_indices_uniform(100, 5)
    assert idx == sorted(idx) and len(set(idx)) == 5
    assert idx[0] > 0 and idx[-1] < 100                          # not clamped to the ends


# ------------------------------------------------------- resolve_video_id
def test_resolve_video_id_by_id_and_by_path():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        assert ann.resolve_video_id(proj, vid) == vid
        # a path whose stem slugifies to the id also resolves
        assert ann.resolve_video_id(proj, "/somewhere/Clip 01.mp4") == vid


def test_resolve_video_id_unknown():
    with tempfile.TemporaryDirectory() as d:
        proj = Project.create(Path(d) / "ws", task="reach", bodyparts=BODYPARTS)
        try:
            ann.resolve_video_id(proj, "nope")
        except FileNotFoundError as err:
            assert "no registered video" in str(err)
        else:
            raise AssertionError("expected FileNotFoundError")


# ------------------------------------------------------------- config + staging
def test_synthesized_config_has_napari_keys():
    with tempfile.TemporaryDirectory() as d:
        proj = Project.create(Path(d) / "ws", task="reach", bodyparts=BODYPARTS,
                              skeleton=[["snout", "paw"]], experimenters=["gravio"])
        cfg = ann.synthesize_config(proj, scorer="gravio")
        for key in ("Task", "scorer", "bodyparts", "multianimalbodyparts", "skeleton",
                    "dotsize", "pcutoff", "colormap", "multianimalproject", "individuals"):
            assert key in cfg
        assert cfg["scorer"] == "gravio" and cfg["bodyparts"] == BODYPARTS


def test_staging_builds_dlc_layout():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        frames = frames_mod.extract_frames(proj, vid, n=6, mode="uniform")
        config_path, dataset_dir = ann.stage_annotation_project(proj, vid, frames, scorer="gravio")
        assert config_path.name == "config.yaml"
        assert dataset_dir == config_path.parent / "labeled-data" / vid
        # frames appear as symlinks in the staging dataset dir
        staged = sorted(dataset_dir.glob("*.png"))
        assert len(staged) == 6 and all(p.is_symlink() for p in staged)
        # config parent is the DLC "project root" napari will save into
        assert (config_path.parent / "labeled-data" / vid).is_dir()


# ------------------------------ full orchestration with napari stubbed out
def _fake_napari_that_labels(config_path: Path, dataset_dir: Path):
    """Stand in for the GUI: write a CollectedData_*.h5 the way napari would on save."""
    scorer = "gravio"
    images = sorted(p.name for p in dataset_dir.glob("*.png"))
    rows = [("labeled-data", dataset_dir.name, name) for name in images]
    index = pd.MultiIndex.from_tuples(rows)
    cols = pd.MultiIndex.from_tuples(
        [(scorer, bp, c) for bp in BODYPARTS for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    data = np.arange(len(images) * len(BODYPARTS) * 2, dtype=float).reshape(len(images), -1)
    df = pd.DataFrame(data, index=index, columns=cols)
    df.to_hdf(dataset_dir / f"CollectedData_{scorer}.h5", key="df_with_missing", mode="w")


def test_annotate_video_full_loop_ingests_labels():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        returned = ann.annotate_video(proj, vid, n=8, _launch=_fake_napari_that_labels)
        assert returned == vid
        labels = proj.layout.labels_parquet(vid)
        assert labels.is_file()                                   # labels landed in the workspace
        df = pd.read_parquet(labels)
        assert set(df["bodypart"]) == set(BODYPARTS)
        assert (proj.layout.frames_dir(vid)).is_dir()


def test_annotate_video_no_labels_saved_is_graceful():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        # a launch that saves nothing (user closed without labelling)
        returned = ann.annotate_video(proj, vid, n=8, _launch=lambda c, ds: None)
        assert returned == vid
        assert not proj.layout.labels_parquet(vid).is_file()     # nothing ingested, no crash


def test_annotate_resolves_via_path_argument():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        returned = ann.annotate_video(proj, "/anywhere/Clip 01.mp4", n=6,
                                      _launch=_fake_napari_that_labels)
        assert returned == vid


# ------------------------------------------------ original/processed split + scale
def _project_with_pair(root: Path, *, orig=(160, 120), proc=(80, 30), n_frames=40):
    """Project with a mirrored original+processed video at deliberately anisotropic sizes."""
    proj = Project.create(root / "ws", task="reach", bodyparts=BODYPARTS)
    o = _make_video(root / "orig" / "Clip 01.mp4", n_frames=n_frames, size=orig)
    p = _make_video(root / "proc" / "Clip 01.mp4", n_frames=n_frames, size=proc)
    proj.add_video(o, link="copy")                    # -> original (default)
    proj.add_video(p, kind="processed", link="copy")
    return proj, "clip-01"


def test_add_video_probes_dimensions():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        rec = proj.video_record(vid, "original")
        assert rec.width and rec.height                   # populated by the cv2 probe


def test_original_and_processed_are_separate_shelves():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_pair(Path(d))
        assert proj.videos("original") == [vid]
        assert proj.videos("processed") == [vid]
        assert proj.has_video(vid, "original") and proj.has_video(vid, "processed")


def test_annotation_scale_is_anisotropic():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_pair(Path(d), orig=(160, 120), proc=(80, 30))
        sx, sy = proj.annotation_scale(vid)
        assert abs(sx - 0.5) < 1e-9 and abs(sy - 0.25) < 1e-9   # 80/160, 30/120 -- x != y


def test_annotation_scale_identity_without_processed():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        assert proj.annotation_scale(vid) == (1.0, 1.0)


def test_extract_writes_both_frame_sets():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_pair(Path(d))
        written = frames_mod.extract_frames(proj, vid, n=6, mode="uniform")
        orig = sorted(proj.layout.frames_dir(vid, "original").glob("*.png"))
        proc = sorted(proj.layout.frames_dir(vid, "processed").glob("*.png"))
        assert written == orig
        assert [p.name for p in orig] == [p.name for p in proc]     # same indices in both
        # processed frames are actually smaller
        import cv2
        assert cv2.imread(str(proc[0])).shape[1] < cv2.imread(str(orig[0])).shape[1]


def test_annotate_scales_labels_into_processed_space():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_pair(Path(d), orig=(160, 120), proc=(80, 30))

        captured = {}

        def fake_launch(config_path, dataset_dir):
            # label every frame at a known ORIGINAL-space point, like napari would
            _fake_napari_at(config_path, dataset_dir, x=100.0, y=80.0)
            captured["dir"] = dataset_dir

        ann.annotate_video(proj, vid, n=5, _launch=fake_launch)
        df = pd.read_parquet(proj.layout.labels_parquet(vid))
        # 100*0.5 = 50, 80*0.25 = 20  -- anisotropic scale applied correctly
        assert (df["x"].dropna().round(6) == 50.0).all()
        assert (df["y"].dropna().round(6) == 20.0).all()


def _fake_napari_at(config_path: Path, dataset_dir: Path, *, x: float, y: float):
    """Like _fake_napari_that_labels but places every marker at a fixed (x, y)."""
    scorer = "labeler"
    images = sorted(p.name for p in dataset_dir.glob("*.png"))
    index = pd.MultiIndex.from_tuples([("labeled-data", dataset_dir.name, n) for n in images])
    cols = pd.MultiIndex.from_tuples(
        [(scorer, bp, c) for bp in BODYPARTS for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    row = []
    for _bp in BODYPARTS:
        row += [x, y]
    df = pd.DataFrame([row] * len(images), index=index, columns=cols)
    df.to_hdf(dataset_dir / f"CollectedData_{scorer}.h5", key="df_with_missing", mode="w")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
    print(f"annotate: {passed}/{passed} checks passed")
