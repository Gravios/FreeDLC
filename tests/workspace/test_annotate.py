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


def _make_video(path: Path, *, n_frames: int = 50, size: int = 32) -> Path:
    """Write a tiny mp4 whose frames change over time (so kmeans has something to cluster)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(n_frames):
        frame = np.full((size, size, 3), (i * 5) % 255, dtype=np.uint8)
        frame[: size // 2, : size // 2] = (255 - (i * 5) % 255)
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


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
    print(f"annotate: {passed}/{passed} checks passed")
