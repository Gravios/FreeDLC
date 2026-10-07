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
        assert proj.labels_record(vid).space == "original"       # no processed video: unscaled
        assert proj.layout.labels_toml(vid).is_file()


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


def test_annotate_stores_labels_in_the_pixels_they_were_drawn_in():
    """A processed counterpart does not change what annotate stores; training converts."""
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_pair(Path(d), orig=(160, 120), proc=(80, 30))

        def fake_launch(config_path, dataset_dir):
            # label every frame at a known ORIGINAL-space point, like napari would
            _fake_napari_at(config_path, dataset_dir, x=100.0, y=80.0)

        ann.annotate_video(proj, vid, n=5, _launch=fake_launch)
        df = pd.read_parquet(proj.layout.labels_parquet(vid))
        assert (df["x"].dropna().round(6) == 100.0).all()          # as placed, not scaled
        assert (df["y"].dropna().round(6) == 80.0).all()
        rec = proj.labels_record(vid)
        assert proj.layout.labels_toml(vid).is_file()
        assert (rec.space, rec.scale_x, rec.scale_y) == ("original", 1.0, 1.0)
        # the anisotropic scale is applied when the processed frames are asked for
        assert proj.labels_scale_to(vid, "processed") == (0.5, 0.25)   # 80/160, 30/120 -- x != y
        assert proj.labels_scale_to(vid, "original") == (1.0, 1.0)
        assert proj.label_frames_kind(vid) == "original"


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


# ---------------------------------------------- extract-frames --all (CLI)
def _run(argv):
    import contextlib
    import io

    from deeplabcut.workspace import cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _project_with_n_videos(root: Path, ids, *, size=(160, 120)):
    proj = Project.create(root / "ws", task="reach", bodyparts=BODYPARTS)
    for name in ids:
        proj.add_video(_make_video(root / "raw" / f"{name}.mp4", n_frames=30, size=size), link="copy")
    return proj


def test_extract_all_covers_every_video():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _project_with_n_videos(d, ["a", "b", "c"])
        code, out = _run(["extract-frames", "--all", "--project", str(d / "ws"), "-n", "5"])
        assert code == 0, out
        assert "3/3 video(s)" in out
        for v in ("a", "b", "c"):
            assert len(list(proj.layout.frames_dir(v, "original").glob("*.png"))) == 5


def test_extract_all_and_video_are_mutually_exclusive():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project_with_n_videos(d, ["a"])
        code, out = _run(["extract-frames", "a", "--all", "--project", str(d / "ws")])
        assert code == 2 and "not both" in out


def test_extract_requires_video_or_all():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project_with_n_videos(d, ["a"])
        code, out = _run(["extract-frames", "--project", str(d / "ws")])
        assert code == 2 and "or --all" in out


def test_extract_all_on_empty_project():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        Project.create(Path(d) / "ws", task="reach", bodyparts=BODYPARTS)
        code, out = _run(["extract-frames", "--all", "--project", str(Path(d) / "ws")])
        assert code == 2 and "no registered videos" in out


def test_extract_single_video_still_works():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project_with_n_videos(d, ["solo"])
        code, out = _run(["extract-frames", "solo", "--project", str(d / "ws"), "-n", "4"])
        assert code == 0 and "solo: extracted 4 frame(s)" in out


# ------------------------------------------- kmeans stride + parallel --jobs
def test_kmeans_stride_decodes_a_sample_not_every_frame():
    # a 300-frame clip; with a stride the sampled set is far smaller than 300
    idx_all = []
    idx_strided = []

    class _Cap:
        def __init__(self, sink):
            self.sink = sink
            self.pos = 0

        def set(self, prop, v):
            self.pos = int(v)
            self.sink.append(self.pos)

        def read(self):
            import numpy as np
            return True, np.zeros((8, 8, 3), np.uint8)

    frames_mod._frame_indices_kmeans(_Cap(idx_all), 300, 5, sample_stride=1, fps=30)
    frames_mod._frame_indices_kmeans(_Cap(idx_strided), 300, 5, sample_stride=30, fps=30)
    assert len(idx_all) == 300               # stride 1 touches every frame
    assert len(idx_strided) == 10            # stride 30 touches ~1/30
    assert len(idx_strided) < len(idx_all)


def test_kmeans_default_stride_is_about_one_per_second():
    # 600 frames at 30 fps -> default stride ~30 -> ~20 sampled, not 600
    touched = []

    class _Cap:
        def set(self, prop, v):
            touched.append(int(v))

        def read(self):
            import numpy as np
            return True, np.zeros((8, 8, 3), np.uint8)

    frames_mod._frame_indices_kmeans(_Cap(), 600, 5, sample_stride=None, fps=30)
    assert 10 <= len(touched) <= 60          # a per-second-ish sample, nowhere near 600


def test_kmeans_is_temporally_spread_despite_a_burst():
    # First 7.5% of the clip is a visually extreme "handling" burst; the rest is a
    # near-static scene. Global appearance k-means bunched picks into the burst
    # (adjacent-in-time frames); temporal stratification must spread them instead.
    total, burst_end, n = 1200, 90, 20

    class _BurstCap:
        def set(self, prop, v):
            self._i = int(v)

        def read(self):
            i = self._i
            if i < burst_end:                          # big fast-moving bright bar -> outliers
                f = np.full((24, 24, 3), 240, np.uint8)
                x = (i * 24) // burst_end
                f[:, max(0, x - 3):x + 3] = 10
                return True, f
            f = np.full((24, 24, 3), 90, np.uint8)     # near-static scene
            f[10:14, 10:14] = 200
            return True, f

    idx = frames_mod._frame_indices_kmeans(_BurstCap(), total, n, sample_stride=10, fps=30)
    assert len(idx) >= n - 2                                      # ~n frames, no collapse
    assert idx == sorted(idx)
    assert sum(i < burst_end for i in idx) <= 2                  # burst can't dominate (was ~5/20)
    assert idx[-1] - idx[0] >= 0.75 * total                      # picks span the timeline
    gaps = [b - a for a, b in zip(idx[:-1], idx[1:], strict=True)]
    assert max(gaps) <= 3 * (total / n)                          # no giant temporal hole


def test_extract_all_parallel_jobs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _project_with_n_videos(d, ["a", "b", "c", "d"])
        code, out = _run(["extract-frames", "--all", "-j", "2", "--project", str(d / "ws"), "-n", "5"])
        assert code == 0, out
        assert "4/4 video(s)" in out and "jobs: 2" in out
        for v in ("a", "b", "c", "d"):
            assert len(list(proj.layout.frames_dir(v, "original").glob("*.png"))) == 5


def test_parallel_and_serial_produce_same_frames():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _project_with_n_videos(d, ["x", "y"])
        assert _run(["extract-frames", "--all", "-j", "2", "--project", str(d / "ws"), "-n", "6"])[0] == 0
        parallel = {v: sorted(p.name for p in proj.layout.frames_dir(v, "original").glob("*.png"))
                    for v in ("x", "y")}
        # wipe and redo serially
        for v in ("x", "y"):
            for f in proj.layout.frames_dir(v, "original").glob("*.png"):
                f.unlink()
        assert _run(["extract-frames", "--all", "-j", "1", "--project", str(d / "ws"), "-n", "6"])[0] == 0
        serial = {v: sorted(p.name for p in proj.layout.frames_dir(v, "original").glob("*.png"))
                  for v in ("x", "y")}
        assert parallel == serial          # parallelism changes speed, not output


# ------------------------------------------------- frames and links stay intact
def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _self_link(path: Path) -> None:
    """Replace ``path`` with a symlink to itself -- what the old ingest left behind."""
    path.unlink()
    path.symlink_to(path)


def test_annotate_leaves_frames_as_real_files():
    """Regression: ingesting through the staging view replaced every frame with a self-link."""
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        before = {p.name: _sha(p) for p in frames_mod.extract_frames(proj, vid, n=6)}

        for _session in range(2):                                  # a second session must still open
            ann.annotate_video(proj, vid, _launch=_fake_napari_that_labels)
            frames = sorted(proj.layout.frames_dir(vid, "original").glob("*.png"))
            assert {p.name: _sha(p) for p in frames} == before     # same files, same pixels
            assert not any(p.is_symlink() for p in frames)
            staged = sorted(proj.layout.staging_dataset_dir(vid).glob("*.png"))
            assert [p.name for p in staged] == sorted(before)
            assert all(p.is_symlink() and p.is_file() for p in staged)   # links, none dangling


def test_annotate_restores_frames_destroyed_by_self_links():
    """A project already damaged by the old ingest heals on the next annotate, labels kept."""
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        ann.annotate_video(proj, vid, n=6, _launch=_fake_napari_that_labels)
        frames = sorted(proj.layout.frames_dir(vid, "original").glob("*.png"))
        before = {p.name: _sha(p) for p in frames}
        labels_before = pd.read_parquet(proj.layout.labels_parquet(vid))
        for p in frames:
            _self_link(p)
        assert not any(p.is_file() for p in frames)

        seen = {}

        def reopen(config_path, dataset_dir):
            seen["readable"] = sorted(p.name for p in dataset_dir.glob("*.png") if p.is_file())
            seen["labels"] = ann.find_collected_data(dataset_dir)

        ann.annotate_video(proj, vid, n=99, _launch=reopen)        # n ignored: selection is kept
        assert seen["readable"] == sorted(before)                  # the annotator sees every frame
        assert seen["labels"] is not None                          # ...and the earlier labels
        frames = sorted(proj.layout.frames_dir(vid, "original").glob("*.png"))
        assert {p.name: _sha(p) for p in frames} == before         # re-read from the same indices
        assert not any(p.is_symlink() for p in frames)
        pd.testing.assert_frame_equal(pd.read_parquet(proj.layout.labels_parquet(vid)), labels_before)


def test_extract_reports_unrestorable_frames():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        for p in frames_mod.extract_frames(proj, vid, n=4):
            _self_link(p)
        for media in proj.video_media_files(vid):                  # the video is gone too
            media.unlink()
        Path(proj.video_record(vid).source_path).unlink()
        try:
            frames_mod.extract_frames(proj, vid, n=4)
        except ValueError as err:
            assert "--overwrite" in str(err)
        else:
            raise AssertionError("expected ValueError")


def test_extract_leaves_external_dangling_links_alone():
    """A link to a missing file elsewhere may come back; it is not silently re-extracted."""
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        frames = frames_mod.extract_frames(proj, vid, n=4)
        elsewhere = Path(d) / "legacy" / frames[0].name
        frames[0].unlink()
        frames[0].symlink_to(elsewhere)
        assert frames_mod.extract_frames(proj, vid, n=4) == frames[1:]
        assert frames[0].is_symlink() and not frames[0].exists()


def test_staging_repairs_and_prunes_links():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        frames = frames_mod.extract_frames(proj, vid, n=6)
        _, dataset_dir = ann.stage_annotation_project(proj, vid, frames, scorer="gravio")
        saved = dataset_dir / "CollectedData_gravio.h5"
        saved.write_bytes(b"labels")

        dangling = dataset_dir / frames[0].name                    # a link that no longer resolves
        dangling.unlink()
        dangling.symlink_to(dataset_dir / "gone.png")
        (dataset_dir / "img9999.png").symlink_to(dataset_dir / "gone.png")   # frame since dropped

        ann.stage_annotation_project(proj, vid, frames, scorer="gravio")
        staged = sorted(dataset_dir.glob("*.png"))
        assert [p.name for p in staged] == [f.name for f in frames]
        assert all(p.resolve() == f.resolve() for p, f in zip(staged, frames, strict=True))
        assert saved.read_bytes() == b"labels"                     # saved labels are never pruned


def test_processed_frames_backfilled_under_original_names():
    """A processed video registered after extraction still gets its frame set."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = Project.create(d / "ws", task="reach", bodyparts=BODYPARTS)
        vid = proj.add_video(_make_video(d / "raw" / "clip.mp4", size=(160, 120)), link="copy")
        orig = frames_mod.extract_frames(proj, vid, n=5)
        assert not proj.layout.frames_dir(vid, "processed").exists()

        proj.add_video(_make_video(d / "small" / "clip.mp4", size=(80, 60)), kind="processed", link="copy")
        assert frames_mod.extract_frames(proj, vid, n=5) == orig   # selection untouched
        proc = sorted(proj.layout.frames_dir(vid, "processed").glob("*.png"))
        assert [p.name for p in proc] == [p.name for p in orig]
        assert proj.label_frames_dir(vid) == proj.layout.frames_dir(vid, "processed")


def test_extract_overwrite_never_writes_through_a_link():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project_with_video(Path(d))
        frames = frames_mod.extract_frames(proj, vid, n=4)
        outside = Path(d) / "outside.png"
        outside.write_bytes(b"keep me")
        frames[0].unlink()
        frames[0].symlink_to(outside)
        redone = frames_mod.extract_frames(proj, vid, n=4, overwrite=True)
        assert outside.read_bytes() == b"keep me"
        assert [p.name for p in redone] == [p.name for p in frames]
        assert not any(p.is_symlink() for p in redone)


def test_resolve_media_ignores_video_toml():
    """``video.toml`` sits beside the media; it must never be mistaken for it."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = Project.create(d / "ws", task="reach", bodyparts=BODYPARTS)
        src = _make_video(d / "raw" / "clip.mp4")
        ref = proj.add_video(src, link="reference", video_id="ref")
        assert frames_mod.resolve_media(proj, ref) == src.resolve()          # nothing materialized
        late = d / "raw" / "late.webm"                                       # sorts after "toml"
        late.write_bytes(src.read_bytes())
        vid = proj.add_video(late, link="symlink")
        assert frames_mod.resolve_media(proj, vid).name == "video.webm"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
    print(f"annotate: {passed}/{passed} checks passed")
