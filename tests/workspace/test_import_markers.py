"""``fdlc import``: markers from pose parquets, as labels or as an analyze run.

A tiny project with an original (64x48) and a processed (32x24) synthetic video; pose
files in FreeDLC's long layout and Mufasa's wide one, written in processed pixels.

Run via ``tests/workspace/run_all.sh`` or directly.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.workspace import Project
from deeplabcut.workspace import import_markers as im
from deeplabcut.workspace.annotate import stage_annotation_project
from deeplabcut.workspace.annotations import collected_data_to_long_df, find_collected_data, read_collected_data
from deeplabcut.workspace.cli import main
from deeplabcut.workspace.refine import match_video, run_poses

BODYPARTS = ["snout", "paw_L", "paw_R"]
N = 40


def _video(path: Path, size) -> Path:
    import cv2

    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(N):
        frame = np.full((h, w, 3), (i * 6) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _project(root: Path):
    proj = Project.create(root / "ws", task="reach", bodyparts=BODYPARTS)
    vid = proj.add_video(_video(root / "raw" / "Session1.mp4", (64, 48)), link="copy")
    proj.add_video(_video(root / "raw" / "Session1-32x24.mp4", (32, 24)), video_id=vid, kind="processed",
                   link="copy")
    return proj, vid


def _long(markers=BODYPARTS, frames=range(N), individual="single"):
    rows = []
    for f in frames:
        for k, m in enumerate(markers):
            rows.append((f, individual, m, 1.0 + k + 0.1 * f, 2.0 + k, 0.9))
    return pd.DataFrame(rows, columns=["frame", "individual", "bodypart", "x", "y", "likelihood"])


def _wide(long):
    out = pd.DataFrame(index=sorted(long.frame.unique()))
    for m, g in long.groupby("bodypart", sort=False):
        g = g.set_index("frame")
        out[f"{m}_x"], out[f"{m}_y"], out[f"{m}_p"] = g.x, g.y, g.likelihood
        out[f"{m}_var_x"], out[f"{m}_var_y"] = 0.1, 0.1            # a smoother's columns
    return out.reset_index(drop=True)


def _run_cli(*args) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main([str(a) for a in args])
    return rc, buf.getvalue()


# ------------------------------------------------------------------ reading
def test_read_long_and_wide():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        long = _long()
        long.drop(columns=["individual"]).to_parquet(d / "a.parquet")
        got, layout = im.read_markers(d / "a.parquet")
        assert layout == "long" and list(got.columns) == im.LONG_COLUMNS and (got.individual == "single").all()
        wide = _wide(long)
        wide.loc[3, ["snout_x", "snout_y", "snout_p"]] = 0.0           # Mufasa's missing point
        wide.to_parquet(d / "b.parquet")
        got, layout = im.read_markers(d / "b.parquet")
        assert layout == "wide" and set(got.bodypart) == set(BODYPARTS)      # no "snout_var"
        snout3 = got[(got.frame == 3) & (got.bodypart == "snout")].iloc[0]
        assert np.isnan(snout3.x) and np.isnan(snout3.y)
        paw = got[(got.frame == 5) & (got.bodypart == "paw_L")].iloc[0]
        assert paw.x == 1.0 + 1 + 0.5 and paw.likelihood == 0.9
        wide.assign(frame=wide.index + 100).to_parquet(d / "c.parquet")    # a frame column wins
        assert im.read_markers(d / "c.parquet")[0].frame.min() == 100
        pd.DataFrame({"a": [1]}).to_parquet(d / "d.parquet")
        try:
            im.read_markers(d / "d.parquet")
            raise AssertionError("a table of neither layout was read")
        except ValueError as err:
            assert "neither a long pose" in str(err)


# ------------------------------------------------------------------ checking
def test_markers_checked_by_name():
    with tempfile.TemporaryDirectory() as d:
        proj, _ = _project(Path(d))
        extra = _long(markers=[*BODYPARTS, "tail"])
        try:
            im.check_markers(extra, proj)
            raise AssertionError("a marker the project doesn't have was taken")
        except ValueError as err:
            assert "tail" in str(err) and "--rename" in str(err) and "--drop-unknown" in str(err)
        assert set(im.check_markers(extra, proj, drop_unknown=True).bodypart) == set(BODYPARTS)
        other = _long(markers=["nose", "paw_L", "paw_R"])
        assert set(im.check_markers(other, proj, rename={"nose": "snout"}).bodypart) == set(BODYPARTS)
        for rename, says in (({"nose": "paw_L"}, "merge"), ({"ear": "snout"}, "doesn't have")):
            try:
                im.check_markers(other, proj, rename=rename)
                raise AssertionError(f"{rename} was accepted")
            except ValueError as err:
                assert says in str(err), err
        assert (im.check_markers(_long(individual="idv0"), proj).individual == "single").all()
        two = pd.concat([_long(individual="a"), _long(individual="b")])
        try:
            im.check_markers(two, proj)
            raise AssertionError("two animals were taken into a one-animal project")
        except ValueError as err:
            assert "2 individuals" in str(err)


# ------------------------------------------------------------------ the video
def test_find_video_and_its_pixels():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj, vid = _project(d)
        small = _long()                                       # fits both videos
        for name in ("Session1.parquet", "Session1-32x24.fdlc.parquet", "session1.cam1-triplet.paws.parquet"):
            try:
                im.find_video(proj, d / name, small)
                raise AssertionError("the kind was guessed for coordinates that fit both videos")
            except ValueError as err:
                assert "give --kind" in str(err)
            assert im.find_video(proj, d / name, small, kind="processed") == (vid, "processed")
        big = small.assign(x=small.x * 10)                    # beyond the processed video: original
        assert im.find_video(proj, d / "Session1.parquet", big) == (vid, "original")
        media = proj.video_media_files(vid, "processed")[0]
        assert im.find_video(proj, d / "x.parquet", small, video=media) == (vid, "processed")
        assert im.find_video(proj, d / "x.parquet", small, video_id=vid, kind="original") == (vid, "original")
        try:
            im.find_video(proj, d / "Other.parquet", small)
            raise AssertionError("an unmatched name found a video")
        except ValueError as err:
            assert "--video" in str(err)
        assert im.to_original_scale(proj, vid, "processed") == (2.0, 2.0)
        assert im.to_original_scale(proj, vid, "original") == (1.0, 1.0)


def test_frames_chosen():
    long = _long()
    long.loc[(long.frame == 7), "likelihood"] = 0.1
    assert im.parse_frames("3,5,10-12") == [3, 5, 10, 11, 12]
    assert im.select_frames(long, n=5) == [0, 10, 20, 29, 39]
    assert im.select_frames(long, frames=[7, 8, 99]) == [7, 8]
    assert 7 not in im.select_frames(long, every=True, min_likelihood=0.5)
    assert im.select_frames(long, labeled={2, 4, 99}) == [2, 4]


# ------------------------------------------------------------------ labels
def test_import_labels_merges_with_the_annotators_copy():
    with tempfile.TemporaryDirectory() as d:
        proj, vid = _project(Path(d))
        long = im.check_markers(_long(), proj)
        s = im.import_labels(proj, long, vid, "processed", [4, 8])
        labels = pd.read_parquet(proj.layout.labels_parquet(vid))
        assert s["placed"] == 6 and s["extracted"] == 2
        assert sorted(labels.image.unique()) == ["img0004.png", "img0008.png"]
        row = labels[(labels.image == "img0004.png") & (labels.bodypart == "paw_L")].iloc[0]
        assert (row.x, row.y) == ((1.0 + 1 + 0.4) * 2, (2.0 + 1) * 2)   # processed -> original pixels
        assert proj.labels_record(vid).space == "original"
        assert (proj.layout.frames_dir(vid, "processed") / "img0004.png").is_file()   # both frame sets

        # the annotator's copy exists: it must get the new markers too (it replaces labels.parquet on close)
        frames = sorted(proj.layout.frames_dir(vid, "original").glob("*.png"))
        stage_annotation_project(proj, vid, frames, scorer="me")
        moved = long.assign(x=long.x + 5)
        s = im.import_labels(proj, moved, vid, "processed", [4, 12])
        assert s["kept"] == 3 and s["placed"] == 3                       # frame 4 kept, frame 12 new
        staged = find_collected_data(proj.layout.staging_dataset_dir(vid))
        staged_long = collected_data_to_long_df(read_collected_data(staged)).dropna(subset=["x"])
        labels = pd.read_parquet(proj.layout.labels_parquet(vid)).dropna(subset=["x"])
        assert len(staged_long) == len(labels) == 9 and staged.with_name(staged.name + ".bak").is_file()
        s = im.import_labels(proj, moved, vid, "processed", [4], overwrite=True)
        labels = pd.read_parquet(proj.layout.labels_parquet(vid))
        row = labels[(labels.image == "img0004.png") & (labels.bodypart == "paw_L")].iloc[0]
        assert s["replaced"] == 3 and row.x == (1.0 + 1 + 0.4 + 5) * 2

        before = proj.layout.labels_parquet(vid).read_bytes()
        s = im.import_labels(proj, long, vid, "processed", [20], propose=True)
        assert s["target"].endswith("machinelabels-iter0.h5") and s["placed"] == 3
        assert proj.layout.labels_parquet(vid).read_bytes() == before            # proposals only


# ------------------------------------------------------------------ runs
def test_import_run_is_a_run_like_applys():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj, vid = _project(d)
        long = im.check_markers(_long(), proj)
        run, paths = im.import_run(proj, [(d / "Session1.parquet", long, vid, "processed")], note="smoothed")
        assert run.manifest().status == "finished" and run.manifest().notes == "smoothed"
        assert run.manifest().params["imported_from"][0].endswith("Session1.parquet")
        (video, pose), = run_poses(proj, run)
        assert pose == paths[0] and match_video(proj, video) == (vid, "processed")
        back = pd.read_parquet(pose)
        assert list(back.columns) == im.LONG_COLUMNS and len(back) == len(long)
        try:
            im.import_run(proj, [(d / "a.parquet", long, vid, "processed"), (d / "b.parquet", long, vid, "processed")])
            raise AssertionError("two poses of one video were put in one run")
        except ValueError as err:
            assert "separate runs" in str(err)


# ------------------------------------------------------------------ the CLI
def test_cli():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj, vid = _project(d)
        _wide(_long()).to_parquet(d / "Session1_smoothed_v2.parquet")
        _long().to_parquet(d / "Session1.paws.parquet")
        rc, out = _run_cli("import", "labels", d / "Session1.paws.parquet", "--project", proj.root,
                           "--kind", "processed", "-n", "3", "--dry-run")
        assert rc == 0 and "would label 3 frame(s)" in out and not proj.layout.labels_parquet(vid).exists()
        rc, out = _run_cli("import", "labels", d / "Session1.paws.parquet", "--project", proj.root,
                           "--kind", "processed", "--frames", "1,2")
        assert rc == 0 and "6 marker(s) labeled in 2 frame(s)" in out
        rc, out = _run_cli("import", "run", d / "Session1_smoothed_v2.parquet", "--project", proj.root,
                           "--kind", "processed")
        assert rc == 0 and "wide, 40 frame(s), 3 marker(s)" in out and len(proj.runs("analyze")) == 1
        rc, out = _run_cli("import", "run", d / "Session1_smoothed_v2.parquet", d / "Session1.paws.parquet",
                           "--project", proj.root, "--video-id", vid)
        assert rc == 2 and "--video/--video-id name one file's video" in out
        _long(markers=["tail"]).to_parquet(d / "Session1.tail.parquet")
        rc, out = _run_cli("import", "labels", d / "Session1.tail.parquet", "--project", proj.root,
                           "--kind", "processed")
        assert rc == 2 and "not the project's" in out


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
    print(f"import_markers: {passed}/{passed} checks passed")
