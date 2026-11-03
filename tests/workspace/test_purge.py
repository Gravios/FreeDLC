#
# FreeDLC workspace layer -- purge tests
#
"""Tests for ``fdlc purge --unlabeled-frames``. Needs pandas, pyarrow and pytables; no torch.

Standalone: ``python tests/workspace/test_purge.py`` -> ``purge: N/N checks passed``.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut import workspace as ws
from deeplabcut.workspace import cli
from deeplabcut.workspace.annotate import MACHINE_LABELS
from deeplabcut.workspace.annotations import (
    collected_data_to_long_df,
    long_df_to_collected_data,
    read_collected_data,
    write_collected_data,
    write_labels_parquet,
)
from deeplabcut.workspace.manifest import write_manifest
from deeplabcut.workspace.purge import plan_unlabeled_frames
from deeplabcut.workspace.schema import VideoRecord

BODYPARTS = ["snout", "paw"]
FRAMES = [f"img{i:03d}.png" for i in range(6)]


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _long(placed: dict[str, float], images, likelihood=False):
    """Long labels for ``images``: a snout at ``placed[image]`` where given, everything else NaN."""
    rows = [(im, "single", bp, placed.get(im, np.nan) if bp == "snout" else np.nan, 1.0 if im in placed else np.nan)
            for im in images for bp in BODYPARTS]
    df = pd.DataFrame(rows, columns=["image", "individual", "bodypart", "x", "y"])
    return df.assign(likelihood=0.9) if likelihood else df


def _project(d: Path, *, legacy_target: Path | None = None):
    """``clip`` with six frames in both sets.

    Labeled: img000 and img001 in labels.parquet, img002 only in the staged
    CollectedData (saved, not ingested). Proposed: img003. Unlabeled: img003-img005.
    """
    proj = ws.Project.create(d / "ws", task="reach", bodyparts=BODYPARTS)
    for kind in ("original", "processed"):
        write_manifest(proj.layout.video_toml("clip", kind),
                       VideoRecord(video_id="clip", source_path=f"{kind}.mp4", link="reference").to_dict())
        frames_dir = proj.layout.frames_dir("clip", kind)
        frames_dir.mkdir(parents=True)
        for name in FRAMES:
            if kind == "original" and name == "img005.png" and legacy_target is not None:
                (frames_dir / name).symlink_to(legacy_target)        # a migrated project's frame
            else:
                (frames_dir / name).write_bytes(b"png")
    write_labels_parquet(_long({"img000.png": 5.0, "img001.png": 6.0}, FRAMES), proj.layout.labels_parquet("clip"))
    staging = proj.layout.staging_dataset_dir("clip")
    staging.mkdir(parents=True)
    for name in FRAMES:
        (staging / name).symlink_to(proj.layout.frames_dir("clip", "original") / name)
    staged = _long({"img000.png": 5.0, "img001.png": 6.0, "img002.png": 7.0}, FRAMES)
    write_collected_data(long_df_to_collected_data(staged, proj, scorer="me", dataset="clip"),
                         staging / "CollectedData_me.h5")
    proposals = _long({"img003.png": 8.0}, ["img003.png"], likelihood=True)
    write_collected_data(long_df_to_collected_data(proposals, proj, scorer="fdlc", dataset="clip", likelihood=True),
                         staging / MACHINE_LABELS)
    return proj


def test_plan_counts_staged_labels_as_labels():
    with tempfile.TemporaryDirectory() as d:
        proj = _project(Path(d))
        plan = plan_unlabeled_frames(proj, "clip")
        assert (plan.n_frames, plan.n_labeled) == (6, 3)
        assert plan.names == ["img003.png", "img004.png", "img005.png"] and plan.proposed == ["img003.png"]
        kept = plan_unlabeled_frames(proj, "clip", keep_proposed=True)
        assert kept.names == ["img004.png", "img005.png"] and kept.kept_proposed == ["img003.png"]


def test_purge_lists_first_and_removes_with_yes():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        legacy = d / "legacy" / "img005.png"
        legacy.parent.mkdir()
        legacy.write_bytes(b"png")
        proj = _project(d, legacy_target=legacy)
        ws_ = str(proj.root)
        lay = proj.layout

        code, out = _run(["purge", "--project", ws_])
        assert code == 2 and "--unlabeled-frames" in out
        code, out = _run(["purge", "--unlabeled-frames", "--project", ws_])
        assert code == 0 and "clip: 6 frame(s), 3 labeled; 3 unlabeled to remove (1 with proposed markers" in out
        assert "add --yes to remove 3 unlabeled frame(s) from 1 video(s)" in out
        assert len(list(lay.frames_dir("clip", "original").glob("*.png"))) == 6       # a dry run

        code, out = _run(["purge", "--unlabeled-frames", "--keep-proposed", "--yes", "--project", ws_])
        assert code == 0 and "removed 2 unlabeled frame(s) from 1 video(s)" in out, out
        for kind in ("original", "processed"):
            left = sorted(p.name for p in lay.frames_dir("clip", kind).glob("*.png"))
            assert left == FRAMES[:4], (kind, left)
        assert legacy.is_file()                                       # the link went, not its target
        staging = lay.staging_dataset_dir("clip")
        assert sorted(p.name for p in staging.glob("*.png")) == FRAMES[:4]
        staged = collected_data_to_long_df(read_collected_data(staging / "CollectedData_me.h5"))
        assert sorted(set(staged["image"])) == FRAMES[:4]
        assert (staging / "CollectedData_me.h5.bak").is_file()
        assert sorted(set(pd.read_parquet(lay.labels_parquet("clip"))["image"])) == FRAMES[:4]
        assert (staging / MACHINE_LABELS).is_file()                   # img003's proposals stay

        code, out = _run(["purge", "--unlabeled-frames", "--yes", "--project", ws_])
        assert code == 0 and "removed 1 unlabeled frame(s)" in out
        assert not (staging / MACHINE_LABELS).exists()                # its last frame went
        labels = pd.read_parquet(lay.labels_parquet("clip"))
        assert labels["x"].notna().sum() == 2 and sorted(set(labels["image"])) == FRAMES[:3]
        code, out = _run(["purge", "--unlabeled-frames", "--project", ws_])
        assert code == 0 and "no unlabeled frame to remove" in out and "nothing to remove" in out


def test_a_video_without_labels_is_left_unless_named():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=BODYPARTS)
        write_manifest(proj.layout.video_toml("fresh"),
                       VideoRecord(video_id="fresh", source_path="fresh.mp4", link="reference").to_dict())
        frames_dir = proj.layout.frames_dir("fresh")
        frames_dir.mkdir(parents=True)
        for name in FRAMES[:2]:
            (frames_dir / name).write_bytes(b"png")
        ws_ = str(proj.root)
        code, out = _run(["purge", "--unlabeled-frames", "--yes", "--project", ws_])
        assert code == 0 and "fresh: no labels yet; its 2 frame(s) are left" in out
        assert len(list(frames_dir.glob("*.png"))) == 2
        code, out = _run(["purge", "fresh", "--unlabeled-frames", "--yes", "--project", ws_])
        assert code == 0 and "removed 2 unlabeled frame(s)" in out and not list(frames_dir.glob("*.png"))
        code, out = _run(["purge", "nope", "--unlabeled-frames", "--project", ws_])
        assert code == 2 and "no registered video matches 'nope'" in out


def _analyze(proj, video: Path, *, labeled: bytes | None = b"x" * 2048, out: Path | None = None):
    """An analyze run with a pose for ``video``, and a labeled video beside it unless ``labeled`` is None."""
    run = proj.new_run("analyze")
    pose_dir = (out or run.dir) / video.stem
    pose_dir.mkdir(parents=True)
    pd.DataFrame({"frame": [0], "individual": ["single"], "bodypart": ["snout"], "x": [1.0], "y": [1.0],
                  "likelihood": [0.9]}).to_parquet(pose_dir / "pose.parquet")
    write_manifest(pose_dir / "run.toml", {"run_id": video.stem, "kind": "analyze", "inputs": [str(video)]})
    if labeled is not None:
        (pose_dir / "labeled.mp4").write_bytes(labeled)
    run.finish(outputs=[str(pose_dir / "pose.parquet")])
    return run, pose_dir


def test_labeled_videos_of_runs_are_removed_and_their_poses_kept():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _project(d)
        ws_ = str(proj.root)
        first, first_dir = _analyze(proj, d / "raw" / "clip.mp4")
        second, second_dir = _analyze(proj, d / "raw" / "other.mp4", out=d / "elsewhere")   # apply --out
        third, third_dir = _analyze(proj, d / "raw" / "clip.mp4", labeled=None)              # no labeled video

        code, out = _run(["purge", "--labeled-videos", "--project", ws_])
        assert code == 0 and f"run {first.run_id}: clip  2.0 kB" in out
        assert f"run {second.run_id}: (unregistered video)" in out and third.run_id not in out
        assert "add --yes to remove 2 labeled video(s) of 2 run(s), 4.1 kB" in out
        assert (first_dir / "labeled.mp4").is_file()                              # a dry run

        code, out = _run(["purge", "clip", "--labeled-videos", "--yes", "--project", ws_])
        assert code == 0 and "removed 1 labeled video(s) of 1 run(s), 2.0 kB" in out
        assert not (first_dir / "labeled.mp4").exists() and (first_dir / "pose.parquet").is_file()
        assert (second_dir / "labeled.mp4").is_file()                             # not clip's

        code, out = _run(["purge", "--labeled-videos", "--run", second.run_id, "--yes", "--project", ws_])
        assert code == 0 and "removed 1 labeled video(s)" in out and not (second_dir / "labeled.mp4").exists()
        code, out = _run(["purge", "--labeled-videos", "--project", ws_])
        assert code == 0 and "no labeled videos of analyze runs" in out and "nothing to remove" in out

        code, out = _run(["purge", "--labeled-videos", "--run", "nope", "--project", ws_])
        assert code == 2 and "no analyze run 'nope'" in out
        code, out = _run(["purge", "--unlabeled-frames", "--run", "latest", "--project", ws_])
        assert code == 2 and "--run only applies with --labeled-videos" in out
        code, out = _run(["purge", "--labeled-videos", "--keep-proposed", "--project", ws_])
        assert code == 2 and "--keep-proposed only applies with --unlabeled-frames" in out


def test_labeled_videos_named_after_their_video_are_removed_too():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _project(d)
        run = proj.new_run("analyze")
        pd.DataFrame({"frame": [0], "individual": ["single"], "bodypart": ["snout"], "x": [1.0], "y": [1.0],
                      "likelihood": [0.9]}).to_parquet(run.dir / "clip.parquet")
        write_manifest(run.dir / "clip.toml", {"run_id": "clip", "inputs": [str(d / "raw" / "clip.mp4")]})
        (run.dir / "clip.mp4").write_bytes(b"x" * 1000)
        run.finish(outputs=[str(run.dir / "clip.parquet")])
        code, out = _run(["purge", "--labeled-videos", "--yes", "--project", str(proj.root)])
        assert code == 0 and "removed 1 labeled video(s) of 1 run(s), 1.0 kB" in out, out
        assert not (run.dir / "clip.mp4").exists() and (run.dir / "clip.parquet").is_file()
        assert (run.dir / "clip.toml").is_file()

        # a run whose --out was the videos' own folder: the source video is never taken for a labeled one
        source = d / "raw2"
        source.mkdir()
        (source / "clip.mp4").write_bytes(b"video")
        other = proj.new_run("analyze")
        pd.DataFrame({"frame": [0], "individual": ["single"], "bodypart": ["snout"], "x": [1.0], "y": [1.0],
                      "likelihood": [0.9]}).to_parquet(source / "clip.parquet")
        write_manifest(source / "clip.toml", {"run_id": "clip", "inputs": [str(source / "clip.mp4")]})
        other.finish(outputs=[str(source / "clip.parquet")])
        code, out = _run(["purge", "--labeled-videos", "--yes", "--project", str(proj.root)])
        assert code == 0 and "no labeled videos" in out and (source / "clip.mp4").is_file()


def test_both_at_once_are_confirmed_together():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _project(d)
        _, pose_dir = _analyze(proj, d / "raw" / "clip.mp4")
        code, out = _run(["purge", "--unlabeled-frames", "--labeled-videos", "--project", str(proj.root)])
        assert code == 0 and ("add --yes to remove 3 unlabeled frame(s) from 1 video(s) and 1 labeled video(s) "
                              "of 1 run(s)") in out
        code, out = _run(["purge", "--unlabeled-frames", "--labeled-videos", "--yes", "--project", str(proj.root)])
        assert code == 0 and "removed 3 unlabeled frame(s)" in out and "removed 1 labeled video(s)" in out
        assert not (pose_dir / "labeled.mp4").exists()


def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"purge: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
