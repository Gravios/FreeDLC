#
# FreeDLC workspace layer -- batch apply tests
#
"""Tests for deeplabcut.workspace.apply batch labeling.

collect_videos (file/dir/glob expansion) runs for real; apply_to_videos is
exercised with the torch/parquet seams (_build_runners, _infer_to_df,
write_pose_parquet) patched, so the loop, the build-once behavior, the per-video
output layout, and error handling are all verified without a GPU.

Standalone: ``python tests/workspace/test_apply.py`` -> ``apply: N/N checks passed``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from deeplabcut import workspace as ws
from deeplabcut.workspace import apply as apply_mod


def _bundle(root: Path, bodyparts=("snout",)):
    proj = ws.Project.create(root / "ws", task="reach", bodyparts=list(bodyparts))
    cfg = root / "pytorch_config.yaml"
    with cfg.open("w") as fh:
        yaml.safe_dump({"net_type": "resnet_50", "metadata": {"bodyparts": list(bodyparts)}}, fh)
    snap = root / "snapshot-050.pt"
    snap.write_bytes(b"w")
    return ws.ModelBundle.create(proj.layout.model_dir("m1"), pose_config_src=cfg, snapshot_src=snap,
                                 architecture="resnet_50", bodyparts=list(bodyparts), model_id="m1")


def _patch_inference(mp, *, fail_on=None):
    """Patch the torch/parquet seams; return a dict tracking build/infer calls."""
    tracker = {"builds": 0, "inferred": []}

    def fake_build(bundle, **kw):
        tracker["builds"] += 1
        return ("POSE_RUNNER", None)

    def fake_infer(bundle, video, pose_runner, detector_runner, *, cropping):
        name = Path(video).name
        tracker["inferred"].append(name)
        if fail_on and name == fail_on:
            raise RuntimeError("boom")
        return pd.DataFrame({"frame": [0], "individual": ["single"], "bodypart": ["snout"],
                             "x": [1.0], "y": [2.0], "likelihood": [0.9]})

    def fake_write_parquet(df, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub-parquet")
        return path

    mp.setattr(apply_mod, "_build_runners", fake_build)
    mp.setattr(apply_mod, "_infer_to_df", fake_infer)
    mp.setattr(apply_mod, "write_pose_parquet", fake_write_parquet)
    return tracker


# ------------------------------------------------------------- collect_videos
def test_collect_videos_files_dirs_globs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.mp4").write_bytes(b"v")
        (d / "b.avi").write_bytes(b"v")
        (d / "notes.txt").write_bytes(b"x")
        sub = d / "more"
        sub.mkdir()
        (sub / "c.mov").write_bytes(b"v")
        got = ws.collect_videos([str(d / "a.mp4"), str(sub), str(d / "*.avi")])
        assert sorted(p.name for p in got) == ["a.mp4", "b.avi", "c.mov"]  # txt filtered from dir


def test_collect_videos_dedups():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.mp4").write_bytes(b"v")
        got = ws.collect_videos([str(d / "a.mp4"), str(d), str(d / "*.mp4")])
        assert len(got) == 1


def test_collect_videos_explicit_file_not_filtered():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "clip.dat").write_bytes(b"v")  # odd extension, but named explicitly
        assert [p.name for p in ws.collect_videos([str(d / "clip.dat")])] == ["clip.dat"]


# ------------------------------------------------------------- apply_to_videos
def test_apply_to_videos_builds_runner_once(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bundle = _bundle(d)
        tracker = _patch_inference(monkeypatch)
        vids = [d / "clip1.mp4", d / "clip2.mp4", d / "clip3.mp4"]
        for v in vids:
            v.write_bytes(b"v")

        results = ws.apply_to_videos(bundle, vids, d / "out", batch_size=2)
        assert tracker["builds"] == 1                                    # runner built ONCE
        assert tracker["inferred"] == ["clip1.mp4", "clip2.mp4", "clip3.mp4"]
        assert len(results) == 3
        for v in vids:
            od = (d / "out") / v.stem
            assert (od / "pose.parquet").exists() and (od / "run.toml").exists()


def test_apply_to_videos_on_error_skip(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bundle = _bundle(d)
        _patch_inference(monkeypatch, fail_on="clip2.mp4")
        vids = [d / "clip1.mp4", d / "clip2.mp4", d / "clip3.mp4"]
        for v in vids:
            v.write_bytes(b"v")

        results = ws.apply_to_videos(bundle, vids, d / "out", on_error="skip")
        assert results[str(d / "clip2.mp4")] is None            # failed video recorded as None
        assert results[str(d / "clip1.mp4")] is not None
        assert results[str(d / "clip3.mp4")] is not None        # loop continued past the failure


def test_apply_to_videos_on_error_raise(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bundle = _bundle(d)
        _patch_inference(monkeypatch, fail_on="clip1.mp4")
        (d / "clip1.mp4").write_bytes(b"v")
        try:
            ws.apply_to_videos(bundle, [d / "clip1.mp4"], d / "out")
        except RuntimeError:
            pass
        else:
            raise AssertionError("default on_error should re-raise")


# ------------------------------------------------------------------ smoke runner
def _run_smoke() -> int:
    class _MP:
        def setattr(self, obj, name, val):
            self._saved = getattr(self, "_saved", [])
            self._saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, val in reversed(getattr(self, "_saved", [])):
                setattr(obj, name, val)

    import inspect

    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        if "monkeypatch" in inspect.signature(chk).parameters:
            mp = _MP()
            try:
                chk(mp)
            finally:
                mp.undo()
        else:
            chk()
    print(f"apply: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
