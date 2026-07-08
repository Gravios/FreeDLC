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


def test_beside_video_path():
    p = apply_mod.beside_video_path("/data/vids/sess/clip-0001-20250901.mp4")
    assert p == Path("/data/vids/sess/clip-0001-20250901.fdlc.parquet")
    t = apply_mod.fdlc_sidecar_path("/data/vids/sess/clip-0001-20250901.mp4")
    assert t == Path("/data/vids/sess/clip-0001-20250901.fdlc.toml")


def test_bundle_carries_skeleton():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bundle = _bundle(d, bodyparts=("snout", "tail"))
        # default is empty; a bundle created with a skeleton persists it
        assert bundle.card.skeleton == []
        cfg = d / "pc.yaml"
        cfg.write_text("net_type: resnet_50\nmetadata: {bodyparts: [snout, tail]}\n")
        snap = d / "s.pt"
        snap.write_bytes(b"w")
        b2 = ws.ModelBundle.create(d / "b2", pose_config_src=cfg, snapshot_src=snap,
                                   architecture="resnet_50", bodyparts=["snout", "tail"],
                                   model_id="b2", skeleton=[["snout", "tail"]])
        assert b2.card.skeleton == [["snout", "tail"]]
        assert ws.ModelBundle.open(d / "b2").card.skeleton == [["snout", "tail"]]


def test_apply_to_videos_beside_video(monkeypatch):
    import tomllib
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bundle = _bundle(d)
        _patch_inference(monkeypatch)
        viddir = d / "source" / "sess1"
        viddir.mkdir(parents=True)
        v = viddir / "stroh-kj-0001-20250901.mp4"
        v.write_bytes(b"v")

        results = ws.apply_to_videos(bundle, [v], d / "ignored", beside_video=True,
                                     skeleton=[["snout", "tail"], ["tail", "tip"]])
        expected = viddir / "stroh-kj-0001-20250901.fdlc.parquet"
        assert results[str(v)] == expected              # <video-stem>.fdlc.parquet next to the video
        assert expected.exists()
        assert not (d / "ignored").exists()             # out_root ignored in beside mode
        assert not (viddir / "run.toml").exists()       # no legacy run.toml in the source dir

        sidecar = viddir / "stroh-kj-0001-20250901.fdlc.toml"
        assert sidecar.exists()                         # <stem>.fdlc.toml sidecar written beside
        meta = tomllib.loads(sidecar.read_text())
        assert meta["skeleton"] == [["snout", "tail"], ["tail", "tip"]]   # skeleton edges included
        assert meta["output"] == expected.name
        assert meta["model_id"] == "m1"
        assert meta["bodyparts"] == ["snout"]


def test_apply_labeled_video_wiring(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bundle = _bundle(d, bodyparts=("snout", "tail"))
        _patch_inference(monkeypatch)
        calls = []

        def fake_render(video, df, out_path, bundle, skeleton, pcutoff):
            calls.append((Path(video).name, Path(out_path).name,
                          tuple(tuple(e) for e in (skeleton or [])), pcutoff))
            Path(out_path).write_bytes(b"mp4")
            return Path(out_path)

        monkeypatch.setattr(apply_mod, "_render_labeled", fake_render)
        viddir = d / "src"
        viddir.mkdir()
        v = viddir / "clip.mp4"
        v.write_bytes(b"v")

        ws.apply_to_videos(bundle, [v], d / "ig", beside_video=True,
                           skeleton=[["snout", "tail"]], labeled_video=True, pcutoff=0.5)
        assert calls == [("clip.mp4", "clip.fdlc.mp4", (("snout", "tail"),), 0.5)]  # routed to .fdlc.mp4
        assert (viddir / "clip.fdlc.mp4").exists()


def test_render_labeled_video_real():
    try:
        import cv2
        import numpy as np
    except ImportError:
        return  # cv2 not present in this env; wiring is covered above
    from deeplabcut.workspace.label_video import render_labeled_video

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        v = d / "clip.mp4"
        w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for _ in range(4):
            w.write(np.full((48, 64, 3), 30, np.uint8))
        w.release()
        rows = [(f, "single", bp, 10.0 + f, 20.0, 0.99)
                for f in range(4) for bp in ("snout", "tail")]
        df = pd.DataFrame(rows, columns=["frame", "individual", "bodypart", "x", "y", "likelihood"])
        out = render_labeled_video(v, df, d / "clip.fdlc.mp4",
                                   bodyparts=["snout", "tail"], skeleton=[["snout", "tail"]],
                                   progress=False)
        assert out.exists() and out.stat().st_size > 0
        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 4


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
