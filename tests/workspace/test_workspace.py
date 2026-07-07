#
# FreeDLC workspace layer -- tests
#
"""Tests for deeplabcut.workspace.

Runs two ways:
  * ``pytest tests/workspace/test_workspace.py`` -- normal CI usage.
  * ``python tests/workspace/test_workspace.py`` -- standalone smoke runner that
    prints ``workspace: N/N checks passed`` (AST-only environments: torch,
    pyarrow, cv2 are never imported here).
"""
from __future__ import annotations

import ast
import tempfile
from datetime import datetime
from pathlib import Path

from deeplabcut import workspace as ws
from deeplabcut.workspace import ids, schema

WS_DIR = Path(ws.__file__).parent


# --------------------------------------------------------------------------- ids
def test_id_format_and_uniqueness():
    fixed = datetime(2026, 7, 7, 14, 15, 30)
    i = ids.new_id(fixed)
    assert i.startswith("20260707-141530-")
    assert ids.is_id(i)
    assert not ids.is_id("nope")
    assert len({ids.new_id() for _ in range(200)}) == 200  # random suffix collision-resistant


def test_slugify():
    assert ids.slugify("Session 01 (reaching)!!") == "session-01-reaching"
    assert ids.video_id_from_path("/raw/Mouse_A/clip 3.mp4") == "clip-3"
    try:
        ids.slugify("!!!")
    except ValueError:
        pass
    else:
        raise AssertionError("empty slug should raise")


# ------------------------------------------------------------------------ schema
def test_schema_validation_and_roundtrip():
    pc = schema.ProjectConfig(task="reach", bodyparts=["snout", "paw"])
    assert schema.ProjectConfig.from_dict(pc.to_dict()).bodyparts == ["snout", "paw"]
    for bad in (
        lambda: schema.ProjectConfig(task="", bodyparts=["a"]),
        lambda: schema.ProjectConfig(task="t", bodyparts=[]),
        lambda: schema.ProjectConfig(task="t", bodyparts=["a", "a"]),
        lambda: schema.RunManifest(run_id="r", kind="bogus"),
        lambda: schema.ModelCard(model_id="m", architecture="a", bodyparts=["b"],
                                 default_snapshot="s.pt", top_down=True),  # no detector
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------- manifest
def test_manifest_roundtrip_drops_none():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.toml"
        from deeplabcut.workspace.manifest import read_manifest, write_manifest

        write_manifest(p, {"a": 1, "b": None, "c": ["x"], "nested": {"k": None, "j": 2}})
        got = read_manifest(p)
        assert got == {"a": 1, "c": ["x"], "nested": {"j": 2}}


# ------------------------------------------------------------------------ layout
def test_layout_paths():
    lay = ws.Layout("/proj")
    assert lay.project_toml == Path("/proj/project.toml")
    assert lay.video_media("v1") == Path("/proj/sources/videos/v1/video.mp4")
    assert lay.model_toml("m1") == Path("/proj/models/m1/model.toml")
    assert lay.run_toml("analyze", "r1") == Path("/proj/runs/analyze/r1/run.toml")
    try:
        lay.runs_kind_dir("train_bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("bad kind should raise")


# ----------------------------------------------------------------------- project
def test_project_lifecycle():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "proj"
        proj = ws.Project.create(root, task="reach", bodyparts=["snout", "paw"],
                                 experimenters=["gravio"])
        assert (root / "project.toml").exists()
        assert (root / "sources" / "videos").is_dir()
        # experimenter is metadata, never a path component:
        assert "gravio" not in str(root) and proj.config.experimenters == ["gravio"]

        reopened = ws.Project.open(root)
        assert reopened.config.task == "reach"

        # register a video three ways
        src = Path(d) / "session01.mp4"
        src.write_bytes(b"not really a video")
        vid = proj.add_video(src)  # symlink default
        assert vid == "session01"
        assert proj.layout.video_media(vid).is_symlink()
        assert proj.video_record(vid).size_bytes == len(b"not really a video")
        assert proj.videos() == ["session01"]

        proj.add_video(src, video_id="copy1", link="copy")
        assert proj.layout.video_media("copy1").is_file()
        proj.add_video(src, video_id="ref1", link="reference")
        assert not proj.layout.video_media("ref1").exists()  # reference materializes nothing
        assert proj.videos() == ["copy1", "ref1", "session01"]

        # duplicate id guarded
        try:
            proj.add_video(src, video_id="ref1")
        except FileExistsError:
            pass
        else:
            raise AssertionError("duplicate video id should raise")

        # runs
        run = proj.new_run("analyze", model_id="m123", inputs=[vid], params={"batch_size": 8})
        assert ids.is_id(run.run_id)
        assert run.manifest().status == "created"
        run.start()
        assert run.manifest().status == "running" and run.manifest().started
        vdir = run.video_dir(vid)
        assert vdir == run.dir / vid and vdir.is_dir()
        run.finish(outputs=["pose.parquet"])
        m = run.manifest()
        assert m.status == "finished" and m.outputs == ["pose.parquet"] and m.model_id == "m123"
        assert [r.run_id for r in proj.runs("analyze")] == [run.run_id]


# ------------------------------------------------------------------- model bundle
def test_model_bundle_create_open():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "pytorch_config.yaml"
        cfg.write_text("metadata:\n  bodyparts: [snout, paw]\n")
        snap = Path(d) / "snapshot-050.pt"
        snap.write_bytes(b"\x00weights")
        det = Path(d) / "snapshot-detector-020.pt"
        det.write_bytes(b"\x00det")

        # bottom-up bundle
        b = ws.ModelBundle.create(Path(d) / "bundle", pose_config_src=cfg, snapshot_src=snap,
                                  architecture="resnet50", bodyparts=["snout", "paw"])
        assert ids.is_id(b.card.model_id) and b.card.num_bodyparts == 2
        assert b.pose_config_path.name == "pose.yaml" and b.pose_config_path.exists()
        assert b.snapshot_path().name == "pose-snapshot-050.pt"
        assert b.snapshot_path().exists()

        b2 = ws.ModelBundle.open(Path(d) / "bundle")
        assert b2.card.architecture == "resnet50"

        # top-down bundle carries a detector
        td = ws.ModelBundle.create(Path(d) / "td", pose_config_src=cfg, snapshot_src=snap,
                                   architecture="dekr", bodyparts=["snout", "paw"],
                                   top_down=True, detector_snapshot_src=det)
        assert td.card.top_down and td.detector_snapshot_path().exists()

        # runner builders exist but are not invoked (torch-free environment)
        assert callable(td.build_pose_runner) and callable(td.build_detector_runner)


# -------------------------------------------------------------------------- apply
def test_predictions_to_long_df():
    import numpy as np

    # 2 frames, 2 individuals, 2 bodyparts, + 1 unique bodypart
    f0 = {
        "bodyparts": np.array([[[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]],
                               [[5.0, 6.0, 0.7], [7.0, 8.0, 0.6]]]),
        "unique_bodyparts": np.array([[[10.0, 11.0, 0.95]]]),
    }
    f1 = {"bodyparts": np.full((2, 2, 3), np.nan)}
    df = ws.predictions_to_long_df([f0, f1], bodyparts=["snout", "paw"],
                                   unique_bodyparts=["tailbase"])
    # frame0: 2 ind * 2 bpt + 1 unique = 5 rows; frame1: 4 rows
    assert len(df) == 9
    assert set(df.columns) == {"frame", "individual", "bodypart", "x", "y", "likelihood"}
    r = df[(df.frame == 0) & (df.individual == "idv1") & (df.bodypart == "paw")].iloc[0]
    assert (r.x, r.y, r.likelihood) == (7.0, 8.0, 0.6)
    uni = df[(df.frame == 0) & (df.bodypart == "tailbase")].iloc[0]
    assert uni.individual == ws.SINGLE_INDIVIDUAL and uni.x == 10.0
    assert df[df.frame == 1].likelihood.isna().all()


def test_apply_entrypoint_is_wired():
    # apply_to_video needs torch at call time; here we only assert it is importable
    # and that the module keeps torch/pyarrow imports lazy (not at module load).
    assert callable(ws.apply_to_video)
    src = (WS_DIR / "apply.py").read_text()
    tree = ast.parse(src)
    top_imports = {
        n.module or ""
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for n in ([node] if isinstance(node, ast.ImportFrom) else node.names)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(m.startswith(("torch", "pyarrow", "cv2")) for m in top_imports), top_imports


def test_no_heavy_imports_at_package_load():
    # Every workspace module must be importable without torch/cv2/pyarrow at load.
    for py in sorted(WS_DIR.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in tree.body:  # module-level only
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            assert not any(m.startswith(("torch", "cv2", "pyarrow", "numpy", "pandas"))
                           for m in mods), f"{py.name}: heavy top-level import {mods}"


# --------------------------------------------------------------- standalone smoke
def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed = 0
    for chk in checks:
        chk()
        passed += 1
    print(f"workspace: {passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
