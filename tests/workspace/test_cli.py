#
# FreeDLC workspace layer -- CLI tests
#
"""Tests for deeplabcut.workspace.cli.

The no-torch commands (migrate/info/models/videos) run for real; the torch-backed
commands (apply/train/evaluate) are checked for correct parsing and dispatch by
patching the workspace function each one calls. Nothing here imports torch.

Standalone: ``python tests/workspace/test_cli.py`` -> ``cli: N/N checks passed``.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import yaml

from deeplabcut import workspace as ws
from deeplabcut.workspace import cli


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _legacy_project(root: Path) -> Path:
    legacy = root / "legacy"
    (legacy / "videos").mkdir(parents=True)
    (legacy / "videos" / "clip1.mp4").write_bytes(b"v")
    train = legacy / "dlc-models-pytorch" / "iteration-0" / "reachJul7-trainset95shuffle1" / "train"
    train.mkdir(parents=True)
    with (train / "pytorch_config.yaml").open("w") as fh:
        yaml.safe_dump({"net_type": "resnet_50", "metadata": {"bodyparts": ["snout", "paw"]}}, fh)
    (train / "snapshot-best-100.pt").write_bytes(b"w")
    with (legacy / "config.yaml").open("w") as fh:
        yaml.safe_dump({"Task": "reach", "scorer": "gravio", "multianimalproject": False,
                        "bodyparts": ["snout", "paw"], "uniquebodyparts": [], "skeleton": [],
                        "video_sets": {str((legacy / "videos" / "clip1.mp4").resolve()): {}}}, fh)
    return legacy


def _model_project(root: Path):
    proj = ws.Project.create(root / "ws", task="reach", bodyparts=["snout", "paw"])
    cfg = root / "pytorch_config.yaml"
    with cfg.open("w") as fh:
        yaml.safe_dump({"net_type": "resnet_50", "metadata": {"bodyparts": ["snout", "paw"]}}, fh)
    snap = root / "snapshot-050.pt"
    snap.write_bytes(b"w")
    ws.ModelBundle.create(proj.layout.model_dir("m1"), pose_config_src=cfg, snapshot_src=snap,
                          architecture="resnet_50", bodyparts=["snout", "paw"], model_id="m1")
    return proj


# --------------------------------------------------------- no-torch commands
def test_migrate_command():
    with tempfile.TemporaryDirectory() as d:
        legacy = _legacy_project(Path(d))
        code, out = _run(["migrate", str(legacy), str(Path(d) / "ws"), "--no-annotations"])
        assert code == 0 and "migrated ->" in out
        proj = ws.Project.open(Path(d) / "ws")
        assert proj.videos() == ["clip1"] and len(proj.models()) == 1


def test_info_command():
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        code, out = _run(["info", str(proj.root)])
        assert code == 0
        assert "task:          reach" in out and "models:        1" in out


def test_models_command():
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        code, out = _run(["models", str(proj.root)])
        assert code == 0 and "m1" in out and "resnet_50" in out and "bottom-up" in out


def test_videos_command():
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        src = Path(d) / "clip1.mp4"
        src.write_bytes(b"v")
        proj.add_video(src)
        code, out = _run(["videos", str(proj.root)])
        assert code == 0 and "clip1" in out


def test_no_command_prints_help():
    code, _ = _run([])
    assert code == 2


# -------------------------------------------------- torch commands (dispatch)
def test_apply_dispatch_project(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        seen = {}

        def fake_apply_videos(bundle, videos, out_root, **kw):
            seen["videos"] = [Path(v) for v in videos]
            seen["batch_size"] = kw.get("batch_size")
            return {str(v): Path(out_root) / "pose.parquet" for v in videos}

        monkeypatch.setattr(cli, "apply_to_videos", fake_apply_videos)
        v1, v2 = Path(d) / "clip1.mp4", Path(d) / "clip2.mp4"
        v1.write_bytes(b"v")
        v2.write_bytes(b"v")
        code, out = _run(["apply", str(v1), str(v2), "--project", str(proj.root),
                          "--model-id", "m1", "--batch-size", "4"])
        assert code == 0
        assert len(seen["videos"]) == 2 and seen["batch_size"] == 4  # both videos, batch flag passed
        assert len(proj.runs("analyze")) == 1                        # one analyze run for the batch
        assert proj.runs("analyze")[0].manifest().status == "finished"


def test_apply_dispatch_dropin_model(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))            # creates a bundle dir at models/m1
        model_dir = proj.layout.model_dir("m1")
        seen = {}

        def fake_apply_videos(bundle, videos, out_root, **kw):
            seen["out_root"] = Path(out_root)
            return {str(v): Path(out_root) / "pose.parquet" for v in videos}

        monkeypatch.setattr(cli, "apply_to_videos", fake_apply_videos)
        v1 = Path(d) / "clip1.mp4"
        v1.write_bytes(b"v")
        code, out = _run(["apply", str(v1), "--model", str(model_dir), "--out", str(Path(d) / "preds")])
        assert code == 0 and "pose.parquet" in out
        assert seen["out_root"] == Path(d) / "preds"
        assert len(proj.runs("analyze")) == 0     # drop-in mode opens no project run


def test_train_dispatch(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        seen = {}

        class _FakeBundle:
            class card:  # noqa: N801
                model_id = "trained1"

        def fake_train(project, config, backend, **kw):
            seen["net"] = config.net_type
            seen["epochs"] = config.epochs
            seen["backend"] = type(backend).__name__
            return _FakeBundle()

        monkeypatch.setattr(cli, "train_model", fake_train)
        code, out = _run(["train", str(proj.root), "--net", "hrnet_w32", "--epochs", "3"])
        assert code == 0 and "trained -> models/trained1" in out
        assert seen == {"net": "hrnet_w32", "epochs": 3, "backend": "WorkspaceTrainBackend"}


def test_evaluate_dispatch(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        captured = {}

        def fake_eval(project, bundle, **kw):
            captured.update(kw)
            return {"n": 2, "mean_error": 2.5}

        monkeypatch.setattr(cli, "evaluate_model", fake_eval)
        code, out = _run(["evaluate", str(proj.root), "m1", "--pck", "5"])
        assert code == 0 and '"mean_error": 2.5' in out
        assert captured["pck_threshold"] == 5.0 and captured["pcutoff"] == 0.6


# ------------------------------------------------------------------ smoke runner
def test_label_dispatch_reads_sidecar(monkeypatch):
    from deeplabcut.workspace import label_video
    from deeplabcut.workspace.manifest import write_manifest
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        v = d / "clip.mp4"
        v.write_bytes(b"v")
        (d / "clip.fdlc.parquet").write_bytes(b"p")   # existence only; render is patched
        write_manifest(d / "clip.fdlc.toml",
                       {"bodyparts": ["snout", "tail"], "skeleton": [["snout", "tail"]]})
        seen = {}

        def fake_render(video, parquet, out_path, *, bodyparts, skeleton, pcutoff, **kw):
            seen.update(parquet=Path(parquet).name, out=Path(out_path).name,
                        bodyparts=bodyparts, skeleton=[list(e) for e in (skeleton or [])])
            Path(out_path).write_bytes(b"mp4")
            return Path(out_path)

        monkeypatch.setattr(label_video, "render_labeled_from_parquet", fake_render)
        code, out = _run(["label", str(v)])
        assert code == 0
        assert seen["parquet"] == "clip.fdlc.parquet"        # defaulted beside the video
        assert seen["out"] == "clip.fdlc.mp4"
        assert seen["bodyparts"] == ["snout", "tail"]        # from the .fdlc.toml sidecar
        assert seen["skeleton"] == [["snout", "tail"]]


def test_label_missing_parquet():
    with tempfile.TemporaryDirectory() as d:
        v = Path(d) / "clip.mp4"
        v.write_bytes(b"v")
        code, out = _run(["label", str(v)])
        assert code == 2 and "no pose parquet" in out       # clear error when the parquet is absent


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
    print(f"cli: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
