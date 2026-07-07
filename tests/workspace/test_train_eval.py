#
# FreeDLC workspace layer -- train / evaluate / metrics tests
#
"""Tests for deeplabcut.workspace.{metrics, train, evaluate}.

The compute seams (the training backend, the inference-based predictions
provider) are injected with fakes; the orchestration (run lifecycle, bundle
harvesting, metric recording) and the metric math run for real. No torch/pyarrow.

Standalone: ``python tests/workspace/test_train_eval.py`` -> ``train_eval: N/N checks passed``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from deeplabcut import workspace as ws
from deeplabcut.workspace import ids


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh)


def _fake_train_dir(root: Path, *, net="resnet_50", bodyparts=("snout", "paw"),
                    pose=("snapshot-050.pt", "snapshot-best-100.pt"), detector=()):
    root.mkdir(parents=True, exist_ok=True)
    _write_yaml(root / "pytorch_config.yaml",
                {"net_type": net, "metadata": {"bodyparts": list(bodyparts), "unique_bodyparts": []}})
    for name in list(pose) + list(detector):
        (root / name).write_bytes(b"w")
    return root


# ------------------------------------------------------------------- metrics
def test_pose_error_math():
    gt = pd.DataFrame({"image": ["i1", "i1"], "individual": ["single", "single"],
                       "bodypart": ["snout", "paw"], "x": [0.0, 10.0], "y": [0.0, 0.0]})
    pred = pd.DataFrame({"image": ["i1", "i1"], "individual": ["single", "single"],
                         "bodypart": ["snout", "paw"], "x": [3.0, 10.0], "y": [4.0, 0.0],
                         "likelihood": [0.9, 0.5]})
    m = ws.pose_error(pred, gt, pcutoff=0.6, pck_threshold=6.0)
    assert m["n"] == 2
    assert abs(m["mean_error"] - 2.5) < 1e-9          # (5 + 0) / 2
    assert abs(m["rmse"] - (12.5 ** 0.5)) < 1e-9      # sqrt((25 + 0) / 2)
    assert abs(m["per_bodypart"]["snout"] - 5.0) < 1e-9 and m["per_bodypart"]["paw"] == 0.0
    assert m["n_confident"] == 1 and m["mean_error_confident"] == 5.0  # only snout >= 0.6
    assert m["pck"] == 1.0                             # both within 6px


def test_pose_error_ignores_unlabeled():
    gt = pd.DataFrame({"image": ["i1"], "individual": ["single"], "bodypart": ["snout"],
                       "x": [float("nan")], "y": [float("nan")]})
    pred = pd.DataFrame({"image": ["i1"], "individual": ["single"], "bodypart": ["snout"],
                         "x": [3.0], "y": [4.0], "likelihood": [0.9]})
    assert ws.pose_error(pred, gt)["n"] == 0


# ------------------------------------------------------------- from_train_dir
def test_bundle_from_train_dir():
    with tempfile.TemporaryDirectory() as d:
        td = _fake_train_dir(Path(d) / "train")
        b = ws.ModelBundle.from_train_dir(Path(d) / "bundle", td, model_id="m1")
        assert b.card.architecture == "resnet_50" and b.card.bodyparts == ["snout", "paw"]
        assert "best" in b.card.default_snapshot            # best preferred as default
        assert (b.snapshots_dir / "pose-snapshot-050.pt").exists()  # other snapshot preserved
        assert not b.card.top_down


def test_bundle_from_train_dir_top_down():
    with tempfile.TemporaryDirectory() as d:
        td = _fake_train_dir(Path(d) / "train", detector=("snapshot-detector-best-020.pt",))
        b = ws.ModelBundle.from_train_dir(Path(d) / "bundle", td)
        assert b.card.top_down and b.detector_snapshot_path().exists()


# --------------------------------------------------------------------- train
def test_train_model_success():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])

        def backend(project, run, config):
            return _fake_train_dir(run.dir / "train", net=config.net_type)

        bundle = ws.train_model(proj, ws.TrainConfig(net_type="hrnet_w32", epochs=1), backend)
        assert bundle.card.architecture == "hrnet_w32"
        assert ids.is_id(bundle.card.model_id) and ids.is_id(bundle.card.train_run_id)
        runs = proj.runs("train")
        assert len(runs) == 1 and runs[0].manifest().status == "finished"
        assert runs[0].manifest().params["epochs"] == 1
        assert proj.models() == [bundle.card.model_id]


def test_train_model_backend_failure_marks_run_failed():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout"])

        def bad_backend(project, run, config):
            raise RuntimeError("boom")

        try:
            ws.train_model(proj, ws.TrainConfig(), bad_backend)
        except RuntimeError:
            pass
        else:
            raise AssertionError("failure should propagate")
        assert proj.runs("train")[0].manifest().status == "failed"
        assert proj.models() == []  # no bundle created


def test_train_config():
    c = ws.TrainConfig(detector_epochs=5)
    assert c.top_down and c.to_dict()["detector_epochs"] == 5
    assert not ws.TrainConfig().top_down


# ------------------------------------------------------------------ evaluate
def test_annotated_videos_detection():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout"])
        proj.layout.annotation_dir("v1").mkdir(parents=True)
        proj.layout.labels_parquet("v1").write_bytes(b"")  # existence is all that's checked
        proj.layout.annotation_dir("v2").mkdir(parents=True)  # no labels -> not annotated
        assert proj.annotated_videos() == ["v1"]


def test_evaluate_model_with_injected_providers():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        cfg = Path(d) / "pytorch_config.yaml"
        _write_yaml(cfg, {"net_type": "resnet_50", "metadata": {"bodyparts": ["snout", "paw"]}})
        snap = Path(d) / "snapshot-050.pt"
        snap.write_bytes(b"w")
        bundle = ws.ModelBundle.create(proj.layout.model_dir("m1"), pose_config_src=cfg,
                                       snapshot_src=snap, architecture="resnet_50",
                                       bodyparts=["snout", "paw"], model_id="m1")

        gt = pd.DataFrame({"image": ["i1", "i1"], "individual": ["single", "single"],
                           "bodypart": ["snout", "paw"], "x": [0.0, 10.0], "y": [0.0, 0.0]})
        pred = gt.assign(x=[3.0, 10.0], y=[4.0, 0.0], likelihood=[0.9, 0.5])

        metrics = ws.evaluate_model(
            proj, bundle, videos=["v1"],
            labels_provider=lambda p, v: gt,
            predictions_provider=lambda p, v, g: pred,
            pcutoff=0.6, pck_threshold=6.0,
        )
        assert abs(metrics["mean_error"] - 2.5) < 1e-9 and metrics["pck"] == 1.0

        run = proj.runs("evaluate")[0]
        assert run.manifest().status == "finished" and run.manifest().metrics["n"] == 2
        assert run.manifest().model_id == "m1"
        # metrics were written back onto the model card
        assert abs(ws.ModelBundle.open(proj.layout.model_dir("m1")).card.metrics["mean_error"] - 2.5) < 1e-9


# ------------------------------------------------------------------ smoke runner
def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"train_eval: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
