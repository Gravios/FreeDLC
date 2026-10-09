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


def test_pose_error_reports_medians_and_per_bodypart_pck():
    """One lost prediction must not hide that a marker is otherwise exact."""
    from deeplabcut.workspace.metrics import pose_error

    images = [f"i{k}" for k in range(5)]
    gt = pd.DataFrame({"image": images * 2, "individual": "single",
                       "bodypart": ["snout"] * 5 + ["paw"] * 5, "x": 100.0, "y": 100.0})
    # snout: 2 px off in four frames, 500 px off in one; paw: 10 px off everywhere
    pred = gt.assign(x=[102.0, 102.0, 102.0, 102.0, 600.0] + [110.0] * 5, likelihood=0.9)
    m = pose_error(pred, gt, pcutoff=0.6, pck_threshold=20.0)
    assert abs(m["per_bodypart"]["snout"] - 101.6) < 1e-9          # the mean is dominated by one miss
    assert m["per_bodypart_median"] == {"paw": 10.0, "snout": 2.0}  # the median is not
    assert m["per_bodypart_pck"] == {"paw": 1.0, "snout": 0.8}
    assert m["median_error"] == 10.0 and m["median_error_confident"] == 10.0 and m["pck"] == 0.9


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
        # the frame set trained on is recorded on the run and the model card
        assert ws.TrainConfig().frames == "processed"
        assert runs[0].manifest().params["frames"] == "processed"
        assert ws.ModelBundle.open(proj.layout.model_dir(bundle.card.model_id)).card.frames == "processed"


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


def test_evaluate_scores_on_the_frames_the_model_was_trained_on():
    from deeplabcut.workspace import evaluate
    from deeplabcut.workspace.manifest import write_manifest
    from deeplabcut.workspace.schema import LabelsRecord, VideoRecord

    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        for kind, (w, h) in (("original", (1920, 1080)), ("processed", (192, 108))):
            rec = VideoRecord(video_id="v1", source_path=f"{kind}.mp4", width=w, height=h, link="reference")
            write_manifest(proj.layout.video_toml("v1", kind), rec.to_dict())
        write_manifest(proj.layout.labels_toml("v1"),
                       LabelsRecord(video_id="v1", space="processed", scale_x=0.1, scale_y=0.1).to_dict())

        def bundle_trained_on(frames, model_id):
            return ws.ModelBundle.from_train_dir(
                proj.layout.model_dir(model_id), _fake_train_dir(Path(d) / model_id), model_id=model_id,
                frames=frames)

        gt = pd.DataFrame({"image": ["i1", "i1"], "individual": ["single", "single"],
                           "bodypart": ["snout", "paw"], "x": [10.0, 20.0], "y": [5.0, 5.0]})
        seen = []

        def fake_infer(bundle, frames_dir, images):
            seen.append(Path(frames_dir))
            return gt.assign(x=gt["x"] * factor, y=gt["y"] * factor, likelihood=1.0)

        real, evaluate.infer_on_frames = evaluate.infer_on_frames, fake_infer
        try:
            factor = 10.0   # a model trained on originals predicts in original pixels
            m = ws.evaluate_model(proj, bundle_trained_on("original", "m-orig"), videos=["v1"],
                                  labels_provider=lambda p, v: gt, write=False)
            assert m["mean_error"] < 1e-9                       # ground truth was scaled up to match
            factor = 1.0
            m = ws.evaluate_model(proj, bundle_trained_on("processed", "m-proc"), videos=["v1"],
                                  labels_provider=lambda p, v: gt, write=False)
            assert m["mean_error"] < 1e-9
            # an explicit choice overrides the card
            ws.evaluate_model(proj, bundle_trained_on("processed", "m-proc2"), videos=["v1"],
                              labels_provider=lambda p, v: gt, write=False, frames="original")
        finally:
            evaluate.infer_on_frames = real
        lay = proj.layout
        assert seen == [lay.frames_dir("v1", "original"), lay.frames_dir("v1", "processed"),
                        lay.frames_dir("v1", "original")]
        # each run records the set it scored on (run ids made in one second are unordered)
        recorded = sorted(run.manifest().params["frames"] for run in proj.runs("evaluate"))
        assert recorded == ["original", "original", "processed"]


def test_infer_on_frames_drives_the_runners():
    """Default predictions: bottom-up feeds image paths, top-down feeds (image, boxes) pairs."""
    import numpy as np

    from deeplabcut.workspace import evaluate

    class _Runner:
        def __init__(self, out):
            self.out, self.seen = out, None

        def inference(self, images):
            self.seen = list(images)
            return [self.out] * len(self.seen)

    class _Card:
        bodyparts = ["snout", "paw"]

        def __init__(self, top_down):
            self.top_down = top_down

    class _Bundle:
        def __init__(self, top_down):
            self.card = _Card(top_down)
            self.pose = _Runner({"bodyparts": np.array([[[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]]])})
            self.detector = _Runner({"bboxes": np.zeros((1, 4))})

        def build_pose_runner(self, **kwargs):
            return self.pose

        def build_detector_runner(self, **kwargs):
            return self.detector

        def _read_pose_config(self):
            return {"metadata": {}}

    frames = Path("/frames")
    paths = [str(frames / "i1.png"), str(frames / "i2.png")]

    bottom_up = _Bundle(top_down=False)
    df = evaluate.infer_on_frames(bottom_up, frames, ["i1.png", "i2.png"])
    assert bottom_up.pose.seen == paths and bottom_up.detector.seen is None
    assert list(df.columns) == ["individual", "bodypart", "x", "y", "likelihood", "image"]
    assert df["image"].tolist() == ["i1.png", "i1.png", "i2.png", "i2.png"]
    assert df[df.bodypart == "paw"].iloc[0][["x", "y"]].tolist() == [3.0, 4.0]

    top_down = _Bundle(top_down=True)
    evaluate.infer_on_frames(top_down, frames, ["i1.png", "i2.png"])
    assert top_down.detector.seen == paths
    assert [image for image, _boxes in top_down.pose.seen] == paths
    assert all("bboxes" in boxes for _image, boxes in top_down.pose.seen)


# ------------------------------------------------------------------ smoke runner
def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"train_eval: {len(checks)}/{len(checks)} checks passed")
    return 0


# --------------------------------------------------------- continuing a model
def _card(**kw):
    from types import SimpleNamespace

    base = dict(model_id="m1", architecture="resnet_50", bodyparts=["snout", "paw"], frames="processed",
                top_down=False)
    return SimpleNamespace(card=SimpleNamespace(**{**base, **kw}))


def test_fine_tune_source_must_fit_the_new_model():
    from deeplabcut.workspace.native_train import check_fine_tune_source

    ok = dict(net_type="resnet_50", bodyparts=["snout", "paw"], frames="processed", top_down=False)
    check_fine_tune_source(_card(), **ok)                                   # fits: no error
    for card, needle in ((_card(architecture="hrnet_w32"), "it is a hrnet_w32 and this run trains a resnet_50"),
                         (_card(bodyparts=["paw", "snout"]), "markers ['paw', 'snout'] are not this project's"),
                         (_card(frames="original"), "learned on the original frames"),
                         (_card(top_down=True), "top-down models cannot be continued")):
        try:
            check_fine_tune_source(card, **ok)
        except ValueError as err:
            assert needle in str(err) and str(err).startswith("cannot continue model m1"), str(err)
        else:
            raise AssertionError(f"expected ValueError: {needle}")


def test_fine_tune_schedule_starts_lower_and_steps_down_late():
    from deeplabcut.workspace.native_train import _fine_tune_schedule

    cfg = {"runner": {"optimizer": {"type": "AdamW", "params": {"lr": 5e-4}},
                      "scheduler": {"type": "LRListScheduler",
                                    "params": {"lr_list": [[1e-4], ["1e-05"]], "milestones": [90, 120]}}}}
    assert _fine_tune_schedule(cfg, 50) == "lr 0.0001, then 1e-05 from epoch 38"
    assert cfg["runner"]["optimizer"]["params"]["lr"] == 1e-4
    assert cfg["runner"]["scheduler"]["params"] == {"lr_list": [[1e-5]], "milestones": [38]}
    other = {"runner": {"optimizer": {"params": {"lr": 1e-3}}, "scheduler": {"type": "CosineAnnealing"}}}
    assert _fine_tune_schedule(other, 50) is None and other["runner"]["optimizer"]["params"]["lr"] == 1e-3


def test_train_from_model_takes_its_architecture_and_a_shorter_run():
    from deeplabcut.workspace import cli

    seen = {}

    def fake_train(project, config, backend, **kw):
        seen.update(net=config.net_type, epochs=config.epochs, from_model=config.from_model)
        return type("B", (), {"card": type("C", (), {"model_id": "new1"})})()

    real, cli.train_model = cli.train_model, fake_train
    try:
        _train_from_model_checks(cli, seen)
    finally:
        cli.train_model = real


def _train_from_model_checks(cli, seen):
    import contextlib
    import io

    from deeplabcut.workspace.model_bundle import ModelBundle

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout", "paw"])
        cfg, snap = d / "pytorch_config.yaml", d / "snapshot-050.pt"
        cfg.write_text("net_type: hrnet_w32\nmetadata:\n  bodyparts: [snout, paw]\n")
        snap.write_bytes(b"w")
        ModelBundle.create(proj.layout.model_dir("m1"), pose_config_src=cfg, snapshot_src=snap,
                           architecture="hrnet_w32", bodyparts=["snout", "paw"], model_id="m1")

        def run(*argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                return cli.main(["train", str(proj.root), *argv]), buf.getvalue()

        assert run("--from-model", "m1")[0] == 0
        assert seen == {"net": "hrnet_w32", "epochs": 50, "from_model": "m1"}
        assert run("--from-model", "m1", "--epochs", "30")[0] == 0 and seen["epochs"] == 30
        assert run()[0] == 0 and seen == {"net": "resnet_50", "epochs": 200, "from_model": None}
        code, out = run("--from-model", "nope")
        assert code == 2 and "no model.toml" in out and "m1" in out
        n_runs = len(proj.runs("train"))
        code, out = run("--from-model", "m1", "--net", "resnet_50")    # refused before a run is opened
        assert code == 2 and "it is a hrnet_w32 and this run trains a resnet_50" in out
        assert len(proj.runs("train")) == n_runs and "net" in seen


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
