#
# FreeDLC workspace layer -- compare tests
#
"""Tests for ``fdlc compare``: two models against the labels, two analyze runs with each other.

Inference is injected; the runs' poses are real parquet files on tiny cv2 videos.
No torch.

Standalone: ``python tests/workspace/test_compare.py`` -> ``compare: N/N checks passed``.
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
from deeplabcut.workspace.compare import (
    compare_models,
    compare_runs,
    format_model_comparison,
    format_run_comparison,
)
from deeplabcut.workspace.manifest import write_manifest
from deeplabcut.workspace.schema import LabelsRecord, VideoRecord

BODYPARTS = ["snout", "paw"]


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _raises(fn, needle):
    try:
        fn()
    except ValueError as err:
        assert needle in str(err), str(err)
        return
    raise AssertionError(f"expected ValueError: {needle}")


# --------------------------------------------------------------------- models
def _model(proj, d: Path, model_id: str, frames: str, trained_on):
    import yaml

    train = d / f"train-{model_id}"
    train.mkdir()
    (train / "pytorch_config.yaml").write_text(yaml.safe_dump(
        {"net_type": "resnet_50", "metadata": {"bodyparts": BODYPARTS, "unique_bodyparts": []}}))
    (train / "snapshot-best-010.pt").write_bytes(b"w")
    bundle = ws.ModelBundle.from_train_dir(proj.layout.model_dir(model_id), train, model_id=model_id,
                                           frames=frames)
    if trained_on is not None:
        bundle.write_train_frames(trained_on)
    return bundle


def _labeled_project(d: Path):
    """One video labeled in original pixels (100x50), with a processed counterpart of 10x5."""
    proj = ws.Project.create(d / "ws", task="reach", bodyparts=BODYPARTS)
    for kind, (w, h) in (("original", (100, 50)), ("processed", (10, 5))):
        rec = VideoRecord(video_id="v1", source_path=f"{kind}.mp4", width=w, height=h, link="reference")
        write_manifest(proj.layout.video_toml("v1", kind), rec.to_dict())
    write_manifest(proj.layout.labels_toml("v1"), LabelsRecord(video_id="v1", space="original").to_dict())
    images = [f"img{i:03d}.png" for i in range(6)]
    gt = pd.DataFrame([(im, "single", bp, 20.0 + i, 10.0) for i, im in enumerate(images) for bp in BODYPARTS],
                      columns=["image", "individual", "bodypart", "x", "y"])
    return proj, gt, images


def test_models_are_scored_on_the_frames_neither_trained_on():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj, gt, images = _labeled_project(d)
        a = _model(proj, d, "ma", "processed", [f"v1/{n}" for n in images[:2]])
        b = _model(proj, d, "mb", "original", [f"v1/{n}" for n in images[1:3]])
        asked = {}

        def predict(project, bundle, vid, kind, names):
            asked[bundle.card.model_id] = (kind, list(names))
            rows = gt[gt["image"].isin(names)].copy()
            if bundle.card.model_id == "ma":   # processed pixels: a tenth; off by 0.2 there = 2 px original
                rows["x"], rows["y"] = rows["x"] / 10 + 0.2, rows["y"] / 10
            else:                              # original pixels: snout exact, paw 6 px off
                rows["x"] = rows["x"] + np.where(rows["bodypart"] == "paw", 6.0, 0.0)
            return rows.assign(likelihood=0.9)

        def compare(x, y, **kw):
            return compare_models(proj, x, y, videos=["v1"], labels_provider=lambda p, v: gt,
                                  predictions_provider=predict, **kw)

        result = compare(a, b, pck_threshold=5.0)
        assert asked == {"ma": ("processed", images[3:]), "mb": ("original", images[3:])}
        assert result["frames"] == "original" and result["n_frames"] == 3 and result["n_labeled"] == 6
        ra, rb = result["a"], result["b"]
        assert ra["trained_on"] == 2 and abs(ra["all"]["median"] - 2.0) < 1e-6 and ra["all"]["pck"] == 1.0
        assert rb["per_bodypart"]["snout"]["median"] == 0.0 and rb["per_bodypart"]["paw"]["pck"] == 0.0
        closer = result["closer"]["per_bodypart"]
        assert closer["snout"] == {"n": 3, "a": 0.0, "b": 1.0} and closer["paw"]["a"] == 1.0
        text = "\n".join(format_model_comparison(result))
        assert "scored on 3 of 6 labeled frame(s), the ones neither model trained on" in text
        assert "errors in original pixels; PCK within 5 px" in text

        # in processed pixels the same errors are a tenth
        small = compare(a, b, frames="processed")
        assert abs(small["a"]["all"]["median"] - 0.2) < 1e-6

        # every frame, training frames included, on request only
        every = compare(a, b, all_frames=True)
        assert every["n_frames"] == 6 and "flatter both models" in format_model_comparison(every)[0]

        c = _model(proj, d, "mc", "original", [f"v1/{n}" for n in images[2:]])
        _raises(lambda: compare(a, c), "all 6 labeled frame(s) were trained on by model ma or mc")
        old = _model(proj, d, "mold", "original", None)
        _raises(lambda: compare(a, old), "model mold does not record which frames it trained on")
        _raises(lambda: compare(a, a), "both are model ma")


# ----------------------------------------------------------------------- runs
def _video(path: Path, size, n_frames=6) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()
    return path


def _analyze(proj, video: Path, poses, *, pcutoff=0.6, model_id=None):
    run = proj.new_run("analyze", model_id=model_id, params={"pcutoff": pcutoff})
    out = run.dir / video.stem
    out.mkdir(parents=True)
    poses.to_parquet(out / "pose.parquet")
    write_manifest(out / "run.toml", {"run_id": video.stem, "kind": "analyze", "inputs": [str(video)]})
    run.finish(outputs=[str(out / "pose.parquet")])
    return run


def _poses(rows):
    return pd.DataFrame(rows, columns=["frame", "individual", "bodypart", "x", "y", "likelihood"])


def test_runs_are_compared_marker_by_marker_and_their_disagreements_listed():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=BODYPARTS)
        original = _video(d / "raw" / "clip.mp4", (64, 48))
        processed = _video(d / "small" / "clip.mp4", (32, 24))
        proj.add_video(original, video_id="clip")
        proj.add_video(processed, video_id="clip", kind="processed")

        # A on the processed video (half size), B on the original
        rows_a, rows_b = [], []
        for f in range(6):
            rows_a += [(f, "single", "snout", 10.0, 10.0, 0.9), (f, "single", "paw", 5.0, 5.0, 0.9)]
            far = 20.0 if f == 4 else 0.0            # frame 4: B puts the snout 20 px away
            rows_b += [(f, "single", "snout", 20.0 + far, 20.0, 0.9),
                       (f, "single", "paw", 10.0, 10.0, 0.3 if f in (1, 2) else 0.9)]   # paw unsure in 1, 2
        run_a = _analyze(proj, processed, _poses(rows_a), model_id="ma")
        run_b = _analyze(proj, original, _poses(rows_b), pcutoff=0.5, model_id="mb")

        result = compare_runs(proj, run_a, run_b, top=2)
        assert result["frames"] == "original" and result["pcutoff"] == [0.6, 0.5]
        snout, paw = result["per_bodypart"]["snout"], result["per_bodypart"]["paw"]
        assert snout["median"] == 0.0 and snout["shown_a"] == 1.0 and abs(snout["differ"] - 1 / 6) < 1e-9
        assert abs(paw["shown_b"] - 4 / 6) < 1e-9 and paw["n_both"] == 4 and abs(paw["differ"] - 2 / 6) < 1e-9
        clip = result["videos"]["clip"]
        assert clip["n_frames"] == 6 and clip["n_differ"] == 3
        assert [f["frame"] for f in clip["frames"]] == [1, 4]       # one per half of the disagreeing frames
        assert clip["frames"][1]["distance"] == 10.0                # mean of snout 20 px and paw 0
        assert abs(clip["frames"][1]["time"] - 0.4) < 1e-9

        text = "\n".join(format_run_comparison(result))
        assert "A = run " in text and "(model ma), pcutoff 0.6" in text
        assert "clip: the runs differ on 3 of 6 frame(s)" in text and "0:00.40" in text

        # in processed pixels, distances halve (so does the --far it takes to differ)
        half = compare_runs(proj, run_a, run_b, frames="processed", far=5)
        assert [f["distance"] for f in half["videos"]["clip"]["frames"] if f["frame"] == 4] == [5.0]
        # one pcutoff for both
        assert compare_runs(proj, run_a, run_b, pcutoff=0.2)["per_bodypart"]["paw"]["differ"] == 0.0
        _raises(lambda: compare_runs(proj, run_a, run_a), "both are run")


def test_the_cli_tells_models_from_runs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj, _, images = _labeled_project(d)
        _model(proj, d, "20260101-000000-aaaaaa", "original", [])
        run = proj.new_run("analyze")
        run.finish()
        ws_ = str(proj.root)
        code, out = _run(["compare", "20260101", run.run_id, "--project", ws_])
        assert code == 2 and "is a model and" in out and "compare two models or two runs" in out
        code, out = _run(["compare", "nope", "latest", "--project", ws_])
        assert code == 2 and "'nope' is neither a model nor an analyze run" in out
        code, out = _run(["compare", "20260101", "20260101", "--project", ws_, "--far", "3"])
        assert code == 2 and "--far only applies when comparing runs" in out
        code, out = _run(["compare", "20260101", "20260101", "--project", ws_])
        assert code == 2 and "both are model 20260101-000000-aaaaaa" in out
        assert cli._no_nan({"a": [float("nan"), 1.0], "b": {"c": float("nan")}}) == {"a": [None, 1.0],
                                                                                     "b": {"c": None}}


def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"compare: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
