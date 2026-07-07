#
# FreeDLC workspace layer
#
"""Evaluation, as a workspace run that scores a model against annotations.

``evaluate_model`` opens a ``runs/evaluate/<run_id>/`` run, gathers predictions on
each annotated video's labeled frames, compares them to the ground-truth
``labels.parquet`` via :mod:`~deeplabcut.workspace.metrics`, records the metrics on
both the run and the model card, and returns them.

Two seams are injectable (with lazy, torch/parquet-backed defaults) so the
orchestration and the metric computation are testable without a GPU:
``predictions_provider(project, video_id, ground_truth) -> long df`` and
``labels_provider(project, video_id) -> long df``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .manifest import update_manifest
from .metrics import pose_error

__all__ = ["read_labels", "infer_on_frames", "evaluate_model"]


def read_labels(project, video_id: str):
    """Default ground-truth provider: read ``sources/annotations/<id>/labels.parquet``."""
    import pandas as pd

    return pd.read_parquet(project.layout.labels_parquet(video_id))


def infer_on_frames(bundle, frames_dir, images, *, device: str | None = None, batch_size: int = 1):
    """Default predictions provider: run the bundle on labeled frames.

    Returns a long DataFrame (``image, individual, bodypart, x, y, likelihood``)
    keyed by frame filename. Requires torch (imported lazily via the bundle
    runner and analyze_images).
    """
    from pathlib import Path

    import pandas as pd

    from deeplabcut.pose_estimation_pytorch import analyze_images

    from .apply import predictions_to_long_df

    frames_dir = Path(frames_dir)
    image_names = list(images)
    paths = [str(frames_dir / name) for name in image_names]

    runner = bundle.build_pose_runner(device=device, batch_size=batch_size)
    detector = bundle.build_detector_runner(device=device) if bundle.card.top_down else None
    predictions = analyze_images(paths, runner, detector_runner=detector)

    meta = bundle._read_pose_config().get("metadata", {})
    frames = []
    for name, pred in zip(image_names, predictions, strict=False):
        long = predictions_to_long_df([pred], bundle.card.bodyparts,
                                      unique_bodyparts=meta.get("unique_bodyparts") or None)
        long["image"] = name
        long = long.drop(columns=["frame"], errors="ignore")
        frames.append(long)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_model(
    project,
    bundle,
    *,
    videos: Sequence[str] | None = None,
    predictions_provider: Callable | None = None,
    labels_provider: Callable | None = None,
    pcutoff: float | None = 0.6,
    pck_threshold: float | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate ``bundle`` on a project's annotated videos and return metrics.

    Records the metrics on the run's ``run.toml`` and (when ``write``) on the
    model card's ``model.toml``.
    """
    import pandas as pd

    labels_provider = labels_provider or read_labels
    if predictions_provider is None:
        def predictions_provider(project, video_id, ground_truth):
            images = list(dict.fromkeys(ground_truth["image"].tolist()))
            return infer_on_frames(bundle, project.layout.frames_dir(video_id), images)

    videos = list(videos) if videos is not None else project.annotated_videos()
    run = project.new_run("evaluate", model_id=bundle.card.model_id, inputs=videos)
    run.start()
    try:
        pred_frames, gt_frames = [], []
        for video_id in videos:
            gt = labels_provider(project, video_id)
            pred = predictions_provider(project, video_id, gt)
            gt_frames.append(gt)
            pred_frames.append(pred)
        predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        ground_truth = pd.concat(gt_frames, ignore_index=True) if gt_frames else pd.DataFrame()
        metrics = pose_error(predictions, ground_truth, pcutoff=pcutoff, pck_threshold=pck_threshold)
    except Exception:
        run.fail()
        raise

    update_manifest(run.manifest_path, metrics=metrics)
    run.finish()
    if write:
        bundle.set_metrics(metrics)
    return metrics
