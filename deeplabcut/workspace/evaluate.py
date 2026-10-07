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

__all__ = ["read_labels", "scale_labels", "infer_on_frames", "evaluate_model"]


def read_labels(project, video_id: str):
    """Default ground-truth provider: read ``sources/annotations/<id>/labels.parquet``."""
    import pandas as pd

    return pd.read_parquet(project.layout.labels_parquet(video_id))


def scale_labels(df, scale: tuple[float, float]):
    """Return ``df`` with ``x``/``y`` multiplied by ``scale`` (``df`` itself for identity)."""
    scale_x, scale_y = scale
    if (scale_x, scale_y) == (1.0, 1.0):
        return df
    df = df.copy()
    df["x"] = df["x"] * scale_x
    df["y"] = df["y"] * scale_y
    return df


def infer_on_frames(bundle, frames_dir, images, *, device: str | None = None, batch_size: int = 1):
    """Default predictions provider: run the bundle on labeled frames.

    Returns a long DataFrame (``image, individual, bodypart, x, y, likelihood``)
    keyed by frame filename. Requires torch (imported lazily via the bundle's
    runners). The runners are driven directly, the way DeepLabCut's own image
    analysis does: a top-down model first detects boxes, and the pose runner then
    takes ``(image, boxes)`` pairs instead of bare image paths.
    """
    from pathlib import Path

    import pandas as pd

    from .apply import predictions_to_long_df

    frames_dir = Path(frames_dir)
    image_names = list(images)
    paths = [str(frames_dir / name) for name in image_names]

    runner = bundle.build_pose_runner(device=device, batch_size=batch_size)
    detector = bundle.build_detector_runner(device=device) if bundle.card.top_down else None
    pose_inputs = paths
    if detector is not None:
        pose_inputs = list(zip(paths, detector.inference(images=paths), strict=True))
    predictions = runner.inference(pose_inputs)

    meta = bundle._read_pose_config().get("metadata", {})
    frames = []
    for name, pred in zip(image_names, predictions, strict=True):
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
    frames: str | None = None,
) -> dict[str, Any]:
    """Evaluate ``bundle`` on a project's annotated videos and return metrics.

    ``frames`` is the frame set to score on (``"original"`` | ``"processed"``). It
    defaults to the one the model was trained on, as recorded on its card, and for
    a model with no record to the set each video's labels are stored in. The
    ground truth is converted into that pixel space, so errors are in its pixels.

    Records the metrics on the run's ``run.toml`` and (when ``write``) on the
    model card's ``model.toml``.

    Raises:
        ValueError: if a video's labels cannot be put on ``frames``.
    """
    import pandas as pd

    frames = frames or getattr(bundle.card, "frames", None)
    labels_provider = labels_provider or read_labels
    if predictions_provider is None:
        def predictions_provider(project, video_id, ground_truth):
            images = list(dict.fromkeys(ground_truth["image"].tolist()))
            kind = frames or project.label_frames_kind(video_id)
            return infer_on_frames(bundle, project.layout.frames_dir(video_id, kind), images)

    videos = list(videos) if videos is not None else project.annotated_videos()
    project.check_frames(videos, frames)
    run = project.new_run("evaluate", model_id=bundle.card.model_id, inputs=videos,
                          params={"frames": frames} if frames else None)
    run.start()
    try:
        pred_frames, gt_frames = [], []
        for video_id in videos:
            gt = scale_labels(labels_provider(project, video_id), project.labels_scale_to(video_id, frames))
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
