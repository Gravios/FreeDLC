#
# FreeDLC workspace layer
#
"""Native PyTorch training, straight from a workspace.

Trains a model without any legacy ``config.yaml`` or ``dlc-models-pytorch/``
layout: it stages a COCO dataset from the workspace's ``labels.parquet`` +
frames, builds a fresh pose config for the requested architecture, and drives
DeepLabCut's own ``train(loader, ...)`` with a ``COCOLoader`` whose
``model_folder`` points at ``runs/train/<id>/train/`` (where snapshots are
written for the run to harvest into a bundle).

This mirrors the config handling of ``pose_estimation_pytorch.apis.train_network``
exactly (the epochs/batch/save overrides and the detector-then-pose train calls),
substituting the data source. Everything torch/DLC is imported lazily, so this
module loads without torch; the ``train_in_workspace`` compute path itself needs
a torch environment to run.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["train_in_workspace", "probe_image_dims"]


def probe_image_dims(project, video_ids) -> dict[str, tuple[int, int]]:
    """Map ``"<video_id>/<frame>" -> (width, height)`` by reading frame headers (PIL)."""
    from PIL import Image

    dims: dict[str, tuple[int, int]] = {}
    for vid in video_ids:
        frames = project.layout.frames_dir(vid)
        if not frames.is_dir():
            continue
        for f in sorted(frames.iterdir()):
            if not f.is_file():
                continue
            try:
                with Image.open(f) as im:
                    dims[f"{vid}/{f.name}"] = im.size
            except Exception:
                continue
    return dims


def train_in_workspace(project, run, config) -> Path:
    """Train a model natively; returns the ``train`` dir holding the snapshots.

    Requires torch. Stages ``runs/train/<id>/dataset/`` (COCO json + linked
    frames), writes the pose config and snapshots into ``runs/train/<id>/train/``.
    """
    from deeplabcut.pose_estimation_pytorch.apis import training as dlc_training
    from deeplabcut.pose_estimation_pytorch.config.make_pose_config import make_pytorch_pose_config
    from deeplabcut.pose_estimation_pytorch.data import COCOLoader
    from deeplabcut.pose_estimation_pytorch.task import Task

    from .coco_export import export_coco_dataset, workspace_to_dlc_project_dict

    dataset_dir = run.dir / "dataset"
    train_dir = run.dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    video_ids = project.annotated_videos()
    export_coco_dataset(
        project, dataset_dir, video_ids=video_ids,
        train_fraction=config.train_fraction, seed=config.seed or 0,
        image_dims=probe_image_dims(project, video_ids),
    )

    pose_config_path = train_dir / "pytorch_config.yaml"
    pose_cfg = make_pytorch_pose_config(
        workspace_to_dlc_project_dict(project.config),
        pose_config_path,
        net_type=config.net_type,
        top_down=config.top_down,
        save=True,
    )

    loader = COCOLoader(dataset_dir, model_config=pose_cfg,
                        train_json_filename="train.json", test_json_filename="test.json")

    # Apply TrainConfig overrides onto the config, exactly as train_network does.
    loader.model_cfg.train_settings.batch_size = config.batch_size
    loader.model_cfg.train_settings.epochs = config.epochs
    loader.model_cfg.runner.snapshots.save_epochs = config.save_epochs
    if config.seed is not None:
        loader.model_cfg.train_settings.seed = config.seed
    if config.top_down and loader.model_cfg.get("detector") is not None:
        loader.model_cfg.detector.train_settings.batch_size = config.detector_batch_size
        loader.model_cfg.detector.train_settings.epochs = config.detector_epochs

    pose_task = Task(loader.model_cfg.get("method", "bu"))
    if pose_task == Task.TOP_DOWN and loader.model_cfg["detector"]["train_settings"]["epochs"] > 0:
        detector_run_config = loader.model_cfg["detector"]
        detector_run_config["device"] = loader.model_cfg.get("device")
        dlc_training.train(
            loader=loader, run_config=detector_run_config, task=Task.DETECT, device=config.device,
        )

    if loader.model_cfg["train_settings"]["epochs"] > 0:
        dlc_training.train(
            loader=loader, run_config=loader.model_cfg, task=pose_task,
            device=config.device, logger_config=loader.model_cfg.get("logger"),
        )

    return train_dir
