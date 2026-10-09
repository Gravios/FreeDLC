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

__all__ = ["TRAIN_LOG", "train_in_workspace", "probe_image_dims"]

#: Training log written into a run's ``train/`` directory (DeepLabCut's own name for it).
TRAIN_LOG = "train.txt"


def probe_image_dims(project, video_ids, frames: str | None = None) -> dict[str, tuple[int, int]]:
    """Map ``"<video_id>/<frame>" -> (width, height)`` by reading frame headers (PIL).

    Reads the frame set the dataset export stages -- ``frames``, or per video the
    one its labels are stored in when ``None`` -- so the sizes describe the images
    the trainer actually loads.
    """
    from PIL import Image

    dims: dict[str, tuple[int, int]] = {}
    for vid in video_ids:
        frames_dir = project.layout.frames_dir(vid, frames or project.label_frames_kind(vid))
        if not frames_dir.is_dir():
            continue
        for f in sorted(frames_dir.iterdir()):
            if not f.is_file():
                continue
            try:
                with Image.open(f) as im:
                    dims[f"{vid}/{f.name}"] = im.size
            except Exception:
                continue
    return dims


#: share of a fine-tuning run spent at the starting learning rate before the last step down
FINE_TUNE_DECAY_AT = 0.75


def check_fine_tune_source(bundle, *, net_type: str, bodyparts: list[str], frames: str | None,
                           top_down: bool) -> None:
    """Refuse to start from ``bundle`` when its weights cannot fit the model being trained.

    Raises:
        ValueError: naming every mismatch -- architecture, markers (their number and
            order fix the output layer), frame set (the resolution it learned at), or
            a top-down model, which is not supported.
    """
    card = bundle.card
    problems = []
    if card.top_down or top_down:
        problems.append("top-down models cannot be continued yet; train without --from-model")
    if card.architecture != net_type:
        problems.append(f"it is a {card.architecture} and this run trains a {net_type} (drop --net, or give "
                        f"--net {card.architecture})")
    if list(card.bodyparts) != list(bodyparts):
        problems.append(f"its markers {list(card.bodyparts)} are not this project's {list(bodyparts)}")
    if card.frames and frames and card.frames != frames:
        problems.append(f"it learned on the {card.frames} frames and this run uses the {frames} frames "
                        f"(add --frames {card.frames})")
    if problems:
        raise ValueError(f"cannot continue model {card.model_id}:\n  " + "\n  ".join(problems))


def fine_tune_snapshot(bundle, dest: Path) -> Path:
    """Write the weights of ``bundle``'s default snapshot to ``dest``, and nothing else.

    DeepLabCut resumes from a snapshot: it restores the optimizer, the learning-rate
    schedule and the epoch counter, so a converged model would carry on at its final,
    decayed rate. A snapshot holding only the weights (and epoch 0) makes it start a
    new run from them instead.
    """
    from deeplabcut.pose_estimation_pytorch.runners.base import attempt_snapshot_load

    snapshot = attempt_snapshot_load(bundle.snapshot_path(), "cpu")
    import torch

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": snapshot["model"], "metadata": {"epoch": 0}}, dest)
    return dest


def _fine_tune_schedule(model_cfg, epochs: int) -> str | None:
    """Fit the learning-rate schedule to a fine-tuning run of ``epochs``; describe it, or None.

    The default schedule steps 5e-4 -> 1e-4 -> 1e-5 at fixed epochs of a 200-epoch
    run. A model that has learned already starts at the middle rate (1e-4) and
    steps down to the last (1e-5) after :data:`FINE_TUNE_DECAY_AT` of the run.
    Other schedules are left as they are.
    """
    runner = model_cfg["runner"]
    scheduler = runner.get("scheduler") or {}
    if scheduler.get("type") != "LRListScheduler":
        return None
    lr_list = scheduler["params"]["lr_list"]
    start, end = float(lr_list[0][0]), float(lr_list[-1][0])
    at = max(1, round(FINE_TUNE_DECAY_AT * epochs))
    runner["optimizer"]["params"]["lr"] = start
    scheduler["params"]["milestones"] = [at]
    scheduler["params"]["lr_list"] = [[end]]
    return f"lr {start:g}, then {end:g} from epoch {at}"


def train_in_workspace(project, run, config) -> Path:
    """Train a model natively; returns the ``train`` dir holding the snapshots.

    Requires torch. Trains on the frame set ``config.frames`` names, with the labels
    converted into its pixel space. Stages ``runs/train/<id>/dataset/`` (COCO json + linked
    frames), writes the pose config and snapshots into ``runs/train/<id>/train/``.
    Progress is logged to the console and to ``runs/train/<id>/train/train.txt``.
    """
    from deeplabcut.pose_estimation_pytorch.apis import training as dlc_training
    from deeplabcut.pose_estimation_pytorch.config.make_pose_config import make_pytorch_pose_config
    from deeplabcut.pose_estimation_pytorch.data import COCOLoader
    from deeplabcut.pose_estimation_pytorch.runners.logger import destroy_file_logging, setup_file_logging
    from deeplabcut.pose_estimation_pytorch.task import Task

    from .coco_export import TEST_JSON, TRAIN_JSON, export_coco_dataset, workspace_to_dlc_project_dict
    from .util import files_log, shown

    # Absolute: the loader rewrites each image's file_name to <dataset>/images/<name>
    # every time it loads the data, and only leaves absolute paths alone. With a
    # relative project root a second load would prefix the paths twice.
    run_dir = run.dir.resolve()
    dataset_dir = run_dir / "dataset"
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    video_ids = project.annotated_videos()
    files_log.info("project %s", shown(project.layout.root))
    files_log.info("frame set: %s", config.frames or "as stored, per video")
    export_coco_dataset(
        project, dataset_dir, video_ids=video_ids,
        train_fraction=config.train_fraction, seed=config.seed or 0,
        image_dims=probe_image_dims(project, video_ids, config.frames), frames=config.frames,
    )

    pose_config_path = train_dir / "pytorch_config.yaml"
    pose_cfg = make_pytorch_pose_config(
        workspace_to_dlc_project_dict(project.config),
        pose_config_path,
        net_type=config.net_type,
        top_down=config.top_down,
        save=True,
    )

    init_snapshot = None
    if config.from_model:
        from . import _snapshots
        from .model_bundle import ModelBundle

        source = ModelBundle.from_project(project, config.from_model)
        check_fine_tune_source(source, net_type=config.net_type, bodyparts=_snapshots.read_bodyparts(pose_config_path),
                               frames=config.frames, top_down=config.top_down)
        init_snapshot = fine_tune_snapshot(source, train_dir / f"init-from-{source.card.model_id}.pt")
        files_log.info("starting from model %s", shown(source.path))
        files_log.info("  snapshot  %s (weights only)", shown(source.snapshot_path()))

    loader = COCOLoader(dataset_dir, model_config=pose_cfg,
                        train_json_filename=TRAIN_JSON, test_json_filename=TEST_JSON)

    # Check what the loader will actually feed the trainer. It filters annotations by
    # its own rules, and a dataset it has emptied still "trains" -- every target is
    # blank, the loss falls to zero, and the model learns to predict nothing.
    n_images = len(loader.train_json["images"])
    n_labeled = len({a["image_id"] for a in loader.load_data("train")["annotations"]})
    if n_labeled < n_images:
        raise ValueError(
            f"the loader kept annotations for only {n_labeled} of {n_images} training image(s) "
            f"in {dataset_dir}; refusing to train on unlabeled images"
        )

    loader.model_cfg.train_settings.batch_size = config.batch_size
    loader.model_cfg.train_settings.epochs = config.epochs
    loader.model_cfg.runner.snapshots.save_epochs = config.save_epochs
    if config.seed is not None:
        loader.model_cfg.train_settings.seed = config.seed
    if init_snapshot is not None and (schedule := _fine_tune_schedule(loader.model_cfg, config.epochs)):
        files_log.info("  schedule  %s", schedule)
    if config.rotate180:
        loader.model_cfg["data"]["train"]["rotate180"] = float(config.rotate180)
        files_log.info("augmentation: half turn with probability %g", config.rotate180)
    if config.top_down and loader.model_cfg.get("detector") is not None:
        loader.model_cfg.detector.train_settings.batch_size = config.detector_batch_size
        loader.model_cfg.detector.train_settings.epochs = config.detector_epochs

    # DeepLabCut reports training progress (per-epoch losses, evaluation metrics)
    # through the root logger and leaves it to the entry point to attach handlers --
    # train_network() does it for the legacy layout. Do the same here, or the run is
    # silent: progress goes to the console and to train/train.txt, as in DeepLabCut.
    files_log.info("training in %s", train_dir)
    files_log.info("  config     %s", pose_config_path)
    files_log.info("  log        %s", train_dir / TRAIN_LOG)
    files_log.info("  stats      %s", train_dir / "learning_stats.csv")
    files_log.info("  snapshots  %s", train_dir / "snapshot-*.pt")
    setup_file_logging(train_dir / TRAIN_LOG)
    try:
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
                snapshot_path=init_snapshot,
            )
    finally:
        destroy_file_logging()

    return train_dir
