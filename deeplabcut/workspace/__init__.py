#
# FreeDLC workspace layer
#
"""FreeDLC workspace: a project/IO layer built on a clean, explicit layout.

The layout separates immutable inputs (``sources/``) from generated artifacts
(``runs/``, ``derived/``) and from portable model bundles (``models/``). Entity
identity lives in TOML manifests, not in directory or file names, and every
generating operation records reproducible provenance in a ``run.toml``.

Typical use::

    from deeplabcut import workspace as ws

    proj = ws.Project.create("/data/reaching", task="reaching", bodyparts=["snout", "paw"])
    vid = proj.add_video("/raw/session01.mp4")

    # A trained model is a portable bundle -- no project required to run it:
    bundle = ws.ModelBundle.open("/models/20260707-141530-a1b9c2")
    run = proj.new_run("analyze", model_id=bundle.card.model_id, inputs=[vid])
    ws.apply_to_video(bundle, proj.layout.video_media(vid), run.video_dir(vid))
    run.finish(outputs=["pose.parquet"])
"""
from __future__ import annotations

from . import ids
from .annotations import (
    collected_data_to_long_df,
    ingest_annotations,
    ingest_video_annotations,
    read_collected_data,
)
from .apply import (
    SINGLE_INDIVIDUAL,
    apply_to_video,
    apply_to_videos,
    collect_videos,
    predictions_to_long_df,
    write_pose_parquet,
)
from .coco_export import (
    export_coco_dataset,
    labels_to_coco,
    split_coco,
    workspace_to_dlc_project_dict,
)
from .evaluate import evaluate_model, infer_on_frames, read_labels
from .layout import Layout
from .metrics import pose_error
from .migrate import (
    LegacyModel,
    discover_legacy_models,
    legacy_config_to_project_config,
    legacy_video_paths,
    migrate_project,
    read_legacy_config,
)
from .model_bundle import ModelBundle
from .project import Project, Run
from .schema import (
    RUN_KINDS,
    SCHEMA_VERSION,
    ModelCard,
    ProjectConfig,
    RunManifest,
    VideoRecord,
)
from .train import TrainBackend, TrainConfig, WorkspaceTrainBackend, train_model

__all__ = [
    "ids",
    "Layout",
    "Project",
    "Run",
    "ModelBundle",
    "ProjectConfig",
    "VideoRecord",
    "ModelCard",
    "RunManifest",
    "SINGLE_INDIVIDUAL",
    "apply_to_video",
    "apply_to_videos",
    "collect_videos",
    "predictions_to_long_df",
    "write_pose_parquet",
    "migrate_project",
    "discover_legacy_models",
    "legacy_config_to_project_config",
    "legacy_video_paths",
    "read_collected_data",
    "collected_data_to_long_df",
    "ingest_annotations",
    "ingest_video_annotations",
    "read_legacy_config",
    "LegacyModel",
    "TrainConfig",
    "TrainBackend",
    "train_model",
    "WorkspaceTrainBackend",
    "export_coco_dataset",
    "labels_to_coco",
    "split_coco",
    "workspace_to_dlc_project_dict",
    "evaluate_model",
    "read_labels",
    "infer_on_frames",
    "pose_error",
    "RUN_KINDS",
    "SCHEMA_VERSION",
]
