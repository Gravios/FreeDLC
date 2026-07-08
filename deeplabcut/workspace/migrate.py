#
# FreeDLC workspace layer
#
"""Migrate a legacy DeepLabCut project into the new workspace layout.

Reads a classic project (``config.yaml`` + ``videos/`` + ``dlc-models-pytorch/``)
and writes a workspace: ``project.toml``, registered source videos, and one
portable :class:`~deeplabcut.workspace.model_bundle.ModelBundle` per trained
(iteration, shuffle) found under ``dlc-models-pytorch/``. The legacy coordinates
(iteration / shuffle / train fraction) are recorded in each model's
``model.toml`` under ``[legacy]`` -- as provenance, not as path identity.

Annotation ingest (``labeled-data`` -> ``sources/annotations``) is a separate
step; see :mod:`~deeplabcut.workspace.annotations`.

Only PyYAML is needed for reading legacy configs; imported lazily so the
workspace layer stays torch/yaml-free at load time.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _snapshots, ids
from .model_bundle import ModelBundle
from .project import Project
from .schema import ProjectConfig

__all__ = [
    "LegacyModel",
    "read_legacy_config",
    "legacy_config_to_project_config",
    "discover_legacy_models",
    "legacy_video_paths",
    "migrate_project",
]

log = logging.getLogger(__name__)

LEGACY_MODELS_DIR = "dlc-models-pytorch"

# Parses "<Task><date>-trainset95shuffle1" -> (95, 1).
_TRAINSET_SHUFFLE_RE = re.compile(r"trainset(\d+)shuffle(\d+)$")
_ITERATION_RE = re.compile(r"iteration-(\d+)")


# ---------------------------------------------------------------- config reading
def read_legacy_config(path: str | Path) -> dict[str, Any]:
    """Load a legacy ``config.yaml`` into a dict (PyYAML, imported lazily)."""
    import yaml

    with Path(path).open() as fh:
        return yaml.safe_load(fh)


def legacy_config_to_project_config(cfg: dict[str, Any]) -> ProjectConfig:
    """Map a legacy config dict onto a :class:`ProjectConfig`.

    Multi-animal projects store their parts in ``multianimalbodyparts`` (with
    ``bodyparts`` set to the sentinel ``"MULTI!"``); single-animal projects store
    a plain list in ``bodyparts``. The experimenter/``scorer`` becomes a member of
    ``experimenters`` -- metadata, not a path component.
    """
    multi = bool(cfg.get("multianimalproject", False))
    if multi:
        bodyparts = list(cfg.get("multianimalbodyparts") or [])
        individuals = list(cfg.get("individuals") or [])
    else:
        raw = cfg.get("bodyparts")
        bodyparts = list(raw) if isinstance(raw, list) else []
        individuals = []
    scorer = cfg.get("scorer")
    return ProjectConfig(
        task=cfg["Task"],
        bodyparts=bodyparts,
        experimenters=[scorer] if scorer else [],
        multi_animal=multi,
        individuals=individuals,
        unique_bodyparts=list(cfg.get("uniquebodyparts") or []),
        skeleton=[list(e) for e in (cfg.get("skeleton") or []) if e],
        notes=f"migrated from a legacy DeepLabCut project (date={cfg.get('date')})",
    )


def legacy_video_paths(cfg: dict[str, Any], legacy_root: str | Path) -> list[Path]:
    """Resolve source video paths from a legacy config.

    Prefers the absolute keys of ``video_sets``; for any that no longer exist,
    falls back to a same-named file under ``<legacy_root>/videos/``. Finally, if
    ``video_sets`` yields nothing, enumerates ``<legacy_root>/videos/``.
    """
    legacy_root = Path(legacy_root)
    videos_dir = legacy_root / "videos"
    found: list[Path] = []
    for key in (cfg.get("video_sets") or {}):
        p = Path(key)
        if p.is_file():
            found.append(p)
        elif (videos_dir / p.name).is_file():
            found.append(videos_dir / p.name)
        else:
            log.warning("video referenced in config but not found on disk: %s", key)
    if not found and videos_dir.is_dir():
        found = sorted(p for p in videos_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    return found


# ------------------------------------------------------------- model discovery
@dataclass
class LegacyModel:
    """A trained model found under ``dlc-models-pytorch/``."""

    train_dir: Path
    pose_config: Path
    pose_snapshots: list[Path]
    detector_snapshots: list[Path] = field(default_factory=list)
    iteration: int | None = None
    shuffle: int | None = None
    train_fraction: float | None = None
    net_type: str | None = None

    @property
    def top_down(self) -> bool:
        return bool(self.detector_snapshots)


def discover_legacy_models(legacy_root: str | Path) -> list[LegacyModel]:
    """Find every trained PyTorch model under ``<legacy_root>/dlc-models-pytorch``."""
    root = Path(legacy_root) / LEGACY_MODELS_DIR
    models: list[LegacyModel] = []
    if not root.is_dir():
        return models
    for train_dir in sorted(root.glob("iteration-*/*/train")):
        pose_config = train_dir / "pytorch_config.yaml"
        if not pose_config.is_file():
            continue
        pose = _snapshots.list_snapshots(train_dir, detector=False)
        if not pose:
            log.warning("skipping %s: no pose snapshots found", train_dir)
            continue
        it_m = _ITERATION_RE.search(str(train_dir))
        ts_m = _TRAINSET_SHUFFLE_RE.search(train_dir.parent.name)
        models.append(
            LegacyModel(
                train_dir=train_dir,
                pose_config=pose_config,
                pose_snapshots=pose,
                detector_snapshots=_snapshots.list_snapshots(train_dir, detector=True),
                iteration=int(it_m.group(1)) if it_m else None,
                shuffle=int(ts_m.group(2)) if ts_m else None,
                train_fraction=int(ts_m.group(1)) / 100 if ts_m else None,
                net_type=_snapshots.read_net_type(pose_config),
            )
        )
    return models


# --------------------------------------------------------------- bundle assembly
def _bundle_from_legacy(model: LegacyModel, dest: Path, model_id: str) -> ModelBundle:
    return ModelBundle.from_train_dir(
        dest,
        model.train_dir,
        model_id=model_id,
        legacy={
            "iteration": model.iteration,
            "shuffle": model.shuffle,
            "train_fraction": model.train_fraction,
            "source": model.train_dir.parent.name,
        },
    )


# --------------------------------------------------------------------- top-level
def migrate_project(
    legacy_root: str | Path,
    dest_root: str | Path,
    *,
    link: str = "symlink",
    include_videos: bool = True,
    include_models: bool = True,
    include_annotations: bool = True,
    exist_ok: bool = False,
) -> Project:
    """Migrate a legacy project at ``legacy_root`` into a workspace at ``dest_root``.

    Returns the opened :class:`Project`. Videos are registered (materialized per
    ``link``); each trained model becomes a portable bundle under ``models/``.
    Annotations are handled separately by the annotations module.
    """
    legacy_root = Path(legacy_root)
    cfg = read_legacy_config(legacy_root / "config.yaml")
    pc = legacy_config_to_project_config(cfg)

    project = Project.create(
        dest_root,
        task=pc.task,
        bodyparts=pc.bodyparts,
        experimenters=pc.experimenters,
        multi_animal=pc.multi_animal,
        individuals=pc.individuals,
        unique_bodyparts=pc.unique_bodyparts,
        skeleton=pc.skeleton,
        exist_ok=exist_ok,
    )
    project.config.notes = pc.notes
    project.save_config()

    if include_videos:
        for vpath in legacy_video_paths(cfg, legacy_root):
            try:
                project.add_video(vpath, link=link, exist_ok=True)
            except FileNotFoundError:
                log.warning("could not register video: %s", vpath)

    if include_models:
        for model in discover_legacy_models(legacy_root):
            model_id = ids.new_model_id()
            _bundle_from_legacy(model, project.layout.model_dir(model_id), model_id)
            log.info("migrated model -> models/%s (shuffle=%s)", model_id, model.shuffle)

    if include_annotations:
        from .annotations import ingest_annotations

        summary = ingest_annotations(project, legacy_root, link=link)
        for video_id, n in summary.items():
            log.info("migrated %d annotated frames -> sources/annotations/%s", n, video_id)

    return project
