"""Launch the napari-deeplabcut annotator on a workspace project's video.

napari-deeplabcut is built for the legacy DLC layout: it opens a ``config.yaml`` plus a
``labeled-data/<dataset>/`` folder of frames, and on save writes ``CollectedData_*`` back
into that folder. A workspace project has neither shape. Rather than depend on the (not
yet applied) napari-side patch that teaches it to read ``project.toml`` directly, this
module bridges from the workspace side, which works with a stock napari-deeplabcut:

1. Frames are ensured in ``sources/annotations/<video_id>/frames/`` (extracted if absent).
2. A DLC-shaped *staging* tree is built under ``<root>/.annotate/<video_id>/`` -- a
   synthesized ``config.yaml`` and a ``labeled-data/<video_id>/`` of symlinks to the
   frames -- which is exactly what napari expects to open and save into.
3. napari is launched on that staging tree (blocking until the window closes).
4. On close, the ``CollectedData_*`` napari wrote is ingested into
   ``sources/annotations/<video_id>/labels.parquet`` via the existing ingest path.

The staging tree persists (it is not a temp dir), so if napari or the ingest step fails
the raw labels are still recoverable and re-running annotate re-ingests them. napari is
imported lazily inside :func:`launch_napari`, so the rest of this module -- and the CLI --
imports and tests without Qt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import ids
from .annotations import find_collected_data, ingest_video_annotations
from .frames import extract_frames

log = logging.getLogger(__name__)

__all__ = [
    "STAGING_DIRNAME",
    "resolve_video_id",
    "synthesize_config",
    "stage_annotation_project",
    "launch_napari",
    "annotate_video",
]

STAGING_DIRNAME = ".annotate"

DEFAULT_DOTSIZE = 6
DEFAULT_PCUTOFF = 0.6
DEFAULT_COLORMAP = "viridis"


def resolve_video_id(project, video: str) -> str:
    """Resolve a CLI ``video`` argument to a registered ``video_id``.

    Accepts an id directly, or a path whose slugified stem matches a registered video.

    Raises:
        FileNotFoundError: if nothing registered matches.
    """
    if project.has_video(video):
        return video
    candidate = ids.slugify(Path(video).stem)
    if project.has_video(candidate):
        return candidate
    raise FileNotFoundError(
        f"no registered video matches {video!r} (looked for id {video!r} and {candidate!r}); "
        f"run `dlc-ws add-video` first"
    )


def synthesize_config(project, *, scorer: str) -> dict:
    """Build the legacy ``config.yaml`` dict napari needs from the project's manifest."""
    cfg = project.config
    return {
        "Task": cfg.task,
        "scorer": scorer,
        "multianimalproject": bool(cfg.multi_animal),
        "individuals": list(cfg.individuals),
        "bodyparts": list(cfg.bodyparts),
        "multianimalbodyparts": list(cfg.bodyparts),
        "uniquebodyparts": list(cfg.unique_bodyparts),
        "skeleton": [list(edge) for edge in cfg.skeleton],
        "dotsize": DEFAULT_DOTSIZE,
        "pcutoff": DEFAULT_PCUTOFF,
        "colormap": DEFAULT_COLORMAP,
    }


def _scorer(project) -> str:
    for name in project.config.experimenters:
        if str(name).strip():
            return str(name).strip()
    return "labeler"


def stage_annotation_project(project, video_id: str, frames: list[Path], *, scorer: str) -> tuple[Path, Path]:
    """Build the DLC-shaped staging tree napari opens; return ``(config_path, dataset_dir)``.

    ``<root>/.annotate/<video_id>/`` gets a ``config.yaml`` and a ``labeled-data/<video_id>/``
    of symlinks to ``frames``. Idempotent: re-linking is safe, and any ``CollectedData_*``
    napari previously wrote there is left in place so existing labels reload.
    """
    import yaml

    staging = project.layout.root / STAGING_DIRNAME / video_id
    dataset_dir = staging / "labeled-data" / video_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    linked = {p.name for p in dataset_dir.glob("*.png")}
    for frame in frames:
        dst = dataset_dir / frame.name
        if frame.name in linked:
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(Path(frame).resolve())

    config_path = staging / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(synthesize_config(project, scorer=scorer), fh, sort_keys=False)
    return config_path, dataset_dir


def launch_napari(config_path: Path, dataset_dir: Path) -> None:
    """Open napari-deeplabcut on ``[dataset_dir, config_path]`` and block until closed.

    Isolated so it is the only part that needs Qt; imported lazily.
    """
    import napari  # noqa: F401
    from napari import Viewer, run

    viewer = Viewer()
    viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")
    viewer.open([str(dataset_dir), str(config_path)], plugin="napari-deeplabcut")
    run()


def annotate_video(
    project,
    video: str,
    *,
    n: int = 20,
    mode: str = "uniform",
    link: str = "symlink",
    _launch=launch_napari,
) -> str:
    """Extract-if-needed, launch napari to annotate ``video``, and ingest labels on close.

    Frames are staged from the *original* video so markers are placed on full-resolution
    images. napari saves a CollectedData in that original pixel space (a self-consistent
    legacy artifact). On close it is ingested into ``sources/annotations/<id>/labels.parquet``
    with coordinates scaled to the *processed* space -- the space the model trains on --
    using the project's per-video ``(scale_x, scale_y)``. The scale transform lives here,
    not in napari: napari shows original frames, so writing processed coordinates into a
    CollectedData that references those frames would make it internally inconsistent.

    ``_launch`` is injected so the orchestration can be tested without Qt. Returns the
    resolved ``video_id``.
    """
    video_id = resolve_video_id(project, video)
    frames = extract_frames(project, video_id, n=n, mode=mode)
    scorer = _scorer(project)
    config_path, dataset_dir = stage_annotation_project(project, video_id, frames, scorer=scorer)

    _launch(config_path, dataset_dir)

    # napari has closed: pull whatever labels were saved into the workspace,
    # scaling original-space coordinates into processed space on the way in.
    collected = find_collected_data(dataset_dir)
    if collected is None:
        log.info("no CollectedData written for %s; nothing to ingest", video_id)
        return video_id
    scale_x, scale_y = project.annotation_scale(video_id)
    long, copied = ingest_video_annotations(
        project, video_id, collected, dataset_dir, link=link, scale=(scale_x, scale_y)
    )
    n_images = len(dict.fromkeys(long["image"].tolist()))
    space = "processed" if (scale_x, scale_y) != (1.0, 1.0) else "original"
    log.info("ingested %d annotated frame(s) in %s space -> %s",
             n_images, space, project.layout.labels_parquet(video_id))
    return video_id
