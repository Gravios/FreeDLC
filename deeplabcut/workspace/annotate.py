"""Launch the napari-deeplabcut annotator on a workspace project's video.

napari-deeplabcut is built for the legacy DLC layout: it opens a ``config.yaml`` plus a
``labeled-data/<dataset>/`` folder of frames, and on save writes ``CollectedData_*`` back
into that folder. A workspace project has neither shape, so this module bridges from the
workspace side, which works with a stock napari-deeplabcut:

1. Frames are ensured in ``sources/annotations/<video_id>/frames/original/`` (extracted
   if absent).
2. A DLC-shaped *staging* tree is built under ``<root>/.annotate/<video_id>/`` -- a
   synthesized ``config.yaml`` and a ``labeled-data/<video_id>/`` of symlinks to the
   frames -- which is exactly what napari expects to open and save into.
3. napari is launched on that staging tree (blocking until the window closes).
4. On close, the ``CollectedData_*`` napari wrote is ingested into
   ``sources/annotations/<video_id>/labels.parquet``.

Data flows one way through the staging tree: frames are *viewed* through it and labels
are *read* out of it. The frames stay where extraction put them; nothing is ever linked
or copied back from the staging view into ``sources/``.

napari shows existing labels from the staged ``CollectedData_*`` file, and what it
saves there replaces ``labels.parquet`` when the window closes. A staging tree without
that file -- a migrated project, or a deleted ``.annotate/`` -- would therefore open
with no labels and lose them on the next save, so staging first writes the file from
``labels.parquet`` (:func:`restore_collected_data`).

A model's proposed positions (``fdlc extract --from-run``) are staged as
``machinelabels-iter0.h5``. napari shows that file as a layer of its own and merges it
into ``CollectedData`` only when that layer is saved, so a proposal is never taken for
a label unless the annotator accepts it. Frames that hold labels after a session are
dropped from the file (:func:`prune_machine_labels`).

The staging tree persists (it is not a temp dir), so if napari or the ingest step fails
the raw labels are still recoverable and re-running annotate re-ingests them. napari is
imported lazily inside :func:`launch_napari`, so the rest of this module -- and the CLI --
imports and tests without Qt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import ids
from .annotations import _image_name, find_collected_data, ingest_video_annotations
from .frames import extract_frames
from .layout import Layout
from .util import materialize

log = logging.getLogger(__name__)

__all__ = [
    "STAGING_DIRNAME",
    "MACHINE_LABELS",
    "restore_collected_data",
    "prune_machine_labels",
    "resolve_video_id",
    "synthesize_config",
    "stage_annotation_project",
    "launch_napari",
    "annotate_video",
]

STAGING_DIRNAME = Layout.STAGING_DIRNAME

#: the staged file of a model's proposed positions (DLC's name for machine predictions)
MACHINE_LABELS = "machinelabels-iter0.h5"

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
        f"run `fdlc add-video` first"
    )


def synthesize_config(project, *, scorer: str) -> dict:
    """Build the legacy ``config.yaml`` dict napari needs from the project's manifest.

    The ``[display]`` table sets ``dotsize`` and ``colormap``, and adds the
    ``bodypart_colors``/``bodypart_sizes`` napari-freedlc reads for markers with a
    style of their own.
    """
    from .display import annotator_config

    cfg = project.config
    config = {
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
    config.update(annotator_config(cfg.style))
    return config


def _scorer(project) -> str:
    for name in project.config.experimenters:
        if str(name).strip():
            return str(name).strip()
    return "labeler"


def labeled_images(long) -> set[str]:
    """Image names in long-form labels that hold at least one placed marker."""
    return set(long.loc[long["x"].notna() & long["y"].notna(), "image"])


def restore_collected_data(project, video_id: str, dataset_dir: Path, *, scorer: str) -> Path | None:
    """Write ``labels.parquet`` into the staging as ``CollectedData_<scorer>.h5`` if it has none.

    The labels are put in original pixels, which is what the annotator shows. Returns
    the file written, or ``None`` when the staging already has a CollectedData file or
    there are no labels.

    Raises:
        ValueError: if the stored labels cannot be put in original pixels.
    """
    import pandas as pd

    from .annotations import long_df_to_collected_data, write_collected_data

    parquet = project.layout.labels_parquet(video_id)
    if find_collected_data(dataset_dir) is not None or not parquet.is_file():
        return None
    labels = pd.read_parquet(parquet)
    sx, sy = project.labels_scale_to(video_id, "original")
    labels = labels.assign(x=labels["x"] * sx, y=labels["y"] * sy)
    wide = long_df_to_collected_data(labels, project, scorer=scorer, dataset=video_id)
    path = write_collected_data(wide, dataset_dir / f"CollectedData_{scorer}.h5")
    log.warning("%s: the annotator had no copy of the labels; wrote %d labeled frame(s) to %s",
                video_id, len(labeled_images(labels)), path)
    return path


def prune_machine_labels(dataset_dir: Path, labeled: set[str]) -> int:
    """Drop the frames in ``labeled`` from the staged proposals; return how many went.

    The file is removed once no frame is left in it.
    """
    from .annotations import read_collected_data, write_collected_data

    path = Path(dataset_dir) / MACHINE_LABELS
    if not path.is_file():
        return 0
    wide = read_collected_data(path)
    names = [_image_name(k) for k in wide.index]
    keep = [name not in labeled for name in names]
    dropped = len(names) - sum(keep)
    if not dropped:
        return 0
    if not any(keep):
        path.unlink()
    else:
        write_collected_data(wide[keep], path)
    return dropped


def stage_annotation_project(project, video_id: str, frames: list[Path], *, scorer: str) -> tuple[Path, Path]:
    """Build the DLC-shaped staging tree napari opens; return ``(config_path, dataset_dir)``.

    ``<root>/.annotate/<video_id>/`` gets a ``config.yaml`` and a ``labeled-data/<video_id>/``
    holding one symlink per frame in ``frames``. Idempotent, and self-correcting: every
    link is checked against its frame and re-made if it is missing, dangling or points
    elsewhere, and links to frames no longer in ``frames`` (dropped by a re-extraction)
    are removed. Only symlinks are ever pruned -- any ``CollectedData_*`` napari
    previously wrote there is left in place so existing labels reload, and when there
    is none, one is written from ``labels.parquet`` (:func:`restore_collected_data`).
    """
    import yaml

    dataset_dir = project.layout.staging_dataset_dir(video_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    wanted = {Path(frame).name for frame in frames}
    for staged in dataset_dir.glob("*.png"):
        if staged.is_symlink() and staged.name not in wanted:
            staged.unlink()
    for frame in frames:
        materialize(frame, dataset_dir / Path(frame).name, "symlink")
    restore_collected_data(project, video_id, dataset_dir, scorer=scorer)

    config_path = project.layout.staging_config(video_id)
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
    _launch=launch_napari,
) -> str:
    """Extract-if-needed, launch napari to annotate ``video``, and ingest labels on close.

    Frames are staged from the *original* video so markers are placed on full-resolution
    images, and the labels are stored as placed: ``labels.parquet`` holds original-pixel
    coordinates and ``labels.toml`` says so. They are not scaled to the processed video
    here. Which frame set a model trains on is chosen at training time, and the
    conversion belongs there: stored in the pixels they were drawn in, the labels stay
    valid when the processed video is replaced by one of another size, and what is on
    disk is what the annotator saw.

    Only the labels are ingested. The frames napari showed are the workspace's own
    ``frames/original/`` files, reached through the staging symlinks, so there is
    nothing to bring in -- and linking them "into" the workspace from the staging
    view would replace each frame with a symlink to itself.

    ``_launch`` is injected so the orchestration can be tested without Qt. Returns the
    resolved ``video_id``.
    """
    video_id = resolve_video_id(project, video)
    frames = extract_frames(project, video_id, n=n, mode=mode)
    scorer = _scorer(project)
    config_path, dataset_dir = stage_annotation_project(project, video_id, frames, scorer=scorer)

    _launch(config_path, dataset_dir)

    # napari has closed: pull whatever labels were saved into the workspace, in the
    # pixels they were placed in. Nothing is scaled here; `train` and `evaluate`
    # convert to the frame set they are asked to use.
    collected = find_collected_data(dataset_dir)
    if collected is None:
        log.info("no CollectedData written for %s; nothing to ingest", video_id)
        return video_id
    long, _ = ingest_video_annotations(project, video_id, collected, None)
    n_images = len(dict.fromkeys(long["image"].tolist()))
    log.info("ingested %d annotated frame(s) in original pixels -> %s",
             n_images, project.layout.labels_parquet(video_id))
    accepted = prune_machine_labels(dataset_dir, labeled_images(long))
    if accepted:
        log.info("%d frame(s) now labeled were dropped from the proposals in %s",
                 accepted, dataset_dir / MACHINE_LABELS)
    return video_id
