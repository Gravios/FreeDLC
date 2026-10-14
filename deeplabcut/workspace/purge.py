#
# FreeDLC workspace layer
#
"""Remove what a project no longer needs.

:func:`plan_unlabeled_frames` finds a video's extracted frames that carry no label:
not one marker placed on them in ``labels.parquet``, nor in the annotator's staged
``CollectedData`` (labels napari saved that were not ingested yet, if napari did not
close cleanly). :func:`purge_unlabeled_frames` removes them -- from both frame sets,
``frames/original/`` and ``frames/processed/``, and from the annotator's staging:

- the staging symlinks to them;
- their rows in the staged ``CollectedData`` and in ``labels.parquet`` (rows with no
  marker placed, which napari writes for every frame it showed);
- the markers a model proposed for them (``machinelabels-iter0.h5``, see
  ``extract --from-run``), unless ``keep_proposed`` keeps the frames that have any.

Labels are never touched: a frame with even one marker placed stays, with all its
rows. A frame is removed by unlinking it, so a frame that is a link (a migrated
project's) loses only the link, never the file it points to.

Frames are named by the source frame they were read from, so the frames removed can
be extracted again later, by ``extract`` with another selection or ``extract
--from-run``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["UnlabeledFrames", "plan_unlabeled_frames", "purge_unlabeled_frames", "videos_with_frames"]

log = logging.getLogger(__name__)


@dataclass
class UnlabeledFrames:
    """What :func:`purge_unlabeled_frames` removes, or would remove, for one video."""

    video_id: str
    n_frames: int = 0                                   # extracted frame names, either set
    n_labeled: int = 0                                  # of those, frames with a marker placed
    names: list[str] = field(default_factory=list)      # the frames to remove
    proposed: list[str] = field(default_factory=list)   # of those, frames with proposed markers
    kept_proposed: list[str] = field(default_factory=list)  # unlabeled, kept for their proposals


def videos_with_frames(project) -> list[str]:
    """Video ids that have extracted frames of either set, sorted."""
    root = project.layout.annotations_dir
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sets = [project.layout.frames_dir(d.name, kind) for kind in project.layout.VIDEO_KINDS]
        if any(s.is_dir() and any(s.glob("*.png")) for s in sets):
            out.append(d.name)
    return out


def _frame_names(project, video_id: str) -> set[str]:
    names: set[str] = set()
    for kind in project.layout.VIDEO_KINDS:
        frames_dir = project.layout.frames_dir(video_id, kind)
        if frames_dir.is_dir():
            names |= {p.name for p in frames_dir.glob("*.png")}  # a dangling link counts too
    return names


def _labeled(project, video_id: str) -> set[str]:
    """Frames of ``video_id`` with a marker placed, ingested or only staged."""
    import pandas as pd

    from .annotate import labeled_images
    from .annotations import collected_data_to_long_df, find_collected_data, read_collected_data

    labeled: set[str] = set()
    parquet = project.layout.labels_parquet(video_id)
    if parquet.is_file():
        labeled |= labeled_images(pd.read_parquet(parquet))
    staged = find_collected_data(project.layout.staging_dataset_dir(video_id))
    if staged is not None:
        labeled |= labeled_images(collected_data_to_long_df(read_collected_data(staged)))
    return labeled


def _proposed(project, video_id: str) -> set[str]:
    """Frames of ``video_id`` with a marker proposed in the staged machine labels."""
    from .annotate import MACHINE_LABELS, labeled_images
    from .annotations import collected_data_to_long_df, read_collected_data

    path = project.layout.staging_dataset_dir(video_id) / MACHINE_LABELS
    if not path.is_file():
        return set()
    return labeled_images(collected_data_to_long_df(read_collected_data(path)))


def plan_unlabeled_frames(project, video_id: str, *, keep_proposed: bool = False) -> UnlabeledFrames:
    """The extracted frames of ``video_id`` that :func:`purge_unlabeled_frames` would remove."""
    names = _frame_names(project, video_id)
    labeled = _labeled(project, video_id) & names
    unlabeled = names - labeled
    proposed = _proposed(project, video_id) & unlabeled
    kept = proposed if keep_proposed else set()
    return UnlabeledFrames(video_id=video_id, n_frames=len(names), n_labeled=len(labeled),
                           names=sorted(unlabeled - kept), proposed=sorted(proposed - kept),
                           kept_proposed=sorted(kept))


def _drop_rows(path: Path, names: set[str]) -> int:
    """Drop the rows of ``names`` from a staged CollectedData file; keep the old one as ``.bak``."""
    import shutil

    from .annotations import _image_name, read_collected_data, write_collected_data

    wide = read_collected_data(path)
    keep = [_image_name(k) not in names for k in wide.index]
    dropped = len(keep) - sum(keep)
    if dropped:
        shutil.copy2(path, path.with_name(path.name + ".bak"))
        write_collected_data(wide[keep], path.with_suffix(".h5"))
    return dropped


def purge_unlabeled_frames(project, plan: UnlabeledFrames) -> int:
    """Remove the frames of ``plan`` (see the module docstring); return how many files went.

    The labels are read again first, so a frame labeled since the plan was made stays.
    """
    import pandas as pd

    from .annotate import prune_machine_labels
    from .annotations import find_collected_data, write_labels_parquet

    vid = plan.video_id
    names = set(plan.names) - _labeled(project, vid)
    if not names:
        return 0
    removed = 0
    for kind in project.layout.VIDEO_KINDS:
        frames_dir = project.layout.frames_dir(vid, kind)
        for name in names:
            path = frames_dir / name
            if path.is_symlink() or path.exists():
                path.unlink()
                removed += 1

    dataset_dir = project.layout.staging_dataset_dir(vid)
    for name in names:
        link = dataset_dir / name
        if link.is_symlink():
            link.unlink()
    staged = find_collected_data(dataset_dir)
    if staged is not None:
        _drop_rows(staged, names)
    prune_machine_labels(dataset_dir, names)

    parquet = project.layout.labels_parquet(vid)
    if parquet.is_file():
        labels = pd.read_parquet(parquet)
        gone = labels["image"].isin(list(names))
        if gone.any():
            write_labels_parquet(labels[~gone], parquet)
    log.info("%s: removed %d unlabeled frame(s)", vid, len(names))
    return removed
