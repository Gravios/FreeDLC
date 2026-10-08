#
# FreeDLC workspace layer
#
"""Choose frames to label next from where an applied model is unsure.

An ``analyze`` run (``dlc-ws apply --project``) leaves one ``pose.parquet`` per
video, with a likelihood for every marker in every frame. A labeled video draws a
marker only where its likelihood reaches ``pcutoff``, so a frame in which few or
none of them do is a frame the model cannot handle yet -- and the most useful
frame to annotate next.

:func:`frame_stats` counts, for every frame, the markers that would be drawn.
:func:`pick_low_confidence` takes the frames at or below ``max_shown`` and spreads
the picks over them: the low-confidence frames, in time order, are split into ``n``
runs of equal length and the worst frame of each run is kept (fewest markers shown,
then lowest mean likelihood). Low-confidence frames come in episodes -- the animal
in a corner, rearing, grooming -- so picking the n worst outright would take n
nearly identical frames from one episode. Splitting by count rather than by time
still gives a long bad episode more picks than a short one.

:func:`run_poses` finds a run's pose files and the video each was computed on, and
:func:`video_id_for` maps that video back to a registered id. Frame numbers are
shared by an original and its processed video, so a run on either serves.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .manifest import read_manifest

__all__ = [
    "resolve_run",
    "run_poses",
    "video_id_for",
    "frame_stats",
    "pick_low_confidence",
    "run_pcutoff",
]

log = logging.getLogger(__name__)

#: likelihood threshold `apply` draws markers at when a run does not record its own
DEFAULT_PCUTOFF = 0.6


def resolve_run(project, run: str):
    """The ``analyze`` :class:`Run` named by ``run``.

    ``run`` is a run id, a prefix unique among the project's analyze runs,
    ``latest``, or the path of a run directory.

    Raises:
        ValueError: if nothing, or more than one run, fits.
    """
    runs = project.runs("analyze")
    if not runs:
        raise ValueError("the project has no analyze runs yet; run `dlc-ws apply --project ...` first")
    if run == "latest":
        return runs[-1]
    path = Path(run)
    if (path / "run.toml").is_file():
        for candidate in runs:
            if candidate.dir.resolve() == path.resolve():
                return candidate
        raise ValueError(f"{path} is not an analyze run of {project.root}")
    hits = [r for r in runs if r.run_id.startswith(run)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no analyze run {run!r} in {project.root} (latest: {runs[-1].run_id})")
    raise ValueError(f"{run!r} fits {len(hits)} analyze runs; give more of the id")


def run_pcutoff(run) -> float:
    """The ``pcutoff`` the run's labeled videos were drawn with, else ``apply``'s default."""
    value = run.manifest().params.get("pcutoff")
    return float(value) if value is not None else DEFAULT_PCUTOFF


def run_poses(project, run) -> list[tuple[Path | None, Path]]:
    """``(video, pose.parquet)`` for every video of ``run``, sorted by pose path.

    The poses are found in the run directory, or where the run's outputs say they
    were written (``apply --out``). Each pose's own ``run.toml`` names its video;
    ``None`` when it does not.
    """
    poses = set(run.dir.glob("*/pose.parquet"))
    for out in run.manifest().outputs:
        for p in (Path(out), project.layout.root / out):
            if p.name == "pose.parquet" and p.is_file():
                poses.add(p)
                break
    pairs: dict[Path, tuple[Path | None, Path]] = {}
    for pose in poses:
        key = pose.resolve()
        if key in pairs:
            continue
        video = None
        record = pose.with_name("run.toml")
        if record.is_file():
            inputs = read_manifest(record).get("inputs") or []
            video = Path(inputs[0]) if inputs else None
        pairs[key] = (video, pose)
    return sorted(pairs.values(), key=lambda pair: str(pair[1]))


def video_id_for(project, video: Path) -> str | None:
    """The registered id ``video`` is the original or processed video of, if any.

    The file itself is looked for first (the media of either kind, or the source a
    video was registered from); failing that, the name is matched as
    ``videos --register`` matches it, which also finds a file of another size.
    """
    target = video.resolve() if video.exists() else None
    if target is not None:
        for kind in ("original", "processed"):
            for vid in project.videos(kind):
                sources = [m.resolve() for m in project.video_media_files(vid, kind)]
                sources.append(Path(project.video_record(vid, kind).source_path).resolve())
                if target in sources:
                    return vid
    pairs, _ = project.match_originals([video])
    return next(iter(pairs), None)


def frame_stats(df, pcutoff: float):
    """Per frame: the markers a labeled video would draw (``shown``) and their mean likelihood.

    A marker is drawn when its likelihood reaches ``pcutoff`` and its position is
    finite. Every individual's markers count. Returns a DataFrame indexed by frame
    with columns ``shown`` and ``likelihood``.
    """
    import numpy as np
    import pandas as pd

    drawn = (df["likelihood"] >= pcutoff) & np.isfinite(df["x"]) & np.isfinite(df["y"])
    grouped = df.assign(drawn=drawn).groupby("frame")
    return pd.DataFrame({
        "shown": grouped["drawn"].sum().astype(int),
        "likelihood": grouped["likelihood"].mean().fillna(0.0),
    })


def pick_low_confidence(stats, n: int, *, max_shown: int, exclude=()) -> tuple[list[int], int]:
    """Choose up to ``n`` frames among those showing at most ``max_shown`` markers.

    Frames in ``exclude`` (those already extracted) are never chosen. Returns the
    chosen frame numbers, sorted, and how many frames qualified.
    """
    import numpy as np

    low = stats[(stats["shown"] <= max_shown) & ~stats.index.isin(list(exclude))].sort_index()
    if n <= 0 or low.empty:
        return [], len(low)
    if len(low) <= n:
        return [int(f) for f in low.index], len(low)
    picks = []
    for chunk in np.array_split(np.arange(len(low)), n):
        part = low.iloc[chunk].sort_values(["shown", "likelihood"], kind="stable")
        picks.append(int(part.index[0]))
    return sorted(picks), len(low)
