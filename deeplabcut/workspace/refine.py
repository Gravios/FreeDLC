#
# FreeDLC workspace layer
#
"""Choose frames to label next from where an applied model is unsure.

An ``analyze`` run (``fdlc apply --project``) leaves one ``pose.parquet`` per
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

:func:`pick_confident` does the same among the frames showing the most markers,
keeping the most confident frame of each run. Those are frames the model already
gets right -- or appears to: labeled with the model's own positions (see
:func:`propose_labels`) they show where the confident predictions are off, and
they widen the coverage of the labeled set at little cost.

:func:`run_poses` finds a run's pose files and the video each was computed on, and
:func:`video_id_for` maps that video back to a registered id. Frame numbers are
shared by an original and its processed video, so a run on either serves.

:func:`propose_labels` writes a run's predicted positions for new frames into the
annotation file napari opens, in original pixels, so they come up as markers to
adjust rather than to place. ``labels.parquet`` is not touched: the proposals
become labels when annotate closes and ingests that file -- as they are left in it
then, adjusted or not.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .apply import SINGLE_INDIVIDUAL as SINGLE
from .manifest import read_manifest

__all__ = [
    "resolve_run",
    "run_poses",
    "match_video",
    "video_id_for",
    "frame_stats",
    "pick_low_confidence",
    "pick_confident",
    "run_pcutoff",
    "video_size",
    "propose_labels",
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
        raise ValueError("the project has no analyze runs yet; run `fdlc apply --project ...` first")
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


def match_video(project, video: Path) -> tuple[str | None, str | None]:
    """``(video_id, kind)`` of the registered video ``video`` is, if any.

    The file itself is looked for first (the media of either kind, or the source a
    video was registered from), which also tells the kind. Failing that, the name
    is matched as ``videos --register`` matches it -- which finds a file of another
    size too, but not its kind (``None``).
    """
    target = video.resolve() if video.exists() else None
    if target is not None:
        for kind in ("original", "processed"):
            for vid in project.videos(kind):
                sources = [m.resolve() for m in project.video_media_files(vid, kind)]
                sources.append(Path(project.video_record(vid, kind).source_path).resolve())
                if target in sources:
                    return vid, kind
    pairs, _ = project.match_originals([video])
    return next(iter(pairs), None), None


def video_id_for(project, video: Path) -> str | None:
    """The registered id ``video`` is the original or processed video of, if any."""
    return match_video(project, video)[0]


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


def _spread(frames, n: int, *, best_first) -> list[int]:
    """Split ``frames`` (time-ordered) into ``n`` equal runs; keep the first of each by ``best_first``."""
    import numpy as np

    if len(frames) <= n:
        return [int(f) for f in frames.index]
    keys, ascending = zip(*best_first, strict=True)
    picks = []
    for chunk in np.array_split(np.arange(len(frames)), n):
        part = frames.iloc[chunk].sort_values(list(keys), ascending=list(ascending), kind="stable")
        picks.append(int(part.index[0]))
    return sorted(picks)


def pick_low_confidence(stats, n: int, *, max_shown: int, exclude=()) -> tuple[list[int], int]:
    """Choose up to ``n`` frames among those showing at most ``max_shown`` markers.

    Frames in ``exclude`` (those already extracted) are never chosen. Returns the
    chosen frame numbers, sorted, and how many frames qualified.
    """
    low = stats[(stats["shown"] <= max_shown) & ~stats.index.isin(list(exclude))].sort_index()
    if n <= 0 or low.empty:
        return [], len(low)
    return _spread(low, n, best_first=[("shown", True), ("likelihood", True)]), len(low)


def pick_confident(stats, n: int, *, exclude=()) -> tuple[list[int], int]:
    """Choose up to ``n`` frames among those showing the most markers any frame shows.

    Within each run the frame with the highest mean likelihood is kept. Frames in
    ``exclude`` are never chosen. Returns the chosen frame numbers, sorted, and how
    many frames qualified.
    """
    free = stats[~stats.index.isin(list(exclude))]
    if n <= 0 or free.empty:
        return [], 0
    top = free[free["shown"] == free["shown"].max()].sort_index()
    return _spread(top, n, best_first=[("likelihood", False)]), len(top)


def video_size(project, video_id: str, video: Path | None, kind: str | None) -> tuple[int, int] | None:
    """``(width, height)`` of the video a run was computed on, if it can be told.

    The file is probed when it is still there; otherwise the registered record of
    ``kind`` (which kind of ``video_id`` the run's video was found to be) is used.
    """
    from .project import _probe_video

    if video is not None and video.is_file():
        width, height, _, _ = _probe_video(video)
        if width and height:
            return width, height
    if kind is not None and project.has_video(video_id, kind):
        rec = project.video_record(video_id, kind)
        if rec.width and rec.height:
            return rec.width, rec.height
    return None


def propose_labels(
    project, video_id: str, poses, frames: dict[str, int], *, scale: tuple[float, float],
    min_likelihood: float,
) -> tuple[Path, int]:
    """Put a run's predicted positions for ``frames`` into the annotation file napari opens.

    ``frames`` maps a frame file name to its frame number in ``poses`` (a run's
    long-form pose DataFrame). Positions are multiplied by ``scale`` -- run-video
    pixels to original pixels, which the annotator shows -- and a marker below
    ``min_likelihood`` is left unplaced. Frames that already hold a label are left
    as they are.

    The file is the staged ``CollectedData_<scorer>.h5`` annotate opens. When there
    is none yet, it is started from ``labels.parquet`` so that existing labels are
    kept; a file that is replaced is kept beside it as ``.bak``. Returns the file and
    the number of markers placed.
    """
    import shutil

    import numpy as np
    import pandas as pd

    from .annotate import _scorer
    from .annotations import (
        collected_data_to_long_df,
        find_collected_data,
        long_df_to_collected_data,
        read_collected_data,
    )

    dataset_dir = project.layout.staging_dataset_dir(video_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    existing = find_collected_data(dataset_dir)
    if existing is not None:
        labels = collected_data_to_long_df(read_collected_data(existing))
        scorer = existing.stem[len("CollectedData_"):]
    else:
        scorer = _scorer(project)
        labels = pd.DataFrame(columns=["image", "individual", "bodypart", "x", "y"])
        parquet = project.layout.labels_parquet(video_id)
        if parquet.is_file():
            labels = pd.read_parquet(parquet)
            sx, sy = project.labels_scale_to(video_id, "original")  # stored space -> what napari shows
            labels = labels.assign(x=labels["x"] * sx, y=labels["y"] * sy)
    labeled = set(labels.loc[labels["x"].notna() & labels["y"].notna(), "image"])

    by_frame = {f: name for name, f in frames.items() if name not in labeled}
    picked = poses[poses["frame"].isin(list(by_frame))].copy()
    if project.config.multi_animal:  # model slots idv0, idv1, ... -> the project's individuals
        slots = [i for i in dict.fromkeys(picked["individual"]) if i != SINGLE]
        rename = dict(zip(slots, project.config.individuals, strict=False))
        picked["individual"] = picked["individual"].map(lambda i: rename.get(i, i))
    keep = picked["likelihood"] >= min_likelihood
    picked.loc[~keep, ["x", "y"]] = np.nan
    proposals = pd.DataFrame({
        "image": picked["frame"].map(by_frame),
        "individual": picked["individual"],
        "bodypart": picked["bodypart"],
        "x": picked["x"] * scale[0],
        "y": picked["y"] * scale[1],
    })
    placed = int((proposals["x"].notna() & proposals["y"].notna()).sum())
    # frames with nothing placed still go in, so napari lists them with the others
    merged = pd.concat([labels[~labels["image"].isin(list(by_frame.values()))], proposals],
                       ignore_index=True)
    wide = long_df_to_collected_data(merged, project, scorer=scorer, dataset=video_id)

    target = existing if existing is not None and existing.suffix == ".h5" else \
        dataset_dir / f"CollectedData_{scorer}.h5"
    if target.exists():
        shutil.copy2(target, target.with_name(target.name + ".bak"))
    partial = target.with_name(target.stem + ".part.h5")
    wide.to_hdf(partial, key="df_with_missing", mode="w")
    partial.replace(target)
    csv = target.with_suffix(".csv")
    if csv.exists():  # keep a CSV napari may have written in step with the h5
        wide.to_csv(csv)
    return target, placed
