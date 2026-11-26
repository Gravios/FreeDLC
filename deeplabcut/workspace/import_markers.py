#
# FreeDLC workspace layer
#
"""Bring markers made elsewhere into a project: as labels, or as an analyze run.

Markers come from many places besides a model of the project's own: another
project's model, a smoother that cleaned a model's output (Mufasa's Kalman
smoother), or a measurement (paw positions from a pressure plate). ``fdlc import``
reads them from a parquet file and puts them where the project's tools find them:

* **as labels** (:func:`import_labels`): positions at chosen frames are added to a
  video's ``labels.parquet``, extracting those frames when they are not extracted
  yet -- training data, ready to check in ``annotate``. With ``propose`` they are
  staged as napari *machine labels* instead (as ``extract --from-run`` does), to be
  accepted frame by frame;
* **as a run** (:func:`import_run`): the whole pose becomes an ``analyze`` run, laid
  out as ``apply`` writes one (``<video>.parquet`` beside its ``<video>.toml``
  record), so ``compare``, ``extract --from-run`` and ``label`` take it like a
  model's.

Two parquet layouts are read (:func:`read_markers`):

* **long**, FreeDLC's own: one row per frame and marker, columns ``frame``,
  ``bodypart``, ``x``, ``y`` and, optionally, ``individual`` and ``likelihood``;
* **wide**, Mufasa's: one row per frame, columns ``<marker>_x``, ``<marker>_y`` and,
  optionally, ``<marker>_p``; a ``frame`` column, or else the row number, is the
  frame. Other columns (a smoother's ``<marker>_var_x``, say) are left out. Mufasa
  writes a missing point as ``0, 0`` with likelihood 0; that is read as missing.

The markers are checked against the project's (:func:`check_markers`): a marker
the project doesn't have is refused, with its name, unless it is renamed
(``rename``) or explicitly dropped (``drop_unknown``). Markers are matched by name,
never by position.

Coordinates are in the pixels of the video the markers were made on: the
original or its processed counterpart (:func:`find_video` tells which). A run keeps
them as they are, as ``apply`` does; labels are converted to the pixels the video's
labels are stored in (original, unless older labels were stored processed).

pandas is imported lazily; cv2 only when frames are extracted.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .apply import SINGLE_INDIVIDUAL as SINGLE

log = logging.getLogger(__name__)

__all__ = [
    "read_markers",
    "check_markers",
    "parse_renames",
    "find_video",
    "to_original_scale",
    "select_frames",
    "parse_frames",
    "import_labels",
    "import_run",
]

#: the columns of a long (FreeDLC) pose, in order
LONG_COLUMNS = ["frame", "individual", "bodypart", "x", "y", "likelihood"]


# ---------------------------------------------------------------------------- reading
def read_markers(path: str | Path):
    """Read a pose parquet in either layout into FreeDLC's long form.

    Returns ``(long, layout)``: a DataFrame with columns ``frame, individual,
    bodypart, x, y, likelihood`` (missing points NaN; likelihood 1.0 where the file
    has none) and ``"long"`` or ``"wide"``.

    Raises:
        ValueError: if the file is neither layout.
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):        # Mufasa's imported form: the last level is flat
        df.columns = [col[-1] for col in df.columns]
    cols = set(map(str, df.columns))
    if {"frame", "bodypart", "x", "y"} <= cols:
        return _from_long(df), "long"
    markers = wide_markers(df.columns)
    if markers:
        return _from_wide(df, markers), "wide"
    shown = ", ".join(list(map(str, df.columns))[:8]) + (" ..." if len(cols) > 8 else "")
    raise ValueError(f"{path}: neither a long pose (frame, bodypart, x, y[, individual, likelihood]) nor "
                     f"a wide one (<marker>_x, <marker>_y[, <marker>_p]); its columns: {shown}")


def wide_markers(columns) -> list[str]:
    """The markers of a wide table, in column order: every ``<name>`` with both ``<name>_x``
    and ``<name>_y``, except a smoother's ``<marker>_var`` beside its ``<marker>``."""
    names = [str(c) for c in columns]
    have = set(names)
    found = []
    for c in names:
        if c.endswith("_x") and c[:-2] + "_y" in have and c[:-2] not in found:
            found.append(c[:-2])
    return [m for m in found if not (m.endswith("_var") and m[:-4] in found)]


def _from_long(df):
    import numpy as np
    import pandas as pd

    out = pd.DataFrame({
        "frame": pd.to_numeric(df["frame"], errors="raise").astype("int64"),
        "individual": df["individual"].astype(str) if "individual" in df else SINGLE,
        "bodypart": df["bodypart"].astype(str),
        "x": pd.to_numeric(df["x"], errors="coerce").astype(float),
        "y": pd.to_numeric(df["y"], errors="coerce").astype(float),
        "likelihood": (pd.to_numeric(df["likelihood"], errors="coerce").astype(float)
                       if "likelihood" in df else 1.0),
    })[LONG_COLUMNS]
    bad = ~(np.isfinite(out["x"]) & np.isfinite(out["y"]))
    out.loc[bad, ["x", "y"]] = np.nan
    return out.reset_index(drop=True)


def _from_wide(df, markers):
    import numpy as np
    import pandas as pd

    if "frame" in df.columns:
        frames = pd.to_numeric(df["frame"], errors="raise").astype("int64").to_numpy()
    elif pd.api.types.is_integer_dtype(df.index):
        frames = df.index.to_numpy().astype("int64")
    else:
        frames = np.arange(len(df), dtype="int64")
    parts = []
    for m in markers:
        x = pd.to_numeric(df[f"{m}_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[f"{m}_y"], errors="coerce").to_numpy(float)
        p = (pd.to_numeric(df[f"{m}_p"], errors="coerce").to_numpy(float) if f"{m}_p" in df
             else np.ones(len(df)))
        missing = ~(np.isfinite(x) & np.isfinite(y)) | ((x == 0) & (y == 0) & ~(p > 0))
        x, y = np.where(missing, np.nan, x), np.where(missing, np.nan, y)
        parts.append(pd.DataFrame({"frame": frames, "individual": SINGLE, "bodypart": m,
                                   "x": x, "y": y, "likelihood": p}))
    return pd.concat(parts, ignore_index=True).sort_values(["frame"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------- checking
def parse_renames(pairs) -> dict[str, str]:
    """``["old=new", ...]`` -> ``{old: new}``."""
    out: dict[str, str] = {}
    for pair in pairs or ():
        old, sep, new = str(pair).partition("=")
        if not sep or not old or not new:
            raise ValueError(f"--rename takes OLD=NEW, got {pair!r}")
        out[old] = new
    return out


def check_markers(long, project, *, rename: dict[str, str] | None = None, drop_unknown: bool = False):
    """Match the file's markers and individuals to the project's, by name.

    ``rename`` maps the file's names to the project's first. A marker the project
    doesn't have is refused unless ``drop_unknown``; then it is left out and named in
    the log. Two of the file's markers becoming one is refused.

    Individuals: in a single-animal project the file must hold one (whatever it is
    called), which becomes ``single``. In a multi-animal project the file's
    individuals must be the project's (``single`` for unique markers); a model's
    numbered slots (``idv0``, ``idv1``...) go to the project's individuals in order.

    Returns the checked DataFrame. Raises ``ValueError`` saying what doesn't match.
    """
    cfg = project.config
    rename = dict(rename or {})
    unknown_renames = sorted(set(rename) - set(long["bodypart"].unique()))
    if unknown_renames:
        raise ValueError(f"--rename names marker(s) the file doesn't have: {', '.join(unknown_renames)}")
    unchanged = set(long["bodypart"].unique()) - set(rename)
    sources: dict[str, list[str]] = {}
    for old, new in rename.items():
        sources.setdefault(new, []).append(old)
    clash = [f"{new} <- {', '.join(sorted(olds) + ([new] if new in unchanged else []))}"
             for new, olds in sources.items() if len(olds) > 1 or new in unchanged]
    if clash:
        raise ValueError(f"--rename would merge markers: {'; '.join(clash)}")
    long = long.assign(bodypart=long["bodypart"].map(lambda b: rename.get(b, b)))

    project_markers = [*cfg.bodyparts, *cfg.unique_bodyparts]
    unknown = [b for b in dict.fromkeys(long["bodypart"]) if b not in project_markers]
    if unknown and not drop_unknown:
        raise ValueError(
            f"{len(unknown)} marker(s) in the file are not the project's: {', '.join(unknown)}\n"
            f"the project's: {', '.join(project_markers)}\n"
            "rename them (--rename FILE_NAME=PROJECT_NAME), leave them out (--drop-unknown), "
            "or add them to project.toml's bodyparts")
    if unknown:
        log.warning("left out %d marker(s) the project doesn't have: %s", len(unknown), ", ".join(unknown))
        long = long[~long["bodypart"].isin(unknown)]
    if long.empty:
        raise ValueError("none of the file's markers are the project's")

    individuals = list(dict.fromkeys(long["individual"]))
    if not cfg.multi_animal:
        if len(individuals) > 1:
            raise ValueError(f"the file has {len(individuals)} individuals ({', '.join(individuals)}); "
                             "the project tracks one")
        return long.assign(individual=SINGLE).reset_index(drop=True)
    allowed = [*cfg.individuals, SINGLE]
    slots = [i for i in individuals if re.fullmatch(r"idv\d+", i)]
    if slots and not set(slots) & set(cfg.individuals):       # a model's numbered slots
        to_project = dict(zip(slots, cfg.individuals, strict=False))
        long = long.assign(individual=long["individual"].map(lambda i: to_project.get(i, i)))
    strange = [i for i in dict.fromkeys(long["individual"]) if i not in allowed]
    if strange:
        raise ValueError(f"individual(s) {', '.join(strange)} are not the project's ({', '.join(cfg.individuals)})")
    return long.reset_index(drop=True)


# ---------------------------------------------------------------------------- the video
def _base_name(path: Path) -> str:
    """A pose file's name without its suffixes: ``s1.cam1-triplet.fdlc.parquet`` -> ``s1``."""
    return path.name.split(".")[0]


def find_video(project, source: str | Path, long=None, *, video=None, video_id: str | None = None,
               kind: str | None = None) -> tuple[str, str]:
    """``(video_id, kind)`` of the registered video the markers in ``source`` were made on.

    ``video`` (a file) is looked up among the registered videos (their media, or the
    sources they were registered from), which also tells the kind; ``video_id`` names
    the id directly. Without either, the file's name is matched as ``videos
    --register`` matches names: ``s1.parquet``, ``s1-triplet.parquet`` and
    ``s1.cam1.fdlc.parquet`` all belong to ``s1``.

    The kind -- which video's pixels the coordinates are in -- is ``kind`` when given,
    else the only kind registered under the id, else told from the coordinates: any
    beyond the processed video's size are original. When the coordinates fit both,
    ``kind`` has to be given.

    Raises:
        ValueError: when no registered video fits, or the kind can't be told.
    """
    from .annotate import resolve_video_id
    from .refine import match_video

    found_kind = None
    if video is not None:
        vid, found_kind = match_video(project, Path(video))
        if vid is None:
            raise ValueError(f"{video} is not a registered video of {project.root} "
                             "(`fdlc videos` lists them; `fdlc add-video` registers one)")
    elif video_id is not None:
        try:
            vid = resolve_video_id(project, video_id)
        except FileNotFoundError:
            if project.has_video(video_id, "processed"):
                vid = video_id
            else:
                raise ValueError(f"no video {video_id!r} is registered in {project.root}") from None
    else:
        base = _base_name(Path(source))
        pairs, _ = project.match_originals([Path(base + ".mp4")])
        vid = next(iter(pairs), None)
        if vid is None:
            from . import ids
            slug = ids.slugify(base)
            fits = [v for v in project.videos("processed") if slug == v or slug.startswith(v + "-")]
            vid = max(fits, key=len) if fits else None
        if vid is None:
            raise ValueError(f"{Path(source).name}: no registered video matches its name ({base}); "
                             "give --video or --video-id")

    if kind is None:
        kind = found_kind
    if kind is None:
        kinds = [k for k in ("original", "processed") if project.has_video(vid, k)]
        if len(kinds) == 1:
            kind = kinds[0]
        elif not kinds:
            raise ValueError(f"{vid} has annotations but no registered video; give --kind")
        else:
            kind = _kind_from_coordinates(project, vid, long)
    if kind not in ("original", "processed"):
        raise ValueError(f"kind must be original or processed, got {kind!r}")
    if not project.has_video(vid, kind) and not (kind == "original" and project.has_video(vid, "processed")):
        raise ValueError(f"no {kind} video is registered under {vid!r}")
    return vid, kind


def _size(project, vid: str, kind: str) -> tuple[int, int] | None:
    from .frames import resolve_media
    from .project import _probe_video

    if not project.has_video(vid, kind):
        return None
    rec = project.video_record(vid, kind)
    if rec.width and rec.height:
        return rec.width, rec.height
    try:
        w, h, _, _ = _probe_video(resolve_media(project, vid, kind))
    except FileNotFoundError:
        return None
    return (w, h) if w and h else None


def _kind_from_coordinates(project, vid: str, long) -> str:
    proc = _size(project, vid, "processed")
    if long is not None and proc is not None and len(long):
        reach = (long["x"].max(), long["y"].max())
        if reach[0] > proc[0] * 1.02 or reach[1] > proc[1] * 1.02:
            return "original"
    raise ValueError(f"{vid} has an original and a processed video and the markers fit both; give "
                     "--kind original or --kind processed (the video they were made on)")


def to_original_scale(project, vid: str, kind: str) -> tuple[float, float]:
    """The factors taking ``kind`` pixels to original pixels."""
    if kind == "original":
        return 1.0, 1.0
    orig, proc = _size(project, vid, "original"), _size(project, vid, "processed")
    if orig is None or proc is None:
        raise ValueError(f"{vid}: the sizes of its original and processed videos are needed to put processed "
                         "pixels on original frames, and one is unknown")
    return orig[0] / proc[0], orig[1] / proc[1]


# ---------------------------------------------------------------------------- frames
def parse_frames(spec: str) -> list[int]:
    """``"10,20,30-40"`` -> ``[10, 20, 30, 31, ..., 40]``."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            out += list(range(int(lo), int(hi) + 1)) if sep else [int(lo)]
        except ValueError:
            raise ValueError(f"frames are numbers and ranges (10,20,30-40), got {part!r}") from None
    return sorted(set(out))


def select_frames(long, *, frames=None, n: int | None = 20, every: bool = False, labeled=None,
                  min_likelihood: float = 0.0) -> list[int]:
    """The frames to label: those with at least one marker at ``min_likelihood``, then

    * ``frames``: those of them (a list of frame numbers);
    * ``labeled``: those already labeled (a set of frame numbers) -- markers added to
      the frames you have;
    * ``every``: all of them;
    * else ``n``, spread evenly over them in time.
    """
    import numpy as np

    seen = long[long["x"].notna() & long["y"].notna() & (long["likelihood"] >= min_likelihood)]
    candidates = sorted(set(seen["frame"].astype(int)))
    if frames is not None:
        return [f for f in sorted(set(map(int, frames))) if f in set(candidates)]
    if labeled is not None:
        return [f for f in candidates if f in set(labeled)]
    if every or n is None or n >= len(candidates):
        return candidates
    if n <= 0:
        return []
    picks = np.array(candidates)[np.linspace(0, len(candidates) - 1, n).round().astype(int)]
    return sorted(set(int(f) for f in picks))


# ---------------------------------------------------------------------------- labels
def import_labels(project, long, vid: str, kind: str, frames: list[int], *, overwrite: bool = False,
                  propose: bool = False, min_likelihood: float = 0.0, scorer: str = "import",
                  dry_run: bool = False) -> dict:
    """Add ``long``'s markers at ``frames`` to ``vid``'s labels (or proposals).

    Frames not extracted yet are extracted (from the original video, and the
    processed one when registered). The positions go from ``kind`` pixels to the
    pixels the labels are stored in. A marker already labeled in a frame keeps its
    position unless ``overwrite``; markers below ``min_likelihood`` are not placed.

    The labels are written to ``labels.parquet`` and to the annotator's copy in
    ``.annotate/`` when there is one -- that copy replaces ``labels.parquet`` when
    napari closes, so it must hold them too; its previous version is kept as
    ``.bak``.

    With ``propose`` the positions are staged as machine labels instead (see
    :func:`~.refine.propose_labels`): neither file is touched, and frames already
    labeled are skipped.

    Returns a summary dict: ``frames`` (selected), ``extracted`` (new frame files),
    ``placed``, ``kept`` (already labeled, left), ``replaced``, ``target``.
    """
    import shutil

    import pandas as pd

    from .annotate import labeled_images
    from .annotations import (
        collected_data_to_long_df,
        find_collected_data,
        long_df_to_collected_data,
        read_collected_data,
        write_collected_data,
        write_labels_parquet,
    )
    from .frames import add_frames, frame_index
    from .manifest import write_manifest
    from .schema import LabelsRecord

    sx, sy = to_original_scale(project, vid, kind)
    summary = {"frames": len(frames), "extracted": 0, "placed": 0, "kept": 0, "replaced": 0, "target": None}
    if dry_run or not frames:
        return summary
    written = add_frames(project, vid, frames)
    summary["extracted"] = len(written)
    names = {}
    for p in project.layout.frames_dir(vid, "original").glob("img*.png"):
        i = frame_index(p.name)
        if i is not None and i in set(frames):
            names[i] = p.name
    missing = sorted(set(frames) - set(names))
    if missing:
        log.warning("%s: %d frame(s) could not be extracted (past the end of the video?): %s",
                    vid, len(missing), missing[:10])
    picked = long[long["frame"].isin(list(names))].copy()

    if propose:
        from .refine import propose_labels

        target, placed = propose_labels(project, vid, picked, {n: f for f, n in names.items()},
                                        scale=(sx, sy), min_likelihood=min_likelihood, scorer=scorer)
        summary.update(placed=placed, target=str(target))
        return summary

    placed_ok = picked["x"].notna() & picked["y"].notna() & (picked["likelihood"] >= min_likelihood)
    picked = picked[placed_ok]
    new = pd.DataFrame({"image": picked["frame"].map(names), "individual": picked["individual"],
                        "bodypart": picked["bodypart"], "x": picked["x"] * sx, "y": picked["y"] * sy})

    # what is labeled now, in original pixels: the annotator's copy when it has one (it
    # replaces labels.parquet on close), else labels.parquet
    parquet = project.layout.labels_parquet(vid)
    stored_to_orig = project.labels_scale_to(vid, "original") if parquet.is_file() else (1.0, 1.0)
    dataset_dir = project.layout.staging_dataset_dir(vid)
    staged = find_collected_data(dataset_dir)
    if staged is not None:
        base = collected_data_to_long_df(read_collected_data(staged))
    elif parquet.is_file():
        base = pd.read_parquet(parquet)
        base = base.assign(x=base["x"] * stored_to_orig[0], y=base["y"] * stored_to_orig[1])
    else:
        base = pd.DataFrame(columns=["image", "individual", "bodypart", "x", "y"])
    base = base[["image", "individual", "bodypart", "x", "y"]]

    key = ["image", "individual", "bodypart"]
    have = base[base["x"].notna() & base["y"].notna()].set_index(key).index
    hit = new.set_index(key).index.isin(have)
    summary["kept" if not overwrite else "replaced"] = int(hit.sum())
    if not overwrite:
        new = new[~hit]
    rest = base.set_index(key).drop(new.set_index(key).index, errors="ignore").reset_index()
    merged = pd.concat([rest, new], ignore_index=True) if len(rest) else new.reset_index(drop=True)
    merged = merged.sort_values(["image", "individual", "bodypart"], kind="stable").reset_index(drop=True)
    summary["placed"] = len(new)

    # labels.parquet in the pixels it is stored in (original for labels made now)
    to_stored = (1.0 / stored_to_orig[0], 1.0 / stored_to_orig[1])
    out = merged.assign(x=merged["x"] * to_stored[0], y=merged["y"] * to_stored[1])
    if not project.layout.labels_toml(vid).is_file():
        write_manifest(project.layout.labels_toml(vid), LabelsRecord(video_id=vid, space="original").to_dict())
    write_labels_parquet(out, parquet)
    summary["target"] = str(parquet)
    if staged is not None:
        shutil.copy2(staged, staged.with_name(staged.name + ".bak"))
        wide = long_df_to_collected_data(merged, project, scorer=_staged_scorer(staged), dataset=vid)
        write_collected_data(wide, staged)
    log.info("%s: %d marker(s) in %d frame(s); %d labeled frame(s) now",
             vid, summary["placed"], len(names), len(labeled_images(merged)))
    return summary


def _staged_scorer(path: Path) -> str:
    stem = Path(path).stem
    return stem[len("CollectedData_"):] if stem.startswith("CollectedData_") else "labeler"


# ---------------------------------------------------------------------------- runs
def import_run(project, items, *, note: str = ""):
    """Make one ``analyze`` run of imported poses.

    ``items`` is a list of ``(source, long, video_id, kind)``. Each pose is written
    as ``apply`` writes one, named after the video it belongs to (its registered
    media of ``kind``): ``<video>.parquet`` beside a ``<video>.toml`` record whose
    ``inputs`` name that media, so ``compare`` and ``extract --from-run`` find the
    video. The record and ``run.toml`` say where the markers came from
    (``params.imported_from``). Coordinates are kept in the video's pixels.

    Returns ``(run, [pose paths])``.

    Raises:
        ValueError: if two items belong to the same video.
    """
    from .apply import pose_record_path, run_pose_path, write_pose_parquet
    from .frames import resolve_media
    from .manifest import write_manifest
    from .schema import RunManifest, now_iso
    from .util import code_version

    by_video: dict[tuple[str, str], str] = {}
    media = []
    for source, _, vid, kind in items:
        if (vid, kind) in by_video:
            raise ValueError(f"{source} and {by_video[(vid, kind)]} both belong to the {kind} video {vid}; "
                             "import them in separate runs")
        by_video[(vid, kind)] = str(source)
        try:
            media.append(resolve_media(project, vid, kind))
        except FileNotFoundError:
            media.append(project.layout.video_media(vid, ".mp4", kind))
    sources = [str(Path(s).resolve()) for s, _, _, _ in items]
    run = project.new_run("analyze", inputs=[str(Path(m).absolute()) for m in media],
                          params={"imported_from": sources})
    run.start()
    paths = []
    for (source, long, vid, kind), video in zip(items, media, strict=True):
        started = now_iso()
        pose = write_pose_parquet(long[["frame", "individual", "bodypart", "x", "y", "likelihood"]],
                                  run_pose_path(run.dir, video))
        record = RunManifest(
            run_id=Path(video).stem, kind="analyze", status="finished", started=started, finished=now_iso(),
            inputs=[str(Path(video).absolute())],
            outputs=[pose.name],
            params={"imported_from": str(Path(source).resolve()), "video_id": vid, "kind": kind},
            code_version=code_version(), notes=note or "imported")
        write_manifest(pose_record_path(pose), record.to_dict())
        paths.append(pose)
    run.finish(outputs=[str(p.absolute()) for p in paths])
    run._patch(notes=note or "imported")
    return run, paths
