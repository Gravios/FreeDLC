#
# FreeDLC workspace layer
#
"""Applying a model to a video -- the project-less inference path.

This is the formalization of the observation that you do not need a project to
run a model: given a :class:`ModelBundle` and a video, produce a tidy pose
parquet plus a provenance record. In a run (or ``--out``) directory a video's
files are named after it: ``<stem>.parquet`` (the pose), ``<stem>.toml`` (the
record) and, with a labeled video, ``<stem>.mp4``. Runs written before that kept
each video in a folder of its own, as ``<slug>/pose.parquet``, ``run.toml`` and
``labeled.mp4``; :func:`pose_record_path` and :func:`pose_labeled_path` find the
companions of a pose in either layout. Under the hood it builds
runners from the bundle and calls the existing, already-project-less
``deeplabcut.pose_estimation_pytorch.video_inference``.

Output schema is deliberately **tidy/long** -- one row per (frame, individual,
bodypart) with columns ``frame, individual, bodypart, x, y, likelihood`` -- a
single stable schema regardless of model, shuffle or snapshot. This replaces the
wide, scorer-named MultiIndex HDF5 of the legacy pipeline.

``video_inference`` (torch) and ``DataFrame.to_parquet`` (pyarrow) are imported
lazily; the pure array->DataFrame conversion is dependency-light and unit-tested.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .manifest import write_manifest
from .schema import SCHEMA_VERSION, RunManifest, now_iso
from .util import code_version, files_log, shown

log = logging.getLogger(__name__)

__all__ = [
    "SINGLE_INDIVIDUAL",
    "VIDEO_EXTENSIONS",
    "predictions_to_long_df",
    "write_pose_parquet",
    "collect_videos",
    "apply_to_video",
    "apply_to_videos",
    "run_pose_path",
    "pose_record_path",
    "pose_labeled_path",
    "check_run_outputs",
    "check_scale",
    "resize_frame",
    "unscale_predictions",
    "scale_note",
]

#: Video file extensions recognized when expanding directories / globs.
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
)

#: Label used for unique (non-identity) bodyparts in the ``individual`` column.
SINGLE_INDIVIDUAL = "single"


def predictions_to_long_df(
    predictions: Sequence[dict[str, Any]],
    bodyparts: Sequence[str],
    *,
    unique_bodyparts: Sequence[str] | None = None,
    individuals: Sequence[str] | None = None,
):
    """Convert per-frame prediction dicts to a tidy long-format DataFrame.

    Args:
        predictions: one dict per frame, each with a ``"bodyparts"`` array of
            shape ``(n_individuals, n_bodyparts, 3)`` holding ``[x, y, likelihood]``,
            and optionally a ``"unique_bodyparts"`` array of shape
            ``(1, n_unique, 3)`` or ``(n_unique, 3)``.
        bodyparts: ordered bodypart names matching axis 1 of ``"bodyparts"``.
        unique_bodyparts: ordered names for the unique-bodypart array, if present.
        individuals: names for axis 0; defaults to ``idv0, idv1, ...`` (or the
            single value ``"single"`` for a one-individual model).

    Returns:
        A pandas DataFrame with columns
        ``[frame, individual, bodypart, x, y, likelihood]``.
    """
    import numpy as np
    import pandas as pd

    rows_frame: list[int] = []
    rows_ind: list[str] = []
    rows_bpt: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    ls: list[float] = []

    def _emit(frame_idx: int, ind_name: str, names: Sequence[str], arr) -> None:
        for b, bpt in enumerate(names):
            x, y, lk = float(arr[b, 0]), float(arr[b, 1]), float(arr[b, 2])
            rows_frame.append(frame_idx)
            rows_ind.append(ind_name)
            rows_bpt.append(bpt)
            xs.append(x)
            ys.append(y)
            ls.append(lk)

    for frame_idx, pred in enumerate(predictions):
        pose = np.asarray(pred["bodyparts"])
        if pose.ndim == 2:  # (n_bodyparts, 3) -> single individual
            pose = pose[None]
        n_ind = pose.shape[0]
        names = list(individuals) if individuals is not None else (
            [SINGLE_INDIVIDUAL] if n_ind == 1 else [f"idv{i}" for i in range(n_ind)]
        )
        if len(names) < n_ind:
            names = names + [f"idv{i}" for i in range(len(names), n_ind)]
        for i in range(n_ind):
            _emit(frame_idx, names[i], bodyparts, pose[i])

        uniq = pred.get("unique_bodyparts")
        if uniq is not None and unique_bodyparts:
            uniq = np.asarray(uniq)
            if uniq.ndim == 3:
                uniq = uniq[0]
            _emit(frame_idx, SINGLE_INDIVIDUAL, unique_bodyparts, uniq)

    return pd.DataFrame(
        {
            "frame": rows_frame,
            "individual": rows_ind,
            "bodypart": rows_bpt,
            "x": xs,
            "y": ys,
            "likelihood": ls,
        }
    )


def write_pose_parquet(df, path: str | Path) -> Path:
    """Write a pose DataFrame to Parquet (requires pyarrow, imported by pandas)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


#: a pose file in a folder of its own: how runs laid out their videos before
LEGACY_POSE = "pose.parquet"


def run_pose_path(out_root: str | Path, video: str | Path) -> Path:
    """Where a run (or ``--out``) puts the pose of ``video``: ``<out_root>/<stem>.parquet``."""
    return Path(out_root) / f"{Path(video).stem}.parquet"


def pose_record_path(pose: str | Path) -> Path:
    """The provenance record of a run's pose: ``<stem>.toml`` (``run.toml`` for a legacy ``pose.parquet``)."""
    pose = Path(pose)
    return pose.with_name("run.toml") if pose.name == LEGACY_POSE else pose.with_suffix(".toml")


def pose_labeled_path(pose: str | Path) -> Path:
    """The labeled video of a run's pose: ``<stem>.mp4`` (``labeled.mp4`` for a legacy ``pose.parquet``)."""
    pose = Path(pose)
    return pose.with_name("labeled.mp4") if pose.name == LEGACY_POSE else pose.with_suffix(".mp4")


def check_run_outputs(videos, out_root: str | Path | None, *, labeled_video: bool = True) -> None:
    """Refuse videos whose files would collide in ``out_root`` (``None``: a new run directory).

    ``labeled_video`` says whether labeled videos will be written as well.

    Raises:
        ValueError: if two videos share a file name stem, a video is named ``run``
            (its record would replace the run's own ``run.toml``), or a labeled
            video would be written over its source video (``--out`` its own folder).
    """
    problems: list[str] = []
    by_stem: dict[str, list[Path]] = {}
    for video in map(Path, videos):
        by_stem.setdefault(video.stem, []).append(video)
        if video.stem == "run":
            problems.append(f"{video}: a video named 'run' would replace the run's own run.toml; rename it")
        labeled = pose_labeled_path(run_pose_path(out_root, video)) if out_root is not None else None
        if labeled_video and labeled is not None and labeled.resolve() == video.resolve():
            problems.append(f"{video}: its labeled video would be written over it; give another --out "
                            "or use --beside-video")
    for stem, same in by_stem.items():
        if len(same) > 1:
            problems.append(f"{len(same)} videos are named {stem} ({', '.join(map(str, same))}); their files "
                            "would overwrite each other -- apply them in separate runs")
    if problems:
        raise ValueError("\n".join(problems))


#: Suffix for pose files written next to their source video (``--beside-video``).
FDLC_SUFFIX = ".fdlc.parquet"


def beside_video_path(video: str | Path) -> Path:
    """Pose output path next to the source video: ``<dir>/<stem>.fdlc.parquet``."""
    video = Path(video)
    return video.parent / f"{video.stem}{FDLC_SUFFIX}"


def fdlc_sidecar_path(video: str | Path) -> Path:
    """Sidecar metadata path next to the source video: ``<dir>/<stem>.fdlc.toml``."""
    video = Path(video)
    return video.parent / f"{video.stem}.fdlc.toml"


def labeled_video_path(video: str | Path) -> Path:
    """Annotated-video path next to the source video: ``<dir>/<stem>.fdlc.mp4``."""
    video = Path(video)
    return video.parent / f"{video.stem}.fdlc.mp4"


def sidecar_for_parquet(parquet: str | Path) -> Path:
    """The ``.fdlc.toml`` sidecar paired with a ``.fdlc.parquet`` pose file."""
    parquet = Path(parquet)
    base = parquet.name[:-len(FDLC_SUFFIX)] if parquet.name.endswith(FDLC_SUFFIX) else parquet.stem
    return parquet.with_name(f"{base}.fdlc.toml")


def read_fdlc_sidecar(path: str | Path) -> tuple[list[str] | None, list[list[str]]]:
    """Read ``bodyparts`` and ``skeleton`` from a ``.fdlc.toml`` sidecar."""
    from .manifest import read_manifest

    meta = read_manifest(path)
    return (meta.get("bodyparts") or None), [list(e) for e in meta.get("skeleton", [])]


def read_fdlc_display(path: str | Path) -> dict:
    """The ``[display]`` table a ``.fdlc.toml`` sidecar carries (empty if none)."""
    from .manifest import read_manifest

    return dict(read_manifest(path).get("display") or {})


def _render_labeled(video, df, out_path, bundle, skeleton, pcutoff, display=None) -> Path:
    from .display import parse_display, video_style
    from .label_video import render_labeled_video  # lazy: pulls cv2

    style = video_style(parse_display(display), list(bundle.card.bodyparts)) if display else {}
    return render_labeled_video(
        video, df, out_path,
        bodyparts=bundle.card.bodyparts, skeleton=skeleton, pcutoff=pcutoff, **style,
    )


def _write_fdlc_sidecar(video, pose_path, bundle, *, skeleton, params, snapshot, started,
                        display=None) -> Path:
    """Write the ``<stem>.fdlc.toml`` sidecar: provenance + bodyparts + skeleton edges (+ display)."""
    video = Path(video)
    card = bundle.card
    data = {
        "schema_version": SCHEMA_VERSION,
        "kind": "analyze",
        "started": started,
        "finished": now_iso(),
        "model_id": card.model_id,
        "snapshot": card.default_snapshot if snapshot == "default" else snapshot,
        "input": str(video.resolve()),
        "output": Path(pose_path).name,
        "bodyparts": list(card.bodyparts),
        "skeleton": [list(e) for e in (skeleton or [])],
        "params": dict(params),
        "code_version": code_version(),
    }
    if display:  # so `fdlc label` draws it the same way later
        data["display"] = dict(display)
    return write_manifest(fdlc_sidecar_path(video), data)


def collect_videos(paths: Sequence[str | Path], *, extensions=VIDEO_EXTENSIONS) -> list[Path]:
    """Expand file / directory / glob paths into a sorted, de-duplicated video list.

    - a directory yields the video files directly inside it (by extension),
    - a glob pattern (containing ``*``, ``?`` or ``[``) is expanded; ``**`` matches any
      number of folders,
    - any other path is taken as-is (so an explicit file is never filtered out).
    """
    from glob import glob

    collected: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            collected += [
                f for f in sorted(p.iterdir())
                if f.is_file() and f.suffix.lower() in extensions
            ]
        elif any(ch in str(raw) for ch in "*?["):
            collected += [
                Path(g) for g in sorted(glob(str(raw), recursive=True)) if Path(g).suffix.lower() in extensions
            ]
        else:
            collected.append(p)

    seen: set[Path] = set()
    unique: list[Path] = []
    for f in collected:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _report_bundle(bundle, snapshot: str, detector_snapshot: str) -> None:
    """Tell the file report which model files a run is about to use."""
    files_log.info("model bundle %s", shown(bundle.path))
    files_log.info("  config    %s", shown(bundle.pose_config_path))
    files_log.info("  snapshot  %s", shown(bundle.snapshot_path(snapshot)))
    if bundle.card.top_down:
        files_log.info("  detector  %s", shown(bundle.detector_snapshot_path(detector_snapshot)))
    if getattr(bundle.card, "frames", None):
        files_log.info("  trained on %s frames -- apply it to video of that resolution", bundle.card.frames)


def _build_runners(bundle, *, snapshot, detector_snapshot, device, batch_size, max_individuals):
    """Build the pose runner (and detector runner for top-down models) once."""
    pose_runner = bundle.build_pose_runner(
        snapshot=snapshot, device=device, batch_size=batch_size, max_individuals=max_individuals
    )
    detector_runner = None
    if bundle.card.top_down:
        detector_runner = bundle.build_detector_runner(
            snapshot=detector_snapshot, device=device, batch_size=batch_size
        )
    return pose_runner, detector_runner


def check_scale(scale: float) -> float:
    """``scale`` as a float, refused unless it is positive and finite."""
    import math

    scale = float(scale)
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"--scale must be a positive number, got {scale}")
    return scale


def resize_frame(frame, scale: float):
    """``frame`` resized by ``scale`` (area averaging to shrink, bilinear to enlarge); as is at 1."""
    if scale == 1.0:
        return frame
    import cv2

    h, w = frame.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def unscale_predictions(predictions, scale: float):
    """Prediction dicts made on frames resized by ``scale``, put back in the video's pixels.

    The ``x, y`` of ``bodyparts`` and ``unique_bodyparts`` (and the boxes of ``bboxes``)
    are divided by ``scale``; likelihoods are left, and so is the ``-1, -1`` a model
    gives for a marker it did not place.
    """
    import numpy as np

    if scale == 1.0:
        return predictions
    out = []
    for pred in predictions:
        pred = dict(pred)
        for key in ("bodyparts", "unique_bodyparts"):
            if pred.get(key) is not None:
                arr = np.array(pred[key], dtype=float, copy=True)
                xy = arr[..., :2]
                unplaced = np.all(xy == -1, axis=-1, keepdims=True)
                arr[..., :2] = np.where(unplaced, xy, xy / scale)
                pred[key] = arr
        if pred.get("bboxes") is not None:
            pred["bboxes"] = np.asarray(pred["bboxes"], dtype=float) / scale
        out.append(pred)
    return out


#: inference transforms after which the model sees a fixed size whatever the frame's
INPUT_RESIZERS = ("resize", "longest_max_size", "top_down_crop")


def scale_note(bundle, scale: float) -> str | None:
    """A warning when ``scale`` can't do what it is for: the bundle's own inference
    transforms resize its input (or crop each animal to a fixed size, top-down) anyway."""
    if scale == 1.0:
        return None
    import yaml

    try:
        with open(bundle.pose_config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        inference = (config.get("data") or {}).get("inference") or {}
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return None
    resizers = [k for k in INPUT_RESIZERS if inference.get(k)]
    if not resizers and not getattr(bundle.card, "top_down", False):
        return None
    what = ", ".join(resizers) or "top-down crops"
    return (f"--scale {scale:g}: model {bundle.card.model_id} resizes its own input ({what}), so the animals "
            "reach it at about the same size whatever the scale")


def _scaled_video(video, cropping, scale: float):
    """A ``VideoIterator`` on ``video`` whose frames come resized by ``scale`` (imports torch)."""
    from deeplabcut.pose_estimation_pytorch.apis.videos import VideoIterator

    class _Scaled(VideoIterator):
        def __next__(self):
            item = super().__next__()
            if isinstance(item, tuple):
                return resize_frame(item[0], scale), item[1]
            return resize_frame(item, scale)

    return _Scaled(str(video), cropping=cropping)


def _infer_to_df(bundle, video, pose_runner, detector_runner, *, cropping, scale: float = 1.0):
    """Run inference on one video with pre-built runners; return a tidy long DataFrame.

    With ``scale`` the frames are resized by it before they reach the model (and the
    detector, for a top-down model), and the positions are put back in the video's
    pixels: for animals filmed larger or smaller than those the model learned.
    """
    from deeplabcut.pose_estimation_pytorch import video_inference

    source = str(video) if scale == 1.0 else _scaled_video(video, cropping, scale)
    predictions = video_inference(
        video=source, pose_runner=pose_runner, detector_runner=detector_runner, cropping=cropping
    )
    predictions = unscale_predictions(predictions, scale)
    meta = bundle._read_pose_config().get("metadata", {})
    return predictions_to_long_df(
        predictions, bodyparts=bundle.card.bodyparts,
        unique_bodyparts=meta.get("unique_bodyparts") or None,
    )


def _write_video_outputs(df, video, out_dir, bundle, *, snapshot, batch_size, device, cropping, started,
                         pcutoff=None, scale=1.0):
    pose_path = write_pose_parquet(df, run_pose_path(out_dir, video))
    run = RunManifest(
        run_id=Path(video).stem,
        kind="analyze",
        status="finished",
        started=started,
        finished=now_iso(),
        model_id=bundle.card.model_id,
        snapshot=bundle.card.default_snapshot if snapshot == "default" else snapshot,
        inputs=[str(Path(video).resolve())],
        outputs=[pose_path.name],
        params={"batch_size": batch_size, "device": device, "cropping": cropping, "pcutoff": pcutoff,
                "scale": scale},
        code_version=code_version(),
    )
    write_manifest(pose_record_path(pose_path), run.to_dict())
    return pose_path


def apply_to_video(
    bundle,
    video: str | Path,
    out_dir: str | Path,
    *,
    device: str | None = None,
    batch_size: int = 1,
    snapshot: str = "default",
    detector_snapshot: str = "default",
    max_individuals: int | None = None,
    cropping: list[int] | None = None,
    write: bool = True,
    beside_video: bool = False,
    skeleton: list[list[str]] | None = None,
    labeled_video: bool = False,
    pcutoff: float = 0.6,
    runners=None,
    display=None,
    scale: float = 1.0,
):
    """Run a :class:`ModelBundle` on one video, writing its pose and record into ``out_dir``.

    The files are named after the video: ``<stem>.parquet`` and ``<stem>.toml``.
    Needs no project. Requires torch at call time (via the runner + video
    inference). Returns the path to the written pose parquet (or the
    in-memory DataFrame when ``write=False``). With ``beside_video=True`` the
    pose is written as ``<video-stem>.fdlc.parquet`` next to the source video,
    together with a ``<video-stem>.fdlc.toml`` sidecar (provenance, bodyparts, and
    ``skeleton`` edges), and no run directory is created. With
    ``labeled_video=True`` an annotated ``.fdlc.mp4`` (beside) / ``<stem>.mp4``
    (``out_dir``) is also rendered, styled by ``display`` (a project's ``[display]``
    table) when given. Pass ``runners`` (from :func:`apply_to_videos`) to reuse a
    runner already built for this bundle. ``scale`` resizes the frames before the model
    sees them (positions are written in the video's pixels all the same): ``0.77`` to
    show a model animals ``1/0.77`` = 1.3x larger than those it was trained on at the
    size it knows.
    """
    started = now_iso()
    scale = check_scale(scale)
    if runners is None:
        _report_bundle(bundle, snapshot, detector_snapshot)
        runners = _build_runners(
            bundle, snapshot=snapshot, detector_snapshot=detector_snapshot,
            device=device, batch_size=batch_size, max_individuals=max_individuals,
        )
    pose_runner, detector_runner = runners
    resized = f" (frames resized by {scale:g} for the model)" if scale != 1 else ""
    files_log.info("video %s%s", shown(video), resized)
    df = _infer_to_df(bundle, video, pose_runner, detector_runner, cropping=cropping, scale=scale)
    if not write:
        return df
    if beside_video:
        pose_path = write_pose_parquet(df, beside_video_path(video))
        sidecar = _write_fdlc_sidecar(
            video, pose_path, bundle, skeleton=skeleton,
            params={"batch_size": batch_size, "device": device, "cropping": cropping, "scale": scale},
            snapshot=snapshot, started=started, display=display,
        )
        files_log.info("  pose     %s (%d frame(s))", shown(pose_path), df["frame"].nunique() if len(df) else 0)
        files_log.info("  sidecar  %s", shown(sidecar))
        if labeled_video:
            _render_labeled(video, df, labeled_video_path(video), bundle, skeleton, pcutoff, display)
            files_log.info("  labeled  %s", shown(labeled_video_path(video)))
        return pose_path
    pose_path = _write_video_outputs(
        df, video, out_dir, bundle,
        snapshot=snapshot, batch_size=batch_size, device=device, cropping=cropping, started=started,
        pcutoff=pcutoff, scale=scale,
    )
    files_log.info("  pose     %s (%d frame(s))", shown(pose_path), df["frame"].nunique() if len(df) else 0)
    files_log.info("  record   %s", shown(pose_record_path(pose_path)))
    if labeled_video:
        _render_labeled(video, df, pose_labeled_path(pose_path), bundle, skeleton, pcutoff, display)
        files_log.info("  labeled  %s", shown(pose_labeled_path(pose_path)))
    return pose_path


def apply_to_videos(
    bundle,
    videos: Sequence[str | Path],
    out_root: str | Path,
    *,
    device: str | None = None,
    batch_size: int = 1,
    snapshot: str = "default",
    detector_snapshot: str = "default",
    max_individuals: int | None = None,
    cropping: list[int] | None = None,
    beside_video: bool = False,
    skeleton: list[list[str]] | None = None,
    labeled_video: bool = False,
    pcutoff: float = 0.6,
    on_error: str = "raise",
    display=None,
    scale: float = 1.0,
) -> dict[str, Path | None]:
    """Label several videos with one bundle, building the runner **once**.

    Each video's outputs are written into ``out_root`` as ``<video-stem>.parquet``
    and ``<video-stem>.toml`` (and ``<video-stem>.mp4`` with ``labeled_video``); videos
    whose files would collide are refused first (:func:`check_run_outputs`). With
    ``beside_video=True`` each pose is instead written as ``<video-stem>.fdlc.parquet``
    next to its source video and ``out_root`` is ignored. Returns
    ``{video_path: pose_parquet_path}``; on a per-video failure, ``on_error="skip"``
    records ``None`` and continues, while ``on_error="raise"`` (default) re-raises.
    ``scale`` resizes every frame before inference (see :func:`apply_to_video`).
    """
    out_root = Path(out_root)
    videos = [Path(v) for v in videos]
    scale = check_scale(scale)
    note = scale_note(bundle, scale)
    if note:
        log.warning(note)
    if not beside_video:
        check_run_outputs(videos, out_root, labeled_video=labeled_video)
    _report_bundle(bundle, snapshot, detector_snapshot)
    runners = _build_runners(
        bundle, snapshot=snapshot, detector_snapshot=detector_snapshot,
        device=device, batch_size=batch_size, max_individuals=max_individuals,
    )
    results: dict[str, Path | None] = {}
    for video in videos:
        out_dir = None if beside_video else out_root
        try:
            results[str(video)] = apply_to_video(
                bundle, video, out_dir,
                snapshot=snapshot, detector_snapshot=detector_snapshot,
                device=device, batch_size=batch_size, cropping=cropping, scale=scale,
                beside_video=beside_video, skeleton=skeleton,
                labeled_video=labeled_video, pcutoff=pcutoff, runners=runners, display=display,
            )
        except Exception:
            if on_error != "skip":
                raise
            log.exception("failed to analyze %s", video)
            results[str(video)] = None
    return results
