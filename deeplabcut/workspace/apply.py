#
# FreeDLC workspace layer
#
"""Applying a model to a video -- the project-less inference path.

This is the formalization of the observation that you do not need a project to
run a model: given a :class:`ModelBundle` and a video, produce a tidy
``pose.parquet`` plus a ``run.toml`` provenance record. Under the hood it builds
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

from . import ids
from .manifest import write_manifest
from .schema import SCHEMA_VERSION, RunManifest, now_iso
from .util import code_version

log = logging.getLogger(__name__)

__all__ = [
    "SINGLE_INDIVIDUAL",
    "VIDEO_EXTENSIONS",
    "predictions_to_long_df",
    "write_pose_parquet",
    "collect_videos",
    "apply_to_video",
    "apply_to_videos",
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


def _write_fdlc_sidecar(video, pose_path, bundle, *, skeleton, params, snapshot, started) -> Path:
    """Write the ``<stem>.fdlc.toml`` sidecar: provenance + bodyparts + skeleton edges."""
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
    return write_manifest(fdlc_sidecar_path(video), data)


def collect_videos(paths: Sequence[str | Path], *, extensions=VIDEO_EXTENSIONS) -> list[Path]:
    """Expand file / directory / glob paths into a sorted, de-duplicated video list.

    - a directory yields the video files directly inside it (by extension),
    - a glob pattern (containing ``*``, ``?`` or ``[``) is expanded,
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
                Path(g) for g in sorted(glob(str(raw))) if Path(g).suffix.lower() in extensions
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


def _infer_to_df(bundle, video, pose_runner, detector_runner, *, cropping):
    """Run inference on one video with pre-built runners; return a tidy long DataFrame."""
    from deeplabcut.pose_estimation_pytorch import video_inference

    predictions = video_inference(
        video=str(video), pose_runner=pose_runner, detector_runner=detector_runner, cropping=cropping
    )
    meta = bundle._read_pose_config().get("metadata", {})
    return predictions_to_long_df(
        predictions, bodyparts=bundle.card.bodyparts,
        unique_bodyparts=meta.get("unique_bodyparts") or None,
    )


def _write_video_outputs(df, video, out_dir, bundle, *, snapshot, batch_size, device, cropping, started):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_path = write_pose_parquet(df, out_dir / "pose.parquet")
    run = RunManifest(
        run_id=out_dir.name,
        kind="analyze",
        status="finished",
        started=started,
        finished=now_iso(),
        model_id=bundle.card.model_id,
        snapshot=bundle.card.default_snapshot if snapshot == "default" else snapshot,
        inputs=[str(Path(video).resolve())],
        outputs=[pose_path.name],
        params={"batch_size": batch_size, "device": device, "cropping": cropping},
        code_version=code_version(),
    )
    write_manifest(out_dir / "run.toml", run.to_dict())
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
    runners=None,
):
    """Run a :class:`ModelBundle` on one video, writing ``pose.parquet`` + ``run.toml``.

    Needs no project. Requires torch at call time (via the runner + video
    inference). Returns the path to the written ``pose.parquet`` (or the
    in-memory DataFrame when ``write=False``). With ``beside_video=True`` the
    pose is written as ``<video-stem>.fdlc.parquet`` next to the source video,
    together with a ``<video-stem>.fdlc.toml`` sidecar (provenance, bodyparts, and
    ``skeleton`` edges), and no run directory is created. Pass ``runners`` (from
    :func:`apply_to_videos`) to reuse a runner already built for this bundle.
    """
    started = now_iso()
    if runners is None:
        runners = _build_runners(
            bundle, snapshot=snapshot, detector_snapshot=detector_snapshot,
            device=device, batch_size=batch_size, max_individuals=max_individuals,
        )
    pose_runner, detector_runner = runners
    df = _infer_to_df(bundle, video, pose_runner, detector_runner, cropping=cropping)
    if not write:
        return df
    if beside_video:
        pose_path = write_pose_parquet(df, beside_video_path(video))
        _write_fdlc_sidecar(
            video, pose_path, bundle, skeleton=skeleton,
            params={"batch_size": batch_size, "device": device, "cropping": cropping},
            snapshot=snapshot, started=started,
        )
        return pose_path
    return _write_video_outputs(
        df, video, out_dir, bundle,
        snapshot=snapshot, batch_size=batch_size, device=device, cropping=cropping, started=started,
    )


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
    on_error: str = "raise",
) -> dict[str, Path | None]:
    """Label several videos with one bundle, building the runner **once**.

    Each video's outputs are written to ``out_root/<video-stem>/pose.parquet``
    (plus ``run.toml``). With ``beside_video=True`` each pose is instead written
    as ``<video-stem>.fdlc.parquet`` next to its source video and ``out_root`` is
    ignored. Returns ``{video_path: pose_parquet_path}``; on a per-video failure,
    ``on_error="skip"`` records ``None`` and continues, while ``on_error="raise"``
    (default) re-raises.
    """
    out_root = Path(out_root)
    runners = _build_runners(
        bundle, snapshot=snapshot, detector_snapshot=detector_snapshot,
        device=device, batch_size=batch_size, max_individuals=max_individuals,
    )
    results: dict[str, Path | None] = {}
    for video in videos:
        video = Path(video)
        out_dir = None if beside_video else out_root / ids.slugify(video.stem)
        try:
            results[str(video)] = apply_to_video(
                bundle, video, out_dir,
                snapshot=snapshot, detector_snapshot=detector_snapshot,
                device=device, batch_size=batch_size, cropping=cropping,
                beside_video=beside_video, skeleton=skeleton, runners=runners,
            )
        except Exception:
            if on_error != "skip":
                raise
            log.exception("failed to analyze %s", video)
            results[str(video)] = None
    return results
