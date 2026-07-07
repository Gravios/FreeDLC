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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .manifest import write_manifest
from .schema import RunManifest, now_iso
from .util import code_version

__all__ = [
    "SINGLE_INDIVIDUAL",
    "predictions_to_long_df",
    "write_pose_parquet",
    "apply_to_video",
]

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
):
    """Run a :class:`ModelBundle` on one video, writing ``pose.parquet`` + ``run.toml``.

    Needs no project. Requires torch at call time (via the runner + video
    inference). Returns the path to the written ``pose.parquet`` (or the
    in-memory DataFrame when ``write=False``).
    """
    from deeplabcut.pose_estimation_pytorch import video_inference

    out_dir = Path(out_dir)
    started = now_iso()

    pose_runner = bundle.build_pose_runner(
        snapshot=snapshot, device=device, batch_size=batch_size, max_individuals=max_individuals
    )
    detector_runner = None
    if bundle.card.top_down:
        detector_runner = bundle.build_detector_runner(
            snapshot=detector_snapshot, device=device, batch_size=batch_size
        )

    predictions = video_inference(
        video=str(video),
        pose_runner=pose_runner,
        detector_runner=detector_runner,
        cropping=cropping,
    )

    meta = bundle._read_pose_config().get("metadata", {})
    df = predictions_to_long_df(
        predictions,
        bodyparts=bundle.card.bodyparts,
        unique_bodyparts=meta.get("unique_bodyparts") or None,
    )

    if not write:
        return df

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
