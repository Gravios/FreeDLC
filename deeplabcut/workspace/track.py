#
# FreeDLC workspace layer -- cross-frame identity tracking
#
"""Assign stable track identities to per-frame pose instances.

The ``apply`` pipeline emits pose *per frame*: for multi-animal top-down models
each frame carries several instances, but their ``individual`` labels are not
linked across frames. This module links them with a simple, dependency-light
tracker so downstream analysis has consistent identities.

Algorithm (v1, greedy centroid association):
  - Each instance's position is the mean of its keypoints above ``pcutoff``.
  - Frame to frame, instances are matched to active tracks by nearest centroid,
    greedily and only within ``max_distance`` px.
  - Unmatched instances start new tracks; tracks unseen for more than ``max_gap``
    frames are retired.

This is intentionally simple. It does not use optical flow, appearance, or
optimal (Hungarian) assignment, so it can swap identities when animals cross or
occlude. Those are the natural next increments (see docs). numpy only; no torch.
"""
from __future__ import annotations


def _instance_centroids(df, pcutoff: float):
    """``(frame, individual) -> (cx, cy)`` from keypoints at or above ``pcutoff``."""
    import numpy as np

    good = df[df["likelihood"] >= pcutoff]
    cent = good.groupby(["frame", "individual"], sort=False)[["x", "y"]].mean()
    out: dict[tuple, tuple[float, float]] = {}
    for (frame, ind), row in cent.iterrows():
        cx, cy = float(row["x"]), float(row["y"])
        if np.isfinite(cx) and np.isfinite(cy):
            out[(int(frame), ind)] = (cx, cy)
    return out


def track_dataframe(df, *, max_distance: float = 50.0, max_gap: int = 10, pcutoff: float = 0.6):
    """Return ``df`` with ``individual`` replaced by stable ``track_N`` ids.

    Instances that never clear ``pcutoff`` (no usable centroid) keep their
    original per-frame label prefixed with ``untracked_``.
    """
    import numpy as np

    centroids = _instance_centroids(df, pcutoff)
    frames = sorted({int(f) for f in df["frame"].unique()})
    instances_per_frame: dict[int, list] = {f: [] for f in frames}
    for f in frames:
        sub = df[df["frame"] == f]
        for ind in dict.fromkeys(sub["individual"]):        # first-seen order, unique
            instances_per_frame[f].append(ind)

    active: list[dict] = []          # {id, last_frame, centroid}
    next_id = 0
    mapping: dict[tuple, str] = {}   # (frame, individual) -> track label

    for f in frames:
        active = [t for t in active if f - t["last_frame"] <= max_gap]
        dets = [(ind, centroids.get((f, ind))) for ind in instances_per_frame[f]]

        pairs = []
        for ti, t in enumerate(active):
            for di, (_ind, c) in enumerate(dets):
                if c is None:
                    continue
                d = float(np.hypot(c[0] - t["centroid"][0], c[1] - t["centroid"][1]))
                if d <= max_distance:
                    pairs.append((d, ti, di))
        pairs.sort()

        used_t, used_d = set(), set()
        for _d, ti, di in pairs:                            # greedy nearest-first
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = active[ti]
            ind, c = dets[di]
            t["last_frame"], t["centroid"] = f, c
            mapping[(f, ind)] = t["label"]

        for di, (ind, c) in enumerate(dets):
            if di in used_d:
                continue
            if c is None:
                mapping[(f, ind)] = f"untracked_{ind}"
                continue
            label = f"track_{next_id}"
            next_id += 1
            active.append({"label": label, "last_frame": f, "centroid": c})
            mapping[(f, ind)] = label

    out = df.copy()
    out["individual"] = [mapping[(int(f), ind)]
                         for f, ind in zip(df["frame"], df["individual"], strict=True)]
    return out


def count_tracks(df) -> int:
    """Number of distinct non-``untracked_`` identities in a tracked df."""
    return sum(1 for t in df["individual"].unique() if not str(t).startswith("untracked_"))


def tracked_parquet_path(parquet):
    """Default output path for a tracked parquet: ``<base>.tracked.fdlc.parquet``."""
    from pathlib import Path

    parquet = Path(parquet)
    if parquet.name.endswith(".fdlc.parquet"):
        base = parquet.name[: -len(".fdlc.parquet")]
        return parquet.with_name(f"{base}.tracked.fdlc.parquet")
    return parquet.with_name(f"{parquet.stem}.tracked.parquet")


def track_parquet(parquet, out_path, *, max_distance: float = 50.0, max_gap: int = 10,
                  pcutoff: float = 0.6):
    """Read a pose parquet, assign track ids, write the result; return ``(path, n_tracks)``."""
    import pandas as pd

    from .apply import write_pose_parquet

    tracked = track_dataframe(
        pd.read_parquet(parquet), max_distance=max_distance, max_gap=max_gap, pcutoff=pcutoff,
    )
    return write_pose_parquet(tracked, out_path), count_tracks(tracked)
