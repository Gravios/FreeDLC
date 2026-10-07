#
# FreeDLC workspace layer
#
"""Pose-estimation metrics.

Operate on tidy long DataFrames (``image, individual, bodypart, x, y[, likelihood]``)
-- the schema produced by both annotation ingest (ground truth) and inference
(predictions). Pure pandas/numpy, so fully unit-tested.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["pose_error", "DEFAULT_KEYS"]

DEFAULT_KEYS = ("image", "individual", "bodypart")


def pose_error(
    predictions,
    ground_truth,
    *,
    on: Sequence[str] = DEFAULT_KEYS,
    pcutoff: float | None = None,
    pck_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare predicted vs. ground-truth keypoints.

    Both inputs are long DataFrames sharing the ``on`` keys plus ``x``/``y``;
    predictions may also carry ``likelihood``. Rows with missing ground truth are
    ignored (unlabeled keypoints).

    Returns a dict with ``n`` (compared keypoints), ``mean_error`` (mean Euclidean
    pixel distance), ``median_error``, ``rmse`` (root-mean-square distance),
    ``per_bodypart`` (mean error by bodypart) and ``per_bodypart_median``. If
    ``pcutoff`` is given, also reports the confident subset
    (``mean_error_confident``, ``median_error_confident``, ``n_confident``); if
    ``pck_threshold`` is given, the PCK (fraction of keypoints within the threshold),
    overall and ``per_bodypart_pck``.

    Means and medians are both reported because pose errors are heavy-tailed: a
    marker that is right in most frames and far off in a few has a small median and
    a large mean. The median says how good the typical prediction is, the PCK how
    often it is usable; the mean alone cannot tell "always mediocre" from "usually
    exact, sometimes lost".
    """
    import numpy as np
    import pandas as pd

    on = list(on)
    merged = pd.merge(ground_truth, predictions, on=on, suffixes=("_gt", "_pred"), how="inner")
    merged = merged.dropna(subset=["x_gt", "y_gt", "x_pred", "y_pred"])
    dist = np.sqrt((merged["x_pred"] - merged["x_gt"]) ** 2 + (merged["y_pred"] - merged["y_gt"]) ** 2)
    merged = merged.assign(_error=dist)

    def _mean(s) -> float:
        return float(s.mean()) if len(s) else float("nan")

    def _median(s) -> float:
        return float(s.median()) if len(s) else float("nan")

    def _rmse(s) -> float:
        return float(np.sqrt((s**2).mean())) if len(s) else float("nan")

    result: dict[str, Any] = {
        "n": int(len(merged)),
        "mean_error": _mean(merged["_error"]),
        "median_error": _median(merged["_error"]),
        "rmse": _rmse(merged["_error"]),
        "per_bodypart": {str(bpt): _mean(g["_error"]) for bpt, g in merged.groupby("bodypart")},
        "per_bodypart_median": {str(bpt): _median(g["_error"]) for bpt, g in merged.groupby("bodypart")},
    }
    if pcutoff is not None and "likelihood" in merged.columns:
        confident = merged[merged["likelihood"] >= pcutoff]
        result["n_confident"] = int(len(confident))
        result["mean_error_confident"] = _mean(confident["_error"])
        result["median_error_confident"] = _median(confident["_error"])
    if pck_threshold is not None:
        result["pck"] = float((merged["_error"] <= pck_threshold).mean()) if len(merged) else float("nan")
        result["pck_threshold"] = pck_threshold
        result["per_bodypart_pck"] = {
            str(bpt): float((g["_error"] <= pck_threshold).mean()) for bpt, g in merged.groupby("bodypart")
        }
    return result
