#
# FreeDLC workspace layer -- labeled video rendering
#
"""Render a tidy pose DataFrame onto its source video.

Draws each bodypart as a colored dot (one color per bodypart) and, when a
skeleton is given, connects the configured bodypart pairs -- each drawn only when
its likelihood meets ``pcutoff``. Multi-animal frames draw every individual.

Uses cv2 + numpy only (both already required), so it stays decoupled from the
legacy ``make_labeled_video`` path and the wide DLC format. cv2/numpy are
imported lazily inside the functions, so importing this module stays light.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def _bodypart_colors(names: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """A distinct BGR color per bodypart, evenly spaced around the hue wheel."""
    import cv2
    import numpy as np

    n = max(len(names), 1)
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (np.arange(n) * 179 // n).astype(np.uint8)
    hsv[:, 0, 1:] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]
    return {name: tuple(int(c) for c in bgr[i]) for i, name in enumerate(names)}


def _index_by_frame(df) -> dict[int, dict]:
    """``frame -> {individual -> {bodypart -> (x, y, likelihood)}}``."""
    col = {c: i for i, c in enumerate(df.columns)}
    lookup: dict[int, dict] = {}
    for row in df.itertuples(index=False, name=None):
        frame = int(row[col["frame"]])
        ind = row[col["individual"]]
        lookup.setdefault(frame, {}).setdefault(ind, {})[row[col["bodypart"]]] = (
            row[col["x"]], row[col["y"]], row[col["likelihood"]],
        )
    return lookup


def render_labeled_video(
    video: str | Path,
    df,
    out_path: str | Path,
    *,
    bodyparts: Sequence[str],
    skeleton: Sequence[Sequence[str]] | None = None,
    pcutoff: float = 0.6,
    dotsize: int = 5,
    line_thickness: int = 1,
) -> Path:
    """Write an annotated copy of ``video`` to ``out_path``; return that path.

    Requires cv2 at call time. Keypoints and skeleton edges below ``pcutoff`` (or
    with non-finite coordinates) are skipped.
    """
    import math

    import cv2

    video, out_path = Path(video), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _bodypart_colors(list(bodyparts))
    edges = [(a, b) for a, b in (skeleton or [])]
    lookup = _index_by_frame(df)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open a video writer for {out_path} (missing mp4v codec?)")

    def _ok(pt) -> bool:
        return pt is not None and pt[2] >= pcutoff and math.isfinite(pt[0]) and math.isfinite(pt[1])

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for kp in lookup.get(idx, {}).values():
                for a, b in edges:                      # skeleton under the dots
                    pa, pb = kp.get(a), kp.get(b)
                    if _ok(pa) and _ok(pb):
                        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                                 (255, 255, 255), line_thickness)
                for bp, pt in kp.items():
                    if _ok(pt):
                        cv2.circle(frame, (int(round(pt[0])), int(round(pt[1]))),
                                   dotsize, colors.get(bp, (0, 0, 255)), -1)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path
