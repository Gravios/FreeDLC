#
# FreeDLC workspace layer -- labeled video rendering
#
"""Render a tidy pose DataFrame onto its source video.

Draws each bodypart as a colored dot (one color per bodypart) and, when a
skeleton is given, connects the configured bodypart pairs -- each drawn only when
its likelihood meets ``pcutoff``. Multi-animal frames draw every individual.
Colors, dot sizes and line colors/widths default to a hue wheel, ``dotsize`` and
white lines, and can be set per marker and per edge (see :mod:`.display`).

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
    progress: bool = True,
    marker_colors: dict[str, tuple[int, int, int]] | None = None,
    marker_radii: dict[str, int] | None = None,
    line_color: tuple[int, int, int] = (255, 255, 255),
    edge_colors: dict[frozenset, tuple[int, int, int]] | None = None,
    edge_widths: dict[frozenset, int] | None = None,
) -> Path:
    """Write an annotated copy of ``video`` to ``out_path``; return that path.

    Requires cv2 at call time. Keypoints and skeleton edges below ``pcutoff`` (or
    with non-finite coordinates) are skipped. A tqdm progress bar over the frames
    is shown unless ``progress=False``.

    ``dotsize`` is the dot radius and ``line_thickness`` the line width, in pixels.
    ``marker_colors`` (BGR) and ``marker_radii`` override them per bodypart, and
    ``edge_colors`` (BGR) and ``edge_widths`` per skeleton edge, keyed by
    ``frozenset({a, b})``; ``line_color`` is the color of the other edges.
    """
    import math

    import cv2
    from tqdm import tqdm

    video, out_path = Path(video), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {**_bodypart_colors(list(bodyparts)), **(marker_colors or {})}
    radii = marker_radii or {}
    edges = [(a, b, (edge_colors or {}).get(frozenset((a, b)), line_color),
              (edge_widths or {}).get(frozenset((a, b)), line_thickness)) for a, b in (skeleton or [])]
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

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    bar = tqdm(total=total, desc="Labeling video", unit="frame", disable=not progress)
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for kp in lookup.get(idx, {}).values():
                for a, b, color, width in edges:        # skeleton under the dots
                    pa, pb = kp.get(a), kp.get(b)
                    if _ok(pa) and _ok(pb):
                        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, width)
                for bp, pt in kp.items():
                    if _ok(pt):
                        cv2.circle(frame, (int(round(pt[0])), int(round(pt[1]))),
                                   radii.get(bp, dotsize), colors.get(bp, (0, 0, 255)), -1)
            writer.write(frame)
            idx += 1
            bar.update(1)
    finally:
        bar.close()
        cap.release()
        writer.release()
    return out_path


def render_labeled_from_parquet(
    video: str | Path,
    parquet: str | Path,
    out_path: str | Path,
    *,
    bodyparts: Sequence[str] | None = None,
    skeleton: Sequence[Sequence[str]] | None = None,
    pcutoff: float = 0.6,
    display=None,
    **kwargs,
) -> Path:
    """Render an annotated video from an already-written pose parquet.

    ``bodyparts`` defaults to the distinct bodyparts in the parquet (in order of
    first appearance) when not supplied. ``display`` is a project's ``[display]``
    table; keyword arguments given explicitly take precedence over it.
    """
    import pandas as pd

    from .display import parse_display, video_style

    df = pd.read_parquet(parquet)
    if bodyparts is None:
        bodyparts = list(dict.fromkeys(df["bodypart"]))
    if display:
        kwargs = {**video_style(parse_display(display), list(bodyparts)), **kwargs}
    return render_labeled_video(
        video, df, out_path, bodyparts=bodyparts, skeleton=skeleton, pcutoff=pcutoff, **kwargs,
    )
