#
# FreeDLC workspace layer -- tracker tests
#
"""Tests for the greedy centroid tracker. Pure pandas -- no torch/cv2/pyarrow.

Standalone: ``python tests/workspace/test_track.py`` -> ``track: N/N checks passed``.
"""
from __future__ import annotations

import pandas as pd

from deeplabcut.workspace import track as tr


def _mk(rows):
    return pd.DataFrame(rows, columns=["frame", "individual", "bodypart", "x", "y", "likelihood"])


def test_links_two_animals_through_label_swaps():
    rows = []
    for f in range(6):
        a, b = (10 + f, 10), (80 - f, 80)
        order = [("i0", a), ("i1", b)] if f % 2 == 0 else [("i0", b), ("i1", a)]  # swap labels
        for ind, (x, y) in order:
            rows.append((f, ind, "nose", x, y, 0.99))
    out = tr.track_dataframe(_mk(rows), max_distance=20, max_gap=3, pcutoff=0.5)
    assert tr.count_tracks(out) == 2

    def tid(frame, approx_x):
        sub = out[out.frame == frame]
        return sub.iloc[(sub.x - approx_x).abs().argmin()].individual

    assert len({tid(f, 10 + f) for f in range(6)}) == 1   # animal A keeps one id
    assert len({tid(f, 80 - f) for f in range(6)}) == 1   # animal B keeps one id
    assert tid(0, 10) != tid(0, 80)                        # A and B are different tracks


def test_birth_and_single_animal():
    rows = []
    for f in range(6):
        rows.append((f, "p", "nose", 10 + f, 10, 0.9))
        if f >= 3:
            rows.append((f, "q", "nose", 50, 50, 0.9))    # a second animal appears at frame 3
    assert tr.count_tracks(tr.track_dataframe(_mk(rows), max_distance=20, max_gap=3)) == 2

    single = _mk([(f, "single", "nose", 5 + f, 5, 0.9) for f in range(5)])
    assert tr.count_tracks(tr.track_dataframe(single)) == 1


def test_gap_retires_track():
    rows = [(0, "a", "nose", 10, 10, 0.9), (1, "a", "nose", 11, 10, 0.9),
            (6, "a", "nose", 12, 10, 0.9)]                 # reappears after a 5-frame gap
    out = tr.track_dataframe(_mk(rows), max_distance=20, max_gap=2)
    assert tr.count_tracks(out) == 2                       # not linked across the gap


def test_untracked_when_below_pcutoff():
    out = tr.track_dataframe(_mk([(0, "a", "nose", 10, 10, 0.1)]), pcutoff=0.6)
    assert out["individual"].iloc[0].startswith("untracked_")
    assert tr.count_tracks(out) == 0


def test_default_tracked_path():
    from pathlib import Path
    assert tr.tracked_parquet_path("/v/clip.fdlc.parquet") == Path("/v/clip.tracked.fdlc.parquet")
    assert tr.tracked_parquet_path("/v/pose.parquet") == Path("/v/pose.tracked.parquet")


def _run() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for c in checks:
        c()
    print(f"track: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
