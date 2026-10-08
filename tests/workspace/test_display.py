#
# FreeDLC workspace layer -- display style tests
#
"""Tests for the ``[display]`` table: parsing, project.toml, the annotator config and
labeled videos. Needs matplotlib, and cv2 for the rendering check -- no torch.

Standalone: ``python tests/workspace/test_display.py`` -> ``display: N/N checks passed``.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut import workspace as ws
from deeplabcut.workspace import annotate as ann
from deeplabcut.workspace import cli
from deeplabcut.workspace.display import annotator_config, parse_display, video_style

BODYPARTS = ["nose", "head", "tail"]
SKELETON = [["nose", "head"], ["head", "tail"]]
TABLE = {
    "dotsize": 6,
    "colormap": "tab10",
    "line_color": "white",
    "line_width": 1,
    "bodyparts": {"nose": {"color": "red", "size": 10}, "tail": {"color": "#00ff00"}},
    "edges": [{"between": ["head", "nose"], "color": "blue", "width": 3}],   # either direction
}


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as err:
        for needle in needles:
            assert needle in str(err), (needle, str(err))
        return str(err)
    raise AssertionError("expected ValueError")


def test_parse_normalizes_colors_and_keys_edges_both_ways():
    style = parse_display(TABLE, bodyparts=BODYPARTS, skeleton=SKELETON)
    assert style.colors == {"nose": "#ff0000", "tail": "#00ff00"}
    assert style.sizes == {"nose": 10.0} and style.dotsize == 6.0 and style.colormap == "tab10"
    assert style.edge_colors == {frozenset({"nose", "head"}): "#0000ff"}
    assert style.edge_widths == {frozenset({"nose", "head"}): 3.0}
    assert parse_display(None) == parse_display({})                  # everything is optional


def test_parse_lists_every_problem_at_once():
    bad = {
        "dotsize": -1,
        "colormap": "no-such-map",
        "colour": "red",                                              # a typo for a key
        "bodyparts": {"nose": {"color": "reddish"}, "paw": {"color": "red"}, "head": {"size": 4, "shape": "x"}},
        "edges": [{"between": ["nose", "tail"], "color": "red"}, {"between": ["head"]}],
    }
    message = _raises(lambda: parse_display(bad, bodyparts=BODYPARTS, skeleton=SKELETON),
                      "[display] dotsize: -1 is not a positive number",
                      "'no-such-map' is not a matplotlib colormap",
                      "unknown key(s) colour",
                      "nose: 'reddish' is not a color",
                      "paw: not a bodypart of this project",
                      "head: unknown key(s) shape",
                      "nose-tail: not an edge of this project's skeleton",
                      "between must name two markers")
    assert message.startswith("project.toml [display]:")


def test_project_toml_keeps_and_checks_the_table():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=BODYPARTS, skeleton=SKELETON)
        assert "[display]" not in proj.layout.project_toml.read_text()   # no empty table written
        proj.config.display = dict(TABLE)
        proj.save_config()
        again = ws.Project.open(proj.root)
        assert again.config.style.colors["nose"] == "#ff0000"
        assert again.config.display["edges"][0]["width"] == 3

        text = proj.layout.project_toml.read_text().replace('"#00ff00"', '"not-a-color"')
        proj.layout.project_toml.write_text(text)
        _raises(lambda: ws.Project.open(proj.root), "tail: 'not-a-color' is not a color")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["info", str(proj.root)])
        assert code == 2 and "project.toml [display]:" in buf.getvalue()   # a message, not a traceback


def test_the_annotator_config_carries_the_style():
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=BODYPARTS, skeleton=SKELETON)
        plain = ann.synthesize_config(proj, scorer="me")
        assert plain["colormap"] == ann.DEFAULT_COLORMAP and "bodypart_colors" not in plain
        proj.config.display = dict(TABLE)
        cfg = ann.synthesize_config(proj, scorer="me")
        assert cfg["dotsize"] == 6 and cfg["colormap"] == "tab10"
        assert cfg["bodypart_colors"] == {"nose": "#ff0000", "tail": "#00ff00"}
        assert cfg["bodypart_sizes"] == {"nose": 10.0}
        assert annotator_config(parse_display({})) == {}


def test_video_style_matches_the_annotator():
    style = parse_display(TABLE)
    kw = video_style(style, BODYPARTS)
    # tab10 is a palette: its listed colors in order; explicit colors win
    assert kw["marker_colors"] == {"nose": (0, 0, 255), "head": (14, 127, 255), "tail": (0, 255, 0)}
    assert kw["marker_radii"] == {"nose": 5} and kw["dotsize"] == 3       # diameters -> radii
    assert kw["line_color"] == (255, 255, 255) and kw["line_thickness"] == 1
    assert kw["edge_colors"] == {frozenset({"nose", "head"}): (255, 0, 0)}
    # viridis is a gradient: sampled across its range, not its first few entries
    grad = video_style(parse_display({"colormap": "viridis"}), BODYPARTS)["marker_colors"]
    first, last = np.array(grad["nose"]), np.array(grad["tail"])
    assert np.abs(first - last).max() > 100


def test_labeled_video_draws_the_style():
    import cv2

    from deeplabcut.workspace.label_video import render_labeled_from_parquet

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        video = d / "clip.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 80))
        for _ in range(3):
            writer.write(np.zeros((80, 120, 3), np.uint8))
        writer.release()
        pose = pd.DataFrame([(f, "single", bp, x, 40.0, 0.99) for f in range(3)
                             for bp, x in (("nose", 20.0), ("head", 60.0), ("tail", 100.0))],
                            columns=["frame", "individual", "bodypart", "x", "y", "likelihood"])
        parquet = d / "clip.fdlc.parquet"
        pose.to_parquet(parquet)
        table = {"line_width": 3, "bodyparts": {"nose": {"color": "red", "size": 16}},
                 "edges": [{"between": ["head", "tail"], "color": "blue"}]}
        out = render_labeled_from_parquet(video, parquet, d / "out.mp4", bodyparts=BODYPARTS,
                                          skeleton=SKELETON, display=table, progress=False)
        cap = cv2.VideoCapture(str(out))
        ok, frame = cap.read()
        cap.release()
        assert ok
        b, g, r = (int(v) for v in frame[40, 20 + 6])                 # inside the 8 px radius nose dot
        assert r > 180 and g < 80 and b < 80, (b, g, r)
        b, g, r = (int(v) for v in frame[40, 80])                     # middle of head-tail: blue line
        assert b > 150 and r < 100, (b, g, r)
        b, g, r = (int(v) for v in frame[40, 40])                     # middle of nose-head: white line
        assert min(b, g, r) > 150, (b, g, r)


def _run() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for c in checks:
        c()
    print(f"display: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
