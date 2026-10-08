#
# FreeDLC workspace layer -- marker and skeleton display style
#
"""How markers and skeleton lines are drawn: the ``[display]`` table of ``project.toml``.

::

    [display]
    dotsize = 6            # markers without a size of their own
    colormap = "tab20"     # colors for markers without a color of their own
    line_color = "white"   # skeleton lines without a color of their own
    line_width = 1

    [display.bodyparts]
    head_nose = { color = "red", size = 8 }
    back_T4 = { color = "#00c0ff" }

    [[display.edges]]
    between = ["head_nose", "head_mid"]
    color = "orange"
    width = 2

Every key is optional. Colors are names or hex codes (anything matplotlib reads).
Sizes are dot diameters in pixels of the image they are drawn on -- the frame in
napari, the video in a labeled video. Marker colors and sizes are used by the
annotator (``fdlc annotate``) and by labeled videos; skeleton lines are drawn only
in labeled videos.

:func:`parse_display` checks a table once, when the project is opened, so a typo
in a marker name or a color stops every command with one message instead of
being silently ignored. matplotlib is imported lazily.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DisplayStyle", "parse_display", "annotator_config", "video_style"]

_TOP_KEYS = {"dotsize", "colormap", "line_color", "line_width", "bodyparts", "edges"}
_MARKER_KEYS = {"color", "size"}
_EDGE_KEYS = {"between", "color", "width"}


@dataclass(frozen=True)
class DisplayStyle:
    """A checked ``[display]`` table; colors are normalized to ``#rrggbb``."""

    dotsize: float | None = None
    colormap: str | None = None
    line_color: str | None = None
    line_width: float | None = None
    colors: dict[str, str] = field(default_factory=dict)            # bodypart -> #rrggbb
    sizes: dict[str, float] = field(default_factory=dict)           # bodypart -> diameter
    edge_colors: dict[frozenset, str] = field(default_factory=dict)  # {a, b} -> #rrggbb
    edge_widths: dict[frozenset, float] = field(default_factory=dict)


def _hex(value, where: str, problems: list[str]) -> str | None:
    from matplotlib.colors import to_hex

    try:
        return to_hex(value)
    except (ValueError, TypeError):
        problems.append(f"{where}: {value!r} is not a color (use a name like 'red' or a hex code like '#ff0000')")
        return None


def _positive(value, where: str, problems: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        problems.append(f"{where}: {value!r} is not a positive number")
        return None
    return float(value)


def _unknown(keys: Iterable[str], allowed: set[str], where: str, problems: list[str]) -> None:
    extra = sorted(set(keys) - allowed)
    if extra:
        problems.append(f"{where}: unknown key(s) {', '.join(extra)} (allowed: {', '.join(sorted(allowed))})")


def parse_display(
    table: Mapping[str, Any] | None,
    *,
    bodyparts: Iterable[str] | None = None,
    skeleton: Iterable[Iterable[str]] | None = None,
) -> DisplayStyle:
    """Check a ``[display]`` table and return it as a :class:`DisplayStyle`.

    With ``bodyparts`` and ``skeleton`` given, every marker named must be one of
    the bodyparts and every edge one of the skeleton's (in either direction).

    Raises:
        ValueError: listing every problem found.
    """
    table = dict(table or {})
    problems: list[str] = []
    _unknown(table, _TOP_KEYS, "[display]", problems)
    known = set(bodyparts) if bodyparts is not None else None
    edges = {frozenset(e) for e in skeleton} if skeleton is not None else None

    dotsize = _positive(table["dotsize"], "[display] dotsize", problems) if "dotsize" in table else None
    width = _positive(table["line_width"], "[display] line_width", problems) if "line_width" in table else None
    line_color = _hex(table["line_color"], "[display] line_color", problems) if "line_color" in table else None
    colormap = table.get("colormap")
    if colormap is not None:
        import matplotlib

        if not isinstance(colormap, str) or colormap not in matplotlib.colormaps:
            problems.append(f"[display] colormap: {colormap!r} is not a matplotlib colormap "
                            "(e.g. 'tab20', 'Set3', 'viridis')")
            colormap = None

    colors: dict[str, str] = {}
    sizes: dict[str, float] = {}
    markers = table.get("bodyparts") or {}
    if not isinstance(markers, Mapping):
        problems.append("[display.bodyparts] must be a table of marker = { color = ..., size = ... }")
        markers = {}
    for name, spec in markers.items():
        where = f"[display.bodyparts] {name}"
        if known is not None and name not in known:
            problems.append(f"{where}: not a bodypart of this project")
            continue
        if not isinstance(spec, Mapping):
            problems.append(f"{where}: must be {{ color = ..., size = ... }}")
            continue
        _unknown(spec, _MARKER_KEYS, where, problems)
        if "color" in spec and (c := _hex(spec["color"], where, problems)) is not None:
            colors[name] = c
        if "size" in spec and (s := _positive(spec["size"], f"{where} size", problems)) is not None:
            sizes[name] = s

    edge_colors: dict[frozenset, str] = {}
    edge_widths: dict[frozenset, float] = {}
    entries = table.get("edges") or []
    if not isinstance(entries, list):
        problems.append("[display] edges must be a list of { between = [a, b], color = ..., width = ... }")
        entries = []
    for i, spec in enumerate(entries):
        where = f"[[display.edges]] #{i + 1}"
        if not isinstance(spec, Mapping):
            problems.append(f"{where}: must be {{ between = [a, b], color = ..., width = ... }}")
            continue
        _unknown(spec, _EDGE_KEYS, where, problems)
        pair = spec.get("between")
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(p, str) for p in pair):
            problems.append(f"{where}: between must name two markers, e.g. [\"head_nose\", \"head_mid\"]")
            continue
        key = frozenset(pair)
        where = f"[[display.edges]] {pair[0]}-{pair[1]}"
        if edges is not None and key not in edges:
            problems.append(f"{where}: not an edge of this project's skeleton")
            continue
        if "color" in spec and (c := _hex(spec["color"], where, problems)) is not None:
            edge_colors[key] = c
        if "width" in spec and (w := _positive(spec["width"], f"{where} width", problems)) is not None:
            edge_widths[key] = w

    if problems:
        raise ValueError("project.toml [display]:\n  " + "\n  ".join(problems))
    return DisplayStyle(dotsize=dotsize, colormap=colormap, line_color=line_color, line_width=width,
                        colors=colors, sizes=sizes, edge_colors=edge_colors, edge_widths=edge_widths)


def annotator_config(style: DisplayStyle) -> dict[str, Any]:
    """The keys napari-freedlc reads from the annotator's ``config.yaml`` for this style.

    ``dotsize`` and ``colormap`` only when set (the caller supplies defaults);
    ``bodypart_colors`` and ``bodypart_sizes`` only when some marker has its own.
    """
    out: dict[str, Any] = {}
    if style.dotsize is not None:
        out["dotsize"] = style.dotsize
    if style.colormap is not None:
        out["colormap"] = style.colormap
    if style.colors:
        out["bodypart_colors"] = dict(style.colors)
    if style.sizes:
        out["bodypart_sizes"] = dict(style.sizes)
    return out


def _bgr(color: str) -> tuple[int, int, int]:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return int(round(b * 255)), int(round(g * 255)), int(round(r * 255))


#: a colormap listing at most this many colors is a palette whose colors are used as
#: listed; a longer list (viridis has 256) is a gradient, sampled evenly instead
PALETTE_MAX = 32


def _colormap_colors(name: str, n: int) -> list[str]:
    """``n`` colors from a matplotlib colormap, as napari-freedlc picks them.

    A palette (tab20, Set3, ...) gives its colors in order, cycling; a gradient is
    sampled at the centers of ``n`` equal bins.
    """
    import matplotlib
    from matplotlib.colors import to_hex

    cmap = matplotlib.colormaps[name]
    listed = getattr(cmap, "colors", None)
    if listed is not None and 0 < len(listed) <= PALETTE_MAX:
        return [to_hex(listed[i % len(listed)]) for i in range(n)]
    return [to_hex(cmap((i + 0.5) / max(n, 1))) for i in range(n)]


def video_style(style: DisplayStyle, bodyparts: list[str]) -> dict[str, Any]:
    """Keyword arguments for :func:`~.label_video.render_labeled_video` from a style.

    Colors become BGR, sizes (diameters) become radii. Markers without a color of
    their own take one from ``colormap`` when it is set, and otherwise keep the
    renderer's default colors.
    """
    out: dict[str, Any] = {}
    colors: dict[str, str] = {}
    if style.colormap:
        colors.update(zip(bodyparts, _colormap_colors(style.colormap, len(bodyparts)), strict=True))
    colors.update(style.colors)
    if colors:
        out["marker_colors"] = {name: _bgr(c) for name, c in colors.items()}
    if style.sizes:
        out["marker_radii"] = {name: max(1, round(s / 2)) for name, s in style.sizes.items()}
    if style.dotsize is not None:
        out["dotsize"] = max(1, round(style.dotsize / 2))
    if style.line_color:
        out["line_color"] = _bgr(style.line_color)
    if style.line_width is not None:
        out["line_thickness"] = max(1, round(style.line_width))
    if style.edge_colors:
        out["edge_colors"] = {k: _bgr(c) for k, c in style.edge_colors.items()}
    if style.edge_widths:
        out["edge_widths"] = {k: max(1, round(w)) for k, w in style.edge_widths.items()}
    return out
