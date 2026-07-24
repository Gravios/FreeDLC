"""Named skeleton configurations: the bundled library, plus export from a project.

A skeleton config is a standalone TOML file naming a rig's markers, its skeleton
graph (``[skeleton] nodes/edges``) and its kinematic tree (``[[pose.segments]]``
rooted at ``[pose.kinematics] root``). Table and key names mirror a mufasa
``project.toml`` exactly, so a config is a drop-in fragment for one and mufasa's
``kalman_pose_smoother_v2.layout_from_config`` consumes it unchanged.

Both halves are kept because they are not interchangeable. mufasa *can* derive a
tree from the skeleton graph alone, but only by taking a breadth-first spanning
tree, and a rig's graph is not a tree -- RodentH5B7T3 has 24 edges over 15 nodes.
The derivation therefore drops ten edges and picks a topology that need not match
the animal: for RodentH5B7T3 it roots at ``head_back`` and yields fifteen
single-marker segments with no rigid trunk. The edges drive rendering; the
segments drive kinematics; a config carries both.

:func:`validate_config` enforces the same constraints mufasa does at layout-build
time (``layout_from_segments`` and ``BodyLayout.__post_init__``), so a bad rig is
rejected here rather than several pipeline stages later.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .manifest import read_manifest, write_manifest

__all__ = [
    "SKELETON_DIR",
    "available_skeletons",
    "load_skeleton",
    "resolve_skeleton",
    "validate_config",
    "config_from_project",
    "write_config",
    "matching_library_config",
]

SKELETON_DIR = Path(__file__).parent / "skeletons"
"""Directory holding the bundled skeleton configs, one ``<name>.toml`` each."""


def available_skeletons(directory: str | Path | None = None) -> list[str]:
    """Return the names of the skeleton configs in ``directory`` (default: bundled)."""
    directory = Path(directory) if directory is not None else SKELETON_DIR
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.toml"))


def load_skeleton(name: str, directory: str | Path | None = None) -> dict[str, Any]:
    """Return the validated skeleton config named ``name``.

    Raises:
        FileNotFoundError: if no such config exists.
        ValueError: if the config is malformed (see :func:`validate_config`).
    """
    directory = Path(directory) if directory is not None else SKELETON_DIR
    path = directory / f"{name}.toml"
    if not path.is_file():
        known = available_skeletons(directory)
        raise FileNotFoundError(f"no skeleton config named {name!r}; available: {', '.join(known) or '(none)'}")
    config = read_manifest(path)
    validate_config(config)
    return config


def resolve_skeleton(ref: str, directory: str | Path | None = None) -> dict[str, Any]:
    """Return the validated skeleton config referred to by ``ref``.

    ``ref`` is normally the name of an installed config (``RodentH5B7T3``), looked
    up in ``directory`` -- by default the bundled :data:`SKELETON_DIR`, which lives
    beside this module, so an installed FreeDLC finds its own configs. A ``ref``
    ending in ``.toml`` or containing a path separator is instead read as a file,
    which is how a config exported elsewhere is used without installing it first.

    Raises:
        FileNotFoundError: if no such config exists.
        ValueError: if the config is malformed (see :func:`validate_config`).
    """
    looks_like_path = ref.endswith(".toml") or "/" in ref or "\\" in ref
    if not looks_like_path:
        return load_skeleton(ref, directory)
    path = Path(ref)
    if not path.is_file():
        raise FileNotFoundError(f"no skeleton config at {ref}")
    config = read_manifest(path)
    validate_config(config)
    return config


def _segments(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``[[pose.segments]]`` entries, or ``[]`` when the config has none."""
    pose = config.get("pose")
    segments = pose.get("segments") if isinstance(pose, dict) else None
    return list(segments) if isinstance(segments, list) else []


def validate_config(config: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``config`` is a well-formed skeleton config.

    Mirrors mufasa's own checks so a rig that passes here builds a ``BodyLayout``
    there: edges are pairs of declared markers; segment names are unique and
    non-empty; every segment lists markers; no marker is attached twice; parents
    exist; there is exactly one root; and the tree is acyclic and fully reachable.
    Segments are optional -- a config may carry only a graph -- but when present
    they must cover the marker set exactly, since a marker missing from the tree
    is silently unusable downstream.
    """
    body_parts = [str(b) for b in (config.get("body_parts") or [])]
    if not body_parts:
        raise ValueError("skeleton config declares no body_parts")
    if len(set(body_parts)) != len(body_parts):
        raise ValueError("skeleton config body_parts contains duplicates")
    declared = set(body_parts)

    skeleton = config.get("skeleton")
    skeleton = skeleton if isinstance(skeleton, dict) else {}
    nodes = [str(n) for n in (skeleton.get("nodes") or [])]
    unknown_nodes = sorted(set(nodes) - declared)
    if unknown_nodes:
        raise ValueError(f"[skeleton] nodes not in body_parts: {', '.join(unknown_nodes)}")
    for edge in skeleton.get("edges") or []:
        pair = [str(p) for p in edge]
        if len(pair) != 2:
            raise ValueError(f"[skeleton] edge {edge!r} must be a pair")
        undeclared = [p for p in pair if p not in declared]
        if undeclared:
            raise ValueError(f"[skeleton] edge {edge!r} names undeclared bodypart(s): {', '.join(undeclared)}")

    segments = _segments(config)
    if not segments:
        return

    names = [str(s.get("name") or "") for s in segments]
    if not all(names):
        raise ValueError("[[pose.segments]] entries must each have a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError(f"[[pose.segments]] names must be unique: {names}")

    attached: dict[str, str] = {}
    for segment in segments:
        markers = [str(m) for m in (segment.get("markers") or [])]
        if not markers:
            raise ValueError(f"segment {segment.get('name')!r} lists no markers")
        for marker in markers:
            if marker not in declared:
                raise ValueError(f"segment {segment['name']!r} names undeclared marker {marker!r}")
            if marker in attached:
                raise ValueError(
                    f"marker {marker!r} is attached to more than one segment: "
                    f"{attached[marker]!r} and {segment['name']!r}"
                )
            attached[marker] = str(segment["name"])

    missing = sorted(declared - set(attached))
    if missing:
        raise ValueError(f"body_parts not attached to any segment: {', '.join(missing)}")

    roots = [n for n, s in zip(names, segments, strict=True) if not s.get("parent")]
    if len(roots) != 1:
        raise ValueError(f"[[pose.segments]] needs exactly one segment with no parent; found {len(roots)}: {roots}")
    known = set(names)
    for segment in segments:
        parent = segment.get("parent")
        if parent and str(parent) not in known:
            raise ValueError(f"segment {segment['name']!r} has unknown parent {parent!r}")

    # Reachability doubles as the cycle check: a cycle is unreachable from the root.
    children: dict[str, list[str]] = {n: [] for n in names}
    for name, segment in zip(names, segments, strict=True):
        parent = segment.get("parent")
        if parent:
            children[str(parent)].append(name)
    reached, stack = set(), [roots[0]]
    while stack:
        node = stack.pop()
        if node in reached:
            continue
        reached.add(node)
        stack.extend(children[node])
    if len(reached) != len(names):
        raise ValueError(f"[[pose.segments]] has cyclic or unreachable segments: {sorted(set(names) - reached)}")

    declared_root = ((config.get("pose") or {}).get("kinematics") or {}).get("root")
    if declared_root and str(declared_root) != roots[0]:
        raise ValueError(
            f"[pose.kinematics] root is {declared_root!r} but the parentless segment is {roots[0]!r}"
        )


def config_from_project(
    project,
    *,
    name: str,
    description: str = "",
    segments_from: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a skeleton config from ``project``'s bodyparts and skeleton edges.

    A FreeDLC project stores a flat edge list and no kinematic tree, so the config
    carries ``[skeleton]`` only. Pass ``segments_from`` -- another config, usually a
    library entry over the same markers -- to copy its ``[[pose.segments]]`` and
    root across, which is what makes a project that was created from a library rig
    export back as a complete one. No tree is ever derived from the edges here:
    mufasa's spanning-tree fallback exists for that, and baking its guess into a
    file would make an inferred topology look authored.
    """
    config: dict[str, Any] = {
        "name": name,
        "description": description,
        "schema_version": 1,
        "body_parts": list(project.config.bodyparts),
    }
    if segments_from is not None:
        segments = _segments(segments_from)
        root = ((segments_from.get("pose") or {}).get("kinematics") or {}).get("root")
        if segments:
            pose: dict[str, Any] = {}
            if root:
                pose["kinematics"] = {"root": str(root)}
            pose["segments"] = [dict(s) for s in segments]
            config["pose"] = pose
    config["skeleton"] = {
        "nodes": list(project.config.bodyparts),
        "edges": [list(edge) for edge in project.config.skeleton],
    }
    validate_config(config)
    return config


def write_config(config: dict[str, Any], directory: str | Path, *, name: str | None = None) -> Path:
    """Validate ``config`` and write it to ``<directory>/<name>.toml``; return that path."""
    validate_config(config)
    name = name or str(config.get("name") or "")
    if not name:
        raise ValueError("skeleton config has no name and none was given")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return write_manifest(directory / f"{name}.toml", config)


def matching_library_config(bodyparts: Iterable[str], edges: Iterable[Iterable[str]]) -> dict[str, Any] | None:
    """Return the bundled config whose markers and edges match, or ``None``.

    Edges compare as an undirected set, so orientation and ordering do not matter.
    """
    want_parts = set(str(b) for b in bodyparts)
    want_edges = {frozenset((str(a), str(b))) for a, b in edges}
    for candidate in available_skeletons():
        config = load_skeleton(candidate)
        skeleton = config.get("skeleton") or {}
        have_edges = {frozenset((str(a), str(b))) for a, b in (skeleton.get("edges") or [])}
        if set(str(b) for b in config.get("body_parts") or []) == want_parts and have_edges == want_edges:
            return config
    return None
