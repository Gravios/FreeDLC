"""Skeleton-config library: bundled entries, validation, and export from a project.

Torch-free. Run directly (``python3 tests/workspace/test_skeleton_lib.py``) or via
``tests/workspace/run_all.sh``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import tempfile
from pathlib import Path

from deeplabcut.workspace import Project, cli
from deeplabcut.workspace import skeleton_lib as sl

RODENT = "RodentH5B7T3"


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _rodent() -> dict:
    return copy.deepcopy(sl.load_skeleton(RODENT))


# ------------------------------------------------------------ bundled library
def test_bundled_library_lists_rodent():
    assert RODENT in sl.available_skeletons()


def test_rodent_config_is_complete():
    cfg = _rodent()
    assert len(cfg["body_parts"]) == 15
    assert len(cfg["skeleton"]["nodes"]) == 15
    assert len(cfg["skeleton"]["edges"]) == 24
    segments = cfg["pose"]["segments"]
    assert [s["name"] for s in segments] == [
        "back", "back_rear", "neck", "head", "tail_1", "tail_2", "tail_3",
    ]
    assert cfg["pose"]["kinematics"]["root"] == "back"
    # every marker attached exactly once -- the constraint mufasa also enforces
    attached = [m for s in segments for m in s["markers"]]
    assert sorted(attached) == sorted(cfg["body_parts"])


def test_rodent_name_decodes_to_its_marker_counts():
    parts = _rodent()["body_parts"]
    assert sum(p.startswith("head_") for p in parts) == 5                      # H5
    assert sum(p.startswith(("back_", "hip_")) for p in parts) == 7            # B7
    assert sum(p.startswith("tail_") for p in parts) == 3                      # T3


def test_load_unknown_skeleton():
    try:
        sl.load_skeleton("NoSuchRig")
    except FileNotFoundError as err:
        assert RODENT in str(err)
    else:
        raise AssertionError("expected FileNotFoundError")


# ----------------------------------------------------------------- validation
def _rejects(mutate, needle):
    cfg = _rodent()
    mutate(cfg)
    try:
        sl.validate_config(cfg)
    except ValueError as err:
        assert needle in str(err), f"expected {needle!r} in {err!r}"
    else:
        raise AssertionError(f"expected ValueError mentioning {needle!r}")


def test_validate_rejects_two_roots():
    _rejects(lambda c: c["pose"]["segments"][2].pop("parent"), "exactly one segment with no parent")


def test_validate_rejects_unknown_parent():
    _rejects(lambda c: c["pose"]["segments"][1].__setitem__("parent", "nope"), "unknown parent")


def test_validate_rejects_marker_on_two_segments():
    _rejects(lambda c: c["pose"]["segments"][1]["markers"].append("head_back"), "more than one segment")


def test_validate_rejects_unattached_marker():
    _rejects(lambda c: c["pose"]["segments"][3]["markers"].remove("head_nose"), "not attached to any segment")


def test_validate_rejects_cycle():
    def mutate(cfg):
        segments = cfg["pose"]["segments"]
        segments[0]["parent"] = "tail_3"          # root gains a parent -> no root at all
        segments[1]["parent"] = "back"
    _rejects(mutate, "exactly one segment with no parent")


def test_validate_rejects_undeclared_edge_bodypart():
    _rejects(lambda c: c["skeleton"]["edges"].append(["head_nose", "wing"]), "undeclared bodypart")


def test_validate_rejects_root_disagreeing_with_tree():
    _rejects(lambda c: c["pose"]["kinematics"].__setitem__("root", "head"), "but the parentless segment is")


def test_validate_accepts_graph_without_segments():
    cfg = _rodent()
    cfg.pop("pose")
    sl.validate_config(cfg)               # segments are optional


# --------------------------------------------------------------------- export
def _project(root: Path, *, with_edges=True):
    cfg = _rodent()
    edges = [list(e) for e in cfg["skeleton"]["edges"]] if with_edges else []
    return Project.create(root, task="rodent", bodyparts=list(cfg["body_parts"]), skeleton=edges)


def test_export_attaches_matching_library_tree():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project(d / "proj")
        code, out = _run(["export-skeleton", str(d / "proj"), "--name", "MyRig", "--out", str(d / ".fdlc")])
        assert code == 0, out
        assert "segments: 7" in out
        written = sl.load_skeleton("MyRig", d / ".fdlc")
        assert written["pose"]["kinematics"]["root"] == "back"
        assert len(written["pose"]["segments"]) == 7


def test_export_without_segments():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project(d / "proj")
        code, out = _run(["export-skeleton", str(d / "proj"), "--out", str(d / ".fdlc"), "--no-segments"])
        assert code == 0 and "segments: 0" in out
        assert "no kinematic tree" in out
        assert "pose" not in sl.load_skeleton("rodent", d / ".fdlc")   # name defaults to the task


def test_export_defaults_to_dot_fdlc_directory():
    import os
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project(d / "proj")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            code, out = _run(["export-skeleton", str(d / "proj"), "--name", "R"])
        finally:
            os.chdir(cwd)
        assert code == 0 and (d / ".fdlc" / "skeletons" / "R.toml").is_file(), out


def test_export_rejects_project_without_skeleton():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project(d / "proj", with_edges=False)
        code, out = _run(["export-skeleton", str(d / "proj"), "--out", str(d / ".fdlc")])
        assert code == 2 and "no skeleton edges" in out


def test_export_segments_from_unknown_config():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _project(d / "proj")
        code, out = _run(["export-skeleton", str(d / "proj"), "--out", str(d / ".fdlc"),
                          "--segments-from", "NoSuchRig"])
        assert code == 2 and "no skeleton config named" in out


# ------------------------------------------------------------------ list/show
def test_list_skeletons_lists_and_shows():
    code, out = _run(["list", "skeletons"])
    assert code == 0 and RODENT in out

    code, out = _run(["list", "skeletons", RODENT])
    assert code == 0
    assert "markers: 15" in out and "edges: 24" in out and "segments: 7" in out
    assert "root: back" in out and "back_rear" in out

    code, out = _run(["list", "skeletons", "NoSuchRig"])
    assert code == 2 and "no skeleton config named" in out


# ---------------------------------------------------------- create --skeleton-config
def test_create_from_installed_skeleton_config():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ws"
        code, out = _run(["create", str(root), "--task", "rodent", "--skeleton-config", RODENT])
        assert code == 0, out
        assert "bodyparts: 15" in out and "skeleton: 24 edge(s)" in out
        assert f"skeleton config: {RODENT}" in out
        cfg = Project.open(root).config
        rig = _rodent()
        assert cfg.bodyparts == rig["body_parts"]
        assert [list(e) for e in cfg.skeleton] == [list(e) for e in rig["skeleton"]["edges"]]


def test_create_from_skeleton_config_path():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = sl.SKELETON_DIR / f"{RODENT}.toml"
        code, out = _run(["create", str(d / "ws"), "--task", "rodent", "--skeleton-config", str(src)])
        assert code == 0 and "bodyparts: 15" in out, out


def test_create_skeleton_config_round_trips_through_export():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert _run(["create", str(d / "ws"), "--task", "rodent", "--skeleton-config", RODENT])[0] == 0
        code, out = _run(["export-skeleton", str(d / "ws"), "--name", RODENT, "--out", str(d / ".fdlc")])
        assert code == 0 and "segments: 7" in out, out
        assert sl.load_skeleton(RODENT, d / ".fdlc")["pose"]["kinematics"]["root"] == "back"


def test_create_rejects_unknown_skeleton_config():
    with tempfile.TemporaryDirectory() as d:
        code, out = _run(["create", str(Path(d) / "ws"), "--task", "t", "--skeleton-config", "NoSuchRig"])
        assert code == 2 and "no skeleton config named" in out
        assert not (Path(d) / "ws" / "project.toml").exists()


def test_create_rejects_skeleton_config_with_bodyparts():
    with tempfile.TemporaryDirectory() as d:
        code, out = _run(["create", str(Path(d) / "ws"), "--task", "t",
                          "--skeleton-config", RODENT, "--bodyparts", "snout"])
        assert code == 2 and "drop --bodyparts" in out


def test_create_requires_bodyparts_or_skeleton_config():
    with tempfile.TemporaryDirectory() as d:
        code, out = _run(["create", str(Path(d) / "ws"), "--task", "t"])
        assert code == 2 and "one of --bodyparts or --skeleton-config is required" in out


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
    print(f"skeleton_lib: {passed}/{passed} checks passed")
