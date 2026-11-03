#
# FreeDLC workspace layer -- upgrade tests
#
"""Tests for ``fdlc upgrade``: videos from ``<kind>/<id>/video.*`` to ``<kind>/<id>.*``. No torch.

Standalone: ``python tests/workspace/test_upgrade.py`` -> ``upgrade: N/N checks passed``.
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path

from deeplabcut import workspace as ws
from deeplabcut.workspace import cli
from deeplabcut.workspace.frames import resolve_media
from deeplabcut.workspace.manifest import write_manifest
from deeplabcut.workspace.schema import VideoRecord


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _legacy_project(d: Path):
    """A project in the old layout: an absolute and a relative symlink, a copy, and a reference."""
    proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
    raw = d / "raw"
    raw.mkdir()
    for name in ("a.mp4", "b.mp4", "c.avi"):
        (raw / name).write_bytes(name.encode())
    videos = proj.layout.videos_dir

    def legacy(kind, vid, media=None, *, link="symlink"):
        folder = videos / kind / vid
        folder.mkdir(parents=True)
        if media is not None:
            media(folder)
        write_manifest(folder / "video.toml",
                       VideoRecord(video_id=vid, source_path=str(raw / f"{vid}.mp4"), link=link).to_dict())
        return folder

    legacy("original", "a", lambda f: (f / "video.mp4").symlink_to(raw / "a.mp4"))                  # absolute
    legacy("processed", "a", lambda f: (f / "video.mp4").symlink_to(os.path.relpath(raw / "b.mp4", f)))
    legacy("original", "c", lambda f: (f / "video.avi").write_bytes(b"own copy"), link="copy")
    legacy("original", "r", link="reference")                                                         # no media
    return proj.root, raw


def test_a_legacy_project_is_refused_until_upgraded():
    with tempfile.TemporaryDirectory() as d:
        root, raw = _legacy_project(Path(d))
        try:
            ws.Project.open(root)
        except ValueError as err:
            assert "old layout" in str(err) and f"fdlc upgrade {root}" in str(err)
        else:
            raise AssertionError("expected ValueError")
        code, out = _run(["videos", str(root)])
        assert code == 2 and "fdlc upgrade" in out                    # a message, not a traceback

        code, out = _run(["upgrade", str(root)])
        assert code == 0 and "sources/videos/original/a/video.mp4 -> sources/videos/original/a.mp4" in out
        assert "add --yes to move the files of 4 video(s)" in out
        assert (root / "sources/videos/original/a/video.mp4").is_symlink()            # a dry run

        code, out = _run(["upgrade", str(root), "--yes"])
        assert code == 0 and "upgraded" in out and "4 video(s), 7 file(s) moved" in out, out
        proj = ws.Project.open(root)
        orig, proc = proj.layout.videos_kind_dir("original"), proj.layout.videos_kind_dir("processed")
        assert sorted(p.name for p in orig.iterdir()) == ["a.mp4", "a.toml", "c.avi", "c.toml", "r.toml"]
        assert sorted(p.name for p in proc.iterdir()) == ["a.mp4", "a.toml"]
        assert proj.videos("original") == ["a", "c", "r"] and proj.videos("processed") == ["a"]
        assert resolve_media(proj, "a").read_bytes() == b"a.mp4"
        assert resolve_media(proj, "a", "processed").read_bytes() == b"b.mp4"     # the relative link still fits
        assert resolve_media(proj, "c").read_bytes() == b"own copy"
        assert sorted(p.name for p in raw.iterdir()) == ["a.mp4", "b.mp4", "c.avi"]  # sources untouched

        code, out = _run(["upgrade", str(root)])
        assert code == 0 and "is up to date" in out


def test_nothing_moves_while_something_is_in_the_way():
    with tempfile.TemporaryDirectory() as d:
        root, _ = _legacy_project(Path(d))
        (root / "sources/videos/original/c/notes.txt").write_text("mine")
        (root / "sources/videos/original/a.mp4").write_bytes(b"in the way")
        code, out = _run(["upgrade", str(root), "--yes"])
        assert code == 2 and "cannot upgrade; nothing was moved" in out
        assert "notes.txt: not a video file of this folder" in out
        assert "sources/videos/original/a.mp4 already exists" in out
        assert (root / "sources/videos/original/a/video.mp4").is_symlink()
        assert (root / "sources/videos/processed/a/video.mp4").is_symlink()


def test_new_videos_are_registered_flat_under_their_id():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        src = d / "Session 1.MP4"
        src.write_bytes(b"v")
        vid = proj.add_video(src)
        assert vid == "session-1"
        assert proj.layout.video_media(vid, ".MP4").is_symlink()
        assert proj.layout.video_toml(vid).is_file() and not proj.needs_upgrade()
        for bad in ("a.b", "x/y"):
            try:
                proj.add_video(src, video_id=bad)
            except ValueError as err:
                assert "must not be empty or contain" in str(err)
            else:
                raise AssertionError(f"expected ValueError for {bad!r}")


def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"upgrade: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
