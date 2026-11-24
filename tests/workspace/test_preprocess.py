#
# FreeDLC workspace layer -- preprocess tests
#
"""Tests for ``fdlc preprocess``: configurations, what is encoded, and the ffmpeg command.

The encode itself runs only where ffmpeg is installed. No torch.

Standalone: ``python tests/workspace/test_preprocess.py`` -> ``preprocess: N/N checks passed``.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from deeplabcut.workspace import cli
from deeplabcut.workspace import preprocess as pp


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


@contextlib.contextmanager
def _sandbox():
    """A temporary cwd and user config folder."""
    old_cwd, old_xdg = Path.cwd(), os.environ.get("XDG_CONFIG_HOME")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        os.environ["XDG_CONFIG_HOME"] = str(d / "config")
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(old_cwd)
            if old_xdg is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old_xdg


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as err:
        for needle in needles:
            assert needle in str(err), (needle, str(err))
        return
    raise AssertionError("expected ValueError")


def test_the_bundled_triplet_config():
    with _sandbox():
        config = pp.load_config("triplet")
        assert (config.size, config.fps, config.crf, config.preset) == ("640x360", 25.0, 14, "slow")
        assert not config.raw_only and config.scale_flags == "bicubic"
        assert pp.parse_config(config.to_dict(), name="triplet") == config          # saves as it loads
        assert "640x360, crf 14, preset slow" in pp.describe(config)


def test_a_bad_config_lists_every_problem():
    _raises(lambda: pp.parse_config({"size": "641x360", "fps": 0, "crf": 60, "preset": "quick",
                                     "scale_flags": "fancy", "raw_only": "yes", "colour": 1}, name="bad"),
            "preprocess config 'bad'", "unknown key(s) colour", "641x360 needs even", "fps: 0",
            "crf: 60", "preset: 'quick'", "scale_flags: 'fancy'", "raw_only: 'yes'")
    _raises(lambda: pp.parse_config({"size": "big"}, name="x"), "must look like 640x360")
    base = pp.parse_config({}, name="plain")
    assert base.with_changes(crf=20, size="320x180") == pp.parse_config({"crf": 20, "size": "320x180"}, name="plain")
    assert base.with_changes() is base
    _raises(lambda: base.with_changes(preset="nope"), "preset: 'nope'")


def test_configs_are_found_local_then_user_then_bundled():
    with _sandbox() as d:
        user = d / "config" / "fdlc" / "preprocess"
        saved = pp.save_config(pp.load_config("triplet").with_changes(fps=30), "cam30")
        assert saved == user / "cam30.toml" and pp.load_config("cam30").fps == 30.0
        pp.save_config(pp.load_config("triplet").with_changes(crf=20), "triplet", folder=d / ".fdlc" / "preprocess")
        assert pp.load_config("triplet").crf == 20                                  # the local one wins
        listed = {name: label for name, label, _ in pp.available_configs()}
        assert listed == {"cam30": "user", "triplet": "local"}
        assert pp.load_config(str(saved)).name == "cam30"                           # a path
        try:
            pp.load_config("nope")
        except FileNotFoundError as err:
            assert "available: cam30, triplet" in str(err)
        else:
            raise AssertionError("expected FileNotFoundError")
        _raises(lambda: pp.save_config(pp.load_config("cam30"), "a/b"), "use letters, digits")


def test_what_is_encoded_and_where():
    with _sandbox() as d:
        root = d / "data"
        for rel in ("cage A/K3_cam.h264", "cage A/K3_cam_640x360.mp4", "b/session.MP4", "b/session-triplet.mp4",
                    "b/old.fdlc.mp4", "b/x_640x360.part.mp4", "b/notes.txt", "b/deep/c.avi"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"v")
        config = pp.load_config("triplet")
        jobs = pp.plan_jobs(root, config)
        assert [(j.source.relative_to(root).as_posix(), j.output.name, j.raw, j.done) for j in jobs] == [
            ("b/deep/c.avi", "c_640x360.mp4", False, False),
            ("b/session.MP4", "session_640x360.mp4", False, False),
            ("cage A/K3_cam.h264", "K3_cam_640x360.mp4", True, True),               # already there
        ]
        raw = pp.plan_jobs(root, config.with_changes(raw_only=True), out=d / "out")
        assert [j.output for j in raw] == [d / "out" / "cage A" / "K3_cam_640x360.mp4"]

        cmd = pp.ffmpeg_command(raw[0], config)
        assert cmd[cmd.index("-i") - 4:cmd.index("-i")] == ["-fflags", "+genpts", "-r", "25"]
        assert cmd[cmd.index("-vf") + 1] == "setpts=N/(25*TB),scale=640:360:flags=bicubic"
        assert ["-fps_mode", "passthrough"] == cmd[cmd.index("-fps_mode"):cmd.index("-fps_mode") + 2]
        assert cmd[cmd.index("-crf") + 1] == "14" and cmd[cmd.index("-preset") + 1] == "slow"
        container = pp.ffmpeg_command(jobs[1], config)
        assert "-fflags" not in container and container[container.index("-vf") + 1].startswith("scale=")


def test_the_cli_lists_saves_and_plans():
    with _sandbox() as d:
        (d / "data").mkdir()
        (d / "data" / "a.h264").write_bytes(b"v")
        code, out = _run(["preprocess", "data"])
        assert code == 2 and "--config NAME (available: triplet" in out
        code, out = _run(["preprocess", "--config", "triplet", "-n", "data"])
        assert code == 0 and "would encode (raw): data/a.h264 -> data/a_640x360.mp4" in out
        assert "dry run: 1 to encode, 0 already done" in out
        code, out = _run(["preprocess", "--config", "triplet", "--fps", "30", "--crf", "16", "--save-config", "cam30"])
        assert code == 0 and "saved config cam30" in out
        code, out = _run(["list", "preprocess", "cam30"])
        assert code == 0 and "crf 16" in out and "30 fps" in out
        code, out = _run(["list", "preprocess"])
        assert code == 0 and "cam30" in out and "triplet" in out and "bundled" in out
        code, out = _run(["preprocess", "--config", "triplet", "--size", "641x360", "data"])
        assert code == 2 and "needs even" in out


def test_a_raw_stream_is_encoded_frame_for_frame():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("  (skipped: no ffmpeg)")
        return
    with _sandbox() as d:
        raw = d / "data" / "cam.h264"
        raw.parent.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
                        "-c:v", "libx264", "-f", "h264", str(raw)], check=True)
        config = pp.load_config("triplet").with_changes(size="160x120", preset="ultrafast")
        (job,) = pp.plan_jobs(d / "data", config)
        ok, message = pp.run_job(job, config)
        assert ok and message == "50 frames", message
        assert job.output.is_file() and not list(d.rglob("*.part.mp4"))
        code, out = _run(["preprocess", "--config", "triplet", "--size", "160x120", "data"])
        assert code == 0 and "skip (exists)" in out and "done: 0 encoded, 1 skipped" in out, out


def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"preprocess: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
