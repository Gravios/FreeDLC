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
        assert not config.raw_only and config.scale_flags == "bicubic" and config.channels == "triplet"
        assert config.stabilize and config.frame_log and config.piped
        assert not pp.parse_config({}, name="plain").piped
        assert pp.parse_config(config.to_dict(), name="triplet") == config          # saves as it loads
        assert ("640x360, triplet: grayscale previous/frame/next in red/green/blue, stabilized, frame log, crf 14"
                in pp.describe(config))


def test_a_bad_config_lists_every_problem():
    _raises(lambda: pp.parse_config({"size": "641x360", "fps": 0, "crf": 60, "preset": "quick",
                                     "scale_flags": "fancy", "raw_only": "yes", "colour": 1, "channels": "rgb",
                                     "stabilize": 1},
                                    name="bad"),
            "preprocess config 'bad'", "unknown key(s) colour", "641x360 needs even", "fps: 0",
            "crf: 60", "preset: 'quick'", "scale_flags: 'fancy'", "raw_only: 'yes'", "channels: 'rgb'",
            "stabilize: 1 is not true or false")
    assert pp.parse_config({}, name="plain").channels == "color"
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
        for rel in ("cage A/K3_cam.h264", "cage A/K3_cam-triplet.mp4", "b/session.MP4", "b/session-cam30.mp4",
                    "b/old.fdlc.mp4", "b/x-triplet.part.mp4", "b/y_640x360.mp4", "b/notes.txt", "b/deep/c.avi"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"v")
        config = pp.load_config("triplet")
        jobs = pp.plan_jobs(root, config, other_configs=["cam30"])
        assert [(j.source.relative_to(root).as_posix(), j.output.relative_to(root).as_posix(), j.raw, j.done)
                for j in jobs] == [
            ("b/deep/c.avi", "b/deep/c-triplet.mp4", False, False),
            ("b/session.MP4", "b/session-triplet.mp4", False, False),
            ("cage A/K3_cam.h264", "cage A/K3_cam-triplet.mp4", True, True),          # already there
        ]
        # another configuration's outputs are inputs only when it is not known
        assert "b/session-cam30.mp4" in [j.source.relative_to(root).as_posix() for j in pp.plan_jobs(root, config)]
        raw = pp.plan_jobs(root, config.with_changes(raw_only=True))
        assert [j.output for j in raw] == [root / "cage A" / "K3_cam-triplet.mp4"]
        assert pp.output_path(root / "a.h264", "cam30") == root / "a-cam30.mp4"

        cmd = pp.ffmpeg_command(raw[0], config)
        assert cmd[cmd.index("-i") - 4:cmd.index("-i")] == ["-fflags", "+genpts", "-r", "25"]
        assert cmd[cmd.index("-vf") + 1] == "setpts=N/(25*TB),scale=640:360:flags=bicubic"
        assert ["-fps_mode", "passthrough"] == cmd[cmd.index("-fps_mode"):cmd.index("-fps_mode") + 2]
        assert cmd[cmd.index("-crf") + 1] == "14" and cmd[cmd.index("-preset") + 1] == "slow"
        container = pp.ffmpeg_command(jobs[1], config)
        assert "-fflags" not in container and container[container.index("-vf") + 1].startswith("scale=")


def test_triplet_frames_stack_each_frame_with_its_neighbours():
    import numpy as np

    frames = [np.full((2, 3), v, np.uint8) for v in (10, 20, 30)]
    out = list(pp.triplet_frames(frames))
    assert [tuple(int(c) for c in f[0, 0]) for f in out] == [(10, 10, 20), (10, 20, 30), (20, 30, 30)]
    assert out[0].shape == (2, 3, 3)
    assert [tuple(int(c) for c in f[0, 0]) for f in pp.triplet_frames(frames[:1])] == [(10, 10, 10)]
    assert list(pp.triplet_frames([])) == []

    job = pp.Job(source=Path("in/K3.h264"), output=Path("in/K3-triplet.mp4"), raw=True)
    decode, encode = pp.pipe_commands(job, pp.load_config("triplet"), "out.part.mp4", "25")
    assert decode[decode.index("-vf") + 1] == "setpts=N/(25*TB),scale=640:360:flags=bicubic"
    assert decode[-5:] == ["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    assert encode[encode.index("-c:v") + 1] == "libx264rgb" and encode[encode.index("-s") + 1] == "640x360"
    color = pp.load_config("triplet").with_changes(channels="color")
    decode, encode = pp.pipe_commands(job, color, "out.part.mp4", "25")
    assert decode[-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    assert encode[encode.index("-c:v") + 1] == "libx264" and encode[encode.index("-pix_fmt") + 1] == "bgr24"


def _scene(width=320, height=240, seed=0):
    """A textured still scene: smoothed noise, so ORB finds corners everywhere."""
    import cv2
    import numpy as np

    noise = np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 1.5)


def _zoom(image, scale, dx=0.0, dy=0.0):
    """``image`` scaled about its centre and shifted, as a refocusing lens does."""
    import cv2

    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test_the_stabilizer_undoes_a_zoom_and_shift_despite_a_moving_animal():
    import numpy as np

    ref = _scene()
    frame = _zoom(ref, 1.03, 4.0, -3.0)
    frame[90:150, 120:200] = 255                                       # the "animal", not in the reference
    stab = pp.Stabilizer(ref)
    warped, transform, inliers, held = stab.align(frame)
    fields = pp._transform_fields(transform)
    assert not held and inliers >= pp.Stabilizer.MIN_INLIERS
    assert abs(fields["scale"] - 1 / 1.03) < 0.003, fields
    inner = (slice(30, 80), slice(30, 290))                            # away from the edges and the animal
    assert np.abs(warped[inner].astype(int) - ref[inner].astype(int)).mean() < 4
    flat = np.full_like(ref, 128)                                      # nothing to match: keep the last transform
    _, again, _, held = stab.align(flat)
    assert held and np.allclose(again, transform)
    assert pp.sharpness(ref) > pp.sharpness(flat) == 0.0
    same, transform, _, held = pp.Stabilizer(ref).align(ref.copy())   # nothing to correct
    assert not held and transform is not None and np.array_equal(same, ref)
    assert np.allclose(transform, [[1, 0, 0], [0, 1, 0]])


def test_the_cli_lists_saves_and_plans():
    with _sandbox() as d:
        (d / "data").mkdir()
        (d / "data" / "a.h264").write_bytes(b"v")
        code, out = _run(["preprocess", "data"])
        assert code == 2 and "--config NAME (available: triplet" in out
        code, out = _run(["preprocess", "--config", "triplet", "-n", "data"])
        assert code == 0 and "would encode (raw): data/a.h264 -> data/a-triplet.mp4" in out
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
        code, out = _run(["preprocess", "--config", "triplet", "--size", "160x120", "--preset", "ultrafast", "data"])
        assert code == 0 and "50 frames" in out and "done: 1 encoded, 0 skipped" in out, out
        out_file = d / "data" / "cam-triplet.mp4"
        assert out_file.is_file() and not list(d.rglob("*.part.mp4"))
        size = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                               str(out_file)], capture_output=True, text=True, check=True).stdout.strip()
        assert size == "160,120"
        code, out = _run(["preprocess", "--config", "triplet", "data"])       # the output is not an input
        assert code == 0 and "skip (exists)" in out and "done: 0 encoded, 1 skipped" in out, out


def test_a_triplet_video_holds_the_previous_current_and_next_frames():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("  (skipped: no ffmpeg)")
        return
    import cv2
    import numpy as np

    with _sandbox() as d:
        src = d / "data" / "steps.mp4"
        src.parent.mkdir()
        levels = [40 + 20 * i for i in range(8)]                      # frame i is a flat grey of levels[i]
        writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for v in levels:
            writer.write(np.full((48, 64, 3), v, np.uint8))
        writer.release()
        code, out = _run(["preprocess", "--config", "triplet", "--size", "32x24", "--preset", "ultrafast",
                          "--crf", "0", "-j", "2", "data"])
        assert code == 0 and "8 frames" in out, out
        cap = cv2.VideoCapture(str(d / "data" / "steps-triplet.mp4"))
        got = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            b, g, r = (int(round(frame[..., c].mean())) for c in range(3))   # OpenCV gives BGR
            got.append((r, g, b))
        cap.release()
        want = [(levels[max(i - 1, 0)], levels[i], levels[min(i + 1, 7)]) for i in range(8)]
        assert len(got) == 8 and all(abs(x - y) <= 3 for a, b in zip(got, want, strict=True)
                                     for x, y in zip(a, b, strict=True)), (got, want)
        # --overwrite encodes again; without it the finished video is skipped
        code, out = _run(["preprocess", "--config", "triplet", "--size", "32x24", "--preset", "ultrafast", "data"])
        assert "skip (exists)" in out and "done: 0 encoded, 1 skipped" in out
        code, out = _run(["preprocess", "--config", "triplet", "--size", "32x24", "--preset", "ultrafast",
                          "--overwrite", "-n", "data"])
        assert "would overwrite (container)" in out and "dry run: 1 to encode" in out
        code, out = _run(["preprocess", "--config", "triplet", "--size", "32x24", "--preset", "ultrafast",
                          "--overwrite", "data"])
        assert code == 0 and "done: 1 encoded, 0 skipped" in out and not list(d.rglob("*.part.mp4"))


def test_a_refocusing_video_is_stabilized_and_logged():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("  (skipped: no ffmpeg)")
        return
    import csv

    import cv2
    import numpy as np

    with _sandbox() as d:
        src = d / "data" / "cam.mp4"
        src.parent.mkdir()
        ref = _scene(seed=1)
        writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
        for i in range(12):                                            # the lens refocuses at frame 6
            gray = ref if i < 6 else _zoom(ref, 1.04, 3.0, 2.0)
            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        writer.release()
        code, out = _run(["preprocess", "--config", "triplet", "--size", "320x240", "--preset", "ultrafast",
                          "--crf", "0", "data"])
        assert code == 0 and "12 frames" in out, out
        with open(d / "data" / "cam-triplet.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert [int(r["frame"]) for r in rows] == list(range(12))
        assert list(rows[0]) == ["frame", "sharpness", "scale", "rotation", "dx", "dy", "inliers", "held"]
        assert all(float(r["scale"]) == 1 and float(r["dx"]) == 0 for r in rows[:6])   # left as they are
        assert all(abs(float(r["scale"]) - 1 / 1.04) < 0.004 for r in rows[6:]), [r["scale"] for r in rows]
        cap = cv2.VideoCapture(str(d / "data" / "cam-triplet.mp4"))
        greens = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            greens.append(frame[..., 1].astype(int))                   # the frame itself (BGR order)
        cap.release()
        inner = (slice(30, 210), slice(30, 290))
        assert len(greens) == 12
        assert np.abs(greens[9][inner] - greens[2][inner]).mean() < 4  # after the refocus, aligned again
        unstab = _run(["preprocess", "--config", "triplet", "--size", "320x240", "--preset", "ultrafast",
                       "--crf", "0", "--no-stabilize", "--no-frame-log", "--overwrite", "data"])
        assert unstab[0] == 0
        cap = cv2.VideoCapture(str(d / "data" / "cam-triplet.mp4"))
        frames = [cap.read()[1][..., 1].astype(int) for _ in range(12)]
        cap.release()
        assert np.abs(frames[9][inner] - frames[2][inner]).mean() > 10  # without it the zoom stays


def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"preprocess: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
