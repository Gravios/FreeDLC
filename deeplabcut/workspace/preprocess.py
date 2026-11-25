#
# FreeDLC workspace layer
#
"""Prepare a folder tree of videos for a model: ``fdlc preprocess``.

Every video under a folder is downscaled to the size a model was trained on and
written as plain H.264 MP4 beside it, named after the configuration used:
``K3_cam.h264`` with ``triplet`` gives ``K3_cam-triplet.mp4``. Every frame is kept.

*How* a video is encoded is a named **configuration**, a small TOML file::

    description = "Raspberry Pi camera streams, 640x360 like the training videos"
    size = "640x360"      # output width x height, even numbers
    fps = 25              # frame rate raw streams are numbered at
    crf = 14              # x264 quality, lower is better
    preset = "slow"       # x264 speed/size trade-off
    scale_flags = "bicubic"
    raw_only = false      # true: leave videos that are already in a container alone
    channels = "triplet"  # "color": the picture as it is; "triplet": see below
    stabilize = true      # align every frame to the static scene (autofocus, see below)
    frame_log = true      # write <base>-<config>.csv: sharpness (and alignment) per frame

Configurations are looked up, first match wins, in ``./.fdlc/preprocess/`` (one
kept with a project or a data folder), the user's
``~/.config/fdlc/preprocess/`` (where ``fdlc preprocess --save-config`` writes),
and the bundled ``preprocess_configs/`` beside this module. A name ending in
``.toml`` or holding a path separator is read as a file.

Raw H.264/H.265 streams (``.h264 .264 .h265 .hevc``), as Raspberry Pi cameras
write them, carry no usable timestamps: ffmpeg may put every frame at time 0 and
keep only a few, and OpenCV cannot tell their length. Their frames are numbered at
``fps`` instead (``-fflags +genpts -r FPS`` and ``setpts=N/(FPS*TB)``). Videos
already in a container keep their own timing and are only scaled. Either way
``-fps_mode passthrough`` keeps frame N of the output frame N of the source.

With ``channels = "triplet"`` each output frame carries three grayscale frames
in its colour channels: red the previous frame, green the frame itself, blue the
next one (the first and last frames stand in for the missing neighbours). The
model then sees motion as well as shape. Frames are decoded to grayscale at the
output size by one ffmpeg, stacked here, and encoded by another with
``libx264rgb``: the three channels are stored as they are, at full resolution,
where the usual YUV 4:2:0 would mix them and halve two of them. Frame N of the
output is still frame N of the source.

With ``stabilize = true`` every frame is aligned to a reference frame of the same
video before it is stacked or written. Cameras that refocus change the image
scale slightly as the lens moves ("focus breathing") and can shift it, which
moves every marker, and in a triplet video shows as colour fringes on still
objects. :class:`Stabilizer` matches ORB features of each frame to those of the
reference and fits a similarity transform (shift, rotation, uniform scale) with
RANSAC, so the moving animal, a minority of the features, does not pull it; the
frame is then warped onto the reference, edges repeated. A fit that moves no
corner of the frame by more than half a pixel is fitting noise, and the frame is
left exactly as it is (no jitter, no interpolation). The reference is the
sharpest of the first :data:`REFERENCE_FRAMES` frames. Alignment works at the
output size. A frame whose fit has too few matching features keeps the previous
frame's transform.

With ``frame_log = true`` a table ``<base>-<config>.csv`` is written beside the
output: per frame its sharpness (variance of the Laplacian of the grayscale
frame -- dips mark refocusing and motion blur) and, when stabilizing, the
transform applied (``scale``, ``rotation`` in degrees, ``dx``/``dy`` in output
pixels, ``inliers``, ``held`` when the previous transform was kept).

Plain colour without either setting is a single ffmpeg; anything else decodes
frames into Python (one ffmpeg), processes them, and encodes them (another).

:func:`plan_jobs`, :func:`ffmpeg_command`, :func:`pipe_commands` and
:func:`triplet_frames` are pure, so what would be done is testable without ffmpeg;
:func:`run_job` runs one encode.
"""
from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .manifest import read_manifest, write_manifest

__all__ = [
    "PreprocessConfig",
    "Job",
    "BUNDLED_DIR",
    "RAW_SUFFIXES",
    "CONTAINER_SUFFIXES",
    "config_dirs",
    "available_configs",
    "load_config",
    "parse_config",
    "save_config",
    "plan_jobs",
    "output_path",
    "ffmpeg_command",
    "pipe_commands",
    "Stabilizer",
    "sharpness",
    "frame_log_path",
    "triplet_frames",
    "run_job",
    "count_frames",
    "describe",
]

BUNDLED_DIR = Path(__file__).parent / "preprocess_configs"
"""The configurations that come with FreeDLC, one ``<name>.toml`` each."""

RAW_SUFFIXES = (".h264", ".264", ".h265", ".hevc")
CONTAINER_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
#: x264's presets, fastest first
PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
#: ffmpeg's scaler algorithms worth choosing between
SCALE_FLAGS = ("bicubic", "bilinear", "area", "lanczos", "neighbor", "spline")
_KEYS = {"description", "size", "fps", "crf", "preset", "scale_flags", "raw_only", "channels", "stabilize",
         "frame_log"}
#: what the colour channels of an output frame hold
CHANNELS = ("color", "triplet")
_OUTPUT = re.compile(r"_\d+x\d+(\.part)?\.mp4$", re.IGNORECASE)
#: the reference for stabilizing is the sharpest of this many first frames
REFERENCE_FRAMES = 50


@dataclass(frozen=True)
class PreprocessConfig:
    """How videos are encoded; see the module docstring for the keys."""

    name: str
    width: int = 640
    height: int = 360
    fps: float = 25.0
    crf: int = 14
    preset: str = "slow"
    scale_flags: str = "bicubic"
    raw_only: bool = False
    channels: str = "color"
    stabilize: bool = False
    frame_log: bool = False
    description: str = ""

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        """The TOML table this configuration is saved as."""
        fps = int(self.fps) if float(self.fps).is_integer() else self.fps
        return {"description": self.description, "size": self.size, "fps": fps, "crf": self.crf,
                "preset": self.preset, "scale_flags": self.scale_flags, "raw_only": self.raw_only,
                "channels": self.channels, "stabilize": self.stabilize, "frame_log": self.frame_log}

    @property
    def piped(self) -> bool:
        """Whether frames pass through Python (anything but a plain scale)."""
        return self.channels == "triplet" or self.stabilize or self.frame_log

    def with_changes(self, **changes) -> PreprocessConfig:
        """A copy with ``changes`` (``None`` values ignored), checked like a file is."""
        given = {k: v for k, v in changes.items() if v is not None}
        if not given:
            return self
        table = self.to_dict()
        table.update(given)
        return parse_config(table, name=self.name)


def parse_config(table: dict[str, Any], *, name: str) -> PreprocessConfig:
    """Check a configuration table and return it.

    Raises:
        ValueError: listing every problem found.
    """
    problems: list[str] = []
    extra = sorted(set(table) - _KEYS)
    if extra:
        problems.append(f"unknown key(s) {', '.join(extra)} (allowed: {', '.join(sorted(_KEYS))})")
    base = PreprocessConfig(name=name)
    width, height = base.width, base.height
    size = table.get("size", base.size)
    m = re.fullmatch(r"(\d+)x(\d+)", str(size))
    if not m:
        problems.append(f"size: {size!r} must look like 640x360")
    else:
        width, height = int(m[1]), int(m[2])
        if width % 2 or height % 2 or not width or not height:
            problems.append(f"size: {size} needs even, non-zero width and height (H.264 halves the colour planes)")
    fps = table.get("fps", base.fps)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        problems.append(f"fps: {fps!r} is not a positive number")
    crf = table.get("crf", base.crf)
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        problems.append(f"crf: {crf!r} is not a whole number from 0 to 51")
    preset = table.get("preset", base.preset)
    if preset not in PRESETS:
        problems.append(f"preset: {preset!r} is not one of {', '.join(PRESETS)}")
    flags = table.get("scale_flags", base.scale_flags)
    if flags not in SCALE_FLAGS:
        problems.append(f"scale_flags: {flags!r} is not one of {', '.join(SCALE_FLAGS)}")
    switches = {}
    for key in ("raw_only", "stabilize", "frame_log"):
        switches[key] = table.get(key, getattr(base, key))
        if not isinstance(switches[key], bool):
            problems.append(f"{key}: {switches[key]!r} is not true or false")
    channels = table.get("channels", base.channels)
    if channels not in CHANNELS:
        problems.append(f"channels: {channels!r} is not one of {', '.join(CHANNELS)}")
    if problems:
        raise ValueError(f"preprocess config {name!r}:\n  " + "\n  ".join(problems))
    return PreprocessConfig(name=name, width=width, height=height, fps=float(fps), crf=crf, preset=preset,
                            scale_flags=flags, channels=channels, **switches,
                            description=str(table.get("description", "")))


def config_dirs(cwd: str | Path | None = None) -> list[tuple[str, Path]]:
    """Where configurations are looked up, first match wins: ``(label, folder)``."""
    home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return [("local", Path(cwd or Path.cwd()) / ".fdlc" / "preprocess"),
            ("user", home / "fdlc" / "preprocess"),
            ("bundled", BUNDLED_DIR)]


def available_configs(cwd: str | Path | None = None) -> list[tuple[str, str, Path]]:
    """Every configuration that can be named: ``(name, label, file)``, sorted; a name hides later ones."""
    seen: dict[str, tuple[str, str, Path]] = {}
    for label, folder in config_dirs(cwd):
        if folder.is_dir():
            for path in sorted(folder.glob("*.toml")):
                seen.setdefault(path.stem, (path.stem, label, path))
    return sorted(seen.values())


def load_config(ref: str, cwd: str | Path | None = None) -> PreprocessConfig:
    """The configuration named ``ref``, or read from ``ref`` when it is a ``.toml`` path.

    Raises:
        FileNotFoundError: if there is no such configuration.
        ValueError: if it is malformed.
    """
    if ref.endswith(".toml") or os.sep in ref or "/" in ref:
        path = Path(ref)
        if not path.is_file():
            raise FileNotFoundError(f"no preprocess config file {path}")
        return parse_config(read_manifest(path), name=path.stem)
    for _, folder in config_dirs(cwd):
        path = folder / f"{ref}.toml"
        if path.is_file():
            return parse_config(read_manifest(path), name=ref)
    names = ", ".join(name for name, _, _ in available_configs(cwd)) or "(none)"
    raise FileNotFoundError(f"no preprocess config named {ref!r}; available: {names}")


def save_config(config: PreprocessConfig, name: str, folder: str | Path | None = None) -> Path:
    """Save ``config`` as ``<folder>/<name>.toml`` (default: the user's folder); return the file.

    Raises:
        ValueError: if ``name`` cannot be a file name.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name.endswith(".toml"):
        raise ValueError(f"config name {name!r}: use letters, digits, '-', '_' and '.', without .toml")
    folder = Path(folder) if folder is not None else config_dirs()[1][1]
    return write_manifest(folder / f"{name}.toml", replace(config, name=name).to_dict())


@dataclass(frozen=True)
class Job:
    """One video to encode."""

    source: Path
    output: Path
    raw: bool

    @property
    def done(self) -> bool:
        return self.output.is_file()


def output_path(source: str | Path, config_name: str) -> Path:
    """Where ``source`` encoded with configuration ``config_name`` goes: ``<base>-<config_name>.mp4`` beside it."""
    source = Path(source)
    return source.with_name(f"{source.stem}-{config_name}.mp4")


def _is_output(name: str, config_names) -> bool:
    """Whether a file is something this command (or an earlier version of it) wrote."""
    lower = name.lower()
    stem = lower.rsplit(".", 1)[0]
    if stem.endswith(".part") or _OUTPUT.search(lower):
        return True
    return any(stem.endswith(f"-{n.lower()}") for n in config_names)


def plan_jobs(root: str | Path, config: PreprocessConfig, *, other_configs=()) -> list[Job]:
    """The videos under ``root`` to encode with ``config``, sorted by path.

    Each output goes beside its video as ``<base>-<config name>.mp4``. Outputs of
    ``config`` and of the configurations named in ``other_configs``, unfinished
    ``.part`` files, earlier ``*_<W>x<H>.mp4`` outputs and labeled videos
    (``*.fdlc.*``) are never inputs.
    """
    root = Path(root)
    names = {config.name, *other_configs}
    jobs = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if _is_output(path.name, names) or ".fdlc." in path.name.lower():
            continue
        suffix = path.suffix.lower()
        raw = suffix in RAW_SUFFIXES
        if not raw and (suffix not in CONTAINER_SUFFIXES or config.raw_only):
            continue
        jobs.append(Job(source=path, output=output_path(path, config.name), raw=raw))
    return jobs


def _input_args(job: Job, config: PreprocessConfig) -> tuple[list[str], str]:
    """ffmpeg options before ``-i`` and the start of the filter chain: raw streams get their frames numbered."""
    if job.raw:
        fps = f"{config.fps:g}"
        return ["-fflags", "+genpts", "-r", fps], f"setpts=N/({fps}*TB),"
    return [], ""


def ffmpeg_command(job: Job, config: PreprocessConfig, target: str | Path | None = None, *,
                   stats: bool = True) -> list[str]:
    """The ffmpeg command for ``job`` with ``channels = "color"``, writing to ``target`` (default: its output)."""
    before_input, retime = _input_args(job, config)
    vf = f"{retime}scale={config.width}:{config.height}:flags={config.scale_flags}"
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats" if stats else "-nostats", "-y",
            *before_input, "-i", str(job.source),
            "-map", "0:v:0", "-an", "-vf", vf, "-fps_mode", "passthrough",
            "-c:v", "libx264", "-crf", str(config.crf), "-preset", config.preset, "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target or job.output)]


def pipe_commands(job: Job, config: PreprocessConfig, target: str | Path, rate: str, *,
                  stats: bool = True) -> tuple[list[str], list[str]]:
    """The two ffmpeg commands of an encode whose frames pass through Python: ``(decode, encode)``.

    The first writes frames at the output size to its stdout -- grayscale for
    ``channels = "triplet"``, BGR otherwise -- and the second reads frames of that
    size from its stdin at ``rate`` frames per second: RGB for a triplet, stored
    with ``libx264rgb``; BGR otherwise, encoded as usual.
    """
    before_input, retime = _input_args(job, config)
    triplet = config.channels == "triplet"
    vf = f"{retime}scale={config.width}:{config.height}:flags={config.scale_flags}"
    decode = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", *before_input, "-i", str(job.source),
              "-map", "0:v:0", "-an", "-vf", vf, "-fps_mode", "passthrough",
              "-f", "rawvideo", "-pix_fmt", "gray" if triplet else "bgr24", "pipe:1"]
    if triplet:
        codec = ["-c:v", "libx264rgb", "-pix_fmt", "rgb24"]
    else:
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    codec += ["-crf", str(config.crf), "-preset", config.preset]
    encode = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats" if stats else "-nostats", "-y",
              "-f", "rawvideo", "-pix_fmt", "rgb24" if triplet else "bgr24", "-s", config.size, "-r", rate,
              "-i", "pipe:0", *codec, "-movflags", "+faststart", str(target)]
    return decode, encode


def triplet_frames(frames):
    """Stack each frame with its neighbours: ``H x W x 3`` with (previous, frame, next).

    ``frames`` yields ``H x W`` grayscale arrays; the first frame is its own
    previous and the last its own next, so as many frames come out as go in.
    """
    import numpy as np

    it = iter(frames)
    try:
        cur = next(it)
    except StopIteration:
        return
    prev = cur
    for nxt in it:
        yield np.dstack((prev, cur, nxt))
        prev, cur = cur, nxt
    yield np.dstack((prev, cur, cur))


def _source_rate(path: Path) -> str | None:
    """A container video's average frame rate as ffprobe gives it (``30000/1001``), or None."""
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
                            capture_output=True, text=True, check=False)
    rate = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    num, _, den = rate.partition("/")
    ok = num.replace(".", "", 1).isdigit() and float(num) > 0 and (not den or (den.isdigit() and int(den) > 0))
    return rate if ok else None


def sharpness(gray) -> float:
    """Variance of the Laplacian of a grayscale frame: higher is sharper."""
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class Stabilizer:
    """Aligns frames to a reference frame of the same scene (see the module docstring).

    :meth:`align` returns a frame warped onto the reference and the transform
    used; transforms map frame pixels to reference pixels.
    """

    #: fewer RANSAC inliers than this and the previous transform is kept
    MIN_INLIERS = 20
    #: a transform moving no corner of the frame further than this (pixels) is not applied
    DEADBAND_PX = 0.5

    def __init__(self, reference_gray, *, n_features: int = 1000, ransac_px: float = 2.0):
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(reference_gray, None)
        self._ransac_px = ransac_px
        self.height, self.width = reference_gray.shape[:2]
        self.identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.last = self.identity
        h, w = self.height, self.width
        self._corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=float)

    def estimate(self, gray):
        """``(transform, inliers, held)`` for a grayscale frame; ``held`` when the previous one was kept."""
        import numpy as np

        cv2 = self._cv2
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or self._ref_desc is None or len(kp) < self.MIN_INLIERS:
            return self.last, 0, True
        matches = self._matcher.match(desc, self._ref_desc)
        if len(matches) < self.MIN_INLIERS:
            return self.last, 0, True
        src = np.float32([kp[m.queryIdx].pt for m in matches])
        dst = np.float32([self._ref_kp[m.trainIdx].pt for m in matches])
        transform, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                      ransacReprojThreshold=self._ransac_px)
        inliers = int(mask.sum()) if mask is not None else 0
        if transform is None or inliers < self.MIN_INLIERS:
            return self.last, inliers, True
        moved = self._corners @ transform.T - self._corners[:, :2]
        if np.hypot(moved[:, 0], moved[:, 1]).max() <= self.DEADBAND_PX:
            transform = self.identity  # fitting noise: leave the frame exactly as it is
        self.last = transform
        return transform, inliers, False

    def align(self, frame, gray=None):
        """``(warped frame, transform, inliers, held)``; ``gray`` is the frame's grayscale, if at hand."""
        cv2 = self._cv2
        if gray is None:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform, inliers, held = self.estimate(gray)
        if transform is self.identity:
            return frame, transform, inliers, held
        warped = cv2.warpAffine(frame, transform, (self.width, self.height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        return warped, transform, inliers, held


def _transform_fields(transform) -> dict[str, float]:
    """Scale, rotation (degrees) and shift of a 2x3 similarity transform."""
    import math

    a, b = float(transform[0, 0]), float(transform[1, 0])
    return {"scale": math.hypot(a, b), "rotation": math.degrees(math.atan2(b, a)),
            "dx": float(transform[0, 2]), "dy": float(transform[1, 2])}


def frame_log_path(output: str | Path) -> Path:
    """The per-frame table written beside an output: ``<base>-<config>.csv``."""
    return Path(output).with_suffix(".csv")


def _processed_frames(frames, config: PreprocessConfig, log_rows: list | None):
    """Stabilize ``frames`` (BGR, or grayscale for a triplet) and record each one's row in ``log_rows``."""
    import cv2

    frames = iter(frames)
    stabilizer = None
    if config.stabilize:  # the sharpest of the first frames is the reference
        head = [f for _, f in zip(range(REFERENCE_FRAMES), frames, strict=False)]
        if not head:
            return
        grays = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in head]
        stabilizer = Stabilizer(max(grays, key=sharpness))
        frames = itertools.chain(head, frames)
    for n, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row = {"frame": n, "sharpness": round(sharpness(gray), 2)} if log_rows is not None else None
        if stabilizer is not None:
            frame, transform, inliers, held = stabilizer.align(frame, gray)
            if row is not None:
                row.update({k: round(v, 5) for k, v in _transform_fields(transform).items()})
                row.update(inliers=inliers, held=int(held))
        if row is not None:
            log_rows.append(row)
        yield frame


def _encode_piped(job: Job, config: PreprocessConfig, target: Path, *, stats: bool) -> tuple[int, list | None]:
    """Decode, process and encode through Python; return ``(exit code, frame log rows or None)``."""
    import numpy as np

    rate = f"{config.fps:g}" if job.raw else (_source_rate(job.source) or f"{config.fps:g}")
    decode_cmd, encode_cmd = pipe_commands(job, config, target, rate, stats=stats)
    triplet = config.channels == "triplet"
    shape = (config.height, config.width) if triplet else (config.height, config.width, 3)
    size = int(np.prod(shape))
    decoder = subprocess.Popen(decode_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    def decoded():
        while True:
            buf = decoder.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(shape)

    log_rows = [] if config.frame_log else None
    frames = _processed_frames(decoded(), config, log_rows)
    if triplet:
        frames = triplet_frames(frames)
    try:
        for frame in frames:
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pass  # the encoder stopped; its exit code says why
    finally:
        try:
            encoder.stdin.close()
        except BrokenPipeError:
            pass
        decoder.stdout.close()
    encoded, decoded_code = encoder.wait(), decoder.wait()
    return decoded_code or encoded, log_rows


def _write_frame_log(rows: list, path: Path) -> None:
    import csv

    part = path.with_name(path.stem + ".part.csv")
    with part.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["frame", "sharpness"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(part, path)


def count_frames(path: str | Path) -> int | None:
    """The frames ffprobe decodes from ``path`` (reads all of it), or None."""
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                            capture_output=True, text=True, check=False)
    text = result.stdout.strip().splitlines()
    return int(text[0]) if text and text[0].isdigit() else None


def run_job(job: Job, config: PreprocessConfig, *, check_frames: bool = True,
            stats: bool = True) -> tuple[bool, str]:
    """Encode ``job``; return ``(ok, message)``.

    The encode is written as ``.part.mp4`` and moved over the output when ffmpeg
    succeeds, so an interrupted run leaves nothing to be mistaken for a finished
    video, and an existing output is replaced only by a complete one. With
    ``check_frames`` the output's frame count is compared with the source's; an
    output that differs is kept and reported, not deleted. ``stats`` shows
    ffmpeg's progress line (off when several videos are encoded at once).

    Raises:
        FileNotFoundError: if ffmpeg or ffprobe is not installed.
    """
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"{tool} is not installed or not on PATH")
    job.output.parent.mkdir(parents=True, exist_ok=True)
    part = job.output.with_name(job.output.stem + ".part.mp4")  # <base>-<config>.part.mp4
    log_rows = None
    if config.piped:
        code, log_rows = _encode_piped(job, config, part, stats=stats)
    else:
        code = subprocess.run(ffmpeg_command(job, config, part, stats=stats), stdin=subprocess.DEVNULL,
                              check=False).returncode
    if code != 0:
        part.unlink(missing_ok=True)
        return False, f"ffmpeg failed (exit {code})"
    if log_rows is not None:
        _write_frame_log(log_rows, frame_log_path(job.output))
    if not check_frames:
        os.replace(part, job.output)
        return True, "done"
    n_in, n_out = count_frames(job.source), count_frames(part)
    os.replace(part, job.output)
    if n_in is not None and n_in == n_out:
        return True, f"{n_out} frames"
    written = n_out if n_out is not None else "?"
    source = n_in if n_in is not None else "?"
    return False, f"CHECK: {written} frames written, the source has {source}; the output is kept"


def describe(config: PreprocessConfig) -> str:
    """One line: what a configuration does."""
    what = "raw streams only" if config.raw_only else "raw streams and container videos"
    pixels = ("triplet: grayscale previous/frame/next in red/green/blue" if config.channels == "triplet"
              else "color")
    if config.stabilize:
        pixels += ", stabilized"
    if config.frame_log:
        pixels += ", frame log"
    return (f"{config.size}, {pixels}, crf {config.crf}, preset {config.preset}, {config.scale_flags} scaling, "
            f"raw streams at {config.fps:g} fps; {what}")
