#
# FreeDLC workspace layer
#
"""Prepare a folder tree of videos for a model: ``fdlc preprocess``.

Every video under a folder is downscaled to the size a model was trained on and
written as plain H.264 MP4, ``<name>_<W>x<H>.mp4`` beside it (or into a copy of
the folder structure), with every frame kept.

*How* a video is encoded is a named **configuration**, a small TOML file::

    description = "Raspberry Pi camera streams, 640x360 like the training videos"
    size = "640x360"      # output width x height, even numbers
    fps = 25              # frame rate raw streams are numbered at
    crf = 14              # x264 quality, lower is better
    preset = "slow"       # x264 speed/size trade-off
    scale_flags = "bicubic"
    raw_only = false      # true: leave videos that are already in a container alone

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

:func:`plan_jobs` and :func:`ffmpeg_command` are pure, so what would be done is
testable without ffmpeg; :func:`run_job` runs one encode.
"""
from __future__ import annotations

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
    "ffmpeg_command",
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
_KEYS = {"description", "size", "fps", "crf", "preset", "scale_flags", "raw_only"}
_OUTPUT = re.compile(r"_\d+x\d+(\.part)?\.mp4$", re.IGNORECASE)


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
    description: str = ""

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        """The TOML table this configuration is saved as."""
        fps = int(self.fps) if float(self.fps).is_integer() else self.fps
        return {"description": self.description, "size": self.size, "fps": fps, "crf": self.crf,
                "preset": self.preset, "scale_flags": self.scale_flags, "raw_only": self.raw_only}

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
    raw_only = table.get("raw_only", base.raw_only)
    if not isinstance(raw_only, bool):
        problems.append(f"raw_only: {raw_only!r} is not true or false")
    if problems:
        raise ValueError(f"preprocess config {name!r}:\n  " + "\n  ".join(problems))
    return PreprocessConfig(name=name, width=width, height=height, fps=float(fps), crf=crf, preset=preset,
                            scale_flags=flags, raw_only=raw_only, description=str(table.get("description", "")))


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


def plan_jobs(root: str | Path, config: PreprocessConfig, *, out: str | Path | None = None) -> list[Job]:
    """The videos under ``root`` to encode with ``config``, sorted by path.

    Outputs go beside each video, or with ``out`` into ``out`` mirroring the
    subfolders of ``root``. Outputs (``*_<W>x<H>.mp4``), labeled videos
    (``*.fdlc.*``) and triplet videos (``*-triplet.*``) are never inputs.
    """
    root = Path(root)
    jobs = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        lower = path.name.lower()
        if _OUTPUT.search(lower) or ".fdlc." in lower or "-triplet." in lower:
            continue
        suffix = path.suffix.lower()
        raw = suffix in RAW_SUFFIXES
        if not raw and (suffix not in CONTAINER_SUFFIXES or config.raw_only):
            continue
        folder = (Path(out) / path.parent.relative_to(root)) if out is not None else path.parent
        jobs.append(Job(source=path, output=folder / f"{path.stem}_{config.size}.mp4", raw=raw))
    return jobs


def ffmpeg_command(job: Job, config: PreprocessConfig, target: str | Path | None = None) -> list[str]:
    """The ffmpeg command for ``job``, writing to ``target`` (default: its output)."""
    fps = f"{config.fps:g}"
    scale = f"scale={config.width}:{config.height}:flags={config.scale_flags}"
    if job.raw:
        before_input = ["-fflags", "+genpts", "-r", fps]
        vf = f"setpts=N/({fps}*TB),{scale}"
    else:
        before_input, vf = [], scale
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y", *before_input, "-i", str(job.source),
            "-map", "0:v:0", "-an", "-vf", vf, "-fps_mode", "passthrough",
            "-c:v", "libx264", "-crf", str(config.crf), "-preset", config.preset, "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target or job.output)]


def count_frames(path: str | Path) -> int | None:
    """The frames ffprobe decodes from ``path`` (reads all of it), or None."""
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                            capture_output=True, text=True, check=False)
    text = result.stdout.strip().splitlines()
    return int(text[0]) if text and text[0].isdigit() else None


def run_job(job: Job, config: PreprocessConfig, *, check_frames: bool = True) -> tuple[bool, str]:
    """Encode ``job``; return ``(ok, message)``.

    The encode is written as ``.part.mp4`` and renamed when ffmpeg succeeds, so an
    interrupted run leaves nothing to be mistaken for a finished video. With
    ``check_frames`` the output's frame count is compared with the source's; an
    output that differs is kept and reported, not deleted.

    Raises:
        FileNotFoundError: if ffmpeg (or, with ``check_frames``, ffprobe) is not installed.
    """
    for tool in ("ffmpeg", "ffprobe") if check_frames else ("ffmpeg",):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"{tool} is not installed or not on PATH")
    job.output.parent.mkdir(parents=True, exist_ok=True)
    part = job.output.with_name(job.output.stem + ".part.mp4")
    result = subprocess.run(ffmpeg_command(job, config, part), stdin=subprocess.DEVNULL, check=False)
    if result.returncode != 0:
        part.unlink(missing_ok=True)
        return False, f"ffmpeg failed (exit {result.returncode})"
    if not check_frames:
        part.rename(job.output)
        return True, "done"
    n_in, n_out = count_frames(job.source), count_frames(part)
    part.rename(job.output)
    if n_in is not None and n_in == n_out:
        return True, f"{n_out} frames"
    written = n_out if n_out is not None else "?"
    source = n_in if n_in is not None else "?"
    return False, f"CHECK: {written} frames written, the source has {source}; the output is kept"


def describe(config: PreprocessConfig) -> str:
    """One line: what a configuration does."""
    what = "raw streams only" if config.raw_only else "raw streams and container videos"
    return (f"{config.size}, crf {config.crf}, preset {config.preset}, {config.scale_flags} scaling, "
            f"raw streams at {config.fps:g} fps; {what}")
