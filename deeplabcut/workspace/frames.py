"""Extract annotation frames from a project's videos into ``sources/annotations/``.

napari-deeplabcut annotates a folder of extracted frames, not a raw video, so a
workspace project needs frames on disk before it can be labelled. This module reads a
project's *original* (full-resolution) video and writes frames into
``sources/annotations/<video_id>/frames/original/``, the set the annotator labels on.

When the video also has a *processed* (downscaled) counterpart registered, the SAME
frame indices are extracted from it into ``.../frames/processed/`` -- the set the model
trains on. Two frame sets are kept on purpose (route 1): annotation happens on the
crisp original for precise marker placement, while training uses the processed frames
so it matches the processed video inference runs on. Extracting the processed frames
from the processed video (rather than downscaling the original PNGs) keeps them
pixel-identical to what inference sees, including the exact scaler the reduction used.

Selection runs once, on the original:

* ``uniform`` -- frames evenly spaced across the video. Deterministic and dependency-
  light (cv2 only), the sensible default.
* ``kmeans`` -- cluster frames by downsampled appearance and take the one nearest each
  centroid (DeepLabCut's classic strategy). Needs scikit-learn, imported lazily.

Frames are named ``img<frame_index>.png`` zero-padded to the video's frame count, so a
name maps back to its source frame, sorts correctly, and matches between the original
and processed sets. cv2 is imported inside the functions, keeping the module (and the
rest of the CLI) importable without it.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "VIDEO_MEDIA_GLOB",
    "resolve_media",
    "extract_frames",
]

VIDEO_MEDIA_GLOB = "video.*"


def resolve_media(project, video_id: str, kind: str = "original") -> Path:
    """Return the on-disk media file for a registered ``video_id`` of ``kind``.

    Raises:
        FileNotFoundError: if the video is not registered under ``kind``, or its media
            is a ``reference`` whose recorded source path is gone.
    """
    if not project.has_video(video_id, kind):
        raise FileNotFoundError(
            f"{kind} video {video_id!r} is not registered; run `dlc-ws add-video` first"
        )
    vdir = project.layout.video_dir(video_id, kind)
    media = sorted(vdir.glob(VIDEO_MEDIA_GLOB))
    if media:
        return media[0]
    record = project.video_record(video_id, kind)  # a "reference" materializes nothing
    src = Path(record.source_path)
    if src.is_file():
        return src
    raise FileNotFoundError(f"no media on disk for {kind} video {video_id!r} (looked in {vdir} and {src})")


def _frame_indices_uniform(total: int, n: int) -> list[int]:
    """Return ``n`` evenly-spaced frame indices across ``[0, total)``."""
    if n >= total:
        return list(range(total))
    return [min(total - 1, int((i + 0.5) * total / n)) for i in range(n)]


def _frame_indices_kmeans(
    video, total: int, n: int, *, resize: int = 32, sample_stride: int | None = None, fps: float | None = None
) -> list[int]:
    """Return ``n`` representative frame indices, spread across the whole video.

    Every ``sample_stride``-th frame is decoded and reduced to a small grayscale
    feature; the sampled (time-ordered) frames are split into ``n`` equal temporal
    segments and the frame nearest each segment's appearance centroid is kept. That
    gives one frame per 1/n of the recording -- so the result always spans the video
    in time -- while still choosing a clean, representative frame within each window
    rather than a blind evenly-spaced grab.

    This deliberately does *not* cluster globally by appearance. Global k-means lets a
    brief but visually extreme episode (handling at the start or end, a light change,
    the animal on the lens) claim a disproportionate share of the clusters: those
    outlier frames each win a centroid while the near-identical majority collapses
    into a few, so the selection bunches up in time -- the opposite of useful
    annotation coverage. Stratifying by time first bounds every window's share to one
    frame. Appearance still chooses *which* frame within a window.

    Only every ``sample_stride``-th frame is decoded, which is the dominant cost on a
    long video. When ``sample_stride`` is not given it defaults to roughly one frame
    per second (from ``fps``), floored so the sample stays at least a few times ``n``.
    """
    import cv2
    import numpy as np

    if sample_stride is None:
        per_second = int(round(fps)) if fps and fps > 0 else 30
        # ~1 fps, but never so coarse that the sample has fewer than a few per segment
        sample_stride = max(1, min(per_second, total // max(n * 3, 1) or 1))

    sampled_idx: list[int] = []
    feats: list[np.ndarray] = []
    for idx in range(0, total, max(1, sample_stride)):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = video.read()
        if not ok:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (resize, resize))
        sampled_idx.append(idx)
        feats.append(small.astype("float32").ravel())

    if len(feats) <= n:
        return sorted(sampled_idx) or _frame_indices_uniform(total, n)

    # The sampled frames are already time-ordered; split them into n equal temporal
    # segments and keep each segment's medoid (frame nearest the segment mean).
    matrix = np.vstack(feats)
    bounds = np.linspace(0, len(sampled_idx), n + 1).astype(int)
    chosen: list[int] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b <= a:
            continue
        segment = matrix[a:b]
        center = segment.mean(axis=0)
        j = a + int(np.argmin(np.linalg.norm(segment - center, axis=1)))
        chosen.append(sampled_idx[j])
    return sorted(dict.fromkeys(chosen))


def _grab(video, indices, out_dir: Path, width: int):
    """Write the given frame ``indices`` from an open capture into ``out_dir``."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = video.read()
        if not ok:
            continue
        out = out_dir / f"img{idx:0{width}d}.png"
        cv2.imwrite(str(out), frame)
        written.append(out)
    return sorted(written)


def extract_frames(
    project,
    video_id: str,
    *,
    n: int = 20,
    mode: str = "uniform",
    overwrite: bool = False,
    sample_stride: int | None = None,
) -> list[Path]:
    """Extract annotation frames for ``video_id`` into ``sources/annotations/``.

    Frames are selected once on the original video and written to
    ``frames/original/``; if a processed counterpart is registered, the same indices
    are also extracted from it into ``frames/processed/`` for training.

    Args:
        n: number of frames to extract.
        mode: ``"uniform"`` (evenly spaced) or ``"kmeans"`` (content-clustered).
        overwrite: re-extract even if original frames already exist.
        sample_stride: for kmeans only -- decode every Nth frame when clustering.
            ``None`` defaults to roughly one frame per second, which is what keeps
            kmeans from decoding the whole video. Ignored by uniform.

    Returns:
        The written *original* frame paths, sorted. If they already exist and
        ``overwrite`` is false, returns them without touching any video.

    Raises:
        ValueError: on an unknown ``mode`` or an unreadable/empty video.
        FileNotFoundError: if the original video is not registered or its media is gone.
    """
    import cv2

    if mode not in ("uniform", "kmeans"):
        raise ValueError(f"mode must be 'uniform' or 'kmeans', got {mode!r}")

    frames_dir = project.layout.frames_dir(video_id, "original")
    existing = sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else []
    if existing and not overwrite:
        return existing

    media = resolve_media(project, video_id, "original")
    video = cv2.VideoCapture(str(media))
    try:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"video {media} reports no frames (unreadable or empty)")
        if mode == "uniform":
            indices = _frame_indices_uniform(total, n)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)
            indices = _frame_indices_kmeans(video, total, n, sample_stride=sample_stride, fps=fps)

        if overwrite:
            for stale in frames_dir.glob("*.png"):
                stale.unlink()
        width = max(4, len(str(total - 1)))
        written = _grab(video, indices, frames_dir, width)
    finally:
        video.release()

    if not written:
        raise ValueError(f"no frames could be read from {media}")

    # mirror the same frames from the processed video, if one is registered
    if project.has_video(video_id, "processed"):
        proc_media = resolve_media(project, video_id, "processed")
        proc_dir = project.layout.frames_dir(video_id, "processed")
        proc_cap = cv2.VideoCapture(str(proc_media))
        try:
            proc_total = int(proc_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or total
            proc_width = max(4, len(str(proc_total - 1)))
            if overwrite and proc_dir.is_dir():
                for stale in proc_dir.glob("*.png"):
                    stale.unlink()
            _grab(proc_cap, indices, proc_dir, proc_width)
        finally:
            proc_cap.release()

    return sorted(written)
