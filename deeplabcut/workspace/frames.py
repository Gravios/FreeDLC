"""Extract annotation frames from a project's videos into ``sources/annotations/``.

napari-deeplabcut annotates a folder of extracted frames, not a raw video, so a
workspace project needs frames on disk before it can be labelled. This module reads a
registered video and writes evenly-spaced or content-clustered frames as PNGs into
``sources/annotations/<video_id>/frames/`` -- the exact directory the labelling flow
and :func:`~deeplabcut.workspace.annotations.ingest_video_annotations` already expect.

Two selection modes:

* ``uniform`` -- frames evenly spaced across the video. Deterministic and dependency-
  light (cv2 only), the sensible default for a first pass.
* ``kmeans`` -- cluster frames by downsampled appearance and take the one nearest each
  centroid, so visually distinct moments are favoured over evenly-spaced near-duplicates
  (DeepLabCut's classic strategy). Needs scikit-learn, imported lazily so ``uniform``
  never pays for it.

Frames are named ``img<frame_index>.png`` zero-padded to the video's frame count, so a
filename maps back to its source frame and sorts correctly. cv2 is imported inside the
functions, keeping the module importable (and the rest of the CLI testable) without it.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "VIDEO_MEDIA_GLOB",
    "resolve_media",
    "extract_frames",
]

VIDEO_MEDIA_GLOB = "video.*"


def resolve_media(project, video_id: str) -> Path:
    """Return the on-disk media file for a registered ``video_id``.

    Raises:
        FileNotFoundError: if the video is not registered, or its media is a
            ``reference`` (recorded path only, nothing materialized) that is gone.
    """
    if not project.has_video(video_id):
        raise FileNotFoundError(f"video {video_id!r} is not registered; run `dlc-ws add-video` first")
    vdir = project.layout.video_dir(video_id)
    media = sorted(vdir.glob(VIDEO_MEDIA_GLOB))
    if media:
        return media[0]
    # a "reference" link materializes nothing; fall back to the recorded source path
    record = project.video_record(video_id)
    src = Path(record.source_path)
    if src.is_file():
        return src
    raise FileNotFoundError(f"no media on disk for video {video_id!r} (looked in {vdir} and {src})")


def _frame_indices_uniform(total: int, n: int) -> list[int]:
    """Return ``n`` evenly-spaced frame indices across ``[0, total)``."""
    if n >= total:
        return list(range(total))
    # midpoints of n equal buckets -> avoids always grabbing the first/last frame
    return [min(total - 1, int((i + 0.5) * total / n)) for i in range(n)]


def _frame_indices_kmeans(video, total: int, n: int, *, resize: int = 32, step: int = 1) -> list[int]:
    """Return ``n`` frame indices chosen as the frames nearest k-means centroids.

    Every ``step``-th frame is read, downscaled to ``resize`` x ``resize`` greyscale and
    clustered; the frame closest to each centroid is kept. Falls back to uniform if
    there is too little material to cluster.
    """
    import cv2
    import numpy as np

    sampled_idx: list[int] = []
    feats: list[np.ndarray] = []
    for idx in range(0, total, max(1, step)):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = video.read()
        if not ok:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (resize, resize))
        sampled_idx.append(idx)
        feats.append(small.astype("float32").ravel())

    if len(feats) <= n:
        return sorted(sampled_idx) or _frame_indices_uniform(total, n)

    from sklearn.cluster import KMeans

    matrix = np.vstack(feats)
    km = KMeans(n_clusters=n, n_init=10, random_state=0).fit(matrix)
    chosen: list[int] = []
    for c in range(n):
        members = np.where(km.labels_ == c)[0]
        if members.size == 0:
            continue
        d = np.linalg.norm(matrix[members] - km.cluster_centers_[c], axis=1)
        chosen.append(sampled_idx[members[int(np.argmin(d))]])
    return sorted(dict.fromkeys(chosen))


def extract_frames(
    project,
    video_id: str,
    *,
    n: int = 20,
    mode: str = "uniform",
    overwrite: bool = False,
) -> list[Path]:
    """Extract frames from ``video_id`` into ``sources/annotations/<video_id>/frames/``.

    Args:
        n: number of frames to extract.
        mode: ``"uniform"`` (evenly spaced) or ``"kmeans"`` (content-clustered).
        overwrite: re-extract even if the frames directory already holds frames.

    Returns:
        The written frame paths, sorted. If frames already exist and ``overwrite`` is
        false, returns the existing frames without touching the video.

    Raises:
        ValueError: on an unknown ``mode`` or an unreadable/empty video.
        FileNotFoundError: if the video is not registered or its media is missing.
    """
    import cv2

    if mode not in ("uniform", "kmeans"):
        raise ValueError(f"mode must be 'uniform' or 'kmeans', got {mode!r}")

    frames_dir = project.layout.frames_dir(video_id)
    existing = sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else []
    if existing and not overwrite:
        return existing

    media = resolve_media(project, video_id)
    video = cv2.VideoCapture(str(media))
    try:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"video {media} reports no frames (unreadable or empty)")
        if mode == "uniform":
            indices = _frame_indices_uniform(total, n)
        else:
            indices = _frame_indices_kmeans(video, total, n)

        frames_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for stale in frames_dir.glob("*.png"):
                stale.unlink()

        width = max(4, len(str(total - 1)))
        written: list[Path] = []
        for idx in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = video.read()
            if not ok:
                continue
            out = frames_dir / f"img{idx:0{width}d}.png"
            cv2.imwrite(str(out), frame)
            written.append(out)
    finally:
        video.release()

    if not written:
        raise ValueError(f"no frames could be read from {media}")
    return sorted(written)
