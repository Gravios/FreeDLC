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

Frames are named ``img<frame_index>.png`` zero-padded to the original video's frame
count, so a name maps back to its source frame and sorts correctly. A processed frame
takes the *file name* of its original, so the two sets always match by name -- which is
how labels (keyed by image name) find their frame in either set. cv2 is imported inside
the functions, keeping the module (and the rest of the CLI) importable without it.

Extraction also keeps an existing frame set whole. A frame that has become a link back
into its own directory -- the self-links an earlier ingest bug left in place of the
images -- is re-read from the video at the index its name carries, and processed frames
missing for an original (the processed video having been registered after extraction)
are filled in. Neither changes which frames are selected, so existing labels keep
pointing at the same images.

``match_original`` goes one step further for a processed video that has been
*replaced* (another size, another encoding): the whole processed set is read again
from the video registered now, at the indices of the original frames. The processed
frames on disk were read from the old video, and nothing about their names says so.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

__all__ = [
    "resolve_media",
    "extract_frames",
]

log = logging.getLogger(__name__)

_FRAME_NAME = re.compile(r"img(\d+)\.png")


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
    for media in project.video_media_files(video_id, kind):
        if media.is_file():  # skips a symlink whose source has moved away
            return media
    record = project.video_record(video_id, kind)  # a "reference" materializes nothing
    src = Path(record.source_path)
    if src.is_file():
        return src
    vdir = project.layout.video_dir(video_id, kind)
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


def _frame_index(name: str) -> int | None:
    """The source frame index an extracted frame's file name carries, if it has one."""
    m = _FRAME_NAME.fullmatch(name)
    return int(m.group(1)) if m else None


def _grab(video, targets) -> list[Path]:
    """Write ``(frame_index, path)`` ``targets`` from an open capture; return those written.

    A path that is currently a symlink is unlinked first: writing through it would put
    the image wherever the link points (or fail outright on a dangling or looping one).
    """
    import cv2

    written: list[Path] = []
    for idx, out in targets:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = video.read()
        if not ok:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_symlink():
            out.unlink()
        if cv2.imwrite(str(out), frame):
            written.append(out)
    return sorted(written)


def _grab_from(project, video_id: str, kind: str, targets) -> list[Path]:
    """Open the registered ``kind`` video of ``video_id`` and :func:`_grab` ``targets``."""
    import cv2

    media = resolve_media(project, video_id, kind)
    video = cv2.VideoCapture(str(media))
    try:
        return _grab(video, targets)
    finally:
        video.release()


def _restore_frames(project, video_id: str, kind: str, names) -> list[str]:
    """Re-read the frames ``names`` of ``kind`` from their video; return the names restored.

    Each is re-read at the index its name carries and written under that same name.
    Best-effort: names that carry no index, and a video that is gone, are reported
    and left alone rather than raised.
    """
    frames_dir = project.layout.frames_dir(video_id, kind)
    targets = [(idx, frames_dir / name) for name in names if (idx := _frame_index(name)) is not None]
    restored: list[str] = []
    if targets:
        try:
            restored = [p.name for p in _grab_from(project, video_id, kind, targets)]
        except FileNotFoundError as err:
            log.warning("cannot restore %s frames of %s: %s", kind, video_id, err)
    lost = sorted(set(names) - set(restored))
    if lost:
        log.warning("%d %s frame(s) of %s are unreadable and were not restored (e.g. %s)",
                    len(lost), kind, video_id, frames_dir / lost[0])
    return restored


def _is_self_link(path: Path) -> bool:
    """True if ``path`` is an unreadable symlink pointing back into its own directory.

    That is a frame which can never become readable again (it names itself, or a
    sibling that is gone), as opposed to a link to a file elsewhere that is merely
    unavailable right now -- an unmounted legacy project, say.
    """
    if not path.is_symlink() or path.is_file():
        return False
    target = Path(os.path.join(path.parent, os.readlink(path)))
    return target.parent.resolve() == path.parent.resolve()


def _complete_existing(
    project, video_id: str, entries: list[Path], *, match_original: bool = False
) -> list[Path]:
    """Make an already-extracted frame set whole again; return the readable originals.

    Restores original frames that have become self-links, then fills in any processed
    frame missing for a readable original -- or, with ``match_original``, replaces the
    processed set with one read afresh for every readable original. The selection is
    left as it is. Links to missing files *outside* the frame set are only reported:
    their images may differ from the video's (a cropped legacy frame) and may yet come
    back.
    """
    self_links = [p.name for p in entries if _is_self_link(p)]
    if self_links:
        log.warning("%d original frame(s) of %s are links to themselves; re-extracting them",
                    len(self_links), video_id)
        _restore_frames(project, video_id, "original", self_links)
    frames = sorted(p for p in entries if p.is_file())
    dangling = [p for p in entries if not p.is_file() and p.name not in self_links]
    if dangling:
        log.warning("%d original frame(s) of %s link to missing files (e.g. %s -> %s)",
                    len(dangling), video_id, dangling[0], os.readlink(dangling[0]))

    if project.has_video(video_id, "processed"):
        proc_dir = project.layout.frames_dir(video_id, "processed")
        if match_original and frames:
            for stale in proc_dir.glob("*.png"):  # read from whatever was registered before
                stale.unlink()
        missing = [p.name for p in frames if not (proc_dir / p.name).is_file()]
        if missing:
            _restore_frames(project, video_id, "processed", missing)
    return frames


def extract_frames(
    project,
    video_id: str,
    *,
    n: int = 20,
    mode: str = "uniform",
    overwrite: bool = False,
    sample_stride: int | None = None,
    match_original: bool = False,
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
        match_original: read the processed frames again from the processed video
            registered now, at the indices of the existing original frames. Use it
            after replacing the processed video: the selection, the original frames
            and the labels keyed by their names all stay as they are. Requires a
            processed video. With no frames extracted yet it changes nothing --
            a fresh extraction writes matching sets anyway.

    Returns:
        The *original* frame paths, sorted. If frames already exist and ``overwrite``
        is false, the selection is kept: no new frames are chosen (``n`` and ``mode``
        are ignored) and the readable existing ones are returned, after restoring
        any that had become self-links and any processed frames missing for them.

    Raises:
        ValueError: on an unknown ``mode``, an unreadable/empty video, or existing
            frames none of which is readable or could be restored.
        FileNotFoundError: if the original video is not registered or its media is gone;
            with ``match_original``, likewise for the processed video.
    """
    import cv2

    if mode not in ("uniform", "kmeans"):
        raise ValueError(f"mode must be 'uniform' or 'kmeans', got {mode!r}")

    frames_dir = project.layout.frames_dir(video_id, "original")
    entries = sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else []
    if match_original:
        if overwrite:
            raise ValueError("match_original keeps the selected frames; it cannot be combined with overwrite")
        resolve_media(project, video_id, "processed")  # fail before any frame is removed
    if entries and not overwrite:
        frames = _complete_existing(project, video_id, entries, match_original=match_original)
        if not frames:
            raise ValueError(
                f"none of the frames in {frames_dir} is readable and they could not be restored; "
                f"re-run `dlc-ws extract --overwrite` for {video_id!r}"
            )
        return frames

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

        for stale in entries:
            stale.unlink()
        width = max(4, len(str(total - 1)))
        written = _grab(video, [(idx, frames_dir / f"img{idx:0{width}d}.png") for idx in indices])
    finally:
        video.release()

    if not written:
        raise ValueError(f"no frames could be read from {media}")

    # mirror the same frames from the processed video, if one is registered, under
    # the same file names so the two sets pair up by name
    if project.has_video(video_id, "processed"):
        proc_dir = project.layout.frames_dir(video_id, "processed")
        if proc_dir.is_dir():
            for stale in proc_dir.glob("*.png"):
                stale.unlink()
        _grab_from(project, video_id, "processed",
                   [(idx, proc_dir / p.name) for p in written if (idx := _frame_index(p.name)) is not None])

    return written
