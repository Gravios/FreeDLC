#
# FreeDLC workspace layer
#
"""Background plates of a project, and floors made from them for other videos.

A model learns the floor its training videos show. Applied to videos filmed on
another floor -- a black track surface, say -- it meets pixels it never saw
around the animal. :func:`make_plates` records what the training floor looks
like, and :class:`Underlay` puts that floor under the animal in other videos
(``fdlc preprocess`` with ``underlay`` set).

**Plates** (``fdlc background``): for each labeled video of a project the empty
scene is estimated as the per-pixel median of frames sampled across the video --
the animal is rarely at the same place, so the median shows the floor without
it -- and the floor is where the labeled markers lie, widened and closed into
one region. Each video's plate and floor mask are written to
``<project>/backgrounds/<name>/<video_id>.png`` and ``<video_id>-floor.png``,
with a ``background.toml`` listing them.

**Underlays**: for a video to be prepared, its own median background is
estimated the same way, and its floor is the dark part of that median (at most
``floor_max``). A new floor of the plates' size is stitched from square patches
taken inside the plates' floors, overlapping and blended so no seams show. In
each frame the floor is swapped for it under the animal: a pixel brighter than
the background (a light animal on a dark floor) counts as animal in proportion to
how much brighter, and only the rest of it is swapped (see :class:`Underlay`). The
new floor is the same in every frame of a video, so in a triplet video it is
still grey and adds no motion; the animal's shadow, darker than the floor
around it, is replaced like the floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BACKGROUND_TOML",
    "Plate",
    "median_background",
    "floor_from_points",
    "dark_floor",
    "make_plates",
    "load_plates",
    "synthesize_floor",
    "Underlay",
]

BACKGROUND_TOML = "background.toml"
#: a floor region smaller than this share of the frame is noise
_MIN_REGION = 0.01


@dataclass
class Plate:
    """One video's empty scene and where its floor is."""

    image: object   # H x W uint8 grayscale
    floor: object   # H x W bool
    name: str = ""


def _gray(frame):
    import cv2

    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def median_background(frames):
    """The per-pixel median of grayscale (or BGR) ``frames``: the scene without what moves."""
    import numpy as np

    stack = np.stack([_gray(f) for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def sample_video_frames(path: str | Path, n: int) -> list:
    """Up to ``n`` grayscale frames spread evenly over a video (by seeking; for container videos)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"{path}: cannot tell how many frames it has")
        frames = []
        for index in np.linspace(0, total - 1, min(n, total)).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(_gray(frame))
        return frames
    finally:
        cap.release()


def _clean(mask, close_px: int, open_px: int = 0):
    """Remove specks narrower than ``open_px``, close gaps up to ``close_px``, fill holes, and keep the
    regions larger than :data:`_MIN_REGION` of the frame."""
    import cv2
    import numpy as np

    def disc(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    mask = mask.astype(np.uint8)
    if open_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disc(open_px))
    if close_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disc(close_px))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(mask)
    least = _MIN_REGION * mask.size
    for contour in contours:
        if cv2.contourArea(contour) >= least:
            cv2.drawContours(keep, [contour], -1, 1, thickness=cv2.FILLED)
    return keep.astype(bool)


def floor_from_points(shape, points, *, radius: int | None = None):
    """The floor of a scene: where ``points`` (labeled markers, ``N x 2`` x/y) lie, widened and closed."""
    import cv2
    import numpy as np

    h, w = shape[:2]
    radius = radius or max(3, round(0.04 * min(h, w)))
    mask = np.zeros((h, w), np.uint8)
    for x, y in np.asarray(points, float):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(mask, (round(x), round(y)), radius, 1, thickness=-1)
    return _clean(mask, close_px=radius * 2)


def dark_floor(median, floor_max: int):
    """The floor of a scene whose floor is dark: its median background at most ``floor_max``, cleaned."""
    h, w = median.shape[:2]
    r = max(1, round(0.005 * min(h, w)))
    return _clean(median <= floor_max, close_px=2 * r, open_px=r)


def make_plates(project, name: str = "default", *, kind: str = "processed", samples: int = 200,
                videos=None) -> Path:
    """Write the plates of ``project``'s labeled videos to ``backgrounds/<name>/``; return that folder.

    ``kind`` is the frame set to read (the one the model was trained on); labels,
    stored in original pixels, are scaled into it.

    Raises:
        ValueError: if no labeled video of ``kind`` gives a plate.
    """
    import cv2
    import numpy as np

    from .evaluate import read_labels, scale_labels
    from .frames import resolve_media
    from .manifest import write_manifest

    out = project.layout.root / "backgrounds" / name
    videos = list(videos) if videos else [v for v in project.annotated_videos() if project.has_video(v, kind)]
    written, skipped = [], []
    for vid in videos:
        try:
            frames = sample_video_frames(resolve_media(project, vid, kind), samples)
        except (FileNotFoundError, ValueError) as err:
            skipped.append(f"{vid}: {err}")
            continue
        if not frames:
            skipped.append(f"{vid}: no frames could be read")
            continue
        plate = median_background(frames)
        labels = scale_labels(read_labels(project, vid).dropna(subset=["x", "y"]),
                              project.labels_scale_to(vid, kind))
        floor = floor_from_points(plate.shape, labels[["x", "y"]].to_numpy())
        if not floor.any():
            skipped.append(f"{vid}: its labels give no floor region")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{vid}.png"), plate)
        cv2.imwrite(str(out / f"{vid}-floor.png"), floor.astype(np.uint8) * 255)
        written.append(vid)
    if not written:
        raise ValueError("no plate could be made" + "".join(f"\n  {s}" for s in skipped))
    write_manifest(out / BACKGROUND_TOML, {"name": name, "frames": kind, "samples": samples, "videos": written,
                                           "skipped": skipped})
    return out


def load_plates(folder: str | Path) -> list[Plate]:
    """The plates written by :func:`make_plates` into ``folder``.

    Raises:
        FileNotFoundError: if ``folder`` holds no plates.
    """
    import cv2

    folder = Path(folder).expanduser()
    plates = []
    for image_path in sorted(folder.glob("*.png")):
        if image_path.stem.endswith("-floor"):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}-floor.png")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if image is not None and mask is not None and image.shape == mask.shape:
            plates.append(Plate(image=image, floor=mask > 127, name=image_path.stem))
    if not plates:
        raise FileNotFoundError(f"no background plates in {folder} (make them with `fdlc background`)")
    return plates


def synthesize_floor(plates, shape, *, patch: int = 48, seed: int = 0):
    """A floor of ``shape`` stitched from ``patch``-sized squares of the plates' floors.

    Squares overlap by half and are blended with a ramp, so no seams show. The
    same ``seed`` gives the same floor.

    Raises:
        ValueError: if no plate's floor holds a whole square.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    kernel = np.ones((patch, patch), np.uint8)
    sources = []
    for plate in plates:  # top-left corners whose square lies wholly on the floor
        inside = cv2.erode(plate.floor.astype(np.uint8), kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT,
                           borderValue=0)
        ys, xs = np.nonzero(inside[: plate.floor.shape[0] - patch + 1, : plate.floor.shape[1] - patch + 1])
        if len(ys):
            sources.append((plate.image, ys, xs))
    if not sources:
        raise ValueError(f"no plate floor is large enough for {patch}x{patch} pixel patches")
    h, w = shape[:2]
    step = patch // 2
    ramp = np.minimum(np.arange(patch) + 1, patch - np.arange(patch)).astype(float)
    weight = np.outer(ramp, ramp)
    total = np.zeros((h + 2 * patch, w + 2 * patch))  # room for squares hanging over every edge
    weights = np.zeros_like(total)
    for top in range(-step, h, step):
        for left in range(-step, w, step):
            image, ys, xs = sources[rng.integers(len(sources))]
            i = rng.integers(len(ys))
            square = image[ys[i]: ys[i] + patch, xs[i]: xs[i] + patch].astype(float)
            t, l = top + step, left + step  # offset into the padded canvas
            total[t: t + patch, l: l + patch] += square * weight
            weights[t: t + patch, l: l + patch] += weight
    floor = total[step: step + h, step: step + w] / np.maximum(weights[step: step + h, step: step + w], 1e-9)
    return np.clip(floor.round(), 0, 255).astype(np.uint8)


class Underlay:
    """Puts a new floor under the animal in the frames of one video (see the module docstring).

    Each floor pixel is taken as a mix of animal and old floor, ``a`` its animal
    share: 0 at the background level, 1 at the animal's brightness (the median
    of the frame's animal pixels). The old floor is swapped for the new one in
    proportion to ``1 - a``: ``out = frame + (1 - a) * (new floor - background)``.
    The animal stays, its anti-aliased edge blends into the new floor instead of
    keeping a rim of the old one, and a shadow -- darker than the background --
    becomes the same shadow on the new floor.
    """

    def __init__(self, median, texture, *, floor_max: int, animal_diff: int):
        import cv2
        import numpy as np

        self.median = median.astype(np.float32)
        self.floor = dark_floor(median, floor_max)
        self.animal_diff = animal_diff
        self._floor_f = cv2.GaussianBlur(self.floor.astype(np.float32), (3, 3), 0)  # no jagged border
        self._swap = texture.astype(np.float32) - self.median

    @property
    def coverage(self) -> float:
        """The share of the frame taken for floor."""
        return float(self.floor.mean())

    def apply(self, frame):
        """``(frame with the new floor, share of the frame replaced)``; ``frame`` is grayscale or BGR."""
        import numpy as np

        gray = _gray(frame).astype(np.float32)
        above = gray - self.median
        animal = above > self.animal_diff
        level = float(np.median(above[animal])) if animal.any() else 255.0
        share = np.clip(above / max(level, 1.0), 0.0, 1.0)
        weight = self._floor_f * (1.0 - share)
        delta = weight * self._swap
        if frame.ndim == 3:
            delta = delta[..., None]
        out = frame.astype(np.float32) + delta
        return np.clip(out.round(), 0, 255).astype(np.uint8), float(weight.mean())
