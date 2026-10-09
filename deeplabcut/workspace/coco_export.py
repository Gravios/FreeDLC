#
# FreeDLC workspace layer
#
"""Export workspace annotations to COCO, for native training.

DeepLabCut's PyTorch trainer consumes a ``Loader``; its ``COCOLoader`` reads a
standard COCO JSON. So training natively -- straight from a workspace, with no
legacy ``config.yaml`` or ``dlc-models-pytorch/`` -- reduces to: convert the
tidy ``labels.parquet`` into a COCO dataset, then hand ``COCOLoader`` to the
trainer.

This module holds the pure conversion pieces (tidy long -> COCO dict, train/test
split, JSON writing, and the workspace-project -> DeepLabCut-project-dict mapping
that :func:`make_pytorch_pose_config` needs). They depend only on the stdlib and
pandas, so they are unit-tested; reading ``labels.parquet`` (pyarrow) and probing
image sizes are deferred to the training driver.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

__all__ = [
    "ANNOTATIONS_DIRNAME",
    "IMAGES_DIRNAME",
    "TRAIN_JSON",
    "TEST_JSON",
    "workspace_to_dlc_project_dict",
    "labels_to_coco",
    "split_coco",
    "write_coco_json",
    "export_coco_dataset",
]


log = logging.getLogger(__name__)

# The dataset-root shape DeepLabCut's COCOLoader expects: it loads
# ``<root>/annotations/<json>`` and resolves each ``file_name`` under ``<root>/images/``.
ANNOTATIONS_DIRNAME = "annotations"
IMAGES_DIRNAME = "images"
TRAIN_JSON = "train.json"
TEST_JSON = "test.json"

#: Pixels added around an individual's labeled keypoints to form its bounding box
#: (DeepLabCut's default ``bbox_margin``).
BBOX_MARGIN = 20


def workspace_to_dlc_project_dict(config) -> dict[str, Any]:
    """Map a workspace :class:`ProjectConfig` to the project dict that
    :func:`make_pytorch_pose_config` expects (the inverse of migration's mapping).
    """
    d: dict[str, Any] = {
        "Task": config.task,
        "scorer": (config.experimenters or ["scorer"])[0],
        "multianimalproject": config.multi_animal,
        "individuals": list(config.individuals) or ["individual1"],
        "uniquebodyparts": list(config.unique_bodyparts),
        "skeleton": [list(e) for e in config.skeleton],
    }
    if config.multi_animal:
        d["multianimalbodyparts"] = list(config.bodyparts)
        d["bodyparts"] = "MULTI!"
    else:
        d["bodyparts"] = list(config.bodyparts)
    return d


def _keypoints_bbox(points, image_w: int, image_h: int, margin: float = BBOX_MARGIN) -> list[float]:
    """COCO ``[x, y, w, h]`` box around labeled ``points``, grown by ``margin`` pixels.

    The margin keeps the box non-degenerate (the loader's augmentation rejects a box
    of zero width or height, which bare keypoints produce whenever they line up),
    and matches DeepLabCut's own keypoint boxes. Clipped to the image when its size
    is known (non-zero).
    """
    xs, ys = [x for x, _ in points], [y for _, y in points]
    x0, y0, x1, y1 = min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin
    if image_w and image_h:
        x0, y0 = min(max(x0, 0.0), image_w - 1.0), min(max(y0, 0.0), image_h - 1.0)
        x1, y1 = max(min(x1, float(image_w)), x0 + 1.0), max(min(y1, float(image_h)), y0 + 1.0)
    return [x0, y0, x1 - x0, y1 - y0]


def labels_to_coco(labels_by_video, bodyparts, *, image_dims: dict | None = None) -> dict[str, Any]:
    """Convert per-video tidy label DataFrames to a single COCO dict.

    Args:
        labels_by_video: ``{video_id: long DataFrame}`` with columns
            ``image, individual, bodypart, x, y``.
        bodyparts: ordered bodypart names; keypoints are emitted in this order.
        image_dims: optional ``{file_name: (width, height)}``; defaults to 0x0.

    Returns a COCO dict with ``images``, ``annotations`` and ``categories``.
    Keypoints use visibility ``2`` for labeled points and ``0`` (at 0,0) for
    unlabeled ones. Each annotation carries the ``bbox`` (``[x, y, w, h]``) and
    ``area`` of its labeled keypoints: DeepLabCut's ``COCOLoader`` discards any
    annotation whose ``bbox`` is empty, so without one nothing reaches the trainer.

    Only what was actually labeled is emitted. An individual with no labeled
    keypoint gets no annotation, and an image left with no annotation is dropped:
    a frame that was extracted but never labeled is not a "nothing here" example,
    and training on it as one teaches the model to predict nothing.
    """
    import pandas as pd

    image_dims = image_dims or {}
    images: list[dict] = []
    annotations: list[dict] = []
    img_id = 0
    ann_id = 0
    for video_id, df in labels_by_video.items():
        for image_name, img_df in df.groupby("image", sort=False):
            file_name = f"{video_id}/{image_name}"
            w, h = image_dims.get(file_name, (0, 0))
            image_annotations: list[dict] = []
            for _individual, ind_df in img_df.groupby("individual", sort=False):
                coords = {row.bodypart: (row.x, row.y) for row in ind_df.itertuples()}
                kpts: list[float] = []
                labeled: list[tuple[float, float]] = []
                for bpt in bodyparts:
                    x, y = coords.get(bpt, (float("nan"), float("nan")))
                    if pd.isna(x) or pd.isna(y):
                        kpts += [0.0, 0.0, 0]
                    else:
                        kpts += [float(x), float(y), 2]
                        labeled.append((float(x), float(y)))
                if not labeled:
                    continue
                x_min, y_min, width, height = _keypoints_bbox(labeled, w, h)
                image_annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "keypoints": kpts, "num_keypoints": len(labeled),
                    "bbox": [x_min, y_min, width, height], "area": width * height, "iscrowd": 0,
                })
                ann_id += 1
            if not image_annotations:
                continue
            images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})
            annotations.extend(image_annotations)
            img_id += 1
    categories = [{"id": 1, "name": "animal", "keypoints": list(bodyparts), "skeleton": []}]
    return {"images": images, "annotations": annotations, "categories": categories}


def split_coco(
    coco: dict, *, train_fraction: float = 0.95, seed: int | None = 0, keep_in_train=None,
) -> tuple[dict, dict]:
    """Split a COCO dict into (train, test) by image (annotations follow their image).

    Images whose ``file_name`` (``<video_id>/<image>``) is in ``keep_in_train`` --
    the frames an earlier model trained on, say -- always go to training, so none of
    them is tested on. The test set, ``1 - train_fraction`` of all images as before,
    is drawn from the others; when they are fewer, all of them are tested on.

    Raises:
        ValueError: if ``keep_in_train`` leaves no image to test on.
    """
    keep = set(keep_in_train or ())
    kept = [im for im in coco["images"] if im["file_name"] in keep]
    rest = [im for im in coco["images"] if im["file_name"] not in keep]
    if keep and not rest:
        raise ValueError(f"all {len(kept)} labeled frame(s) are in the training list; "
                         "label new frames to have something to test on")
    random.Random(seed).shuffle(rest)
    n_test = min(len(rest), len(coco["images"]) - round(len(coco["images"]) * train_fraction))
    # the test frames are taken from the end, so without a list the split is the one it always was
    train_ids = {im["id"] for im in kept} | {im["id"] for im in rest[:len(rest) - n_test]}

    def _subset(ids: set[int]) -> dict:
        return {
            "images": [im for im in coco["images"] if im["id"] in ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
            "categories": coco["categories"],
        }

    all_ids = {im["id"] for im in coco["images"]}
    return _subset(train_ids), _subset(all_ids - train_ids)


def write_coco_json(coco: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco))
    return path


def export_coco_dataset(
    project,
    dest: str | Path,
    *,
    video_ids=None,
    train_fraction: float = 0.95,
    seed: int | None = 0,
    link: str = "symlink",
    image_dims: dict | None = None,
    labels_provider=None,
    frames: str | None = None,
    keep_in_train=None,
) -> tuple[Path, Path]:
    """Stage a COCO dataset for training under ``dest``.

    Writes ``dest/annotations/train.json`` and ``dest/annotations/test.json`` and
    materializes the labeled frames under ``dest/images/<video_id>/`` (the COCO
    ``file_name``s) -- the ``annotations/`` + ``images/`` shape DeepLabCut's
    ``COCOLoader`` reads a dataset root as.
    ``labels_provider(project, video_id) -> long DataFrame`` defaults to reading
    ``labels.parquet`` (pyarrow, lazy).

    ``frames`` picks the frame set the dataset is built from, for every video:
    ``"processed"`` or ``"original"``. Label coordinates are converted into that
    pixel space, so either set can be trained on whichever one the labels are
    stored in. ``None`` uses, per video, the set the labels are stored in.

    ``keep_in_train`` names frames (``<video_id>/<image>``) that must be trained on,
    never tested on (see :func:`split_coco`).

    Returns ``(train_json_path, test_json_path)``.

    Raises:
        ValueError: if a video's labels cannot be put on ``frames`` (see
            :meth:`Project.check_frames`), or no labeled frame has a readable image.
    """
    from .evaluate import read_labels, scale_labels
    from .util import files_log, materialize, shown

    dest = Path(dest)
    images_root = dest / IMAGES_DIRNAME
    labels_provider = labels_provider or read_labels
    video_ids = list(video_ids) if video_ids is not None else project.annotated_videos()

    project.check_frames(video_ids, frames)

    # Stage each labeled frame at dest/images/<video_id>/<image> from the requested
    # frame set (by default the one the labels are stored in), and bring the label
    # coordinates into that set's pixel space, so frames and coordinates agree.
    #
    # Labels whose frame is not a readable file are left out of the dataset, and said
    # so here with the remedy: the loader would drop them anyway, but only after the
    # train/test split, and with a list of paths in the run directory that says
    # nothing about where the frame should have come from.
    labels_by_video = {}
    for vid in video_ids:
        df = scale_labels(labels_provider(project, vid), project.labels_scale_to(vid, frames))
        n_frames = df["image"].nunique()
        df = df.dropna(subset=["x", "y"])  # frames with no labeled keypoint drop out here
        if df["image"].nunique() < n_frames:
            log.info("%s: %d extracted frame(s) carry no labels and are not trained on",
                     vid, n_frames - df["image"].nunique())
        src_dir = project.layout.frames_dir(vid, frames or project.label_frames_kind(vid))
        names = list(dict.fromkeys(df["image"].tolist()))
        present = [name for name in names if (src_dir / name).is_file()]
        for name in present:
            materialize(src_dir / name, images_root / vid / name, link)
        if len(present) < len(names):
            log.warning(
                "%s: %d of %d labeled frame(s) have no readable image in %s and are left out "
                "of training; run `fdlc extract %s` to restore them",
                vid, len(names) - len(present), len(names), src_dir, vid,
            )
            df = df[df["image"].isin(present)]
        labels_by_video[vid] = df
        record, (scale_x, scale_y) = project.labels_record(vid), project.labels_scale_to(vid, frames)
        converted = "" if (scale_x, scale_y) == (1.0, 1.0) else f", coordinates x{scale_x:g} y{scale_y:g}"
        files_log.info("video %s: %d labeled frame(s)", vid, len(present))
        files_log.info("  labels  %s (stored in %s pixels%s)",
                       shown(project.layout.labels_parquet(vid)), record.space, converted)
        files_log.info("  frames  %s", shown(src_dir))

    coco = labels_to_coco(labels_by_video, project.config.bodyparts, image_dims=image_dims)
    if not coco["images"]:
        raise ValueError("no labeled frame has a readable image; there is nothing to train on")
    train, test = split_coco(coco, train_fraction=train_fraction, seed=seed, keep_in_train=keep_in_train)
    if keep_in_train:
        present = {im["file_name"] for im in coco["images"]}
        held = len(set(keep_in_train) & present)
        files_log.info("  kept in training: %d frame(s) from the training list%s", held,
                       f" ({len(set(keep_in_train)) - held} of the list no longer labeled)"
                       if held < len(set(keep_in_train)) else "")
    train_path = write_coco_json(train, dest / ANNOTATIONS_DIRNAME / TRAIN_JSON)
    test_path = write_coco_json(test, dest / ANNOTATIONS_DIRNAME / TEST_JSON)
    files_log.info("dataset %s", shown(dest))
    files_log.info("  train   %s (%d image(s))", shown(train_path), len(train["images"]))
    files_log.info("  test    %s (%d image(s))", shown(test_path), len(test["images"]))
    files_log.info("  images  %s (%s to the frames above)",
                   shown(images_root), "symlinks" if link == "symlink" else "copies")
    return train_path, test_path
