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
import random
from pathlib import Path
from typing import Any

__all__ = [
    "workspace_to_dlc_project_dict",
    "labels_to_coco",
    "split_coco",
    "write_coco_json",
    "export_coco_dataset",
]


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


def labels_to_coco(labels_by_video, bodyparts, *, image_dims: dict | None = None) -> dict[str, Any]:
    """Convert per-video tidy label DataFrames to a single COCO dict.

    Args:
        labels_by_video: ``{video_id: long DataFrame}`` with columns
            ``image, individual, bodypart, x, y``.
        bodyparts: ordered bodypart names; keypoints are emitted in this order.
        image_dims: optional ``{file_name: (width, height)}``; defaults to 0x0
            (the trainer computes ground-truth bboxes from keypoints regardless).

    Returns a COCO dict with ``images``, ``annotations`` and ``categories``.
    Keypoints use visibility ``2`` for labeled points and ``0`` (at 0,0) for
    unlabeled ones, so fully-unlabeled individuals are dropped by the loader.
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
            images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})
            for _individual, ind_df in img_df.groupby("individual", sort=False):
                coords = {row.bodypart: (row.x, row.y) for row in ind_df.itertuples()}
                kpts: list[float] = []
                n_labeled = 0
                for bpt in bodyparts:
                    x, y = coords.get(bpt, (float("nan"), float("nan")))
                    if pd.isna(x) or pd.isna(y):
                        kpts += [0.0, 0.0, 0]
                    else:
                        kpts += [float(x), float(y), 2]
                        n_labeled += 1
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "keypoints": kpts, "num_keypoints": n_labeled, "bbox": [], "iscrowd": 0,
                })
                ann_id += 1
            img_id += 1
    categories = [{"id": 1, "name": "animal", "keypoints": list(bodyparts), "skeleton": []}]
    return {"images": images, "annotations": annotations, "categories": categories}


def split_coco(coco: dict, *, train_fraction: float = 0.95, seed: int | None = 0) -> tuple[dict, dict]:
    """Split a COCO dict into (train, test) by image (annotations follow their image)."""
    images = list(coco["images"])
    random.Random(seed).shuffle(images)
    n_train = round(len(images) * train_fraction)
    train_ids = {im["id"] for im in images[:n_train]}

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
) -> tuple[Path, Path]:
    """Stage a COCO dataset for training under ``dest``.

    Writes ``dest/train.json`` and ``dest/test.json`` and materializes the
    labeled frames under ``dest/images/<video_id>/`` (the COCO ``file_name``s).
    ``labels_provider(project, video_id) -> long DataFrame`` defaults to reading
    ``labels.parquet`` (pyarrow, lazy).

    Returns ``(train_json_path, test_json_path)``.
    """
    import shutil

    from .evaluate import read_labels

    dest = Path(dest)
    images_root = dest / "images"
    labels_provider = labels_provider or read_labels
    video_ids = list(video_ids) if video_ids is not None else project.annotated_videos()

    labels_by_video = {vid: labels_provider(project, vid) for vid in video_ids}
    coco = labels_to_coco(labels_by_video, project.config.bodyparts, image_dims=image_dims)
    train, test = split_coco(coco, train_fraction=train_fraction, seed=seed)
    train_path = write_coco_json(train, dest / "train.json")
    test_path = write_coco_json(test, dest / "test.json")

    # materialize frames at dest/images/<video_id>/<image>
    for vid in video_ids:
        src_dir = project.layout.frames_dir(vid)
        dst_dir = images_root / vid
        dst_dir.mkdir(parents=True, exist_ok=True)
        for frame in (src_dir.iterdir() if src_dir.is_dir() else []):
            if not frame.is_file():
                continue
            dst = dst_dir / frame.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if link == "symlink":
                dst.symlink_to(frame.resolve())
            elif link == "copy":
                shutil.copy2(frame, dst)
            else:
                raise ValueError(f"link must be symlink|copy, got {link!r}")
    return train_path, test_path
