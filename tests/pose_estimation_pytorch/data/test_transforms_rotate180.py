"""The `rotate180` augmentation: a half turn that keeps keypoints on their pixels and their names."""

import random

import numpy as np

from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms


def _image_with_dots(h, w, points, colors):
    img = np.zeros((h, w, 3), np.uint8)
    for (x, y), c in zip(points, colors, strict=True):
        img[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] = c
    return img


def _apply(aug, img, points, labels):
    t = build_transforms(aug)
    return t(image=img, keypoints=points, class_labels=labels, bboxes=[[90, 40, 420, 270]], bbox_labels=[0])


def test_half_turn_is_exact_and_keeps_names():
    h, w = 360, 640
    points = [(100.0, 50.0), (500.0, 300.0)]
    colors = [(255, 0, 0), (0, 255, 0)]
    out = _apply({"rotate180": 1.0}, _image_with_dots(h, w, points, colors), points, ["left", "right"])
    assert out["class_labels"] == ["left", "right"]                   # a rotation, not a mirror
    np.testing.assert_allclose(out["keypoints"], [(w - 1 - 100, h - 1 - 50), (w - 1 - 500, h - 1 - 300)])
    for (x, y), c in zip(out["keypoints"], colors, strict=True):
        assert tuple(out["image"][int(y), int(x)]) == c                 # the dot moved with its keypoint


def test_half_turn_happens_with_its_probability():
    points = [(100.0, 50.0), (500.0, 300.0)]
    img = np.zeros((360, 640, 3), np.uint8)
    turned = 0
    for seed in range(400):
        random.seed(seed)
        np.random.seed(seed)
        kp = _apply({"rotate180": 0.5}, img, points, ["nose", "tail"])["keypoints"]
        turned += kp[0][0] > kp[1][0]
    assert 150 < turned < 250
    assert _apply({}, img, points, ["nose", "tail"])["keypoints"][0][0] == 100.0   # off unless asked for


def test_the_training_config_accepts_rotate180():
    """`fdlc train --rotate180` sets it on DeepLabCut's typed training config, which must have the field."""
    from deeplabcut.pose_estimation_pytorch.config.data import DataConfig

    cfg = DataConfig(train={"affine": {"p": 0.5, "rotation": 30, "scaling": [0.5, 1.25]}})
    cfg["train"]["rotate180"] = 0.5
    names = [type(t).__name__ for t in build_transforms(cfg["train"]).transforms]
    assert names[:2] == ["Sequential", "Affine"]                      # the half turn comes before the affine
    assert DataConfig(train={"rotate180": 0.25})["train"]["rotate180"] == 0.25
