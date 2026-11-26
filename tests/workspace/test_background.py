#
# FreeDLC workspace layer -- background plates and underlay tests
#
"""Tests for ``fdlc background`` and ``fdlc preprocess --underlay``. Needs OpenCV; the
end-to-end check needs ffmpeg. No torch.

Standalone: ``python tests/workspace/test_background.py`` -> ``background: N/N checks passed``.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from deeplabcut import workspace as ws
from deeplabcut.workspace import background as bg
from deeplabcut.workspace import cli
from deeplabcut.workspace.manifest import read_manifest, write_manifest
from deeplabcut.workspace.schema import LabelsRecord

W, H = 160, 120


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _bedding(seed=0, mean=150):
    """A light, grainy floor like bedding."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, 18, (H, W))
    return np.clip(cv2.GaussianBlur(noise, (3, 3), 0.8), 0, 255).astype(np.uint8)


def _track_frame(t, n=20):
    """A black track floor (10) between grey walls (110), a white animal (235) moving along it."""
    frame = np.full((H, W), 110, np.uint8)
    frame[30:90, :] = 10
    x = 20 + int((W - 40) * t / max(n - 1, 1))
    cv2.circle(frame, (x, 60), 9, 235, -1)
    return frame


def test_floors_from_labels_and_from_darkness():
    points = [(x, 60) for x in range(20, 141, 10)]
    floor = bg.floor_from_points((H, W), points)
    assert floor[60, 80] and floor[60, 20] and not floor[5, 80]          # along the markers, not the walls
    median = bg.median_background([_track_frame(t) for t in range(20)])
    assert int(median[60, 80]) == 10                                     # the animal is gone
    dark = bg.dark_floor(median, 40)
    assert dark[60, 5] and dark[35, 80] and not dark[10, 80] and not dark[100, 80]
    assert not bg.floor_from_points((H, W), [(np.nan, np.nan)]).any()


def test_a_new_floor_is_stitched_from_plate_floors():
    floor = np.zeros((H, W), bool)
    floor[20:100, 20:140] = True
    image = _bedding(seed=1)
    image[~floor] = 255                                                  # walls must never leak in
    plates = [bg.Plate(image=image, floor=floor)]
    made = bg.synthesize_floor(plates, (90, 200), patch=24, seed=3)
    assert made.shape == (90, 200) and made.max() < 250
    assert abs(float(made.mean()) - float(image[floor].mean())) < 6
    assert made.std() > 4                                                # textured, not flat
    assert np.array_equal(made, bg.synthesize_floor(plates, (90, 200), patch=24, seed=3))
    try:
        bg.synthesize_floor(plates, (90, 200), patch=200)
    except ValueError as err:
        assert "large enough" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_the_underlay_replaces_the_floor_but_not_the_animal():
    frames = [_track_frame(t) for t in range(20)]
    median = bg.median_background(frames)
    texture = _bedding(seed=2)
    under = bg.Underlay(median, texture, floor_max=40, animal_diff=25)
    assert 0.4 < under.coverage < 0.6
    out, share = under.apply(frames[10])
    assert 0.3 < share < 0.6
    x = 20 + int((W - 40) * 10 / 19)
    assert out[60, x] == 235                                             # the animal stays
    assert out[10, 80] == 110 and out[110, 80] == 110                    # so do the walls
    far = (slice(40, 80), slice(0, 10))                                  # floor away from the animal
    assert np.abs(out[far].astype(int) - texture[far].astype(int)).max() <= 1
    colour, _ = under.apply(cv2.cvtColor(frames[10], cv2.COLOR_GRAY2BGR))
    assert colour.shape == (H, W, 3) and np.array_equal(colour[..., 0], out)


def test_plates_are_made_from_a_projects_labeled_videos():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout", "tail"])
        video = d / "cage.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
        floor = _bedding(seed=4)
        for t in range(30):
            frame = floor.copy()
            cv2.circle(frame, (20 + 4 * t, 60), 8, 250, -1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        writer.release()
        vid = proj.add_video(video)
        write_manifest(proj.layout.labels_toml(vid), LabelsRecord(video_id=vid, space="original").to_dict())
        labels = pd.DataFrame([(f"img{t:03d}.png", "single", bp, 20.0 + 4 * t + dx, 60.0)
                               for t in range(30) for bp, dx in (("snout", 6), ("tail", -6))],
                              columns=["image", "individual", "bodypart", "x", "y"])
        lab = proj.layout.labels_parquet(vid)
        lab.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(lab)
        code, out = _run(["background", "--project", str(proj.root), "--frames", "original", "--samples", "15"])
        assert code == 0 and "plates of 1 video(s) (original frames)" in out, out
        folder = proj.root / "backgrounds" / "default"
        assert read_manifest(folder / "background.toml")["videos"] == [vid]
        (plate,) = bg.load_plates(folder)
        assert plate.image.shape == (H, W) and plate.floor[60, 80] and not plate.floor[5, 5]
        assert abs(int(plate.image[60, 80]) - int(floor[60, 80])) < 25        # the animal is gone
        code, out = _run(["background", "--project", str(proj.root), "--frames", "processed"])
        assert code == 2 and "no plate could be made" in out                    # no processed video


def test_preprocess_puts_the_plate_floor_under_the_animal():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("  (skipped: no ffmpeg)")
        return
    import csv

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        plates = d / "plates"
        plates.mkdir()
        floor = np.ones((H, W), bool)
        floor[:10], floor[-10:] = False, False
        cv2.imwrite(str(plates / "cage.png"), _bedding(seed=5))
        cv2.imwrite(str(plates / "cage-floor.png"), floor.astype(np.uint8) * 255)
        data = d / "data"
        data.mkdir()
        writer = cv2.VideoWriter(str(data / "track.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
        for t in range(20):
            writer.write(cv2.cvtColor(_track_frame(t), cv2.COLOR_GRAY2BGR))
        writer.release()
        old = Path.cwd()
        os.chdir(d)
        try:
            code, out = _run(["preprocess", "--config", "triplet", "--size", f"{W}x{H}", "--preset", "ultrafast",
                              "--crf", "0", "--no-stabilize", "--underlay", str(plates), "data"])
            missing = _run(["preprocess", "--config", "triplet", "--underlay", str(d / "none"), "data"])
        finally:
            os.chdir(old)
        assert code == 0 and "20 frames" in out and "dark floor (<= 40) replaced from" in out, out
        assert missing[0] == 2 and "no background plates" in missing[1]
        cap = cv2.VideoCapture(str(data / "track-triplet.mp4"))
        ok, frame = cap.read()
        for _ in range(9):
            ok, frame = cap.read()
        cap.release()
        assert ok
        b, g, r = (frame[..., c].astype(int) for c in range(3))
        far = (slice(40, 80), slice(0, 10))
        assert g[far].mean() > 100 and g[far].std() > 4                   # bedding now, not black (10)
        assert np.abs(r[far] - b[far]).max() <= 3                         # still still: grey in the triplet
        x = 20 + int((W - 40) * 9 / 19)
        assert g[60, x] > 220                                             # the animal stays
        with open(data / "track-triplet.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 20 and "floor" in rows[0] and 0.3 < float(rows[0]["floor"]) < 0.6


def _run_smoke() -> int:
    checks = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for chk in checks:
        chk()
    print(f"background: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
