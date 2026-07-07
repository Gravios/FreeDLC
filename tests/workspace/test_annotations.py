#
# FreeDLC workspace layer -- annotation ingest tests
#
"""Tests for deeplabcut.workspace.annotations.

Covers the wide CollectedData -> tidy long conversion, CSV reading (via a pandas
round-trip that reproduces DLC's header layout), frame materialization, and the
per-video / per-project ingest orchestration. Parquet writing (pyarrow) is the
only untested step and is monkeypatched where a full path is exercised.

Standalone: ``python tests/workspace/test_annotations.py`` -> ``annotations: N/N checks passed``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut import workspace as ws
from deeplabcut.workspace import annotations as ann


def _wide_single(images, bodyparts, scorer="gravio"):
    cols = pd.MultiIndex.from_product([[scorer], bodyparts, ["x", "y"]],
                                      names=["scorer", "bodyparts", "coords"])
    data = np.arange(len(images) * len(cols), dtype=float).reshape(len(images), len(cols))
    return pd.DataFrame(data, index=list(images), columns=cols)


def _wide_multi(images, individuals, bodyparts, scorer="lab"):
    cols = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, ["x", "y"]],
                                      names=["scorer", "individuals", "bodyparts", "coords"])
    data = np.arange(len(images) * len(cols), dtype=float).reshape(len(images), len(cols))
    return pd.DataFrame(data, index=list(images), columns=cols)


# ---------------------------------------------------------------- conversion
def test_image_name():
    assert ann._image_name("labeled-data/clip/img7.png") == "img7.png"
    assert ann._image_name(("labeled-data", "clip", "img7.png")) == "img7.png"


def test_convert_single_animal():
    df = _wide_single(["img1.png", "img2.png"], ["snout", "paw"])
    long = ann.collected_data_to_long_df(df)
    assert list(long.columns) == ["image", "individual", "bodypart", "x", "y"]
    assert len(long) == 2 * 2  # 2 frames x 2 bodyparts
    assert set(long["individual"]) == {ws.SINGLE_INDIVIDUAL}
    # img1 row values were [snout.x=0, snout.y=1, paw.x=2, paw.y=3]
    r = long[(long.image == "img1.png") & (long.bodypart == "paw")].iloc[0]
    assert (r.x, r.y) == (2.0, 3.0)
    # bodypart order preserved
    assert list(long[long.image == "img1.png"]["bodypart"]) == ["snout", "paw"]


def test_convert_multi_animal():
    df = _wide_multi(["img1.png"], ["m1", "m2"], ["snout", "tail"])
    long = ann.collected_data_to_long_df(df)
    assert len(long) == 1 * 2 * 2  # frames x individuals x bodyparts
    assert set(long["individual"]) == {"m1", "m2"}
    r = long[(long.individual == "m2") & (long.bodypart == "tail")].iloc[0]
    # columns order: m1.snout.xy, m1.tail.xy, m2.snout.xy, m2.tail.xy -> m2.tail = cols 6,7
    assert (r.x, r.y) == (6.0, 7.0)


# ------------------------------------------------------------------ csv read
def test_read_collected_data_csv_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        df = _wide_single(["img1.png", "img2.png"], ["snout", "paw"])
        csv = Path(d) / "CollectedData_gravio.csv"
        df.to_csv(csv)
        assert ann._csv_header_rows(csv) == 3
        back = ann.read_collected_data(csv)
        long = ann.collected_data_to_long_df(back)
        assert len(long) == 4 and long[(long.image == "img2.png") & (long.bodypart == "snout")].iloc[0].x == 4.0


# ---------------------------------------------------------------- frame copy
def test_copy_frames_symlink_and_missing():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        (src / "img1.png").write_bytes(b"px")
        dest = Path(d) / "dest"
        n = ann.copy_frames(src, dest, ["img1.png", "missing.png"], link="symlink")
        assert n == 1 and (dest / "img1.png").is_symlink() and not (dest / "missing.png").exists()


def test_find_collected_data_prefers_h5():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "CollectedData_x.csv").write_text("")
        (p / "CollectedData_x.h5").write_bytes(b"")
        assert ann.find_collected_data(p).suffix == ".h5"


# ----------------------------------------------------------------- ingest E2E
def _make_labeled_project(root: Path, video="clip1", bodyparts=("snout", "paw")):
    """A minimal workspace + a legacy labeled-data dir with CSV annotations + frames."""
    proj = ws.Project.create(root / "ws", task="reach", bodyparts=list(bodyparts))
    legacy = root / "legacy"
    ldir = legacy / "labeled-data" / video
    ldir.mkdir(parents=True)
    images = ["img1.png", "img2.png"]
    _wide_single(images, list(bodyparts)).to_csv(ldir / "CollectedData_gravio.csv")
    for name in images:
        (ldir / name).write_bytes(b"px")
    return proj, legacy, ldir, images


def test_ingest_video_annotations_no_write():
    with tempfile.TemporaryDirectory() as d:
        proj, _, ldir, images = _make_labeled_project(Path(d))
        long, copied = ann.ingest_video_annotations(
            proj, "clip1", ldir / "CollectedData_gravio.csv", ldir, write=False)
        assert copied == 2
        assert proj.layout.frames_dir("clip1").joinpath("img1.png").is_symlink()
        assert set(long["image"]) == set(images)


def test_ingest_annotations_writes_parquet(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj, legacy, _, _ = _make_labeled_project(Path(d))
        written = []
        monkeypatch.setattr(ann, "write_labels_parquet", lambda df, path: written.append(Path(path)))
        summary = ann.ingest_annotations(proj, legacy)  # write=True, but write fn is stubbed
        assert summary == {"clip1": 2}
        assert written == [proj.layout.labels_parquet("clip1")]
        assert proj.layout.frames_dir("clip1").joinpath("img2.png").is_symlink()


def test_migrate_includes_annotations(monkeypatch):
    # migrate_project should route through ingest_annotations when annotations exist.
    from deeplabcut.workspace import migrate

    with tempfile.TemporaryDirectory() as d:
        legacy = Path(d) / "legacy"
        (legacy / "videos").mkdir(parents=True)
        (legacy / "videos" / "clip1.mp4").write_bytes(b"v")
        ldir = legacy / "labeled-data" / "clip1"
        ldir.mkdir(parents=True)
        _wide_single(["img1.png"], ["snout", "paw"]).to_csv(ldir / "CollectedData_gravio.csv")
        (ldir / "img1.png").write_bytes(b"px")
        import yaml
        (legacy / "config.yaml").write_text(yaml.safe_dump({
            "Task": "reach", "scorer": "gravio", "multianimalproject": False,
            "bodyparts": ["snout", "paw"], "uniquebodyparts": [], "skeleton": [],
            "video_sets": {str((legacy / "videos" / "clip1.mp4").resolve()): {}},
        }))
        written = []
        monkeypatch.setattr(ann, "write_labels_parquet", lambda df, path: written.append(Path(path)))

        proj = migrate.migrate_project(legacy, Path(d) / "ws", include_annotations=True)
        assert written == [proj.layout.labels_parquet("clip1")]
        assert proj.layout.frames_dir("clip1").joinpath("img1.png").is_symlink()


# ------------------------------------------------------------------ smoke runner
def _run_smoke() -> int:
    class _MP:
        def setattr(self, obj, name, val):
            self._saved = getattr(self, "_saved", [])
            self._saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, val in reversed(getattr(self, "_saved", [])):
                setattr(obj, name, val)

    import inspect

    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        if "monkeypatch" in inspect.signature(chk).parameters:
            mp = _MP()
            try:
                chk(mp)
            finally:
                mp.undo()
        else:
            chk()
    print(f"annotations: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
