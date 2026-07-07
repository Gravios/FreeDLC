#
# FreeDLC workspace layer
#
"""Ingest legacy annotations into the workspace.

Converts a project's ``labeled-data/<video>/CollectedData_<scorer>.h5`` (or
``.csv``) -- a wide DataFrame with a scorer/[individuals]/bodyparts/coords column
MultiIndex -- into a tidy long ``sources/annotations/<video_id>/labels.parquet``
with columns ``image, individual, bodypart, x, y``, and materializes the labeled
frames under ``.../frames/``.

The scorer, which the legacy format bakes into every file name and column, is
dropped from the data (it lives once in ``project.toml``'s ``experimenters``).

pandas is imported lazily. Reading ``.h5`` needs pytables and writing Parquet
needs pyarrow; both are deferred to call time. The wide->long conversion and CSV
reading depend only on pandas and are unit-tested.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import ids
from .apply import SINGLE_INDIVIDUAL

__all__ = [
    "read_collected_data",
    "collected_data_to_long_df",
    "write_labels_parquet",
    "copy_frames",
    "find_collected_data",
    "ingest_video_annotations",
    "ingest_annotations",
]

log = logging.getLogger(__name__)


def _image_name(index_value) -> str:
    """Reduce a DataFrame index entry to a frame filename.

    Handles both the flat string index (``labeled-data/clip/img1.png``) and the
    MultiIndex form (``("labeled-data", "clip", "img1.png")``) used by DLC.
    """
    if isinstance(index_value, tuple):
        return str(index_value[-1])
    return Path(str(index_value)).name


def _csv_header_rows(path: Path) -> int:
    """Number of header rows in a CollectedData CSV (up to and incl. the ``coords`` row)."""
    with path.open() as fh:
        for i, line in enumerate(fh):
            if line.split(",", 1)[0].strip().strip('"') == "coords":
                return i + 1
    # Fall back to the single-animal default (scorer, bodyparts, coords).
    return 3


def read_collected_data(path: str | Path):
    """Read a ``CollectedData_*`` annotation file into a wide DataFrame.

    ``.h5`` is read via :func:`pandas.read_hdf` (needs pytables); ``.csv`` via
    :func:`pandas.read_csv` with the header depth detected from the ``coords`` row.
    """
    import pandas as pd

    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        return pd.read_hdf(path)
    if path.suffix == ".csv":
        n = _csv_header_rows(path)
        return pd.read_csv(path, header=list(range(n)), index_col=0)
    raise ValueError(f"unsupported annotation file: {path}")


def collected_data_to_long_df(df):
    """Convert a wide CollectedData DataFrame to tidy long form.

    Returns a DataFrame with columns ``[image, individual, bodypart, x, y]``,
    preserving bodypart/individual order. The scorer level is dropped.
    """
    import numpy as np
    import pandas as pd

    names = list(df.columns.names)
    coord_name = "coords" if "coords" in names else names[-1]
    bpt_name = "bodyparts" if "bodyparts" in names else names[-2]
    ind_name = "individuals" if "individuals" in names else None

    images = [_image_name(k) for k in df.index]

    # group columns by (individual, bodypart), preserving first-seen order
    groups: dict[tuple[str, str], dict[str, tuple]] = {}
    for col in df.columns:
        fields = dict(zip(names, col, strict=True))
        individual = fields.get(ind_name, SINGLE_INDIVIDUAL) if ind_name else SINGLE_INDIVIDUAL
        bodypart = fields[bpt_name]
        coord = fields[coord_name]
        if coord in ("x", "y"):
            groups.setdefault((individual, bodypart), {})[coord] = col

    n = len(df)
    nan = np.full(n, np.nan)
    parts = []
    for (individual, bodypart), coords in groups.items():
        x = df[coords["x"]].to_numpy() if "x" in coords else nan
        y = df[coords["y"]].to_numpy() if "y" in coords else nan
        parts.append(pd.DataFrame({"image": images, "individual": individual,
                                   "bodypart": bodypart, "x": x, "y": y}))
    if not parts:
        return pd.DataFrame(columns=["image", "individual", "bodypart", "x", "y"])
    return pd.concat(parts, ignore_index=True)[["image", "individual", "bodypart", "x", "y"]]


def write_labels_parquet(df, path: str | Path) -> Path:
    """Write a long-format labels DataFrame to Parquet (needs pyarrow)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def copy_frames(src_dir: str | Path, dest_dir: str | Path, images, *, link: str = "symlink") -> int:
    """Materialize the given frame files from ``src_dir`` into ``dest_dir``.

    ``link`` is ``"symlink"`` (default) or ``"copy"``. Returns the number of
    frames materialized; missing frames are warned about and skipped.
    """
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in images:
        src = src_dir / name
        if not src.is_file():
            log.warning("labeled frame not found: %s", src)
            continue
        dst = dest_dir / name
        if link == "symlink":
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        elif link == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"link must be symlink|copy, got {link!r}")
        count += 1
    return count


def find_collected_data(labeled_dir: str | Path) -> Path | None:
    """Locate the ``CollectedData_*`` file in a ``labeled-data/<video>`` dir (h5 preferred)."""
    labeled_dir = Path(labeled_dir)
    for pattern in ("CollectedData_*.h5", "CollectedData_*.csv"):
        hits = sorted(labeled_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def ingest_video_annotations(
    project,
    video_id: str,
    collected_data: str | Path,
    frames_dir: str | Path,
    *,
    link: str = "symlink",
    write: bool = True,
):
    """Ingest one video's annotations: write ``labels.parquet`` + copy frames.

    Returns ``(long_df, n_frames_materialized)``.
    """
    long = collected_data_to_long_df(read_collected_data(collected_data))
    images = list(dict.fromkeys(long["image"].tolist()))
    copied = copy_frames(frames_dir, project.layout.frames_dir(video_id), images, link=link)
    if write:
        write_labels_parquet(long, project.layout.labels_parquet(video_id))
    return long, copied


def ingest_annotations(project, legacy_root: str | Path, *, link: str = "symlink",
                       write: bool = True) -> dict[str, int]:
    """Ingest every ``labeled-data/<video>`` folder found under ``legacy_root``.

    Returns a mapping of ``video_id -> number of annotated frames``.
    """
    labeled = Path(legacy_root) / "labeled-data"
    summary: dict[str, int] = {}
    if not labeled.is_dir():
        return summary
    for vdir in sorted(p for p in labeled.iterdir() if p.is_dir()):
        collected = find_collected_data(vdir)
        if collected is None:
            continue
        video_id = ids.slugify(vdir.name)
        long, _ = ingest_video_annotations(project, video_id, collected, vdir, link=link, write=write)
        summary[video_id] = len(dict.fromkeys(long["image"].tolist()))
        log.info("ingested %d annotated frames for %s", summary[video_id], video_id)
    return summary
