#
# FreeDLC workspace layer
#
"""Path resolution for the new workspace layout.

All knowledge of *where things live on disk* is concentrated here, so no other
module hard-codes directory names. The layout:

    <root>/
    |- project.toml
    |- sources/                     immutable inputs; the pipeline never writes here
    |  |- videos/<video_id>/video.mp4 (+ video.toml)
    |  '- annotations/<video_id>/{frames/, labels.parquet}
    |- models/<model_id>/           portable model bundles
    |  |- model.toml
    |  |- pose.yaml
    |  '- snapshots/
    |- runs/<kind>/<run_id>/        one isolated dir per generating operation
    |  '- run.toml (+ per-video outputs)
    '- derived/<video_id>/          stable "latest" views into runs/ (symlinks)

Filenames are intentionally boring and stable (``pose.parquet``, ``labels.parquet``,
``video.mp4``); the *directory* carries the coordinates. This is a Layout object
rather than loose functions so a root is resolved once and passed around.
"""
from __future__ import annotations

from pathlib import Path

from .schema import RUN_KINDS

__all__ = ["Layout"]


class Layout:
    """Resolves every workspace path relative to a project ``root``.

    Purely computational: constructing paths never touches the filesystem.
    Use :meth:`create_skeleton` to materialize the top-level directories.
    """

    #: Top-level directories created for a new project.
    TOP_LEVEL = ("sources/videos/original", "sources/videos/processed",
                 "sources/annotations", "models", "runs", "derived")

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"Layout(root={self.root!r})"

    # -- project ----------------------------------------------------------
    @property
    def project_toml(self) -> Path:
        return self.root / "project.toml"

    # -- sources: videos --------------------------------------------------
    #
    # Videos are split by kind: `original` holds full-resolution reference footage
    # (what frames are extracted from and annotated on), `processed` holds the
    # downscaled videos the model trains and runs on. A video id may appear under
    # both -- a mirrored pair -- and the original/processed resolution difference
    # is what defines a project's annotation scale.
    VIDEO_KINDS = ("original", "processed")

    @property
    def videos_dir(self) -> Path:
        return self.root / "sources" / "videos"

    def videos_kind_dir(self, kind: str = "original") -> Path:
        if kind not in self.VIDEO_KINDS:
            raise ValueError(f"video kind must be one of {self.VIDEO_KINDS}, got {kind!r}")
        return self.videos_dir / kind

    def video_dir(self, video_id: str, kind: str = "original") -> Path:
        return self.videos_kind_dir(kind) / video_id

    def video_media(self, video_id: str, suffix: str = ".mp4", kind: str = "original") -> Path:
        return self.video_dir(video_id, kind) / f"video{suffix}"

    def video_toml(self, video_id: str, kind: str = "original") -> Path:
        return self.video_dir(video_id, kind) / "video.toml"

    # -- sources: annotations --------------------------------------------
    @property
    def annotations_dir(self) -> Path:
        return self.root / "sources" / "annotations"

    def annotation_dir(self, video_id: str) -> Path:
        return self.annotations_dir / video_id

    def frames_dir(self, video_id: str, kind: str = "original") -> Path:
        if kind not in self.VIDEO_KINDS:
            raise ValueError(f"frame kind must be one of {self.VIDEO_KINDS}, got {kind!r}")
        return self.annotation_dir(video_id) / "frames" / kind

    def labels_parquet(self, video_id: str) -> Path:
        return self.annotation_dir(video_id) / "labels.parquet"

    # -- models -----------------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def model_toml(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "model.toml"

    def pose_config(self, model_id: str, name: str = "pose.yaml") -> Path:
        return self.model_dir(model_id) / name

    def snapshots_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "snapshots"

    # -- runs -------------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def runs_kind_dir(self, kind: str) -> Path:
        self._check_kind(kind)
        return self.runs_dir / kind

    def run_dir(self, kind: str, run_id: str) -> Path:
        return self.runs_kind_dir(kind) / run_id

    def run_toml(self, kind: str, run_id: str) -> Path:
        return self.run_dir(kind, run_id) / "run.toml"

    def run_video_dir(self, kind: str, run_id: str, video_id: str) -> Path:
        return self.run_dir(kind, run_id) / video_id

    # -- derived ----------------------------------------------------------
    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    def derived_video_dir(self, video_id: str) -> Path:
        return self.derived_dir / video_id

    # -- helpers ----------------------------------------------------------
    def create_skeleton(self, exist_ok: bool = True) -> None:
        """Create the top-level directory skeleton under ``root``."""
        self.root.mkdir(parents=True, exist_ok=exist_ok)
        for rel in self.TOP_LEVEL:
            (self.root / rel).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in RUN_KINDS:
            raise ValueError(f"unknown run kind {kind!r}; expected one of {RUN_KINDS}")
