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
    TOP_LEVEL = ("sources/videos", "sources/annotations", "models", "runs", "derived")

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"Layout(root={self.root!r})"

    # -- project ----------------------------------------------------------
    @property
    def project_toml(self) -> Path:
        return self.root / "project.toml"

    # -- sources: videos --------------------------------------------------
    @property
    def videos_dir(self) -> Path:
        return self.root / "sources" / "videos"

    def video_dir(self, video_id: str) -> Path:
        return self.videos_dir / video_id

    def video_media(self, video_id: str, suffix: str = ".mp4") -> Path:
        return self.video_dir(video_id) / f"video{suffix}"

    def video_toml(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "video.toml"

    # -- sources: annotations --------------------------------------------
    @property
    def annotations_dir(self) -> Path:
        return self.root / "sources" / "annotations"

    def annotation_dir(self, video_id: str) -> Path:
        return self.annotations_dir / video_id

    def frames_dir(self, video_id: str) -> Path:
        return self.annotation_dir(video_id) / "frames"

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
