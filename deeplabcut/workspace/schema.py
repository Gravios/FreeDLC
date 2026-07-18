#
# FreeDLC workspace layer
#
"""Typed schemas for the four workspace manifests.

These dataclasses are the *contract* of the new layout. Each maps 1:1 to a TOML
manifest and knows how to (de)serialize and validate itself:

    project.toml  -> ProjectConfig   (the experiment: bodyparts, individuals, ...)
    video.toml    -> VideoRecord     (one immutable source video)
    model.toml    -> ModelCard       (one portable, project-optional model bundle)
    run.toml      -> RunManifest      (one generating operation + its provenance)

Guiding principles baked into the fields:

  * Identity lives here, not in paths/filenames. The experimenter is a *field*
    (``ProjectConfig.experimenters``), never a path component.
  * Model coordinates (architecture, split, training run) are *fields* of the
    ModelCard, addressed by an opaque ``model_id`` -- not ``trainset95shuffle1``.
  * Every run records enough provenance (model, snapshot, inputs, code version,
    timestamps) to reproduce it without parsing any filename.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "RUN_KINDS",
    "now_iso",
    "ProjectConfig",
    "VideoRecord",
    "ModelCard",
    "RunManifest",
]

#: Bumped when a manifest's on-disk shape changes; drives migrations.
SCHEMA_VERSION = 1

#: The kinds of generating operation that get their own ``runs/<kind>/`` tree.
RUN_KINDS = ("train", "evaluate", "analyze")


def now_iso() -> str:
    """Current UTC time as a second-resolution ISO-8601 string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _require(d: dict[str, Any], *keys: str, ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{ctx} manifest is missing required key(s): {', '.join(missing)}")


@dataclass
class ProjectConfig:
    """``project.toml`` -- the single source of truth for an experiment."""

    task: str
    bodyparts: list[str]
    created: str = field(default_factory=now_iso)
    experimenters: list[str] = field(default_factory=list)
    multi_animal: bool = False
    individuals: list[str] = field(default_factory=list)
    unique_bodyparts: list[str] = field(default_factory=list)
    skeleton: list[list[str]] = field(default_factory=list)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("ProjectConfig.task must be a non-empty string")
        if not self.bodyparts:
            raise ValueError("ProjectConfig.bodyparts must list at least one bodypart")
        if len(set(self.bodyparts)) != len(self.bodyparts):
            raise ValueError("ProjectConfig.bodyparts contains duplicates")
        for edge in self.skeleton:
            if len(edge) != 2:
                raise ValueError(f"skeleton edges must be [a, b] pairs, got {edge!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProjectConfig:
        _require(d, "task", "bodyparts", ctx="project")
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class VideoRecord:
    """``video.toml`` -- provenance for one immutable source video.

    ``fps``/``width``/``height`` are optional because probing them requires a
    video backend (OpenCV); they are populated when a backend is available and
    left unset otherwise.
    """

    video_id: str
    source_path: str
    added: str = field(default_factory=now_iso)
    size_bytes: int | None = None
    sha256: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    n_frames: int | None = None
    link: str = "symlink"  # how the media is materialized: symlink | copy | reference
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VideoRecord:
        _require(d, "video_id", "source_path", ctx="video")
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ModelCard:
    """``model.toml`` -- a portable, project-optional trained model.

    A model bundle (``models/<model_id>/``) is self-contained: this card plus the
    pose config (``pose.yaml``) and snapshot(s) are everything needed to run
    inference, on any machine, with no surrounding project.
    """

    model_id: str
    architecture: str
    bodyparts: list[str]
    default_snapshot: str
    created: str = field(default_factory=now_iso)
    engine: str = "pytorch"
    top_down: bool = False
    pose_config: str = "pose.yaml"
    default_detector_snapshot: str | None = None
    train_run_id: str | None = None
    code_version: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    legacy: dict[str, Any] = field(default_factory=dict)
    skeleton: list[list[str]] = field(default_factory=list)
    pose_onnx: str | None = None
    detector_onnx: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.top_down and not self.default_detector_snapshot:
            raise ValueError("a top-down ModelCard requires default_detector_snapshot")

    @property
    def num_bodyparts(self) -> int:
        return len(self.bodyparts)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelCard:
        _require(d, "model_id", "architecture", "bodyparts", "default_snapshot", ctx="model")
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RunManifest:
    """``run.toml`` -- one generating operation and its provenance."""

    run_id: str
    kind: str
    created: str = field(default_factory=now_iso)
    status: str = "created"  # created | running | finished | failed
    started: str | None = None
    finished: str | None = None
    model_id: str | None = None
    snapshot: str | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    code_version: str | None = None
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in RUN_KINDS:
            raise ValueError(f"run kind must be one of {RUN_KINDS}, got {self.kind!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunManifest:
        _require(d, "run_id", "kind", ctx="run")
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
