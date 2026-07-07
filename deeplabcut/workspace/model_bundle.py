#
# FreeDLC workspace layer
#
"""Portable model bundles.

A :class:`ModelBundle` is a self-contained directory (``models/<model_id>/``)
holding everything needed to run inference -- a :class:`ModelCard`, the pose
config (``pose.yaml``) and the snapshot(s). Crucially it is *project-optional*:
a bundle can be copied to any machine and run on any video with no surrounding
project. This is the concrete form of the observation that a trained DeepLabCut
model is really just ``pose_config + snapshot``; the bundle makes that explicit
instead of burying the two files under ``dlc-models-pytorch/.../train/``.

The methods that actually build torch runners import the ``deeplabcut``
pose-estimation package lazily, so the workspace layer itself stays importable
without torch installed.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import ids
from .layout import Layout
from .manifest import read_manifest, write_manifest
from .schema import ModelCard
from .util import code_version

__all__ = ["ModelBundle"]


class ModelBundle:
    """A portable trained-model directory."""

    def __init__(self, path: str | Path, card: ModelCard):
        self.path = Path(path)
        self.card = card

    def __repr__(self) -> str:
        return f"ModelBundle(model_id={self.card.model_id!r}, arch={self.card.architecture!r})"

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path) -> ModelBundle:
        """Open an existing bundle directory (must contain ``model.toml``)."""
        path = Path(path)
        toml = path / "model.toml"
        if not toml.exists():
            raise FileNotFoundError(f"no model.toml in {path}")
        return cls(path, ModelCard.from_dict(read_manifest(toml)))

    @classmethod
    def create(
        cls,
        dest: str | Path,
        *,
        pose_config_src: str | Path,
        snapshot_src: str | Path,
        architecture: str,
        bodyparts: list[str],
        top_down: bool = False,
        detector_snapshot_src: str | Path | None = None,
        model_id: str | None = None,
        train_run_id: str | None = None,
        metrics: dict | None = None,
        exist_ok: bool = False,
    ) -> ModelBundle:
        """Assemble a portable bundle at ``dest`` from a trained model's files.

        Copies the pose config to ``dest/pose.yaml`` and the snapshot to
        ``dest/snapshots/`` (recorded as the default snapshot), optionally a
        detector snapshot for top-down models, and writes ``model.toml``.

        This is the "lift the two files out of a project into a portable bundle"
        operation. Nothing here needs torch.
        """
        dest = Path(dest)
        if (dest / "model.toml").exists() and not exist_ok:
            raise FileExistsError(f"a model bundle already exists at {dest}")
        if top_down and detector_snapshot_src is None:
            raise ValueError("top_down=True requires detector_snapshot_src")

        snapshots = dest / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)

        shutil.copy2(pose_config_src, dest / "pose.yaml")
        pose_name = f"pose-{Path(snapshot_src).name}"
        shutil.copy2(snapshot_src, snapshots / pose_name)

        detector_name: str | None = None
        if detector_snapshot_src is not None:
            detector_name = f"detector-{Path(detector_snapshot_src).name}"
            shutil.copy2(detector_snapshot_src, snapshots / detector_name)

        card = ModelCard(
            model_id=model_id or ids.new_model_id(),
            architecture=architecture,
            bodyparts=list(bodyparts),
            default_snapshot=pose_name,
            top_down=top_down,
            default_detector_snapshot=detector_name,
            train_run_id=train_run_id,
            code_version=code_version(),
            metrics=dict(metrics or {}),
        )
        write_manifest(dest / "model.toml", card.to_dict())
        return cls(dest, card)

    @classmethod
    def from_project(cls, project, model_id: str) -> ModelBundle:
        """Open the bundle stored at ``models/<model_id>/`` inside a project."""
        return cls.open(Layout(project.root).model_dir(model_id))

    # -- paths ------------------------------------------------------------
    @property
    def pose_config_path(self) -> Path:
        return self.path / self.card.pose_config

    @property
    def snapshots_dir(self) -> Path:
        return self.path / "snapshots"

    def snapshot_path(self, which: str = "default") -> Path:
        """Resolve a pose snapshot path.

        ``which`` is ``"default"`` (the card's default) or an explicit filename
        inside ``snapshots/``.
        """
        name = self.card.default_snapshot if which == "default" else which
        p = self.snapshots_dir / name
        if not p.exists():
            raise FileNotFoundError(f"snapshot {name!r} not found in {self.snapshots_dir}")
        return p

    def detector_snapshot_path(self, which: str = "default") -> Path:
        if which == "default":
            name = self.card.default_detector_snapshot
            if name is None:
                raise ValueError("this bundle has no detector snapshot (bottom-up model)")
        else:
            name = which
        p = self.snapshots_dir / name
        if not p.exists():
            raise FileNotFoundError(f"detector snapshot {name!r} not found in {self.snapshots_dir}")
        return p

    # -- runners (lazy torch) --------------------------------------------
    def _read_pose_config(self) -> dict:
        # Lazy import: config reader lives in the pytorch engine.
        from deeplabcut.core.config import read_config_as_dict

        return read_config_as_dict(str(self.pose_config_path))

    def build_pose_runner(
        self,
        *,
        snapshot: str = "default",
        device: str | None = None,
        batch_size: int = 1,
        max_individuals: int | None = None,
        **kwargs,
    ):
        """Build a pose :class:`InferenceRunner` from this bundle.

        Requires torch (imported lazily). Equivalent to the documented
        project-less path ``get_pose_inference_runner(model_cfg, snapshot)``.
        """
        from deeplabcut.pose_estimation_pytorch import get_pose_inference_runner

        return get_pose_inference_runner(
            model_config=self._read_pose_config(),
            snapshot_path=str(self.snapshot_path(snapshot)),
            batch_size=batch_size,
            device=device,
            max_individuals=max_individuals,
            **kwargs,
        )

    def build_detector_runner(
        self,
        *,
        snapshot: str = "default",
        device: str | None = None,
        batch_size: int = 1,
        **kwargs,
    ):
        """Build a detector :class:`InferenceRunner` (top-down models only)."""
        from deeplabcut.pose_estimation_pytorch import get_detector_inference_runner

        return get_detector_inference_runner(
            model_config=self._read_pose_config(),
            snapshot_path=str(self.detector_snapshot_path(snapshot)),
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
