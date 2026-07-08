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


def _portable_pose_config(cfg: dict) -> dict:
    """Return a copy of a pose config with machine-specific absolute paths removed.

    A trained model's ``pytorch_config.yaml`` embeds absolute paths (the source
    project, the config's own location, and the pretrained ``weight_init``
    checkpoints used at training time). None of these are needed to run
    inference, and keeping them would make the bundle non-relocatable -- so they
    are stripped. ``pose_config_path`` is left as the relative ``pose.yaml`` and
    resolved to the bundle's real location at runtime.
    """
    import copy

    cfg = copy.deepcopy(cfg)
    meta = cfg.setdefault("metadata", {})
    meta["project_path"] = ""
    meta["pose_config_path"] = "pose.yaml"
    # weight_init is a training-only concern and carries absolute checkpoint paths.
    train_settings = cfg.get("train_settings")
    if isinstance(train_settings, dict):
        train_settings.pop("weight_init", None)
    return cfg


def _place_snapshot(src: str | Path, dst: Path, link: str) -> None:
    """Copy or symlink a snapshot into the bundle (``link`` is ``copy``|``symlink``)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link == "symlink":
        dst.symlink_to(Path(src).resolve())
    elif link == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"link must be 'copy' or 'symlink', got {link!r}")


def _write_portable_pose_config(src: str | Path, dst: Path) -> None:
    """Read a pose config, strip absolute paths, and write it to ``dst``."""
    import yaml

    with Path(src).open() as fh:
        cfg = yaml.safe_load(fh) or {}
    with dst.open("w") as fh:
        yaml.safe_dump(_portable_pose_config(cfg), fh, sort_keys=False)


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
        legacy: dict | None = None,
        link: str = "copy",
        exist_ok: bool = False,
    ) -> ModelBundle:
        """Assemble a portable bundle at ``dest`` from a trained model's files.

        Writes a path-stripped copy of the pose config to ``dest/pose.yaml`` and
        places the snapshot into ``dest/snapshots/`` (recorded as the default
        snapshot), optionally a detector snapshot for top-down models, and writes
        ``model.toml``. ``link`` places snapshots by ``copy`` (default) or
        ``symlink`` (useful for large checkpoints). The bundle contains no
        absolute paths, so it can be moved anywhere. Nothing here needs torch.
        """
        dest = Path(dest)
        if (dest / "model.toml").exists() and not exist_ok:
            raise FileExistsError(f"a model bundle already exists at {dest}")
        if top_down and detector_snapshot_src is None:
            raise ValueError("top_down=True requires detector_snapshot_src")

        snapshots = dest / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)

        _write_portable_pose_config(pose_config_src, dest / "pose.yaml")
        pose_name = f"pose-{Path(snapshot_src).name}"
        _place_snapshot(snapshot_src, snapshots / pose_name, link)

        detector_name: str | None = None
        if detector_snapshot_src is not None:
            detector_name = f"detector-{Path(detector_snapshot_src).name}"
            _place_snapshot(detector_snapshot_src, snapshots / detector_name, link)

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
            legacy=dict(legacy or {}),
        )
        write_manifest(dest / "model.toml", card.to_dict())
        return cls(dest, card)

    @classmethod
    def from_project(cls, project, model_id: str) -> ModelBundle:
        """Open the bundle stored at ``models/<model_id>/`` inside a project."""
        return cls.open(Layout(project.root).model_dir(model_id))

    @classmethod
    def from_train_dir(
        cls,
        dest: str | Path,
        train_dir: str | Path,
        *,
        model_id: str | None = None,
        train_run_id: str | None = None,
        metrics: dict | None = None,
        legacy: dict | None = None,
        snapshots: str = "all",
        link: str = "copy",
        exist_ok: bool = False,
    ) -> ModelBundle:
        """Harvest a PyTorch ``train`` directory into a portable bundle.

        Reads ``pytorch_config.yaml`` and the ``snapshot-*.pt`` checkpoints,
        picks the default snapshot (best, else highest epoch), and writes
        ``model.toml``. ``snapshots="all"`` brings every checkpoint across;
        ``snapshots="best"`` keeps only the default pose (and detector) snapshot
        -- useful when the training run holds many large checkpoints. ``link``
        places snapshots by ``copy`` (default) or ``symlink``. The resulting
        bundle is path-free and relocatable. Shared by migration and training.
        """
        from . import _snapshots

        train_dir = Path(train_dir)
        pose_config = train_dir / "pytorch_config.yaml"
        if not pose_config.is_file():
            raise FileNotFoundError(f"no pytorch_config.yaml in {train_dir}")
        pose = _snapshots.list_snapshots(train_dir, detector=False)
        if not pose:
            raise FileNotFoundError(f"no pose snapshots in {train_dir}")
        detector = _snapshots.list_snapshots(train_dir, detector=True)

        default_pose = _snapshots.pick_default_snapshot(pose)
        default_det = _snapshots.pick_default_snapshot(detector) if detector else None
        bundle = cls.create(
            dest,
            pose_config_src=pose_config,
            snapshot_src=default_pose,
            architecture=_snapshots.read_net_type(pose_config) or "unknown",
            bodyparts=_snapshots.read_bodyparts(pose_config),
            top_down=bool(detector),
            detector_snapshot_src=default_det,
            model_id=model_id,
            train_run_id=train_run_id,
            metrics=metrics,
            legacy=legacy,
            link=link,
            exist_ok=exist_ok,
        )
        if snapshots == "best":
            return bundle
        for src in pose:
            if src != default_pose:
                _place_snapshot(src, bundle.snapshots_dir / f"pose-{src.name}", link)
        for src in detector:
            if src != default_det:
                _place_snapshot(src, bundle.snapshots_dir / f"detector-{src.name}", link)
        return bundle

    # -- paths ------------------------------------------------------------
    @property
    def pose_config_path(self) -> Path:
        return self.path / self.card.pose_config

    def set_metrics(self, metrics: dict) -> None:
        """Record evaluation metrics on the card and persist ``model.toml``."""
        self.card.metrics = dict(metrics)
        write_manifest(self.path / "model.toml", self.card.to_dict())

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
            model_config=str(self.pose_config_path),
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
            model_config=str(self.pose_config_path),
            snapshot_path=str(self.detector_snapshot_path(snapshot)),
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
