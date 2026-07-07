#
# FreeDLC workspace layer
#
"""Training, as a workspace run that produces a model bundle.

``train_model`` owns the run lifecycle and the workspace side of training: it
opens a ``runs/train/<run_id>/`` run, hands off to a pluggable *backend* that
performs the actual training and produces a PyTorch ``train`` directory, then
harvests that directory into a portable ``models/<model_id>/`` bundle linked back
to the run. The backend seam keeps the (torch-heavy, environment-specific)
compute out of the orchestration, so the orchestration is small and testable and
the trainer can be swapped without touching it.

``DlcPytorchBackend`` is the provided backend that drives DeepLabCut's PyTorch
trainer. It requires a torch environment and is imported lazily; it is the one
piece here that is not exercised without a GPU.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import ids
from .model_bundle import ModelBundle

__all__ = ["TrainConfig", "TrainBackend", "train_model", "DlcPytorchBackend"]


@dataclass
class TrainConfig:
    """Workspace-level training configuration.

    A small, explicit config in place of DeepLabCut's 20+ positional training
    arguments. ``detector_epochs > 0`` requests a top-down (detector + pose) model.
    """

    net_type: str = "resnet_50"
    epochs: int = 200
    batch_size: int = 8
    save_epochs: int = 25
    train_fraction: float = 0.95
    device: str | None = None
    detector_epochs: int = 0
    detector_batch_size: int = 8
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def top_down(self) -> bool:
        return self.detector_epochs > 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class TrainBackend(Protocol):
    """Performs training and returns the PyTorch ``train`` directory it produced.

    The returned directory must contain ``pytorch_config.yaml`` and one or more
    ``snapshot-*.pt`` checkpoints (and ``snapshot-detector-*.pt`` for top-down).
    """

    def __call__(self, project, run, config: TrainConfig) -> Path: ...


def train_model(project, config: TrainConfig, backend: TrainBackend, *,
                model_id: str | None = None) -> ModelBundle:
    """Run training and return the resulting portable :class:`ModelBundle`.

    Opens a train run, delegates to ``backend``, harvests the produced train
    directory into ``models/<model_id>/`` (linked to the run), and marks the run
    finished -- or failed, re-raising, if the backend errors.
    """
    run = project.new_run("train", params=config.to_dict())
    run.start()
    try:
        train_dir = backend(project, run, config)
    except Exception:
        run.fail()
        raise

    model_id = model_id or ids.new_model_id()
    bundle = ModelBundle.from_train_dir(
        project.layout.model_dir(model_id),
        train_dir,
        model_id=model_id,
        train_run_id=run.run_id,
    )
    run.finish(outputs=[f"models/{model_id}"])
    return bundle


class DlcPytorchBackend:
    """Backend that drives DeepLabCut's PyTorch trainer (requires torch).

    DeepLabCut's trainer writes into a legacy project's ``dlc-models-pytorch/``
    tree, so this backend is constructed against a legacy ``config.yaml`` and a
    shuffle; it runs the training and returns that shuffle's ``train`` directory
    for :func:`train_model` to harvest into the workspace. Imports of the trainer
    are deferred to call time.
    """

    def __init__(self, legacy_config: str | Path, *, shuffle: int = 1,
                 trainingsetindex: int = 0, modelprefix: str = ""):
        self.legacy_config = str(legacy_config)
        self.shuffle = shuffle
        self.trainingsetindex = trainingsetindex
        self.modelprefix = modelprefix

    def __call__(self, project, run, config: TrainConfig) -> Path:
        from deeplabcut.compat import train_network
        from deeplabcut.core.engine import Engine
        from deeplabcut.utils.auxiliaryfunctions import get_model_folder, read_config

        train_network(
            self.legacy_config,
            shuffle=self.shuffle,
            trainingsetindex=self.trainingsetindex,
            epochs=config.epochs,
            save_epochs=config.save_epochs,
            batch_size=config.batch_size,
            detector_epochs=config.detector_epochs,
            detector_batch_size=config.detector_batch_size,
            device=config.device,
            modelprefix=self.modelprefix,
            engine=Engine.PYTORCH,
            **config.extra,
        )
        cfg = read_config(self.legacy_config)
        model_folder = get_model_folder(
            config.train_fraction, self.shuffle, cfg,
            modelprefix=self.modelprefix, engine=Engine.PYTORCH,
        )
        return Path(cfg["project_path"]) / model_folder / "train"
