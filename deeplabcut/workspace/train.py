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

``WorkspaceTrainBackend`` is the provided backend: it trains natively from the
workspace's own annotations, with no legacy ``config.yaml`` or ``dlc-models``
layout. It requires torch and defers that work to
:mod:`~deeplabcut.workspace.native_train`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import ids
from .model_bundle import ModelBundle

__all__ = ["TrainConfig", "TrainBackend", "train_model", "WorkspaceTrainBackend"]


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


class WorkspaceTrainBackend:
    """Native training backend.

    Trains straight from the workspace's annotations (``sources/annotations/``)
    into ``runs/train/<id>/train/`` -- no legacy ``config.yaml`` and no
    ``dlc-models-pytorch/`` layout. Requires torch; the compute is deferred to
    :func:`deeplabcut.workspace.native_train.train_in_workspace`.
    """

    def __call__(self, project, run, config: TrainConfig) -> Path:
        from .native_train import train_in_workspace

        return train_in_workspace(project, run, config)
