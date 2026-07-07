#
# FreeDLC workspace layer
#
"""``dlc-ws`` -- a command-line interface over the workspace.

Thin wrappers around the workspace API: ``migrate``, ``info``, ``models``,
``videos``, ``apply``, ``train``, ``evaluate``. Uses only argparse (no extra
dependencies), and each handler calls a single workspace function, so parsing and
dispatch are testable without torch; the torch-backed commands (apply/train/
evaluate) simply call their (lazily torch-importing) workspace functions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ids
from .apply import apply_to_video
from .evaluate import evaluate_model
from .migrate import migrate_project
from .model_bundle import ModelBundle
from .project import Project
from .train import DlcPytorchBackend, TrainConfig, train_model

__all__ = ["main", "build_parser"]


def cmd_migrate(args) -> int:
    project = migrate_project(
        args.legacy_root,
        args.dest,
        link=args.link,
        include_videos=not args.no_videos,
        include_models=not args.no_models,
        include_annotations=not args.no_annotations,
    )
    print(f"migrated -> {project.root}")
    print(f"  videos: {len(project.videos())}  "
          f"annotated: {len(project.annotated_videos())}  "
          f"models: {len(project.models())}")
    return 0


def cmd_info(args) -> int:
    project = Project.open(args.project)
    c = project.config
    print(f"task:          {c.task}")
    print(f"experimenters: {', '.join(c.experimenters) or '-'}")
    print(f"bodyparts:     {', '.join(c.bodyparts)}")
    print(f"multi_animal:  {c.multi_animal}")
    if c.individuals:
        print(f"individuals:   {', '.join(c.individuals)}")
    print(f"videos:        {len(project.videos())} ({len(project.annotated_videos())} annotated)")
    print(f"models:        {len(project.models())}")
    for kind in ("train", "evaluate", "analyze"):
        n = len(project.runs(kind))
        if n:
            print(f"runs/{kind}:  {n}")
    return 0


def cmd_models(args) -> int:
    project = Project.open(args.project)
    for model_id in project.models():
        card = ModelBundle.from_project(project, model_id).card
        parts = [model_id, card.architecture, "top-down" if card.top_down else "bottom-up"]
        if card.legacy.get("shuffle") is not None:
            parts.append(f"shuffle={card.legacy['shuffle']}")
        if card.metrics.get("mean_error") is not None:
            parts.append(f"mean_error={card.metrics['mean_error']:.2f}")
        print("  ".join(parts))
    return 0


def cmd_videos(args) -> int:
    project = Project.open(args.project)
    annotated = set(project.annotated_videos())
    for vid in project.videos():
        print(f"{vid}{'  [annotated]' if vid in annotated else ''}")
    return 0


def cmd_apply(args) -> int:
    project = Project.open(args.project)
    bundle = ModelBundle.from_project(project, args.model_id)
    video_id = ids.slugify(Path(args.video).stem)
    run = project.new_run("analyze", model_id=args.model_id, inputs=[args.video])
    run.start()
    out_path = apply_to_video(bundle, args.video, run.video_dir(video_id),
                              device=args.device, batch_size=args.batch_size)
    run.finish(outputs=[str(out_path)])
    print(f"wrote {out_path}")
    return 0


def cmd_train(args) -> int:
    project = Project.open(args.project)
    config = TrainConfig(net_type=args.net, epochs=args.epochs, batch_size=args.batch_size,
                         detector_epochs=args.detector_epochs, device=args.device)
    backend = DlcPytorchBackend(args.legacy_config, shuffle=args.shuffle)
    bundle = train_model(project, config, backend)
    print(f"trained -> models/{bundle.card.model_id}")
    return 0


def cmd_evaluate(args) -> int:
    project = Project.open(args.project)
    bundle = ModelBundle.from_project(project, args.model_id)
    metrics = evaluate_model(project, bundle, videos=args.videos or None,
                             pcutoff=args.pcutoff, pck_threshold=args.pck)
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlc-ws", description="FreeDLC workspace CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("migrate", help="migrate a legacy DeepLabCut project")
    p.add_argument("legacy_root")
    p.add_argument("dest")
    p.add_argument("--link", choices=("symlink", "copy", "reference"), default="symlink")
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--no-models", action="store_true")
    p.add_argument("--no-annotations", action="store_true")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("info", help="summarize a workspace project")
    p.add_argument("project")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("models", help="list model bundles")
    p.add_argument("project")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("videos", help="list registered videos")
    p.add_argument("project")
    p.set_defaults(func=cmd_videos)

    p = sub.add_parser("apply", help="run a model on a video (no project scaffolding needed)")
    p.add_argument("project")
    p.add_argument("model_id")
    p.add_argument("video")
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("train", help="train a model (requires torch)")
    p.add_argument("project")
    p.add_argument("--legacy-config", required=True, dest="legacy_config",
                   help="legacy config.yaml driving the DeepLabCut PyTorch trainer")
    p.add_argument("--net", default="resnet_50")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--detector-epochs", type=int, default=0, dest="detector_epochs")
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--device")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a model against annotations")
    p.add_argument("project")
    p.add_argument("model_id")
    p.add_argument("--videos", nargs="*")
    p.add_argument("--pcutoff", type=float, default=0.6)
    p.add_argument("--pck", type=float, default=None)
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
