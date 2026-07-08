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

from .apply import (
    apply_to_videos,
    beside_video_path,
    collect_videos,
    labeled_video_path,
    read_fdlc_sidecar,
    sidecar_for_parquet,
)
from .evaluate import evaluate_model
from .migrate import migrate_project
from .model_bundle import ModelBundle
from .project import Project
from .train import TrainConfig, WorkspaceTrainBackend, train_model

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
    videos = collect_videos(args.videos)
    if not videos:
        print("no videos found")
        return 2

    if args.model:  # project-less drop-in model
        bundle = ModelBundle.open(args.model)
    else:
        if not args.model_id:
            print("--model-id is required with --project")
            return 2
        project = Project.open(args.project)
        bundle = ModelBundle.from_project(project, args.model_id)

    # skeleton edges: from the bundle (drop-in) else the project (project mode)
    skeleton = list(bundle.card.skeleton)
    if not args.model and not skeleton:
        skeleton = list(project.config.skeleton)
    common = dict(device=args.device, batch_size=args.batch_size, skeleton=skeleton,
                  labeled_video=args.labeled_video, pcutoff=args.pcutoff)

    def _report(results):
        for video, pose in results.items():
            print(f"{video} -> {pose}")

    # --beside-video: <stem>.fdlc.{parquet,toml}(+ .fdlc.mp4) next to each source video
    if args.beside_video:
        _report(apply_to_videos(bundle, videos, Path("."), beside_video=True, **common))
        return 0

    if args.model:  # drop-in, run-dir output
        out_root = Path(args.out) if args.out else Path("dlc-predictions")
        _report(apply_to_videos(bundle, videos, out_root, **common))
        return 0

    run = project.new_run("analyze", model_id=args.model_id, inputs=[str(v) for v in videos])
    run.start()
    out_root = Path(args.out) if args.out else run.dir
    results = apply_to_videos(bundle, videos, out_root, **common)
    run.finish(outputs=[str(p) for p in results.values() if p])
    _report(results)
    return 0


def cmd_label(args) -> int:
    from .label_video import render_labeled_from_parquet

    video = Path(args.video)
    parquet = Path(args.parquet) if args.parquet else beside_video_path(video)
    if not parquet.is_file():
        print(f"no pose parquet found at {parquet}")
        return 2

    # bodyparts + skeleton: explicit model/project, else the .fdlc.toml sidecar,
    # else derive bodyparts from the parquet (skeleton absent).
    bodyparts: list[str] | None = None
    skeleton: list[list[str]] = []
    if args.model:
        card = ModelBundle.open(args.model).card
        bodyparts, skeleton = list(card.bodyparts), list(card.skeleton)
    elif args.project:
        if not args.model_id:
            print("--model-id is required with --project")
            return 2
        project = Project.open(args.project)
        card = ModelBundle.from_project(project, args.model_id).card
        bodyparts = list(card.bodyparts)
        skeleton = list(card.skeleton) or list(project.config.skeleton)
    else:
        for sc in (sidecar_for_parquet(parquet), sidecar_for_parquet(beside_video_path(video))):
            if sc.is_file():
                bodyparts, skeleton = read_fdlc_sidecar(sc)
                break

    out = Path(args.out) if args.out else labeled_video_path(video)
    result = render_labeled_from_parquet(
        video, parquet, out, bodyparts=bodyparts, skeleton=skeleton,
        pcutoff=args.pcutoff, dotsize=args.dotsize,
    )
    print(f"{video} -> {result}")
    return 0


def cmd_train(args) -> int:
    project = Project.open(args.project)
    config = TrainConfig(net_type=args.net, epochs=args.epochs, batch_size=args.batch_size,
                         detector_epochs=args.detector_epochs, device=args.device,
                         train_fraction=args.train_fraction, seed=args.seed)
    bundle = train_model(project, config, WorkspaceTrainBackend())
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

    p = sub.add_parser("apply", help="label one or more videos (files, folders, or globs)")
    p.add_argument("videos", nargs="+", help="video files, directories, or glob patterns")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--project", help="workspace project (use with --model-id)")
    source.add_argument("--model", help="a model bundle directory (project-less drop-in)")
    p.add_argument("--model-id", dest="model_id", help="model id inside --project")
    p.add_argument("--out", help="output root directory")
    p.add_argument("--beside-video", action="store_true", dest="beside_video",
                   help="write <video-stem>.fdlc.parquet next to each source video (no run dir)")
    p.add_argument("--labeled-video", action="store_true", dest="labeled_video",
                   help="also render an annotated .fdlc.mp4 with keypoints + skeleton")
    p.add_argument("--pcutoff", type=float, default=0.6,
                   help="likelihood threshold for drawing keypoints in the labeled video")
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("label", help="render an annotated video from an existing .fdlc.parquet")
    p.add_argument("video", help="source video to annotate")
    p.add_argument("--parquet", help="pose parquet (default: <video-stem>.fdlc.parquet beside the video)")
    p.add_argument("--out", help="output video (default: <video-stem>.fdlc.mp4 beside the video)")
    p.add_argument("--project", help="project for bodyparts/skeleton (use with --model-id)")
    p.add_argument("--model", help="model bundle for bodyparts/skeleton")
    p.add_argument("--model-id", dest="model_id", help="model id inside --project")
    p.add_argument("--pcutoff", type=float, default=0.6)
    p.add_argument("--dotsize", type=int, default=5)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("train", help="train a model natively from annotations (requires torch)")
    p.add_argument("project")
    p.add_argument("--net", default="resnet_50")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--detector-epochs", type=int, default=0, dest="detector_epochs")
    p.add_argument("--train-fraction", type=float, default=0.95, dest="train_fraction")
    p.add_argument("--seed", type=int, default=0)
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
