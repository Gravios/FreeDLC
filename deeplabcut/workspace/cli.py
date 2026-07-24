#
# FreeDLC workspace layer
#
"""``dlc-ws`` -- a command-line interface over the workspace.

Thin wrappers around the workspace API: ``create``, ``list``,
``export-skeleton``, ``migrate``, ``info``, ``models``, ``videos``, ``apply``,
``label``, ``track``, ``export``, ``train``, ``evaluate``. Uses only argparse (no
extra dependencies), and each handler calls a single workspace function, so
parsing and dispatch are testable without torch; the
torch-backed commands (apply/train/evaluate) simply call their (lazily
torch-importing) workspace functions.

``create`` and ``migrate`` are the two ways to obtain a project: the former builds
one from scratch, the latter converts a legacy DeepLabCut tree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import skeleton_lib
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


def _parse_skeleton(
    tokens: list[str], bodyparts: list[str], unique_bodyparts: list[str]
) -> list[list[str]]:
    """Parse ``A,B`` command-line tokens into validated ``[a, b]`` skeleton edges.

    Edges may reference any declared part (``bodyparts`` or ``unique_bodyparts``);
    naming a part that was never declared is almost always a typo, so it is
    rejected here rather than silently producing edges that never render. This
    check lives in the CLI, not in :class:`~deeplabcut.workspace.schema.ProjectConfig`,
    so that migrating a legacy project with a stale ``skeleton`` keeps working.

    Raises:
        ValueError: if a token is not a comma-separated pair, or names an
            undeclared bodypart.
    """
    known = set(bodyparts) | set(unique_bodyparts)
    edges: list[list[str]] = []
    for token in tokens:
        pair = [part.strip() for part in token.split(",")]
        if len(pair) != 2 or not all(pair):
            raise ValueError(
                f"skeleton edge {token!r} must be two bodyparts separated by a comma, e.g. snout,tailbase"
            )
        undeclared = [part for part in pair if part not in known]
        if undeclared:
            raise ValueError(f"skeleton edge {token!r} names undeclared bodypart(s): {', '.join(undeclared)}")
        edges.append(pair)
    return edges


def cmd_create(args) -> int:
    if args.individuals and not args.multi_animal:
        print("--individuals requires --multi-animal")
        return 2
    if args.skeleton_config and (args.bodyparts or args.skeleton):
        print("--skeleton-config supplies the markers and edges; drop --bodyparts / --skeleton")
        return 2
    if not args.skeleton_config and not args.bodyparts:
        print("one of --bodyparts or --skeleton-config is required")
        return 2
    try:
        if args.skeleton_config:
            rig = skeleton_lib.resolve_skeleton(args.skeleton_config)
            bodyparts = [str(b) for b in rig["body_parts"]]
            skeleton = [list(edge) for edge in (rig.get("skeleton") or {}).get("edges") or []]
        else:
            bodyparts = args.bodyparts
            skeleton = _parse_skeleton(args.skeleton, args.bodyparts, args.unique_bodyparts)
        project = Project.create(
            args.root,
            task=args.task,
            bodyparts=bodyparts,
            experimenters=args.experimenters,
            multi_animal=args.multi_animal,
            individuals=args.individuals,
            unique_bodyparts=args.unique_bodyparts,
            skeleton=skeleton,
            exist_ok=args.exist_ok,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as err:  # bad input -> a message, not a traceback
        print(err)
        return 2
    config = project.config
    print(f"created -> {project.root}")
    print(f"  task: {config.task}  bodyparts: {len(config.bodyparts)}  skeleton: {len(config.skeleton)} edge(s)")
    if args.skeleton_config:
        print(f"  skeleton config: {rig.get('name') or args.skeleton_config}")
    if config.multi_animal:
        print(f"  individuals: {', '.join(config.individuals) or '-'}")
    return 0


def cmd_list_skeletons(args) -> int:
    if not args.name:
        names = skeleton_lib.available_skeletons()
        print("\n".join(names) if names else "no skeleton configs bundled")
        return 0
    try:
        config = skeleton_lib.load_skeleton(args.name)
    except (FileNotFoundError, ValueError) as err:
        print(err)
        return 2
    pose = config.get("pose") or {}
    segments = pose.get("segments") or []
    edges = (config.get("skeleton") or {}).get("edges") or []
    print(f"{config['name']} -- {config.get('description', '')}")
    print(f"  markers: {len(config['body_parts'])}  edges: {len(edges)}  segments: {len(segments)}")
    if segments:
        print(f"  root: {(pose.get('kinematics') or {}).get('root') or '-'}")
        for segment in segments:
            parent = segment.get("parent") or "-"
            print(f"    {segment['name']:<10} parent={parent:<10} {', '.join(segment['markers'])}")
    return 0


def cmd_export_skeleton(args) -> int:
    try:
        project = Project.open(args.project)
    except (FileNotFoundError, ValueError) as err:
        print(err)
        return 2
    if not project.config.skeleton:
        print(f"project has no skeleton edges to export: {project.layout.project_toml}")
        return 2
    segments_from = None
    try:
        if args.segments_from:
            segments_from = skeleton_lib.load_skeleton(args.segments_from)
        elif not args.no_segments:
            # A project created from a library rig exports back as a complete one.
            segments_from = skeleton_lib.matching_library_config(
                project.config.bodyparts, project.config.skeleton
            )
        config = skeleton_lib.config_from_project(
            project,
            name=args.name or project.config.task,
            description=args.description,
            segments_from=segments_from,
        )
        path = skeleton_lib.write_config(config, args.out)
    except (FileNotFoundError, ValueError, OSError) as err:
        print(err)
        return 2
    n_segments = len((config.get("pose") or {}).get("segments") or [])
    print(f"exported -> {path}")
    print(f"  markers: {len(config['body_parts'])}  edges: {len(config['skeleton']['edges'])}  "
          f"segments: {n_segments}")
    if not n_segments:
        print("  no kinematic tree: add [[pose.segments]] by hand, or re-run with --segments-from NAME")
    return 0


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


def cmd_track(args) -> int:
    from .track import track_parquet, tracked_parquet_path

    parquet = Path(args.parquet)
    if not parquet.is_file():
        print(f"no pose parquet found at {parquet}")
        return 2
    out = Path(args.out) if args.out else tracked_parquet_path(parquet)
    result, n = track_parquet(parquet, out, max_distance=args.max_distance,
                              max_gap=args.max_gap, pcutoff=args.pcutoff)
    print(f"{parquet} -> {result}  ({n} tracks)")
    return 0


def cmd_export(args) -> int:
    from .onnx_export import check_onnx_parity

    bundle = ModelBundle.open(args.bundle)
    if args.check:
        result = check_onnx_parity(bundle, opset=args.opset, atol=args.atol,
                                   rtol=args.rtol, batch=args.batch)
        for label, rep in result["reports"].items():
            for row in rep["rows"]:
                mark = "OK" if row["passed"] else "MISMATCH"
                print(f"  [{label}] {row['name']}: max|delta|={row['max_diff']:.2e}  {mark}")
        print("PARITY:", "PASS" if result["ok"] else "FAIL")
        if not result["ok"]:
            return 1                                     # don't export a model that doesn't match
    out = bundle.export_onnx(opset=args.opset)
    print(f"exported {out}")
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

    p = sub.add_parser("create", help="create a new workspace project")
    p.add_argument("root", help="directory to create the project in")
    p.add_argument("--task", required=True, help="experiment name recorded in project.toml")
    p.add_argument("--bodyparts", nargs="+", default=[], metavar="NAME",
                   help="keypoint names to track (or use --skeleton-config)")
    p.add_argument("--skeleton-config", dest="skeleton_config", metavar="NAME",
                   help="take the markers and edges from an installed skeleton config, or a path to one "
                        "(see: dlc-ws list skeletons)")
    p.add_argument("--experimenters", nargs="+", default=[], metavar="NAME")
    p.add_argument("--multi-animal", action="store_true", dest="multi_animal")
    p.add_argument("--individuals", nargs="+", default=[], metavar="NAME",
                   help="animal names (requires --multi-animal)")
    p.add_argument("--unique-bodyparts", nargs="+", default=[], dest="unique_bodyparts", metavar="NAME",
                   help="landmarks that do not belong to an individual (e.g. arena corners)")
    p.add_argument("--skeleton", action="extend", nargs="+", default=[], metavar="A,B",
                   help="skeleton edges as comma-separated bodypart pairs, e.g. --skeleton snout,tailbase")
    p.add_argument("--exist-ok", action="store_true", dest="exist_ok",
                   help="overwrite an existing project.toml instead of failing (sources/ and runs/ are untouched)")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("list", help="list saved configurations")
    listed = p.add_subparsers(dest="what", metavar="{skeletons}", required=True)
    q = listed.add_parser("skeletons", help="list the installed skeleton configs, or show one")
    q.add_argument("name", nargs="?", help="skeleton config to describe (default: list every name)")
    q.set_defaults(func=cmd_list_skeletons)

    p = sub.add_parser("export-skeleton", help="export a project's skeleton as a named config")
    p.add_argument("project", help="workspace project root")
    p.add_argument("--name", help="config name (default: the project's task)")
    p.add_argument("--out", default=".fdlc/skeletons", help="output directory (default: .fdlc/skeletons)")
    p.add_argument("--description", default="", help="description recorded in the config")
    p.add_argument("--segments-from", dest="segments_from", metavar="NAME",
                   help="copy the kinematic tree from this bundled config")
    p.add_argument("--no-segments", action="store_true", dest="no_segments",
                   help="write the graph only, without looking for a matching library tree")
    p.set_defaults(func=cmd_export_skeleton)

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

    p = sub.add_parser("track", help="assign cross-frame identities to a pose parquet")
    p.add_argument("parquet", help="pose parquet from apply (per-frame instances)")
    p.add_argument("--out", help="output parquet (default: <base>.tracked.fdlc.parquet)")
    p.add_argument("--max-distance", type=float, default=50.0, dest="max_distance",
                   help="max centroid movement (px) to link an instance to a track")
    p.add_argument("--max-gap", type=int, default=10, dest="max_gap",
                   help="frames a track may be unseen before it is retired")
    p.add_argument("--pcutoff", type=float, default=0.6)
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("export", help="export a bundle's pose model to ONNX (requires torch)")
    p.add_argument("bundle", help="model bundle directory")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--check", action="store_true",
                   help="verify onnxruntime matches torch forward before exporting")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8, help="batch size for the parity check")
    p.set_defaults(func=cmd_export)

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
