#
# FreeDLC workspace layer
#
"""Compare two models, or two ``apply`` runs, side by side.

:func:`compare_models` scores two models against the labels, on the labeled frames
**neither** of them trained on (each model's ``train_frames.txt``, see
:meth:`~.model_bundle.ModelBundle.train_frames`), so neither is flattered by
frames it has seen. Both are scored in one pixel space, so their errors compare
directly; besides each model's median, mean, RMSE and PCK it reports, frame by
frame and marker by marker, which model came closer to the label.

:func:`compare_runs` needs no labels. For every video two ``analyze`` runs share,
it reports per marker how often each run draws it (likelihood at or above the
run's ``pcutoff``), its mean likelihood, and how far apart the two runs put it
where both draw it. It also picks the frames where the runs disagree most -- a
marker drawn by one run only, or drawn by both more than ``far`` pixels apart --
spread over each video as ``extract --from-run`` spreads its picks. Those frames
are where at least one of the models is wrong, which makes them good frames to
label next.

The inference (models) and the pose files (runs) are injectable, so the
comparison itself is testable without torch.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .evaluate import infer_on_frames, read_labels, scale_labels

__all__ = [
    "compare_models",
    "compare_runs",
    "format_model_comparison",
    "format_run_comparison",
    "DEFAULT_PCK",
    "DEFAULT_FAR",
]

KEYS = ["image", "individual", "bodypart"]
#: PCK threshold, in pixels of the frame set compared in
DEFAULT_PCK = 10.0
#: two runs disagree on a marker both draw when they put it further apart than this, in pixels
DEFAULT_FAR = 10.0


def _markers(project) -> list[str]:
    return list(project.config.bodyparts) + list(project.config.unique_bodyparts)


def _space(frames: str | None) -> str:
    return f"{frames} pixels" if frames else "the pixels the labels are stored in"


def _summary(errors, pck: float) -> dict[str, float]:
    """``n``, ``median``, ``mean``, ``rmse`` and ``pck`` of a Series of pixel errors."""
    import numpy as np

    n = int(len(errors))
    if not n:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "rmse": float("nan"), "pck": float("nan")}
    return {"n": n, "median": float(errors.median()), "mean": float(errors.mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())), "pck": float((errors <= pck).mean())}


def _errors(predictions, ground_truth):
    """Ground truth joined to predictions on image, individual and bodypart, with the pixel ``error``."""
    import numpy as np
    import pandas as pd

    if predictions.empty or ground_truth.empty:
        return pd.DataFrame(columns=[*KEYS, "error"])
    merged = pd.merge(ground_truth[[*KEYS, "x", "y"]], predictions[[*KEYS, "x", "y"]],
                      on=KEYS, suffixes=("_gt", "_pred"), how="inner")
    merged = merged.dropna(subset=["x_gt", "y_gt", "x_pred", "y_pred"])
    error = np.hypot(merged["x_pred"] - merged["x_gt"], merged["y_pred"] - merged["y_gt"])
    return merged[KEYS].assign(error=error.to_numpy())


def compare_models(
    project,
    bundle_a,
    bundle_b,
    *,
    videos: Sequence[str] | None = None,
    frames: str | None = None,
    pck_threshold: float = DEFAULT_PCK,
    all_frames: bool = False,
    predictions_provider: Callable | None = None,
    labels_provider: Callable | None = None,
) -> dict[str, Any]:
    """Score two models on the labeled frames neither trained on.

    Each model is run on the frame set it was trained on; its predictions and the
    labels are then brought into ``frames``, the pixel space everything is reported
    in. ``frames`` defaults to the set both models were trained on, or
    ``"original"`` when they differ. With ``all_frames`` every labeled frame is
    scored, training frames included.

    ``predictions_provider(project, bundle, video_id, frames, images)`` returns a
    long DataFrame in the pixels of ``frames`` (``None``: the set the labels are
    stored in); it defaults to running the model on those frames.

    Raises:
        ValueError: if the models are the same, a video's labels cannot be put on a
            frame set, a model does not record its training frames (without
            ``all_frames``), or no labeled frame is left to score.
    """
    import pandas as pd

    a, b = bundle_a.card, bundle_b.card
    if a.model_id == b.model_id:
        raise ValueError(f"both are model {a.model_id}; name two different models")
    labels_provider = labels_provider or read_labels
    if predictions_provider is None:
        def predictions_provider(project, bundle, video_id, kind, images):
            kind = kind or project.label_frames_kind(video_id)
            return infer_on_frames(bundle, project.layout.frames_dir(video_id, kind), images)

    if frames is None:
        frames = a.frames if a.frames == b.frames else "original"
    videos = list(videos) if videos is not None else project.annotated_videos()
    if not videos:
        raise ValueError("the project has no labeled videos; there is nothing to score on")
    for kind in dict.fromkeys((frames, a.frames, b.frames)):
        project.check_frames(videos, kind)

    trained: dict[str, set[str]] = {}
    for bundle in (bundle_a, bundle_b):
        listed = bundle.train_frames(project)
        if listed is None and not all_frames:
            raise ValueError(f"model {bundle.card.model_id} does not record which frames it trained on (no "
                             "train_frames.txt, and its training run is gone); give --all-frames to score on "
                             "every labeled frame, training frames included")
        trained[bundle.card.model_id] = set(listed or ())
    exclude = set() if all_frames else set().union(*trained.values())

    truth, scored = [], {a.model_id: [], b.model_id: []}
    n_labeled = 0
    for vid in videos:
        gt = labels_provider(project, vid).dropna(subset=["x", "y"])
        names = list(dict.fromkeys(gt["image"].tolist()))
        n_labeled += len(names)
        keep = [n for n in names if f"{vid}/{n}" not in exclude]
        if not keep:
            continue
        to_x, to_y = project.labels_scale_to(vid, frames)
        gt = scale_labels(gt[gt["image"].isin(keep)], (to_x, to_y))
        truth.append(gt.assign(image=vid + "/" + gt["image"]))
        for bundle in (bundle_a, bundle_b):
            own_x, own_y = project.labels_scale_to(vid, bundle.card.frames)
            pred = predictions_provider(project, bundle, vid, bundle.card.frames, keep)
            if pred.empty:
                continue
            pred = scale_labels(pred, (to_x / own_x, to_y / own_y))
            scored[bundle.card.model_id].append(pred.assign(image=vid + "/" + pred["image"]))
    if not truth:
        raise ValueError(f"all {n_labeled} labeled frame(s) were trained on by model {a.model_id} or "
                         f"{b.model_id}; label new frames to compare them on")
    ground_truth = pd.concat(truth, ignore_index=True)
    err = {mid: _errors(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), ground_truth)
           for mid, parts in scored.items()}

    paired = pd.merge(err[a.model_id], err[b.model_id], on=KEYS, suffixes=("_a", "_b"))

    def _closer(df) -> dict[str, float]:
        n = len(df)
        if not n:
            return {"n": 0, "a": float("nan"), "b": float("nan")}
        return {"n": n, "a": float((df["error_a"] < df["error_b"]).mean()),
                "b": float((df["error_b"] < df["error_a"]).mean())}

    def _model(mid: str) -> dict[str, Any]:
        e = err[mid]
        return {
            "model_id": mid,
            "trained_on": len(trained[mid]),
            "all": _summary(e["error"], pck_threshold),
            "per_bodypart": {bp: _summary(e.loc[e["bodypart"] == bp, "error"], pck_threshold)
                             for bp in _markers(project)},
        }

    return {
        "frames": frames,
        "pck_threshold": pck_threshold,
        "all_frames": all_frames,
        "n_labeled": n_labeled,
        "n_frames": int(ground_truth["image"].nunique()),
        "a": _model(a.model_id),
        "b": _model(b.model_id),
        "closer": {"all": _closer(paired),
                   "per_bodypart": {bp: _closer(paired[paired["bodypart"] == bp]) for bp in _markers(project)}},
    }


def _fmt(value: float, spec: str) -> str:
    return "-" if value != value else format(value, spec)  # NaN -> "-"


def format_model_comparison(result: dict[str, Any]) -> list[str]:
    """The lines ``fdlc compare`` prints for :func:`compare_models`."""
    a, b = result["a"], result["b"]
    if result["all_frames"]:
        lines = [f"scored on all {result['n_frames']} labeled frame(s), training frames included: "
                 "the scores flatter both models"]
    else:
        lines = [f"scored on {result['n_frames']} of {result['n_labeled']} labeled frame(s), the ones neither "
                 f"model trained on (A trained on {a['trained_on']}, B on {b['trained_on']})"]
    lines += [f"A = {a['model_id']}", f"B = {b['model_id']}",
              f"errors in {_space(result['frames'])}; PCK within {result['pck_threshold']:g} px; "
              "closer: share of labeled markers each model put nearer the label", ""]
    groups = ("median", "mean", "RMSE", "PCK", "closer")
    lines.append((f"{'':<22}" + "".join(f"{g:^15}" for g in groups)).rstrip())
    head = f"{'marker':<16}{'n':>6}" + f"{'A':>8}{'B':>7}" * len(groups)
    lines += [head, "-" * len(head)]

    def row(name, sa, sb, closer):
        cells = [(sa["median"], sb["median"], ".2f"), (sa["mean"], sb["mean"], ".2f"),
                 (sa["rmse"], sb["rmse"], ".2f"), (sa["pck"], sb["pck"], ".0%"),
                 (closer["a"], closer["b"], ".0%")]
        return f"{name:<16}{closer['n']:>6}" + "".join(
            f"{_fmt(x, spec):>8}{_fmt(y, spec):>7}" for x, y, spec in cells)

    lines.append(row("all", a["all"], b["all"], result["closer"]["all"]))
    for bp, sa in a["per_bodypart"].items():
        lines.append(row(bp, sa, b["per_bodypart"][bp], result["closer"]["per_bodypart"][bp]))
    return lines


# ---------------------------------------------------------------------- runs
def _kind_size(project, video_id: str, kind: str) -> tuple[int, int] | None:
    """``(width, height)`` of the registered ``kind`` video of ``video_id``, if known."""
    import contextlib

    from .frames import resolve_media
    from .project import _probe_video

    if not project.has_video(video_id, kind):
        return None
    rec = project.video_record(video_id, kind)
    if rec.width and rec.height:
        return rec.width, rec.height
    with contextlib.suppress(FileNotFoundError, OSError):
        width, height, _, _ = _probe_video(resolve_media(project, video_id, kind))
        if width and height:
            return width, height
    return None


def _run_videos(project, run) -> dict[str, tuple]:
    """``video_id -> (pose path, video, kind)`` for the videos of an analyze run that are registered."""
    from . import refine

    out = {}
    for video, pose in refine.run_poses(project, run):
        vid, kind = refine.match_video(project, video) if video is not None else (None, None)
        if vid is not None and vid not in out:
            out[vid] = (pose, video, kind)
    return out


def compare_runs(
    project,
    run_a,
    run_b,
    *,
    frames: str | None = None,
    pcutoff: float | None = None,
    far: float = DEFAULT_FAR,
    top: int = 5,
    poses_reader: Callable | None = None,
) -> dict[str, Any]:
    """Compare the predictions of two ``analyze`` runs on the videos they share.

    Positions are brought into the pixels of ``frames`` -- by default the kind of
    video both runs were computed on, or ``"original"`` when they differ. A marker
    is drawn by a run when its likelihood reaches the run's ``pcutoff`` (or
    ``pcutoff``, when given). ``top`` frames per video are picked where the runs
    disagree most (see the module docstring).

    ``poses_reader(path) -> long DataFrame`` defaults to reading the parquet.

    Raises:
        ValueError: if the runs are the same, the project is multi-animal, or the
            runs share no video whose size is known.
    """
    import numpy as np
    import pandas as pd

    from . import refine

    if run_a.run_id == run_b.run_id:
        raise ValueError(f"both are run {run_a.run_id}; name two different runs")
    if project.config.multi_animal:
        raise ValueError("comparing runs of a multi-animal project is not supported yet: the two runs' "
                         "individuals are not matched to each other")
    poses_reader = poses_reader or pd.read_parquet
    cut = {"a": pcutoff if pcutoff is not None else refine.run_pcutoff(run_a),
           "b": pcutoff if pcutoff is not None else refine.run_pcutoff(run_b)}
    found = {"a": _run_videos(project, run_a), "b": _run_videos(project, run_b)}
    shared = sorted(set(found["a"]) & set(found["b"]))
    only = {side: sorted(set(found[side]) - set(shared)) for side in found}
    if frames is None:
        kinds = {found[side][vid][2] for side in found for vid in shared}
        frames = kinds.pop() if len(kinds) == 1 and None not in kinds else "original"

    skipped, joined, videos = [], [], {}
    for vid in shared:
        target = _kind_size(project, vid, frames)
        sides = {}
        for side in ("a", "b"):
            pose, video, kind = found[side][vid]
            size = refine.video_size(project, vid, video, kind)
            if size is None or target is None:
                break
            df = poses_reader(pose)
            sides[side] = df.assign(x=df["x"] * target[0] / size[0], y=df["y"] * target[1] / size[1])
        if len(sides) < 2:
            skipped.append(f"{vid}: the size of a run's video or of the {frames} video is unknown")
            continue
        keys = ["frame", "individual", "bodypart"]
        both = pd.merge(sides["a"][[*keys, "x", "y", "likelihood"]], sides["b"][[*keys, "x", "y", "likelihood"]],
                        on=keys, suffixes=("_a", "_b"))
        for side in ("a", "b"):
            both[f"drawn_{side}"] = ((both[f"likelihood_{side}"] >= cut[side]) & np.isfinite(both[f"x_{side}"])
                                     & np.isfinite(both[f"y_{side}"]))
        both["distance"] = np.hypot(both["x_a"] - both["x_b"], both["y_a"] - both["y_b"])
        drawn = both["drawn_a"] & both["drawn_b"]
        both["differ"] = (both["drawn_a"] != both["drawn_b"]) | (drawn & (both["distance"] > far))
        both["distance"] = both["distance"].where(drawn)
        joined.append(both.assign(video=vid))

        per_frame = both.groupby("frame").agg(differ=("differ", "sum"), distance=("distance", "mean"))
        per_frame["distance"] = per_frame["distance"].fillna(0.0)
        candidates = per_frame[per_frame["differ"] > 0]
        picks = []
        if top > 0 and len(candidates):
            picks = refine.spread(candidates, top, best_first=[("differ", False), ("distance", False)])
        fps = project.video_record(vid, "original").fps if project.has_video(vid, "original") else None
        videos[vid] = {
            "n_frames": int(both["frame"].nunique()),
            "n_differ": int(len(candidates)),
            "frames": [{"frame": f, "time": f / fps if fps else None, "differ": int(per_frame.at[f, "differ"]),
                        "distance": float(per_frame.at[f, "distance"])} for f in picks],
        }
    if not joined:
        raise ValueError(f"runs {run_a.run_id} and {run_b.run_id} share no video to compare"
                         + ("".join(f"\n  {s}" for s in skipped) if skipped else ""))
    data = pd.concat(joined, ignore_index=True)

    def _stats(df) -> dict[str, float]:
        dist = df["distance"].dropna()
        return {
            "n": int(len(df)),
            "shown_a": float(df["drawn_a"].mean()), "shown_b": float(df["drawn_b"].mean()),
            "likelihood_a": float(df["likelihood_a"].mean()), "likelihood_b": float(df["likelihood_b"].mean()),
            "n_both": int(len(dist)),
            "median": float(dist.median()) if len(dist) else float("nan"),
            "p95": float(dist.quantile(0.95)) if len(dist) else float("nan"),
            "differ": float(df["differ"].mean()),
        }

    return {
        "runs": [run_a.run_id, run_b.run_id],
        "models": [run_a.manifest().model_id, run_b.manifest().model_id],
        "frames": frames,
        "pcutoff": [cut["a"], cut["b"]],
        "far": far,
        "only": [only["a"], only["b"]],
        "skipped": skipped,
        "all": _stats(data),
        "per_bodypart": {bp: _stats(data[data["bodypart"] == bp]) for bp in _markers(project)
                         if (data["bodypart"] == bp).any()},
        "videos": videos,
    }


def _clock(seconds: float | None) -> str:
    """``m:ss.ss``, or ``-`` when the frame rate is unknown."""
    if seconds is None:
        return "-"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}:{sec:05.2f}"


def format_run_comparison(result: dict[str, Any]) -> list[str]:
    """The lines ``fdlc compare`` prints for :func:`compare_runs`."""
    (ra, rb), (ma, mb), (pa, pb) = result["runs"], result["models"], result["pcutoff"]
    lines = [f"A = run {ra}" + (f" (model {ma})" if ma else "") + f", pcutoff {pa:g}",
             f"B = run {rb}" + (f" (model {mb})" if mb else "") + f", pcutoff {pb:g}",
             f"{len(result['videos'])} video(s) in both, {sum(v['n_frames'] for v in result['videos'].values())}"
             f" frame(s); distances in {_space(result['frames'])}, where both runs draw the marker"]
    for side, names in zip("AB", result["only"], strict=True):
        if names:
            lines.append(f"only in {side}: {', '.join(names)}")
    lines += [f"skipped {s}" for s in result["skipped"]]
    lines.append(f"drawn: share of frames the marker is drawn (likelihood at or above pcutoff); differ: share "
                 f"drawn by one run only, or by both more than {result['far']:g} px apart")
    lines.append("")
    lines.append(f"{'':<16}{'drawn':^15}{'likelihood':^15}{'distance':^15}".rstrip())
    head = f"{'marker':<16}{'A':>8}{'B':>7}{'A':>8}{'B':>7}{'median':>8}{'p95':>7}{'differ':>8}"
    lines += [head, "-" * len(head)]

    def row(name, s):
        return (f"{name:<16}{_fmt(s['shown_a'], '.0%'):>8}{_fmt(s['shown_b'], '.0%'):>7}"
                f"{_fmt(s['likelihood_a'], '.2f'):>8}{_fmt(s['likelihood_b'], '.2f'):>7}"
                f"{_fmt(s['median'], '.2f'):>8}{_fmt(s['p95'], '.1f'):>7}{_fmt(s['differ'], '.0%'):>8}")

    lines.append(row("all", result["all"]))
    lines += [row(bp, s) for bp, s in result["per_bodypart"].items()]
    for vid, info in result["videos"].items():
        lines += ["", f"{vid}: the runs differ on {info['n_differ']} of {info['n_frames']} frame(s)"]
        if info["frames"]:
            lines.append(f"  {'frame':>7}  {'time':>9}  {'differing':>9}  {'mean distance':>13}")
        for f in info["frames"]:
            lines.append(f"  {f['frame']:>7}  {_clock(f['time']):>9}  {f['differ']:>9}  {f['distance']:>13.1f}")
    return lines
