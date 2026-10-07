#
# FreeDLC workspace layer -- CLI tests
#
"""Tests for deeplabcut.workspace.cli.

The no-torch commands (migrate/info/models/videos) run for real; the torch-backed
commands (apply/train/evaluate) are checked for correct parsing and dispatch by
patching the workspace function each one calls. Nothing here imports torch.

Standalone: ``python tests/workspace/test_cli.py`` -> ``cli: N/N checks passed``.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import yaml

from deeplabcut import workspace as ws
from deeplabcut.workspace import cli


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _legacy_project(root: Path) -> Path:
    legacy = root / "legacy"
    (legacy / "videos").mkdir(parents=True)
    (legacy / "videos" / "clip1.mp4").write_bytes(b"v")
    train = legacy / "dlc-models-pytorch" / "iteration-0" / "reachJul7-trainset95shuffle1" / "train"
    train.mkdir(parents=True)
    with (train / "pytorch_config.yaml").open("w") as fh:
        yaml.safe_dump({"net_type": "resnet_50", "metadata": {"bodyparts": ["snout", "paw"]}}, fh)
    (train / "snapshot-best-100.pt").write_bytes(b"w")
    with (legacy / "config.yaml").open("w") as fh:
        yaml.safe_dump({"Task": "reach", "scorer": "gravio", "multianimalproject": False,
                        "bodyparts": ["snout", "paw"], "uniquebodyparts": [], "skeleton": [],
                        "video_sets": {str((legacy / "videos" / "clip1.mp4").resolve()): {}}}, fh)
    return legacy


def _model_project(root: Path):
    proj = ws.Project.create(root / "ws", task="reach", bodyparts=["snout", "paw"])
    cfg = root / "pytorch_config.yaml"
    with cfg.open("w") as fh:
        yaml.safe_dump({"net_type": "resnet_50", "metadata": {"bodyparts": ["snout", "paw"]}}, fh)
    snap = root / "snapshot-050.pt"
    snap.write_bytes(b"w")
    ws.ModelBundle.create(proj.layout.model_dir("m1"), pose_config_src=cfg, snapshot_src=snap,
                          architecture="resnet_50", bodyparts=["snout", "paw"], model_id="m1")
    return proj


# --------------------------------------------------------- no-torch commands
def test_create_command():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ws"
        code, out = _run(["create", str(root), "--task", "reach",
                          "--bodyparts", "snout", "paw", "--experimenters", "gravio"])
        assert code == 0 and "created ->" in out
        assert "bodyparts: 2" in out and "skeleton: 0 edge(s)" in out
        proj = ws.Project.open(root)
        assert proj.config.task == "reach"
        assert proj.config.bodyparts == ["snout", "paw"]
        assert proj.config.experimenters == ["gravio"]
        assert not proj.config.multi_animal
        for rel in ("sources/videos/original", "sources/videos/processed",
                    "sources/annotations", "models", "runs", "derived"):
            assert (root / rel).is_dir()          # full skeleton, not just project.toml


def test_create_with_skeleton():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ws"
        # edges accepted both as several values to one flag and as a repeated flag
        code, out = _run(["create", str(root), "--task", "reach",
                          "--bodyparts", "snout", "paw", "tailbase",
                          "--skeleton", "snout,paw", "paw,tailbase",
                          "--skeleton", "snout,tailbase"])
        assert code == 0 and "skeleton: 3 edge(s)" in out
        assert ws.Project.open(root).config.skeleton == [
            ["snout", "paw"], ["paw", "tailbase"], ["snout", "tailbase"],
        ]


def test_create_skeleton_may_reference_unique_bodyparts():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ws"
        code, _ = _run(["create", str(root), "--task", "reach", "--bodyparts", "snout", "paw",
                        "--unique-bodyparts", "corner1", "--skeleton", "snout,corner1"])
        assert code == 0
        assert ws.Project.open(root).config.skeleton == [["snout", "corner1"]]


def test_create_rejects_undeclared_skeleton_bodypart():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ws"
        code, out = _run(["create", str(root), "--task", "reach", "--bodyparts", "snout", "paw",
                          "--skeleton", "snout,tialbase"])
        assert code == 2 and "undeclared bodypart(s): tialbase" in out
        assert not (root / "project.toml").exists()   # nothing written on a rejected edge


def test_create_rejects_malformed_skeleton_edge():
    with tempfile.TemporaryDirectory() as d:
        for token in ("snout", "snout,paw,tailbase", "snout,"):
            code, out = _run(["create", str(Path(d) / token.replace(",", "_")), "--task", "reach",
                              "--bodyparts", "snout", "paw", "tailbase", "--skeleton", token])
            assert code == 2 and "separated by a comma" in out


def test_create_rejects_duplicate_bodyparts():
    with tempfile.TemporaryDirectory() as d:
        code, out = _run(["create", str(Path(d) / "ws"), "--task", "reach",
                          "--bodyparts", "snout", "snout"])
        assert code == 2 and "duplicates" in out       # ProjectConfig's error, not a traceback


def test_create_individuals_require_multi_animal():
    with tempfile.TemporaryDirectory() as d:
        code, out = _run(["create", str(Path(d) / "ws"), "--task", "reach",
                          "--bodyparts", "snout", "--individuals", "mouse1"])
        assert code == 2 and "--individuals requires --multi-animal" in out


def test_create_multi_animal():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ws"
        code, out = _run(["create", str(root), "--task", "reach", "--bodyparts", "snout", "paw",
                          "--multi-animal", "--individuals", "mouse1", "mouse2"])
        assert code == 0 and "individuals: mouse1, mouse2" in out
        cfg = ws.Project.open(root).config
        assert cfg.multi_animal and cfg.individuals == ["mouse1", "mouse2"]


def test_create_existing_project():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ws"
        base = ["create", str(root), "--task", "reach", "--bodyparts", "snout", "paw"]
        assert _run(base)[0] == 0
        code, out = _run(base)
        assert code == 2 and "already exists" in out
        # --exist-ok rewrites project.toml in place
        code, _ = _run(["create", str(root), "--task", "grasp", "--bodyparts", "snout", "--exist-ok"])
        assert code == 0 and ws.Project.open(root).config.task == "grasp"


def test_migrate_command():
    with tempfile.TemporaryDirectory() as d:
        legacy = _legacy_project(Path(d))
        code, out = _run(["migrate", str(legacy), str(Path(d) / "ws"), "--no-annotations"])
        assert code == 0 and "migrated ->" in out
        proj = ws.Project.open(Path(d) / "ws")
        assert proj.videos() == ["clip1"] and len(proj.models()) == 1


def test_info_command():
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        code, out = _run(["info", str(proj.root)])
        assert code == 0
        assert "task:          reach" in out and "models:        1" in out


def test_models_command():
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        code, out = _run(["models", str(proj.root)])
        assert code == 0 and "m1" in out and "resnet_50" in out and "bottom-up" in out


def test_videos_command():
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        src = Path(d) / "clip1.mp4"
        src.write_bytes(b"v")
        proj.add_video(src)
        code, out = _run(["videos", str(proj.root)])
        assert code == 0 and "clip1" in out


def test_videos_table_shows_pairing_labels_and_problems():
    from deeplabcut.workspace.manifest import write_manifest
    from deeplabcut.workspace.schema import LabelsRecord, VideoRecord

    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout"])
        lay = proj.layout

        def register(vid, kind, w, h):
            rec = VideoRecord(video_id=vid, source_path=f"{kind}.mp4", width=w, height=h, link="reference")
            write_manifest(lay.video_toml(vid, kind), rec.to_dict())

        def label(vid):
            lay.annotation_dir(vid).mkdir(parents=True, exist_ok=True)
            lay.labels_parquet(vid).write_bytes(b"")

        register("paired", "original", 1920, 1080)
        register("paired", "processed", 192, 108)
        label("paired")
        write_manifest(lay.labels_toml("paired"),
                       LabelsRecord(video_id="paired", space="processed", scale_x=0.1, scale_y=0.1).to_dict())
        fdir = lay.frames_dir("paired", "original")
        fdir.mkdir(parents=True)
        (fdir / "img1.png").write_bytes(b"px")
        (fdir / "img2.png").symlink_to(fdir / "img2.png")           # a link to itself

        register("small-192x108", "processed", 192, 108)             # registered under its own id
        label("small-192x108")                                       # ...with labels, but no original

        code, out = _run(["videos", str(proj.root)])
        assert code == 0
        rows = {line.split()[0]: line.split() for line in out.splitlines() if not line.startswith("  !")}
        assert rows["paired"][1:] == ["1920x1080", "192x108", "processed", "px", "1/0"]
        assert rows["small-192x108"][1:] == ["-", "192x108", "original", "px?", "0/0"]   # "?": inferred
        assert "! paired: 1 original frame(s) are broken links" in out
        assert "! small-192x108: has labels but no original video is registered under this id" in out
        assert "! small-192x108: processed video has no original with the same id" in out

        # training on it fails with a message, not a traceback
        code, out = _run(["train", str(proj.root), "--epochs", "1"])
        assert code == 2
        assert "small-192x108: its labels are in original pixels and no original video is registered" in out
        assert "dlc-ws videos" in out
        assert proj.runs("train") == []


def _png_header(width: int, height: int) -> bytes:
    return (b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR"
            + width.to_bytes(4, "big") + height.to_bytes(4, "big"))


def test_videos_register_pairs_a_folder_with_the_originals():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        for src in _make_videos(d / "original", ["Session_1.mp4", "Session_1_b.mp4", "Session_2.mp4"]):
            proj.add_video(src)
        small = d / "reduced-640x360"
        a, b, labeled = _make_videos(
            small, ["Session_1_640x360.mp4", "Session_1_b-triplet.mp4", "Session_1_640x360.fdlc.mp4"])
        (small / "notes.txt").write_text("not a video")

        code, out = _run(["videos", str(proj.root), "--register", str(small)])
        assert code == 0, out
        assert proj.videos("processed") == ["session-1", "session-1-b"]      # longest id wins for "_b"
        media = {v: proj.video_media_files(v, "processed")[0] for v in proj.videos("processed")}
        assert media["session-1"].resolve() == a.resolve() and media["session-1"].is_symlink()
        assert media["session-1-b"].resolve() == b.resolve()
        assert "registered 2 processed video(s)" in out and "ignored 1 labeled .fdlc video(s)" in out
        assert "1 original(s) have no video there and keep what they had (e.g. session-2)" in out
        assert "--match-original" not in out                                  # no frames extracted yet
        assert "video id" in out                                               # the table follows
        stale = proj.layout.frames_dir("session-1", "processed")
        stale.mkdir(parents=True)
        (stale / "img1.png").write_bytes(_png_header(640, 360))

        # another folder takes over; the files of the first are left alone
        other = d / "triplet"
        (c,) = _make_videos(other, ["Session_1-triplet.mp4"])
        code, out = _run(["videos", str(proj.root), "--register", str(other), "--link", "copy"])
        assert code == 0, out
        assert "re-read them with `dlc-ws extract --all --match-original`" in out
        now = proj.video_media_files("session-1", "processed")
        assert len(now) == 1 and not now[0].is_symlink() and now[0].read_bytes() == c.read_bytes()
        assert a.read_bytes() == b"fake video"


def test_videos_register_changes_nothing_unless_the_whole_folder_matches():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (src,) = _make_videos(d / "original", ["Session_1.mp4"])
        proj.add_video(src)

        mixed = d / "mixed"
        _make_videos(mixed, ["Session_1_640x360.mp4", "Stranger.mp4"])
        code, out = _run(["videos", str(proj.root), "--register", str(mixed)])
        assert code == 2 and "nothing was changed" in out
        assert "Stranger.mp4: no original video matches this name" in out
        assert proj.videos("processed") == []

        twice = d / "twice"
        _make_videos(twice, ["Session_1_640x360.mp4", "Session_1_320x180.mp4"])
        code, out = _run(["videos", str(proj.root), "--register", str(twice)])
        assert code == 2 and "'session-1' is already matched by Session_1_320x180.mp4" in out
        assert proj.videos("processed") == []

        code, out = _run(["videos", str(proj.root), "--register", str(d / "missing")])
        assert code == 2 and "not a directory" in out
        (d / "empty").mkdir()
        code, out = _run(["videos", str(proj.root), "--register", str(d / "empty")])
        assert code == 2 and "no video files found" in out


def test_videos_notes_processed_frames_and_counts_that_no_longer_fit():
    from deeplabcut.workspace.manifest import write_manifest
    from deeplabcut.workspace.schema import VideoRecord

    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout"])
        lay = proj.layout

        def register(vid, kind, w, h, n):
            rec = VideoRecord(video_id=vid, source_path=f"{kind}.mp4", width=w, height=h, n_frames=n,
                              link="reference")
            write_manifest(lay.video_toml(vid, kind), rec.to_dict())

        def frame(vid, w, h):
            fdir = lay.frames_dir(vid, "processed")
            fdir.mkdir(parents=True)
            (fdir / "img1.png").write_bytes(_png_header(w, h))

        for vid, n_processed, frame_size in (("fits", 100, (640, 360)), ("stale", 100, (192, 108)),
                                             ("short", 90, (640, 360))):
            register(vid, "original", 1920, 1080, 100)
            register(vid, "processed", 640, 360, n_processed)
            frame(vid, *frame_size)

        code, out = _run(["videos", str(proj.root)])
        assert code == 0
        notes = [line for line in out.splitlines() if line.startswith("  !")]
        assert len(notes) == 2, out
        assert ("! stale: processed frames are 192x108 but the processed video is 640x360 "
                "(run `dlc-ws extract stale --match-original`)") in out
        assert "! short: processed video has 90 frames, the original 100" in out


def test_no_command_prints_help():
    code, _ = _run([])
    assert code == 2


# -------------------------------------------------- torch commands (dispatch)
def test_apply_dispatch_project(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        seen = {}

        def fake_apply_videos(bundle, videos, out_root, **kw):
            seen["videos"] = [Path(v) for v in videos]
            seen["batch_size"] = kw.get("batch_size")
            return {str(v): Path(out_root) / "pose.parquet" for v in videos}

        monkeypatch.setattr(cli, "apply_to_videos", fake_apply_videos)
        v1, v2 = Path(d) / "clip1.mp4", Path(d) / "clip2.mp4"
        v1.write_bytes(b"v")
        v2.write_bytes(b"v")
        code, out = _run(["apply", str(v1), str(v2), "--project", str(proj.root),
                          "--model-id", "m1", "--batch-size", "4"])
        assert code == 0
        assert len(seen["videos"]) == 2 and seen["batch_size"] == 4  # both videos, batch flag passed
        assert len(proj.runs("analyze")) == 1                        # one analyze run for the batch
        assert proj.runs("analyze")[0].manifest().status == "finished"


def test_apply_dispatch_dropin_model(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))            # creates a bundle dir at models/m1
        model_dir = proj.layout.model_dir("m1")
        seen = {}

        def fake_apply_videos(bundle, videos, out_root, **kw):
            seen["out_root"] = Path(out_root)
            return {str(v): Path(out_root) / "pose.parquet" for v in videos}

        monkeypatch.setattr(cli, "apply_to_videos", fake_apply_videos)
        v1 = Path(d) / "clip1.mp4"
        v1.write_bytes(b"v")
        code, out = _run(["apply", str(v1), "--model", str(model_dir), "--out", str(Path(d) / "preds")])
        assert code == 0 and "pose.parquet" in out
        assert seen["out_root"] == Path(d) / "preds"
        assert len(proj.runs("analyze")) == 0     # drop-in mode opens no project run


def test_train_dispatch(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        seen = {}

        class _FakeBundle:
            class card:  # noqa: N801
                model_id = "trained1"

        def fake_train(project, config, backend, **kw):
            seen["net"] = config.net_type
            seen["epochs"] = config.epochs
            seen["backend"] = type(backend).__name__
            return _FakeBundle()

        monkeypatch.setattr(cli, "train_model", fake_train)
        code, out = _run(["train", str(proj.root), "--net", "hrnet_w32", "--epochs", "3"])
        assert code == 0 and "trained -> models/trained1" in out
        assert seen == {"net": "hrnet_w32", "epochs": 3, "backend": "WorkspaceTrainBackend"}


def test_evaluate_dispatch(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        proj = _model_project(Path(d))
        captured = {}

        def fake_eval(project, bundle, **kw):
            captured.update(kw)
            return {"n": 2, "mean_error": 2.5}

        monkeypatch.setattr(cli, "evaluate_model", fake_eval)
        code, out = _run(["evaluate", str(proj.root), "m1", "--pck", "5"])
        assert code == 0 and '"mean_error": 2.5' in out
        assert captured["pck_threshold"] == 5.0 and captured["pcutoff"] == 0.6


# ------------------------------------------------------------------ smoke runner
def test_label_dispatch_reads_sidecar(monkeypatch):
    from deeplabcut.workspace import label_video
    from deeplabcut.workspace.manifest import write_manifest
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        v = d / "clip.mp4"
        v.write_bytes(b"v")
        (d / "clip.fdlc.parquet").write_bytes(b"p")   # existence only; render is patched
        write_manifest(d / "clip.fdlc.toml",
                       {"bodyparts": ["snout", "tail"], "skeleton": [["snout", "tail"]]})
        seen = {}

        def fake_render(video, parquet, out_path, *, bodyparts, skeleton, pcutoff, **kw):
            seen.update(parquet=Path(parquet).name, out=Path(out_path).name,
                        bodyparts=bodyparts, skeleton=[list(e) for e in (skeleton or [])])
            Path(out_path).write_bytes(b"mp4")
            return Path(out_path)

        monkeypatch.setattr(label_video, "render_labeled_from_parquet", fake_render)
        code, out = _run(["label", str(v)])
        assert code == 0
        assert seen["parquet"] == "clip.fdlc.parquet"        # defaulted beside the video
        assert seen["out"] == "clip.fdlc.mp4"
        assert seen["bodyparts"] == ["snout", "tail"]        # from the .fdlc.toml sidecar
        assert seen["skeleton"] == [["snout", "tail"]]
        assert str(d / "clip.fdlc.parquet") not in out       # quiet by default

        code, out = _run(["label", str(v), "--verbose"])     # ...and names every file when asked
        assert code == 0
        for needed in (str(v), str(d / "clip.fdlc.parquet"), str(d / "clip.fdlc.toml"), str(d / "clip.fdlc.mp4")):
            assert needed in out, f"{needed!r} missing from:\n{out}"
        parser = cli.build_parser()
        assert parser.parse_args(["train", "x", "-v"]).verbose
        assert parser.parse_args(["apply", "--model", "m", "x", "-v"]).verbose
        assert not parser.parse_args(["label", "x"]).verbose


def test_label_missing_parquet():
    with tempfile.TemporaryDirectory() as d:
        v = Path(d) / "clip.mp4"
        v.write_bytes(b"v")
        code, out = _run(["label", str(v)])
        assert code == 2 and "no pose parquet" in out       # clear error when the parquet is absent


def test_track_dispatch(monkeypatch):
    from deeplabcut.workspace import track as track_mod
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pq = d / "clip.fdlc.parquet"
        pq.write_bytes(b"p")
        seen = {}

        def fake_track_parquet(parquet, out_path, **kw):
            seen.update(out=Path(out_path).name, kw=kw)
            Path(out_path).write_bytes(b"t")
            return Path(out_path), 3

        monkeypatch.setattr(track_mod, "track_parquet", fake_track_parquet)
        code, out = _run(["track", str(pq), "--max-distance", "30"])
        assert code == 0
        assert seen["out"] == "clip.tracked.fdlc.parquet"      # default tracked name
        assert seen["kw"]["max_distance"] == 30.0
        assert "3 tracks" in out


def test_export_dispatch(monkeypatch):
    from deeplabcut.workspace import onnx_export
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _model_project(d)
        bundle_dir = proj.layout.model_dir("m1")

        called = {}

        def fake_export(bundle, out_path, *, opset, dynamic):
            called["opset"] = opset
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"onnx")
            return Path(out_path)

        monkeypatch.setattr(onnx_export, "export_pose_onnx", fake_export)
        code, out = _run(["export", str(bundle_dir), "--opset", "18"])
        assert code == 0 and "exported" in out
        assert called["opset"] == 18
        assert ws.ModelBundle.open(bundle_dir).card.pose_onnx == "pose.onnx"   # card recorded


def test_export_check_dispatch(monkeypatch):
    from deeplabcut.workspace import onnx_export
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = _model_project(d)
        bundle_dir = proj.layout.model_dir("m1")

        def fake_export(bundle, out_path, **kw):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"onnx")
            return Path(out_path)

        monkeypatch.setattr(onnx_export, "export_pose_onnx", fake_export)

        # PASS -> exports, exit 0
        monkeypatch.setattr(onnx_export, "check_onnx_parity", lambda bundle, **kw: {
            "ok": True,
            "reports": {"batch=1": {"ok": True,
                                    "rows": [{"name": "bp.heatmap", "max_diff": 1e-7, "passed": True}]}},
        })
        code, out = _run(["export", str(bundle_dir), "--check"])
        assert code == 0 and "PARITY: PASS" in out and "exported" in out

        # FAIL -> no export, exit 1
        monkeypatch.setattr(onnx_export, "check_onnx_parity", lambda bundle, **kw: {
            "ok": False,
            "reports": {"batch=1": {"ok": False,
                                    "rows": [{"name": "bp.heatmap", "max_diff": 0.5, "passed": False}]}},
        })
        code, out = _run(["export", str(bundle_dir), "--check"])
        assert code == 1 and "PARITY: FAIL" in out


# --------------------------------------------------------------- add-video
def test_train_refuses_a_frame_set_the_labels_cannot_reach():
    """`--frames processed` is the default; without a processed video it says so, opens no run."""
    with tempfile.TemporaryDirectory() as d:
        proj = ws.Project.create(Path(d) / "ws", task="reach", bodyparts=["snout", "paw"])
        proj.add_video(_make_videos(Path(d) / "raw", ["clip.mp4"])[0], link="reference")
        (proj.layout.annotation_dir("clip")).mkdir(parents=True)
        proj.layout.labels_parquet("clip").write_bytes(b"")     # marks the video as annotated
        code, out = _run(["train", str(Path(d) / "ws"), "--epochs", "1"])
        assert code == 2
        assert "clip: no processed video is registered" in out and "--frames original" in out
        assert proj.runs("train") == []
        assert cli.build_parser().parse_args(["train", "x"]).frames == "processed"
        assert cli.build_parser().parse_args(["evaluate", "x", "m"]).frames is None


def _make_videos(root: Path, names) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in names:
        p = root / n
        p.write_bytes(b"fake video")
        paths.append(p)
    return paths


def test_add_single_video():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (vid,) = _make_videos(d / "raw", ["Session 01.mp4"])
        code, out = _run(["add-video", str(d / "ws"), str(vid)])
        assert code == 0, out
        assert "added 1 original video(s)" in out and "session-01" in out
        assert ws.Project.open(d / "ws").videos() == ["session-01"]


def test_add_video_directory():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        _make_videos(d / "raw", ["a.mp4", "b.avi", "c.mov", "notes.txt"])
        code, out = _run(["add-video", str(d / "ws"), str(d / "raw")])
        assert code == 0, out
        assert "added 3 original video(s)" in out                # the .txt is ignored
        assert ws.Project.open(d / "ws").videos() == ["a", "b", "c"]


def test_add_video_default_link_is_symlink():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (vid,) = _make_videos(d / "raw", ["clip.mp4"])
        assert _run(["add-video", str(d / "ws"), str(vid)])[0] == 0
        media = proj.layout.video_media("clip", ".mp4")
        assert media.is_symlink() and media.resolve() == vid.resolve()


def test_add_video_copy_link():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proj = ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (vid,) = _make_videos(d / "raw", ["clip.mp4"])
        assert _run(["add-video", str(d / "ws"), str(vid), "--link", "copy"])[0] == 0
        media = proj.layout.video_media("clip", ".mp4")
        assert media.is_file() and not media.is_symlink()


def test_add_video_rejects_slug_collision_within_batch():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        # 'clip.mp4' and 'clip.avi' both slugify to 'clip'
        _make_videos(d / "raw", ["clip.mp4", "clip.avi"])
        code, out = _run(["add-video", str(d / "ws"), str(d / "raw")])
        assert code == 2 and "conflicting video ids" in out and "also derived from" in out
        assert ws.Project.open(d / "ws").videos() == []      # nothing added on conflict


def test_add_video_rejects_collision_across_folders():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (a,) = _make_videos(d / "one", ["clip.mp4"])
        (b,) = _make_videos(d / "two", ["clip.mp4"])
        code, out = _run(["add-video", str(d / "ws"), str(a), str(b)])
        assert code == 2 and "also derived from" in out
        assert ws.Project.open(d / "ws").videos() == []


def test_add_video_rejects_id_already_registered():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (vid,) = _make_videos(d / "raw", ["clip.mp4"])
        assert _run(["add-video", str(d / "ws"), str(vid)])[0] == 0
        code, out = _run(["add-video", str(d / "ws"), str(vid)])
        assert code == 2 and "already registered" in out


def test_add_video_exist_ok_re_registers():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (vid,) = _make_videos(d / "raw", ["clip.mp4"])
        assert _run(["add-video", str(d / "ws"), str(vid)])[0] == 0
        code, out = _run(["add-video", str(d / "ws"), str(vid), "--exist-ok"])
        assert code == 0 and "added 1 original video(s)" in out


def test_add_video_explicit_id_single():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (vid,) = _make_videos(d / "raw", ["Session 01.mp4"])
        code, out = _run(["add-video", str(d / "ws"), str(vid), "--video-id", "trial-a"])
        assert code == 0 and "trial-a" in out
        assert ws.Project.open(d / "ws").videos() == ["trial-a"]


def test_add_video_id_rejected_for_multiple():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        _make_videos(d / "raw", ["a.mp4", "b.mp4"])
        code, out = _run(["add-video", str(d / "ws"), str(d / "raw"), "--video-id", "x"])
        assert code == 2 and "single video" in out
        assert ws.Project.open(d / "ws").videos() == []


def test_add_video_no_videos_found():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ws.Project.create(d / "ws", task="reach", bodyparts=["snout"])
        (d / "empty").mkdir()
        code, out = _run(["add-video", str(d / "ws"), str(d / "empty")])
        assert code == 2 and "no video files found" in out


def _run_smoke() -> int:
    class _MP:
        def setattr(self, obj, name, val):
            self._saved = getattr(self, "_saved", [])
            self._saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, val in reversed(getattr(self, "_saved", [])):
                setattr(obj, name, val)

    import inspect

    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    for chk in checks:
        if "monkeypatch" in inspect.signature(chk).parameters:
            mp = _MP()
            try:
                chk(mp)
            finally:
                mp.undo()
        else:
            chk()
    print(f"cli: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
