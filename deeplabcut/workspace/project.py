#
# FreeDLC workspace layer
#
"""The :class:`Project` -- the top-level handle for a workspace.

A project owns the ``sources/`` (immutable inputs), ``models/`` (portable
bundles), ``runs/`` (generating operations) and ``derived/`` (stable views)
trees, and its ``project.toml``. It is deliberately thin: it registers sources,
enumerates entities, and opens runs. Model assembly lives in
:mod:`~deeplabcut.workspace.model_bundle`; inference lives in
:mod:`~deeplabcut.workspace.apply`.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from . import ids
from .layout import Layout
from .manifest import read_manifest, write_manifest
from .schema import LabelsRecord, ProjectConfig, RunManifest, VideoRecord, now_iso
from .util import code_version, materialize, same_file, sha256_file

__all__ = ["Project", "Run"]


def _probe_video(path: Path) -> tuple[int | None, int | None, float | None, int | None]:
    """Return ``(width, height, fps, n_frames)`` for a video, or ``None``s if unprobeable.

    Uses OpenCV when importable. Probing is best-effort: any failure yields all
    ``None`` rather than raising, so registering a video never depends on a backend.
    """
    try:
        import cv2
    except ImportError:
        return (None, None, None, None)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return (None, None, None, None)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        return (width, height, fps, n_frames)
    finally:
        cap.release()


class Project:
    """A FreeDLC workspace rooted at a directory."""

    def __init__(self, root: str | Path, config: ProjectConfig):
        self.layout = Layout(root)
        self.config = config

    @property
    def root(self) -> Path:
        return self.layout.root

    def __repr__(self) -> str:
        return f"Project(root={self.root!r}, task={self.config.task!r})"

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        task: str,
        bodyparts: list[str],
        experimenters: Iterable[str] = (),
        multi_animal: bool = False,
        individuals: Iterable[str] = (),
        unique_bodyparts: Iterable[str] = (),
        skeleton: Iterable[list[str]] = (),
        exist_ok: bool = False,
    ) -> Project:
        """Create a new project on disk and return an open handle.

        Raises:
            FileExistsError: if ``project.toml`` already exists and not ``exist_ok``.
        """
        layout = Layout(root)
        if layout.project_toml.exists() and not exist_ok:
            raise FileExistsError(f"a project already exists at {layout.root}")
        config = ProjectConfig(
            task=task,
            bodyparts=list(bodyparts),
            experimenters=list(experimenters),
            multi_animal=multi_animal,
            individuals=list(individuals),
            unique_bodyparts=list(unique_bodyparts),
            skeleton=[list(e) for e in skeleton],
        )
        layout.create_skeleton()
        write_manifest(layout.project_toml, config.to_dict())
        return cls(layout.root, config)

    @classmethod
    def open(cls, root: str | Path) -> Project:
        """Open an existing project.

        Raises:
            FileNotFoundError: if no ``project.toml`` is found under ``root``.
        """
        layout = Layout(root)
        if not layout.project_toml.exists():
            raise FileNotFoundError(f"no project.toml under {layout.root}")
        config = ProjectConfig.from_dict(read_manifest(layout.project_toml))
        return cls(layout.root, config)

    def save_config(self) -> None:
        """Persist the in-memory :class:`ProjectConfig` back to ``project.toml``."""
        write_manifest(self.layout.project_toml, self.config.to_dict())

    # -- sources: videos --------------------------------------------------
    def add_video(
        self,
        path: str | Path,
        *,
        video_id: str | None = None,
        kind: str = "original",
        link: str = "symlink",
        hash: bool = False,
        exist_ok: bool = False,
    ) -> str:
        """Register a source video and return its ``video_id``.

        ``kind`` selects the shelf: ``"original"`` (default) for full-resolution
        reference footage, or ``"processed"`` for the downscaled video the model
        trains on. The media is materialized under
        ``sources/videos/<kind>/<video_id>/`` according to ``link``: ``"symlink"``
        (default), ``"copy"``, or ``"reference"`` (record the path only). A
        ``video.toml`` provenance record is always written, with width/height/fps
        probed when a video backend is available -- these dimensions are what the
        original/processed scale is later derived from.

        Args:
            kind: which shelf to register under (original | processed).
            link: how to materialize the media (symlink | copy | reference).
            hash: also compute and record the source SHA-256 (streams the file).
        """
        if link not in ("symlink", "copy", "reference"):
            raise ValueError(f"link must be symlink|copy|reference, got {link!r}")
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"video not found: {src}")
        vid = video_id or ids.video_id_from_path(src)
        if self.has_video(vid, kind) and not exist_ok:
            raise FileExistsError(f"{kind} video id {vid!r} already registered")

        vdir = self.layout.video_dir(vid, kind)
        media = self.layout.video_media(vid, src.suffix or ".mp4", kind)
        # Re-registering replaces whatever media the id had. Refuse when the source
        # is that media itself (a copy living in the workspace): replacing it would
        # delete the only copy and, for a symlink, leave a link pointing at itself.
        previous = self.video_media_files(vid, kind)
        if any(not p.is_symlink() and same_file(p, src) for p in previous):
            raise ValueError(f"{src} is already this project's own media for {kind} video {vid!r}")
        vdir.mkdir(parents=True, exist_ok=True)
        for stale in previous:
            if stale != media or link == "reference":
                stale.unlink()  # another extension, or media a reference must not keep
        if link != "reference":
            materialize(src, media, link)

        width, height, fps, n_frames = _probe_video(src)
        record = VideoRecord(
            video_id=vid,
            source_path=str(src),
            size_bytes=src.stat().st_size,
            sha256=sha256_file(src) if hash else None,
            link=link,
            width=width,
            height=height,
            fps=fps,
            n_frames=n_frames,
        )
        write_manifest(self.layout.video_toml(vid, kind), record.to_dict())
        return vid

    def video_media_files(self, video_id: str, kind: str = "original") -> list[Path]:
        """Media materialized under a video's directory (``video.<ext>``), sorted.

        Empty for an unregistered video and for a ``reference``, which records the
        source path without materializing it. ``video.toml``, the provenance record
        beside the media, is not media.
        """
        vdir = self.layout.video_dir(video_id, kind)
        toml = self.layout.video_toml(video_id, kind)
        return sorted(p for p in vdir.glob("video.*") if p != toml) if vdir.is_dir() else []

    def has_video(self, video_id: str, kind: str = "original") -> bool:
        return self.layout.video_toml(video_id, kind).exists()

    def videos(self, kind: str = "original") -> list[str]:
        """Registered video ids for ``kind`` (default: original), sorted."""
        d = self.layout.videos_kind_dir(kind)
        return sorted(p.name for p in d.iterdir() if (p / "video.toml").exists()) if d.exists() else []

    def video_record(self, video_id: str, kind: str = "original") -> VideoRecord:
        return VideoRecord.from_dict(read_manifest(self.layout.video_toml(video_id, kind)))

    def annotation_scale(self, video_id: str) -> tuple[float, float]:
        """Return the ``(scale_x, scale_y)`` mapping original pixels to processed pixels.

        Derived from the two registered videos' dimensions. ``(1.0, 1.0)`` when there
        is no processed counterpart, or when either video's dimensions are unknown --
        in both cases annotation coordinates are left in original space.

        The scale is anisotropic on purpose: a 240x136 reduction of 1920x1080 is
        0.125 in x but ~0.1259 in y, so a single ratio would skew y.
        """
        if not (self.has_video(video_id, "original") and self.has_video(video_id, "processed")):
            return (1.0, 1.0)
        orig = self.video_record(video_id, "original")
        proc = self.video_record(video_id, "processed")
        if not (orig.width and orig.height and proc.width and proc.height):
            return (1.0, 1.0)
        return (proc.width / orig.width, proc.height / orig.height)

    def labels_record(self, video_id: str) -> LabelsRecord:
        """The pixel space ``labels.parquet`` is in, and the scale that put it there.

        Read from ``labels.toml``, written when annotations are ingested. Labels
        ingested before that record existed have none; for those the space is
        inferred the way ingest decided it -- processed whenever a processed
        counterpart currently gives a non-identity scale, original otherwise.
        """
        path = self.layout.labels_toml(video_id)
        if path.exists():
            return LabelsRecord.from_dict(read_manifest(path))
        scale_x, scale_y = self.annotation_scale(video_id)
        space = "processed" if (scale_x, scale_y) != (1.0, 1.0) else "original"
        return LabelsRecord(video_id=video_id, space=space, scale_x=scale_x, scale_y=scale_y)

    def label_frames_kind(self, video_id: str) -> str:
        """Which frame set is in the same pixel space as ``labels.parquet``.

        Everything that reads frames *by label* without asking for a particular
        frame set (dataset export, image sizes, evaluation) must use this, not a
        fixed kind. See :meth:`labels_record`.
        """
        return self.labels_record(video_id).space

    def label_frames_dir(self, video_id: str) -> Path:
        """The frames directory matching ``labels.parquet`` (see :meth:`label_frames_kind`)."""
        return self.layout.frames_dir(video_id, self.label_frames_kind(video_id))

    def labels_scale_to(self, video_id: str, frames: str | None) -> tuple[float, float]:
        """The ``(x, y)`` factors mapping stored label coordinates onto ``frames``.

        ``frames`` is ``"original"``, ``"processed"``, or ``None`` for "whatever the
        labels are stored in" (identity). The stored space comes from
        :meth:`labels_record`; going to the other one multiplies or divides by the
        original->processed scale.

        Raises:
            ValueError: on an unknown ``frames``, or when the other space cannot be
                reached -- ``processed`` without a registered processed counterpart
                of known dimensions, or ``original`` for processed-space labels
                whose scale is unknown.
        """
        if frames is None:
            return (1.0, 1.0)
        if frames not in self.layout.VIDEO_KINDS:
            raise ValueError(f"frames must be one of {self.layout.VIDEO_KINDS}, got {frames!r}")
        record = self.labels_record(video_id)
        if frames == record.space:
            return (1.0, 1.0)
        if frames == "processed":
            if not self.has_video(video_id, "processed"):
                raise ValueError(f"{video_id}: no processed video is registered")
            orig, proc = self.video_record(video_id, "original"), self.video_record(video_id, "processed")
            if not (orig.width and orig.height and proc.width and proc.height):
                raise ValueError(f"{video_id}: original/processed video dimensions are unknown")
            return (proc.width / orig.width, proc.height / orig.height)
        if not (record.scale_x and record.scale_y):
            raise ValueError(f"{video_id}: the scale its processed-space labels were written with is unknown")
        return (1.0 / record.scale_x, 1.0 / record.scale_y)

    def check_frames(self, video_ids: Iterable[str], frames: str | None) -> None:
        """Raise one ``ValueError`` naming every video whose labels cannot be put on ``frames``."""
        problems = []
        for video_id in video_ids:
            try:
                self.labels_scale_to(video_id, frames)
            except ValueError as err:
                problems.append(str(err))
        if problems:
            other = "original" if frames == "processed" else "processed"
            raise ValueError(
                f"cannot use the {frames} frames for {len(problems)} video(s):\n  "
                + "\n  ".join(problems)
                + "\nregister the missing counterpart "
                "(`dlc-ws add-video <project> <video> --processed --video-id <id>`) "
                f"or choose `--frames {other}`"
            )

    def annotated_videos(self) -> list[str]:
        """Video ids that have ingested annotations (``labels.parquet``), sorted."""
        d = self.layout.annotations_dir
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if (p / "labels.parquet").exists())

    # -- models -----------------------------------------------------------
    def models(self) -> list[str]:
        """All model ids present under ``models/``, sorted (i.e. oldest first)."""
        d = self.layout.models_dir
        return sorted(p.name for p in d.iterdir() if (p / "model.toml").exists()) if d.exists() else []

    # -- runs -------------------------------------------------------------
    def new_run(
        self,
        kind: str,
        *,
        model_id: str | None = None,
        snapshot: str | None = None,
        inputs: Iterable[str] = (),
        params: dict | None = None,
    ) -> Run:
        """Create an isolated run directory and its initial ``run.toml``.

        Returns a :class:`Run` handle whose status starts at ``"created"``; call
        :meth:`Run.start` / :meth:`Run.finish` around the work.
        """
        run_id = ids.new_run_id()
        manifest = RunManifest(
            run_id=run_id,
            kind=kind,  # validated by RunManifest
            model_id=model_id,
            snapshot=snapshot,
            inputs=list(inputs),
            params=dict(params or {}),
            code_version=code_version(),
        )
        run = Run(self, kind, run_id)
        run.dir.mkdir(parents=True, exist_ok=True)
        write_manifest(run.manifest_path, manifest.to_dict())
        return run

    def runs(self, kind: str | None = None) -> list[Run]:
        """Existing runs, oldest first. Filter by ``kind`` if given."""
        from .schema import RUN_KINDS

        out: list[Run] = []
        for k in ((kind,) if kind else RUN_KINDS):
            kdir = self.layout.runs_kind_dir(k)
            if kdir.exists():
                for rid in sorted(p.name for p in kdir.iterdir() if (p / "run.toml").exists()):
                    out.append(Run(self, k, rid))
        return out


class Run:
    """Handle to a single run directory (``runs/<kind>/<run_id>/``)."""

    def __init__(self, project: Project, kind: str, run_id: str):
        self.project = project
        self.kind = kind
        self.run_id = run_id

    def __repr__(self) -> str:
        return f"Run(kind={self.kind!r}, run_id={self.run_id!r})"

    @property
    def dir(self) -> Path:
        return self.project.layout.run_dir(self.kind, self.run_id)

    @property
    def manifest_path(self) -> Path:
        return self.project.layout.run_toml(self.kind, self.run_id)

    def manifest(self) -> RunManifest:
        return RunManifest.from_dict(read_manifest(self.manifest_path))

    def video_dir(self, video_id: str) -> Path:
        d = self.project.layout.run_video_dir(self.kind, self.run_id, video_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _patch(self, **changes) -> None:
        data = read_manifest(self.manifest_path)
        data.update({k: v for k, v in changes.items() if v is not None})
        write_manifest(self.manifest_path, data)

    def start(self) -> Run:
        self._patch(status="running", started=now_iso())
        return self

    def finish(self, outputs: Iterable[str] = ()) -> Run:
        self._patch(status="finished", finished=now_iso(), outputs=list(outputs))
        return self

    def fail(self) -> Run:
        self._patch(status="failed", finished=now_iso())
        return self
