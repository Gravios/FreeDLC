#
# FreeDLC workspace layer
#
"""Convert a project to the current on-disk layout (``fdlc upgrade``).

Projects made before kept every registered video in a folder of its own::

    sources/videos/<kind>/<video_id>/video.<ext>
    sources/videos/<kind>/<video_id>/video.toml

Now a video's files sit side by side in its kind's folder, named by its id::

    sources/videos/<kind>/<video_id>.<ext>
    sources/videos/<kind>/<video_id>.toml

:func:`plan_upgrade` lists the moves and anything in the way; :func:`apply_upgrade`
makes them. Nothing is moved while anything is in the way, a target is never
overwritten, and the source videos a symlink points at are never touched: a link
is moved as a link (one with a relative target is re-made pointing at the same
file). An emptied folder is removed; a folder still holding other files is left,
and named.

The annotations, models and runs are not changed. Runs of ``apply`` made before
kept a folder per video too; every command still reads those (see
:mod:`.apply`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["UpgradePlan", "plan_upgrade", "apply_upgrade"]


@dataclass
class UpgradePlan:
    """The moves :func:`apply_upgrade` makes, and what stops them."""

    moves: list[tuple[Path, Path]] = field(default_factory=list)  # (from, to)
    folders: list[Path] = field(default_factory=list)              # the video folders emptied
    problems: list[str] = field(default_factory=list)              # anything in the way

    @property
    def n_videos(self) -> int:
        return len(self.folders)


def plan_upgrade(project) -> UpgradePlan:
    """What converting ``project``'s videos to the current layout takes."""
    lay = project.layout
    plan = UpgradePlan()
    taken: set[Path] = set()
    for kind in lay.VIDEO_KINDS:
        for folder in lay.legacy_video_dirs(kind):
            vid = folder.name
            if "." in vid:
                plan.problems.append(f"{folder}: the id {vid!r} holds a '.', so it cannot name the video's "
                                     "files; re-register the video under another id")
                continue
            plan.folders.append(folder)
            for item in sorted(folder.iterdir()):
                if item.name == lay.LEGACY_VIDEO_TOML:
                    target = lay.video_toml(vid, kind)
                elif item.name.startswith("video."):
                    target = lay.video_media(vid, item.name[len("video"):], kind)
                else:
                    plan.problems.append(f"{item}: not a video file of this folder; move or remove it first")
                    continue
                if target.exists() or target.is_symlink() or target in taken:
                    plan.problems.append(f"{item}: {target} already exists")
                    continue
                taken.add(target)
                plan.moves.append((item, target))
    return plan


def _move(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst``; a symlink with a relative target is re-made to point at the same file."""
    if src.is_symlink() and not os.path.isabs(os.readlink(src)):
        target = os.path.normpath(os.path.join(src.parent, os.readlink(src)))
        os.symlink(os.path.relpath(target, dst.parent), dst)
        src.unlink()
        return
    os.rename(src, dst)  # same file system: a rename, never a copy of the video


def apply_upgrade(plan: UpgradePlan) -> list[Path]:
    """Make the moves of ``plan``; return the folders left because they still hold files.

    Raises:
        ValueError: if the plan has problems (nothing is moved then).
    """
    if plan.problems:
        raise ValueError("cannot upgrade:\n  " + "\n  ".join(plan.problems))
    for src, dst in plan.moves:
        _move(src, dst)
    left = []
    for folder in plan.folders:
        try:
            folder.rmdir()
        except OSError:
            left.append(folder)
    return left
