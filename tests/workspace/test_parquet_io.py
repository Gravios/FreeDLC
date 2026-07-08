#
# FreeDLC workspace layer -- parquet engine declaration guard
#
"""Guard that a parquet engine is declared whenever the workspace writes parquet.

The workspace's native format is parquet (``labels.parquet``, ``pose.parquet``).
``DataFrame.to_parquet`` needs pyarrow (or fastparquet), and that is NOT pulled in
by ``pandas[hdf5,performance]`` -- so a real inference run crashed at the final
write with "Unable to find a usable engine". The AST/smoke suites never execute
parquet I/O (pyarrow isn't importable at sandbox test time), so this text-level
guard encodes the invariant instead: if the code writes parquet, the dependency
must be declared.

Standalone: ``python tests/workspace/test_parquet_io.py`` -> ``parquet_io: N/N checks passed``.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "deeplabcut" / "workspace"


def test_workspace_actually_uses_parquet():
    hits = [p.name for p in WORKSPACE.glob("*.py")
            if "to_parquet" in p.read_text(encoding="utf-8")
            or "read_parquet" in p.read_text(encoding="utf-8")]
    assert hits, "expected parquet I/O somewhere in the workspace layer"


def test_a_parquet_engine_is_declared():
    deps = tomllib.load((ROOT / "pyproject.toml").open("rb"))["project"]["dependencies"]
    joined = " ".join(deps)
    has_engine = (
        any(d.split("[")[0].split(">")[0].split("=")[0].strip() in {"pyarrow", "fastparquet"}
            for d in deps)
        or "[parquet" in joined
        or ",parquet" in joined
    )
    assert has_engine, (
        "the workspace writes parquet but pyproject declares no engine "
        "(pyarrow / fastparquet / pandas[parquet])"
    )


def _run() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for c in checks:
        c()
    print(f"parquet_io: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
