#
# FreeDLC -- import-time decorator validation
#
"""Statically validate every ``@renamed_parameter`` decorator in the package.

``@renamed_parameter`` validates its target's signature *at import time*: it
raises if ``new`` is not a parameter (or ``old`` still is). That means a
mismatch crashes ``import deeplabcut`` -- but it is invisible to ``compileall``
and to any test that doesn't actually import the heavy package. This is exactly
how a dangling decorator (left behind when its function was deleted) slipped
through: it slid onto the next function and only blew up on real import.

This test replicates the decorator's guards by parsing the AST, so the whole
class of bug is caught without importing torch/cv2/etc.

Standalone: ``python tests/workspace/test_decorators.py`` -> ``decorators: N/N checks passed``.
"""
from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "deeplabcut"


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    names = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args] + [p.arg for p in a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _renamed_parameter_calls(fn):
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "renamed_parameter":
            kw = {k.arg: (k.value.value if isinstance(k.value, ast.Constant) else None)
                  for k in dec.keywords}
            yield kw.get("old"), kw.get("new")


def _audit() -> list[str]:
    problems = []
    for path in PKG.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = _params(node)
            for old, new in _renamed_parameter_calls(node):
                rel = path.relative_to(PKG.parent)
                if new is not None and new not in params:
                    problems.append(f"{rel}:{node.lineno} {node.name}(): new={new!r} not in params {params}")
                if old is not None and old in params:
                    problems.append(f"{rel}:{node.lineno} {node.name}(): old={old!r} still in params")
    return problems


def test_no_dangling_renamed_parameter():
    problems = _audit()
    assert not problems, "invalid @renamed_parameter decorators (crash import):\n" + "\n".join(problems)


def test_audit_actually_found_decorators():
    # guard against the audit silently matching nothing (e.g. import name drift)
    count = sum(
        1
        for path in PKG.rglob("*.py")
        if "renamed_parameter" in path.read_text(encoding="utf-8", errors="ignore")
    )
    assert count > 0, "audit found no renamed_parameter usages at all -- check the matcher"


def _run() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for c in checks:
        c()
    print(f"decorators: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
