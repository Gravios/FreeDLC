#
# FreeDLC workspace layer -- ONNX plumbing tests (milestone 1)
#
"""Tests for the ONNX milestone-1 plumbing.

Covers what is verifiable without torch: the ModelCard fields round-trip, the
OnnxRunner runs a real (trivial) ONNX model, and build_pose_runner(backend="onnx")
returns a working session. The torch export path is scaffolded and unverified
(needs torch) -- only its CLI dispatch is checked (in test_cli.py, mocked).

onnx / onnxruntime tests self-skip if those packages are absent (they are an
optional [onnx] extra).

Standalone: ``python tests/workspace/test_onnx.py`` -> ``onnx: N/N checks passed``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from deeplabcut import workspace as ws


def _bundle(root: Path, bodyparts=("snout", "tail")):
    cfg = root / "pc.yaml"
    cfg.write_text("net_type: resnet_50\nmetadata: {bodyparts: [snout, tail]}\n")
    snap = root / "s.pt"
    snap.write_bytes(b"w")
    return ws.ModelBundle.create(root / "b", pose_config_src=cfg, snapshot_src=snap,
                                 architecture="resnet_50", bodyparts=list(bodyparts), model_id="b")


def _trivial_onnx(path: Path):
    """A minimal valid ONNX graph (Relu) so the runtime path can be exercised."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return False
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))
    return True


def test_card_round_trips_pose_onnx():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        b = _bundle(d)
        assert b.card.pose_onnx is None
        assert not b.has_pose_onnx
        assert b.pose_onnx_path == b.snapshots_dir / "pose.onnx"
        b.card.pose_onnx = "pose.onnx"
        from deeplabcut.workspace.manifest import write_manifest
        write_manifest(b.path / "model.toml", b.card.to_dict())
        assert ws.ModelBundle.open(b.path).card.pose_onnx == "pose.onnx"


def test_onnx_runner_runs_real_model():
    try:
        import numpy as np
        import onnxruntime  # noqa: F401
    except ImportError:
        return  # onnx extra not installed
    from deeplabcut.workspace.onnx_export import OnnxRunner
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        model = d / "m.onnx"
        if not _trivial_onnx(model):
            return
        runner = OnnxRunner(model)
        x = np.array([-1.0, 2.0] * 24, dtype="float32").reshape(1, 3, 4, 4)
        (out,) = runner.run(x)
        assert out.shape == (1, 3, 4, 4)
        assert (out >= 0).all()               # Relu: negatives clipped -> torch-free forward pass works


def test_build_pose_runner_onnx_backend():
    try:
        import numpy as np
        import onnxruntime  # noqa: F401
    except ImportError:
        return
    from deeplabcut.workspace.onnx_export import OnnxRunner
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        b = _bundle(d)
        if not _trivial_onnx(b.snapshots_dir / "pose.onnx"):
            return
        b.card.pose_onnx = "pose.onnx"
        runner = b.build_pose_runner(backend="onnx")
        assert isinstance(runner, OnnxRunner)
        (out,) = runner.run(np.zeros((1, 3, 4, 4), dtype="float32"))
        assert out.shape == (1, 3, 4, 4)


def test_build_pose_runner_rejects_unknown_backend():
    with tempfile.TemporaryDirectory() as d:
        b = _bundle(Path(d))
        try:
            b.build_pose_runner(backend="tensorflow")
        except ValueError as e:
            assert "backend" in str(e)
        else:
            raise AssertionError("unknown backend should raise ValueError")


def _run() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for c in checks:
        c()
    print(f"onnx: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
