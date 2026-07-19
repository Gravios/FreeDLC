#
# FreeDLC workspace layer -- ONNX export + runtime + parity check
#
"""Export bundle pose models to ONNX, run them with onnxruntime, and check parity.

What is verifiable without torch (and tested):
  - ``_flatten_outputs``: PoseModel's nested head-output dict -> ordered leaves.
  - ``_parity_report``: element-wise torch-vs-onnx comparison (pure numpy).
  - ``OnnxRunner``: onnxruntime session wrapper (the torch-free forward pass).

Torch-coupled and UNVERIFIED here (needs torch; scaffolded against DLC's real
builders and forward shape, see docs/onnx_export.md):
  - ``export_pose_onnx``: builds the model (``PoseModel.build`` + load_state_dict),
    wraps it so ``torch.onnx.export`` sees a flat tuple of head tensors, and
    traces it.
  - ``check_onnx_parity``: exports, then asserts the onnxruntime forward matches
    the torch forward tensor-for-tensor. This is the gate `dlc-ws export --check`
    runs on a torch box.

torch / onnxruntime are imported lazily, so importing this module stays light.
"""
from __future__ import annotations

from pathlib import Path


def _flatten_outputs(out: dict) -> list[tuple[str, object]]:
    """Nested head-output dict -> ``[(name, value)]`` leaves in stable sorted order.

    ``{"bp": {"heatmap": H, "locref": L}}`` -> ``[("bp.heatmap", H), ("bp.locref", L)]``.
    Pure; no torch.
    """
    leaves: list[tuple[str, object]] = []
    for head in sorted(out):
        sub = out[head]
        if isinstance(sub, dict):
            for key in sorted(sub):
                leaves.append((f"{head}.{key}", sub[key]))
        else:
            leaves.append((head, sub))
    return leaves


def _parity_report(names, ref, test, *, atol: float = 1e-3, rtol: float = 1e-3) -> dict:
    """Element-wise comparison of two ordered sequences of arrays. Pure numpy."""
    import numpy as np

    rows = []
    ok = True
    for name, r, t in zip(names, ref, test, strict=True):
        r, t = np.asarray(r), np.asarray(t)
        if r.shape != t.shape:
            rows.append({"name": name, "max_diff": float("inf"), "passed": False})
            ok = False
            continue
        max_diff = float(np.abs(r - t).max()) if r.size else 0.0
        passed = bool(np.allclose(r, t, atol=atol, rtol=rtol))
        ok = ok and passed
        rows.append({"name": name, "max_diff": max_diff, "passed": passed})
    return {"ok": ok, "rows": rows}


def _build_pose_model(bundle):
    """Build + load the bundle's default pose model. Requires torch."""
    import torch

    from deeplabcut.pose_estimation_pytorch.models import PoseModel

    cfg = bundle._read_pose_config()
    model = PoseModel.build(cfg["model"])
    snap = torch.load(str(bundle.snapshot_path("default")), map_location="cpu", weights_only=False)
    model.load_state_dict(snap["model"])
    model.eval()
    return model, cfg


def _export_wrapper(model):
    """Wrap a PoseModel so its forward returns a flat tuple of head tensors."""
    import torch
    from torch import nn

    class ExportWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
            self.names: list[str] = []

        def forward(self, x):
            leaves = _flatten_outputs(self.model(x))
            keep = [(n, v) for n, v in leaves if torch.is_tensor(v)]
            self.names = [n for n, _ in keep]
            return tuple(v for _, v in keep)

    return ExportWrapper(model)


def _crop_hw(cfg) -> tuple[int, int]:
    crop = cfg["data"]["inference"].get("top_down_crop") or {"height": 256, "width": 256}
    return int(crop["height"]), int(crop["width"])


def export_pose_onnx(bundle, out_path: str | Path, *, opset: int = 17, dynamic: bool = True) -> Path:
    """Trace the bundle's default pose model to ONNX. Requires torch; UNVERIFIED here."""
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = _build_pose_model(bundle)
    wrapper = _export_wrapper(model).eval()
    h, w = _crop_hw(cfg)
    dummy = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        wrapper(dummy)                                   # populate output names
    names = wrapper.names
    axes = {"input": {0: "batch"}}
    axes.update({n: {0: "batch"} for n in names})
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["input"], output_names=names,
        opset_version=opset, dynamic_axes=axes if dynamic else None,
    )
    return out_path


def check_onnx_parity(bundle, *, opset: int = 17, atol: float = 1e-3, rtol: float = 1e-3,
                      batch: int = 8) -> dict:
    """Export, then assert onnxruntime matches torch on the same input. Requires torch.

    Returns ``{"ok": bool, "reports": {label: parity_report}}``. UNVERIFIED here --
    this is the gate that runs on a torch box (``dlc-ws export --check``).
    """
    import tempfile

    import torch

    model, cfg = _build_pose_model(bundle)
    wrapper = _export_wrapper(model).eval()
    h, w = _crop_hw(cfg)
    x = torch.rand(1, 3, h, w)
    with torch.no_grad():
        wrapper(x)
    names = wrapper.names

    onnx_path = Path(tempfile.mkdtemp()) / "pose.onnx"
    axes = {"input": {0: "batch"}}
    axes.update({n: {0: "batch"} for n in names})
    torch.onnx.export(wrapper, x, str(onnx_path), input_names=["input"],
                      output_names=names, opset_version=opset, dynamic_axes=axes)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = [("batch=1", x)]
    if batch > 1:
        inputs.append((f"batch={batch}", torch.rand(batch, 3, h, w)))

    reports = {}
    for label, inp in inputs:
        with torch.no_grad():
            ref = [t.numpy() for t in wrapper(inp)]
        test = sess.run(None, {"input": inp.numpy()})
        reports[label] = _parity_report(names, ref, test, atol=atol, rtol=rtol)
    return {"ok": all(r["ok"] for r in reports.values()), "reports": reports}


class OnnxRunner:
    """Thin onnxruntime session wrapper: the torch-free forward pass.

    Milestone-1 primitive -- builds a session and runs it. Decoding outputs into
    keypoints and matching the torch runner's ``.inference()`` interface (with
    numerical parity) is milestone 2.
    """

    def __init__(self, onnx_path: str | Path, *, device: str | None = None):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device and "cuda" in str(device).lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def run(self, batch):
        """Run the forward pass on an ``NCHW`` float32 array; return the output list."""
        return self.session.run(None, {self.input_name: batch})
