#
# FreeDLC workspace layer -- ONNX export + runtime (milestone 1: plumbing)
#
"""Export bundle pose models to ONNX and run them with onnxruntime.

MILESTONE 1 -- PLUMBING ONLY.

``export_pose_onnx`` traces a torch model and therefore cannot run in a
torch-free environment. It is scaffolded against DLC's real builders
(``PoseModel.build`` + ``load_state_dict``) but is **unverified** here -- the
torch-vs-onnxruntime numerical parity check that actually matters needs a torch
environment (see docs/onnx_export.md).

``OnnxRunner`` (the onnxruntime session wrapper -- the torch-free forward pass) IS
exercised, against a real trivial model. Turning its raw outputs into keypoints
(the heatmap decoder) and matching the torch runner's per-frame ``.inference()``
interface is milestone 2.

torch / onnxruntime are imported lazily, so importing this module stays light.
"""
from __future__ import annotations

from pathlib import Path


def export_pose_onnx(bundle, out_path: str | Path, *, opset: int = 17, dynamic: bool = True) -> Path:
    """Trace the bundle's default pose model to ONNX. Requires torch; UNVERIFIED here.

    Mirrors how inference builds the model: ``PoseModel.build(cfg["model"])`` then
    ``load_state_dict(torch.load(snapshot)["model"])``. Only the network forward
    pass is exported; the heatmap decoder stays in Python (milestone 2).
    """
    import torch

    from deeplabcut.pose_estimation_pytorch.models import PoseModel

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = bundle._read_pose_config()
    model = PoseModel.build(cfg["model"])
    snapshot = torch.load(str(bundle.snapshot_path("default")), map_location="cpu", weights_only=False)
    model.load_state_dict(snapshot["model"])
    model.eval()

    crop = cfg["data"]["inference"].get("top_down_crop") or {"height": 256, "width": 256}
    dummy = torch.zeros(1, 3, int(crop["height"]), int(crop["width"]))
    dyn = {"input": {0: "batch"}, "heatmaps": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["heatmaps"],
        opset_version=opset, dynamic_axes=dyn,
    )
    return out_path


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
