# ONNX export for workspace bundles — design sketch

Status: **milestone 1 is implemented; nothing yet runs a model through ONNX.**

What exists (`deeplabcut/workspace/onnx_export.py`, install with `pip install -e ".[onnx]"`):

- `fdlc export <bundle> [--opset 18]` writes the bundle's pose model to
  `snapshots/pose.onnx` and records it on the card (`ModelCard.pose_onnx`).
- `fdlc export <bundle> --check [--atol 1e-3 --rtol 1e-3 --batch N]` first runs the
  torch model and the ONNX model on the same input, prints the difference for every
  output tensor, and exports only if they match.
- `ModelBundle.build_pose_runner(backend="onnx")` returns an `OnnxRunner`: an
  onnxruntime session that runs the forward pass and returns the raw head outputs.

What does not exist yet: decoding those outputs into keypoints, so `fdlc apply`
always uses torch; and detector export for top-down models. The export and the
parity check are covered only by tests that mock torch -- run `fdlc export --check`
on a torch machine before relying on an exported model.

The rest of this page is the design the implementation started from. Its code
blocks are sketches, not the code as written.

## Goal

Make a `ModelBundle` runnable without torch, via
[onnxruntime](https://onnxruntime.ai/). Today a bundle is portable in *files*
(path-free `pose.yaml` + `.pt` snapshots) but not in *runtime* — you still need
the full torch + torchvision + DLC stack to run it. An ONNX bundle would:

- run inference with only `onnxruntime` (CPU or GPU) — no torch, no DLC engine;
- be genuinely cross-platform / deployable (the same `.onnx` runs anywhere ORT does);
- open the door to the realtime path SLEAP has (ORT + I/O binding is low-latency);
- keep the *exact* two-stage structure — we export the two nets, the Python glue stays.

## Where it grafts

Two seams, mirroring what already exists for the torch path:

1. **Export** — a new `ModelBundle.export_onnx()` and a `fdlc export` command.
   It writes `pose.onnx` (and `detector.onnx` for top-down) next to the snapshots
   and records their names on the `ModelCard`.
2. **Inference** — an `OnnxPoseRunner` / `OnnxDetectorRunner` that
   `build_pose_runner` / `build_detector_runner` return when the caller asks for
   the ONNX backend. `apply_to_video` and everything above it are unchanged —
   they already talk to runners through `video_inference`, so the swap is below
   that line.

```
ModelCard:  + pose_onnx: str | None      + detector_onnx: str | None
ModelBundle: + export_onnx(opset=18, dynamic=True)
             + build_pose_runner(..., backend="torch"|"onnx")
apply:       unchanged (runner is an interface)
cli:         + fdlc export <bundle> [--opset 18] [--check]
```

## Export path (sketch)

The torch model is already built from `pose.yaml` + snapshot by DLC's builders —
the same ones `get_pose_inference_runner` uses internally. Export is: build it,
put it in eval mode, trace with a dummy input, write ONNX with dynamic axes.

```python
def export_pose_onnx(bundle, out_path, *, opset=18, dynamic=True):
    import torch
    from deeplabcut.pose_estimation_pytorch.models import PoseModel  # the builder DLC uses

    cfg = bundle._read_pose_config()
    model = PoseModel.build(cfg["model"])                 # same construction as inference
    state = torch.load(bundle.snapshot_path("default"), map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    h, w = cfg["data"]["inference"]["top_down_crop"].values()   # e.g. 256x256
    dummy = torch.zeros(1, 3, h, w)
    axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["heatmaps", "locref"],
        opset_version=opset, dynamic_axes=axes,
    )
    return out_path
```

The **pose** net (HRNet → heatmap/locref heads) is a clean CNN and exports
straightforwardly. Two decisions:

- **Postprocessing (heatmap → keypoints).** The argmax + local-refinement step
  that turns heatmaps into (x, y, score) can either be (a) baked into the graph
  (cleaner single artifact, but argmax/soft-argmax export is fiddly) or (b) kept
  in Python/numpy after ORT returns the heatmaps. Recommend **(b)** first — it's
  the smaller, safer diff and matches how the torch runner already splits model
  vs decoder.
- **Normalization.** Fold the mean/std normalize into the graph or keep it in the
  pre-processing; keeping it out mirrors the current transform pipeline.

## Inference path (sketch)

```python
class OnnxPoseRunner:
    def __init__(self, onnx_path, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=list(providers))

    def predict(self, batch):                              # batch: NCHW float32
        heatmaps, locref = self.sess.run(None, {"input": batch})
        return decode_heatmaps(heatmaps, locref)           # the SAME numpy decoder as torch path
```

Reusing the existing heatmap decoder for both backends is the key to
correctness: same postprocess, only the forward pass differs.

## The top-down complication (the hard part)

Top-down is two nets with Python glue between them:

```
detector(frame) -> boxes  ──►  crop+resize each box  ──►  pose(crops) -> keypoints
```

- **Pose stage:** easy, per above.
- **Detector stage (FasterRCNN):** this is where ONNX export gets genuinely hard.
  torchvision detection models export to ONNX, but the graph includes NMS and
  produces a *dynamic* number of boxes; historically this needs a recent opset,
  careful `dynamic_axes`, and sometimes `torch.onnx.dynamo_export` rather than the
  legacy tracer. Expect iteration here, and validate box-for-box against the torch
  detector before trusting it.
- **The crop/resize glue stays in Python** either way — it is not part of either
  graph. So an "ONNX bundle" is really two `.onnx` files plus the existing Python
  orchestration, not one end-to-end graph. That's fine and actually keeps
  `apply_to_video` untouched.

A pragmatic first milestone: **pose-only ONNX** (bottom-up or single-animal
models, and the pose stage of top-down), leaving the FasterRCNN detector on torch.
That already removes torch from the majority of the compute for many models and
de-risks the easy 80% before tackling detector export.

## Verification plan (what this sketch does NOT prove)

None of the above is runtime-checked here. Before it can be trusted, on a torch box:

1. Export `pose.onnx`; run ORT and the torch runner on the **same** crops; assert
   keypoints match within a tight tolerance (e.g. < 1e-3 px, allowing for
   fp rounding). This is the real correctness gate.
2. Confirm dynamic batch works (batch 1 and 8 give identical per-item results).
3. Only then attempt `detector.onnx`, with the same box-level parity check.
4. Wire `build_pose_runner(backend="onnx")` and re-run one real video end to end;
   diff the `pose.parquet` against the torch run.

The AST/smoke tests here can cover the *plumbing* (card fields, CLI dispatch, path
resolution) with mocks, exactly as the torch runner is covered — but the numerical
parity checks above are the ones that matter and they need torch + onnxruntime.

## Suggested increments

1. *(done)* `ModelCard.pose_onnx` field + `fdlc export` writing `pose.onnx`
   (pose-only) with a `--check` parity gate, plumbing tests with mocks.
2. *(started)* `build_pose_runner(backend="onnx")` returns a forward-pass-only
   `OnnxRunner`. Still to do: decode its outputs with the numpy heatmap decoder
   behind the torch runner's interface, and a parity test vs torch on real crops.
3. Detector export (`detector.onnx`) — the hard, iterate-until-it-matches step.
4. *(partly done)* `onnx`/`onnxruntime` are an optional extra (`.[onnx]`); torch is
   still a core dependency, so a torch-free deployment needs its own slim package.
