# Workspace projects from the command line (`dlc-ws`)

A *workspace* is FreeDLC's project layout: a `project.toml` plus four directories
(`sources/`, `models/`, `runs/`, `derived/`). It replaces the legacy
`config.yaml` + `labeled-data/` + `dlc-models-pytorch/` tree, and is driven from
the `dlc-ws` command.

This page walks the pipeline from an empty directory to a trained, evaluated
model, and then covers the cases that need a decision: which resolution to train
on, what to do when a video has no downscaled counterpart, and how to recover
from broken frame links.

```text
create -> add-video -> extract-frames -> annotate -> train -> evaluate -> apply
```

Every command has `--help`. Note that the project is given in two ways: most
commands take it as the first positional argument (`dlc-ws train <project>`),
while `extract-frames` and `annotate` take the *video* positionally and the
project as `--project` (default: the current directory).

## Layout

```text
<project>/
|- project.toml
|- sources/
|  |- videos/original/<video_id>/video.<ext>  (+ video.toml)
|  |- videos/processed/<video_id>/video.<ext> (+ video.toml)
|  '- annotations/<video_id>/
|     |- frames/original/     frames you annotate on
|     |- frames/processed/    the same frames from the processed video
|     |- labels.parquet       the annotations
|     '- labels.toml          which pixel space labels.parquet is in
|- models/<model_id>/         portable model bundles (model.toml, pose.yaml, snapshots/)
|- runs/<kind>/<run_id>/      one directory per train / evaluate / analyze run
|- derived/
'- .annotate/<video_id>/      staging for the annotator (see "Annotate")
```

## Original and processed videos

A video can be registered twice under the same id:

- **original** -- the full-resolution recording. Frames for annotation are taken
  from it, so markers are placed on sharp images.
- **processed** -- a downscaled copy. This is the video the model is meant to run
  on, so by default it is also what the model is trained on.

The two are a pair only if they share a **video id**. The id is derived from the
file name, so a downscaled file with a different name must be given the
original's id explicitly:

```bash
dlc-ws create ws --task reach --bodyparts snout paw tail --experimenters gravio

dlc-ws add-video ws session1.mp4
dlc-ws add-video ws session1-320x240.mp4 --processed --video-id session1
```

Without `--video-id`, the second file would be registered as a separate video
`session1-320x240` and nothing would connect it to `session1`.

`dlc-ws videos <project>` shows, for every id, what is registered, labeled and
extracted under it, and points out what does not fit together:

```text
video id          original   processed  labels in     frames o/p
session1          1920x1080  192x108    original px   20/20
session2          1920x1080  -          original px   20/0
session3-192x108  -          192x108    original px?  0/0
  ! session3-192x108: has labels but no original video is registered under this id
  ! session3-192x108: processed video has no original with the same id (see `add-video --video-id`)
```

`labels in` is the pixel space the coordinates are *stored* in -- not which video
was annotated, which is always the original. It does not limit what you can train
on. A trailing `?` means there is no `labels.toml` and the space is inferred from
the current pairing.
`frames o/p` counts the readable frames in `frames/original/` and
`frames/processed/`.

`--link` controls how the media is stored: `symlink` (default), `copy`, or
`reference` (record the path only). `--exist-ok` re-registers an id; it replaces
the stored media but never writes to the file a previous symlink pointed at.

A processed video is optional. A project with only originals works; see
"Projects without processed videos" under Use cases.

## Extract frames

```bash
dlc-ws extract-frames session1 --project ws -n 20
dlc-ws extract-frames --all --project ws -n 20 --mode kmeans -j 4
```

Frames are selected once, on the original video, and written to
`frames/original/`. If a processed video is registered, the same frame indices
are also read from it into `frames/processed/`, under the same file names.

Running it again is safe and **does not select new frames**. It keeps the
existing set and completes it:

- a frame that has become a link to itself is re-read from the video;
- processed frames missing for an original are filled in, which is what happens
  when the processed video was registered after extraction.

`--overwrite` discards the set and selects again. Labels are keyed by frame file
name, so after `--overwrite` existing labels may point at frames that no longer
exist. Do not use it on a video you have already annotated unless you intend to
re-annotate.

## Annotate

```bash
dlc-ws annotate session1 --project ws
```

This opens the napari annotator (the `napari-freedlc` plugin) on the original
frames. Save the keypoints layer in napari (File > Save Selected Layer(s), Ctrl+S),
then close the window. On close the labels are
read into the workspace:

- `labels.parquet` -- the annotations in long form, in the pixels of the original
  frames you placed them on;
- `labels.toml` -- a record of that pixel space (`space = "original"`).

Having a processed video does not change what is stored. The coordinates are
converted to the processed frames when a model is trained or evaluated on them
(see `--frames`), so the labels stay valid if you later replace the processed
video with one of a different size.

Projects annotated with earlier versions may hold labels that were scaled to the
processed video when they were saved (`space = "processed"`, with the scale
used). They keep working: the record says how to convert them.

The annotator itself only ever sees `.annotate/<video_id>/`, a staging directory
holding a synthesized `config.yaml`, symlinks to the frames, and the
`CollectedData_*` file napari saves. Frames are only viewed through it; nothing
is copied or linked back from it into `sources/`. It is kept between sessions so
earlier labels reload.

## Train

```bash
dlc-ws train ws --net resnet_50 --epochs 200 --batch-size 8 --device cuda:0
```

Training uses every video that has a `labels.parquet`.

| Option | Default | Meaning |
|---|---|---|
| `--frames` | `processed` | frame set to train on: `processed` or `original` |
| `--net` | `resnet_50` | architecture |
| `--epochs` | `200` | pose-model epochs |
| `--batch-size` | `8` | |
| `--detector-epochs` | `0` | above 0 trains a top-down model (detector, then pose) |
| `--train-fraction` | `0.95` | share of images used for training; the rest are the test set |
| `--seed` | `0` | |
| `--device` | auto | e.g. `cuda:0`, `cpu` |

### Which resolution: `--frames`

`--frames` selects the frame set for the **whole run**, so one dataset never
mixes resolutions.

| `--frames` | Images | Coordinates |
|---|---|---|
| `processed` (default) | `frames/processed/` | in processed pixels |
| `original` | `frames/original/` | in original pixels |

The label coordinates are converted into the chosen set's pixel space when the
dataset is staged. It does not matter which space `labels.parquet` is stored in:
`labels.toml` records that, and the conversion goes whichever way is needed.

Train on the resolution you will run the model on. A model trained on
full-resolution frames performs poorly on downscaled video and the reverse,
because the animal's apparent size differs. See Use cases below for when
`original` is the right choice.

`--frames processed` needs a processed video, with known dimensions, for every
annotated video. If one is missing, training stops before anything is written
and names each video:

```text
cannot use the processed frames for 1 video(s):
  session2: no processed video is registered
register the missing counterpart (`dlc-ws add-video <project> <video> --processed --video-id <id>`) or choose `--frames original`
```

### What a run produces

Progress is printed while training runs:

```text
Using 57 images and 3 for testing
Starting pose model training...
Epoch 1/200 (lr=0.0005), train loss 0.15812
Epoch 2/200 (lr=0.0005), train loss 0.02725
...
trained -> models/20261007-063121-76fd3f (processed frames)
```

```text
runs/train/<run_id>/
|- run.toml                    parameters, including the frame set
|- dataset/
|  |- annotations/{train,test}.json
|  '- images/<video_id>/       links to the labeled frames
'- train/
   |- train.txt                the same log that was printed
   |- learning_stats.csv       per-epoch numbers
   |- pytorch_config.yaml
   '- snapshot-*.pt
models/<model_id>/             the resulting bundle
```

`models/<model_id>/model.toml` records `frames = "processed"` or `"original"`.

Only frames with at least one labeled keypoint are trained on; frames that were
extracted but never labeled are skipped. Of those, only frames whose image is a
readable file are used. If some are
not, they are left out and the run says so, per video, with the fix:

```text
session1: 3 of 20 labeled frame(s) have no readable image in sources/annotations/session1/frames/original and are left out of training; run `dlc-ws extract-frames session1` to restore them
```

### Seeing which files are used: `--verbose`

`train`, `apply` and `label` take `-v` / `--verbose`, which lists every file the
command reads and writes, as it gets to them:

```text
$ dlc-ws train ws --verbose
project /data/ws
frame set: processed
video session1: 20 labeled frame(s)
  labels  /data/ws/sources/annotations/session1/labels.parquet (stored in original pixels, coordinates x0.1 y0.1)
  frames  /data/ws/sources/annotations/session1/frames/processed
dataset /data/ws/runs/train/<run_id>/dataset
  train   /data/ws/runs/train/<run_id>/dataset/annotations/train.json (19 image(s))
  test    /data/ws/runs/train/<run_id>/dataset/annotations/test.json (1 image(s))
  images  /data/ws/runs/train/<run_id>/dataset/images (symlinks to the frames above)
training in /data/ws/runs/train/<run_id>/train
  config     /data/ws/runs/train/<run_id>/train/pytorch_config.yaml
  log        /data/ws/runs/train/<run_id>/train/train.txt
  stats      /data/ws/runs/train/<run_id>/train/learning_stats.csv
  snapshots  /data/ws/runs/train/<run_id>/train/snapshot-*.pt
Using 19 images and 1 for testing
...
model bundle /data/ws/models/<model_id>
  card      /data/ws/models/<model_id>/model.toml
  config    /data/ws/models/<model_id>/pose.yaml
  snapshot  /data/ws/models/<model_id>/snapshots/pose-snapshot-best-190.pt  (default)
  from run  /data/ws/runs/train/<run_id>/run.toml
trained -> models/<model_id> (processed frames)
```

Paths are absolute. When the labels are converted for the chosen frame set, the
`labels` line says so (`stored in original pixels, coordinates x0.1 y0.1`). For `apply` the report
names the bundle's config and the snapshot used, each input video (with the file
a symlink resolves to), and the pose file, record and labeled video written for
it. For `label` it names the video, the pose file, where the marker names and
skeleton came from, and the output.

### Augmentation

Images are augmented on the fly with DeepLabCut's defaults: rotation up to 30
degrees and rescaling between 0.5x and 1.25x (applied to half the images), a
448x448 keypoint-aware crop, Gaussian noise and motion blur. Horizontal flips
are off. These are not yet configurable from `dlc-ws train`; the resolved
settings are written to `train/pytorch_config.yaml` and printed at the start of
the log.

## Evaluate

```bash
dlc-ws evaluate ws <model_id>
```

Runs the model on the labeled frames and reports the error against the
annotations, in pixels. It scores on the frame set the model was trained on,
read from `model.toml`; the annotations are converted to match. The metrics are
printed and stored on the run and the model card.

The report gives both the mean and the median error, overall and per marker
(`per_bodypart`, `per_bodypart_median`). Read them together: pose errors are
heavy-tailed, so a marker that is exact in most frames and lost in a few shows a
small median and a large mean. `--pck 20` adds the share of predictions within
20 px, overall and per marker (`per_bodypart_pck`), which says how often a marker
is usable.

`--frames original|processed` overrides the frame set. Errors are then in that
set's pixels, so numbers from different frame sets are not directly comparable:
the same miss is 2x larger in pixels on frames twice the size.

## Apply to videos

```bash
dlc-ws apply --project ws --model-id <model_id> session1-320x240.mp4 --labeled-video
```

`apply` runs on the video files you pass, so the resolution is your choice of
file. Pass video of the resolution the model was trained on: processed-size
video for a model trained with `--frames processed`, full-resolution video for
one trained with `--frames original`.

## Use cases

### Train at the processed resolution (the default)

Register both videos under one id, annotate, train:

```bash
dlc-ws add-video ws session1.mp4
dlc-ws add-video ws session1-320x240.mp4 --processed --video-id session1
dlc-ws annotate session1 --project ws
dlc-ws train ws
```

You annotate on full-resolution frames; the model learns from, and is later run
on, the downscaled video.

### Train at the original resolution

```bash
dlc-ws train ws --frames original
```

Works whether or not processed videos exist. Use it when the model will be
applied to full-resolution video, or to check whether the downscaling is what
limits accuracy. Compare the two models by applying each to its own resolution,
not by their pixel errors.

### Projects without processed videos

There are no processed frames to train on, so the
default cannot be used. Pass the flag on every run:

```bash
dlc-ws train ws --frames original
```

This is also the right command when the only videos you have are already small
and were registered as originals: "original" means "the frames you annotated
on", not "high resolution".

### Adding a processed video after annotating

```bash
dlc-ws add-video ws session1-320x240.mp4 --processed --video-id session1
dlc-ws extract-frames session1 --project ws      # fills in frames/processed/
dlc-ws train ws
```

No re-annotation is needed. `labels.toml` says the labels are in original
pixels, and training scales them down.

This relies on `labels.toml`. Labels ingested before that file existed have no
record, and their space is then inferred from the videos registered *now* --
which would wrongly read original-space labels as processed once a processed
video is added. If `sources/annotations/<video_id>/labels.toml` is missing,
write it first by opening and closing the annotator **before** registering the
processed video:

```bash
dlc-ws annotate session1 --project ws    # save, close
```

### Replacing the processed video with one of another size

```bash
dlc-ws add-video ws session1-640x360.mp4 --processed --video-id session1 --exist-ok
rm -r ws/sources/annotations/session1/frames/processed
dlc-ws extract-frames session1 --project ws      # re-reads the same frames
dlc-ws train ws
```

The labels need no attention: they are converted to whatever processed video is
registered when you train, including labels that an earlier version stored
already scaled for the old size.

The processed *frames* do need replacing, since the ones on disk were read from
the old video: delete `frames/processed/` as above and `extract-frames` re-reads
the same frame indices from the new video. It never reselects frames.

The conversion is a pure scale. The processed video must show the same field of
view as the original; a processed video that is *cropped* is not supported.

### Which space are my labels in?

```bash
cat ws/sources/annotations/session1/labels.toml
```

```toml
video_id = "session1"
space = "original"
scale_x = 1.0
scale_y = 1.0
```

## Troubleshooting

### The model predicts nothing: likelihoods near 0, points on a regular grid

Look at `runs/train/<run_id>/train/learning_stats.csv`. If
`losses/train.bodypart_locref` is exactly `0.0` and the heatmap loss falls to
around `1e-9` within a few epochs, the model was trained without labels: every
target was blank, and it learned to output zeros. Versions before the fix wrote
an empty bounding box for each annotation, which DeepLabCut's loader discards
without a message. Models from those versions are unusable; update and retrain.
A healthy run shows a non-zero locref loss and a heatmap loss that decreases
gradually. Training now refuses to start if the loader keeps fewer labeled
images than the dataset holds.

### `cannot use the processed frames for N video(s)`

The default `--frames processed` needs, for every annotated video, an original
and a processed video registered under the same id. The message names each video
and what is missing; `dlc-ws videos <project>` shows the whole picture. Register
the missing counterpart under the same id (`--processed --video-id`), or train
with `--frames original`. The usual cause is a downscaled file that was
registered under its own id instead of the original's.

### `N of M labeled frame(s) have no readable image ... left out of training`

Frame files are missing or are broken links. Run
`dlc-ws extract-frames <video> --project <project>` (without `--overwrite`) and
train again.

### Frames that are symlinks to themselves (`Too many levels of symbolic links`)

Left behind by `dlc-ws annotate` in versions before the link fix, which
replaced each frame with a link to itself when the annotator closed. The
labels are intact. `dlc-ws extract-frames` or `dlc-ws annotate` re-reads those
frames from the video at the index in their file name. Do not use
`--overwrite`.

### napari: `No supported images were found`

The staged frame links point at missing files; newer `napari-freedlc` versions
say how many and show one. Same fix as above: re-run `dlc-ws annotate`, which
restores the frames and re-makes the links.

### Training prints nothing between the dataset warnings and `trained -> ...`

An older version that attached no log handler. Per-epoch numbers are still in
`runs/train/<run_id>/train/learning_stats.csv`.

### `File .../dataset/annotations/train.json does not exist`, or `analyze_images() got an unexpected keyword argument 'detector_runner'`

Bugs in older versions of `dlc-ws train` and `dlc-ws evaluate`; update.

### napari-freedlc fails to import with `cannot import name 'SYMBOL_TRANSLATION_INVERTED'`

napari 0.9 removed a name the plugin uses. Install `napari<0.9`.
