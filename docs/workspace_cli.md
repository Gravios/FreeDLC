# Workspace projects from the command line (`fdlc`)

A *workspace* is FreeDLC's project layout: a `project.toml` plus four directories
(`sources/`, `models/`, `runs/`, `derived/`). It replaces the legacy
`config.yaml` + `labeled-data/` + `dlc-models-pytorch/` tree, and is driven from
the `fdlc` command. (`dlc-ws`, its earlier name, still works.)

This page walks the pipeline from an empty directory to a trained, evaluated
model, and then covers the cases that need a decision: which resolution to train
on, what to do when a video has no downscaled counterpart, and how to recover
from broken frame links.

```text
create -> add-video -> extract -> annotate -> train -> evaluate -> apply
   ^                                                                 |
   '------------- extract --from-run (more frames to label) <--------'
```

Installation is described in the [README](../README.md#install).

## Commands at a glance

| Command | Does |
|---|---|
| `create <dir>` | start a project: task, markers, skeleton |
| `list skeletons [name]` | the bundled skeleton configs, or one in detail |
| `export-skeleton <project>` | write the project's markers and skeleton as a reusable config |
| `migrate <legacy> <dest>` | convert a legacy DeepLabCut project (config.yaml tree) |
| `info <project>` | the task and markers, and how many videos, annotated videos, models and runs |
| `add-video <project> <videos...>` | register original (or, with `--processed`, downscaled) videos |
| `videos <project>` | what is registered, labeled and extracted per video; `--register DIR` pairs a folder of processed videos |
| `extract [<video>] --project P` | frames to annotate; `--from-run` adds frames chosen from a model's results |
| `annotate <video> --project P` | napari on the full-resolution frames |
| `train <project>` | train a model on the labeled frames |
| `models <project>` | the trained model bundles |
| `evaluate <project> <model_id>` | pixel error against the annotations |
| `apply <videos...>` | run a model on videos (`--project` + `--model-id`, or `--model <bundle>`) |
| `label <video>` | render a labeled video from a pose file |
| `track <pose.parquet>` | link multi-animal detections into identities across frames |
| `export <bundle>` | write the pose model as ONNX (see [onnx_export.md](onnx_export.md)) |

Every command has `--help`.

**The project can be left out.** Run inside a project -- in its root or any folder
below it -- and every command finds it, the way git finds a repository:

```bash
cd ws
fdlc videos
fdlc train --device cuda:0
fdlc evaluate <model_id> --pck 20
fdlc extract --from-run latest -n 10
```

To name it instead, most commands take it as the first positional argument
(`fdlc train ws`), while `extract` and `annotate` take the *video* positionally
and the project as `--project`, and `apply` takes it as `--project`. Either form
accepts the project root or its `project.toml`. `add-video` tells the two apart
by `project.toml`: inside a project, `fdlc add-video a.mp4 b.mp4` adds both files.
The examples on this page name the project (`ws`) so they work from anywhere.

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
|- derived/                   reserved; nothing writes here yet
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
fdlc create ws --task reach --bodyparts snout paw tail --experimenters gravio

fdlc add-video ws session1.mp4
fdlc add-video ws session1-320x240.mp4 --processed --video-id session1
```

Without `--video-id`, the second file would be registered as a separate video
`session1-320x240` and nothing would connect it to `session1`.

`fdlc videos <project>` shows, for every id, what is registered, labeled and
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

### Registering a whole folder: `videos --register`

Processed videos usually come as a folder made from the originals -- `reduced/`,
`reduced-640x360/`, `triplet/`. `--register` pairs such a folder with the
originals by name, so no `--video-id` is needed:

```bash
fdlc videos ws --register /data/video/reduced-640x360
fdlc extract --all --project ws --match-original
```

A file belongs to an original when its name is the original's name, optionally
followed by a suffix: `Session1_640x360.mp4` and `Session1-triplet.mp4` both
belong to `session1`. The folder is taken as a whole. If any video in it matches
no original, or two match the same one, the mismatches are listed and **nothing
is changed**. Labeled `.fdlc.mp4` videos in the folder are ignored.

Each matched original gets the file as its processed video, replacing the one it
had; the previous file is not touched, only the link to it. Originals with no
video in the folder keep what they had. Any number of folders can be kept side
by side and switched between this way -- the project points at one at a time.

The frames in `frames/processed/` were read from the previous videos, so run
`extract --match-original` after switching (see "Extract frames"). `videos`
points out frames whose size no longer fits:

```text
  ! session1: processed frames are 192x108 but the processed video is 640x360 (run `fdlc extract session1 --match-original`)
```

It also notes a processed video whose frame count differs from its original's.
Frame N of one is then not necessarily frame N of the other, and labels would sit
on the wrong moment; re-encode it without dropping or duplicating frames
(`ffmpeg -fps_mode passthrough`).

## Extract frames

```bash
fdlc extract session1 --project ws -n 20
fdlc extract --all --project ws -n 20 --mode kmeans -j 4
```

(`extract-frames`, the command's earlier name, still works.)

Frames are selected once, on the original video, and written to
`frames/original/`. If a processed video is registered, the same frame indices
are also read from it into `frames/processed/`, under the same file names.

Running it again is safe and **does not select new frames**. It keeps the
existing set and completes it:

- a frame that has become a link to itself is re-read from the video;
- processed frames missing for an original are filled in, which is what happens
  when the processed video was registered after extraction.

`--match-original` reads the whole processed set again, from the processed video
registered now, at the frames already extracted from the original. Use it after
replacing the processed video. The selection, the original frames and the labels
are untouched, so no annotation is lost. With `--all` it skips videos that have
no processed video.

To add frames chosen from where a trained model is unsure or confident, see
`extract --from-run` below.

`--overwrite` discards the set and selects again. Labels are keyed by frame file
name, so after `--overwrite` existing labels may point at frames that no longer
exist. Do not use it on a video you have already annotated unless you intend to
re-annotate.

## Annotate

```bash
fdlc annotate session1 --project ws
```

This opens the napari annotator (the `napari-freedlc` plugin) on the original
frames, extracting them first if there are none yet (`-n`, `--mode` as for
`extract`). Save the keypoints layer in napari (File > Save Selected Layer(s),
Ctrl+S), then close the window. On close, what was last saved is read into the
workspace -- changes made after the last save are not:

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
earlier labels reload. If it has no `CollectedData_*` file -- a migrated project,
or after `.annotate/` was deleted -- one is written from `labels.parquet` before
napari opens, so the existing labels are shown and kept.

Marker names are hidden in napari; hold **N** to show them. Marker colors and
sizes come from the project's `[display]` table (see "Marker colors and sizes").

## Train

```bash
fdlc train ws --net resnet_50 --epochs 200 --batch-size 8 --device cuda:0
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
register the missing counterpart (`fdlc add-video <project> <video> --processed --video-id <id>`) or choose `--frames original`
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
session1: 3 of 20 labeled frame(s) have no readable image in sources/annotations/session1/frames/original and are left out of training; run `fdlc extract session1` to restore them
```

### Seeing which files are used: `--verbose`

`train`, `apply` and `label` take `-v` / `--verbose`, which lists every file the
command reads and writes, as it gets to them:

```text
$ fdlc train ws --verbose
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
are off. Frames smaller than the crop are padded to it first, so at 640x360 a
training image is a padded full frame and at 192x108 it is mostly padding. These are not yet configurable from `fdlc train`; the resolved
settings are written to `train/pytorch_config.yaml` and printed at the start of
the log.

## Evaluate

```bash
fdlc evaluate ws <model_id>
fdlc evaluate ws <model_id> --videos session1 session2 --pck 20
```

Runs the model on the labeled frames and reports the error against the
annotations, in pixels. It scores on the frame set the model was trained on,
read from `model.toml`; the annotations are converted to match. The metrics are
printed and stored on the run and the model card. `--videos` limits it to some
videos; by default every annotated video is scored.

**Every labeled frame is scored, including the ones the model was trained on.**
That makes it a check of how well the model fits its own annotations, not of how
well it generalizes; a large error here means the model cannot even reproduce
its training data. The held-out numbers are the test metrics computed during
training, on the frames `--train-fraction` left out, in
`runs/train/<run_id>/train/learning_stats.csv` and printed in the log.
`--pcutoff` (default 0.6) sets which predictions count as confident.

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
fdlc apply --project ws --model-id <model_id> session1-320x240.mp4 --labeled-video
fdlc apply --project ws --model-id <model_id> reduced-640x360/ --device cuda:0 --batch-size 32
```

`apply` runs on the video files you pass (files, folders or globs), so the
resolution is your choice of file. Pass video of the resolution the model was
trained on: processed-size video for a model trained with `--frames processed`,
full-resolution video for one trained with `--frames original`.

Where the results go:

- with `--project`, to an analyze run: `runs/analyze/<run_id>/<video>/`, named
  after the video file (lower case, other characters as `-`), holds `pose.parquet`, a `run.toml` naming the video, and with `--labeled-video`
  a `labeled.mp4`. This is what `extract --from-run` reads (see the next
  section). The run records the `--pcutoff` its labeled videos were drawn with
  (default 0.6: a marker is drawn when its likelihood reaches it). `--out DIR`
  writes the per-video folders under `DIR` instead; the run still lists them.
- with `--beside-video`, next to each video as `<stem>.fdlc.parquet`,
  `<stem>.fdlc.toml` and `<stem>.fdlc.mp4`. No run is made, so these cannot be
  used with `extract --from-run`.
- with `--model <bundle>` instead of `--project`, a model bundle is used on its
  own, without a project, and the results go to `--out` (default
  `dlc-predictions/`).

`pose.parquet` has one row per frame, individual and marker: `frame, individual,
bodypart, x, y, likelihood`, in the pixels of the video it was run on.

## More frames from a model's own results: `extract --from-run`

```bash
fdlc apply --project ws --model-id <model_id> reduced/*.mp4 --labeled-video
fdlc extract --from-run latest --project ws -n 10
fdlc annotate session1 --project ws
fdlc train ws
```

`--from-run` reads the poses of an analyze run and adds up to `-n` frames to each
video's frame set, with the model's marker positions proposed for them:

- **unsure** frames, in which few or none of the markers would be drawn in the
  labeled video -- what the model handles worst, and where the labeled set lacks
  coverage;
- **confident** frames, in which the most markers are drawn with the highest
  likelihood -- where the model looks right. Checking those shows whether its
  confident predictions really are accurate.

`--best K` sets how many of the `-n` are confident (default: half, rounded down,
so `-n 10` gives 5 of each and `-n 5` gives 2 confident, 3 unsure); `--best 0`
adds unsure frames only. The run is named by its id, a unique start of it,
`latest`, or its directory. Give a video to limit it to that one.

A frame is unsure when at most `--max-shown` markers reach the pcutoff (default:
a third of the markers, rounded down; 5 of 15), and confident when it shows as
many markers as any frame does. The pcutoff is the run's own unless `--pcutoff`
is given. The picks are spread over each group: its frames are split, in time
order, into as many runs of equal length as there are picks, and the best of
each run is taken -- for unsure frames the fewest markers drawn, then the lowest
mean likelihood; for confident ones the highest mean likelihood. A long episode
therefore gets more picks than a short one, but never neighbouring frames of it.

The frames already extracted, and their labels, are kept; frames already in the
set are not picked again. The new frames are read from the original video and,
when one is registered, from the processed video, under the same names. A run
on either video serves, since both have the same frame numbers.

### The proposed markers

The run's positions are staged as DeepLabCut *machine labels*,
`.annotate/<video>/labeled-data/<video>/machinelabels-iter0.h5`, scaled to original
pixels and with their likelihoods. Only markers that reach the pcutoff -- those
the labeled video draws -- are placed; `--propose-all` places every marker, and
`--no-propose` none. Proposals already staged for other frames are kept, and the
previous file is kept beside it as `.bak`. Your labels and `labels.parquet` are
not touched.

In napari the proposals are a layer of their own, `machinelabels-iter0`, drawn as
`x` beside your labels layer (`CollectedData_<scorer>`). They become labels only
when you save **that layer**: napari then merges it into your labels -- adding its
points, never deleting a label you already have -- and `annotate` reads them in
when you close the window. So:

1. select the `machinelabels-iter0` layer and go through every proposed frame:
   move the markers that are wrong, delete those you cannot place, place the
   missing ones;
2. save that layer (Ctrl+S); every frame in it is added to your labels, so do
   this only once you have checked them all;
3. close the window.

Closing without saving the machine layer leaves the proposals where they are, as
proposals, for the next session. Frames that hold labels after a session are
dropped from the proposals file, and it is removed once none are left.

```text
run 20261008-094012-3fa2c1, pcutoff 0.6: per video, up to 5 frame(s) showing at most 5 of 15 marker(s) and 5 showing the most
session1: added 5 unsure (of 1312, showing 0-4) and 5 confident (of 80211, showing 15) -> ws/sources/annotations/session1/frames/original
  proposed 87 marker(s) as machine labels -> ws/.annotate/session1/labeled-data/session1/machinelabels-iter0.h5
```

Every individual's markers count towards the total, so in a multi-animal
project a frame in which one animal is seen well is not unsure even if another
is missed. Proposed positions go to the project's individuals in the order the
model numbers them.

## Labeled videos and tracking

```bash
fdlc label session1.mp4                                  # from session1.fdlc.parquet beside it
fdlc label session1.mp4 --parquet runs/analyze/<run_id>/session1/pose.parquet --project ws --model-id <model_id>
fdlc track session1.fdlc.parquet --max-distance 50 --max-gap 10
```

`label` draws a pose file onto its video: markers at or above `--pcutoff` and
the skeleton between them. The marker names and skeleton come from `--model`
or `--project`/`--model-id` when given, and otherwise from the `.fdlc.toml`
beside the pose file. `track` is for multi-animal output: it links each frame's detections
into identities by centroid distance and writes `<base>.tracked.fdlc.parquet`.

## Marker colors and sizes

A `[display]` table in `project.toml` sets how markers are drawn in the annotator
and in labeled videos, and how skeleton lines are drawn in labeled videos (the
annotator draws no lines):

```toml
[display]
dotsize = 6            # markers without a size of their own
colormap = "tab20"     # colors for markers without a color of their own
line_color = "white"   # skeleton lines without a color of their own
line_width = 1

[display.bodyparts]
head_nose = { color = "red", size = 10 }
head_mid  = { color = "orange" }
back_T4   = { color = "#00c0ff", size = 8 }

[[display.edges]]
between = ["head_nose", "head_mid"]
color = "red"
width = 2

[[display.edges]]
between = ["back_T4", "back_T8"]
color = "#00c0ff"
```

Every key is optional. Colors are names (`red`, `orange`, `tab:blue`) or hex
codes. Sizes are dot diameters in pixels of the image they are drawn on: the
full-resolution frame in the annotator, the video in a labeled video -- so the
same size looks three times larger on a 640x360 video than on 1920x1080 frames.
An edge is named by its two markers, in either order, and must be in the
skeleton.

Markers without a color of their own take one from `colormap`, in the order of
`bodyparts`. A palette (`tab10`, `tab20`, `Set3`, `Dark2`, ...) gives its colors in
order, which keeps neighbouring markers distinct; a gradient (`viridis`,
`plasma`, ...) is sampled evenly across its range. Without `colormap`, the
annotator uses `viridis` and labeled videos a hue wheel.

The table is checked whenever the project is opened: a misspelled marker, an
edge not in the skeleton, an unreadable color or an unknown key stops the
command with a list of what is wrong.

In napari, the point-size control sets the size of markers without a size of
their own; changes made there are not written back to `project.toml`. `fdlc
label --dotsize R` draws every marker with radius `R`, overriding the table.
`apply --beside-video` stores the table in the `.fdlc.toml` sidecar, so a later
`fdlc label` draws the same style.

## Skeleton configs

```bash
fdlc list skeletons                      # the bundled configs
fdlc list skeletons RodentH5B7T3         # one config's markers, edges and segments
fdlc create ws --task reach --skeleton-config RodentH5B7T3
fdlc export-skeleton ws --name MyRig     # this project's markers and edges as a config
```

A skeleton config names a rig's markers, its skeleton edges and its kinematic
tree (segments), in the table layout mufasa's `project.toml` uses.
`--skeleton-config` takes the markers and edges from one; `export-skeleton`
writes one from a project, to `.fdlc/skeletons/` by default.

## Legacy projects

```bash
fdlc migrate /path/to/legacy-project ws
```

Converts a DeepLabCut project (`config.yaml`, `labeled-data/`,
`dlc-models-pytorch/`) into a workspace: its videos, annotations and trained
models. `--link` sets how files are brought across (`symlink` by default),
and `--no-videos`, `--no-annotations`, `--no-models` leave parts out. The
legacy project is only read.

## Use cases

### Train at the processed resolution (the default)

Register both videos under one id, annotate, train:

```bash
fdlc add-video ws session1.mp4
fdlc add-video ws session1-320x240.mp4 --processed --video-id session1
fdlc annotate session1 --project ws
fdlc train ws
```

You annotate on full-resolution frames; the model learns from, and is later run
on, the downscaled video.

### Train at the original resolution

```bash
fdlc train ws --frames original
```

Works whether or not processed videos exist. Use it when the model will be
applied to full-resolution video, or to check whether the downscaling is what
limits accuracy. Compare the two models by applying each to its own resolution,
not by their pixel errors.

### Projects without processed videos

There are no processed frames to train on, so the
default cannot be used. Pass the flag on every run:

```bash
fdlc train ws --frames original
```

This is also the right command when the only videos you have are already small
and were registered as originals: "original" means "the frames you annotated
on", not "high resolution".

### Adding a processed video after annotating

```bash
fdlc add-video ws session1-320x240.mp4 --processed --video-id session1
fdlc extract session1 --project ws      # fills in frames/processed/
fdlc train ws
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
fdlc annotate session1 --project ws    # save, close
```

### Replacing the processed video with one of another size

```bash
fdlc add-video ws session1-640x360.mp4 --processed --video-id session1 --exist-ok
fdlc extract session1 --project ws --match-original
fdlc train ws
```

For a folder of them, `fdlc videos ws --register <folder>` replaces the first
command for every video at once, followed by `fdlc extract --all --project ws
--match-original`.

The labels need no attention: they are converted to whatever processed video is
registered when you train, including labels that an earlier version stored
already scaled for the old size.

The processed *frames* do need replacing, since the ones on disk were read from
the old video: `--match-original` re-reads the same frame indices from the new
video. It never reselects frames.

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
and what is missing; `fdlc videos <project>` shows the whole picture. Register
the missing counterpart under the same id (`--processed --video-id`), or train
with `--frames original`. The usual cause is a downscaled file that was
registered under its own id instead of the original's.

### `N of M labeled frame(s) have no readable image ... left out of training`

Frame files are missing or are broken links. Run
`fdlc extract <video> --project <project>` (without `--overwrite`) and
train again.

### Frames that are symlinks to themselves (`Too many levels of symbolic links`)

Left behind by `fdlc annotate` in versions before the link fix, which
replaced each frame with a link to itself when the annotator closed. The
labels are intact. `fdlc extract` or `fdlc annotate` re-reads those
frames from the video at the index in their file name. Do not use
`--overwrite`.

### napari: `No supported images were found`

The staged frame links point at missing files; newer `napari-freedlc` versions
say how many and show one. Same fix as above: re-run `fdlc annotate`, which
restores the frames and re-makes the links.

### Training prints nothing between the dataset warnings and `trained -> ...`

An older version that attached no log handler. Per-epoch numbers are still in
`runs/train/<run_id>/train/learning_stats.csv`.

### `File .../dataset/annotations/train.json does not exist`, or `analyze_images() got an unexpected keyword argument 'detector_runner'`

Bugs in older versions of `fdlc train` and `fdlc evaluate`; update.

### napari: `super-class __init__() of type KeypointControls was never called`

napari is running on PyQt5 (or PyQt6) instead of PySide6: when both are installed,
qtpy, which napari uses to pick one, prefers PyQt5. `fdlc annotate` now asks for
PySide6 (it sets `QT_API=pyside6` unless you have set it), and napari-freedlc no
longer fails on PyQt. With older versions, `export QT_API=pyside6`, or remove the
extra binding: `pip list | grep -i pyqt`, then `pip uninstall pyqt5 pyqt5-qt5
pyqt5-sip`.

### napari-freedlc fails to import with `cannot import name 'SYMBOL_TRANSLATION_INVERTED'`

napari 0.9 removed a name the plugin uses. napari-freedlc requires `napari<0.9`,
so this usually means napari was upgraded by another install, or the PyPI
napari-deeplabcut replaced napari-freedlc (re-installing `FreeDLC[gui]` does
that; see the README's Install section). Check with `pip show napari-deeplabcut`:
the location should be your napari-freedlc checkout. Re-run
`pip install -e ./napari-freedlc`.
