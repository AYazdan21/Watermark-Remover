# wm_testset -- real hand-labelled watermark test set (placeholder)

This directory is empty on purpose. `scripts/train_segmenter.py` evaluates
the trained segmenter here, on real documents, to check for a synthetic
train/test domain gap (the training data in `wm_dataset_out/` is entirely
synthetic -- composited watermarks over stock document backgrounds). Until
this directory is populated with real hand-labelled documents, running the
training script's evaluation step against it will fail with an explicit,
instructive error rather than silently reporting all-zero metrics.

## Expected layout

```
wm_testset/
  images/
    doc001.jpg
    doc002.png
    ...
  labels/
    doc001.txt
    doc002.txt
    ...
```

Flat directories -- **no** `train/`/`val/` split subfolders (unlike
`wm_dataset_out/`, this set is only ever used for evaluation, never for
training).

- `images/`: real scanned or exported documents, `.jpg` / `.jpeg` / `.png`
  (`.bmp` / `.tif` / `.tiff` / `.webp` are also accepted). Any resolution.
- `labels/`: one `.txt` file per image, **same base filename as the image**
  (e.g. `images/doc001.jpg` <-> `labels/doc001.txt`), regardless of the
  image's extension.

## Label format (YOLO-seg)

Single class, id `0` ("watermark"). Each line in a label file is one
watermark instance, as a normalized polygon:

```
0 x1 y1 x2 y2 x3 y3 ... xn yn
```

- `0` is the class id (always `0` -- there is only one class).
- `x1 y1 x2 y2 ...` are polygon vertices, in order (need not be closed --
  don't repeat the first point at the end), each coordinate normalized to
  `[0, 1]` by dividing by the image's width (`x`) or height (`y`).
- At least 3 points (6 numbers after the class id) per line.
- A file may contain multiple lines if a document has multiple separate
  watermark instances/regions.

Example, for a triangular watermark region on a 2000x3000px image with
pixel vertices (200,300), (800,300), (500,900):

```
0 0.100000 0.100000 0.400000 0.100000 0.250000 0.300000
```

**An image with NO watermark is a negative and gets an EMPTY label
file** -- zero bytes, not a missing file and not a placeholder line. This
matches the convention already used by `wm_dataset_out/` (see
`scripts/wm_dataset/labels.py`). Negatives matter as much as positives
here: the evaluation script reports a false-positive rate computed
specifically on these negative images, because the failure mode this
whole project has fought hardest is the model touching pixels that are
not watermark.

A *missing* label file (no `.txt` at all for a given image) is also
tolerated by the evaluation script and treated the same as an empty one,
but prefer creating an explicit empty file -- it makes clear the negative
was intentional rather than an oversight.

## How to produce these polygons

Any tool that exports YOLO-seg format works (e.g. Roboflow, CVAT, Label
Studio, or `scripts/wm_dataset/labels.py`'s own `mask_to_polygons` /
`polygons_to_yolo_lines` helpers if you're going mask-first). Trace the
visible watermark region(s) as closely as practical -- the evaluation
script's pixel-level IoU/Dice metrics are only as good as this ground
truth.

## Running the evaluation

Once this directory has real images + labels:

```
../.venv/Scripts/python.exe scripts/train_segmenter.py \
    --data-dir wm_dataset_out --test-dir wm_testset \
    --model yolo11n-seg.pt --epochs 100 --imgsz 768 --batch 4
```

or, to re-score an already-trained checkpoint without retraining:

```
../.venv/Scripts/python.exe scripts/train_segmenter.py \
    --eval-only --weights runs/segment/wm_seg/weights/best.pt \
    --data-dir wm_dataset_out --test-dir wm_testset
```
