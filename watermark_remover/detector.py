"""Method 4's detection stage: a plain YOLO *detection* model (boxes only,
no masks) run over a document page. This module OWNS detection and touches
NO PIXELS -- see doc_detect.py for removal and doc_debug.py's
``debug_detect_boxes`` for the inspection-only overlay that shares this
module's output.

This is deliberately much simpler than segmenter.py: there is exactly one
registered model, it always returns boxes (never masks -- the checkpoint's
head is Detect, not Segment; ``r.masks is None`` for every result), and
there is no false-positive filter here. Segmenter.py's filter
(``_classify_instance``) needs a refined per-instance MASK to measure things
like median ink alpha and background saturation inside the mark's own
shape; a bare axis-aligned box doesn't carry enough signal for that same
filter to be meaningful, so Method 4 accepts every box above the confidence
threshold as-is. (A future deblending strategy could add its own box-level
heuristic here, but none exists yet.)

Verified facts about the registered checkpoint
(weights/yolo11s-det-freeze-new-dataset.pt): ultralytics task=detect
(Detect head -- ``r.masks is None``), base yolo11s.pt, single class
{0: 'watermark'}, trained at imgsz=1024, freeze=11. On
wm_testset/images/0_000bf78605.jpg at conf=0.25, imgsz=1024 it returns 15
boxes (confs 0.34-0.95) tightly covering the "AriaTender" wordmark letters
and shield glyph. Box coordinates from ``r.boxes.xyxy`` are already in
original-image pixel space (ultralytics undoes its own letterboxing before
returning them), so no rescaling is needed here.
"""

import os
import time

import numpy as np
import torch
from ultralytics import YOLO

from .config import BASE_DIR

# --- model registry --------------------------------------------------------
#
# One entry today; more detection checkpoints can be added the same way
# segmenter.py's _DIRECT_MASK_MODELS registry grew over time -- name ->
# (path relative to BASE_DIR, training imgsz).
_DETECT_MODELS = {
    "YOLO11s Detect (Half-Frozen, New Dataset)": (os.path.join("weights", "yolo11s-det-freeze-new-dataset.pt"), 1024),
}
DETECT_MODEL_CHOICES = list(_DETECT_MODELS)
DEFAULT_DETECT_MODEL = DETECT_MODEL_CHOICES[0]

_detect_models = {}


def get_detect_model(name: str = DEFAULT_DETECT_MODEL):
    """Lazily loads and caches a detection-only YOLO model by its
    _DETECT_MODELS display name. Mirrors segmenter.get_direct_mask_model's
    caching pattern (one shared cache dict keyed by name). An unknown name
    falls back to the default model, same convention as the segmentation
    registries."""
    if name not in _detect_models:
        rel_path, _imgsz = _DETECT_MODELS.get(name, _DETECT_MODELS[DEFAULT_DETECT_MODEL])
        path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Detection model file {path} not found (model: {name!r}).")
        _detect_models[name] = YOLO(path)
    return _detect_models[name]


def detect_watermark_boxes(
    img_np: np.ndarray,
    conf: float = 0.25,
    model_choice: str = DEFAULT_DETECT_MODEL,
    box_padding: int = 0,
):
    """Runs the selected detection model and returns (instances, meta).

    instances -- list of dicts, sorted by confidence descending:
      {
        "box": (x1, y1, x2, y2)      -- ints, padded + clipped, EXCLUSIVE
                                         end (i.e. the region is
                                         img[y1:y2, x1:x2]),
        "raw_box": (x1, y1, x2, y2)  -- floats, the model's own unpadded
                                         box in original-image pixels,
        "conf": float,
        "source": model display name,
      }
    Instances whose padded box is empty after clipping to the image are
    dropped.

    meta -- dict:
      detect_ms, imgsz, model, conf, box_padding,
      coverage -- fraction of the page inside the union of the (padded)
                  boxes.

    No false-positive filter runs on this path -- see the module docstring
    for why: segmenter.py's filter needs a refined mask, and this path only
    ever has axis-aligned boxes.
    """
    t0 = time.time()
    h, w = img_np.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_choice in _DETECT_MODELS:
        source_name = model_choice
    else:
        source_name = DEFAULT_DETECT_MODEL
    _rel_path, imgsz = _DETECT_MODELS[source_name]

    model = get_detect_model(source_name)
    results = model(img_np, conf=conf, imgsz=imgsz, device=device, verbose=False)
    r = results[0]

    detect_ms = (time.time() - t0) * 1000

    instances = []
    if r.boxes is not None and len(r.boxes) > 0:
        raw_boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        for (rx1, ry1, rx2, ry2), c in zip(raw_boxes, scores):
            x1 = max(0, int(np.floor(rx1)) - box_padding)
            y1 = max(0, int(np.floor(ry1)) - box_padding)
            x2 = min(w, int(np.ceil(rx2)) + box_padding)
            y2 = min(h, int(np.ceil(ry2)) + box_padding)
            if x2 <= x1 or y2 <= y1:
                continue
            instances.append({
                "box": (x1, y1, x2, y2),
                "raw_box": (float(rx1), float(ry1), float(rx2), float(ry2)),
                "conf": float(c),
                "source": source_name,
            })

    instances.sort(key=lambda inst: -inst["conf"])

    union_mask = np.zeros((h, w), dtype=bool)
    for inst in instances:
        x1, y1, x2, y2 = inst["box"]
        union_mask[y1:y2, x1:x2] = True
    coverage = float(np.mean(union_mask)) if h * w else 0.0

    meta = {
        "detect_ms": detect_ms,
        "imgsz": imgsz,
        "model": source_name,
        "conf": conf,
        "box_padding": box_padding,
        "coverage": coverage,
    }
    return instances, meta
