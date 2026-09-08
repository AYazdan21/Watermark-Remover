"""Pure, dependency-light pixel-level segmentation metrics for the watermark
segmenter's train/eval script (scripts/train_segmenter.py).

Deliberately independent of ultralytics: everything here operates on plain
numpy arrays / YOLO-seg label text, so it can be unit-tested on tiny
hand-made arrays without loading a model or touching a GPU.

Why pixel metrics matter more than detection mAP for this project: the
segmenter's output feeds a pixel-level alpha-unmixing removal step
(watermark_remover/unmixer.py), so "which pixels are watermark" predicts
removal quality far better than a detection-centric mAP does. mAP rewards
"found a box that overlaps enough"; it does not punish a mask that bleeds
5px into surrounding text (which alpha-unmixing would then wrongly touch)
or one that undershoots a thin stroke (which would then be left behind).
Hence: rasterize both GT and predicted polygons to full-resolution binary
masks and score them pixel-by-pixel.

Also tracked separately: false positives on NEGATIVE images (empty label
file = no watermark present at all). The single failure mode this whole
project has fought hardest is touching pixels that are not watermark, so a
model that is accurate on positives but trigger-happy on clean documents is
still a bad model -- aggregate IoU/Dice alone would hide that (negatives
contribute no GT-positive pixels to the union, so their false positives
only show up in the pixel-precision denominator, diluted by every
true-positive pixel in the rest of the dataset).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - cv2 is a hard project dependency
    raise ImportError(
        "seg_metrics requires opencv-python (cv2), already a project dependency "
        "-- see requirements.txt"
    ) from exc


class LabelParseWarning(RuntimeWarning):
    """Raised (as a returned string, not actually raised) when a YOLO-seg
    label line is malformed. Collected rather than thrown so one bad line in
    a large hand-labelled test set doesn't abort the whole evaluation."""


def read_yolo_seg_polygons(
    label_path: Path,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> Tuple[List[np.ndarray], List[str]]:
    """Parses a YOLO-seg label file into pixel-coordinate polygons.

    Returns (polygons, warnings). `polygons` is a list of (N, 2) float32
    arrays in pixel space, one per instance of `class_id` (this dataset is
    single-class, but the filter is kept for safety). A missing or empty
    file yields ([], []) -- the correct representation of a negative
    (no-watermark) image, matching scripts/wm_dataset/labels.py's
    write_yolo_label convention.

    Malformed lines (wrong token count, too few points, unparsable floats)
    are skipped and reported as warning strings rather than raising, since
    this path also parses real hand-labelled test data the user supplies.
    """
    warnings: List[str] = []
    if not Path(label_path).exists():
        return [], warnings
    text = Path(label_path).read_text(encoding="utf-8").strip()
    if not text:
        return [], warnings

    polygons: List[np.ndarray] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except (ValueError, IndexError):
            warnings.append(f"{label_path}:{lineno}: unparsable line, skipped: {line!r}")
            continue
        if cls != class_id:
            continue
        if len(coords) % 2 != 0 or len(coords) < 6:
            warnings.append(
                f"{label_path}:{lineno}: expected an even count of >=6 normalized "
                f"coords, got {len(coords)}, skipped"
            )
            continue
        pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * img_w
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * img_h
        polygons.append(pts.astype(np.float32))
    return polygons, warnings


def polygons_to_binary_mask(
    polygons: Sequence[np.ndarray], height: int, width: int
) -> np.ndarray:
    """Rasterizes a list of pixel-coordinate (N, 2) polygons to a single
    unioned binary mask (uint8, values in {0, 1}), shape (height, width).

    Uses cv2.fillPoly, which fills self-intersecting / concave polygons
    correctly via an even-odd-ish scanline rule adequate for coverage
    masks (matches the rasterization convention already used by
    scripts/wm_dataset/labels.py for label generation, so GT rasterized
    here is consistent with how the labels were produced).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    polys_i32 = [
        np.round(p).astype(np.int32) for p in polygons if p is not None and len(p) >= 3
    ]
    if polys_i32:
        cv2.fillPoly(mask, polys_i32, 1)
    return mask


class PixelMetricAccumulator:
    """Running aggregator for pixel-level segmentation metrics over a whole
    dataset split. Aggregates raw TP/FP/FN/TN pixel counts (not per-image
    averages) so the final IoU/Dice/precision/recall are dataset-level
    (large images and dense watermark instances contribute proportionally
    to their pixel count, which is what actually matters for the
    downstream removal step operating on those same pixels).
    """

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        # Negative-image (empty-label) bookkeeping, tracked separately.
        self.neg_images_total = 0
        self.neg_images_with_fp = 0
        self.neg_fp_pixels = 0
        self.neg_total_pixels = 0
        self.pos_images_total = 0
        self.n_images = 0

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray, is_negative: bool) -> None:
        """Folds one image's GT/predicted binary masks into the running
        totals. `gt_mask` and `pred_mask` must be the same (H, W) shape,
        any dtype castable to bool. `is_negative` should be True iff the
        image's GT label file was empty (no watermark instances)."""
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(
                f"gt_mask shape {gt_mask.shape} != pred_mask shape {pred_mask.shape}"
            )
        gt = gt_mask.astype(bool)
        pred = pred_mask.astype(bool)

        tp = int(np.logical_and(gt, pred).sum())
        fp = int(np.logical_and(~gt, pred).sum())
        fn = int(np.logical_and(gt, ~pred).sum())
        tn = int(np.logical_and(~gt, ~pred).sum())

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.n_images += 1

        if is_negative:
            self.neg_images_total += 1
            self.neg_fp_pixels += fp
            self.neg_total_pixels += int(gt.size)
            if fp > 0:
                self.neg_images_with_fp += 1
        else:
            self.pos_images_total += 1

    def compute(self) -> dict:
        """Returns the aggregated metric dict. Safe to call at any point
        (e.g. on an empty accumulator, which yields zeros rather than
        raising a ZeroDivisionError)."""
        eps = 1e-9
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / (tp + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        neg_fp_rate_pixels: Optional[float] = (
            self.neg_fp_pixels / self.neg_total_pixels if self.neg_total_pixels > 0 else None
        )
        neg_fp_rate_images: Optional[float] = (
            self.neg_images_with_fp / self.neg_images_total if self.neg_images_total > 0 else None
        )

        return {
            "n_images": self.n_images,
            "n_positive_images": self.pos_images_total,
            "n_negative_images": self.neg_images_total,
            "iou": iou,
            "dice": dice,
            "pixel_precision": precision,
            "pixel_recall": recall,
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
            "tn_pixels": self.tn,
            # False positives on genuinely-empty (negative) images only --
            # the "touching pixels that aren't watermark" failure mode.
            "negative_fp_pixel_rate": neg_fp_rate_pixels,
            "negative_fp_image_rate": neg_fp_rate_images,
        }
