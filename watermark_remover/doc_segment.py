"""Method 3: segmentation-driven document watermark removal.

Detect the watermark with models (segmenter.py), refine to a tight mask,
then remove it by alpha-unmixing INSIDE that mask only. This inverts the
older approach (erase everything brighter than a global threshold, then
try to protect structure you accidentally hit), which caused a long series
of regressions: fabricated table lines, gray form boxes turned white, and
watermark fragments preserved as if they were gridlines. Method 3 instead
touches only pixels a model positively identified as watermark -- that is
its entire value, so it deliberately has no global threshold, no gridline
detection/snapping, and no bg_mode: those are exactly the machinery that
caused the regressions this rewrite exists to avoid.

HARD INVARIANT: every pixel outside the union of accepted instance masks
is byte-identical to the input -- verified directly with np.array_equal
against dataset/document_originals/{69,70,75}_original.png during
development (see the segmenter/doc_segment verification notes for results);
see tests/ (owned by another agent) for any checked-in automated version.
"""

import time

import cv2
import numpy as np

from .segmenter import detect_watermark_masks, local_ring_background
from .unmixer import unmix_region


def clean_document_segment(img_np: np.ndarray, conf: float = 0.15, model_choice: str = "Both (Union)", use_sam: bool = True):
    """Removes watermark instances found by detect_watermark_masks, one
    instance at a time, using a per-instance local background (never a
    page-wide flat estimate -- see segmenter.local_ring_background) and
    unmix_region's alpha-unmixing (never a flat replace).

    Returns (cleaned_np uint8 HxWx3, status) where status is a dict with:
      instances_found, instances_accepted, instances_rejected,
      coverage (fraction of page cleaned), used_sam, sam_error,
      detect_ms, refine_ms, unmix_ms, total_ms, message (human string)
    """
    t0 = time.time()
    mask, meta = detect_watermark_masks(img_np, conf=conf, model_choice=model_choice, use_sam=use_sam)

    cleaned = img_np.copy()
    handled = np.zeros(img_np.shape[:2], dtype=bool)

    accepted_instances = [inst for inst in meta["instances"] if inst["accepted"]]
    # Higher-confidence instances win any overlap (rare post-NMS, but SAM's
    # oriented masks can still overlap slightly where boxes from the two
    # models nearly but not quite matched) -- process most confident first
    # and only ever touch pixels no earlier (more confident) instance has
    # already resolved.
    accepted_instances.sort(key=lambda inst: -inst["conf"])

    # A detected instance's mask is an *envelope* around the watermark, so it
    # routinely also covers whatever real content sits under it. The
    # alpha-unmix model (observed = a*mark + (1-a)*paper) only holds where the
    # true content is paper: over solid ink it is not a no-op, it discolors
    # the ink. Confirmed on a real spreadsheet, where unmixing envelopes that
    # crossed table cells wiped "West"/"Central" and row numbers to a pale
    # yellow-green. A semi-transparent watermark is by definition lighter than
    # ink, so excluding solidly-dark pixels costs no real watermark coverage
    # while making it structurally impossible to damage text underneath.
    # (Same fix, same reasoning as the is_dark_text guard in document_cleaner.)
    _gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _otsu, _ = cv2.threshold(_gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    is_dark_ink = _gray_full < (float(_otsu) - 15.0)

    t_unmix0 = time.time()
    for inst in accepted_instances:
        inst_mask_bool = (inst["mask"] > 0) & (~handled) & (~is_dark_ink)
        if not np.any(inst_mask_bool):
            continue
        background = inst.get("background")
        if background is None:
            background = local_ring_background(img_np, inst["mask"])
        result = unmix_region(img_np, inst_mask_bool.astype(np.uint8) * 255, mark_color=inst.get("mark_color"), background=background)
        cleaned[inst_mask_bool] = result.recovered[inst_mask_bool]
        handled |= inst_mask_bool
    unmix_ms = (time.time() - t_unmix0) * 1000

    total_ms = (time.time() - t0) * 1000

    n_found = len(meta["instances"])
    n_accepted = meta["accepted_count"]
    n_rejected = meta["rejected_count"]
    coverage_pct = meta["coverage"] * 100

    sam_note = "MobileSAM refinement" if meta["used_sam"] else f"raw YOLO boxes (SAM unavailable: {meta['sam_error']})" if meta["sam_error"] else "raw YOLO boxes (SAM disabled)"
    message = (
        f"Method 3 (segmentation-driven): {n_found} candidate instance(s) detected, "
        f"{n_accepted} accepted / {n_rejected} rejected by the opaque-ink filter, "
        f"using {sam_note}. Cleaned {coverage_pct:.2f}% of the page in {total_ms:.1f} ms "
        f"(detect {meta['detect_ms']:.1f} ms, refine {meta['refine_ms']:.1f} ms, unmix {unmix_ms:.1f} ms)."
    )

    status = {
        "instances_found": n_found,
        "instances_accepted": n_accepted,
        "instances_rejected": n_rejected,
        "coverage": meta["coverage"],
        "used_sam": meta["used_sam"],
        "sam_error": meta["sam_error"],
        "detect_ms": meta["detect_ms"],
        "refine_ms": meta["refine_ms"],
        "unmix_ms": unmix_ms,
        "total_ms": total_ms,
        "message": message,
    }
    return cleaned, status
