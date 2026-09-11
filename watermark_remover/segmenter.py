"""Segmentation-driven watermark detection: YOLO box detection (unioned
across two complementary fine-tuned checkpoints) refined to tight
per-instance masks with MobileSAM, then filtered for false positives using
the same alpha-unmixing machinery Method 3 uses to remove the watermark.

This module OWNS detection + mask refinement + the false-positive filter.
It touches no pixels itself -- see doc_segment.py for removal.

Background, from real experiments on dataset/document_originals (69 =
Persian web form with a thin-ring shield watermark, 70 = tender notice,
75 = tiled "CONFIDENTIAL" spreadsheet):

- yolo11s_watermark.pt and yolo11_watermark_general.pt are COMPLEMENTARY:
  the general model is far stronger on the tiled spreadsheet (75) but
  misses the tender notice (70) entirely, which the s model catches.
  Both are run and their boxes unioned, then deduplicated with NMS.
- Detection is precise but under-recalls (misses watermark instances
  overlapping dense text) and throws a handful of false positives on
  document/screenshot UI chrome (title bars, sheet tabs, status bars,
  column headers) -- see _OPAQUE_REJECT_ALPHA below for how those are
  told apart from real watermark instances.
- MobileSAM, box-prompted, tightens a coarse axis-aligned YOLO box down
  to an oriented envelope hugging the actual mark -- roughly 15-25% of
  the box's area for a real watermark word. This is what makes it safe
  to unmix inside the mask without flattening real text: a box alone is
  much too coarse for that.

A third, later-added model -- weights/best-yolo11-seg.pt, display name
"Finetuned (AriaTender)" -- takes a different shape entirely: it is a
yolo11n-SEG checkpoint (task=segment, single class {0: 'watermark'})
finetuned on synthetic composites of the two AriaTender marks over varied
Persian document pages, trained at imgsz 1024. Because it is a
segmentation model, it emits an instance mask directly per detection --
there is no axis-aligned box to refine, so this path skips _detect_boxes
and _sam_refine entirely (see detect_watermark_masks's routing and
_detect_seg_masks below). Measured over the 73 real images in
wm_testset/images at conf 0.25/imgsz 1024: 72/73 produced detections,
mean 19 instances/image, mean page coverage 4.96%, and visual inspection
of 6 pages confirmed masks land on the watermark glyph/wordmark without
firing on body text or table rules -- the failure mode that made the two
legacy detectors' raw output unusable before MobileSAM+the filter below
were added (they flagged 40/40 clean pages). Because this model's masks
hug the glyph strokes far more tightly than a MobileSAM box envelope
does, the opaque-ink cutoff below needs a separate, higher value on this
path -- see _SEG_OPAQUE_REJECT_ALPHA.
"""

import os
import time

import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO

from .config import BASE_DIR
from .unmixer import estimate_mark_color, unmix_region

# --- model files -------------------------------------------------------

_MODEL_FILES = {
    "YOLO11s": "yolo11s_watermark.pt",
    "YOLO11 General": "yolo11_watermark_general.pt",
}
_SAM_WEIGHTS = "mobile_sam.pt"

# The finetuned direct-mask model. Unlike the two legacy checkpoints above
# (which sit at BASE_DIR root), this one lives in a weights/ subdirectory --
# it was dropped there separately after the legacy models were already
# wired up, so its path is assembled on its own rather than added to
# _MODEL_FILES (which every legacy caller assumes resolves directly under
# BASE_DIR).
_SEG_MODEL_NAME = "Finetuned (AriaTender)"
_SEG_MODEL_REL_PATH = os.path.join("weights", "best-yolo11-seg.pt")
_SEG_IMGSZ = 1024  # trained at this resolution on Colab -- do not change.

_yolo_models = {}
_sam_model = None
_seg_model = None


def get_yolo_model(name: str):
    """Lazily loads and caches a YOLO11 watermark detector by display name
    ("YOLO11s" or "YOLO11 General"). Mirrors photo_inpainter.get_yolo_model's
    caching pattern."""
    if name not in _yolo_models:
        path = os.path.join(BASE_DIR, _MODEL_FILES.get(name, _MODEL_FILES["YOLO11s"]))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found.")
        _yolo_models[name] = YOLO(path)
    return _yolo_models[name]


def get_sam_model():
    """Lazily loads and caches the MobileSAM box-prompted segmenter."""
    global _sam_model
    if _sam_model is None:
        path = os.path.join(BASE_DIR, _SAM_WEIGHTS)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SAM weights {path} not found.")
        _sam_model = SAM(path)
    return _sam_model


def get_seg_model():
    """Lazily loads and caches the finetuned direct-mask watermark model
    (weights/best-yolo11-seg.pt, display name "Finetuned (AriaTender)").
    Mirrors get_yolo_model's caching pattern, but resolves under weights/
    instead of BASE_DIR -- see _SEG_MODEL_REL_PATH."""
    global _seg_model
    if _seg_model is None:
        path = os.path.join(BASE_DIR, _SEG_MODEL_REL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Finetuned segmentation model file {path} not found.")
        _seg_model = YOLO(path)
    return _seg_model


# --- box union + NMS -----------------------------------------------------

_NMS_IOU = 0.5


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = _NMS_IOU):
    """Greedy NMS, highest confidence first. Used to dedupe the union of
    two detectors' boxes where they agree on the same instance."""
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        idxs = rest[iou <= iou_thr]
    return keep


def _models_for_choice(model_choice: str):
    if model_choice == "YOLO11s":
        return ["YOLO11s"]
    if model_choice == "YOLO11 General":
        return ["YOLO11 General"]
    return ["YOLO11s", "YOLO11 General"]  # "Both (Union)" and any other value


def _detect_boxes(img_np: np.ndarray, conf: float, model_choice: str, device: str):
    """Runs the selected model(s), returns parallel lists (boxes Nx4, confs,
    source model name) after unioning and NMS-deduping across models."""
    all_boxes, all_scores, all_src = [], [], []
    for name in _models_for_choice(model_choice):
        model = get_yolo_model(name)
        results = model(img_np, conf=conf, imgsz=800, device=device, verbose=False)
        for b in results[0].boxes:
            all_boxes.append(b.xyxy[0].cpu().numpy().tolist())
            all_scores.append(float(b.conf[0]))
            all_src.append(name)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), []

    boxes_arr = np.array(all_boxes, dtype=np.float64)
    scores_arr = np.array(all_scores, dtype=np.float64)
    keep = _nms(boxes_arr, scores_arr, _NMS_IOU)
    return boxes_arr[keep], scores_arr[keep], [all_src[i] for i in keep]


# --- SAM refinement -----------------------------------------------------

_BOX_MARGIN = 4  # px of context given to SAM around each YOLO box


def _sam_refine(img_np: np.ndarray, boxes: np.ndarray, device: str):
    """Box-prompts MobileSAM for every box in one call so mask order lines
    up with `boxes`. Returns (list_of_uint8_masks, error_or_None). Never
    raises -- a SAM failure is reported back so the caller can fall back to
    the raw boxes instead of crashing (the whole point of the mask being a
    *refinement*, not a hard dependency)."""
    h, w = img_np.shape[:2]
    if len(boxes) == 0:
        return [], None
    try:
        sam = get_sam_model()
        padded = boxes.copy()
        padded[:, 0] = np.clip(padded[:, 0] - _BOX_MARGIN, 0, w)
        padded[:, 1] = np.clip(padded[:, 1] - _BOX_MARGIN, 0, h)
        padded[:, 2] = np.clip(padded[:, 2] + _BOX_MARGIN, 0, w)
        padded[:, 3] = np.clip(padded[:, 3] + _BOX_MARGIN, 0, h)
        res = sam(img_np, bboxes=padded.tolist(), device=device, verbose=False)
        if res[0].masks is None:
            return [np.zeros((h, w), dtype=np.uint8) for _ in boxes], "SAM returned no masks"
        raw = res[0].masks.data.cpu().numpy()  # (N, H', W') bool, may be at model's own resolution
        masks = []
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for m in raw:
            m8 = (m.astype(np.uint8)) * 255
            if m8.shape[:2] != (h, w):
                m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_NEAREST)
            # Light close + 1px dilate: covers anti-aliased mark edges without
            # meaningfully growing onto neighboring real content.
            m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, close_k)
            m8 = cv2.dilate(m8, close_k)
            masks.append(m8)
        return masks, None
    except Exception as exc:  # pragma: no cover - defensive, see docstring
        return [np.zeros((h, w), dtype=np.uint8) for _ in boxes], str(exc)


def _boxes_to_masks(img_np: np.ndarray, boxes: np.ndarray):
    """Fallback when SAM isn't used/available: each box becomes a filled
    rectangle mask. Much coarser than a SAM refinement -- expected to be
    less safe to unmix inside -- but keeps the pipeline functional."""
    h, w = img_np.shape[:2]
    masks = []
    for x1, y1, x2, y2 in boxes:
        m = np.zeros((h, w), dtype=np.uint8)
        m[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))] = 255
        masks.append(m)
    return masks


# --- direct-mask model path (finetuned) ----------------------------------


def _detect_seg_masks(img_np: np.ndarray, conf: float, device: str):
    """Runs the finetuned direct-mask model at imgsz=1024 (the resolution
    it was trained at -- unlike _detect_boxes's hardcoded imgsz=800 for the
    two legacy detectors, which is untouched by this addition) and returns
    parallel lists: masks (list of uint8 HxW, 0/255, already resized to
    the input image and thresholded), boxes (Nx4), scores, source (the
    model's display name, repeated -- there is only one model on this
    path, so _nms's cross-model dedup has nothing to do).

    ultralytics returns `masks.data` as an (N, h', w') array at the model's
    own (letterboxed) inference resolution, NOT the input image size --
    confirmed directly (a 713x788 input produced masks at 928x1024). Each
    mask is resized to the full image with nearest-neighbor (matching how
    _sam_refine resizes its own raw masks) and re-thresholded at >127
    since resizing a binary mask can otherwise leave edge pixels at
    intermediate values.
    """
    h, w = img_np.shape[:2]
    model = get_seg_model()
    results = model(img_np, conf=conf, imgsz=_SEG_IMGSZ, device=device, verbose=False)
    r = results[0]

    if r.masks is None or len(r.masks.data) == 0:
        return [], np.zeros((0, 4)), np.zeros((0,)), []

    raw = r.masks.data.cpu().numpy()  # (N, h', w'), model resolution
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()

    masks = []
    for m in raw:
        m8 = (m.astype(np.uint8)) * 255
        if m8.shape[:2] != (h, w):
            m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_NEAREST)
        m8 = ((m8 > 127).astype(np.uint8)) * 255
        masks.append(m8)

    sources = [_SEG_MODEL_NAME] * len(masks)
    return masks, boxes, scores, sources


# --- local background + false-positive filter ----------------------------

_RING_PX = 20
_MIN_RING_PIXELS = 30


def local_ring_background(img_np: np.ndarray, inst_mask: np.ndarray, ring_px: int = _RING_PX) -> np.ndarray:
    """Estimates the background immediately behind one instance from a ring
    of pixels around it: dilate the mask, subtract the mask itself, take the
    per-channel median of what's left. Deliberately NOT a page-wide estimate
    (document_cleaner.py's flat target_bg) -- a page can have multiple zones
    (white margin vs. tinted paper vs. a gray form box), and a watermark
    instance is small enough that its own immediate surroundings are a safe,
    locally-uniform stand-in for "what's behind it", without assuming
    anything about the rest of the page.
    """
    h, w = inst_mask.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    dilated = cv2.dilate(inst_mask, k)
    ring = (dilated > 0) & (inst_mask == 0)
    if int(np.sum(ring)) < _MIN_RING_PIXELS:
        # Instance fills (almost) its whole neighborhood (e.g. touches the
        # page edge) -- fall back to inpainting from whatever border exists.
        return cv2.inpaint(img_np, inst_mask, ring_px, cv2.INPAINT_TELEA)
    ring_color = np.median(img_np[ring].reshape(-1, 3), axis=0)
    bg = img_np.astype(np.float64).copy()
    bg[inst_mask > 0] = ring_color
    return np.clip(bg, 0, 255).astype(np.uint8)


# Two independent false-positive signals, both measured on the instance's
# own local-ring background (see local_ring_background) so they cost
# nothing extra to compute once the removal machinery already runs.
#
# Calibrated against dataset/document_originals 69/70/75 at conf=0.15,
# model_choice="Both (Union)", use_sam=True (see doc_segment's verification
# notes for the exact numbers). Two clearly-different kinds of chrome false
# positive showed up on image 75 (an Excel screenshot):
#
# 1. COLORED chrome -- the green title bar's icon and its "Watermark.xlsx"
#    filename text. These sit on a saturated green background, which no
#    real document in the set ever does (plain paper, margins, and even a
#    tinted/cream page all measured near-zero saturation; the one legitimate
#    exception, the tender notice's cream paper under its own watermark,
#    measured 22). Median HSV saturation of the local background here (182
#    for both) is nowhere near the real-document range, so instances whose
#    local background is this saturated are rejected regardless of alpha
#    (their alpha alone -- 0.70-0.80 -- overlaps genuine watermark instances
#    too closely to trust alone; see below).
# 2. NEUTRAL opaque ink -- a bold column header ("Employee", median ink
#    alpha 0.81) sitting on plain white, indistinguishable by color from a
#    watermark on paper. Here alpha is the only signal, and it does
#    distinguish: every real watermark instance found -- the tiled
#    "CONFIDENTIAL" spreadsheet mark (~0.54-0.61), the tender notice's
#    diagonal text even where it overlaps real black body text (0.73, pulled
#    up by the real ink sharing the same SAM mask) -- topped out at 0.73.
#    The column header (0.81) and the title-bar icon/text (0.70-0.80, also
#    caught by the saturation rule above) sat at or above that. 0.78 sits in
#    the gap between the highest real instance (tender, 0.73) and the
#    lowest confirmed opaque-chrome instance (0.81), with the saturated
#    cases doubly covered by rule 1.
#
# Neither rule catches everything (a few low-alpha, low-saturation slivers
# on plain white -- row-number-column edges, a status bar -- slip through),
# but those score low alpha precisely because there's little real signal
# there, so unmixing them is close to a no-op; the cutoffs are aimed at the
# cases where acceptance would visibly flatten or discolor real content.
_OPAQUE_REJECT_ALPHA = 0.78
_CHROME_SATURATION_REJECT = 60.0
_INK_PERCENTILE = 70
_INK_FLOOR = 10.0
_MIN_INK_PIXELS = 15

# A third, independent rule: reject a single instance whose SAM-refined
# mask still covers an implausibly large share of the page. A real
# watermark occurrence -- one tile of a repeating pattern, one stamp, one
# logo -- is a local feature; MobileSAM box-prompted on a real one hugs it
# tightly (see the module docstring: ~15-25% of even a generously-padded
# box). On dataset/document_originals/69 (a thin-ring shield watermark
# tiled across a data table -- the hardest case in the set) the general
# model's boxes are so imprecise that SAM cannot find a tight sub-object
# and instead fills 80%+ of the box, covering 20-26% of the *entire page*
# in a single instance. Confirmed by direct diff against the input: over
# both such instances, roughly a fifth of the covered pixels changed by a
# visible amount, and the changes land squarely on real table gridlines
# and header text, not the watermark -- exactly the fabricated-gridline /
# flattened-content failure mode this rewrite exists to avoid. No real
# watermark instance across all three test documents came anywhere close
# to this share of the page (the largest, a genuine over-text instance,
# covered 0.46%), so the cap costs nothing on the cases that work and
# stops the single worst observed failure mode.
_MAX_INSTANCE_COVERAGE = 0.05

# A second opaque-ink cutoff, used only on the finetuned direct-mask path
# (model_choice == "Finetuned (AriaTender)"). _OPAQUE_REJECT_ALPHA above
# was calibrated against MobileSAM envelopes, which -- being a coarse
# box-derived shape -- include a lot of untouched paper between glyph
# strokes, so a real watermark's median ink alpha stays low even where it
# crosses darker content (this module's own docstring already notes the
# pull-up effect on the tender notice: 0.73 where the mark's SAM mask
# shared pixels with real black body text). This model's masks hug the
# glyph strokes themselves, so that same pull-up effect is much stronger:
# running _classify_instance with 0.78 over all 1389 instances found
# across the 73 wm_testset/images images (conf=0.25, imgsz=1024) rejected
# 238 (17.1%), and on 3 of those images EVERY instance was rejected --
# i.e. Method 3 would silently do nothing on them. Rendering
# accepted/rejected overlays and inspecting them directly showed the
# rejected instances sit squarely on the watermark -- parts of the
# shield/gavel glyph, and the wordmark where it crosses a dark red
# decorative border -- not on real content; they are not false positives.
# The measured alpha distribution (p50 0.257, p90 0.834, p99 0.980) shows
# the 0.78-0.96 band is exactly where mark-over-dark-content cases land,
# so 0.97 keeps them while still rejecting fully-opaque instances (>=0.97),
# which are the ones most likely to be solid real ink rather than a
# translucent overlay, mirroring the reasoning behind 0.78 above just
# recalibrated for this model's tighter masks. The deeper safety net on
# this path is doc_segment.py's existing per-pixel `is_dark_ink` guard,
# which excludes dark-ink pixels from modification regardless of which
# instance's mask covers them -- a per-pixel rule that is strictly better
# than this per-instance alpha rule, so a higher alpha cutoff here costs
# little: anything that slips through still can't touch real dark ink.
_SEG_OPAQUE_REJECT_ALPHA = 0.97


def _ink_subset(residual: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """The subset of an instance's mask most likely to be actual mark/ink,
    rather than untouched paper the mask happens to also cover (a SAM
    envelope around diagonal text includes plenty of the letter gaps too).
    Falls back to the whole mask if too little of it stands out."""
    res_in_mask = residual[mask_bool]
    if res_in_mask.size == 0 or float(res_in_mask.max()) < 1e-6:
        return mask_bool
    ink_thresh = max(_INK_FLOOR, float(np.percentile(res_in_mask, _INK_PERCENTILE)))
    ink_bool = mask_bool & (residual >= ink_thresh)
    if int(np.sum(ink_bool)) < _MIN_INK_PIXELS:
        return mask_bool
    return ink_bool


def _classify_instance(img_np: np.ndarray, inst_mask: np.ndarray, opaque_reject_alpha: float = _OPAQUE_REJECT_ALPHA):
    """Computes all false-positive-filter signals for one instance.
    Returns a dict with median_ink_alpha, bg_saturation, page_coverage,
    accepted (bool), and the background/mark_color already computed
    (reused later so doc_segment.py never has to fit the model twice).

    `opaque_reject_alpha`: the cutoff applied to median_ink_alpha below.
    Defaults to _OPAQUE_REJECT_ALPHA (the legacy MobileSAM-envelope
    calibration) so every existing caller is unaffected; the direct-mask
    path passes _SEG_OPAQUE_REJECT_ALPHA instead -- see that constant's
    comment for why the two models need different cutoffs.
    """
    mask_bool = inst_mask > 0
    h, w = inst_mask.shape[:2]
    page_coverage = float(np.sum(mask_bool)) / float(h * w)

    # Cheapest check first: an instance this large is a SAM-refinement
    # failure (see _MAX_INSTANCE_COVERAGE), not a real watermark occurrence
    # -- skip the alpha fit entirely rather than unmix across a fifth of
    # the page with an admittedly-unreliable single flat background.
    if page_coverage > _MAX_INSTANCE_COVERAGE:
        return {
            "median_ink_alpha": 0.0, "bg_saturation": 0.0, "page_coverage": page_coverage,
            "accepted": False,
            "reject_reason": f"mask too large ({page_coverage * 100:.1f}% of page >= {_MAX_INSTANCE_COVERAGE * 100:.0f}%, SAM likely failed to isolate a tight object)",
            "background": None, "mark_color": None,
        }

    bg_est = local_ring_background(img_np, inst_mask)
    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float64)
    bg_gray = cv2.cvtColor(bg_est, cv2.COLOR_RGB2GRAY).astype(np.float64)
    residual = np.abs(gray_full - bg_gray)

    ink_bool = _ink_subset(residual, mask_bool)
    mark_color = estimate_mark_color(img_np, inst_mask, residual)
    result = unmix_region(img_np, inst_mask, mark_color=mark_color, background=bg_est)

    alpha_ink = result.alpha[ink_bool]
    median_ink_alpha = float(np.median(alpha_ink)) if alpha_ink.size else 0.0

    bg_hsv = cv2.cvtColor(bg_est, cv2.COLOR_RGB2HSV)
    bg_saturation = float(np.median(bg_hsv[:, :, 1][mask_bool])) if np.any(mask_bool) else 0.0

    reject_reason = None
    if bg_saturation >= _CHROME_SATURATION_REJECT:
        reject_reason = f"colored UI chrome (local background saturation {bg_saturation:.0f} >= {_CHROME_SATURATION_REJECT:.0f})"
    elif median_ink_alpha >= opaque_reject_alpha:
        reject_reason = f"opaque ink (median alpha {median_ink_alpha:.2f} >= {opaque_reject_alpha})"

    return {
        "median_ink_alpha": median_ink_alpha,
        "bg_saturation": bg_saturation,
        "page_coverage": page_coverage,
        "accepted": reject_reason is None,
        "reject_reason": reject_reason,
        "background": bg_est,
        "mark_color": mark_color,
    }


# --- public entry point ---------------------------------------------------

def detect_watermark_masks(img_np: np.ndarray, conf: float = 0.15, model_choice: str = "Both (Union)", use_sam: bool = True):
    """Detects watermark instances and refines them to tight masks.

    Returns (mask, meta):
      mask -- uint8 HxW, 0/255. Union of ONLY the accepted instances (the
              false-positive filter has already been applied). This is the
              mask doc_segment.py's removal is scoped to.
      meta -- dict with:
        instances     -- list of per-instance dicts: box (x1,y1,x2,y2),
                          conf, source (model name), mask (uint8 HxW),
                          accepted (bool), reject_reason, median_ink_alpha
        accepted_count, rejected_count
        coverage      -- fraction of the page covered by the accepted mask
        used_sam      -- bool, whether SAM refinement actually ran
        sam_error     -- str or None
        detect_ms, refine_ms, total_ms

    ``model_choice == "Finetuned (AriaTender)"`` routes to a completely
    different, simpler path: that model emits instance masks directly (see
    _detect_seg_masks), so _detect_boxes and _sam_refine are skipped
    entirely -- there is no box stage and no separate refinement stage to
    time, so ``refine_ms`` is always 0.0 and ``used_sam`` is always False
    on this path. Everything downstream (the false-positive filter, the
    accepted-mask union, the returned shape) is identical to the legacy
    paths; only the opaque-ink cutoff passed into _classify_instance
    differs (see _SEG_OPAQUE_REJECT_ALPHA).
    """
    t0 = time.time()
    h, w = img_np.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_seg_path = model_choice == _SEG_MODEL_NAME
    opaque_reject_alpha = _SEG_OPAQUE_REJECT_ALPHA if is_seg_path else _OPAQUE_REJECT_ALPHA

    if is_seg_path:
        masks, boxes, scores, sources = _detect_seg_masks(img_np, conf, device)
        detect_ms = (time.time() - t0) * 1000
        used_sam = False
        sam_error = None
        refine_ms = 0.0
    else:
        boxes, scores, sources = _detect_boxes(img_np, conf, model_choice, device)
        detect_ms = (time.time() - t0) * 1000

    empty_mask = np.zeros((h, w), dtype=np.uint8)
    if len(boxes) == 0:
        return empty_mask, {
            "instances": [], "accepted_count": 0, "rejected_count": 0,
            "coverage": 0.0, "used_sam": False, "sam_error": None,
            "detect_ms": detect_ms, "refine_ms": 0.0, "total_ms": detect_ms,
        }

    if not is_seg_path:
        t1 = time.time()
        sam_error = None
        used_sam = False
        if use_sam:
            raw_masks, sam_error = _sam_refine(img_np, boxes, device)
            if sam_error is None:
                used_sam = True
                masks = raw_masks
            else:
                masks = _boxes_to_masks(img_np, boxes)
        else:
            masks = _boxes_to_masks(img_np, boxes)
        refine_ms = (time.time() - t1) * 1000
    # else: masks/used_sam/sam_error/refine_ms already set above by the
    # direct-mask branch -- nothing left to refine.

    instances = []
    accepted_mask = np.zeros((h, w), dtype=np.uint8)
    accepted_count = 0
    rejected_count = 0
    for box, conf_i, src, inst_mask in zip(boxes, scores, sources, masks):
        if not np.any(inst_mask > 0):
            rejected_count += 1
            instances.append({
                "box": tuple(float(v) for v in box), "conf": float(conf_i), "source": src,
                "mask": inst_mask, "accepted": False, "reject_reason": "empty mask",
                "median_ink_alpha": 0.0,
            })
            continue

        cls = _classify_instance(img_np, inst_mask, opaque_reject_alpha=opaque_reject_alpha)
        instances.append({
            "box": tuple(float(v) for v in box), "conf": float(conf_i), "source": src,
            "mask": inst_mask, "accepted": cls["accepted"], "reject_reason": cls["reject_reason"],
            "median_ink_alpha": cls["median_ink_alpha"], "bg_saturation": cls["bg_saturation"],
            "page_coverage": cls["page_coverage"],
            "background": cls["background"], "mark_color": cls["mark_color"],
        })
        if cls["accepted"]:
            accepted_mask = np.maximum(accepted_mask, inst_mask)
            accepted_count += 1
        else:
            rejected_count += 1

    coverage = float(np.mean(accepted_mask > 0))
    total_ms = (time.time() - t0) * 1000

    meta = {
        "instances": instances,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "coverage": coverage,
        "used_sam": used_sam,
        "sam_error": sam_error,
        "detect_ms": detect_ms,
        "refine_ms": refine_ms,
        "total_ms": total_ms,
    }
    return accepted_mask, meta
