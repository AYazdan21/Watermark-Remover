"""Method 4: detection-driven document watermark removal.

Where Method 3 (doc_segment.py) detects a tight per-instance MASK and
deblends only the pixels the mask actually covers, Method 4 detects plain
axis-aligned BOXES (detector.py, a Detect-head YOLO checkpoint with no
Segment head at all) and removes watermarks by working inside the box
itself. This module owns removal; detector.py owns detection and touches no
pixels.

Removal strategies are registered explicitly in STRATEGY_CHOICES below and
dispatched from the single shared implementation, ``apply_box_strategy``,
mirroring doc_segment.py's STRATEGY_CHOICES/removal_strategy pattern, so
adding a further deblending strategy later is a matter of appending to the
list and adding another ``elif``/``else`` branch inside
``apply_box_strategy`` -- nothing about detector.py or the calling
convention needs to change.

There are three strategies. The first two -- unlike the flat-box-fill this
module used to ship -- leave real content under a box alone as much as
their own math allows, and both read their background not as a single flat
colour per box but as a per-pixel background MAP (``estimate_background_map``
below): a box frequently straddles two different real zones of the page --
e.g. a grey form panel and a white text field, both under one detected
watermark box -- and a single ring-median colour for the whole box lands
BETWEEN the two (measured: 225 and 255 paper -> one box colour of 240),
which then visibly paints the darker zone's real, untouched-by-any-mark
pixels the wrong shade. The background map instead estimates, independently
for every pixel, the light background of its own row segment between
borders, so a grey panel pixel keeps reading ~225 and a white field pixel
keeps reading ~255. See ``estimate_background_map``'s docstring for exactly
how, and the "Light patches" defect this fixes. The third strategy,
STRATEGY_ALPHA_NET, doesn't use a background estimate at all -- it inverts
a trained network's predicted per-pixel watermark opacity directly (see its
own entry below and ``watermark_remover/alpha_net.py``'s module docstring
for the physics).

- STRATEGY_THRESHOLD_FILL ("Threshold + Flat Fill (per box)", the default):
  Method 1's exact anti-aliased threshold-and-flatten math (see
  ``doc_core._remove_flat_fill``), restricted to run only inside each box
  instead of over the whole page. A page-level Otsu threshold decides,
  pixel by pixel, whether a box pixel looks like watermark-on-paper (gets
  flattened to the background map's own value at that pixel) or looks
  darker than the page's own ink/paper split (left alone). Honest limits:
  anything darker than the threshold survives, so a watermark component
  that is itself darker than the page's Otsu split (e.g. a saturated
  coloured glyph) is kept unless the matching stamp-channel filter is
  chosen to make it read as bright; and light real content inside a box --
  a faint rule, faint text -- above the threshold is flattened along with
  the mark, exactly like Method 1 on the page it's restricted to (the
  background map fixes WHAT it's flattened to, not WHETHER it survives).
- STRATEGY_BOUNDED_SUBTRACTIVE ("Bounded Subtractive (per box)"): Method
  3's bounded subtractive correction (``doc_segment._estimate_darkening`` /
  ``_bounded_subtractive_correct``), adapted to a box instead of a tight
  mask. Never replaces a pixel outright -- it adds back (or subtracts, for
  a bright mark on a dark banner) at most one estimated darkening vector D,
  so a pixel can move toward its own background-map value but can never
  jump straight to it. The correction is applied only to pixels NOT
  classified as real ink (see ``apply_box_strategy``'s "Protect ink in
  Bounded Subtractive" comment below for why this differs from Method 3) --
  a won ink pixel is left byte-identical. Honest limits: D is sized to the
  TYPICAL mark pixel (an 85th percentile, same as Method 3), so the darkest
  mark pixels are capped short of full removal and leave a faint grey
  ghost -- identical trade-off to Method 3's bounded path; and pixels near
  the ink/non-ink cutoff (anti-aliased text edges, light rules just above
  the Otsu-15 line) can still shift by up to D even though they read as
  real content, because the ink guard is a hard per-pixel classification,
  not a soft one.
- STRATEGY_ALPHA_NET ("Alpha Network (per box)"): the only strategy here
  that doesn't estimate a background colour and threshold/subtract toward
  it -- it runs ``watermark_remover/alpha_net.py``'s trained ``AlphaUNet``
  once over the (context-padded) union of all boxes, then, per box, inverts
  the predicted per-pixel watermark opacity with the closed-form
  ``alpha_net.recover``: ``true = (observed - alpha*ink) / (1 - alpha)``.
  Where the network predicts alpha ~= 0 for a won pixel, the recovered
  pixel is (numerically) the observed pixel -- there is no threshold and no
  possibility of hallucinating content, only ever inverting a modelled
  linear blend. Requires a trained checkpoint at ``ALPHA_NET_WEIGHTS``
  (``weights/alpha_net_best_final.pt``, or the older name
  ``weights/alpha_net_best.pt``); without one this strategy degrades
  gracefully (page returned unchanged, with an explanatory message -- see
  ``clean_document_detect``) rather than crashing. Honest limits: quality
  depends on how well the network's synthetic training (the Kaggle
  ``wm_dataset_out`` generator, replayed for exact targets) transfers to
  real scanned pages -- its validation numbers are on synthetic crops only;
  and, like the other two
  strategies, only pixels inside a detected box's won region are ever
  touched -- a mark the detector misses is untouched here too.

HARD INVARIANT: every pixel outside the union of the (padded) detection
boxes is byte-identical to the input. Zero detections returns an unmodified
copy of the input. This is a direct consequence of only ever writing inside
`cleaned[y1:y2, x1:x2]` slices coming straight from detector.py's already
clipped, already padded box coordinates -- never touched, never verified
after the fact by a separate mask step.
"""

import os
import time

import cv2
import numpy as np
import torch

from . import alpha_net, detector
from .config import BASE_DIR
from .container_cleaner import fill_masked_area_rgb
from .doc_core import _remove_flat_fill
from .doc_segment import (
    _LUMA_WEIGHTS,
    _bounded_subtractive_correct,
    _estimate_darkening,
    _kmeans_two_tone_masks,
)

STRATEGY_THRESHOLD_FILL = "Threshold + Flat Fill (per box)"
STRATEGY_BOUNDED_SUBTRACTIVE = "Bounded Subtractive (per box)"
STRATEGY_ALPHA_NET = "Alpha Network (per box)"

# New deblending strategies get appended here and dispatched with a new
# ``elif removal_strategy == STRATEGY_...:`` branch in apply_box_strategy
# below -- see the module docstring.
STRATEGY_CHOICES = [STRATEGY_THRESHOLD_FILL, STRATEGY_BOUNDED_SUBTRACTIVE, STRATEGY_ALPHA_NET]

# Trained alpha-regression checkpoint (watermark_remover/alpha_net.py,
# train_watermark_seg_kaggle.ipynb section 10). The notebook's final-download
# cell names it alpha_net_best_final.pt; the older name alpha_net_best.pt is
# still accepted. If neither exists, STRATEGY_ALPHA_NET degrades gracefully
# (unchanged page + explanatory message, see clean_document_detect /
# debug_detect_boxes) rather than crashing.
_ALPHA_NET_CANDIDATES = [
    os.path.join(BASE_DIR, "weights", "alpha_net_best_final.pt"),
    os.path.join(BASE_DIR, "weights", "alpha_net_best.pt"),
]
ALPHA_NET_WEIGHTS = next((p for p in _ALPHA_NET_CANDIDATES if os.path.exists(p)), _ALPHA_NET_CANDIDATES[0])

_alpha_net_models = {}


def alpha_net_weights_missing_message(path: str) -> str:
    return (
        f"Alpha Network weights not found at {path}. Train them with the Kaggle "
        f"notebook (train_watermark_seg_kaggle.ipynb, section 10 -- the "
        f"alpha network), then place the downloaded checkpoint at "
        f"weights/alpha_net_best_final.pt and restart the app."
    )


def get_alpha_net_model(path: str = None):
    """Lazily loads and caches an ``alpha_net.AlphaUNet`` checkpoint, keyed
    by `path` (default ``ALPHA_NET_WEIGHTS``) so a different checkpoint
    (e.g. a test one) doesn't evict the default. Raises FileNotFoundError
    with a clear message if the file is missing -- ``clean_document_detect``
    and ``debug_detect_boxes`` both check existence themselves first and
    return a friendly, non-crashing message before ever reaching this, but
    this guard covers any other/direct caller too."""
    path = path or ALPHA_NET_WEIGHTS
    if path not in _alpha_net_models:
        if not os.path.exists(path):
            raise FileNotFoundError(alpha_net_weights_missing_message(path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _alpha_net_models[path] = alpha_net.load_model(path, device=device)
    return _alpha_net_models[path]

# Bounded Subtractive (per box) only: minimum signed luminance difference
# (background vs. observed, in the mark's own direction) for a won box
# pixel to count as a "mark candidate" when ESTIMATING D. A tight
# per-instance mask (Method 3) is mostly mark pixels, so estimating D over
# every non-ink pixel works there; a box is mostly plain paper between
# strokes, so without this floor the candidate sample would be dominated by
# paper noise around 0, dragging _estimate_darkening's median-based trim
# toward ~0 and making the correction do nothing. Restricting the
# ESTIMATION sample to pixels that are genuinely darker/brighter than the
# fill colour (not the pixels actually corrected -- every won pixel still
# gets corrected) fixes that without changing what gets touched.
_MARK_NOISE_FLOOR = 3.0


def estimate_background_map(img_np: np.ndarray, instances: list) -> np.ndarray:
    """Computes a PER-PIXEL background estimate for every pixel inside any
    detected box, WITHOUT touching any pixels of the real image. This is
    the single source of truth for Method 4's background -- both
    apply_box_strategy (removal) and doc_debug.debug_detect_boxes
    (inspection) call this exact function, so the debugger always shows
    precisely what removal will use.

    WHY per-pixel instead of Method 4's old one-colour-per-box
    (``plan_box_fills``, removed): a single detected box frequently
    straddles two genuinely different real zones of the page -- e.g. a grey
    form panel (RGB ~225) and an adjacent white text field (RGB ~255).
    ``plan_box_fills``' ring-median put ONE colour on the whole box, which
    landed in between the two zones (measured ~240) and then visibly
    overpainted the grey panel's own untouched pixels the wrong shade --
    e.g. pixel (y=276, x=270): input 225 -> wrongly filled 240 under the
    old scheme. A per-pixel map keeps the grey panel reading ~225 and the
    white field reading ~255, because each pixel's estimate comes from its
    own row segment, not the box as a whole.

    Implementation: build the union mask of every (already padded) detected
    box, then hand it to ``container_cleaner.fill_masked_area_rgb`` with
    its default parameters. That function -- built for exactly this
    "estimate the real background under a masked region" problem for
    Method 3's container-fill strategy -- already does the right thing here
    for free: for each row, it takes the median of the LIGHT, NEUTRAL
    pixels OUTSIDE the mask in that row's own border-bounded container
    segment (falling back to a +/-12-row window, then the row, then the
    whole image, only when a segment has too few candidates), so:
      - a grey panel row keeps sampling other grey-panel pixels and a white
        field row keeps sampling other white-field pixels -- they never mix
        because a border (a rule, a box edge) between them splits them into
        separate containers;
      - every candidate pixel is, by construction, OUTSIDE every detected
        box, so watermark pixels themselves never contaminate the estimate
        (unlike naively median-ing the box's own pixels).

    Returns an HxWx3 uint8 array the same shape as img_np. Only pixels
    inside the union of detected boxes are meaningful; outside it, the
    returned value equals the input pixel (fill_masked_area_rgb leaves
    unmasked pixels untouched) and is never read by callers, since removal
    only ever indexes this map with a box's own crop.
    """
    h, w = img_np.shape[:2]

    union_mask = np.zeros((h, w), dtype=bool)
    for inst in instances:
        x1, y1, x2, y2 = inst["box"]
        union_mask[y1:y2, x1:x2] = True

    if not np.any(union_mask):
        return img_np.copy()

    return fill_masked_area_rgb(img_np, union_mask)


def apply_box_strategy(
    img_np: np.ndarray,
    instances: list,
    removal_strategy: str,
    thresh_offset: int = 0,
    anti_alias: bool = True,
    stamp_filter: str = "None (Standard)",
    background: np.ndarray = None,
):
    """The ONE Method 4 removal implementation. clean_document_detect calls
    this after detection to actually remove watermarks; doc_debug.
    debug_detect_boxes calls it too (purely to compute the cleaned array
    for its overlay diff) so the debugger always shows exactly the pixels
    removal would change.

    Instances are processed in the order given (detector.detect_watermark_
    boxes already sorts by confidence descending) with a page-wide
    `handled` mask, so a higher-confidence box's own pixels always win any
    overlap and every pixel is only ever touched by the one box that "won"
    it -- see the module docstring for the two strategies' math and honest
    limits.

    `background`, if given, is an HxWx3 uint8 array from
    estimate_background_map -- the per-pixel background BOTH strategies now
    read instead of one flat colour per box (see that function's docstring
    and the module docstring for why). If None (the default), this function
    computes it itself via estimate_background_map(img_np, instances).
    Callers that already have it (doc_debug.debug_detect_boxes, so its
    report can show the exact same map removal used) pass it in instead of
    paying to recompute it.

    Returns (cleaned_np, page_info, per_box_info):
      page_info -- dict computed once for the whole page:
        {"channel": "gray"/"red"/"blue", "otsu": int, "final_thresh": int,
         "gray_otsu": int}
        `otsu`/`final_thresh` are on the stamp_filter-selected channel
        (Threshold + Flat Fill's own math); `gray_otsu` is always the plain
        grayscale Otsu split, independent of stamp_filter, used by Bounded
        Subtractive to decide dark-ink protection regardless of which
        channel the OTHER strategy would have read.
      per_box_info -- list parallel to `instances`, both strategies include
        {"bg_median": [r,g,b] (median of the background map over this
          box's WON pixels), "bg_gray_min": int, "bg_gray_max": int (min/max
          luma of that same background-map slice -- a wide min-max, e.g.
          "225-255", is the signature of a box straddling two real zones of
          the page), "changed_px": int}, plus:
        Threshold + Flat Fill: {"replaced_frac": float}
        Bounded Subtractive: {"polarity": "dark"/"bright",
          "candidate_px": int, "n_clusters": int,
          "D": [[r,g,b], ...] (one rounded-int triple per cluster),
          "max_abs_change": int, "ink_px_kept": int (won pixels classified
          as real ink and therefore left byte-identical -- see the "Protect
          ink" comment below)}
        Alpha Network: {"alpha_mean": float, "alpha_max": float,
          "alpha_px": int (won pixels with predicted alpha > 0.02),
          "changed_px": int} -- no bg_median/bg_gray_* here, since this
          strategy never estimates a background colour at all (see the
          module docstring); page_info gains "alpha_net_weights" (the
          checkpoint path) and "alpha_net_infer_ms" (one predict_image call
          over the union-of-boxes crop, shared by every box).

    Raises ValueError for an unknown removal_strategy. For STRATEGY_ALPHA_NET
    specifically, raises FileNotFoundError if ALPHA_NET_WEIGHTS is missing --
    clean_document_detect and debug_detect_boxes both check for that
    themselves first and never actually reach this raise in normal use (see
    their own docstrings), but a direct caller of apply_box_strategy is not
    protected by that and will see the raise.
    """
    if removal_strategy not in STRATEGY_CHOICES:
        raise ValueError(
            f"Unknown Method 4 removal strategy {removal_strategy!r}. "
            f"Valid choices: {STRATEGY_CHOICES}."
        )

    h, w = img_np.shape[:2]
    cleaned = img_np.copy()
    handled = np.zeros((h, w), dtype=bool)

    if background is None:
        background = estimate_background_map(img_np, instances)

    # Page-level channel selection, exactly as doc_core.clean_document (M1)
    # picks it -- see that function's "Select channel based on stamp_filter"
    # comment. Otsu is computed ONCE for the whole page, not per box: a box
    # containing only watermark on paper would have Otsu split the
    # watermark itself into "ink" and "paper" and keep the darker half of
    # the very thing we're trying to remove.
    if stamp_filter == "Red Stamp Filter":
        channel_name = "red"
        channel = img_np[:, :, 0]
    elif stamp_filter == "Blue Stamp Filter":
        channel_name = "blue"
        channel = img_np[:, :, 2]
    else:
        channel_name = "gray"
        channel = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    otsu, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = int(otsu)
    final_thresh = int(np.clip(otsu + thresh_offset, 60, 245))

    # Independent grayscale Otsu, used only by Bounded Subtractive's
    # dark-paper polarity branch (see _get_inst_ink_and_polarity in
    # doc_segment.py, which this mirrors) -- deliberately NOT the
    # stamp_filter channel, so ink protection doesn't change depending on a
    # control that strategy has no other use for.
    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray_otsu, _ = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray_otsu = int(gray_otsu)

    page_info = {
        "channel": channel_name,
        "otsu": otsu,
        "final_thresh": final_thresh,
        "gray_otsu": gray_otsu,
    }

    # Alpha Network: run the model ONCE over the (context-padded) union of
    # all boxes, up front -- not per box -- both because the network's
    # receptive field benefits from seeing a bit past each box's own edge
    # (the 48px margin) and because re-running a whole U-Net per box would
    # be wasteful when most boxes on a page sit close together anyway.
    # Each instance below just indexes its own slice out of this shared
    # prediction.
    alpha_net_alpha_full = None
    alpha_net_recovered_full = None
    alpha_net_offset = None
    if removal_strategy == STRATEGY_ALPHA_NET:
        if not os.path.exists(ALPHA_NET_WEIGHTS):
            raise FileNotFoundError(alpha_net_weights_missing_message(ALPHA_NET_WEIGHTS))
        page_info["alpha_net_weights"] = ALPHA_NET_WEIGHTS
        page_info["alpha_net_infer_ms"] = 0.0
        if instances:
            model = get_alpha_net_model(ALPHA_NET_WEIGHTS)
            margin = 48
            ux1 = max(0, min(inst["box"][0] for inst in instances) - margin)
            uy1 = max(0, min(inst["box"][1] for inst in instances) - margin)
            ux2 = min(w, max(inst["box"][2] for inst in instances) + margin)
            uy2 = min(h, max(inst["box"][3] for inst in instances) + margin)
            union_crop = img_np[uy1:uy2, ux1:ux2]

            t_anet0 = time.time()
            alpha_net_alpha_full, _alpha_net_ink_full, alpha_net_recovered_full = alpha_net.predict_image(
                model, union_crop
            )
            page_info["alpha_net_infer_ms"] = (time.time() - t_anet0) * 1000
            alpha_net_offset = (ux1, uy1)

    per_box_info = []

    for inst in instances:
        x1, y1, x2, y2 = inst["box"]
        won = ~handled[y1:y2, x1:x2]

        orig_crop = img_np[y1:y2, x1:x2]
        bg_crop = background[y1:y2, x1:x2]

        # Shared background stats reported for BOTH strategies (see
        # apply_box_strategy's docstring) -- computed once here so the
        # debugger and the invariants a "box straddles two zones" bug would
        # violate (defect 1) are visible regardless of which strategy ran.
        won_bg = bg_crop[won]
        if won_bg.size:
            bg_median = [int(v) for v in np.round(np.median(won_bg.astype(np.float64), axis=0))]
            won_bg_gray = won_bg.astype(np.float64) @ _LUMA_WEIGHTS
            bg_gray_min = int(round(float(won_bg_gray.min())))
            bg_gray_max = int(round(float(won_bg_gray.max())))
        else:
            bg_median = [0, 0, 0]
            bg_gray_min = 0
            bg_gray_max = 0

        if removal_strategy == STRATEGY_THRESHOLD_FILL:
            channel_crop = channel[y1:y2, x1:x2]
            no_grid = np.zeros(orig_crop.shape[:2], dtype=bool)
            target_bg_crop = bg_crop

            fill_result = _remove_flat_fill(
                orig_crop, channel_crop, final_thresh, no_grid, target_bg_crop, anti_alias
            )
            cleaned[y1:y2, x1:x2][won] = fill_result[won]

            won_channel = channel_crop[won]
            replaced_frac = float(np.mean(won_channel >= final_thresh)) if won_channel.size else 0.0
            changed_px = int(np.sum(np.any(fill_result[won] != orig_crop[won], axis=-1)))
            per_box_info.append({
                "replaced_frac": replaced_frac,
                "changed_px": changed_px,
                "bg_median": bg_median,
                "bg_gray_min": bg_gray_min,
                "bg_gray_max": bg_gray_max,
            })

        elif removal_strategy == STRATEGY_BOUNDED_SUBTRACTIVE:
            gray_crop = gray_full[y1:y2, x1:x2]

            # Per-pixel background luminance (was a single scalar from the
            # box's one flat fill colour before the background-map fix) --
            # polarity is still one decision per box (the median over won
            # pixels), but ink protection below now reads bg_lum_crop
            # PIXEL BY PIXEL, so it tracks the real local background at
            # each pixel instead of one compromise value for the box.
            bg_lum_crop = bg_crop.astype(np.float64) @ _LUMA_WEIGHTS
            won_bg_lum = bg_lum_crop[won]
            is_bright_mark = bool(np.median(won_bg_lum) < 140.0) if won_bg_lum.size else False
            if is_bright_mark:
                is_ink = gray_crop.astype(np.float64) > (bg_lum_crop + 40.0)
                signed_lum_diff = (orig_crop.astype(np.float64) - bg_crop.astype(np.float64)) @ _LUMA_WEIGHTS
            else:
                is_ink = gray_crop.astype(np.float64) < (gray_otsu - 15.0)
                signed_lum_diff = (bg_crop.astype(np.float64) - orig_crop.astype(np.float64)) @ _LUMA_WEIGHTS

            # Restrict the ESTIMATION sample to mark candidates -- see the
            # module-level comment on _MARK_NOISE_FLOOR.
            candidate = won & (~is_ink) & (signed_lum_diff >= _MARK_NOISE_FLOOR)

            clusters = _kmeans_two_tone_masks(orig_crop, candidate, won)

            box_cleaned = orig_crop.copy()
            D_list = []
            for cluster_mask in clusters:
                cluster_candidate = cluster_mask & candidate
                D = _estimate_darkening(
                    orig_crop,
                    bg_crop,
                    cluster_candidate,
                    fallback_mask_bool=candidate,
                    is_bright_mark=is_bright_mark,
                )
                # Protect ink: correct only pixels this cluster owns that
                # are NOT classified as real ink. A tight Method 3 mask is
                # mostly mark pixels to begin with, so running the
                # correction on every mask pixel (ink included) barely
                # matters there -- and M3 deliberately dropped an ink guard
                # from its own correction because rule pixels straddling
                # the Otsu-15 cutoff came out dashed (half corrected, half
                # not -- see doc_segment.py's module docstring). An M4 BOX
                # is a different shape of problem: it contains lots of
                # genuine text between mark strokes, not mostly mark, so
                # without this guard every won pixel -- including solid
                # black ink -- gets the correction too. For black text on
                # paper, bg - obs is large, so the projection onto D clips
                # to c=1 and the pixel is lightened by the FULL D (measured:
                # dark text in the shield box moved from mean gray 47.1 to
                # 107.4). Excluding is_ink pixels from the write-back fixes
                # that; the known trade-off is the inverse of M3's dashing
                # risk -- an anti-aliased text edge or a light rule pixel
                # just above the is_ink cutoff still gets corrected and can
                # shift by up to D, because is_ink is a hard per-pixel
                # classification, not a soft one.
                correct_mask = cluster_mask & (~is_ink)
                if np.any(correct_mask):
                    corrected = _bounded_subtractive_correct(
                        orig_crop, bg_crop, correct_mask, D, is_bright_mark=is_bright_mark
                    )
                    box_cleaned[correct_mask] = corrected
                D_list.append([int(round(v)) for v in D])

            cleaned[y1:y2, x1:x2][won] = box_cleaned[won]

            won_before = orig_crop[won]
            won_after = box_cleaned[won]
            changed_px = int(np.sum(np.any(won_after != won_before, axis=-1)))
            if won_before.size:
                max_abs_change = int(
                    np.max(np.abs(won_after.astype(np.int32) - won_before.astype(np.int32)))
                )
            else:
                max_abs_change = 0

            per_box_info.append({
                "polarity": "bright" if is_bright_mark else "dark",
                "candidate_px": int(np.sum(candidate)),
                "n_clusters": len(clusters),
                "D": D_list,
                "changed_px": changed_px,
                "max_abs_change": max_abs_change,
                "ink_px_kept": int(np.sum(won & is_ink)),
                "bg_median": bg_median,
                "bg_gray_min": bg_gray_min,
                "bg_gray_max": bg_gray_max,
            })

        else:  # STRATEGY_ALPHA_NET
            ox, oy = alpha_net_offset
            lx1, ly1, lx2, ly2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy
            rec_box = alpha_net_recovered_full[ly1:ly2, lx1:lx2]
            alpha_box = alpha_net_alpha_full[ly1:ly2, lx1:lx2]

            box_cleaned = orig_crop.copy()
            box_cleaned[won] = rec_box[won]
            cleaned[y1:y2, x1:x2][won] = box_cleaned[won]

            won_alpha = alpha_box[won]
            changed_px = int(np.sum(np.any(box_cleaned[won] != orig_crop[won], axis=-1)))
            per_box_info.append({
                "alpha_mean": float(won_alpha.mean()) if won_alpha.size else 0.0,
                "alpha_max": float(won_alpha.max()) if won_alpha.size else 0.0,
                "alpha_px": int(np.sum(won_alpha > 0.02)),
                "changed_px": changed_px,
            })

        handled[y1:y2, x1:x2] = True

    return cleaned, page_info, per_box_info


def clean_document_detect(
    img_np: np.ndarray,
    conf: float = 0.25,
    model_choice: str = detector.DEFAULT_DETECT_MODEL,
    box_padding: int = 0,
    removal_strategy: str = STRATEGY_THRESHOLD_FILL,
    thresh_offset: int = 0,
    anti_alias: bool = True,
    stamp_filter: str = "None (Standard)",
):
    """Detects watermark boxes (detector.detect_watermark_boxes) and removes
    them with `removal_strategy` via apply_box_strategy. See the module
    docstring for the hard invariant this function preserves and for all
    three strategies' honest limits.

    Returns (cleaned_np, status) where status is a dict:
      instances_found, coverage, detect_ms, fill_ms, total_ms,
      removal_strategy, model, box_padding, thresh_offset, anti_alias,
      stamp_filter, page_info, per_box_info, changed_px, message.

    STRATEGY_ALPHA_NET without a trained checkpoint present at
    ALPHA_NET_WEIGHTS is handled here BEFORE detection even runs: this
    function returns an unmodified copy of `img_np` and a status dict whose
    `message` explains what's missing and how to produce it, rather than
    crashing (or than detecting boxes it can never actually remove).
    """
    if removal_strategy not in STRATEGY_CHOICES:
        raise ValueError(
            f"Unknown Method 4 removal strategy {removal_strategy!r}. "
            f"Valid choices: {STRATEGY_CHOICES}."
        )

    if removal_strategy == STRATEGY_ALPHA_NET and not os.path.exists(ALPHA_NET_WEIGHTS):
        message = f"Method 4 (detection-driven): {alpha_net_weights_missing_message(ALPHA_NET_WEIGHTS)}"
        status = {
            "instances_found": 0,
            "coverage": 0.0,
            "detect_ms": 0.0,
            "fill_ms": 0.0,
            "total_ms": 0.0,
            "removal_strategy": removal_strategy,
            "model": model_choice,
            "box_padding": box_padding,
            "thresh_offset": thresh_offset,
            "anti_alias": anti_alias,
            "stamp_filter": stamp_filter,
            "page_info": {},
            "per_box_info": [],
            "changed_px": 0,
            "message": message,
        }
        return img_np.copy(), status

    t0 = time.time()
    instances, meta = detector.detect_watermark_boxes(
        img_np, conf=conf, model_choice=model_choice, box_padding=box_padding
    )
    detect_ms = meta["detect_ms"]

    t_fill0 = time.time()
    cleaned, page_info, per_box_info = apply_box_strategy(
        img_np,
        instances,
        removal_strategy,
        thresh_offset=thresh_offset,
        anti_alias=anti_alias,
        stamp_filter=stamp_filter,
    )
    fill_ms = (time.time() - t_fill0) * 1000
    total_ms = (time.time() - t0) * 1000

    n_found = len(instances)
    coverage_pct = meta["coverage"] * 100
    changed_px = sum(pb.get("changed_px", 0) for pb in per_box_info)

    if n_found == 0:
        message = (
            f"Method 4 (detection-driven): no boxes detected by {meta['model']} "
            f"at conf >= {conf} (padding {box_padding} px) -- nothing was removed. "
            "Try lowering the confidence slider."
        )
    elif removal_strategy == STRATEGY_THRESHOLD_FILL:
        message = (
            f"Method 4 (detection-driven): {n_found} box(es) detected by {meta['model']} "
            f"at conf >= {conf} (padding {box_padding} px). "
            f"{STRATEGY_THRESHOLD_FILL} at threshold {page_info['final_thresh']} "
            f"(Otsu {page_info['otsu']}, offset {thresh_offset:+d}, {page_info['channel']} channel, "
            f"{'anti-aliased' if anti_alias else 'hard-thresholded'}). "
            f"Changed {changed_px} px ({coverage_pct:.2f}% of page inside boxes) "
            f"in {total_ms:.1f} ms (detect {detect_ms:.1f} ms, fill {fill_ms:.1f} ms)."
        )
    elif removal_strategy == STRATEGY_ALPHA_NET:
        infer_ms = page_info.get("alpha_net_infer_ms", 0.0)
        message = (
            f"Method 4 (detection-driven): {n_found} box(es) detected by {meta['model']} "
            f"at conf >= {conf} (padding {box_padding} px). "
            f"{STRATEGY_ALPHA_NET}: inverted the predicted per-pixel watermark opacity "
            f"(closed-form recovery, {infer_ms:.1f} ms network inference). "
            f"Changed {changed_px} px ({coverage_pct:.2f}% of page inside boxes) "
            f"in {total_ms:.1f} ms (detect {detect_ms:.1f} ms, fill {fill_ms:.1f} ms)."
        )
    else:
        n_clusters_total = sum(pb.get("n_clusters", 0) for pb in per_box_info)
        message = (
            f"Method 4 (detection-driven): {n_found} box(es) detected by {meta['model']} "
            f"at conf >= {conf} (padding {box_padding} px). "
            f"{STRATEGY_BOUNDED_SUBTRACTIVE}: corrected {n_found} box(es) across "
            f"{n_clusters_total} colour cluster(s), changed {changed_px} px "
            f"({coverage_pct:.2f}% of page inside boxes) in {total_ms:.1f} ms "
            f"(detect {detect_ms:.1f} ms, fill {fill_ms:.1f} ms)."
        )

    status = {
        "instances_found": n_found,
        "coverage": meta["coverage"],
        "detect_ms": detect_ms,
        "fill_ms": fill_ms,
        "total_ms": total_ms,
        "removal_strategy": removal_strategy,
        "model": meta["model"],
        "box_padding": box_padding,
        "thresh_offset": thresh_offset,
        "anti_alias": anti_alias,
        "stamp_filter": stamp_filter,
        "page_info": page_info,
        "per_box_info": per_box_info,
        "changed_px": changed_px,
        "message": message,
    }
    return cleaned, status
