"""Template registration for Method 3's watermark removal.

doc_segment.py's bounded-subtractive correction (see that module's
docstring) infers per-pixel coverage from brightness alone: `c =
clip(dot(bg-obs, D)/dot(D,D), 0, 1)`. Because D is sized to the typical
mark pixel, roughly half of every instance's pixels are darker than a
"typical" mark pixel and get capped short of full recovery -- measured on
wm_testset/images/0_3fafdb3957.jpg: 51%, 47%, 18%, 49%, 46%, 45% of pixels
per instance have d > D. That capped half is what remains as a visible
grey ghost of the letterforms (mark pixels lift 213 -> 222 against paper
225, not to 225). The cap can't simply be raised: the same brightness-only
evidence makes a dense watermark stroke and a light table rule
indistinguishable from ONE observation -- removing the cap entirely was
measured to destroy 52.1% of rule pixels under gray 175 (see doc_segment's
module docstring).

This module breaks that tie with a second observation: the real mark,
extracted from an actual document and photometrically calibrated (see
scripts/wm_dataset/asset_prep.py's extract_flat_screenshot_stamp and
clean_stamp). Registering that template to a detected instance gives a
per-pixel alpha and ink colour that are KNOWN rather than inferred from
one brightness value, so the plain compositing inverse
`true = (obs - alpha*ink) / (1 - alpha)` (see `deblend` below) does the
right thing for both a mark pixel (high alpha, ink close to background ->
clears to background) and a rule pixel the mask happens to also cover
(near-zero alpha at that location -> left essentially untouched) with no
cap and no tradeoff between them.

Search strategy (deliberately NOT a brute-force scan of the page): the
segmenter's own accepted instances are used as the prior.
  1. `fit_page_registration` estimates ONE (mark_id, scale, rotation) for
     the whole page from the highest-confidence accepted instances -- the
     mark is one tiled pattern per page (see compositor.py's placement
     patterns: a single `composite()` call, hence one scale/rotation,
     governs every instance on a synthetic page), so scale and rotation
     are shared and only translation varies per instance. This keeps the
     search tractable: the expensive (scale x rotation) grid only runs on
     a handful of instances, not all of them, and not brute-force over the
     page.
  2. `register_instance` then refines only TRANSLATION for every other
     instance, in a small window around that instance's own box centre,
     at the page-level scale/rotation already fixed. It still tries both
     registered marks (`ariatender_wide`, `ariatender_stacked`) per
     instance, since the page-level fit picks the better AGGREGATE mark
     but the segmenter itself never records which mark it found for any
     one instance.

Matching is done via normalised cross-correlation (cv2.matchTemplate,
TM_CCOEFF_NORMED) between the observed local darkening `d = local_bg -
observed` (luminance) and the candidate-warped template's alpha channel.
NCC is invariant to the unknown scalar opacity multiplier and to an
additive offset, which is exactly what makes it usable here: `d` also
carries whatever real document content sits under the mask, and the
template's alpha is only known up to the eventual opacity fit (see
`_fit_opacity`) -- a plain dot-product or SSD score would confound both of
those with genuine misregistration.
"""

import concurrent.futures
import math
import os
import sys
import time

import cv2
import numpy as np

# --- template loading ------------------------------------------------------
#
# scripts/wm_dataset/compositor.py + asset_prep.py already implement (and,
# for asset_prep, calibrate) exactly the stamp extraction this module needs
# -- see the task's own worked numbers (full-coverage alpha 0.3125/0.2222,
# round-trip mean|err| 1.11) for why those functions are trusted rather than
# re-derived here. They are scripts (no package `__init__.py`), and
# compositor.py itself does a bare `import asset_prep`, so they can only be
# imported with scripts/wm_dataset on sys.path -- done lazily, once, inside
# `_load_templates` rather than at module import time, so a Watermark-Remover
# import never has a hard dependency on that sibling scripts/ directory
# unless template registration is actually used.
_WM_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "wm_dataset"
)

_TEMPLATES = None  # dict mark_id -> {"ink": HxWx3 float64, "alpha": HxW float64 [0,1]}, or {} on failure
_TEMPLATE_LOAD_ERROR = None  # str or None -- set once, on the first (and only) load attempt


def _load_templates():
    """Lazily loads and caches both registered marks' calibrated RGBA
    stamps as (ink, alpha) float arrays, via compositor.load_base_stamp --
    mirrors segmenter.py's get_yolo_model/get_sam_model caching pattern.

    Never raises: a missing scripts/wm_dataset checkout or missing
    _wm_extract_deliverable source assets (compositor.DEFAULT_ASSETS_DIR,
    one level above the repo root -- see that module's docstring) is a real
    possibility in some deployments, and the whole point of this feature is
    that it degrades to doc_segment.py's existing bounded-subtractive path
    when unavailable, not that it crashes Method 3. Failure is cached too
    (as `{}`), so a broken environment doesn't retry-and-fail on every call.
    """
    global _TEMPLATES, _TEMPLATE_LOAD_ERROR
    if _TEMPLATES is not None:
        return _TEMPLATES
    try:
        if _WM_DATASET_DIR not in sys.path:
            sys.path.insert(0, _WM_DATASET_DIR)
        import compositor  # noqa: PLC0415 -- see module-level comment above

        templates = {}
        for mark_id in (compositor.WIDE_MARK_ID, compositor.STACKED_MARK_ID):
            stamp = compositor.load_base_stamp(mark_id)  # RGBA PIL.Image, cached inside compositor too
            rgb = np.asarray(stamp.convert("RGB"), dtype=np.float64)
            alpha = np.asarray(stamp.split()[-1], dtype=np.float64) / 255.0
            templates[mark_id] = {"ink": rgb, "alpha": alpha}
        _TEMPLATES = templates
    except Exception as exc:  # pragma: no cover - defensive, see docstring
        _TEMPLATES = {}
        _TEMPLATE_LOAD_ERROR = str(exc)
    return _TEMPLATES


def templates_available() -> bool:
    return bool(_load_templates())


def template_load_error():
    _load_templates()
    return _TEMPLATE_LOAD_ERROR


# --- tunables ---------------------------------------------------------------
#
# Matches doc_segment.py's own _LUMA_WEIGHTS -- kept as a separate constant
# here (rather than imported) so this module has no import-time dependency
# on doc_segment, which imports THIS module.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])

# Sub-pixel registration error (translation is only searched/scored at
# integer-pixel resolution -- see _match_translation) otherwise leaves a
# one-pixel-wide sliver where the warped alpha doesn't quite line up with
# the glyph edge in the real image; deblending through that sliver at full
# strength recovers the WRONG ink-vs-background split there and shows up as
# a thin colour fringe (the pink shield ink bleeding a pixel into paper, or
# vice versa). A small blur spreads that one-pixel error over ~2px instead
# of concentrating it, which is the guard the task calls for.
_ALPHA_BLUR_SIGMA = 0.8

# Ceiling on the FITTED alpha, well below 1 so `deblend`'s division can
# never blow up. Both registered marks' calibrated full-coverage alpha is
# only 0.22-0.31 (see _load_templates' docstring / the module import-time
# self-check), and compositor.py's own opacity_mult sampling range for
# synthetic training data tops out at 1.5x that (~0.47) -- so a fitted
# alpha anywhere near 1.0 means the registration or opacity fit is wrong,
# never that the real mark is genuinely that opaque. 0.6 sits comfortably
# above the highest plausible real value while leaving `deblend`'s
# denominator (1-alpha >= 0.4) far from the danger zone.
ALPHA_CEILING = 0.6

# Registration acceptance threshold on the TM_CCOEFF_NORMED score (see
# `register_instance`). Calibrated by measurement -- see
# verify_template_registration.py's score-distribution report for the
# actual numbers this was set against; kept here as a single named
# constant so it's easy to find and re-tune without hunting through the
# matching code.
REGISTRATION_SCORE_THRESHOLD = 0.35

# Page-level (scale, rotation) fit: only the this-many highest-confidence
# accepted instances pay for the full (scale x angle) grid search --
# everyone else only ever does a cheap translation-only refine at the
# already-fixed page scale/angle (see module docstring, step 1 vs step 3).
_TOP_K_FOR_PAGE_FIT = 5

# Coarse-to-fine grids for the page-level search. Coarse first to find the
# right neighbourhood cheaply, then a short fine pass around the coarse
# winner -- a full fine-resolution grid over the whole range would cost
# ~10x more per instance for no benefit once the coarse pass has already
# localised the true optimum (verified empirically: the fine pass never
# lands outside the coarse cell adjacent to its seed on the real test
# image -- see the verification report).
_SCALE_FACTORS_COARSE = (0.75, 0.85, 0.95, 1.05, 1.15, 1.3)
_SCALE_FACTOR_FINE_SPAN = 0.12  # +/- around the coarse winner
_SCALE_FACTOR_FINE_STEP = 0.03

# The marks in this dataset are near-horizontal in the common case but the
# synthetic generator deliberately also places "diagonal" lattices at
# 25-50 degrees (see compositor.py's PATTERNS docstring) -- so the search
# must cover that range and must not hard-code 0.
_ANGLE_COARSE_DEG = tuple(range(-50, 51, 10))
_ANGLE_FINE_SPAN_DEG = 8.0
_ANGLE_FINE_STEP_DEG = 2.0

# Translation search margin, as a fraction of the (already scale/rotation-
# fixed) warped template's own size -- generous at the page-fit stage
# (translation is not yet known at all beyond the instance's box centre)
# and used identically at the per-instance refine stage, since the box
# centre is already a decent translation prior either way.
_TRANSLATION_MARGIN_FRAC = 0.35

# Below this many pixels, a least-squares opacity fit (see _fit_opacity) is
# too small a sample to trust -- mirrors doc_segment.py's
# _MIN_DARKENING_SAMPLE_PIXELS reasoning for the same class of problem.
_MIN_OPACITY_FIT_PIXELS = 20

# Floor on the warped template's larger dimension, applied ONLY at the
# page-level (scale, angle) search (_hypothesis_score) -- NOT at the
# per-instance translation refine, where the page scale is already fixed
# and a tiny instance simply keeps whatever page scale won.
#
# Necessary because NCC on a sufficiently tiny patch is not a meaningful
# geometric fit at all: shrinking the wide wordmark template to, say,
# 23x5px (scale ~0.011, seeded from a single small glyph fragment's own
# box width) destroys essentially all of its internal structure --
# resizing collapses it to a near-featureless blurry gradient blob, which
# then correlates spuriously well (measured 0.5-0.88 TM_CCOEFF_NORMED)
# against almost ANY same-sized local window, since a few-pixel patch has
# too few independent samples to discriminate a real match from noise.
# Confirmed directly on wm_testset/images/0_3fafdb3957.jpg: without this
# floor, that spurious tiny-scale hypothesis's SUMMED score across the
# top-K candidates (see the long comment above _hypothesis_score) beat
# the true scale's (~0.278, confirmed by visual inspection: one
# "ariatender_wide" occurrence spans x=17-619) summed score even after
# fixing the per-instance-independent-selection bug -- summing across
# instances rejects a hypothesis that fits ONE window well by chance, but
# not one so degenerate it fits MOST windows passably by chance. 40px is
# comfortably above the ~23px failure case and comfortably below the
# smallest real, legitimately-scaled instance seen on that page (the
# shield fragment alone warps to ~139px tall at the true page scale).
_MIN_WARPED_TEMPLATE_DIM = 40

# Thread count for scoring independent (scale, angle) hypotheses in
# _fit_scale_angle_for_mark's coarse/fine grids (see that function). Not
# tied to cv2.getNumThreads() -- measured directly (see the M3 speed
# refactor notes): this module's own registration path runs with OpenCV
# left single-threaded by torch/ultralytics, so 4 python threads each
# making their own cv2.matchTemplate/warpAffine calls (which release the
# GIL) give a real ~46-48% wall-time reduction on real pages, not just
# oversubscription of an already-parallel C++ call.
_REGISTRATION_THREADS = 4


def _luma(img_float: np.ndarray) -> np.ndarray:
    return img_float @ _LUMA_WEIGHTS


# --- template warping --------------------------------------------------------

def _warp_template(ink: np.ndarray, alpha: np.ndarray, scale: float, angle_deg: float,
                    blur_sigma: float = _ALPHA_BLUR_SIGMA):
    """Resizes then rotates (ink, alpha) about their own centre, expanding
    the canvas so nothing is cropped -- the numeric equivalent of
    compositor._rotate_scale_stamp's PIL resize+rotate(expand=True), done
    in cv2/float arrays here since this path calls it far more often (once
    per (scale, angle) hypothesis during the page-level search) and PIL's
    per-call Python-level overhead would dominate.
    """
    h0, w0 = alpha.shape
    w1 = max(1, int(round(w0 * scale)))
    h1 = max(1, int(round(h0 * scale)))
    ink_r = cv2.resize(ink, (w1, h1), interpolation=cv2.INTER_LINEAR)
    alpha_r = cv2.resize(alpha, (w1, h1), interpolation=cv2.INTER_LINEAR)

    if abs(angle_deg) < 1e-6:
        warped_ink, warped_alpha = ink_r, alpha_r
    else:
        theta = math.radians(angle_deg)
        cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
        new_w = int(math.ceil(w1 * cos_t + h1 * sin_t))
        new_h = int(math.ceil(w1 * sin_t + h1 * cos_t))
        M = cv2.getRotationMatrix2D((w1 / 2.0, h1 / 2.0), angle_deg, 1.0)
        M[0, 2] += (new_w - w1) / 2.0
        M[1, 2] += (new_h - h1) / 2.0
        warped_ink = cv2.warpAffine(ink_r, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        warped_alpha = cv2.warpAffine(alpha_r, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0.0)

    if blur_sigma > 0:
        warped_alpha = cv2.GaussianBlur(warped_alpha, (0, 0), blur_sigma)
    return warped_ink, warped_alpha


def _extract_window(arr: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Grabs an (h, w) window from `arr` at top-left (x0, y0), edge-padding
    whatever falls outside `arr`'s bounds instead of raising -- an
    instance near the page border must not crash the search, and edge
    replication is a far more neutral fill for a correlation search window
    than zeros (which would inject a fake strong edge into the score).
    """
    H, W = arr.shape[:2]
    x1, y1 = x0 + w, y0 + h
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return np.zeros((h, w), dtype=arr.dtype)
    patch = arr[cy0:cy1, cx0:cx1]
    pad_top, pad_bottom = cy0 - y0, y1 - cy1
    pad_left, pad_right = cx0 - x0, x1 - cx1
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    return patch


def _match_translation(d_window: np.ndarray, warped_alpha: np.ndarray):
    """NCC of the observed darkening window against the candidate warped
    template alpha (see module docstring for why NCC, not raw correlation).
    Returns (best_score, (best_x, best_y)) where (best_x, best_y) is the
    warped template's top-left offset WITHIN d_window. `d_window` must be
    at least as large as `warped_alpha` in both dimensions (guaranteed by
    every caller: windows are always sized as template size + margin)."""
    th, tw = warped_alpha.shape
    if d_window.shape[0] < th or d_window.shape[1] < tw:
        return -1.0, (0, 0)
    result = cv2.matchTemplate(d_window.astype(np.float32), warped_alpha.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    idx = int(np.argmax(result))
    y, x = np.unravel_index(idx, result.shape)
    return float(result[y, x]), (int(x), int(y))


def _window_for_instance(box, margin_w: int, margin_h: int):
    """Search-window top-left/size for one instance's box, centred on the
    box centre with the given margins -- shared by both the page-level
    search (large margin, translation unknown) and the per-instance refine
    (small margin, translation already close from the box centre prior)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return cx, cy


# --- page-level (scale, rotation) fit ---------------------------------------
#
# IMPORTANT design points, found by measurement (not in the original
# plan) -- both bugs below were caught by actually reading the numbers
# on wm_testset/images/0_3fafdb3957.jpg, not by inspection of the code:
#
# 1. Candidates are scored by the SUM (well, average -- see
#    fit_page_registration) of per-candidate translation-only scores
#    under ONE SHARED (scale, angle) hypothesis, never by letting each
#    candidate pick its own best (scale, angle) independently and only
#    reconciling the winners afterwards. On the test image above, the
#    page's true single "ariatender_wide" occurrence spans x=17-619
#    (confirmed by visual inspection), i.e. scale ~0.278 -- but the
#    segmenter reports it as 19 small FRAGMENTS (individual words/glyphs
#    a dense table broke its own mask into, box widths 15-130px), and
#    letting each fragment pick its own best (scale, angle) independently
#    let a physically nonsensical tiny scale (~0.014, an order of
#    magnitude off) win on several of them: a heavily downsampled
#    template has so little internal structure left that it correlates
#    well against almost any same-sized patch by chance, regardless of
#    whether the scale means anything. A wrong shared scale has no reason
#    to also fit every OTHER fragment's own window at the same time,
#    while the true scale/angle does (it's the actual geometry of the
#    one real mark) -- so scoring jointly under one hypothesis, not
#    per-candidate independently, is what makes the true answer
#    findable at all.
#
# 2. Even scored jointly, a window centred on ONE fragment's own (small)
#    box is too small to ever reach the correct GLOBAL alignment for a
#    fragment far from that placement's centre -- the true (scale, angle)
#    still lost to a wrong small one when every fragment used its own
#    box-centred window, because the correct alignment fell outside
#    several fragments' own search windows entirely and they landed on
#    spurious local optima instead. Fixed by treating the fragmented case
#    as a SEPARATE, single "large" probe (the union of the top-K boxes,
#    searched in ONE window centred on that union, not on any individual
#    fragment's box) and comparing it against the per-fragment "small"
#    probes by AVERAGE score per probe (fair regardless of how many
#    probes contributed) -- see fit_page_registration for the "small" vs
#    "large" regime split this produced, and register_instance for why
#    the "large" regime also has to override where the per-instance
#    TRANSLATION refine centres its own window (same root cause: a
#    fragment far from the union's centre still can't reach the correct
#    alignment from its own box).

def _hypothesis_score(top_k, ink: np.ndarray, alpha: np.ndarray, scale: float, angle: float):
    """Warps the template ONCE for this (scale, angle) hypothesis and
    scores it against every candidate in `top_k` (list of (bg_luma,
    img_luma, box)), each via its own translation-only search. Returns
    (summed_score, per_instance) where per_instance is a list of (score,
    temp_x0, temp_y0) in full-image pixel coordinates, parallel to
    `top_k`.
    """
    if scale <= 0.01:
        return -1.0, None
    w_ink, w_alpha = _warp_template(ink, alpha, scale, angle)
    th, tw = w_alpha.shape
    if max(th, tw) < _MIN_WARPED_TEMPLATE_DIM:
        # Too small to be a meaningful geometric fit -- see
        # _MIN_WARPED_TEMPLATE_DIM's docstring for the measured failure
        # this excludes.
        return -1.0, None
    margin_w = max(4, int(round(tw * _TRANSLATION_MARGIN_FRAC)))
    margin_h = max(4, int(round(th * _TRANSLATION_MARGIN_FRAC)))
    win_w, win_h = tw + 2 * margin_w, th + 2 * margin_h

    total = 0.0
    per_instance = []
    for bg_luma, img_luma, box in top_k:
        cx, cy = _window_for_instance(box, 0, 0)
        win_x0 = int(round(cx - win_w / 2.0))
        win_y0 = int(round(cy - win_h / 2.0))
        bg_win = _extract_window(bg_luma, win_x0, win_y0, win_w, win_h)
        img_win = _extract_window(img_luma, win_x0, win_y0, win_w, win_h)
        d_win = bg_win - img_win
        score, (dx_local, dy_local) = _match_translation(d_win, w_alpha)
        total += score
        per_instance.append((score, win_x0 + dx_local, win_y0 + dy_local))
    return total, per_instance


def _score_hypotheses_ordered(top_k, ink: np.ndarray, alpha: np.ndarray, hyps: list):
    """Scores every (scale, angle) in `hyps` -- each one an independent
    _hypothesis_score call -- on a small thread pool, and returns the
    (total, per_instance) results in the SAME order as `hyps` (not
    completion order: concurrent.futures.Executor.map yields results in
    the order its inputs were given). Each hypothesis's own cv2.matchTemplate
    / cv2.warpAffine calls release the GIL, and this module's registration
    path runs with OpenCV left single-threaded by torch/ultralytics (see
    _REGISTRATION_THREADS), so this is real parallelism, not oversubscription
    of an already-threaded C++ call. No RNG is involved anywhere in
    registration (unlike doc_segment's per-page cv2.kmeans calls, which must
    never be threaded -- see that module's docstring), so scoring
    independent hypotheses out of order is safe; the caller then reduces
    the ordered results with the exact same sequential comparison the
    serial loop used, so the winner -- ties included -- is unaffected.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=_REGISTRATION_THREADS) as ex:
        return list(ex.map(lambda sa: _hypothesis_score(top_k, ink, alpha, sa[0], sa[1]), hyps))


def _fit_scale_angle_for_mark(top_k, ink: np.ndarray, alpha: np.ndarray, scale_seeds):
    """Coarse-to-fine (scale, angle) search for ONE template, maximising
    `_hypothesis_score`'s SUMMED score across `top_k` (see the module-level
    comment above for why summed, not per-instance-independent). Returns
    (best_total_score, best_scale, best_angle_deg).
    """
    best_total, best_scale, best_angle = -1.0, scale_seeds[0], 0.0

    coarse_hyps = [
        (seed_scale * factor, float(angle))
        for seed_scale in scale_seeds
        for factor in _SCALE_FACTORS_COARSE
        for angle in _ANGLE_COARSE_DEG
    ]
    coarse_results = _score_hypotheses_ordered(top_k, ink, alpha, coarse_hyps)
    for (scale, angle), (total, _) in zip(coarse_hyps, coarse_results):
        if total > best_total:
            best_total, best_scale, best_angle = total, scale, angle

    fine_scale_lo = best_scale * (1.0 - _SCALE_FACTOR_FINE_SPAN)
    fine_scale_hi = best_scale * (1.0 + _SCALE_FACTOR_FINE_SPAN)
    fine_scales = np.arange(fine_scale_lo, fine_scale_hi + 1e-9, best_scale * _SCALE_FACTOR_FINE_STEP)
    fine_angles = np.arange(best_angle - _ANGLE_FINE_SPAN_DEG, best_angle + _ANGLE_FINE_SPAN_DEG + 1e-9,
                             _ANGLE_FINE_STEP_DEG)
    fine_hyps = [(float(scale), float(angle)) for scale in fine_scales for angle in fine_angles]
    fine_results = _score_hypotheses_ordered(top_k, ink, alpha, fine_hyps)
    for (scale, angle), (total, _) in zip(fine_hyps, fine_results):
        if total > best_total:
            best_total, best_scale, best_angle = total, scale, angle

    return best_total, best_scale, best_angle


def fit_page_registration(img_np: np.ndarray, accepted_instances: list, templates: dict = None):
    """Estimates ONE (mark_id, scale, angle) for the whole page from the
    `_TOP_K_FOR_PAGE_FIT` highest-confidence accepted instances (module
    docstring, step 1).

    Runs TWO separate searches per mark and keeps whichever scores higher
    (by AVERAGE score per probe, not raw sum -- see below for why average):

    - "small": each top-K instance is its OWN probe, searched around its
      own box_width/template_width seed (module docstring step 4), in a
      window centred on ITS OWN box -- correct when the top-K instances
      really are independent occurrences (e.g. a tiled lattice of small
      marks).
    - "large": ONE probe -- the union of all top-K boxes -- searched
      around union_width/template_width, in a window centred on the
      UNION box.

    The "large" probe exists because the top-K "instances" are sometimes
    FRAGMENTS of one larger occurrence, not independent copies of the
    mark: confirmed directly on wm_testset/images/0_3fafdb3957.jpg by visual
    inspection -- one "ariatender_wide" occurrence spans x=17-619 (scale
    ~0.28), but the segmenter reports it as 19 small instances (individual
    words/glyphs a dense table broke its own mask into). A per-instance
    window centred on ONE fragment's own (small) box is too small to ever
    reach the correct GLOBAL alignment for a fragment far from that
    placement's centre -- the true alignment simply falls outside the
    search window -- so every fragment's own "small" search lands on some
    other, spuriously-good local optimum instead (small-template NCC
    scores are inflated in general: a heavily downsampled template has
    little internal structure left, so it correlates well against almost
    any same-sized patch by chance). Measured: at the correct scale/angle,
    each fragment's own small-window score averaged ~0.10, while a small
    (wrong) scale averaged ~0.22 across the same 5 fragments -- the small
    scale would win a naive comparison despite being off by an order of
    magnitude. Scoring the union region as ONE probe in a window sized to
    actually contain the true placement finds a clean, sharply localised
    peak at the correct scale/angle (score ~0.34, essentially tied across
    a tight neighbourhood and falling off outside it -- see
    verify_template_registration.py's report), which is what makes it a
    fair, comparable alternative once compared by PER-PROBE AVERAGE
    rather than raw sum (a sum of 5 small scores vs. a "sum" of 1 large
    score is not otherwise comparable).

    This does NOT special-case which answer is "right" -- a genuine
    lattice of small independent tiles has no real large mark spanning
    their union, so the "large" probe there should (and does, by the same
    logic) score worse than the correct "small" probe.

    Returns {"available": False} if there are no accepted instances, the
    templates failed to load, or every (mark, scale, angle) hypothesis was
    rejected by _hypothesis_score for every candidate mark (e.g. the page's
    accepted instances are all too small to fit any template against);
    otherwise {"available": True, "mark_id", "scale", "angle_deg", "score",
    "regime" ("small" or "large"), "candidate_scores"} (the last two purely
    for the verification report -- not consumed by register_instance).
    """
    if templates is None:
        templates = _load_templates()
    if not templates or not accepted_instances:
        return {"available": False}

    img_luma = _luma(img_np.astype(np.float64))
    ranked = sorted(accepted_instances, key=lambda inst: -inst["conf"])[:_TOP_K_FOR_PAGE_FIT]

    top_k = []
    for inst in ranked:
        bg = inst.get("background")
        bg_luma = _luma(bg.astype(np.float64)) if bg is not None else img_luma
        top_k.append((bg_luma, img_luma, inst["box"]))

    union_x1 = min(inst["box"][0] for inst in ranked)
    union_y1 = min(inst["box"][1] for inst in ranked)
    union_x2 = max(inst["box"][2] for inst in ranked)
    union_y2 = max(inst["box"][3] for inst in ranked)
    union_w = max(1.0, union_x2 - union_x1)
    # Representative local background for the union probe: the top-K
    # instance with the largest own box area, on the premise that a
    # bigger fragment gives the most reliable local-ring estimate (see
    # segmenter.local_ring_background) -- reused here rather than trying
    # to synthesise one composite background over a region several
    # instances' local rings only partially cover.
    areas = [(inst["box"][2] - inst["box"][0]) * (inst["box"][3] - inst["box"][1]) for inst in ranked]
    rep_bg_luma = top_k[int(np.argmax(areas))][0]
    union_probe = [(rep_bg_luma, img_luma, (union_x1, union_y1, union_x2, union_y2))]

    own_widths = [max(1.0, inst["box"][2] - inst["box"][0]) for inst in ranked]

    best_mark_id, best_avg, best_scale, best_angle, best_regime = None, -1.0, None, None, None
    candidate_scores = {}
    for mark_id, tmpl in templates.items():
        tmpl_w = tmpl["alpha"].shape[1]

        small_seeds = sorted({w / tmpl_w for w in own_widths})
        total_small, scale_small, angle_small = _fit_scale_angle_for_mark(
            top_k, tmpl["ink"], tmpl["alpha"], small_seeds)
        avg_small = total_small / len(top_k) if total_small >= 0 else -1.0

        total_large, scale_large, angle_large = _fit_scale_angle_for_mark(
            union_probe, tmpl["ink"], tmpl["alpha"], [union_w / tmpl_w])
        avg_large = total_large if total_large >= 0 else -1.0

        if avg_large >= avg_small:
            avg, scale, angle, regime = avg_large, scale_large, angle_large, "large"
        else:
            avg, scale, angle, regime = avg_small, scale_small, angle_small, "small"

        candidate_scores[mark_id] = {"small_avg": avg_small, "large_avg": avg_large}
        if avg > best_avg:
            best_mark_id, best_avg, best_scale, best_angle, best_regime = mark_id, avg, scale, angle, regime

    if best_mark_id is None:
        # Every hypothesis for every candidate mark was rejected by
        # _hypothesis_score (warped template under _MIN_WARPED_TEMPLATE_DIM,
        # or scale <= 0.01) -- e.g. the page's accepted instances are all too
        # small to fit any template against. best_scale/best_angle are still
        # None here; returning an "available" result would hand
        # register_instance a None scale and crash in _warp_template. There
        # is no usable page registration, so report that instead.
        return {"available": False, "candidate_scores": candidate_scores}

    result = {
        "available": True,
        "mark_id": best_mark_id,
        "scale": best_scale,
        "angle_deg": best_angle,
        "score": best_avg,
        "regime": best_regime,
        "candidate_scores": candidate_scores,
    }

    if best_regime == "large":
        # The whole point of the "large" regime is that every top-K
        # fragment shares not just scale/angle but this exact PLACEMENT
        # too (they're pieces of the same one occurrence) -- recover the
        # union probe's own winning translation here so register_instance
        # can anchor every instance's refine window on it directly,
        # instead of on that instance's own (potentially far-away, see
        # the module-level comment above) box centre.
        tmpl = templates[best_mark_id]
        _, per = _hypothesis_score(union_probe, tmpl["ink"], tmpl["alpha"], best_scale, best_angle)
        if per:
            result["anchor_x"], result["anchor_y"] = per[0][1], per[0][2]

    return result


# --- opacity fit + deblend ---------------------------------------------------

def deblend(obs: np.ndarray, ink: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Inverse of the standard compositing equation
    `observed = alpha*ink + (1-alpha)*true`, given KNOWN per-pixel alpha
    and ink (from registration), rather than doc_segment.py's bounded
    subtractive guess. `alpha` carries no trailing channel axis; `obs`/
    `ink` do -- broadcasts across channels so a single per-pixel alpha
    still recovers each channel's own true value (obs/ink already differ
    per channel wherever the template's ink is non-grey, e.g. the pink
    shield). Denominator is floored at 1e-3 purely as a last-resort divide
    guard; ALPHA_CEILING is the real guard (see that constant's docstring)
    and keeps this floor from ever actually being hit on a real fit.
    """
    denom = np.clip(1.0 - alpha, 1e-3, None)
    return np.clip((obs - alpha * ink) / denom, 0.0, 255.0)


def _fit_opacity(bg_luma_px: np.ndarray, obs_luma_px: np.ndarray,
                  ink_luma_px: np.ndarray, tmpl_alpha_px: np.ndarray) -> float:
    """Least-squares opacity multiplier k in
    `observed_darkening ~= k * template_alpha * (background - ink)`
    (task-specified fit), i.e. k = <x, y> / <x, x> with
    x = template_alpha * (bg_luma - ink_luma), y = bg_luma - obs_luma, over
    one instance's mask pixels. Returns 0.0 if the sample is too small or
    degenerate (x effectively zero everywhere -- e.g. the registered patch
    barely overlaps the mask) rather than dividing by ~0.
    """
    if tmpl_alpha_px.size < _MIN_OPACITY_FIT_PIXELS:
        return 0.0
    x = tmpl_alpha_px * (bg_luma_px - ink_luma_px)
    y = bg_luma_px - obs_luma_px
    denom = float(np.dot(x, x))
    if denom < 1e-6:
        return 0.0
    k = float(np.dot(x, y) / denom)
    return max(0.0, k)


def register_instance(img_np: np.ndarray, inst: dict, page_reg: dict, mask_bool: np.ndarray,
                       templates: dict = None, page_luma: np.ndarray = None):
    """Registers one accepted instance against the page-level (scale,
    angle) already fixed by `fit_page_registration`.

    Two modes, matching that function's "small"/"large" regime split (see
    the long comment above `_hypothesis_score` for why the split exists):

    - regime == "large": every top-K candidate was a FRAGMENT of the same
      one occurrence, so `page_reg` already carries that occurrence's
      exact placement (`anchor_x`/`anchor_y`, from the union-region fit --
      far more reliable than anything a single small fragment's own
      window could find, see that comment). Every instance reuses this
      SAME placement directly, with only a small jitter margin (not the
      instance's own box centre) -- searching a fresh, fragment-centred
      window here would reintroduce exactly the bug that made the union
      probe necessary in the first place (the true placement can fall
      outside a small fragment's own window). Only the page's own winning
      mark is tried; trying the other one has no basis once every
      instance is assumed to be a piece of the SAME single occurrence.
    - regime == "small" (or missing): the page-level candidates really
      were independent occurrences (e.g. a tiled lattice), so translation
      genuinely varies per instance -- refine it in a small window around
      THIS instance's own box centre (module docstring, step 3), trying
      both registered marks (step 5) since the segmenter never records
      which mark produced any one detection.

    On success, fits the opacity multiplier (`_fit_opacity`) and deblends
    (`deblend`) every pixel of `mask_bool`, IN PLACE nowhere -- returns the
    corrected values for `mask_bool`'s True pixels plus a diagnostics dict,
    mirroring doc_segment._bounded_subtractive_correct's calling
    convention so the caller assigns `cleaned[mask_bool] = corrected`.

    Returns (corrected_or_None, info). `corrected` is None (caller should
    fall back to the bounded-subtractive path) whenever: templates aren't
    available, `page_reg` isn't available, the instance's mask is empty, or
    the best NCC score across both marks is below REGISTRATION_SCORE_THRESHOLD.
    `info` always carries at least {"attempted": bool, "fell_back": bool,
    "score": float}; on success it adds {"mark_id", "scale", "angle_deg",
    "k", "alpha_mean"}.

    `page_luma`: optional pre-computed `_luma(img_np.astype(np.float64))`
    for the WHOLE page, from doc_segment.py (computed once per page rather
    than once per instance -- every instance's img_luma is identical
    anyway). When given, it's also reused to build bg_luma cheaply: bg
    differs from img_np only at the pixels local_ring_background actually
    replaced, and _luma is a per-pixel dot product with no cross-pixel
    dependency, so `page_luma`'s value at every OTHER pixel already equals
    `_luma(bg)` there -- confirmed bit-identical against recomputing
    `_luma(bg)` over the whole array on every accepted instance of every
    test page (see the M3 speed-refactor verification notes). When
    `page_luma` is None, img_luma/bg_luma are computed exactly as before.
    """
    if templates is None:
        templates = _load_templates()
    if not templates or not page_reg.get("available"):
        return None, {"attempted": False, "fell_back": True, "score": None}

    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None, {"attempted": False, "fell_back": True, "score": None}

    bg = inst.get("background")
    img_f = img_np.astype(np.float64)
    bg_f = bg.astype(np.float64) if bg is not None else img_f
    img_luma = page_luma if page_luma is not None else _luma(img_f)
    if bg is None:
        bg_luma = img_luma
    elif page_luma is not None:
        diff = np.any(bg != img_np, axis=2)
        bg_luma = page_luma.copy()
        if np.any(diff):
            bg_luma[diff] = _luma(bg[diff].astype(np.float64))
    else:
        bg_luma = _luma(bg_f)

    scale, angle = page_reg["scale"], page_reg["angle_deg"]

    use_anchor = page_reg.get("regime") == "large" and "anchor_x" in page_reg

    if use_anchor:
        # The union-region fit already validated this placement far more
        # reliably than any one small fragment's own window ever could
        # (see the long comment above _hypothesis_score) -- reuse it
        # directly, with NO per-instance NCC re-score. Re-scoring here
        # would be actively misleading, not just redundant: this
        # fragment's own local-ring background is a flat fill valid only
        # INSIDE its own small mask, so a d_win covering the anchor's full
        # (page-scale) window is ~0 everywhere outside that small mask
        # and a real value only inside it -- NCC against the template's
        # full-width alpha structure then scores low (measured 0.03-0.16
        # across this page's 19 fragments) even though the placement
        # itself is exactly correct, because NCC is penalising a mismatch
        # in AREA COVERED, not a placement error. The opacity fit's own
        # k<=0 guard below is the right gate for this mode: it fails
        # exactly when this fragment's mask doesn't usefully overlap the
        # template's ink at this (correct) placement, which is the only
        # real failure mode left once placement is already known.
        mark_id = page_reg["mark_id"]
        tmpl = templates[mark_id]
        w_ink, w_alpha = _warp_template(tmpl["ink"], tmpl["alpha"], scale, angle)
        temp_x0, temp_y0 = page_reg["anchor_x"], page_reg["anchor_y"]
        score = page_reg.get("score", REGISTRATION_SCORE_THRESHOLD)
    else:
        best = None  # (score, mark_id, warped_ink, warped_alpha, temp_x0, temp_y0)
        for mark_id, tmpl in templates.items():
            w_ink, w_alpha = _warp_template(tmpl["ink"], tmpl["alpha"], scale, angle)
            th, tw = w_alpha.shape
            margin_w = max(4, int(round(tw * _TRANSLATION_MARGIN_FRAC)))
            margin_h = max(4, int(round(th * _TRANSLATION_MARGIN_FRAC)))
            win_w, win_h = tw + 2 * margin_w, th + 2 * margin_h
            cx, cy = _window_for_instance(inst["box"], 0, 0)
            win_x0 = int(round(cx - win_w / 2.0))
            win_y0 = int(round(cy - win_h / 2.0))
            bg_win = _extract_window(bg_luma, win_x0, win_y0, win_w, win_h)
            img_win = _extract_window(img_luma, win_x0, win_y0, win_w, win_h)
            d_win = bg_win - img_win
            score_i, (dx_local, dy_local) = _match_translation(d_win, w_alpha)
            if best is None or score_i > best[0]:
                best = (score_i, mark_id, w_ink, w_alpha, win_x0 + dx_local, win_y0 + dy_local)

        score, mark_id, w_ink, w_alpha, temp_x0, temp_y0 = best
        if score < REGISTRATION_SCORE_THRESHOLD:
            return None, {"attempted": True, "fell_back": True, "score": score, "mark_id": mark_id,
                           "scale": scale, "angle_deg": angle}

    th, tw = w_alpha.shape
    ly = ys - temp_y0
    lx = xs - temp_x0
    valid = (ly >= 0) & (ly < th) & (lx >= 0) & (lx < tw)

    tmpl_alpha_px = np.zeros(ys.shape[0], dtype=np.float64)
    ink_px = np.zeros((ys.shape[0], 3), dtype=np.float64)
    if np.any(valid):
        tmpl_alpha_px[valid] = w_alpha[ly[valid], lx[valid]]
        ink_px[valid] = w_ink[ly[valid], lx[valid]]

    obs_px = img_f[ys, xs]
    bg_px = bg_f[ys, xs]
    obs_luma_px = img_luma[ys, xs]
    bg_luma_px = bg_luma[ys, xs]
    ink_luma_px = _luma(ink_px)

    k = _fit_opacity(bg_luma_px[valid], obs_luma_px[valid], ink_luma_px[valid], tmpl_alpha_px[valid]) \
        if np.any(valid) else 0.0

    # A geometrically good NCC score (this instance's mask really does sit
    # where the template predicts) does not guarantee a USABLE opacity fit
    # -- k clips to exactly 0.0 whenever the least-squares numerator comes
    # out non-positive (_fit_opacity), which happens e.g. when the mask is
    # dominated by pixels the warped alpha barely covers. Accepting that as
    # "success" would silently do nothing to those pixels while reporting
    # them as handled -- worse than falling back, since the bounded path
    # would have at least partially cleared them. Measured on
    # wm_testset/images/0_3fafdb3957.jpg: 8 of 19 instances hit k == 0.0
    # despite scores above threshold before this guard was added -- see
    # the verification report.
    if k <= 0.0:
        return None, {"attempted": True, "fell_back": True, "score": score, "mark_id": mark_id,
                       "scale": scale, "angle_deg": angle, "k": k}

    alpha_px = np.clip(k * tmpl_alpha_px, 0.0, ALPHA_CEILING)

    corrected = deblend(obs_px, ink_px, alpha_px[:, None]).astype(np.uint8)

    info = {
        "attempted": True, "fell_back": False, "score": score, "mark_id": mark_id,
        "scale": scale, "angle_deg": angle, "k": k,
        "alpha_mean": float(np.mean(alpha_px)) if alpha_px.size else 0.0,
    }
    return corrected, info
