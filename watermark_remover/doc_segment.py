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

Three removal-quality fixes live here (detection + the false-positive filter
are untouched, see segmenter.py):
  1. Each instance's own local-ring background (segmenter.local_ring_background)
     is used, never a page-wide flat estimate.
  2. Two-tone marks (this project's AriaTender watermark is neutral grey
     lettering plus a dusty-pink shield/gavel) are split into up to 2 color
     clusters (see _kmeans_two_tone_masks) and corrected once per cluster,
     because a single darkening direction cannot remove a second,
     differently-colored component of the same instance.
  3. **Removal no longer calls unmixer.unmix_region at all.** It used to:
     each instance was alpha-unmixed against a corrected "removal_mark_color"
     (segmenter._estimate_removal_mark_color). That broke down completely
     once the mark color was corrected: with mark ~(217,217,217) against a
     typical local background of 225, `mark_color - background` is only
     about -8, which makes unmix_region's per-pixel least-squares alpha fit
     ill-conditioned (dividing by a near-zero direction vector) -- measured
     on wm_testset/images/0_3fafdb3957.jpg, alpha clipped to 1.0 on 96.5% of
     instance pixels. Above unmix_region's own _TRUST_CEILING (0.75) it
     stops trusting the unmixed value and blends toward the flat local
     background instead, so in practice Method 3 had degenerated into a
     flat fill -- which erases whatever real content (table rules, text)
     sits under the mask, exactly the failure mode this module exists to
     avoid. Measured rule-pixel loss on that same image: 52.1% of pixels
     with gray < 175 inside the touched mask were gone; Otsu on that page
     is 151, so `is_dark_ink` only protects gray < 136, and the lost rule
     pixels measured gray 132-140 -- straddling that guard -- so half of
     each rule survived and half was painted over, leaving it visibly
     dashed.

     The fix (see _estimate_darkening / _bounded_subtractive_correct below)
     replaces the alpha-unmix with a bounded SUBTRACTIVE correction: add
     back only the mark's own estimated darkening `D` (a per-channel
     3-vector), never replace a pixel with a background estimate. The
     largest change any pixel can undergo is exactly `D` -- a rule pixel at
     138 can move to ~150-175 (still plainly a rule) but can never jump to
     225 (paper), because nothing is ever replaced outright. This also
     means the per-pixel `is_dark_ink` exclusion that used to gate which
     pixels got touched at all is no longer needed for safety (dropping it
     is what fixes the dashing -- see clean_document_segment) -- it is kept
     only for ESTIMATING `D` and the two-tone clusters, where dark real ink
     mixed into the sample would bias the estimate.

Bounding D still leaves a visible artifact of its own: D is sized to the
TYPICAL mark pixel, so roughly half of every instance's pixels (those
darker than typical) are capped short of full recovery -- a grey ghost of
the letterforms remains. Registering the real, calibrated mark template
(see template_match.py) resolves this by making per-pixel alpha KNOWN
instead of inferred from brightness, at which point mark and rule pixels
stop being a coin flip decided by "how dark is this" (see that module's
docstring for the full argument and the worked numbers). clean_document_
segment tries template registration FIRST per instance and only falls
back to the bounded correction above when registration doesn't clear
threshold -- see that function's docstring, and the ``use_template``
parameter for reverting to the bounded path everywhere at once.
"""

import time

import cv2
import numpy as np

from .segmenter import detect_watermark_masks, is_direct_mask_model, local_ring_background
from . import template_match
from .container_cleaner import fill_masked_area_rgb, telea_inpaint_masked_area

STRATEGY_TEMPLATE = "Template Deblending"
STRATEGY_CONTAINER_FILL = "Container-Aware Adaptive Fill (remover)"
STRATEGY_SUBTRACTIVE = "Bounded Subtractive"
STRATEGY_TELEA = "Telea Inpainting"

STRATEGY_CHOICES = [
    STRATEGY_TEMPLATE,
    STRATEGY_CONTAINER_FILL,
    STRATEGY_SUBTRACTIVE,
    STRATEGY_TELEA,
]

# --- two-tone split (Defect 2 fix) ----------------------------------------
#
# A single darkening vector D is one direction in RGB space, so any
# component of the mark's color perpendicular to that direction survives
# untouched. The AriaTender watermark is two-tone -- neutral grey lettering
# plus a dusty-pink shield/gavel glyph (measured ink colors ~(128,128,128)
# and ~(187,97,100)) -- so one D per instance structurally cannot fully
# remove it.
#
# Fix: split each instance's mask into at most 2 color clusters via k-means
# (k=2, RGB, fit on the instance's non-dark-ink pixels so real ink doesn't
# bias the cluster centers) and correct each cluster separately with its
# own estimated D. Two guards decide whether the split is trustworthy
# enough to use:
#
# - _CLUSTER_MIN_PIXELS: below this, a cluster's own color estimate is too
#   small a sample to trust -- merge back to a single correction for the
#   whole instance.
# - _CLUSTER_MIN_CHROMA_DIST: k-means on raw RGB reliably finds a 2-way
#   split for EVERY instance, single-tone included -- an anti-aliased glyph
#   always has a lighter edge and a darker core, so plain center-to-center
#   RGB distance cannot tell "two tones" from "one tone, two shades" apart
#   (measured raw RGB center distance across all 19 accepted instances on
#   0_3fafdb3957.jpg: 47.8-104.5, with NO separation between the two
#   genuinely two-tone instances (52.7, 47.8) and the seventeen single-tone
#   ones (50.1-104.5) -- unusable as a discriminator). Stripping each
#   center's own luminance first (center - mean(center), i.e. comparing
#   hue/tint only) fixes this cleanly: the seventeen single-tone instances
#   measured chroma distance 0.0-1.6 (their two clusters are just a
#   lighter/darker shade of the SAME grey), while the two two-tone
#   grey+pink instances measured 16.4 and 19.2 -- an order of magnitude
#   higher, with a wide gap on either side. 8.0 sits in that gap. A
#   single-tone mark must not be split into "light grey / dark grey" (which
#   would gain nothing and add risk of a visible seam at the cluster
#   boundary), so the threshold is set well above the single-tone ceiling.
_CLUSTER_MIN_PIXELS = 30
_CLUSTER_MIN_CHROMA_DIST = 8.0

# --- bounded subtractive correction (Defect 3 fix) ------------------------
#
# D is estimated as a high percentile of (background - observed) over the
# instance's own non-dark-ink pixels: the most-covered pixels in that
# subset are the best available estimate of the mark's full-coverage
# darkening. A high percentile is required, not a median/low percentile --
# most mask pixels are lightly-covered edges/anti-aliasing, so the median
# under-estimates full coverage. BUT a plain percentile over the whole
# non-dark-ink subset is itself contaminated the same way estimate_mark_color
# was (Defect 1): is_dark_ink only excludes gray < Otsu-15 (136 on this
# page), and a table rule or text edge at gray 136-140 -- legitimately
# excluded-as-ink one shade darker, but NOT excluded here -- is far darker
# than the mark itself, so it dominates the upper percentile. Measured
# directly: on wm_testset/images/0_3fafdb3957.jpg's neutral grey-wordmark
# instances, the untrimmed q90 landed at 32-36 per channel against a true
# mark darkening of ~12 (confirmed by isolating the same instance's pixels
# that DON'T overlap the rule-line region: q90 there drops back to ~12) --
# a 3x inflation entirely explained by rule pixels straddling the ink
# cutoff, not by the mark being genuinely that dark. A bounded correction
# with a 3x-too-large D still visibly fades a rule it was supposed to only
# lighten by ~12, so the contamination has to be trimmed before the
# percentile, not just bounded after it.
#
# Fix: before taking the percentile, drop pixels whose per-pixel luminance
# darkening is more than ~2x the sample's own median darkening (the sample
# is dominated by mark-over-paper pixels, so the median tracks the mark;
# real content sitting under a handful of mask pixels is the outlier).
# Confirmed this recovers ~12 for the grey wordmark instances and correctly
# keeps the pink shield instance's larger, genuine (12, 42, 41) -- its own
# median already reflects its stronger tint, so the trim doesn't cut it.
_DARKENING_PERCENTILE = 85
_TRIM_MULT = 2.0
_TRIM_OFFSET = 4.0
_MIN_TRIM_KEEP_PIXELS = 10
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])
# Below this many non-dark-ink pixels, a percentile estimate is too noisy
# to trust (mirrors segmenter._MIN_NON_INK_PIXELS's reasoning for the same
# problem on the old mark-color estimate) -- fall back to a larger, more
# stable sample: the cluster's own full pixel set (ink included) first,
# then the whole instance's non-ink pixels.
_MIN_DARKENING_SAMPLE_PIXELS = 20
# Sane ceiling on D, per channel. Guards against a corrupted estimate (a
# bad local-ring background, or a tiny/noisy sample) producing a runaway
# correction -- this is the whole point of bounding the correction in the
# first place, so the cap itself must not be able to reintroduce an
# unbounded change. 100 sits comfortably above every measured real
# instance (grey wordmark ~12, pink shield ~12-42) while still being a
# small fraction of the 0-255 range, so it costs nothing on real data and
# only fires on a genuinely pathological estimate.
_D_CEILING = 100.0


def _kmeans_two_tone_masks(img_np: np.ndarray, fit_mask_bool: np.ndarray, full_mask_bool: np.ndarray) -> list:
    """Splits full_mask_bool into at most 2 color-cluster boolean masks
    (see module-level comment above), fitting k-means on fit_mask_bool
    (the instance's non-dark-ink pixels) but assigning EVERY pixel of
    full_mask_bool -- ink included -- to its nearest cluster center, so a
    dark-ink pixel that happens to sit inside e.g. the pink glyph still
    gets that glyph's own D rather than falling through a gap between
    clusters. Returns [full_mask_bool] unchanged -- i.e. "don't split" --
    whenever there are too few pixels to cluster, a cluster comes out too
    small to trust, or the two cluster centers don't differ enough in
    hue/tint to be two genuine tones rather than one tone's light/dark
    shading.
    """
    ys, xs = np.where(fit_mask_bool)
    n = len(ys)
    if n < _CLUSTER_MIN_PIXELS * 2:
        return [full_mask_bool]

    samples = img_np[ys, xs].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(samples, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    counts = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]
    if min(counts) < _CLUSTER_MIN_PIXELS:
        return [full_mask_bool]

    chroma0 = centers[0] - float(np.mean(centers[0]))
    chroma1 = centers[1] - float(np.mean(centers[1]))
    chroma_dist = float(np.linalg.norm(chroma0 - chroma1))
    if chroma_dist < _CLUSTER_MIN_CHROMA_DIST:
        return [full_mask_bool]

    # Split is trustworthy -- assign every pixel of the FULL mask (ink
    # included) to whichever fitted center it's closer to in RGB.
    fys, fxs = np.where(full_mask_bool)
    fsamples = img_np[fys, fxs].astype(np.float32)
    d0 = np.linalg.norm(fsamples - centers[0], axis=1)
    d1 = np.linalg.norm(fsamples - centers[1], axis=1)
    assign = (d1 < d0)

    masks = []
    for want_one in (False, True):
        m = np.zeros(full_mask_bool.shape, dtype=bool)
        sel = assign == want_one
        m[fys[sel], fxs[sel]] = True
        masks.append(m)
    return masks


def _estimate_darkening(
    img_np: np.ndarray,
    background: np.ndarray,
    sample_mask_bool: np.ndarray,
    fallback_mask_bool: np.ndarray,
    q: float = _DARKENING_PERCENTILE,
    is_bright_mark: bool = False,
) -> np.ndarray:
    """Per-channel darkening/brightening vector D over sample_mask_bool
    (expected to be a non-ink subset), with content outliers trimmed.

    When `is_bright_mark` is True (e.g. translucent white watermark on dark/colored banner),
    diff is (observed - background) and D measures excess lightness to subtract.
    When `is_bright_mark` is False (standard mark on paper), diff is (background - observed)
    and D measures darkening to add back.
    """
    use_mask = sample_mask_bool if int(np.sum(sample_mask_bool)) >= _MIN_DARKENING_SAMPLE_PIXELS else fallback_mask_bool
    if not np.any(use_mask):
        return np.zeros(3, dtype=np.float64)

    if is_bright_mark:
        diff = img_np[use_mask].astype(np.float64) - background[use_mask].astype(np.float64)
    else:
        diff = background[use_mask].astype(np.float64) - img_np[use_mask].astype(np.float64)

    lum_diff = diff @ _LUMA_WEIGHTS
    med = float(np.median(lum_diff))
    keep = lum_diff <= (_TRIM_MULT * med + _TRIM_OFFSET)
    diff_for_stat = diff[keep] if int(np.sum(keep)) >= _MIN_TRIM_KEEP_PIXELS else diff

    if diff_for_stat.shape[0] < _MIN_DARKENING_SAMPLE_PIXELS:
        D = np.median(diff_for_stat, axis=0)
    else:
        D = np.percentile(diff_for_stat, q, axis=0)
    return np.clip(D, 0.0, _D_CEILING)


def _bounded_subtractive_correct(
    img_np: np.ndarray,
    background: np.ndarray,
    mask_bool: np.ndarray,
    D: np.ndarray,
    is_bright_mark: bool = False,
) -> np.ndarray:
    """Recovers pixels under `mask_bool` by bounded correction along vector `D`.

    When `is_bright_mark` is False (dark mark on light paper):
      adds back at most D (out = obs + c * D).
    When `is_bright_mark` is True (light mark on dark banner):
      subtracts at most D (out = obs - c * D).
    """
    obs = img_np[mask_bool].astype(np.float64)
    Dv = D.astype(np.float64)
    denom = float(np.dot(Dv, Dv))
    if denom < 1e-6:
        return obs.astype(np.uint8)
    bg = background[mask_bool].astype(np.float64)

    if is_bright_mark:
        diff = obs - bg
        c = np.clip((diff @ Dv) / denom, 0.0, 1.0)
        out = obs - c[:, None] * Dv[None, :]
    else:
        diff = bg - obs
        c = np.clip((diff @ Dv) / denom, 0.0, 1.0)
        out = obs + c[:, None] * Dv[None, :]

    return np.clip(out, 0, 255).astype(np.uint8)


def clean_document_segment(
    img_np: np.ndarray,
    conf: float = 0.25,
    model_choice: str = "Finetuned (AriaTender)",
    use_sam: bool = True,
    use_template: bool = True,
    removal_strategy: str = STRATEGY_TEMPLATE,
    allow_colored_bg: bool = True,
):
    """Removes watermark instances found by detect_watermark_masks.

    Supported removal strategies:
    - STRATEGY_TEMPLATE ("Template Deblending"): Registers calibrated AriaTender mark,
      deblending mathematically (clears mark completely with zero grey ghost). Falls back
      to bounded subtractive per-instance if registration score is low.
    - STRATEGY_CONTAINER_FILL ("Container-Aware Adaptive Fill (remover)"): Ports Desktop/remover's
      border-bounded local fill, preventing color leakage across cells/borders without templates.
    - STRATEGY_SUBTRACTIVE ("Bounded Subtractive"): Adds back estimated darkening vector D,
      bounded so it cannot erase rules or text.
    - STRATEGY_TELEA ("Telea Inpainting"): Fast classical inpainting inside the mask.
    """
    t0 = time.time()
    mask, meta = detect_watermark_masks(
        img_np,
        conf=conf,
        model_choice=model_choice,
        use_sam=use_sam,
        allow_colored_bg=allow_colored_bg,
    )

    cleaned = img_np.copy()
    handled = np.zeros(img_np.shape[:2], dtype=bool)

    accepted_instances = [inst for inst in meta["instances"] if inst["accepted"]]
    accepted_instances.sort(key=lambda inst: -inst["conf"])

    _gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _otsu, _ = cv2.threshold(_gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    page_dark_ink = _gray_full < (float(_otsu) - 15.0)

    def _get_inst_ink_and_polarity(inst):
        m_bool = inst["mask"] > 0
        bg = inst.get("background")
        if bg is None:
            bg = local_ring_background(img_np, inst["mask"])
        bg_g = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        bg_lum = float(np.median(bg_g[m_bool])) if np.any(m_bool) else 255.0
        if bg_lum < 140.0:
            # Dark or saturated colored background: protect solid bright foreground text
            is_ink = _gray_full > (bg_g + 40.0)
            is_bright = True
        else:
            # Light paper background: protect dark body text and table rules
            is_ink = page_dark_ink
            is_bright = False
        return bg, is_ink, is_bright

    # Respect legacy use_template=False toggle
    if not use_template and removal_strategy == STRATEGY_TEMPLATE:
        removal_strategy = STRATEGY_SUBTRACTIVE

    template_page_reg = {"available": False}
    instance_registrations = []

    t_unmix0 = time.time()

    if removal_strategy == STRATEGY_CONTAINER_FILL:
        union_mask = np.zeros(img_np.shape[:2], dtype=bool)
        protected_ink = np.zeros(img_np.shape[:2], dtype=bool)
        for inst in accepted_instances:
            m_bool = inst["mask"] > 0
            union_mask |= m_bool
            _, is_ink, _ = _get_inst_ink_and_polarity(inst)
            protected_ink |= (m_bool & is_ink)
        fill_mask = union_mask & (~protected_ink)
        cleaned = fill_masked_area_rgb(img_np, fill_mask)
        unmix_ms = (time.time() - t_unmix0) * 1000

    elif removal_strategy == STRATEGY_TELEA:
        union_mask = np.zeros(img_np.shape[:2], dtype=bool)
        protected_ink = np.zeros(img_np.shape[:2], dtype=bool)
        for inst in accepted_instances:
            m_bool = inst["mask"] > 0
            union_mask |= m_bool
            _, is_ink, _ = _get_inst_ink_and_polarity(inst)
            protected_ink |= (m_bool & is_ink)
        fill_mask = union_mask & (~protected_ink)
        cleaned = telea_inpaint_masked_area(img_np, fill_mask)
        unmix_ms = (time.time() - t_unmix0) * 1000

    else:
        do_template = (removal_strategy == STRATEGY_TEMPLATE)
        if do_template:
            template_page_reg = template_match.fit_page_registration(img_np, accepted_instances)
            # Every instance's own img_luma is the same full-page value --
            # computed once here instead of once per instance inside
            # register_instance (see that function's page_luma parameter).
            page_luma = template_match._luma(img_np.astype(np.float64))

        for inst in accepted_instances:
            full_mask_bool = (inst["mask"] > 0) & (~handled)
            if not np.any(full_mask_bool):
                continue
            background, inst_ink, is_bright_mark = _get_inst_ink_and_polarity(inst)

            used_template = False
            if do_template and not is_bright_mark:
                corrected_t, treg_info = template_match.register_instance(
                    img_np, inst, template_page_reg, full_mask_bool, page_luma=page_luma
                )
                instance_registrations.append({"box": inst["box"], "conf": inst["conf"], **treg_info})
                if corrected_t is not None:
                    cleaned[full_mask_bool] = corrected_t
                    used_template = True

            if not used_template:
                non_ink_mask_bool = full_mask_bool & (~inst_ink)
                cluster_masks = _kmeans_two_tone_masks(img_np, non_ink_mask_bool, full_mask_bool)
                for cluster_mask_bool in cluster_masks:
                    if not np.any(cluster_mask_bool):
                        continue
                    cluster_non_ink_bool = cluster_mask_bool & (~inst_ink)
                    D = _estimate_darkening(
                        img_np,
                        background,
                        cluster_non_ink_bool,
                        fallback_mask_bool=non_ink_mask_bool,
                        is_bright_mark=is_bright_mark,
                    )
                    corrected = _bounded_subtractive_correct(
                        img_np,
                        background,
                        cluster_mask_bool,
                        D,
                        is_bright_mark=is_bright_mark,
                    )
                    cleaned[cluster_mask_bool] = corrected
            handled |= full_mask_bool
        unmix_ms = (time.time() - t_unmix0) * 1000

    total_ms = (time.time() - t0) * 1000

    n_found = len(meta["instances"])
    n_accepted = meta["accepted_count"]
    n_rejected = meta["rejected_count"]
    coverage_pct = meta["coverage"] * 100

    if is_direct_mask_model(model_choice):
        sam_note = "the finetuned model's own direct instance masks (no YOLO boxes, no SAM)"
    elif meta["used_sam"]:
        sam_note = "MobileSAM refinement"
    elif meta["sam_error"]:
        sam_note = f"raw YOLO boxes (SAM unavailable: {meta['sam_error']})"
    else:
        sam_note = "raw YOLO boxes (SAM disabled)"

    template_used_count = sum(1 for r in instance_registrations if not r.get("fell_back"))
    template_fallback_count = sum(1 for r in instance_registrations if r.get("fell_back"))

    if removal_strategy == STRATEGY_CONTAINER_FILL:
        strategy_note = " Applied container-aware adaptive fill (remover) bounded by cell/table borders."
    elif removal_strategy == STRATEGY_TELEA:
        strategy_note = " Applied Telea inpainting."
    elif removal_strategy == STRATEGY_SUBTRACTIVE:
        strategy_note = " Applied bounded subtractive correction."
    else:
        if template_page_reg.get("available"):
            strategy_note = (
                f" Template registration ({template_page_reg['mark_id']}, "
                f"scale {template_page_reg['scale']:.3f}, angle {template_page_reg['angle_deg']:.1f} deg): "
                f"{template_used_count}/{len(instance_registrations)} instance(s) registered, "
                f"{template_fallback_count} fell back to the bounded correction."
            )
        else:
            strategy_note = " Template registration unavailable -- all instances used the bounded correction."

    message = (
        f"Method 3 (segmentation-driven): {n_found} candidate instance(s) detected, "
        f"{n_accepted} accepted / {n_rejected} rejected by the opaque-ink filter, "
        f"using {sam_note}. Cleaned {coverage_pct:.2f}% of the page in {total_ms:.1f} ms "
        f"(detect {meta['detect_ms']:.1f} ms, refine {meta['refine_ms']:.1f} ms, unmix {unmix_ms:.1f} ms)."
        f"{strategy_note}"
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
        "removal_strategy": removal_strategy,
        "use_template": use_template,
        "template_page_registration": template_page_reg,
        "instance_registrations": instance_registrations,
        "template_used_count": template_used_count,
        "template_fallback_count": template_fallback_count,
    }
    return cleaned, status
