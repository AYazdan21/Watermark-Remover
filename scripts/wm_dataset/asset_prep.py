"""Turns a watermark crop lifted off a real document into a clean, reusable
compositing stamp.

Why this exists: the raw alpha extracted from a real page carries the imprint
of whatever was underneath the mark -- ghost outlines of body text, JPEG
speckle, holes where black glyphs crossed it. Composite that 3,000 times and
every training image shares one identical noise fingerprint, which a
segmentation model can learn instead of learning the watermark. Cleaning it
once is amortized across the whole dataset.

The real mark is a flat-opacity graphic, so the alpha *variation* we measured
off the page is an artifact of the content behind it, not a property of the
mark. This module therefore rebuilds the stamp as a clean silhouette at a
measured, uniform opacity rather than preserving the noisy per-pixel alpha.

Calibration (measured on dataset/fulllogo.jpg, see MEASURED_* below): the
mark reads 226 where it sits on plain 254 paper, consistently across the
shield body and thick letter strokes -- a 28-level darkening.
"""

import cv2
import numpy as np
from PIL import Image

# Measured off the source document; see module docstring.
MEASURED_PAPER = 254.0
MEASURED_MARK_ON_PAPER = 226.0
# observed = a*ink + (1-a)*paper. Ink/alpha aren't separably identifiable from
# one observation, so we fix a plausible mid-gray ink and solve for the alpha
# that reproduces the measured appearance exactly. Any (ink, alpha) pair on
# this line renders identically on white; this one also behaves sensibly when
# composited onto darker backgrounds.
INK_GRAY = 128
BASE_ALPHA = (MEASURED_PAPER - MEASURED_MARK_ON_PAPER) / (MEASURED_PAPER - INK_GRAY)


def clean_stamp(rgba_path, min_component_frac=0.002, feather=1.0, close_k=5):
    """Cleans a raw extracted RGBA crop into a flat-opacity stamp.

    min_component_frac: drop connected components smaller than this fraction
    of the largest one -- this is what removes the ghosted body-text fragments
    while keeping the logo's own strokes (which are large and connected).
    """
    im = Image.open(rgba_path).convert("RGBA")
    alpha = np.array(im.split()[-1])

    binary = (alpha > 20).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_min = max(20, int(areas.max() * min_component_frac))
        cleaned = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= keep_min:
                cleaned[labels == i] = 1
        binary = cleaned

    # Feather the silhouette edge back on: a hard binary edge composites with
    # visible aliasing that a real printed/rendered mark never has.
    soft = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), feather) if feather > 0 else binary.astype(np.float32)
    soft = np.clip(soft, 0.0, 1.0)

    out = np.zeros((soft.shape[0], soft.shape[1], 4), dtype=np.uint8)
    out[:, :, 0:3] = INK_GRAY
    out[:, :, 3] = (soft * BASE_ALPHA * 255).astype(np.uint8)
    return _trim(Image.fromarray(out, "RGBA"))


def _trim(im, pad=2):
    bbox = im.split()[-1].getbbox()
    if bbox is None:
        return im
    x1, y1, x2, y2 = bbox
    return im.crop((max(0, x1 - pad), max(0, y1 - pad), min(im.width, x2 + pad), min(im.height, y2 + pad)))


def _local_paper_estimate(observed_rgb, observed_gray, paper_thr):
    """Estimates the page's ground colour AT EACH PIXEL, not as one global
    constant.

    Why per-pixel: the screenshot this feeds (ariatender_wide_wordmark.png)
    is NOT a uniform page -- it is level-240 gray over most of the frame but
    carries a white (255) band across part of it (a UI highlight strip in
    the original screenshot). A single global paper value would mis-measure
    darkening inside that band: pixels there would appear "darker than
    paper" even where there is no ink at all. Instead, pixels that are
    unambiguously bare paper are identified directly, and every other pixel
    borrows the paper colour of its NEAREST such pixel via a distance
    transform. This is only valid because the ground is piecewise-constant
    in large blocks (a page background, not a photograph) -- nearest-known-
    paper is a good estimator precisely because paper regions are large and
    contiguous relative to any stroke.

    `paper_thr` (definitely-paper brightness floor) is a calibrated constant,
    not a free knob -- measured directly against this source image:
      - 239.0 (chosen): 70.3% of pixels classified as known-paper, and the
        resulting silhouette is clean (see extract_flat_screenshot_stamp).
      - 236.0: anti-aliased fringe pixels around strokes are bright enough
        to be misclassified as paper, which erodes the silhouette's core --
        core-coverage fraction collapses from 0.219 to 0.130.
      - 244.0: destroys the estimate almost entirely (known-paper drops to
        4.4%), because the paper itself sits at 240, i.e. below this floor.
    The companion neutrality test (max channel - min channel < 6) is what
    keeps the pink shield's own paper-bright pixels, if any, from being
    misread as chromatic ink; it excludes nothing here in practice since the
    page ground is genuinely gray, but guards against a colored highlight
    band being mistaken for gray paper.
    """
    known = (observed_gray >= paper_thr) & (
        (observed_rgb.max(axis=2) - observed_rgb.min(axis=2)) < 6
    )
    # cv2's distance transform gives us, for every unknown pixel, the pixel
    # coordinates (via labelType=DIST_LABEL_PIXEL) of its nearest zero
    # (known) pixel in a single pass -- far cheaper than any per-pixel
    # nearest-neighbour search over a 2184x534 image.
    _, label_map = cv2.distanceTransformWithLabels(
        (~known).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    h, w = observed_gray.shape
    ys, xs = np.nonzero(known)
    order = np.argsort(ys * w + xs)  # matches cv2's raster-scan label indexing
    known_y, known_x = ys[order], xs[order]
    paper = observed_rgb[known_y[label_map - 1], known_x[label_map - 1]]
    paper[known] = observed_rgb[known]
    return paper, known


def extract_flat_screenshot_stamp(path, paper_thr=239.0, d_core=35.0,
                                   min_component_frac=0.002, noise_floor=2.0):
    """Extracts a clean RGBA stamp from a RAW, FULLY-OPAQUE screenshot crop.

    Why this is a different code path from clean_stamp(): clean_stamp reads
    the alpha channel to find the silhouette, which only works if the source
    file's alpha already encodes "ink vs. not-ink" (a background-removed
    crop). ariatender_wide_wordmark.png is a plain screenshot -- alpha=255
    everywhere -- so its only usable signal is colour DIFFERENCE against the
    page it was captured on. This function derives alpha (and, unlike
    clean_stamp, per-pixel INK COLOUR) from that difference instead.

    Preserving per-pixel ink colour (step 4 below) matters here specifically
    because this mark is two-tone: a dusty-pink shield/gavel glyph next to
    grey lettering. clean_stamp's flat-gray-silhouette approach is correct
    for the stacked mark (which really is one uniform ink) but would
    flatten this mark's pink glyph to the same grey as its text, destroying
    the one visual feature that makes the two watermarks distinguishable.

    Algorithm (each constant below is measured against this source image,
    not assumed):

    1. Local paper estimate -- see _local_paper_estimate().

    2. Coverage: how much of this pixel is ink, on a 0..1 scale, from how
       far the observed gray falls below its local paper estimate:
           d = local_paper_gray - observed_gray
           coverage = clip(d / D_CORE, 0, 1)
       D_CORE=35.0 is the measured paper-to-ink-core drop: paper reads 240,
       the grey stroke's solid core reads ~205. Pixels with d < noise_floor
       (2.0) are zeroed -- JPEG-era screenshot noise on flat paper is a
       fraction of a level, well under that, so this only suppresses noise
       and does not bite into real anti-aliased fringes (which sit well
       above 2.0 by the time they're visually distinguishable from paper).

    3. Alpha: coverage is a fraction of full ink darkening, so it is put on
       the same alpha SCALE the rest of this module already uses --
       A_REF = D_CORE / (240.0 - INK_GRAY) = 35.0 / (240 - 128) = 0.3125,
       i.e. "the alpha a flat INK_GRAY=128 ink would need to reproduce a
       35-level darkening on 240 paper", exactly the same reasoning BASE_ALPHA
       uses for the stacked mark's 28-level darkening on 254 paper. Both
       marks' alpha therefore live on one consistent scale even though they
       were measured from different source images.
           alpha = coverage * A_REF

    4. Per-pixel ink colour -- this is what keeps the shield pink instead of
       collapsing every stroke to one grey. Solving the standard compositing
       equation `observed = alpha*ink + (1-alpha)*paper` for `ink` per pixel:
           ink = paper + (observed - paper) / alpha
                = paper + (observed - paper) * (D_CORE / A_REF) / d
       (using coverage's own d/D_CORE in place of the true alpha, since that
       is the only estimate available, then rescaling to the A_REF scale --
       algebraically: ink = 240 + (observed-paper) * ((D_CORE/A_REF) / max(d, noise_floor))).
       The key stability property: (observed - paper) / d is a UNIT CHROMA
       DIRECTION -- both its numerator and denominator shrink together on
       faint, mostly-paper anti-aliased edge pixels (small d), so the ratio
       stays a well-conditioned direction vector rather than blowing up the
       way a naive `observed / alpha` unmix would as alpha -> 0. `max(d,
       noise_floor)` in the denominator is what prevents an actual divide-
       by-zero on the (already-zeroed-out) sub-noise-floor pixels.

    5. Speckle removal + trim, mirroring clean_stamp: binarise coverage at
       0.12, close with a 5x5 ellipse to bridge thin anti-aliased gaps,
       drop connected components under `min_component_frac` of the largest
       (removes JPEG-block and label-rendering speckle outside the glyphs
       while keeping every real letter/glyph stroke, all large and
       connected), multiply coverage by the surviving silhouette, build the
       final RGBA, and trim to content.

    Returns a trimmed RGBA PIL.Image.
    """
    im = Image.open(path).convert("RGB")
    observed = np.asarray(im, dtype=np.float32)
    observed_gray = observed.mean(axis=2)

    paper, _known = _local_paper_estimate(observed, observed_gray, paper_thr)
    paper_gray = paper.mean(axis=2)

    a_ref = d_core / (240.0 - INK_GRAY)

    d = paper_gray - observed_gray
    coverage = np.clip(d / d_core, 0.0, 1.0)
    coverage[d < noise_floor] = 0.0

    ink = np.clip(
        240.0 + (observed - paper) * ((d_core / a_ref) / np.maximum(d, noise_floor))[..., None],
        0.0, 255.0,
    )

    binary = (coverage > 0.12).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, cc_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_min = max(20, int(areas.max() * min_component_frac))
        silhouette = np.isin(cc_labels, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= keep_min])
    else:
        silhouette = binary.astype(bool)

    coverage = coverage * silhouette
    alpha = coverage * a_ref

    h, w = observed_gray.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, 0:3] = ink.astype(np.uint8)
    out[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return _trim(Image.fromarray(out, "RGBA"))


def combine_stamp(logo_im, subtitle_im, gap_frac=0.055, subtitle_scale=None, align="left"):
    """Stacks the logo and its subtitle into one stamp at their real relative
    geometry, since that is how the mark actually appears on a page.

    Defaults are derived from dataset/fulllogo.jpg, where the logo occupies
    y~245-350 and the subtitle y~358-400, both starting at x~40 -- i.e. a gap
    of roughly 5-6% of the logo's width, left-aligned, with the subtitle
    already at its natural relative size in the source crops.
    """
    if subtitle_scale is not None:
        w = int(subtitle_im.width * subtitle_scale)
        h = int(subtitle_im.height * subtitle_scale)
        subtitle_im = subtitle_im.resize((w, h), Image.LANCZOS)

    gap = int(logo_im.width * gap_frac)
    w = max(logo_im.width, subtitle_im.width)
    h = logo_im.height + gap + subtitle_im.height
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def x_for(im):
        if align == "center":
            return (w - im.width) // 2
        if align == "right":
            return w - im.width
        return 0

    canvas.alpha_composite(logo_im, (x_for(logo_im), 0))
    canvas.alpha_composite(subtitle_im, (x_for(subtitle_im), logo_im.height + gap))
    return _trim(canvas)
