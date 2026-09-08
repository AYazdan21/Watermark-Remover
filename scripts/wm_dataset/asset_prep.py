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
