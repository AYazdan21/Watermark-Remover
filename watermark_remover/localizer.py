import cv2
import numpy as np

# Below this confidence/coverage, localize_watermark's mask isn't trusted
# and callers should fall back to their previous global-threshold behavior
# -- localization can only ever ADD protection for non-watermark content
# (see document_cleaner.py), so an untrusted mask must never gate erasure,
# or the tool could silently stop removing the watermark at all.
MIN_CONFIDENCE = 0.12
MIN_COVERAGE = 0.0008
MAX_COVERAGE = 0.65

# Candidate watermark scales, as a fraction of the page's shorter side --
# from a small logo/stamp up to a large diagonal banner. The watermark's
# own stroke width isn't known ahead of time, so these are swept rather
# than derived as a fixed multiple of the (much thinner) body-text stroke
# width, which under-covers large watermarks (see localize_watermark).
_SCALE_FRACTIONS = [0.025, 0.05, 0.09, 0.15, 0.22]

# cv2.MORPH_CLOSE cost grows sharply with kernel size, and the largest
# candidate scale above needs a kernel dozens of pixels wide -- at full
# scan resolution that's slow for no quality benefit, since this is only
# estimating coarse structure. The whole sweep instead runs on a copy
# downscaled so the short side is at most this many pixels; only the
# final winning mask is upsampled back to the input's resolution.
_WORK_MAX_DIM = 220


def _closed(gray, k):
    k = max(1, int(k) | 1)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _estimate_stroke_scale(gray: np.ndarray) -> float:
    """Estimates the typical dark-stroke half-width (roughly, body-text
    thickness) via an Otsu ink mask and a distance transform, so the
    text-closing kernel below scales with the image's effective resolution
    instead of being a fixed pixel count."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    vals = dt[dt > 0]
    if vals.size == 0:
        return 1.5
    return float(np.median(vals))


def _band_stats(band: np.ndarray):
    band_vals = band[band > 0]
    if band_vals.size < 50:
        return None
    noise_floor = float(np.percentile(band_vals, 60))
    strong = float(np.percentile(band_vals, 92))
    band_std = float(band_vals.std()) + 1e-6
    confidence = float(np.clip((strong - noise_floor) / (3.0 * band_std), 0.0, 1.0))
    thresh = max(4.0, noise_floor + 0.35 * (strong - noise_floor))
    coverage_est = float(np.mean(band >= thresh))
    return confidence, thresh, coverage_est


def localize_watermark(gray: np.ndarray):
    """Finds a soft evidence mask for a semi-transparent watermark or stamp
    overlaid on document text/paper, at whatever scale it turns out to be.

    The idea: compare two morphological paper estimates -- one that closes
    away only text-scale dark strokes, and one that additionally closes
    away watermark-scale strokes. Where the coarse estimate is much
    brighter than the fine one, a mark wider than a letter but narrower
    than the page background is likely present. Since a watermark can be
    anything from a small corner logo to a page-spanning diagonal banner,
    several candidate coarse scales are tried and the one that produces the
    most distinct band (vs. its own residual noise floor) is kept -- a
    single fixed multiple of the body-text stroke width under-covers large
    watermarks, since the closing kernel it implies never fully closes over
    a much bigger mark.

    Returns (mask, coverage, confidence):
      mask       -- uint8 0/255, watermark evidence dilated for anti-aliased
                    edges; all zero if no confident watermark was found.
      coverage   -- fraction of the page flagged as watermark.
      confidence -- 0..1, how distinct the winning band is from the
                    residual morphological noise floor.
    """
    h, w = gray.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)

    min_dim = min(h, w)
    work_scale = min(1.0, _WORK_MAX_DIM / min_dim)
    if work_scale < 1.0:
        gray_work = cv2.resize(gray, (max(1, int(round(w * work_scale))), max(1, int(round(h * work_scale)))), interpolation=cv2.INTER_AREA)
    else:
        gray_work = gray
    wh, ww = gray_work.shape[:2]
    work_min_dim = min(wh, ww)

    stroke = float(np.clip(_estimate_stroke_scale(gray_work), 1.0, 12.0))
    fine_k = max(3, int(round(stroke * 3)))
    paper_fine = _closed(gray_work, fine_k)

    best = None  # (confidence, band, thresh, coarse_k)
    for frac in _SCALE_FRACTIONS:
        coarse_k = int(round(work_min_dim * frac)) | 1
        if coarse_k <= fine_k + 4:
            continue
        median_k = max(3, (coarse_k // 3) | 1)
        paper_coarse = cv2.medianBlur(_closed(gray_work, coarse_k), median_k)
        band = cv2.subtract(paper_coarse, paper_fine).astype(np.float32)

        stats = _band_stats(band)
        if stats is None:
            continue
        confidence, thresh, coverage_est = stats
        if not (0.0003 <= coverage_est <= 0.5):
            continue
        if best is None or confidence > best[0]:
            best = (confidence, band, thresh, coarse_k)

    if best is None:
        return empty, 0.0, 0.0

    confidence, band, thresh, coarse_k = best
    candidate = (band >= thresh).astype(np.uint8) * 255
    # Drop speckle smaller than the text stroke scale -- a real watermark
    # glyph is broader than a single character stem at this residual scale.
    open_k = max(2, int(round(stroke)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))

    if float(np.mean(candidate > 0)) < 1e-6:
        return empty, 0.0, 0.0

    if work_scale < 1.0:
        candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_LINEAR)
        candidate = (candidate >= 127).astype(np.uint8) * 255

    # Dilate to cover anti-aliased/blurred watermark edges and thin
    # connecting strokes the open() pass may have removed, proportional to
    # the winning watermark scale itself (not the much smaller text
    # stroke), converted back to full-resolution pixels.
    dilate_k = max(3, int(round((coarse_k / work_scale) * 0.35)))
    mask = cv2.dilate(candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))

    coverage = float(np.mean(mask > 0))
    if coverage < 1e-6:
        return empty, 0.0, 0.0

    return mask, coverage, confidence


def is_trusted(coverage: float, confidence: float) -> bool:
    return confidence >= MIN_CONFIDENCE and MIN_COVERAGE <= coverage <= MAX_COVERAGE
