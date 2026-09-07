from dataclasses import dataclass

import cv2
import numpy as np

from .document_cleaner import auto_detect_document_profile

# Boundaries and weights were fit by measuring these features on the app's
# own dataset/ (documents from dataset/cleaned_documents, photos from
# dataset/originals) and checking the resulting scores against the true
# content of each image -- not against which UI tab had produced it, since
# several images in that dataset were originally run through the wrong tab
# (that mis-routing is exactly what this classifier exists to prevent).
_TERMS = [
    # (feature, boundary, steepness, weight) -- higher feature value pushes
    # the score toward "document" except for sat_med, which is inverted.
    ("dom_color_frac", 0.72, 22.0, 2.5),
    ("paper_frac", 0.65, 12.0, 2.0),
    ("sat_med", 45.0, 0.10, 2.0),
    ("flat_frac", 0.45, 8.0, 0.8),
]
_FEATURE_LABELS = {
    "dom_color_frac": "dominant paper color coverage",
    "paper_frac": "paper coverage",
    "sat_med": "color saturation",
    "flat_frac": "flat/uniform area",
}
# A detected orthogonal line grid (table/spreadsheet) is very hard to
# produce by accident in a natural photo, so treat it as strong evidence
# and let it override a borderline score from the other features.
_TABLE_OVERRIDE_SCORE = 0.88
_ANALYZE_MAX_DIM = 800

# The two routes fail differently: sending a photo to the document cleaner
# just over-flattens some bright regions, but sending a real document (esp.
# one with graphics/logos that drag its color stats toward "photo") to the
# LaMa pipeline fabricates plausible-looking fake text over real content --
# the exact failure this classifier exists to prevent. On a genuine toss-up
# that asymmetry means the safer error is to default to "document", so the
# route threshold sits below the neutral 0.5 point. Confidence is still
# measured against 0.5, so a tie-broken call is honestly reported as low
# confidence rather than hidden behind the shifted threshold.
_DOCUMENT_ROUTE_THRESHOLD = 0.40


@dataclass
class ImageProfile:
    paper_frac: float
    sat_med: float
    flat_frac: float
    dom_color_frac: float
    has_table: bool


@dataclass
class RouteDecision:
    route: str  # "document" or "photo"
    confidence: float  # 0..1, distance from the undecided midpoint
    document_score: float  # 0..1, raw weighted score
    reason: str
    profile: ImageProfile


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def analyze_image(img_np: np.ndarray) -> ImageProfile:
    """Computes a small set of color/geometry features used to tell scanned
    documents (letters, tables, notices) apart from natural photos."""
    h, w = img_np.shape[:2]
    scale = _ANALYZE_MAX_DIM / max(h, w)
    small = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img_np
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # Paper fraction: share of pixels clustered around the brightest dominant tone.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    bright_region = hist[128:]
    bright_mode = int(np.argmax(bright_region)) + 128 if bright_region.sum() > 0 else int(np.argmax(hist))
    lo, hi = max(0, bright_mode - 12), min(256, bright_mode + 13)
    paper_frac = float(hist[lo:hi].sum() / (sh * sw))

    # Median saturation: scanned documents are near-grayscale; photos are not.
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat_med = float(np.median(hsv[:, :, 1]))

    # Flat-gradient fraction: documents are mostly uniform paper plus thin strokes.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    flat_frac = float(np.mean(gm < 6))

    # Dominant quantized-color coverage: a real paper background collapses
    # to one tight RGB bin; a photographed gradient (sky, lighting) does not,
    # even where it is locally flat pixel-to-pixel.
    quant = (small // 16).astype(np.int32)
    keys = quant[:, :, 0] * 256 + quant[:, :, 1] * 16 + quant[:, :, 2]
    _, counts = np.unique(keys, return_counts=True)
    dom_color_frac = float(counts.max() / keys.size)

    has_table = bool(auto_detect_document_profile(small)["has_table"])

    return ImageProfile(
        paper_frac=paper_frac,
        sat_med=sat_med,
        flat_frac=flat_frac,
        dom_color_frac=dom_color_frac,
        has_table=has_table,
    )


def classify(profile: ImageProfile) -> RouteDecision:
    """Scores an ImageProfile toward "document" or "photo". Uses a smooth
    weighted sum rather than hard cutoffs so no single feature outlier (e.g.
    a photo with a large plain-colored background) can flip the route on
    its own; the confidence is reported honestly and low when features
    disagree, so the UI can surface uncertainty instead of guessing."""
    score = 0.0
    total_weight = 0.0
    parts = []
    for name, boundary, k, weight in _TERMS:
        value = getattr(profile, name)
        if name == "sat_med":
            s = 1.0 - _sigmoid(k * (value - boundary))
        else:
            s = _sigmoid(k * (value - boundary))
        score += weight * s
        total_weight += weight
        parts.append((name, value, s))

    document_score = score / total_weight
    if profile.has_table:
        document_score = max(document_score, _TABLE_OVERRIDE_SCORE)

    route = "document" if document_score >= _DOCUMENT_ROUTE_THRESHOLD else "photo"
    confidence = abs(document_score - 0.5) * 2.0

    if profile.has_table:
        reason = "Detected a regular table/spreadsheet grid."
    else:
        leading = max(parts, key=lambda p: abs(p[2] - 0.5))
        reason = f"Based mainly on {_FEATURE_LABELS[leading[0]]} ({leading[1]:.2f})."
        if _DOCUMENT_ROUTE_THRESHOLD <= document_score < 0.5:
            reason += " Too close to call, defaulted to the safer Document mode."

    return RouteDecision(
        route=route,
        confidence=confidence,
        document_score=document_score,
        reason=reason,
        profile=profile,
    )


def classify_image(img_np: np.ndarray) -> RouteDecision:
    return classify(analyze_image(img_np))
