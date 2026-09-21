"""Stamp Fit: locate the known AriaTender stamp artwork on a page and remove it
with the exact compositing inverse.

Why this exists
---------------
Every other removal strategy decides pixel by pixel what is watermark, from
local evidence only, and a faint grey pixel on its own is genuinely
ambiguous (watermark? shading? a light UI element?). The AriaTender mark is
not an unknown shape: it is one of two fixed pieces of artwork. So this
module asks a much easier question -- WHERE is this known shape -- and
answers it with a single global fit per mark (scale, position; rotation is
fixed at 0, every real mark measured so far is level). A whole wordmark
lining up letter by letter is unambiguous, so page content cannot be
mistaken for the mark, and the mask is the artwork's own coverage rather
than a per-pixel guess.

Once the shape and position are known, only two things per mark remain
unknown: its strength ``o`` and its ink colour. The mark's per-pixel
opacity is then ``a = o * coverage`` and the page is recovered exactly:

    true = (observed - a * ink) / (1 - a)

-- so text and lines UNDER the mark come back intact, which no
fill/threshold strategy can do.

Pipeline (``remove_stamps``)
----------------------------
1. Locate (``locate_stamps``): the alpha network's per-pixel opacity map is
   used as the signal to match the artwork against (the network localises
   the mark well; its errors -- stray blobs on real UI, ragged edges -- do
   not look like a wordmark, so the shape fit overrules them). Candidates
   come from template matching over 40 scales; each is scored by soft IoU
   (how much of the signal the rendered stamp explains -- plain normalised
   correlation favours tiny templates sitting on small blobs), then refined
   by pattern search and a local exhaustive search at full resolution.
   Two kinds are tried: the wide wordmark (shield + "AriaTender.neT") and
   the stacked mark (logo + Persian subtitle, the subtitle placed at the
   fixed layout offset ``SUB_OFFSET`` measured on real pages).
2. Accept (``_evidence``): a fit is kept only if the PAGE ITSELF shows the
   mark -- the image's colour must change along the fitted strokes
   (relative to a background inpainted from their surroundings) clearly
   more than along the same shape shifted off the mark. Colour, not just
   brightness: a grey mark over a blue banner mostly shifts hue. This is
   independent of the network, and rejects fits on pages that carry no
   AriaTender mark at all.
3. Refine: 1/4-pixel position and an edge blur (sigma <= MAX_EDGE_BLUR), by
   least squares against the page's darkening signal. (Wider blur/stroke-
   weight searches were tried and rejected: the objective runs away to
   over-broad coverage that leaves halos.)
4. Strength and ink per ink region (the wide mark's pink shield and grey
   letters are separate regions) by robust least squares against an
   inpainted background. On a flat background only the product
   ``o * (background - ink)`` is observable -- a faint dark ink and a
   strong light ink darken paper identically -- so a SOFT prior on the ink's
   luminance (``INK_LUM_PRIOR``) decides that one direction; everything the
   data does constrain (the strength, the ink's hue, e.g. the shield's red)
   comes from the page. A wrong split only matters for content under the
   mark and is bounded: +-40 levels of ink brightness moves black text
   under the mark by about +-12 levels. Regions with too few pixels borrow
   the parameters of a sibling region.
4b. Edge profile (``EDGE_PROFILE``): the Gaussian edge blur from step 3 is
    right in the interior but wrong exactly at the letter edges (see Honest
    limits below), so it is replaced, per ink region, by a small per-page
    fitted curve ``a = P(d)``, ``d`` = signed distance (sub-pixel, page px)
    from each pixel to the template's own 0.5 iso-contour at the fitted
    position/scale. ``P`` is piecewise-linear on fixed 0.5px knots from
    ``d = -3`` (hard-pinned to 0 -- caps how far outside the edge the mark
    may reach) to ``d = min(4, region depth)``, fit by regularised linear
    least squares (smoothness + a prior pulling knots with little data back
    to today's blurred model) with 3-4 rounds of trimming against page
    text/lines near the mark. Ink stays fixed at its step-4 value. A region
    keeps its fitted curve only if it lowers the same trimmed residual the
    old model gets by >= 3%; otherwise (or with too few edge pixels to fit
    at all) it falls back to a sibling region's curve, or to today's model
    unchanged -- so a page can only get better, never worse, byte-for-byte
    verified with ``EDGE_PROFILE = False``.
5. Remove: the exact inverse, written ONLY where the fitted stamp has
   non-zero opacity. Every pixel outside the stamps' own footprint is
   byte-identical to the input.

Honest limits
-------------
- Only the two AriaTender designs are known. Any other watermark is not
  found here (``doc_detect``'s Stamp Fit strategy falls back to the Alpha
  Network for detection boxes no fitted stamp covers).
- The artwork files' edges are slightly softer than some sites' crisp
  rendering, so before step 4b a faint rim remained along letter edges on
  some pages (measured: residual mark contrast ~1-10 grey levels, from ~35
  before removal at all). The fitted edge profile (step 4b) removes most of
  this on pages with enough clean edge pixels to fit it (see
  ``scripts/alpha_net/eval_stamp_rim.py`` for per-page ghost-score numbers);
  it still cannot undo JPEG ringing baked into the page's own pixels, and a
  rim over dense page text is fit from fewer usable pixels so improves less.
- Rotation is fixed at 0 and the subtitle layout is fixed; a rotated or
  re-laid-out stamp will be rejected by the evidence check (and fall back)
  rather than fitted wrongly.
"""

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import lsq_linear
from scipy.special import erf

from .config import BASE_DIR

STAMP_DIR = os.path.join(BASE_DIR, "assets", "stamps")
_FILES = {
    "wide": "ariatender_wide.png",
    "logo": os.path.join("sources", "ariatender-black-clean.png"),
    "sub": os.path.join("sources", "ariatender_subtitle_persian-black-clean.png"),
}
# Subtitle top-left relative to the logo's top-left, in logo-template pixels
# (multiply by the fitted logo scale). Measured on wm_realtune/0_552cff21fd
# (both parts clearly visible) and confirmed within +-1 px on 70_original and
# 0_e6b580e44b by an independent darkening signal.
SUB_OFFSET = (4.98, 147.55)
# Soft prior on the ink's luminance (0-1). Only decides the one direction a
# flat background leaves unobservable -- see the module docstring, step 4.
# From the cleanest real measurement (0_23a4bae7d5: text under the mark
# gives the strength directly; shield and letters agree at o ~= 0.20).
INK_LUM_PRIOR = 100.0 / 255.0
DEFAULT_STRENGTH = 0.2
MAX_ALPHA = 0.9
# Upper bound on the fitted edge blur (px). Small/low-res JPEG screenshots need
# ~1-1.5 px; an unbounded search (with a stroke-weight term) ran away to
# over-broad coverage that left halos, so this stays capped.
MAX_EDGE_BLUR = 1.6
# A mark whose surroundings are at least this bright (0-255) is on 'paper':
# its background for fitting comes from a closing, otherwise from inpainting.
LIGHT_BACKGROUND_LUM = 150.0
MAX_MARKS = 4
# Acceptance (see _evidence): the page must change along the fitted strokes
# by at least MIN_CHANGE grey levels (RGB distance), at least MIN_CHANGE_RATIO
# times as much as along the same shape shifted off the mark, with at least
# MIN_CHANGED_FRACTION of stroke pixels changed. Calibrated on wm_realtune
# (see the Stamp Fit section of scripts/alpha_net/benchmark.py's history).
MIN_CHANGE = 6.0
MIN_CHANGE_RATIO = 1.8
MIN_CHANGED_FRACTION = 0.6
# ...and the shape must explain most of the network's signal in its own
# window (real marks: 0.51-0.89 on wm_realtune; fits on leftover noise after
# the real mark was taken: <= 0.40), and be a plausible size.
MIN_LOCAL_IOU = 0.45
MIN_WIDTH_PX = 60

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)

# ---------------------------------------------------------------------------
# Edge profile (step 4b): a = P(d) instead of a = o * cov_blurred
# ---------------------------------------------------------------------------
# Master switch: False reproduces the pre-edge-profile output exactly,
# byte-for-byte (step 4b is skipped entirely).
EDGE_PROFILE = True
# Knot grid: fixed 0.5px spacing from -D_OUT (hard-pinned to alpha 0) to
# min(MAX_D_IN, the region's own interior depth).
D_OUT = 3.0
KNOT_STEP = 0.5
MAX_D_IN = 4.0
# A region needs at least this many pixels with d > -D_OUT (before, and
# again after, the outlier pre-filter) to fit its own curve; otherwise it
# borrows an accepted sibling region's curve (evaluated on its own distance
# map), or -- if no sibling has one -- keeps today's model.
MIN_EDGE_PIXELS = 400
# A fitted (or borrowed) curve is used only if it lowers the same trimmed
# residual today's model gets, by at least this fraction.
ACCEPT_MARGIN = 0.03
# Up-front outlier cap on |obs - B| (0-1 scale): page content far darker
# than any mark can produce is dropped before fitting (matches the 80-level
# cap `_colour_change` uses on the same 0-1 scale, 80/255 ~= 0.314).
OUTLIER_ABS = 0.35
# Acceptance uses a tighter, PER-MODEL cap on the POST-removal luminance
# residual (0-1 scale, 25 grey levels/255) -- same cut
# scripts/alpha_net/eval_stamp_rim.py's ghost score uses -- so a region's own
# accept/reject decision tracks the externally-reported metric exactly,
# rather than a looser fitting-time cap that can leave the accept test
# insensitive to a rim hiding in the pixels the fit itself keeps.
CONTENT_CUT_LUM = 25.0 / 255.0
# Regularisation weights (see module docstring, step 4b), scaled by the
# number of kept edge pixels so they stay dimensionless. There are only
# K-2 smoothness rows (~13) and K-1 prior rows (~13) against 3*n_eff data
# rows (tens of thousands), so lam_smooth/lam_prior must stay far below
# n_eff or a handful of regularisation rows outweigh the entire data term
# and crush P(d) into a near-flat ramp with no real edge transition
# (measured: at the original 0.5/0.02 -- i.e. lam_smooth, lam_prior on the
# same order as n_eff itself -- the fitted curve was uniformly WORSE than
# today's model on every region tested, by the acceptance test's own exact-
# reconstruction metric). Tuned on wm_realtune (scripts/alpha_net/
# eval_stamp_rim.py): this range gives a real median ~12% residual
# reduction on the regions with enough clean edge pixels to fit, while the
# acceptance test (not this constant) is what keeps every other region and
# page safe.
LAM_SMOOTH_PER_PIXEL = 1e-5
LAM_PRIOR_PER_PIXEL = 1e-4
# Window margin (page px) around each part's bbox: must cover D_OUT+MAX_D_IN
# plus slack for the 4x-supersampled distance transform's own border effects.
EDGE_WINDOW_MARGIN = 9
# Supersampling factor for the sub-pixel signed-distance map.
SS = 4


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def templates():
    """{name: {"alpha": (H,W) float32 0-1, "regions": {region: (H,W) float32}}}.
    Faint specks away from the strokes are removed (same rule as
    scripts/alpha_net/export_stamps.py). The wide mark is split into its pink
    shield and grey letters, which have different inks."""
    out = {}
    for name, rel in _FILES.items():
        rgba = np.asarray(Image.open(os.path.join(STAMP_DIR, rel)).convert("RGBA"), np.float32) / 255.0
        a = rgba[..., 3] / max(float(rgba[..., 3].max()), 1e-6)
        keep = cv2.dilate((a >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        a = np.where(keep, a, 0).astype(np.float32)
        if name == "wide":
            sat = rgba[..., :3].max(2) - rgba[..., :3].min(2)
            regions = {"shield": np.where(sat > 0.06, a, 0).astype(np.float32),
                       "letters": np.where(sat <= 0.06, a, 0).astype(np.float32)}
        else:
            regions = {name: a}
        out[name] = {"alpha": a, "regions": regions}
    return out


def _resized(name, region, tw, th):
    key = (name, region, tw, th)
    cache = _resized.cache
    if key not in cache:
        t = templates()[name]["alpha"] if region is None else templates()[name]["regions"][region]
        interp = cv2.INTER_AREA if tw < t.shape[1] else cv2.INTER_LINEAR
        cache[key] = cv2.resize(t, (tw, th), interpolation=interp)
        if len(cache) > 256:
            cache.pop(next(iter(cache)))
    return cache[key]


_resized.cache = {}


def _size(name, scale):
    t = templates()[name]["alpha"]
    return max(2, int(round(t.shape[1] * scale))), max(2, int(round(t.shape[0] * scale)))


def render(part, shape, region=None, window=None):
    """Coverage (0-1) of one fitted part, float position, optional edge blur.
    `window` = (x0, y0, w, h) renders only that page window (fast path for
    search); default renders the whole page of `shape` = (H, W)."""
    tw, th = _size(part["name"], part["scale"])
    t = _resized(part["name"], region, tw, th)
    x0, y0, w, h = window if window is not None else (0, 0, shape[1], shape[0])
    M = np.float32([[1, 0, part["x"] - x0], [0, 1, part["y"] - y0]])
    out = cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if part.get("sigma", 0) > 0:
        out = cv2.GaussianBlur(out, (0, 0), part["sigma"])
    return out


def _render_parts(parts, shape, window=None):
    cov = None
    for p in parts:
        c = render(p, shape, window=window)
        cov = c if cov is None else np.maximum(cov, c)
    return cov


def _parts_bbox(parts, shape, margin=0):
    H, W = shape
    xs0, ys0, xs1, ys1 = [], [], [], []
    for p in parts:
        tw, th = _size(p["name"], p["scale"])
        xs0.append(p["x"]); ys0.append(p["y"]); xs1.append(p["x"] + tw); ys1.append(p["y"] + th)
    x0 = int(max(0, np.floor(min(xs0)) - margin)); y0 = int(max(0, np.floor(min(ys0)) - margin))
    x1 = int(min(W, np.ceil(max(xs1)) + margin)); y1 = int(min(H, np.ceil(max(ys1)) + margin))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _soft_iou(t, a):
    return float(np.minimum(t, a).sum() / max(1e-6, np.maximum(t, a).sum()))


def _with_subtitle(logo):
    sx, sy = SUB_OFFSET
    return [logo, dict(name="sub", scale=logo["scale"], x=logo["x"] + sx * logo["scale"],
                       y=logo["y"] + sy * logo["scale"], sigma=0.0)]


def _expand(kind, main):
    return [main] if kind == "wide" else _with_subtitle(main)


# ---------------------------------------------------------------------------
# Locating
# ---------------------------------------------------------------------------

def _candidates(sig, name, n_scales=40, per_scale=3, max_templ_w=160):
    """Template-matching candidates for one kind on a (downscaled) signal.
    Each scale is matched at a resolution where the template is at most
    `max_templ_w` px wide -- this stage only proposes positions (the full-
    resolution refinement comes later), and matching a page-sized template
    against a padded page is what dominated the runtime."""
    H, W = sig.shape
    t = templates()[name]["alpha"]
    out = []
    shrunk = {}
    for sc in np.geomspace(0.10 * W / t.shape[1], 4.0 * W / t.shape[1], n_scales):
        tw, th = _size(name, sc)
        if th < 6 or tw < 6:
            continue
        f = min(1.0, max_templ_w / tw)
        f = round(f, 2) if f < 1 else 1.0
        if f not in shrunk:
            shrunk[f] = sig if f == 1.0 else cv2.resize(sig, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA)
        s_sig = shrunk[f]
        stw, sth = _size(name, sc * f)
        if sth < 4 or stw < 4:
            continue
        tt = _resized(name, None, stw, sth)
        py, px = int(sth * 0.8), int(stw * 0.8)
        padded = cv2.copyMakeBorder(s_sig, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
        r = cv2.matchTemplate(padded, tt, cv2.TM_CCORR)
        for _ in range(per_scale):
            _, v, _, loc = cv2.minMaxLoc(r)
            if v <= 0:
                break
            out.append(dict(name=name, scale=float(sc), x=float((loc[0] - px) / f), y=float((loc[1] - py) / f), sigma=0.0))
            r[max(0, loc[1] - sth // 4):loc[1] + sth // 4 + 1, max(0, loc[0] - stw // 4):loc[0] + stw // 4 + 1] = 0
    return out


def _pattern_refine(sig, kind, main, iters=40):
    """Pattern search on (x, y, scale) of the main part, soft IoU of the whole mark."""
    step = dict(x=max(1.0, 0.03 * _size(main["name"], main["scale"])[0]), y=None, scale=0.04)
    step["y"] = step["x"]

    def score(m):
        return _soft_iou(_render_parts(_expand(kind, m), sig.shape), sig)

    best = score(main)
    for _ in range(iters):
        improved = False
        for key in ("x", "y", "scale"):
            for d in (1, -1):
                q = dict(main)
                q[key] = main[key] * (1 + d * step[key]) if key == "scale" else main[key] + d * step[key]
                s = score(q)
                if s > best + 1e-5:
                    best, main, improved = s, q, True
        if not improved:
            step = {k: v / 2 for k, v in step.items()}
            if step["scale"] < 0.002 and step["x"] < 1:
                break
    return main, best


def _local_refine(sig, kind, main, radius=6, scales=(0.985, 0.9925, 1.0, 1.0075, 1.015)):
    """Search around `main` at full resolution, soft IoU measured in a window
    around the mark: integer position first, then scale at that position,
    then a +-1 px / scale pass (staged -- all combinations at once was the
    second-largest cost)."""
    shape = sig.shape
    x0, y0, w, h = _parts_bbox(_expand(kind, main), shape, margin=radius + 12)
    sw = sig[y0:y0 + h, x0:x0 + w]

    def score(m):
        return _soft_iou(_render_parts(_expand(kind, m), shape, window=(x0, y0, w, h)), sw)

    base = dict(main, x=float(round(main["x"])), y=float(round(main["y"])))
    best = max((dict(base, x=base["x"] + dx, y=base["y"] + dy)
                for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)), key=score)
    best = max((dict(best, scale=main["scale"] * sf) for sf in scales), key=score)
    best = max((dict(best, x=best["x"] + dx, y=best["y"] + dy, scale=best["scale"] * sf)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) for sf in (0.996, 1.0, 1.004)), key=score)
    return best, score(best)


def _normalise_signal(alpha_map):
    a = alpha_map.astype(np.float32)
    pos = a[a > 0.02]
    norm = float(np.percentile(pos, 90)) if pos.size > 50 else max(float(a.max()), 1e-3)
    return np.clip(a / norm, 0, 1).astype(np.float32)


def _fit_one(sig_full):
    """Best single mark on the (normalised, full-res) signal: (kind, parts, iou) or None."""
    H, W = sig_full.shape
    k = min(1.0, 600.0 / max(H, W))
    sig = cv2.resize(sig_full, (max(1, int(W * k)), max(1, int(H * k))), interpolation=cv2.INTER_AREA)
    best = None
    for kind, name in (("wide", "wide"), ("stacked", "logo")):
        cands = _candidates(sig, name)
        if not cands:
            continue
        scored = sorted(cands, key=lambda c: -_soft_iou(_render_parts(_expand(kind, c), sig.shape), sig))[:4]
        for c in scored:
            m, s = _pattern_refine(sig, kind, c)
            if best is None or s > best[0]:
                best = (s, kind, m, k)
    if best is None:
        return None
    _, kind, m, k = best
    m = dict(m, scale=m["scale"] / k, x=m["x"] / k, y=m["y"] / k)
    m, iou = _local_refine(sig_full, kind, m, radius=max(3, int(round(2 / k))))
    return kind, _expand(kind, m), iou


def _colour_change(img, parts, shape):
    """Median colour change (RGB distance, grey levels) along the parts'
    strokes relative to a background inpainted from their immediate
    surroundings, and the fraction of stroke pixels that change by > 3.
    Page content much stronger than any watermark (text, rules: > 80) is
    ignored either way."""
    x0, y0, w, h = _parts_bbox(parts, shape, margin=10)
    if w < 8 or h < 8:
        return 0.0, 0.0
    cov = _render_parts(parts, shape, window=(x0, y0, w, h))
    stroke = cov >= 0.5
    if stroke.sum() < 40:
        return 0.0, 0.0
    crop = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
    hole = cv2.dilate((cov > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8))
    B = cv2.inpaint(crop, hole, 3, cv2.INPAINT_TELEA).astype(np.float32)
    diff = np.linalg.norm(crop.astype(np.float32) - B, axis=2)[stroke]
    diff = diff[diff <= 80]
    if diff.size < 40:
        return 0.0, 0.0
    return float(np.median(diff)), float((diff > 3).mean())


def _evidence(img, parts):
    """Independent of the network: does the PAGE change along the fitted
    strokes? Colour change (not just darkening -- a grey mark over a blue
    banner shifts hue while barely changing brightness), compared with the
    same shape shifted off the mark on the same page as a control. Returns
    (median change on the fit, median change on the control, fraction of
    fit stroke pixels changed)."""
    shape = img.shape[:2]
    on, frac = _colour_change(img, parts, shape)
    h = max(_size(p["name"], p["scale"])[1] for p in parts)
    ctrl = []
    for dy in (-0.55 * h, 0.55 * h):
        shifted = [dict(p, y=p["y"] + dy) for p in parts]
        c, _ = _colour_change(img, shifted, shape)
        ctrl.append(c)
    return on, float(np.median(ctrl)), frac


def locate_stamps(img, alpha_map, max_marks=MAX_MARKS):
    """Fits up to `max_marks` stamps. Returns (accepted, rejected): lists of
    dicts {kind, parts, iou (local), change, control, changed_fraction, width_px}."""
    sig = _normalise_signal(alpha_map)
    accepted, rejected = [], []
    strong0 = int((sig > 0.3).sum())
    for i in range(max_marks):
        # after the first mark, only keep looking if a substantial amount of
        # strong signal is still unexplained
        if float(sig.max()) <= 0.05 or (i > 0 and (sig > 0.3).sum() < 0.25 * strong0 * (1.0 / (i + 1))):
            break
        fit = _fit_one(sig)
        if fit is None:
            break
        kind, parts, iou = fit
        change, control, frac = _evidence(img, parts)
        # how much of the signal IN THIS MARK'S OWN WINDOW the shape explains
        # (page-global IoU would penalise pages that carry several real marks)
        wx, wy, ww, wh = _parts_bbox(parts, sig.shape, margin=10)
        local_iou = _soft_iou(_render_parts(parts, sig.shape, window=(wx, wy, ww, wh)), sig[wy:wy + wh, wx:wx + ww])
        width = max(_size(p["name"], p["scale"])[0] for p in parts)
        rec = dict(kind=kind, parts=parts, iou=round(local_iou, 4), change=round(change, 2),
                   control=round(control, 2), changed_fraction=round(frac, 3), width_px=int(width))
        # remove what this fit explains from the signal before looking again
        x0, y0, w, h = _parts_bbox(parts, sig.shape, margin=6)
        foot = _render_parts(parts, sig.shape, window=(x0, y0, w, h)) > 0.02
        foot = cv2.dilate(foot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        sig[y0:y0 + h, x0:x0 + w][foot] = 0
        if (local_iou >= MIN_LOCAL_IOU and width >= MIN_WIDTH_PX and change >= MIN_CHANGE
                and change >= MIN_CHANGE_RATIO * control and frac >= MIN_CHANGED_FRACTION):
            accepted.append(rec)
        else:
            rejected.append(rec)
            break
    return accepted, rejected


# ---------------------------------------------------------------------------
# Sub-pixel refinement and exact removal
# ---------------------------------------------------------------------------

def _subpixel(part, d, shape):
    """1/4-px position, tiny scale change and edge blur for one part, by least
    squares of the page's darkening signal `d` against k * coverage."""
    x0, y0, w, h = _parts_bbox([part], shape, margin=8)
    dw = d[y0:y0 + h, x0:x0 + w]

    def score(q):
        cov = render(q, shape, window=(x0, y0, w, h))
        m = cv2.dilate((cov > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        c, v = cov[m], dw[m]
        if c.size < 20:
            return np.inf
        k = (c * v).sum() / max((c * c).sum(), 1e-6)
        r = np.abs(v - k * c)
        r = np.minimum(r, np.percentile(r, 90))          # ignore page text
        return float((r * r).mean())

    grid = [dict(part, x=part["x"] + dx, y=part["y"] + dy, sigma=sg)
            for dx in np.arange(-2, 2.01, 0.5) for dy in np.arange(-2, 2.01, 0.5) for sg in (0.0, 0.5, 0.9, 1.3)]
    best = min(grid, key=score)
    for step in (0.25, 0.125):
        cand = [dict(best, x=best["x"] + a * step, y=best["y"] + b * step, scale=best["scale"] * sf,
                     sigma=float(min(MAX_EDGE_BLUR, max(0.0, best["sigma"] + sd))))
                for a in (-1, 0, 1) for b in (-1, 0, 1) for sf in (0.997, 1.0, 1.003) for sd in (-0.15, 0, 0.15)]
        best = min(cand, key=score)
    return best


def _fit_strength_ink(obs, B, cov):
    """Robust least squares for one ink region: obs - B = cov*(p - o*B),
    p = o*ink, with a soft prior lum(ink) = INK_LUM_PRIOR (see module
    docstring). Returns (o, ink_rgb 0-1) or None when there is too little data."""
    m = cov > 0.5
    if m.sum() < 300:
        return None
    y = (obs - B)[m]
    c = cov[m]
    b = B[m]
    keep = np.ones(len(c), bool)
    o, p = DEFAULT_STRENGTH, np.full(3, INK_LUM_PRIOR * DEFAULT_STRENGTH)
    for _ in range(4):
        n = int(keep.sum())
        A = np.zeros((3 * n + 1, 4))
        r = np.zeros(3 * n + 1)
        for ch in range(3):
            A[ch * n:(ch + 1) * n, 0] = -c[keep] * b[keep, ch]
            A[ch * n:(ch + 1) * n, 1 + ch] = c[keep]
            r[ch * n:(ch + 1) * n] = y[keep, ch]
        w = 0.3 * np.sqrt(n)
        A[-1, 0] = -w * INK_LUM_PRIOR
        A[-1, 1:] = w * _LUMA
        sol, *_ = np.linalg.lstsq(A, r, rcond=None)
        o, p = float(sol[0]), sol[1:]
        res = np.abs(y - (c[:, None] * p[None] - o * c[:, None] * b)).mean(1)
        keep = res <= np.percentile(res[keep], 75)
    if not (0.02 <= o <= 0.8):
        return None
    return o, np.clip(p / o, 0, 1), float(np.median(res[keep]))


def _stroke_width(parts):
    """Widest stroke (px) of the fitted parts, from the templates' distance transform."""
    w = 0.0
    for p in parts:
        tw, th = _size(p["name"], p["scale"])
        m = (_resized(p["name"], None, tw, th) >= 0.5).astype(np.uint8)
        if m.any():
            w = max(w, 2.0 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max()))
    return w


def _surround_luminance(img, parts):
    """Median luminance of a ring just outside the mark's footprint."""
    shape = img.shape[:2]
    x0, y0, w, h = _parts_bbox(parts, shape, margin=12)
    foot = (_render_parts(parts, shape, window=(x0, y0, w, h)) > 0.02).astype(np.uint8)
    ring = (cv2.dilate(foot, np.ones((15, 15), np.uint8)) > 0) & ~(cv2.dilate(foot, np.ones((3, 3), np.uint8)) > 0)
    lum = img[y0:y0 + h, x0:x0 + w].astype(np.float32) @ _LUMA
    return float(np.median(lum[ring])) if ring.any() else 255.0


def _inpaint_background(img, marks):
    """Background for fitting: the stamps' footprint inpainted from its
    immediate surroundings (good on flat or bright-on-dark backgrounds; on
    text-dense pages it smears text into the estimate -- see
    _paper_background)."""
    H, W = img.shape[:2]
    foot = np.zeros((H, W), np.uint8)
    for mk in marks:
        foot |= (_render_parts(mk["parts"], (H, W)) > 0.01).astype(np.uint8)
    return cv2.inpaint(img, cv2.dilate(foot, np.ones((7, 7), np.uint8)), 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0


def _paper_background(img, marks):
    """Local background for FITTING strength/ink (the removal itself never
    uses it). A morphological closing a little wider than the stamp's
    thickest stroke erases thin dark features -- page text AND the mark's
    own strokes -- and keeps the paper/banner/ornament colour underneath.
    Inpainting the footprint from its surroundings (the first version) smears
    nearby text into the estimate on text-dense pages, making the background
    too dark, the mark's darkening too small, and its strength
    under-estimated (measured: visible leftover mark on exactly the dense-
    text pages). Outside the marks' windows the image itself is returned."""
    H, W = img.shape[:2]
    B = img.astype(np.float32) / 255.0
    for mk in marks:
        k = int(round(_stroke_width(mk["parts"]) * 1.4)) | 1
        k = max(5, k)
        x0, y0, w, h = _parts_bbox(mk["parts"], (H, W), margin=k + 4)
        crop = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
        closed = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        B[y0:y0 + h, x0:x0 + w] = closed.astype(np.float32) / 255.0
    return B


# ---------------------------------------------------------------------------
# Edge profile: a = P(d) (see module docstring, step 4b)
# ---------------------------------------------------------------------------

def _render_region_crisp_4x(part, region, x0, y0, w, h):
    """CRISP (sigma=0) coverage of one region, rendered at SS x supersampling
    in the page window (x0, y0, w, h). Used only for the geometry (the 0.5
    iso-contour), never as the alpha value itself."""
    tw, th = _size(part["name"], part["scale"])
    t = _resized(part["name"], region, tw * SS, th * SS)
    M = np.float32([[1, 0, (part["x"] - x0) * SS], [0, 1, (part["y"] - y0) * SS]])
    return cv2.warpAffine(t, M, (w * SS, h * SS), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _signed_distance(part, region, x0, y0, w, h):
    """Sub-pixel signed distance (page px, +inside/-outside) from each page
    pixel centre in the window to the region's own 0.5 iso-contour, via a
    4x-supersampled distance transform, area-averaged back down to page
    pixels (tested against sampling the centre sub-pixel directly; area
    averaging was smoother and made no visible difference to the fit)."""
    cov4 = _render_region_crisp_4x(part, region, x0, y0, w, h)
    mask4 = (cov4 >= 0.5).astype(np.uint8)
    dist_in = cv2.distanceTransform(mask4, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - mask4, cv2.DIST_L2, 5)
    # distanceTransform gives >=1 for the innermost/outermost pixel layer
    # (distance to the nearest pixel of the OTHER class); the true geometric
    # edge sits half a pixel closer, hence the -0.5 on both sides.
    d4 = np.where(mask4 > 0, dist_in - 0.5, 0.5 - dist_out).astype(np.float32)
    d = cv2.resize(d4, (w, h), interpolation=cv2.INTER_AREA) / float(SS)
    return d


def _knots_for(d_in):
    """Fixed 0.5px knots from -D_OUT (pinned to 0) to d_in, capped."""
    d_in = float(np.clip(d_in, KNOT_STEP, MAX_D_IN))
    d_in = min(MAX_D_IN, float(np.ceil(d_in / KNOT_STEP) * KNOT_STEP))
    n = int(round((d_in + D_OUT) / KNOT_STEP)) + 1
    return np.linspace(-D_OUT, d_in, n).astype(np.float32)


def _pl_basis(d, knots):
    """Piecewise-linear hat-function basis: (len(d), len(knots)); P(d) =
    Phi @ P_knots. Clamped outside the knot range (P = P(first)/P(last))."""
    d = np.clip(d, knots[0], knots[-1])
    idx = np.searchsorted(knots, d, side="right") - 1
    idx = np.clip(idx, 0, len(knots) - 2)
    d0, d1 = knots[idx], knots[idx + 1]
    t = np.where(d1 > d0, (d - d0) / np.maximum(d1 - d0, 1e-9), 0.0)
    Phi = np.zeros((d.shape[0], len(knots)), np.float32)
    rows = np.arange(d.shape[0])
    Phi[rows, idx] = (1 - t).astype(np.float32)
    Phi[rows, idx + 1] = t.astype(np.float32)
    return Phi


# Same range scripts/alpha_net/eval_stamp_rim.py's ghost score bins over
# (fixed, not tied to a region's own D_OUT/D_in) so acceptance here tracks
# exactly what gets reported, including the sliver just past the -D_OUT
# hard cutoff where a small stale tail of the OLD blurred model can still
# sit.
_BIAS_BINS = np.arange(-4.0, 4.0, KNOT_STEP)


def _binned_bias_rms(d_vals, res_lum):
    """RMS of the per-0.5px-bin MEAN of a SIGNED luminance residual, over
    `d_vals` in [-D_OUT, MAX_D_IN]. This is what a visible rim looks like: a
    systematic bias that survives binning, as opposed to per-pixel noise
    that averages out. Matters because a per-pixel residual NORM (unsigned,
    averaged over the whole footprint -- mostly deep-interior/deep-exterior
    pixels both models already fit well) barely moves even when a fit trades
    a small overall improvement for a bigger bias concentrated right at the
    edge; this metric is directly sensitive to that trade and is the same
    one `scripts/alpha_net/eval_stamp_rim.py`'s ghost score uses, so
    acceptance here tracks what is actually measured/reported."""
    means = []
    for lo in _BIAS_BINS:
        m = (d_vals >= lo) & (d_vals < lo + KNOT_STEP)
        if m.any():
            means.append(float(res_lum[m].mean()))
    if not means:
        return 0.0
    means = np.asarray(means)
    return float(np.sqrt(np.mean(means * means)))


def _recon_lum_residual(obs_flat, B_flat, a_flat, ink):
    """Exact lum(B) - lum(cleaned) for the SAME exact inverse `remove_stamps`
    writes out (and scripts/alpha_net/eval_stamp_rim.py's ghost score
    measures) -- NOT the linear approximation `(obs-B) - a*(ink-B)` (which
    equals `-(1-a) * (B-cleaned)`). Two models being compared here generally
    have different alpha at the same pixel, so that missing `(1-a)` factor
    is not a shared constant: it lets a model with a locally larger alpha
    look better under the linear residual while its true post-division
    residual (what actually lands on the page) is worse. Must match exactly
    so acceptance here cannot diverge from what gets reported/shipped."""
    a = np.clip(a_flat, 0.0, MAX_ALPHA).astype(np.float64)
    ink64 = np.asarray(ink, np.float64)
    cleaned = np.clip((obs_flat - a[:, None] * ink64[None, :]) / (1.0 - a[:, None]), 0.0, 1.0)
    return (B_flat - cleaned) @ _LUMA.astype(np.float64)


def _smoothness_rows(K):
    """Second-difference rows over the K-1 FREE knots (knot 0 is pinned to
    alpha 0, not a variable -- its (always zero) contribution is simply
    omitted rather than shifted onto the RHS)."""
    rows = []
    for j in range(1, K - 1):
        row = np.zeros(K - 1)
        for full_idx, coef in ((j - 1, 1.0), (j, -2.0), (j + 1, 1.0)):
            if full_idx >= 1:
                row[full_idx - 1] = coef
        rows.append(row)
    return np.array(rows) if rows else np.zeros((0, K - 1))


def _fit_edge_profile_region(d_w, obs_w, B_w, ink, o_old, cov_w, sigma_old):
    """Fit P(d) for one ink region in its own window. Returns
    {"status": "insufficient"} when there are too few usable edge pixels,
    else {"status": "ok", "knots", "P_free", "alpha_win", "residual_before",
    "residual_after", "accept"}. `alpha_win` and the residuals are always
    computed when status is "ok", regardless of `accept` -- the caller
    decides whether to use them."""
    footprint = d_w > -D_OUT
    if int(footprint.sum()) < MIN_EDGE_PIXELS:
        return dict(status="insufficient")
    ys, xs = np.nonzero(footprint)
    dv = d_w[ys, xs].astype(np.float32)
    y = (obs_w[ys, xs] - B_w[ys, xs]).astype(np.float64)
    imb = (ink[None, :] - B_w[ys, xs]).astype(np.float64)
    covv = cov_w[ys, xs].astype(np.float64)

    keep0 = np.linalg.norm(y, axis=1) <= OUTLIER_ABS
    if int(keep0.sum()) < MIN_EDGE_PIXELS:
        return dict(status="insufficient")

    knots = _knots_for(min(MAX_D_IN, float(dv.max())))
    K = len(knots)
    Phi_all = _pl_basis(dv, knots)              # (n, K)
    Phi = Phi_all[:, 1:].astype(np.float64)     # (n, K-1): free knots only

    n_eff = int(keep0.sum())
    lam_smooth = LAM_SMOOTH_PER_PIXEL * n_eff
    lam_prior = LAM_PRIOR_PER_PIXEL * n_eff

    # Prior P0: today's blurred model projected onto the same basis; where a
    # knot has almost no support in this region's window, fall back to the
    # analytic Gaussian-edge profile with the same strength/blur instead of
    # an undefined/noisy bin.
    model_v = o_old * covv
    denom = Phi_all.sum(0)
    numer = Phi_all.T @ model_v
    if sigma_old > 1e-6:
        analytic = o_old * 0.5 * (1.0 + erf(knots / (sigma_old * np.sqrt(2.0))))
    else:
        analytic = o_old * (knots >= 0).astype(np.float64)
    P0_full = np.where(denom >= 5, numer / np.maximum(denom, 1e-9), analytic).astype(np.float64)
    prior_target = P0_full[1:]

    S = _smoothness_rows(K)
    keep_idx = keep0.copy()
    P_free = prior_target.copy()
    for _ in range(4):
        sel = np.nonzero(keep_idx)[0]
        if sel.size < 50:
            break
        rows = [Phi[sel] * imb[sel, ch:ch + 1] for ch in range(3)]
        rhs = [y[sel, ch] for ch in range(3)]
        A = np.vstack(rows + [np.sqrt(lam_smooth) * S, np.sqrt(lam_prior) * np.eye(K - 1)])
        b = np.concatenate(rhs + [np.zeros(S.shape[0]), np.sqrt(lam_prior) * prior_target])
        sol = lsq_linear(A, b, bounds=(0.0, MAX_ALPHA))
        P_free = sol.x
        pred_a = Phi @ P_free
        res_all = np.linalg.norm(y - pred_a[:, None] * imb, axis=1)
        thresh = np.percentile(res_all[keep_idx], 80)
        keep_idx = (res_all <= thresh) & keep0

    Phi_win = _pl_basis(d_w.ravel().astype(np.float32), knots)[:, 1:]
    alpha_win = (Phi_win.astype(np.float64) @ P_free).reshape(d_w.shape)
    alpha_win = np.clip(np.where(d_w > -D_OUT, alpha_win, 0.0), 0, MAX_ALPHA).astype(np.float32)

    # Acceptance: over the FULL window (not just the footprint used to fit),
    # same range and pixel-selection style as the external ghost score -- a
    # SHARED mask (safe under both models' own post-removal residual, capped
    # at CONTENT_CUT_LUM), not the looser IRLS trim used only to keep the
    # least-squares solve itself robust, and not a separate per-model mask
    # (which can quietly compare different pixel sets and make a fit look
    # better than it is). The full window also catches a stale tail the OLD
    # blurred model can still leave just past the new model's hard -D_OUT
    # cutoff. See CONTENT_CUT_LUM's comment.
    d_full = d_w.ravel().astype(np.float32)
    obs_full = obs_w.reshape(-1, 3).astype(np.float64)
    B_full = B_w.reshape(-1, 3).astype(np.float64)
    res_lum_new = _recon_lum_residual(obs_full, B_full, alpha_win.ravel(), ink)
    res_lum_old = _recon_lum_residual(obs_full, B_full, o_old * cov_w.ravel(), ink)
    mask_cmp = (np.abs(res_lum_new) <= CONTENT_CUT_LUM) & (np.abs(res_lum_old) <= CONTENT_CUT_LUM)
    res_new = _binned_bias_rms(d_full[mask_cmp], res_lum_new[mask_cmp])
    res_old = _binned_bias_rms(d_full[mask_cmp], res_lum_old[mask_cmp])
    accept = res_old > 1e-9 and res_new <= res_old * (1.0 - ACCEPT_MARGIN)

    return dict(status="ok", knots=knots, P_free=P_free, alpha_win=alpha_win,
                residual_before=res_old, residual_after=res_new, accept=bool(accept))


def _profile_info(pr, used):
    """JSON-serialisable per-region report of the edge-profile step."""
    if pr.get("status") != "ok":
        return dict(used=False, reason="insufficient_data", knots_d=[], knots_a=[],
                    residual_before=None, residual_after=None)
    if pr.get("mark_reverted"):
        reason = "reverted_pooled_regression"
    else:
        reason = ("borrowed" if pr.get("borrowed") else "accepted") if used else \
                 ("borrowed_not_improved" if pr.get("borrowed") else "not_improved")
    knots_a = [0.0] + [round(float(v), 3) for v in pr["P_free"]]
    return dict(used=bool(used), reason=reason,
                knots_d=[round(float(v), 3) for v in pr["knots"]], knots_a=knots_a,
                residual_before=round(float(pr["residual_before"]), 4),
                residual_after=round(float(pr["residual_after"]), 4))


def remove_stamps(img, marks):
    """Exact removal of already-located `marks` (from locate_stamps).
    Returns (cleaned uint8, alpha map float32, per-mark info list). Pixels
    where the fitted opacity is 0 are byte-identical to `img`."""
    H, W = img.shape[:2]
    if not marks:
        return img.copy(), np.zeros((H, W), np.float32), []
    obs = img.astype(np.float32) / 255.0
    foot = np.zeros((H, W), np.uint8)
    for mk in marks:
        foot |= (_render_parts(mk["parts"], (H, W)) > 0.01).astype(np.uint8)
    # Two background estimates, each right in a different setting: a closing
    # (_paper_background) assumes the background is the LIGHTEST thing
    # locally -- true for dark text on light paper, where inpainting smears
    # text into the fill -- and is wrong on a dark background carrying light
    # content (white chips on a blue banner get inflated into it), where
    # inpainting is right. Each mark picks by the luminance of its own
    # surroundings. (Choosing by fit residual instead was tried: it picked
    # the closing on the blue-banner page, the one case where it is wrong.)
    bg_inpaint = _inpaint_background(img, marks)
    bg_closing = _paper_background(img, marks)
    A = np.zeros((H, W), np.float32)
    INK = np.zeros((H, W, 3), np.float32)
    infos = []
    for mk in marks:
        bg_name = "closing" if _surround_luminance(img, mk["parts"]) >= LIGHT_BACKGROUND_LUM else "inpaint"
        B = bg_closing if bg_name == "closing" else bg_inpaint
        d = (B - obs) @ _LUMA
        parts = [_subpixel(p, d, (H, W)) for p in mk["parts"]]
        regions = []
        for p in parts:
            names = list(templates()[p["name"]]["regions"])
            for rname in names:
                regions.append((f"{p['name']}:{rname}" if len(names) > 1 else p["name"], p, rname,
                                render(p, (H, W), region=rname)))
        fits = {rn: _fit_strength_ink(obs, B, cov) for rn, p, rname, cov in regions}
        own = [f for f in fits.values() if f is not None]

        def _resolved(rn):
            f = fits[rn] or (own[0] if own else (DEFAULT_STRENGTH, np.full(3, INK_LUM_PRIOR, np.float32), 0.0))
            return f[0], f[1]

        # Edge profile (step 4b): fit per region in its own window, first
        # pass only for regions with enough edge pixels of their own.
        profiles = {}
        windows = {}
        if EDGE_PROFILE:
            for rn, p, rname, cov in regions:
                o, ink = _resolved(rn)
                x0, y0, w, h = _parts_bbox([p], (H, W), margin=EDGE_WINDOW_MARGIN)
                windows[rn] = (x0, y0, w, h)
                d_w = _signed_distance(p, rname, x0, y0, w, h)
                obs_w = obs[y0:y0 + h, x0:x0 + w]
                B_w = B[y0:y0 + h, x0:x0 + w]
                cov_w = cov[y0:y0 + h, x0:x0 + w]
                profiles[rn] = dict(_fit_edge_profile_region(d_w, obs_w, B_w, ink, o, cov_w, p.get("sigma", 0.0)),
                                    d_w=d_w, obs_w=obs_w, B_w=B_w, cov_w=cov_w)
            # Second pass: regions with too few edge pixels of their own
            # borrow an accepted sibling's curve, evaluated on their OWN
            # distance map, and are safety-checked the same way before use.
            donors = [rn for rn, pr in profiles.items() if pr["status"] == "ok" and pr["accept"]]
            for rn, pr in profiles.items():
                if pr["status"] != "insufficient" or not donors:
                    continue
                o, ink = _resolved(rn)
                donor = profiles[donors[0]]
                d_w, obs_w, B_w, cov_w = pr["d_w"], pr["obs_w"], pr["B_w"], pr["cov_w"]
                Phi = _pl_basis(d_w.ravel().astype(np.float32), donor["knots"])[:, 1:]
                alpha_win = (Phi.astype(np.float64) @ donor["P_free"]).reshape(d_w.shape)
                alpha_win = np.clip(np.where(d_w > -D_OUT, alpha_win, 0.0), 0, MAX_ALPHA).astype(np.float32)
                # Acceptance over the FULL window, same as _fit_edge_profile_region.
                d_full = d_w.ravel().astype(np.float32)
                obs_full = obs_w.reshape(-1, 3).astype(np.float64)
                B_full = B_w.reshape(-1, 3).astype(np.float64)
                res_lum_new = _recon_lum_residual(obs_full, B_full, alpha_win.ravel(), ink)
                res_lum_old = _recon_lum_residual(obs_full, B_full, o * cov_w.ravel(), ink)
                mask_cmp = (np.abs(res_lum_new) <= CONTENT_CUT_LUM) & (np.abs(res_lum_old) <= CONTENT_CUT_LUM)
                res_new = _binned_bias_rms(d_full[mask_cmp], res_lum_new[mask_cmp])
                res_old = _binned_bias_rms(d_full[mask_cmp], res_lum_old[mask_cmp])
                accept = res_old > 1e-9 and res_new <= res_old * (1.0 - ACCEPT_MARGIN)
                profiles[rn] = dict(pr, status="ok", knots=donor["knots"], P_free=donor["P_free"],
                                    alpha_win=alpha_win, residual_before=res_old, residual_after=res_new,
                                    accept=accept, borrowed=True)

            # Final per-MARK safety net. Each region's own accept test only
            # guarantees ITS OWN binned residual improved; pooled together
            # with an unrelated sibling region's edges (e.g. the stacked
            # mark's logo + subtitle), two independently-improving curves
            # can still average to something worse at a shared d bin (the
            # same Simpson's-paradox trap `scripts/alpha_net/
            # eval_stamp_rim.py` pools raw samples, not curves, to avoid).
            # So verify the mark's regions' decisions TOGETHER do not raise
            # the pooled residual above today's-model baseline; if they do,
            # revert every region of this mark to today's model.
            pooled_d, pooled_old, pooled_new = [], [], []
            for rn, p, rname, cov in regions:
                pr = profiles.get(rn)
                if not (pr and pr["status"] == "ok"):
                    continue
                o, ink = _resolved(rn)
                d_w, obs_w, B_w, cov_w = pr["d_w"], pr["obs_w"], pr["B_w"], pr["cov_w"]
                d_full = d_w.ravel().astype(np.float32)
                obs_full = obs_w.reshape(-1, 3).astype(np.float64)
                B_full = B_w.reshape(-1, 3).astype(np.float64)
                res_old_v = _recon_lum_residual(obs_full, B_full, o * cov_w.ravel(), ink)
                res_new_v = _recon_lum_residual(obs_full, B_full, pr["alpha_win"].ravel(), ink) \
                    if pr["accept"] else res_old_v
                keep = (np.abs(res_old_v) <= CONTENT_CUT_LUM) & (np.abs(res_new_v) <= CONTENT_CUT_LUM)
                pooled_d.append(d_full[keep])
                pooled_old.append(res_old_v[keep])
                pooled_new.append(res_new_v[keep])
            if pooled_d:
                pd, po, pn = np.concatenate(pooled_d), np.concatenate(pooled_old), np.concatenate(pooled_new)
                if _binned_bias_rms(pd, pn) > _binned_bias_rms(pd, po):
                    for rn in profiles:
                        if profiles[rn]["status"] == "ok" and profiles[rn]["accept"]:
                            profiles[rn] = dict(profiles[rn], accept=False, mark_reverted=True)

        region_info = {}
        for rn, p, rname, cov in regions:
            o, ink = _resolved(rn)
            a = np.clip(o * cov, 0, MAX_ALPHA)
            pr = profiles.get(rn)
            used = bool(pr and pr["status"] == "ok" and pr["accept"])
            if used:
                x0, y0, w, h = windows[rn]
                a = a.copy()
                a[y0:y0 + h, x0:x0 + w] = pr["alpha_win"]
            sel = a > A
            A = np.where(sel, a, A)
            INK[sel] = ink
            region_info[rn] = dict(strength=round(o, 3), ink=[int(v) for v in np.round(ink * 255)],
                                   source="own" if fits[rn] else ("borrowed" if own else "default"),
                                   edge_profile=_profile_info(pr, used) if EDGE_PROFILE else
                                   dict(used=False, reason="off", knots_d=[], knots_a=[],
                                        residual_before=None, residual_after=None))
        mk["parts"] = parts
        infos.append(dict(kind=mk["kind"], parts=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for p in parts],
                          regions=region_info, background=bg_name, iou=mk.get("iou"), change=mk.get("change"),
                          control=mk.get("control"), changed_fraction=mk.get("changed_fraction")))
    rec = np.clip((obs - A[..., None] * INK) / (1.0 - A[..., None]), 0, 1)
    out = img.copy()
    m = A > 1e-4
    out[m] = np.round(rec[m] * 255).astype(np.uint8)
    return out, A, infos
