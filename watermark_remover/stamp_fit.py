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
5. Remove: the exact inverse, written ONLY where the fitted stamp has
   non-zero opacity. Every pixel outside the stamps' own footprint is
   byte-identical to the input.

Honest limits
-------------
- Only the two AriaTender designs are known. Any other watermark is not
  found here (``doc_detect``'s Stamp Fit strategy falls back to the Alpha
  Network for detection boxes no fitted stamp covers).
- The artwork files' edges are slightly softer than some sites' crisp
  rendering, so a faint rim can remain along letter edges on some pages
  (measured: residual mark contrast ~1-10 grey levels, from ~35 before).
- Rotation is fixed at 0 and the subtitle layout is fixed; a rotated or
  re-laid-out stamp will be rejected by the evidence check (and fall back)
  rather than fitted wrongly.
"""

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

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
                regions.append((f"{p['name']}:{rname}" if len(names) > 1 else p["name"], render(p, (H, W), region=rname)))
        fits = {rn: _fit_strength_ink(obs, B, cov) for rn, cov in regions}
        own = [f for f in fits.values() if f is not None]
        region_info = {}
        for rn, cov in regions:
            f = fits[rn] or (own[0] if own else (DEFAULT_STRENGTH, np.full(3, INK_LUM_PRIOR, np.float32), 0.0))
            o, ink = f[0], f[1]
            a = np.clip(o * cov, 0, MAX_ALPHA)
            sel = a > A
            A = np.where(sel, a, A)
            INK[sel] = ink
            region_info[rn] = dict(strength=round(o, 3), ink=[int(v) for v in np.round(ink * 255)],
                                   source="own" if fits[rn] else ("borrowed" if own else "default"))
        mk["parts"] = parts
        infos.append(dict(kind=mk["kind"], parts=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for p in parts],
                          regions=region_info, background=bg_name, iou=mk.get("iou"), change=mk.get("change"),
                          control=mk.get("control"), changed_fraction=mk.get("changed_fraction")))
    rec = np.clip((obs - A[..., None] * INK) / (1.0 - A[..., None]), 0, 1)
    out = img.copy()
    m = A > 1e-4
    out[m] = np.round(rec[m] * 255).astype(np.uint8)
    return out, A, infos
