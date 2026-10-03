"""Model-free 'faint darkening' signal shared by the Template Builder and
Method 5 (Template Stamp Fit).

A semi-transparent mark that a site stamps on every page only ever makes the
page a little darker than the paper underneath it. The paper underneath is
easy to estimate without any model, in two ways:

* **Pose not known yet** (the first locate pass): a morphological closing with
  a kernel wider than the mark's thickest SOLID area erases everything thin
  and dark (the mark's own strokes AND page text) and keeps the lightest thing
  locally, which is the paper (``paper_background``). The kernel comes from the
  template's largest solid thickness (a big filled disc needs a big kernel), not
  from a stroke width.
* **Pose known** (builder warps, Method 5 removal): the mark's own footprint is
  simply left out and the paper is interpolated across it from the locally
  bright pixels around it (``paper_background_masked``: normalised
  convolution + a coarser pyramid for holes bigger than the blur). Inside a
  solid area the closing's idea of 'background' is the mark itself, which makes
  the darkening ~0 there; the masked estimate has no such hole.

What the page is darker than that paper is the 'darkening' (3 channels: a teal
mark darkens R far more than G/B, black text darkens all of them equally, which
is what lets the locator tell them apart). Real ink (text, rules) is far
darker than any watermark, so pixels above a high threshold, and a thin rim
around them (anti-aliasing), are zeroed. What is left is the faint mark and
little else, and that is what templates are matched against -- no trained
network involved.

Contents
--------
* ``paper_background`` / ``paper_background_masked`` / ``footprint_mask``: the
  two paper estimates.
* ``darkening`` / ``band_signal`` / ``normalise`` (luminance-like, 1 channel,
  the v1 signal) and ``band_signal3`` / ``normalise3`` / ``page_signal3``
  (3 channels, colour-matched).
* ``colour_template``: the 3-channel matched-filter template ``alpha*(1-ink)``.
* ``ncc_match``: multi-scale normalised cross-correlation of a template
  against a signal (padded so marks hanging off the page edge still match);
  works on 1- or 3-channel signals.
* ``refine_pose``: local scale / sub-pixel search around a candidate.
* ``locate_template``: the page-level driver both callers use (coarse match
  on a downscaled page, then a full-resolution refinement in a window).

Pose convention (identical to ``stamp_fit``): ``scale`` is page pixels per
template pixel, ``(x, y)`` is where the (round-sized) resized template's
top-left lands on the page, float.

Known limits: the mark must be DARKER than the page; rotation is fixed at 0.
"""

import threading

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
MAX_KERNEL = 61            # cap of the v1 stroke-width kernel (``kernel_for`` default)
MAX_SOLID_KERNEL = 151     # cap of the solid-thickness kernel used by the locate passes
_EXCL_DILATE = 9
_CLOSE_DIRECT = 41         # closing kernels wider than this run on a shrunken copy (speed)
STAGE1_W = 200             # px: the coarse scale/position pass works on a signal where the mark is at most this wide
MAX_REFINE_W = 320         # px: the final pose search never works finer than this across the mark
KERNEL_GROUP_RATIO = 1.6       # scales searched together share one paper-background kernel when they differ by at most this factor
COARSE_POOL = 2            # the coarse stage proposes this x n_cands candidates; the cheap coarse refinement keeps n_cands


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------

def stroke_width(alpha, thresh=0.5):
    """Widest stroke (px) of a coverage map, from its distance transform
    (same idea as ``stamp_fit._stroke_width``)."""
    m = (alpha >= thresh).astype(np.uint8)
    if not m.any():
        return 0.0
    return 2.0 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max())


def solid_thickness(alpha, thresh=0.3):
    """Largest SOLID thickness (px) of a coverage map: twice the maximum of the
    distance transform of ``alpha >= thresh``. A filled disc of diameter d gives
    d; a letter stroke gives its stroke width."""
    return stroke_width(alpha, thresh)


def kernel_for(thickness_at_scale, cap=MAX_KERNEL):
    """k = max(7, round(1.4 * thickness)), capped at ``cap``, odd. The v1
    callers pass a stroke width with the default cap; the v2 locate passes pass
    ``solid_thickness * scale`` with ``cap=MAX_SOLID_KERNEL``."""
    k = int(max(7, round(1.4 * float(thickness_at_scale))))
    return int(min(int(cap), k) | 1)


def paper_background(img_rgb_uint8, kernel_px):
    """Local paper colour: closing with an elliptical kernel wider than the
    strokes / solid areas. float32 (H, W, 3) in 0-1. Kernels wider than
    ``_CLOSE_DIRECT`` run on a shrunken copy and are upsampled (the paper is
    smooth by construction; this keeps a 151 px kernel as cheap as a 41 px one)."""
    k = max(5, int(kernel_px) | 1)
    img = np.ascontiguousarray(img_rgb_uint8)
    if k > _CLOSE_DIRECT:
        H, W = img.shape[:2]
        q = _CLOSE_DIRECT / float(k)
        sw, sh = max(8, int(round(W * q))), max(8, int(round(H * q)))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        ks = max(5, int(round(k * sw / float(W))) | 1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)
        return cv2.resize(closed, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    return closed.astype(np.float32) / 255.0


def _as_float(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def darkening(img, B):
    """max over RGB of (B_c - I_c), clipped at 0. Max over channels (not luma)
    so a pink mark that mostly darkens G/B still shows. float32 (H, W) 0-1."""
    d = (_as_float(B) - _as_float(img)).max(axis=2)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def band_signal(img, B, hi=0.35, text_dilate=2):
    """Darkening with real ink (> ``hi``) and a ``text_dilate``-px rim around
    it zeroed: the faint mark and little else."""
    d = darkening(img, B)
    text = (d > hi).astype(np.uint8)
    if text_dilate > 0 and text.any():
        k = 2 * int(text_dilate) + 1
        text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    d[text > 0] = 0.0
    return d


def normalise(sig):
    """Divide by the 90th percentile of values > 0.02 (as
    ``stamp_fit._normalise_signal``), clip to [0, 1]."""
    a = sig.astype(np.float32)
    pos = a[a > 0.02]
    norm = float(np.percentile(pos, 90)) if pos.size > 50 else max(float(a.max()), 1e-3)
    return np.clip(a / max(norm, 1e-3), 0, 1).astype(np.float32)


def band_signal3(img, B, hi=0.35, text_dilate=2):
    """3-channel darkening ``D = B - I`` (clipped at 0), with real ink zeroed in
    ALL channels: pixels whose LUMINANCE darkening exceeds ``hi`` and a
    ``text_dilate``-px rim around them. float32 (H, W, 3)."""
    Dr = _as_float(B) - _as_float(img)
    lum = Dr @ LUMA
    text = (lum > hi).astype(np.uint8)
    if text_dilate > 0 and text.any():
        k = 2 * int(text_dilate) + 1
        text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    D = np.clip(Dr, 0.0, 1.0)
    D[text > 0] = 0.0
    return D.astype(np.float32)


def normalise3(sig3):
    """Divide all channels by the 90th percentile (of values > 0.02) of the
    per-pixel channel maximum, so channel ratios survive; clip to [0, 1]."""
    a = sig3.astype(np.float32)
    mx = a.max(axis=2)
    pos = mx[mx > 0.02]
    norm = float(np.percentile(pos, 90)) if pos.size > 50 else max(float(mx.max()), 1e-3)
    return np.clip(a / max(norm, 1e-3), 0, 1).astype(np.float32)


def page_signal(img_rgb_uint8, kernel_px, max_side=800):
    """Normalised band signal of a whole page at working resolution.

    The paper background is estimated on a downscaled copy (cheap) and
    upsampled; the darkening, the text zeroing and its rim are done at FULL
    resolution and only then is the signal shrunk -- zeroing text after
    shrinking leaves blurred text bars below the threshold that look like
    mark strokes. Returns (signal, f) with ``f`` = working px per page px."""
    return _page_signal(img_rgb_uint8, kernel_px, max_side, band_signal, normalise)


def page_signal3(img_rgb_uint8, kernel_px, max_side=800):
    """Same as ``page_signal`` but 3-channel and colour-preserving (float32
    (h, w, 3)); the page signal the colour-matched locator correlates against."""
    return _page_signal(img_rgb_uint8, kernel_px, max_side, band_signal3, normalise3)


def _page_signal(img, kernel_px, max_side, band, norm):
    H, W = img.shape[:2]
    f = min(1.0, float(max_side) / max(H, W))
    if f >= 1.0:
        B = paper_background(img, kernel_px)
        return norm(band(img, B)), 1.0
    sw, sh = max(1, int(round(W * f))), max(1, int(round(H * f)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    kc = max(5, int(round(kernel_px * f)) | 1)
    Bs = paper_background(small, kc)
    B = cv2.resize(Bs, (W, H), interpolation=cv2.INTER_LINEAR)
    sig = cv2.resize(band(img, B), (sw, sh), interpolation=cv2.INTER_AREA)
    return norm(sig), sw / W


def colour_template(alpha, ink):
    """3-channel matched-filter template: the darkening a unit-opacity mark
    leaves on white paper, ``alpha * (1 - ink)`` per pixel. ``alpha`` (h, w) 0-1,
    ``ink`` (h, w, 3) 0-1. float32 (h, w, 3)."""
    return (alpha.astype(np.float32)[..., None] * (1.0 - ink.astype(np.float32))).astype(np.float32)


def colour_spread(W3):
    """Channel spread of a colour template relative to its luminance, from the
    template's total mass per channel: ~0 for a grey mark, large for a teal one."""
    m = W3.reshape(-1, 3).sum(0).astype(np.float64)
    lum = float(m @ LUMA)
    return float((m.max() - m.min()) / max(lum, 1e-9))


# ---------------------------------------------------------------------------
# Mark-aware paper background (pose known)
# ---------------------------------------------------------------------------

def footprint_mask(alpha, pose, shape, dilate_px=3, thresh=0.02):
    """Boolean page mask of the mark's footprint: ``alpha > thresh`` rendered at
    ``pose`` (dict scale, x, y), dilated by ``dilate_px``."""
    H, W = shape
    cov = render_template(alpha, pose["scale"], pose["x"], pose["y"], (0, 0, W, H))
    m = (cov > thresh).astype(np.uint8)
    if dilate_px > 0 and m.any():
        k = 2 * int(dilate_px) + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m > 0


def _fill_pyramid(num, den, sigma, tau=0.25):
    """Normalised convolution ``G*(I m) / G*(m)`` with a coarser-pyramid fill:
    where ``G*(m)`` is small (a hole wider than the blur) the estimate falls back
    to the same quantity one octave coarser, repeatedly, until it is defined."""
    levels = []
    n, d = num, den
    while True:
        bn = cv2.GaussianBlur(n, (0, 0), sigma)
        bd = cv2.GaussianBlur(d, (0, 0), sigma)
        levels.append((bn, bd))
        h, w = d.shape
        if min(h, w) < 16 or len(levels) >= 9:
            break
        size = (max(1, w // 2), max(1, h // 2))
        n = cv2.resize(n, size, interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, size, interpolation=cv2.INTER_AREA)
    tot = float(levels[0][1].sum()) or 0.0
    fallback = (levels[0][0].reshape(-1, 3).sum(0) / tot) if tot > 1e-6 else np.ones(3, np.float32)
    B = None
    for bn, bd in reversed(levels):
        est = bn / np.maximum(bd, 1e-6)[..., None]
        w = np.clip(bd / tau, 0.0, 1.0)[..., None]
        if B is None:
            B = w * est + (1.0 - w) * fallback[None, None, :]
        else:
            up = cv2.resize(B, (bd.shape[1], bd.shape[0]), interpolation=cv2.INTER_LINEAR)
            B = w * est + (1.0 - w) * up
    return B.astype(np.float32)


def paper_background_masked(img_rgb_uint8, footprint, sigma=None, bright_win=15, bright_tol=25.0):
    """Paper colour with the mark's footprint left out. float32 (H, W, 3), 0-1.

    'Paper' pixels are outside ``footprint`` AND locally bright
    (``lum >= max_filter(lum, bright_win) - bright_tol`` in 0-255 grey levels:
    drops text, rules, dark graphics). ``B = G*(I m) / G*(m)`` per channel with
    a Gaussian of ``sigma`` (default 1.5% of max(H, W), at least 6 px); holes
    bigger than the blur (a solid disc) are filled from a coarser pyramid level
    until every pixel is defined. Computed at a reduced resolution (the paper
    is smooth) and upsampled."""
    img = np.ascontiguousarray(img_rgb_uint8)
    H, W = img.shape[:2]
    lum = img.astype(np.float32) @ LUMA
    mx = cv2.dilate(lum, np.ones((int(bright_win), int(bright_win)), np.uint8))
    paper = lum >= (mx - float(bright_tol))
    if footprint is not None:
        paper &= ~np.asarray(footprint, bool)
    if sigma is None:
        sigma = max(6.0, 0.015 * max(H, W))
    q = min(1.0, 4.0 / float(sigma))
    m = paper.astype(np.float32)
    num = img.astype(np.float32) * (m[..., None] / 255.0)
    if q < 1.0:
        sw, sh = max(8, int(round(W * q))), max(8, int(round(H * q)))
        num = cv2.resize(num, (sw, sh), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
        sig_w = float(sigma) * sw / W
    else:
        sig_w = float(sigma)
    Bs = _fill_pyramid(np.ascontiguousarray(num), np.ascontiguousarray(m), sig_w)
    if Bs.shape[:2] != (H, W):
        Bs = cv2.resize(Bs, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(Bs, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Template rendering (same resize / placement rules as stamp_fit.render)
# ---------------------------------------------------------------------------

_RS_CACHE = {}
_RS_LOCK = threading.Lock()


def template_size(template, scale):
    return max(2, int(round(template.shape[1] * scale))), max(2, int(round(template.shape[0] * scale)))


def resized_template(template, scale):
    tw, th = template_size(template, scale)
    key = (id(template), template.shape, tw, th)
    with _RS_LOCK:
        hit = _RS_CACHE.get(key)
    if hit is not None and hit[0] is template:
        return hit[1]
    interp = cv2.INTER_AREA if tw < template.shape[1] else cv2.INTER_LINEAR
    out = cv2.resize(template, (tw, th), interpolation=interp)
    with _RS_LOCK:
        if len(_RS_CACHE) > 96:
            _RS_CACHE.pop(next(iter(_RS_CACHE)))
        _RS_CACHE[key] = (template, out)
    return out


def render_template(template, scale, x, y, window):
    """Coverage of ``template`` at (scale, x, y) in page window (x0, y0, w, h)."""
    t = resized_template(template, scale)
    x0, y0, w, h = window
    M = np.float32([[1, 0, x - x0], [0, 1, y - y0]])
    return cv2.warpAffine(t, M, (int(w), int(h)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _crop_pad(a, x0, y0, w, h):
    """a[y0:y0+h, x0:x0+w] with zeros where the window leaves the array."""
    H, W = a.shape[:2]
    out = np.zeros((h, w) + a.shape[2:], a.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + w), min(H, y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


def _pearson(a, b):
    """Pearson correlation over all pixels; for multi-channel inputs the mean of
    each channel is removed separately (what ``cv2.matchTemplate`` does)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.ndim == 3:
        a = a - a.mean(axis=(0, 1), keepdims=True)
        b = b - b.mean(axis=(0, 1), keepdims=True)
    else:
        a = a - a.mean()
        b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def _centre(a):
    """float32 copy with the mean of each channel removed."""
    a = a.astype(np.float32)
    return a - a.mean(axis=(0, 1), keepdims=True) if a.ndim == 3 else a - a.mean()


def _pearson_c(tc, wc, wn):
    """Pearson of an already-centred template ``tc`` with a centred window ``wc``
    of norm ``wn`` (what ``_pearson`` computes, without redoing the window)."""
    den = float(np.sqrt(np.vdot(tc, tc))) * wn
    return float(np.vdot(tc, wc) / den) if den > 1e-12 else 0.0


def inside_fraction(alpha, pose, page_shape, thresh=0.02):
    """Fraction of the template's footprint pixels (``alpha > thresh``) that land
    inside the page at ``pose`` (a mark partly off the page has less than 1)."""
    t = resized_template(alpha, pose["scale"])
    th, tw = t.shape[:2]
    H, W = page_shape
    x0, y0 = int(round(pose["x"])), int(round(pose["y"]))
    sel = t > thresh
    tot = int(sel.sum())
    if tot == 0:
        return 0.0
    ix0, iy0 = max(0, -x0), max(0, -y0)
    ix1, iy1 = min(tw, W - x0), min(th, H - y0)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float(sel[iy0:iy1, ix0:ix1].sum()) / tot


def on_page_pixels(alpha, pose, page_shape, thresh=0.3):
    """Number of on-page pixels where the template rendered at ``pose`` has
    coverage >= ``thresh`` (the size term of Method 5's candidate ranking: a tiny
    fit at the page edge scores a high NCC by chance but has few such pixels)."""
    t = resized_template(alpha, pose["scale"])
    th, tw = t.shape[:2]
    H, W = page_shape
    x0, y0 = int(round(pose["x"])), int(round(pose["y"]))
    ix0, iy0 = max(0, -x0), max(0, -y0)
    ix1, iy1 = min(tw, W - x0), min(th, H - y0)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return int((t[iy0:iy1, ix0:ix1] >= thresh).sum())


# ---------------------------------------------------------------------------
# Multi-scale NCC
# ---------------------------------------------------------------------------

def _dup(c, o):
    cx, cy = c["x"] + c["w"] / 2, c["y"] + c["h"] / 2
    ox, oy = o["x"] + o["w"] / 2, o["y"] + o["h"] / 2
    return (abs(cx - ox) < 0.3 * min(c["w"], o["w"]) and abs(cy - oy) < 0.3 * min(c["h"], o["h"])
            and abs(np.log(c["scale"] / o["scale"])) < 0.2)


def _nms(cands, n_best):
    out = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if not any(_dup(c, o) for o in out):
            out.append(c)
        if len(out) >= n_best:
            break
    return out


def ncc_match(signal, template, scales, per_scale=2, n_best=6, max_side=600, pad_frac=0.5,
              min_mass=0.03, max_templ_w=160):
    """Multi-scale ``cv2.matchTemplate(..., TM_CCOEFF_NORMED)`` of ``template``
    against ``signal`` (both float32, both (h, w) or both (h, w, 3)). ``scales``
    are signal-pixels per template-pixel. The signal is matched at long side <=
    ``max_side`` and zero-padded by ``pad_frac`` of the template size on every
    side, so a mark partly off the page still matches. A scale whose template
    would be wider than ``max_templ_w`` px is matched on a further shrunken
    signal (this stage only proposes positions; the refinement that follows is
    full resolution -- and a page-sized template is what dominated the runtime).
    Windows holding less than ``min_mass`` of the template's own mass are skipped
    (NCC of a near-empty window is numerical noise). Returns up to ``n_best``
    dicts {scale, x, y, score, fm} in the coordinates of ``signal`` (x, y may be
    negative; ``fm`` = the extra shrink factor used, 1 = none), best first."""
    H, W = signal.shape[:2]
    k = min(1.0, float(max_side) / max(H, W))
    sig0 = signal if k >= 1.0 else cv2.resize(signal, (max(1, int(round(W * k))), max(1, int(round(H * k)))),
                                               interpolation=cv2.INTER_AREA)
    sig0 = np.ascontiguousarray(sig0, np.float32)
    kx0, ky0 = sig0.shape[1] / W, sig0.shape[0] / H
    shrunk = {1.0: sig0}
    cands = []
    for s in scales:
        tw0 = template.shape[1] * s * kx0
        fm = 1.0 if tw0 <= max_templ_w else round(max_templ_w / tw0, 2)
        if fm not in shrunk:
            shrunk[fm] = cv2.resize(sig0, (max(1, int(round(sig0.shape[1] * fm))), max(1, int(round(sig0.shape[0] * fm)))),
                                    interpolation=cv2.INTER_AREA)
        sig = shrunk[fm]
        kx, ky = sig.shape[1] / W, sig.shape[0] / H
        tw = int(round(template.shape[1] * s * kx))
        th = int(round(template.shape[0] * s * ky))
        if tw < 6 or th < 6:
            continue
        interp = cv2.INTER_AREA if tw < template.shape[1] else cv2.INTER_LINEAR
        t = cv2.resize(template, (tw, th), interpolation=interp).astype(np.float32)
        tmass = float(t.sum())
        if tmass < 1e-3:
            continue
        px, py = int(tw * pad_frac), int(th * pad_frac)
        padded = cv2.copyMakeBorder(sig, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
        r = cv2.matchTemplate(padded, t, cv2.TM_CCOEFF_NORMED)
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
        ii = cv2.integral(padded if padded.ndim == 2 else padded.sum(axis=2))
        rh, rw = r.shape
        mass = (ii[th:th + rh, tw:tw + rw] - ii[0:rh, tw:tw + rw]
                - ii[th:th + rh, 0:rw] + ii[0:rh, 0:rw])
        r = np.where(mass >= min_mass * tmass, r, -1.0).astype(np.float32)
        for _ in range(per_scale):
            _, v, _, loc = cv2.minMaxLoc(r)
            if v <= 0:
                break
            cands.append(dict(scale=float(s), x=float((loc[0] - px) / kx), y=float((loc[1] - py) / ky),
                              score=float(min(v, 1.0)), w=tw / kx, h=th / ky, fm=float(fm)))
            r[max(0, loc[1] - th // 4):loc[1] + th // 4 + 1, max(0, loc[0] - tw // 4):loc[0] + tw // 4 + 1] = -1.0
    out = _nms(cands, n_best)
    for c in out:
        c.pop("w"), c.pop("h")
    return out


# ---------------------------------------------------------------------------
# Local refinement
# ---------------------------------------------------------------------------

def refine_pose(signal, template, pose, radius=6, scale_span=0.02, scale_step=0.005,
                subpixel=True, origin=(0, 0)):
    """Local search around ``pose`` = dict(scale, x, y) (page coordinates).

    ``signal`` covers the page region whose top-left is ``origin`` (pixels
    outside it count as zero); ``signal`` and ``template`` are both 1-channel or
    both 3-channel (the score is then the colour-matched correlation). Stages:
    integer (x, y) within +-radius; scale in 1 +- scale_span (step scale_step,
    centre kept); then quarter-pixel (dx, dy) in -0.5..0.5. Score = NCC of the
    rendered template against the signal over the template's box. Returns
    (pose, score)."""
    ox, oy = origin
    s = float(pose["scale"])
    cx = pose["x"] - ox
    cy = pose["y"] - oy

    def best_shift(sc, px, py, rad):
        t = resized_template(template, sc).astype(np.float32)
        th, tw = t.shape[:2]
        ix, iy = int(round(px)), int(round(py))
        win = _crop_pad(signal, ix - rad, iy - rad, tw + 2 * rad, th + 2 * rad)
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, v, _, loc = cv2.minMaxLoc(r)
        return float(v), ix - rad + loc[0], iy - rad + loc[1]

    score, bx, by = best_shift(s, cx, cy, int(radius))
    if scale_span > 0:
        factors = np.arange(-scale_span, scale_span + 1e-9, scale_step)
        tw0, th0 = template_size(template, s)
        for f in factors:
            if abs(f) < 1e-9:
                continue
            s2 = s * (1.0 + f)
            tw2, th2 = template_size(template, s2)
            # keep the centre fixed while the scale changes
            px = bx + (tw0 - tw2) / 2.0
            py = by + (th0 - th2) / 2.0
            v, x2, y2 = best_shift(s2, px, py, 2)
            if v > score + 1e-6:
                score, s, bx, by = v, s2, x2, y2
                tw0, th0 = tw2, th2
    if subpixel and scale_span > 0:
        # finer scale passes (a 0.25% scale error is already ~1.4 px at the far end of a 560 px mark)
        for step in (scale_step / 2.0, scale_step / 4.0):
            for f in (-step, step):
                tw0, th0 = template_size(template, s)
                s2 = s * (1.0 + f)
                tw2, th2 = template_size(template, s2)
                if (tw2, th2) == (tw0, th0):
                    continue
                v, x2, y2 = best_shift(s2, bx + (tw0 - tw2) / 2.0, by + (th0 - th2) / 2.0, 1)
                if v > score + 1e-6:
                    score, s, bx, by = v, s2, x2, y2
    fx = fy = 0.0
    if subpixel:
        t = resized_template(template, s).astype(np.float32)
        th, tw = t.shape[:2]
        win = _crop_pad(signal, int(bx), int(by), tw, th).astype(np.float32)
        wc = _centre(win)
        wn = float(np.sqrt(np.vdot(wc, wc)))
        best = (_pearson_c(_centre(t), wc, wn), 0.0, 0.0)
        for dy in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for dx in (-0.5, -0.25, 0.0, 0.25, 0.5):
                if dx == 0.0 and dy == 0.0:
                    continue
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                ts_ = cv2.warpAffine(t, M, (tw, th), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                v = _pearson_c(_centre(ts_), wc, wn)
                if v > best[0] + 1e-6:
                    best = (v, dx, dy)
        score, fx, fy = best
    return dict(scale=float(s), x=float(bx + fx + ox), y=float(by + fy + oy)), float(score)


# ---------------------------------------------------------------------------
# Page-level driver
# ---------------------------------------------------------------------------

def _exclude(sig, exclusions, f, origin=(0, 0)):
    """Zero the dilated footprints of already-accepted marks (page-coordinate
    poses) in ``sig``, a signal at ``f`` working px per page px whose top-left
    is page pixel ``origin``."""
    if not exclusions:
        return
    h, w = sig.shape[:2]
    ox, oy = origin
    for tpl, pose in exclusions:
        cov = render_template(tpl, pose["scale"] * f, (pose["x"] - ox) * f, (pose["y"] - oy) * f, (0, 0, w, h))
        foot = (cov > 0.02).astype(np.uint8)
        foot = cv2.dilate(foot, np.ones((_EXCL_DILATE, _EXCL_DILATE), np.uint8)) > 0
        sig[foot] = 0.0


def _kernel_groups(scales, kfun, ratio=KERNEL_GROUP_RATIO):
    """Scales grouped (ascending, each group spanning at most ``ratio``) with the
    closing kernel of each group's LARGEST scale; groups with the same kernel
    merge. A page's background is estimated once per kernel."""
    s = sorted(float(v) for v in scales)
    groups, cur = [], []
    for v in s:
        if cur and v / cur[0] > ratio:
            groups.append(cur)
            cur = []
        cur.append(v)
    if cur:
        groups.append(cur)
    merged = {}
    for g in groups:
        merged.setdefault(int(kfun(g[-1])), []).extend(g)
    return merged


def locate_template(img_rgb_uint8, template, scales, kernel_px, n_cands=3, exclusions=None, scan_side=450,
                    coarse_side=800, color=None, polish=False):
    """Best placement of ``template`` on a page, or None.

    ``template``: (h, w) coverage (peak 1). ``color``: optional (h, w, 3)
    colour template (``colour_template``); when given the match is colour-matched
    (3-channel signal, 3-channel correlation), otherwise it is the v1
    luminance-like match of the coverage. ``scales``: page-pixels per
    template-pixel to search. ``kernel_px``: the paper-background closing kernel
    in page pixels, an int or a callable ``scale -> int`` (scales are searched in
    groups, each with the kernel of its largest scale). ``exclusions``: list of
    (coverage template, pose) already accepted on this page, whose footprint is
    zeroed in the signal first. ``polish``: after the full-resolution
    refinement, re-score each candidate against the MARK-AWARE background (its
    footprint left out) and refine once more. Returns dict(scale, x, y, score)
    in page coordinates, with the other tried candidates in ``"candidates"``."""
    img = img_rgb_uint8
    H, W = img.shape[:2]
    colour = color is not None
    mt = color if colour else template
    sig_fn = page_signal3 if colour else page_signal
    kfun = kernel_px if callable(kernel_px) else (lambda s, _k=int(kernel_px): _k)
    groups = _kernel_groups(scales, kfun)
    Tw, Th = template.shape[1], template.shape[0]
    sigs, cands = {}, []
    for kern, sc in groups.items():
        sig, f = sig_fn(img, kern, coarse_side)
        _exclude(sig, exclusions, f)
        sigs[kern] = (sig, f)
        for c in ncc_match(sig, mt, [s * f for s in sc], max_side=scan_side, n_best=max(n_cands, 3)):
            sc_p = c["scale"] / f
            cands.append(dict(scale=sc_p, x=c["x"] / f, y=c["y"] / f, score=c["score"],
                              w=Tw * sc_p, h=Th * sc_p, kern=kern, f=f, c_scale=c["scale"], c_x=c["x"], c_y=c["y"],
                              fm=c["fm"]))
    cands = _nms(cands, COARSE_POOL * n_cands)
    if not cands:
        return None
    # ---- stage 1: refine every candidate on the coarse signal (covers the scale-grid gap) ----
    stage1 = []
    for c in cands:
        sig, f = sigs[c["kern"]]
        q1 = min(1.0, STAGE1_W / max(Tw * c["c_scale"], 1.0))      # a 400 px template does not need a 400 px search
        if q1 < 0.95:
            sg = cv2.resize(sig, None, fx=q1, fy=q1, interpolation=cv2.INTER_AREA)
            q1 = sg.shape[1] / float(sig.shape[1])
        else:
            sg, q1 = sig, 1.0
        pose_c = dict(scale=c["c_scale"] * q1, x=c["c_x"] * q1, y=c["c_y"] * q1)
        pose_c, sc1 = refine_pose(sg, mt, pose_c, radius=max(3, int(np.ceil(1.5 / c["fm"])) + 1), scale_span=0.06,
                                  scale_step=0.01, subpixel=False)
        stage1.append((c, dict(scale=pose_c["scale"] / q1 / f, x=pose_c["x"] / q1 / f, y=pose_c["y"] / q1 / f), sc1))
    # the coarse (closing-background) ranking is noisy: take a wider pool through the cheap coarse refinement and
    # keep the best n_cands of it for the full-resolution and mark-aware passes
    stage1 = sorted(stage1, key=lambda t: -t[2])[:n_cands]
    best = None
    tried = []
    for c, pose, sc1 in stage1:
        sig, f = sigs[c["kern"]]
        kernel = c["kern"]
        tw, th = template_size(template, pose["scale"])
        # full-resolution pass, but never finer than ~MAX_REFINE_W px across the mark (1 px of a 500 px mark is noise)
        gw = min(1.0, MAX_REFINE_W / float(max(tw, 1)))
        if f >= 0.999:
            q = min(1.0, gw)
            sg = sig if q >= 0.98 else cv2.resize(sig, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
            fw = sg.shape[1] / float(W)
            pose_w = dict(scale=pose["scale"] * fw, x=pose["x"] * fw, y=pose["y"] * fw)
            fine_w, score = refine_pose(sg, mt, pose_w, radius=3, scale_span=0.02, scale_step=0.005)
            fine_pose = dict(scale=fine_w["scale"] / fw, x=fine_w["x"] / fw, y=fine_w["y"] / fw)
        elif gw <= f * 1.02:
            q = min(1.0, gw / f)
            sg = sig if q >= 0.98 else cv2.resize(sig, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
            fw = f * sg.shape[1] / float(sig.shape[1])
            pose_w = dict(scale=pose["scale"] * fw, x=pose["x"] * fw, y=pose["y"] * fw)
            fine_w, score = refine_pose(sg, mt, pose_w, radius=3, scale_span=0.02, scale_step=0.005)
            fine_pose = dict(scale=fine_w["scale"] / fw, x=fine_w["x"] / fw, y=fine_w["y"] / fw)
        else:
            rad = int(np.ceil(1.5 / gw)) + 2
            m = int(kernel) + rad + 4
            wx0 = int(max(0, np.floor(pose["x"]) - m)); wy0 = int(max(0, np.floor(pose["y"]) - m))
            wx1 = int(min(W, np.ceil(pose["x"] + tw) + m)); wy1 = int(min(H, np.ceil(pose["y"] + th) + m))
            if wx1 - wx0 < 8 or wy1 - wy0 < 8:
                continue
            crop = np.ascontiguousarray(img[wy0:wy1, wx0:wx1])
            if colour:
                sf = band_signal3(crop, paper_background(crop, kernel))
            else:
                sf = band_signal(crop, paper_background(crop, kernel))
            sf = normalise3(sf) if colour else normalise(sf)
            _exclude(sf, exclusions, 1.0, origin=(wx0, wy0))
            if gw < 0.999:
                sf = cv2.resize(sf, None, fx=gw, fy=gw, interpolation=cv2.INTER_AREA)
            fw = sf.shape[1] / float(wx1 - wx0)
            pose_w = dict(scale=pose["scale"] * fw, x=pose["x"] * fw, y=pose["y"] * fw)
            fine_w, score = refine_pose(sf, mt, pose_w, radius=max(2, int(np.ceil(1.5 * fw / gw)) + 1), scale_span=0.02,
                                        scale_step=0.005, origin=(wx0 * fw, wy0 * fw))
            fine_pose = dict(scale=fine_w["scale"] / fw, x=fine_w["x"] / fw, y=fine_w["y"] / fw)
        if polish and score > 0.05:
            fp = footprint_mask(template, fine_pose, (H, W))
            Bm = paper_background_masked(img, fp)
            sp = normalise3(band_signal3(img, Bm)) if colour else normalise(band_signal(img, Bm))
            _exclude(sp, exclusions, 1.0)
            tw2 = template_size(template, fine_pose["scale"])[0]
            q2 = min(1.0, MAX_REFINE_W / float(max(tw2, 1)))
            if q2 < 0.95:
                sp = cv2.resize(sp, None, fx=q2, fy=q2, interpolation=cv2.INTER_AREA)
                q2 = sp.shape[1] / float(W)
            else:
                q2 = 1.0
            pw = dict(scale=fine_pose["scale"] * q2, x=fine_pose["x"] * q2, y=fine_pose["y"] * q2)
            p2, s2 = refine_pose(sp, mt, pw, radius=2, scale_span=0.01, scale_step=0.0025)
            if s2 > score:
                fine_pose, score = dict(scale=p2["scale"] / q2, x=p2["x"] / q2, y=p2["y"] / q2), s2
        tried.append(dict(fine_pose, score=score))
        if best is None or score > best["score"]:
            best = dict(fine_pose, score=score)
    if best is None:
        return None
    best["candidates"] = tried
    return best


def dark_fraction(img_rgb_uint8, template, pose, kernel_px, min_dark=0.02):
    """Fraction of the template's stroke pixels (coverage >= 0.5) at ``pose``
    where the page is darker than its paper background by more than
    ``min_dark`` (0-1). A real darker-than-paper mark is ~1; a lighter mark,
    or a fit on unrelated clutter on a dark page, is much lower."""
    H, W = img_rgb_uint8.shape[:2]
    tw, th = template_size(template, pose["scale"])
    m = int(kernel_px) + 4
    x0, y0 = max(0, int(np.floor(pose["x"])) - m), max(0, int(np.floor(pose["y"])) - m)
    x1, y1 = min(W, int(np.ceil(pose["x"] + tw)) + m), min(H, int(np.ceil(pose["y"] + th)) + m)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    crop = np.ascontiguousarray(img_rgb_uint8[y0:y1, x0:x1])
    B = paper_background(crop, kernel_px)
    D = (B - crop.astype(np.float32) / 255.0).max(axis=2)
    cov = render_template(template, pose["scale"], pose["x"], pose["y"], (x0, y0, x1 - x0, y1 - y0))
    stroke = cov >= 0.5
    if stroke.sum() < 20:
        return 0.0
    return float((D[stroke] > min_dark).mean())
