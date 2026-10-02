"""Model-free 'faint darkening' signal shared by the Template Builder and
Method 5 (Template Stamp Fit).

A semi-transparent mark that a site stamps on every page only ever makes the
page a little darker than the paper underneath it. The paper underneath is
easy to estimate without any model: a morphological closing with a kernel a
bit wider than the mark's strokes erases everything thin and dark (the mark's
own strokes AND page text) and keeps the lightest thing locally, which is the
paper. What the page is darker than that paper is the 'darkening'. Real ink
(text, rules) is far darker than any watermark, so pixels above a high
threshold, and a thin rim around them (anti-aliasing), are zeroed. What is
left is the faint mark and little else, and that is what templates are
matched against -- no trained network involved.

Contents
--------
* ``paper_background`` / ``darkening`` / ``band_signal`` / ``normalise``:
  the signal itself.
* ``ncc_match``: multi-scale normalised cross-correlation of a template
  against a signal (padded so marks hanging off the page edge still match).
* ``refine_pose``: local scale / sub-pixel search around a candidate.
* ``locate_template``: the page-level driver both callers use (coarse match
  on a downscaled page, then a full-resolution refinement in a window).

Pose convention (identical to ``stamp_fit``): ``scale`` is page pixels per
template pixel, ``(x, y)`` is where the (round-sized) resized template's
top-left lands on the page, float.

Known limits: the mark must be DARKER than the page; rotation is fixed at 0.
"""

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
MAX_KERNEL = 61
_EXCL_DILATE = 9


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


def kernel_for(stroke_width_at_scale):
    """k = max(7, round(1.4 * stroke width)), capped at MAX_KERNEL, odd."""
    k = int(max(7, round(1.4 * float(stroke_width_at_scale))))
    return int(min(MAX_KERNEL, k) | 1)


def paper_background(img_rgb_uint8, kernel_px):
    """Local paper colour: closing with an elliptical kernel wider than the
    strokes. float32 (H, W, 3) in 0-1."""
    k = max(5, int(kernel_px) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(np.ascontiguousarray(img_rgb_uint8), cv2.MORPH_CLOSE, se)
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


def page_signal(img_rgb_uint8, kernel_px, max_side=800):
    """Normalised band signal of a whole page at working resolution.

    The paper background is estimated on a downscaled copy (cheap) and
    upsampled; the darkening, the text zeroing and its rim are done at FULL
    resolution and only then is the signal shrunk -- zeroing text after
    shrinking leaves blurred text bars below the threshold that look like
    mark strokes. Returns (signal, f) with ``f`` = working px per page px."""
    H, W = img_rgb_uint8.shape[:2]
    f = min(1.0, float(max_side) / max(H, W))
    if f >= 1.0:
        B = paper_background(img_rgb_uint8, kernel_px)
        return normalise(band_signal(img_rgb_uint8, B)), 1.0
    sw, sh = max(1, int(round(W * f))), max(1, int(round(H * f)))
    small = cv2.resize(img_rgb_uint8, (sw, sh), interpolation=cv2.INTER_AREA)
    kc = max(5, int(round(kernel_px * f)) | 1)
    Bs = paper_background(small, kc)
    B = cv2.resize(Bs, (W, H), interpolation=cv2.INTER_LINEAR)
    sig = cv2.resize(band_signal(img_rgb_uint8, B), (sw, sh), interpolation=cv2.INTER_AREA)
    return normalise(sig), sw / W


# ---------------------------------------------------------------------------
# Template rendering (same resize / placement rules as stamp_fit.render)
# ---------------------------------------------------------------------------

_RS_CACHE = {}


def template_size(template, scale):
    return max(2, int(round(template.shape[1] * scale))), max(2, int(round(template.shape[0] * scale)))


def resized_template(template, scale):
    tw, th = template_size(template, scale)
    key = (id(template), template.shape, tw, th)
    hit = _RS_CACHE.get(key)
    if hit is not None and hit[0] is template:
        return hit[1]
    interp = cv2.INTER_AREA if tw < template.shape[1] else cv2.INTER_LINEAR
    out = cv2.resize(template, (tw, th), interpolation=interp)
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
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Multi-scale NCC
# ---------------------------------------------------------------------------

def ncc_match(signal, template, scales, per_scale=2, n_best=6, max_side=600, pad_frac=0.5,
              min_mass=0.03):
    """Multi-scale ``cv2.matchTemplate(..., TM_CCOEFF_NORMED)`` of ``template``
    against ``signal`` (both float32). ``scales`` are signal-pixels per
    template-pixel. The signal is matched at long side <= ``max_side`` and
    zero-padded by ``pad_frac`` of the template size on every side, so a mark
    partly off the page still matches. Windows holding less than ``min_mass``
    of the template's own mass are skipped (NCC of a near-empty window is
    numerical noise). Returns up to ``n_best`` dicts {scale, x, y, score} in
    the coordinates of ``signal`` (x, y may be negative), best first."""
    H, W = signal.shape
    k = min(1.0, float(max_side) / max(H, W))
    sig = signal if k >= 1.0 else cv2.resize(signal, (max(1, int(round(W * k))), max(1, int(round(H * k)))),
                                              interpolation=cv2.INTER_AREA)
    kx, ky = sig.shape[1] / W, sig.shape[0] / H
    sig = np.ascontiguousarray(sig, np.float32)
    cands = []
    for s in scales:
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
        ii = cv2.integral(padded)
        rh, rw = r.shape
        mass = (ii[th:th + rh, tw:tw + rw] - ii[0:rh, tw:tw + rw]
                - ii[th:th + rh, 0:rw] + ii[0:rh, 0:rw])
        r = np.where(mass >= min_mass * tmass, r, -1.0).astype(np.float32)
        for _ in range(per_scale):
            _, v, _, loc = cv2.minMaxLoc(r)
            if v <= 0:
                break
            cands.append(dict(scale=float(s), x=float((loc[0] - px) / kx), y=float((loc[1] - py) / ky),
                              score=float(min(v, 1.0)), w=tw / kx, h=th / ky))
            r[max(0, loc[1] - th // 4):loc[1] + th // 4 + 1, max(0, loc[0] - tw // 4):loc[0] + tw // 4 + 1] = -1.0
    cands.sort(key=lambda c: -c["score"])
    out = []
    for c in cands:
        cx, cy = c["x"] + c["w"] / 2, c["y"] + c["h"] / 2
        dup = False
        for o in out:
            ox, oy = o["x"] + o["w"] / 2, o["y"] + o["h"] / 2
            if (abs(cx - ox) < 0.3 * min(c["w"], o["w"]) and abs(cy - oy) < 0.3 * min(c["h"], o["h"])
                    and abs(np.log(c["scale"] / o["scale"])) < 0.2):
                dup = True
                break
        if not dup:
            out.append(c)
        if len(out) >= n_best:
            break
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
    outside it count as zero). Stages: integer (x, y) within +-radius; scale
    in 1 +- scale_span (step scale_step, centre kept); then quarter-pixel
    (dx, dy) in -0.5..0.5. Score = NCC of the rendered template against the
    signal over the template's box. Returns (pose, score)."""
    ox, oy = origin
    s = float(pose["scale"])
    cx = pose["x"] - ox
    cy = pose["y"] - oy

    def best_shift(sc, px, py, rad):
        t = resized_template(template, sc).astype(np.float32)
        th, tw = t.shape
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
        th, tw = t.shape
        win = _crop_pad(signal, int(bx), int(by), tw, th)
        best = (_pearson(t, win), 0.0, 0.0)
        for dy in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for dx in (-0.5, -0.25, 0.0, 0.25, 0.5):
                if dx == 0.0 and dy == 0.0:
                    continue
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                ts = cv2.warpAffine(t, M, (tw, th), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                v = _pearson(ts, win)
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
    h, w = sig.shape
    ox, oy = origin
    for tpl, pose in exclusions:
        cov = render_template(tpl, pose["scale"] * f, (pose["x"] - ox) * f, (pose["y"] - oy) * f, (0, 0, w, h))
        foot = (cov > 0.02).astype(np.uint8)
        foot = cv2.dilate(foot, np.ones((_EXCL_DILATE, _EXCL_DILATE), np.uint8)) > 0
        sig[foot] = 0.0


def locate_template(img_rgb_uint8, template, scales, kernel_px, n_cands=3, exclusions=None, scan_side=450,
                    coarse_side=800):
    """Best placement of ``template`` on a page, or None.

    ``scales``: page-pixels per template-pixel to search. ``kernel_px``: the
    paper-background closing kernel in page pixels. ``exclusions``: list of
    (template, pose) already accepted on this page, whose footprint is
    zeroed in the signal first. Returns dict(scale, x, y, score) in page
    coordinates, with the other tried candidates in ``"candidates"``."""
    H, W = img_rgb_uint8.shape[:2]
    sig, f = page_signal(img_rgb_uint8, kernel_px, coarse_side)
    _exclude(sig, exclusions, f)
    cands = ncc_match(sig, template, [s * f for s in scales], max_side=scan_side, n_best=max(n_cands, 3))
    cands = cands[:n_cands]
    if not cands:
        return None
    best = None
    tried = []
    for c in cands:
        # stage 1: refine on the coarse signal (covers the scale-grid gap)
        pose_c = dict(scale=c["scale"], x=c["x"], y=c["y"])
        pose_c, _ = refine_pose(sig, template, pose_c, radius=3, scale_span=0.05, scale_step=0.01,
                                subpixel=False)
        pose = dict(scale=pose_c["scale"] / f, x=pose_c["x"] / f, y=pose_c["y"] / f)
        if f >= 0.999:
            fine_pose, score = refine_pose(sig, template, pose, radius=3, scale_span=0.02, scale_step=0.005)
        else:
            tw, th = template_size(template, pose["scale"])
            rad = int(np.ceil(1.5 / f)) + 2
            m = int(kernel_px) + rad + 4
            wx0 = int(max(0, np.floor(pose["x"]) - m)); wy0 = int(max(0, np.floor(pose["y"]) - m))
            wx1 = int(min(W, np.ceil(pose["x"] + tw) + m)); wy1 = int(min(H, np.ceil(pose["y"] + th) + m))
            if wx1 - wx0 < 8 or wy1 - wy0 < 8:
                continue
            crop = np.ascontiguousarray(img_rgb_uint8[wy0:wy1, wx0:wx1])
            sf = normalise(band_signal(crop, paper_background(crop, kernel_px)))
            _exclude(sf, exclusions, 1.0, origin=(wx0, wy0))
            fine_pose, score = refine_pose(sf, template, pose, radius=rad, scale_span=0.02, scale_step=0.005,
                                           origin=(wx0, wy0))
        tried.append(dict(fine_pose, score=score))
        if best is None or score > best["score"]:
            best = dict(fine_pose, score=score)
    if best is None:
        return None
    best["candidates"] = tried
    best["dark_frac"] = dark_fraction(img_rgb_uint8, template, best, kernel_px)
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
