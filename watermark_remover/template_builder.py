"""Template Builder: estimate a site's watermark from a folder of its pages.

A watermark that a site stamps on every page is the one thing those pages
have in common: the page content differs, the mark does not. Given 20+ pages
and one box around the mark on one of them, this module

1. makes a rough seed template from the faint darkening inside the box
   (``template_signal``: paper background by closing, text zeroed),
2. registers every page against it (multi-scale NCC + local refinement; pages
   with a different mark, or none, fail the score and are rejected),
3. warps all accepted pages into one common frame and takes the per-pixel
   MEDIAN of their luminance gradients (Dekel et al., "On the Effectiveness of
   Visible Watermarks", CVPR 2017, stage 1): the page content's gradients
   differ page to page and cancel in the median, the mark's gradient is the
   same on every page and survives; Poisson integration (DST-I solver, zero
   Dirichlet boundary at the frame edge) turns the gradients back into an
   image of the mark,
4. refines it with an alternating minimisation of the image-formation model
   ``B_i - I_i = o_i * u(p) * (B_i - k_r)`` (B_i = the page's paper
   background, o_i = the page's strength, u = coverage shape, k_r = the ink of
   the pixel's region): per-page strength by trimmed least squares, per-pixel
   coverage by a weighted, Tukey-biweight-reweighted least squares across
   pages and channels,
5. repeats registration with the refined template (``outer_iters`` times).

The result is a library template (``template_library``) that Method 5
(``template_stamp_fit``) removes with Stamp Fit's exact inverse.

Known limits
------------
- Assumes the mark is DARKER than the page (true for AriaTender and
  ETENDER). A light mark on a dark banner is not estimated.
- Rotation is fixed at 0; the layout of a multi-part mark is whatever the
  seed page shows (one rigid template).
- Ink luminance is fixed by ``stamp_fit.INK_LUM_PRIOR`` (unobservable on flat
  paper); it only matters for content under the mark.
- Pages where the site placed a different variant of the mark are rejected
  during the build: build one template per variant.

Everything here is deterministic for a fixed input and page order.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from scipy import fft as sfft
from scipy import ndimage as ndi

from . import stamp_fit
from . import template_library as tl
from . import template_signal as ts

BUILDER_VERSION = 1
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LUMA = ts.LUMA
MAX_DARK = 0.35        # |B - I| beyond this is page content, not the mark
MIN_PAGES = 5
MIN_DARK_FRAC = 0.4    # share of stroke pixels that must be darker than the paper (mark is darker, see docstring)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def list_pages(pages_dir):
    """Sorted image files directly inside ``pages_dir`` (non-recursive)."""
    d = tl.resolve_path(pages_dir)
    if not d or not os.path.isdir(d):
        return []
    return [os.path.join(d, n) for n in sorted(os.listdir(d))
            if n.lower().endswith(IMG_EXT) and os.path.isfile(os.path.join(d, n))]


def read_rgb(path):
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def _nanmedian0(a):
    """Median over axis 0 ignoring NaN. Returns (median, count)."""
    n = np.sum(~np.isnan(a), axis=0)
    srt = np.sort(a, axis=0)                   # NaN sorts last
    last = a.shape[0] - 1
    lo = np.clip((n - 1) // 2, 0, last)
    hi = np.clip(n // 2, 0, last)
    m = 0.5 * (np.take_along_axis(srt, lo[None], 0)[0] + np.take_along_axis(srt, hi[None], 0)[0])
    m = np.where(n > 0, m, np.nan)
    return m.astype(np.float32), n


def _bbox(mask, margin=0, shape=None):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1, y0, y1 = xs.min() - margin, xs.max() + 1 + margin, ys.min() - margin, ys.max() + 1 + margin
    if shape is not None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1], x1), min(shape[0], y1)
    return int(x0), int(y0), int(x1), int(y1)


def _remove_small(u, thresh, min_px):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((u > thresh).astype(np.uint8), connectivity=8)
    out = u.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            out[lab == i] = 0
    return out


def _prune(u):
    """Zero pixels below 0.03, drop connected components of u > 0.1 smaller than
    max(20, 0.0005 x frame area, 0.3% of all such pixels), and zero everything
    that is not within 3 px of a kept component (isolated faint speckle)."""
    u = u.copy()
    u[u < 0.03] = 0
    m = (u > 0.1).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return u
    total = int(stats[1:, cv2.CC_STAT_AREA].sum())
    min_px = max(20, 0.0005 * u.size, 0.003 * total)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    kept = keep[lab]
    near = cv2.dilate(kept.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    u[~near] = 0
    return u


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-9)


# ---------------------------------------------------------------------------
# Seed template
# ---------------------------------------------------------------------------

def _seed_template(img, box):
    """Rough template from the faint darkening inside ``box`` = (x, y, w, h):
    the box plus a 20% zero margin on each side, at seed-page scale."""
    H, W = img.shape[:2]
    x, y, w, h = box
    k = int(np.clip(round(0.12 * min(w, h)), 15, 61)) | 1
    m = k + 6
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    sig = ts.normalise(ts.band_signal(crop, ts.paper_background(crop, k)))
    c0 = sig[y - y0:y - y0 + h, x - x0:x - x0 + w].copy()
    c0[c0 < 0.15] = 0
    c0 = _remove_small(c0, 0.1, 20)
    mx, my = int(round(0.2 * w)), int(round(0.2 * h))
    tpl = np.zeros((h + 2 * my, w + 2 * mx), np.float32)
    tpl[my:my + h, mx:mx + w] = c0
    return tpl


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _register_one(path, tpl, tsw, s_nom, lo, hi, n_scales, min_score):
    img = read_rgb(path)
    if img is None:
        return dict(accepted=False, score=0.0, scale=0.0, x=0.0, y=0.0, reason="unreadable file", size=None)
    H, W = img.shape[:2]
    sn = s_nom(W)
    scales = sn * np.geomspace(lo, hi, n_scales)
    kernel = ts.kernel_for(tsw * sn * hi)
    res = ts.locate_template(img, tpl, scales, kernel)
    if res is None:
        return dict(accepted=False, score=0.0, scale=0.0, x=0.0, y=0.0, reason="no candidate found", size=(W, H))
    ok = res["score"] >= min_score and res["dark_frac"] >= MIN_DARK_FRAC
    if res["score"] < min_score:
        reason = f"best registration score {res['score']:.2f} < {min_score:.2f} (different mark or none)"
    elif not ok:
        reason = (f"the page is not darker than its paper along the matched strokes "
                  f"({res['dark_frac']:.2f} of stroke pixels < {MIN_DARK_FRAC}); lighter marks are not supported")
    else:
        reason = ""
    return dict(accepted=bool(ok), score=float(res["score"]), scale=float(res["scale"]), x=float(res["x"]),
                y=float(res["y"]), size=(W, H), reason=reason)


def _register_all(files, tpl, s_nom, lo, hi, n_scales, min_score, workers):
    """Registers every page. A second, scale-constrained pass follows: the mark
    is stamped at a size proportional to the page width, so once a few pages
    matched confidently (score >= 0.5) their median relative width is a
    strong prior. Pages that did not match, or matched at a size far from it
    (a fit on page clutter), are searched again within +-25% of that width;
    pages still inconsistent are rejected."""
    tsw = ts.stroke_width(tpl)
    Tw = tpl.shape[1]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda p: _register_one(p, tpl, tsw, s_nom, lo, hi, n_scales, min_score), files))
    for p, r in zip(files, res):
        r["file"] = os.path.basename(p)
        r["path"] = p

    def rel(r):
        return r["scale"] * Tw / r["size"][0]

    conf = [rel(r) for r in res if r["accepted"] and r["score"] >= 0.5]
    if len(conf) >= 3:
        m = float(np.median(conf))
        redo = [i for i, r in enumerate(res) if r["size"] is not None and
                (not r["accepted"] or not (0.8 <= rel(r) / m <= 1.25))]
        prior = lambda Wp, _m=m, _t=Tw: _m * Wp / float(_t)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            redone = list(ex.map(lambda i: _register_one(files[i], tpl, tsw, prior, 0.8, 1.25, 12, min_score), redo))
        for i, r2 in zip(redo, redone):
            old = res[i]
            r2["file"], r2["path"] = old["file"], old["path"]
            if r2["accepted"] and 0.8 <= rel(r2) / m <= 1.25:
                res[i] = r2
            elif old["accepted"]:
                old["accepted"] = False
                old["reason"] = (f"matched at {rel(old) / m:.2f}x the typical mark width, "
                                 f"inconsistent with the other pages (probably clutter)")
    return res


# ---------------------------------------------------------------------------
# Frame + warp
# ---------------------------------------------------------------------------

def _frame_geometry(tpl_shape, rows, pad_frac, frame_max_width):
    Th, Tw = tpl_shape
    px, py = int(round(pad_frac * Tw)), int(round(pad_frac * Th))
    G = float(np.percentile([r["scale"] for r in rows], 75))
    G = min(G, frame_max_width / float(Tw + 2 * px))
    G = max(G, 8.0 / float(Tw + 2 * px))
    Fw, Fh = int(round((Tw + 2 * px) * G)), int(round((Th + 2 * py) * G))
    return dict(px=px, py=py, G=G, Fw=max(8, Fw), Fh=max(8, Fh), Tw=Tw, Th=Th)


def _to_frame(src, x0, y0, ox, oy, r, geo, fill_valid=False):
    """Warp a page crop (top-left at page (x0, y0)) into the frame whose
    top-left is page (ox, oy) and which has ``r`` page px per frame px."""
    Fw, Fh = geo["Fw"], geo["Fh"]
    q = min(1.0, 1.0 / r)
    if q < 1.0:
        sw, sh = max(1, int(round(src.shape[1] * q))), max(1, int(round(src.shape[0] * q)))
        small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        qx, qy = sw / src.shape[1], sh / src.shape[0]
    else:
        small, qx, qy = src, 1.0, 1.0
    M = np.float32([[r * qx, 0, (ox - x0) * qx + (r * qx - 1) / 2.0],
                    [0, r * qy, (oy - y0) * qy + (r * qy - 1) / 2.0]])
    return cv2.warpAffine(small, M, (Fw, Fh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0), (small, M)


def _warp_page(row, geo, tsw):
    img = read_rgb(row["path"])
    if img is None:
        return None
    H, W = img.shape[:2]
    s, x, y = row["scale"], row["x"], row["y"]
    G, Fw, Fh = geo["G"], geo["Fw"], geo["Fh"]
    r = s / G
    ox, oy = x - s * geo["px"], y - s * geo["py"]
    k = ts.kernel_for(tsw * s)
    x0 = max(0, int(np.floor(ox)) - k - 2); y0 = max(0, int(np.floor(oy)) - k - 2)
    x1 = min(W, int(np.ceil(ox + r * Fw)) + k + 2); y1 = min(H, int(np.ceil(oy + r * Fh)) + k + 2)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    Bc = np.rint(ts.paper_background(crop, k) * 255).astype(np.uint8)
    I, (small, M) = _to_frame(crop, x0, y0, ox, oy, r, geo)
    Bf, _ = _to_frame(Bc, x0, y0, ox, oy, r, geo)
    ones = np.full(small.shape[:2], 255, np.uint8)
    V = cv2.warpAffine(ones, M, (Fw, Fh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 250
    return I, Bf, V


# ---------------------------------------------------------------------------
# Stage 1: Dekel et al. 2017 -- median gradients + Poisson integration
# ---------------------------------------------------------------------------

def _dekel_initial(I, V, min_valid):
    N, Fh, Fw = V.shape
    Y = np.empty((N, Fh, Fw), np.float32)
    for i in range(N):
        Y[i] = I[i].astype(np.float32) @ (LUMA / 255.0)
    Y[~V] = np.nan
    Gx = np.zeros((Fh, Fw), np.float32)
    Gy = np.zeros((Fh, Fw), np.float32)
    R = max(8, int(6e6 // max(1, N * Fw)))
    for r0 in range(0, Fh, R):
        r1 = min(Fh, r0 + R)
        gx = Y[:, r0:r1, 1:] - Y[:, r0:r1, :-1]
        m, n = _nanmedian0(gx)
        Gx[r0:r1, :-1] = np.where(n >= min_valid, np.nan_to_num(m), 0)
        r1b = min(Fh - 1, r1)
        if r1b > r0:
            gy = Y[:, r0 + 1:r1b + 1, :] - Y[:, r0:r1b, :]
            m, n = _nanmedian0(gy)
            Gy[r0:r1b, :] = np.where(n >= min_valid, np.nan_to_num(m), 0)
    div = np.zeros((Fh, Fw), np.float64)
    div += Gx
    div[:, 1:] -= Gx[:, :-1]
    div += Gy
    div[1:, :] -= Gy[:-1, :]
    n_, m_ = Fh - 2, Fw - 2
    W = np.zeros((Fh, Fw), np.float64)
    if n_ >= 2 and m_ >= 2:
        f = sfft.dstn(div[1:-1, 1:-1], type=1)
        jj = np.arange(1, n_ + 1); kk = np.arange(1, m_ + 1)
        lam = (2 * np.cos(np.pi * jj / (n_ + 1)) - 2)[:, None] + (2 * np.cos(np.pi * kk / (m_ + 1)) - 2)[None, :]
        W[1:-1, 1:-1] = sfft.idstn(f / lam, type=1)
    c1 = ts.normalise(np.maximum(-W, 0).astype(np.float32))
    c1[c1 < 0.03] = 0
    return c1, W.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 2: regions, ink, alternating minimisation
# ---------------------------------------------------------------------------

def _two_means(X):
    mean = X.mean(0)
    a = int(np.argmax(np.linalg.norm(X - mean, axis=1)))
    b = int(np.argmax(np.linalg.norm(X - X[a], axis=1)))
    c = [X[a].copy(), X[b].copy()]
    lab = np.zeros(len(X), bool)
    for _ in range(15):
        lab = np.linalg.norm(X - c[1], axis=1) < np.linalg.norm(X - c[0], axis=1)
        if lab.all() or (~lab).all():
            break
        c = [_unit(X[~lab].mean(0)), _unit(X[lab].mean(0))]
    return c, lab


def _estimate_regions(I, B, V, c1):
    """labels (Fh, Fw) uint8 in 1..K, ink (K, 3) 0-1, dirs (K, 3), region pixel counts."""
    N, Fh, Fw = V.shape
    P = np.argwhere(c1 > 0.3)
    if len(P) > 150000:
        P = P[np.linspace(0, len(P) - 1, 150000).astype(int)]
    ys, xs = P[:, 0], P[:, 1]
    D = (B[:, ys, xs].astype(np.float32) - I[:, ys, xs].astype(np.float32)) / 255.0     # (N, M, 3)
    nrm = np.linalg.norm(D, axis=2)
    ok = V[:, ys, xs] & (nrm >= 0.02)
    U = np.where(ok[..., None], D / np.maximum(nrm, 1e-6)[..., None], np.nan)
    med = np.stack([_nanmedian0(U[..., c])[0] for c in range(3)], axis=1)               # (M, 3)
    has = ~np.isnan(med).any(1)
    med = med[has]
    ys, xs = ys[has], xs[has]
    if len(med) < 10:
        med = np.tile(_unit(np.ones(3)), (10, 1)); ys = xs = np.zeros(10, int)
    med = med / np.maximum(np.linalg.norm(med, axis=1, keepdims=True), 1e-9)
    gdir = _unit(med.mean(0))
    K, dirs, lab = 1, [gdir], np.zeros(len(med), int)
    c, l2 = _two_means(med)
    if l2.any() and (~l2).any():
        ang = np.degrees(np.arccos(np.clip(float(c[0] @ c[1]), -1, 1)))
        small = min(l2.sum(), (~l2).sum()) / len(med)
        if ang > 10.0 and small >= 0.03:
            order = [0, 1] if (~l2).sum() >= l2.sum() else [1, 0]      # region 1 = larger cluster
            K, dirs = 2, [c[o] for o in order]
            lab = np.where(l2, 1, 0) if order == [0, 1] else np.where(l2, 0, 1)
    seed = np.zeros((Fh, Fw), np.uint8)
    seed[ys, xs] = lab + 1
    if K == 1:
        labels = np.ones((Fh, Fw), np.uint8)
    else:
        idx = ndi.distance_transform_edt(seed == 0, return_distances=False, return_indices=True)
        labels = seed[idx[0], idx[1]]
        m2 = cv2.blur((labels == 2).astype(np.float32), (7, 7)) > 0.5      # majority smoothing
        labels = np.where(m2, 2, 1).astype(np.uint8)
    # ink per region: paper colour minus t * darkening direction, luma fixed by the prior
    ink = []
    for r in range(1, K + 1):
        sel = (labels == r) & (c1 > 0.3)
        yy, xx = np.nonzero(sel)
        if yy.size == 0:
            yy, xx = np.nonzero(labels == r)
        if yy.size > 4000:
            sub = np.linspace(0, yy.size - 1, 4000).astype(int); yy, xx = yy[sub], xx[sub]
        pix = B[:, yy, xx].astype(np.float32) / 255.0
        v = V[:, yy, xx]
        paper = np.array([np.median(pix[..., c][v]) if v.any() else 1.0 for c in range(3)], np.float32)
        d = dirs[r - 1].astype(np.float32)
        ld = float(d @ LUMA)
        t = (float(paper @ LUMA) - stamp_fit.INK_LUM_PRIOR) / ld if ld > 0.05 else 0.5
        t = max(t, 0.05)
        ink.append(np.clip(paper - t * d, 0, 1))
    return labels, np.asarray(ink, np.float32), np.asarray(dirs, np.float32)


def _fit_strengths(I, B, V, u, kmap):
    N = V.shape[0]
    idx = np.flatnonzero(u.ravel() > 0.5)
    if idx.size < 30:
        idx = np.flatnonzero(u.ravel() > 0.2)
    if idx.size < 30:
        return np.ones(N, np.float32)
    uu = u.ravel()[idx]
    kk = kmap.reshape(-1, 3)[idx]
    o = np.zeros(N, np.float32)
    for i in range(N):
        Bi = B[i].reshape(-1, 3)[idx].astype(np.float32) / 255.0
        Ii = I[i].reshape(-1, 3)[idx].astype(np.float32) / 255.0
        D = Bi - Ii
        x = uu[:, None] * (Bi - kk)
        use = V[i].ravel()[idx] & (np.abs(D).max(1) <= MAX_DARK)
        if use.sum() < 20:
            o[i] = 0.0
            continue
        keep = use.copy()
        val = 0.0
        for _ in range(3):
            den = float((x[keep] * x[keep]).sum())
            if den < 1e-9:
                break
            val = float((x[keep] * D[keep]).sum() / den)
            res = np.linalg.norm(D - val * x, axis=1)
            thr = np.percentile(res[use], 75)
            keep = use & (res <= thr)
        o[i] = max(val, 0.0)
    return o


def _fit_coverage(I, B, V, o, kmap, rounds=2):
    N, Fh, Fw = V.shape
    u = np.zeros((Fh, Fw), np.float32)
    R = max(4, int(4e6 // max(1, N * Fw * 3)))
    oo = o[:, None, None, None]
    for r0 in range(0, Fh, R):
        r1 = min(Fh, r0 + R)
        Bc = B[:, r0:r1].astype(np.float32) / 255.0
        D = Bc - I[:, r0:r1].astype(np.float32) / 255.0
        BK = oo * (Bc - kmap[None, r0:r1])
        w = (V[:, r0:r1] & (np.abs(D).max(-1) <= MAX_DARK)).astype(np.float32)[..., None]
        wt = w
        for rnd in range(rounds + 1):
            den = (wt * BK * BK).sum((0, 3))
            num = (wt * BK * D).sum((0, 3))
            uc = np.where(den > 1e-8, num / np.maximum(den, 1e-8), 0).astype(np.float32)
            if rnd == rounds:
                break
            res = D - uc[None, ..., None] * BK
            a = np.where(w > 0, np.abs(res), np.nan)
            a = np.moveaxis(a, -1, 1).reshape(N * 3, r1 - r0, Fw)
            mad, _ = _nanmedian0(a)
            sig = np.maximum(1.4826 * np.nan_to_num(mad), 0.01)
            t = np.clip(res / (4.685 * sig[None, ..., None]), -1, 1)
            wt = w * (1 - t * t) ** 2
        u[r0:r1] = uc
    return u


def _refine(I, B, V, c1, labels, ink, iters=4):
    N = V.shape[0]
    kmap = ink[labels.astype(int) - 1].astype(np.float32)
    ref = c1 > 0.3
    if ref.sum() < 10:
        ref = c1 > 0.1
    min_valid = max(5, 0.3 * N)
    u = c1.copy()
    for _ in range(iters):
        o = _fit_strengths(I, B, V, u, kmap)
        u = _fit_coverage(I, B, V, o, kmap)
        peak = float(np.percentile(u[ref], 99.5)) if ref.any() else float(u.max())
        u = np.clip(u / max(peak, 1e-3), 0, 1)
    # Noise floor: with a few dozen pages every pixel keeps a little positive
    # noise (JPEG texture / text rims divided by the mark's faint contrast).
    # Estimate it where there is no mark and subtract it, so the support is
    # the mark and not the whole frame.
    core = cv2.dilate((c1 > 0.1).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    bgm = (~core) & (V.sum(0) >= min_valid)
    tau = 0.03
    if bgm.sum() > 500:
        tau = float(np.clip(np.percentile(u[bgm], 97), 0.03, 0.4))
    u = np.clip((u - tau) / (1.0 - tau), 0, 1).astype(np.float32)
    o = _fit_strengths(I, B, V, u, kmap)
    u[u < 0.03] = 0
    u[V.sum(0) < min_valid] = 0
    u = _prune(u)
    return u.astype(np.float32), o


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

def make_preview(alpha, labels, ink_rgb, max_side=700):
    """Left: coverage as grey on white. Right: the mark (region ink colours) on a checkerboard."""
    H, W = alpha.shape
    f = min(3.0, max_side / max(H, W))
    a = cv2.resize(alpha, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    a = np.clip(a, 0, 1)
    h, w = a.shape
    left = np.repeat((255 * (1 - a))[..., None], 3, 2)
    rgb = np.zeros((H, W, 3), np.float32)
    for i, k in enumerate(ink_rgb, 1):
        rgb[labels == i] = k
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[0:h, 0:w]
    cb = np.where(((yy // 12 + xx // 12) % 2) == 0, 235.0, 175.0)[..., None]
    right = rgb * a[..., None] + cb * (1 - a[..., None])
    gap = np.full((h, 8, 3), 128.0)
    return np.clip(np.hstack([left, gap, right]), 0, 255).astype(np.uint8)


def _overlay(path, tpl, pose, out_path):
    img = read_rgb(path)
    if img is None:
        return False
    H, W = img.shape[:2]
    tw, th = ts.template_size(tpl, pose["scale"])
    m = int(0.2 * max(tw, th))
    x0, y0 = max(0, int(pose["x"]) - m), max(0, int(pose["y"]) - m)
    x1, y1 = min(W, int(pose["x"]) + tw + m), min(H, int(pose["y"]) + th + m)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    cov = ts.render_template(tpl, pose["scale"], pose["x"], pose["y"], (x0, y0, x1 - x0, y1 - y0))
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    cnts, _ = cv2.findContours((cov >= 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnts, -1, (255, 0, 0), max(1, int(round(0.004 * max(crop.shape[:2])))))
    f = min(1.0, 1100.0 / max(crop.shape[:2]))
    if f < 1:
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    Image.fromarray(crop).save(out_path)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_template(pages_dir, seed_page, seed_box=None, seed_mask=None, name=None, library_dir=None,
                   max_pages=60, outer_iters=2, min_reg_score=0.30, frame_max_width=1000,
                   overwrite=False, progress=None, workers=None):
    """Builds a template from the pages in ``pages_dir`` and saves it into the
    library. ``seed_box`` = (x, y, w, h) in seed-page pixels, or ``seed_mask`` =
    boolean scribble mask of the seed page (its bounding box is used).

    Returns {"ok", "template_dir", "report", "preview", "message", ...};
    ``report`` has the per-page table (``rows``), summary and warnings;
    ``preview`` is an RGB array; ``overlays`` lists overlay image paths."""
    t_start = time.time()
    prog = (lambda f, msg="": progress(f, desc=msg)) if progress else (lambda f, msg="": None)
    params = dict(max_pages=int(max_pages), outer_iters=int(outer_iters), min_reg_score=float(min_reg_score),
                  frame_max_width=int(frame_max_width))

    def fail(msg):
        return dict(ok=False, template_dir=None, report=None, preview=None, overlays=[],
                    message=f"**Build failed:** {msg}")

    if not tl.is_safe_name(name):
        return fail("template name must be 1-64 characters: letters, digits, `_`, `-`, `.`")
    root = tl.library_root(library_dir)
    if os.path.exists(os.path.join(root, name, "template.png")) and not overwrite:
        return fail(f"template `{name}` already exists in `{root}` (tick overwrite to replace it)")
    pages_dir_r = tl.resolve_path(pages_dir)
    files = list_pages(pages_dir_r)
    if not files:
        return fail(f"no images found in `{pages_dir}`")
    seed_p = tl.resolve_path(seed_page)
    if seed_p and not os.path.isfile(seed_p):
        cand = os.path.join(pages_dir_r, str(seed_page))
        seed_p = cand if os.path.isfile(cand) else seed_p
    if not seed_p or not os.path.isfile(seed_p):
        return fail(f"seed page `{seed_page}` not found")
    seed_img = read_rgb(seed_p)
    if seed_img is None:
        return fail("seed page could not be read")
    SH, SW = seed_img.shape[:2]
    if seed_mask is not None:
        m = np.asarray(seed_mask) > 0
        if m.shape != (SH, SW):
            m = cv2.resize(m.astype(np.uint8), (SW, SH), interpolation=cv2.INTER_NEAREST) > 0
        bb = _bbox(m)
        if bb is None:
            return fail("the scribble is empty")
        seed_box = (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
    if seed_box is None or len(seed_box) != 4 or seed_box[2] < 20 or seed_box[3] < 10:
        return fail("no usable seed box (mark the watermark on the seed page)")
    x, y, w, h = [float(v) for v in seed_box]
    px_, py_ = 0.05 * w, 0.05 * h
    x0, y0 = max(0, int(np.floor(x - px_))), max(0, int(np.floor(y - py_)))
    x1, y1 = min(SW, int(np.ceil(x + w + px_))), min(SH, int(np.ceil(y + h + py_)))
    box = (x0, y0, x1 - x0, y1 - y0)

    order = [seed_p] + [f for f in files if os.path.normcase(f) != os.path.normcase(seed_p)]
    order = order[:max(MIN_PAGES, int(max_pages))]
    if len(order) < MIN_PAGES:
        return fail(f"need at least {MIN_PAGES} pages, found {len(order)}")
    workers = workers or max(1, min(4, os.cpu_count() or 2))

    prog(0.02, "seed template")
    tpl = _seed_template(seed_img, box)
    if (tpl > 0.3).sum() < 30:
        return fail("no faint darkening found inside the box (is the mark darker than the page, and inside the box?)")

    warnings = []
    rows_final = None
    N_read = 0
    prev_rows = None
    final = None
    iters = max(1, int(outer_iters))
    for it in range(iters):
        base = 0.05 + 0.9 * it / iters
        span = 0.9 / iters
        # ---- 5.3 register every page ------------------------------------
        prog(base, f"registering {len(order)} pages (round {it + 1}/{iters})")
        Th, Tw = tpl.shape
        if it == 0:
            s_nom = lambda Wp, _w=SW: (Wp / float(_w))
            lo, hi, ns = 0.4, 2.5, 30
        else:
            rel = float(np.median([r["scale"] * Tw / r["size"][0] for r in rows_final]))
            s_nom = lambda Wp, _r=rel, _t=Tw: _r * Wp / float(_t)
            lo, hi, ns = 0.6, 1.7, 20
        rows = _register_all(order, tpl, s_nom, lo, hi, ns, min_reg_score, workers)
        acc = [r for r in rows if r["accepted"]]
        N_read = sum(1 for r in rows if r["reason"] != "unreadable file")
        if N_read < MIN_PAGES:
            return fail(f"only {N_read} readable pages")
        if len(acc) < MIN_PAGES:
            return fail(f"only {len(acc)} of {N_read} pages matched the seed (need {MIN_PAGES}). Draw a tighter box "
                        f"around the mark, pick a seed page where it is clearly visible, or lower the registration score.")
        # early stop when nothing moved (rows_final is already in this template's coordinates)
        if it > 0 and rows_final is not None:
            prev = {r["file"]: r for r in rows_final}
            if sorted(prev) == sorted(r["file"] for r in acc):
                move = max(np.hypot((a["x"] + a["scale"] * Tw / 2) - (prev[a["file"]]["x"] + prev[a["file"]]["scale"] * Tw / 2),
                                    (a["y"] + a["scale"] * Th / 2) - (prev[a["file"]]["y"] + prev[a["file"]]["scale"] * Th / 2))
                           for a in acc)
                if move < 0.5:
                    break
        prev_rows = rows
        # ---- 5.4 frame resolution and warp ------------------------------
        prog(base + 0.35 * span, "warping accepted pages into the frame")
        geo = _frame_geometry(tpl.shape, acc, 0.0 if it == 0 else 0.2, frame_max_width)
        tsw = ts.stroke_width(tpl)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            warped = list(ex.map(lambda r: _warp_page(r, geo, tsw), acc))
        keep = [(r, wp) for r, wp in zip(acc, warped) if wp is not None]
        acc = [r for r, _ in keep]
        if len(acc) < MIN_PAGES:
            return fail("too few pages could be warped into the frame")
        I = np.stack([wp[0] for _, wp in keep]); Bf = np.stack([wp[1] for _, wp in keep]); V = np.stack([wp[2] for _, wp in keep])
        del warped, keep
        N = len(acc)
        # ---- 5.5 Dekel stage 1 -------------------------------------------
        prog(base + 0.55 * span, "median gradients + Poisson integration")
        c1, _ = _dekel_initial(I, V, max(5, 0.3 * N))
        if (c1 > 0.3).sum() < 30:
            return fail("the pages' common gradient is empty: the registration did not find a shared mark")
        # ---- 5.6 regions, ink, alternating minimisation ------------------
        prog(base + 0.7 * span, "refining coverage")
        labels, ink, _ = _estimate_regions(I, Bf, V, c1)
        u, o = _refine(I, Bf, V, c1, labels, ink)
        if (u > 0.1).sum() < 30:
            return fail("refinement produced an empty template")
        edge = np.zeros_like(u, bool); edge[0, :] = edge[-1, :] = True; edge[:, 0] = edge[:, -1] = True
        touches = bool((u[edge] > 0.03).any())
        # ---- crop to support (+4 px) --------------------------------------
        bb = _bbox(u > 0.03, margin=4, shape=u.shape)
        cx0, cy0, cx1, cy1 = bb
        u_c = np.ascontiguousarray(u[cy0:cy1, cx0:cx1])
        lab_c = np.ascontiguousarray(labels[cy0:cy1, cx0:cx1])
        # poses relative to the cropped template
        G = geo["G"]
        new_rows = []
        for r, oi in zip(acc, o):
            s2 = r["scale"] / G
            ox = r["x"] - r["scale"] * geo["px"]; oy = r["y"] - r["scale"] * geo["py"]
            new_rows.append(dict(file=r["file"], path=r["path"], scale=s2, x=ox + s2 * cx0, y=oy + s2 * cy0,
                                 score=r["score"], strength=float(oi), size=r["size"]))
        final = dict(u=u_c, labels=lab_c, ink=ink, geo=geo, touches=touches, n=N, rows=new_rows,
                     frame=(cx0, cy0, cx1, cy1))
        rows_final = new_rows
        tpl = u_c
        prog(base + span, f"round {it + 1} done")

    # ---- 5.8 finish ---------------------------------------------------
    prog(0.96, "saving")
    u, labels = final["u"], final["labels"]
    ink = final["ink"]
    Th, Tw = u.shape
    # drop empty regions, relabel in order of size
    ids = [i for i in range(1, len(ink) + 1) if (u[(labels == i)] > 0.03).sum() > 0]
    counts = [int((u[(labels == i)] > 0.03).sum()) for i in ids]
    ids = [i for _, i in sorted(zip([-c for c in counts], ids))]
    new_lab = np.zeros_like(labels)
    for j, i in enumerate(ids, 1):
        new_lab[(labels == i) & (u > 0.03)] = j
    ink_u8 = [tuple(int(v) for v in np.round(ink[i - 1] * 255)) for i in ids] or [(100, 100, 100)]
    pix = [int((new_lab == j).sum()) for j in range(1, len(ids) + 1)]
    # a page whose fitted strength is ~0 does not actually show the mark at the fitted pose
    # (a fit on clutter, or a lighter-than-page mark): report it as rejected
    med_o = float(np.median([r["strength"] for r in rows_final]))
    weak = {r["file"]: r for r in rows_final if r["strength"] < 0.3 * med_o}
    rows_final = [r for r in rows_final if r["file"] not in weak]
    accepted_rows = {r["file"]: r for r in rows_final}
    all_rows = []
    for r in prev_rows if prev_rows is not None else []:
        a = accepted_rows.get(r["file"])
        if a is not None:
            all_rows.append(dict(file=r["file"], accepted=True, score=round(a["score"], 4), scale=round(a["scale"], 5),
                                 x=round(a["x"], 2), y=round(a["y"], 2), strength=round(a["strength"], 4), reason=""))
        else:
            w_ = weak.get(r["file"])
            reason = (f"no measurable mark at the fitted pose (strength {w_['strength']:.3f} vs median {med_o:.3f})"
                      if w_ else (r["reason"] or "could not be warped into the frame"))
            all_rows.append(dict(file=r["file"], accepted=False, score=round(r["score"], 4), scale=None, x=None,
                                 y=None, strength=None, reason=reason))
    # prev_rows may be the last registration (poses relative to the previous template): accepted rows above
    # use the converted poses, rejected rows keep their reason.
    acc_rows = list(rows_final)
    rel = [r["scale"] * Tw / r["size"][0] for r in acc_rows]
    stren = [r["strength"] for r in acc_rows]
    if final["touches"]:
        warnings.append("the mark may extend past your box; draw a bigger box")
    if len(acc_rows) < 20:
        warnings.append(f"only {len(acc_rows)} pages were used; 20+ gives a cleaner estimate")
    nrej = len(all_rows) - len(acc_rows)
    sw = ts.stroke_width(u)
    elapsed = time.time() - t_start
    meta = dict(name=name, builder_version=BUILDER_VERSION, source_folder=pages_dir_r,
                seed_page=os.path.basename(seed_p), seed_box=[int(v) for v in box],
                n_pages_used=len(acc_rows), n_pages_rejected=int(nrej), template_size=[int(Tw), int(Th)],
                rel_width=dict(median=float(np.median(rel)), min=float(np.min(rel)), max=float(np.max(rel))),
                strength=dict(median=float(np.median(stren)), min=float(np.min(stren)), max=float(np.max(stren))),
                regions=[dict(id=j, ink_rgb=list(ink_u8[j - 1]), pixels=pix[j - 1]) for j in range(1, len(ids) + 1)],
                stroke_width_px=float(sw), build_seconds=round(elapsed, 1), warnings=warnings,
                params=dict(params, workers=workers, frame_scale_G=float(final["geo"]["G"])))
    try:
        folder = tl.save_template(name, u, new_lab if len(ids) > 1 else None, ink_u8, meta,
                                  library_dir=library_dir, overwrite=overwrite)
    except Exception as e:
        return fail(str(e))
    preview = make_preview(u, np.maximum(new_lab, 1) if len(ids) <= 1 else new_lab, ink_u8)
    Image.fromarray(preview).save(os.path.join(folder, "preview.png"))
    overlays = []
    if acc_rows:
        pick = np.unique(np.linspace(0, len(acc_rows) - 1, min(6, len(acc_rows))).round().astype(int))
        for n, i in enumerate(pick):
            r = acc_rows[i]
            op = os.path.join(folder, f"overlay_{n:02d}.png")
            if _overlay(r["path"], u, dict(scale=r["scale"], x=r["x"], y=r["y"]), op):
                overlays.append(op)
    report = dict(rows=all_rows, summary=meta, warnings=warnings, elapsed=elapsed)
    with open(os.path.join(folder, "build_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    prog(1.0, "done")
    message = _summary_md(name, folder, meta, all_rows, elapsed, warnings)
    return dict(ok=True, template_dir=folder, report=report, preview=preview, overlays=overlays, message=message)


def _summary_md(name, folder, meta, rows, elapsed, warnings):
    rej = [r for r in rows if not r["accepted"]]
    lines = [f"### Template `{name}` built in {elapsed:.0f} s",
             f"- saved to `{folder}`",
             f"- pages used **{meta['n_pages_used']}**, rejected **{meta['n_pages_rejected']}**",
             f"- template size {meta['template_size'][0]}x{meta['template_size'][1]} px, "
             f"instance width / page width: median {meta['rel_width']['median']:.3f} "
             f"(min {meta['rel_width']['min']:.3f}, max {meta['rel_width']['max']:.3f})",
             f"- strength: median {meta['strength']['median']:.3f} "
             f"(min {meta['strength']['min']:.3f}, max {meta['strength']['max']:.3f})",
             f"- regions: " + ", ".join(f"#{g['id']} ink rgb{tuple(g['ink_rgb'])} ({g['pixels']} px)" for g in meta["regions"])]
    for wmsg in warnings:
        lines.append(f"- WARNING: {wmsg}")
    if rej:
        lines.append("\n**Rejected pages**\n")
        for r in rej[:30]:
            lines.append(f"- `{r['file']}`: {r['reason']}")
        if len(rej) > 30:
            lines.append(f"- ... and {len(rej) - 30} more (see build_report.json)")
    return "\n".join(lines)
