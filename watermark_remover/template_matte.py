"""Per-pixel matting of a watermark from registered pages (Template Builder v2).

Everything here works in the template FRAME: ``N`` registered pages warped
into one common frame, with their own mark-aware paper background
(``template_signal.paper_background_masked``) and a validity mask.

Image-formation model, per page ``i`` and pixel ``p``::

    I_i = (1 - m_i a(p)) B_i + m_i a(p) k(p)        (B_i = the page's own paper)

so the darkening ``D_i = B_i - I_i = m_i a(p) (B_i - k(p))``. At reference paper
white that is the *matted darkening*::

    W(p) = a(p) (1 - k(p))                           (3 channels)

and ``D_i = W - m_i a (1 - B_i)``: dividing by ``B_i`` (per channel) is a first
approximation (right for white paper, ``a k (1/B - 1)`` too small otherwise); once
a first opacity is known the exact per-page sample ``D_i + a (1 - B_i)`` is used
(``estimate(a_prior=...)``). ``W`` is what the pages have in common at ``p`` and is
estimated robustly:
pages that carry text (or anything much darker than the mark) at ``p`` are
excluded, then a per-channel median and two rounds of Tukey-biweight IRLS give
``W``. On flat paper only ``W`` is observable -- a faint dark ink and a strong
light ink darken paper identically -- so one template-wide ink luminance
``L_ink`` fixes the split::

    a(p) = lum(W) / (1 - L_ink)   (>= max_c W_c so the ink stays >= 0),   k = 1 - W / a

which reproduces the page exactly on paper for ANY ``L_ink``; ``L_ink`` only
matters for page content under the mark. It is calibrated (Dekel-style) from
text strokes that cross the mark when enough of them are seen (``calibrate_ink``),
else it is ``stamp_fit.INK_LUM_PRIOR``.

The support (which pixels belong to the mark) is decided by statistical
significance across pages, not by a noise floor (``support_mask``): thin, faint
letters that are the same on every page survive.

Deterministic for a fixed input.
"""

import cv2
import numpy as np

from . import stamp_fit
from . import template_signal as ts

LUMA = ts.LUMA
TEXT_ABS = 0.12           # a page is 'text-under-mark' at p when its lum darkening exceeds
TEXT_REL = 2.5            # max(TEXT_ABS, TEXT_REL x the pixel's running median)
MIN_LUM_W = 0.012         # support: lum(W) at least this
MIN_Z = 4.0               # support: significance z = median / (1.4826 MAD / sqrt(n_valid))
MIN_COMPONENT = 6         # support: components smaller than this are dropped
MAX_ALPHA = stamp_fit.MAX_ALPHA


def nanmedian0(a):
    """Median over axis 0 ignoring NaN (exact, via a sort; NaN sorts last).
    Returns (median float32, count)."""
    n = np.sum(~np.isnan(a), axis=0)
    srt = np.sort(a, axis=0)
    last = a.shape[0] - 1
    lo = np.clip((n - 1) // 2, 0, last)
    hi = np.clip(n // 2, 0, last)
    m = 0.5 * (np.take_along_axis(srt, lo[None], 0)[0] + np.take_along_axis(srt, hi[None], 0)[0])
    m = np.where(n > 0, m, np.nan)
    return m.astype(np.float32), n


# ---------------------------------------------------------------------------
# Matted darkening W
# ---------------------------------------------------------------------------

def estimate(I, B, V, irls_rounds=2, sigma_floor=0.01, want_crossings=True, a_prior=None):
    """Robust per-pixel matted darkening across pages.

    Per-page sample of ``W``: ``(B_i - I_i) / B_i`` per channel, or - when an
    opacity map ``a_prior`` (Fh, Fw) from a first pass is given - the exact
    ``(B_i - I_i) + a_prior (1 - B_i)``.

    ``I`` (N, Fh, Fw, 3) uint8, ``B`` (N, Fh, Fw, 3) float (0-1), ``V``
    (N, Fh, Fw) bool. Returns a dict:
      ``W`` (Fh, Fw, 3) float32 -- matted darkening at paper white (can be slightly
      negative where there is no mark);
      ``n_valid`` (Fh, Fw) -- pages that contributed at the pixel;
      ``z`` (Fh, Fw) -- significance of the luminance darkening;
      ``text`` (N, Fh, Fw) bool -- page i shows something much darker than the
      mark at the pixel (text under the mark);
      ``ip_lum`` / ``bp_lum`` (Fh, Fw) -- median over contributing pages of the
      page luminance / paper luminance (for ``calibrate_ink``)."""
    N, Fh, Fw = V.shape
    W = np.zeros((Fh, Fw, 3), np.float32)
    nvalid = np.zeros((Fh, Fw), np.int16)
    zmap = np.zeros((Fh, Fw), np.float32)
    text = np.zeros((N, Fh, Fw), bool)
    ip = np.zeros((Fh, Fw), np.float32)
    bp = np.zeros((Fh, Fw), np.float32)
    R = max(4, int(2.4e6 // max(1, N * Fw * 3)))
    for r0 in range(0, Fh, R):
        r1 = min(Fh, r0 + R)
        Bc = B[:, r0:r1].astype(np.float32)
        Ic = I[:, r0:r1].astype(np.float32) * (1.0 / 255.0)
        v = V[:, r0:r1]
        if a_prior is None:
            D = (Bc - Ic) / np.maximum(Bc, 0.05)
        else:
            D = (Bc - Ic) + a_prior[None, r0:r1, :, None] * (1.0 - Bc)
        lumD = D @ LUMA
        med1, _ = nanmedian0(np.where(v, lumD, np.nan))
        thr = np.maximum(TEXT_ABS, TEXT_REL * np.nan_to_num(med1))
        txt = v & (lumD > thr[None])
        use = v & ~txt & (lumD > -0.15)
        text[:, r0:r1] = txt
        nv = use.sum(0)
        Dm = np.where(use[..., None], D, np.nan)
        W0 = np.nan_to_num(np.stack([nanmedian0(Dm[..., c])[0] for c in range(3)], -1))
        for _ in range(irls_rounds):
            res = Dm - W0[None]
            mad = np.nan_to_num(np.stack([nanmedian0(np.abs(res[..., c]))[0] for c in range(3)], -1))
            sig = np.maximum(1.4826 * mad, sigma_floor)
            t = res / (4.685 * sig[None])
            w = np.where(np.isnan(t), 0.0, np.clip(1.0 - t * t, 0.0, None) ** 2)
            sw = w.sum(0)
            W0 = np.where(sw > 1e-6, (w * np.nan_to_num(Dm)).sum(0) / np.maximum(sw, 1e-6), W0)
        W[r0:r1] = W0
        nvalid[r0:r1] = nv
        lu = np.where(use, lumD, np.nan)
        medl, _ = nanmedian0(lu)
        madl, _ = nanmedian0(np.abs(lu - medl[None]))
        zmap[r0:r1] = np.nan_to_num(medl) / (1.4826 * np.maximum(np.nan_to_num(madl), 1e-3)
                                             / np.sqrt(np.maximum(nv, 1)))
        if want_crossings:
            ip[r0:r1] = np.nan_to_num(nanmedian0(np.where(use, Ic @ LUMA, np.nan))[0])
            bp[r0:r1] = np.nan_to_num(nanmedian0(np.where(use, Bc @ LUMA, np.nan))[0])
    return dict(W=W, n_valid=nvalid, z=zmap, text=text, ip_lum=ip, bp_lum=bp)


def support_mask(est, n_pages, min_comp=MIN_COMPONENT):
    """Pixels that belong to the mark: ``n_valid >= max(5, 0.3 N)``,
    ``lum(W) >= 0.012`` and significance ``z >= 4``; components smaller than
    ``min_comp`` px removed; 1-px gaps closed (3x3). Returns (mask, counts) where
    ``counts`` lists how many pixels each rule removed, applied in this order."""
    lumW = est["W"] @ LUMA
    r1 = est["n_valid"] >= max(5.0, 0.3 * n_pages)
    r2 = lumW >= MIN_LUM_W
    r3 = est["z"] >= MIN_Z
    keep = r1 & r2 & r3
    n, lab, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < min_comp
    removed_small = int(stats[1:, cv2.CC_STAT_AREA][small[1:]].sum()) if n > 1 else 0
    keep2 = keep & ~small[lab]
    closed = cv2.morphologyEx(keep2.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    counts = dict(frame_pixels=int(keep.size),
                  removed_by_valid_pages=int((~r1).sum()),
                  removed_by_min_darkening=int((r1 & ~r2).sum()),
                  removed_by_significance=int((r1 & r2 & ~r3).sum()),
                  removed_small_components=removed_small,
                  added_by_gap_closing=int((closed & ~keep2).sum()),
                  kept=int(closed.sum()))
    return closed, counts


# ---------------------------------------------------------------------------
# Opacity + ink from W
# ---------------------------------------------------------------------------

def opacity_and_ink(W, L_ink, mask=None):
    """``a = lum(W) / (1 - L_ink)`` (and at least ``max_c W_c`` so the ink stays
    non-negative), ``k = 1 - W / a``. Returns (a float32 (Fh, Fw) clipped to
    [0, MAX_ALPHA], k float32 (Fh, Fw, 3) in 0-1). Pixels outside ``mask`` get
    a = 0 and a neutral ink of luminance ``L_ink``."""
    lumW = W @ LUMA
    a = np.maximum(lumW / max(1.0 - float(L_ink), 0.05), W.max(axis=2))
    a = np.clip(a, 0.0, MAX_ALPHA).astype(np.float32)
    if mask is not None:
        a = np.where(mask, a, 0.0).astype(np.float32)
    ok = a > 0.01
    k = np.where(ok[..., None], 1.0 - W / np.maximum(a, 1e-6)[..., None], float(L_ink))
    return a, np.clip(k, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Opacity calibration from text crossings (Dekel-style)
# ---------------------------------------------------------------------------

def calibrate_ink(I, V, est, support, W, prior, min_pixels=300, min_cross=2):
    """Estimate the template-wide ink luminance ``L_ink`` from text strokes that
    cross the mark.

    At a pixel seen both with text under the mark (``I_text``) and on bare paper
    (``I_paper``): ``I_paper - I_text = (1 - a)(B - T)`` with ``T`` the page's text
    ink luminance (median of the strongly dark pixels in a ring around the mark),
    so ``a = 1 - (I_paper - I_text) / (B - T)``. Pooling pixels in the mark's core
    that have at least ``min_cross`` crossing pages gives a robust opacity ``a_hat``;
    since ``lum(W) = a (1 - L_ink)``, ``1 - L_ink = median(lum W) / median(a_hat)``.
    Used only if at least ``min_pixels`` pixels qualify and ``median(a_hat)`` is in
    [0.05, 0.9] and the resulting ``L_ink`` in [0, 0.9]; otherwise ``prior``.
    Returns dict(L_ink, source, n_pixels, a_hat, reason)."""
    N = I.shape[0]
    lumW = W @ LUMA
    out = dict(L_ink=float(prior), source="prior", n_pixels=0, a_hat=None, reason="")
    if not support.any():
        out["reason"] = "empty support"
        return out
    a0 = lumW / max(1.0 - prior, 0.05)
    peak0 = float(np.percentile(a0[support], 99.5))
    core = support & (a0 >= 0.5 * peak0)
    idx = np.flatnonzero(core.ravel())
    if idx.size < min_pixels:
        out["reason"] = f"only {idx.size} core pixels"
        return out
    se = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sup_u8 = support.astype(np.uint8)
    ring = (cv2.dilate(sup_u8, se(10)) > 0) & ~(cv2.dilate(sup_u8, se(2)) > 0)
    ip = est["ip_lum"].ravel()[idx]
    bp = est["bp_lum"].ravel()[idx]
    Aj = np.full((N, idx.size), np.nan, np.float32)
    n_text_pages = 0
    for j in range(N):
        lj = (I[j].astype(np.float32) @ LUMA) * (1.0 / 255.0)
        r_ok = ring & V[j] & (lj < 0.35)
        if int(r_ok.sum()) < 50:
            continue
        T = float(np.median(lj[r_ok]))
        Ij = lj.ravel()[idx]
        sel = est["text"][j].ravel()[idx] & V[j].ravel()[idx] & (Ij < 0.35) & ((bp - T) > 0.3)
        if not sel.any():
            continue
        n_text_pages += 1
        Aj[j, sel] = 1.0 - (ip[sel] - Ij[sel]) / (bp[sel] - T)
    cnt = np.sum(~np.isnan(Aj), axis=0)
    ahat, _ = nanmedian0(Aj)
    valid = (cnt >= min_cross) & np.isfinite(ahat) & (ahat > 0.0) & (ahat < 1.0)
    out["n_pixels"] = int(valid.sum())
    if valid.sum() < min_pixels:
        out["reason"] = f"only {int(valid.sum())} pixels with >= {min_cross} text crossings (need {min_pixels})"
        return out
    a_med = float(np.median(ahat[valid]))
    out["a_hat"] = a_med
    if not (0.05 <= a_med <= 0.9):
        out["reason"] = f"crossing opacity estimate {a_med:.3f} outside [0.05, 0.9]"
        return out
    ratio = float(np.median(lumW.ravel()[idx][valid])) / a_med
    L = 1.0 - ratio
    if not (0.0 <= L <= 0.9):
        out["reason"] = f"implied ink luminance {L:.2f} outside [0, 0.9]"
        return out
    out.update(L_ink=float(L), source="text_crossings", reason=f"{int(valid.sum())} px, {n_text_pages} pages")
    return out


# ---------------------------------------------------------------------------
# Per-page strength multiplier
# ---------------------------------------------------------------------------

def fit_multiplier(y, x, prior=0.2, rounds=3, min_pixels=300, clip=(0.3, 2.0)):
    """Robust scalar ``m`` with ``y ~ m x`` (rows = pixels, 3 channels), a soft
    prior ``m ~ 1`` of weight ``prior * sqrt(n)`` (in units where the data rows
    have unit rms, so the prior is a few percent of the data, not half of it) and
    ``rounds`` rounds of trimming at the 75th percentile of the residual norm.
    ``prior=0`` and ``clip=None`` give the raw least-squares value. Returns
    (m, n_pixels_used); with fewer than ``min_pixels`` pixels returns (1, n)."""
    n = int(len(x))
    if n < min_pixels:
        return 1.0, n
    s = float(np.sqrt((x * x).mean()))
    if s < 1e-6:
        return 1.0, n
    xt, yt = x / s, y / s
    keep = np.ones(n, bool)
    m = 1.0
    for _ in range(int(rounds) + 1):
        k = int(keep.sum())
        w2 = (float(prior) ** 2) * k
        m = float(((xt[keep] * yt[keep]).sum() + w2) / max((xt[keep] ** 2).sum() + w2, 1e-9))
        res = np.linalg.norm(yt - m * xt, axis=1)
        thr = np.percentile(res[keep], 75)
        keep = res <= thr
    used = int(keep.sum())
    if clip is not None:
        m = float(np.clip(m, clip[0], clip[1]))
    return m, used


# ---------------------------------------------------------------------------
# Ink regions (for the per-region Stamp Fit removal option only)
# ---------------------------------------------------------------------------

def _kmeans(X, w, K, iters=25):
    """Deterministic weighted k-means: farthest-point init (first centre = the
    point farthest from the weighted mean), Lloyd iterations, argmin ties go to
    the lower index. Returns (centres (K, d), labels (n,), sse)."""
    mean = (X * w[:, None]).sum(0) / max(float(w.sum()), 1e-9)
    c = [X[int(np.argmax(((X - mean) ** 2).sum(1)))].copy()]
    dmin = ((X - c[0]) ** 2).sum(1)
    for _ in range(1, K):
        c.append(X[int(np.argmax(dmin))].copy())
        dmin = np.minimum(dmin, ((X - c[-1]) ** 2).sum(1))
    C = np.asarray(c, np.float64)
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(2)
        new = np.argmin(d, axis=1)
        if _ > 0 and (new == lab).all():
            break
        lab = new
        for j in range(K):
            sel = lab == j
            if sel.any():
                C[j] = (X[sel] * w[sel, None]).sum(0) / max(float(w[sel].sum()), 1e-9)
    d = ((X[:, None, :] - C[None]) ** 2).sum(2)
    lab = np.argmin(d, axis=1)
    sse = float((w * d[np.arange(len(X)), lab]).sum())
    return C, lab, sse


def cluster_regions(a, ink, A, support, max_k=5, min_px=300, gain=0.25, a_min=0.05):
    """Cluster the mark's pixels (``a > a_min``) by ink colour into 1..``max_k``
    regions: a cluster is added while it lowers the within-cluster spread by more
    than ``gain`` AND every cluster keeps at least ``min_px`` pixels. Returns
    (labels uint8 (Fh, Fw): 0 = outside the support, 1..K = region by decreasing
    size; ink_mean (K, 3) float 0-1, opacity-weighted)."""
    Fh, Fw = a.shape
    sel = support & (a > a_min)
    labels = np.zeros((Fh, Fw), np.uint8)
    if sel.sum() < 10:
        labels[support] = 1
        return labels, np.full((1, 3), 0.4, np.float32)
    aw = cv2.GaussianBlur(a * sel, (0, 0), 1.2)
    ks = np.stack([cv2.GaussianBlur(ink[..., c] * a * sel, (0, 0), 1.2) for c in range(3)], -1) / np.maximum(aw, 1e-6)[..., None]
    ys, xs = np.nonzero(sel)
    if len(ys) > 60000:
        sub = np.linspace(0, len(ys) - 1, 60000).astype(int)
    else:
        sub = np.arange(len(ys))
    X = ks[ys[sub], xs[sub]].astype(np.float64)
    w = (A[ys[sub], xs[sub]] + 0.05).astype(np.float64)
    best_K, best = 1, _kmeans(X, w, 1)
    for K in range(2, max_k + 1):
        C, lab, sse = _kmeans(X, w, K)
        sizes = np.bincount(lab, minlength=K) * (len(ys) / len(sub))
        if sse < (1.0 - gain) * best[2] and sizes.min() >= min_px:
            best_K, best = K, (C, lab, sse)
        else:
            break
    C = best[0]
    yy, xx = np.nonzero(support)
    f = ks[yy, xx].astype(np.float64)
    f = np.where((aw[yy, xx] > 1e-6)[:, None], f, C.mean(0)[None])
    lab_all = np.argmin(((f[:, None, :] - C[None]) ** 2).sum(2), axis=1)
    sizes = np.bincount(lab_all, minlength=best_K)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty(best_K, int)
    rank[order] = np.arange(best_K)
    labels[yy, xx] = (rank[lab_all] + 1).astype(np.uint8)
    ink_mean = np.zeros((best_K, 3), np.float32)
    for r in range(best_K):
        m = labels == (r + 1)
        wsum = float(A[m].sum())
        ink_mean[r] = (ink[m] * A[m][:, None]).sum(0) / wsum if wsum > 1e-9 else C[order[r]]
    return labels, ink_mean
