"""Page-adaptive removal ("Page-adaptive (v3)") for Method 5.

The per-pixel model (``template_remove``) keeps the template's opacity and ink
and fits ONE strength per page. Real pages differ from the template in ways one
scalar cannot absorb: the page's resampler may have sharpened the mark (a dark
inner overshoot at every letter edge, slightly different in x and y), and the
page's ink may be darker or bluer than the template's. Over paper a darker grey
and a lighter grey are indistinguishable; over a blue banner they are not. This
module fits a handful of parameters PER MARK, on the page itself, and then reads
the opacity back from the page wherever it can be read:

1. Window: the template's box at the pose + 10 px; everything below is local.
2. Local background ``B``: ``cv2.inpaint`` of the footprint ``F`` from its
   surroundings, plus a reliability mask (known pixels = outside ``F`` and not on
   a strong edge; low spread of the known pixels in a 15x15 box; within 7 px of a
   known pixel). White text on a banner is not 'paper' to a closing, but it is
   known pixels to inpainting.
3. Model, per spatial part (connected component of the footprint)::

       a = peak * (m * a_u + a_int * sum_k DW_k * Q_k(d))

   ``a_u`` the template coverage at the pose (no blur), ``d`` the signed distance
   to the template's own half-level contour, ``DW`` four direction weights (the
   edge normal points right / left / down / up; they sum to 1) and ``Q_k`` four
   piecewise-linear edge-profile curves pinned to 0 at both ends (an edge-only
   correction: a sharpened resampler gives a different overshoot on vertical and
   horizontal edges), plus a per-ink-region tint ``k = k_t + dk``.
4. Fit: 4 rounds of Tukey-weighted least squares on the reliable pixels
   (``scipy.optimize.lsq_linear`` with bounds, ridge on ``Q`` and on its second
   differences). ``lam_k`` pulls the tint hard towards the template ink: a weak
   pull is what failed on the blue banner.
5. Polish: where the page can be read (reliable, not text, background clearly
   differs from the ink, the pixel's own colour is consistent with the ink) the
   per-pixel opacity ``a_d = <y, B - k> / |B - k|^2`` replaces the parametric one.
6. Exact inverse ``(I - a k) / (1 - a)``, written only where ``a > 1e-4``.
7. Guard: the leftover-ghost score (the statistic of
   ``scripts/alpha_net/eval_stamp_rim.py``) of the result is compared with the
   per-pixel (v2) model's on the same pixels; a mark where v3 is worse by more
   than ``GUARD_MARGIN`` grey levels keeps the v2 result.

``fit_quality`` runs the cheap global-part-only version (one strength, no edge
profile, no tint, no polish, 2 rounds) and returns the fraction of the page's
darkening the template explains; Method 5 uses it to reject a located 'mark'
that the page does not actually show, and to decide between two templates that
describe the same mark.

Marks are removed one after another on the running output. Pixels outside every
mark's ``zone`` are byte-identical to the input. Deterministic.
"""

import time

import cv2
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear

from . import stamp_fit
from . import template_remove as trm
from . import template_signal as ts

LUMA = ts.LUMA

MARGIN = 10                 # window margin around the template box (px)
SS = 4                      # supersampling of the signed-distance map
KNOTS = np.arange(-3.0, 3.51, 0.5)     # edge-profile knots (px from the contour); the two ends are pinned to 0
STEP = 0.5
NQ = len(KNOTS) - 2         # free knots per direction
N_DIR = 4                   # edge normal pointing right, left, down, up
ROUNDS = 4                  # IRLS rounds of the full fit
MIN_PART_PX = 600           # weighted pixels a part needs for its own fit
MIN_RELIABLE = 300          # fewer reliable pixels: fall back to the v2 model
MAX_FIT_PIXELS = 60000      # per solve (seeded subsample)
LAM_RIDGE = 1e-3            # ridge on Q, per sqrt(sum w); second differences get 3x
LAM_K = 0.002               # pull of the ink tint towards the template ink
M_BOUNDS = (0.2, 3.0)
Q_BOUNDS = (-3.0, 3.0)
DK_CLIP = 0.5
MIN_BASE = 0.06             # |B - k| below this carries no information on the opacity
GRAD_TOL = 0.06             # |grad lum| / 4 above this is a strong edge (not a known pixel)
STD_TOL = 0.025             # max-channel spread of the known pixels in the box
RELIABLE_BOX = 15
MIN_KNOWN_IN_BOX = 8
MAX_KNOWN_DIST = 7.0
EXTEND_INK = True           # ink of the edge ring (coverage ~0) = neighbouring stroke ink, not black
GHOST_CUT = 25.0            # grey levels: larger residuals are page content, not the mark
GUARD_MARGIN = 0.3          # grey levels: v3 worse than v2 by more than this -> keep v2
D_LO, D_HI, D_STEP = -4.0, 4.0, 0.5


def _tukey(r, c):
    u = r / c
    return np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)


# ---------------------------------------------------------------------------
# Geometry: signed distance, direction weights, edge-profile basis
# ---------------------------------------------------------------------------

def render_ss(tpl, pose, x0, y0, w, h, ss=SS):
    """Unit coverage at ``ss`` x supersampling in the page window (x0, y0, w, h)."""
    a = tpl["alpha"]
    tw, th = ts.template_size(a, pose["scale"])
    interp = cv2.INTER_LINEAR if ss * tw > a.shape[1] else cv2.INTER_AREA
    t = cv2.resize(a, (tw * ss, th * ss), interpolation=interp)
    M = np.float32([[1, 0, (pose["x"] - x0) * ss], [0, 1, (pose["y"] - y0) * ss]])
    return cv2.warpAffine(t, M, (w * ss, h * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def signed_distance(a4, ss=SS):
    """Signed distance (page px, + inside) from each page pixel to the half-level
    contour of the coverage normalised by its LOCAL maximum (so a faint thin
    letter and a strong solid disc both get a contour), via a supersampled
    distance transform averaged back to 1x (same idea as
    ``stamp_fit._signed_distance``)."""
    loc = cv2.dilate(a4, np.ones((5 * ss + 1, 5 * ss + 1), np.uint8))
    n = a4 / np.maximum(loc, 1e-3)
    mask = ((n >= 0.5) & (a4 > 0.02)).astype(np.uint8)
    din = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    d4 = np.where(mask > 0, din - 0.5, 0.5 - dout).astype(np.float32)
    h, w = a4.shape[0] // ss, a4.shape[1] // ss
    return cv2.resize(d4, (w, h), interpolation=cv2.INTER_AREA) / ss


def dir_weights(d):
    """(h, w, 4) weights of the edge normal pointing into the stroke to the
    right, left, down, up: squared positive / negative parts of the unit
    gradient of ``GaussianBlur(d, 0.7)``. They sum to 1."""
    g = cv2.GaussianBlur(d, (0, 0), 0.7)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    n = np.sqrt(gx * gx + gy * gy) + 1e-6
    c, s = gx / n, gy / n
    return np.stack([np.maximum(c, 0) ** 2, np.maximum(-c, 0) ** 2, np.maximum(s, 0) ** 2, np.maximum(-s, 0) ** 2], -1)


def _basis(d, a_int, DW):
    """(N, N_DIR * NQ) design of the edge-profile curves: for direction k and free
    knot j, ``a_int * DW_k * hat_j(d)`` (piecewise-linear hat on KNOTS)."""
    dd = np.clip(d, KNOTS[0], KNOTS[-1])
    idx = np.clip(np.searchsorted(KNOTS, dd, side="right") - 1, 0, len(KNOTS) - 2)
    t = (dd - KNOTS[idx]) / STEP
    N = d.size
    out = np.zeros((N, N_DIR * NQ), np.float64)
    rows = np.arange(N)
    for k in range(N_DIR):
        wk = a_int * DW[:, k]
        for off, wt in ((0, 1.0 - t), (1, t)):
            j = idx + off
            ok = (j >= 1) & (j <= len(KNOTS) - 2)          # the pinned end knots carry no parameter
            out[rows[ok], k * NQ + j[ok] - 1] += (wk * wt)[ok]
    return out


def _q_alpha(Q, d, a_int, DW):
    """Edge correction ``a_int * sum_k DW_k * Q_k(d)`` on the whole window (the
    pinned ends make ``np.interp``'s clamping exactly 0 outside the knots)."""
    out = np.zeros(d.shape, np.float32)
    for k in range(N_DIR):
        q = np.concatenate([[0.0], Q[k * NQ:(k + 1) * NQ], [0.0]])
        out += DW[..., k] * np.interp(d, KNOTS, q).astype(np.float32)
    return a_int * out


def _alpha(m, Q, a_u, a_int, d, DW, peak):
    return peak * (m * a_u + _q_alpha(Q, d, a_int, DW))


def _reg_rows(P):
    """Ridge on Q (rows 0..P-2 of the parameter vector after m) and second
    differences within each direction block."""
    R = np.zeros((P - 1, P))
    R[:, 1:] = np.eye(P - 1)
    rows = []
    for blk in range((P - 1) // NQ):
        for j in range(NQ - 2):
            r = np.zeros(P)
            r[1 + blk * NQ + j:1 + blk * NQ + j + 3] = [1.0, -2.0, 1.0]
            rows.append(r)
    return R, np.array(rows)


def _fit_part(y, base, a_u, a_int, d, DW, w, peak, edge=True, rng=None):
    """Weighted least squares of ``y = a * base`` (3 channels stacked, rows
    weighted by sqrt(w)) for one part: returns (m, Q). ``edge=False``: only ``m``.

    The stacked system is solved through its sufficient statistics (``A^T A`` and
    ``A^T b``, Cholesky-reduced to an equivalent small system) so the bounded
    ``lsq_linear`` stays fast on 60k pixels; the objective is identical."""
    N = len(w)
    if N > MAX_FIT_PIXELS:
        rng = rng or np.random.default_rng(0)
        sel = np.sort(rng.choice(N, MAX_FIT_PIXELS, replace=False))
        y, base, a_u, a_int, d, w = y[sel], base[sel], a_u[sel], a_int[sel], d[sel], w[sel]
        DW = None if DW is None else DW[sel]
    sw = np.sqrt(w).astype(np.float64)
    if edge:
        X = np.concatenate([a_u[:, None].astype(np.float64), _basis(d, a_int, DW)], 1) * peak
    else:
        X = a_u[:, None].astype(np.float64) * peak
    P = X.shape[1]
    G = np.zeros((P, P))
    g = np.zeros(P)
    for c in range(3):
        A = X * (base[:, c].astype(np.float64) * sw)[:, None]
        b = y[:, c].astype(np.float64) * sw
        G += A.T @ A
        g += A.T @ b
    if edge:
        R, S = _reg_rows(P)
        nrm = np.sqrt(max(float(w.sum()), 1.0))
        G += (LAM_RIDGE * nrm) ** 2 * (R.T @ R) + (3.0 * LAM_RIDGE * nrm) ** 2 * (S.T @ S)
        lo = np.r_[M_BOUNDS[0], np.full(P - 1, Q_BOUNDS[0])]
        hi = np.r_[M_BOUNDS[1], np.full(P - 1, Q_BOUNDS[1])]
    else:
        lo, hi = np.array([M_BOUNDS[0]]), np.array([M_BOUNDS[1]])
    G += 1e-12 * max(float(np.trace(G)), 1e-12) / P * np.eye(P)
    try:
        L = cholesky(G, lower=True)
        sol = lsq_linear(L.T, solve_triangular(L, g, lower=True), bounds=(lo, hi)).x
    except Exception:                                    # degenerate system: keep the template as it is
        sol = np.r_[1.0, np.zeros(P - 1)]
    return float(sol[0]), (sol[1:] if edge else np.zeros(N_DIR * NQ))


# ---------------------------------------------------------------------------
# Per-mark preparation
# ---------------------------------------------------------------------------

def _extend_ink(a_u, k):
    """Ink where the coverage does not reach (the edge ring): the neighbouring
    stroke's ink, not black. Exact ``k`` wherever ``a_u >= 0.1``."""
    den = cv2.GaussianBlur(a_u, (0, 0), 2.0)
    num = cv2.GaussianBlur(k * a_u[..., None], (0, 0), 2.0)
    ext = num / np.maximum(den, 1e-6)[..., None]
    ext = np.where((den > 1e-4)[..., None], ext, k)
    mix = np.clip(a_u / 0.1, 0.0, 1.0)[..., None]
    return np.clip(mix * k + (1.0 - mix) * ext, 0.0, 1.0).astype(np.float32)


def _ink_regions(tpl, pose, x0, y0, w, h, k):
    """Ink-region label map (1..R) over the window, and R. The template's own
    region labels at the pose when it has them (library templates built with
    several regions), else colour vs grey by the ink's saturation."""
    lab = tpl.get("labels")
    if lab is not None and int(lab.max()) >= 2:
        K = int(lab.max())
        tw, th = ts.template_size(tpl["alpha"], pose["scale"])
        L = cv2.resize(lab, (tw, th), interpolation=cv2.INTER_NEAREST)
        M = np.float32([[1, 0, pose["x"] - x0], [0, 1, pose["y"] - y0]])
        win = cv2.warpAffine(L, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if (win == 0).any() and (win > 0).any():
            dist = np.stack([cv2.distanceTransform((win != r).astype(np.uint8), cv2.DIST_L2, 3) for r in range(1, K + 1)], 0)
            win = np.where(win == 0, np.argmin(dist, 0) + 1, win)
        return win.astype(np.int32), K
    sat = k.max(2) - k.min(2)
    return np.where(sat > 0.08, 2, 1).astype(np.int32), 2


def _reliability(I, F):
    """(known-pixel mask, reliable-background mask): known = outside the footprint
    and not on a strong edge; reliable = enough known pixels in a 15x15 box, low
    spread among them, and close to a known pixel."""
    lum = I @ LUMA
    g = np.hypot(cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    edge = cv2.dilate((g > GRAD_TOL).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    known = (~F) & (~edge)
    m = known.astype(np.float32)
    box = (RELIABLE_BOX, RELIABLE_BOX)
    bm = cv2.boxFilter(m, -1, box, normalize=False)
    s1 = cv2.boxFilter(I * m[..., None], -1, box, normalize=False)
    s2 = cv2.boxFilter(I * I * m[..., None], -1, box, normalize=False)
    den = np.maximum(bm, 1e-6)[..., None]
    mu = s1 / den
    var = (s2 / den - mu * mu).max(2)
    dist = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 5)
    return known, (bm >= MIN_KNOWN_IN_BOX) & (np.sqrt(np.maximum(var, 0.0)) < STD_TOL) & (dist <= MAX_KNOWN_DIST)


def prepare(img, tpl, pose):
    """Everything per mark that does not depend on the fit: window, template
    coverage / ink at the pose, signed distance, footprint / zone, local
    background and reliability, text mask, parts and ink regions. ``None`` when
    the window is degenerate."""
    H, W = img.shape[:2]
    x0, y0, a_u, k_t = trm.render_pixel(tpl, dict(pose, sigma=0.0), (H, W), margin=MARGIN)
    h, w = a_u.shape
    if w < 8 or h < 8 or float(a_u.max()) < 0.05:
        return None
    if EXTEND_INK:
        k_t = _extend_ink(a_u, k_t)
    I = img[y0:y0 + h, x0:x0 + w].astype(np.float32) / 255.0
    d = signed_distance(render_ss(tpl, pose, x0, y0, w, h))
    a_int = cv2.dilate(a_u, np.ones((5, 5), np.uint8))
    sup = (a_u > 0.02).astype(np.uint8)
    F = (cv2.dilate(sup, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0) | ((d > -2.0) & (a_int > 0.02))
    zone = F & ((a_u > 0.02) | (d > -3.0))
    u8 = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
    B = cv2.inpaint(u8, F.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    known, rel_bg = _reliability(I, F)
    peak = float(tpl["opacity_peak"])
    y = B - I
    text = cv2.dilate(((y @ LUMA) > np.maximum(0.12, 2.5 * peak * a_int)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rel0 = zone & rel_bg & ~text
    rr = max(2, int(0.012 * np.hypot(h, w)))
    n_lab, plab = cv2.connectedComponents(cv2.dilate(sup, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))))
    rlab, n_reg = _ink_regions(tpl, pose, x0, y0, w, h, k_t)
    return dict(x0=x0, y0=y0, w=w, h=h, I=I, a_u=a_u, k_t=k_t, d=d, a_int=a_int, sup=sup, F=F, zone=zone, B=B,
                rel_bg=rel_bg, text=text, rel0=rel0, y=y, peak=peak, n_parts=n_lab - 1, plab=plab,
                rlab=rlab, n_reg=n_reg, DW=None)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit(P, rounds=ROUNDS, edge=True, per_part=True, tint=True):
    """IRLS fit of the model on ``prepare``'s reliable pixels. Returns a dict
    (alpha ``a_now`` over the window, ink ``kk``, tint ``dk`` (R + 1, 3), ``m`` per
    part, Q, final Tukey weights ``wts``, reliable count ``n_rel``) or ``None``
    when fewer than ``MIN_RELIABLE`` reliable pixels exist."""
    a_u, a_int, d, B, y, k_t, peak = P["a_u"], P["a_int"], P["d"], P["B"], P["y"], P["k_t"], P["peak"]
    plab, rlab = P["plab"], P["rlab"]
    if edge and P["DW"] is None:
        P["DW"] = dir_weights(d)
    DW = P["DW"] if edge else None
    dk = np.zeros((P["n_reg"] + 1, 3), np.float64)
    nQ = N_DIR * NQ
    params = {0: (1.0, np.zeros(nQ))}
    zero_q = np.zeros(nQ)

    def alpha_of(m, Q):
        if edge:
            return _alpha(m, Q, a_u, a_int, d, DW, peak)
        return peak * m * a_u

    a_now = alpha_of(*params[0])
    rng = np.random.default_rng(0)
    dwf = (lambda sel: DW[sel]) if edge else (lambda sel: None)
    wts = None
    n_rel = 0
    for it in range(rounds):
        kk = np.clip(k_t + dk[rlab], 0.0, 1.0)
        base = B - kk
        rel = P["rel0"] & (np.sqrt((base * base).sum(2)) > MIN_BASE)
        n_rel = int(rel.sum())
        if n_rel < MIN_RELIABLE:
            return None
        r = y - a_now[..., None] * base
        rl = np.sqrt((r * r).sum(2))
        c = 4.685 * max(1.4826 * float(np.median(rl[rel])), 0.008)
        wts = (_tukey(rl, c) * rel).astype(np.float32)
        s = wts > 0
        if int(s.sum()) < 200:
            return None
        params = {0: _fit_part(y[s], base[s], a_u[s], a_int[s], d[s], dwf(s), wts[s], peak, edge, rng)}
        a_now = alpha_of(*params[0])
        if per_part:
            for p in range(1, P["n_parts"] + 1):
                sp = s & (plab == p)
                if int(sp.sum()) >= MIN_PART_PX:
                    params[p] = _fit_part(y[sp], base[sp], a_u[sp], a_int[sp], d[sp], dwf(sp), wts[sp], peak, edge, rng)
                    a_now = np.where(plab == p, alpha_of(*params[p]), a_now)
        if tint:
            for g in range(1, P["n_reg"] + 1):
                sg = s & (rlab == g)
                if int(sg.sum()) < 50:
                    continue
                ag = a_now[sg][:, None]
                wg = wts[sg][:, None]
                tgt = ag * (B[sg] - k_t[sg]) - y[sg]
                dk[g] = np.clip((wg * ag * tgt).sum(0) / ((wg * ag * ag).sum(0) + LAM_K * wg.sum()), -DK_CLIP, DK_CLIP)
    return dict(a_now=a_now, kk=np.clip(k_t + dk[rlab], 0.0, 1.0), dk=dk, params=params, wts=wts, n_rel=n_rel)


def fit_quality(img, tpl, pose):
    """Cheap fit (global part only, no edge profile, no tint, no polish, 2
    rounds) of ``tpl`` at ``pose``: how much of the page's darkening there the
    template explains. Returns dict(E, energy, n_rel, m) with
    ``E = 1 - sum w |y - a base|^2 / sum w |y|^2`` on the reliable pixels and
    ``energy = sum w (|y|^2 - |y - a base|^2)`` (the explained energy; comparable
    between templates on the same mark), or ``E = None`` when the page has too few
    reliable pixels there to judge."""
    P = prepare(img, tpl, pose)
    out = dict(E=None, energy=0.0, n_rel=0, m=None)
    if P is None:
        return out
    f = fit(P, rounds=2, edge=False, per_part=False, tint=False)
    if f is None:
        return out
    base = P["B"] - f["kk"]
    rel = P["rel0"] & (np.sqrt((base * base).sum(2)) > MIN_BASE)
    r = P["y"] - f["a_now"][..., None] * base
    rl = np.sqrt((r * r).sum(2))
    c = 4.685 * max(1.4826 * float(np.median(rl[rel])), 0.008)
    w = _tukey(rl, c) * rel
    yy = (P["y"] ** 2).sum(2)
    num = float((w * (rl ** 2)).sum())
    den = float((w * yy).sum())
    if den < 1e-9:
        return out
    return dict(E=1.0 - num / den, energy=den - num, n_rel=f["n_rel"], m=float(f["params"][0][0]))


# ---------------------------------------------------------------------------
# Polish, guard
# ---------------------------------------------------------------------------

def _polish(P, a_par, kk):
    """Per-pixel opacity read from the page where it can be read (see module
    docstring, 5). Returns (alpha, number of polished pixels)."""
    I, B, y, zone = P["I"], P["B"], P["y"], P["zone"]
    dvec = B - kk
    dd = (dvec * dvec).sum(2)
    a_d = (y * dvec).sum(2) / np.maximum(dd, 1e-6)
    perp = np.sqrt(np.maximum(((y - a_d[..., None] * dvec) ** 2).sum(2), 0.0))
    k3 = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(a_par, k3) * 1.3 + 0.03
    lo = cv2.erode(a_par, k3) * 0.75 - 0.03
    ok = zone & P["rel_bg"] & ~P["text"] & (dd > 0.08 ** 2) & (perp < 0.03 + 0.1 * a_par) & (a_d >= lo) & (a_d <= hi)
    wgt = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), 0.7) * ok
    a = np.clip(wgt * np.clip(a_d, 0.0, stamp_fit.MAX_ALPHA) + (1.0 - wgt) * a_par, 0.0, stamp_fit.MAX_ALPHA) * zone
    return a.astype(np.float32), int(ok.sum())


def ghost_score(res_lum255, d, mask, cut=GHOST_CUT):
    """RMS over 0.5 px bins of the signed distance ``d`` in [-4, 4) of the mean
    luminance residual (grey levels) on ``mask`` pixels with |residual| <= ``cut``:
    the leftover-rim statistic of ``scripts/alpha_net/eval_stamp_rim.py``. 0 for an
    empty selection."""
    sel = mask & (np.abs(res_lum255) <= cut) & (d >= D_LO) & (d < D_HI)
    if not sel.any():
        return 0.0
    nb = int(round((D_HI - D_LO) / D_STEP))
    idx = np.clip(((d[sel] - D_LO) / D_STEP).astype(np.int64), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    sm = np.bincount(idx, weights=res_lum255[sel].astype(np.float64), minlength=nb)
    v = sm[cnt > 0] / cnt[cnt > 0]
    return float(np.sqrt(np.mean(v * v)))


def _inverse(I, a, k):
    return np.clip((I - a[..., None] * k) / (1.0 - a[..., None]), 0.0, 1.0)


def _apply(out_win, I, a, k):
    """Write the exact inverse into ``out_win`` (uint8 view) where a > 1e-4."""
    msk = a > trm.MIN_ALPHA
    if msk.any():
        rec = _inverse(I, a, k)
        out_win[msk] = np.round(rec[msk] * 255).astype(np.uint8)


def _r(v, n=3):
    return None if v is None else round(float(v), n)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def remove_adaptive(img, marks, polish=True):
    """Remove already-located template marks with the page-adaptive model.

    ``marks``: list of dicts with ``kind``, ``parts`` (one pose dict: ``name``
    registered in ``stamp_fit.templates()``, ``scale``, ``x``, ``y``, ``sigma``) and
    ``tpl``, as for ``template_remove.remove_pixel`` (which this has the contract
    of). Each mark's ``parts`` is replaced by its sub-pixel refined pose. Returns
    (cleaned uint8, alpha float32 = the per-pixel max of the opacities removed,
    per-mark info list). Pixels outside every mark's zone are byte-identical to
    ``img``."""
    H, W = img.shape[:2]
    if not marks:
        return img.copy(), np.zeros((H, W), np.float32), []
    obs0 = img.astype(np.float32) / 255.0
    foot = np.zeros((H, W), bool)
    for mk in marks:
        foot |= ts.footprint_mask(mk["tpl"]["alpha"], mk["parts"][0], (H, W), 3)
    Bm = ts.paper_background_masked(img, foot)
    d0 = (Bm - obs0) @ LUMA
    cur = img.copy()
    obs_cur = obs0.copy()                                   # float view of the running output
    A = np.zeros((H, W), np.float32)
    infos = []
    for mk in marks:
        t0 = time.time()
        tpl = mk["tpl"]
        part = stamp_fit._subpixel(mk["parts"][0], d0, (H, W))
        pose = dict(part, sigma=0.0)                       # the edge profile replaces the blur
        P = prepare(cur, tpl, pose)
        info = dict(kind=mk["kind"], model="adaptive", iou=mk.get("iou"), change=mk.get("change"),
                    control=mk.get("control"), changed_fraction=mk.get("changed_fraction"), reverted=False,
                    polished=0, reason=None)
        if P is None:
            mk["parts"] = [pose]
            info.update(parts=[_part_info(pose)], reason="degenerate window", m={}, dk=[], reliable=0, regions={},
                        ghost=None, seconds=round(time.time() - t0, 2))
            infos.append(info)
            continue
        x0, y0, w, h = P["x0"], P["y0"], P["w"], P["h"]
        I, B, zone = P["I"], P["B"], P["zone"]
        # the v2 per-pixel model of this mark (guard and fallback): its own blur, restricted to the zone
        v2 = trm.pixel_model(tpl, part, obs_cur, Bm, (H, W))
        v2_a = _crop(v2["a_w"], v2["x0"], v2["y0"], x0, y0, w, h) * zone
        v2_k = _crop(v2["k"], v2["x0"], v2["y0"], x0, y0, w, h)
        f = fit(P)
        if f is None:
            a, k, used = v2_a, v2_k, "v2"
            info.update(reason="too few reliable pixels", m={}, dk=[], reliable=int(P["rel0"].sum()), ghost=None)
            mk["parts"] = [part]
        else:
            a_par = (np.clip(f["a_now"], 0.0, stamp_fit.MAX_ALPHA) * zone).astype(np.float32)
            kk = f["kk"]
            a3, n_pol = _polish(P, a_par, kk) if polish else (a_par, 0)
            # guard: v3 must not leave a stronger rim than the v2 model on the same pixels
            mask = P["rel_bg"] & ~P["text"]
            lum_I = (I @ LUMA) * 255.0
            lum_B = (B @ LUMA) * 255.0
            g_before = ghost_score(lum_I - lum_B, P["d"], mask)
            g_v3 = ghost_score(_lum255(I, a3, kk) - lum_B, P["d"], mask)
            g_v2 = ghost_score(_lum255(I, v2_a, v2_k) - lum_B, P["d"], mask)
            info.update(reliable=f["n_rel"], polished=n_pol, ghost=dict(before=_r(g_before, 2), v3=_r(g_v3, 2), v2=_r(g_v2, 2)))
            if g_v3 > g_v2 + GUARD_MARGIN:
                a, k, used = v2_a, v2_k, "v2"
                info["reverted"] = True
                mk["parts"] = [part]
            else:
                a, k, used = a3, kk, "v3"
                mk["parts"] = [pose]
            ms = {p: float(v[0]) for p, v in f["params"].items()}
            info.update(m={int(p): round(v, 3) for p, v in ms.items()},
                        dk=[[int(round(v * 255)) for v in row] for row in f["dk"][1:]])
        sub = cur[y0:y0 + h, x0:x0 + w]
        _apply(sub, I, a, k)
        obs_cur[y0:y0 + h, x0:x0 + w] = sub.astype(np.float32) / 255.0
        Asub = A[y0:y0 + h, x0:x0 + w]
        np.maximum(Asub, a, out=Asub)
        core = P["a_u"] >= 0.5
        mean_ink = k[core].mean(0) if core.any() else np.full(3, stamp_fit.INK_LUM_PRIOR, np.float32)
        strength = float(np.median(a[core])) if core.any() else 0.0
        info.update(parts=[_part_info(mk["parts"][0])], used=used, zone_px=int(zone.sum()), sigma=_r(mk["parts"][0].get("sigma", 0.0)),
                    background="local inpaint", regions={"adaptive": dict(strength=round(strength, 3),
                                                                          ink=[int(v) for v in np.round(mean_ink * 255)],
                                                                          source="page-adaptive")})
        info["seconds"] = round(time.time() - t0, 2)
        infos.append(info)
    for mk, inf in zip(marks, infos):
        inf["change_after"] = round(float(stamp_fit._colour_change(cur, mk["parts"], (H, W))[0]), 2)
    return cur, A, infos


def _part_info(p):
    return {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in p.items()}


def _lum255(I, a, k):
    """Luminance (grey levels) of the exact inverse, 0 where a <= 1e-4 left as I."""
    rec = np.where((a > trm.MIN_ALPHA)[..., None], _inverse(I, a, k), I)
    # the written image is uint8: the guard sees the same rounding
    return (np.round(rec * 255.0) / 255.0 @ LUMA) * 255.0


def _crop(arr, ax0, ay0, x0, y0, w, h):
    """``arr`` (a window whose top-left is page (ax0, ay0)) restricted to the
    window (x0, y0, w, h) of the page, zero where ``arr`` does not reach."""
    out = np.zeros((h, w) + arr.shape[2:], arr.dtype)
    sx0, sy0 = max(x0, ax0), max(y0, ay0)
    sx1, sy1 = min(x0 + w, ax0 + arr.shape[1]), min(y0 + h, ay0 + arr.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0 - ay0:sy1 - ay0, sx0 - ax0:sx1 - ax0]
    return out
