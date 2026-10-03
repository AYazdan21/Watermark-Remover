"""Per-pixel colour removal ("Per-pixel colour (v2)") for Method 5.

Stamp Fit's removal (``stamp_fit.remove_stamps``) fits one strength and one flat
ink colour per ink REGION and estimates the paper with a closing that is only
right for thin strokes. A coloured, partly solid mark (ETENDER's teal disc) is
neither: the closing's 'background' inside the disc is the disc itself, and one
grey ink per region leaves a teal ghost. This module removes a template whose
opacity ``a(p)`` and ink ``k(p)`` are known PER PIXEL (``template_matte``):

1. Paper: ``template_signal.paper_background_masked`` -- the located marks'
   footprints are left out and the paper is interpolated across them.
2. Pose: ``stamp_fit._subpixel`` on the page's darkening against that paper
   (1/4 px position, tiny scale change, edge blur ``sigma``); it works through
   the registered key, so it needs no edit of ``stamp_fit``.
3. Per-page strength: one scalar ``m`` (the template's opacity is ``m a(p)``) by
   robust least squares of ``B - I = m a (B - k)`` over the footprint pixels
   that show paper under the mark (page text excluded), 3 channels, soft prior
   ``m ~ 1``, trimmed, clipped to [0.3, 2.0] (``template_matte.fit_multiplier``).
4. Exact inverse ``(I - m a k) / (1 - m a)`` written only where ``m a > 1e-4``;
   every other pixel stays byte-identical to the input. The template is
   rendered at the fitted pose with the fitted blur on the PREMULTIPLIED
   channels (``a`` and ``a k`` are blurred, then divided), so a blurred edge
   keeps its ink.

Works for v1 templates too (their ink is constant per region, their opacity
peak is the median build strength).
"""

import cv2
import numpy as np

from . import stamp_fit
from . import template_matte as tm
from . import template_signal as ts

LUMA = ts.LUMA
MIN_ALPHA = 1e-4


def _premult(tpl):
    """(h, w, 4) float32: coverage and coverage x ink, cached on the template."""
    S = tpl.get("_premult")
    if S is None:
        a = tpl["alpha"].astype(np.float32)
        ink = np.asarray(tpl["ink"], np.float32)
        S = np.ascontiguousarray(np.dstack([a, a[..., None] * ink]).astype(np.float32))
        tpl["_premult"] = S
    return S


def render_pixel(tpl, part, shape, margin=None):
    """Template at ``part`` (scale, x, y, sigma) in a page window.

    Returns (x0, y0, a_unit, k): ``a_unit`` (h, w) coverage with peak 1 and ``k``
    (h, w, 3) the ink, both blurred on the premultiplied channels, for the window
    whose top-left is page (x0, y0)."""
    sigma = float(part.get("sigma", 0.0))
    if margin is None:
        margin = int(np.ceil(3.0 * sigma)) + 4
    tw, th = ts.template_size(tpl["alpha"], part["scale"])
    H, W = shape
    x0 = int(max(0, np.floor(part["x"]) - margin)); y0 = int(max(0, np.floor(part["y"]) - margin))
    x1 = int(min(W, np.ceil(part["x"] + tw) + margin)); y1 = int(min(H, np.ceil(part["y"] + th) + margin))
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    t = ts.resized_template(_premult(tpl), part["scale"])
    M = np.float32([[1, 0, part["x"] - x0], [0, 1, part["y"] - y0]])
    win = cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if sigma > 0:
        win = cv2.GaussianBlur(win, (0, 0), sigma)
    a = win[..., 0]
    k = win[..., 1:4] / np.maximum(a, 1e-6)[..., None]
    return x0, y0, np.clip(a, 0.0, 1.0), np.clip(k, 0.0, 1.0)


def text_under_mark(y, x):
    """Pixels where the page shows something much darker than the mark can
    explain (text, rules): luminance darkening above ``max(0.12, 2.5 x`` the
    expected mark darkening``)``, plus a 1 px rim. ``y`` = B - I, ``x`` = a (B - k),
    both (h, w, 3)."""
    ly, lx = y @ LUMA, x @ LUMA
    txt = (ly > np.maximum(tm.TEXT_ABS, tm.TEXT_REL * lx)).astype(np.uint8)
    return cv2.dilate(txt, np.ones((3, 3), np.uint8)) > 0


def remove_pixel(img, marks):
    """Remove already-located template marks with the per-pixel colour model.

    ``marks``: list of dicts with ``kind``, ``parts`` (one pose dict with the
    key ``name`` registered in ``stamp_fit.templates()``, ``scale``, ``x``, ``y``,
    ``sigma``) and ``tpl`` (the loaded template). Each mark's ``parts`` is
    replaced by the sub-pixel refined pose. Returns (cleaned uint8, alpha
    float32 = the opacity actually removed, per-mark info list). Pixels where
    the removed opacity is 0 are byte-identical to ``img``."""
    H, W = img.shape[:2]
    if not marks:
        return img.copy(), np.zeros((H, W), np.float32), []
    obs = img.astype(np.float32) / 255.0
    foot = np.zeros((H, W), bool)
    for mk in marks:
        pose = mk["parts"][0]
        foot |= ts.footprint_mask(mk["tpl"]["alpha"], pose, (H, W), 3)
    B = ts.paper_background_masked(img, foot)
    d = (B - obs) @ LUMA
    A = np.zeros((H, W), np.float32)
    INK = np.zeros((H, W, 3), np.float32)
    infos = []
    for mk in marks:
        tpl = mk["tpl"]
        part = stamp_fit._subpixel(mk["parts"][0], d, (H, W))
        x0, y0, a_unit, k = render_pixel(tpl, part, (H, W))
        h, w = a_unit.shape
        a_abs = (tpl["opacity_peak"] * a_unit)[..., None]
        Bw = B[y0:y0 + h, x0:x0 + w]
        y = Bw - obs[y0:y0 + h, x0:x0 + w]
        x = a_abs * (Bw - k)
        use = (a_unit > 0.1) & ~text_under_mark(y, x)
        m, n_used = tm.fit_multiplier(y[use], x[use], prior=0.2, min_pixels=300, clip=(0.3, 2.0))
        a_w = np.clip(m * a_abs[..., 0], 0.0, stamp_fit.MAX_ALPHA)
        sub = A[y0:y0 + h, x0:x0 + w]
        sel = a_w > sub
        A[y0:y0 + h, x0:x0 + w] = np.where(sel, a_w, sub)
        ink_sub = INK[y0:y0 + h, x0:x0 + w]
        ink_sub[sel] = k[sel]
        core = a_unit >= 0.5
        mean_ink = (k[core].mean(0) if core.any() else np.full(3, stamp_fit.INK_LUM_PRIOR, np.float32))
        mk["parts"] = [part]
        infos.append(dict(kind=mk["kind"], parts=[{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in part.items()}],
                          regions={"pixel": dict(strength=round(float(m * tpl["opacity_peak"]), 3),
                                                 ink=[int(v) for v in np.round(mean_ink * 255)],
                                                 source="per-pixel")},
                          m=round(float(m), 3), n_pixels_used=int(n_used), sigma=round(float(part.get("sigma", 0.0)), 3),
                          background="mark-aware paper", iou=mk.get("iou"), change=mk.get("change"),
                          control=mk.get("control"), changed_fraction=mk.get("changed_fraction")))
    rec = np.clip((obs - A[..., None] * INK) / (1.0 - A[..., None]), 0.0, 1.0)
    out = img.copy()
    mask = A > MIN_ALPHA
    out[mask] = np.round(rec[mask] * 255).astype(np.uint8)
    shape = (H, W)
    for mk, inf in zip(marks, infos):
        inf["change_after"] = round(float(stamp_fit._colour_change(out, mk["parts"], shape)[0]), 2)
    return out, A, infos
