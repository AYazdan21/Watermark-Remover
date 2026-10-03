"""Template Builder: estimate a site's watermark from a folder of its pages.

A watermark that a site stamps on every page is the one thing those pages
have in common: the page content differs, the mark does not. Given 20+ pages
and one box around the mark on one of them, this module (builder v2)

1. makes a rough colour seed template from the faint darkening inside the box
   (``template_signal``: paper background interpolated across the box, text
   zeroed),
2. registers every page against it: multi-scale colour-matched correlation over
   ABSOLUTE scales (a site's mark need not scale with the page width) plus local
   refinement; a candidate pose is accepted only if the page itself shows the
   mark (Stamp Fit's evidence check) -- pages with a different mark, or none,
   are rejected there, not by a size-consistency rule,
3. warps all accepted pages into one common frame together with their own
   MARK-AWARE paper background (the mark's footprint left out and interpolated,
   so a big solid area is not mistaken for background) and estimates the mark
   per pixel with ``template_matte``: the robust (median + Tukey IRLS) matted
   darkening ``W = a (1 - k)`` over the pages that show bare paper at the
   pixel, a statistical-significance support instead of a noise floor, one
   template-wide ink luminance (calibrated from text strokes crossing the mark
   when possible), hence a per-pixel opacity ``a`` and a per-pixel ink ``k``,
4. repeats registration with the refined template (``outer_iters`` times).

The result is a library template (``template_library``, ``builder_version`` 2)
that Method 5 (``template_stamp_fit``) removes with the per-pixel colour model.

Dekel et al., "On the Effectiveness of Visible Watermarks" (CVPR 2017), is the
basis of the idea (many pages stamped with the same mark; estimate the mark
from what they share). v1 followed its stage 1 literally (median gradients +
Poisson integration) and fitted two flat ink colours; on a coloured, partly
solid mark that left holes and a grey ghost, so v2 estimates the darkening
directly against a background that does not contain the mark.

Known limits
------------
- Assumes the mark is DARKER than the page (true for AriaTender and
  ETENDER). A light mark on a dark banner is not estimated.
- Rotation is fixed at 0; the layout of a multi-part mark is whatever the
  seed page shows (one rigid template).
- Only the product ``a (1 - k)`` is observable on bare paper; the split into
  opacity and ink uses one template-wide ink luminance (calibrated from text
  crossings when >= 300 pixels see them, else ``stamp_fit.INK_LUM_PRIOR``). It
  only matters for page content under the mark.
- Pages where the site placed a different variant of the mark are rejected
  during the build: build one template per variant.

Everything here is deterministic for a fixed input and page order.
"""

import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from . import stamp_fit
from . import template_library as tl
from . import template_matte as tm
from . import template_signal as ts

BUILDER_VERSION = 2
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LUMA = ts.LUMA
MIN_PAGES = 5
SCALE_RANGE = (0.3, 3.5)     # absolute scales searched in every round (page px per template px)
N_SCALES = 30
N_CANDS = 5                  # candidates refined and evidence-checked per page
MIN_INSIDE = 0.4             # a mark may hang off the page, but >= 40% of its footprint must be on it
MIN_AGREE_FRAC = 0.85        # a page whose agreement with the template is below this x the median is rejected
FOOT_DILATE = 3              # px, footprint dilation for the mark-aware background
MAX_EXPANSIONS = 2           # frame growths when the support touches the frame edge
_counter = itertools.count()
_EV_LOCK = threading.Lock()  # stamp_fit's resize cache is a plain dict


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


def _bbox(mask, margin=0, shape=None):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1, y0, y1 = xs.min() - margin, xs.max() + 1 + margin, ys.min() - margin, ys.max() + 1 + margin
    if shape is not None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1], x1), min(shape[0], y1)
    return int(x0), int(y0), int(x1), int(y1)


def _remove_small(mask, min_px):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            out[lab == i] = False
    return out


class _Template:
    """The template a registration round matches against: ``alpha`` (h, w)
    coverage (peak 1), ``W3`` (h, w, 3) colour template or None (grey: match the
    coverage), plus what the locate pass needs."""

    def __init__(self, alpha, W3=None):
        self.alpha = np.ascontiguousarray(alpha, np.float32)
        self.W3 = None
        if W3 is not None and ts.colour_spread(W3) >= 0.10:
            self.W3 = np.ascontiguousarray(W3, np.float32)
        self.thickness = ts.solid_thickness(self.alpha)
        self.key = f"builder:{os.getpid()}:{next(_counter)}"
        stamp_fit.templates()[self.key] = {"alpha": self.alpha, "regions": {"mark": self.alpha}}

    def kernel(self, scale):
        return ts.kernel_for(self.thickness * scale, ts.MAX_SOLID_KERNEL)

    def close(self):
        stamp_fit.templates().pop(self.key, None)


# ---------------------------------------------------------------------------
# Seed template
# ---------------------------------------------------------------------------

def _seed_template(img, box):
    """Rough colour template from the faint darkening inside ``box`` = (x, y, w,
    h): the paper is interpolated across the whole box from the bright pixels
    around it (the closing used for locating needs a kernel wider than the
    mark's thickest solid area, which is unknown here). The result is the box
    plus a 20% zero margin on each side, at seed-page scale: (alpha, W3)."""
    H, W = img.shape[:2]
    x, y, w, h = box
    fp = np.zeros((H, W), bool)
    fp[y:y + h, x:x + w] = True
    B = ts.paper_background_masked(img, fp)
    c3 = ts.normalise3(ts.band_signal3(img, B)[y:y + h, x:x + w])
    c0 = c3.max(axis=2)
    keep = _remove_small(c0 >= 0.15, 20)
    mx, my = int(round(0.2 * w)), int(round(0.2 * h))
    alpha = np.zeros((h + 2 * my, w + 2 * mx), np.float32)
    W3 = np.zeros((h + 2 * my, w + 2 * mx, 3), np.float32)
    alpha[my:my + h, mx:mx + w] = np.where(keep, c0, 0)
    W3[my:my + h, mx:mx + w] = c3 * keep[..., None]
    return alpha, W3


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _locate_page(path, T, scales):
    img = read_rgb(path)
    if img is None:
        return dict(size=None, cands=[], unreadable=True)
    H, W = img.shape[:2]
    res = ts.locate_template(img, T.alpha, scales, T.kernel, n_cands=N_CANDS, color=T.W3, polish=True)
    cands = sorted(res["candidates"], key=lambda c: -c["score"]) if res else []
    return dict(size=(W, H), cands=cands, unreadable=False)


def _judge_page(path, T, loc, min_score):
    """First candidate (best score first) that scores >= ``min_score``, keeps >=
    40% of its footprint on the page AND passes Stamp Fit's evidence check; else
    a rejection that quotes the best candidate's numbers."""
    if loc["unreadable"]:
        return dict(accepted=False, score=0.0, scale=0.0, x=0.0, y=0.0, size=None, reason="unreadable file")
    W, H = loc["size"]
    if not loc["cands"]:
        return dict(accepted=False, score=0.0, scale=0.0, x=0.0, y=0.0, size=(W, H), reason="no candidate found")
    img = read_rgb(path)
    first = None
    for c in loc["cands"]:
        rec = dict(score=float(c["score"]), scale=float(c["scale"]), x=float(c["x"]), y=float(c["y"]), size=(W, H))
        if c["score"] < min_score:
            if first is None:
                first = dict(rec, accepted=False,
                             reason=f"best registration score {c['score']:.2f} < {min_score:.2f} (different mark or none)")
            break
        inside = ts.inside_fraction(T.alpha, c, (H, W))
        if inside < MIN_INSIDE:
            if first is None:
                first = dict(rec, accepted=False,
                             reason=f"best candidate (score {c['score']:.2f}) has only {inside:.0%} of the mark on the page")
            continue
        pose = dict(name=T.key, scale=c["scale"], x=c["x"], y=c["y"], sigma=0.0)
        with _EV_LOCK:
            change, control, frac = stamp_fit._evidence(img, [pose])
        ev = dict(change=round(float(change), 2), control=round(float(control), 2), changed_fraction=round(float(frac), 3))
        why = []
        if change < stamp_fit.MIN_CHANGE:
            why.append(f"change {change:.1f} < {stamp_fit.MIN_CHANGE}")
        if change < stamp_fit.MIN_CHANGE_RATIO * control:
            why.append(f"change {change:.1f} < {stamp_fit.MIN_CHANGE_RATIO}x control {control:.1f}")
        if frac < stamp_fit.MIN_CHANGED_FRACTION:
            why.append(f"changed fraction {frac:.2f} < {stamp_fit.MIN_CHANGED_FRACTION}")
        if not why:
            return dict(rec, accepted=True, reason="", evidence=ev, inside=round(inside, 3))
        if first is None:
            first = dict(rec, accepted=False, evidence=ev, inside=round(inside, 3),
                         reason=f"best candidate (score {c['score']:.2f}) fails the evidence check: " + "; ".join(why))
    return first or dict(accepted=False, score=0.0, scale=0.0, x=0.0, y=0.0, size=(W, H), reason="no usable candidate")


def _register_all(files, T, min_score, workers):
    scales = np.geomspace(SCALE_RANGE[0], SCALE_RANGE[1], N_SCALES)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        locs = list(ex.map(lambda p: _locate_page(p, T, scales), files))
    rows = []
    for p, loc in zip(files, locs):
        r = _judge_page(p, T, loc, min_score)
        r["file"] = os.path.basename(p)
        r["path"] = p
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Frame + warp
# ---------------------------------------------------------------------------

def _frame_geometry(Th, Tw, pads, G, frame_max_width):
    """Frame = template extent plus per-side padding (template px), at ``G`` frame
    px per template px (capped so the frame is at most ``frame_max_width`` wide)."""
    ext_w = Tw + pads["l"] + pads["r"]
    ext_h = Th + pads["t"] + pads["b"]
    G = max(min(G, frame_max_width / float(ext_w)), 8.0 / float(ext_w))
    return dict(pads=dict(pads), G=float(G), Fw=max(8, int(round(ext_w * G))), Fh=max(8, int(round(ext_h * G))),
                Tw=Tw, Th=Th)


def _to_frame(src, x0, y0, ox, oy, r, geo):
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


def _warp_page(row, geo, foot_alpha):
    """The page's crop ``I`` (uint8), its mark-aware paper background ``B``
    (float16) and the validity mask ``V``, all in the frame."""
    img = read_rgb(row["path"])
    if img is None:
        return None
    H, W = img.shape[:2]
    s, x, y = row["scale"], row["x"], row["y"]
    G, Fw, Fh, pads = geo["G"], geo["Fw"], geo["Fh"], geo["pads"]
    r = s / G
    ox, oy = x - s * pads["l"], y - s * pads["t"]
    x0 = max(0, int(np.floor(ox)) - 3); y0 = max(0, int(np.floor(oy)) - 3)
    x1 = min(W, int(np.ceil(ox + r * Fw)) + 3); y1 = min(H, int(np.ceil(oy + r * Fh)) + 3)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    fp = ts.footprint_mask(foot_alpha, dict(scale=s, x=x, y=y), (H, W), FOOT_DILATE)
    Bfull = ts.paper_background_masked(img, fp)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    Bc = np.ascontiguousarray(Bfull[y0:y1, x0:x1])
    I, (small, M) = _to_frame(crop, x0, y0, ox, oy, r, geo)
    Bf, _ = _to_frame(Bc, x0, y0, ox, oy, r, geo)
    ones = np.full(small.shape[:2], 255, np.uint8)
    V = cv2.warpAffine(ones, M, (Fw, Fh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 250
    return I, Bf.astype(np.float16), V


def _page_strengths(I, B, V, est, A, k, a_peak):
    """Per page, the least-squares multiplier ``m_i`` of the template's opacity
    (``B_i - I_i = m_i a (B_i - k)``) over the mark's pixels that show bare paper
    on that page, and the cosine ``c_i`` between the observed darkening and the
    template's over the same pixels. Raw (no prior, no clip): ``m_i`` ~ 0 means
    the page does not show the mark at the fitted pose, a low ``c_i`` that it shows
    something else there (a pose on clutter that still fits some strength)."""
    N = V.shape[0]
    core = A > 0.1
    a_abs = (a_peak * A)[..., None]
    out = np.zeros(N, np.float32)
    cos = np.zeros(N, np.float32)
    se = np.ones((3, 3), np.uint8)
    for i in range(N):
        txt = cv2.dilate(est["text"][i].astype(np.uint8), se) > 0
        sel = core & V[i] & ~txt
        if int(sel.sum()) < 20:
            continue
        Bi = B[i][sel].astype(np.float32)
        y = Bi - I[i][sel].astype(np.float32) * (1.0 / 255.0)
        x = a_abs[sel] * (Bi - k[sel])
        out[i], _ = tm.fit_multiplier(y, x, prior=0.0, min_pixels=20, clip=None)
        cos[i] = float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-9))
    return out, cos


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

def make_preview(A, a, ink, support, max_side=640):
    """Three panels side by side: the opacity ``a`` as grey on white (normalised
    to its peak), the mark as it appears on paper (``a k + (1 - a) white``) and
    the support mask."""
    H, W = A.shape
    f = min(3.0, max_side / max(H, W))
    sz = (max(1, int(W * f)), max(1, int(H * f)))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    An = np.clip(cv2.resize(A, sz, interpolation=interp), 0, 1)
    a_ = np.clip(cv2.resize(a, sz, interpolation=interp), 0, 1)
    ink_ = np.stack([cv2.resize(ink[..., c], sz, interpolation=interp) for c in range(3)], -1)
    sup = cv2.resize(support.astype(np.uint8) * 255, sz, interpolation=cv2.INTER_NEAREST)
    left = np.repeat((255 * (1 - An))[..., None], 3, 2)
    mid = 255.0 * (a_[..., None] * ink_ + (1 - a_[..., None]))
    right = np.repeat(255 - sup[..., None], 3, 2).astype(np.float32)
    gap = np.full((sz[1], 8, 3), 128.0)
    return np.clip(np.hstack([left, gap, mid, gap, right]), 0, 255).astype(np.uint8)


def _overlay(path, alpha, pose, out_path):
    img = read_rgb(path)
    if img is None:
        return False
    H, W = img.shape[:2]
    tw, th = ts.template_size(alpha, pose["scale"])
    m = int(0.2 * max(tw, th))
    x0, y0 = max(0, int(pose["x"]) - m), max(0, int(pose["y"]) - m)
    x1, y1 = min(W, int(pose["x"]) + tw + m), min(H, int(pose["y"]) + th + m)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    cov = ts.render_template(alpha, pose["scale"], pose["x"], pose["y"], (x0, y0, x1 - x0, y1 - y0))
    crop = np.array(img[y0:y1, x0:x1])          # a copy: read_rgb returns a read-only array, which a full-width crop would stay
    cnts, _ = cv2.findContours((cov >= 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnts, -1, (255, 0, 0), max(1, int(round(0.004 * max(crop.shape[:2])))))
    f = min(1.0, 1100.0 / max(crop.shape[:2]))
    if f < 1:
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    Image.fromarray(crop).save(out_path)
    return True


# ---------------------------------------------------------------------------
# One estimation round (warp + estimate + support + opacity/ink)
# ---------------------------------------------------------------------------

def _estimate_round(acc, T, foot_alpha, pads, frame_max_width, workers, prog, base, span):
    """Warps the accepted pages into a frame (growing it up to twice when the
    support touches its edge), estimates the mark and returns a dict with the
    full-frame maps, the geometry and the per-page rows (None when it fails)."""
    Th, Tw = T.alpha.shape
    G0 = float(np.median([r["scale"] for r in acc]))
    expansions = 0
    pads = dict(pads)
    while True:
        geo = _frame_geometry(Th, Tw, pads, G0, frame_max_width)
        prog(base + 0.30 * span, f"warping {len(acc)} pages into the frame")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            warped = list(ex.map(lambda r: _warp_page(r, geo, foot_alpha), acc))
        keep = [(r, wp) for r, wp in zip(acc, warped) if wp is not None]
        if len(keep) < MIN_PAGES:
            return None
        rows = [r for r, _ in keep]
        I = np.stack([wp[0] for _, wp in keep])
        B = np.stack([wp[1] for _, wp in keep])
        V = np.stack([wp[2] for _, wp in keep])
        del warped, keep
        N = len(rows)
        prog(base + 0.55 * span, "estimating the matted darkening")
        est = tm.estimate(I, B, V)
        sup, counts = tm.support_mask(est, N)
        if sup.sum() >= 30:
            # second pass: with a first opacity known, the exact per-page sample D + a (1 - B) replaces D / B
            L1 = tm.calibrate_ink(I, V, est, sup, est["W"], stamp_fit.INK_LUM_PRIOR)["L_ink"]
            a1, _ = tm.opacity_and_ink(est["W"], L1, sup)
            est = tm.estimate(I, B, V, a_prior=a1)
            sup, counts = tm.support_mask(est, N)
        edge = {"l": bool(sup[:, 0].any()), "r": bool(sup[:, -1].any()),
                "t": bool(sup[0, :].any()), "b": bool(sup[-1, :].any())}
        touched = [s for s, v in edge.items() if v]
        if touched and expansions < MAX_EXPANSIONS:
            ext_w = Tw + pads["l"] + pads["r"]
            ext_h = Th + pads["t"] + pads["b"]
            for s in touched:
                pads[s] += 0.15 * (ext_w if s in "lr" else ext_h)
            expansions += 1
            continue
        return dict(rows=rows, I=I, B=B, V=V, est=est, sup=sup, counts=counts, geo=geo, N=N,
                    expansions=expansions, touches=touched, pads=pads)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_template(pages_dir, seed_page, seed_box=None, seed_mask=None, name=None, library_dir=None,
                   max_pages=60, outer_iters=3, min_reg_score=0.20, frame_max_width=1000,
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
    seed_box_input = [int(round(v)) for v in (x, y, w, h)]      # as given, before the 5% padding (what a rebuild needs)
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
    a0, W3_0 = _seed_template(seed_img, box)
    if (a0 > 0.3).sum() < 30:
        return fail("no faint darkening found inside the box (is the mark darker than the page, and inside the box?)")

    T = _Template(a0, W3_0)
    keys = [T.key]
    warnings = []
    final = None
    reg_rows = None
    iters = max(1, int(outer_iters))
    try:
        for it in range(iters):
            base = 0.05 + 0.9 * it / iters
            span = 0.9 / iters
            prog(base, f"registering {len(order)} pages (round {it + 1}/{iters})")
            Th, Tw = T.alpha.shape
            rows = _register_all(order, T, min_reg_score, workers)
            acc = [r for r in rows if r["accepted"]]
            n_read = sum(1 for r in rows if r["reason"] != "unreadable file")
            if n_read < MIN_PAGES:
                return fail(f"only {n_read} readable pages")
            if len(acc) < MIN_PAGES:
                return fail(f"only {len(acc)} of {n_read} pages showed the seed mark (need {MIN_PAGES}). Draw a tighter box "
                            f"around the mark, pick a seed page where it is clearly visible, or lower the registration score.")
            # early stop when nothing moved (final rows are already in this template's coordinates)
            if it > 0 and final is not None:
                prev = {r["file"]: r for r in final["rows"]}
                if sorted(prev) == sorted(r["file"] for r in acc):
                    move = max(np.hypot((a["x"] + a["scale"] * Tw / 2) - (prev[a["file"]]["x"] + prev[a["file"]]["scale"] * Tw / 2),
                                        (a["y"] + a["scale"] * Th / 2) - (prev[a["file"]]["y"] + prev[a["file"]]["scale"] * Th / 2))
                               for a in acc)
                    if move < 0.5:
                        reg_rows = rows
                        break
            reg_rows = rows
            pads = {s: 0.0 for s in "ltrb"} if it == 0 else {"l": 0.2 * Tw, "r": 0.2 * Tw, "t": 0.2 * Th, "b": 0.2 * Th}
            foot = np.ones_like(T.alpha) if it == 0 else T.alpha
            rd = _estimate_round(acc, T, foot, pads, frame_max_width, workers, prog, base, span)
            if rd is None:
                return fail("too few pages could be warped into the frame")
            est, sup, N = rd["est"], rd["sup"], rd["N"]
            if sup.sum() < 30:
                return fail("the pages share no consistent darkening at the registered poses: the registration did not find a shared mark")
            prog(base + 0.75 * span, "opacity, ink and regions")
            W = est["W"]
            cal = tm.calibrate_ink(rd["I"], rd["V"], est, sup, W, stamp_fit.INK_LUM_PRIOR)
            L_ink = cal["L_ink"]
            a, k = tm.opacity_and_ink(W, L_ink, sup)
            a_peak = float(np.percentile(a[sup], 99.5))
            if a_peak < 0.01:
                return fail("the estimated mark is too faint to use")
            A = np.clip(a / a_peak, 0, 1).astype(np.float32)
            m_i, cos_i = _page_strengths(rd["I"], rd["B"], rd["V"], est, A, k, a_peak)
            # crop to the support (+4 px)
            bb = _bbox(sup, margin=4, shape=sup.shape)
            cx0, cy0, cx1, cy1 = bb
            sl = (slice(cy0, cy1), slice(cx0, cx1))
            geo = rd["geo"]
            G = geo["G"]
            new_rows = []
            for r, mi, ci in zip(rd["rows"], m_i, cos_i):
                s2 = r["scale"] / G
                ox = r["x"] - r["scale"] * rd["pads"]["l"]
                oy = r["y"] - r["scale"] * rd["pads"]["t"]
                new_rows.append(dict(file=r["file"], path=r["path"], scale=s2, x=ox + s2 * cx0, y=oy + s2 * cy0,
                                     score=r["score"], strength=float(mi * a_peak), mult=float(mi), cos=float(ci), size=r["size"],
                                     evidence=r.get("evidence")))
            final = dict(A=A[sl].copy(), a=a[sl].copy(), k=k[sl].copy(), sup=sup[sl].copy(), a_peak=a_peak,
                         L_ink=L_ink, cal=cal, geo=geo, N=N, rows=new_rows, counts=rd["counts"],
                         expansions=rd["expansions"], touches=rd["touches"])
            T.close()
            T = _Template(final["A"], ts.colour_template(final["A"], final["k"]))
            keys.append(T.key)
            prog(base + span, f"round {it + 1} done")
    finally:
        for kk in keys:
            stamp_fit.templates().pop(kk, None)

    # ---- finish ------------------------------------------------------------
    prog(0.96, "saving")
    A, a, k, sup = final["A"], final["a"], final["k"], final["sup"]
    a_peak, L_ink = final["a_peak"], final["L_ink"]
    Th, Tw = A.shape
    labels, ink_mean = tm.cluster_regions(a, k, A, sup)
    K = int(labels.max())
    ink_u8 = [tuple(int(v) for v in np.round(c * 255)) for c in ink_mean]
    pix = [int((labels == j).sum()) for j in range(1, K + 1)]
    ink_map = np.where(sup[..., None], k, ink_mean[0][None, None, :])
    ink_map = np.round(ink_map * 255).astype(np.uint8)
    # a page whose fitted strength is ~0 does not actually show the mark at the fitted pose
    rows_final = final["rows"]
    med_m = float(np.median([r["mult"] for r in rows_final]))
    med_c = float(np.median([r["cos"] for r in rows_final]))
    weak = {r["file"]: r for r in rows_final if r["mult"] < 0.3 * med_m or r["cos"] < MIN_AGREE_FRAC * med_c}
    rows_final = [r for r in rows_final if r["file"] not in weak]
    accepted_rows = {r["file"]: r for r in rows_final}
    all_rows = []
    for r in reg_rows or []:
        acc_r = accepted_rows.get(r["file"])
        if acc_r is not None:
            all_rows.append(dict(file=r["file"], accepted=True, score=round(acc_r["score"], 4), scale=round(acc_r["scale"], 5),
                                 x=round(acc_r["x"], 2), y=round(acc_r["y"], 2), strength=round(acc_r["strength"], 4),
                                 agreement=round(acc_r["cos"], 4), reason="", evidence=acc_r.get("evidence")))
        else:
            w_ = weak.get(r["file"])
            if w_ and w_["mult"] < 0.3 * med_m:
                reason = (f"no measurable mark at the fitted pose (strength {w_['strength']:.3f} vs median "
                          f"{med_m * a_peak:.3f})")
            elif w_:
                reason = (f"the page disagrees with the template at the fitted pose (agreement {w_['cos']:.2f} vs "
                          f"median {med_c:.2f}; probably a fit on clutter)")
            else:
                reason = r.get("reason") or "could not be warped into the frame"
            all_rows.append(dict(file=r["file"], accepted=False, score=round(r.get("score", 0.0), 4), scale=None, x=None,
                                 y=None, strength=None, reason=reason, evidence=r.get("evidence")))
    acc_rows = list(rows_final)
    widths = [r["scale"] * Tw for r in acc_rows]
    rel = [r["scale"] * Tw / r["size"][0] for r in acc_rows]
    stren = [r["strength"] for r in acc_rows]
    if final["touches"]:
        warnings.append("the mark may extend past your box (the support still touches the frame edge after "
                        f"{final['expansions']} expansion(s)); draw a bigger box")
    if len(acc_rows) < 20:
        warnings.append(f"only {len(acc_rows)} pages were used; 20+ gives a cleaner estimate")
    if final["cal"]["source"] == "prior":
        warnings.append("ink luminance is the prior (" + (final["cal"]["reason"] or "no text crossings")
                        + "); it only matters for page content under the mark")
    nrej = len(all_rows) - len(acc_rows)
    sw = ts.stroke_width(A)
    elapsed = time.time() - t_start
    stat = lambda v: dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))
    meta = dict(name=name, builder_version=BUILDER_VERSION, source_folder=pages_dir_r,
                seed_page=os.path.basename(seed_p), seed_box=[int(v) for v in box], seed_box_input=seed_box_input,
                n_pages_used=len(acc_rows), n_pages_rejected=int(nrej), template_size=[int(Tw), int(Th)],
                rel_width=stat(rel), instance_width_px=stat(widths), strength=stat(stren),
                opacity_peak=float(a_peak), ink_luminance=float(L_ink), ink_luminance_source=final["cal"]["source"],
                ink_luminance_detail=final["cal"].get("reason", ""),
                regions=[dict(id=j, ink_rgb=list(ink_u8[j - 1]), pixels=pix[j - 1]) for j in range(1, K + 1)],
                stroke_width_px=float(sw), support_rules=final["counts"], frame_expansions=int(final["expansions"]),
                build_seconds=round(elapsed, 1), warnings=warnings,
                params=dict(params, workers=workers, frame_scale_G=float(final["geo"]["G"]),
                            scale_range=list(SCALE_RANGE), n_scales=N_SCALES))
    try:
        folder = tl.save_template(name, A, labels if K > 1 else None, ink_u8, meta, library_dir=library_dir,
                                  overwrite=overwrite, ink_map=ink_map)
    except Exception as e:
        return fail(str(e))
    preview = make_preview(A, a, k, sup)
    Image.fromarray(preview).save(os.path.join(folder, "preview.png"))
    overlays = []
    if acc_rows:
        pick = np.unique(np.linspace(0, len(acc_rows) - 1, min(6, len(acc_rows))).round().astype(int))
        for n, i in enumerate(pick):
            r = acc_rows[i]
            op = os.path.join(folder, f"overlay_{n:02d}.png")
            if _overlay(r["path"], A, dict(scale=r["scale"], x=r["x"], y=r["y"]), op):
                overlays.append(op)
    report = dict(rows=all_rows, summary=meta, warnings=warnings, elapsed=elapsed)
    with open(os.path.join(folder, "build_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    prog(1.0, "done")
    message = _summary_md(name, folder, meta, all_rows, elapsed, warnings)
    return dict(ok=True, template_dir=folder, report=report, preview=preview, overlays=overlays, message=message)


def _summary_md(name, folder, meta, rows, elapsed, warnings):
    rej = [r for r in rows if not r["accepted"]]
    iw = meta["instance_width_px"]
    cnt = meta["support_rules"]
    lines = [f"### Template `{name}` built in {elapsed:.0f} s",
             f"- saved to `{folder}`",
             f"- pages used **{meta['n_pages_used']}**, rejected **{meta['n_pages_rejected']}**",
             f"- template size {meta['template_size'][0]}x{meta['template_size'][1]} px, "
             f"instance width: median {iw['median']:.0f} px (min {iw['min']:.0f}, max {iw['max']:.0f}); "
             f"as a fraction of page width: median {meta['rel_width']['median']:.3f} "
             f"(min {meta['rel_width']['min']:.3f}, max {meta['rel_width']['max']:.3f})",
             f"- opacity peak **{meta['opacity_peak']:.3f}**; per-page strength: median {meta['strength']['median']:.3f} "
             f"(min {meta['strength']['min']:.3f}, max {meta['strength']['max']:.3f})",
             f"- ink luminance **{meta['ink_luminance']:.2f}** (source: {meta['ink_luminance_source']}"
             + (f", {meta['ink_luminance_detail']}" if meta.get("ink_luminance_detail") else "") + ")",
             f"- support: {cnt['kept']} px kept; rules removed {cnt['removed_by_valid_pages']} (too few pages), "
             f"{cnt['removed_by_min_darkening']} (darkening < 0.012), {cnt['removed_by_significance']} (z < 4), "
             f"{cnt['removed_small_components']} (components < 6 px); gap closing added {cnt['added_by_gap_closing']}",
             f"- frame expansions: {meta['frame_expansions']}",
             f"- ink regions (for the per-region removal option): "
             + ", ".join(f"#{g['id']} rgb{tuple(g['ink_rgb'])} ({g['pixels']} px)" for g in meta["regions"])]
    for wmsg in warnings:
        lines.append(f"- WARNING: {wmsg}")
    if rej:
        lines.append("\n**Rejected pages**\n")
        for r in rej[:30]:
            ev = r.get("evidence")
            evs = (f" [change {ev['change']}, control {ev['control']}, changed fraction {ev['changed_fraction']}]"
                   if ev else "")
            lines.append(f"- `{r['file']}`: {r['reason']}{evs}")
        if len(rej) > 30:
            lines.append(f"- ... and {len(rej) - 30} more (see build_report.json)")
    return "\n".join(lines)
