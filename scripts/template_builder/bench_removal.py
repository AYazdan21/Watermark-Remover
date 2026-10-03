"""Synthetic ground-truth benchmark of Method 5's removal models (CLI).

Run from the repo root with the project venv:
    .venv/Scripts/python.exe scripts/template_builder/bench_removal.py --out DIR_OUTSIDE_THE_REPO \
        [--backgrounds wm_backgrounds_v2] [--templates NAME ...] [--library DIR] [--n 40] [--seed 0]

The known clean page lets us measure what is left of a mark, including the
degradations seen on real pages. Per sample: a light background page (mean
luminance >= 150) resized to a width of 700-1300 px; with p = 0.35 a coloured
banner (random saturated colour, horizontal gradient) under the mark area with
white text-like bars and a rounded semi-transparent white pill; the template
composited at 0.35-0.95 x the page width (15% of the samples partly off the
page) with its opacity x U(0.7, 1.4) and an ink jitter of +-15 levels (grey inks
also get a darker 55-100 variant half of the time), all at **2x** resolution and
then downscaled with INTER_AREA, INTER_CUBIC or INTER_LANCZOS4 (the last two give
the dark inner overshoot seen on real pages); p = 0.3 an unsharp mask (amount
0.5, sigma 1); JPEG quality 70 / 85 / 95 with 4:2:0. **The ground truth is the
clean page through the identical pipeline.**

Method 5 is run with only the sample's template; the mark is located once and
removed with every removal model (adaptive, pixel, region). Per sample and model:

* ``located``: the located footprint has an IoU >= 0.5 with the true one;
* ``mae``: mean |cleaned - GT| (grey levels, 3 channels) inside the true footprint
  dilated by 3 px (``mae0`` = the same for the untouched page);
* ``ghost``: the leftover-rim statistic of ``scripts/alpha_net/eval_stamp_rim.py``
  against the GT: RMS over 0.5 px bins of the TRUE pose's signed distance in
  [-4, 4] of the mean luminance residual ``lum(cleaned) - lum(GT)`` (pixels with
  |residual| <= 25 grey levels); ``ghost0`` = the same for the untouched page;
* ``out_alpha``: changed pixels outside the returned alpha support (must be 0),
  ``out_foot``: changed pixels more than 9 px outside the true footprint.

Writes ``samples.csv``, ``summary.json`` / ``summary.md`` (median and p90 per
model x template x {paper, banner} x resampler, over the located samples) and a
few example strips (page | GT | removals) into ``--out``. Refuses an ``--out``
inside the repo (the pages are synthetic but large). Never writes into the
backgrounds folder or ``dataset/``.
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from watermark_remover import template_adaptive as tad  # noqa: E402
from watermark_remover import template_library as tl  # noqa: E402
from watermark_remover import template_signal as ts  # noqa: E402
from watermark_remover import template_stamp_fit as tsf  # noqa: E402

MODELS = ("adaptive", "pixel", "region")
RESAMPLERS = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
DEFAULT_TEMPLATES = ["AriaTender wide (built-in)", "AriaTender stacked (built-in)"]
P_BANNER, P_OFFPAGE, P_SHARP = 0.35, 0.15, 0.3
GHOST_CUT = tad.GHOST_CUT


# ---------------------------------------------------------------------------
# Synthetic samples
# ---------------------------------------------------------------------------

def light_backgrounds(folder, limit=None):
    """Sorted image files whose mean luminance (on a thumbnail) is >= 150."""
    out = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        im = Image.open(os.path.join(folder, f)).convert("L")
        im.thumbnail((160, 160))
        if float(np.asarray(im).mean()) >= 150:
            out.append(os.path.join(folder, f))
        if limit and len(out) >= limit:
            break
    return out


def _hsv_rgb(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(h) % 180, int(s * 255), int(v * 255)]]]), cv2.COLOR_HSV2RGB)[0, 0])


def draw_banner(canvas, box, rng):
    """Coloured banner (horizontal gradient) with white text-like bars and a rounded
    semi-transparent white pill, drawn on the 2x canvas inside ``box`` = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return
    h0 = rng.uniform(0, 180)
    c0 = np.array(_hsv_rgb(h0, rng.uniform(0.7, 1.0), rng.uniform(0.35, 0.9)), np.float32)
    c1 = np.array(_hsv_rgb(h0 + rng.uniform(-20, 20), rng.uniform(0.7, 1.0), rng.uniform(0.35, 0.9)), np.float32)
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    canvas[y0:y1, x0:x1] = np.round(c0[None, None] * (1 - t) + c1[None, None] * t).astype(np.uint8)
    reg = canvas[y0:y1, x0:x1]
    for _ in range(int(rng.integers(6, 13))):                       # white text-like bars
        bh = max(2, int(h * rng.uniform(0.035, 0.07)))
        bw = max(6, int(w * rng.uniform(0.08, 0.35)))
        bx = int(rng.uniform(0, max(1, w - bw)))
        by = int(rng.uniform(0, max(1, h - bh)))
        cv2.rectangle(reg, (bx, by), (bx + bw, by + bh), (255, 255, 255), -1, cv2.LINE_AA)
    pw, ph = int(w * rng.uniform(0.35, 0.6)), int(h * rng.uniform(0.25, 0.4))   # rounded pill
    px, py = int(rng.uniform(0, max(1, w - pw))), int(rng.uniform(0, max(1, h - ph)))
    m = np.zeros((h, w), np.uint8)
    r = max(2, ph // 2)
    cv2.rectangle(m, (px + r, py), (px + pw - r, py + ph), 255, -1)
    cv2.circle(m, (px + r, py + r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(m, (px + pw - r, py + r), r, 255, -1, cv2.LINE_AA)
    a = (cv2.GaussianBlur(m, (0, 0), 1.2).astype(np.float32) / 255.0 * rng.uniform(0.2, 0.35))[..., None]
    reg[:] = np.round(reg.astype(np.float32) * (1 - a) + 255.0 * a).astype(np.uint8)


def _finish(x, resampler, size, sharp, quality):
    """The identical pipeline for page and ground truth: downscale, optional
    unsharp mask, JPEG 4:2:0."""
    y = cv2.resize(x, size, interpolation=resampler)
    if sharp:
        yf = y.astype(np.float32)
        y = np.clip(yf + 0.5 * (yf - cv2.GaussianBlur(yf, (0, 0), 1.0)), 0, 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(y).save(buf, "JPEG", quality=int(quality), subsampling=2)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def make_sample(bg_path, tpl, rng, p_banner=P_BANNER):
    """One synthetic sample: dict(obs, gt, cov_true (unit coverage at 1x), d_true window info, params)."""
    img0 = np.asarray(Image.open(bg_path).convert("RGB"))
    Wp = int(rng.uniform(700, 1300))
    Hp = int(round(img0.shape[0] * Wp / img0.shape[1]))
    if Hp > 1700:                                                    # keep the runtime bounded
        img0 = img0[:int(round(img0.shape[0] * 1700 / Hp))]
        Hp = 1700
    img0 = np.ascontiguousarray(img0)
    interp0 = cv2.INTER_AREA if img0.shape[1] > 2 * Wp else cv2.INTER_CUBIC
    base2 = cv2.resize(img0, (2 * Wp, 2 * Hp), interpolation=interp0)
    alpha_t = tpl["alpha"].astype(np.float32)
    Th, Tw = alpha_t.shape
    # mark geometry at 1x
    frac = rng.uniform(0.35, 0.95)
    s = frac * Wp / Tw
    if s * Th > 0.9 * Hp:
        s = 0.9 * Hp / Th
    tw, th = s * Tw, s * Th
    off = bool(rng.random() < P_OFFPAGE)
    if off:
        side = int(rng.integers(0, 4))
        f = rng.uniform(0.15, 0.4)
        x = rng.uniform(0, max(1.0, Wp - tw)); y = rng.uniform(0, max(1.0, Hp - th))
        if side == 0:
            x = -f * tw
        elif side == 1:
            x = Wp - (1 - f) * tw
        elif side == 2:
            y = -f * th
        else:
            y = Hp - (1 - f) * th
    else:
        x = rng.uniform(0, max(1.0, Wp - tw)); y = rng.uniform(0, max(1.0, Hp - th))
    x, y = round(float(x) * 4) / 4, round(float(y) * 4) / 4
    # 2x window around the mark
    X2, Y2, s2 = 2 * x, 2 * y, 2 * s
    w2, h2 = ts.template_size(alpha_t, s2)
    wx0, wy0 = int(max(0, np.floor(X2) - 2)), int(max(0, np.floor(Y2) - 2))
    wx1, wy1 = int(min(2 * Wp, np.ceil(X2 + w2) + 2)), int(min(2 * Hp, np.ceil(Y2 + h2) + 2))
    ww, wh = wx1 - wx0, wy1 - wy0
    # banner under the mark area
    banner = bool(rng.random() < p_banner)
    clean2 = base2.copy()
    if banner:
        mx, my = rng.uniform(0.06, 0.15) * w2, rng.uniform(0.06, 0.15) * h2
        bb = (int(max(0, X2 - mx)), int(max(0, Y2 - my)), int(min(2 * Wp, X2 + w2 + mx)), int(min(2 * Hp, Y2 + h2 + my)))
        draw_banner(clean2, bb, rng)
    # ink and opacity
    ink = tpl["ink"].astype(np.float32).copy()
    sat = ink.max(2) - ink.min(2)
    grey = (sat < 0.08) & (alpha_t > 0.02)
    ink_variant = "template"
    if grey.any() and rng.random() < 0.5:
        ink[grey] = rng.uniform(55, 100) / 255.0
        ink_variant = "dark grey"
    ink = np.clip(ink + rng.uniform(-15, 15, 3).astype(np.float32)[None, None] / 255.0, 0, 1)
    mult = float(rng.uniform(0.7, 1.4))
    peak = float(tpl["opacity_peak"]) * mult
    prem = np.ascontiguousarray(np.dstack([alpha_t, alpha_t[..., None] * ink]).astype(np.float32))
    pt = ts.resized_template(prem, s2)
    M = np.float32([[1, 0, X2 - wx0], [0, 1, Y2 - wy0]])
    win = cv2.warpAffine(pt, M, (ww, wh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a_u2 = np.clip(win[..., 0], 0, 1)
    a2 = np.clip(peak * a_u2, 0, stamp_max())[..., None]
    pm = peak * win[..., 1:4]
    obs2 = clean2.copy()
    reg = clean2[wy0:wy1, wx0:wx1].astype(np.float32)
    obs2[wy0:wy1, wx0:wx1] = np.round(np.clip(reg * (1 - a2) + pm * 255.0 * (a2 / np.maximum(peak * a_u2[..., None], 1e-6)), 0, 255)).astype(np.uint8)
    # identical pipeline
    resamp_name = str(rng.choice(list(RESAMPLERS)))
    sharp = bool(rng.random() < P_SHARP)
    q = int(rng.choice([70, 85, 95]))
    obs = _finish(obs2, RESAMPLERS[resamp_name], (Wp, Hp), sharp, q)
    gt = _finish(clean2, RESAMPLERS[resamp_name], (Wp, Hp), sharp, q)
    # true unit coverage at 1x (and the window of its 2x map for the true signed distance)
    cov2 = np.zeros((2 * Hp, 2 * Wp), np.float32)
    cov2[wy0:wy1, wx0:wx1] = a_u2
    cov1 = cv2.resize(cov2, (Wp, Hp), interpolation=cv2.INTER_AREA)
    params = dict(page_w=Wp, page_h=Hp, width_frac=round(float(frac), 3), scale=round(float(s), 4), x=x, y=y,
                  offpage=off, banner=banner, resampler=resamp_name, sharpen=sharp, jpeg=q, opacity_mult=round(mult, 3),
                  ink=ink_variant, bg=os.path.basename(bg_path))
    return dict(obs=obs, gt=gt, cov1=cov1, cov2=cov2, params=params)


def stamp_max():
    return float(tad.stamp_fit.MAX_ALPHA)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def true_distance(cov2, win):
    """Signed distance (page px, + inside) to the true half-level contour over the
    1x window ``win`` = (x0, y0, w, h), from the 2x coverage map."""
    x0, y0, w, h = win
    c = np.ascontiguousarray(cov2[2 * y0:2 * (y0 + h), 2 * x0:2 * (x0 + w)])
    return tad.signed_distance(cv2.resize(c, (4 * w, 4 * h), interpolation=cv2.INTER_LINEAR))


def lum(a):
    return a.astype(np.float32) @ ts.LUMA


def sample_metrics(obs, gt, cleaned, alpha, sup, foot_mask, win, d_true):
    x0, y0, w, h = win
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    f = foot_mask[sl]
    def mae(a):
        return float(np.abs(a[sl].astype(np.float32) - gt[sl].astype(np.float32))[f].mean()) if f.any() else 0.0
    def ghost(a):
        return tad.ghost_score(lum(a[sl]) - lum(gt[sl]), d_true, np.ones(f.shape, bool), GHOST_CUT)
    changed = (cleaned != obs).any(2)
    far = ~(cv2.dilate(sup.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0)
    return dict(mae=mae(cleaned), mae0=mae(obs), ghost=ghost(cleaned), ghost0=ghost(obs),
                changed=int(changed.sum()), out_alpha=int((changed & ~(alpha > 1e-4)).sum()), out_foot=int((changed & far).sum()))


def located_iou(tpl, marks, cov1):
    """IoU of the best located footprint (coverage >= 0.3) with the true one."""
    H, W = cov1.shape
    true = cov1 >= 0.3
    best = 0.0
    for mk in marks:
        x0, y0, m = tsf._footprint(tpl["alpha"], mk["parts"][0], (H, W))
        if m is None:
            continue
        P = np.zeros((H, W), bool)
        P[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        inter = int((P & true).sum())
        best = max(best, inter / max(1, int((P | true).sum())))
    return best


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def summarise(rows, models, templates):
    """Median / p90 of ghost and MAE per model x template x {paper, banner} x resampler
    (located samples), plus the targets of the plan."""
    loc = [r for r in rows if r["located"]]
    S = dict(n_samples=len(rows), n_located=len(loc), groups={}, targets={})
    def grp(sel):
        out = {}
        for m in models:
            g = [r[f"{m}_ghost"] for r in sel]
            e = [r[f"{m}_mae"] for r in sel]
            out[m] = dict(n=len(sel), ghost_med=_q(g, 50), ghost_p90=_q(g, 90), mae_med=_q(e, 50), mae_p90=_q(e, 90))
        out["untouched"] = dict(n=len(sel), ghost_med=_q([r["ghost0"] for r in sel], 50), mae_med=_q([r["mae0"] for r in sel], 50))
        return out
    for t in list(templates) + ["ALL"]:
        ts_ = [r for r in loc if t == "ALL" or r["template"] == t]
        S["groups"][f"{t}|all"] = grp(ts_)
        for kind, sel in (("paper", [r for r in ts_ if not r["banner"]]), ("banner", [r for r in ts_ if r["banner"]])):
            S["groups"][f"{t}|{kind}"] = grp(sel)
        for rs in RESAMPLERS:
            S["groups"][f"{t}|{rs}"] = grp([r for r in ts_ if r["resampler"] == rs])
    if "adaptive" in models and "pixel" in models and loc:
        a, p = S["groups"]["ALL|all"]["adaptive"], S["groups"]["ALL|all"]["pixel"]
        b = S["groups"]["ALL|banner"]
        worse = sum(1 for r in loc if r["adaptive_mae"] > r["pixel_mae"] + 1.0)
        S["targets"] = dict(
            ghost_ratio=a["ghost_med"] / max(p["ghost_med"], 1e-9),
            mae_ratio=a["mae_med"] / max(p["mae_med"], 1e-9),
            banner_mae_ratio=(b["adaptive"]["mae_med"] / max(b["pixel"]["mae_med"], 1e-9)) if b["adaptive"]["n"] else None,
            frac_adaptive_worse_by_1=worse / len(loc),
            changed_outside_alpha=sum(r[f"{m}_out_alpha"] for r in rows for m in models),
            changed_outside_foot_9px=sum(r[f"{m}_out_foot"] for r in rows for m in models))
        T = S["targets"]
        T["met"] = dict(ghost=T["ghost_ratio"] <= 0.6, mae=T["mae_ratio"] <= 0.8,
                        banner=(T["banner_mae_ratio"] is not None and T["banner_mae_ratio"] <= 0.6),
                        worse=T["frac_adaptive_worse_by_1"] <= 0.05, outside=T["changed_outside_alpha"] == 0)
    return S


def summary_md(S, models):
    L = [f"## Removal benchmark: {S['n_samples']} samples, {S['n_located']} located", "",
         "Median / p90 over the LOCATED samples. ghost = leftover rim vs the ground truth (grey levels), mae = mean abs error inside the dilated true footprint (grey levels).", "",
         "| group | n | " + " | ".join(f"{m} ghost med/p90 | {m} MAE med/p90" for m in models) + " | untouched ghost/MAE |",
         "|---|---|" + "---|---|" * len(models) + "---|"]
    for k, g in S["groups"].items():
        n = g[models[0]]["n"]
        if n == 0:
            continue
        cells = " | ".join(f"{g[m]['ghost_med']:.2f} / {g[m]['ghost_p90']:.2f} | {g[m]['mae_med']:.2f} / {g[m]['mae_p90']:.2f}" for m in models)
        L.append(f"| {k} | {n} | {cells} | {g['untouched']['ghost_med']:.2f} / {g['untouched']['mae_med']:.2f} |")
    T = S.get("targets")
    if T:
        L += ["", "### Targets",
              f"- adaptive median ghost / pixel median ghost = {T['ghost_ratio']:.2f} (target <= 0.60): {'met' if T['met']['ghost'] else 'MISSED'}",
              f"- adaptive median MAE / pixel median MAE = {T['mae_ratio']:.2f} (target <= 0.80): {'met' if T['met']['mae'] else 'MISSED'}",
              f"- banner subset MAE ratio = {T['banner_mae_ratio'] if T['banner_mae_ratio'] is None else round(T['banner_mae_ratio'], 2)} (target <= 0.60): {'met' if T['met']['banner'] else 'MISSED'}",
              f"- adaptive MAE > pixel MAE + 1.0 on {100 * T['frac_adaptive_worse_by_1']:.1f}% of located samples (target <= 5%): {'met' if T['met']['worse'] else 'MISSED'}",
              f"- changed pixels outside the returned alpha: {T['changed_outside_alpha']} (target 0): {'met' if T['met']['outside'] else 'MISSED'}",
              f"- changed pixels > 9 px outside the true footprint (all models): {T['changed_outside_foot_9px']}"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _inside_repo(p):
    p = os.path.normcase(os.path.abspath(p))
    r = os.path.normcase(str(REPO))
    return p == r or p.startswith(r + os.sep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backgrounds", default="wm_backgrounds_v2", help="folder with background pages (default wm_backgrounds_v2)")
    ap.add_argument("--out", required=True, help="output folder; must be OUTSIDE the repository")
    ap.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES,
                    help="built-in names, library names or png paths (default: wide and stacked built-ins)")
    ap.add_argument("--library", default=None, help="library folder for library template names")
    ap.add_argument("--n", type=int, default=40, help="samples per template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--examples", type=int, default=4, help="example strips saved per template (plus the worst adaptive ones)")
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--p-banner", type=float, default=P_BANNER, help="probability of a coloured banner under the mark")
    ap.add_argument("--oracle", action="store_true",
                    help="diagnostic: remove at the TRUE pose instead of locating (isolates the removal from the locator)")
    a = ap.parse_args()

    out = tl.resolve_path(a.out)
    if _inside_repo(out):
        sys.exit(f"--out must be outside the repository ({REPO}); got {out}")
    bgd = tl.resolve_path(a.backgrounds)
    if not os.path.isdir(bgd):
        sys.exit(f"backgrounds folder not found: {bgd}")
    os.makedirs(out, exist_ok=True)
    models = list(a.models)
    bgs = light_backgrounds(bgd)
    if not bgs:
        sys.exit(f"no light backgrounds (mean luminance >= 150) in {bgd}")
    print(f"{len(bgs)} light backgrounds in {bgd}", file=sys.stderr, flush=True)

    rows, examples = [], []
    t_all = time.time()
    names = []
    for ti, name in enumerate(a.templates):
        tpl = tl.load_template(name, a.library)
        label = tl.template_label(tpl)
        names.append(label)
        for i in range(a.n):
            rng = np.random.default_rng([a.seed, ti, i])
            bg = bgs[int(rng.integers(0, len(bgs)))]
            smp = make_sample(bg, tpl, rng, a.p_banner)
            obs, gt, cov1 = smp["obs"], smp["gt"], smp["cov1"]
            H, W = obs.shape[:2]
            if a.oracle:
                key = tl.register_with_stamp_fit(tpl)
                pr = smp["params"]
                marks = [dict(kind=f"template:{label}", parts=[dict(name=key, scale=pr["scale"], x=pr["x"], y=pr["y"], sigma=0.0)], tpl=tpl)]
            else:
                marks, info = tsf.locate_marks(obs, [name], a.library, min_score=a.min_score)
            iou = located_iou(tpl, marks, cov1)
            ok = iou >= 0.5
            sup = cov1 > 0.02
            foot = cv2.dilate(sup.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            ys, xs = np.nonzero(sup)
            row = dict(template=label, i=i, located=bool(ok), iou=round(float(iou), 3), n_marks=len(marks), **smp["params"])
            strip = {}
            if sup.any():
                wx0, wy0 = max(0, int(xs.min()) - 10), max(0, int(ys.min()) - 10)
                win = (wx0, wy0, min(W, int(xs.max()) + 11) - wx0, min(H, int(ys.max()) + 11) - wy0)
                d_true = true_distance(smp["cov2"], win)
            else:
                win, d_true = None, None
            for m in models:
                if ok and win is not None:
                    cleaned, alpha, rm = tsf.remove_marks(obs, tsf.copy_marks(marks), m)
                    row[f"{m}_secs"] = round(sum(r.get("seconds", 0.0) for r in rm), 2) if m == "adaptive" else None
                    row[f"{m}_reverted"] = any(r.get("reverted") or r.get("reason") for r in rm) if m == "adaptive" else None
                else:
                    cleaned, alpha = obs, np.zeros((H, W), np.float32)
                    row[f"{m}_secs"], row[f"{m}_reverted"] = None, None
                if win is not None:
                    met = sample_metrics(obs, gt, cleaned, alpha, sup, foot, win, d_true)
                else:
                    met = dict(mae=0.0, mae0=0.0, ghost=0.0, ghost0=0.0, changed=0, out_alpha=0, out_foot=0)
                for k, v in met.items():
                    if k in ("mae0", "ghost0"):
                        row[k] = round(v, 3)
                    else:
                        row[f"{m}_{k}"] = round(v, 3) if isinstance(v, float) else v
                strip[m] = cleaned
            rows.append(row)
            print(f"[{label} {i + 1}/{a.n}] {smp['params']['resampler']:7s} q{smp['params']['jpeg']} {'banner' if smp['params']['banner'] else 'paper '} "
                  f"{'off ' if smp['params']['offpage'] else ''}located={ok} iou={iou:.2f} "
                  + " ".join(f"{m}: ghost {row[f'{m}_ghost']:.2f} mae {row[f'{m}_mae']:.2f}" for m in models), file=sys.stderr, flush=True)
            worse = ("adaptive" in models and "pixel" in models and ok and row["adaptive_mae"] > row["pixel_mae"] + 1.0)
            if win is not None and (i < a.examples or (worse and sum(1 for e in examples if e[0] == "worse") < a.examples)):
                x0, y0, w, h = win
                sl = (slice(y0, y0 + h), slice(x0, x0 + w))
                strip_img = np.concatenate([obs[sl], gt[sl]] + [strip[m][sl] for m in models], 1)
                examples.append(("worse" if worse and i >= a.examples else "first", f"{label.replace(' ', '_')}_{i:03d}", strip_img))
    secs = time.time() - t_all
    S = summarise(rows, models, names)
    S["seconds"] = round(secs, 1)
    S["args"] = dict(vars(a), out=out)
    with open(os.path.join(out, "samples.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = list(rows[0].keys())
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=2, default=float)
    md = summary_md(S, models)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    ex_dir = os.path.join(out, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    for kind, nm, im in examples:
        Image.fromarray(im).save(os.path.join(ex_dir, f"{kind}_{nm}.png"))
    print(md)
    print(f"\n{secs / 60:.1f} min; samples.csv, summary.json, summary.md and examples/ in {out}")


if __name__ == "__main__":
    main()
