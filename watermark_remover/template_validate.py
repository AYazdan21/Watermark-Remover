"""Validate a template on any folder of pages (AriaTender, ETENDER, ...).

Three independent checks, all model-free:

1. **Reference comparison** (optional): where does a known-good reference
   (e.g. ``assets/stamps/ariatender_wide.png`` for an AriaTender-built
   template) sit on the built template, and how well do the two coverage maps
   agree (Pearson correlation, soft / binary IoU, mean |delta|)?
2. **Removal over the folder**: Method 5 with only this template on every page,
   with the chosen removal model (``removal="pixel"``: per-pixel colour, the
   default; ``"region"``: Stamp Fit's per-region fit).
   ``change_before`` is the leftover mark contrast along the strokes
   (``stamp_fit._colour_change``) on the page, ``change_after`` the same on the
   cleaned page at the same fitted parts -- no ground truth needed.
3. **Clean targets** (optional): when ``<stem>_clean.png`` / ``<stem>.png``
   exist in a folder, mean absolute error and PSNR inside the dilated
   footprint and on the whole page, before vs after.

Outputs go to ``out_dir`` (default ``outputs/template_validation/<template>_<time>/``,
git-ignored); nothing is ever written into the pages folder.
"""

import csv
import json
import os
import re
import time

import cv2
import numpy as np
from PIL import Image

from . import stamp_fit
from . import template_library as tl
from . import template_signal as ts
from .config import BASE_DIR
from .template_builder import list_pages, read_rgb
from .template_stamp_fit import clean_document_template, footprint_overlay

CSV_FIELDS = ["file", "accepted", "score", "x", "y", "scale", "width_px", "change_before", "change_after",
              "control", "changed_fraction", "strengths", "inks", "changed_px", "time_s"]


def default_out_dir(template):
    stem = re.sub(r"[^A-Za-z0-9_.\-]+", "_", os.path.splitext(os.path.basename(str(template).rstrip("/\\")))[0])
    return os.path.join(BASE_DIR, "outputs", "template_validation", f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}")


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------

def compare_to_reference(built, ref, out_png=None):
    """``built`` and ``ref`` are template dicts (``load_template``). Finds the
    scale + translation of the reference onto the built template's coverage
    map and reports agreement. Returns dict(pose, pearson, soft_iou,
    binary_iou, mean_abs_delta, ncc)."""
    A = built["alpha"].astype(np.float32)
    R = ref["alpha"].astype(np.float32)
    H, W = A.shape
    s0 = W / R.shape[1]
    # pad so a reference larger than the built frame (or only partly inside it) can still match
    scales = s0 * np.geomspace(0.35, 2.8, 40)
    cands = ts.ncc_match(A, R, scales, per_scale=2, n_best=3, max_side=600, pad_frac=0.5)
    if not cands:
        return None
    best = None
    for c in cands:
        pose = dict(scale=c["scale"], x=c["x"], y=c["y"])
        pose, _ = ts.refine_pose(A, R, pose, radius=6, scale_span=0.05, scale_step=0.01, subpixel=False)
        pose, score = ts.refine_pose(A, R, pose, radius=2, scale_span=0.01, scale_step=0.0025)
        if best is None or score > best[1]:
            best = (pose, score)
    pose, score = best
    tw, th = ts.template_size(R, pose["scale"])
    x0 = int(min(0, np.floor(pose["x"]))); y0 = int(min(0, np.floor(pose["y"])))
    x1 = int(max(W, np.ceil(pose["x"] + tw))); y1 = int(max(H, np.ceil(pose["y"] + th)))
    cw, ch = x1 - x0, y1 - y0
    refc = ts.render_template(R, pose["scale"], pose["x"], pose["y"], (x0, y0, cw, ch))
    bc = np.zeros((ch, cw), np.float32)
    bc[-y0:-y0 + H, -x0:-x0 + W] = A
    union = (refc > 0.02) | (bc > 0.02)
    a, b = refc[union].astype(np.float64), bc[union].astype(np.float64)
    pearson = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 1e-9 and b.std() > 1e-9 else 0.0
    soft = float(np.minimum(refc, bc).sum() / max(np.maximum(refc, bc).sum(), 1e-9))
    bi = (refc >= 0.5) & (bc >= 0.5)
    bu = (refc >= 0.5) | (bc >= 0.5)
    biou = float(bi.sum() / max(bu.sum(), 1))
    mad = float(np.abs(a - b).mean()) if a.size else 0.0
    if out_png:
        img = np.zeros((ch, cw, 3), np.uint8)
        img[..., 0] = np.clip(refc * 255, 0, 255).astype(np.uint8)
        img[..., 1] = np.clip(bc * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(out_png)
    return dict(pose=dict(pose, w=tw, h=th), ncc=float(score), pearson=pearson, soft_iou=soft,
                binary_iou=biou, mean_abs_delta=mad)


# ---------------------------------------------------------------------------
# Clean targets
# ---------------------------------------------------------------------------

def _find_clean(clean_dir, stem):
    for cand in (f"{stem}_clean.png", f"{stem}.png", f"{stem}_clean.jpg", f"{stem}.jpg"):
        p = os.path.join(clean_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def _err(a, b, mask=None):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is not None:
        if not mask.any():
            return None, None
        d = d[mask]
    mae = float(d.mean())
    mse = float((d * d).mean())
    psnr = float(10 * np.log10(255.0 ** 2 / mse)) if mse > 1e-9 else 99.0
    return mae, psnr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_template(template, pages_dir, out_dir=None, reference=None, clean_dir=None, max_pages=100,
                      min_score=0.30, progress=None, library_dir=None, removal="pixel"):
    """Runs the three checks; returns dict(ok, message, summary, rows, out_dir, csv,
    gallery (list of image paths, overlay/cleaned pairs), reference_image)."""
    prog = (lambda f, msg="": progress(f, desc=msg)) if progress else (lambda f, msg="": None)
    try:
        tpl = tl.load_template(template, library_dir)
    except Exception as e:
        return dict(ok=False, message=f"**Validation failed:** {e}", rows=[], gallery=[])
    label = tl.template_label(tpl)
    files = list_pages(pages_dir)[:max(1, int(max_pages))]
    if not files:
        return dict(ok=False, message=f"**Validation failed:** no images in `{pages_dir}`", rows=[], gallery=[])
    out_dir = tl.resolve_path(out_dir) or default_out_dir(template)
    pd = os.path.normcase(os.path.abspath(tl.resolve_path(pages_dir)))
    if os.path.normcase(os.path.abspath(out_dir)) == pd:
        return dict(ok=False, message="**Validation failed:** the output folder must not be the pages folder", rows=[], gallery=[])
    os.makedirs(out_dir, exist_ok=True)
    clean_dir = tl.resolve_path(clean_dir)
    if clean_dir and not os.path.isdir(clean_dir):
        return dict(ok=False, message=f"**Validation failed:** clean-targets folder `{clean_dir}` not found", rows=[], gallery=[])

    summary = dict(template=label, template_path=tpl["path"], pages_dir=tl.resolve_path(pages_dir),
                   out_dir=out_dir, n_pages=len(files), min_score=min_score, removal=removal)
    ref_res, ref_png = None, None
    if reference and str(reference) not in ("None", ""):
        prog(0.02, "reference comparison")
        try:
            ref = tl.load_template(reference, library_dir)
            ref_png = os.path.join(out_dir, "reference_compare.png")
            ref_res = compare_to_reference(tpl, ref, ref_png)
            if ref_res is None:
                ref_png = None
        except Exception as e:
            summary["reference_error"] = str(e)
            ref_png = None
        summary["reference"] = ref_res

    rows, gallery_acc, gallery_rej = [], [], []
    clean_acc = []
    t_run = time.time()
    for i, path in enumerate(files):
        prog(0.05 + 0.9 * i / len(files), f"page {i + 1}/{len(files)}")
        stem = os.path.splitext(os.path.basename(path))[0]
        img = read_rgb(path)
        if img is None:
            rows.append(dict(file=os.path.basename(path), accepted=False, time_s=0.0))
            continue
        t0 = time.time()
        cleaned, alpha, info, _ = clean_document_template(img, [tpl["path"]], library_dir, min_score=min_score,
                                                          removal=removal)
        dt = time.time() - t0
        acc = info["accepted"]
        row = dict(file=os.path.basename(path), accepted=bool(acc), time_s=round(dt, 3))
        if acc:
            a0 = acc[0]
            parts = info["marks"][0]["parts"]
            shape = img.shape[:2]
            before = stamp_fit._colour_change(img, parts, shape)[0]
            after = stamp_fit._colour_change(cleaned, parts, shape)[0]
            rm = info["removal"][0]["regions"]
            row.update(score=a0["score"], x=a0["x"], y=a0["y"], scale=a0["scale"], width_px=a0["width_px"],
                       change_before=round(before, 2), change_after=round(after, 2), control=a0["control"],
                       changed_fraction=a0["changed_fraction"],
                       strengths=";".join(f"{k}={v['strength']}" for k, v in rm.items()),
                       inks=";".join(f"{k}={tuple(v['ink'])}" for k, v in rm.items()),
                       changed_px=int((cleaned != img).any(2).sum()))
        else:
            rej = info["rejected"][0] if info["rejected"] else {}
            row.update(score=rej.get("score"), x=rej.get("x"), y=rej.get("y"), scale=rej.get("scale"),
                       width_px=rej.get("width_px"), control=rej.get("control"),
                       changed_fraction=rej.get("changed_fraction"), changed_px=0)
        if clean_dir:
            cp = _find_clean(clean_dir, stem)
            if cp is not None:
                tgt = read_rgb(cp)
                if tgt is not None and tgt.shape == img.shape:
                    foot = cv2.dilate((alpha > 1e-4).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                    ce = dict(foot_before=_err(img, tgt, foot[..., None].repeat(3, 2)),
                              foot_after=_err(cleaned, tgt, foot[..., None].repeat(3, 2)),
                              page_before=_err(img, tgt), page_after=_err(cleaned, tgt))
                    clean_acc.append(ce)
                    row.update(clean_foot_mae_before=ce["foot_before"][0], clean_foot_mae_after=ce["foot_after"][0],
                               clean_page_mae_before=ce["page_before"][0], clean_page_mae_after=ce["page_after"][0])
        rows.append(row)
        Image.fromarray(cleaned).save(os.path.join(out_dir, f"{stem}_cleaned.png"))
        ov = footprint_overlay(img, alpha) if acc else img
        Image.fromarray(ov).save(os.path.join(out_dir, f"{stem}_overlay.png"))
        diff = np.clip(np.abs(cleaned.astype(np.int16) - img.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        Image.fromarray(diff).save(os.path.join(out_dir, f"{stem}_diff.png"))
        pair = (os.path.join(out_dir, f"{stem}_overlay.png"), os.path.join(out_dir, f"{stem}_cleaned.png"))
        (gallery_acc if acc else gallery_rej).append(pair)
    total = time.time() - t_run

    acc_rows = [r for r in rows if r.get("accepted")]
    cb = [r["change_before"] for r in acc_rows if r.get("change_before") is not None]
    ca = [r["change_after"] for r in acc_rows if r.get("change_after") is not None]
    summary.update(n_accepted=len(acc_rows), n_rejected=len(rows) - len(acc_rows),
                   median_change_before=float(np.median(cb)) if cb else None,
                   median_change_after=float(np.median(ca)) if ca else None,
                   median_score=float(np.median([r["score"] for r in acc_rows])) if acc_rows else None,
                   time_per_page_s=round(total / max(1, len(rows)), 3))
    if clean_acc:
        def avg(k, j):
            v = [c[k][j] for c in clean_acc if c[k][0] is not None]
            return float(np.mean(v)) if v else None
        summary["clean_targets"] = dict(n=len(clean_acc), **{f"{k}_{m}": avg(k, j) for k in
                                        ("foot_before", "foot_after", "page_before", "page_after") for j, m in ((0, "mae"), (1, "psnr"))})
    csv_path = os.path.join(out_dir, "summary.csv")
    cols = CSV_FIELDS + [c for c in ("clean_foot_mae_before", "clean_foot_mae_after", "clean_page_mae_before",
                                      "clean_page_mae_after") if any(c in r for r in rows)]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    prog(1.0, "done")
    pairs = (gallery_acc + gallery_rej)[:12]
    gallery = [p for pr in pairs for p in pr]
    return dict(ok=True, message=_markdown(summary, rows), summary=summary, rows=rows, out_dir=out_dir,
                csv=csv_path, gallery=gallery, reference_image=ref_png)


def _markdown(s, rows):
    L = [f"### Validation of `{s['template']}` on {s['n_pages']} page(s)",
         f"- accepted **{s['n_accepted']}**, rejected **{s['n_rejected']}** (min score {s['min_score']:.2f}, "
         f"{'per-pixel colour' if s.get('removal', 'pixel') == 'pixel' else 'per-region (Stamp Fit)'} removal)"]
    if s.get("median_change_before") is not None:
        L.append(f"- leftover mark contrast along the strokes (median, grey levels): "
                 f"**{s['median_change_before']:.1f} -> {s['median_change_after']:.1f}**")
        L.append(f"- median fit score {s['median_score']:.2f}")
    L.append(f"- {s['time_per_page_s']:.1f} s per page")
    r = s.get("reference")
    if r:
        L.append(f"- vs reference: Pearson **{r['pearson']:.3f}**, soft IoU {r['soft_iou']:.3f}, binary IoU@0.5 "
                 f"{r['binary_iou']:.3f}, mean |delta| {r['mean_abs_delta']:.3f} (reference at scale {r['pose']['scale']:.3f})")
    elif s.get("reference_error"):
        L.append(f"- reference comparison failed: {s['reference_error']}")
    c = s.get("clean_targets")
    if c:
        L.append(f"- clean targets ({c['n']} matched), mean abs error / PSNR: footprint "
                 f"{c['foot_before_mae']:.2f}/{c['foot_before_psnr']:.1f} dB -> {c['foot_after_mae']:.2f}/{c['foot_after_psnr']:.1f} dB; "
                 f"whole page {c['page_before_mae']:.3f}/{c['page_before_psnr']:.1f} dB -> {c['page_after_mae']:.3f}/{c['page_after_psnr']:.1f} dB"
                 if c.get("foot_before_mae") is not None else f"- clean targets: {c['n']} matched, no footprint pixels")
    rej = [x for x in rows if not x.get("accepted")]
    if rej:
        L.append("\n**Rejected pages:** " + ", ".join(f"`{x['file']}`" for x in rej[:20]) + (" ..." if len(rej) > 20 else ""))
    L.append(f"\nOutputs: `{s['out_dir']}`")
    return "\n".join(L)
