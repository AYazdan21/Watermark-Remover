"""Repeatable measurement of the alpha-regression network (``watermark_
remover/alpha_net.py``), built from the ad hoc diagnostic scripts
(``diagnose.py`` / ``noise_test.py`` / ``scale_test.py``, scratchpad
``alpha_diag/``) that first found the checkpoint's weaknesses: it removes
only ~62-65% of the mark even on its own synthetic training distribution
(under-predicts alpha), is sensitive to page scale/sharpness on real crisp
screenshots, and -- without an ink guard -- damages real dark text inside
detected boxes on some real pages. This script turns those one-off
investigations into a CLI anyone can re-run against any checkpoint.

Two measurement tracks:

1. **Real pages** (``wm_testset/images/*``) -- NO ground truth exists for
   these (see ``wm_testset/README.md``: it is a placeholder directory with
   no labels checked in for this repo state). Everything reported here is
   inferred: M4 boxes from the detector, "mark pixels" from M3's finetuned
   segmenter mask intersected with those boxes and a page-Otsu ink
   exclusion, background from ``doc_detect.estimate_background_map``. Take
   real-page numbers as approximate, self-consistent trend indicators, not
   absolute ground truth -- this is stated again in ``summary.md``.
2. **Synthetic with exact ground truth** -- the calibrated real stamps
   exported by ``scripts/alpha_net/export_stamps.py`` (``assets/stamps/``)
   composited onto held-out ``wm_backgrounds_v2/`` pages with a known alpha
   map and a known clean target, so ``removed_pct``/``alpha_true`` vs.
   ``alpha_pred`` are exact, not inferred.

Removal for the real-page track runs THROUGH THE APP PATH
(``doc_detect.apply_box_strategy(..., removal_strategy=STRATEGY_ALPHA_NET)``)
so the ink guard (``doc_detect.ALPHA_NET_INK_GUARD``) is exercised exactly as
the app would exercise it -- not a reimplementation of the recovery math.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WM_DATASET_DIR = REPO_ROOT / "scripts" / "wm_dataset"
sys.path.insert(0, str(WM_DATASET_DIR))
sys.path.insert(0, str(REPO_ROOT))

import compositor  # noqa: E402 (needs WM_DATASET_DIR on sys.path)
from watermark_remover import alpha_net, detector, doc_detect, segmenter  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114])


def lum(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64) @ LUMA


# ---------------------------------------------------------------------------
# Real-page mark/ink/background classification -- exact logic of
# scale_test.py, cleaned up (see that script for the original ad hoc form).
# ---------------------------------------------------------------------------

def classify_real_page(img: np.ndarray, boxes_conf: float = 0.25):
    """Returns a dict with detection/segmentation-derived quantities for one
    real page, or {"skip_reason": str} if it can't be scored.

    Mirrors scale_test.py exactly:
      - union: boolean union of M4's detected boxes.
      - ink_px: union & (gray < page_gray_otsu - 15) -- real dark content,
        regardless of mark polarity (scale_test.py uses this one fixed
        formula for classifying "ink" when carving out mark pixels, distinct
        from doc_detect.py's polarity-aware ink guard used for REMOVAL).
      - is_bright_mark: True when the box-averaged background is dark (< 140
        luma) -- a bright mark on a dark banner, so darkening is signed the
        other way.
      - d: signed darkening (background minus observed, or observed minus
        background for a bright mark), i.e. how much the mark visibly darkens
        (or, for bright-on-dark, lightens) the page at each pixel.
      - M: mark pixels = M3 finetuned segmenter mask & union & ~ink_px &
        (d > 3).
    """
    seg_mask, _ = segmenter.detect_watermark_masks(img, conf=boxes_conf, model_choice="Finetuned (AriaTender)")
    inst, _ = detector.detect_watermark_boxes(img, conf=boxes_conf)
    if not inst:
        return {"skip_reason": "no detection boxes", "n_boxes": 0}

    bg = doc_detect.estimate_background_map(img, inst)
    h, w = img.shape[:2]
    union = np.zeros((h, w), dtype=bool)
    for i in inst:
        x1, y1, x2, y2 = i["box"]
        union[y1:y2, x1:x2] = True

    g = lum(img)
    otsu, _ = cv2.threshold(g.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_px = union & (g < otsu - 15)

    bgl = lum(bg)
    is_bright_mark = bool(np.median(bgl[union]) < 140.0) if union.any() else False
    d = (g - bgl) if is_bright_mark else (bgl - g)

    M = (seg_mask > 0) & union & (~ink_px) & (d > 3)
    if M.sum() < 200:
        return {"skip_reason": f"too few mark pixels ({int(M.sum())})", "n_boxes": len(inst)}

    return {
        "skip_reason": None,
        "instances": inst,
        "n_boxes": len(inst),
        "union": union,
        "ink_px": ink_px,
        "bg": bg,
        "bgl": bgl,
        "g": g,
        "d": d,
        "M": M,
        "is_bright_mark": is_bright_mark,
        "seg_mask": seg_mask,
    }


def score_real_page(img: np.ndarray, cls: dict, cleaned: np.ndarray,
                     model, device) -> dict:
    """Computes the four real-page metrics from the task brief given the
    classification (classify_real_page) and the removal already produced by
    doc_detect.apply_box_strategy (`cleaned`, full page, same shape as img).
    """
    union, ink_px, M = cls["union"], cls["ink_px"], cls["M"]
    g, bgl, d = cls["g"], cls["bgl"], cls["d"]
    is_bright = cls["is_bright_mark"]
    seg_mask = cls["seg_mask"]

    gr = lum(cleaned)
    d_after = (gr - bgl) if is_bright else (bgl - gr)
    removed_pct = 100.0 * (1.0 - np.median(d_after[M]) / np.median(d[M]))

    ink_kept_pct = 100.0 * np.mean(np.abs(gr - g)[ink_px] <= 10.0) if ink_px.any() else float("nan")

    clean_px = union & (~(seg_mask > 0)) & (~ink_px)
    clean_change = float(np.abs(gr - g)[clean_px].mean()) if clean_px.any() else float("nan")

    # Raw network alpha, for reporting only (not what removal used pixel-for-
    # pixel -- apply_box_strategy runs the net over a padded box-union crop;
    # this is the same "call predict_image over the whole page" scale_test.py/
    # diagnose.py both do, close enough for a summary statistic).
    a_pred, _ink, _rec = alpha_net.predict_image(model, img, device=device)
    alpha_pred_median = float(np.median(a_pred[M]))

    return {
        "removed_pct": float(removed_pct),
        "ink_kept_pct": float(ink_kept_pct),
        "clean_change": clean_change,
        "alpha_pred_median": alpha_pred_median,
        "polarity": "bright" if is_bright else "dark",
        "n_mark_px": int(M.sum()),
        "n_ink_px": int(ink_px.sum()),
    }


# ---------------------------------------------------------------------------
# Synthetic ground-truth generation: composite a calibrated real stamp
# (assets/stamps/) onto a held-out wm_backgrounds_v2 page. Reuses
# compositor's own rotate/scale/over-blend primitives (the exact math the
# training/eval data was built with) rather than reimplementing them, but
# with THIS script's own placement sampling (width fraction / rotation /
# position ranges from the task brief, not compositor's registry ranges --
# see module docstring).
# ---------------------------------------------------------------------------

STAMP_FILES = {
    "ariatender_stacked": "ariatender_stacked.png",
    "ariatender_wide": "ariatender_wide.png",
}


def _load_stamp(assets_dir: Path, mark_id: str) -> Image.Image:
    path = assets_dir / STAMP_FILES[mark_id]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/alpha_net/export_stamps.py` first "
            f"to generate assets/stamps/."
        )
    return Image.open(path).convert("RGBA")


def make_synthetic_sample(bg_img: Image.Image, stamp_rgba: Image.Image, rng: np.random.Generator,
                           jpeg_q90_prob: float = 0.5):
    """Composites `stamp_rgba` once onto `bg_img` (single-placement, 0.4-0.9
    of page width, rotation within +/-30deg, opacity_mult ~ U(0.5, 1.5)),
    normal alpha-over blend, native resolution, no noise. Returns
    (obs_uint8, clean_uint8, alpha_map float32 HxW, meta dict).
    """
    bg = bg_img.convert("RGB")
    w, h = bg.size

    frac = float(rng.uniform(0.4, 0.9))
    angle = float(rng.uniform(-30.0, 30.0))
    opacity_mult = float(rng.uniform(0.5, 1.5))

    scale = (frac * w) / stamp_rgba.width
    tile = compositor._rotate_scale_stamp(stamp_rgba, scale, angle, opacity_mult, tint_shift=0)
    tw, th = tile.size

    cx = rng.uniform(0.5 - 0.1, 0.5 + 0.1) * w
    cy = rng.uniform(0.5 - 0.1, 0.5 + 0.1) * h
    x = int(round(cx - tw / 2.0))
    y = int(round(cy - th / 2.0))

    canvas_rgb = np.asarray(bg, dtype=np.float32) / 255.0
    canvas_a = np.zeros((h, w), dtype=np.float32)
    tile_arr = np.asarray(tile, dtype=np.float32) / 255.0
    compositor._paste_over(canvas_rgb, canvas_a, tile_arr, x, y)

    obs = np.clip(canvas_rgb * 255.0, 0, 255).astype(np.uint8)
    clean = np.asarray(bg, dtype=np.uint8)

    jpeg_applied = bool(rng.random() < jpeg_q90_prob)
    if jpeg_applied:
        buf = io.BytesIO()
        Image.fromarray(obs).save(buf, "JPEG", quality=90)
        buf.seek(0)
        obs = np.array(Image.open(buf).convert("RGB"))

    meta = dict(frac=frac, angle_deg=angle, opacity_mult=opacity_mult, jpeg_q90=jpeg_applied)
    return obs, clean, canvas_a.astype(np.float32), meta


# Ink saturation threshold splitting mark pixels into "coloured" vs "grey"
# ink for the breakdown below -- matches `INK_MIN_ALPHA`'s sibling constant
# in scripts/alpha_net/alpha_train.py (`alpha_loss`/`evaluate`'s coloured-
# subset definition), so a training run's coloured-subset metrics and this
# benchmark's coloured-subset metrics are measuring the same thing.
INK_SAT_COLORED_THRESHOLD = 0.06
_MIN_GROUP_PX = 20


def _ink_saturation(rgb) -> np.ndarray:
    """max(channel) - min(channel) over the last axis; 0 for perfect grey,
    up to 1 for a fully saturated primary/secondary colour."""
    return rgb.max(axis=-1) - rgb.min(axis=-1)


def score_synthetic_sample(obs, clean, alpha_map, model, device):
    a_pred, ink_pred, rec = alpha_net.predict_image(model, obs, device=device)

    M = (alpha_map > 0.05) & (alpha_map <= 0.9)
    if M.sum() < 50:
        return None

    diff = np.abs(rec.astype(np.float64) - clean.astype(np.float64)).mean(axis=2)
    signal = np.abs(obs.astype(np.float64) - clean.astype(np.float64)).mean(axis=2)
    removed_pct = 100.0 * (1.0 - diff[M].mean() / signal[M].mean())

    alpha_true_median = float(np.median(alpha_map[M]))
    alpha_pred_median = float(np.median(a_pred[M]))

    zero_px = alpha_map <= 1e-6
    clean_change = float(np.abs(lum(rec) - lum(clean))[zero_px].mean()) if zero_px.any() else float("nan")

    # --- coloured-vs-grey ink breakdown -------------------------------
    # True ink, inverted from the exact compositing equation `obs = a*ink +
    # (1-a)*clean` -- the same closed-form target alpha_train.py's L_ink
    # and coloured-subset eval metrics use (see that module's alpha_loss
    # docstring): exact when `obs` is the pre-JPEG composite, ~5.7/255 MAE
    # under q85 JPEG for a >= 0.10. `alpha_map` is clamped away from 0 in
    # the denominator so this stays finite even outside the stamp's
    # footprint; only pixels in `M` (alpha in (0.05, 0.9]) are ever used
    # below, so that noise floor doesn't leak into the reported numbers.
    obs_f = obs.astype(np.float64) / 255.0
    clean_f = clean.astype(np.float64) / 255.0
    alpha_f = alpha_map.astype(np.float64)
    a_safe = np.clip(alpha_f, 0.10, None)[..., None]
    ink_true = np.clip((obs_f - (1.0 - alpha_f[..., None]) * clean_f) / a_safe, 0.0, 1.0)
    sat_true = _ink_saturation(ink_true)
    sat_pred = _ink_saturation(ink_pred.astype(np.float64))

    colored_mask = M & (sat_true > INK_SAT_COLORED_THRESHOLD)
    grey_mask = M & ~colored_mask

    def _group(mask):
        if mask.sum() < _MIN_GROUP_PX:
            return {"removed_pct": float("nan"), "sat_pred": float("nan"), "sat_true": float("nan")}
        return {
            "removed_pct": float(100.0 * (1.0 - diff[mask].mean() / signal[mask].mean())),
            "sat_pred": float(sat_pred[mask].mean()),
            "sat_true": float(sat_true[mask].mean()),
        }

    colored_g = _group(colored_mask)
    grey_g = _group(grey_mask)

    return {
        "removed_pct": float(removed_pct),
        "alpha_true_median": alpha_true_median,
        "alpha_pred_median": alpha_pred_median,
        "clean_change": clean_change,
        "n_mark_px": int(M.sum()),
        "removed_pct_colored": colored_g["removed_pct"],
        "ink_sat_pred_colored": colored_g["sat_pred"],
        "ink_sat_true_colored": colored_g["sat_true"],
        "n_colored_px": int(colored_mask.sum()),
        "removed_pct_grey": grey_g["removed_pct"],
        "ink_sat_pred_grey": grey_g["sat_pred"],
        "ink_sat_true_grey": grey_g["sat_true"],
        "n_grey_px": int(grey_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Panels: before/after zoom crops for qualitative inspection.
# ---------------------------------------------------------------------------

def save_panel(out_path: Path, orig: np.ndarray, cleaned: np.ndarray, box, pad: int = 24):
    h, w = orig.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    before = orig[y1:y2, x1:x2]
    after = cleaned[y1:y2, x1:x2]
    if before.size == 0:
        return
    gap = np.full((before.shape[0], 8, 3), 255, dtype=np.uint8)
    panel = np.concatenate([before, gap, after], axis=1)
    Image.fromarray(panel).save(out_path)


# ---------------------------------------------------------------------------
# Strategy comparison (--compare-strategies)
# ---------------------------------------------------------------------------

STRATEGIES_FOR_COMPARE = [
    doc_detect.STRATEGY_THRESHOLD_FILL,
    doc_detect.STRATEGY_BOUNDED_SUBTRACTIVE,
    doc_detect.STRATEGY_ALPHA_NET,
]


def run_strategy(img: np.ndarray, cls: dict, strategy: str, model, device):
    inst = cls["instances"]
    background = cls["bg"]
    cleaned, page_info, per_box_info = doc_detect.apply_box_strategy(
        img, inst, strategy, background=background,
    )
    g, bgl, d = cls["g"], cls["bgl"], cls["d"]
    M, ink_px, union = cls["M"], cls["ink_px"], cls["union"]
    is_bright = cls["is_bright_mark"]
    gr = lum(cleaned)
    d_after = (gr - bgl) if is_bright else (bgl - gr)
    removed_pct = 100.0 * (1.0 - np.median(d_after[M]) / np.median(d[M]))
    ink_kept_pct = 100.0 * np.mean(np.abs(gr - g)[ink_px] <= 10.0) if ink_px.any() else float("nan")
    seg_mask = cls["seg_mask"]
    clean_px = union & (~(seg_mask > 0)) & (~ink_px)
    clean_change = float(np.abs(gr - g)[clean_px].mean()) if clean_px.any() else float("nan")
    return cleaned, {
        "strategy": strategy,
        "removed_pct": float(removed_pct),
        "ink_kept_pct": float(ink_kept_pct),
        "clean_change": clean_change,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _stat(values, fn):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float("nan")
    return fn(vals)


def summarize(rows, keys):
    out = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        out[k] = {
            "median": _stat(vals, statistics.median),
            "mean": _stat(vals, statistics.mean),
            "n": sum(1 for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))),
        }
    return out


def fmt_summary_table(title, summary, keys):
    lines = [f"### {title}", "", "| metric | median | mean | n |", "|---|---|---|---|"]
    for k in keys:
        s = summary[k]
        lines.append(f"| {k} | {s['median']:.3f} | {s['mean']:.3f} | {s['n']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=doc_detect.ALPHA_NET_WEIGHTS)
    ap.add_argument("--pages", nargs="*", default=None,
                     help="filenames (basenames) inside wm_testset/images to score; default: all of them")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthetic", type=int, default=40)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmark_out"))
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--no-ink-guard", action="store_true")
    ap.add_argument("--compare-strategies", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    doc_detect.ALPHA_NET_WEIGHTS = args.weights
    doc_detect.ALPHA_NET_INK_GUARD = not args.no_ink_guard
    model = doc_detect.get_alpha_net_model(args.weights)
    print(f"Loaded {args.weights} on {device}. ALPHA_NET_INK_GUARD={doc_detect.ALPHA_NET_INK_GUARD}")

    t_start = time.time()

    # ---- Real pages ----
    images_dir = REPO_ROOT / "wm_testset" / "images"
    if args.pages:
        page_paths = [images_dir / p for p in args.pages]
    else:
        page_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg")) + \
            sorted(images_dir.glob("*.png"))
        page_paths = sorted(set(page_paths))
    if args.limit:
        page_paths = page_paths[: args.limit]

    page_rows = []
    panel_count = 0
    for p in page_paths:
        t0 = time.time()
        img = np.array(Image.open(p).convert("RGB"))
        cls = classify_real_page(img)
        row = {"page": p.name, "w": img.shape[1], "h": img.shape[0], "n_boxes": cls.get("n_boxes", 0)}
        if cls.get("skip_reason"):
            row["skipped"] = cls["skip_reason"]
            row["seconds"] = time.time() - t0
            page_rows.append(row)
            print(f"  SKIP {p.name}: {cls['skip_reason']}")
            continue

        cleaned, page_info, per_box_info = doc_detect.apply_box_strategy(
            img, cls["instances"], doc_detect.STRATEGY_ALPHA_NET, background=cls["bg"],
        )
        metrics = score_real_page(img, cls, cleaned, model, device)
        row.update(metrics)
        row["skipped"] = ""
        row["seconds"] = time.time() - t0
        page_rows.append(row)
        print(f"  {p.name}: removed {metrics['removed_pct']:5.1f}% | ink_kept {metrics['ink_kept_pct']:5.1f}% | "
              f"clean_change {metrics['clean_change']:.2f} | alpha_pred {metrics['alpha_pred_median']:.3f} | "
              f"{row['seconds']:.2f}s")

        if panel_count < args.panels and cls["instances"]:
            box = max(cls["instances"], key=lambda i: (i["box"][2] - i["box"][0]) * (i["box"][3] - i["box"][1]))["box"]
            save_panel(panels_dir / f"{p.stem}_panel.png", img, cleaned, box)
            panel_count += 1

    pages_csv = out_dir / "pages.csv"
    all_keys = sorted({k for r in page_rows for k in r.keys()})
    with open(pages_csv, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=all_keys)
        w_.writeheader()
        for r in page_rows:
            w_.writerow(r)

    # ---- Synthetic ----
    bg_dir = REPO_ROOT / "wm_backgrounds_v2"
    bg_paths = sorted(bg_dir.glob("*.png")) + sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.jpeg"))
    assets_dir = REPO_ROOT / "assets" / "stamps"
    mark_ids = list(STAMP_FILES.keys())
    stamps = {m: _load_stamp(assets_dir, m) for m in mark_ids}

    rng = np.random.default_rng(args.seed)
    synth_rows = []
    n_synth = args.synthetic
    if bg_paths and n_synth:
        pool_size = min(n_synth * 3, len(bg_paths))
        bg_choices = rng.choice(len(bg_paths), size=pool_size, replace=(pool_size < n_synth))
        made = 0
        idx = 0
        while made < n_synth and idx < len(bg_choices):
            bg_path = bg_paths[bg_choices[idx]]
            idx += 1
            mark_id = mark_ids[made % len(mark_ids)]
            bg_img = Image.open(bg_path)
            attempt = 0
            result = None
            while attempt < 5:
                obs, clean, alpha_map, meta = make_synthetic_sample(bg_img, stamps[mark_id], rng)
                m = score_synthetic_sample(obs, clean, alpha_map, model, device)
                if m is not None:
                    result = (m, meta)
                    break
                attempt += 1
            if result is None:
                continue
            m, meta = result
            row = {"bg": bg_path.name, "mark_id": mark_id, **meta, **m}
            synth_rows.append(row)
            made += 1

    synthetic_csv = out_dir / "synthetic.csv"
    if synth_rows:
        s_keys = sorted({k for r in synth_rows for k in r.keys()})
        with open(synthetic_csv, "w", newline="", encoding="utf-8") as f:
            w_ = csv.DictWriter(f, fieldnames=s_keys)
            w_.writeheader()
            for r in synth_rows:
                w_.writerow(r)

    # ---- compare-strategies ----
    compare_lines = []
    if args.compare_strategies:
        compare_rows = []
        cmp_pages = page_paths if not args.pages else page_paths
        for p in cmp_pages:
            img = np.array(Image.open(p).convert("RGB"))
            cls = classify_real_page(img)
            if cls.get("skip_reason"):
                continue
            for strategy in STRATEGIES_FOR_COMPARE:
                _cleaned, m = run_strategy(img, cls, strategy, model, device)
                m["page"] = p.name
                compare_rows.append(m)
        if compare_rows:
            compare_csv = out_dir / "compare_strategies.csv"
            c_keys = ["page", "strategy", "removed_pct", "ink_kept_pct", "clean_change"]
            with open(compare_csv, "w", newline="", encoding="utf-8") as f:
                w_ = csv.DictWriter(f, fieldnames=c_keys)
                w_.writeheader()
                for r in compare_rows:
                    w_.writerow(r)
            compare_lines.append("### Strategy comparison (per page)")
            compare_lines.append("")
            compare_lines.append("| page | strategy | removed_pct | ink_kept_pct | clean_change |")
            compare_lines.append("|---|---|---|---|---|")
            for r in compare_rows:
                compare_lines.append(
                    f"| {r['page']} | {r['strategy']} | {r['removed_pct']:.1f} | {r['ink_kept_pct']:.1f} | "
                    f"{r['clean_change']:.2f} |"
                )

    total_seconds = time.time() - t_start

    # ---- Summary ----
    scored_pages = [r for r in page_rows if not r.get("skipped")]
    skipped_pages = [r for r in page_rows if r.get("skipped")]
    real_keys = ["removed_pct", "ink_kept_pct", "clean_change", "alpha_pred_median"]
    real_summary = summarize(scored_pages, real_keys)
    synth_keys = ["removed_pct", "alpha_true_median", "alpha_pred_median", "clean_change"]
    synth_summary = summarize(synth_rows, synth_keys)
    color_keys = ["removed_pct_colored", "ink_sat_pred_colored", "ink_sat_true_colored",
                  "removed_pct_grey", "ink_sat_pred_grey", "ink_sat_true_grey"]
    color_summary = summarize(synth_rows, color_keys) if synth_rows else None

    md = []
    md.append("# Alpha-network benchmark")
    md.append("")
    md.append(f"Weights: `{args.weights}`  \nInk guard: `{doc_detect.ALPHA_NET_INK_GUARD}`  \n"
              f"Real pages scored: {len(scored_pages)} (skipped: {len(skipped_pages)})  \n"
              f"Synthetic samples: {len(synth_rows)}  \nTotal runtime: {total_seconds:.1f}s")
    md.append("")
    md.append("**Real pages have NO ground truth** (see `wm_testset/README.md`) -- `removed_pct`/`ink_kept_pct`/"
               "`clean_change`/`alpha_pred_median` are all inferred from M4 detection boxes, M3's finetuned "
               "segmenter mask, and a page-level Otsu ink split, not from labelled watermark opacity. Treat these "
               "as trend indicators, not ground truth. Synthetic numbers ARE exact (known alpha map + clean "
               "target from compositing the calibrated `assets/stamps/` marks).")
    md.append("")
    md.append(fmt_summary_table("Real pages (inferred)", real_summary, real_keys))
    md.append("")
    md.append(fmt_summary_table("Synthetic (exact ground truth)", synth_summary, synth_keys))
    md.append("")
    if color_summary is not None:
        md.append(f"### Coloured vs grey ink breakdown (synthetic, threshold sat_true > "
                   f"{INK_SAT_COLORED_THRESHOLD})")
        md.append("")
        md.append("Mark pixels (`M`, same as the synthetic table above) split by true ink saturation "
                   "(`max(channel) - min(channel)` of the closed-form true ink -- see `score_synthetic_sample`): "
                   f"`sat_true > {INK_SAT_COLORED_THRESHOLD}` is \"coloured\" ink (e.g. the wide stamp's pink "
                   "shield), else \"grey\" (e.g. its lettering). This is the number that shows whether a retrain "
                   "actually fixed the ink head's grey collapse: `removed_pct_grey` was never the problem, "
                   "`removed_pct_colored` was.")
        md.append("")
        md.append(fmt_summary_table("Coloured ink", color_summary,
                                     ["removed_pct_colored", "ink_sat_pred_colored", "ink_sat_true_colored"]))
        md.append("")
        md.append(fmt_summary_table("Grey ink", color_summary,
                                     ["removed_pct_grey", "ink_sat_pred_grey", "ink_sat_true_grey"]))
        md.append("")
    if skipped_pages:
        md.append("### Skipped real pages")
        md.append("")
        md.append("| page | reason |")
        md.append("|---|---|")
        for r in skipped_pages:
            md.append(f"| {r['page']} | {r['skipped']} |")
        md.append("")
    if compare_lines:
        md.extend(compare_lines)
        md.append("")

    summary_md = out_dir / "summary.md"
    summary_md.write_text("\n".join(md), encoding="utf-8")

    print("\n" + "\n".join(md))
    print(f"\nWrote {pages_csv}, {synthetic_csv if synth_rows else '(no synthetic rows)'}, {summary_md}, "
          f"{panel_count} panels to {panels_dir}")
    print(f"Total runtime: {total_seconds:.1f}s")


if __name__ == "__main__":
    main()
