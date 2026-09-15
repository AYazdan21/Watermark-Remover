"""Computes automatic, no-ground-truth metrics per (image, model) from the
raw outputs of run_models.py (model_eval/<model>/<stem>.json + _mask.png),
writes model_eval/metrics.csv, then applies simple documented thresholds to
flag candidate-problem rows into model_eval/auto_flags.csv.

These flags are HINTS for what to look at during visual review -- they are
not verdicts. Ground truth for this eval is human visual judgement (there
is no labelled test set), recorded separately in model_eval/review.csv.

Usage:
    ../.venv/Scripts/python.exe scripts/eval_models/metrics.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO / "wm_testset" / "images"
OUT_ROOT = REPO / "model_eval"

SEG_MODELS = ["segmentations-freeze", "segmentations-full"]
DET_MODELS = ["detection-freeze"]
ALL_MODELS = SEG_MODELS + DET_MODELS

CONF_HIGH = 0.25

# --- thresholds for containment / overlap bookkeeping (documented) --------
# A seg instance counts as "covered" by a det box if at least this fraction
# of the instance's own mask pixels fall inside some det box.
SEG_COVERED_BY_DET_FRAC = 0.30
# A det box counts as "empty of seg signal" if fewer than this fraction of
# its pixels are inside the seg model's union mask (checked against BOTH
# seg models; a box must miss both to count).
DET_BOX_EMPTY_SEG_FRAC = 0.05

# --- auto-flag thresholds (documented) -------------------------------------
FLAG_HIGH_DARK_INK = 0.50       # dark_ink_frac above this -> "looks like text"
FLAG_HIGH_COVERAGE = 0.35       # coverage above this -> suspiciously large
FLAG_HEAVY_FRAGMENTATION = 8.0  # instances per union-blob above this
FLAG_LOW_IOU_DISAGREEMENT = 0.20  # seg-freeze vs seg-full union IoU below this (both nonempty)
FLAG_DET_SEG_DISAGREEMENT = 0.5   # frac_det_box_not_in_seg above this


def list_stems():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p.stem for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts)


def load_json(model, stem):
    p = OUT_ROOT / model / f"{stem}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_union_mask(model, stem):
    p = OUT_ROOT / model / f"{stem}_mask.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.uint8)


def otsu_thresh(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def pixel_stats_in_mask(gray, hsv_sat, mask_bool, otsu_t):
    n = int(mask_bool.sum())
    if n == 0:
        return 0.0, 0.0
    dark = (gray[mask_bool].astype(np.float32) < (otsu_t - 15)).mean()
    sat = (hsv_sat[mask_bool] > 60).mean()
    return float(dark), float(sat)


def pixel_stats_in_boxes(gray, hsv_sat, boxes, h, w, otsu_t):
    if not boxes:
        return 0.0, 0.0
    box_mask = np.zeros((h, w), dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(round(x2))), min(h, int(round(y2)))
        if x2i > x1i and y2i > y1i:
            box_mask[y1i:y2i, x1i:x2i] = True
    return pixel_stats_in_mask(gray, hsv_sat, box_mask, otsu_t)


def edge_touch_count(items_hi, w, h, margin=2):
    n = 0
    for it in items_hi:
        x1, y1, x2, y2 = it["box_xyxy"]
        if x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin:
            n += 1
    return n


def conn_components(mask):
    if mask is None or mask.sum() == 0:
        return 0
    n, _ = cv2.connectedComponents(mask, connectivity=8)
    return n - 1  # subtract background label


def iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return None
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return None
    return float((a & b).sum() / union)


def frac_box_not_in_masks(box, masks, h, w):
    x1, y1, x2, y2 = box
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2i <= x1i or y2i <= y1i:
        return True
    area = (x2i - x1i) * (y2i - y1i)
    for m in masks:
        if m is None:
            continue
        inter = int(m[y1i:y2i, x1i:x2i].sum())
        if area > 0 and inter / area >= DET_BOX_EMPTY_SEG_FRAC:
            return False  # has enough overlap with at least one seg model
    return True


def frac_inst_not_in_boxes(polygon_or_mask_area_fn, inst, det_boxes, h, w, inst_mask=None):
    if inst_mask is None or inst_mask.sum() == 0:
        return True
    inst_area = int(inst_mask.sum())
    covered = np.zeros((h, w), dtype=bool)
    for (x1, y1, x2, y2) in det_boxes:
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(round(x2))), min(h, int(round(y2)))
        if x2i > x1i and y2i > y1i:
            covered[y1i:y2i, x1i:x2i] = True
    inter = int((inst_mask.astype(bool) & covered).sum())
    return (inter / inst_area) < SEG_COVERED_BY_DET_FRAC


def instance_mask_from_polygon(poly, h, w):
    if not poly or len(poly) < 3:
        return None
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m


def main():
    stems = list_stems()
    print(f"Computing metrics for {len(stems)} images x {len(ALL_MODELS)} models")

    rows = []
    for stem in stems:
        img_path_candidates = list(IMAGES_DIR.glob(stem + ".*"))
        if not img_path_candidates:
            continue
        img_bgr = cv2.imread(str(img_path_candidates[0]), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv_sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        otsu_t = otsu_thresh(gray)

        data = {m: load_json(m, stem) for m in ALL_MODELS}
        seg_union = {m: load_union_mask(m, stem) for m in SEG_MODELS}

        seg_iou = iou(seg_union.get("segmentations-freeze"), seg_union.get("segmentations-full"))

        for model in ALL_MODELS:
            d = data.get(model)
            is_seg = model in SEG_MODELS
            if d is None:
                rows.append({"image": stem, "model": model, "n_inst": 0, "n_inst_low": 0})
                continue

            insts = d.get("instances", [])
            hi = [it for it in insts if it["conf"] >= CONF_HIGH]
            lo = [it for it in insts if it["conf"] < CONF_HIGH]
            confs = [it["conf"] for it in hi]

            row = {
                "image": stem, "model": model,
                "n_inst": len(hi), "n_inst_low": len(lo),
                "conf_max": round(max(confs), 4) if confs else 0.0,
                "conf_mean": round(float(np.mean(confs)), 4) if confs else 0.0,
                "edge_touch": edge_touch_count(hi, w, h),
            }

            if is_seg:
                union = seg_union.get(model)
                cov = float(union.sum()) / (w * h) if union is not None else 0.0
                dark, sat = pixel_stats_in_mask(gray, hsv_sat, union.astype(bool), otsu_t) if union is not None else (0.0, 0.0)
                n_blobs = conn_components(union) if union is not None else 0
                frag = (len(hi) / n_blobs) if n_blobs > 0 else 0.0
                areas = [it["mask_area_px"] for it in hi if it.get("mask_area_px")]
                mean_area_frac = float(np.mean(areas)) / (w * h) if areas else 0.0

                det_boxes_hi = []
                if data.get("detection-freeze"):
                    det_boxes_hi = [it["box_xyxy"] for it in data["detection-freeze"]["instances"] if it["conf"] >= CONF_HIGH]
                not_covered = 0
                for it in hi:
                    inst_mask = instance_mask_from_polygon(it.get("polygon"), h, w)
                    if frac_inst_not_in_boxes(None, it, det_boxes_hi, h, w, inst_mask=inst_mask):
                        not_covered += 1
                frac_not_in_det = (not_covered / len(hi)) if hi else None

                row.update({
                    "coverage": round(cov, 4),
                    "dark_ink_frac": round(dark, 4),
                    "saturated_frac": round(sat, 4),
                    "mean_inst_area_frac": round(mean_area_frac, 4),
                    "fragmentation": round(frag, 3),
                    "iou_segfreeze_segfull": round(seg_iou, 4) if seg_iou is not None else "",
                    "frac_seg_inst_not_in_det": round(frac_not_in_det, 4) if frac_not_in_det is not None else "",
                    "frac_det_box_not_in_seg": "",
                })
            else:
                boxes_hi = [it["box_xyxy"] for it in hi]
                cov_mask = np.zeros((h, w), dtype=bool)
                for (x1, y1, x2, y2) in boxes_hi:
                    x1i, y1i = max(0, int(x1)), max(0, int(y1))
                    x2i, y2i = min(w, int(round(x2))), min(h, int(round(y2)))
                    if x2i > x1i and y2i > y1i:
                        cov_mask[y1i:y2i, x1i:x2i] = True
                cov = float(cov_mask.sum()) / (w * h)
                dark, sat = pixel_stats_in_boxes(gray, hsv_sat, boxes_hi, h, w, otsu_t)

                seg_masks = [seg_union.get(m) for m in SEG_MODELS]
                if boxes_hi:
                    n_empty = sum(1 for b in boxes_hi if frac_box_not_in_masks(b, seg_masks, h, w))
                    frac_det_not_in_seg = n_empty / len(boxes_hi)
                else:
                    frac_det_not_in_seg = None

                row.update({
                    "coverage": round(cov, 4),
                    "dark_ink_frac": round(dark, 4),
                    "saturated_frac": round(sat, 4),
                    "mean_inst_area_frac": "",
                    "fragmentation": "",
                    "iou_segfreeze_segfull": "",
                    "frac_seg_inst_not_in_det": "",
                    "frac_det_box_not_in_seg": round(frac_det_not_in_seg, 4) if frac_det_not_in_seg is not None else "",
                })
            rows.append(row)

    fieldnames = ["image", "model", "n_inst", "n_inst_low", "conf_max", "conf_mean",
                  "coverage", "dark_ink_frac", "saturated_frac", "edge_touch",
                  "mean_inst_area_frac", "fragmentation",
                  "iou_segfreeze_segfull", "frac_det_box_not_in_seg", "frac_seg_inst_not_in_det"]

    metrics_path = OUT_ROOT / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w_.writeheader()
        for r in rows:
            for k in fieldnames:
                r.setdefault(k, "")
            w_.writerow(r)
    print(f"Wrote {len(rows)} rows to {metrics_path}")

    # ---- auto flags ----
    flag_rows = []
    for r in rows:
        flags = []
        n_inst = r.get("n_inst", 0)
        n_lo = r.get("n_inst_low", 0)
        if n_inst == 0 and n_lo == 0:
            flags.append("no_detections")
        elif n_inst == 0 and n_lo > 0:
            flags.append("low_conf_only")
        if r.get("dark_ink_frac", "") != "" and r["dark_ink_frac"] > FLAG_HIGH_DARK_INK and n_inst > 0:
            flags.append("high_dark_ink")
        if r.get("coverage", "") != "" and r["coverage"] > FLAG_HIGH_COVERAGE:
            flags.append("high_coverage")
        if r.get("fragmentation", "") not in ("", None) and r["fragmentation"] > FLAG_HEAVY_FRAGMENTATION:
            flags.append("heavy_fragmentation")
        if r.get("iou_segfreeze_segfull", "") not in ("", None) and r["iou_segfreeze_segfull"] < FLAG_LOW_IOU_DISAGREEMENT:
            flags.append("seg_models_disagree")
        if r.get("frac_det_box_not_in_seg", "") not in ("", None) and r["frac_det_box_not_in_seg"] > FLAG_DET_SEG_DISAGREEMENT:
            flags.append("det_seg_disagree")
        if r.get("frac_seg_inst_not_in_det", "") not in ("", None) and r["frac_seg_inst_not_in_det"] > FLAG_DET_SEG_DISAGREEMENT:
            flags.append("det_seg_disagree")
        if flags:
            flag_rows.append({"image": r["image"], "model": r["model"], "flags": ";".join(flags)})

    flags_path = OUT_ROOT / "auto_flags.csv"
    with open(flags_path, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["image", "model", "flags"])
        w_.writeheader()
        for r in flag_rows:
            w_.writerow(r)
    print(f"Wrote {len(flag_rows)} flagged rows (of {len(rows)}) to {flags_path}")

    # compact summary
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    print("\n--- per-model quick summary (conf>=0.25) ---")
    for model, rs in by_model.items():
        n_zero = sum(1 for r in rs if r.get("n_inst", 0) == 0)
        print(f"{model}: {len(rs)} images, {n_zero} with zero detections")


if __name__ == "__main__":
    main()
