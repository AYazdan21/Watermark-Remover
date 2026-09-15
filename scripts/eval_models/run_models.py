"""Runs the three trained watermark models over wm_testset/images and dumps
raw-model outputs (overlay png, binary mask png for seg, json of instances)
per model, plus a 2x2 compare panel per image. No false-positive filtering
(watermark_remover/segmenter.py's post-processing is intentionally NOT used
here) -- this is meant to show what the bare model actually predicts.

Usage:
    ../.venv/Scripts/python.exe scripts/eval_models/run_models.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO / "wm_testset" / "images"
OUT_ROOT = REPO / "model_eval"

MODELS = [
    ("segmentations-freeze", REPO / "weights" / "yolo11-seg-freeze-new-dataset.pt", "seg"),
    ("segmentations-full", REPO / "weights" / "yolo11-seg-full-new-dataset.pt", "seg"),
    ("detection-freeze", REPO / "weights" / "yolo11s-det-freeze-new-dataset.pt", "det"),
]

CONF_LOW = 0.10
CONF_HIGH = 0.25
IMGSZ = 1024

RED = (0, 0, 255)     # BGR
YELLOW = (0, 255, 255)


def list_images():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts)


def simplify_polygon(poly_xy: np.ndarray, epsilon_frac: float = 0.003) -> list:
    """poly_xy: (N,2) float array of pixel coords. Returns simplified list of [x,y]."""
    if poly_xy is None or len(poly_xy) < 3:
        return []
    pts = poly_xy.astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(pts, True)
    eps = max(0.5, epsilon_frac * peri)
    approx = cv2.approxPolyDP(pts, eps, True)
    return approx.reshape(-1, 2).round(1).tolist()


def draw_label(img, x, y, text, color):
    cv2.putText(img, text, (int(x), int(max(12, y))), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(max(12, y))), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def run_seg_model(model, img_bgr, img_path, out_dir):
    h, w = img_bgr.shape[:2]
    res = model.predict(source=str(img_path), imgsz=IMGSZ, conf=CONF_LOW,
                         verbose=False, device=0)[0]

    overlay = img_bgr.copy()
    mask_union = np.zeros((h, w), dtype=np.uint8)
    instances = []

    if res.masks is not None and len(res.masks) > 0:
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        masks_data = res.masks.data.cpu().numpy()  # (N, mh, mw) at model res
        order = np.argsort(-confs)
        for i in order:
            conf = float(confs[i])
            m = masks_data[i]
            m_full = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            m_bin = (m_full >= 0.5).astype(np.uint8)
            area = int(m_bin.sum())
            x1, y1, x2, y2 = boxes[i].tolist()

            contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            if contours:
                c = max(contours, key=cv2.contourArea)
                polygon = simplify_polygon(c.reshape(-1, 2))

            is_high = conf >= CONF_HIGH
            if is_high:
                mask_union |= m_bin
                tint = np.zeros_like(overlay)
                tint[m_bin.astype(bool)] = RED
                overlay = cv2.addWeighted(overlay, 1.0, tint, 0.45, 0)
                cv2.drawContours(overlay, contours, -1, RED, 2)
            else:
                # faint/dashed yellow outline for low-conf instances
                dash_img = np.zeros_like(overlay)
                cv2.drawContours(dash_img, contours, -1, YELLOW, 2)
                overlay = cv2.addWeighted(overlay, 1.0, dash_img, 0.35, 0)

            ys, xs = np.where(m_bin > 0)
            ly, lx = (float(ys.min()), float(xs.min())) if len(ys) else (y1, x1)
            draw_label(overlay, lx, ly - 4, f"#{i} {conf:.2f}", RED if is_high else YELLOW)

            instances.append({
                "idx": int(i), "conf": conf, "box_xyxy": [x1, y1, x2, y2],
                "mask_area_px": area, "polygon": polygon,
            })

    stem = img_path.stem
    cv2.imwrite(str(out_dir / f"{stem}.png"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask_union * 255)
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"image": img_path.name, "width": w, "height": h,
                    "conf_low": CONF_LOW, "conf_high": CONF_HIGH,
                    "instances": instances}, f)
    return overlay, mask_union


def run_det_model(model, img_bgr, img_path, out_dir):
    h, w = img_bgr.shape[:2]
    res = model.predict(source=str(img_path), imgsz=IMGSZ, conf=CONF_LOW,
                         verbose=False, device=0)[0]

    overlay = img_bgr.copy()
    instances = []
    if res.boxes is not None and len(res.boxes) > 0:
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        order = np.argsort(-confs)
        for i in order:
            conf = float(confs[i])
            x1, y1, x2, y2 = boxes[i].tolist()
            is_high = conf >= CONF_HIGH
            color = RED if is_high else YELLOW
            thickness = 2 if is_high else 1
            if is_high:
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            else:
                # dashed rect
                pts = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
                for k in range(4):
                    p1, p2 = pts[k], pts[(k + 1) % 4]
                    n = 12
                    for t in range(0, n, 2):
                        a = t / n
                        b = min(1.0, (t + 1) / n)
                        pa = (int(p1[0] + (p2[0] - p1[0]) * a), int(p1[1] + (p2[1] - p1[1]) * a))
                        pb = (int(p1[0] + (p2[0] - p1[0]) * b), int(p1[1] + (p2[1] - p1[1]) * b))
                        cv2.line(overlay, pa, pb, color, 1)
            draw_label(overlay, x1, y1 - 4, f"#{i} {conf:.2f}", color)
            instances.append({
                "idx": int(i), "conf": conf, "box_xyxy": [x1, y1, x2, y2],
                "mask_area_px": None, "polygon": [],
            })

    stem = img_path.stem
    cv2.imwrite(str(out_dir / f"{stem}.png"), overlay)
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"image": img_path.name, "width": w, "height": h,
                    "conf_low": CONF_LOW, "conf_high": CONF_HIGH,
                    "instances": instances}, f)
    return overlay


def make_compare_panel(orig_bgr, overlays_titled, out_path, tile_long_side=800):
    """overlays_titled: list of (title, bgr_img) length 4 (orig + 3 models)."""
    def fit(img, long_side):
        h, w = img.shape[:2]
        s = long_side / max(h, w)
        return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))

    tiles = []
    for title, im in overlays_titled:
        t = fit(im, tile_long_side)
        th, tw = t.shape[:2]
        banner = np.full((26, tw, 3), 255, dtype=np.uint8)
        cv2.putText(banner, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([banner, t]))

    # pad tiles to same size
    max_h = max(t.shape[0] for t in tiles)
    max_w = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        th, tw = t.shape[:2]
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        canvas[:th, :tw] = t
        padded.append(canvas)

    top = np.hstack(padded[0:2])
    bottom = np.hstack(padded[2:4])
    grid = np.vstack([top, bottom])

    # final downscale so long side <= 1600
    gh, gw = grid.shape[:2]
    long_side = max(gh, gw)
    if long_side > 1600:
        s = 1600 / long_side
        grid = cv2.resize(grid, (int(gw * s), int(gh * s)))
    cv2.imwrite(str(out_path), grid)


def main():
    images = list_images()
    n = len(images)
    print(f"Found {n} test images in {IMAGES_DIR}")

    out_dirs = {}
    for name, weight_path, kind in MODELS:
        d = OUT_ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[name] = d
    compare_dir = OUT_ROOT / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    loaded = []
    for name, weight_path, kind in MODELS:
        t0 = time.time()
        m = YOLO(str(weight_path))
        loaded.append((name, m, kind))
        print(f"  loaded {name} ({kind}) from {weight_path.name} in {time.time()-t0:.1f}s")

    t_start = time.time()
    for idx, img_path in enumerate(images, 1):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[{idx}/{n}] WARN could not read {img_path.name}, skipping")
            continue

        panel_tiles = [("original", img_bgr)]
        for name, model, kind in loaded:
            out_dir = out_dirs[name]
            if kind == "seg":
                overlay, _ = run_seg_model(model, img_bgr, img_path, out_dir)
            else:
                overlay = run_det_model(model, img_bgr, img_path, out_dir)
            panel_tiles.append((name, overlay))

        make_compare_panel(img_bgr, panel_tiles, compare_dir / f"{img_path.stem}.png")

        elapsed = time.time() - t_start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (n - idx) / rate if rate > 0 else 0
        print(f"[{idx}/{n}] {img_path.name}  elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    print(f"Done. Total time {time.time()-t_start:.1f}s for {n} images x {len(loaded)} models.")


if __name__ == "__main__":
    main()
