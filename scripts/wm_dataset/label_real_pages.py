"""Auto-labels real watermarked pages for fine-tuning, in the same YOLO-seg
format as the synthetic datasets (single class 0 = "watermark").

How a mask is made (per page):
  1. The detection model (watermark_remover.detector) finds the watermark
     boxes, padded by --box-padding px.
  2. The alpha network (default weights/alpha_net_v2_best_final.pt) predicts
     per-pixel watermark opacity. v2's alpha is accurate (96-101% of ground
     truth on synthetic composites); only its ink colour is wrong, and the
     mask only needs alpha.
  3. Alpha outside the boxes is zeroed, then the repo's own
     labels.alpha_to_polygons turns alpha into mask + polygons -- the SAME
     thresholding and polygon simplification the synthetic labels use, so
     real and synthetic labels are drawn the same way.

These are MODEL-GENERATED labels, not hand labels. Review overlays/ before
training on them: fine-tuning on uncorrected predictions mostly teaches the
model its own mistakes. Any alpha left outside the detection boxes is
reported per page, since a large value there usually means the detector
missed part of a mark.

Output (--out):
  images/  copies of the chosen pages
  labels/  YOLO-seg .txt, same base name (empty file = negative)
  masks/   binary PNG masks (255 = watermark)
  overlays/ review images: page | mask in red | polygon outlines
  data.yaml, report.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import labels  # noqa: E402  (scripts/wm_dataset/labels.py)
from watermark_remover import alpha_net, detector  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def page_features(img: np.ndarray) -> np.ndarray:
    """Cheap appearance descriptor used only to pick a varied subset."""
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    return np.concatenate([small.mean((0, 1)), small.std((0, 1)), [h / 2000.0, w / 2000.0]])


def pick_diverse(paths, n, seed=0):
    """Greedy farthest-point selection on page_features: n pages that differ
    from each other as much as possible, instead of n near-duplicates."""
    feats = np.stack([page_features(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) for p in paths])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    chosen = [int(np.random.default_rng(seed).integers(len(paths)))]
    dist = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < min(n, len(paths)):
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[nxt], axis=1))
    return [paths[i] for i in sorted(chosen)]


def overlay(img: np.ndarray, mask: np.ndarray, polygons) -> Image.Image:
    tint = img.copy()
    tint[mask > 0] = (0.45 * tint[mask > 0] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    outl = np.asarray(labels.draw_polygons_overlay(Image.fromarray(img), polygons))
    return Image.fromarray(np.concatenate([img, tint, outl], axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(REPO_ROOT / "wm_testset" / "images"))
    ap.add_argument("--out", default=str(REPO_ROOT / "wm_realtune"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--weights", default=str(REPO_ROOT / "weights" / "alpha_net_v2_best_final.pt"))
    ap.add_argument("--det-model", default=detector.DEFAULT_DETECT_MODEL)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--box-padding", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = sorted(p for p in Path(args.src).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    pages = pick_diverse(src, args.n, args.seed)

    out = Path(args.out)
    for sub in ("images", "labels", "masks", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = "cuda" if alpha_net.torch.cuda.is_available() else "cpu"
    model = alpha_net.load_model(args.weights, device=device)

    rows = []
    for p in pages:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        instances, _meta = detector.detect_watermark_boxes(img, args.det_conf, args.det_model, args.box_padding)

        alpha, _ink, _rec = alpha_net.predict_image(model, img, device=device)
        in_boxes = np.zeros((h, w), bool)
        for inst in instances:
            x1, y1, x2, y2 = inst["box"]
            in_boxes[y1:y2, x1:x2] = True
        outside = alpha * ~in_boxes
        alpha_boxed = np.where(in_boxes, alpha, 0.0).astype(np.float32)

        mask, polygons = labels.alpha_to_polygons(alpha_boxed)
        lines = labels.polygons_to_yolo_lines(polygons, w, h)

        shutil.copy2(p, out / "images" / p.name)
        labels.write_yolo_label(out / "labels" / f"{p.stem}.txt", lines)
        Image.fromarray(mask).save(out / "masks" / f"{p.stem}.png")
        overlay(img, mask, polygons).save(out / "overlays" / f"{p.stem}.jpg", quality=90)

        rows.append({
            "image": p.name, "width": w, "height": h, "det_boxes": len(instances),
            "polygons": len(polygons), "mask_px_pct": round(100 * (mask > 0).mean(), 3),
            "alpha_peak": round(float(alpha_boxed.max()), 3),
            "alpha_outside_boxes_px": int((outside > 0.05).sum()),
        })
        print(f"{p.name}: {len(instances)} boxes, {len(polygons)} polygons, "
              f"mask {rows[-1]['mask_px_pct']}%, alpha>0.05 outside boxes: {rows[-1]['alpha_outside_boxes_px']} px")

    with open(out / "report.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images\nval: images\nnc: 1\nnames: ['watermark']\n", encoding="utf-8")
    print(f"\nwrote {len(rows)} pages to {out}")


if __name__ == "__main__":
    main()
