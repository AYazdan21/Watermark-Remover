"""Runs a trained watermark segmenter over UNLABELLED real images and writes
prediction overlays plus descriptive statistics.

Why this exists separately from train_segmenter.py: accuracy metrics (IoU,
precision, recall, mAP) are comparisons against ground truth, so they are
mathematically undefined without labels. Real document images typically
arrive unlabelled. But the single most important question about a model
trained on synthetic composites -- did it transfer to real documents at all,
or did it learn the compositing artifacts? -- is answerable by *looking* at
predictions, with no labels required. That check costs minutes and should
happen before anyone invests hours in hand-labelling.

What you get:
  - Side-by-side overlay PNGs (original | prediction) per image.
  - Descriptive stats that need no ground truth: how many instances were
    found per image, what fraction of the page they cover, and the
    confidence distribution.

What you deliberately do NOT get: any claim of accuracy. A model can fire
confidently on the wrong pixels. These outputs are for human judgement, not
scoring -- read them as "did it find the mark, and did it fire anywhere it
obviously shouldn't", not as evidence of correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_COLOR = (255, 40, 40)
MASK_ALPHA = 0.45


def find_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def overlay_masks(img: Image.Image, masks: np.ndarray) -> Image.Image:
    """Tints every predicted mask pixel. masks: (N, H, W) in {0,1}, already
    resized to the image."""
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    if masks is not None and len(masks):
        union = (masks.sum(axis=0) > 0)
        tint = np.array(MASK_COLOR, dtype=np.float32)
        base[union] = (1 - MASK_ALPHA) * base[union] + MASK_ALPHA * tint
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def side_by_side(original: Image.Image, overlaid: Image.Image, label: str) -> Image.Image:
    w, h = original.size
    pad = 8
    canvas = Image.new("RGB", (w * 2 + pad, h + 26), (245, 245, 245))
    canvas.paste(original.convert("RGB"), (0, 26))
    canvas.paste(overlaid, (w + pad, 26))
    d = ImageDraw.Draw(canvas)
    d.text((4, 6), "original", fill=(40, 40, 40))
    d.text((w + pad + 4, 6), label, fill=(180, 30, 30))
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="Trained YOLO-seg .pt checkpoint.")
    ap.add_argument("--images", required=True, help="Directory of unlabelled images (recursive).")
    ap.add_argument("--out", default="runs/wm_seg/overlays", help="Where to write overlays + stats.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N images (0 = all).")
    ap.add_argument("--max-width", type=int, default=1400,
                     help="Downscale overlays wider than this, to keep files viewable.")
    args = ap.parse_args(argv)

    weights = Path(args.weights).resolve()
    img_dir = Path(args.images).resolve()
    out_dir = Path(args.out).resolve()

    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        return 1
    if not img_dir.is_dir():
        print(f"ERROR: image directory not found: {img_dir}")
        return 1
    images = find_images(img_dir)
    if not images:
        print(f"ERROR: no images found under {img_dir}")
        return 1
    if args.limit:
        images = images[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"weights : {weights}")
    print(f"images  : {img_dir}  ({len(images)} files)")
    print(f"out     : {out_dir}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: ultralytics not importable: {exc}")
        return 1

    model = YOLO(str(weights))

    rows = []
    for i, path in enumerate(images, 1):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        res = model.predict(source=str(path), conf=args.conf, iou=args.iou,
                            imgsz=args.imgsz, device=args.device, verbose=False)[0]

        confs = []
        masks_arr = None
        if res.masks is not None and res.masks.data is not None and len(res.masks.data):
            m = res.masks.data.cpu().numpy()  # (N, h', w') at model resolution
            resized = []
            for single in m:
                mi = Image.fromarray((single * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
                resized.append((np.asarray(mi) > 127).astype(np.uint8))
            masks_arr = np.stack(resized) if resized else None
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf):
            confs = [float(c) for c in res.boxes.conf.cpu().numpy()]

        n_inst = 0 if masks_arr is None else int(masks_arr.shape[0])
        coverage = 0.0 if masks_arr is None else float((masks_arr.sum(axis=0) > 0).mean())

        overlaid = overlay_masks(img, masks_arr)
        label = f"prediction: {n_inst} inst, {coverage * 100:.2f}% of page"
        panel = side_by_side(img, overlaid, label)
        if panel.width > args.max_width:
            panel.thumbnail((args.max_width, args.max_width * 4))
        panel.save(out_dir / f"{path.stem}_overlay.png")

        rows.append({
            "image": str(path.relative_to(img_dir)),
            "width": W, "height": H,
            "instances": n_inst,
            "coverage_frac": round(coverage, 5),
            "conf_max": round(max(confs), 4) if confs else 0.0,
            "conf_mean": round(float(np.mean(confs)), 4) if confs else 0.0,
        })
        if i % 10 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}]")

    stats_path = out_dir / "predictions.json"
    stats_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n_any = sum(1 for r in rows if r["instances"] > 0)
    covs = [r["coverage_frac"] for r in rows]
    inst = [r["instances"] for r in rows]
    print("\n" + "=" * 62)
    print("QUALITATIVE SUMMARY (no ground truth -- these are NOT accuracy metrics)")
    print("=" * 62)
    print(f"  images processed          : {len(rows)}")
    print(f"  images with >=1 detection : {n_any}  ({100.0 * n_any / len(rows):.1f}%)")
    print(f"  instances/image           : mean {np.mean(inst):.1f}   median {np.median(inst):.0f}   max {max(inst)}")
    print(f"  page coverage             : mean {100 * np.mean(covs):.2f}%   median {100 * np.median(covs):.2f}%   max {100 * max(covs):.2f}%")
    print(f"\n  overlays -> {out_dir}")
    print(f"  stats    -> {stats_path}")
    print("\n  Open the overlays and judge by eye: did it find the mark, and did it")
    print("  fire on anything it obviously should not (body text, table rules)?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
