"""Builds contrast-enhanced review panels from run_models.py's outputs.

The plain compare/ panels are downscaled originals, and at that size a large,
faint grey watermark (e.g. wm_testset 70_original, 82_original) is close to
invisible -- the first review pass misjudged exactly those pages. Here the
base image is local-background-normalised (gray / max-filtered gray, then
stretched) so a mark only a few grey levels darker than its surroundings
shows up as clearly as the text, on white paper and on tinted banners alike.

Panel: enhanced | enhanced + seg-freeze | enhanced + seg-full | enhanced + det
(conf >= 0.25 only). Written to model_eval/review_enhanced/<stem>.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
IMAGES = REPO / "wm_testset" / "images"
EVAL = REPO / "model_eval"
OUT = EVAL / "review_enhanced"
CONF = 0.25
MAX_SIDE = 1600
BG_KERNEL = 31      # wider than a text stroke, so the max filter sees paper
RATIO_LO = 0.70     # ratio mapped to black; 1.0 (= local paper) maps to white
MODELS = [
    ("segmentations-freeze", "seg-freeze", (0, 0, 255)),
    ("segmentations-full", "seg-full", (255, 90, 0)),
    ("detection-freeze", "det-freeze", (0, 160, 0)),
]


def enhance(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (BG_KERNEL, BG_KERNEL))
    bg = cv2.GaussianBlur(cv2.dilate(gray, k), (0, 0), BG_KERNEL / 3)
    ratio = np.clip(gray / bg, RATIO_LO, 1.0)
    out = ((ratio - RATIO_LO) / (1.0 - RATIO_LO) * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def draw(base: np.ndarray, folder: str, stem: str, color, is_det: bool) -> np.ndarray:
    tile = base.copy()
    lw = max(2, round(max(base.shape[:2]) / 500))
    if is_det:
        data = json.loads((EVAL / folder / f"{stem}.json").read_text(encoding="utf-8"))
        for inst in data["instances"]:
            if inst["conf"] >= CONF:
                x1, y1, x2, y2 = map(int, inst["box_xyxy"])
                cv2.rectangle(tile, (x1, y1), (x2, y2), color, lw)
    else:
        mask = cv2.imread(str(EVAL / folder / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.any():
            if mask.shape != base.shape[:2]:
                mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
            sel = mask > 0
            tint = np.array(color, np.float32)
            tile[sel] = (0.65 * tile[sel] + 0.35 * tint).astype(np.uint8)
            contours, _ = cv2.findContours((sel * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tile, contours, -1, color, lw)
    return tile


def label(tile: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((34, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, tile])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for i, p in enumerate(paths, 1):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        base = enhance(bgr)
        tiles = [label(base, "enhanced")]
        for folder, name, color in MODELS:
            tiles.append(label(draw(base, folder, p.stem, color, folder.startswith("detection")), name))
        h, w = tiles[0].shape[:2]
        if h / w > 1.3:  # tall page: one row of four
            panel = np.hstack(tiles)
        else:
            panel = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        s = MAX_SIDE / max(panel.shape[:2])
        if s < 1:
            panel = cv2.resize(panel, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{p.stem}.png"), panel)
        print(f"\r[{i}/{len(paths)}] {p.name}", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
