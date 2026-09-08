"""Turns a continuous alpha coverage map (from compositor.py) into YOLO-seg
polygon labels.

Pipeline: alpha map -> binary mask (threshold) -> contours (cv2) ->
simplified polygons -> normalized YOLO-seg lines.

Mask threshold: RELATIVE to each sample's own peak alpha, not absolute.

This distinction is load-bearing and an absolute cut was measured to fail.
The calibrated base alpha at 1.0x opacity is only ~0.222 (the mark is
semi-transparent by design, see asset_prep.MEASURED_*), and generation
applies a 0.5x-1.5x opacity multiplier on top, so a dim sample peaks around
0.11. An absolute 0.08 cut looks safe against that peak -- but the threshold
applies PER PIXEL, and most pixels of a glyph sit well below its peak
(anti-aliased edges, thin strokes, the feathered silhouette). Measured on a
real failing sample: at 0.08 the mask still held 21k pixels but shattered
into 759 disconnected specks whose largest was 202px, every one of them
under MIN_AREA_FRAC's floor -- so all were dropped and the label came out
EMPTY for an image that visibly contains the watermark. That is worse than
useless as training data: it actively teaches "no watermark here" on a
positive. At a relative cut the same sample yields ~39 solid components.

So: threshold = max(ABS_FLOOR, REL_PEAK_FRAC * peak_alpha_of_this_sample),
which self-calibrates across the whole opacity range without needing the
generator to plumb its multiplier through. Samples whose peak is under
ABS_MIN_PEAK are treated as genuinely empty (true negatives).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

MASK_ALPHA_THRESHOLD = 0.08  # retained for callers that want an absolute cut

# Relative-threshold parameters (see module docstring).
REL_PEAK_FRAC = 0.25
ABS_FLOOR = 0.012
ABS_MIN_PEAK = 0.02


def resolve_threshold(alpha_map: np.ndarray) -> float:
    """Per-sample threshold, scaled to this map's own peak alpha. Returns a
    value above any achievable alpha when the map is effectively empty, so a
    true negative yields an empty mask rather than selecting noise."""
    peak = float(np.clip(alpha_map, 0.0, 1.0).max()) if alpha_map.size else 0.0
    if peak < ABS_MIN_PEAK:
        return 1.1
    return max(ABS_FLOOR, REL_PEAK_FRAC * peak)

# Drop components smaller than this fraction of the image area -- removes
# JPEG/blur speckle noise near threshold and any near-zero-area slivers left
# by rotation clipping at the image border.
MIN_AREA_FRAC = 0.00035
# Polygon simplification strength, as a fraction of the contour perimeter.
EPSILON_FRAC = 0.01
# A simplified polygon with fewer than this many vertices after clamping
# isn't a usable region.
MIN_VERTICES = 3


def alpha_to_mask(alpha_map: np.ndarray, threshold: float = MASK_ALPHA_THRESHOLD) -> np.ndarray:
    """Binary uint8 mask (0/255), same HxW as alpha_map."""
    return (np.clip(alpha_map, 0.0, 1.0) >= threshold).astype(np.uint8) * 255


def mask_to_polygons(mask: np.ndarray, min_area_frac: float = MIN_AREA_FRAC,
                      epsilon_frac: float = EPSILON_FRAC,
                      ) -> List[np.ndarray]:
    """Finds connected watermark regions in `mask` and returns simplified
    polygons (each an (N, 2) float array of pixel coordinates).

    Uses RETR_EXTERNAL: internal holes (e.g. inside closed letterforms like
    "A" or "e") are not represented as separate rings -- YOLO-seg polygons
    are single un-holed contours, and for a coverage mask (not a legibility
    mask) filling those small holes is the correct, standard choice.

    Instances clipped by the image border are handled automatically: their
    contour simply follows the border, since findContours operates on the
    full-frame mask rather than per-instance crops.
    """
    h, w = mask.shape[:2]
    min_area = max(20.0, min_area_frac * h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[np.ndarray] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(0.5, epsilon_frac * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < MIN_VERTICES:
            # Simplification collapsed a real (large-area) region to a
            # degenerate shape -- fall back to a lighter simplification
            # rather than dropping a legitimate instance.
            approx = cv2.approxPolyDP(c, max(0.5, eps * 0.3), True)
            pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < MIN_VERTICES:
            continue
        polygons.append(pts)
    return polygons


def alpha_to_polygons(alpha_map: np.ndarray, threshold: float = None,
                       min_area_frac: float = MIN_AREA_FRAC,
                       epsilon_frac: float = EPSILON_FRAC,
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Convenience wrapper: alpha map -> (binary mask, polygon list).

    threshold=None (the default) self-calibrates per sample via
    resolve_threshold; pass a float only to force an absolute cut.
    """
    if threshold is None:
        threshold = resolve_threshold(alpha_map)
    mask = alpha_to_mask(alpha_map, threshold)
    polys = mask_to_polygons(mask, min_area_frac, epsilon_frac)
    return mask, polys


def polygons_to_yolo_lines(polygons: Sequence[np.ndarray], img_w: int, img_h: int,
                            class_id: int = 0) -> List[str]:
    """Normalizes pixel-coordinate polygons to YOLO-seg label lines:
    "<class_id> x1 y1 x2 y2 ..." with coordinates in [0, 1]."""
    lines = []
    for poly in polygons:
        if len(poly) < MIN_VERTICES:
            continue
        coords = []
        for x, y in poly:
            nx = min(1.0, max(0.0, x / img_w))
            ny = min(1.0, max(0.0, y / img_h))
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")
        lines.append(" ".join([str(class_id)] + coords))
    return lines


def write_yolo_label(path: str | Path, lines: Sequence[str]) -> None:
    """Writes label lines to `path`; an empty sequence writes an empty file
    (the correct YOLO representation of a negative / no-object image)."""
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_label(path: str | Path) -> List[Tuple[int, np.ndarray]]:
    """Parses a YOLO-seg label file back into [(class_id, (N,2) points), ...]
    with points still normalized [0, 1]. Used for verification/round-trip
    checks. Raises ValueError on a malformed line (odd coordinate count)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0 or len(coords) < 6:
            raise ValueError(f"malformed YOLO-seg line in {path}: {line!r}")
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        out.append((class_id, pts))
    return out


def draw_polygons_overlay(image: Image.Image, polygons: Sequence[np.ndarray],
                           color: Tuple[int, int, int] = (255, 32, 32),
                           width: int = 3) -> Image.Image:
    """Draws polygons (pixel coords) over a copy of `image`, for visual
    verification that labels land on the watermark."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for poly in polygons:
        pts = [(float(x), float(y)) for x, y in poly]
        if len(pts) >= 2:
            draw.polygon(pts, outline=color, width=width)
        for x, y in pts:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    return out
