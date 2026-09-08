"""Places the cleaned AriaTender stamp (see asset_prep.py) onto background
document images, producing a composited RGB image plus a *continuous* alpha
coverage map that is an exact record of which pixels were touched and by how
much. That alpha map is the free, perfect ground truth this whole synthetic
pipeline exists to generate -- labels.py turns it into YOLO-seg polygons, and
generate.py also saves it directly for a future alpha-regression head.

Compositing math: the source mark is semi-transparent ink over paper, so it
must darken the background *proportionally* to alpha, not paste over it:

    observed = a * ink + (1 - a) * background

Multiple overlapping instances (e.g. a lattice tile touching its neighbour)
are accumulated with the standard "over" operator so alpha never exceeds 1
and the darkening compounds the way real overlapping ink would.

Placement patterns, chosen to match what's actually observed on real
watermarked documents (see module docstring context in the task brief):
  - single:   one large instance (~0.6-1.0x page width), mild rotation --
              the "big mark centred on the page" case of the source document.
  - oversize: one instance scaled PAST the page (1.05-1.9x) and off-centre,
              so the frame clips it -- the subtitle drops off the bottom, or
              part of the logo runs past an edge. Trains partial-mark recall.
  - corner:   one small instance tucked near a page corner.
  - lattice:  a regular tiled grid of small instances (~0.06-0.14x page
              width), the "repeated small mark" case, usually near-horizontal.
  - diagonal: a tiled grid rotated as a whole, 25-50 degrees, covering the
              full page -- the "diagonal repeating banner" case.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import asset_prep

# ---------------------------------------------------------------------------
# Stamp loading
# ---------------------------------------------------------------------------

# _wm_extract_deliverable lives one level above the repo root (sibling of
# Watermark-Remover), matching the paths the task brief gives relative to cwd
# "Watermark-Remover": "../_wm_extract_deliverable/...".
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parents[3] / "_wm_extract_deliverable"
LOGO_NAME = "ariatender_logo_hammer_text.png"
SUBTITLE_NAME = "ariatender_subtitle_persian.png"

PATTERNS = ("single", "oversize", "lattice", "diagonal", "corner")


@functools.lru_cache(maxsize=4)
def load_base_stamp(assets_dir: Optional[str] = None) -> Image.Image:
    """Loads and cleans the two source crops into one combined RGBA stamp.

    Cached (by assets_dir) since asset_prep's cleaning involves connected-
    component analysis and Gaussian feathering -- no need to redo it per
    sample. Returns a fresh copy per call so callers can mutate freely.
    """
    d = Path(assets_dir) if assets_dir else DEFAULT_ASSETS_DIR
    logo = asset_prep.clean_stamp(str(d / LOGO_NAME))
    subtitle = asset_prep.clean_stamp(str(d / SUBTITLE_NAME))
    return asset_prep.combine_stamp(logo, subtitle)


def get_stamp(assets_dir: Optional[str] = None) -> Image.Image:
    """Public entry point: a fresh copy of the base stamp, safe to mutate."""
    return load_base_stamp(assets_dir).copy()


# ---------------------------------------------------------------------------
# Instance record
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """One placed copy of the stamp, in final-image pixel coordinates."""

    polygon: List[Tuple[float, float]]  # oriented rect corners, image coords
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (axis-aligned)
    scale: float
    rotation_deg: float
    opacity_mult: float


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotate_scale_stamp(stamp: Image.Image, scale: float, angle_deg: float,
                         opacity_mult: float, tint_shift: int) -> Image.Image:
    """Returns a resized + rotated + re-tinted copy of the stamp, RGBA, with
    the alpha channel already scaled by opacity_mult (clipped to [0, 255])."""
    w = max(1, int(round(stamp.width * scale)))
    h = max(1, int(round(stamp.height * scale)))
    resized = stamp.resize((w, h), Image.LANCZOS)

    r, g, b, a = resized.split()
    if tint_shift:
        ink = np.array(r, dtype=np.int16)  # r == g == b (flat gray ink)
        ink = np.clip(ink + tint_shift, 0, 255).astype(np.uint8)
        r = g = b = Image.fromarray(ink)
    if opacity_mult != 1.0:
        a_arr = np.array(a, dtype=np.float32) * opacity_mult
        a = Image.fromarray(np.clip(a_arr, 0, 255).astype(np.uint8))
    resized = Image.merge("RGBA", (r, g, b, a))

    if angle_deg:
        resized = resized.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
    return resized


def _oriented_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float
                       ) -> List[Tuple[float, float]]:
    """Corners of a w x h rect centered at (cx, cy), rotated by angle_deg
    (degrees, counter-clockwise, matching PIL's rotate convention)."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    hw, hh = w / 2.0, h / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    out = []
    for dx, dy in corners:
        # PIL rotate() is counter-clockwise for positive angles in image
        # coords (y-down); rotating (dx, dy) by +theta accordingly:
        rx = dx * cos_t + dy * sin_t
        ry = -dx * sin_t + dy * cos_t
        out.append((cx + rx, cy + ry))
    return out


# ---------------------------------------------------------------------------
# Alpha-over accumulation onto float buffers
# ---------------------------------------------------------------------------

def _paste_over(canvas_rgb: np.ndarray, canvas_a: np.ndarray,
                 tile_rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-composites tile_rgba (float, 0..1) onto canvas_rgb/canvas_a
    (float, 0..1) in place, at top-left offset (x, y), clipped to bounds.

    canvas_rgb always holds a fully-resolved *straight* (non-premultiplied)
    color -- it starts as the opaque background and, after each tile, is the
    already-blended visual result -- so a new tile blends over it with the
    plain "src over opaque dst" formula, independent of canvas_a. canvas_a is
    separate bookkeeping: the *coverage* map for labels, accumulated with the
    standard over-alpha rule so overlapping/repeated stamps compound (more
    coverage) even though the RGB blend above always sees an opaque backdrop.
    """
    th, tw = tile_rgba.shape[:2]
    ch, cw = canvas_a.shape[:2]

    src_x0, src_y0 = max(0, -x), max(0, -y)
    dst_x0, dst_y0 = max(0, x), max(0, y)
    dst_x1, dst_y1 = min(cw, x + tw), min(ch, y + th)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    t_rgb = tile_rgba[src_y0:src_y1, src_x0:src_x1, :3]
    t_a = tile_rgba[src_y0:src_y1, src_x0:src_x1, 3]

    dst_rgb = canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1, :]
    dst_a = canvas_a[dst_y0:dst_y1, dst_x0:dst_x1]

    rgb_out = t_rgb * t_a[..., None] + dst_rgb * (1.0 - t_a[..., None])
    a_out = t_a + dst_a * (1.0 - t_a)

    canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1, :] = rgb_out
    canvas_a[dst_y0:dst_y1, dst_x0:dst_x1] = a_out


# ---------------------------------------------------------------------------
# Placement pattern generators -- each yields (cx, cy, scale, angle) tuples
# in final-image pixel coordinates for one sample.
# ---------------------------------------------------------------------------

def _placements_single(w: int, h: int, rng: np.random.Generator
                        ) -> Tuple[List[Tuple[float, float, float, float]], float]:
    frac = rng.uniform(0.60, 1.00)
    angle = rng.uniform(-8, 8) if rng.random() < 0.7 else rng.uniform(20, 45) * rng.choice([-1, 1])
    cx = rng.uniform(0.5 - 0.15, 0.5 + 0.15) * w
    cy = rng.uniform(0.5 - 0.15, 0.5 + 0.15) * h
    return [(cx, cy, frac, angle)], frac


def _placements_oversize(w: int, h: int, rng: np.random.Generator
                          ) -> Tuple[List[Tuple[float, float, float, float]], float]:
    """One mark scaled BEYOND the page, so it is clipped by the frame.

    Real scans and screenshots are frequently cropped mid-watermark: the
    subtitle line falls off the bottom, or one end of the logo runs past the
    edge. A model trained only on fully-visible marks learns the whole
    silhouette as one rigid template and degrades on partial ones -- which is
    exactly the case a user hits when they crop a region out of a page. The
    label follows the visible part only, since the compositor's alpha map
    records what actually landed inside the frame.
    """
    frac = rng.uniform(1.05, 1.9)
    angle = rng.uniform(-10, 10) if rng.random() < 0.75 else rng.uniform(15, 40) * rng.choice([-1, 1])
    # Push the centre well off the middle so the clipping is asymmetric --
    # sometimes the subtitle is gone, sometimes half the logo.
    cx = rng.uniform(0.20, 0.80) * w
    cy = rng.uniform(0.18, 0.82) * h
    return [(cx, cy, frac, angle)], frac


def _placements_corner(w: int, h: int, rng: np.random.Generator
                        ) -> Tuple[List[Tuple[float, float, float, float]], float]:
    frac = rng.uniform(0.12, 0.28)
    angle = rng.uniform(-10, 10)
    margin = 0.06
    corner = rng.integers(0, 4)
    cx = margin * w if corner in (0, 3) else (1 - margin) * w
    cy = margin * h if corner in (0, 1) else (1 - margin) * h
    return [(cx, cy, frac, angle)], frac


def _placements_lattice(w: int, h: int, rng: np.random.Generator, diagonal: bool
                         ) -> Tuple[List[Tuple[float, float, float, float]], float]:
    frac = rng.uniform(0.07, 0.14)
    grid_angle = rng.uniform(30, 50) * rng.choice([-1, 1]) if diagonal else rng.uniform(-6, 6)
    stamp_w = frac * w
    spacing_x = stamp_w * rng.uniform(1.6, 2.4)
    spacing_y = spacing_x * rng.uniform(0.7, 1.1)
    jitter = spacing_x * 0.08

    # Overscan so the rotated grid still covers corners after rotation.
    diag = math.hypot(w, h)
    n_cols = int(diag / spacing_x) + 4
    n_rows = int(diag / spacing_y) + 4
    theta = math.radians(grid_angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx0, cy0 = w / 2.0, h / 2.0

    placements = []
    for i in range(-n_rows // 2, n_rows // 2 + 1):
        for j in range(-n_cols // 2, n_cols // 2 + 1):
            gx = j * spacing_x + rng.uniform(-jitter, jitter)
            gy = i * spacing_y + rng.uniform(-jitter, jitter)
            px = cx0 + gx * cos_t - gy * sin_t
            py = cy0 + gx * sin_t + gy * cos_t
            if -stamp_w < px < w + stamp_w and -stamp_w < py < w + stamp_w + h:
                placements.append((px, py, frac, grid_angle))
    return placements, frac


def _choose_pattern(rng: np.random.Generator, pattern: Optional[str]) -> str:
    if pattern is not None:
        assert pattern in PATTERNS, f"unknown pattern {pattern!r}"
        return pattern
    # Roughly matches observed real-world mix: large single marks and
    # lattices are both common; corner and diagonal are less so.
    return str(rng.choice(PATTERNS, p=[0.30, 0.18, 0.24, 0.16, 0.12]))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def composite(background: Image.Image, stamp: Image.Image,
              rng: np.random.Generator, pattern: Optional[str] = None,
              opacity_mult: Optional[float] = None,
              tint_shift: Optional[int] = None,
              ) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """Composites `stamp` onto `background` using a sampled placement
    pattern. Returns (composited_rgb [PIL RGB], alpha_map [float32 HxW,
    0..1], meta dict).

    `background` must be a PIL Image (any mode; converted to RGB).
    `stamp` is the RGBA stamp from get_stamp().
    `opacity_mult` / `tint_shift`, if given, fix those sample-level values
    (used by the calibration self-test); otherwise opacity_mult is sampled
    ~U(0.5, 1.5) around the calibrated base alpha as specified in the task
    brief, and tint_shift ~U(-18, 18).
    """
    bg = background.convert("RGB")
    w, h = bg.size

    chosen_pattern = _choose_pattern(rng, pattern)
    if chosen_pattern == "single":
        placements, base_frac = _placements_single(w, h, rng)
    elif chosen_pattern == "corner":
        placements, base_frac = _placements_corner(w, h, rng)
    elif chosen_pattern == "oversize":
        placements, base_frac = _placements_oversize(w, h, rng)
    elif chosen_pattern == "lattice":
        placements, base_frac = _placements_lattice(w, h, rng, diagonal=False)
    else:  # diagonal
        placements, base_frac = _placements_lattice(w, h, rng, diagonal=True)

    if opacity_mult is None:
        opacity_mult = float(rng.uniform(0.5, 1.5))
    # Small per-sample ink tint jitter (documents scan/photograph the mark
    # with slightly different gray balance); shared across all instances in
    # a sample so it reads as one consistent scan, not per-instance noise.
    if tint_shift is None:
        tint_shift = int(rng.integers(-18, 19))

    canvas_rgb = np.asarray(bg, dtype=np.float32) / 255.0
    canvas_a = np.zeros((h, w), dtype=np.float32)

    instances: List[Instance] = []
    aspect = stamp.height / stamp.width
    for (cx, cy, frac, angle) in placements:
        scale = (frac * w) / stamp.width
        if scale <= 0.01:
            continue
        tile = _rotate_scale_stamp(stamp, scale, angle, opacity_mult, tint_shift)
        tile_arr = np.asarray(tile, dtype=np.float32) / 255.0
        tw, th = tile.size
        x = int(round(cx - tw / 2.0))
        y = int(round(cy - th / 2.0))

        # Skip instances that don't touch the canvas at all.
        if x + tw <= 0 or x >= w or y + th <= 0 or y >= h:
            continue

        _paste_over(canvas_rgb, canvas_a, tile_arr, x, y)

        poly = _oriented_polygon(cx, cy, frac * w, frac * w * aspect, angle)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = (max(0.0, min(xs)), max(0.0, min(ys)), min(float(w), max(xs)), min(float(h), max(ys)))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        instances.append(Instance(polygon=poly, bbox=bbox, scale=scale,
                                   rotation_deg=angle, opacity_mult=opacity_mult))

    out_rgb = Image.fromarray(np.clip(canvas_rgb * 255.0, 0, 255).astype(np.uint8), "RGB")

    meta: Dict[str, Any] = {
        "pattern": chosen_pattern,
        "base_scale_frac": base_frac,
        "opacity_mult": opacity_mult,
        "tint_shift": tint_shift,
        "n_instances": len(instances),
        "instances": [
            {
                "polygon": inst.polygon,
                "bbox": inst.bbox,
                "scale": inst.scale,
                "rotation_deg": inst.rotation_deg,
                "opacity_mult": inst.opacity_mult,
            }
            for inst in instances
        ],
    }
    return out_rgb, canvas_a, meta


# ---------------------------------------------------------------------------
# Post-compositing augmentations (applied to the whole sample, watermarked
# or not, so the watermark region carries no compositing "tell").
# ---------------------------------------------------------------------------

def apply_scan_augmentations(image: Image.Image, alpha_map: np.ndarray,
                              rng: np.random.Generator,
                              ) -> Tuple[Image.Image, np.ndarray]:
    """Mild Gaussian blur (scanned look) and an occasional downscale/upscale
    round trip (screenshot resampling). Applied identically to the image and
    the alpha map so labels stay aligned with what's visible. JPEG
    re-encoding is deliberately NOT done here -- that happens once, at save
    time, in generate.py, after both train/val paths reuse this image."""
    w, h = image.size

    if rng.random() < 0.5:
        sigma = float(rng.uniform(0.3, 1.1))
        image = image.filter(ImageFilter.GaussianBlur(sigma))
        alpha_map = cv2_gaussian_blur(alpha_map, sigma)

    if rng.random() < 0.3:
        factor = float(rng.uniform(0.5, 0.85))
        small = (max(1, int(w * factor)), max(1, int(h * factor)))
        image = image.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
        alpha_img = Image.fromarray((np.clip(alpha_map, 0, 1) * 255).astype(np.uint8))
        alpha_img = alpha_img.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
        alpha_map = np.asarray(alpha_img, dtype=np.float32) / 255.0

    return image, alpha_map


def cv2_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(arr, (k, k), sigma)
