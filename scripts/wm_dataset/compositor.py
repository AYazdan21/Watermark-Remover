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

Single-layer alpha-regression model (for generate.py's --save-clean /
"clean" target): the equation `observed = a*ink + (1-a)*clean` is EXACT
before apply_scan_augmentations -- composite()'s over-accumulated alpha
(canvas_a) and its RGB blend (canvas_rgb) are built from the same per-tile
alpha and the same "over" accumulation, so they are mutually consistent by
construction. apply_scan_augmentations then blurs/resamples image, alpha_map
and clean identically, but blur does not commute with the multiply in
`a*ink + (1-a)*clean` (blur(a*ink) != blur(a)*blur(ink) in general), so the
equation becomes only approximate after augmentation, and only near mark
edges where alpha changes quickly over the blur kernel's support -- flat
interior regions (alpha locally constant) are unaffected. This is why the
alpha-regression network trains against the pre-JPEG `clean` target with an
L1/robust loss rather than assuming the closed form is exact everywhere.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import asset_prep

# ---------------------------------------------------------------------------
# Stamp loading / mark registry
# ---------------------------------------------------------------------------

# _wm_extract_deliverable lives one level above the repo root (sibling of
# Watermark-Remover), matching the paths the task brief gives relative to cwd
# "Watermark-Remover": "../_wm_extract_deliverable/...".
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parents[3] / "_wm_extract_deliverable"
LOGO_NAME = "ariatender_logo_hammer_text.png"
SUBTITLE_NAME = "ariatender_subtitle_persian.png"
WIDE_WORDMARK_NAME = "ariatender_wide_wordmark.png"

PATTERNS = ("single", "oversize", "lattice", "diagonal", "corner")

STACKED_MARK_ID = "ariatender_stacked"
WIDE_MARK_ID = "ariatender_wide"


@dataclass(frozen=True)
class MarkSpec:
    """Everything pattern-placement needs to know about one watermark mark,
    besides its pixels (those are loaded separately and cached -- see
    load_base_stamp). Bundling this per-mark, rather than as compositor-wide
    globals, is what makes it possible to run two visually unrelated marks
    (a near-square stacked logo and a 4.3:1 wide wordmark) through the same
    placement code without one mark's calibration leaking into the other's.

    scale_ranges: pattern name -> (frac_lo, frac_hi), where frac is the
    fraction of PAGE WIDTH the placed instance's width should span. This is
    mark-specific because the same frac produces a very different apparent
    size (and, for a wide mark, a very different HEIGHT) depending on the
    mark's own aspect ratio -- see ariatender_wide's scale_ranges below for
    the concrete case this bit.

    pattern_weights: pattern name -> relative weight, normalised at
    selection time (so callers need not pre-normalise, e.g. when only
    overriding one entry).
    """

    mark_id: str
    scale_ranges: Dict[str, Tuple[float, float]]
    pattern_weights: Dict[str, float]


def _load_stacked_mark(assets_dir: Path) -> Image.Image:
    """Loads the original AriaTender mark: logo crop + Persian subtitle crop,
    each background-removed (real alpha already usable) and cleaned by
    clean_stamp, then stacked at their real relative geometry."""
    logo = asset_prep.clean_stamp(str(assets_dir / LOGO_NAME))
    subtitle = asset_prep.clean_stamp(str(assets_dir / SUBTITLE_NAME))
    return asset_prep.combine_stamp(logo, subtitle)


def _load_wide_mark(assets_dir: Path) -> Image.Image:
    """Loads the new wide wordmark. Unlike the stacked mark's source crops,
    this file is a raw, fully-opaque screenshot (alpha=255 everywhere), so
    it goes through extract_flat_screenshot_stamp -- a colour-difference
    extraction, not an alpha-based one -- rather than clean_stamp. See that
    function's docstring in asset_prep.py for why, and for the measured
    calibration constants."""
    return asset_prep.extract_flat_screenshot_stamp(str(assets_dir / WIDE_WORDMARK_NAME))


_MARK_LOADERS: Dict[str, Callable[[Path], Image.Image]] = {
    STACKED_MARK_ID: _load_stacked_mark,
    WIDE_MARK_ID: _load_wide_mark,
}

# Registered marks. Each entry's scale_ranges/pattern_weights are the single
# source of truth for how that mark gets placed -- composite() no longer
# hard-codes any of this.
MARK_REGISTRY: Dict[str, MarkSpec] = {
    # The original mark. These ranges/weights are copied verbatim from the
    # pre-registry hard-coded values in _placements_*/_choose_pattern, so
    # registering it changes nothing about existing dataset generation.
    STACKED_MARK_ID: MarkSpec(
        mark_id=STACKED_MARK_ID,
        scale_ranges={
            "single": (0.60, 1.00),
            "oversize": (1.05, 1.9),
            "corner": (0.12, 0.28),
            "lattice": (0.07, 0.14),
            "diagonal": (0.07, 0.14),
        },
        pattern_weights={
            "single": 0.30,
            "oversize": 0.18,
            "lattice": 0.24,
            "diagonal": 0.16,
            "corner": 0.12,
        },
    ),
    # The new wide wordmark. Its aspect ratio is ~4.3:1 (2163x501 after
    # extraction) versus the stacked mark's near-square ~1:1 -- and `frac`
    # controls WIDTH as a fraction of page width, so the same frac used for
    # the stacked mark would make this mark's height collapse. Concretely:
    # at the stacked mark's lattice frac of 0.07, on a 1240px-wide page this
    # mark would render 87px wide and only ~20px tall -- illegible, and
    # small enough that individual glyph components risk being dropped by
    # labels.MIN_AREA_ABS_CAP (120px). Every range below is shifted up
    # relative to the stacked mark's so this mark stays legible and its
    # components stay comfortably above that floor at every pattern's
    # smallest end. pattern_weights are left equal to the stacked mark's --
    # no measurement suggests real documents favour one pattern differently
    # for a wide wordmark versus a stacked logo, so there is no basis yet
    # to diverge; revisit if/when real wide-mark documents are collected.
    WIDE_MARK_ID: MarkSpec(
        mark_id=WIDE_MARK_ID,
        scale_ranges={
            "single": (0.55, 1.00),
            "oversize": (1.05, 1.90),
            "corner": (0.20, 0.42),
            "lattice": (0.14, 0.28),
            "diagonal": (0.14, 0.28),
        },
        pattern_weights={
            "single": 0.30,
            "oversize": 0.18,
            "lattice": 0.24,
            "diagonal": 0.16,
            "corner": 0.12,
        },
    ),
}


@functools.lru_cache(maxsize=8)
def load_base_stamp(mark_id: str = STACKED_MARK_ID, assets_dir: Optional[str] = None) -> Image.Image:
    """Loads and cleans one registered mark's source asset(s) into an RGBA
    stamp.

    Cached (by (mark_id, assets_dir)) since both extraction paths --
    clean_stamp's connected-component analysis and feathering, and
    extract_flat_screenshot_stamp's distance-transform paper estimate -- are
    expensive enough that redoing them per sample would dominate dataset
    generation time. Returns a fresh copy per call so callers can mutate
    freely without corrupting the cached original.
    """
    if mark_id not in _MARK_LOADERS:
        raise ValueError(f"unknown mark_id {mark_id!r}; registered marks: {sorted(_MARK_LOADERS)}")
    d = Path(assets_dir) if assets_dir else DEFAULT_ASSETS_DIR
    return _MARK_LOADERS[mark_id](d)


def get_stamp(assets_dir: Optional[str] = None, mark_id: Optional[str] = None) -> Image.Image:
    """Public entry point: a fresh copy of one registered mark's stamp,
    safe to mutate.

    `assets_dir` stays the first positional parameter (rather than
    `mark_id`) so existing calls of the form `get_stamp(assets_dir)`
    continue to return the original stacked mark unchanged -- that call
    shape predates the mark registry and other code still uses it.
    """
    resolved_mark_id = mark_id if mark_id is not None else STACKED_MARK_ID
    return load_base_stamp(resolved_mark_id, assets_dir).copy()


def resolve_mark(mark: Union["MarkSpec", str, None]) -> "MarkSpec":
    """Normalises the `mark` argument composite() accepts (a MarkSpec, a
    registered mark_id string, or None) to a MarkSpec. None means "the
    original mark", both for backwards compatibility with callers written
    before a second mark existed and because it's a sensible default."""
    if mark is None:
        return MARK_REGISTRY[STACKED_MARK_ID]
    if isinstance(mark, MarkSpec):
        return mark
    if mark not in MARK_REGISTRY:
        raise ValueError(f"unknown mark {mark!r}; registered marks: {sorted(MARK_REGISTRY)}")
    return MARK_REGISTRY[mark]


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
        # Additive per-channel offset, applied to R/G/B independently. The
        # previous implementation read ONLY the red channel and wrote it
        # back to all three ("ink = r; r=g=b=ink") -- a harmless no-op for
        # the stacked mark, whose ink genuinely is flat gray (r==g==b
        # already), but for the wide wordmark's pink shield that would have
        # thrown away the red/green/blue difference entirely and rendered
        # every instance of it grey. Shifting each channel by the same
        # scalar preserves each pixel's chroma (its channel-to-channel
        # differences) while still moving its overall brightness, which is
        # what "the whole scan is a bit lighter/darker" should mean.
        rgb = np.dstack([
            np.array(r, dtype=np.int16),
            np.array(g, dtype=np.int16),
            np.array(b, dtype=np.int16),
        ])
        rgb = np.clip(rgb + tint_shift, 0, 255).astype(np.uint8)
        r = Image.fromarray(rgb[..., 0])
        g = Image.fromarray(rgb[..., 1])
        b = Image.fromarray(rgb[..., 2])
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

def _placements_single(w: int, h: int, rng: np.random.Generator, frac_range: Tuple[float, float]
                        ) -> Tuple[List[Tuple[float, float, float, float]], float]:
    frac = rng.uniform(*frac_range)
    angle = rng.uniform(-8, 8) if rng.random() < 0.7 else rng.uniform(20, 45) * rng.choice([-1, 1])
    cx = rng.uniform(0.5 - 0.15, 0.5 + 0.15) * w
    cy = rng.uniform(0.5 - 0.15, 0.5 + 0.15) * h
    return [(cx, cy, frac, angle)], frac


def _placements_oversize(w: int, h: int, rng: np.random.Generator, frac_range: Tuple[float, float]
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
    frac = rng.uniform(*frac_range)
    angle = rng.uniform(-10, 10) if rng.random() < 0.75 else rng.uniform(15, 40) * rng.choice([-1, 1])
    # Push the centre well off the middle so the clipping is asymmetric --
    # sometimes the subtitle is gone, sometimes half the logo.
    cx = rng.uniform(0.20, 0.80) * w
    cy = rng.uniform(0.18, 0.82) * h
    return [(cx, cy, frac, angle)], frac


def _placements_corner(w: int, h: int, rng: np.random.Generator, frac_range: Tuple[float, float]
                        ) -> Tuple[List[Tuple[float, float, float, float]], float]:
    frac = rng.uniform(*frac_range)
    angle = rng.uniform(-10, 10)
    margin = 0.06
    corner = rng.integers(0, 4)
    cx = margin * w if corner in (0, 3) else (1 - margin) * w
    cy = margin * h if corner in (0, 1) else (1 - margin) * h
    return [(cx, cy, frac, angle)], frac


def _placements_lattice(w: int, h: int, rng: np.random.Generator, diagonal: bool,
                         frac_range: Tuple[float, float]
                         ) -> Tuple[List[Tuple[float, float, float, float]], float]:
    frac = rng.uniform(*frac_range)
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


def _choose_pattern(rng: np.random.Generator, pattern: Optional[str],
                     pattern_weights: Dict[str, float]) -> str:
    if pattern is not None:
        assert pattern in PATTERNS, f"unknown pattern {pattern!r}"
        return pattern
    # Roughly matches observed real-world mix: large single marks and
    # lattices are both common; corner and diagonal are less so. Weights
    # come from the active MarkSpec rather than being fixed here, so a
    # differently-shaped mark can prefer a different pattern mix (today
    # both registered marks share the same mix -- see MARK_REGISTRY).
    weights = np.array([pattern_weights[p] for p in PATTERNS], dtype=np.float64)
    probs = weights / weights.sum()
    return str(rng.choice(PATTERNS, p=probs))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def composite(background: Image.Image, stamp: Image.Image,
              rng: np.random.Generator, pattern: Optional[str] = None,
              opacity_mult: Optional[float] = None,
              tint_shift: Optional[int] = None,
              mark: Union["MarkSpec", str, None] = None,
              ) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """Composites `stamp` onto `background` using a sampled placement
    pattern. Returns (composited_rgb [PIL RGB], alpha_map [float32 HxW,
    0..1], meta dict).

    `background` must be a PIL Image (any mode; converted to RGB).
    `stamp` is the RGBA stamp from get_stamp() -- note this is NOT derived
    from `mark` automatically; callers pick the stamp pixels and the mark's
    placement calibration separately (generate.py loads each registered
    mark's stamp once and passes both together per sample).
    `mark` selects which MarkSpec's scale_ranges/pattern_weights govern this
    call: a MarkSpec, a registered mark_id string, or None for the original
    stacked mark (see resolve_mark) -- this is what lets composite() place
    a mark whose aspect ratio and pattern preferences differ from the
    original without composite() itself knowing about specific marks.
    `opacity_mult` / `tint_shift`, if given, fix those sample-level values
    (used by the calibration self-test); otherwise opacity_mult is sampled
    ~U(0.5, 1.5) around the calibrated base alpha as specified in the task
    brief, and tint_shift ~U(-18, 18).
    """
    bg = background.convert("RGB")
    w, h = bg.size

    mark_spec = resolve_mark(mark)
    chosen_pattern = _choose_pattern(rng, pattern, mark_spec.pattern_weights)
    frac_range = mark_spec.scale_ranges[chosen_pattern]
    if chosen_pattern == "single":
        placements, base_frac = _placements_single(w, h, rng, frac_range)
    elif chosen_pattern == "corner":
        placements, base_frac = _placements_corner(w, h, rng, frac_range)
    elif chosen_pattern == "oversize":
        placements, base_frac = _placements_oversize(w, h, rng, frac_range)
    elif chosen_pattern == "lattice":
        placements, base_frac = _placements_lattice(w, h, rng, diagonal=False, frac_range=frac_range)
    else:  # diagonal
        placements, base_frac = _placements_lattice(w, h, rng, diagonal=True, frac_range=frac_range)

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
        "mark_id": mark_spec.mark_id,
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
                              clean: Optional[Image.Image] = None,
                              ) -> Tuple[Image.Image, np.ndarray, Optional[Image.Image]]:
    """Mild Gaussian blur (scanned look) and an occasional downscale/upscale
    round trip (screenshot resampling). Applied identically to the image and
    the alpha map so labels stay aligned with what's visible. JPEG
    re-encoding is deliberately NOT done here -- that happens once, at save
    time, in generate.py, after both train/val paths reuse this image.

    `clean` (optional) is the pre-composite background, as a PIL RGB image
    of the same size as `image` -- the alpha-regression training target. If
    given, the SAME sampled parameters (blur sigma, downscale factor) are
    applied to it too, via the same PIL GaussianBlur path used for `image`
    (not the cv2 path used for `alpha_map`, since `clean` is an RGB photo,
    not a coverage map). The RNG is consumed in exactly the same order/
    amount whether or not `clean` is passed -- every `rng.random()`/
    `rng.uniform()` call below is unconditional on `clean`'s presence -- so
    an existing `--seed` still reproduces byte-identical `image`/`alpha_map`
    output regardless of whether `--save-clean` is also given. Returns
    (image, alpha_map, clean); `clean` in the return is None when not
    passed in.
    """
    w, h = image.size

    if rng.random() < 0.5:
        sigma = float(rng.uniform(0.3, 1.1))
        image = image.filter(ImageFilter.GaussianBlur(sigma))
        alpha_map = cv2_gaussian_blur(alpha_map, sigma)
        if clean is not None:
            clean = clean.filter(ImageFilter.GaussianBlur(sigma))

    if rng.random() < 0.3:
        factor = float(rng.uniform(0.5, 0.85))
        small = (max(1, int(w * factor)), max(1, int(h * factor)))
        image = image.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
        alpha_img = Image.fromarray((np.clip(alpha_map, 0, 1) * 255).astype(np.uint8))
        alpha_img = alpha_img.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
        alpha_map = np.asarray(alpha_img, dtype=np.float32) / 255.0
        if clean is not None:
            clean = clean.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)

    return image, alpha_map, clean


def cv2_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(arr, (k, k), sigma)
