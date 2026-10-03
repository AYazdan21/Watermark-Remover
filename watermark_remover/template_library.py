"""Template library: save / load / list watermark templates and make them
usable by ``stamp_fit``.

Layout (one folder per template under ``assets/stamps/library/`` by default)::

    <name>/template.png       RGBA uint8. A = coverage (peak 255). v1: RGB = ink colour
                              of the pixel's region (constant per region). v2: A =
                              opacity / opacity_peak and RGB = the PER-PIXEL ink colour
    <name>/regions.png        uint8 label map, 0 = outside, 1..K = ink region
                              (absent = one region); v2 uses it only for the
                              per-region Stamp Fit removal option
    <name>/meta.json          see ``save_template`` / the Template Builder
                              (``builder_version`` 1 or 2; v2 adds ``opacity_peak``,
                              ``ink_luminance``, ``instance_width_px``, ...)
    <name>/preview.png        human-readable preview
    <name>/build_report.json  per-page registration table from the build

A loaded template always offers ``alpha`` (coverage, peak 1), ``ink`` (H, W, 3
float, the ink at every pixel) and ``opacity_peak`` (the opacity a coverage of 1
stands for: ``meta.opacity_peak`` for v2, the median build strength for v1, a
nominal value for bare PNGs), so Method 5's per-pixel removal treats v1 and v2
templates alike.

Deliberate design choice -- registering into ``stamp_fit``
----------------------------------------------------------
``stamp_fit.templates()`` is ``lru_cache(maxsize=1)`` and returns a mutable
dict, and every helper of Stamp Fit (``_size``, ``_resized``, ``render``,
``_evidence``, ``_subpixel``, ``remove_stamps``) looks a template up by name in
that dict. ``register_with_stamp_fit`` therefore inserts a library template into
that dict under a unique key (``lib:<name>@<mtime_ns>`` or ``png:<path>@<mtime_ns>``)
so all of Stamp Fit's machinery -- sub-pixel refinement, per-region strength and
ink fit, edge profile and the exact inverse -- works for library templates
WITHOUT editing ``stamp_fit.py``. The mtime in the key keeps
``stamp_fit._resized.cache`` from ever serving a stale, rebuilt template.
"""

import json
import os
import re
import time

import cv2
import numpy as np
from PIL import Image

from . import stamp_fit
from .config import BASE_DIR

DEFAULT_LIBRARY_DIR = os.path.join(BASE_DIR, "assets", "stamps", "library")
BUILTIN_PNGS = {
    "AriaTender wide (built-in)": os.path.join(BASE_DIR, "assets", "stamps", "ariatender_wide.png"),
    "AriaTender stacked (built-in)": os.path.join(BASE_DIR, "assets", "stamps", "ariatender_stacked.png"),
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


def resolve_path(p, default=None):
    """Quoted paths, either slash style; relative paths resolve against the repo root."""
    if p is None or not str(p).strip():
        return default
    p = str(p).strip().strip('"').strip("'").strip()
    p = os.path.expanduser(p)
    if not os.path.isabs(p):
        p = os.path.join(BASE_DIR, p)
    return os.path.normpath(p)


def library_root(library_dir=None):
    return resolve_path(library_dir, DEFAULT_LIBRARY_DIR)


def is_safe_name(name):
    return bool(name) and bool(_SAFE_NAME.match(str(name))) and str(name) not in (".", "..")


def list_templates(library_dir=None):
    """Folder names holding a valid template.png + meta.json, sorted."""
    root = library_root(library_dir)
    if not os.path.isdir(root):
        return []
    out = []
    for n in sorted(os.listdir(root)):
        d = os.path.join(root, n)
        if os.path.isfile(os.path.join(d, "template.png")) and os.path.isfile(os.path.join(d, "meta.json")):
            out.append(n)
    return out


def _regions_from_labels(alpha, labels):
    n = int(labels.max()) if labels is not None and labels.size else 0
    if n <= 1:
        return {"mark": alpha.astype(np.float32)}
    return {f"region{i}": np.where(labels == i, alpha, 0).astype(np.float32) for i in range(1, n + 1)}


def save_template(name, alpha, region_labels, ink_rgb_per_region, meta, library_dir=None,
                  overwrite=False, ink_map=None):
    """Writes template.png (+ regions.png when K > 1) and meta.json. Returns the folder.
    ``alpha`` float 0-1 (H, W); ``region_labels`` uint8 (H, W) or None;
    ``ink_rgb_per_region`` list of (r, g, b) 0-255, index i = region i+1.
    ``ink_map`` (H, W, 3) uint8 (v2): the per-pixel ink; when given it is the RGB
    of template.png and ``ink_rgb_per_region`` only fills meta-level summaries."""
    if not is_safe_name(name):
        raise ValueError(f"'{name}' is not a safe template name (letters, digits, _ - . ; up to 64 chars)")
    folder = os.path.join(library_root(library_dir), name)
    if os.path.exists(os.path.join(folder, "template.png")) and not overwrite:
        raise FileExistsError(f"template '{name}' already exists in {os.path.dirname(folder)} (use overwrite)")
    os.makedirs(folder, exist_ok=True)
    H, W = alpha.shape
    labels = region_labels if region_labels is not None else (alpha > 0).astype(np.uint8)
    K = max(1, int(labels.max()))
    if ink_map is not None:
        rgb = np.ascontiguousarray(np.asarray(ink_map, np.uint8))
    else:
        rgb = np.zeros((H, W, 3), np.uint8)
        for i in range(1, K + 1):
            ink = ink_rgb_per_region[min(i - 1, len(ink_rgb_per_region) - 1)] if len(ink_rgb_per_region) else (100, 100, 100)
            rgb[labels == i] = np.asarray(ink, np.uint8)
        rgb[labels == 0] = np.asarray(ink_rgb_per_region[0] if len(ink_rgb_per_region) else (100, 100, 100), np.uint8)
    rgba = np.dstack([rgb, np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(os.path.join(folder, "template.png"))
    reg_path = os.path.join(folder, "regions.png")
    if K > 1:
        Image.fromarray(labels.astype(np.uint8), "L").save(reg_path)
    elif os.path.exists(reg_path):
        os.remove(reg_path)
    meta = dict(meta)
    meta.setdefault("name", name)
    meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("builder_version", 1)
    meta["template_size"] = [int(W), int(H)]
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return folder


def _load_bare_png(path):
    rgba = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0
    a = rgba[..., 3] / max(float(rgba[..., 3].max()), 1e-6)
    keep = cv2.dilate((a >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = np.where(keep, a, 0).astype(np.float32)
    sat = rgba[..., :3].max(2) - rgba[..., :3].min(2)
    colour = np.where(sat > 0.06, a, 0).astype(np.float32)
    grey = np.where(sat <= 0.06, a, 0).astype(np.float32)
    if colour.sum() > 0 and grey.sum() > 0:
        regions = {"colour": colour, "grey": grey}
    else:
        regions = {"mark": a}
    name = os.path.splitext(os.path.basename(path))[0]
    H, W = a.shape
    meta = dict(name=name, builder_version=0, source="bare png", template_size=[W, H], rel_width=None)
    return dict(alpha=a, regions=regions, meta=meta, path=os.path.abspath(path), kind="png",
                mtime_ns=os.stat(path).st_mtime_ns, ink=np.ascontiguousarray(rgba[..., :3]).astype(np.float32),
                opacity_peak=float(stamp_fit.DEFAULT_STRENGTH))


def load_template(name_or_path, library_dir=None):
    """Library name, template folder path, or bare RGBA PNG path ->
    dict(alpha, regions, meta, path, kind, mtime_ns). Bare PNGs get the same
    speck rule and saturation split ``stamp_fit.templates()`` applies; library
    templates do not (the builder cleans its own output)."""
    s = str(name_or_path).strip().strip('"').strip("'")
    if s in BUILTIN_PNGS:
        return _load_bare_png(BUILTIN_PNGS[s])
    cand = resolve_path(s)
    if os.path.isfile(cand) and cand.lower().endswith((".png", ".webp")):
        return _load_bare_png(cand)
    folder = None
    if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "template.png")):
        folder = cand
    else:
        lib = os.path.join(library_root(library_dir), s)
        if os.path.isdir(lib) and os.path.isfile(os.path.join(lib, "template.png")):
            folder = lib
    if folder is None:
        raise FileNotFoundError(f"template '{name_or_path}' not found (library: {library_root(library_dir)})")
    tp = os.path.join(folder, "template.png")
    rgba = np.asarray(Image.open(tp).convert("RGBA"))
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    rp = os.path.join(folder, "regions.png")
    labels = np.asarray(Image.open(rp).convert("L")) if os.path.isfile(rp) else None
    if labels is not None and labels.shape != alpha.shape:
        labels = None
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return dict(alpha=alpha, regions=_regions_from_labels(alpha, labels), meta=meta,
                path=os.path.abspath(folder), kind="lib", mtime_ns=os.stat(tp).st_mtime_ns,
                labels=labels, ink=rgba[..., :3].astype(np.float32) / 255.0,
                opacity_peak=_opacity_peak(meta))


def _opacity_peak(meta):
    """Opacity that a coverage of 1 stands for: ``opacity_peak`` of a v2 build, the
    median per-page strength of a v1 build, else Stamp Fit's nominal strength."""
    try:
        v = meta.get("opacity_peak")
        if v is None:
            v = (meta.get("strength") or {}).get("median")
        v = float(v)
        if 0.02 <= v <= 0.95:
            return v
    except (TypeError, ValueError):
        pass
    return float(stamp_fit.DEFAULT_STRENGTH)


def template_label(tpl):
    return str(tpl["meta"].get("name") or os.path.basename(tpl["path"]))


def register_with_stamp_fit(tpl):
    """Insert ``tpl`` into ``stamp_fit.templates()`` (see module docstring);
    returns the key under which Stamp Fit's helpers find it."""
    if tpl["kind"] == "lib":
        prefix = f"lib:{os.path.basename(tpl['path'])}@"
    else:
        prefix = f"png:{tpl['path']}@"
    key = f"{prefix}{tpl['mtime_ns']}"
    store = stamp_fit.templates()
    for old in [k for k in store if k.startswith(prefix) and k != key]:
        store.pop(old, None)           # a rebuilt template: drop the stale entry
    if key not in store:
        store[key] = {"alpha": tpl["alpha"], "regions": dict(tpl["regions"])}
    return key


def builtin_names():
    return list(BUILTIN_PNGS)
