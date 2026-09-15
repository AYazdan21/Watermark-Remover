"""CLI driver: composites AriaTender watermark stamp(s) onto background
documents and writes a YOLO-seg dataset (images, polygon labels, continuous
alpha maps, and a JSONL metadata sidecar).

Two marks are registered in compositor.MARK_REGISTRY: the original
"ariatender_stacked" (logo + Persian subtitle) and "ariatender_wide" (the
wide two-tone wordmark). By default every positive sample draws from a
mix of both (equally weighted) -- see --marks / --mark-weights / --mark-id
below to change that.

Backgrounds are read from a plain directory of images (glob'd recursively) --
deliberately NOT importing backgrounds.py / gen_persian_docs.py, which are
being written in parallel by another agent. Point --backgrounds-dir at
whatever directory they end up filling; this script only cares that it
contains image files. If no such directory is given (or it's empty), a
small set of procedural placeholder backgrounds is generated in-memory so
the pipeline is runnable and testable end-to-end -- clearly a fallback path,
never silently used for a "real" dataset build (a warning is printed).

Usage:
    python generate.py --backgrounds-dir ../../dataset/some_real_dir \
        --out C:/scratch/wm_out --n 2000 --seed 0

Output layout:
    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt   (YOLO-seg polygons; empty file = negative)
    <out>/alpha/{train,val}/*.png    (continuous alpha coverage maps, 0-255)
    <out>/clean/{train,val}/*.png    (only with --save-clean: the pre-composite,
                                       pre-JPEG background page -- the alpha-
                                       regression training target; lossless PNG)
    <out>/data.yaml                  (ultralytics dataset config)
    <out>/meta.jsonl                 (one JSON record per sample)
    <out>/generation_args.json       (every CLI arg this run was called with,
                                       including --seed, for reproducing it)

Alpha-regression target, honest note: the single-layer compositing model
`observed = a*ink + (1-a)*clean` is EXACT before scan augmentation (see
compositor.py's module docstring) -- alpha and the RGB blend come from the
same "over" accumulation, so they're consistent by construction. Blur/
resample (apply_scan_augmentations) then make the equation only approximate
at mark edges, since blur doesn't commute with the multiply. `clean` is
saved BEFORE JPEG re-encoding of the composite, so the alpha-regression
target is the true page, not a JPEG-degraded one -- the network learns
watermark removal, not JPEG artifact repair; residual JPEG noise in the
`images/` composite is an irreducible floor on measured recovery error, not
something the network is trained to fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

import compositor
import labels as labels_mod

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Watermark-Remover
PROTECTED_DATASET_DIR = (REPO_ROOT / "dataset").resolve()


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------

def collect_background_paths(backgrounds_dir: Optional[Path]) -> List[Path]:
    if backgrounds_dir is None or not Path(backgrounds_dir).exists():
        return []
    return sorted(
        p for p in Path(backgrounds_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def make_placeholder_background(rng: np.random.Generator) -> Image.Image:
    """Plain procedural paper-like background: NOT a stand-in for real
    document data, only exists so the compositing/labeling pipeline can be
    exercised end-to-end without a backgrounds directory yet."""
    w = int(rng.integers(900, 1400))
    h = int(rng.integers(1100, 1700))
    base = int(rng.integers(245, 253))
    arr = np.full((h, w, 3), base, dtype=np.float32)
    arr += rng.normal(0, 2.0, (h, w, 1)).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    if rng.random() < 0.6:
        draw = ImageDraw.Draw(img)
        n_lines = int(rng.integers(15, 35))
        for i in range(n_lines):
            y = int((i + 1) * h / (n_lines + 1))
            g = int(rng.integers(225, 242))
            draw.line([(int(w * 0.08), y), (int(w * 0.92), y)], fill=(g, g, g), width=1)
    return img


def _maybe_resize_background(img: Image.Image, rng: np.random.Generator,
                              max_dim: int = 1800, min_dim: int = 700) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        s = max_dim / longest
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    elif longest < min_dim:
        s = min_dim / longest
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(out_dir: Path, backgrounds_dir: Optional[Path], n_samples: int,
                      val_frac: float = 0.15, negative_frac: float = 0.15,
                      seed: int = 0, jpeg_quality_range=(55, 95),
                      assets_dir: Optional[str] = None,
                      mark_ids: Optional[List[str]] = None,
                      mark_weights: Optional[List[float]] = None,
                      class_name: str = "watermark",
                      save_clean: bool = False) -> dict:
    out_dir = Path(out_dir)
    _guard_output_path(out_dir)

    rng = np.random.default_rng(seed)

    # Which mark(s) this run can draw from. Defaulting to both registered
    # marks (rather than just the original) is deliberate: the whole point
    # of this multi-mark extension is that a real dataset build should mix
    # them, not that every caller must remember to ask for both.
    if mark_ids is None:
        mark_ids = list(compositor.MARK_REGISTRY.keys())
    if mark_weights is None:
        mark_weights = [1.0] * len(mark_ids)
    if len(mark_weights) != len(mark_ids):
        raise ValueError(f"mark_weights ({len(mark_weights)}) must match mark_ids ({len(mark_ids)})")
    mark_probs = np.array(mark_weights, dtype=np.float64)
    mark_probs = mark_probs / mark_probs.sum()

    # Load every requested stamp ONCE, up front. load_base_stamp is already
    # lru_cache'd, but re-extracting per sample would still mean re-running
    # (and re-copying) the extraction machinery n_samples times instead of
    # len(mark_ids) times -- for extract_flat_screenshot_stamp specifically,
    # that's a distance-transform over a 2184x534 image, not something to
    # redo thousands of times when the source pixels never change.
    stamps = {mid: compositor.get_stamp(assets_dir, mark_id=mid) for mid in mark_ids}

    bg_paths = collect_background_paths(backgrounds_dir)
    use_placeholders = len(bg_paths) == 0
    if use_placeholders:
        print(f"[generate] WARNING: no backgrounds found in {backgrounds_dir!r} -- "
              f"falling back to procedural placeholder backgrounds. This is a "
              f"pipeline smoke-test path, not a real dataset build.",
              file=sys.stderr)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "alpha" / split).mkdir(parents=True, exist_ok=True)
        if save_clean:
            (out_dir / "clean" / split).mkdir(parents=True, exist_ok=True)

    order = np.arange(n_samples)
    rng.shuffle(order)
    n_val = int(round(n_samples * val_frac)) if n_samples >= 4 else 0
    val_ids = set(order[:n_val].tolist())

    meta_records = []
    n_negative = 0
    n_empty_positive = 0
    empty_positive_ids = []
    mark_counts = {mid: 0 for mid in mark_ids}
    for i in range(n_samples):
        split = "val" if i in val_ids else "train"
        is_negative = rng.random() < negative_frac

        if use_placeholders:
            bg = make_placeholder_background(rng)
            bg_file = "<placeholder>"
        else:
            bg_path = bg_paths[int(rng.integers(0, len(bg_paths)))]
            bg = Image.open(bg_path).convert("RGB")
            bg = _maybe_resize_background(bg, rng)
            bg_file = str(bg_path)

        if is_negative:
            n_negative += 1
            composited = bg.convert("RGB")
            # Same image as the composite for negatives -- there is nothing
            # to remove, so the "clean" target and the observed page are
            # identical (both are just `composited`; sharing the object is
            # safe since apply_scan_augmentations never mutates in place).
            clean_img = composited if save_clean else None
            alpha_map = np.zeros((bg.height, bg.width), dtype=np.float32)
            sample_mark_id = None
            comp_meta = {"mark_id": None, "pattern": None, "opacity_mult": None, "tint_shift": 0,
                         "base_scale_frac": None, "n_instances": 0, "instances": []}
        else:
            sample_mark_id = str(rng.choice(mark_ids, p=mark_probs))
            mark_counts[sample_mark_id] += 1
            composited, alpha_map, comp_meta = compositor.composite(
                bg, stamps[sample_mark_id], rng, mark=sample_mark_id)
            # The resized background BEFORE compositing -- composite() never
            # mutates `bg` itself (it works on its own float32 copy), so this
            # is a fresh, independent RGB copy of the true page.
            clean_img = bg.convert("RGB") if save_clean else None

        composited, alpha_map, clean_img = compositor.apply_scan_augmentations(
            composited, alpha_map, rng, clean=clean_img)

        mask, polygons = labels_mod.alpha_to_polygons(alpha_map)
        yolo_lines = labels_mod.polygons_to_yolo_lines(polygons, composited.width, composited.height)

        # A positive that yields no polygons is the single worst thing this
        # pipeline can emit: the image visibly contains the watermark but the
        # label says it does not, so the model is actively trained to ignore
        # it. This was a real bug (an absolute mask threshold shattered dim
        # small-scale marks into sub-min-area specks), so it is now counted
        # and surfaced rather than written out silently.
        if not is_negative and not yolo_lines:
            n_empty_positive += 1
            empty_positive_ids.append(f"wm_{i:06d}")

        jpeg_quality = int(rng.integers(jpeg_quality_range[0], jpeg_quality_range[1] + 1))
        sample_id = f"wm_{i:06d}"

        img_path = out_dir / "images" / split / f"{sample_id}.jpg"
        label_path = out_dir / "labels" / split / f"{sample_id}.txt"
        alpha_path = out_dir / "alpha" / split / f"{sample_id}.png"
        clean_path = out_dir / "clean" / split / f"{sample_id}.png" if save_clean else None

        composited.convert("RGB").save(img_path, "JPEG", quality=jpeg_quality)
        labels_mod.write_yolo_label(label_path, yolo_lines)
        Image.fromarray((np.clip(alpha_map, 0, 1) * 255).astype(np.uint8)).save(alpha_path)
        if save_clean:
            # Lossless: this is the alpha-regression training target, saved
            # BEFORE the composite's JPEG re-encoding above -- see the
            # module docstring's honest note on why.
            clean_img.convert("RGB").save(clean_path, "PNG")

        meta_records.append({
            "id": sample_id,
            "split": split,
            "image": _rel(img_path, out_dir),
            "label": _rel(label_path, out_dir),
            "alpha": _rel(alpha_path, out_dir),
            "clean": _rel(clean_path, out_dir) if save_clean else None,
            "class_name": class_name,
            # The mark ACTUALLY used for this sample -- None for negatives.
            # This must come from comp_meta (i.e. from what composite()
            # itself resolved), not from a single CLI-wide constant: once a
            # sample can draw from more than one mark, a constant would be
            # wrong for every sample that didn't happen to get the first one.
            "mark_id": comp_meta.get("mark_id"),
            "negative": is_negative,
            "background_file": bg_file,
            "jpeg_quality": jpeg_quality,
            "pattern": comp_meta.get("pattern"),
            "opacity_mult": comp_meta.get("opacity_mult"),
            "tint_shift": comp_meta.get("tint_shift"),
            "base_scale_frac": comp_meta.get("base_scale_frac"),
            "n_instances_placed": comp_meta.get("n_instances", 0),
            "n_polygons": len(polygons),
            "instances": comp_meta.get("instances", []),
        })

    meta_path = out_dir / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec) + "\n")

    data_yaml_path = out_dir / "data.yaml"
    data_yaml_path.write_text(
        "path: " + str(out_dir).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n",
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "n_samples": n_samples,
        "n_negative": n_negative,
        "n_empty_positive": n_empty_positive,
        "empty_positive_ids": empty_positive_ids[:20],
        "n_train": sum(1 for r in meta_records if r["split"] == "train"),
        "n_val": sum(1 for r in meta_records if r["split"] == "val"),
        "used_placeholders": use_placeholders,
        "mark_counts": mark_counts,
        "meta_path": str(meta_path),
        "data_yaml": str(data_yaml_path),
    }


def _rel(p: Path, base: Path) -> str:
    return str(p.relative_to(base)).replace("\\", "/")


def _guard_output_path(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    if resolved == PROTECTED_DATASET_DIR or PROTECTED_DATASET_DIR in resolved.parents:
        raise SystemExit(
            f"Refusing to write into {PROTECTED_DATASET_DIR} (the real data "
            f"directory) or any path under it. Choose a different --out."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output dataset root (must not be under dataset/).")
    ap.add_argument("--backgrounds-dir", default=None, help="Directory of background images (glob'd recursively). "
                                                              "Falls back to procedural placeholders if omitted/empty.")
    ap.add_argument("--n", type=int, default=200, help="Number of samples to generate.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--negative-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg-quality-min", type=int, default=55)
    ap.add_argument("--jpeg-quality-max", type=int, default=95)
    ap.add_argument("--assets-dir", default=None, help="Override the _wm_extract_deliverable source-crop directory.")
    ap.add_argument("--mark-id", default=None,
                     help="Single-mark shorthand: generate using only this one registered mark "
                          "(e.g. 'ariatender_stacked' or 'ariatender_wide'). Ignored if --marks is "
                          "also given -- see --marks.")
    ap.add_argument("--marks", default=None,
                     help="Comma-separated list of registered mark ids to draw from per positive "
                          "sample (default: 'ariatender_stacked,ariatender_wide', i.e. both). "
                          "If both --marks and --mark-id are given, --marks wins and a note is "
                          "printed saying so.")
    ap.add_argument("--mark-weights", default=None,
                     help="Comma-separated floats, one per entry in --marks, giving each mark's "
                          "relative sampling weight (default: equal weight for every mark). "
                          "Normalised automatically, so they need not sum to 1.")
    ap.add_argument("--save-clean", action="store_true",
                     help="Also save the pre-composite, pre-JPEG background page as "
                          "clean/{split}/{id}.png -- the alpha-regression training target. "
                          "Off by default (extra disk + write time); does not change --seed "
                          "reproducibility of images/alpha when omitted.")
    args = ap.parse_args(argv)

    if args.marks is not None:
        mark_ids = [m.strip() for m in args.marks.split(",") if m.strip()]
        if args.mark_id is not None:
            print(f"[generate] NOTE: both --marks and --mark-id were given; --marks "
                  f"({mark_ids}) wins, --mark-id ({args.mark_id!r}) is ignored.", file=sys.stderr)
    elif args.mark_id is not None:
        mark_ids = [args.mark_id]
    else:
        mark_ids = None  # generate_dataset defaults this to every registered mark.

    mark_weights = None
    if args.mark_weights is not None:
        mark_weights = [float(x.strip()) for x in args.mark_weights.split(",") if x.strip()]

    summary = generate_dataset(
        out_dir=Path(args.out),
        backgrounds_dir=Path(args.backgrounds_dir) if args.backgrounds_dir else None,
        n_samples=args.n,
        val_frac=args.val_frac,
        negative_frac=args.negative_frac,
        seed=args.seed,
        jpeg_quality_range=(args.jpeg_quality_min, args.jpeg_quality_max),
        assets_dir=args.assets_dir,
        mark_ids=mark_ids,
        mark_weights=mark_weights,
        save_clean=args.save_clean,
    )

    # Every CLI arg this run was called with (incl. seed, n, backgrounds
    # dir, save_clean) -- so a dataset build can always be reproduced or
    # audited later without digging through shell history.
    args_path = Path(args.out) / "generation_args.json"
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    summary["generation_args"] = str(args_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
