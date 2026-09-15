"""Deterministic replay of ``generate_dataset.py`` to recover EXACT
per-pixel alpha-regression targets (``clean``, per-pixel accumulated
watermark opacity ``A``) for the Kaggle "train and test2" dataset, whose
reference JPEGs/masks give no such ground truth on their own (see
``scripts/kaggle_dataset/README.md`` for why).

STANDALONE by design: stdlib + numpy + PIL + cv2 only, no repo imports.

How it works
------------
``generate_dataset.py`` seeds ``random``/``np.random`` with a fixed seed
and then draws every augmentation parameter (background shuffle, watermark
combo assignment, scale/rotation/blur/opacity/blend-mode/perspective/
feather/tiling/JPEG quality/noise/colour-shift/resize) from those two
global generators, in a fixed call order. If we run the *exact same file*
against the *exact same input assets* with the *exact same library
versions*, every draw reproduces bit-for-bit, so we can recover the exact
per-image compositing that produced each reference JPEG -- including the
otherwise-unrecorded per-pixel alpha and the pre-post-processing clean
background -- by monkeypatching four of its functions to *also* carry a
parallel "clean" image and an alpha accumulator ``A`` through the same
sequence of operations, using ONLY parameters that are already fixed by
the point they're read (no extra ``random``/``np.random`` calls are ever
added, removed, or reordered by the patches -- see the exactness checks
below, which are what prove that promise held).

This script imports the generator module fresh via ``importlib`` (so it
never pollutes ``sys.modules`` with a stale copy across repeated runs in
the same interpreter... except it does register under a fixed name --
re-running this script's ``run_replay`` twice in one process reloads the
module each time, which is fine since generator functions are pure
w.r.t. the patches installed just before each run), points its
``BG_DIR``/``WM_DIR``/``OUTPUT_DIR`` globals at the given paths, and
calls its unmodified ``main()``.

Exactness proof, per verified image (every ``--verify-every`` images):
JPEG-encode the replayed composite in memory at the SAME quality
``generate_dataset.py`` drew for it, decode it back, and compare pixel-
for-pixel against the reference JPEG decoded the same way. Only an
environment whose Pillow/libjpeg encoder is bit-identical to whatever
produced the reference dataset will show ``exact: true`` for every
verified image -- see the module docstring's honesty note and this run's
``replay_report.json`` for which local Python actually achieves that.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import PIL
    _PIL_VERSION = getattr(PIL, "__version__", "unknown")
except Exception:  # pragma: no cover
    _PIL_VERSION = "unknown"


class _StopReplay(BaseException):
    """Raised (as a BaseException, NOT Exception) once ``--limit`` images
    have been produced, so it escapes the generator's own
    ``except Exception: continue`` inside ``main()``'s per-image loop
    instead of being silently swallowed there -- see module docstring.
    Never caught anywhere except around the ``main()`` call in
    ``run_replay``.
    """


# ---------------------------------------------------------------------------
# Generator module loading
# ---------------------------------------------------------------------------

def _load_generator_module(path: Path):
    spec = importlib.util.spec_from_file_location("kaggle_replay_generator", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Histogram-based percentile (memory-safe over arbitrarily many images)
# ---------------------------------------------------------------------------

def _percentile_from_hist(hist_counts: np.ndarray, bin_edges: np.ndarray, q: float) -> float:
    total = int(hist_counts.sum())
    if total == 0:
        return 0.0
    target = q / 100.0 * total
    cum = np.cumsum(hist_counts)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(bin_edges) - 2)
    return float(bin_edges[idx + 1])


# ---------------------------------------------------------------------------
# Core replay
# ---------------------------------------------------------------------------

def run_replay(args) -> dict:
    out_root = Path(args.out)
    gen_out_dir = out_root / "_replay_generator_out"
    clean_root = out_root / "clean"
    alpha_root = out_root / "alpha"

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    for split in ("train", "val"):
        (clean_root / split).mkdir(parents=True, exist_ok=True)
        (alpha_root / split).mkdir(parents=True, exist_ok=True)
    gen_out_dir.mkdir(parents=True, exist_ok=True)

    gen_path = Path(args.generator) if args.generator else (Path(__file__).parent / "generate_dataset.py")
    mod = _load_generator_module(gen_path)
    mod.BG_DIR = str(Path(args.backgrounds_dir))
    mod.WM_DIR = str(Path(args.watermarks_dir))
    mod.OUTPUT_DIR = str(gen_out_dir)

    orig_composite = mod.composite_watermark_on_background
    orig_gen_single = mod.generate_single_image

    limit = args.limit
    state = {"clean": None, "A": None}
    captured_results = []  # one dict per successfully-generated image, in idx order
    per_image_records = []  # verification / stats records, one per image, same order

    hist_counts = np.zeros(256, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, 257)
    stats_acc = {"total_pixels": 0, "total_pos_pixels": 0, "total_gt_09": 0, "per_image_max": []}

    # -- patched composite_watermark_on_background ---------------------------
    def patched_composite(bg, wm_aug, opacity, blend_mode, position):
        bg_w, bg_h = bg.size
        wm_w, wm_h = wm_aug.size
        px, py = position
        x1 = max(0, px)
        y1 = max(0, py)
        x2 = min(bg_w, px + wm_w)
        y2 = min(bg_h, py + wm_h)
        if x2 > x1 and y2 > y1:
            wm_x1 = x1 - px
            wm_y1 = y1 - py
            wm_x2 = x2 - px
            wm_y2 = y2 - py
            wm_crop = wm_aug.crop((wm_x1, wm_y1, wm_x2, wm_y2))
            if wm_crop.size[0] > 0 and wm_crop.size[1] > 0:
                wm_a = wm_crop.split()[3]
                # Exactly mirrors generate_dataset.composite_watermark_on_background
                # lines computing wm_a_np -- no RNG involved, deterministic
                # given (wm_aug, opacity, position), which are already fixed
                # by the time this function is called.
                wm_a_np = np.asarray(wm_a, dtype=np.float32) / 255.0 * opacity
                A = state["A"]
                region = A[y1:y2, x1:x2]
                A[y1:y2, x1:x2] = 1.0 - (1.0 - region) * (1.0 - wm_a_np)
        return orig_composite(bg, wm_aug, opacity, blend_mode, position)

    # -- patched apply_post_processing ---------------------------------------
    def patched_post_processing(img):
        # MUST mirror generate_dataset.apply_post_processing LINE BY LINE:
        # same random()/np.random calls, in the same order, with the same
        # short-circuiting. Any deviation here desyncs random.Random's
        # internal state from the reference run and silently corrupts every
        # draw for the rest of the dataset -- the per-image exact-decode
        # checks in this script's report are what prove this mirror is
        # faithful; never trust it without them.
        clean = state["clean"]
        A = state["A"]

        img_np = np.array(img, dtype=np.float32)
        clean_np = np.array(clean, dtype=np.float32)

        # 1. Gaussian noise (60% of the time) -- identical noise array
        # added to both img and clean, then independently clipped (a real
        # source of small clipping-induced nonlinearity right at 0/255,
        # same as JPEG's own inability to represent out-of-range values).
        if random.random() < 0.6:
            noise_sigma = random.uniform(*mod.NOISE_SIGMA_RANGE)
            if noise_sigma > 1:
                noise = np.random.normal(0, noise_sigma, img_np.shape)
                img_np = np.clip(img_np + noise, 0, 255)
                clean_np = np.clip(clean_np + noise, 0, 255)

        # 2. Colour-temperature shift (30% of the time) -- identical shift.
        if random.random() < 0.3:
            shift = random.uniform(-15, 15)
            img_np[:, :, 0] = np.clip(img_np[:, :, 0] + shift, 0, 255)
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] - shift * 0.7, 0, 255)
            clean_np[:, :, 0] = np.clip(clean_np[:, :, 0] + shift, 0, 255)
            clean_np[:, :, 2] = np.clip(clean_np[:, :, 2] - shift * 0.7, 0, 255)

        img_out = Image.fromarray(img_np.astype(np.uint8))
        clean_out = Image.fromarray(clean_np.astype(np.uint8))

        # 3. Down/up resize (20% of the time) -- same factor/filter applied
        # to img, clean AND the alpha accumulator (PIL mode "F", BILINEAR).
        if random.random() < 0.2:
            factor = random.uniform(0.4, 0.8)
            small_size = (max(64, int(img_out.width * factor)), max(64, int(img_out.height * factor)))
            img_out = img_out.resize(small_size, Image.BILINEAR).resize(img_out.size, Image.BILINEAR)
            clean_out = clean_out.resize(small_size, Image.BILINEAR).resize(clean_out.size, Image.BILINEAR)
            a_img = Image.fromarray(np.ascontiguousarray(A, dtype=np.float32), mode="F")
            a_img = a_img.resize(small_size, Image.BILINEAR).resize(mod.TARGET_SIZE, Image.BILINEAR)
            A = np.asarray(a_img, dtype=np.float32)

        state["clean"] = clean_out
        state["A"] = np.clip(A, 0.0, 1.0).astype(np.float32)
        return img_out

    # -- patched generate_single_image ---------------------------------------
    def patched_generate_single_image(idx, bg_path, watermarks, wm_combo):
        if limit is not None and idx >= limit:
            raise _StopReplay()
        clean0 = Image.open(bg_path).convert("RGB").resize(mod.TARGET_SIZE, Image.LANCZOS)
        state["clean"] = clean0
        state["A"] = np.zeros((mod.TARGET_SIZE[1], mod.TARGET_SIZE[0]), dtype=np.float32)
        result = orig_gen_single(idx, bg_path, watermarks, wm_combo)
        captured_results.append({
            "idx": idx,
            "img_filename": result["img_filename"],
            "bg_filename": result["bg_filename"],
            "wm_combo": list(result["wm_combo"]),
            "has_watermark": result["has_watermark"],
            "annotations": result["annotations"],
        })
        return result

    # -- patched save_as_jpeg_with_quality ------------------------------------
    def patched_save_as_jpeg(img, path):
        # Same draw, same position in the RNG sequence as the original --
        # we just redirect what gets written.
        quality = random.randint(*mod.JPEG_QUALITY_RANGE)
        path = Path(path)
        split = path.parent.name
        stem = path.stem

        clean = state["clean"]
        A = np.clip(state["A"], 0.0, 1.0).astype(np.float32)

        clean.save(clean_root / split / f"{stem}.png", "PNG")
        alpha_u8 = np.clip(np.round(A * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(alpha_u8, mode="L").save(alpha_root / split / f"{stem}.png", "PNG")

        if captured_results:
            captured_results[-1]["split"] = split
            captured_results[-1]["stem"] = stem
            captured_results[-1]["quality"] = int(quality)

        rec = {
            "idx": captured_results[-1]["idx"] if captured_results else None,
            "stem": stem,
            "split": split,
            "quality": int(quality),
            "alpha_max": float(A.max()) if A.size else 0.0,
            "alpha_p50": float(np.percentile(A, 50)),
            "alpha_p99": float(np.percentile(A, 99)),
        }
        pos = A[A > 0]
        rec["alpha_pos_p99"] = float(np.percentile(pos, 99)) if pos.size else 0.0

        stats_acc["per_image_max"].append(rec["alpha_max"])
        stats_acc["total_pixels"] += int(A.size)
        stats_acc["total_pos_pixels"] += int(pos.size)
        stats_acc["total_gt_09"] += int(np.sum(A > 0.9))
        if pos.size:
            h, _ = np.histogram(pos, bins=256, range=(0.0, 1.0))
            hist_counts[:] += h

        n_so_far = len(per_image_records)
        do_verify = (n_so_far % max(1, args.verify_every) == 0)
        if do_verify:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=quality)
            buf.seek(0)
            replay_decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)

            ref_path = Path(args.reference_root) / "images" / split / f"{stem}.jpg"
            if ref_path.exists():
                ref_decoded = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.int16)
                if replay_decoded.shape == ref_decoded.shape:
                    diff = np.abs(replay_decoded - ref_decoded)
                    rec["max_abs_diff"] = int(diff.max())
                    rec["mean_abs_diff"] = float(diff.mean())
                    rec["exact"] = bool(diff.max() == 0)
                else:
                    rec["max_abs_diff"] = None
                    rec["mean_abs_diff"] = None
                    rec["exact"] = False
                    rec["shape_mismatch"] = [list(replay_decoded.shape), list(ref_decoded.shape)]

                pre_jpeg = np.asarray(img.convert("RGB"), dtype=np.float64)
                ref_f = ref_decoded.astype(np.float64)
                if pre_jpeg.shape == ref_f.shape:
                    mse = float(np.mean((pre_jpeg - ref_f) ** 2))
                    rec["psnr_pre_jpeg_vs_ref"] = (float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse))
                rec["verified"] = True
            else:
                rec["verified"] = False
                rec["verify_error"] = f"reference image not found: {ref_path}"
        else:
            rec["verified"] = False

        per_image_records.append(rec)
        return None

    mod.composite_watermark_on_background = patched_composite
    mod.apply_post_processing = patched_post_processing
    mod.generate_single_image = patched_generate_single_image
    mod.save_as_jpeg_with_quality = patched_save_as_jpeg

    old_argv = sys.argv
    sys.argv = [str(gen_path)]
    t0 = time.time()
    try:
        mod.main()
    except _StopReplay:
        pass
    finally:
        sys.argv = old_argv
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Cross-check against reference CSV / COCO attributes. Built from our
    # OWN captured_results rather than gen_out_dir's own annotations/*.json,
    # because when --limit triggers an early _StopReplay, main()'s own
    # post-loop COCO/CSV-writing code (which sits after the per-image for
    # loop) never runs -- captured_results is populated incrementally
    # during the loop instead and works identically with or without
    # --limit.
    # ------------------------------------------------------------------
    csv_mismatches = []
    attr_mismatches = []
    ATTR_KEYS = ("scale", "rotation", "opacity", "blend_mode", "is_tiled")

    ref_csv_path = Path(args.reference_root) / "annotations" / "image_labels.csv"
    ref_rows = {}
    if ref_csv_path.exists():
        with open(ref_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref_rows[row["image_name"]] = row
    else:
        csv_mismatches.append({"reason": f"reference CSV not found: {ref_csv_path}"})

    ref_coco = {}
    for split in ("train", "val"):
        p = Path(args.reference_root) / "annotations" / f"{split}.json"
        if not p.exists():
            attr_mismatches.append({"reason": f"reference COCO file not found: {p}"})
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        by_img = {img["id"]: [] for img in data["images"]}
        for ann in data["annotations"]:
            by_img.setdefault(ann["image_id"], []).append(ann)
        for anns in by_img.values():
            anns.sort(key=lambda a: a["id"])
        ref_coco[split] = by_img

    for r in captured_results:
        fname = r["img_filename"]
        split = r.get("split")

        ref_row = ref_rows.get(fname)
        if ref_row is None:
            csv_mismatches.append({"image": fname, "reason": "missing in reference CSV"})
        elif ref_row["background_source"] != r["bg_filename"]:
            csv_mismatches.append({
                "image": fname, "reason": "background_source mismatch",
                "reference": ref_row["background_source"], "replay": r["bg_filename"],
            })

        idx = r["idx"]
        ref_anns = ref_coco.get(split, {}).get(idx) if split else None
        if ref_anns is None:
            if split is not None:
                attr_mismatches.append({"image": fname, "reason": f"image id {idx} not found in reference {split}.json"})
            continue
        my_anns = r["annotations"]
        if len(ref_anns) != len(my_anns):
            attr_mismatches.append({
                "image": fname, "reason": "annotation count mismatch",
                "reference_count": len(ref_anns), "replay_count": len(my_anns),
            })
            continue
        for j, (ra, ma) in enumerate(zip(ref_anns, my_anns)):
            rattrs = ra.get("attributes", {})
            mattrs = ma.get("attributes", {})
            for k in ATTR_KEYS:
                rv, mv = rattrs.get(k), mattrs.get(k)
                if rv != mv:
                    attr_mismatches.append({
                        "image": fname, "ann_index": j, "key": k,
                        "reference": rv, "replay": mv,
                    })

    # ------------------------------------------------------------------
    # Alpha distribution stats
    # ------------------------------------------------------------------
    per_image_max = np.asarray(stats_acc["per_image_max"], dtype=np.float64)
    alpha_stats = {
        "per_image_max_p50": float(np.percentile(per_image_max, 50)) if per_image_max.size else 0.0,
        "per_image_max_p99": float(np.percentile(per_image_max, 99)) if per_image_max.size else 0.0,
        "per_image_max_true_max": float(per_image_max.max()) if per_image_max.size else 0.0,
        "positive_pixel_p99": _percentile_from_hist(hist_counts, bin_edges, 99.0),
        "frac_pixels_gt_0.9": (stats_acc["total_gt_09"] / stats_acc["total_pixels"]) if stats_acc["total_pixels"] else 0.0,
        "frac_pixels_positive": (stats_acc["total_pos_pixels"] / stats_acc["total_pixels"]) if stats_acc["total_pixels"] else 0.0,
    }

    # ------------------------------------------------------------------
    # Disk usage
    # ------------------------------------------------------------------
    def _dir_bytes(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    n_total = len(captured_results)
    disk_bytes = _dir_bytes(clean_root) + _dir_bytes(alpha_root)

    verified = [r for r in per_image_records if r.get("verified")]
    n_exact = sum(1 for r in verified if r.get("exact"))
    max_abs_diffs = [r["max_abs_diff"] for r in verified if r.get("max_abs_diff") is not None]
    mean_abs_diffs = [r["mean_abs_diff"] for r in verified if r.get("mean_abs_diff") is not None]

    n_train = sum(1 for r in captured_results if r.get("split") == "train")
    n_val = sum(1 for r in captured_results if r.get("split") == "val")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version": sys.version,
        "pillow_version": _PIL_VERSION,
        "numpy_version": np.__version__,
        "cv2_version": cv2.__version__,
        "generator_path": str(gen_path),
        "backgrounds_dir": str(args.backgrounds_dir),
        "watermarks_dir": str(args.watermarks_dir),
        "reference_root": str(args.reference_root),
        "limit": limit,
        "verify_every": args.verify_every,
        "elapsed_seconds": elapsed,
        "seconds_per_image": (elapsed / n_total) if n_total else None,
        "disk_bytes_total": disk_bytes,
        "disk_bytes_per_image": (disk_bytes / n_total) if n_total else None,
        "counts": {"total_processed": n_total, "train": n_train, "val": n_val},
        "exact_match": {
            "n_verified": len(verified),
            "n_exact": n_exact,
            "rate": (n_exact / len(verified)) if verified else None,
            "max_abs_diff_overall": max(max_abs_diffs) if max_abs_diffs else None,
            "mean_abs_diff_mean": (sum(mean_abs_diffs) / len(mean_abs_diffs)) if mean_abs_diffs else None,
        },
        "csv_mismatches": {"count": len(csv_mismatches), "examples": csv_mismatches[:10]},
        "attr_mismatches": {"count": len(attr_mismatches), "examples": attr_mismatches[:10]},
        "alpha_stats": alpha_stats,
        "images": per_image_records,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "replay_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)

    ok = (report["exact_match"]["rate"] in (None, 1.0)) and not csv_mismatches and not attr_mismatches
    report["_ok"] = ok
    return report


def _print_summary(report: dict) -> None:
    print("=" * 70)
    print("  REPLAY REPORT SUMMARY")
    print("=" * 70)
    print(f"  Python: {report['python_version'].splitlines()[0]}")
    print(f"  Pillow {report['pillow_version']}  numpy {report['numpy_version']}  cv2 {report['cv2_version']}")
    c = report["counts"]
    print(f"  Processed: {c['total_processed']} (train={c['train']}, val={c['val']})")
    em = report["exact_match"]
    rate = f"{em['rate']*100:.2f}%" if em["rate"] is not None else "n/a"
    print(f"  Exact-decode match: {em['n_exact']}/{em['n_verified']} ({rate})")
    print(f"  max_abs_diff (over verified): {em['max_abs_diff_overall']}")
    print(f"  mean_abs_diff (avg over verified): {em['mean_abs_diff_mean']}")
    print(f"  CSV mismatches: {report['csv_mismatches']['count']}")
    print(f"  COCO attribute mismatches: {report['attr_mismatches']['count']}")
    a = report["alpha_stats"]
    print(f"  Alpha (per-image max): p50={a['per_image_max_p50']:.4f} p99={a['per_image_max_p99']:.4f} "
          f"max={a['per_image_max_true_max']:.4f}")
    print(f"  Alpha (positive pixels p99): {a['positive_pixel_p99']:.4f}  "
          f"frac>0.9: {a['frac_pixels_gt_0.9']:.6f}")
    print(f"  Elapsed: {report['elapsed_seconds']:.1f}s "
          f"({report['seconds_per_image']:.3f}s/image)" if report['seconds_per_image'] else "")
    if report["disk_bytes_per_image"]:
        print(f"  Disk: {report['disk_bytes_total']/1e6:.1f} MB total, "
              f"{report['disk_bytes_per_image']/1e3:.1f} KB/image")
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Deterministically replay generate_dataset.py to recover exact alpha/clean targets."
    )
    ap.add_argument("--generator", default=None, help="Path to generate_dataset.py (default: sibling file)")
    ap.add_argument("--backgrounds-dir", required=True)
    ap.add_argument("--watermarks-dir", required=True)
    ap.add_argument("--reference-root", required=True,
                     help="A wm_dataset_out dir (contains images/, annotations/) to verify replay against")
    ap.add_argument("--out", required=True, help="Output targets root (clean/, alpha/, replay_report.json)")
    ap.add_argument("--limit", type=int, default=None, help="Stop after the first N indices")
    ap.add_argument("--verify-every", type=int, default=1, help="Verify every Kth processed image (default 1)")
    ap.add_argument("--overwrite", action="store_true", help="Delete --out first")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    report = run_replay(args)
    sys.exit(0 if report.get("_ok") else 1)


if __name__ == "__main__":
    main()
