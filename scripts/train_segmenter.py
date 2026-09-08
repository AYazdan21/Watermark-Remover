"""Train + evaluate a YOLO11-seg watermark segmentation model.

Trains a single-class ("watermark") Ultralytics YOLO11-seg model on the
synthetic dataset produced by scripts/wm_dataset/generate.py
(wm_dataset_out/), then scores it -- via the SAME evaluation code path --
on three domains so results are directly comparable:

    train         wm_dataset_out/images/train (+labels/train)  -- the exact
                  images the model was fit on. Mainly a memorization sanity
                  check: if metrics here are near-perfect but val/test are
                  much worse, the model is memorizing samples rather than
                  learning the mark.
    val_synthetic wm_dataset_out/images/val (+labels/val)       -- held out,
                  but still synthetic (same compositing pipeline/artifacts
                  as train).
    test_real     --test-dir (default wm_testset/)              -- real,
                  hand-labelled documents, wholly different distribution.

Two families of metrics are computed for every domain:
  (a) Ultralytics native: mask mAP50, mask mAP50-95, precision, recall.
      Standard, comparable to other published segmentation work.
  (b) Pixel-level (scripts/wm_dataset/seg_metrics.py): IoU, Dice, pixel
      precision/recall over the union mask, plus a false-positive rate
      computed ONLY on negative (no-watermark) images. These matter more
      here: the segmenter's output feeds a pixel-level alpha-unmixing
      removal step (watermark_remover/unmixer.py), so "which pixels are
      watermark" predicts removal quality far better than a
      detection-centric mAP does, and the single failure mode this whole
      project has fought hardest is touching pixels that are not
      watermark -- which aggregate IoU/Dice alone can hide (see
      seg_metrics.py's module docstring).

The val_synthetic-vs-test_real gap is the headline: training data is
synthetic, real documents are the target distribution, and a large gap
there is the signature of the model learning compositing artifacts (JPEG
ringing, a too-clean edge, a background tell) rather than the watermark
itself. That delta is computed and printed loudly -- see --gap-threshold.

--------------------------------------------------------------------------
Two landmines this script exists specifically to route around
--------------------------------------------------------------------------

1. wm_dataset_out/data.yaml has a RELATIVE `path: wm_dataset_out`.
   Ultralytics resolves a relative `path` against ITS OWN
   `settings['datasets_dir']` (some unrelated directory configured once,
   globally, on this machine) -- NOT the current working directory -- via
   ultralytics.data.utils.check_det_dataset. Worse, if a directory that
   happens to be named `wm_dataset_out` exists relative to the CURRENT
   working directory, `Path("wm_dataset_out").exists()` short-circuits that
   resolution and it silently uses THAT instead, so the failure mode
   depends on which directory you happened to run this script from. This
   script never passes the original data.yaml to ultralytics: it always
   reads --data-dir, resolves it to an ABSOLUTE path, and writes a derived
   temporary data.yaml with that absolute `path` (see build_data_yaml()).
   The resolved paths are printed before training/eval so you can verify
   them at a glance.

2. The target GPU is an NVIDIA GeForce MX150 (~2GB VRAM) -- a low-end
   laptop GPU. imgsz=1024 with a normal batch size WILL OOM on it. There is
   a genuine accuracy tension worth knowing about: the watermark includes a
   thin-ring shield logo whose strokes get destroyed at low resolution (a
   768px-long edge of a page holding a small stamp can put that ring's
   stroke at 1-2px, right where JPEG ringing and antialiasing already live),
   so a bigger --imgsz genuinely helps quality -- but the hardware caps it.
   Defaults below (yolo11n-seg, imgsz=768, batch=4, AMP on, workers=2) are
   chosen to actually run on this card; every one is a CLI flag so you can
   push --imgsz up (with --batch down) on a bigger GPU. A CUDA
   out-of-memory error during train/val/predict is caught and reported as
   an actionable message instead of a raw traceback.

--------------------------------------------------------------------------
Usage (from the Watermark-Remover directory; adjust the python path if
your venv lives somewhere else)
--------------------------------------------------------------------------

Train, then evaluate on train / val_synthetic / test_real:
    ../.venv/Scripts/python.exe scripts/train_segmenter.py \\
        --data-dir wm_dataset_out --test-dir wm_testset \\
        --model yolo11n-seg.pt --epochs 100 --imgsz 768 --batch 4

Re-score an already-trained checkpoint without retraining:
    ../.venv/Scripts/python.exe scripts/train_segmenter.py \\
        --eval-only --weights runs/segment/wm_seg/weights/best.pt \\
        --data-dir wm_dataset_out --test-dir wm_testset

Both write runs/segment/<name>/metrics.json, a handful of
image/GT/prediction visualization PNGs under
runs/segment/<name>/eval_visualizations/<domain>/, and print a summary
table to stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "wm_dataset"))
import seg_metrics  # noqa: E402  (local module, see sys.path insert above)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ["watermark"]
NC = 1

# Absolute IoU/Dice drop from val_synthetic -> test_real above which the
# domain-gap warning is printed loudly. Tunable via --gap-threshold; 0.15
# is a fairly generous "this is clearly more than noise" bar for a ~35 to
# a few dozen image real test set.
DEFAULT_GAP_THRESHOLD = 0.15


class WMScriptError(RuntimeError):
    """Raised for user-actionable configuration problems (missing/empty
    dataset directories, bad flags, etc). Caught in main() and printed
    without a traceback -- these are expected, fixable situations, not
    bugs in this script."""


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Landmine #1: absolute-path dataset yaml resolution
# ---------------------------------------------------------------------------

def build_data_yaml(
    root_abs: Path,
    val_rel: str,
    train_rel: Optional[str],
    tmp_dir: Path,
    tag: str,
) -> Path:
    """Writes a derived data.yaml with an ABSOLUTE `path`, sidestepping
    ultralytics' settings['datasets_dir'] resolution entirely (see module
    docstring, landmine #1). `train_rel` is required by ultralytics'
    check_det_dataset even for a val-only call, so when there is no
    meaningful train split (the flat --test-dir case) callers pass the
    same value as `val_rel`.
    """
    if not root_abs.is_absolute():
        raise WMScriptError(f"internal error: build_data_yaml got a non-absolute root: {root_abs}")
    data = {
        "path": str(root_abs),
        "train": train_rel or val_rel,
        "val": val_rel,
        "nc": NC,
        "names": CLASS_NAMES,
    }
    out_path = tmp_dir / f"data_{tag}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_path


def load_source_data_yaml(data_dir: Path) -> dict:
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        raise WMScriptError(
            f"--data-dir {data_dir} has no data.yaml (expected {yaml_path}). "
            f"Point --data-dir at the output of scripts/wm_dataset/generate.py."
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key in ("train", "val"):
        if key not in data:
            raise WMScriptError(f"{yaml_path} is missing required key '{key}:'")
    return data


# ---------------------------------------------------------------------------
# Dataset directory discovery / validation
# ---------------------------------------------------------------------------

def labels_rel_from_images_rel(images_rel: str) -> str:
    """Derives a split's labels/ path from its own RELATIVE images/ path
    (e.g. "images/train" -> "labels/train"). Operates on the relative
    string only, not the full absolute path -- an ancestor directory that
    happens to contain "images" anywhere (e.g. a username) would otherwise
    make a full-path substring replace silently point at the wrong
    folder."""
    return images_rel.replace("images", "labels", 1)


def _list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def discover_split(images_dir: Path, labels_dir: Path, domain_label: str) -> List[Path]:
    """Validates an images/+labels/ pair and returns the sorted image list.
    Raises WMScriptError with an instructive message if the directory is
    missing or empty -- silently reporting zero metrics on an unpopulated
    test set would be worse than an error (this is exactly the situation
    for a fresh wm_testset/ the user hasn't filled in yet)."""
    problems: List[str] = []
    if not images_dir.exists():
        problems.append(f"missing images directory: {images_dir}")
    if not labels_dir.exists():
        problems.append(f"missing labels directory: {labels_dir}")
    image_paths = _list_images(images_dir) if images_dir.exists() else []
    if images_dir.exists() and not image_paths:
        problems.append(f"images directory is empty (no {sorted(IMAGE_EXTS)} files): {images_dir}")

    if problems:
        raise WMScriptError(
            f"\n{domain_label} is not ready for evaluation:\n  - "
            + "\n  - ".join(problems)
            + "\n\nExpected layout: <dir>/images/*.jpg|png + <dir>/labels/*.txt "
            "(YOLO-seg polygons, class 0; an EMPTY .txt file means 'no watermark' "
            "-- a true negative, not a missing label).\n"
            "See wm_testset/README.md for the exact format if this is the test set."
        )
    return image_paths


# ---------------------------------------------------------------------------
# OOM handling
# ---------------------------------------------------------------------------

def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def run_guarded(fn, call_kwargs: dict, stage: str):
    """Runs `fn(**call_kwargs)`, turning a CUDA OOM into an actionable
    WMScriptError instead of a raw stack trace. Anything else re-raises
    unchanged (only OOM has a scripted, obvious fix). `imgsz`/`batch` for
    the error message are read out of call_kwargs itself, so callers pass
    exactly one kwargs dict -- no risk of the duplicate-keyword collision
    you'd get from also passing imgsz=/batch= alongside **call_kwargs."""
    imgsz = call_kwargs.get("imgsz", "?")
    batch = call_kwargs.get("batch", "?")
    try:
        return fn(**call_kwargs)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, re-narrowed below
        if _is_cuda_oom(exc):
            raise WMScriptError(
                f"\nCUDA out of memory during {stage} (imgsz={imgsz}, batch={batch}).\n"
                "This GPU (e.g. an MX150-class ~2GB card) cannot fit this "
                "combination. Try, in order of first resort:\n"
                f"  1. reduce --batch (currently {batch}) -- e.g. --batch {max(1, batch // 2)}\n"
                f"  2. reduce --imgsz (currently {imgsz}) -- e.g. --imgsz {max(320, (imgsz // 2 // 32) * 32)}\n"
                "  3. train on a bigger GPU, or with --device cpu (slow, but always fits)\n"
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Evaluation: native ultralytics metrics + pixel metrics, one code path
# ---------------------------------------------------------------------------

def native_metrics_from_result(metrics_obj) -> dict:
    """Extracts the scalar fields we care about from an ultralytics
    SegmentMetrics object (the return value of YOLO.val() for a -seg
    model). See ultralytics.utils.metrics.Metric for map/map50/mp/mr."""
    return {
        "mask_mAP50": float(metrics_obj.seg.map50),
        "mask_mAP50_95": float(metrics_obj.seg.map),
        "precision": float(metrics_obj.seg.mp),
        "recall": float(metrics_obj.seg.mr),
        "box_mAP50": float(metrics_obj.box.map50),
        "box_mAP50_95": float(metrics_obj.box.map),
    }


def make_visualization_panel(
    image_path: Path,
    gt_polys: List[np.ndarray],
    pred_polys: List[np.ndarray],
    out_path: Path,
    max_side: int = 640,
) -> None:
    """Saves an [image | GT overlay | prediction overlay] side-by-side PNG
    for eyeballing. Downscaled to max_side on the long edge purely so the
    panel files stay small; overlays are drawn before the downscale so line
    width stays visually reasonable."""
    img = Image.open(image_path).convert("RGB")

    def overlay(polys: List[np.ndarray], color) -> Image.Image:
        im = img.copy()
        draw = ImageDraw.Draw(im)
        for poly in polys:
            pts = [(float(x), float(y)) for x, y in poly]
            if len(pts) >= 2:
                draw.polygon(pts, outline=color, width=max(2, img.width // 400))
        return im

    gt_panel = overlay(gt_polys, (40, 200, 40))
    pred_panel = overlay(pred_polys, (230, 40, 40))

    scale = min(1.0, max_side / max(img.width, img.height))
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    panels = [img.resize(new_size), gt_panel.resize(new_size), pred_panel.resize(new_size)]

    gap = 6
    combined = Image.new(
        "RGB", (new_size[0] * 3 + gap * 2, new_size[1]), (255, 255, 255)
    )
    for i, p in enumerate(panels):
        combined.paste(p, (i * (new_size[0] + gap), 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)


def evaluate_domain(
    model,
    domain_key: str,
    domain_label: str,
    images_dir: Path,
    labels_dir: Path,
    native_yaml: Path,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    """Evaluates `model` on one domain (train / val_synthetic / test_real)
    via a single shared code path, returning both metric families. This
    same function is called for all three domains -- the only thing that
    differs between calls is which images/labels/yaml it's pointed at."""
    print(f"\n{'=' * 70}\nEvaluating domain: {domain_label} ({images_dir})\n{'=' * 70}")

    image_paths = discover_split(images_dir, labels_dir, domain_label)
    print(f"  {len(image_paths)} images found.")

    # (a) Ultralytics native metrics (mask mAP50/50-95, precision, recall).
    native_name = f"{args.name}_eval_{domain_key}"
    val_kwargs = dict(
        data=str(native_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(run_dir),
        name=native_name,
        plots=args.native_plots,
        save_json=False,
        verbose=False,
        exist_ok=True,
    )
    native_result = run_guarded(
        model.val, val_kwargs, stage=f"native val() on {domain_label}",
    )
    native = native_metrics_from_result(native_result)
    print(
        f"  native: mask mAP50={native['mask_mAP50']:.4f} "
        f"mAP50-95={native['mask_mAP50_95']:.4f} "
        f"P={native['precision']:.4f} R={native['recall']:.4f}"
    )

    # (b) Pixel-level metrics + visualizations, via a single predict() pass.
    accumulator = seg_metrics.PixelMetricAccumulator()
    label_warnings: List[str] = []
    vis_dir = run_dir / "eval_visualizations" / domain_key
    n_vis_written = 0

    predict_kwargs = dict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=min(args.batch, 8),
        verbose=False,
        stream=True,
    )
    results_iter = run_guarded(
        model.predict, predict_kwargs, stage=f"predict() on {domain_label}",
    )

    for image_path, r in zip(image_paths, results_iter):
        h, w = r.orig_shape
        label_path = labels_dir / (image_path.stem + ".txt")
        gt_polys, warns = seg_metrics.read_yolo_seg_polygons(label_path, w, h)
        label_warnings.extend(warns)
        gt_mask = seg_metrics.polygons_to_binary_mask(gt_polys, h, w)
        is_negative = len(gt_polys) == 0

        pred_polys: List[np.ndarray] = []
        if r.masks is not None and len(r.masks.xy) > 0:
            pred_polys = [poly for poly in r.masks.xy if len(poly) >= 3]
        pred_mask = seg_metrics.polygons_to_binary_mask(pred_polys, h, w)

        accumulator.update(gt_mask, pred_mask, is_negative)

        if n_vis_written < args.num_vis:
            try:
                make_visualization_panel(
                    image_path, gt_polys, pred_polys,
                    vis_dir / f"{image_path.stem}.png",
                )
                n_vis_written += 1
            except Exception as exc:  # noqa: BLE001 - visualization must never abort evaluation
                print(f"  WARNING: could not write visualization for {image_path.name}: {exc}")

    if label_warnings:
        print(f"  WARNING: {len(label_warnings)} malformed label line(s) skipped, e.g.:")
        for w in label_warnings[:5]:
            print(f"    {w}")

    pixel = accumulator.compute()
    print(
        f"  pixel:  IoU={pixel['iou']:.4f} Dice={pixel['dice']:.4f} "
        f"P={pixel['pixel_precision']:.4f} R={pixel['pixel_recall']:.4f} "
        f"| negatives={pixel['n_negative_images']} "
        f"FP-pixel-rate={_fmt_opt(pixel['negative_fp_pixel_rate'])} "
        f"FP-image-rate={_fmt_opt(pixel['negative_fp_image_rate'])}"
    )
    print(f"  wrote {n_vis_written} visualization panel(s) to {vis_dir}")

    return {
        "n_images": len(image_paths),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "native": native,
        "pixel": pixel,
        "n_label_warnings": len(label_warnings),
    }


def _fmt_opt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


# ---------------------------------------------------------------------------
# Domain-gap summary
# ---------------------------------------------------------------------------

def compute_gap(a: dict, b: dict) -> dict:
    """delta = a - b, on the pixel-metric fields that matter most for the
    overfitting/domain-gap check. Positive delta means `a` scored higher."""
    pa, pb = a["pixel"], b["pixel"]
    return {
        "iou_delta": pa["iou"] - pb["iou"],
        "dice_delta": pa["dice"] - pb["dice"],
        "mask_mAP50_delta": a["native"]["mask_mAP50"] - b["native"]["mask_mAP50"],
    }


def print_domain_gap(label_a: str, label_b: str, gap: dict, threshold: float) -> None:
    flagged = abs(gap["iou_delta"]) >= threshold or abs(gap["dice_delta"]) >= threshold
    banner = "!" * 70
    if flagged:
        print(f"\n{banner}\nLARGE GAP: {label_a} vs {label_b}\n{banner}")
    else:
        print(f"\n--- gap: {label_a} vs {label_b} ---")
    print(
        f"  IoU delta:        {gap['iou_delta']:+.4f}\n"
        f"  Dice delta:        {gap['dice_delta']:+.4f}\n"
        f"  mask mAP50 delta:  {gap['mask_mAP50_delta']:+.4f}"
        f"   (threshold={threshold:.2f})"
    )
    if flagged:
        print(
            "  This gap is at or above --gap-threshold. If val_synthetic is high\n"
            "  and test_real is low, the model likely learned synthetic\n"
            "  compositing artifacts (JPEG ringing, an edge that's too clean,\n"
            "  a background tell) rather than the watermark itself -- treat\n"
            "  train-set metrics with suspicion and prioritize collecting more\n"
            "  / more varied real labelled data over more synthetic epochs."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="wm_dataset_out",
                     help="YOLO-seg dataset root produced by scripts/wm_dataset/generate.py "
                          "(must contain data.yaml, images/{train,val}, labels/{train,val}). "
                          "Default: wm_dataset_out")
    ap.add_argument("--test-dir", default="wm_testset",
                     help="Real hand-labelled test set: <dir>/images/*.jpg|png + "
                          "<dir>/labels/*.txt, flat (no split subfolders). Default: wm_testset")
    ap.add_argument("--weights", default=None,
                     help="Checkpoint (.pt) path. Required with --eval-only. Without "
                          "--eval-only, overrides --model as the training starting point "
                          "(e.g. to resume/continue-fine-tune from a prior run).")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                     help="Base COCO-pretrained starting weights for training (ignored if "
                          "--weights is given, or in --eval-only mode). Default: yolo11n-seg.pt "
                          "(the nano variant -- smallest/fastest, lowest VRAM; the right "
                          "starting point for a ~2GB laptop GPU).")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4,
                     help="Training/eval batch size. Default 4 -- conservative for a "
                          "~2GB-VRAM GPU (e.g. MX150) at --imgsz 768. Drop to 2 or 1 if you "
                          "hit a CUDA out-of-memory error.")
    ap.add_argument("--imgsz", type=int, default=768,
                     help="Training/inference image size. Default 768: a compromise between "
                          "the thin-ring shield logo needing resolution to survive (bigger is "
                          "better for that) and a ~2GB GPU's VRAM ceiling (1024 WILL OOM at any "
                          "reasonable batch size on such a card). Raise this on a bigger GPU.")
    ap.add_argument("--device", default=None,
                     help="Ultralytics device string ('0' for first CUDA GPU, 'cpu', etc). "
                          "Default: None (ultralytics auto-selects: GPU if available, else CPU).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs/segment",
                     help="Ultralytics project dir (also where metrics.json and "
                          "eval_visualizations/ are written, under --name). Default: runs/segment")
    ap.add_argument("--name", default="wm_seg",
                     help="Run name under --project. Default: wm_seg")
    ap.add_argument("--eval-only", action="store_true",
                     help="Skip training; just evaluate --weights on all three domains. Use "
                          "this to re-score a checkpoint without retraining.")
    ap.add_argument("--conf", type=float, default=0.25,
                     help="Confidence threshold used for eval predictions (native val() and "
                          "the pixel-metric/visualization predict() pass). Default: 0.25")
    ap.add_argument("--iou", type=float, default=0.7,
                     help="NMS IoU threshold for eval predictions. Default: 0.7 (ultralytics "
                          "default).")
    ap.add_argument("--workers", type=int, default=2,
                     help="Dataloader worker processes. Default: 2 -- kept low because "
                          "ultralytics/PyTorch multi-worker DataLoaders are noticeably less "
                          "reliable on Windows than Linux (higher counts have been observed to "
                          "hang or thrash); raise if your setup handles it fine.")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                     help="Automatic mixed precision during training. Default: on -- roughly "
                          "halves activation memory, which is the difference between fitting "
                          "and OOMing on a ~2GB GPU at --imgsz 768. Use --no-amp only to debug "
                          "a suspected precision issue.")
    ap.add_argument("--patience", type=int, default=30,
                     help="Early-stopping patience (epochs with no val improvement). Default: "
                          "30 -- the training set is small (a couple hundred synthetic images), "
                          "so it can overfit well before --epochs is reached.")
    ap.add_argument("--num-vis", type=int, default=6,
                     help="Number of [image | GT | prediction] visualization panels to save "
                          "per domain. Default: 6")
    ap.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD,
                     help=f"Absolute IoU/Dice delta between val_synthetic and test_real above "
                          f"which the domain-gap warning is printed loudly. Default: "
                          f"{DEFAULT_GAP_THRESHOLD}")
    ap.add_argument("--no-test-eval", action="store_true",
                     help="Skip the real test set entirely (do not require --test-dir to exist "
                          "or be labelled). Use when the real images have no ground-truth labels "
                          "yet: accuracy metrics are undefined without labels, so train first and "
                          "inspect predictions qualitatively with scripts/predict_overlays.py.")
    ap.add_argument("--skip-train-eval", action="store_true",
                     help="Skip evaluating on the images/train split (the memorization sanity "
                          "check). Evaluation still runs on val_synthetic and test_real.")
    ap.add_argument("--native-plots", action="store_true",
                     help="Have ultralytics also write its own PR-curve/confusion-matrix plots "
                          "during each native val() call. Off by default to save time/IO; this "
                          "script's own eval_visualizations/ panels are the primary output.")
    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.eval_only and not args.weights:
        print("ERROR: --eval-only requires --weights <path to a trained .pt checkpoint>.")
        return 2

    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    test_dir = Path(args.test_dir).resolve()

    print("Resolved dataset paths (landmine #1 mitigation -- these are ABSOLUTE, "
          "independent of ultralytics' settings['datasets_dir'] and of cwd):")
    print(f"  data-dir : {data_dir}")
    print(f"  test-dir : {test_dir}")

    try:
        source_yaml = load_source_data_yaml(data_dir)
        train_rel = str(source_yaml["train"])
        val_rel = str(source_yaml["val"])
        print(f"  train images -> {(data_dir / train_rel).resolve()}")
        print(f"  val images   -> {(data_dir / val_rel).resolve()}")
        print(f"  test images  -> {(test_dir / 'images').resolve()}")
    except WMScriptError as exc:
        print(f"\nERROR: {exc}")
        return 1

    # Validate every dataset directory that will be needed -- INCLUDING the
    # (possibly still-placeholder) --test-dir -- before doing anything
    # expensive. Without this, an unpopulated wm_testset/ would only be
    # discovered after a full (possibly multi-hour, on an MX150) training
    # run, right at the last evaluation step. Fail fast instead.
    try:
        if not args.skip_train_eval:
            discover_split(
                data_dir / train_rel, data_dir / labels_rel_from_images_rel(train_rel),
                "train split (--data-dir)",
            )
        discover_split(
            data_dir / val_rel, data_dir / labels_rel_from_images_rel(val_rel),
            "val split (--data-dir)",
        )
        if not args.no_test_eval:
            discover_split(test_dir / "images", test_dir / "labels", "test set (--test-dir)")
    except WMScriptError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print("All dataset directories present and non-empty.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="wm_seg_yaml_"))
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: ultralytics is not importable in this Python environment: {exc}")
        return 1

    try:
        # ---- resolve model / weights ----------------------------------
        if args.eval_only:
            weights_path = Path(args.weights).resolve()
            if not weights_path.exists():
                raise WMScriptError(f"--weights not found: {weights_path}")
            print(f"\n--eval-only: loading {weights_path} (training skipped).")
            model = YOLO(str(weights_path))
        else:
            start_weights = args.weights if args.weights else args.model
            print(f"\nStarting weights: {start_weights}")
            model = YOLO(start_weights)

            train_yaml = build_data_yaml(data_dir, val_rel, train_rel, tmp_dir, "train")
            print(f"Training data.yaml (derived, absolute path): {train_yaml}")
            print(f"  contents: {train_yaml.read_text(encoding='utf-8')}")

            train_kwargs = dict(
                data=str(train_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                amp=args.amp,
                patience=args.patience,
                seed=args.seed,
                deterministic=True,
                project=str(args.project),
                name=args.name,
                exist_ok=True,
                plots=True,
                val=True,
            )
            print(f"\nTraining: {train_kwargs}")
            t0 = time.time()
            run_guarded(model.train, train_kwargs, stage="training")
            print(f"Training finished in {time.time() - t0:.1f}s")

            best_path = run_dir / "weights" / "best.pt"
            if not best_path.exists():
                raise WMScriptError(
                    f"Training finished but no best.pt was found at {best_path} -- "
                    "check the ultralytics training log above for what went wrong."
                )
            print(f"Reloading best checkpoint for evaluation: {best_path}")
            model = YOLO(str(best_path))
            weights_path = best_path

        # ---- evaluate on all three domains, same code path -------------
        domains: Dict[str, dict] = {}

        if not args.skip_train_eval:
            train_native_yaml = build_data_yaml(data_dir, train_rel, val_rel, tmp_dir, "eval_train")
            domains["train"] = evaluate_domain(
                model, "train", "train (synthetic, seen during training)",
                data_dir / train_rel, data_dir / labels_rel_from_images_rel(train_rel),
                train_native_yaml, args, run_dir,
            )

        val_native_yaml = build_data_yaml(data_dir, val_rel, train_rel, tmp_dir, "eval_val")
        domains["val_synthetic"] = evaluate_domain(
            model, "val_synthetic", "val_synthetic (synthetic, held out)",
            data_dir / val_rel, data_dir / labels_rel_from_images_rel(val_rel),
            val_native_yaml, args, run_dir,
        )

        if not args.no_test_eval:
            test_native_yaml = build_data_yaml(test_dir, "images", "images", tmp_dir, "eval_test")
            domains["test_real"] = evaluate_domain(
                model, "test_real", "test_real (real, hand-labelled, held out)",
                test_dir / "images", test_dir / "labels",
                test_native_yaml, args, run_dir,
            )
        else:
            print("\n--no-test-eval: skipping the real test set. Accuracy metrics are "
                  "undefined without ground-truth labels; inspect predictions on real "
                  "images qualitatively with scripts/predict_overlays.py instead.")

    except WMScriptError as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- domain-gap summary --------------------------------------------
    gaps = {}
    if "val_synthetic" in domains and "test_real" in domains:
        gap_vt = compute_gap(domains["val_synthetic"], domains["test_real"])
        gaps["val_synthetic_minus_test_real"] = gap_vt
        print_domain_gap("val_synthetic", "test_real", gap_vt, args.gap_threshold)
    if "train" in domains and "val_synthetic" in domains:
        gap_tv = compute_gap(domains["train"], domains["val_synthetic"])
        gaps["train_minus_val_synthetic"] = gap_tv
        print_domain_gap("train", "val_synthetic", gap_tv, args.gap_threshold)

    # ---- write metrics.json ---------------------------------------------
    out = {
        "run": {
            "weights": str(weights_path),
            "eval_only": args.eval_only,
            "data_dir": str(data_dir),
            "test_dir": str(test_dir),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "seed": args.seed,
            "epochs": args.epochs if not args.eval_only else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "domains": domains,
        "domain_gap": gaps,
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {metrics_path}")

    # ---- human-readable summary ------------------------------------------
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    header = f"{'domain':<15}{'n':>5}  {'IoU':>7}{'Dice':>7}{'pxP':>7}{'pxR':>7}  " \
             f"{'mAP50':>7}{'mAP50-95':>9}  {'negFP%':>8}"
    print(header)
    for key in ("train", "val_synthetic", "test_real"):
        if key not in domains:
            continue
        d = domains[key]
        p, n = d["pixel"], d["native"]
        neg_fp = p["negative_fp_pixel_rate"]
        neg_fp_str = f"{neg_fp * 100:.2f}" if neg_fp is not None else "n/a"
        print(
            f"{key:<15}{d['n_images']:>5}  {p['iou']:>7.4f}{p['dice']:>7.4f}"
            f"{p['pixel_precision']:>7.4f}{p['pixel_recall']:>7.4f}  "
            f"{n['mask_mAP50']:>7.4f}{n['mask_mAP50_95']:>9.4f}  {neg_fp_str:>8}"
        )
    print(
        f"\nWeights used: {weights_path}\n"
        f"Full metrics: {metrics_path}\n"
        f"Visual panels: {run_dir / 'eval_visualizations'}\n"
    )
    print(
        "Re-run evaluation only (no retraining):\n"
        f"  ../.venv/Scripts/python.exe scripts/train_segmenter.py --eval-only "
        f"--weights {weights_path} --data-dir {args.data_dir} --test-dir {args.test_dir}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
