"""
generate_dataset.py
Main dataset generation pipeline:
  1. Loads all backgrounds (2000 total) and 3 processed watermarks
  2. Assigns watermark combinations (0/1/2/3 watermarks per image)
  3. Applies extensive augmentations to watermarks
  4. Composites watermarks onto backgrounds
  5. Generates COCO JSON annotations, binary segmentation masks, and CSV labels
  6. Organizes into train/val splits (85/15)

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --verify   # run verification checks
"""

import os
import sys
import json
import random
import math
import argparse
import csv
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageChops
import cv2


# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG_DIR = os.path.join(BASE_DIR, "wm_backgrounds_v2")
WM_DIR = os.path.join(BASE_DIR, "watermarks_processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TARGET_SIZE = (1024, 1024)  # All output images are 1024x1024
SEED = 42
TOTAL_IMAGES = 2000

# Watermark combination distribution
DIST_NO_WM = 0.10    # 10% = 200 images
DIST_1_WM = 0.40     # 40% = 800 images
DIST_2_WM = 0.30     # 30% = 600 images
DIST_3_WM = 0.20     # 20% = 400 images

# Split ratio
TRAIN_RATIO = 0.85    # 85% train, 15% val

# Augmentation ranges
OPACITY_RANGE = (0.10, 0.70)
ROTATION_RANGE = (-45, 45)
SCALE_RANGE = (0.15, 3.0)
BLUR_SIGMA_RANGE = (0, 3.0)
BRIGHTNESS_RANGE = (0.5, 1.5)
CONTRAST_RANGE = (0.5, 1.5)
PERSPECTIVE_SKEW = 5  # degrees
FEATHER_SIGMA_RANGE = (0, 5)
JPEG_QUALITY_RANGE = (50, 95)
NOISE_SIGMA_RANGE = (0, 10)

# Blend mode weights
BLEND_MODES = {
    "normal": 0.50,
    "multiply": 0.20,
    "screen": 0.20,
    "overlay": 0.10,
}

# Watermark names and IDs
WATERMARK_INFO = {
    "AriaTendernet": {"id": 1, "file": "AriaTendernet.png"},
    "ariatender_logo": {"id": 2, "file": "ariatender_logo.png"},
    "ariatender_persian": {"id": 3, "file": "ariatender_persian.png"},
}

CATEGORIES = [
    {"id": 0, "name": "no_watermark", "supercategory": "background"},
    {"id": 1, "name": "AriaTendernet", "supercategory": "watermark"},
    {"id": 2, "name": "ariatender_logo", "supercategory": "watermark"},
    {"id": 3, "name": "ariatender_persian", "supercategory": "watermark"},
]


# ─── Utility Functions ───────────────────────────────────────────────────────

def load_backgrounds() -> List[str]:
    """Load all background image paths."""
    files = sorted([
        os.path.join(BG_DIR, f)
        for f in os.listdir(BG_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"Found {len(files)} background images")
    return files


def load_watermarks() -> Dict[str, Image.Image]:
    """Load all processed watermark images."""
    wms = {}
    for name, info in WATERMARK_INFO.items():
        path = os.path.join(WM_DIR, info["file"])
        wm = Image.open(path).convert("RGBA")
        wms[name] = wm
        print(f"  Loaded watermark '{name}': {wm.size}")
    return wms


def choose_blend_mode() -> str:
    """Randomly choose a blend mode based on weights."""
    modes = list(BLEND_MODES.keys())
    weights = list(BLEND_MODES.values())
    return random.choices(modes, weights=weights, k=1)[0]


def apply_blend(base_rgb: np.ndarray, wm_rgb: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply blending between base (background under watermark) and watermark RGB.
    Both inputs are float [0, 1]. Returns blended RGB [0, 1].
    """
    if mode == "normal":
        return wm_rgb
    elif mode == "multiply":
        return base_rgb * wm_rgb
    elif mode == "screen":
        return 1.0 - (1.0 - base_rgb) * (1.0 - wm_rgb)
    elif mode == "overlay":
        # Overlay: multiply if base < 0.5, screen if base >= 0.5
        mask = base_rgb < 0.5
        result = np.where(
            mask,
            2.0 * base_rgb * wm_rgb,
            1.0 - 2.0 * (1.0 - base_rgb) * (1.0 - wm_rgb)
        )
        return result
    return wm_rgb


def apply_perspective_warp(img: Image.Image, max_skew_deg: float = 5) -> Image.Image:
    """Apply a slight perspective/affine warp to the image."""
    w, h = img.size
    
    # Random skew amounts in pixels
    max_offset = int(max(w, h) * math.tan(math.radians(max_skew_deg)) * 0.1)
    if max_offset < 1:
        return img
    
    # Define source and destination corners
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, max_offset), random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
        [random.randint(0, max_offset), h - random.randint(0, max_offset)],
    ])
    
    # Convert to numpy for cv2
    img_np = np.array(img)
    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img_np, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    return Image.fromarray(result)


def apply_color_jitter(img: Image.Image) -> Image.Image:
    """Apply slight color jitter (hue/saturation shift) to watermark."""
    # Only jitter sometimes
    if random.random() > 0.3:
        return img
    
    # Split channels
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    
    # Slight hue shift via HSV
    hsv = np.array(rgb.convert("HSV"), dtype=np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
    hsv = hsv.astype(np.uint8)
    
    rgb_shifted = Image.fromarray(hsv, "HSV").convert("RGB")
    return Image.merge("RGBA", (*rgb_shifted.split(), a))


def augment_watermark(
    wm: Image.Image,
    bg_size: Tuple[int, int],
) -> Tuple[Image.Image, dict]:
    """
    Apply all augmentations to a single watermark instance.
    Returns the augmented watermark (RGBA) and a dict of augmentation parameters.
    """
    params = {}
    
    # 1. Scale
    scale = random.uniform(*SCALE_RANGE)
    # Bias toward moderate scales (0.3-1.5) with 70% probability
    if random.random() < 0.7:
        scale = random.uniform(0.3, 1.5)
    params["scale"] = round(scale, 3)
    
    new_w = max(1, int(wm.width * scale))
    new_h = max(1, int(wm.height * scale))
    # Cap to prevent memory issues
    new_w = min(new_w, bg_size[0] * 4)
    new_h = min(new_h, bg_size[1] * 4)
    wm_aug = wm.resize((new_w, new_h), Image.LANCZOS)
    
    # 2. Rotation
    rotation = random.uniform(*ROTATION_RANGE)
    # Bias toward common angles
    if random.random() < 0.4:
        rotation = random.choice([0, 0, 0, -45, 45, -30, 30, -15, 15])
    params["rotation"] = round(rotation, 1)
    wm_aug = wm_aug.rotate(rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    
    # 3. Resolution/blur
    blur_sigma = random.uniform(*BLUR_SIGMA_RANGE)
    if blur_sigma > 0.3:
        wm_aug = wm_aug.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    params["blur_sigma"] = round(blur_sigma, 2)
    
    # 4. Brightness
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    params["brightness"] = round(brightness, 2)
    r, g, b, a = wm_aug.split()
    rgb = Image.merge("RGB", (r, g, b))
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    wm_aug = Image.merge("RGBA", (*rgb.split(), a))
    
    # 5. Contrast
    contrast = random.uniform(*CONTRAST_RANGE)
    params["contrast"] = round(contrast, 2)
    r, g, b, a = wm_aug.split()
    rgb = Image.merge("RGB", (r, g, b))
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    wm_aug = Image.merge("RGBA", (*rgb.split(), a))
    
    # 6. Color jitter
    wm_aug = apply_color_jitter(wm_aug)
    
    # 7. Opacity
    opacity = random.uniform(*OPACITY_RANGE)
    params["opacity"] = round(opacity, 2)
    # We'll apply opacity during compositing, not here
    
    # 8. Perspective warp (30% of the time)
    if random.random() < 0.3:
        wm_aug = apply_perspective_warp(wm_aug, PERSPECTIVE_SKEW)
        params["perspective_warp"] = True
    else:
        params["perspective_warp"] = False
    
    # 9. Edge feathering (40% of the time)
    feather = random.uniform(*FEATHER_SIGMA_RANGE)
    if feather > 0.5 and random.random() < 0.4:
        r, g, b, a = wm_aug.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=feather))
        wm_aug = Image.merge("RGBA", (r, g, b, a))
        params["edge_feather"] = round(feather, 2)
    else:
        params["edge_feather"] = 0
    
    # 10. Blend mode
    blend_mode = choose_blend_mode()
    params["blend_mode"] = blend_mode
    
    return wm_aug, params


def decide_tiling(bg_size: Tuple[int, int]) -> Tuple[bool, int, int, int, int]:
    """Decide if and how to tile the watermark."""
    # 20% chance of tiling
    if random.random() < 0.2:
        cols = random.randint(2, 5)
        rows = random.randint(2, 5)
        gap_x = random.randint(20, 150)
        gap_y = random.randint(20, 100)
        return True, rows, cols, gap_x, gap_y
    return False, 1, 1, 0, 0


def composite_watermark_on_background(
    bg: Image.Image,
    wm_aug: Image.Image,
    opacity: float,
    blend_mode: str,
    position: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    """
    Composite a single augmented watermark onto the background.
    Returns (composited_bg, mask_layer).
    
    mask_layer is a single-channel image where non-zero pixels indicate watermark presence.
    """
    bg_w, bg_h = bg.size
    wm_w, wm_h = wm_aug.size
    px, py = position
    
    # Create the mask layer (same size as bg)
    mask_layer = Image.new("L", (bg_w, bg_h), 0)
    
    # Calculate the visible region (intersection of watermark and background)
    # Watermark region in bg coordinates
    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(bg_w, px + wm_w)
    y2 = min(bg_h, py + wm_h)
    
    if x2 <= x1 or y2 <= y1:
        return bg, mask_layer  # No overlap
    
    # Crop the watermark to the visible region
    wm_x1 = x1 - px
    wm_y1 = y1 - py
    wm_x2 = x2 - px
    wm_y2 = y2 - py
    
    wm_crop = wm_aug.crop((wm_x1, wm_y1, wm_x2, wm_y2))
    
    if wm_crop.size[0] == 0 or wm_crop.size[1] == 0:
        return bg, mask_layer
    
    # Extract channels
    wm_r, wm_g, wm_b, wm_a = wm_crop.split()
    
    # Apply opacity to alpha
    wm_a_np = np.array(wm_a, dtype=np.float32) / 255.0
    wm_a_np = wm_a_np * opacity
    
    # Get the background region
    bg_region = bg.crop((x1, y1, x2, y2)).convert("RGB")
    bg_np = np.array(bg_region, dtype=np.float32) / 255.0
    
    # Get watermark RGB
    wm_rgb_np = np.stack([
        np.array(wm_r, dtype=np.float32) / 255.0,
        np.array(wm_g, dtype=np.float32) / 255.0,
        np.array(wm_b, dtype=np.float32) / 255.0,
    ], axis=-1)
    
    # Apply blend mode
    blended = apply_blend(bg_np, wm_rgb_np, blend_mode)
    
    # Alpha composite: result = bg * (1 - alpha) + blended * alpha
    alpha_3ch = np.stack([wm_a_np] * 3, axis=-1)
    result = bg_np * (1.0 - alpha_3ch) + blended * alpha_3ch
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    # Paste result back
    bg_result = bg.copy()
    bg_result.paste(Image.fromarray(result), (x1, y1))
    
    # Build mask: where alpha > threshold
    mask_region = (wm_a_np > 0.02).astype(np.uint8) * 255
    mask_layer.paste(Image.fromarray(mask_region, "L"), (x1, y1))
    
    return bg_result, mask_layer


def apply_post_processing(img: Image.Image) -> Image.Image:
    """Apply post-composition augmentations to the final image."""
    # Convert to numpy for some ops
    img_np = np.array(img, dtype=np.float32)
    
    # 1. Gaussian noise (60% of the time)
    if random.random() < 0.6:
        noise_sigma = random.uniform(*NOISE_SIGMA_RANGE)
        if noise_sigma > 1:
            noise = np.random.normal(0, noise_sigma, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
    
    # 2. Color temperature shift (30% of the time)
    if random.random() < 0.3:
        shift = random.uniform(-15, 15)
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] + shift, 0, 255)      # R channel
        img_np[:, :, 2] = np.clip(img_np[:, :, 2] - shift * 0.7, 0, 255)  # B channel
    
    img = Image.fromarray(img_np.astype(np.uint8))
    
    # 3. Background quality degradation (resize down then up) - 20% of the time
    if random.random() < 0.2:
        factor = random.uniform(0.4, 0.8)
        small_size = (max(64, int(img.width * factor)), max(64, int(img.height * factor)))
        img = img.resize(small_size, Image.BILINEAR).resize(img.size, Image.BILINEAR)
    
    return img


def save_as_jpeg_with_quality(img: Image.Image, path: str) -> None:
    """Save image as JPEG with random quality to simulate compression artifacts."""
    quality = random.randint(*JPEG_QUALITY_RANGE)
    img.convert("RGB").save(path, "JPEG", quality=quality)


def generate_single_image(
    idx: int,
    bg_path: str,
    watermarks: Dict[str, Image.Image],
    wm_combo: List[str],  # list of watermark names to apply
) -> dict:
    """
    Generate a single dataset image with watermarks and masks.
    Returns annotation metadata dict.
    """
    # Load and resize background to 1024x1024
    bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize(TARGET_SIZE, Image.LANCZOS)
    
    bg_w, bg_h = bg.size
    
    # Determine output filename
    img_filename = f"wm_{idx:04d}.jpg"
    mask_filename = f"wm_{idx:04d}_mask.png"
    
    # Track annotations for this image
    annotations = []
    mask_layers = {}  # category_id -> combined mask
    
    if len(wm_combo) == 0:
        # No watermark — just save the clean image with post-processing
        bg = apply_post_processing(bg)
    else:
        for wm_name in wm_combo:
            wm_src = watermarks[wm_name].copy()
            cat_id = WATERMARK_INFO[wm_name]["id"]
            
            # Decide tiling
            is_tiled, tile_rows, tile_cols, gap_x, gap_y = decide_tiling((bg_w, bg_h))
            
            # Augment the watermark
            wm_aug, aug_params = augment_watermark(wm_src, (bg_w, bg_h))
            opacity = aug_params["opacity"]
            blend_mode = aug_params["blend_mode"]
            
            if is_tiled:
                # Place in a grid pattern
                wm_w, wm_h = wm_aug.size
                total_w = tile_cols * wm_w + (tile_cols - 1) * gap_x
                total_h = tile_rows * wm_h + (tile_rows - 1) * gap_y
                
                # Center the grid on the image
                start_x = (bg_w - total_w) // 2
                start_y = (bg_h - total_h) // 2
                
                combined_mask = Image.new("L", (bg_w, bg_h), 0)
                
                for r in range(tile_rows):
                    for c in range(tile_cols):
                        px = start_x + c * (wm_w + gap_x)
                        py = start_y + r * (wm_h + gap_y)
                        bg, ml = composite_watermark_on_background(
                            bg, wm_aug, opacity, blend_mode, (px, py)
                        )
                        # Merge mask
                        combined_mask = Image.fromarray(
                            np.maximum(np.array(combined_mask), np.array(ml))
                        )
                
                mask_layers[cat_id] = combined_mask
                aug_params["is_tiled"] = True
                aug_params["tile_grid"] = f"{tile_rows}x{tile_cols}"
                
                # Bounding box = extent of all tiles
                mask_np = np.array(combined_mask)
                if mask_np.max() > 0:
                    ys, xs = np.where(mask_np > 0)
                    bbox = [int(xs.min()), int(ys.min()),
                            int(xs.max() - xs.min()), int(ys.max() - ys.min())]
                    area = int(mask_np.sum() / 255)
                    is_partial = (bbox[0] <= 0 or bbox[1] <= 0 or
                                  bbox[0] + bbox[2] >= bg_w - 1 or
                                  bbox[1] + bbox[3] >= bg_h - 1)
                    
                    # Generate polygon from mask (simplified contour)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = []
                    for contour in contours:
                        if len(contour) >= 3:
                            # Simplify contour
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            if len(approx) >= 3:
                                seg = approx.flatten().tolist()
                                segmentation.append(seg)
                    
                    annotations.append({
                        "category_id": cat_id,
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "attributes": {
                            "watermark_name": wm_name,
                            "is_partial": is_partial,
                            **aug_params,
                        }
                    })
            else:
                # Single placement
                wm_w, wm_h = wm_aug.size
                
                # Random position (allow partial off-screen)
                margin = 50
                px = random.randint(-wm_w + margin, bg_w - margin)
                py = random.randint(-wm_h + margin, bg_h - margin)
                
                bg, mask_layer = composite_watermark_on_background(
                    bg, wm_aug, opacity, blend_mode, (px, py)
                )
                mask_layers[cat_id] = mask_layer
                aug_params["is_tiled"] = False
                
                # Compute bounding box from mask
                mask_np = np.array(mask_layer)
                if mask_np.max() > 0:
                    ys, xs = np.where(mask_np > 0)
                    bbox = [int(xs.min()), int(ys.min()),
                            int(xs.max() - xs.min()), int(ys.max() - ys.min())]
                    area = int(mask_np.sum() / 255)
                    is_partial = (bbox[0] <= 0 or bbox[1] <= 0 or
                                  bbox[0] + bbox[2] >= bg_w - 1 or
                                  bbox[1] + bbox[3] >= bg_h - 1)
                    
                    # Generate polygon
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = []
                    for contour in contours:
                        if len(contour) >= 3:
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            if len(approx) >= 3:
                                seg = approx.flatten().tolist()
                                segmentation.append(seg)
                    
                    annotations.append({
                        "category_id": cat_id,
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "attributes": {
                            "watermark_name": wm_name,
                            "is_partial": is_partial,
                            **aug_params,
                        }
                    })
        
        # Post-processing
        bg = apply_post_processing(bg)
    
    # Build combined segmentation mask (class-encoded)
    # 0 = background, 1/2/3 = watermark class
    combined_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    for cat_id, ml in mask_layers.items():
        ml_np = np.array(ml)
        combined_mask[ml_np > 0] = cat_id
    
    # Infer background type from filename
    bg_basename = os.path.basename(bg_path)
    
    return {
        "image": bg,
        "mask": combined_mask,
        "img_filename": img_filename,
        "mask_filename": mask_filename,
        "annotations": annotations,
        "bg_filename": bg_basename,
        "wm_combo": wm_combo,
        "has_watermark": len(wm_combo) > 0,
    }


def build_coco_dataset(
    all_results: List[dict],
    split_name: str,
) -> dict:
    """Build a COCO-format annotation dict from results."""
    images_list = []
    annotations_list = []
    ann_id = 1
    
    for r in all_results:
        img_id = int(r["img_filename"].split("_")[1].split(".")[0])
        images_list.append({
            "id": img_id,
            "file_name": r["img_filename"],
            "width": TARGET_SIZE[0],
            "height": TARGET_SIZE[1],
        })
        
        for ann in r["annotations"]:
            ann_entry = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
                "attributes": ann.get("attributes", {}),
            }
            annotations_list.append(ann_entry)
            ann_id += 1
    
    return {
        "info": {
            "description": f"Watermark Detection Dataset - {split_name}",
            "version": "1.0",
            "year": 2026,
            "date_created": datetime.now().isoformat(),
        },
        "images": images_list,
        "annotations": annotations_list,
        "categories": CATEGORIES,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run verification checks only")
    args = parser.parse_args()
    
    if args.verify:
        run_verification()
        return
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    print("=" * 70)
    print("  WATERMARK DATASET GENERATOR")
    print("=" * 70)
    print(f"  Target: {TOTAL_IMAGES} images @ {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"  Split: {TRAIN_RATIO*100:.0f}% train / {(1-TRAIN_RATIO)*100:.0f}% val")
    print()
    
    # ─── Load assets ──────────────────────────────────────────────────────
    print("Loading backgrounds...")
    bg_paths = load_backgrounds()
    
    if len(bg_paths) < TOTAL_IMAGES:
        print(f"WARNING: Only {len(bg_paths)} backgrounds found, need {TOTAL_IMAGES}.")
        print("Will reuse backgrounds with different augmentations.")
        # Extend by cycling
        while len(bg_paths) < TOTAL_IMAGES:
            bg_paths.extend(bg_paths[:TOTAL_IMAGES - len(bg_paths)])
        bg_paths = bg_paths[:TOTAL_IMAGES]
    else:
        bg_paths = bg_paths[:TOTAL_IMAGES]
    
    random.shuffle(bg_paths)
    
    print("\nLoading watermarks...")
    watermarks = load_watermarks()
    wm_names = list(watermarks.keys())
    
    # ─── Assign watermark combinations ────────────────────────────────────
    print("\nAssigning watermark combinations...")
    n_no_wm = int(TOTAL_IMAGES * DIST_NO_WM)
    n_1_wm = int(TOTAL_IMAGES * DIST_1_WM)
    n_2_wm = int(TOTAL_IMAGES * DIST_2_WM)
    n_3_wm = TOTAL_IMAGES - n_no_wm - n_1_wm - n_2_wm
    
    combos = []
    # No watermark
    combos.extend([[] for _ in range(n_no_wm)])
    # 1 watermark
    for _ in range(n_1_wm):
        combos.append([random.choice(wm_names)])
    # 2 watermarks
    for _ in range(n_2_wm):
        combos.append(random.sample(wm_names, 2))
    # 3 watermarks
    combos.extend([list(wm_names) for _ in range(n_3_wm)])
    
    random.shuffle(combos)
    
    combo_counts = Counter([len(c) for c in combos])
    print(f"  No watermark: {combo_counts[0]}")
    print(f"  1 watermark:  {combo_counts[1]}")
    print(f"  2 watermarks: {combo_counts[2]}")
    print(f"  3 watermarks: {combo_counts[3]}")
    
    # ─── Create output directories ────────────────────────────────────────
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "masks", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)
    
    # ─── Split assignments ────────────────────────────────────────────────
    indices = list(range(TOTAL_IMAGES))
    random.shuffle(indices)
    n_train = int(TOTAL_IMAGES * TRAIN_RATIO)
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:])
    
    print(f"\n  Train: {len(train_indices)} images")
    print(f"  Val:   {len(val_indices)} images")
    
    # ─── Generate images ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GENERATING DATASET")
    print(f"{'='*70}\n")
    
    train_results = []
    val_results = []
    csv_rows = []
    
    for idx in range(TOTAL_IMAGES):
        try:
            result = generate_single_image(
                idx, bg_paths[idx], watermarks, combos[idx]
            )
            
            # Determine split
            split = "train" if idx in train_indices else "val"
            
            # Save image (JPEG with quality augmentation)
            img_path = os.path.join(OUTPUT_DIR, "images", split, result["img_filename"])
            save_as_jpeg_with_quality(result["image"], img_path)
            
            # Save mask (PNG, lossless)
            mask_path = os.path.join(OUTPUT_DIR, "masks", split, result["mask_filename"])
            Image.fromarray(result["mask"]).save(mask_path, "PNG")
            
            # Track for COCO JSON
            if split == "train":
                train_results.append(result)
            else:
                val_results.append(result)
            
            # CSV row
            wm_types = ";".join(result["wm_combo"]) if result["wm_combo"] else ""
            csv_rows.append({
                "image_name": result["img_filename"],
                "split": split,
                "has_watermark": result["has_watermark"],
                "num_watermarks": len(result["wm_combo"]),
                "watermark_types": wm_types,
                "background_source": result["bg_filename"],
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1:4d}/{TOTAL_IMAGES}] Generated {result['img_filename']} "
                      f"({split}, {len(result['wm_combo'])} wm)")
        
        except Exception as e:
            print(f"  ERROR generating image {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ─── Save COCO JSON annotations ──────────────────────────────────────
    print("\nSaving COCO JSON annotations...")
    
    train_coco = build_coco_dataset(train_results, "train")
    val_coco = build_coco_dataset(val_results, "val")
    
    train_json_path = os.path.join(OUTPUT_DIR, "annotations", "train.json")
    val_json_path = os.path.join(OUTPUT_DIR, "annotations", "val.json")
    
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {train_json_path} ({len(train_coco['annotations'])} annotations)")
    
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {val_json_path} ({len(val_coco['annotations'])} annotations)")
    
    # ─── Save CSV labels ─────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "annotations", "image_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "split", "has_watermark", "num_watermarks",
            "watermark_types", "background_source"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved: {csv_path} ({len(csv_rows)} rows)")
    
    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total images: {len(csv_rows)}")
    print(f"  Train: {len(train_results)}, Val: {len(val_results)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    
    # Print distribution stats
    wm_count_dist = Counter(r["num_watermarks"] for r in csv_rows)
    print("  Watermark count distribution:")
    for k in sorted(wm_count_dist.keys()):
        print(f"    {k} watermarks: {wm_count_dist[k]}")
    print()


def run_verification():
    """Run verification checks on the generated dataset."""
    print("=" * 70)
    print("  DATASET VERIFICATION")
    print("=" * 70)
    
    errors = []
    
    # Check image counts
    for split in ["train", "val"]:
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        mask_dir = os.path.join(OUTPUT_DIR, "masks", split)
        
        if not os.path.exists(img_dir):
            errors.append(f"Missing directory: {img_dir}")
            continue
        
        imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        print(f"\n  {split}: {len(imgs)} images, {len(masks)} masks")
        
        # Check mask-image correspondence
        img_bases = {os.path.splitext(f)[0] for f in imgs}
        mask_bases = {f.replace("_mask.png", "") for f in masks}
        
        missing_masks = img_bases - mask_bases
        if missing_masks:
            errors.append(f"{split}: {len(missing_masks)} images missing masks")
        
        # Check mask dimensions
        for mask_file in masks[:10]:  # spot check
            mask = Image.open(os.path.join(mask_dir, mask_file))
            if mask.size != TARGET_SIZE:
                errors.append(f"{split}/{mask_file}: mask size {mask.size} != {TARGET_SIZE}")
    
    # Check COCO JSON
    for split in ["train", "val"]:
        json_path = os.path.join(OUTPUT_DIR, "annotations", f"{split}.json")
        if not os.path.exists(json_path):
            errors.append(f"Missing annotation file: {json_path}")
            continue
        
        with open(json_path, "r") as f:
            coco = json.load(f)
        
        print(f"\n  {split}.json: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
        print(f"  Categories: {[c['name'] for c in coco['categories']]}")
    
    # Check CSV
    csv_path = os.path.join(OUTPUT_DIR, "annotations", "image_labels.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"\n  image_labels.csv: {len(rows)} rows")
        
        wm_dist = Counter(int(r["num_watermarks"]) for r in rows)
        print("  Watermark distribution:")
        for k in sorted(wm_dist.keys()):
            print(f"    {k} watermarks: {wm_dist[k]}")
    
    # Report
    if errors:
        print(f"\n  [FAIL] VERIFICATION FAILED: {len(errors)} errors")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  [SUCCESS] VERIFICATION PASSED")


if __name__ == "__main__":
    main()
