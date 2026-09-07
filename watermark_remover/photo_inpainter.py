import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .config import BASE_DIR
from .lama_engine import lama
from .storage import save_image_triple
from .unmixer import unmix_region

# Above this median alpha within the detected mask, a mark is treated as
# effectively opaque -- there's little real structure left to recover, and
# unmixing a genuinely opaque mark is a plausible source of bad output
# (dividing by a near-zero 1-alpha, or a garbage color estimate with no
# real blend to anchor it -- unlike a mistaken *opaque* classification,
# which just keeps using the already-validated LaMa path unchanged, so
# this threshold is deliberately conservative). Recovering real detail
# only pays off where there's real transparency to exploit.
_TRANSPARENT_ALPHA_CEILING = 0.5

_yolo_models = {}

def get_yolo_model(model_name: str):
    """Lazily loads and caches YOLO11 watermark detector models."""
    if model_name not in _yolo_models:
        if model_name == "YOLO11 General Watermarks":
            path = os.path.join(BASE_DIR, "yolo11_watermark_general.pt")
        else:
            path = os.path.join(BASE_DIR, "yolo11s_watermark.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found.")
        print(f"Loading {model_name} from {path}...")
        _yolo_models[model_name] = YOLO(path)
    return _yolo_models[model_name]


def extract_bg_from_editor(editor_data):
    """Extracts background RGB numpy array from Gradio ImageEditor dictionary."""
    if editor_data is None:
        return None
    background = editor_data.get("background")
    if background is None:
        return None

    if isinstance(background, Image.Image):
        bg_np = np.array(background.convert("RGB"))
    else:
        bg_np = np.array(background)
        if bg_np.ndim == 3 and bg_np.shape[2] == 4:
            bg_np = cv2.cvtColor(bg_np, cv2.COLOR_RGBA2RGB)
        elif bg_np.ndim == 2:
            bg_np = cv2.cvtColor(bg_np, cv2.COLOR_GRAY2RGB)
    return bg_np


def build_watermark_mask(bg_np, boxes, box_dilation: int, smart_mask: bool = True):
    """
    Constructs binary inpainting mask and RGBA editor overlay.
    If smart_mask is True, refines bounding boxes containing text to avoid erasing
    underlying foreground objects (like cars, people, scenery).
    """
    h, w = bg_np.shape[:2]
    gray = cv2.cvtColor(bg_np, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)
    rgba_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        bx1 = max(0, x1 - box_dilation)
        by1 = max(0, y1 - box_dilation)
        bx2 = min(w, x2 + box_dilation)
        by2 = min(h, y2 + box_dilation)

        applied = False
        if smart_mask:
            roi_gray = gray[by1:by2, bx1:bx2]
            bright = (roi_gray >= 205)
            dark = (roi_gray <= 45)

            if np.sum(bright) > 60:
                text_roi = bright.astype(np.uint8) * 255
                text_roi = cv2.dilate(text_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
                mask[by1:by2, bx1:bx2] = np.maximum(mask[by1:by2, bx1:bx2], text_roi)
                rgba_layer[by1:by2, bx1:bx2][text_roi > 0] = [255, 0, 0, 180]
                applied = True
            elif np.sum(dark) > 60:
                text_roi = dark.astype(np.uint8) * 255
                text_roi = cv2.dilate(text_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
                mask[by1:by2, bx1:bx2] = np.maximum(mask[by1:by2, bx1:bx2], text_roi)
                rgba_layer[by1:by2, bx1:bx2][text_roi > 0] = [255, 0, 0, 180]
                applied = True

        if not applied:
            mask[by1:by2, bx1:bx2] = 255
            rgba_layer[by1:by2, bx1:bx2] = [255, 0, 0, 160]

    return mask, rgba_layer


def _local_residual_mask(bg_np, search_region):
    """Finds watermark-like pixels within `search_region` by comparing each
    pixel to its own local surroundings (a large-kernel median), rather
    than an absolute brightness threshold. build_watermark_mask's bright/
    dark cutoffs (fixed at 205/45 regardless of context) miss a watermark
    that spans both bright and dark parts of a photo -- e.g. text crossing
    both sky and road -- since a mid-brightness blended pixel over dark
    road never reaches the same absolute threshold a blended pixel over
    bright sky does. Comparing locally catches both. Returns (mask,
    evidence) where evidence is signed so estimate_mark_color can find the
    watermark's own strongest pixels regardless of whether it reads
    lighter or darker than its surroundings.
    """
    gray = cv2.cvtColor(bg_np, cv2.COLOR_RGB2GRAY)
    local_bg_gray = cv2.medianBlur(gray, 31)
    signed_diff = gray.astype(np.float64) - local_bg_gray.astype(np.float64)

    region_bool = search_region > 0
    direction = 1.0 if np.mean(signed_diff[region_bool]) >= 0 else -1.0
    evidence = direction * signed_diff

    candidate = ((evidence > 15) & region_bool).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    mask = cv2.dilate(candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask, evidence


def _remove_watermark(bg_np, mask):
    """Removes the watermark covered by `mask`, choosing alpha-unmixing
    (recovers real detail underneath) when the mark shows genuine
    transparency, or LaMa inpainting (hallucinates plausible content)
    when it doesn't -- see unmixer.py and _TRANSPARENT_ALPHA_CEILING.
    Returns (result_np, method_note); does not resize.

    `mask` (from build_watermark_mask) is used directly for the LaMa
    fallback, but only as a search region for the unmix attempt -- its
    own fixed 205/45 brightness cutoffs are too coarse for a mark that
    reads differently over different parts of the photo (see
    _local_residual_mask).
    """
    search_region = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    unmix_mask, evidence = _local_residual_mask(bg_np, search_region)
    unmix_mask_bool = unmix_mask > 0

    if not np.any(unmix_mask_bool):
        result_pil = lama(bg_np, mask)
        return np.array(result_pil.convert("RGB")), "LaMa Inpainted (no local watermark structure found)"

    unmixed = unmix_region(bg_np, unmix_mask, evidence=evidence)
    median_alpha = float(np.median(unmixed.alpha[unmix_mask_bool]))

    if median_alpha < _TRANSPARENT_ALPHA_CEILING:
        method_note = f"Alpha-Unmixed (transparent mark, median α={median_alpha:.2f})"
        return unmixed.recovered, method_note

    result_pil = lama(bg_np, mask)
    method_note = f"LaMa Inpainted (opaque mark, median α={median_alpha:.2f})"
    return np.array(result_pil.convert("RGB")), method_note


def auto_detect_watermark(editor_data, conf: float, box_dilation: int, model_name: str, smart_mask: bool):
    """
    Detects watermarks using YOLO11 and overlays red mask directly onto Gradio ImageEditor canvas.
    """
    bg_np = extract_bg_from_editor(editor_data)
    if bg_np is None:
        return editor_data, None, "Please upload an image first."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_yolo_model(model_name)

    t0 = time.time()
    results = model(bg_np, conf=conf, imgsz=800, device=device)
    boxes = results[0].boxes

    # Auto fallback to lower confidence if nothing found
    used_conf = conf
    if len(boxes) == 0 and conf > 0.08:
        fallback_res = model(bg_np, conf=0.08, imgsz=800, device=device)
        if len(fallback_res[0].boxes) > 0:
            results = fallback_res
            boxes = results[0].boxes
            used_conf = 0.08

    elapsed_ms = (time.time() - t0) * 1000

    if len(boxes) == 0:
        return editor_data, None, f"⚠️ No watermarks detected (searched down to conf 0.08). Try manual brush tool."

    mask, rgba_layer = build_watermark_mask(bg_np, boxes, box_dilation, smart_mask)

    updated_editor = {
        "background": bg_np,
        "layers": [rgba_layer],
        "composite": None
    }
    fallback_note = f" (auto-recovered at conf {used_conf:.2f})" if used_conf != conf else ""
    status = f"🎯 Detected **{len(boxes)} watermark(s)** in **{elapsed_ms:.1f} ms**{fallback_note}! Click **Inpaint** or adjust brush."
    return updated_editor, mask, status


def auto_detect_and_inpaint(editor_data, conf: float, box_dilation: int, model_name: str, smart_mask: bool, auto_save: bool):
    """
    1-Click Auto Remove: Runs YOLO11 detection and immediately inpaints with LaMa.
    """
    bg_np = extract_bg_from_editor(editor_data)
    if bg_np is None:
        return None, None, "Please upload an image first.", None, None, None

    h, w = bg_np.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_yolo_model(model_name)

    t0 = time.time()
    results = model(bg_np, conf=conf, imgsz=800, device=device)
    boxes = results[0].boxes

    # Auto fallback to lower confidence if nothing found
    used_conf = conf
    if len(boxes) == 0 and conf > 0.08:
        fallback_res = model(bg_np, conf=0.08, imgsz=800, device=device)
        if len(fallback_res[0].boxes) > 0:
            results = fallback_res
            boxes = results[0].boxes
            used_conf = 0.08

    detect_ms = (time.time() - t0) * 1000

    if len(boxes) == 0:
        # Return input image instead of None so the UI never displays broken grey box
        return Image.fromarray(bg_np), None, f"⚠️ No watermarks detected down to conf 0.08. Please use the brush tool.", bg_np, None, None

    mask, _ = build_watermark_mask(bg_np, boxes, box_dilation, smart_mask)

    t1 = time.time()
    result_np, method_note = _remove_watermark(bg_np, mask)
    result_pil = Image.fromarray(result_np).resize((w, h), Image.LANCZOS)
    inpaint_ms = (time.time() - t1) * 1000

    fallback_note = f" (auto-recovered at conf {used_conf:.2f})" if used_conf != conf else ""
    status = (
        f"⚡ 1-Click Removal Complete: Detected **{len(boxes)} watermark(s)** in **{detect_ms:.1f} ms**{fallback_note}, "
        f"{method_note} in **{inpaint_ms:.1f} ms**!"
    )

    if auto_save:
        idx, save_msg = save_image_triple(bg_np, mask, result_pil)
        if idx:
            status += f"  \n💾 **Auto-saved triple #{idx}** to `dataset/`."

    return result_pil, mask, status, bg_np, mask, result_pil


def inpaint_watermark(editor_data, dilation: int, auto_save: bool):
    """
    Takes the image and brush strokes from Gradio ImageEditor,
    extracts the mask, dilates it, and runs LaMa inpainting.
    """
    bg_np = extract_bg_from_editor(editor_data)
    if bg_np is None:
        return None, None, "Please upload an image first.", None, None, None

    h, w = bg_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Extract brush strokes from layers
    layers = editor_data.get("layers", [])
    has_brush = False

    for layer in layers:
        if layer is None:
            continue
        layer_np = np.array(layer)
        if layer_np.shape[:2] != (h, w):
            layer_np = cv2.resize(layer_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # In RGBA layers, alpha > 0 corresponds to brushed regions
        if layer_np.ndim == 3 and layer_np.shape[2] == 4:
            alpha = layer_np[:, :, 3]
            if np.any(alpha > 0):
                has_brush = True
                mask = np.maximum(mask, (alpha > 0).astype(np.uint8) * 255)
        elif layer_np.ndim == 2 and np.any(layer_np > 0):
            has_brush = True
            mask = np.maximum(mask, (layer_np > 0).astype(np.uint8) * 255)

    if not has_brush:
        return bg_np, mask, "No watermark selected! Click 'Auto-Detect' or brush over the watermark.", None, None, None

    # Apply dilation to cleanly cover anti-aliased watermark borders
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    t0 = time.time()
    result_np, method_note = _remove_watermark(bg_np, mask)
    result_pil = Image.fromarray(result_np).resize((w, h), Image.LANCZOS)
    elapsed_ms = (time.time() - t0) * 1000

    status = f"{method_note} in {elapsed_ms:.1f} ms! (Resolution: {w}x{h})"

    if auto_save:
        idx, save_msg = save_image_triple(bg_np, mask, result_pil)
        if idx:
            status += f"  \n💾 **Auto-saved triple #{idx}** to `dataset/originals/`, `dataset/masks/`, and `dataset/results/`."

    return result_pil, mask, status, bg_np, mask, result_pil

