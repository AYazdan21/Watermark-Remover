import os

from PIL import Image

from .config import ORIGINALS_DIR, MASKS_DIR, RESULTS_DIR


def get_next_index():
    """Finds the next sequential integer index by checking existing files in ORIGINALS_DIR."""
    existing_ids = []
    if os.path.exists(ORIGINALS_DIR):
        for fname in os.listdir(ORIGINALS_DIR):
            name, _ = os.path.splitext(fname)
            prefix = name.split("_")[0]
            if prefix.isdigit():
                existing_ids.append(int(prefix))
    return max(existing_ids) + 1 if existing_ids else 1


def save_image_triple(bg_np, mask_np, result_pil):
    """Saves original, mask, and result into their respective folders."""
    if bg_np is None or mask_np is None or result_pil is None:
        return None, "No result available to save."

    idx = get_next_index()
    orig_path = os.path.join(ORIGINALS_DIR, f"{idx}_original.png")
    mask_path = os.path.join(MASKS_DIR, f"{idx}_mask.png")
    res_path = os.path.join(RESULTS_DIR, f"{idx}_result.png")

    # 1. Save original image
    Image.fromarray(bg_np).save(orig_path)
    # 2. Save binary mask (grayscale)
    Image.fromarray(mask_np).convert("L").save(mask_path)
    # 3. Save inpainted result
    result_pil.save(res_path)

    msg = f"Saved triple #{idx}:\n- `originals/{idx}_original.png`\n- `masks/{idx}_mask.png`\n- `results/{idx}_result.png`"
    return idx, msg


def manual_save_triple(bg_np, mask_np, result_pil):
    if bg_np is None or mask_np is None or result_pil is None:
        return "⚠️ No processed image to save. Please run watermark removal first."
    idx, msg = save_image_triple(bg_np, mask_np, result_pil)
    return f"💾 **Successfully {msg}**"
