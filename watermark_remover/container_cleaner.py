"""Container-aware adaptive background filling for document watermark removal.

Adapted from Desktop/remover (modules/utils.py) for the Watermark-Remover suite.
Operates on standard RGB images and binary masks.

Key Features:
1. Border-bounded horizontal containers: Segments rows by detected table rules and
   text-box outlines, taking median unmasked background within each container to prevent
   color leakage between different document areas (e.g. gray header vs white cell).
2. 2D Container Window: If a row segment is fully obscured, samples neighboring rows
   within the exact same horizontal container boundaries [x_min, x_max).
3. Neutral Background Filtering: Excludes chromatic watermark pixels (pink stamps,
   colored logos) from being considered valid paper background candidates.
"""

import cv2
import numpy as np


def get_most_frequent_color(image: np.ndarray) -> np.ndarray:
    """Returns the most frequent color per channel as an RGB uint8 array."""
    channels = [image[:, :, c] for c in range(image.shape[2])]
    most_frequent = [int(np.bincount(ch.flatten()).argmax()) for ch in channels]
    return np.array(most_frequent, dtype=np.uint8)


def fill_masked_area_rgb(
    image: np.ndarray,
    mask: np.ndarray,
    light_thresh: int = 180,
    border_thresh: int = 130,
    r_window: int = 12,
) -> np.ndarray:
    """Fills masked pixels with the local background color estimated from its container.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        mask: uint8 or boolean array of shape (H, W) where True/255 represents watermark.
        light_thresh: Minimum luminance to qualify as potential document paper background.
        border_thresh: Maximum luminance below which pixels are treated as container borders/text.
        r_window: Vertical search radius for the 2D container window.

    Returns:
        Cleaned RGB uint8 array with masked pixels infilled without cross-border color bleeding.
    """
    if image is None or mask is None:
        return image

    mask_bool = mask > 0
    if not np.any(mask_bool):
        return image.copy()

    img = image.copy()
    h, w = image.shape[:2]
    binary = mask_bool.astype(np.uint8)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 and image.shape[2] == 3 else image

    # Check row background luminance over unmasked pixels to differentiate
    # standard white paper from dark/colored headers/banners.
    row_bg_lum = np.zeros(h, dtype=np.float32)
    for r_idx in range(h):
        unm = binary[r_idx] == 0
        if np.any(unm):
            row_bg_lum[r_idx] = float(np.median(gray_image[r_idx, unm]))
        else:
            row_bg_lum[r_idx] = float(np.median(gray_image[r_idx]))

    # In scanned/administrative documents, legitimate backgrounds are neutral (gray or white).
    # Colored watermarks (pink/red/blue stamps or logos) have chroma and must NOT be used as background candidates.
    cand_mask = np.zeros((h, w), dtype=bool)
    border_mask = np.zeros((h, w), dtype=bool)

    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    bhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, k_h)
    v_line = cv2.morphologyEx((bhat >= 15).astype(np.uint8), cv2.MORPH_OPEN, k_v)

    if image.ndim == 3 and image.shape[2] == 3:
        r = image[:, :, 0].astype(int)
        g = image[:, :, 1].astype(int)
        b = image[:, :, 2].astype(int)
        is_neutral = (np.abs(r - g) <= 8) & (np.abs(r - b) <= 8) & (np.abs(g - b) <= 8)
        light_pixels = gray_image >= light_thresh
        is_mostly_neutral = np.sum(light_pixels & is_neutral) > 0.85 * np.sum(light_pixels)
    else:
        is_mostly_neutral = False
        is_neutral = np.ones((h, w), dtype=bool)

    for r_idx in range(h):
        if row_bg_lum[r_idx] >= 140:
            # Standard light paper row: borders are dark rules/text (< border_thresh)
            border_mask[r_idx] = (gray_image[r_idx] < border_thresh) | (v_line[r_idx] > 0)
            if is_mostly_neutral:
                cand_mask[r_idx] = (binary[r_idx] == 0) & (gray_image[r_idx] >= light_thresh) & is_neutral[r_idx]
            else:
                cand_mask[r_idx] = (binary[r_idx] == 0) & (gray_image[r_idx] >= light_thresh)
        else:
            # Dark / colored banner row: borders are high-contrast foreground text/icons,
            # NOT the colored background itself!
            border_mask[r_idx] = (gray_image[r_idx] > (row_bg_lum[r_idx] + 40)) | (gray_image[r_idx] < (row_bg_lum[r_idx] - 40))
            cand_mask[r_idx] = (binary[r_idx] == 0) & (~border_mask[r_idx])

    row_borders = [np.where(border_mask[row_idx])[0] for row_idx in range(h)]

    cont_cache = {}
    fallback_row_color = {}
    for row_idx in range(h):
        cands = np.where(cand_mask[row_idx])[0]
        if len(cands) >= 3:
            fallback_row_color[row_idx] = np.median(image[row_idx, cands], axis=0).astype(np.uint8)
        else:
            unm_row = np.where((binary[row_idx] == 0) & (~border_mask[row_idx]))[0]
            if len(unm_row) >= 3:
                fallback_row_color[row_idx] = np.median(image[row_idx, unm_row], axis=0).astype(np.uint8)

    if np.sum(cand_mask) > 0:
        global_fallback = np.median(image[cand_mask], axis=0).astype(np.uint8)
    else:
        global_fallback = get_most_frequent_color(image)

    unique_ys = np.where(binary.any(axis=1))[0]
    for py in unique_ys:
        row_mask_cols = np.where(binary[py] == 1)[0]
        rb = row_borders[py]

        if len(rb) == 0:
            segments = [(0, w, row_mask_cols)]
        else:
            split_points = np.searchsorted(row_mask_cols, rb)
            col_splits = np.split(row_mask_cols, split_points)
            segments = []
            for chunk in col_splits:
                if len(chunk) == 0:
                    continue
                px = chunk[0]
                left_b = rb[rb < px]
                x_min = int(left_b[-1] + 1) if len(left_b) > 0 else 0
                right_b = rb[rb > px]
                x_max = int(right_b[0]) if len(right_b) > 0 else w
                segments.append((x_min, x_max, chunk))

        for x_min, x_max, chunk in segments:
            # 1. Try row segment bounded by borders
            seg_cands = np.where(cand_mask[py, x_min:x_max])[0] + x_min
            if len(seg_cands) >= 3:
                col = np.median(image[py, seg_cands], axis=0).astype(np.uint8)
            else:
                # 2. Local 2D window within the SAME container boundaries [x_min, x_max)
                cont_key = (x_min, x_max, py // r_window)
                if cont_key in cont_cache:
                    col = cont_cache[cont_key]
                else:
                    y1 = max(0, py - r_window)
                    y2 = min(h, py + r_window + 1)
                    win_cands = cand_mask[y1:y2, x_min:x_max]
                    if np.sum(win_cands) >= 3:
                        col = np.median(image[y1:y2, x_min:x_max][win_cands], axis=0).astype(np.uint8)
                    elif py in fallback_row_color:
                        col = fallback_row_color[py]
                    else:
                        col = global_fallback
                    cont_cache[cont_key] = col

            img[py, chunk] = col

    return img


def telea_inpaint_masked_area(image: np.ndarray, mask: np.ndarray, inpaint_radius: int = 2) -> np.ndarray:
    """Inpaints masked pixels using cv2.INPAINT_TELEA."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    return cv2.inpaint(image, mask_u8, inpaint_radius, cv2.INPAINT_TELEA)
