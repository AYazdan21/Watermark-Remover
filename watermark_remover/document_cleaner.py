import os
import time

import cv2
import numpy as np
from PIL import Image

from .config import CLEANED_DOCS_DIR


def auto_detect_document_profile(img_np):
    """
    Analyzes document image geometry, lines, margins, and paper colors to automatically
    configure optimal settings for 1-click watermark removal with zero manual setup.
    """
    h, w = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Detect Tables & Grids via 1D Top-Hat filters
    tophat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    tophat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))

    h_lines = cv2.morphologyEx((tophat_h >= 7).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    v_lines = cv2.morphologyEx((tophat_v >= 7).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))

    h_proj = np.sum(h_lines > 0, axis=1)
    v_proj = np.sum(v_lines > 0, axis=0)

    row_peaks = [y for y in range(1, h - 1) if h_proj[y] > w * 0.25 and h_proj[y] >= h_proj[y - 1] and h_proj[y] >= h_proj[y + 1]]
    col_peaks = [x for x in range(1, w - 1) if v_proj[x] > h * 0.25 and v_proj[x] >= v_proj[x - 1] and v_proj[x] >= v_proj[x + 1]]

    has_table = (len(row_peaks) >= 3 and len(col_peaks) >= 2)

    # 2. Line thickness based on resolution
    line_thickness = "2px (Standard)" if max(h, w) > 1600 else "1px (Hairline)"

    # 3. Detect Margins & Background Tint
    m_h = max(2, int(h * 0.025))
    m_w = max(2, int(w * 0.025))
    outer_samples = np.vstack([
        img_np[:m_h, :].reshape(-1, 3),
        img_np[-m_h:, :].reshape(-1, 3),
        img_np[:, :m_w].reshape(-1, 3),
        img_np[:, -m_w:].reshape(-1, 3)
    ])
    outer_color = np.median(outer_samples, axis=0)

    inner_crop = img_np[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    inner_gray = gray[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    bright_inner = inner_crop[inner_gray > np.percentile(inner_gray, 85)]
    inner_color = np.median(bright_inner, axis=0) if len(bright_inner) > 0 else outer_color

    color_diff = float(np.linalg.norm(outer_color.astype(float) - inner_color.astype(float)))

    edges = cv2.Canny(gray, 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_frame = any(cv2.boundingRect(cnt)[2] > w * 0.65 and cv2.boundingRect(cnt)[3] > h * 0.65 for cnt in contours)

    if has_frame and color_diff > 8.0:
        bg_mode = "Dual-Zone Auto (White Margin + Cream Paper)"
    elif color_diff > 12.0:
        bg_mode = "Inner Paper Tint Only"
    else:
        bg_mode = "Pure White Everywhere"

    # 4. Gridline contrast
    if has_table:
        clean_grid = np.zeros((h, w), dtype=np.uint8)
        min_x, max_x = min(col_peaks), max(col_peaks)
        min_y, max_y = min(row_peaks), max(row_peaks)
        for y in row_peaks:
            cv2.line(clean_grid, (min_x, y), (max_x, y), 255, 1)
        for x in col_peaks:
            cv2.line(clean_grid, (x, min_y), (x, max_y), 255, 1)
        line_samples = gray[(clean_grid > 0) & (gray > 180) & (gray < 240)]
        med_val = int(np.median(line_samples)) if len(line_samples) > 0 else 220
        grid_contrast = 60 if med_val > 215 else 40
    else:
        grid_contrast = 50

    return {
        "has_table": has_table,
        "protect_tables": has_table,
        "snap_gridlines": has_table,
        "line_thickness": line_thickness,
        "bg_mode": bg_mode,
        "grid_contrast": grid_contrast,
        "anti_alias": True,
        "stamp_filter": "None (Standard)",
        "sensitivity_offset": 0,
    }


def clean_document_auto(
    doc_image,
    sensitivity_offset: int,
    bg_mode: str,
    protect_tables: bool = True,
    snap_gridlines: bool = True,
    anti_alias: bool = True,
    line_thickness: str = "1px (Hairline)",
    stamp_filter: str = "None (Standard)",
    grid_contrast: int = 50,
    smart_auto: bool = True,
):
    """
    De-blends semi-transparent watermarks and colored stamps from document scans
    while preserving outer white margins, inner paper tints, and optional table gridlines.
    Supports mathematical grid snapping, soft anti-aliasing, and optical channel decomposition.
    """
    if doc_image is None:
        return None, "Please upload a document image first."

    t0 = time.time()
    img_np = np.array(doc_image.convert("RGB"))
    h, w = img_np.shape[:2]

    auto_profile_note = ""
    if smart_auto:
        profile = auto_detect_document_profile(img_np)
        protect_tables = profile["protect_tables"]
        snap_gridlines = profile["snap_gridlines"]
        bg_mode = profile["bg_mode"]
        grid_contrast = profile["grid_contrast"]
        line_thickness = profile["line_thickness"]
        anti_alias = profile["anti_alias"]
        doc_type_str = "Spreadsheet / Table" if profile["has_table"] else "Document Notice"
        auto_profile_note = f"✨ Auto-Pilot: Detected **{doc_type_str}** | "

    # Select channel based on stamp_filter
    if stamp_filter == "Red Stamp Filter":
        # In red channel, red ink reflects ~255 (invisible), while black ink absorbs (~30)
        gray = img_np[:, :, 0]
    elif stamp_filter == "Blue Stamp Filter":
        gray = img_np[:, :, 2]
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Sample outer margin (top/bottom/left/right 2% border)
    m_h = max(2, int(h * 0.025))
    m_w = max(2, int(w * 0.025))
    outer_samples = np.vstack([
        img_np[:m_h, :].reshape(-1, 3),
        img_np[-m_h:, :].reshape(-1, 3),
        img_np[:, :m_w].reshape(-1, 3),
        img_np[:, -m_w:].reshape(-1, 3)
    ])
    outer_color = np.median(outer_samples, axis=0).astype(np.uint8)

    # 2. Sample inner paper color (inside central 20%-80% area, top 15% brightest)
    inner_crop = img_np[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    inner_gray = gray[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    bright_inner = inner_crop[inner_gray > np.percentile(inner_gray, 85)]
    if len(bright_inner) > 0:
        inner_color = np.median(bright_inner, axis=0).astype(np.uint8)
    else:
        inner_color = outer_color

    # Base thresholding (Otsu)
    base_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_thresh = int(np.clip(base_thresh + sensitivity_offset, 60, 245))

    # Detect and protect table gridlines if enabled
    clean_grid = np.zeros((h, w), dtype=np.uint8)
    snapped_grid_applied = False
    med_val = 220
    grid_mask_dilated = np.zeros((h, w), dtype=bool)

    if protect_tables:
        tophat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
        tophat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))

        h_cand = (tophat_h >= 7).astype(np.uint8) * 255
        v_cand = (tophat_v >= 7).astype(np.uint8) * 255

        # Use 25px minimum continuous length to filter out character stems and commas
        h_lines = cv2.morphologyEx(h_cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
        v_lines = cv2.morphologyEx(v_cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))

        if snap_gridlines:
            h_proj = np.sum(h_lines > 0, axis=1)
            row_peaks = []
            for y in range(1, h - 1):
                if h_proj[y] > w * 0.25 and h_proj[y] >= h_proj[y - 1] and h_proj[y] >= h_proj[y + 1]:
                    if not row_peaks or y - row_peaks[-1] > 3:
                        row_peaks.append(y)

            v_proj = np.sum(v_lines > 0, axis=0)
            col_peaks = []
            for x in range(1, w - 1):
                if v_proj[x] > h * 0.25 and v_proj[x] >= v_proj[x - 1] and v_proj[x] >= v_proj[x + 1]:
                    if not col_peaks or x - col_peaks[-1] > 6:
                        col_peaks.append(x)

            if len(row_peaks) >= 2 or len(col_peaks) >= 2:
                thick_px = 2 if line_thickness == "2px (Standard)" else 1
                min_x = min(col_peaks) if col_peaks else 0
                max_x = max(col_peaks) if col_peaks else w - 1
                min_y = min(row_peaks) if row_peaks else 0
                max_y = max(row_peaks) if row_peaks else h - 1

                for y in row_peaks:
                    cv2.line(clean_grid, (min_x, y), (max_x, y), 255, thick_px)
                for x in col_peaks:
                    cv2.line(clean_grid, (x, min_y), (x, max_y), 255, thick_px)

                grid_mask_dilated = clean_grid > 0
                clean_lines = gray[(clean_grid > 0) & (gray > 180) & (gray < 240)]
                med_val = int(np.median(clean_lines)) if len(clean_lines) > 0 else 220
                snapped_grid_applied = True

        if not snapped_grid_applied:
            h_lines_c = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
            v_lines_c = cv2.morphologyEx(v_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
            grid_mask = cv2.bitwise_or(h_lines_c, v_lines_c)
            grid_mask_dilated = cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1) > 0
            clean_lines = gray[grid_mask_dilated & (gray > 190) & (gray < 240)]
            med_val = int(np.median(clean_lines)) if len(clean_lines) > 0 else 210

    # Build target background buffer
    target_bg = np.zeros_like(img_np)
    if bg_mode == "Pure White Everywhere":
        target_bg[:] = [255, 255, 255]
        zone_info = "Single-Zone: Pure White (#FFFFFF)"

    elif bg_mode == "Inner Paper Tint Only":
        target_bg[:] = inner_color
        zone_info = f"Single-Zone: Inner Tint RGB {tuple(inner_color)}"

    else:  # "Dual-Zone Auto (White Margin + Cream Paper)"
        edges = cv2.Canny(gray, 40, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_box = None
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.65 and bh > h * 0.65:
                frame_box = (x, y, bw, bh)
                break

        if frame_box:
            fx, fy, bw, bh = frame_box
            inner_mask = np.zeros((h, w), dtype=bool)
            inner_mask[fy:fy + bh, fx:fx + bw] = True
            target_bg[inner_mask] = inner_color
            target_bg[~inner_mask] = outer_color
            zone_info = f"Dual-Zone Frame Detected: Margin RGB {tuple(outer_color)} | Paper RGB {tuple(inner_color)}"
        else:
            target_bg[:] = inner_color
            zone_info = f"Paper Tint RGB {tuple(inner_color)}"

    # Apply background fill with soft anti-aliasing or hard threshold
    if anti_alias:
        t_low = float(final_thresh - 6)
        t_high = float(final_thresh + 6)
        alpha = np.clip((t_high - gray.astype(float)) / (t_high - t_low), 0.0, 1.0)[:, :, None]
        cleaned = (alpha * img_np.astype(float) + (1.0 - alpha) * target_bg.astype(float)).astype(np.uint8)
    else:
        erasable = (gray >= final_thresh) & (~grid_mask_dilated)
        cleaned = img_np.copy()
        cleaned[erasable] = target_bg[erasable]

    # Re-apply or normalize protected gridlines
    if protect_tables:
        # User-tunable contrast: 0% = original faint shade; 50% = crisp clear (180); 100% = bold dark (130)
        contrast_factor = np.clip(grid_contrast / 100.0, 0.0, 1.0)
        dark_floor = 130
        line_render_val = int(round(med_val - contrast_factor * max(0, med_val - dark_floor)))

        if snapped_grid_applied:
            is_dark_text = (gray < final_thresh - 15)
            grid_pixels = (clean_grid > 0) & (~is_dark_text)
            cleaned[grid_pixels] = [line_render_val, line_render_val, line_render_val]
        else:
            darkened = grid_mask_dilated & (gray < med_val) & (gray >= final_thresh - 20)
            cleaned[darkened] = [line_render_val, line_render_val, line_render_val]

    elapsed_ms = (time.time() - t0) * 1000

    # Auto-save cleaned document
    doc_index = len(os.listdir(CLEANED_DOCS_DIR)) + 1
    save_path = os.path.join(CLEANED_DOCS_DIR, f"{doc_index}_cleaned.png")
    Image.fromarray(cleaned).save(save_path)

    extra_notes = []
    if snapped_grid_applied:
        extra_notes.append("Grid Snapped 100% Straight")
    if anti_alias:
        extra_notes.append("Anti-Aliased")
    if stamp_filter != "None (Standard)":
        extra_notes.append(stamp_filter)
    notes_str = f" | {', '.join(extra_notes)}" if extra_notes else ""

    status = (
        f"{auto_profile_note}Cleaned in **{elapsed_ms:.1f} ms**! ({zone_info}, Threshold: {final_thresh}{notes_str})\n"
        f"Saved to `dataset/cleaned_documents/{doc_index}_cleaned.png`"
    )
    return Image.fromarray(cleaned), status
