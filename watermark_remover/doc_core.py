"""Shared core for the Document tab's three selectable cleaning methods.

The pipeline below is structured exactly like the project's ORIGINAL
document-cleaning algorithm (frozen, byte-for-byte, at
``tests/reference/legacy_document_cleaner.py`` -- see that file's docstring
for provenance). Every stage -- profile auto-detect, channel/stamp-filter
selection, margin+paper colour sampling, Otsu threshold, gridline
detect/protect/snap/restore, and dataset saving -- is written ONCE here and
shared by both manual methods. The two methods differ in exactly one
stage, the removal step itself:

- ``METHOD_THRESHOLD`` ("Threshold + Flat Fill (Original)"): the original
  anti-aliased alpha blend toward a flat background color (or the hard
  threshold branch when anti-aliasing is off). This intentionally
  reproduces the original algorithm bug-for-bug -- see the docstring on
  ``auto_detect_document_profile`` below and on ``_remove_flat_fill`` for
  the specific bugs preserved on purpose.
- ``METHOD_UNMIX`` ("Threshold + Alpha Unmixing"): identical in every other
  respect, but recovers the true pixel under the mark via alpha unmixing
  (see ``unmixer.unmix_region``) instead of flattening it to the
  background estimate. Keeping every other stage (including the original's
  gridline bugs) identical to METHOD_THRESHOLD means any output difference
  between the two methods isolates that one variable.

``METHOD_SEGMENT`` ("Segmentation + Deblending") is a third method built in
parallel (``doc_segment.py``) and is only dispatched to here, not
implemented here.

``METHOD_DETECT`` ("Detection + Box Deblending") is a fourth method
(``doc_detect.py``), also only dispatched to here. Where METHOD_SEGMENT
detects a tight per-instance mask, METHOD_DETECT detects plain axis-aligned
boxes with a Detect-head YOLO model (no masks at all) and removes the
watermark by working inside each box with one of two strategies -- Method
1's threshold-and-flatten math restricted to the box, or Method 3's bounded
subtractive correction adapted to the box -- both of which leave real
content under the mark alone as much as their own math allows, unlike the
whole-box flat fill this used to be. See ``doc_detect.py``'s module
docstring for the full story, both strategies' honest limits, and the hard
invariant.
"""
import os
import time

import cv2
import numpy as np
from PIL import Image

from .config import CLEANED_DOCS_DIR, DOCUMENT_ORIGINALS_DIR
from .unmixer import unmix_region

METHOD_THRESHOLD = "Threshold + Flat Fill (Original)"
METHOD_UNMIX = "Threshold + Alpha Unmixing"
METHOD_SEGMENT = "Segmentation + Deblending"
METHOD_DETECT = "Detection + Box Deblending"

METHOD_CHOICES = [METHOD_THRESHOLD, METHOD_UNMIX, METHOD_SEGMENT, METHOD_DETECT]


def auto_detect_document_profile(img_np):
    """
    Analyzes document image geometry, lines, margins, and paper colors to automatically
    configure optimal settings for 1-click watermark removal with zero manual setup.

    This is a verbatim copy of the original algorithm's profiler (see
    ``tests/reference/legacy_document_cleaner.py``), including its known
    quirks: gridline peaks are corroborated only by projection-count peaks
    (no continuous-run check), so a tiled watermark's own strokes can in
    rare cases be counted toward ``has_table``. Preserved deliberately so
    M1/M2 match the original bug-for-bug.
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


def _remove_flat_fill(img_np, gray, final_thresh, grid_mask_dilated, target_bg, anti_alias):
    """M1's removal stage: the original anti-aliased alpha blend toward a
    flat background color (or the hard-threshold branch when anti-aliasing
    is off). Bug-for-bug faithful to the original: when ``anti_alias`` is
    True (the default), ``grid_mask_dilated`` is computed by the caller but
    is NOT consulted here at all -- gridlines get alpha-blended away like
    everything else and rely entirely on the later redraw step to look
    right. This is a known original bug, preserved deliberately.
    """
    if anti_alias:
        t_low = float(final_thresh - 6)
        t_high = float(final_thresh + 6)
        alpha = np.clip((t_high - gray.astype(float)) / (t_high - t_low), 0.0, 1.0)[:, :, None]
        cleaned = (alpha * img_np.astype(float) + (1.0 - alpha) * target_bg.astype(float)).astype(np.uint8)
    else:
        erasable = (gray >= final_thresh) & (~grid_mask_dilated)
        cleaned = img_np.copy()
        cleaned[erasable] = target_bg[erasable]
    return cleaned


def _remove_unmix(img_np, gray, final_thresh, grid_mask_dilated, target_bg):
    """M2's removal stage: alpha-unmix the region M1 would have erased,
    instead of flattening it to ``target_bg``.

    Region is exactly ``(gray >= final_thresh) & ~grid_mask_dilated`` -- the
    same discrete "would erase" set the hard-threshold branch uses -- so
    it's well defined regardless of the ``anti_alias`` setting (M1's
    anti-alias branch has no discrete erasure set of its own, only a
    continuous blend weight). ``background=target_bg`` is passed explicitly
    so ``unmix_region`` never falls back to ``cv2.inpaint`` -- inpainting
    from a text-riddled mask caused visible smudging historically. Evidence
    for mark-color estimation is how much darker each pixel is than its
    local neighbourhood (document watermarks are darker than paper); no
    dark-text exclusion is needed since the region is, by construction,
    only pixels brighter than the Otsu threshold, so real ink is already
    excluded.
    """
    h, w = gray.shape[:2]
    erasable = (gray >= final_thresh) & (~grid_mask_dilated)
    cleaned = img_np.copy()
    if np.any(erasable):
        local_bg = cv2.medianBlur(gray, max(15, (min(h, w) // 12) | 1))
        evidence = local_bg.astype(np.float64) - gray.astype(np.float64)
        result = unmix_region(img_np, erasable.astype(np.uint8) * 255, evidence=evidence, background=target_bg)
        cleaned[erasable] = result.recovered[erasable]
    return cleaned


def clean_document(
    doc_image,
    method=METHOD_THRESHOLD,
    sensitivity_offset: int = 0,
    bg_mode: str = "Dual-Zone Auto (White Margin + Cream Paper)",
    protect_tables: bool = True,
    snap_gridlines: bool = True,
    anti_alias: bool = True,
    line_thickness: str = "1px (Hairline)",
    stamp_filter: str = "None (Standard)",
    grid_contrast: int = 50,
    smart_auto: bool = True,
    seg_conf: float = 0.25,
    seg_model: str = "Finetuned (AriaTender)",
    seg_use_sam: bool = True,
    seg_use_template: bool = True,
    seg_strategy: str = "Template Deblending",
    seg_allow_colored_bg: bool = True,
    det_conf: float = 0.25,
    det_model: str = "YOLO11s Detect (Half-Frozen, New Dataset)",
    det_box_padding: int = 0,
    det_strategy: str = "Threshold + Flat Fill (per box)",
    det_thresh_offset: int = 0,
    det_anti_alias: bool = True,
    det_stamp_filter: str = "None (Standard)",
    save_dataset: bool = True,
):
    """
    De-blends semi-transparent watermarks and colored stamps from document scans
    while preserving outer white margins, inner paper tints, and optional table gridlines.

    ``method`` selects which of the four Document-tab algorithms runs:
    METHOD_THRESHOLD (M1), METHOD_UNMIX (M2), METHOD_SEGMENT (M3, delegated
    to ``doc_segment.py``), or METHOD_DETECT (M4, delegated to
    ``doc_detect.py``). ``save_dataset`` gates ALL writes to ``dataset/``
    -- when False, nothing is written, for any method.

    ``seg_conf``/``seg_model``/``seg_use_sam``/``seg_use_template``/``seg_strategy``/``seg_allow_colored_bg`` are
    M3-only and passed straight through to
    ``doc_segment.clean_document_segment``.

    ``det_conf``/``det_model``/``det_box_padding``/``det_strategy``/
    ``det_thresh_offset``/``det_anti_alias``/``det_stamp_filter`` are
    M4-only and passed straight through to
    ``doc_detect.clean_document_detect``. They sit after the M3 params and
    before ``save_dataset`` in this signature -- the UI's
    ``btn_clean_doc.click`` wires its inputs to this function positionally,
    so that order must be kept in sync with ``ui.py``.
    """
    if method == METHOD_SEGMENT:
        if doc_image is None:
            return None, "Please upload a document image first."
        try:
            from .doc_segment import clean_document_segment
        except ImportError:
            return None, (
                "⚠️ Method 3 (Segmentation + Deblending) is not available yet -- "
                "its implementation module (`watermark_remover/doc_segment.py`) hasn't landed."
            )
        # doc_segment works in numpy and reports a structured dict; this
        # dispatcher owns the PIL<->numpy conversion, the status rendering and
        # the dataset save, so all three methods present one interface to the
        # UI and honour save_dataset identically.
        seg_np = np.array(doc_image.convert("RGB"))
        cleaned_np, seg_status = clean_document_segment(
            seg_np,
            conf=seg_conf,
            model_choice=seg_model,
            use_sam=seg_use_sam,
            use_template=seg_use_template,
            removal_strategy=seg_strategy,
            allow_colored_bg=seg_allow_colored_bg,
        )

        seg_index = None
        if save_dataset:
            seg_index = len(os.listdir(CLEANED_DOCS_DIR)) + 1
            Image.fromarray(cleaned_np).save(os.path.join(CLEANED_DOCS_DIR, f"{seg_index}_cleaned.png"))
            Image.fromarray(seg_np).save(os.path.join(DOCUMENT_ORIGINALS_DIR, f"{seg_index}_original.png"))

        save_note = f"\nSaved to `dataset/cleaned_documents/{seg_index}_cleaned.png`" if seg_index is not None else ""
        message = seg_status.get("message") if isinstance(seg_status, dict) else str(seg_status)
        return Image.fromarray(cleaned_np), f"[M3: Segmentation + Deblending] {message}{save_note}"

    if method == METHOD_DETECT:
        if doc_image is None:
            return None, "Please upload a document image first."
        try:
            from .doc_detect import clean_document_detect
        except ImportError:
            return None, (
                "⚠️ Method 4 (Detection + Box Deblending) is not available yet -- "
                "its implementation module (`watermark_remover/doc_detect.py`) hasn't landed."
            )
        # doc_detect works in numpy and reports a structured dict, same
        # convention as doc_segment -- this dispatcher owns the PIL<->numpy
        # conversion, the status rendering and the dataset save, so all four
        # methods present one interface to the UI and honour save_dataset
        # identically.
        det_np = np.array(doc_image.convert("RGB"))
        cleaned_np, det_status = clean_document_detect(
            det_np,
            conf=det_conf,
            model_choice=det_model,
            box_padding=int(det_box_padding),
            removal_strategy=det_strategy,
            thresh_offset=int(det_thresh_offset),
            anti_alias=det_anti_alias,
            stamp_filter=det_stamp_filter,
        )

        det_index = None
        if save_dataset:
            det_index = len(os.listdir(CLEANED_DOCS_DIR)) + 1
            Image.fromarray(cleaned_np).save(os.path.join(CLEANED_DOCS_DIR, f"{det_index}_cleaned.png"))
            Image.fromarray(det_np).save(os.path.join(DOCUMENT_ORIGINALS_DIR, f"{det_index}_original.png"))

        save_note = f"\nSaved to `dataset/cleaned_documents/{det_index}_cleaned.png`" if det_index is not None else ""
        message = det_status.get("message") if isinstance(det_status, dict) else str(det_status)
        return Image.fromarray(cleaned_np), f"[M4: Detection + Box Deblending] {message}{save_note}"

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
                # NOTE (original bug, preserved deliberately for M1/M2):
                # this draws an UNCONDITIONAL line from min_x->max_x for
                # every row peak (and min_y->max_y per column peak),
                # fabricating a line across empty space wherever a row/col
                # doesn't actually have a continuous border the full span.
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

    # ---- The ONE stage that differs between M1 and M2 ----
    if method == METHOD_UNMIX:
        cleaned = _remove_unmix(img_np, gray, final_thresh, grid_mask_dilated, target_bg)
    else:
        cleaned = _remove_flat_fill(img_np, gray, final_thresh, grid_mask_dilated, target_bg, anti_alias)

    # Re-apply or normalize protected gridlines (identical for M1 and M2,
    # bugs included: grid_contrast=0 still redraws/flattens gridlines below
    # via line_render_val == med_val, rather than leaving them untouched).
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

    # Auto-save cleaned document -- gated entirely by save_dataset. When
    # False, nothing under dataset/ is touched (no listdir, no write).
    doc_index = None
    if save_dataset:
        doc_index = len(os.listdir(CLEANED_DOCS_DIR)) + 1
        save_path = os.path.join(CLEANED_DOCS_DIR, f"{doc_index}_cleaned.png")
        Image.fromarray(cleaned).save(save_path)
        Image.fromarray(img_np).save(os.path.join(DOCUMENT_ORIGINALS_DIR, f"{doc_index}_original.png"))

    extra_notes = []
    if snapped_grid_applied:
        extra_notes.append("Grid Snapped 100% Straight")
    if method == METHOD_UNMIX:
        extra_notes.append("Alpha Unmixed")
    elif anti_alias:
        extra_notes.append("Anti-Aliased")
    if stamp_filter != "None (Standard)":
        extra_notes.append(stamp_filter)
    notes_str = f" | {', '.join(extra_notes)}" if extra_notes else ""

    method_label = "M1: Flat Fill" if method != METHOD_UNMIX else "M2: Alpha Unmixing"
    save_note = f"\nSaved to `dataset/cleaned_documents/{doc_index}_cleaned.png`" if doc_index is not None else ""
    status = (
        f"{auto_profile_note}[{method_label}] Cleaned in **{elapsed_ms:.1f} ms**! "
        f"({zone_info}, Threshold: {final_thresh}{notes_str}){save_note}"
    )
    return Image.fromarray(cleaned), status
