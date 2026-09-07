import numpy as np

from .classifier import classify_image
from .document_cleaner import clean_document_auto
from .photo_inpainter import auto_detect_and_inpaint

DOCUMENT_LABEL = "📄 Document"
PHOTO_LABEL = "🎨 Photo"


def detect_mode(image):
    """Runs the classifier on an uploaded image for the Auto tab. Returns
    the pre-selected mode label and a status line explaining why, so the
    user can see and override the guess before anything is processed."""
    if image is None:
        return DOCUMENT_LABEL, "Upload an image to auto-detect its type."

    img_np = np.array(image.convert("RGB"))
    decision = classify_image(img_np)
    label = DOCUMENT_LABEL if decision.route == "document" else PHOTO_LABEL
    pct = int(round(decision.confidence * 100))
    status = (
        f"🔍 Detected **{label}** ({pct}% confidence). {decision.reason} "
        f"Wrong? Change the mode below before removing the watermark."
    )
    return label, status


def auto_remove_watermark(image, mode_label: str, auto_save: bool):
    """Dispatches to the document cleaner or the photo inpainter using each
    pipeline's smart-auto defaults, based on the (possibly user-corrected)
    detected mode. Both underlying pipelines are untouched -- this only
    picks between them and supplies sensible default settings."""
    if image is None:
        return None, "Please upload an image first."

    if mode_label == DOCUMENT_LABEL:
        result, status = clean_document_auto(
            image,
            sensitivity_offset=0,
            bg_mode="Dual-Zone Auto (White Margin + Cream Paper)",
            protect_tables=True,
            snap_gridlines=True,
            anti_alias=True,
            line_thickness="1px (Hairline)",
            stamp_filter="None (Standard)",
            grid_contrast=50,
            smart_auto=True,
        )
        return result, f"📄 **Document mode** — {status}"

    img_np = np.array(image.convert("RGB"))
    editor_data = {"background": img_np, "layers": [], "composite": None}
    result, _mask, status, _bg, _m, _res = auto_detect_and_inpaint(
        editor_data,
        conf=0.12,
        box_dilation=10,
        model_name="YOLO11 General Watermarks",
        smart_mask=True,
        auto_save=auto_save,
    )
    return result, f"🎨 **Photo mode** — {status}"
