"""Visual debugger for Method 3's detection + segmentation pipeline.

"The watermark wasn't removed" has at least three different root causes
that look IDENTICAL from Method 3's cleaned output alone:
  1. YOLO found nothing at all (wrong model, confidence too high, or this
     watermark style just isn't in either checkpoint's training data).
  2. YOLO found it, but the false-positive filter in segmenter.py rejected
     every instance (its saturation/alpha heuristics, calibrated on a
     different set of documents, don't always transfer).
  3. Detection + filtering are both fine, but SAM's refined mask doesn't
     actually cover the mark (or covers too much / too little of it).

This module runs the exact same `detect_watermark_masks` doc_segment.py
uses for removal, but renders an annotated overlay + a per-instance report
instead of touching any pixels, so which of the three it is is visible at
a glance instead of guessed at.
"""

import cv2
import numpy as np
from PIL import Image

from .segmenter import detect_watermark_masks

# Box outline color by source model -- lets you see at a glance whether a
# detection came from the checkpoint that's strong on tiled marks or the
# one that's strong on isolated stamps/logos (see segmenter.py's module
# docstring: the two are complementary, not redundant).
_SOURCE_COLORS = {
    "YOLO11s": (66, 133, 244),  # blue
    "YOLO11 General": (255, 152, 0),  # orange
}
_DEFAULT_BOX_COLOR = (128, 128, 128)
_ACCEPT_TINT = (52, 199, 89)  # green
_REJECT_TINT = (255, 59, 48)  # red
_TINT_STRENGTH = 0.45


def debug_detect(doc_image, conf: float = 0.15, model_choice: str = "Both (Union)", use_sam: bool = True):
    """Runs detection+refinement+filtering and returns (overlay_image,
    markdown_report). Never runs removal -- this is inspection only.

    overlay: raw YOLO boxes drawn as outlines (colored by source model,
    labeled with confidence), with each instance's SAM-refined mask tinted
    green if the false-positive filter accepted it or red if it rejected
    it (see segmenter._classify_instance for the criteria).
    """
    if doc_image is None:
        return None, "Please upload a document image first."

    img_np = np.array(doc_image.convert("RGB"))

    _, meta = detect_watermark_masks(img_np, conf=conf, model_choice=model_choice, use_sam=use_sam)

    overlay = _render_overlay(img_np, meta["instances"])
    report = _render_report(meta, conf, model_choice, use_sam)

    return Image.fromarray(overlay), report


def _render_overlay(img_np: np.ndarray, instances: list) -> np.ndarray:
    overlay = img_np.astype(np.float64).copy()

    # Mask tints first, so the box outlines/labels drawn next stay crisp
    # on top instead of being softened by the blend.
    for inst in instances:
        tint = np.array(_ACCEPT_TINT if inst["accepted"] else _REJECT_TINT, dtype=np.float64)
        m = inst["mask"] > 0
        if np.any(m):
            overlay[m] = (1 - _TINT_STRENGTH) * overlay[m] + _TINT_STRENGTH * tint

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    h, w = img_np.shape[:2]
    for i, inst in enumerate(instances, 1):
        x1, y1, x2, y2 = (int(round(v)) for v in inst["box"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        color = _SOURCE_COLORS.get(inst["source"], _DEFAULT_BOX_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"#{i} {inst['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y1 if y1 - th - 6 >= 0 else min(h, y2 + th + 6)
        cv2.rectangle(overlay, (x1, ly - th - 6), (x1 + tw + 6, ly), color, -1)
        cv2.putText(overlay, label, (x1 + 3, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def _render_report(meta: dict, conf: float, model_choice: str, use_sam: bool) -> str:
    instances = meta["instances"]
    lines = [
        f"**Detection:** {meta['detect_ms']:.0f} ms &nbsp;|&nbsp; **SAM refine:** {meta['refine_ms']:.0f} ms "
        f"(used_sam={meta['used_sam']}" + (f", error: `{meta['sam_error']}`" if meta["sam_error"] else "") + ")",
        f"**Raw candidates:** {len(instances)} &nbsp;|&nbsp; "
        f"**Accepted:** {meta['accepted_count']} &nbsp;|&nbsp; "
        f"**Rejected:** {meta['rejected_count']} &nbsp;|&nbsp; "
        f"**Final mask coverage:** {meta['coverage'] * 100:.2f}% of page",
        "",
    ]

    if not instances:
        lines.append(
            f"⚠️ **No detections at all** at confidence ≥ {conf} with model(s) = *{model_choice}*. "
            "This means Method 3 removed nothing because it never saw a candidate in the first place -- "
            "not that a candidate was found and rejected. Try lowering the confidence slider first; if that "
            "doesn't help, try switching model choice (`YOLO11s` vs `YOLO11 General`) -- the two checkpoints "
            "are trained on different watermark styles and are not redundant with each other."
        )
        return "\n".join(lines)

    lines += [
        "| # | Source | Conf | Box (x1,y1,x2,y2) | Status | Ink α (median) | BG saturation | Page cov. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, inst in enumerate(instances, 1):
        box_str = ", ".join(str(int(round(v))) for v in inst["box"])
        status = "✅ accepted" if inst["accepted"] else f"❌ {inst['reject_reason']}"
        alpha = inst.get("median_ink_alpha")
        alpha_str = f"{alpha:.2f}" if alpha is not None else "—"
        sat = inst.get("bg_saturation")
        sat_str = f"{sat:.0f}" if sat is not None else "—"
        cov = inst.get("page_coverage")
        cov_str = f"{cov * 100:.2f}%" if cov is not None else "—"
        lines.append(f"| {i} | {inst['source']} | {inst['conf']:.2f} | {box_str} | {status} | {alpha_str} | {sat_str} | {cov_str} |")

    if meta["accepted_count"] == 0:
        lines.append(
            "\n⚠️ **Every candidate was rejected** by the false-positive filter (see the Status column above). "
            "This is why the cleaned output looks unchanged even though detections exist. The filter's cutoffs "
            "(`_OPAQUE_REJECT_ALPHA`, `_CHROME_SATURATION_REJECT`, `_MAX_INSTANCE_COVERAGE` in `segmenter.py`) "
            "were calibrated on a different set of documents and may need adjusting for this one."
        )

    lines.append(
        "\n\n**Legend:** blue box = YOLO11s, orange box = YOLO11 General, gray = unrecognized source &nbsp;|&nbsp; "
        "green tint = accepted instance mask, red tint = rejected instance mask (SAM's refined shape, not the raw box)."
    )
    return "\n".join(lines)
