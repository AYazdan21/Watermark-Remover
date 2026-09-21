"""Visual debuggers for Method 3's and Method 4's detection pipelines.

"The watermark wasn't removed" has at least three different root causes
that look IDENTICAL from Method 3's cleaned output alone:
  1. YOLO found nothing at all (wrong model, confidence too high, or this
     watermark style just isn't in either checkpoint's training data).
  2. YOLO found it, but the false-positive filter in segmenter.py rejected
     every instance (its saturation/alpha heuristics, calibrated on a
     different set of documents, don't always transfer).
  3. Detection + filtering are both fine, but SAM's refined mask doesn't
     actually cover the mark (or covers too much / too little of it).

`debug_detect` runs the exact same `detect_watermark_masks` doc_segment.py
uses for removal, but renders an annotated overlay + a per-instance report
instead of touching any pixels, so which of the three it is is visible at
a glance instead of guessed at.

`debug_detect_boxes` is Method 4's counterpart: it runs the exact same
`detector.detect_watermark_boxes` + `doc_detect.estimate_background_map` +
`doc_detect.apply_box_strategy` pipeline Method 4 uses for removal (no
false-positive filter exists on that path -- see detector.py's module
docstring for why), and renders which pixels the chosen strategy actually
changed inside each box, plus a per-box report -- so what you see here is
exactly what removal will do, not an approximation of it.
"""

import os

import cv2
import numpy as np
from PIL import Image

from . import detector, doc_detect
from .segmenter import detect_watermark_masks

# Box outline color by source model -- lets you see at a glance whether a
# detection came from the checkpoint that's strong on tiled marks, the one
# that's strong on isolated stamps/logos (see segmenter.py's module
# docstring: the two are complementary, not redundant), or the finetuned
# direct-mask model that is now the primary path.
_SOURCE_COLORS = {
    "YOLO11s": (66, 133, 244),  # blue
    "YOLO11 General": (255, 152, 0),  # orange
    "Finetuned (AriaTender)": (0, 150, 136),  # teal
    "Finetuned (Half-Frozen)": (156, 39, 176),  # purple
    "Finetuned (Full, New Dataset)": (121, 85, 72),  # brown
    "Finetuned (Half-Frozen, New Dataset)": (130, 119, 23),  # olive
    "YOLO11s Detect (Half-Frozen, New Dataset)": (216, 27, 96),  # magenta
}
_DEFAULT_BOX_COLOR = (128, 128, 128)
_ACCEPT_TINT = (52, 199, 89)  # green
_REJECT_TINT = (255, 59, 48)  # red
_TINT_STRENGTH = 0.45

# Method 4's overlay tints, with this amber highlight, ONLY the pixels the
# chosen strategy actually changed inside each box (not the whole box
# interior -- Threshold + Flat Fill and Bounded Subtractive both
# deliberately leave some box pixels alone, and that's the whole point of
# this rewrite, so the overlay has to show it truthfully).
_M4_FILL_TINT = (255, 193, 7)  # amber
_M4_TINT_STRENGTH = 0.30
_M4_RAW_BOX_COLOR = (0, 0, 0)  # thin outline showing the box BEFORE padding


def debug_detect(
    doc_image,
    conf: float = 0.25,
    model_choice: str = "Finetuned (AriaTender)",
    use_sam: bool = True,
    allow_colored_bg: bool = True,
):
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

    _, meta = detect_watermark_masks(
        img_np,
        conf=conf,
        model_choice=model_choice,
        use_sam=use_sam,
        allow_colored_bg=allow_colored_bg,
    )

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
            "doesn't help, try switching model choice. `Finetuned (AriaTender)` is the primary path now -- "
            "trained directly on this project's synthetic AriaTender composites at imgsz 1024, it detects "
            "real watermark instances the two legacy detectors miss entirely (measured 72/73 hit rate over "
            "wm_testset/images at conf=0.25). `YOLO11s` / `YOLO11 General` remain available for watermark "
            "styles outside that training set -- the two are trained on different styles and are not "
            "redundant with each other, but neither is the first thing to reach for anymore."
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
            "(`_OPAQUE_REJECT_ALPHA` / `_SEG_OPAQUE_REJECT_ALPHA`, `_CHROME_SATURATION_REJECT`, "
            "`_MAX_INSTANCE_COVERAGE` / `_SEG_MAX_INSTANCE_COVERAGE` in `segmenter.py`) "
            "were calibrated on a different set of documents and may need adjusting for this one."
        )

    lines.append(
        "\n\n**Legend:** blue box = YOLO11s, orange box = YOLO11 General, teal box = Finetuned (AriaTender), "
        "purple box = Finetuned (Half-Frozen), brown box = Finetuned (Full, New Dataset), "
        "olive box = Finetuned (Half-Frozen, New Dataset), "
        "gray = unrecognized source &nbsp;|&nbsp; "
        "green tint = accepted instance mask, red tint = rejected instance mask (SAM's refined shape for the "
        "two legacy models, or the model's own direct mask for the finetuned models -- SAM never runs on "
        "those paths)."
    )
    return "\n".join(lines)


# ===========================================================================
# Method 4: detection + box deblending debugger
# ===========================================================================


def debug_detect_boxes(
    doc_image,
    conf: float = 0.25,
    model_choice: str = detector.DEFAULT_DETECT_MODEL,
    box_padding: int = 0,
    removal_strategy: str = doc_detect.STRATEGY_THRESHOLD_FILL,
    thresh_offset: int = 0,
    anti_alias: bool = True,
    stamp_filter: str = "None (Standard)",
):
    """Runs Method 4's exact detection (detector.detect_watermark_boxes),
    background-map estimation (doc_detect.estimate_background_map) and
    removal (doc_detect.apply_box_strategy) and returns (overlay_image,
    markdown_report). The cleaned array apply_box_strategy returns is used
    ONLY to compute the overlay's diff -- no pixels are written back to
    doc_image, so this is inspection only, but it calls the exact same
    functions clean_document_detect calls, so the overlay/report show
    exactly what M4 will change before it changes it.
    """
    if doc_image is None:
        return None, "Please upload a document image first."

    if removal_strategy == doc_detect.STRATEGY_ALPHA_NET and not os.path.exists(doc_detect.ALPHA_NET_WEIGHTS):
        # Same non-crashing behaviour as clean_document_detect: skip
        # detection entirely (there's nothing this strategy could do with
        # it anyway) and show the exact same missing-weights message,
        # returning the input unchanged rather than None so there is still
        # something to look at.
        return doc_image, doc_detect.alpha_net_weights_missing_message(doc_detect.ALPHA_NET_WEIGHTS)
    if removal_strategy == doc_detect.STRATEGY_STAMP_FIT and doc_detect.stamp_fit_locator_weights() is None:
        return doc_image, doc_detect.stamp_fit_weights_missing_message()

    img_np = np.array(doc_image.convert("RGB"))

    instances, meta = detector.detect_watermark_boxes(
        img_np, conf=conf, model_choice=model_choice, box_padding=int(box_padding)
    )
    background = doc_detect.estimate_background_map(img_np, instances)
    cleaned, page_info, per_box_info = doc_detect.apply_box_strategy(
        img_np,
        instances,
        removal_strategy,
        thresh_offset=thresh_offset,
        anti_alias=anti_alias,
        stamp_filter=stamp_filter,
        background=background,
    )

    overlay = _render_detect_overlay(img_np, cleaned, instances, box_padding=int(box_padding),
                                     stamp_alpha=page_info.get("stamp_alpha"))
    report = _render_detect_report(
        img_np, instances, meta, conf, model_choice, page_info, per_box_info,
        removal_strategy, anti_alias, box_padding=int(box_padding),
    )

    return Image.fromarray(overlay), report


def _render_detect_overlay(img_np: np.ndarray, cleaned: np.ndarray, instances: list, box_padding: int = 0,
                           stamp_alpha: np.ndarray = None) -> np.ndarray:
    overlay = img_np.astype(np.float64).copy()
    h, w = img_np.shape[:2]

    # Amber tint ONLY the pixels the chosen strategy actually changed
    # (cleaned != img_np), not every pixel inside a box -- both M4
    # strategies deliberately leave some box pixels (real ink, content
    # brighter/darker than they touch) untouched, and this overlay exists
    # to show that truthfully rather than implying the whole box is wiped.
    changed = np.any(cleaned != img_np, axis=2)
    tint = np.array(_M4_FILL_TINT, dtype=np.float64)
    if np.any(changed):
        overlay[changed] = (1 - _M4_TINT_STRENGTH) * overlay[changed] + _M4_TINT_STRENGTH * tint

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Stamp Fit only: outline of every accepted stamp's fitted footprint, so
    # the fit itself (not just the pixels it changed) can be checked by eye.
    if stamp_alpha is not None and np.any(stamp_alpha > 1e-4):
        foot = (stamp_alpha > 0.02).astype(np.uint8)
        edge = cv2.morphologyEx(foot, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)) > 0
        overlay[edge] = _M4_STAMP_OUTLINE_COLOR

    for i, inst in enumerate(instances, 1):
        x1, y1, x2, y2 = inst["box"]
        box_color = _SOURCE_COLORS.get(inst["source"], _DEFAULT_BOX_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

        if box_padding > 0:
            rx1, ry1, rx2, ry2 = (int(round(v)) for v in inst["raw_box"])
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), _M4_RAW_BOX_COLOR, 1)

        label = f"#{i} {inst['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y1 if y1 - th - 6 >= 0 else min(h, y2 + th + 6)
        cv2.rectangle(overlay, (x1, ly - th - 6), (x1 + tw + 6, ly), box_color, -1)
        cv2.putText(overlay, label, (x1 + 3, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


_M4_STAMP_OUTLINE_COLOR = (0, 200, 0)


def _render_stamp_fit_report(lines: list, instances: list, per_box_info: list, page_info: dict) -> str:
    sf = page_info["stamp_fit"]
    lines += [
        f"**Stamp Fit:** {len(sf['marks'])} stamp(s) accepted, {len(sf['rejected'])} candidate fit(s) rejected "
        f"&nbsp;|&nbsp; network {sf['infer_ms']:.0f} ms, locate {sf['locate_ms']:.0f} ms, remove "
        f"{sf['remove_ms']:.0f} ms &nbsp;|&nbsp; locator weights: `{sf['locator_weights']}`",
        "",
    ]
    if sf["marks"]:
        lines += [
            "| # | Kind | Parts (name: scale, x, y, blur) | Local IoU | Page change on strokes / control | "
            "Background | Strength & ink per region |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, m in enumerate(sf["marks"], 1):
            parts = "; ".join(f"{p['name']}: {p['scale']:.3f}, {p['x']:.2f}, {p['y']:.2f}, {p.get('sigma', 0):.2f}"
                              for p in m["parts"])
            regions = "; ".join(f"{k}: o={v['strength']} ink=({v['ink'][0]}, {v['ink'][1]}, {v['ink'][2]})"
                                + ("" if v["source"] == "own" else f" [{v['source']}]")
                                for k, v in m["regions"].items())
            lines.append(f"| {i} | {m['kind']} | {parts} | {m['iou']} | {m['change']} / {m['control']} | "
                         f"{m.get('background', '')} | {regions} |")
        ep_lines = []
        for i, m in enumerate(sf["marks"], 1):
            for k, v in m["regions"].items():
                ep = v.get("edge_profile")
                if not ep:
                    continue
                detail = (f"{ep['reason']}, residual {ep['residual_before']} -> {ep['residual_after']}"
                          if ep.get("residual_before") is not None else ep["reason"])
                ep_lines.append(f"- mark {i} `{k}`: edge profile {'used' if ep['used'] else 'not used'} ({detail})")
        if ep_lines:
            lines += [""] + ep_lines
    else:
        lines.append("No AriaTender stamp was accepted on this page.")
    if sf["rejected"]:
        rej = "; ".join(f"{r['kind']} (IoU {r['iou']}, change {r['change']} vs control {r['control']}, "
                        f"width {r['width_px']} px)" for r in sf["rejected"])
        lines += ["", f"**Rejected fits:** {rej}"]
    if instances:
        lines += ["", "| # | Conf | Box (x1,y1,x2,y2) | Covered by stamp | Handled by | Changed px |", "|---|---|---|---|---|---|"]
        for i, (inst, pbi) in enumerate(zip(instances, per_box_info), 1):
            x1, y1, x2, y2 = inst["box"]
            if pbi["covered_by_stamp"]:
                handler = "Stamp Fit"
            elif pbi["fallback"]:
                handler = sf["fallback_strategy"]
            elif sf["marks"]:
                handler = "nothing (stamp confirmed on page; likely false positive)"
            else:
                handler = "nothing (no fallback weights)"
            lines.append(f"| {i} | {inst['conf']:.2f} | {x1}, {y1}, {x2}, {y2} | "
                         f"{pbi['stamp_cover_frac'] * 100:.0f}% | {handler} | {pbi['changed_px']} |")
    lines.append(
        "\n\n**Legend:** green outline = footprint of each accepted stamp, fitted from the known AriaTender "
        "artwork (the mask is the artwork's own shape, not a per-pixel guess); amber tint = pixels actually "
        "changed. A fit is accepted only if the page itself changes along the fitted strokes clearly more "
        "than along the same shape shifted off the mark (change / control). Strength o and ink are "
        "estimated per region from the page; removal is the exact inverse (observed - a*ink)/(1 - a) with "
        "a = o * coverage, so text under the mark is recovered, not painted over. If no stamp is accepted, "
        "every detection box falls back to the Alpha Network strategy; if one is, uncovered boxes are left "
        "alone as likely false positives. See `watermark_remover/stamp_fit.py`."
    )
    return "\n".join(lines)


def _render_detect_report(
    img_np: np.ndarray,
    instances: list,
    meta: dict,
    conf: float,
    model_choice: str,
    page_info: dict,
    per_box_info: list,
    removal_strategy: str,
    anti_alias: bool,
    box_padding: int = 0,
) -> str:
    lines = [
        f"**Detection:** {meta['detect_ms']:.0f} ms &nbsp;|&nbsp; **Boxes found:** {len(instances)} "
        f"&nbsp;|&nbsp; **Union coverage:** {meta['coverage'] * 100:.2f}% of page",
        f"**Model:** {meta['model']} &nbsp;|&nbsp; **Conf >=** {conf} &nbsp;|&nbsp; **Box padding:** {box_padding} px",
        f"**Strategy:** {removal_strategy}"
        + (
            f" &nbsp;|&nbsp; **Channel:** {page_info['channel']} &nbsp;|&nbsp; "
            f"**Otsu:** {page_info['otsu']} &nbsp;|&nbsp; **Final threshold:** {page_info['final_thresh']} "
            f"&nbsp;|&nbsp; **Anti-alias:** {anti_alias}"
            if instances and removal_strategy == doc_detect.STRATEGY_THRESHOLD_FILL
            else ""
        ),
        "",
        "ℹ️ Method 4 has **no false-positive filter** -- every box shown below is exactly what "
        "removal will act on; there is no accepted/rejected split like Method 3's debugger.",
        "",
    ]

    if removal_strategy == doc_detect.STRATEGY_STAMP_FIT:
        return _render_stamp_fit_report(lines, instances, per_box_info, page_info)

    if not instances:
        lines.append(
            f"⚠️ **No detections at all** at confidence ≥ {conf} with model = *{model_choice}*. "
            "This means Method 4 removed nothing because it never saw a candidate box in the first "
            "place. Try lowering the confidence slider."
        )
        return "\n".join(lines)

    is_threshold = removal_strategy == doc_detect.STRATEGY_THRESHOLD_FILL
    is_alpha_net = removal_strategy == doc_detect.STRATEGY_ALPHA_NET
    if is_threshold:
        lines += [
            "| # | Conf | Box (x1,y1,x2,y2) | Size (w x h) | Page % | BG median (RGB) | BG gray min-max | "
            "Changed px | Replaced % |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    elif is_alpha_net:
        lines += [
            "| # | Conf | Box (x1,y1,x2,y2) | Size (w x h) | Page % | Alpha mean | Alpha max | Alpha px | "
            "Changed px | Polarity | Ink px kept |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| # | Conf | Box (x1,y1,x2,y2) | Size (w x h) | Page % | BG median (RGB) | BG gray min-max | "
            "Changed px | Polarity | Candidates | Clusters | D (RGB) | Ink px kept |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

    page_area = float(img_np.shape[0] * img_np.shape[1]) or 1.0
    for i, (inst, pbi) in enumerate(zip(instances, per_box_info), 1):
        x1, y1, x2, y2 = inst["box"]
        bw, bh = x2 - x1, y2 - y1
        box_str = f"{x1}, {y1}, {x2}, {y2}"
        size_str = f"{bw} x {bh}"
        page_pct_str = f"{(bw * bh) / page_area * 100:.3f}%"
        changed_px = pbi.get("changed_px", 0)

        if is_alpha_net:
            alpha_mean_str = f"{pbi.get('alpha_mean', 0.0):.4f}"
            alpha_max_str = f"{pbi.get('alpha_max', 0.0):.4f}"
            alpha_px = pbi.get("alpha_px", 0)
            polarity = pbi.get("polarity", "—")
            ink_kept = pbi.get("ink_px_kept", 0)
            lines.append(
                f"| {i} | {inst['conf']:.2f} | {box_str} | {size_str} | {page_pct_str} | "
                f"{alpha_mean_str} | {alpha_max_str} | {alpha_px} | {changed_px} | "
                f"{polarity} | {ink_kept} |"
            )
            continue

        bgm = pbi.get("bg_median", [0, 0, 0])
        bg_median_str = f"({bgm[0]}, {bgm[1]}, {bgm[2]})"
        bg_minmax_str = f"{pbi.get('bg_gray_min', 0)}-{pbi.get('bg_gray_max', 0)}"

        if is_threshold:
            replaced_pct_str = f"{pbi.get('replaced_frac', 0.0) * 100:.1f}%"
            lines.append(
                f"| {i} | {inst['conf']:.2f} | {box_str} | {size_str} | {page_pct_str} | "
                f"{bg_median_str} | {bg_minmax_str} | {changed_px} | {replaced_pct_str} |"
            )
        else:
            polarity = pbi.get("polarity", "—")
            candidates = pbi.get("candidate_px", 0)
            n_clusters = pbi.get("n_clusters", 0)
            d_str = "; ".join(f"({d[0]}, {d[1]}, {d[2]})" for d in pbi.get("D", [])) or "—"
            ink_kept = pbi.get("ink_px_kept", 0)
            lines.append(
                f"| {i} | {inst['conf']:.2f} | {box_str} | {size_str} | {page_pct_str} | "
                f"{bg_median_str} | {bg_minmax_str} | {changed_px} | "
                f"{polarity} | {candidates} | {n_clusters} | {d_str} | {ink_kept} |"
            )

    if is_alpha_net:
        infer_ms = page_info.get("alpha_net_infer_ms", 0.0)
        weights_path = page_info.get("alpha_net_weights", doc_detect.ALPHA_NET_WEIGHTS)
        lines.append(
            f"\n\n**Legend:** amber tint = the pixels this strategy actually changed inside each box "
            f"(where predicted alpha was low it may change nothing at all); Alpha mean/max/px come from "
            f"the network's own predicted per-pixel opacity over this box's own won pixels (px = won "
            f"pixels with alpha > 0.02); no background estimate is used by this strategy's OWN removal "
            f"math -- it inverts the compositing equation directly instead, but Polarity/Ink px kept come "
            f"from the same background-map-derived ink test Bounded Subtractive uses "
            f"(`doc_detect.ALPHA_NET_INK_GUARD`, on by default): Ink px kept = won pixels classified as "
            f"real ink and therefore left byte-identical rather than overwritten by the network's "
            f"recovery. Weights: `{weights_path}` "
            f"(inference {infer_ms:.1f} ms, shared across every box on this page). Outline colour = "
            f"detection model (magenta = YOLO11s Detect (Half-Frozen, New Dataset)); thin black outline "
            f"(only shown when box padding > 0) = the raw, unpadded box before padding was applied."
        )
        return "\n".join(lines)

    lines.append(
        "\n\n**Legend:** amber tint = the pixels this strategy actually changed inside each box (not the "
        "whole box -- both strategies deliberately leave some pixels alone); BG median/min-max come from "
        "`doc_detect.estimate_background_map`'s PER-PIXEL background estimate over this box's own won "
        "pixels (not one ring colour for the whole box) -- a wide gray min-max (e.g. \"225-255\") means this "
        "box straddles two real zones of the page, and the map is what keeps each zone's own real pixels "
        "reading their own shade instead of an in-between compromise; outline colour = detection model "
        "(magenta = YOLO11s Detect (Half-Frozen, New Dataset)); thin black outline (only shown when box "
        "padding > 0) = the raw, unpadded box before padding was applied."
    )
    return "\n".join(lines)
