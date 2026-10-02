"""Gradio tabs for Method 5 (Template Stamp Fit) and the Template Builder.

``build_template_tabs()`` is called from ``ui.build_ui`` inside its
``gr.Blocks`` context and adds two tabs. Every callback catches its own
exceptions and shows them in the markdown output instead of crashing the app.
Nothing here writes into ``dataset/``.
"""

import os
import traceback

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from . import template_library as tl
from .template_builder import build_template, list_pages, read_rgb
from .template_stamp_fit import clean_document_template, footprint_overlay
from .template_validate import validate_template

ALL_LABEL = "All library templates"
DEFAULT_LIB = os.path.join("assets", "stamps", "library")


def _err(e):
    return f"**Error:** {type(e).__name__}: {e}\n\n```\n{traceback.format_exc(limit=4)}\n```"


def _lib_names(library_dir):
    try:
        return tl.list_templates(library_dir)
    except Exception:
        return []


def _m5_choices(library_dir):
    ch = [(ALL_LABEL, "__all__")]
    ch += [(f"{n} (library)", n) for n in _lib_names(library_dir)]
    ch += [(n, n) for n in tl.builtin_names()]
    return ch


def _single_choices(library_dir, with_none=False):
    ch = [("None", "None")] if with_none else []
    ch += [(f"{n} (library)", n) for n in _lib_names(library_dir)]
    ch += [(n, n) for n in tl.builtin_names()]
    return ch


# ---------------------------------------------------------------------------
# Method 5
# ---------------------------------------------------------------------------

def m5_refresh(library_dir, current):
    ch = _m5_choices(library_dir)
    valid = {v for _, v in ch}
    keep = [c for c in (current or []) if c in valid]
    return gr.update(choices=ch, value=keep)


def m5_clean(image, templates, library_dir, min_score):
    if image is None:
        return None, None, "Upload an image first."
    if not templates:
        return None, None, "Choose at least one template (or *All library templates*)."
    try:
        img = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)[..., :3]
        cleaned, alpha, info, msg = clean_document_template(img, list(templates), library_dir or None,
                                                            min_score=float(min_score))
        overlay = footprint_overlay(img, alpha) if info["accepted"] else img
        return Image.fromarray(cleaned), Image.fromarray(np.ascontiguousarray(overlay)), msg
    except Exception as e:
        return None, None, _err(e)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _scribble_mask(editor, shape):
    """Brush layers -> boolean mask at ``shape`` = (h, w), the way
    photo_inpainter.inpaint_watermark reads them (alpha > 0)."""
    h, w = shape
    mask = np.zeros((h, w), bool)
    for layer in (editor or {}).get("layers", []) or []:
        if layer is None:
            continue
        a = np.array(layer)
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)
        if a.ndim == 3 and a.shape[2] == 4:
            mask |= a[:, :, 3] > 0
        elif a.ndim == 2:
            mask |= a > 0
    return mask


def b_load_folder(pages_dir):
    try:
        files = list_pages(pages_dir)
        if not files:
            return (gr.update(choices=[], value=None), None,
                    f"No images found in `{pages_dir}` (a folder of .jpg/.png/.bmp/.webp pages).")
        names = [os.path.basename(f) for f in files]
        img = read_rgb(files[0])
        editor = {"background": img, "layers": [], "composite": None} if img is not None else None
        return (gr.update(choices=names, value=names[0]), editor,
                f"Found **{len(files)}** pages. Outline the watermark on the seed page "
                f"(or type x, y, w, h below), name the template and click Build.")
    except Exception as e:
        return gr.update(), None, _err(e)


def b_seed_changed(pages_dir, seed_name):
    try:
        if not seed_name:
            return None
        p = os.path.join(tl.resolve_path(pages_dir) or "", seed_name)
        img = read_rgb(p)
        return {"background": img, "layers": [], "composite": None} if img is not None else None
    except Exception:
        return None


def b_build(pages_dir, seed_name, editor, bx, by, bw, bh, name, library_dir, max_pages, outer_iters,
            min_score, frame_w, overwrite, progress=gr.Progress()):
    try:
        if not pages_dir or not seed_name:
            return None, [], "Load a pages folder and choose a seed page first."
        seed_path = os.path.join(tl.resolve_path(pages_dir), seed_name)
        seed = read_rgb(seed_path)
        if seed is None:
            return None, [], f"Seed page `{seed_name}` could not be read."
        box, mask = None, None
        if all(v is not None for v in (bx, by, bw, bh)) and bw > 0 and bh > 0 and (bx >= 0 and by >= 0):
            box = (float(bx), float(by), float(bw), float(bh))
        else:
            bg = (editor or {}).get("background")
            if bg is None:
                return None, [], "Brush over the watermark on the seed page, or enter x, y, w, h."
            bg = np.array(bg)
            m = _scribble_mask(editor, bg.shape[:2])
            if not m.any():
                return None, [], "Nothing is marked. Brush over the watermark (a rough outline is enough), or enter x, y, w, h."
            if m.shape != seed.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (seed.shape[1], seed.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            mask = m
        res = build_template(pages_dir, seed_path, seed_box=box, seed_mask=mask, name=(name or "").strip(),
                             library_dir=library_dir or None, max_pages=int(max_pages), outer_iters=int(outer_iters),
                             min_reg_score=float(min_score), frame_max_width=int(frame_w), overwrite=bool(overwrite),
                             progress=progress)
        prev = Image.fromarray(res["preview"]) if res.get("preview") is not None else None
        return prev, res.get("overlays", []), res["message"]
    except Exception as e:
        return None, [], _err(e)


def v_refresh(library_dir, cur_t, cur_r):
    ct, cr = _single_choices(library_dir), _single_choices(library_dir, with_none=True)
    vt = cur_t if cur_t in {v for _, v in ct} else None
    vr = cur_r if cur_r in {v for _, v in cr} else "None"
    return gr.update(choices=ct, value=vt), gr.update(choices=cr, value=vr)


def v_run(template, pages_dir, out_dir, reference, clean_dir, max_pages, min_score, library_dir,
          progress=gr.Progress()):
    try:
        if not template:
            return "Choose a template.", [], None, None
        if not pages_dir:
            return "Enter the pages folder.", [], None, None
        res = validate_template(template, pages_dir, out_dir or None,
                                reference=None if reference in (None, "", "None") else reference,
                                clean_dir=clean_dir or None, max_pages=int(max_pages), min_score=float(min_score),
                                progress=progress, library_dir=library_dir or None)
        ref_img = res.get("reference_image")
        ref_img = Image.open(ref_img) if ref_img and os.path.isfile(ref_img) else None
        return res["message"], res.get("gallery", []), ref_img, res.get("csv")
    except Exception as e:
        return _err(e), [], None, None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_template_tabs():
    """Adds the 'Template Stamp Fit (M5)' and 'Template Builder' tabs to the
    enclosing gr.Blocks."""
    # ---------------- Method 5 ----------------
    with gr.Tab("🧩 Template Stamp Fit (M5)"):
        gr.Markdown(
            """
            **Method 5** removes a watermark from a **template library** (no trained model). It finds the
            template on the page with a plain correlation on a faint-darkening signal, checks that the page
            really shows it, and removes it with Stamp Fit's exact inverse. Build templates in the
            **Template Builder** tab; the two AriaTender PNGs work too.
            """
        )
        with gr.Row():
            with gr.Column(scale=5):
                m5_in = gr.Image(label="1. Upload document image", type="pil")
                with gr.Row():
                    m5_tpl = gr.Dropdown(choices=_m5_choices(DEFAULT_LIB), value=[], multiselect=True,
                                         label="Templates", scale=5, allow_custom_value=True)
                    m5_refresh_btn = gr.Button("↻", scale=1, min_width=40)
                m5_lib = gr.Textbox(value=DEFAULT_LIB, label="Library folder")
                m5_score = gr.Slider(0.10, 0.80, value=0.30, step=0.01, label="Min match score")
                m5_btn = gr.Button("🧩 Clean", variant="primary", size="lg")
            with gr.Column(scale=5):
                m5_out = gr.Image(label="2. Cleaned", type="pil")
                m5_overlay = gr.Image(label="Fitted footprint", type="pil")
                m5_report = gr.Markdown(value="Pick a template and click Clean.")
        m5_refresh_btn.click(m5_refresh, [m5_lib, m5_tpl], [m5_tpl])
        m5_btn.click(m5_clean, [m5_in, m5_tpl, m5_lib, m5_score], [m5_out, m5_overlay, m5_report])

    # ---------------- Template Builder ----------------
    with gr.Tab("🛠️ Template Builder"):
        gr.Markdown(
            """
            Build a template for a watermark that a site stamps on **every page**: give it a folder of 20+
            pages from that site, mark the watermark once on one page, click **Build**. The template is saved
            into the library and can then be used by Method 5. The mark must be **darker** than the page.
            Use **Validate** afterwards to check it on any folder.
            """
        )
        with gr.Tabs():
            with gr.Tab("Build"):
                with gr.Row():
                    with gr.Column(scale=6):
                        b_dir = gr.Textbox(label="Pages folder", placeholder=r"D:\pages\site_x")
                        b_load = gr.Button("Load folder")
                        b_seed = gr.Dropdown(choices=[], label="Seed page", interactive=True, allow_custom_value=True)
                        b_editor = gr.ImageEditor(
                            label="Outline or brush over the watermark on the seed page (the bounding box of your strokes is used)",
                            type="numpy",
                            brush=gr.Brush(colors=["#ff0000"], default_size=25),
                            eraser=gr.Eraser(default_size=25),
                        )
                        with gr.Row():
                            b_x = gr.Number(value=0, label="x", precision=0)
                            b_y = gr.Number(value=0, label="y", precision=0)
                            b_w = gr.Number(value=0, label="w", precision=0)
                            b_h = gr.Number(value=0, label="h", precision=0)
                        gr.Markdown("If x, y, w, h are all set (w, h > 0) they override the brush.")
                    with gr.Column(scale=4):
                        b_name = gr.Textbox(label="Template name", placeholder="mysite_wide")
                        b_lib = gr.Textbox(value=DEFAULT_LIB, label="Library folder")
                        with gr.Accordion("Advanced", open=False):
                            b_maxp = gr.Slider(10, 200, value=60, step=1, label="Max pages")
                            b_iters = gr.Slider(1, 4, value=2, step=1, label="Outer iterations")
                            b_minscore = gr.Slider(0.10, 0.80, value=0.30, step=0.01, label="Min registration score")
                            b_framew = gr.Slider(300, 2000, value=1000, step=50, label="Frame max width (px)")
                            b_over = gr.Checkbox(value=False, label="Overwrite an existing template of this name")
                        b_btn = gr.Button("🛠️ Build template", variant="primary", size="lg")
                        b_prev = gr.Image(label="Template preview (coverage | on checkerboard)", type="pil")
                        b_gal = gr.Gallery(label="Fit overlays on accepted pages", columns=3, height="auto")
                        b_report = gr.Markdown(value="Load a folder to start.")
                b_load.click(b_load_folder, [b_dir], [b_seed, b_editor, b_report])
                b_seed.input(b_seed_changed, [b_dir, b_seed], [b_editor])
                b_btn.click(b_build, [b_dir, b_seed, b_editor, b_x, b_y, b_w, b_h, b_name, b_lib, b_maxp, b_iters,
                                      b_minscore, b_framew, b_over], [b_prev, b_gal, b_report])

            with gr.Tab("Validate"):
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            v_tpl = gr.Dropdown(choices=_single_choices(DEFAULT_LIB), value=None, label="Template", scale=5,
                                                allow_custom_value=True)
                            v_refresh_btn = gr.Button("↻", scale=1, min_width=40)
                        v_ref = gr.Dropdown(choices=_single_choices(DEFAULT_LIB, with_none=True), value="None",
                                            label="Reference template (optional)", allow_custom_value=True)
                        v_lib = gr.Textbox(value=DEFAULT_LIB, label="Library folder")
                        v_dir = gr.Textbox(label="Pages folder", placeholder=r"D:\pages\site_x")
                        v_out = gr.Textbox(label="Output folder (blank = outputs/template_validation/...)")
                        v_clean = gr.Textbox(label="Clean targets folder (optional)")
                        with gr.Row():
                            v_maxp = gr.Slider(5, 300, value=100, step=1, label="Max pages")
                            v_score = gr.Slider(0.10, 0.80, value=0.30, step=0.01, label="Min score")
                        v_btn = gr.Button("✅ Run validation", variant="primary", size="lg")
                    with gr.Column(scale=5):
                        v_report = gr.Markdown(value="Choose a template and a pages folder.")
                        v_ref_img = gr.Image(label="Reference comparison (reference red, built green, overlap yellow)", type="pil")
                        v_gal = gr.Gallery(label="(overlay, cleaned) pairs", columns=4, height="auto")
                        v_csv = gr.File(label="Per-page CSV")
                v_refresh_btn.click(v_refresh, [v_lib, v_tpl, v_ref], [v_tpl, v_ref])
                v_btn.click(v_run, [v_tpl, v_dir, v_out, v_ref, v_clean, v_maxp, v_score, v_lib],
                            [v_report, v_gal, v_ref_img, v_csv])
