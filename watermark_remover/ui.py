import gradio as gr

from . import doc_core
from .doc_debug import debug_detect
from .photo_inpainter import (
    auto_detect_watermark,
    auto_detect_and_inpaint,
    inpaint_watermark,
)
from .router import DOCUMENT_LABEL, PHOTO_LABEL, auto_remove_watermark, detect_mode
from .storage import manual_save_triple

custom_css = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}
"""


def build_ui():
    with gr.Blocks(title="Watermark Remover Suite") as demo:
        gr.Markdown(
            """
            # 🦙 Watermark Remover Suite
            Upload an image on the **✨ Auto** tab and it detects whether it's a scanned
            document or a natural photo and picks the right tool for you. Prefer full manual
            control? The tools it dispatches to are still available on their own tabs:
            - **📄 1-Click Document Cleaner (No Mask)**: For scanned documents, notices, PDF pages, contracts. Removes semi-transparent watermarks with **1 click in 3ms** while preserving 100% of the underlying text.
            - **🎨 Photo Inpainter (LaMa + Brush)**: For natural photos, scenery, people, or solid opaque watermarks/objects.
            """
        )

        # ==========================================
        # TAB 0: Auto (detects document vs. photo)
        # ==========================================
        with gr.Tab("✨ Auto (Recommended)"):
            with gr.Row():
                with gr.Column(scale=5):
                    auto_input = gr.Image(label="1. Upload Image", type="pil")
                    auto_mode_radio = gr.Radio(
                        choices=[DOCUMENT_LABEL, PHOTO_LABEL],
                        value=DOCUMENT_LABEL,
                        label="Detected Mode (auto-filled — override if it looks wrong)",
                    )
                    auto_status = gr.Markdown(value="Upload an image to auto-detect its type.")
                    auto_save_chk_top = gr.Checkbox(
                        label="Auto-save triple on removal (Photo mode only)",
                        value=True,
                        info="Writes {N}_original.png, {N}_mask.png, {N}_result.png to dataset/ when Photo mode runs.",
                    )
                    btn_auto_remove = gr.Button("✨ Remove Watermark", variant="primary", size="lg")

                with gr.Column(scale=5):
                    auto_output = gr.Image(label="2. Result", type="pil")

            auto_input.change(
                fn=detect_mode,
                inputs=[auto_input],
                outputs=[auto_mode_radio, auto_status],
            )

            btn_auto_remove.click(
                fn=auto_remove_watermark,
                inputs=[auto_input, auto_mode_radio, auto_save_chk_top],
                outputs=[auto_output, auto_status],
            )

        # ==========================================
        # TAB 1: 1-Click Document Cleaner
        # ==========================================
        with gr.Tab("📄 1-Click Document Cleaner (Zero Mask)"):
            with gr.Row():
                with gr.Column(scale=5):
                    doc_input = gr.Image(label="1. Upload Document Image", type="pil")

                    method_radio = gr.Radio(
                        choices=[doc_core.METHOD_THRESHOLD, doc_core.METHOD_UNMIX, doc_core.METHOD_SEGMENT],
                        value=doc_core.METHOD_THRESHOLD,
                        label="Cleaning Method",
                        info=(
                            "M1 is the original bug-for-bug algorithm (flat background fill). "
                            "M2 is identical to M1 except it recovers the true pixel via alpha "
                            "unmixing instead of flattening it. M3 segments and deblends the "
                            "watermark (separate settings below)."
                        ),
                    )

                    smart_auto_chk = gr.Checkbox(
                        label="✨ Smart Auto-Pilot (Zero Setup - auto-configures all settings from image)",
                        value=True,
                        info="Automatically analyzes image geometry, spreadsheets, margins, paper tints, and contrast. Uncheck to manually override settings below.",
                    )

                    with gr.Group(visible=True) as m1_m2_settings_group:
                        with gr.Row():
                            bg_mode_select = gr.Radio(
                                choices=[
                                    "Dual-Zone Auto (White Margin + Cream Paper)",
                                    "Inner Paper Tint Only",
                                    "Pure White Everywhere",
                                ],
                                value="Dual-Zone Auto (White Margin + Cream Paper)",
                                label="Background Fill Mode",
                                info="Dual-Zone automatically keeps the white margin outside the border and the natural cream color inside.",
                            )
                        with gr.Row():
                            thresh_slider = gr.Slider(
                                minimum=-40,
                                maximum=40,
                                value=0,
                                step=2,
                                label="Threshold Fine-Tuning",
                                info="0 = Automatic (Otsu). Move left if faint watermark remains; move right if dark text fades.",
                            )
                        with gr.Row():
                            protect_tables_chk = gr.Checkbox(
                                label="Protect Table / Spreadsheet Gridlines",
                                value=True,
                                info="Detects and preserves thin horizontal and vertical Excel/table borders.",
                            )
                        with gr.Accordion("📐 Advanced Table & Stamp Settings", open=False):
                            with gr.Row():
                                snap_gridlines_chk = gr.Checkbox(
                                    label="Snap & Straighten Gridlines (Mathematical Grid)",
                                    value=True,
                                    info="Replaces bumpy/distorted watermark intersections with perfectly straight lines.",
                                )
                                anti_alias_chk = gr.Checkbox(
                                    label="Soft Anti-Aliasing",
                                    value=True,
                                    info="Eliminates jagged sawtooth pixel steps on low-res screenshots. (M1 only -- M2's removal region is defined by the threshold regardless of this setting.)",
                                )
                            with gr.Row():
                                thickness_select = gr.Radio(
                                    choices=["1px (Hairline)", "2px (Standard)", "Auto"],
                                    value="1px (Hairline)",
                                    label="Gridline Target Thickness",
                                )
                                stamp_filter_select = gr.Dropdown(
                                    choices=["None (Standard)", "Red Stamp Filter", "Blue Stamp Filter"],
                                    value="None (Standard)",
                                    label="Stamp Color Filter",
                                    info="Mathematically erases colored rubber stamps over black text via optical channel separation.",
                                )
                            with gr.Row():
                                grid_contrast_slider = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=50,
                                    step=5,
                                    label="Gridline Contrast / Darkness (%)",
                                    info="0% = Original faint shade; 50% = Crisp & clear (Recommended); 100% = Bold dark borders.",
                                )

                    with gr.Group(visible=False) as m3_settings_group:
                        with gr.Row():
                            seg_conf_slider = gr.Slider(
                                minimum=0.05,
                                maximum=0.9,
                                value=0.25,
                                step=0.01,
                                label="Segmentation Confidence",
                                info="Lower catches fainter/smaller watermark regions; higher is stricter. 0.25 measured clean for Finetuned (AriaTender).",
                            )
                        with gr.Row():
                            seg_model_select = gr.Dropdown(
                                choices=["Finetuned (AriaTender)", "Both (Union)", "YOLO11s", "YOLO11 General"],
                                value="Finetuned (AriaTender)",
                                label="Segmentation Model",
                                info="Finetuned (AriaTender) emits masks directly and is the recommended default; the other three are legacy box detectors refined with SAM.",
                            )
                            seg_use_sam_chk = gr.Checkbox(
                                label="Refine with SAM",
                                value=True,
                                info="Uses Mobile-SAM to refine detected boxes into precise masks before deblending. Ignored when Segmentation Model = Finetuned (AriaTender), which emits masks directly and never runs SAM.",
                            )
                        with gr.Row():
                            seg_use_template_chk = gr.Checkbox(
                                label="🧪 Experimental: Template-registered removal (recommended, revertible)",
                                value=True,
                                info=(
                                    "Registers the real, calibrated AriaTender mark to each detected instance so "
                                    "per-pixel opacity is known instead of guessed from brightness -- clears the "
                                    "watermark fully instead of leaving a faint grey ghost, without the risk of "
                                    "erasing table rules. Falls back to the older bounded correction per-instance "
                                    "when registration doesn't score well. Uncheck to revert Method 3 entirely to "
                                    "its previous behaviour (bit-identical) if this path misbehaves on your documents."
                                ),
                            )

                    save_dataset_chk = gr.Checkbox(
                        label="💾 Auto-save to dataset/ (raw original + cleaned result)",
                        value=True,
                        info="Writes {N}_original.png / {N}_cleaned.png to dataset/. Uncheck to run without touching dataset/.",
                    )
                    btn_clean_doc = gr.Button("⚡ Clean Document (1-Click)", variant="primary", size="lg")
                    doc_status = gr.Markdown(value="Upload a document and click Clean.")

                with gr.Column(scale=5):
                    doc_output = gr.Image(label="2. Cleaned Document (100% Text Preserved)", type="pil")

            def _toggle_doc_method_groups(method):
                return (
                    gr.update(visible=(method != doc_core.METHOD_SEGMENT)),
                    gr.update(visible=(method == doc_core.METHOD_SEGMENT)),
                )

            method_radio.change(
                fn=_toggle_doc_method_groups,
                inputs=[method_radio],
                outputs=[m1_m2_settings_group, m3_settings_group],
            )

            btn_clean_doc.click(
                fn=doc_core.clean_document,
                inputs=[
                    doc_input,
                    method_radio,
                    thresh_slider,
                    bg_mode_select,
                    protect_tables_chk,
                    snap_gridlines_chk,
                    anti_alias_chk,
                    thickness_select,
                    stamp_filter_select,
                    grid_contrast_slider,
                    smart_auto_chk,
                    seg_conf_slider,
                    seg_model_select,
                    seg_use_sam_chk,
                    seg_use_template_chk,
                    save_dataset_chk,
                ],
                outputs=[doc_output, doc_status],
            )

        # ==========================================
        # TAB 1b: Method 3 Detection Debugger
        # ==========================================
        with gr.Tab("🔍 M3 Detection Debug"):
            gr.Markdown(
                """
                Runs the exact same detection → SAM refinement → false-positive filter
                pipeline Method 3 uses for removal, but **removes nothing** -- it shows
                you what was found and why each candidate was kept or thrown out, so a
                "nothing was removed" result can be told apart from a "everything was
                found but rejected" result or a "SAM's mask is wrong" result, which all
                look identical from the cleaned output alone.
                """
            )
            with gr.Row():
                with gr.Column(scale=5):
                    debug_input = gr.Image(label="1. Upload Document Image", type="pil")
                    with gr.Row():
                        debug_conf_slider = gr.Slider(
                            minimum=0.05,
                            maximum=0.9,
                            value=0.25,
                            step=0.01,
                            label="Segmentation Confidence",
                            info="Same control as Method 3 -- lower catches fainter/smaller regions, at the cost of more false positives to filter.",
                        )
                    with gr.Row():
                        debug_model_select = gr.Dropdown(
                            choices=["Finetuned (AriaTender)", "Both (Union)", "YOLO11s", "YOLO11 General"],
                            value="Finetuned (AriaTender)",
                            label="Segmentation Model",
                            info="Finetuned (AriaTender) emits masks directly and is the recommended default; the other three are legacy box detectors refined with SAM.",
                        )
                        debug_use_sam_chk = gr.Checkbox(
                            label="Refine with SAM",
                            value=True,
                            info="Uncheck to see the raw YOLO boxes without mask refinement. Ignored when Segmentation Model = Finetuned (AriaTender), which emits masks directly and never runs SAM.",
                        )
                    btn_debug_detect = gr.Button("🔍 Run Detection", variant="primary", size="lg")

                with gr.Column(scale=5):
                    debug_output = gr.Image(label="2. Detections (boxes + accepted/rejected masks)", type="pil")

            debug_report = gr.Markdown(value="Upload a document and click Run Detection.")

            btn_debug_detect.click(
                fn=debug_detect,
                inputs=[debug_input, debug_conf_slider, debug_model_select, debug_use_sam_chk],
                outputs=[debug_output, debug_report],
            )

        # ==========================================
        # TAB 2: Photo Inpainter (LaMa + YOLO11)
        # ==========================================
        with gr.Tab("🎨 Photo Inpainter (LaMa + YOLO11)"):
            state_bg = gr.State(None)
            state_mask = gr.State(None)
            state_result = gr.State(None)

            with gr.Row():
                with gr.Column(scale=5):
                    editor = gr.ImageEditor(
                        label="1. Watermarked Image (Auto-detect or brush manually)",
                        type="numpy",
                        brush=gr.Brush(colors=["#ff0000"], default_size=25),
                        eraser=gr.Eraser(default_size=25),
                    )
                    with gr.Accordion("🤖 YOLO11 Auto-Detection Settings", open=True):
                        with gr.Row():
                            model_select = gr.Radio(
                                choices=[
                                    "YOLO11 General Watermarks",
                                    "YOLO11s (Sora Video Watermarks)",
                                ],
                                value="YOLO11 General Watermarks",
                                label="YOLO11 Detector Model",
                                info="YOLO11 General detects stock photo/logo watermarks; YOLO11s detects AI/Sora emblems.",
                            )
                        with gr.Row():
                            conf_slider = gr.Slider(
                                minimum=0.05,
                                maximum=0.85,
                                value=0.12,
                                step=0.01,
                                label="Detection Confidence",
                                info="Default 0.12 catches faint/semi-transparent text watermarks. Auto-falls back to 0.08 if needed.",
                            )
                            box_dilation_slider = gr.Slider(
                                minimum=0,
                                maximum=30,
                                value=10,
                                step=2,
                                label="Auto-Box Dilation (px)",
                                info="Expands detected box to ensure zero blurry watermark edges remain.",
                            )
                        with gr.Row():
                            smart_mask_chk = gr.Checkbox(
                                label="Smart Text/Logo Masking (Preserves objects under text)",
                                value=True,
                                info="Isolates high-contrast text/logo strokes inside the box so underlying objects (e.g. cars, faces) aren't erased.",
                            )

                    with gr.Row():
                        btn_auto_detect = gr.Button("🤖 Auto-Detect (Draw Mask)", variant="secondary", size="lg")
                        btn_auto_inpaint = gr.Button("⚡ 1-Click Auto Remove", variant="primary", size="lg")

                    with gr.Accordion("🖌️ Manual Brush Settings", open=False):
                        with gr.Row():
                            dilation_slider = gr.Slider(
                                minimum=0,
                                maximum=15,
                                value=4,
                                step=1,
                                label="Manual Brush Dilation (px)",
                                info="Expands brush stroke slightly to erase blurry watermark edges.",
                            )

                    with gr.Row():
                        btn_run_lama = gr.Button("✨ Inpaint Current Mask (LaMa)", variant="secondary", size="lg")
                        btn_save_triple = gr.Button("💾 Save Triple (Manual)", variant="secondary", size="lg")

                    with gr.Row():
                        auto_save_chk = gr.Checkbox(
                            label="Auto-save triple on removal",
                            value=True,
                            info="Automatically writes {N}_original.png, {N}_mask.png, {N}_result.png to dataset/",
                        )

                    status_text = gr.Markdown(value="Ready.")

                with gr.Column(scale=5):
                    output_image = gr.Image(label="2. Result (Inpainted Image)", type="pil")
                    output_mask = gr.Image(label="Generated Mask", type="numpy")

            btn_auto_detect.click(
                fn=auto_detect_watermark,
                inputs=[editor, conf_slider, box_dilation_slider, model_select, smart_mask_chk],
                outputs=[editor, output_mask, status_text],
            )

            btn_auto_inpaint.click(
                fn=auto_detect_and_inpaint,
                inputs=[editor, conf_slider, box_dilation_slider, model_select, smart_mask_chk, auto_save_chk],
                outputs=[output_image, output_mask, status_text, state_bg, state_mask, state_result],
            )

            btn_run_lama.click(
                fn=inpaint_watermark,
                inputs=[editor, dilation_slider, auto_save_chk],
                outputs=[output_image, output_mask, status_text, state_bg, state_mask, state_result],
            )

            btn_save_triple.click(
                fn=manual_save_triple,
                inputs=[state_bg, state_mask, state_result],
                outputs=[status_text],
            )
    return demo
