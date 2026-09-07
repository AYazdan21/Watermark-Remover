import gradio as gr

from .document_cleaner import clean_document_auto
from .photo_inpainter import (
    auto_detect_watermark,
    auto_detect_and_inpaint,
    inpaint_watermark,
)
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
            Choose the best tool based on your image type:
            - **📄 1-Click Document Cleaner (No Mask)**: For scanned documents, notices, PDF pages, contracts. Removes semi-transparent watermarks with **1 click in 3ms** while preserving 100% of the underlying text.
            - **🎨 Photo Inpainter (LaMa + Brush)**: For natural photos, scenery, people, or solid opaque watermarks/objects.
            """
        )

        # ==========================================
        # TAB 1: 1-Click Document Cleaner
        # ==========================================
        with gr.Tab("📄 1-Click Document Cleaner (Zero Mask)"):
            with gr.Row():
                with gr.Column(scale=5):
                    doc_input = gr.Image(label="1. Upload Document Image", type="pil")
                    smart_auto_chk = gr.Checkbox(
                        label="✨ Smart Auto-Pilot (Zero Setup - auto-configures all settings from image)",
                        value=True,
                        info="Automatically analyzes image geometry, spreadsheets, margins, paper tints, and contrast. Uncheck to manually override settings below.",
                    )
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
                                info="Eliminates jagged sawtooth pixel steps on low-res screenshots.",
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
                    btn_clean_doc = gr.Button("⚡ Clean Document (1-Click)", variant="primary", size="lg")
                    doc_status = gr.Markdown(value="Upload a document and click Clean.")

                with gr.Column(scale=5):
                    doc_output = gr.Image(label="2. Cleaned Document (100% Text Preserved)", type="pil")

            btn_clean_doc.click(
                fn=clean_document_auto,
                inputs=[
                    doc_input,
                    thresh_slider,
                    bg_mode_select,
                    protect_tables_chk,
                    snap_gridlines_chk,
                    anti_alias_chk,
                    thickness_select,
                    stamp_filter_select,
                    grid_contrast_slider,
                    smart_auto_chk,
                ],
                outputs=[doc_output, doc_status],
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
