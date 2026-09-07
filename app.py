from watermark_remover import config  # noqa: F401 - sets up env vars, encoding, dataset dirs
from watermark_remover.ui import build_ui, custom_css

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(inbrowser=True, css=custom_css)
