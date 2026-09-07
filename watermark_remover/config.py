import os

# Bypass system proxy for local loopback so Gradio can connect to 127.0.0.1
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1,0.0.0.0"
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(_k, None)

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Configure dataset save directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ORIGINALS_DIR = os.path.join(DATASET_DIR, "originals")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")
RESULTS_DIR = os.path.join(DATASET_DIR, "results")
CLEANED_DOCS_DIR = os.path.join(DATASET_DIR, "cleaned_documents")

for folder in [ORIGINALS_DIR, MASKS_DIR, RESULTS_DIR, CLEANED_DOCS_DIR]:
    os.makedirs(folder, exist_ok=True)
