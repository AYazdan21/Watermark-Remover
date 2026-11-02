"""Download-on-first-use for model files too large to commit.

``weights/yolo11_watermark_general.pt`` (110 MB, over GitHub's 100 MB limit)
is ``best.pt`` of the public Hugging Face model
corzent/yolo11x_watermark_detection, byte-identical (SHA-256 checked). When it
is missing, ``ensure_general_model`` downloads it from there, verifies the
hash, and only then moves it into place. If the download fails (no network,
an SSL certificate error behind a proxy, the host blocked), it raises with the
link so the file can be fetched by hand.

Note the model's own licence on Hugging Face: research and educational use.
"""

import hashlib
import os
import urllib.request

GENERAL_MODEL_URL = "https://huggingface.co/corzent/yolo11x_watermark_detection/resolve/main/best.pt"
GENERAL_MODEL_SHA256 = "6ac71b6ab8db27ec7928b5176e60a359c65e1579a5c1d58cf2f98df30cf3085e"
GENERAL_MODEL_NAME = "yolo11_watermark_general.pt"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_general_model(path):
    """Make sure the general YOLO11 watermark detector exists at ``path``,
    downloading it on first use. Returns ``path``."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    manual = (f"Download it by hand from {GENERAL_MODEL_URL} and save it as {path}.")
    print(f"Downloading {GENERAL_MODEL_NAME} (110 MB) from {GENERAL_MODEL_URL} ...")
    try:
        with urllib.request.urlopen(GENERAL_MODEL_URL, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last = 0, -1
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(100 * done / total)
                    if pct // 10 != last:
                        last = pct // 10
                        print(f"  {pct}%")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise FileNotFoundError(f"{GENERAL_MODEL_NAME} is missing and the automatic download failed "
                                f"({type(e).__name__}: {e}). {manual}") from e
    digest = _sha256(tmp)
    if digest != GENERAL_MODEL_SHA256:
        os.remove(tmp)
        raise FileNotFoundError(f"The downloaded {GENERAL_MODEL_NAME} does not match the expected file "
                                f"(SHA-256 {digest}). {manual}")
    os.replace(tmp, path)
    print(f"Saved {path}")
    return path
