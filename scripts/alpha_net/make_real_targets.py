"""Builds alpha-net training targets for real pages with Stamp Fit.

For every page in ``--images`` (default ``wm_realtune/images``):
  - an AriaTender stamp is located and removed exactly
    (``watermark_remover/stamp_fit.py``); the fitted per-pixel opacity is
    written as ``{stem}_alpha.png`` (uint16, opacity * 65535) and the
    recovered page as ``{stem}_clean.png``;
  - a page listed in ``--negatives`` becomes a negative: opacity 0 everywhere,
    clean = the page itself;
  - any other page where no stamp is accepted is skipped (no opacity ground
    truth), and so is every page in ``--skip``.
A ``manifest.json`` records what happened to each page.

These targets feed ``scripts/alpha_net/finetune_real.py``. They inherit Stamp
Fit's limits (a faint rim can remain along letter edges on some pages).

Run from the repo root in the app's venv:
    ../.venv/Scripts/python.exe scripts/alpha_net/make_real_targets.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from watermark_remover import alpha_net, doc_detect, stamp_fit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(REPO / "wm_realtune" / "images"))
    ap.add_argument("--out", default=str(REPO / "wm_realtune" / "alpha_targets"))
    ap.add_argument("--negatives", nargs="*", default=["0_355a9bf621"], help="stems of pages with no watermark")
    ap.add_argument("--skip", nargs="*", default=["75_original"], help="stems to leave out (e.g. non-AriaTender marks)")
    ap.add_argument("--locator", default=None,
                    help="alpha-net checkpoint used to locate stamps (default: the app's Stamp Fit locator). "
                         "The targets alpha_net_v4_realft.pt was trained on were made with "
                         "weights/alpha_net_v2_best_final.pt -- pass that to reproduce them exactly.")
    args = ap.parse_args()

    device = "cuda" if alpha_net.torch.cuda.is_available() else "cpu"
    locator = alpha_net.load_model(args.locator or doc_detect.stamp_fit_locator_weights(), device=device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for p in sorted(Path(args.images).iterdir()):
        if p.stem in args.skip:
            manifest[p.name] = "skipped (--skip)"
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if p.stem in args.negatives:
            clean, A, desc = img, np.zeros(img.shape[:2], np.float32), "negative (alpha=0 everywhere)"
        else:
            a, _ink, _rec = alpha_net.predict_image(locator, img, device=device)
            accepted, _rejected = stamp_fit.locate_stamps(img, a)
            if not accepted:
                manifest[p.name] = "skipped: no stamp accepted"
                continue
            clean, A, info = stamp_fit.remove_stamps(img, accepted)
            desc = f"{len(accepted)} stamp(s): " + ", ".join(i["kind"] for i in info)
        Image.fromarray(np.round(np.clip(A, 0, 1) * 65535).astype(np.uint16)).save(out / f"{p.stem}_alpha.png")
        Image.fromarray(clean).save(out / f"{p.stem}_clean.png")
        manifest[p.name] = desc
        print(f"{p.name}: {desc}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
