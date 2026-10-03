"""Run Method 5 (Template Stamp Fit) over a folder of pages (CLI).

Run from the repo root with the project venv:
    .venv/Scripts/python.exe scripts/template_builder/run_m5.py \
        --pages path/to/pages --out path/to/cleaned [--templates mysite other | all] [--removal adaptive|pixel|region]

Writes <stem>_cleaned.png for every page into --out and prints a summary.
Does not touch dataset/ and loads no model.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

from watermark_remover import template_library as tl  # noqa: E402
from watermark_remover.template_builder import list_pages, read_rgb  # noqa: E402
from watermark_remover.template_stamp_fit import clean_document_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--templates", nargs="+", default=["all"], help="library names / png paths, or 'all'")
    ap.add_argument("--library", default=None)
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--removal", choices=["adaptive", "pixel", "region"], default="adaptive",
                    help="adaptive = page-adaptive model (default); pixel = per-pixel colour model (v2); region = Stamp Fit's per-region fit")
    a = ap.parse_args()

    files = list_pages(a.pages)
    if not files:
        sys.exit(f"no images in {a.pages}")
    out = tl.resolve_path(a.out)
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(tl.resolve_path(a.pages))):
        sys.exit("--out must not be the pages folder")
    os.makedirs(out, exist_ok=True)
    names = ["__all__"] if (len(a.templates) == 1 and a.templates[0].lower() == "all") else a.templates
    n_acc, t0 = 0, time.time()
    for i, f in enumerate(files, 1):
        img = read_rgb(f)
        if img is None:
            print(f"[{i}/{len(files)}] {os.path.basename(f)}: unreadable, skipped")
            continue
        cleaned, alpha, info, msg = clean_document_template(img, names, a.library, min_score=a.min_score, removal=a.removal)
        stem = os.path.splitext(os.path.basename(f))[0]
        Image.fromarray(cleaned).save(os.path.join(out, f"{stem}_cleaned.png"))
        k = len(info["accepted"])
        n_acc += k > 0
        what = ", ".join(f"{m['template']} ({m['score']:.2f})" for m in info["accepted"]) or \
            ("; ".join(f"{r['template']}: {r['reason']}" for r in info["rejected"]) or "no match")
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {what}", flush=True)
    print(f"\n### Method 5 over {len(files)} page(s): {n_acc} with a mark removed, "
          f"{(time.time() - t0) / len(files):.1f} s/page. Cleaned pages in `{out}`")


if __name__ == "__main__":
    main()
