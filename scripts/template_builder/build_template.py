"""Build a watermark template from a folder of pages (Template Builder CLI).

Run from the repo root with the project venv:
    .venv/Scripts/python.exe scripts/template_builder/build_template.py \
        --pages path/to/pages --seed-page 0001.jpg --seed-box 800,450,560,130 --name mysite

--seed-box x,y,w,h is a box around the watermark on the seed page, in that
page's pixels. The template is written to assets/stamps/library/<name>/
(or --library DIR). See docs/template_builder_plan.md.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from watermark_remover.template_builder import build_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", required=True, help="folder with 20+ pages of the site")
    ap.add_argument("--seed-page", required=True, help="file name inside --pages (or a full path) of the page you boxed")
    ap.add_argument("--seed-box", required=True, help="x,y,w,h around the watermark on the seed page (pixels)")
    ap.add_argument("--name", required=True, help="template name (letters, digits, _ - .)")
    ap.add_argument("--library", default=None, help="library folder (default assets/stamps/library)")
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--outer-iters", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.30, help="minimum registration score")
    ap.add_argument("--frame-max-width", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    try:
        box = tuple(float(v) for v in a.seed_box.replace(" ", "").split(","))
        assert len(box) == 4
    except Exception:
        ap.error("--seed-box must be x,y,w,h")

    def progress(frac, desc=""):
        print(f"[{frac * 100:5.1f}%] {desc}", file=sys.stderr, flush=True)

    res = build_template(a.pages, a.seed_page, seed_box=box, name=a.name, library_dir=a.library,
                         max_pages=a.max_pages, outer_iters=a.outer_iters, min_reg_score=a.min_score,
                         frame_max_width=a.frame_max_width, overwrite=a.overwrite, progress=progress)
    print(res["message"])
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
