"""Validate a template on a folder of pages (Template Builder CLI).

Run from the repo root with the project venv:
    .venv/Scripts/python.exe scripts/template_builder/validate_template.py \
        --template mysite --pages path/to/pages [--out DIR] \
        [--reference assets/stamps/ariatender_wide.png] [--clean-dir DIR] [--removal adaptive|pixel|region]

Runs Method 5 with only this template on every page, writes
<stem>_cleaned.png / _overlay.png / _diff.png plus summary.csv / summary.json
and prints the same summary the UI shows. Default output folder:
outputs/template_validation/<template>_<time>/ (never the pages folder).
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from watermark_remover.template_validate import validate_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="library name or path to a template folder / RGBA png")
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default=None, help="library name or png to compare the template against")
    ap.add_argument("--clean-dir", default=None, help="folder with <stem>_clean.png (or <stem>.png) targets")
    ap.add_argument("--max-pages", type=int, default=100)
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--library", default=None)
    ap.add_argument("--removal", choices=["adaptive", "pixel", "region"], default="adaptive",
                    help="adaptive = page-adaptive model (default); pixel = per-pixel colour model (v2); region = Stamp Fit's per-region fit")
    a = ap.parse_args()

    def progress(frac, desc=""):
        print(f"[{frac * 100:5.1f}%] {desc}", file=sys.stderr, flush=True)

    res = validate_template(a.template, a.pages, a.out, reference=a.reference, clean_dir=a.clean_dir,
                            max_pages=a.max_pages, min_score=a.min_score, progress=progress,
                            library_dir=a.library, removal=a.removal)
    print(res["message"])
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
