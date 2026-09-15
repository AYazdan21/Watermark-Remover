"""Reads model_eval/review.csv (the human visual-review verdicts) and copies
each problematic (model, image) overlay into model_eval/issues/<category>/,
so failures of a given kind can be flipped through together. An item with
several categories is copied into each of them. Also writes a README.md per
category folder listing what's in it.

"Problematic" = verdict in {minor, bad}, OR verdict is "good" but categories
is non-empty (e.g. an ood_watermark row that was still judged good -- kept
so ood behaviour stays visible even when it worked).

Usage:
    ../.venv/Scripts/python.exe scripts/eval_models/build_issue_folders.py
"""
from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL_EVAL = REPO / "model_eval"
REVIEW_CSV = MODEL_EVAL / "review.csv"
ISSUES_DIR = MODEL_EVAL / "issues"

CATEGORY_BLURBS = {
    "missed_watermark": "A visible watermark with no prediction at all.",
    "partial_coverage": "Part of the mark is found but not the rest (subtitle, letters, faint areas, parts crossing text).",
    "low_conf_only": "Correct, but only at conf 0.10-0.25, so it would be lost at the default 0.25 threshold.",
    "fp_text": "Fires on real body or heading text.",
    "fp_logo_emblem": "Fires on a real organisation logo, emblem or seal (not AriaTender).",
    "fp_stamp_signature": "Fires on a stamp, signature or handwriting.",
    "fp_table_lines_ui": "Fires on table rules, borders, form boxes or UI chrome.",
    "fp_color_banner_image": "Fires on coloured banners, photos or graphics.",
    "fp_graphic_icon": "Fires on a grey non-watermark graphic, e.g. the gavel 'preview unavailable' placeholder on setad portal pages.",
    "mask_bleed": "The mask on a real watermark spills onto neighbouring content.",
    "fragmented": "One mark split into many pieces or holes.",
    "box_too_large_or_merged": "(det only) One box spans several marks or lots of background.",
    "duplicate": "Overlapping duplicate predictions.",
    "ood_watermark": "A non-AriaTender watermark (real doc); note records whether it was found.",
}


def main():
    rows = list(csv.DictReader(open(REVIEW_CSV, encoding="utf-8")))
    print(f"Loaded {len(rows)} review rows")

    if ISSUES_DIR.exists():
        shutil.rmtree(ISSUES_DIR)
    ISSUES_DIR.mkdir(parents=True)

    by_category = defaultdict(list)  # category -> list of (model, image, verdict, note)

    n_copied = 0
    for r in rows:
        verdict = r["verdict"].strip()
        cats = [c.strip() for c in r["categories"].split(";") if c.strip()]
        if verdict == "good" and not cats:
            continue  # nothing wrong to file away
        if not cats:
            # minor/bad with no explicit category -- skip, shouldn't happen but be safe
            continue
        model = r["model"]
        image = r["image"]
        src = MODEL_EVAL / model / f"{image}.png"
        if not src.exists():
            print(f"WARNING missing overlay: {src}")
            continue
        for cat in cats:
            cat_dir = ISSUES_DIR / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            dst = cat_dir / f"{model}__{image}.png"
            shutil.copy2(src, dst)
            n_copied += 1
            by_category[cat].append((model, image, verdict, r["note"]))

    for cat, items in by_category.items():
        readme = ISSUES_DIR / cat / "README.md"
        blurb = CATEGORY_BLURBS.get(cat, "(no description on file)")
        lines = [f"# {cat}", "", blurb, "", f"{len(items)} item(s):", ""]
        for model, image, verdict, note in sorted(items):
            lines.append(f"- `{model}__{image}.png` -- **{model}**, {image}, verdict={verdict}: {note}")
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Copied {n_copied} overlays into {len(by_category)} category folders under {ISSUES_DIR}")
    for cat in sorted(by_category):
        print(f"  {cat}: {len(by_category[cat])}")


if __name__ == "__main__":
    main()
