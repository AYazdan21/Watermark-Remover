"""Regenerates the data tables inside model_eval/REPORT.md from review.csv and
metrics.csv, so the prose can be edited by hand without the numbers drifting.

Each table lives between `<!--NAME:start-->` and `<!--NAME:end-->` markers;
everything between a pair is replaced on every run.

Usage:
    ../.venv/Scripts/python.exe scripts/eval_models/fill_report_tables.py
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "model_eval"
REPORT = EVAL / "REPORT.md"
MODELS = ["segmentations-freeze", "segmentations-full", "detection-freeze"]
SHORT = {"segmentations-freeze": "seg-freeze", "segmentations-full": "seg-full", "detection-freeze": "det-freeze"}
TEMPLATE_DOCS = {"standard_form", "web_card"}  # the near-identical easy pages


def verdict_table(rows):
    out = ["| model | good | minor | bad |", "|---|---|---|---|"]
    for m in MODELS:
        c = Counter(r["verdict"] for r in rows if r["model"] == m)
        out.append(f"| {SHORT[m]} | {c['good']} | {c['minor']} | {c['bad']} |")
    return out


def main():
    review = list(csv.DictReader(open(EVAL / "review.csv", encoding="utf-8")))
    metrics = {(r["image"], r["model"]): r for r in csv.DictReader(open(EVAL / "metrics.csv", encoding="utf-8"))}
    hard = [r for r in review if r["doc_type"] not in TEMPLATE_DOCS]
    n_hard = len({r["image"] for r in hard})

    tables = {}
    tables["SUMMARY"] = (["**All 73 pages**", ""] + verdict_table(review)
                         + ["", f"**The {n_hard} non-template pages only** (everything except `standard_form` and `web_card`)", ""]
                         + verdict_table(hard))

    cats = sorted({c for r in review for c in r["categories"].split(";") if c})
    lines = ["| category | " + " | ".join(SHORT[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for cat in cats:
        cells = []
        for m in MODELS:
            imgs = [r["image"] for r in review if r["model"] == m and cat in r["categories"].split(";")]
            cells.append(str(len(imgs)))
        lines.append(f"| `{cat}` | " + " | ".join(cells) + " |")
    tables["CATEGORIES"] = lines

    by_doc = defaultdict(lambda: defaultdict(Counter))
    wm_of_doc = {}
    for r in review:
        by_doc[r["doc_type"]][r["model"]][r["verdict"]] += 1
        wm_of_doc.setdefault(r["doc_type"], set()).add(r["watermark_type"])
    lines = ["| doc type | pages | watermark | " + " | ".join(f"{SHORT[m]} (g/m/b)" for m in MODELS) + " |",
             "|---|---|---|" + "---|" * len(MODELS)]
    for doc in sorted(by_doc, key=lambda d: -sum(by_doc[d][MODELS[0]].values())):
        n = sum(by_doc[doc][MODELS[0]].values())
        cells = [f"{by_doc[doc][m]['good']}/{by_doc[doc][m]['minor']}/{by_doc[doc][m]['bad']}" for m in MODELS]
        lines.append(f"| `{doc}` | {n} | {', '.join(sorted(wm_of_doc[doc]))} | " + " | ".join(cells) + " |")
    tables["DOCTYPE"] = lines

    lines = ["| model | mean instances (conf>=0.25) | mean instances (0.10-0.25) | mean conf | mean coverage |", "|---|---|---|---|---|"]
    for m in MODELS:
        ms = [metrics[(img, m)] for img in {r["image"] for r in review} if (img, m) in metrics]
        f = lambda k: sum(float(x[k] or 0) for x in ms) / max(1, len(ms))
        lines.append(f"| {SHORT[m]} | {f('n_inst'):.1f} | {f('n_inst_low'):.1f} | {f('conf_mean'):.2f} | {f('coverage') * 100:.2f}% |")
    tables["METRICS"] = lines

    by_img = defaultdict(dict)
    for r in review:
        by_img[r["image"]][r["model"]] = r
    lines = ["| image | doc type | watermark | " + " | ".join(SHORT[m] for m in MODELS) + " |",
             "|---|---|---|" + "---|" * len(MODELS)]
    order = sorted(by_img, key=lambda i: (by_img[i][MODELS[0]]["doc_type"] in TEMPLATE_DOCS, i))
    for img in order:
        first = by_img[img][MODELS[0]]
        cells = []
        for m in MODELS:
            r = by_img[img][m]
            cells.append(r["verdict"] + (f" ({r['categories'].replace(';', ', ')})" if r["categories"] else ""))
        lines.append(f"| [{img}](compare/{img}.png) | {first['doc_type']} | {first['watermark_type']} | " + " | ".join(cells) + " |")
    tables["APPENDIX"] = lines

    text = REPORT.read_text(encoding="utf-8")
    for name, body in tables.items():
        pat = re.compile(rf"(<!--{name}:start-->).*?(<!--{name}:end-->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"marker {name} missing from {REPORT}")
        text = pat.sub(lambda mo: mo.group(1) + "\n" + "\n".join(body) + "\n" + mo.group(2), text)
    REPORT.write_text(text, encoding="utf-8")
    print(f"filled {len(tables)} tables in {REPORT}")


if __name__ == "__main__":
    main()
