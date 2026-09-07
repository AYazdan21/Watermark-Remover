"""Verification script for the Document tab's M1/M2 methods.

Runs both methods (plus the frozen original reference implementation) over
a fixed corpus, with save_dataset=False throughout, and writes any output
images to an output directory OUTSIDE dataset/ so nothing in dataset/ is
ever touched by this script.

Hard acceptance gate for M1 ("Threshold + Flat Fill (Original)"): its
output must be byte-identical (np.array_equal) to
tests/reference/legacy_document_cleaner.clean_document_auto given the same
settings, for every image in the corpus. If even one differs, M1 is wrong.

Also sanity-checks M2 ("Threshold + Alpha Unmixing"): runs without error on
the whole corpus, and differs from M1 (reports mean absolute diff per
image). A handful of outputs are saved for eyeballing.

Usage (from the Watermark-Remover directory):
    ../.venv/Scripts/python.exe scripts/compare_methods.py
"""
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from watermark_remover import doc_core  # noqa: E402
from tests.reference import legacy_document_cleaner as legacy  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_ROOT, "scratch_compare_methods_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Redirect the frozen legacy module's save-directory global to a scratch
# folder so calling it never writes into dataset/cleaned_documents. This is
# a runtime monkeypatch of the *imported name* in legacy's own module
# namespace -- it does not modify the frozen file's content.
_LEGACY_SCRATCH_DIR = os.path.join(OUTPUT_DIR, "_legacy_scratch_saves")
os.makedirs(_LEGACY_SCRATCH_DIR, exist_ok=True)
legacy.CLEANED_DOCS_DIR = _LEGACY_SCRATCH_DIR


def build_corpus():
    paths = []
    doc_originals_dir = os.path.join(REPO_ROOT, "dataset", "document_originals")
    for name in ["69_original.png", "70_original.png", "75_original.png"]:
        p = os.path.join(doc_originals_dir, name)
        if os.path.exists(p):
            paths.append(p)
        else:
            print(f"WARNING: corpus file missing: {p}")

    for name in sorted(os.listdir(REPO_ROOT)):
        if name.startswith("test_") and name.endswith(".png"):
            paths.append(os.path.join(REPO_ROOT, name))

    return paths


# Fixed settings shared across all three calls for a given image. smart_auto
# stays True so the auto-profiler path (also part of the original pipeline)
# is exercised identically by both implementations.
COMMON_KWARGS = dict(
    sensitivity_offset=0,
    bg_mode="Dual-Zone Auto (White Margin + Cream Paper)",
    protect_tables=True,
    snap_gridlines=True,
    anti_alias=True,
    line_thickness="1px (Hairline)",
    stamp_filter="None (Standard)",
    grid_contrast=50,
    smart_auto=True,
)


def main():
    corpus = build_corpus()
    print(f"Corpus: {len(corpus)} images")

    m1_pass = 0
    m1_fail = []
    m2_ok = 0
    m2_fail = []
    m2_diffs = []

    for path in corpus:
        name = os.path.basename(path)
        img = Image.open(path)

        # --- M1 vs frozen legacy reference: hard byte-identical gate ---
        try:
            legacy_out, _ = legacy.clean_document_auto(img, **COMMON_KWARGS)
            m1_out, _ = doc_core.clean_document(
                img, method=doc_core.METHOD_THRESHOLD, save_dataset=False, **COMMON_KWARGS
            )
            legacy_arr = np.array(legacy_out)
            m1_arr = np.array(m1_out)
            identical = np.array_equal(legacy_arr, m1_arr)
            if identical:
                m1_pass += 1
            else:
                diff = np.abs(legacy_arr.astype(int) - m1_arr.astype(int))
                m1_fail.append((name, float(diff.mean()), int(diff.max())))
                print(f"[M1 MISMATCH] {name}: mean_abs_diff={diff.mean():.4f} max_abs_diff={diff.max()}")
        except Exception as e:
            m1_fail.append((name, "EXCEPTION", str(e)))
            print(f"[M1 EXCEPTION] {name}: {e}")

        # --- M2 sanity: runs cleanly, differs from M1 ---
        try:
            m2_out, m2_status = doc_core.clean_document(
                img, method=doc_core.METHOD_UNMIX, save_dataset=False, **COMMON_KWARGS
            )
            m2_arr = np.array(m2_out)
            m1_arr_for_diff = np.array(
                doc_core.clean_document(img, method=doc_core.METHOD_THRESHOLD, save_dataset=False, **COMMON_KWARGS)[0]
            )
            diff = np.abs(m2_arr.astype(int) - m1_arr_for_diff.astype(int))
            mean_diff = float(diff.mean())
            m2_diffs.append((name, mean_diff))
            m2_ok += 1
            print(f"[M2 OK] {name}: mean_abs_diff_vs_M1={mean_diff:.4f} status={m2_status[:80]!r}")

            stem = os.path.splitext(name)[0]
            Image.fromarray(m1_arr_for_diff.astype(np.uint8)).save(os.path.join(OUTPUT_DIR, f"{stem}_M1.png"))
            Image.fromarray(m2_arr.astype(np.uint8)).save(os.path.join(OUTPUT_DIR, f"{stem}_M2.png"))
        except Exception as e:
            m2_fail.append((name, str(e)))
            print(f"[M2 EXCEPTION] {name}: {e}")

    print("\n===== SUMMARY =====")
    print(f"M1 byte-identical to frozen legacy reference: {m1_pass}/{len(corpus)}")
    if m1_fail:
        print("M1 failures:")
        for entry in m1_fail:
            print(f"  {entry}")

    print(f"\nM2 ran without error: {m2_ok}/{len(corpus)}")
    if m2_fail:
        print("M2 failures:")
        for entry in m2_fail:
            print(f"  {entry}")

    zero_diff = [n for n, d in m2_diffs if d == 0.0]
    print(f"M2 images with ZERO diff from M1 (suspicious -- unmixing had no effect): {len(zero_diff)}")
    if zero_diff:
        for n in zero_diff:
            print(f"  {n}")

    print(f"\nOutputs written to: {OUTPUT_DIR}")

    # Hard gate: fail loudly (non-zero exit) if M1 isn't perfect.
    if m1_pass != len(corpus):
        print("\nFAIL: M1 is not byte-identical to the frozen legacy reference on the full corpus.")
        sys.exit(1)
    else:
        print("\nPASS: M1 is byte-identical to the frozen legacy reference on the full corpus.")


if __name__ == "__main__":
    main()
