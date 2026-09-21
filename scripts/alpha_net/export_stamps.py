"""Exports the two calibrated AriaTender stamps as standalone RGBA PNGs under
``assets/stamps/``.

- ``ariatender_stacked.png`` is built from the clean, anti-aliased source art
  in ``assets/stamps/sources/`` (``ariatender-black-clean.png`` = logo,
  ``ariatender_subtitle_persian-black-clean.png`` = Persian subtitle): each
  file's alpha channel is used as ink coverage, re-coloured to the calibrated
  ink (``asset_prep.INK_GRAY``) at the calibrated opacity
  (``coverage * asset_prep.BASE_ALPHA``), then stacked logo-over-subtitle with
  ``asset_prep.combine_stamp``. This replaced the earlier export of
  ``compositor.get_stamp("ariatender_stacked")``, whose shape came from crops
  lifted off a real page and carried broken strokes and speckle.
- ``ariatender_wide.png`` is still ``compositor.get_stamp("ariatender_wide")``.

Note: ``scripts/wm_dataset/compositor.py`` (the repo's own dataset generator)
still builds its stacked mark from the older crops; only these exported PNGs
changed.

Why this exists: the alpha network's synthetic training/benchmark data needs
the SAME stamp pixels the dataset generator uses (not a re-derivation), so
that a "synthetic with exact ground truth" benchmark (``scripts/alpha_net/
benchmark.py``) is testing the network against the calibrated real-world
opacity, not a differently-tuned stand-in. It is also what the Kaggle
training notebook's section 11 reads from a ``stamps/`` folder uploaded to
the Kaggle dataset (see ``assets/stamps/README.md``, written by this script).

Each output PNG is written directly from ``compositor.get_stamp(mark_id=...)``
-- an RGBA image whose RGB channels are the stamp's own per-pixel ink colour
and whose alpha channel is ``round(calibrated_alpha * 255)`` (see
``asset_prep.clean_stamp`` / ``asset_prep.extract_flat_screenshot_stamp`` for
where that calibration comes from: BASE_ALPHA ~= 0.222 for the stacked mark,
up to ~0.3125 for the wide wordmark's ink core). No further processing is
applied here -- this script's only job is to freeze that in-memory array to
disk and prove, byte-for-byte, that saving+re-reading did not change it.

Run from anywhere (paths are resolved relative to this file); requires
``scripts/wm_dataset`` importable (this script inserts it on sys.path) and
the ``_wm_extract_deliverable`` source-crop directory next to the repo (see
``compositor.DEFAULT_ASSETS_DIR``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WM_DATASET_DIR = REPO_ROOT / "scripts" / "wm_dataset"
sys.path.insert(0, str(WM_DATASET_DIR))

import asset_prep  # noqa: E402  (needs WM_DATASET_DIR on sys.path, see above)
import compositor  # noqa: E402

OUT_DIR = REPO_ROOT / "assets" / "stamps"
SOURCES_DIR = OUT_DIR / "sources"
STACKED_LOGO_SRC = SOURCES_DIR / "ariatender-black-clean.png"
STACKED_SUBTITLE_SRC = SOURCES_DIR / "ariatender_subtitle_persian-black-clean.png"


def _calibrated_from_clean_art(path: Path) -> Image.Image:
    """Clean RGBA art (any ink colour, alpha = anti-aliased coverage) ->
    calibrated stamp: RGB = INK_GRAY, A = coverage * BASE_ALPHA, trimmed.

    The source art carries faint stray alpha specks away from the strokes
    (~2% of pixels, up to 69/255 in the logo file). Left in, they'd become
    faint specks of "watermark" in every training composite, so coverage is
    kept only on the strokes (coverage >= 0.5) and a 2-px band around them
    (their anti-aliased edge); everything else is zeroed."""
    coverage = np.asarray(Image.open(path).convert("RGBA").split()[-1], dtype=np.float64) / 255.0
    near_strokes = cv2.dilate((coverage >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    coverage = np.where(near_strokes, coverage, 0.0)
    out = np.zeros(coverage.shape + (4,), dtype=np.uint8)
    out[..., :3] = asset_prep.INK_GRAY
    out[..., 3] = np.round(coverage * asset_prep.BASE_ALPHA * 255.0).astype(np.uint8)
    return asset_prep._trim(Image.fromarray(out, "RGBA"))


def build_stacked_stamp() -> Image.Image:
    logo = _calibrated_from_clean_art(STACKED_LOGO_SRC)
    subtitle = _calibrated_from_clean_art(STACKED_SUBTITLE_SRC)
    return asset_prep.combine_stamp(logo, subtitle)


# filename -> function returning a fresh RGBA PIL.Image
STAMPS = {
    "ariatender_stacked.png": build_stacked_stamp,
    "ariatender_wide.png": lambda: compositor.get_stamp(mark_id=compositor.WIDE_MARK_ID),
}

README_TEXT = """# assets/stamps

Calibrated, semi-transparent AriaTender watermark stamps, written by
`scripts/alpha_net/export_stamps.py`.

## What these are

Each PNG is an RGBA image:
- **RGB** = the stamp's per-pixel ink colour (flat mid-gray 128 for the
  stacked mark; per-pixel for the wide wordmark, whose pink shield glyph and
  grey lettering are two different colours).
- **A** = `round(calibrated_alpha * 255)`, the real measured opacity of the
  mark at `opacity_mult = 1.0` (before the per-sample `opacity_mult ~
  U(0.5, 1.5)` jitter used in training):
  - `ariatender_stacked.png`: coverage x ~0.222
    (`asset_prep.BASE_ALPHA = (254 - 226) / (254 - 128)`, calibrated against
    `dataset/fulllogo.jpg`: the mark reads 226 on 254 paper).
  - `ariatender_wide.png`: up to ~0.3125 at full ink coverage, varying
    per-pixel with coverage (`asset_prep.extract_flat_screenshot_stamp`,
    `A_REF = 35.0 / (240 - 128)`, calibrated against
    `ariatender_wide_wordmark.png`).

Compositing one onto a background with `observed = a*ink + (1-a)*background`
reproduces the mark at its real-world strength.

## How they were made

```
python scripts/alpha_net/export_stamps.py
```

- `ariatender_stacked.png`: built from the clean, anti-aliased source art in
  `sources/` -- `ariatender-black-clean.png` (logo) stacked above
  `ariatender_subtitle_persian-black-clean.png` (Persian subtitle) with
  `asset_prep.combine_stamp`. Each file's alpha channel is used as ink
  coverage (stray faint specks away from the strokes are removed); the
  source colour is replaced by the calibrated ink and opacity above. (Earlier versions used the compositor's stacked mark, whose shape
  came from crops lifted off a real page and had broken strokes and speckle.)
- `ariatender_wide.png`: `compositor.get_stamp(mark_id="ariatender_wide")`
  (source crop in `../_wm_extract_deliverable`, next to the repo).

The script re-reads each PNG and asserts it is byte-identical to a fresh
build. Note: `scripts/wm_dataset/compositor.py` (the repo's own dataset
generator) still builds its stacked mark from the older crops.

## Where else these are used

- `scripts/alpha_net/benchmark.py`'s synthetic-with-ground-truth path
  composites these stamps onto held-out `wm_backgrounds_v2/` pages to
  measure the alpha network against known alpha/clean targets.
- The Kaggle training notebook (`train_watermark_seg_kaggle.ipynb`, section
  11) reads these two PNGs from a `stamps/` folder uploaded to the Kaggle
  dataset, so Kaggle-side code can composite the same real, calibrated marks
  without needing `_wm_extract_deliverable` (which is not uploaded).
"""


def export():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for filename, build in STAMPS.items():
        stamp = build()  # fresh RGBA PIL.Image
        out_path = OUT_DIR / filename
        stamp.save(out_path)

        # Byte-level round trip: the array we just saved, the array re-read
        # off disk, and a brand-new build (in case anything cached was
        # mutated) must all agree exactly.
        arr_saved = np.array(stamp)
        arr_reread = np.array(Image.open(out_path).convert("RGBA"))
        arr_fresh = np.array(build())
        if not np.array_equal(arr_saved, arr_fresh):
            raise AssertionError(f"{filename}: saved array != fresh build")
        if not np.array_equal(arr_reread, arr_fresh):
            raise AssertionError(f"{filename}: re-read PNG != fresh build")
        if arr_reread.dtype != np.uint8 or arr_reread.shape[-1] != 4:
            raise AssertionError(f"{filename}: unexpected re-read dtype/shape {arr_reread.dtype} {arr_reread.shape}")

        alpha = arr_fresh[:, :, 3].astype(np.float64) / 255.0
        ink_px = alpha > 0.05
        mean_ink_rgb = (
            arr_fresh[:, :, :3][ink_px].mean(axis=0).tolist() if ink_px.any() else [0.0, 0.0, 0.0]
        )
        stats[filename] = {
            "size_wh": stamp.size,
            "file_bytes": out_path.stat().st_size,
            "max_alpha": float(alpha.max()),
            "n_pixels_alpha_gt_0.05": int(ink_px.sum()),
            "mean_ink_rgb_where_alpha_gt_0.05": [round(v, 1) for v in mean_ink_rgb],
        }

    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")
    stats["README.md"] = {"file_bytes": readme_path.stat().st_size}
    return stats


if __name__ == "__main__":
    result = export()
    print(f"Wrote stamps + README to {OUT_DIR}")
    for name, s in result.items():
        print(f"  {name}: {s}")
