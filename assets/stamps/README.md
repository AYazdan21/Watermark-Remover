# assets/stamps

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
