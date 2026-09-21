# wm_realtune -- 10 real pages labelled for fine-tuning

YOLO-seg format, single class `0` = watermark (same as `wm_dataset_v2/`).
Picked from `wm_testset/images` -- if you fine-tune on these, evaluate on the
OTHER `wm_testset` pages only.

- `images/`   the 10 pages (copies)
- `labels/`   YOLO-seg polygons, same base name (empty file = negative)
- `masks/`    binary masks, 255 = watermark (exact; labels are their polygons)
- `overlays/` review images: page | mask in red | label polygons
- `report.csv` per page: how the mask was made, mask %, polygon count, and how
  closely the label polygons reproduce the mask (IoU)
- `fits.json` fitted stamp pose per AriaTender page

## What is committed

Only derived files: `labels/`, `masks/`, `fits.json`, `report.csv`, `data.yaml`,
this README and `alpha_targets/*_alpha.png` + `manifest.json`. The page images,
`overlays/` and `alpha_targets/*_clean.png` contain the real pages themselves
and stay local, like `wm_testset/images` (see `.gitignore`). To rebuild them:
copy the 10 pages from `wm_testset/images` into `images/`, then run
`scripts/alpha_net/make_real_targets.py --locator weights/alpha_net_v2_best_final.pt`
(v2 is the locator the committed targets and `weights/alpha_net_v4_realft.pt`
were made with).

## How the masks were made (not model predictions)

**AriaTender pages (8):** the known stamp artwork was fitted to each page --
scale, position, rotation fixed at 0 -- and the stamp's own coverage
(>= 25% of peak, the same rule as the synthetic labels) drawn at that pose.
Templates: `assets/stamps/ariatender_wide.png` (shield + AriaTender.neT), or
`assets/stamps/sources/ariatender-black-clean.png` (logo) + the Persian
subtitle placed 5 px right / 147.5 px below the logo (logo-scale units,
measured on 0_552cff21fd and confirmed on the others). Each fitted part was
checked against an independent signal (page darkening vs. local background):
the fit is the optimum within +-6 px for every part except 0_e6b580e44b
(1 px, corrected) and 82_original's subtitle (only its top edge is on the
page; kept at the layout position).

**75_original (Excel "CONFIDENTIAL"):** no stamp asset, so the template is the
per-pixel median of the 13 copies in empty cells (gridlines removed first; the
median cancels cell text), placed at all 21 positions of the repeat lattice
(fit residual < 0.7 px), clipped to the sheet's cell area.

**0_355a9bf621:** no watermark -> empty label (a negative).

Polygons are traced on a 4x-upsampled mask so thin strokes are not lost;
label-vs-mask IoU is 0.98-0.995 on every page.

Note: labels use the repo's `labels.mask_to_polygons` (outer contours), so
enclosed holes -- e.g. the inside of the shield -- are filled in the polygon,
exactly as in the synthetic training labels. `masks/` keeps the holes.
