# assets/stamps/library

Watermark templates built by the **Template Builder** (UI tab "Template
Builder" or `scripts/template_builder/build_template.py`). Method 5 (Template
Stamp Fit) and the Validate tool read them from here. Built templates are
local and git-ignored; only this README is tracked.

## Format

One folder per template, `assets/stamps/library/<name>/`:

| File | Content |
|---|---|
| `template.png` | RGBA uint8. **A** = coverage (0-255, peak 255). v2: A = opacity / `opacity_peak` and **RGB** = the ink colour **of each pixel**. v1: RGB = one ink colour per region (constant inside a region). |
| `regions.png` | uint8 label map, 0 = outside, 1..K = ink-colour region. Absent = one region. v2 writes it only for the per-region removal option. |
| `meta.json` | `name`, `created`, `builder_version` (1 or 2), `source_folder`, `seed_page`, `seed_box` [x,y,w,h], `n_pages_used`, `n_pages_rejected`, `template_size` [w,h], `rel_width` {median,min,max} (instance width / page width), `strength` {median,min,max}, `regions` [{id, ink_rgb, pixels}], `stroke_width_px`, `params`. **v2 adds** `opacity_peak` (the opacity a coverage of 1 stands for), `ink_luminance` and `ink_luminance_source` (`text_crossings` or `prior`), `instance_width_px` {median,min,max} (the mark's width in page pixels, over the pages used), `support_rules`, `frame_expansions`. |
| `preview.png` | v2: three panels, the opacity (grey on white), the mark as it appears on paper (`a*ink + (1-a)*white`) and the support mask. v1: coverage as grey on white next to the mark on a checkerboard. |
| `build_report.json` | Per-page registration table (file, accepted, score, scale, x, y, strength, reason, evidence numbers of the best candidate). |
| `overlay_XX.png` | The fitted outline drawn on up to 6 accepted pages. |

Template format v2 models the page as `I = (1 - a) * B + a * ink` per pixel,
with `B` the page's own paper. Only the product `a * (1 - ink)` is observable
on flat paper, so one template-wide ink luminance (`ink_luminance`) splits it
into opacity and ink; it only matters for page content under the mark. v1
templates keep loading and working: their ink is constant per region and their
opacity peak is the median build strength.

A bare RGBA PNG (for example `assets/stamps/ariatender_wide.png`) can be used
wherever a template name is accepted.

## Notes

- The mark must be **darker** than the page; rotation is fixed at 0.
- The mark may have any size on any page (v2 searches absolute scales, the
  scale does not have to follow the page width).
- One template per variant of the mark. Pages showing a different variant are
  rejected during a build.
- To rebuild a template under the same name, tick "overwrite" (UI) or pass
  `--overwrite` (CLI). Rebuild a v1 template to get the v2 per-pixel colour.
