# assets/stamps/library

Watermark templates built by the **Template Builder** (UI tab "Template
Builder" or `scripts/template_builder/build_template.py`). Method 5 (Template
Stamp Fit) and the Validate tool read them from here. Built templates are
local and git-ignored; only this README is tracked.

## Format

One folder per template, `assets/stamps/library/<name>/`:

| File | Content |
|---|---|
| `template.png` | RGBA uint8. **A** = coverage (0-255, peak 255). **RGB** = ink colour of the pixel's region (constant per region). |
| `regions.png` | uint8 label map, 0 = outside, 1..K = ink region. Absent = one region. |
| `meta.json` | `name`, `created`, `builder_version`, `source_folder`, `seed_page`, `seed_box` [x,y,w,h], `n_pages_used`, `n_pages_rejected`, `template_size` [w,h], `rel_width` {median,min,max} (instance width / page width), `strength` {median,min,max}, `regions` [{id, ink_rgb, pixels}], `stroke_width_px`, `params`. |
| `preview.png` | Coverage as grey on white, next to the mark on a checkerboard. |
| `build_report.json` | Per-page registration table (file, accepted, score, scale, x, y, strength, reason). |
| `overlay_XX.png` | The fitted outline drawn on up to 6 accepted pages. |

A bare RGBA PNG (for example `assets/stamps/ariatender_wide.png`) can be used
wherever a template name is accepted.

## Notes

- The mark must be **darker** than the page; rotation is fixed at 0.
- One template per variant of the mark. Pages showing a different variant are
  rejected during a build.
- To rebuild a template under the same name, tick "overwrite" (UI) or pass
  `--overwrite` (CLI).
