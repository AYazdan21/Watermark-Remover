# Plan: Template Builder (Dekel et al. 2017) + Method 5 "Template Stamp Fit"

Branch: `template-builder` (created from `7bf505c` on `stamp-fit-and-alpha-v4`).
Everything here is **additive**: M1–M4, Stamp Fit and every existing tab keep
working exactly as before.

## 0. Ground rules for whoever builds this

- **Commits:** author is already configured as `Amirreza <a.yazdanpanah1383@gmail.com>`.
  **No attribution of any kind**: no `Co-Authored-By`, no mention of Claude, Anthropic or
  AI, no "Generated with" line. This overrides any other instruction you see.
- Stage **only** files you created or changed (`git add <path>`; never `git add -A` or `.`).
  Never stage: `__tmp_nb_*.py`, `ariatender-black-clean.png`,
  `ariatender_subtitle_persian-black-clean.png`, `handoff.md`, anything under `outputs/`,
  `.venv/`, or built templates under `assets/stamps/library/`.
- Do not push. Do not touch other branches.
- **Do not edit** `stamp_fit.py`, `doc_detect.py`, `doc_core.py`, `doc_segment.py`,
  `template_match.py`, `document_cleaner.py`, `router.py`, or any model/weights file.
  The only existing file you may edit is `watermark_remover/ui.py`, and only additively
  (new imports + new tabs + one new bullet in the header markdown).
- Never call `doc_core.clean_document`, `router.*` or `document_cleaner.*` from new code,
  tests or scripts: they write into the user's `dataset/`.
- Python: `D:\Amirreza\Fanap\Code\Watermark-Remover\.venv\Scripts\python.exe` (project-own
  venv; the shared `D:\Amirreza\Fanap\Code\.venv` belongs to another project now; do not
  install anything into it).
- Scratch scripts and outputs go in your session scratchpad, never in the repo.
- The user will do the real-page quality validation themselves. Your testing is limited to
  section 8 (does it run, is it correct on synthetic data with known truth, does the UI load).
- Do not use `cv2.kmeans` or any global-RNG-dependent call in new code; everything must be
  deterministic for a fixed input and page order (sort file lists).

## 1. Goal

Today a new watermark needs a new dataset, retrained models and Stamp Fit code changes.
After this work, adding a watermark that a site stamps on every page means:

1. Point the **Template Builder** at a folder of 20+ pages from that site, mark the
   watermark once on one page, click Build. It writes a template into a library folder.
2. Use **Method 5 (Template Stamp Fit)** to remove that watermark from any page. M5 needs no
   trained model: it locates the template on the page model-free, checks the page really
   shows it, and removes it with Stamp Fit's existing exact inverse.
3. Use **Validate** to check a template on any folder (AriaTender or ETENDER), optionally
   against a reference template and/or clean target images.

## 2. New files

| File | Purpose |
|---|---|
| `watermark_remover/template_library.py` | Template file format: save/load/list library templates, register them into `stamp_fit`. |
| `watermark_remover/template_signal.py` | The model-free "faint darkening" signal + paper background, shared by builder and M5. |
| `watermark_remover/template_builder.py` | The builder (registration, Dekel gradient-median + Poisson init, alternating refinement). |
| `watermark_remover/template_stamp_fit.py` | Method 5: model-free locate + evidence check + `stamp_fit.remove_stamps`. |
| `watermark_remover/template_validate.py` | Validation over a folder (+ reference template comparison, + optional clean targets). |
| `watermark_remover/template_ui.py` | Gradio callbacks + `build_template_tabs()` that `ui.py` calls. Keeps `ui.py` edits tiny. |
| `scripts/template_builder/build_template.py` | CLI for the builder. |
| `scripts/template_builder/validate_template.py` | CLI for validation. |
| `scripts/template_builder/run_m5.py` | CLI: run M5 over a folder, write cleaned pages to an output folder. |
| `assets/stamps/library/README.md` | Explains the library format (the README is committed; built templates are not, add a `.gitignore` inside `assets/stamps/library/` that ignores everything except `README.md` and `.gitignore`). |

## 3. Library format (`template_library.py`)

Default library root: `assets/stamps/library/` (every UI/CLI entry point takes a path
override). One folder per template:

```
assets/stamps/library/<name>/
  template.png   RGBA uint8. A = coverage (0-255, peak = 255). RGB = ink colour of the
                 pixel's region (constant per region).
  regions.png    uint8 label map, 0 = outside, 1..K = ink region. Optional: absent = 1 region.
  meta.json      see below
  preview.png    human-readable preview (coverage as grey on white | on checkerboard)
  build_report.json  per-page registration/strength table from the build
```

`meta.json` keys: `name`, `created` (ISO time), `builder_version` (start at 1),
`source_folder`, `seed_page`, `seed_box` [x,y,w,h], `n_pages_used`, `n_pages_rejected`,
`template_size` [w,h], `rel_width` {`median`,`min`,`max`} (instance width in page px /
page width, over accepted pages), `strength` {`median`,`min`,`max`},
`regions` [{`id`, `ink_rgb`, `pixels`}], `stroke_width_px` (template frame),
`params` (all builder parameters used).

API:
- `list_templates(library_dir=None) -> list[str]` (folder names that contain a valid
  `template.png` + `meta.json`, sorted).
- `load_template(name_or_path, library_dir=None) -> dict` with `alpha` (H,W float32 0-1),
  `regions` {region_name: (H,W) float32 coverage}, `meta`, `path`.
  Accepts a library name, a folder path, or a bare RGBA PNG path (then single region, meta
  synthesised; this lets `assets/stamps/ariatender_wide.png` / `ariatender_stacked.png` be
  used as M5 templates and as validation references).
  For bare PNGs only, apply the same speck rule `stamp_fit.templates()` applies
  (normalise by max, keep `dilate(a >= 0.5, 5x5)`) and the same saturation split
  (`sat > 0.06` → separate "colour" region) so built-ins behave like they do in Stamp Fit.
  For library templates do **not** apply the speck rule (the builder cleans its own output,
  and thin faint strokes must survive).
- `save_template(name, alpha, region_labels, ink_rgb_per_region, meta, library_dir=None)`.
  Refuse to overwrite an existing name unless `overwrite=True`.
- `register_with_stamp_fit(tpl) -> str`: inserts the template into the dict returned by
  `stamp_fit.templates()` under a unique key `"lib:<name>@<mtime_ns of template.png>"`
  (or `"png:<abs path>@<mtime_ns>"` for bare PNGs) as `{"alpha": ..., "regions": ...}` and
  returns the key. `stamp_fit.templates()` is `lru_cache(maxsize=1)` and returns a mutable
  dict, and all of Stamp Fit's helpers (`_size`, `_resized`, `render`, `_evidence`,
  `_subpixel`, `remove_stamps`) look templates up by name in that dict, so this makes them
  work for library templates **without editing `stamp_fit.py`**. The mtime in the key keeps
  `stamp_fit._resized.cache` from ever serving a stale rebuilt template. Document this in
  the module docstring as a deliberate design choice.

## 4. Shared signal (`template_signal.py`)

Both the builder's registration and M5's locator match the template against this signal.

- `paper_background(img_rgb_uint8, kernel_px) -> float32 (H,W,3) 0-1`: morphological
  closing with an elliptical kernel (`kernel_px | 1`, at least 5), same idea as
  `stamp_fit._paper_background`: the lightest thing locally is paper; text and the mark's
  strokes are erased if the kernel is wider than their strokes.
- `darkening(img, B) -> float32 (H,W)`: `max over RGB of (B_c - I_c)`, 0-1, clipped at 0.
  Max over channels (not luma) so a pink mark that mostly darkens G/B still shows.
- `band_signal(img, B, hi=0.35, text_dilate=2) -> float32 (H,W)`: the darkening, with pixels
  whose darkening exceeds `hi` (real ink: text, rules) **and a `text_dilate`-px dilation
  around them** set to 0 (kills anti-aliased text rims). This leaves the faint mark and
  little else.
- `normalise(sig)`: divide by the 90th percentile of values > 0.02 (as
  `stamp_fit._normalise_signal`), clip to [0,1].
- `ncc_match(signal, template, scales, ...)`: multi-scale `cv2.matchTemplate(...,
  TM_CCOEFF_NORMED)` on a downscaled signal (long side ≤ 600 px, as `stamp_fit._fit_one`
  does), returning the top candidates (scale, x, y, score) in full-resolution page
  coordinates, with non-maximum suppression per scale. Pad the signal (constant 0) by 50% of
  the template size so marks partly off the page can still be matched (AriaTender pages have
  marks hanging off the edge, e.g. `82_original.png`).
- `refine_pose(signal, template, pose, radius)`: full-resolution local search: integer
  (x,y) within ±radius, then scale ×{0.98..1.02 in 0.5% steps}, then quarter-pixel
  (dx,dy) ∈ {-0.5..0.5 step 0.25}; score = NCC of the rendered template against the signal
  in a window around the pose (ignore pixels outside the page). Return pose + score.

Kernel size rule (used by both callers): `k = max(7, round(1.4 * stroke_width_at_scale))`,
where stroke width comes from the template's distance transform (as
`stamp_fit._stroke_width`), capped at 61. When the scale is not known yet, use the largest
scale being searched.

## 5. Builder (`template_builder.py`)

Entry point:

```python
build_template(pages_dir, seed_page, seed_box=None, seed_mask=None, name=..., library_dir=None,
               max_pages=60, outer_iters=2, min_reg_score=0.30, frame_max_width=1000,
               progress=None) -> dict   # {"ok", "template_dir", "report", "preview", "message"}
```

`seed_box` = (x,y,w,h) in seed-page pixels, or `seed_mask` = boolean scribble mask (box =
its bounding box). Pad the box by 5% each side, clamp to the page.

### 5.1 Load
Sorted list of `*.jpg|*.jpeg|*.png|*.bmp|*.webp` in `pages_dir` (non-recursive), seed page
first, then the rest, truncated to `max_pages`. Skip unreadable files (record them). Need at
least 5 readable pages; otherwise fail with a clear message.

### 5.2 Seed template
On the seed page: `B` = `paper_background` with `k = clamp(round(0.12 * min(box_w, box_h)), 15, 61)`;
`c0` = `normalise(band_signal)` cropped to the box. The template frame is the box at seed
scale (frame scale 1.0). Add a frame margin of 20% of the box on each side (zero coverage
there initially) so the estimate can extend past a slightly-too-small box.

### 5.3 Register every page (pose = scale, x, y; rotation fixed at 0, same as Stamp Fit)
For each page: compute its band signal (kernel from the current template's stroke width at
the largest searched scale), `ncc_match` over scales
`geomspace(0.4, 2.5, 30) × (page_w / seed_page_w)` (the mark usually scales with the page),
take the top 3 candidates, `refine_pose` each, keep the best. A page is **accepted** if its
refined score ≥ `min_reg_score`; otherwise it is rejected (record score + reason; a page with
a different watermark or none lands here).

### 5.4 Choose the frame resolution and warp
Frame scale = the 75th percentile of accepted instance widths divided by the template frame
width, capped so the frame is at most `frame_max_width` px wide (detail from the larger
pages is kept, most pages are down-sampled into it). For every accepted page, warp into the
frame (`cv2.warpAffine` with the inverse pose, `INTER_AREA` when shrinking else
`INTER_LINEAR`): the RGB crop `I_i`, its paper background `B_i` (computed on the full page
first, then warped), and a validity mask (inside the page). Store crops as **uint8** and
process later steps in row chunks (~64 rows) to keep memory bounded (60 pages × 1000×700 ×3
is fine as uint8, not as float64).

### 5.5 Initial estimate: Dekel et al. 2017, stage 1
- Per page, luminance gradients `gx, gy` of `I_i` (float, Sobel or simple forward
  differences, consistent with the Poisson solver below), invalid pixels = NaN.
- `Gx, Gy` = per-pixel `nanmedian` over pages. Pixels with fewer than `max(5, 0.3·N)` valid
  samples → 0. Background gradients differ page to page and cancel in the median; the
  watermark's gradient is the same on every page and survives.
- Poisson-integrate `(Gx, Gy)` to `W` with zero Dirichlet boundary at the frame edge (the
  mark does not touch the frame edge thanks to the margin): solve `∇²W = div(G)` with a
  DST-I solver (`scipy.fft.dstn` / `idstn`). `W` ≈ (mark − paper) in luminance, negative on
  the mark.
- `c1 = normalise(max(-W, 0))`; zero values < 0.03.

### 5.6 Refinement: alternating minimisation on the image-formation model
Model per page `i`, pixel `p`, channel `c`:
`B_i − I_i = o_i · u(p) · (B_i − k_r)` where `u` = coverage shape (to be normalised to peak
1), `o_i` = that page's strength, `k_r` = ink colour of the pixel's region `r`.

1. **Regions.** Over pixels with `c1 > 0.3`, take the per-pixel median over pages of the
   unit darkening direction `(B−I)/|B−I|` (skip pixels where `|B−I|` < 0.02). Split into 2
   regions with a deterministic 2-means on that direction (init: the two most distant
   directions; no `cv2.kmeans`) **only** if the centroids differ by > 10° and the smaller
   cluster has ≥ 3% of the pixels; otherwise 1 region. Assign every pixel with coverage to
   its nearest centroid. (AriaTender wide = grey letters + pink shield → 2 regions; ETENDER
   grey → 1.)
2. **Ink per region.** Ink is not separable from strength on flat paper (see the
   `stamp_fit` module docstring, step 4), so fix the ink's luminance at
   `stamp_fit.INK_LUM_PRIOR` and take its chroma from the region's median darkening
   direction: `k_r = paper_median − t · dir_r`, with `t` chosen so `luma(k_r) = INK_LUM_PRIOR`,
   clipped to [0,1].
3. Repeat 4 times:
   a. **Per-page strength** `o_i`: robust 1-D least squares over pixels with `u > 0.5`,
      `|B−I| ≤ 0.35`, trimmed to the 75th percentile of residuals for 3 rounds (same spirit
      as `stamp_fit._fit_strength_ink`).
   b. **Per-pixel coverage** `u(p)`: weighted least squares across pages and channels,
      `u = Σ w·o_i(B−k)·(B−I) / Σ w·o_i²(B−k)²`, with weights 0 where `|B−I| > 0.35`
      (text under the mark on that page) or invalid, then 2 rounds of Tukey-biweight IRLS on
      the residuals (scale = 1.4826 × MAD per pixel, floor 0.01). Vectorised in row chunks.
   c. Normalise `u` to peak 1 (99.5th percentile of `u` over pixels with `c1 > 0.3`), clip
      [0,1].
4. Clean: zero `u < 0.03`; remove connected components (of `u > 0.1`) smaller than
   `max(20, 0.0005 × frame area)` px; zero pixels with fewer than `max(5, 0.3·N)` valid
   samples.

### 5.7 Outer loop
Repeat 5.3–5.6 `outer_iters` times using the refined coverage as the template (re-register
**all** pages, including ones rejected earlier). Stop early if the set of accepted pages and
all poses change by < 0.5 px.

### 5.8 Finish
Crop to the support bbox (+4 px). If the support touches the frame edge, add a warning to
the report ("the mark may extend past your box; draw a bigger box"). Compute meta (section
3), the preview, and the per-page report rows: `file, accepted, score, scale, x, y,
strength, reason`. Save with `save_template`. Also save overlay previews of the fitted
template outline on up to 6 accepted pages into the template folder as `overlay_XX.png`.

Target runtime: < 3 min for 60 pages on this laptop (i7-8550U). Report the elapsed time.

## 6. Method 5: Template Stamp Fit (`template_stamp_fit.py`)

```python
clean_document_template(img_rgb_uint8, template_names, library_dir=None, min_score=0.30,
                        max_marks=4) -> (cleaned uint8, alpha float32, info dict, message str)
```

- `template_names`: list of library names / PNG paths, or `["__all__"]` = every library
  template.
- Per template: `load_template` + `register_with_stamp_fit`. Scale range for the search:
  from `meta.rel_width` (`[0.6·min, 1.6·max] × page_w / template_w`); for bare PNGs with no
  meta, `geomspace(0.1, 4.0)·page_w/template_w` like Stamp Fit.
- Locate: page band signal (kernel from the template stroke width at the largest searched
  scale), `ncc_match` + `refine_pose` (top 3 candidates), best pose per template.
- Accept a fit only if **all** hold: score ≥ `min_score`; rendered width ≥
  `stamp_fit.MIN_WIDTH_PX`; and `stamp_fit._evidence(img, parts)` passes the same
  thresholds Stamp Fit uses (`MIN_CHANGE`, `MIN_CHANGE_RATIO`, `MIN_CHANGED_FRACTION`).
  Read these constants from `stamp_fit` (do not copy their values).
- Multiple marks: after accepting a fit, zero its dilated footprint in the signal and search
  again (up to `max_marks`), same pattern as `stamp_fit.locate_stamps`. Across several
  templates, process them in the given order on the same signal.
- Remove: `stamp_fit.remove_stamps(img, marks)` with
  `marks = [{"kind": "template:<name>", "parts": [{"name": key, "scale", "x", "y",
  "sigma": 0.0}], "iou": score, "change", "control", "changed_fraction"}]`. This reuses the
  sub-pixel refinement, per-region strength/ink fit, edge profile and exact inverse
  unchanged. Pixels outside the stamps' footprint stay byte-identical (Stamp Fit's
  invariant); assert it in the smoke test.
- `info`: accepted and rejected fits (template, score, pose, evidence numbers, width),
  `remove_stamps` per-mark info, timings. `message`: a short markdown summary like the M4
  Stamp Fit message.
- No dataset saving, no model loading.

## 7. Validation (`template_validate.py`): works for AriaTender and ETENDER alike

```python
validate_template(template, pages_dir, out_dir, reference=None, clean_dir=None,
                  max_pages=100, min_score=0.30, progress=None) -> dict
```

1. **Reference comparison** (when `reference` is given, e.g.
   `assets/stamps/ariatender_wide.png` for an AriaTender-built template): find the best
   scale+translation of the reference onto the built template (`ncc_match` + `refine_pose`
   on the coverage maps), then report Pearson correlation of coverage over the union
   support, soft IoU, binary IoU at 0.5, and mean |Δcoverage|. Save
   `reference_compare.png` (reference in red channel, built in green, overlap yellow).
2. **Removal over the folder**: run M5 with only this template on each page (sorted,
   truncated to `max_pages`). Write to `out_dir`: `<stem>_cleaned.png`,
   `<stem>_overlay.png` (page with the accepted footprint outlined) and `<stem>_diff.png`
   (|cleaned − page| × 4). Per page CSV row: `file, accepted, score, x, y, scale, width_px,
   change_before, change_after, control, changed_fraction, strengths, inks, changed_px,
   time_s`, where `change_before` = `stamp_fit._colour_change(page, parts)` and
   `change_after` = the same on the cleaned page at the same parts: the leftover mark
   contrast along the strokes, with no ground truth needed.
3. **Clean targets** (when `clean_dir` is given): match by stem (`<stem>_clean.png` or
   `<stem>.png`), and report mean absolute error and PSNR inside the dilated footprint and
   on the whole page, before vs after.
4. Write `summary.json` + `summary.csv` and return a markdown summary (pages accepted/
   rejected, median change_before → change_after, reference metrics, clean-target metrics,
   time per page).

`out_dir` default: `outputs/template_validation/<template>_<YYYYmmdd-HHMMSS>/`
(`outputs/` is git-ignored). Never write into the pages folder.

## 8. UI (`template_ui.py`, wired from `ui.py`)

`ui.py` change: import `build_template_tabs` and call it inside `gr.Blocks`, after the
"M4 Detection Debug" tab and before "Photo Inpainter"; add one bullet to the header markdown
for Method 5 / Template Builder. Nothing else in `ui.py` changes.

### Tab "🧩 Template Stamp Fit (M5)"
- Image upload; template dropdown (multiselect, choices = library templates +
  "All library templates" + the two built-in AriaTender PNGs), a ↻ refresh button;
  library folder textbox (default `assets/stamps/library`); min score slider (0.10–0.80,
  default 0.30); "Clean" button.
- Outputs: cleaned image, an overlay image (fitted footprint outline), markdown report.

### Tab "🛠️ Template Builder"
Two sections (sub-tabs or accordions):

**Build**
- Textbox "Pages folder" (path the user types/pastes), button "Load folder" → shows the page
  count and fills a dropdown "Seed page" (default: first file).
- Selecting the seed page loads it into a `gr.ImageEditor` (brush, same setup as the Photo
  tab). The user scribbles over / outlines the watermark. Read the brush layers the way
  `photo_inpainter.inpaint_watermark` does (alpha > 0 of the layers). Also offer optional
  numeric x, y, w, h fields: if all four are > 0 they override the scribble.
- Textboxes: template name (required, validated as a safe folder name), library folder
  (default `assets/stamps/library`). Advanced accordion: max pages (default 60), outer
  iterations (default 2), min registration score (default 0.30), frame max width (1000),
  overwrite checkbox.
- "Build template" button with `gr.Progress`. Outputs: preview image, a gallery of overlay
  images, markdown report (pages used/rejected and why, rel width, strengths, regions, time,
  warnings, and the saved folder path).

**Validate**
- Template dropdown (library + built-ins, refresh button), pages folder textbox, output
  folder textbox (blank = default under `outputs/`), reference dropdown ("None" + library +
  built-ins), optional clean-targets folder textbox, max pages, min score.
- "Run validation" button with progress. Outputs: markdown summary, gallery of up to 12
  (overlay, cleaned) pairs, reference comparison image, and the path to the CSV.

All callbacks must catch exceptions and show them in the markdown output instead of
crashing the app. Paths: accept quoted paths and both slash styles; resolve relative paths
against the repo root (`config.BASE_DIR`).

## 9. CLIs (`scripts/template_builder/`)

- `build_template.py --pages DIR --seed-page FILE --seed-box x,y,w,h --name NAME
  [--library DIR] [--max-pages 60] [--outer-iters 2] [--min-score 0.30] [--overwrite]`
- `validate_template.py --template NAME|PATH --pages DIR [--out DIR] [--reference NAME|PATH]
  [--clean-dir DIR] [--max-pages 100] [--min-score 0.30] [--library DIR]`
- `run_m5.py --pages DIR --out DIR [--templates NAME ...|all] [--library DIR]
  [--min-score 0.30]`

Each prints the same markdown summary the UI shows. They must work when run from the repo
root with the project venv (`sys.path` insert of the repo root, as other scripts do; check
`scripts/alpha_net/*.py` for the existing pattern).

## 10. Builder's own testing (keep it short; the user validates on real pages)

All test scripts live in the scratchpad, not the repo.

1. **Synthetic truth test.** Composite `assets/stamps/ariatender_wide.png` (with its real
   alpha and ink, `observed = a·ink + (1−a)·bg`) onto 25 pages from `wm_backgrounds_v2/`
   (sorted, first 25), random but seeded position and scale (scale ∝ page width ± 10%),
   opacity multiplier U(0.8, 1.2), saved as JPEG quality 90. Build a template with the seed
   box taken from the known pose on page 0. Pass if: ≥ 22/25 pages accepted, registration
   error ≤ 1 px vs the true pose (after mapping), and reference-comparison Pearson
   correlation ≥ 0.90 against `ariatender_wide.png`. Repeat with `ariatender_stacked.png`.
   Report the numbers you got, whatever they are.
2. **M5 invariant.** On 5 of those synthetic pages, run M5 with the built template: every
   pixel where the returned alpha is 0 must equal the input exactly; report the median
   `change_before → change_after`.
3. **Regression guard.** `git diff --stat 7bf505c` must show only new files plus the
   additive `ui.py` change. Import-check the app: `python -c "from watermark_remover.ui import
   build_ui; build_ui()"` succeeds (this loads LaMa; fine).
4. **UI smoke.** Start the app with the project venv and confirm the two new tabs render
   and a build on the synthetic folder works through the UI (the Gradio client or a
   browser).

Do **not** run quality evaluations on `wm_testset`, `wm_realtune` or `images_scraped2`: the
user does that.

## 11. Commits (on `template-builder`)

Suggested sequence, each with a plain descriptive message in the repo's existing style
(subject line, blank line, wrapped body explaining what and why) and **no attribution**:

1. `Template library and shared darkening signal` (template_library, template_signal,
   library README + .gitignore, this plan file).
2. `Template builder: register pages and estimate the mark (Dekel et al. 2017)`
   (template_builder + build CLI).
3. `Method 5: Template Stamp Fit using library templates` (template_stamp_fit + run_m5 CLI).
4. `Template validation for any watermark` (template_validate + validate CLI).
5. `UI: Template Builder and Method 5 tabs` (template_ui + ui.py).

## 12. Known limits (put these in the module docstrings too)

- Assumes the mark is **darker** than the page (true for AriaTender and ETENDER). A light
  mark on a dark banner is not estimated.
- Rotation fixed at 0; layout of multi-part marks is whatever the seed page shows (one
  rigid template, unlike Stamp Fit's two-part stacked AriaTender fit).
- Ink luminance is fixed by `stamp_fit.INK_LUM_PRIOR` (unobservable on flat paper); only
  matters for content under the mark.
- Pages where the site placed a different variant of the mark are rejected during the build,
  so build one template per variant (seed on a page of that variant).
- M5's locator is a plain correlation on a hand-made signal, not a trained network; the
  evidence check is what keeps false fits out. Tune `min_score` with Validate.
