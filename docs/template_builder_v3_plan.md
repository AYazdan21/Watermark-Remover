# Plan v3: Method 5 locate + removal fixes (diagnosed on four real pages)

Branch: `template-builder` (continue on it; v2 is commits `8169b3d..70d68f5`).
Read `docs/template_builder_plan.md` (v1) and `docs/template_builder_v2_plan.md` (v2) first;
everything there still holds unless this file changes it. Ground rules of v1 section 0 and v2
section 0 apply unchanged.

## 0. Extra ground rules for v3

- **Never write into `assets/stamps/library/`** (the user's library). The user's `etender`
  template there is still a **v1** build and must stay untouched. Build/test templates in a
  library folder in your scratchpad. The Rebuild button (4.2) must only ever be clicked/called
  by you against a scratch library.
- `etender_template/` is not git-ignored: never stage it. Never stage `handoff.md`,
  `__tmp_nb_*.py`, `ariatender-black-clean.png`, `ariatender_subtitle_persian-black-clean.png`.
- Never call `doc_core.clean_document`, `router.*` or `document_cleaner.*` (they write into the
  user's `dataset/`). No threading near `cv2.kmeans`.
- Python: `D:\Amirreza\Fanap\Code\Watermark-Remover\.venv\Scripts\python.exe` only. Never
  install anything into `D:\Amirreza\Fanap\Code\.venv`.
- Keep every existing option reachable: new behaviour becomes the default, old behaviour stays
  selectable (removal models `pixel` and `region`, and the old stacked PNG as a template choice).
- `stamp_fit.py` and every M1-M4 file stay unchanged. `ui.py` stays unchanged.
- Commits: plain repo-style messages, **no attribution of any kind** (no Co-Authored-By, no
  mention of Claude/Anthropic/AI, no "Generated with"). Author is already configured. Do not push.

## 1. What is wrong (measured on the user's four pages)

The user ran M5 v2 on four pages and sent the results. Reproduced exactly:

| Case | Page | What happens | Root cause |
|---|---|---|---|
| ETENDER | `etender_template/0_01a2762f36.jpg` | disc + slogan stay strongly visible | the user's library `etender` is the **v1** build (missing parts, 2 flat inks). With the v2 scratch template the disc goes, a faint slogan / `wWw` ghost stays |
| AriaTender wide, grey form | `images_scraped/0_000bf78605.jpg` | thin outlines along every letter | page's mark edges differ from the template's: sharper, with a dark inner overshoot (a sharpening resampler), slightly direction-dependent. One strength `m` + Gaussian blur cannot model it |
| AriaTender wide, blue banner | `images_scraped/0_23a4bae7d5.jpg` | mark clearly visible after removal (residual change 29) | (a) the per-pixel model keeps the template's ink; the page's ink is a dark grey (~[62,60,65] at a~0.2), which on paper is indistinguishable from grey 100 at a~0.25 but over blue is not; (b) the "locally bright" paper background is contaminated by white text on the banner |
| AriaTender stacked, newspaper | `Images-wm/photo_2026-09-06_12-15-20.jpg` | nothing removed; two tiny 63 px "marks" accepted at the bottom page edge | (a) `assets/stamps/ariatender_stacked.png` is a **synthetic** composite whose subtitle sits ~41 px (at scale 1) lower than on real stamps. Stamp Fit itself uses `stamp_fit.SUB_OFFSET` (measured on real pages); the true subtitle on this page matches that offset. (b) candidates are taken in raw-NCC order, and tiny templates get high NCC by chance (0.68 for a 58 px fit at the edge vs 0.62 for the real 692 px fit with far better evidence 44.5/4.1) |

Prototypes that already show the fixes work are in
`C:\Users\AMIRRE~1\AppData\Local\Temp\claude\D--Amirreza-Fanap-Code\93133996-539d-43df-a76f-a2ed125b924f\scratchpad\v3_diag\`:
`proto_v3.py` (the removal; run with env `BG_MODE=telea`, `lam_k=0.002`), `proto_edge.py`
(edge-profile helpers it imports), `proto_param.py` (helpers). They are **reference code**:
read them, re-implement cleanly, do not import from the scratchpad. Measured with them:

- blue banner: mark essentially gone (v2 pixel: clearly visible; v2 region: outlines);
- grey form: letters gone on paper, faint shield outline only (v2: outlines everywhere);
- newspaper with the real-layout stacked template: located at 705 px, (-5, 175), score 0.65,
  mark removed down to faint traces;
- ETENDER with the v2 scratch template: disc and letters gone, slogan ghost reduced
  (median residual -4.9 -> about -2.8 grey levels on slogan pixels), `wWw` line -14.7 -> -8.8.

## 2. Locate fixes (`template_library.py`, `template_stamp_fit.py`)

### 2.1 Real-layout built-in stacked template

- `"AriaTender stacked (built-in)"` now builds its template **at load time** from Stamp Fit's own
  parts: `stamp_fit.templates()["logo"]["alpha"]` and `["sub"]["alpha"]`, the subtitle placed at
  `stamp_fit.SUB_OFFSET` (sub-pixel, `cv2.warpAffine` INTER_LINEAR), alpha = max of the two,
  ink = flat grey `round(stamp_fit.INK_LUM_PRIOR * 255)`, `opacity_peak = stamp_fit.DEFAULT_STRENGTH`,
  regions `{"mark": alpha}`. Cache it (module-level). `path`/`mtime_ns` must give a unique,
  stable `register_with_stamp_fit` key (e.g. path = `builtin:ariatender_stacked_real`, mtime =
  max of the two source files' mtimes). Reference: `stacked_composite_png` in `proto_v3.py`.
- Keep the old file selectable as `"AriaTender stacked (old PNG layout)"` (same loader as today).
- `builtin_names()` lists: wide, stacked (real layout), stacked (old PNG layout).

### 2.2 Size-aware candidate order

In `clean_document_template`, candidates are tried in order of significance, not raw NCC:
`z = score * sqrt(n_on / 1000)`, `n_on` = number of on-page pixels where the rendered coverage at
the candidate pose is >= 0.3. Keep the `min_score` and `EXTRA_SCORE_FRAC` gates on raw `score`
as today. Report `z` in each accepted/rejected record.

### 2.3 Fit-quality acceptance

After the evidence check passes, run the cheap version of the v3 fit (3.2 steps 1-10 with only
the global part, no edge profile, no polish, 2 IRLS rounds) on the candidate and compute the
explained fraction on the reliable pixels
`E = 1 - sum w |y - a*base|^2 / sum w |y|^2`. Reject with reason `fit explains only E` if
`E < MIN_EXPLAINED` (start 0.25; calibrate with the negatives in 5.3 so that the real marks of
5.2 all pass and false accepts drop). Report `E` per record.

### 2.4 Cross-template overlap

When several templates are selected (or `__all__`), two accepted marks of DIFFERENT templates
whose footprints (coverage >= 0.3) overlap by >= 50% of the smaller one describe the same mark
(e.g. wide and stacked on one AriaTender logo). Keep the one with the larger explained energy
`sum w (|y|^2 - |y - a*base|^2)` from 2.3, drop the other into `info["overlap_dropped"]` with
both numbers.

## 3. Removal v3: "Page-adaptive (v3)" (new module `template_adaptive.py`)

### 3.1 Contract

`remove_adaptive(img, marks, polish=True)` -> `(cleaned uint8, alpha float32, infos)`, the same
contract as `template_remove.remove_pixel` (marks carry `parts=[pose]` and `tpl`). Marks are
removed one after another on the running output; the returned alpha is the per-pixel max.
Pixels outside every mark's `zone` are byte-identical to the input. Works for v2 library
templates, v1 library templates and bare/built-in PNGs.

### 3.2 Per mark

1. **Pose:** as v2 (`paper_background_masked` -> luminance darkening -> `stamp_fit._subpixel`);
   keep `x, y, scale`, **ignore its `sigma`** (the edge profile replaces it).
2. **Window:** template bbox at the pose + 10 px, clipped to the page. All work is in the window.
3. **Template in the window:** `a_u` (unit coverage) and `k_t` (ink) from the premultiplied
   template, no blur (`render_win` in the prototype).
4. **Signed distance `d`** (page px, + inside) to the half-level contour of the coverage
   normalised by its local maximum: render coverage at 4x supersampling, `n = a4 / dilate(a4,
   (5*4+1)^2)`, mask `n >= 0.5 & a4 > 0.02`, inner/outer `distanceTransform` with the -0.5
   corrections, INTER_AREA back to 1x, /4 (prototype `render_ss`, `signed_distance`; same idea as
   `stamp_fit._signed_distance`).
5. `a_int = dilate(a_u, 5x5)`. **Edge direction weights** `DW` (4 channels: into-stroke right,
   left, down, up) = squared positive/negative parts of the unit gradient of
   `GaussianBlur(d, 0.7)` (prototype `dir_weights`); they sum to 1.
6. **Footprint:** `F = dilate(a_u > 0.02, ellipse 5x5) | (d > -2 & a_int > 0.02)`;
   `zone = F & (a_u > 0.02 | d > -3)`.
7. **Local background `B`:** `cv2.inpaint(window_uint8, F, 3, INPAINT_TELEA)`. Reliability
   `rel_bg`: known pixels = outside `F` and not on a strong edge (`|grad lum|/4 > 0.06`, dilated
   3x3); in a 15x15 box at least 8 known pixels, the max-channel std of the known pixels in the
   box < 0.025, and distance to the nearest known pixel <= 7 px.
8. `y = B - I`. **Text:** `dilate(lum(y) > max(0.12, 2.5 * peak * a_int), 3x3)`.
   `rel0 = zone & rel_bg & ~text`.
9. **Parts** (spatial): connected components of `dilate(a_u > 0.02, ellipse radius
   max(2, 0.012 * hypot(h, w)))`. **Ink regions:** the template's own region labels at the pose
   when it has them (v1/v2 library `regions`), else colour vs grey by ink saturation
   (`max - min > 0.08`).
10. **Model:** `a = peak * (m_p * a_u + a_int * sum_k DW_k * Q_{p,k}(d))`, `Q` piecewise linear on
    knots `d = -3, -2.5, ..., 3.5` with both ends pinned to 0 (edge-only correction), one `m` and
    four `Q` curves per part; ink `k = clip(k_t + dk_r)` per ink region.
11. **Fit:** 4 IRLS rounds. Each round: `base = B - k`, reliable = `rel0 & |base| > 0.06`;
    Tukey weights on `|y - a*base|` with `c = 4.685 * max(1.4826 * median, 0.008)`. Fit the
    global part (all pixels) with `scipy.optimize.lsq_linear` on the 3 channels stacked, rows
    weighted by `sqrt(w)`; ridge `1e-3 * sqrt(sum w)` on the Q coefficients and 3x that on their
    second differences within each direction block; bounds `m in [0.2, 3]`, `Q in [-3, 3]`.
    Then each part with >= 600 weighted pixels gets its own fit; smaller parts use the global
    one. Then the ink tint per region: `dk = sum w a (a (B - k_t) - y) / (sum w a^2 + lam_k sum w)`,
    `lam_k = 0.002` (a strong pull towards the template ink is exactly what failed on the blue
    banner), clipped to +-0.5 per channel. Subsample to <= 60k pixels per solve (seed 0).
    If fewer than 300 reliable pixels: fall back to the v2 pixel model for this mark
    (`info.reason = "too few reliable pixels"`).
12. `a_par = clip(a, 0, stamp_fit.MAX_ALPHA) * zone`.
13. **Polish** (per-pixel opacity read from the page where it can be read): with
    `dvec = B - k`, `a_d = <y, dvec> / |dvec|^2`, `perp = |y - a_d dvec|`. Use `a_d` where
    `zone & rel_bg & ~text & |dvec| > 0.08 & perp < 0.03 + 0.1 * a_par` and
    `0.75 * erode3(a_par) - 0.03 <= a_d <= 1.3 * dilate3(a_par) + 0.03`; blend
    `w = GaussianBlur(ok, 0.7) * ok`, `a = w * clip(a_d) + (1 - w) * a_par`.
14. **Inverse:** `(I - a k) / (1 - a)`, written only where `a > 1e-4`.
15. **Guard:** ghost score of this mark = RMS over 0.5 px bins of `d` in [-4, 4] of the mean
    luminance residual `lum(cleaned) - lum(B)` on `rel_bg & ~text` pixels with
    `|residual| <= 25` grey levels (same statistic as `scripts/alpha_net/eval_stamp_rim.py`).
    Compute it for the v3 result and for the v2 pixel model of the same mark (same `B`). If v3
    is worse by more than 0.3 grey levels, keep the v2 result for this mark
    (`info.reverted = true`). Report both scores.

Info per mark: `model: "adaptive"`, `m` per part, `dk` per region (0-255), reliable pixel count,
polished pixel count, ghost before/after (v3 and v2), reverted, seconds.

### 3.3 Wiring

`clean_document_template(..., removal="adaptive")` is the new default; `REMOVALS =
("adaptive", "pixel", "region")`. `_message` prints the model name and per-mark `m`, `dk`,
ghost scores, and a line for any v1 template (4.1).

## 4. UI / CLI (`template_ui.py`, CLIs)

1. **v1 notice:** when a selected template has `builder_version < 2`, M5's message says: "built
   with builder v1 (missing parts, 2 flat inks) -- rebuild it in the Template Builder tab for
   much better results".
2. **Rebuild button** in the Builder tab: dropdown of library templates whose `meta.json` has
   `source_folder`, `seed_page`, `seed_box`; "Rebuild with current builder" calls
   `build_template(pages_dir=source_folder, seed_page, seed_box, name=same, overwrite=True,
   library_dir=current library)`, showing the same report as Build. Missing folder/page ->
   clear error, nothing touched.
3. Removal radio in M5 and Validate: `[("Page-adaptive (v3)", "adaptive"), ("Per-pixel colour
   (v2)", "pixel"), ("Per-region (Stamp Fit)", "region")]`, default adaptive.
   `run_m5.py` / `validate_template.py`: `--removal adaptive|pixel|region` (default adaptive).
4. New CLI `scripts/template_builder/bench_removal.py` (section 5.1), committed.

## 5. Your testing (all scratch in `...\scratchpad\template_builder_v3\`)

### 5.1 Synthetic ground-truth benchmark (`bench_removal.py`)

Purpose: measure residual against a known clean page, including the degradations seen on real
pages. Args: `--backgrounds` (default `wm_backgrounds_v2`), `--out` (required; refuse any path
inside the repo), `--templates` (default: wide built-in, stacked built-in; any library template
name/path may be added with `--library`), `--n` per template (default 40), `--seed`.

Per sample: a light background page (mean lum >= 150), resized to width U(700, 1300); with
p = 0.35 a coloured banner (random saturated colour, horizontal gradient) under the mark area
with some white text-like bars and a rounded semi-transparent white pill. Composite the
template at width U(0.35, 0.95) x page width (15% of samples partly off the page), opacity x
U(0.7, 1.4), ink jitter +-15 levels (for grey marks also a darker ink 55-100), done at **2x**
resolution then downscaled with one of {INTER_AREA, INTER_CUBIC, INTER_LANCZOS4} (the last two
give the overshoot seen on real pages); p = 0.3 unsharp mask (amount 0.5, sigma 1); JPEG q in
{70, 85, 95} with 4:2:0. **Ground truth = the clean page through the identical pipeline.**

Run M5 with `[that template]` for each removal model. Per sample and model: located (footprint
IoU with truth >= 0.5), MAE inside the dilated true footprint vs GT, ghost score vs GT (as 3.2.15
but against GT, with the TRUE pose's `d`), pixels changed outside the footprint (must be 0).
Write a CSV and a summary (median / p90 per model x template x {paper, banner} x resampler).

Targets (report all, explain any miss):
- adaptive median ghost <= 0.6 x pixel(v2) median ghost; median MAE <= 0.8 x v2;
- banner subset: adaptive median MAE <= 0.6 x v2;
- adaptive MAE > v2 MAE + 1.0 on at most 5% of samples;
- 0 pixels changed outside the footprint, every model.

### 5.2 The user's four pages

Build an ETENDER v2 template from `etender_template/` (seed `0_0122bde289.jpg`, box
`186,87,298,276`) into your scratch library (the v2 builder is unchanged). Then run M5
(`adaptive`, `pixel`, `region`) on:
- `etender_template/0_01a2762f36.jpg` with that template (and once with the user's v1 library
  template read-only, to show the v1 notice);
- `images_scraped/0_000bf78605.jpg` and `images_scraped/0_23a4bae7d5.jpg` with the wide built-in;
- `Images-wm/photo_2026-09-06_12-15-20.jpg` with the stacked built-in, and with
  `[wide, stacked]` together (2.4 must leave exactly one mark: the stacked one).
Save original | v2 | v3 crops of each mark (2x zoom for small pages) in the scratchpad and list
the paths; report the per-mark info lines. The stacked must be found at ~705 px wide near
(-5, 175) and no tiny marks at the bottom edge.

### 5.3 Negatives and regressions

- False accepts: the ETENDER scratch template on 25 AriaTender pages from `images_scraped/`
  (pick pages with an AriaTender mark by eye or by the wide template's acceptance), and the wide
  and stacked built-ins on the 31 `etender_template/` pages. Report counts with v2 locate vs v3
  locate (2.2-2.4). v3 must not accept more false marks than v2.
- Real AriaTender pages: M5 with wide + stacked on 15 pages of `wm_realtune/images` (skip
  `75_original`, `0_355a9bf621`): marks found by v2 must still be found; report ghost scores
  (3.2.15, against the inpainted B) for pixel vs adaptive.
- Runtime: per-page M5 time for v2 vs v3 on the four pages (v3 should add <= ~4 s per mark).

### 5.4 Hygiene

`git diff --stat 70d68f5` shows only `template_*` modules, the CLIs in
`scripts/template_builder/`, and this plan file. `stamp_fit.py`, `ui.py` and every M1-M4 file
unchanged. `build_ui()` builds. Start the app once on a free port (Windows reserves 7861-7960;
use e.g. 7840), check that the radio shows the three models, the stacked choices show, the
Rebuild dropdown lists a scratch-library template and rebuilding it works; stop the app.

## 6. Commits

Plain messages, repo style, **no attribution**. Suggested:
1. `Method 5: real-layout stacked template and size-aware candidate ranking` (2.1-2.4 + this plan)
2. `Method 5: page-adaptive removal` (3)
3. `UI and CLIs: page-adaptive removal option, template rebuild, removal benchmark` (4)
