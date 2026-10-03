# Plan v2: Template Builder + M5 quality fixes (diagnosed on ETENDER)

Branch: `template-builder` (continue on it; v1 is commits `731c0b1..5aafef7`).
Read `docs/template_builder_plan.md` (v1) first; everything there still holds unless this file
changes it. Ground rules in v1 section 0 apply unchanged, plus the extra rules in section 0 below.

## 0. Extra ground rules for v2

- The user's ETENDER pages are in `etender_template/` (31 JPEGs; the folder is **not**
  git-ignored, so take extra care: never stage it, never edit `.gitignore` for it). The user's own built template is
  `assets/stamps/library/etender/` (v1). **Never write into `assets/stamps/library/`**: build
  into a library folder in your scratchpad. Do not delete or overwrite the user's template.
- You may use `etender_template/` to **debug** (build templates, look at previews, run M5 on a
  few pages, print numbers). The user does the formal validation; do not write reports for
  them beyond your final message.
- Existing v1 templates (template.png + meta.json with `builder_version: 1`) must keep loading
  and working in M5.
- Keep every v1 option reachable: new behaviour becomes the default, the old one stays
  selectable where this plan says so.

## 1. What is wrong (measured on `etender_template/`, v1 template built by the user)

The real ETENDER mark: a **teal** "e" logo (a large, **solid** light-teal disc with a white "e",
a darker teal crescent and a page-curl corner), **light-cyan** "ETENDER" letters, a thin
teal slogan "FAST, FLEXIBLE AND EFFECTIVE", and a **dark-grey** serif "wWw.Etender.ir". It is
translucent: page text under the disc stays visible. Opacity is very consistent across pages
(v1 per-page strength 0.50-0.53 on 20 of 24 pages).

v1 results: M5 locates it correctly (score 0.63, evidence 64 vs control 16 on
`0_f0779d27f5.jpg`), but the removal is poor: median colour change along the strokes goes
**64 -> 21** grey levels (synthetic AriaTender: 69 -> 7). Visible residue: the teal disc and
letters stay as a teal ghost; the slogan is not touched at all. Causes:

1. **Wrong colour.** v1 splits the mark into at most 2 colour regions with one ink each. Region 1
   (disc + letters + www, 15.8k px) got ink rgb(90, 99, 99): grey. Removing a grey mark from a
   teal one leaves teal.
2. **Holes and streaks in the coverage.** Paper background is a morphological closing ~27 px
   wide (1.4 x a 19.6 px "stroke width"), but the disc is a ~150 px solid area. Inside it the
   "background" is the mark itself, so the darkening is ~0 there. The template preview shows a
   blotchy, streaked disc. The same closing is used by `stamp_fit.remove_stamps` at removal.
3. **Missing parts.** The slogan survives only as fragments ("IB", "ND", "ECTI") and the first
   "E" is faint: the 97th-percentile noise-floor subtraction plus the small-component pruning in
   `_refine`/`_prune` delete thin faint letters.
4. **Seven pages wrongly rejected; all seven carry the mark** (checked visually):
   - `0_17bd4d10ad` (340x522), `0_8fb366aa09` (314x452), `0_92280a6ca0` (356x516): ETENDER's mark
     does **not** scale with page width (instances are ~173, 201, 242, 256, 284, 506 px wide on
     pages 424-1330 px wide; on ~320 px pages it is ~0.5-0.8 of the width). v1's second pass
     rejects anything outside 0.8-1.25x the median *relative* width, and later rounds search only
     0.6-1.7x. `0_92280a6ca0` was matched at the right size and still rejected (M5 accepts that
     same pose). M5's own scale window (`_scales_for`: 0.6*min .. 1.6*max of `rel_width`) also
     cannot reach the mark on narrow pages.
   - `0_19bc3eb415` (0.26), `0_5c0b98c051` (0.21, table grid), `0_a1aa061f46` (0.297, mark on the
     right): correlation below 0.30 on cluttered pages. The template is blotchy and incomplete,
     and the signal ignores colour.
   - `0_2613dee841` (527x233): the mark is partly off the page; the match landed on clutter
     (strength 0.04).

## 2. Builder changes (`template_builder.py`, `template_signal.py`, `template_library.py`)

### 2.1 Mark-aware paper background (new `template_signal.paper_background_masked`)
Replace the closing wherever the mark's footprint is known (builder warps, M5 removal):
- Inputs: page, footprint mask (rendered template at the pose, `> 0.02`, dilated 3 px).
- "Paper" pixels: outside the footprint AND locally bright: `lum >= max_filter(lum, 15 px) - 25`
  (drops text, rules, dark graphics).
- `B` = normalised convolution of the paper pixels (Gaussian, sigma ~ 1.5% of max(H, W), min 6 px):
  `B = G*(I·m) / G*(m)`; where `G*(m)` is tiny (big holes such as the disc), fill from a coarser
  pyramid level, repeated until every pixel is defined. Per channel, float 0-1.
- Keep the old closing as `paper_background` (used only where the pose is not known yet, e.g.
  the first locate pass), with its kernel from the template's **largest solid thickness**
  (2 x max of the distance transform of `alpha >= 0.3`), not a stroke width, capped at 151.

### 2.2 Per-pixel ink colour (template format v2)
- After registration and warping, per frame pixel `p` and page `i` with `B_i` from 2.1, the
  observed darkening is `D_i(p) = B_i(p) - I_i(p)` (3 channels). Exclude a page at `p` when it is
  text-under-mark (lum darkening more than `max(0.12, 2.5 x the pixel's running median)`) or
  invalid. Per pixel, robust estimate over the remaining pages: per-channel median of
  `D_i / (B_i lum)`-normalised darkening, then 2 rounds of Tukey IRLS (as v1 5.6b). Result: the
  per-pixel, per-channel **matted darkening** `W(p)` at reference paper white.
- Opacity and ink from `W`: fix one template-wide ink luminance `L_ink` (section 2.4), then
  `a(p) = lum(W(p)) / (1 - L_ink)` (clip to [0, MAX_ALPHA]) and `k(p) = 1 - W(p)/a(p)` (clip
  [0,1]) where `a(p) > 0.01`. On paper this reproduces the page exactly for any `L_ink`;
  `L_ink` only matters for content under the mark.
- Template v2 files: `template.png` A = `a / a_peak` (peak 255), RGB = per-pixel ink `k(p)`
  (no longer constant per region); `meta.json` gets `builder_version: 2`, `opacity_peak:
  a_peak`, `ink_luminance: L_ink`, `ink_luminance_source: "text_crossings" | "prior"`.
  `regions.png` is still written (2.6) for the per-region Stamp Fit removal option.
- Drop the v1 2-means region split as the ink model.

### 2.3 Support: significance instead of noise floor + component pruning
Keep pixel `p` when its darkening is consistent across pages: `n_valid(p) >= max(5, 0.3 N)`,
`lum(W(p)) >= 0.012`, and `z(p) = median / (1.4826 MAD / sqrt(n_valid)) >= 4`. Then only remove
components smaller than 6 px. No percentile noise-floor subtraction. Thin faint slogan letters
are consistent across pages, so they survive. Lightly close 1-px gaps (3x3) inside the kept
support. Report how many pixels each rule removed.

### 2.4 Opacity calibration from text crossings (Dekel-style matting, optional)
Per page, take pixels inside the footprint where a text stroke crosses the mark: the page's text
ink colour `T_i` is measured as the median of strongly dark pixels (lum < 0.35) in a ring
around the footprint. At a pixel with both crossing pages (`I_text`) and paper pages
(`I_paper`) we have `I_paper - I_text = (1 - a)(B - T)`, so `a = 1 - (I_paper - I_text)/(B - T)`.
Pool these into a robust estimate of the template-wide `a_peak` (median over pixels with
`a(p)/a_peak > 0.5` and >= 2 crossing pages), and solve `L_ink` from it. Use it only if at least
300 pixels qualify and the estimate is in [0.05, 0.9]; otherwise `L_ink = stamp_fit.INK_LUM_PRIOR`.
Record which was used.

### 2.5 Registration without the page-width assumption
- Search scales in **absolute** terms (relative to the template frame), wide, in every round:
  `geomspace(0.3, 3.5, 30)`, regardless of page width (the cost is fine on a 450 px scan).
- **Remove** the relative-width consistency rejection (second pass at 0.8-1.25x).
- Acceptance per page = best candidate (top 5 by score, in order) that passes
  `stamp_fit._evidence` with Stamp Fit's thresholds AND has score >= `min_reg_score`
  (default lowered to 0.20). Evidence, not size consistency, keeps clutter out.
- Keep the weak-strength rejection (< 0.3 x median), and report it as before.
- Partial marks off the page: evidence and strength use only the in-page part (already true
  for `_evidence`); require at least 40% of the footprint inside the page.

### 2.6 Colour-matched locating signal (builder and M5)
- Page signal = 3-channel darkening `D(p) = B(p) - I(p)` (closing background, 2.1 last bullet,
  at the 450-600 px scan resolution), with text suppressed (lum darkening > 0.35 → all channels 0,
  dilated 2 px), same as v1's band rule.
- Template signal = 3-channel `W(p)` at the candidate scale (v1 templates: `alpha x (1 - ink)`
  from their RGB).
- Match with `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` on the 3-channel float32 images (OpenCV
  sums over channels). Black text darkens all channels equally; a teal mark darkens R much more
  than G/B, so text correlates less. Keep the v1 luminance signal as a fallback when the
  template is (near-)grey (max channel spread of `W` < 10% of its luminance).
- `refine_pose` scores with the same 3-channel correlation.

### 2.7 Frame and box handling
- If the kept support touches the frame edge after a round, expand the frame by 15% on that
  side and redo the warp + estimation (max 2 expansions); only warn if it still touches.
- Frame resolution: the median of accepted instance widths (not the 75th percentile), capped
  at `frame_max_width`.

### 2.8 Regions (for the per-region removal option only)
Cluster pixels with `a(p) > 0.05` by ink chroma (deterministic k-means in numpy, farthest-point
init, K chosen 1-5 by: add a cluster while it lowers the within-cluster spread by > 25% and every
cluster has >= 300 px). Write `regions.png` and per-region mean ink into meta. Not used by the
default removal.

## 3. M5 changes (`template_stamp_fit.py`)

1. **Scale window.** Scales come from the template's recorded absolute instance sizes:
   meta gains `instance_width_px` {min, max, median} (v2 builds); search
   `geomspace(0.5 x min, 2.0 x max)` in template-frame scale, union with the v1 `rel_width`
   window, so narrow pages are covered. v1 templates without the new field:
   `geomspace(0.1, 4.0)·page_w/template_w` like Stamp Fit.
2. **Locate** with the colour-matched signal (2.6) and the same candidate/evidence loop.
3. **New removal, "Per-pixel colour (v2)", default:**
   - Pose: `stamp_fit._subpixel(part, d, shape)` (works through the registered key), with `d`
     computed from the mark-aware background 2.1.
   - Per page, one scalar strength multiplier `m` (template opacity is `m x a(p)`): robust LS over
     footprint pixels that are paper under the mark (not text: same text test as 2.2), 3 channels,
     `B - I = m · a(p) · (B - k(p))`, with a soft prior `m ~ 1` (weight `0.2 x sqrt(n)`), trimmed
     3 rounds at the 75th percentile. Clip `m` to [0.3, 2.0]; if fewer than 300 usable pixels,
     `m = 1`.
   - Edge blur: use the `sigma` that `_subpixel` returns; render `a` and `k·a` at the pose with
     that blur (blur the premultiplied `a·k` and `a`, then divide).
   - Remove with the exact inverse `(I - m a k) / (1 - m a)`, only where `m a > 1e-4`; every
     other pixel stays byte-identical. Clip `m a` to `stamp_fit.MAX_ALPHA`.
   - Info per mark: `m`, `n_pixels_used`, sigma, background method, residual change.
4. **Option "Per-region (Stamp Fit)"**: the v1 path (`stamp_fit.remove_stamps`) stays
   selectable. For a v2 template it uses `regions.png` from 2.8.
5. UI (`template_ui.py`): M5 tab gets a "Removal model" radio (Per-pixel colour (v2) default /
   Per-region (Stamp Fit)). Validate gets the same radio. `run_m5.py` and
   `validate_template.py` get `--removal pixel|region`.

## 4. Builder UI/report

- Report additions: per rejected page, the evidence numbers of its best candidate; `L_ink` and
  its source; how many support pixels each rule removed; frame expansions.
- Advanced: `min registration score` default 0.20.
- Preview image: 3 panels: opacity `a` (grey), per-pixel ink on white (`a·k + (1-a)·white`, i.e.
  the mark as it appears on paper), and the support mask.

## 5. Your testing (short; the user validates)

All scripts in your scratchpad.

1. **Synthetic coloured solid mark.** Make a test mark that has the ETENDER failure modes: a
   solid teal disc ~150 px with a white letter cut out, a darker crescent, light-cyan block
   letters, a 1-2 px thin slogan line of small letters, and dark-grey serif text; translucent,
   true `a_peak = 0.3`, per-pixel ink. Composite onto 25 light pages from `wm_backgrounds_v2`
   (skip pages with mean lum < 150) at **absolute** sizes from {0.7, 1.0, 1.2, 2.0} x 250 px,
   random positions incl. 3 partly off the page, JPEG q85. Build with the seed box from page 0.
   Report: pages accepted (target >= 23/25), Pearson of `a` vs truth, mean |Δink| over
   `a > 0.1` pixels, slogan pixels kept (fraction of true slogan support), `L_ink` vs truth.
   Then M5 (pixel model) on 5 pages: median change before -> after (target after <= 8),
   byte-identical outside the footprint.
2. **Regression:** rebuild the AriaTender synthetic sets from v1 testing (the generator is in your
   scratchpad under `template_builder/gen_synth.py`; reuse it) and report the same numbers as v1
   section 10 for wide + stacked, to show nothing got worse.
3. **ETENDER debug run:** build from `etender_template/` (seed `0_0122bde289.jpg`, box
   `186,87,298,276`, the user's box) into your scratch library. Report pages accepted/rejected
   with reasons, the preview (describe it: is the disc solid, slogan present, colours teal?),
   `L_ink` source, and M5 pixel-model change before -> after on `0_f0779d27f5.jpg`,
   `0_0122bde289.jpg`, `0_92280a6ca0.jpg`, `0_8fb366aa09.jpg` (v1: 64 -> 21, 57 -> 24, 57 -> 25,
   not found). Save the before/after images in the scratchpad and list their paths.
4. `git diff --stat 5aafef7` shows only template_* modules, the three CLIs, `template_ui.py`, this
   plan file, and the library README if you touch it. `stamp_fit.py` and every M1-M4 file
   unchanged. `build_ui()` builds. Start the app once (project venv), confirm the new radio and
   a v2 build work through the UI, stop it.

## 6. Commits

Plain messages in the repo style, no attribution of any kind. Suggested:
1. `Template signal: mark-aware paper background and colour-matched matching`
2. `Template builder v2: per-pixel ink, significance support, width-free registration`
3. `Method 5: per-pixel colour removal and absolute scale window`
4. `UI and CLIs: removal model option and v2 builder report`
(This plan file goes in commit 1.)
