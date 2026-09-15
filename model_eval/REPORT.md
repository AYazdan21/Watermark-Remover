# Watermark model evaluation on real documents

Three checkpoints, run as raw model output (no `segmenter.py` false-positive filter) at
`imgsz=1024` on the 73 real, unlabelled pages in `wm_testset/images/`:

| model | task | training | outputs |
|---|---|---|---|
| `yolo11-seg-freeze-new-dataset.pt` | segmentation | `freeze=11` | [`segmentations-freeze/`](segmentations-freeze/) |
| `yolo11-seg-full-new-dataset.pt` | segmentation | full finetune | [`segmentations-full/`](segmentations-full/) |
| `yolo11s-det-freeze-new-dataset.pt` | detection | `freeze=11` | [`detection-freeze/`](detection-freeze/) |

> **This is the second, corrected version of this report.** The first pass reviewed downscaled
> colour panels, where a large, faint grey watermark is nearly invisible. It got the two most
> informative pages wrong. It called the real AriaTender mark on `70_original` a
> "hallucination over headline text", and it called the cropped AriaTender mark on `82_original`
> a "thin-ring non-AriaTender watermark". Both patterns it built on those calls are withdrawn
> (see [§4.8](#48-claims-from-the-first-pass-that-do-not-hold)). Every page was re-reviewed on
> contrast-enhanced panels. The first-pass verdicts are kept for reference in
> `review_first_pass_superseded.csv`.

## TL;DR

- **The test set is dominated by one easy template.** 58 of 73 pages are the same
  "جزئیات اطلاعات نیاز" tender form, with the wide AriaTender mark at the same scale and
  position. 3 more are a near-identical web card. All three models find the mark on these. The
  headline "good" counts mostly measure that one template, so the real signal is in the other
  12 pages.
- **On those 12 hard pages every model fails more often than it succeeds.** Two failure modes
  cover most of it:
  1. **A large, faint, stacked mark** (grey "AriaTender" plus the Persian subtitle, often cropped
     by the page and lying over dense text). There are 4 such pages. Usually only the shield
     outline is found, and the subtitle line is found on 0 of 12 model-page pairs.
  2. **The grey gavel "preview unavailable" placeholder** on setad portal pages. There are 5 such
     pages, and the models mask or box it in 14 of 15 model-page pairs. It shares the gavel motif
     of the AriaTender shield.
- **Each model has its own false-positive habit.**
  - **seg-full** marks the form's radio buttons on all 58 template pages. The blobs are small
    (3–5 per page), but they appear on every page.
  - **seg-freeze** covered a bold, colourful text-only page with 58 masks.
  - **det-freeze** draws huge boxes over body text whenever the mark is large and faint.
- **No model wins overall.** On the four large-mark pages the best model changes from page to
  page (seg-full on `70_original`, seg-freeze on `82_original`, and none on the other two).
- **Top fixes:**
  1. Build a small *labelled and diverse* real test set. The current one can't tell the models
     apart.
  2. Add real-page hard negatives: the gavel placeholder, form widgets, bold display text and org
     logos.
  3. Generate the stacked mark at page-spanning scale, faint and cropped, over dense text,
     subtitle included.

## 1. Method

**Inference** (`scripts/eval_models/run_models.py`): each checkpoint runs at `conf=0.10`. For each
image and model it writes an overlay (solid red at conf ≥ 0.25, dashed yellow for 0.10–0.25), a
union mask at conf ≥ 0.25 (seg models only) and a JSON file with every instance. It also writes
`compare/<image>.png`, a 2×2 colour panel.

**Metrics** (`scripts/eval_models/metrics.py`): instance counts, confidence, coverage,
dark-ink fraction, saturation, fragmentation and cross-model agreement. These go to
`metrics.csv`, with the flags in `auto_flags.csv`.

**Review panels** (`scripts/eval_models/make_review_panels.py`, new in this pass):
`review_enhanced/<image>.png`. The base image is divided by a local-background estimate and
contrast-stretched, so a mark only a few grey levels darker than its paper shows as clearly as
text. Masks are tinted and outlined, and det boxes are drawn, all at conf ≥ 0.25. On coloured
banners the enhancement turns the banner black, so those pages were also zoomed with the original
colour image beside the overlays.

**How verdicts were assigned** (`review.csv`, columns `verdict`, `categories`, `watermark_type`,
`doc_type` and `note`):
- **Viewed by eye (24 pages).** The 12 non-template pages were all viewed, with zoomed crops of
  the mark region wherever it was small. So were 10 standard-form pages and 2 web cards.
- **Assigned from metrics (49 pages).** These are the other 48 standard-form pages and 1 web card.
  A page only got the template verdict if its metrics matched the pages viewed by eye:
  - seg-freeze vs seg-full union-mask IoU ≥ 0.85
  - seg-freeze has no mask area outside seg-full's
  - seg-full has only small extra blobs (≤ 0.2% of the page)
  - det boxes cover ≥ 90% of the seg masks, with ≤ 1% of the page boxed away from them

  All 49 passed. Their `note` column says they were not individually viewed.
- **Verdict rules:**
  - `bad` means most of the mark was missed, or a false positive covers an area comparable to a
    mark component.
  - `minor` means small misses (e.g. the subtitle only) or small false-positive blobs.
  - `good` means the whole mark is covered with no meaningful false positive.
- **Det scoring changed from the first pass.** Det splitting one mark into 8–17 per-word boxes no
  longer lowers its verdict by itself. When the boxes cover the mark, it is fit for removal. This
  is the main reason det's counts differ from the first pass.

**No ground truth exists** (`wm_testset/` has no `labels/`), so every verdict is visual judgement.

## 2. Results

<!--SUMMARY:start-->
**All 73 pages**

| model | good | minor | bad |
|---|---|---|---|
| seg-freeze | 62 | 3 | 8 |
| seg-full | 5 | 60 | 8 |
| det-freeze | 61 | 3 | 9 |

**The 12 non-template pages only** (everything except `standard_form` and `web_card`)

| model | good | minor | bad |
|---|---|---|---|
| seg-freeze | 1 | 3 | 8 |
| seg-full | 2 | 2 | 8 |
| det-freeze | 0 | 3 | 9 |
<!--SUMMARY:end-->

Category counts (number of pages per model; a page can be in several categories):

<!--CATEGORIES:start-->
| category | seg-freeze | seg-full | det-freeze |
|---|---|---|---|
| `box_too_large_or_merged` | 0 | 0 | 3 |
| `fp_color_banner_image` | 1 | 0 | 0 |
| `fp_graphic_icon` | 4 | 5 | 5 |
| `fp_logo_emblem` | 0 | 0 | 1 |
| `fp_table_lines_ui` | 1 | 61 | 2 |
| `fp_text` | 1 | 1 | 3 |
| `mask_bleed` | 1 | 0 | 0 |
| `missed_watermark` | 0 | 1 | 0 |
| `ood_watermark` | 1 | 1 | 1 |
| `partial_coverage` | 5 | 6 | 4 |
<!--CATEGORIES:end-->

By document type:

<!--DOCTYPE:start-->
| doc type | pages | watermark | seg-freeze (g/m/b) | seg-full (g/m/b) | det-freeze (g/m/b) |
|---|---|---|---|---|---|
| `standard_form` | 58 | wide | 58/0/0 | 0/58/0 | 58/0/0 |
| `portal_long_page` | 5 | wide | 1/0/4 | 0/0/5 | 0/0/5 |
| `web_card` | 3 | wide | 3/0/0 | 3/0/0 | 3/0/0 |
| `newspaper_notice` | 2 | stacked_large_faint | 0/0/2 | 0/0/2 | 0/0/2 |
| `web_card_colored` | 1 | wide | 0/1/0 | 1/0/0 | 0/1/0 |
| `text_image_no_mark` | 1 | none | 0/0/1 | 1/0/0 | 0/1/0 |
| `notice_colored_frame` | 1 | stacked_large_faint | 0/0/1 | 0/1/0 | 0/0/1 |
| `excel_screenshot` | 1 | ood_confidential_tiled | 0/1/0 | 0/1/0 | 0/1/0 |
| `notice_bordered` | 1 | stacked_large_faint | 0/1/0 | 0/0/1 | 0/0/1 |
<!--DOCTYPE:end-->

Raw-output statistics (all 73 pages; det coverage is box area, so it isn't comparable with seg):

<!--METRICS:start-->
| model | mean instances (conf>=0.25) | mean instances (0.10-0.25) | mean conf | mean coverage |
|---|---|---|---|---|
| seg-freeze | 14.2 | 7.5 | 0.70 | 5.53% |
| seg-full | 18.5 | 6.3 | 0.71 | 5.18% |
| det-freeze | 13.3 | 4.6 | 0.69 | 10.37% |
<!--METRICS:end-->

Issue folders: [`issues/<category>/`](issues/). Each holds the colour overlays with the
`<model>__<image>.png` naming and a README listing each item's verdict and note. The
`fp_table_lines_ui` folder is large because it holds seg-full's radio-button blobs on all 58
template pages.

## 3. Head-to-head

**seg-freeze**
- **Wins:**
  - The cleanest output on the template pages, with no widget blobs.
  - The only model to cover the whole cropped mark on
    [`82_original`](review_enhanced/82_original.png), although its masks are coarse blobs that
    also cover text between strokes and part of the TCI logo.
  - The only model to ignore the gavel placeholder on one portal page
    ([`0_f57fd0e5d7`](review_enhanced/0_f57fd0e5d7.png)).
- **Loses:**
  - Produced the worst false positive in the set: 58 masks over the bold blue display text of
    [`0_355a9bf621`](review_enhanced/0_355a9bf621.png), a page with no watermark (checked on the
    enhanced panel).
  - Masks the placeholder on 4 of 5 portal pages, once adding a strip down the page margin
    ([`0_9f374fecce`](review_enhanced/0_9f374fecce.png)).
  - Finds only fragments of the large marks on `70_original`, `0_552cff21fd` and `0_e6b580e44b`.

**seg-full**
- **Wins:**
  - The best trace of the huge faint wordmark on [`70_original`](review_enhanced/70_original.png),
    at conf 0.93.
  - Correctly empty on `0_355a9bf621`.
- **Loses:**
  - Marks 3–5 radio buttons on every template page.
  - Masks the placeholder on 5 of 5 portal pages, and on 2 of them also drops most of the
    wordmark's letters ([`0_a63948e17f`](review_enhanced/0_a63948e17f.png), `0_f57fd0e5d7`).
  - Finds the shield only on `82_original`, `0_e6b580e44b` and `0_552cff21fd`.
  - Marks a red headline word on `70_original`.

**det-freeze**
- **Wins:** its per-word boxes cover the mark on every template page, and it finds most
  "CONFIDENTIAL" repeats on the Excel screenshot.
- **Loses:**
  - Boxes the placeholder on 5 of 5 portal pages.
  - When the mark is large and faint, the boxes grow to cover whole text blocks and still miss
    letters (`70_original`, `0_552cff21fd`, `82_original`).
  - Boxes the SAIPAGAM logo on [`0_e6b580e44b`](review_enhanced/0_e6b580e44b.png).
  - Box-based removal on those pages would damage body text.

On the hard pages the models fail on *different* pixels. That suggests combining their outputs,
but a plain union would also combine their false positives (widgets from full, text from freeze,
placeholders from all three). See §5.4.

## 4. Patterns

### 4.1 The test set is one template repeated, so it can't rank the models

58 pages are the same setad tender form, and 3 are a near-identical web card. The wide mark
(pink shield plus "AriaTender.neT") sits at the same place and scale on all of them. The metrics
show this directly. Seg-freeze and seg-full agree at IoU 0.86–0.93, and page coverage is almost
constant (4.3–6.3%), varying only with page height. All three models are essentially perfect on
the mark here. The only thing that separates them on this template is seg-full's radio-button
blobs. **Consequence:** any comparison over "all 73 pages" is about 84% driven by this template.
All the conclusions below come from the 12 remaining pages. That is a small sample, so each
pattern states its count.

### 4.2 The large, faint, stacked mark is the main recall failure (4 of 4 pages)

These are `70_original`, `82_original`, `0_552cff21fd` and `0_e6b580e44b`. The mark is the
*stacked* AriaTender variant: a grey "AriaTender" wordmark in a rounded font, with the Persian
subtitle "زنجیره معاملاتی آریاتندر" spaced out underneath. It spans most of the page width, is
often cropped by the page edge, sits at low contrast over dense black text, and on
`0_552cff21fd` also over a grey emblem.

- **Outcome:** 10 of the 12 model-page pairs are `bad`. The two exceptions are seg-full on `70` and
  seg-freeze on `82` (`minor`). The typical partial result is *the shield outline only*. The
  subtitle is never found.
- **Confidence behaviour:** these pages produce many 0.10–0.25 instances, while the template pages
  produce almost none. The ≥0.25 → ≥0.10 coverage change for seg-freeze:

  | page | coverage at ≥0.25 | coverage at ≥0.10 |
  |---|---|---|
  | `0_552cff21fd` | 1.6% | 18.2% |
  | `0_e6b580e44b` | 1.9% | 5.5% |
  | `70_original` | 4.1% | 8.3% |

  I did not check where each low-confidence instance lands. On `0_e6b580e44b` some fall on the
  black header banner, so lowering the threshold would recover some of the mark along with new
  false positives. Treat that as an experiment, not a fix.
- **Likely causes** (hypotheses; the training set isn't in the repo, see the note below):
  - *Scale is relative to the asset, not the page.* `scripts/kaggle_dataset/generate_dataset.py`
    samples `SCALE_RANGE = (0.15, 3.0)` of the watermark PNG's own size, with 70% of draws in
    0.3–1.5. Every background is also squashed to 1024×1024 first. So a stacked mark spanning
    and cropped by the page, as on these four pages, is a thin tail of the distribution.
  - *Watermark parts are placed independently.* The same generator treats `ariatender_logo` and
    `ariatender_persian` as separate watermarks placed at unrelated random positions, so the
    subtitle-under-wordmark layout is never shown as one object. A thin, widely spaced subtitle
    at low alpha is also the hardest thing to label (mask = alpha > 0.02).
  - *Backgrounds are synthetic.* They come from `wm_backgrounds_v2`, pages procedurally generated
    by `gen_persian_docs.py`. Real scanned newspaper notices, with halftone noise and heavy bold
    type under the mark, are not represented.

> **Which generator built the training data isn't certain.** The notebook describes the
> "new dataset" as 2000 images at 1024×1024 with a 1700/300 split. That matches
> `scripts/kaggle_dataset/generate_dataset.py` (2000 images, `TARGET_SIZE=(1024,1024)`,
> 85/15 split), not `scripts/wm_dataset/generate.py`, which keeps page aspect up to 1800px.
> Neither that dataset nor its `watermarks_processed/` assets are in the repo, so the causes
> above come from reading the generator code, not from inspecting the actual training images.

### 4.3 The grey gavel placeholder is mistaken for the mark (14 of 15 model-page pairs)

All 5 long setad portal pages carry a grey "پیش نمایش در دسترس نیست" (preview unavailable)
placeholder graphic just below the mark. It is two tilted cards, each with a gavel. seg-full and
det mark it on 5 of 5 pages, and seg-freeze on 4 of 5. The mark itself is usually found too.
The AriaTender shield *is* a gavel inside a shield outline, and the training data has no negative
with a gavel or similar judicial/legal icon. So the models learned "grey gavel" as a sufficient
cue. Best example: [`0_cc8bf23373`](review_enhanced/0_cc8bf23373.png).

This is the most fixable pattern. The placeholder is the same asset on every portal page, so a
handful of real crops used as hard negatives should remove it (§5.2).

### 4.4 seg-full marks form widgets and UI chrome

- **Template pages:** 3–5 small blobs on the round radio buttons above the mark, on all 58 pages
  (0.12–0.19% of the page).
- **Other UI:** the bottom "بازگشت" (back) button on 2 portal pages, and the title-bar icons of
  the Excel screenshot.
- **seg-freeze doesn't do this** on the template, even though both were trained on the same data.
  One plausible reading: the frozen backbone keeps generic COCO features, while the full finetune
  learned a low-level "small grey rounded outline" cue from the shield's curves. That is a
  hypothesis, not tested here.
- **Impact:** the blobs are small and would mostly survive `segmenter.py`'s filter. Unmixing a
  radio button is only a small visual change, but it happens on every template page.

### 4.5 Bold display text can trigger dense false positives (3 pages, but severe)

- **seg-freeze, `0_355a9bf621`:** a page of large, bold, blue Persian display text with no
  watermark gets 58 masks over the letterforms.
- **seg-full, `70_original`:** fires on the red headline word "آگهی".
- **det-freeze, `0_23a4bae7d5`:** boxes the "MTPP-1288" reference text.

Thick, rounded glyph strokes at display size look like the AriaTender wordmark's stroke width.
The synthetic backgrounds rarely have display-size bold type without a mark.

### 4.6 Detection boxes balloon when the mark is large and faint

On clean pages det localises well, with one box per word or letter group. When it can't resolve
the letters of a large faint mark (`70_original`, `0_552cff21fd`, `82_original`), it falls back to
a few large boxes. Those boxes cover paragraphs, the TCI logo, or half the page (24% on
`0_552cff21fd`). Separately, it boxed the SAIPAGAM logo on `0_e6b580e44b` (1 page). Any
box-scoped removal (Method 4) is unsafe on exactly these pages.

### 4.7 The tiled out-of-distribution mark is mostly found (1 page)

On `75_original`, the Excel screenshot with a tiled diagonal "CONFIDENTIAL", all three models
find most repeats and all three are `minor`:
- **seg-freeze** misses repeats over the dense left columns.
- **seg-full** finds more but adds title-bar blobs.
- **det-freeze** boxes are loose and miss a few.

The 20% tiling in the generator plausibly teaches the "repeated text in a grid" layout. With
one page, that is suggestive only.

### 4.8 Claims from the first pass that do not hold

- **"Positional-prior hallucination over headline text" (`70_original`): withdrawn.** A real,
  large, faint stacked AriaTender mark crosses the middle of that page. It is plainly visible on
  the enhanced panel. seg-full's 0.93-confidence silhouette traces it.
- **"Thin-ring non-AriaTender watermark" (`82_original`): withdrawn.** `82_original` is a cropped
  stacked AriaTender mark (shield plus "Aria"), the same page as the earlier `bad_examples`
  input. The note in the `segmenter.py` docstring that "69 = thin-ring shield" refers to an older
  `dataset/document_originals/69`, not `wm_testset/images/69_original.png`. The latter is an
  ordinary standard form.
- **"Recall degrades with non-paper (coloured) backgrounds": not supported.** The only
  coloured-banner page (`0_23a4bae7d5`) had its mark found by all three models. The first pass's
  other examples (`0_552cff21fd`, `0_e6b580e44b`) are §4.2 faint-large-stacked cases on paper.
- **"Det fragmentation is a near-universal defect": reframed.** It is structural (per-word boxes)
  and harmless where the boxes cover the mark. Det's real problems are §4.3 and §4.6.

## 5. Solutions, ranked by expected impact vs. effort

### 5.1 Build a small labelled, *diverse* real test set (foundational, low–medium effort)

Fixes the fact that the current set can't distinguish models (§4.1).
- **What to label:** the 12 non-template pages plus 5 template pages as a sanity check, in the
  YOLO-seg format described in `wm_testset/README.md`.
- **What to collect:** more stacked-mark newspaper notices, portal pages and other layouts. Aim
  for ≥ 10 pages per document type, and cap any one template at ~10% of the set.
- **What to report:** mask IoU and recall per document type, plus false-positive area on
  mark-free pages. Keep `0_355a9bf621` as a negative and add more clean pages.
- **How to tell it worked:** `scripts/train_segmenter.py --eval-only` gives real numbers, and
  every fix below is judged by per-document-type metrics on this set, not by panels.

### 5.2 Real-page hard negatives (high impact, low effort)

Fixes §4.3, §4.4, §4.5 and the logo false positive.
- **Crops to collect:** the gavel placeholder (identical on every setad page), radio buttons and
  "back" buttons, bold display-text crops like `0_355a9bf621`, and org logos (SAIPAGAM, TCI,
  Rahavard Tamin).
- **How to use them:** as mark-free backgrounds (or regions pasted into backgrounds) in the
  generator, so they appear with empty labels. Also place some *next to* real marks, the way the
  placeholder sits under the mark on portal pages.
- **Target:** `fp_graphic_icon` 14 → ~0, seg-full `fp_table_lines_ui` 61 → ~0, and seg-freeze
  empty on `0_355a9bf621`, with template-page recall unchanged. Rerun
  `run_models.py → make_review_panels.py` and compare.

### 5.3 Generate the stacked mark as it appears on real notices (high impact, medium effort)

Fixes §4.2.
- **Composite as one object:** logo plus subtitle at their real relative geometry
  (`compositor._load_stacked_mark` already does this in the other generator).
- **Scale relative to the page:** 0.8–1.9× page width, deliberately cropped. The `oversize`
  pattern in `scripts/wm_dataset/compositor.py` is the right shape to borrow.
- **Faint and over real content:** low opacity, over dense bold text and halftone-noisy scanned
  notices.
- **Label threshold:** check that the thin subtitle survives the mask threshold. The relative
  threshold in `scripts/wm_dataset/labels.py` exists for exactly this.
- **Target:** the 4 stacked pages move from `bad` to at least `minor`, with the subtitle covered.

### 5.4 Inference-side mitigations while the data work lands (medium impact, low effort)

- **Keep the post-filter on.** These numbers are raw output. `segmenter.py`'s filter (with the new
  seg-path coverage cap) is still worth measuring on the 12 hard pages once labels exist.
- **Don't use det boxes for removal on large marks.** Reject or clip boxes where most pixels are
  dark text or the box is much larger than its seg-mask content, or use det only as a presence
  gate for the seg mask.
- **Try a guarded union of the two seg models.** Accept a seg-full instance with no seg-freeze
  support only if it is large or high-confidence, which drops the radio-button blobs. Accept a
  seg-freeze instance only if its dark-ink fraction is low, which drops `0_355a9bf621`. Evaluate
  on the labelled set; it is not verified here.
- **Test a lower threshold on pages with many 0.10–0.25 instances.** Per §4.2 it adds recall and
  false positives together, so measure before adopting.
- **Consider a template fast path.** 61 of 73 pages are two fixed templates, and
  `watermark_remover/template_match.py` already exists. Deterministic matching there would free
  the model for the hard pages. Not evaluated here.

### 5.5 Training regime (low impact on its own)

Freeze vs. full is not the lever. Each trades one false-positive habit for another (widgets vs.
display text), and neither wins on the stacked mark. Re-compare them after 5.2 and 5.3, on the
labelled set.

## 6. Appendix — per-image verdicts

Links open the colour 2×2 panel. The contrast-enhanced version of each is in
`review_enhanced/<image>.png`. Non-template pages come first.

<!--APPENDIX:start-->
| image | doc type | watermark | seg-freeze | seg-full | det-freeze |
|---|---|---|---|---|---|
| [0_23a4bae7d5](compare/0_23a4bae7d5.png) | web_card_colored | wide | minor (fp_color_banner_image) | good | minor (fp_text) |
| [0_2573513521](compare/0_2573513521.png) | portal_long_page | wide | bad (fp_graphic_icon, partial_coverage) | bad (fp_graphic_icon, partial_coverage) | bad (fp_graphic_icon, fp_table_lines_ui) |
| [0_355a9bf621](compare/0_355a9bf621.png) | text_image_no_mark | none | bad (fp_text) | good | minor (fp_text) |
| [0_552cff21fd](compare/0_552cff21fd.png) | newspaper_notice | stacked_large_faint | bad (partial_coverage) | bad (missed_watermark) | bad (box_too_large_or_merged, fp_text) |
| [0_9f374fecce](compare/0_9f374fecce.png) | portal_long_page | wide | bad (fp_graphic_icon, fp_table_lines_ui) | bad (fp_graphic_icon, fp_table_lines_ui) | bad (fp_graphic_icon) |
| [0_a63948e17f](compare/0_a63948e17f.png) | portal_long_page | wide | bad (fp_graphic_icon) | bad (partial_coverage, fp_graphic_icon, fp_table_lines_ui) | bad (fp_graphic_icon) |
| [0_cc8bf23373](compare/0_cc8bf23373.png) | portal_long_page | wide | bad (fp_graphic_icon) | bad (fp_graphic_icon) | bad (fp_graphic_icon) |
| [0_e6b580e44b](compare/0_e6b580e44b.png) | newspaper_notice | stacked_large_faint | bad (partial_coverage) | bad (partial_coverage) | bad (partial_coverage, fp_logo_emblem) |
| [0_f57fd0e5d7](compare/0_f57fd0e5d7.png) | portal_long_page | wide | good | bad (partial_coverage, fp_graphic_icon) | bad (fp_graphic_icon) |
| [70_original](compare/70_original.png) | notice_colored_frame | stacked_large_faint | bad (partial_coverage) | minor (partial_coverage, fp_text) | bad (box_too_large_or_merged, partial_coverage) |
| [75_original](compare/75_original.png) | excel_screenshot | ood_confidential_tiled | minor (ood_watermark, partial_coverage) | minor (ood_watermark, fp_table_lines_ui) | minor (ood_watermark, partial_coverage, fp_table_lines_ui) |
| [82_original](compare/82_original.png) | notice_bordered | stacked_large_faint | minor (mask_bleed) | bad (partial_coverage) | bad (box_too_large_or_merged, partial_coverage) |
| [0_000bf78605](compare/0_000bf78605.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_02b607fa88](compare/0_02b607fa88.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_0bac2afc07](compare/0_0bac2afc07.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_1883524d63](compare/0_1883524d63.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_192b4e909c](compare/0_192b4e909c.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_24cbbf8903](compare/0_24cbbf8903.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_3104e71c19](compare/0_3104e71c19.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_363cf03d05](compare/0_363cf03d05.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_3ba779527b](compare/0_3ba779527b.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_3fafdb3957](compare/0_3fafdb3957.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_421284cfff](compare/0_421284cfff.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_43c9f722e2](compare/0_43c9f722e2.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_454be6f65a](compare/0_454be6f65a.png) | web_card | wide | good | good | good |
| [0_5058b6138d](compare/0_5058b6138d.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_5e2a16c39b](compare/0_5e2a16c39b.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_63dcc4a1e3](compare/0_63dcc4a1e3.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_659ce518ac](compare/0_659ce518ac.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_67f0d1eb00](compare/0_67f0d1eb00.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_69212ea400](compare/0_69212ea400.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_693a13316c](compare/0_693a13316c.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_6a08735940](compare/0_6a08735940.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_71cfedcee6](compare/0_71cfedcee6.png) | web_card | wide | good | good | good |
| [0_74286b3d33](compare/0_74286b3d33.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_7882b61e2c](compare/0_7882b61e2c.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_7a547cc01e](compare/0_7a547cc01e.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_867338eedb](compare/0_867338eedb.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_876860feef](compare/0_876860feef.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_93a11281eb](compare/0_93a11281eb.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_9b3249c2b8](compare/0_9b3249c2b8.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_9f02983b8d](compare/0_9f02983b8d.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_9f380d21d5](compare/0_9f380d21d5.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_a20f0fcd3f](compare/0_a20f0fcd3f.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_a376b5d68f](compare/0_a376b5d68f.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_a5b2ed4727](compare/0_a5b2ed4727.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_a5fe2c63d7](compare/0_a5fe2c63d7.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_a6cf7f72c8](compare/0_a6cf7f72c8.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_a887bfdadb](compare/0_a887bfdadb.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_ab50b935d6](compare/0_ab50b935d6.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_b2d7a9cb6d](compare/0_b2d7a9cb6d.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_c2f8c127b9](compare/0_c2f8c127b9.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_cf8939c6e4](compare/0_cf8939c6e4.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_d35830f41a](compare/0_d35830f41a.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_d4a8b722eb](compare/0_d4a8b722eb.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_d513362f80](compare/0_d513362f80.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_dad9cf8e8a](compare/0_dad9cf8e8a.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_e2245d56f8](compare/0_e2245d56f8.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_e22a62e49c](compare/0_e22a62e49c.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_e246822b77](compare/0_e246822b77.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_e42bc467d6](compare/0_e42bc467d6.png) | web_card | wide | good | good | good |
| [0_e436733f34](compare/0_e436733f34.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_e58075890c](compare/0_e58075890c.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_e631dc143d](compare/0_e631dc143d.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_e9225ef657](compare/0_e9225ef657.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_f711f9d448](compare/0_f711f9d448.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_f872257aa5](compare/0_f872257aa5.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_fa5f871138](compare/0_fa5f871138.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_fad11fd1dd](compare/0_fad11fd1dd.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_fbc79a7fb0](compare/0_fbc79a7fb0.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_fcf7aa3b06](compare/0_fcf7aa3b06.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [0_fd8cdb0fd4](compare/0_fd8cdb0fd4.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
| [69_original](compare/69_original.png) | standard_form | wide | good | minor (fp_table_lines_ui) | good |
<!--APPENDIX:end-->
