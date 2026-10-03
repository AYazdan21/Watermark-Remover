"""Method 5: Template Stamp Fit -- remove a library watermark from a page,
no trained model.

Stamp Fit (``stamp_fit``) removes the two known AriaTender marks with the
exact compositing inverse, but finds them with a neural alpha network. This
method keeps the exact inverse and replaces the finding with the model-free
signal of ``template_signal``:

1. Locate: the page's 3-channel faint-darkening signal (paper background by
   closing, text zeroed) is correlated, colour-matched, with the template's
   matted darkening ``alpha (1 - ink)`` over the scale range the template was
   built with, coarse NCC on a downscaled page then a full-resolution scale /
   sub-pixel refinement, polished against the mark-aware paper background.
   The scale window comes from the ABSOLUTE instance sizes recorded in the
   template (``meta.instance_width_px``; a site's mark need not scale with the
   page width), united with the v1 relative-width window; a template without
   that field (v1) is searched over ``geomspace(0.1, 4.0) * page_w / template_w``
   like Stamp Fit. A grey template falls back to the luminance-like match.
   The candidates are tried in order of significance ``z = score * sqrt(n_on /
   1000)`` (``n_on`` = on-page pixels with coverage >= 0.3), not raw NCC: a tiny fit
   at the page edge gets a high NCC by chance, the real mark has far more pixels.
2. Accept a fit only if ALL hold: NCC score >= ``min_score`` and significance
   ``z >= MIN_Z`` (the score of a fit on a handful of pixels is not evidence); the rendered
   width >= ``stamp_fit.MIN_WIDTH_PX``; at least 40% of the footprint is on the
   page; the PAGE ITSELF shows the mark (``stamp_fit._evidence``: the colour
   must change along the fitted strokes clearly more than along the same shape
   shifted off the mark), with the thresholds read from ``stamp_fit``
   (``MIN_CHANGE``, ``MIN_CHANGE_RATIO``, ``MIN_CHANGED_FRACTION``); and the
   template, fitted with one strength on the page's own pixels, explains at least
   ``MIN_EXPLAINED`` of the darkening there (``template_adaptive.fit_quality``).
   The evidence and fit-quality checks are what keep false fits out: the locator
   is a plain correlation on a hand-made signal.
3. Several templates are located independently (a template's own further
   instances exclude each other, other templates do not). Two accepted marks of
   DIFFERENT templates whose footprints overlap by >= 50% of the smaller one
   describe the same mark (the wide and the stacked AriaTender template on one
   logo): the one that explains more of the page's darkening is kept, the other
   goes to ``info["overlap_dropped"]``.
4. Remove, ``removal=`` one of
   ``"adaptive"`` (default, "Page-adaptive (v3)", ``template_adaptive``): per-mark
   fit on the page itself (strength per part, direction-aware edge profile, ink
   tint per ink region, local background, per-pixel opacity read from the page
   where it can be read) with a guard that falls back to the per-pixel model on a
   mark where it would be worse;
   ``"pixel"`` ("Per-pixel colour (v2)", ``template_remove``): per-pixel
   opacity and ink, one scalar strength per page, paper interpolated across the
   mark's footprint; ``"region"`` ("Per-region (Stamp Fit)"): the v1 path,
   ``stamp_fit.remove_stamps`` with the template's regions (for a v2 template the
   ink-colour regions of its ``regions.png``). Whichever is used, pixels outside
   the stamps' footprint stay byte-identical to the input.

Library templates reach ``stamp_fit`` through
``template_library.register_with_stamp_fit`` (see that module's docstring).
No dataset saving, no model loading.

Known limits: the mark must be darker than the page; rotation is fixed at 0;
a template is one rigid layout (one template per variant of a mark).
"""

import time

import cv2
import numpy as np

from . import stamp_fit
from . import template_adaptive as tad
from . import template_library as tl
from . import template_remove as trm
from . import template_signal as ts

N_CANDS = 12           # candidates refined and ranked by significance per search (5 missed a 872 px mark behind five chance fits at the page edge)
MIN_INSIDE = 0.4       # share of the footprint that must be on the page
EXTRA_SIZE_TOL = 1.3   # further instances of a template on a page: within this factor of the first one's size...
EXTRA_SCORE_FRAC = 0.6  # ...and scoring at least this fraction of the first one's score
ON_COVERAGE = 0.3      # coverage that counts a pixel as 'on' in the size term of the candidate ranking
MIN_Z = 1.0            # significance gate z = score * sqrt(n_on / 1000): real marks measured 1.45-6.1, chance fits <= 0.63
MIN_EXPLAINED = 0.25  # fit-quality gate: the template must explain at least this share of the darkening
OVERLAP_FRAC = 0.5     # marks of different templates overlapping by this share of the smaller footprint are one mark
REMOVALS = ("adaptive", "pixel", "region")
REMOVAL_NAMES = {"adaptive": "page-adaptive", "pixel": "per-pixel colour", "region": "per-region (Stamp Fit)"}


def _scales_for(tpl, page_w, n_default=40):
    """Page-pixels per template-pixel to search (see the module docstring)."""
    tw = tpl["alpha"].shape[1]
    meta = tpl["meta"] or {}
    iw = meta.get("instance_width_px")
    rw = meta.get("rel_width")
    if iw and iw.get("min") and iw.get("max"):
        lo, hi = 0.5 * float(iw["min"]) / tw, 2.0 * float(iw["max"]) / tw
        if rw and rw.get("min") and rw.get("max"):
            lo = min(lo, 0.6 * float(rw["min"]) * page_w / tw)
            hi = max(hi, 1.6 * float(rw["max"]) * page_w / tw)
        n = int(np.clip(round(np.log(hi / lo) / 0.04), 12, 40))
        return np.geomspace(lo, hi, n)
    return np.geomspace(0.1, 4.0, n_default) * page_w / tw


def _pose_dict(key, pose):
    return dict(name=key, scale=float(pose["scale"]), x=float(pose["x"]), y=float(pose["y"]), sigma=0.0)


def footprint_overlay(img, alpha, color=(255, 0, 0)):
    """Page with the outline of the removed footprint (alpha > 0.02) drawn on it."""
    out = np.ascontiguousarray(img.copy())
    cnts, _ = cv2.findContours((alpha > 0.02).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, max(1, int(round(0.0025 * max(img.shape[:2])))))
    return out


def colour_template_of(tpl):
    """The 3-channel locating template of a loaded template, or None when it is
    (near-)grey and the luminance-like match should be used."""
    W3 = ts.colour_template(tpl["alpha"], tpl["ink"])
    return W3 if ts.colour_spread(W3) >= 0.10 else None


def _is_v1(tpl):
    """A library template built by the v1 builder (bare / built-in PNGs have no builder)."""
    return tpl.get("kind") == "lib" and int((tpl.get("meta") or {}).get("builder_version", 1) or 1) < 2


def _footprint(alpha, pose, shape, thresh=ON_COVERAGE):
    """(x0, y0, mask) of the on-page part of the template's footprint (coverage >=
    ``thresh``) at ``pose``; mask is None when nothing is on the page."""
    t = ts.resized_template(alpha, pose["scale"])
    th, tw = t.shape[:2]
    H, W = shape
    x0, y0 = int(round(pose["x"])), int(round(pose["y"]))
    ix0, iy0 = max(0, -x0), max(0, -y0)
    ix1, iy1 = min(tw, W - x0), min(th, H - y0)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0, 0, None
    return x0 + ix0, y0 + iy0, t[iy0:iy1, ix0:ix1] >= thresh


def _overlap(a, b):
    """Overlap of two footprints (``_footprint`` triples) as a share of the smaller one."""
    ax, ay, am = a
    bx, by, bm = b
    if am is None or bm is None:
        return 0.0
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + am.shape[1], bx + bm.shape[1]), min(ay + am.shape[0], by + bm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = int((am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).sum())
    return inter / max(1, min(int(am.sum()), int(bm.sum())))


def _resolve_overlaps(marks, accepted, shape):
    """Cross-template overlap (module docstring, 3). Returns (kept marks, kept
    accepted records, dropped list of dicts). Greedy by explained energy."""
    foots = [_footprint(mk["tpl"]["alpha"], mk["parts"][0], shape) for mk in marks]
    labels = [tl.template_label(mk["tpl"]) for mk in marks]
    order = sorted(range(len(marks)), key=lambda i: (-marks[i].get("energy", 0.0), -accepted[i].get("z", 0.0)))
    keep, dropped = [], []
    for i in order:
        clash = None
        for j in keep:
            if labels[i] == labels[j]:
                continue
            ov = _overlap(foots[i], foots[j])
            if ov >= OVERLAP_FRAC:
                clash = (j, ov)
                break
        if clash is None:
            keep.append(i)
        else:
            j, ov = clash
            dropped.append(dict(dropped=accepted[i]["template"], kept=accepted[j]["template"], overlap=round(float(ov), 3),
                                energy_dropped=round(float(marks[i].get("energy", 0.0)), 2),
                                energy_kept=round(float(marks[j].get("energy", 0.0)), 2),
                                explained_dropped=accepted[i].get("explained"), explained_kept=accepted[j].get("explained"),
                                score_dropped=accepted[i]["score"], score_kept=accepted[j]["score"]))
    keep.sort()
    return [marks[i] for i in keep], [accepted[i] for i in keep], dropped


def locate_marks(img_rgb_uint8, template_names, library_dir=None, min_score=0.30, max_marks=4):
    """Locate library watermarks on a page (steps 1-3 of the module docstring)
    without removing them. Returns (marks, info): ``marks`` carry ``kind``, ``parts``
    (one pose dict), ``tpl`` and the evidence / fit-quality numbers; ``info`` has
    the accepted / rejected records and timings."""
    t_loc = time.time()
    img = np.ascontiguousarray(np.asarray(img_rgb_uint8)[..., :3], np.uint8)
    H, W = img.shape[:2]
    names = list(template_names or [])
    if any(str(n) == "__all__" for n in names):
        names = tl.list_templates(library_dir)
    info = dict(accepted=[], rejected=[], extra_rejected=[], templates=[], removal=[], timings={}, errors=[],
                overlap_dropped=[], v1_templates=[])
    marks = []
    for name in names:
        try:
            tpl = tl.load_template(name, library_dir)
        except Exception as e:
            info["errors"].append(f"{name}: {e}")
            continue
        key = tl.register_with_stamp_fit(tpl)
        label = tl.template_label(tpl)
        info["templates"].append(label)
        if _is_v1(tpl):
            info["v1_templates"].append(label)
        scales = _scales_for(tpl, W)
        W3 = colour_template_of(tpl)
        thick = ts.solid_thickness(tpl["alpha"])
        kfun = lambda s, _t=thick: ts.kernel_for(_t * s, ts.MAX_SOLID_KERNEL)
        first_scale, first_score = None, None
        exclusions = []          # this template's own accepted instances only (see the module docstring, 3)
        for i in range(max_marks):
            # a second instance of the same template on a page is stamped at the same size as the first:
            # searching the whole window again mostly finds page clutter that clears the evidence check
            sc_i = scales if first_scale is None else np.geomspace(first_scale / EXTRA_SIZE_TOL, first_scale * EXTRA_SIZE_TOL, 9)
            loc = ts.locate_template(img, tpl["alpha"], sc_i, kfun, n_cands=N_CANDS, exclusions=exclusions,
                                     color=W3, polish=True)
            if loc is None:
                if i == 0:
                    info["rejected"].append(dict(template=label, reason="no candidate", score=0.0))
                break
            # Candidates in order of SIGNIFICANCE, z = score * sqrt(n_on / 1000), not raw NCC (a tiny fit
            # at the page edge scores a high NCC by chance: 0.68 for 58 px vs 0.62 for the real 692 px
            # mark); the gates on the raw score stay. The first one whose evidence passes wins (a false
            # best candidate on page clutter must not hide the real mark).
            chosen, first_fail = None, None
            ranked = []
            for c in loc["candidates"]:
                if c["score"] < min_score or (first_score is not None and c["score"] < EXTRA_SCORE_FRAC * first_score):
                    continue
                n_on = ts.on_page_pixels(tpl["alpha"], c, (H, W), ON_COVERAGE)
                ranked.append((c["score"] * np.sqrt(n_on / 1000.0), n_on, c))
            ranked.sort(key=lambda t: -t[0])
            for z, n_on, c in ranked:
                pose = _pose_dict(key, c)
                width = stamp_fit._size(key, pose["scale"])[0]
                rec = dict(template=label, score=round(c["score"], 4), z=round(float(z), 3), n_on=int(n_on),
                           scale=round(pose["scale"], 4), x=round(pose["x"], 2), y=round(pose["y"], 2), width_px=int(width))
                inside = ts.inside_fraction(tpl["alpha"], pose, (H, W))
                change, control, frac = stamp_fit._evidence(img, [pose])
                rec.update(change=round(change, 2), control=round(control, 2), changed_fraction=round(frac, 3),
                           inside=round(inside, 3))
                ok = (z >= MIN_Z and width >= stamp_fit.MIN_WIDTH_PX and inside >= MIN_INSIDE and change >= stamp_fit.MIN_CHANGE
                      and change >= stamp_fit.MIN_CHANGE_RATIO * control and frac >= stamp_fit.MIN_CHANGED_FRACTION)
                fq = None
                if ok:
                    fq = tad.fit_quality(img, tpl, pose)
                    rec.update(explained=None if fq["E"] is None else round(float(fq["E"]), 3),
                               energy=round(float(fq["energy"]), 2))
                    if fq["E"] is not None and fq["E"] < MIN_EXPLAINED:
                        ok = False
                if ok:
                    chosen = (pose, rec, change, control, frac, width, fq)
                    break
                if first_fail is None:
                    why = []
                    if z < MIN_Z:
                        why.append(f"significance z {z:.2f} < {MIN_Z} (too few on-page pixels for the score)")
                    if width < stamp_fit.MIN_WIDTH_PX:
                        why.append(f"width {width}px < {stamp_fit.MIN_WIDTH_PX}")
                    if inside < MIN_INSIDE:
                        why.append(f"only {inside:.0%} of the mark on the page")
                    if change < stamp_fit.MIN_CHANGE:
                        why.append(f"change {change:.1f} < {stamp_fit.MIN_CHANGE}")
                    if change < stamp_fit.MIN_CHANGE_RATIO * control:
                        why.append(f"change {change:.1f} < {stamp_fit.MIN_CHANGE_RATIO}x control {control:.1f}")
                    if frac < stamp_fit.MIN_CHANGED_FRACTION:
                        why.append(f"changed fraction {frac:.2f} < {stamp_fit.MIN_CHANGED_FRACTION}")
                    if fq is not None and fq["E"] is not None and fq["E"] < MIN_EXPLAINED:
                        why.append(f"fit explains only {fq['E']:.2f} of the darkening (< {MIN_EXPLAINED})")
                    first_fail = dict(rec, reason="; ".join(why))
            if chosen is None:
                if first_fail is not None:
                    (info["rejected"] if i == 0 else info["extra_rejected"]).append(first_fail)
                elif i == 0:
                    info["rejected"].append(dict(template=label, score=round(loc["score"], 4),
                                                 reason=f"score {loc['score']:.2f} < {min_score:.2f}"))
                break
            pose, rec, change, control, frac, width, fq = chosen
            if first_scale is None:
                first_scale, first_score = pose["scale"], rec["score"]
            exclusions.append((tpl["alpha"], dict(scale=pose["scale"], x=pose["x"], y=pose["y"])))
            info["accepted"].append(rec)
            marks.append(dict(kind=f"template:{label}", parts=[pose], iou=rec["score"],
                              change=round(change, 2), control=round(control, 2),
                              changed_fraction=round(frac, 3), width_px=int(width), tpl=tpl,
                              explained=rec.get("explained"), energy=float(fq["energy"]) if fq else 0.0))
    if len({tl.template_label(mk["tpl"]) for mk in marks}) > 1:
        marks, info["accepted"], info["overlap_dropped"] = _resolve_overlaps(marks, info["accepted"], (H, W))
    info["timings"]["locate_s"] = round(time.time() - t_loc, 3)
    return marks, info


def copy_marks(marks):
    """Marks that can be removed again with another model (``remove_marks``
    replaces each mark's ``parts`` by the refined pose)."""
    return [dict(mk, parts=[dict(p) for p in mk["parts"]]) for mk in marks]


def remove_marks(img_rgb_uint8, marks, removal="adaptive", polish=True):
    """Remove located marks with the chosen model. Returns (cleaned uint8, alpha
    float32, per-mark info list); ``mk["parts"]`` of every mark is replaced by the
    refined pose. Pixels where the returned alpha is 0 are byte-identical to the input."""
    if removal not in REMOVALS:
        raise ValueError(f"removal must be one of {REMOVALS}, not {removal!r}")
    img = np.ascontiguousarray(np.asarray(img_rgb_uint8)[..., :3], np.uint8)
    if removal == "adaptive":
        return tad.remove_adaptive(img, marks, polish=polish)
    if removal == "pixel":
        return trm.remove_pixel(img, marks)
    plain = [{k: v for k, v in mk.items() if k != "tpl"} for mk in marks]
    out = stamp_fit.remove_stamps(img, plain)
    for mk, p in zip(marks, plain):
        mk["parts"] = p["parts"]
    return out


def clean_document_template(img_rgb_uint8, template_names, library_dir=None, min_score=0.30, max_marks=4,
                            removal="adaptive"):
    """Locate and remove library watermarks. Returns (cleaned uint8, alpha
    float32, info dict, message str). ``template_names``: library names / PNG
    paths, or ["__all__"] for every library template. ``removal``: "adaptive"
    (page-adaptive, default), "pixel" (per-pixel colour) or "region" (Stamp Fit's
    per-region fit)."""
    if removal not in REMOVALS:
        raise ValueError(f"removal must be one of {REMOVALS}, not {removal!r}")
    t_all = time.time()
    img = np.ascontiguousarray(np.asarray(img_rgb_uint8)[..., :3], np.uint8)
    H, W = img.shape[:2]
    marks, info = locate_marks(img, template_names, library_dir, min_score=min_score, max_marks=max_marks)
    info["removal_model"] = removal
    t_rm = time.time()
    cleaned, alpha, rm_infos = remove_marks(img, marks, removal)
    info["timings"]["remove_s"] = round(time.time() - t_rm, 3)
    info["timings"]["total_s"] = round(time.time() - t_all, 3)
    info["removal"] = rm_infos
    for mk in marks:
        mk.pop("tpl", None)
    info["marks"] = marks            # parts after sub-pixel refinement
    return cleaned, alpha, info, _message(info, W, H)


def _mark_line(a, rm):
    """One accepted mark's markdown line."""
    fit = f", explains {a['explained']:.2f}" if a.get("explained") is not None else ""
    head = (f"- **{a['template']}**: score {a['score']:.2f}, {a['width_px']} px wide at ({a['x']:.0f}, {a['y']:.0f}), "
            f"change {a['change']:.1f} vs control {a['control']:.1f} (changed {a['changed_fraction']:.2f}){fit}; ")
    if rm.get("model") == "adaptive":
        g = rm.get("ghost") or {}
        ghost = (f", ghost {g['before']} -> {g['v3']} (v2 {g['v2']}) grey levels" if g else "")
        if rm.get("reason"):
            note = f"; kept the per-pixel model ({rm['reason']})"
        elif rm.get("reverted"):
            note = "; reverted to the per-pixel model (v3 left a stronger rim)"
        else:
            note = ""
        m = ", ".join(f"{k}={v}" for k, v in rm.get("m", {}).items())
        dk = ", ".join(str(v) for v in rm.get("dk", []))
        body = (f"adaptive: m [{m}], ink tint dk [{dk}] (0-255), {rm.get('reliable', 0)} reliable px, "
                f"{rm.get('polished', 0)} polished{ghost}{note}, residual change {rm.get('change_after')}, {rm.get('seconds')} s")
        return head + body
    regs = ", ".join(f"{k}: o={v['strength']}, ink={v['ink']}" for k, v in rm["regions"].items())
    extra = ""
    if "m" in rm:
        extra = f"; m={rm['m']} ({rm['n_pixels_used']} px), blur {rm['sigma']}, residual change {rm.get('change_after')}"
    return head + regs + extra


def _message(info, W, H):
    acc, rej = info["accepted"], info["rejected"]
    model = REMOVAL_NAMES.get(info.get("removal_model"), str(info.get("removal_model")))
    head = (f"### Template Stamp Fit: {len(acc)} mark(s) removed ({W}x{H}, {info['timings']['total_s']:.1f} s, "
            f"{model} removal)")
    lines = [head]
    for a, rm in zip(acc, info.get("removal", [])):
        lines.append(_mark_line(a, rm))
    for r in rej:
        lines.append(f"- not removed, `{r['template']}`: {r['reason']}"
                     + (f" (score {r['score']:.2f})" if r.get("score") else ""))
    for o in info.get("overlap_dropped", []):
        lines.append(f"- dropped `{o['dropped']}` (explained energy {o['energy_dropped']}) as the same mark as "
                     f"`{o['kept']}` (energy {o['energy_kept']}), footprints overlap {o['overlap']:.0%}")
    for t in info.get("v1_templates", []):
        lines.append(f"- **{t}**: built with builder v1 (missing parts, 2 flat inks) -- rebuild it in the Template "
                     f"Builder tab for much better results")
    for e in info.get("errors", []):
        lines.append(f"- error: {e}")
    if not acc and not rej and not info.get("errors"):
        lines.append("- no templates selected")
    lines.append(f"\nlocate {info['timings']['locate_s']:.1f} s, remove {info['timings']['remove_s']:.1f} s")
    return "\n".join(lines)
