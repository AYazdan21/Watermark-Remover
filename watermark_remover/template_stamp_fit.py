"""Method 5: Template Stamp Fit -- remove a library watermark from a page,
no trained model.

Stamp Fit (``stamp_fit``) removes the two known AriaTender marks with the
exact compositing inverse, but finds them with a neural alpha network. This
method keeps everything after the finding and replaces the finding with the
model-free signal of ``template_signal``:

1. Locate: the page's faint-darkening signal (paper background by closing,
   text zeroed) is matched against each requested template over the scale
   range the template was built with (``meta.rel_width``), coarse NCC on a
   downscaled page, then a full-resolution scale / sub-pixel refinement.
2. Accept a fit only if ALL hold: NCC score >= ``min_score``; the rendered
   width >= ``stamp_fit.MIN_WIDTH_PX``; and the PAGE ITSELF shows the mark
   (``stamp_fit._evidence``: the colour must change along the fitted strokes
   clearly more than along the same shape shifted off the mark), with the
   thresholds read from ``stamp_fit`` (``MIN_CHANGE``, ``MIN_CHANGE_RATIO``,
   ``MIN_CHANGED_FRACTION``). The evidence check is what keeps false fits out:
   the locator is a plain correlation on a hand-made signal.
3. Remove: ``stamp_fit.remove_stamps`` -- sub-pixel refinement, per-region
   strength / ink fit, edge profile and the exact inverse, unchanged. Pixels
   outside the stamps' footprint stay byte-identical to the input.

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
from . import template_library as tl
from . import template_signal as ts

N_CANDS = 5            # candidates refined and evidence-checked per search


def _scales_for(tpl, page_w, n_default=40):
    tw = tpl["alpha"].shape[1]
    rw = (tpl["meta"] or {}).get("rel_width")
    if rw and rw.get("min") and rw.get("max"):
        lo, hi = 0.6 * float(rw["min"]), 1.6 * float(rw["max"])
        n = int(np.clip(round(np.log(hi / lo) / 0.04), 12, 40))
    else:
        lo, hi, n = 0.1, 4.0, n_default
    return np.geomspace(lo, hi, n) * page_w / tw


def _pose_dict(key, pose):
    return dict(name=key, scale=float(pose["scale"]), x=float(pose["x"]), y=float(pose["y"]), sigma=0.0)


def footprint_overlay(img, alpha, color=(255, 0, 0)):
    """Page with the outline of the removed footprint (alpha > 0.02) drawn on it."""
    out = np.ascontiguousarray(img.copy())
    cnts, _ = cv2.findContours((alpha > 0.02).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, max(1, int(round(0.0025 * max(img.shape[:2])))))
    return out


def clean_document_template(img_rgb_uint8, template_names, library_dir=None, min_score=0.30, max_marks=4):
    """Locate and remove library watermarks. Returns (cleaned uint8, alpha
    float32, info dict, message str). ``template_names``: library names / PNG
    paths, or ["__all__"] for every library template."""
    t_all = time.time()
    img = np.ascontiguousarray(np.asarray(img_rgb_uint8)[..., :3], np.uint8)
    H, W = img.shape[:2]
    names = list(template_names or [])
    if any(str(n) == "__all__" for n in names):
        names = tl.list_templates(library_dir)
    info = dict(accepted=[], rejected=[], extra_rejected=[], templates=[], removal=[], timings={}, errors=[])
    marks, exclusions = [], []
    t_loc = time.time()
    for name in names:
        try:
            tpl = tl.load_template(name, library_dir)
        except Exception as e:
            info["errors"].append(f"{name}: {e}")
            continue
        key = tl.register_with_stamp_fit(tpl)
        label = tl.template_label(tpl)
        info["templates"].append(label)
        scales = _scales_for(tpl, W)
        kernel = ts.kernel_for(ts.stroke_width(tpl["alpha"]) * float(scales.max()))
        for i in range(max_marks):
            loc = ts.locate_template(img, tpl["alpha"], scales, kernel, n_cands=N_CANDS, exclusions=exclusions)
            if loc is None:
                if i == 0:
                    info["rejected"].append(dict(template=label, reason="no candidate", score=0.0))
                break
            # Candidates best score first; the first one whose evidence passes wins (a false best
            # candidate on page clutter must not hide the real mark).
            chosen, first_fail = None, None
            for c in sorted(loc["candidates"], key=lambda c: -c["score"]):
                if c["score"] < min_score:
                    break
                pose = _pose_dict(key, c)
                width = stamp_fit._size(key, pose["scale"])[0]
                rec = dict(template=label, score=round(c["score"], 4), scale=round(pose["scale"], 4),
                           x=round(pose["x"], 2), y=round(pose["y"], 2), width_px=int(width))
                change, control, frac = stamp_fit._evidence(img, [pose])
                rec.update(change=round(change, 2), control=round(control, 2), changed_fraction=round(frac, 3))
                ok = (width >= stamp_fit.MIN_WIDTH_PX and change >= stamp_fit.MIN_CHANGE
                      and change >= stamp_fit.MIN_CHANGE_RATIO * control and frac >= stamp_fit.MIN_CHANGED_FRACTION)
                if ok:
                    chosen = (pose, rec, change, control, frac, width)
                    break
                if first_fail is None:
                    why = []
                    if width < stamp_fit.MIN_WIDTH_PX:
                        why.append(f"width {width}px < {stamp_fit.MIN_WIDTH_PX}")
                    if change < stamp_fit.MIN_CHANGE:
                        why.append(f"change {change:.1f} < {stamp_fit.MIN_CHANGE}")
                    if change < stamp_fit.MIN_CHANGE_RATIO * control:
                        why.append(f"change {change:.1f} < {stamp_fit.MIN_CHANGE_RATIO}x control {control:.1f}")
                    if frac < stamp_fit.MIN_CHANGED_FRACTION:
                        why.append(f"changed fraction {frac:.2f} < {stamp_fit.MIN_CHANGED_FRACTION}")
                    first_fail = dict(rec, reason="; ".join(why))
            if chosen is None:
                if first_fail is not None:
                    (info["rejected"] if i == 0 else info["extra_rejected"]).append(first_fail)
                elif i == 0:
                    info["rejected"].append(dict(template=label, score=round(loc["score"], 4),
                                                 reason=f"score {loc['score']:.2f} < {min_score:.2f}"))
                break
            pose, rec, change, control, frac, width = chosen
            exclusions.append((tpl["alpha"], dict(scale=pose["scale"], x=pose["x"], y=pose["y"])))
            info["accepted"].append(rec)
            marks.append(dict(kind=f"template:{label}", parts=[pose], iou=rec["score"],
                              change=round(change, 2), control=round(control, 2),
                              changed_fraction=round(frac, 3), width_px=int(width)))
    info["timings"]["locate_s"] = round(time.time() - t_loc, 3)
    t_rm = time.time()
    cleaned, alpha, rm_infos = stamp_fit.remove_stamps(img, marks)
    info["timings"]["remove_s"] = round(time.time() - t_rm, 3)
    info["timings"]["total_s"] = round(time.time() - t_all, 3)
    info["removal"] = rm_infos
    info["marks"] = marks            # parts after sub-pixel refinement
    return cleaned, alpha, info, _message(info, W, H)


def _message(info, W, H):
    acc, rej = info["accepted"], info["rejected"]
    head = f"### Template Stamp Fit: {len(acc)} mark(s) removed ({W}x{H}, {info['timings']['total_s']:.1f} s)"
    lines = [head]
    for a, rm in zip(acc, info.get("removal", [])):
        regs = ", ".join(f"{k}: o={v['strength']}, ink={v['ink']}" for k, v in rm["regions"].items())
        lines.append(f"- **{a['template']}**: score {a['score']:.2f}, {a['width_px']} px wide at ({a['x']:.0f}, {a['y']:.0f}), "
                     f"change {a['change']:.1f} vs control {a['control']:.1f} (changed {a['changed_fraction']:.2f}); {regs}")
    for r in rej:
        lines.append(f"- not removed, `{r['template']}`: {r['reason']}"
                     + (f" (score {r['score']:.2f})" if r.get("score") else ""))
    for e in info.get("errors", []):
        lines.append(f"- error: {e}")
    if not acc and not rej and not info.get("errors"):
        lines.append("- no templates selected")
    lines.append(f"\nlocate {info['timings']['locate_s']:.1f} s, remove {info['timings']['remove_s']:.1f} s")
    return "\n".join(lines)
