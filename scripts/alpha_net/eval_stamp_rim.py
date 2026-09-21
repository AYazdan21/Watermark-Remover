"""Evaluate Stamp Fit's fitted edge profile (``stamp_fit.EDGE_PROFILE``): does
it remove the faint rim left along letter edges by the old Gaussian-blur
model, without ever making a page worse?

For every AriaTender page in ``--images`` (default ``wm_realtune/images``,
skipping ``75_original`` -- not an AriaTender mark -- and ``0_355a9bf621`` --
no mark), a stamp is located ONCE with the app's own locator
(``doc_detect.stamp_fit_locator_weights()`` + ``alpha_net.predict_image`` +
``stamp_fit.locate_stamps``, exactly as ``make_real_targets.py`` does), then
``remove_stamps`` is run twice on deep copies of the same located marks: once
with ``EDGE_PROFILE`` off, once on.

Per page, per mark region, pixels in the region's own window are pooled by
their signed distance ``d`` to the fitted letter edge (0.5px bins, d in
[-4, 4]) and the residual ``r = lum(B) - lum(cleaned)`` (B = the same
background estimate the fit used) is averaged in each bin, EXCLUDING page-
content pixels (|r| > 25 grey levels -- text/lines, not anything a watermark
can produce). A structured rim shows up as a non-flat mean curve; noise
averages out. The page-level **ghost score** is the RMS of these per-bin
means (also reported: the max |bin mean|). A **noise floor** is the same
statistic computed with the same window shifted off the mark (like
``stamp_fit._evidence``'s control) -- what "zero rim" looks like on that
exact page.

Safety checks: (1) zero pixels may change outside the final (EDGE_PROFILE=on)
footprint; (2) the mean interior change (d > 1.5, i.e. deep inside a stroke,
away from any edge) between the on/off outputs should be small -- flagged if
it exceeds 3 grey levels, since the interior model does not change on
purpose, only the edge does.

Side-by-side crops (observed | off | on, plus contrast-stretched |residual|
views for off/on) are written to ``--crops-dir``, OUTSIDE the repo by default
(real page content must not be committed).

Run from the repo root in the app's venv:
    ../.venv/Scripts/python.exe scripts/alpha_net/eval_stamp_rim.py
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from watermark_remover import alpha_net, doc_detect, stamp_fit  # noqa: E402

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
D_LO, D_HI, D_STEP = -4.0, 4.0, 0.5
BIN_LO = np.arange(D_LO, D_HI, D_STEP)
CONTENT_CUT = 25.0          # |r| above this = page content, not the mark
INTERIOR_D = 1.5             # d > this = "deep interior", away from any edge
INTERIOR_FLAG = 3.0           # grey levels
TEXT_UNDER_MARK_PAGES = {"70_original", "0_e6b580e44b", "0_552cff21fd"}


def _lum(img_float_or_uint8):
    return img_float_or_uint8.astype(np.float32) @ _LUMA


def _region_list(mk):
    """[(rn, part, region_name)] for one located mark's (already refined)
    parts, same naming as stamp_fit.remove_stamps."""
    out = []
    for p in mk["parts"]:
        names = list(stamp_fit.templates()[p["name"]]["regions"])
        for rname in names:
            rn = f"{p['name']}:{rname}" if len(names) > 1 else p["name"]
            out.append((rn, p, rname))
    return out


def _bin_means(d, r):
    """Per-bin mean of r over 0.5px bins of d in [-4, 4). NaN where empty."""
    means = np.full(len(BIN_LO), np.nan, np.float64)
    for i, lo in enumerate(BIN_LO):
        m = (d >= lo) & (d < lo + D_STEP)
        if m.any():
            means[i] = float(r[m].mean())
    return means


def _ghost_score(means):
    v = means[~np.isnan(means)]
    if v.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(v * v))), float(np.abs(v).max())


def _region_samples(part, rname, shape, bg_lum_full, obs_lum_full):
    """(d, r) sample pairs for one region's own window, and the window."""
    x0, y0, w, h = stamp_fit._parts_bbox([part], shape, margin=stamp_fit.EDGE_WINDOW_MARGIN)
    d = stamp_fit._signed_distance(part, rname, x0, y0, w, h)
    r = bg_lum_full[y0:y0 + h, x0:x0 + w] - obs_lum_full[y0:y0 + h, x0:x0 + w]
    return d, r, (x0, y0, w, h)


def _control_raw_samples(part, rname, shape, bg_lum_full, obs_lum_full):
    """Same shape's RAW (d, r) sample pairs from a shifted window off the
    mark, like stamp_fit._evidence's control (dy = +-0.55 * part height),
    concatenated over both shifts. Returned raw (not pre-binned) so callers
    can pool them with other regions' samples before binning -- averaging
    already-binned curves from unrelated regions is a Simpson's-paradox trap
    (two independently-improving curves can average to something worse)."""
    tw, th = stamp_fit._size(part["name"], part["scale"])
    x0, y0, w, h = stamp_fit._parts_bbox([part], shape, margin=stamp_fit.EDGE_WINDOW_MARGIN)
    d = stamp_fit._signed_distance(part, rname, x0, y0, w, h)
    H, W = shape
    ds, rs = [], []
    for dy in (-0.55 * th, 0.55 * th):
        sy0 = int(round(y0 + dy))
        if sy0 < 0 or sy0 + h > H:
            continue
        r = bg_lum_full[sy0:sy0 + h, x0:x0 + w] - obs_lum_full[sy0:sy0 + h, x0:x0 + w]
        ds.append(d.ravel())
        rs.append(r.ravel())
    if not ds:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ds), np.concatenate(rs)


def _stretch(a, lo_pct=1, hi_pct=99):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
    if hi <= lo:
        return np.zeros_like(a, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _label(img, text):
    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def _save_crop(path, img, clean_off, clean_on, bg_lum, x0, y0, w, h, scale=3):
    def up(a):
        return cv2.resize(a, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    obs_c = up(img[y0:y0 + h, x0:x0 + w])
    off_c = up(clean_off[y0:y0 + h, x0:x0 + w])
    on_c = up(clean_on[y0:y0 + h, x0:x0 + w])
    r_off = _stretch(np.abs(bg_lum - _lum(clean_off))[y0:y0 + h, x0:x0 + w])
    r_on = _stretch(np.abs(bg_lum - _lum(clean_on))[y0:y0 + h, x0:x0 + w])
    r_off_c = cv2.cvtColor(up(r_off), cv2.COLOR_GRAY2RGB)
    r_on_c = cv2.cvtColor(up(r_on), cv2.COLOR_GRAY2RGB)
    top = np.hstack([_label(obs_c, "observed"), _label(off_c, "off (today)"), _label(on_c, "on (edge profile)")])
    bot = np.hstack([np.zeros_like(r_off_c), _label(r_off_c, "|residual| off"), _label(r_on_c, "|residual| on")])
    out = np.vstack([top, bot])
    Image.fromarray(out).save(path)


def evaluate_page(path, locator, device, crops_dir):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    shape = img.shape[:2]
    a, _ink, _rec = alpha_net.predict_image(locator, img, device=device)
    accepted, _rejected = stamp_fit.locate_stamps(img, a)
    if not accepted:
        return dict(page=path.stem, skipped="no stamp accepted")

    marks_off = copy.deepcopy(accepted)
    marks_on = copy.deepcopy(accepted)
    # remove_stamps computes bg_inpaint/bg_closing ONCE, from the marks'
    # positions BEFORE _subpixel refines them (the refinement loop mutates
    # mk["parts"] in place afterwards) -- so recompute from a snapshot taken
    # before the call, not from marks_on/marks_off after it returns, or the
    # background estimate (and hence every residual computed against it)
    # would silently disagree with what the fit itself used.
    bg_snapshot = copy.deepcopy(accepted)
    stamp_fit.EDGE_PROFILE = False
    clean_off, A_off, info_off = stamp_fit.remove_stamps(img, marks_off)
    stamp_fit.EDGE_PROFILE = True
    clean_on, A_on, info_on = stamp_fit.remove_stamps(img, marks_on)

    # Recompute the SAME background estimate(s) the fit used (not returned
    # by remove_stamps), keyed per mark by info_on[i]["background"].
    bg_inpaint = _lum(stamp_fit._inpaint_background(img, bg_snapshot) * 255.0)
    bg_closing = _lum(stamp_fit._paper_background(img, bg_snapshot) * 255.0)
    obs_lum = _lum(img)
    off_lum = _lum(clean_off)
    on_lum = _lum(clean_on)

    pooled_d, pooled_r_off, pooled_r_on = [], [], []
    pooled_d_ctrl, pooled_r_ctrl = [], []
    used_regions = []
    interior_diffs = []
    best_crop = None  # (score_improvement, path-args) to pick the best-showing crop
    for mi, (mk, info) in enumerate(zip(marks_on, info_on)):
        bg_lum_full = bg_closing if info["background"] == "closing" else bg_inpaint
        for rn, part, rname in _region_list(mk):
            d, r_off_w, win = _region_samples(part, rname, shape, bg_lum_full, off_lum)
            _, r_on_w, _ = _region_samples(part, rname, shape, bg_lum_full, on_lum)
            dv = d.ravel()
            # A SHARED mask (safe under both the off and on residual, not a
            # separate per-model mask) so both curves are compared on the
            # same pixel set -- otherwise a model can look better only
            # because it quietly excludes different pixels.
            valid = (np.abs(r_off_w.ravel()) <= CONTENT_CUT) & (np.abs(r_on_w.ravel()) <= CONTENT_CUT)
            pooled_d.append(dv[valid])
            pooled_r_off.append(r_off_w.ravel()[valid])
            pooled_r_on.append(r_on_w.ravel()[valid])
            dc, rc = _control_raw_samples(part, rname, shape, bg_lum_full, off_lum)
            # Same page-content exclusion as the on-mark samples (`valid`
            # above) -- otherwise the "noise floor" is not noise at all, it's
            # whatever text/lines happen to sit in the shifted control
            # window, which swamps it and makes it useless as a "zero rim"
            # baseline (observed: floors of 20-30 grey levels on text-heavy
            # pages, many times bigger than the on-mark residual itself).
            valid_c = np.abs(rc) <= CONTENT_CUT
            pooled_d_ctrl.append(dc[valid_c])
            pooled_r_ctrl.append(rc[valid_c])
            ep = info["regions"][rn]["edge_profile"]
            used_regions.append((rn, ep["used"], ep["reason"]))

            interior2d = d > INTERIOR_D
            if interior2d.any():
                x0, y0, w, h = win
                diff = np.abs(clean_on[y0:y0 + h, x0:x0 + w].astype(np.float32)
                              - clean_off[y0:y0 + h, x0:x0 + w].astype(np.float32))
                interior_diffs.append(float(diff[interior2d].mean()))

            gs_off_r, _ = _ghost_score(_bin_means(dv[valid], r_off_w.ravel()[valid]))
            gs_on_r, _ = _ghost_score(_bin_means(dv[valid], r_on_w.ravel()[valid]))
            improvement = gs_off_r - gs_on_r
            if best_crop is None or improvement > best_crop[0]:
                x0, y0, w, h = stamp_fit._parts_bbox(mk["parts"], shape, margin=10)
                best_crop = (improvement, x0, y0, w, h, mi)

    # Pool RAW samples across all of the page's regions before binning (not
    # an average of independently-binned curves -- see
    # `_control_raw_samples`'s docstring for why that would be unsound).
    pooled_d = np.concatenate(pooled_d)
    pooled_r_off = np.concatenate(pooled_r_off)
    pooled_r_on = np.concatenate(pooled_r_on)
    pooled_d_ctrl = np.concatenate([a for a in pooled_d_ctrl if a.size])
    pooled_r_ctrl = np.concatenate([a for a in pooled_r_ctrl if a.size])
    means_off = _bin_means(pooled_d, pooled_r_off)
    means_on = _bin_means(pooled_d, pooled_r_on)
    means_ctrl = _bin_means(pooled_d_ctrl, pooled_r_ctrl)
    gs_off, mx_off = _ghost_score(means_off)
    gs_on, mx_on = _ghost_score(means_on)
    gs_floor, mx_floor = _ghost_score(means_ctrl)

    outside = int((((clean_on != img).any(axis=2)) & ~(A_on > 1e-4)).sum())
    interior_change = float(np.mean(interior_diffs)) if interior_diffs else 0.0

    crop_path = None
    if crops_dir is not None and best_crop is not None:
        _, x0, y0, w, h, mi = best_crop
        crop_path = crops_dir / f"{path.stem}_crop.png"
        bg_for_crop = bg_closing if info_on[mi]["background"] == "closing" else bg_inpaint
        _save_crop(crop_path, img, clean_off, clean_on, bg_for_crop, x0, y0, w, h)

    return dict(
        page=path.stem,
        ghost_off=round(gs_off, 3), ghost_on=round(gs_on, 3), ghost_floor=round(gs_floor, 3),
        max_off=round(mx_off, 3), max_on=round(mx_on, 3), max_floor=round(mx_floor, 3),
        regions=used_regions,
        outside_footprint_changed=outside,
        interior_change=round(interior_change, 3),
        interior_flag=interior_change > INTERIOR_FLAG,
        curve_off=[None if np.isnan(v) else round(float(v), 3) for v in means_off],
        curve_on=[None if np.isnan(v) else round(float(v), 3) for v in means_on],
        curve_floor=[None if np.isnan(v) else round(float(v), 3) for v in means_ctrl],
        bins=[round(float(v), 2) for v in BIN_LO],
        crop=str(crop_path) if crop_path else None,
        text_under_mark=path.stem in TEXT_UNDER_MARK_PAGES,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(REPO / "wm_realtune" / "images"))
    ap.add_argument("--skip", nargs="*", default=["75_original"])
    ap.add_argument("--negatives", nargs="*", default=["0_355a9bf621"])
    ap.add_argument("--locator", default=None)
    ap.add_argument("--crops-dir", default=None, help="where to save comparison crops (outside the repo)")
    ap.add_argument("--json-out", default=None, help="optional path to dump the full per-page results as JSON")
    args = ap.parse_args()

    device = "cuda" if alpha_net.torch.cuda.is_available() else "cpu"
    locator = alpha_net.load_model(args.locator or doc_detect.stamp_fit_locator_weights(), device=device)
    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in sorted(Path(args.images).iterdir()):
        if p.stem in args.skip or p.stem in args.negatives:
            print(f"{p.name}: skipped")
            continue
        res = evaluate_page(p, locator, device, crops_dir)
        results.append(res)
        if res.get("skipped"):
            print(f"{res['page']}: {res['skipped']}")
            continue
        print(f"{res['page']}: ghost off={res['ghost_off']} on={res['ghost_on']} floor={res['ghost_floor']} "
              f"| max off={res['max_off']} on={res['max_on']} "
              f"| outside_footprint={res['outside_footprint_changed']} interior_change={res['interior_change']}"
              f"{' [FLAG]' if res['interior_flag'] else ''}")
        for rn, used, reason in res["regions"]:
            print(f"    {rn}: {'used' if used else 'not used'} ({reason})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
