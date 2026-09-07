from dataclasses import dataclass

import cv2
import numpy as np

# Above this alpha, unmixing is numerically unstable: recovering the true
# pixel divides by (1-alpha), so any small error in the estimated mark
# color or background is amplified without bound as alpha -> 1. Rather
# than clip-and-hope, the recovery fades smoothly from "trust the unmixed
# value" to "trust the plain inpainted estimate" across this band, so
# near-opaque stroke centers degrade to a smooth guess instead of
# producing numerical garbage. Validated against a real semi-transparent
# text watermark, where ~13% of masked pixels landed above this band.
_TRUST_CEILING = 0.75
_TRUST_BAND = 0.25
_INPAINT_RADIUS = 7


@dataclass
class UnmixResult:
    recovered: np.ndarray  # uint8 HxWx3, only mask pixels differ from the input
    alpha: np.ndarray  # float32 HxW, 0 outside mask
    mark_color: np.ndarray  # float32 (3,)


def _background_prior(img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A smooth, content-plausible guess at what's behind the mask, used
    both as the fallback for near-opaque pixels and as one side of the
    unmixing equation."""
    return cv2.inpaint(img_np, mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA).astype(np.float64)


def estimate_mark_color(img_np: np.ndarray, mask: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    """Estimates the watermark's own color by sampling the pixels within
    `mask` with the strongest `evidence` (e.g. brightness residual against
    local surroundings) -- those pixels are least diluted by whatever is
    underneath, so they're the best available proxy for the mark's true
    color."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.array([255.0, 255.0, 255.0])
    strengths = evidence[ys, xs]
    n = max(20, len(ys) // 20)
    top = np.argsort(-strengths)[:n]
    return img_np[ys[top], xs[top]].astype(np.float64).mean(axis=0)


def unmix_region(img_np: np.ndarray, mask: np.ndarray, mark_color: np.ndarray = None, evidence: np.ndarray = None, background: np.ndarray = None) -> UnmixResult:
    """Recovers true pixel values under a semi-transparent watermark inside
    `mask`, instead of replacing them with a flat background color.

    Model: observed = alpha*mark_color + (1-alpha)*true_pixel. A background
    estimate turns the per-pixel unmix into a solvable least-squares fit for
    alpha (3 equations -- one per channel -- for the single unknown alpha,
    given the known/estimated mark_color and background). The true pixel is
    then recovered directly from the solved alpha via the same equation,
    which is more faithful than the background estimate alone wherever
    alpha is small-to-moderate (the estimate is only a rough guess; the
    recovered value uses the actual observed pixel). It's also the fallback
    for near-opaque pixels, where recovery is numerically unreliable (see
    _TRUST_CEILING below) -- so a bad background estimate shows up twice.

    `background`: a caller-supplied background estimate, when one already
    exists and is more reliable than a generic guess -- e.g. a document's
    already-estimated flat paper color, which holds almost everywhere on a
    real page and is cheaper and more robust than interpolating one from
    context. If omitted, one is estimated by inpainting around `mask`,
    appropriate for photos where there's no flat "paper" to fall back on.
    Inpainting from a mask riddled with small excluded holes (e.g. real
    text pixels carved out of a watermark region) can itself be unreliable
    -- passing a real background estimate sidesteps that entirely.
    `mark_color`: the watermark's own color, if already known (e.g. from a
    caller that already isolated bright watermark pixels). If omitted, it's
    estimated from the pixels in `mask` with the strongest `evidence`.
    `evidence`: per-pixel "how strongly does this look like watermark"
    score, used only for color estimation when `mark_color` is omitted
    (e.g. brightness residual against local surroundings). Defaults to
    plain grayscale brightness if not given.
    """
    h, w = mask.shape[:2]
    img_f = img_np.astype(np.float64)

    if not np.any(mask > 0):
        return UnmixResult(recovered=img_np.copy(), alpha=np.zeros((h, w), dtype=np.float32), mark_color=np.array([255.0, 255.0, 255.0]))

    b_est = background.astype(np.float64) if background is not None else _background_prior(img_np, mask)

    if mark_color is None:
        if evidence is None:
            evidence = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float64)
        mark_color = estimate_mark_color(img_np, mask, evidence)
    mark_color = np.asarray(mark_color, dtype=np.float64)

    diff_obs = img_f - b_est
    diff_dir = mark_color[None, None, :] - b_est
    num = np.sum(diff_obs * diff_dir, axis=2)
    den = np.sum(diff_dir * diff_dir, axis=2) + 1e-6
    alpha = np.clip(num / den, 0.0, 1.0)

    recovered_raw = (img_f - alpha[:, :, None] * mark_color[None, None, :]) / np.clip(1 - alpha[:, :, None], 1e-3, None)
    recovered_raw = np.clip(recovered_raw, 0, 255)
    trust = np.clip((_TRUST_CEILING - alpha) / _TRUST_BAND, 0.0, 1.0)[:, :, None]
    blended = trust * recovered_raw + (1 - trust) * b_est

    mask_bool = mask > 0
    out = img_f.copy()
    out[mask_bool] = blended[mask_bool]
    out = np.clip(out, 0, 255).astype(np.uint8)

    alpha_out = np.where(mask_bool, alpha, 0.0).astype(np.float32)
    return UnmixResult(recovered=out, alpha=alpha_out, mark_color=mark_color.astype(np.float32))
