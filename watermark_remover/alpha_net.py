"""Per-pixel watermark-opacity ("alpha") regression network, plus the
closed-form removal it enables.

STANDALONE by design: this module imports only ``math``, ``numpy``,
``torch``, ``torch.nn`` and ``torch.nn.functional`` -- no package-relative
imports, and specifically no ``import config`` (``watermark_remover.config``
mutates process-wide environment variables and creates ``dataset/``
directories on import, neither of which may happen on a Kaggle kernel that
only wants the model definition). The Kaggle training notebook
(``train_watermark_seg_kaggle.ipynb``, section 10a) writes this exact file
verbatim via ``%%writefile``, so this module is the single source of truth
for the architecture -- the notebook's copy and this one must never diverge.

The physics
-----------
A semi-transparent watermark is ink composited over a page:

    observed = a * ink + (1 - a) * true

where ``a`` (alpha, in [0, 1]) is the per-pixel opacity of the mark and
``ink`` is the mark's own colour at that pixel. If ``a`` and ``ink`` are
known, the true page is recovered EXACTLY by inverting that equation:

    true = (observed - a * ink) / (1 - a)

-- see ``recover`` below. Real text, rules, and any other content that sits
under the mark survives this inversion undamaged: there is no thresholding,
no inpainting, no hallucinated pixels. Where the network predicts ``a = 0``
(a page pixel with no mark at all), ``recover`` returns ``observed``
unchanged by construction -- the network cannot invent content, only ever
estimate how much of a known-linear blend to undo. This is the key
difference from the project's other removal strategies (M1-M3, M4's other
strategies), which classify+fill or classify+subtract rather than solving
the compositing equation directly.

Why the alpha ceiling
----------------------
``AlphaUNet`` bounds its alpha prediction to ``[0, alpha_max]`` via
``alpha_max * sigmoid(...)`` rather than a free ``sigmoid`` in ``[0, 1]``.
This is not a modelling nicety: as ``a -> 1`` the denominator of the inverse
above, ``(1 - a)``, goes to zero and any error in the ``ink`` estimate is
amplified without bound.

Two datasets have been measured against this module, with two different
generators and two different calibrated ceilings:

- The project's original synthetic dataset (``wm_dataset_v2/alpha``, 588
  alpha maps, 80 negatives), from ``scripts/wm_dataset``'s compositor
  (``opacity_mult ~ U(0.5, 1.5)`` around a calibrated base alpha of
  ~0.222, single watermark layer only): per-image max alpha p50 = 0.239,
  p99 = 0.546, true max = 0.569; per-watermark-pixel alpha p99.9 = 0.475.
  Nothing here ever approaches fully opaque ink.
- The Kaggle "train and test2" dataset (``scripts/kaggle_dataset/
  generate_dataset.py``, per-mark opacity ``U(0.10, 0.70)``, up to 3
  stacked watermark layers per image, four blend modes): measured by
  ``scripts/kaggle_dataset/replay_alpha_targets.py`` replaying 60 of the
  2000 images exactly (bit-identical decoded JPEGs; see that script's
  ``replay_report.json``) -- per-image max alpha p50 = 0.430, p99 = 0.851,
  true max = 0.873; per-positive-pixel alpha p99 = 0.699; no pixel in this
  60-image sample exceeded 0.9. Stacking three layers at the range's own
  ceiling (opacity 0.70 each, via the over rule ``1 - (1-a)^3``) can reach
  ~0.973 in principle, so more of the 2000 images than were sampled here
  could plausibly exceed 0.9 -- this ceiling is a best-effort calibration
  from a partial replay, not an exhaustive bound.

``DEFAULT_ALPHA_MAX = 0.9`` is calibrated for the Kaggle dataset (the
0.7 value that suited the single-layer, lower-opacity original dataset
would clip a meaningful fraction of this one's real alpha values). At
``alpha_max = 0.9`` the denominator floor is ``1 - 0.9 = 0.1``, i.e. a
10x worst-case noise-amplification factor -- an order of magnitude worse
than the old ceiling's 3.3x. Recovery is genuinely ill-conditioned for
pixels the network predicts near this ceiling: expect visibly noisier
``recover`` output there than in the low/mid-alpha regime one dataset's
worth of training previously saw. If a checkpoint is ever trained
specifically on the original, lower-opacity dataset again, prefer
constructing it with the lower ``alpha_max = 0.7`` explicitly -- this
default now serves the harder (Kaggle) case, and ``load_model`` always
reads the ceiling that was actually trained with from the checkpoint
regardless of this default (see ``checkpoint_dict``/``load_model``).

Honest limits
--------------
- Edges: ``compositor.apply_scan_augmentations`` (original dataset) and
  ``generate_dataset.apply_post_processing`` (Kaggle dataset, replayed by
  ``scripts/kaggle_dataset/replay_alpha_targets.py``) both blur/resample
  the composite -- and, in the replay, the clean target and alpha map too,
  with the SAME sampled parameters -- but blur does not commute with the
  multiply in the compositing equation, so the equation is only
  approximate right at mark edges (where alpha changes quickly over the
  blur kernel's support), not in flat interior regions. The network is
  trained against this post-augmentation reality, not the pre-augmentation
  exact physics.
- Kaggle dataset only -- overlapping layers break the single-``ink``
  model: ``generate_dataset.py`` composites each watermark placement (and,
  for a tiled watermark, each tile instance) against whatever the
  background *already looks like after earlier placements*, not against
  the pristine clean page -- so where two placements' non-"normal" blend
  modes (multiply/screen/overlay) overlap the same pixel, the effective
  ink at that pixel depends on placement order and is not exactly
  recoverable as a single ``(alpha, ink)`` pair. Measured on 5 replayed
  Kaggle images (``physics_check.py``, ad hoc verification script): a
  single non-overlapping placement recovers to within ~1-3 (0-255 units)
  MAE of the true clean pixels using an oracle (alpha, ink) pair, close to
  the ambient JPEG-noise floor; the worst observed case (a single "screen"
  placement at alpha_max=0.6, on an unusually noisy image whose JPEG-only
  background MAE was itself ~5.4) reached ~10.7 MAE -- consistent with the
  noise-amplification factor of ``1/(1-0.6) = 2.5x`` predicted above, not
  evidence of a broken model. Overlapping/tiled non-normal-blend regions
  are expected to be harder than this best case.
- ``ink`` is only meaningful where ``alpha`` is large enough to matter --
  around and below ``alpha ~ 0.02`` the compositing equation is dominated by
  ``true`` regardless of what ``ink`` says, so an ink prediction on a
  near-zero-alpha pixel carries essentially no information and should not
  be trusted or visualised on its own.
- Synthetic-to-real gap: every gram of training signal here comes from
  synthetic compositing -- either ``scripts/wm_dataset``'s AriaTender stamp
  onto scanned/procedural backgrounds, or ``scripts/kaggle_dataset``'s
  three AriaTender variants onto ``wm_backgrounds_v2``. Neither has been
  validated against a real, human-labelled watermark-opacity ground truth
  -- how well either transfers to a real scanned page (e.g. the Kaggle
  dataset's own unlabelled ``images_scraped/``) is unmeasured and unknown
  until real weights are trained and qualitatively checked (see
  ``train_watermark_seg_kaggle.ipynb`` section 10f).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT_VERSION = 1
DEFAULT_ALPHA_MAX = 0.9
DEFAULT_WIDTHS = (32, 64, 128, 256)

# Alpha channel bias, initialised so an untrained network starts near
# identity (predicts ~0 alpha everywhere): sigmoid(-4) ~= 0.0180, so
# alpha_max * sigmoid(-4) ~= 0.0162 at alpha_max = 0.9 -- small enough that
# `recover` barely perturbs the input before any training has happened.
_ALPHA_BIAS_INIT = -4.0


class _ConvBlock(nn.Module):
    """Two x (Conv3x3 -> GroupNorm(8) -> SiLU), same spatial size in/out.

    `dilation` widens the receptive field without changing resolution --
    used once, at the bottleneck, to see more page context (e.g. to tell a
    grey form panel from watermark ink) without another downsample.
    """

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        return x


class AlphaUNet(nn.Module):
    """Small U-Net predicting per-pixel watermark opacity (alpha) and ink
    colour from an RGB page crop.

    Input: ``(N, 3, H, W)`` RGB in ``[0, 1]``, H and W multiples of 16 (see
    ``predict_image`` for the reflect-pad/tiling wrapper that handles
    arbitrary page sizes).

    Architecture: an encoder of ``len(widths)`` levels, each a
    ``_ConvBlock``, with ``MaxPool2d(2)`` between consecutive levels
    (``len(widths) - 1`` downsamples -- 3 for the default 4 widths); a
    bottleneck ``_ConvBlock`` at ``widths[-1]`` with ``dilation=2`` (same
    resolution as the last encoder level, wider receptive field); a decoder
    that mirrors the encoder, each stage doing bilinear x2 upsample ->
    concat the matching encoder skip -> ``_ConvBlock`` back down to that
    level's width. A 1x1 conv head maps the final decoder width to 4
    channels: channel 0 -> alpha via ``alpha_max * sigmoid``, channels 1:4
    -> ink (RGB) via plain ``sigmoid``.
    """

    def __init__(self, widths=DEFAULT_WIDTHS, alpha_max: float = DEFAULT_ALPHA_MAX):
        super().__init__()
        widths = tuple(widths)
        if len(widths) < 2:
            raise ValueError(f"AlphaUNet needs at least 2 widths, got {widths!r}")
        self.widths = widths
        self.alpha_max = float(alpha_max)

        self.pool = nn.MaxPool2d(2)

        # Encoder: one block per width, first one taking 3-channel RGB.
        enc_in = (3,) + widths[:-1]
        self.encoders = nn.ModuleList([
            _ConvBlock(c_in, c_out) for c_in, c_out in zip(enc_in, widths)
        ])

        # Bottleneck: extra block at the last (smallest-resolution) width,
        # dilation=2, no further downsample.
        self.bottleneck = _ConvBlock(widths[-1], widths[-1], dilation=2)

        # Decoder: mirrors the encoder from the bottleneck back up to
        # widths[0], concatenating the matching encoder skip at each stage.
        # decoder[i] takes (bottleneck-or-previous-decoder-output at
        # widths[i+1]) upsampled + skip at widths[i], outputs widths[i].
        dec_in_ch = [widths[i + 1] + widths[i] for i in range(len(widths) - 1)]
        dec_out_ch = list(widths[:-1])
        self.decoders = nn.ModuleList([
            _ConvBlock(c_in, c_out) for c_in, c_out in zip(dec_in_ch, dec_out_ch)
        ])

        self.head = nn.Conv2d(widths[0], 4, kernel_size=1)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = _ALPHA_BIAS_INIT

    def forward(self, x: torch.Tensor):
        skips = []
        h = x
        n = len(self.encoders)
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < n - 1:
                h = self.pool(h)
        # h is now at the last (deepest) encoder level's resolution --
        # len(widths) - 1 pools total, matching "3 downsamples for 4
        # widths". The bottleneck widens the receptive field via dilation
        # at that SAME resolution, no further downsample.
        h = self.bottleneck(h)

        # Decoder: len(widths) - 1 upsample stages, each concatenating the
        # matching shallower skip. skips[-1] (the deepest level) is not
        # concatenated separately -- it already flows into the bottleneck
        # directly, at the same resolution, so there is nothing to skip
        # across for it.
        for dec, skip in zip(self.decoders[::-1], skips[-2::-1]):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.head(h)
        alpha = self.alpha_max * torch.sigmoid(out[:, 0:1])
        ink = torch.sigmoid(out[:, 1:4])
        return alpha, ink

    def num_params(self) -> int:
        """Total parameter count, e.g. ``print(f"{model.num_params():,}")``."""
        return sum(p.numel() for p in self.parameters())


def recover(obs: torch.Tensor, alpha: torch.Tensor, ink: torch.Tensor,
            alpha_max: float) -> torch.Tensor:
    """Closed-form inverse of ``obs = alpha*ink + (1-alpha)*true``, solved
    for ``true``. Broadcasts over any matching shape (``alpha`` is typically
    single-channel and broadcasts against ``obs``/``ink``'s 3 channels).

    The denominator is floored at ``1 - alpha_max`` (not merely away from
    exactly 0) so a worst-case alpha prediction can never blow up the
    division beyond what ``alpha_max`` itself bounds -- consistent with
    ``AlphaUNet`` never predicting above ``alpha_max`` in the first place.
    """
    denom = torch.clamp(1.0 - alpha, min=1.0 - alpha_max)
    return torch.clamp((obs - alpha * ink) / denom, 0.0, 1.0)


def checkpoint_dict(model: AlphaUNet, epoch, val_metrics: dict, train_config: dict) -> dict:
    """Bundles everything needed to reconstruct and describe a trained
    model into one plain dict, suitable for ``torch.save``."""
    return {
        "format_version": FORMAT_VERSION,
        "arch": "AlphaUNet",
        "widths": list(model.widths),
        "alpha_max": float(model.alpha_max),
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "train_config": train_config,
    }


def load_model(path, device="cpu") -> AlphaUNet:
    """Loads a checkpoint written by ``checkpoint_dict`` and returns an
    ``AlphaUNet`` in eval mode on ``device``.

    Uses ``torch.load(..., weights_only=True)`` -- every value inside the
    checkpoint dict (epoch, val_metrics, train_config, widths, alpha_max)
    must therefore be plain tensors/Python primitives (int/float/str/bool/
    None/list/dict of those), never arbitrary objects, or loading will
    raise. Rejects a checkpoint whose ``format_version`` doesn't match this
    module's ``FORMAT_VERSION``, with a clear error, rather than silently
    trying (and possibly failing confusingly) to load an incompatible
    architecture.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    version = ckpt.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported alpha_net checkpoint format_version={version!r} "
            f"(expected {FORMAT_VERSION}) at {path!r}. This checkpoint was "
            f"likely written by an incompatible version of alpha_net.py."
        )
    widths = tuple(ckpt["widths"])
    alpha_max = float(ckpt["alpha_max"])
    model = AlphaUNet(widths=widths, alpha_max=alpha_max)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _feather_weight(tile: int, overlap: int) -> np.ndarray:
    """Separable linear-ramp ("feather") blend weight for one tile: a 1-D
    ramp that rises linearly from just-above-zero to 1 over the first
    `overlap` pixels, stays at 1 in the middle, and ramps back down over the
    last `overlap` pixels; the 2-D weight is the outer product of that ramp
    with itself (separable). Adjacent tiles' overlapping regions then sum to
    a smooth blend instead of a hard seam. Never exactly zero, so a pixel
    covered by only one tile (e.g. at the padded canvas's true edge) still
    gets a well-defined (if locally low) weight -- the weighted average over
    a single sample equals that sample regardless of the weight's
    magnitude, so this never biases the result, only the numerical
    conditioning of the average.
    """
    overlap = max(1, min(overlap, tile // 2))
    ramp = np.linspace(1.0 / (overlap + 1), 1.0, overlap, dtype=np.float32)
    w = np.ones(tile, dtype=np.float32)
    w[:overlap] = ramp
    w[-overlap:] = ramp[::-1]
    return np.outer(w, w)


def predict_image(model: AlphaUNet, img_rgb_uint8: np.ndarray, device=None,
                   tile: int = 512, overlap: int = 64):
    """Runs `model` over an arbitrary-sized RGB page and returns
    ``(alpha, ink, recovered)``:
      - ``alpha``: ``(H, W)`` float32 in ``[0, alpha_max]``.
      - ``ink``: ``(H, W, 3)`` float32 in ``[0, 1]``.
      - ``recovered``: ``(H, W, 3)`` uint8 -- the closed-form removal
        (``recover``), computed ONCE from the merged alpha/ink maps (not
        per-tile, so there is no seam in the recovered image beyond
        whatever seam already exists in the blended alpha/ink).

    The image is reflect-padded so every side is >= `tile` and a multiple
    of 16 (AlphaUNet's resolution requirement), tiled with `overlap`-pixel
    overlap, and each tile's alpha/ink is blended into the full-size buffer
    with `_feather_weight` so tile seams don't show. Works unchanged for
    images smaller than one tile (a single tile covers the whole padded
    canvas). Runs under ``torch.no_grad()``; uses fp16 autocast only when
    `device` is CUDA.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    is_cuda = device.type == "cuda"

    model = model.to(device)
    model.eval()
    alpha_max = model.alpha_max

    h, w = img_rgb_uint8.shape[:2]
    pad_h = max(tile, int(math.ceil(h / 16.0) * 16))
    pad_w = max(tile, int(math.ceil(w / 16.0) * 16))
    pad_bottom = pad_h - h
    pad_right = pad_w - w

    img_f = img_rgb_uint8.astype(np.float32) / 255.0
    img_padded = np.pad(img_f, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="reflect")

    alpha_acc = np.zeros((pad_h, pad_w), dtype=np.float32)
    ink_acc = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    weight_acc = np.zeros((pad_h, pad_w), dtype=np.float32)

    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, pad_h - tile + 1), stride))
    if ys[-1] != pad_h - tile:
        ys.append(pad_h - tile)
    xs = list(range(0, max(1, pad_w - tile + 1), stride))
    if xs[-1] != pad_w - tile:
        xs.append(pad_w - tile)

    ramp2d = _feather_weight(tile, overlap)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                crop = img_padded[y:y + tile, x:x + tile, :]
                t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
                if is_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        a, ink = model(t)
                else:
                    a, ink = model(t)
                a_np = a[0, 0].float().cpu().numpy()
                ink_np = ink[0].float().permute(1, 2, 0).cpu().numpy()

                alpha_acc[y:y + tile, x:x + tile] += a_np * ramp2d
                ink_acc[y:y + tile, x:x + tile, :] += ink_np * ramp2d[..., None]
                weight_acc[y:y + tile, x:x + tile] += ramp2d

    weight_safe = np.clip(weight_acc, 1e-6, None)
    alpha_full = (alpha_acc / weight_safe)[:h, :w]
    ink_full = (ink_acc / weight_safe[..., None])[:h, :w, :]

    obs_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
    alpha_t = torch.from_numpy(alpha_full).unsqueeze(0).unsqueeze(0)
    ink_t = torch.from_numpy(ink_full).permute(2, 0, 1).unsqueeze(0)
    recovered_t = recover(obs_t, alpha_t, ink_t, alpha_max)
    recovered = (recovered_t[0].permute(1, 2, 0).numpy() * 255.0)
    recovered = np.clip(np.round(recovered), 0, 255).astype(np.uint8)

    return alpha_full.astype(np.float32), ink_full.astype(np.float32), recovered
