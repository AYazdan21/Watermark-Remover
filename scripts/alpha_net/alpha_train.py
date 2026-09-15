"""Dependency-light training library for the alpha-regression network
(``alpha_net.AlphaUNet``). This is the canonical copy -- the Kaggle
training notebook (section 10b, ``%%writefile /kaggle/working/
alpha_train.py``) writes this exact file verbatim, so this file and the
notebook's copy must never diverge. Written to be testable from a plain
Python process (see the training notebook's local smoke test) -- it
deliberately does NOT import IPython or anything notebook-specific.

Imports only: os, csv, math, pathlib, random, shutil, time, cv2, numpy,
torch, and ``alpha_net`` (expected importable next to this file, e.g. both
written to /kaggle/working by the notebook's writefile cells).

Dataset layout expected (two SEPARATE roots -- see ``find_kaggle_inputs``
and ``AlphaCropDataset``):

- ``images_root``: the Kaggle "train and test2" dataset's
  ``wm_dataset_colab/wm_dataset_out`` (or a local copy of it) --
  ``images_root/images/<split>/wm_NNNN.jpg``.
- ``targets_root``: EXACT per-pixel alpha/clean targets produced by
  ``scripts/kaggle_dataset/replay_alpha_targets.py`` (the reference JPEGs
  alone carry no such ground truth -- see that script's module docstring
  for why) -- ``targets_root/alpha/<split>/wm_NNNN.png`` (per-pixel
  accumulated watermark opacity, uint8) and ``targets_root/clean/<split>/
  wm_NNNN.png`` (the unwatermarked page, lossless).

Only image stems present under BOTH roots are used; a replay run with
``--limit`` (partial targets) still trains, just on fewer stems -- see
``AlphaCropDataset``.
"""

from __future__ import annotations

import csv
import math
import os
import pathlib
import random
import shutil
import time

import cv2
import numpy as np
import torch

import alpha_net


# ---------------------------------------------------------------------------
# Kaggle input discovery
# ---------------------------------------------------------------------------

def _looks_like_wm_dataset_out(p: pathlib.Path) -> bool:
    return (p / "images" / "train").is_dir() and (p / "labels" / "train").is_dir() and (p / "annotations").is_dir()


def find_kaggle_inputs(search_roots=("/kaggle/input",)) -> dict:
    """Locates the pieces of the Kaggle "train and test2" dataset that the
    alpha-net notebook section needs: the YOLO/annotation dataset root
    (``wm_dataset_out``), the unlabelled real test pages
    (``images_scraped``), and the two extra folders needed for a local
    ``replay_alpha_targets.py`` run (``wm_backgrounds_v2``,
    ``watermarks_processed`` -- not needed if pre-built ``alpha``/``clean``
    targets were uploaded directly instead).

    Returns a dict with keys ``wm_dataset_out``, ``images_scraped``,
    ``backgrounds``, ``watermarks`` -> ``pathlib.Path`` or ``None``. A
    missing entry maps to ``None`` with a clear printed note rather than
    raising -- training only strictly needs ``wm_dataset_out`` (plus a
    ``targets_root`` built separately); the other two are only needed to
    run the replay on Kaggle itself.

    Resolution order per key: a preferred candidate under
    ``<root>/train-and-test2/...`` first, then a recursive search under
    each of `search_roots`.
    """
    roots = [pathlib.Path(r) for r in search_roots]
    result = {"wm_dataset_out": None, "images_scraped": None, "backgrounds": None, "watermarks": None}

    preferred_rel = {
        "wm_dataset_out": "train-and-test2/wm_dataset_colab/wm_dataset_out",
        "images_scraped": "train-and-test2/images_scraped/images_scraped",
        "backgrounds": "train-and-test2/wm_backgrounds_v2",
        "watermarks": "train-and-test2/watermarks_processed",
    }
    for root in roots:
        for key, rel in preferred_rel.items():
            if result[key] is not None:
                continue
            cand = root / rel
            if key == "wm_dataset_out":
                if _looks_like_wm_dataset_out(cand):
                    result[key] = cand
            elif cand.is_dir():
                result[key] = cand

    for root in roots:
        if not root.exists():
            continue
        if result["wm_dataset_out"] is None:
            for p in sorted(root.rglob("wm_dataset_out")):
                if p.is_dir() and _looks_like_wm_dataset_out(p):
                    result["wm_dataset_out"] = p
                    break
        if result["images_scraped"] is None:
            deepest = None
            for p in root.rglob("images_scraped"):
                if not p.is_dir():
                    continue
                try:
                    has_imgs = any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in p.iterdir() if f.is_file())
                except OSError:
                    has_imgs = False
                if has_imgs and (deepest is None or len(p.parts) > len(deepest.parts)):
                    deepest = p
            result["images_scraped"] = deepest
        if result["backgrounds"] is None:
            for p in root.rglob("wm_backgrounds_v2"):
                if p.is_dir():
                    result["backgrounds"] = p
                    break
        if result["watermarks"] is None:
            for p in root.rglob("watermarks_processed"):
                if p.is_dir():
                    result["watermarks"] = p
                    break

    for key, val in result.items():
        if val is None:
            print(f"[find_kaggle_inputs] NOTE: could not locate '{key}' under {[str(r) for r in roots]}.")
        else:
            print(f"[find_kaggle_inputs] {key} -> {val}")
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _dilate_bool_mask(mask_bool: np.ndarray, ksize: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(np.float32)


class AlphaCropDataset(torch.utils.data.Dataset):
    """Random (train) or deterministic (val) crops pairing
    ``images_root/images/<split>`` with ``targets_root/{alpha,clean}/
    <split>`` by filename stem.

    Returns, per item: `obs` (3,crop,crop), `clean` (3,crop,crop), `alpha`
    (1,crop,crop, RAW/unclamped, as saved), `w_alpha` (1,crop,crop),
    `w_rec` (1,crop,crop) -- all float32 tensors, obs/clean in [0, 1].

    ``w_alpha`` and ``w_rec`` both start from the same base pixel weight
    ``weight = 1 + 4 * dilated(alpha > 0.01)`` (mark pixels, and a halo
    around them, count more), then diverge:

    - ``w_alpha = weight * ((alpha <= 0.01) | visible)`` where ``visible
      = dilate(mean_c|obs - clean| > 4/255, 3x3)`` (computed AFTER gain
      jitter, so it reflects what the jittered crop actually shows).
      Rationale: a pixel with ``alpha > 0`` but that isn't actually
      visibly different from `clean` (e.g. a "multiply" blend against a
      near-white patch) carries no recoverable signal about `alpha` in
      the observed pixel -- supervising the network to predict a specific
      nonzero alpha there would be teaching it to hallucinate structure
      that isn't observable. Pixels genuinely at alpha<=0.01 (true
      background) ARE supervised, so the network still learns to predict
      ~0 there.
    - ``w_rec = weight * (alpha <= alpha_max)``. Rationale: alpha above
      the network's own ceiling is architecturally unrecoverable (`recover`
      floors its denominator at ``1 - alpha_max`` regardless of what's
      asked of it) -- including those pixels in the reconstruction losses
      would just be penalising the network for a ceiling ``alpha_max``
      itself imposes, so they're excluded from `L_rec`/`L_comp` entirely.
    """

    def __init__(self, images_root, targets_root, split, crop, samples_per_epoch=None, p_watermark=0.7,
                 gain_jitter=(0.85, 1.0), deterministic=False, seed=0, alpha_max=None, log=print):
        self.images_root = pathlib.Path(images_root)
        self.targets_root = pathlib.Path(targets_root)
        self.split = split
        self.crop = int(crop)
        self.samples_per_epoch = samples_per_epoch
        self.p_watermark = p_watermark
        self.gain_jitter = gain_jitter
        self.deterministic = deterministic
        self.seed = seed
        self.alpha_max = float(alpha_max) if alpha_max is not None else alpha_net.DEFAULT_ALPHA_MAX

        img_dir = self.images_root / "images" / split
        alpha_dir = self.targets_root / "alpha" / split
        clean_dir = self.targets_root / "clean" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"AlphaCropDataset: missing images dir {img_dir}")
        if not (alpha_dir.is_dir() and clean_dir.is_dir()):
            raise FileNotFoundError(
                f"AlphaCropDataset: missing alpha/{split} or clean/{split} under {self.targets_root} "
                f"-- run replay_alpha_targets.py first (a partial replay with --limit is fine as long "
                f"as it covers at least a few images of this split)."
            )

        imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS}
        alphas = {p.stem: p for p in alpha_dir.glob("*.png")}
        cleans = {p.stem: p for p in clean_dir.glob("*.png")}

        stems = sorted(set(imgs) & set(alphas) & set(cleans))
        self.n_skipped = len(set(imgs) - set(stems))
        if self.n_skipped:
            log(f"[AlphaCropDataset:{split}] {self.n_skipped} image(s) under {img_dir} have no matching "
                f"alpha/clean target (partial replay?) -- skipped; using {len(stems)} stem(s).")
        if not stems:
            raise ValueError(
                f"AlphaCropDataset({self.images_root}, {self.targets_root}, split={split!r}): no stems "
                f"present in images AND alpha AND clean -- targets_root needs at least one replayed "
                f"image of this split."
            )

        self.stems = stems
        self.items = [(imgs[s], alphas[s], cleans[s]) for s in self.stems]

    def __len__(self):
        if self.samples_per_epoch is not None:
            return int(self.samples_per_epoch)
        if self.deterministic:
            return 2 * len(self.items)
        return len(self.items)

    def _read_triple(self, i):
        img_p, alpha_p, clean_p = self.items[i]
        obs_bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if obs_bgr is None:
            raise IOError(f"Failed to read {img_p}")
        obs = cv2.cvtColor(obs_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        clean_bgr = cv2.imread(str(clean_p), cv2.IMREAD_COLOR)
        if clean_bgr is None:
            raise IOError(f"Failed to read {clean_p}")
        clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        alpha_gray = cv2.imread(str(alpha_p), cv2.IMREAD_GRAYSCALE)
        if alpha_gray is None:
            raise IOError(f"Failed to read {alpha_p}")
        alpha = alpha_gray.astype(np.float32) / 255.0

        return obs, clean, alpha

    @staticmethod
    def _pad_if_needed(obs, clean, alpha, crop):
        h, w = alpha.shape[:2]
        pad_h = max(0, crop - h)
        pad_w = max(0, crop - w)
        if pad_h or pad_w:
            obs = np.pad(obs, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            alpha = np.pad(alpha, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        return obs, clean, alpha

    @staticmethod
    def _crop_origin(alpha, crop, rng, prefer_watermark):
        h, w = alpha.shape[:2]
        max_y0 = max(0, h - crop)
        max_x0 = max(0, w - crop)
        if prefer_watermark:
            ys, xs = np.nonzero(alpha > 0.05)
        else:
            ys, xs = (np.empty(0), np.empty(0))
        if prefer_watermark and len(ys) > 0:
            i = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[i]), int(xs[i])
            y0 = int(np.clip(cy - crop // 2, 0, max_y0))
            x0 = int(np.clip(cx - crop // 2, 0, max_x0))
        else:
            y0 = int(rng.integers(0, max_y0 + 1))
            x0 = int(rng.integers(0, max_x0 + 1))
        return y0, x0

    def __getitem__(self, idx):
        n = len(self.items)
        if self.deterministic:
            # Fixed function of (seed, idx): independent of worker count,
            # call order, or how many epochs have run before it.
            rng = np.random.default_rng([int(self.seed), int(idx)])
            img_idx = (idx // 2) % n
            prefer_wm = (idx % 2 == 0)
        else:
            rng = np.random.default_rng()
            img_idx = int(rng.integers(0, n))
            prefer_wm = rng.random() < self.p_watermark

        obs, clean, alpha = self._read_triple(img_idx)
        crop = self.crop
        obs, clean, alpha = self._pad_if_needed(obs, clean, alpha, crop)

        has_wm_px = bool(np.any(alpha > 0.05))
        y0, x0 = self._crop_origin(alpha, crop, rng, prefer_wm and has_wm_px)

        obs_c = obs[y0:y0 + crop, x0:x0 + crop, :].astype(np.float32).copy()
        clean_c = clean[y0:y0 + crop, x0:x0 + crop, :].astype(np.float32).copy()
        alpha_c = alpha[y0:y0 + crop, x0:x0 + crop].astype(np.float32).copy()

        gain = rng.uniform(self.gain_jitter[0], self.gain_jitter[1], size=(1, 1, 3)).astype(np.float32)
        obs_c = obs_c * gain
        clean_c = clean_c * gain

        weight_c = 1.0 + 4.0 * _dilate_bool_mask(alpha_c > 0.01, 7)

        # A>0 but invisible after gain jitter (e.g. "multiply" onto a
        # near-white patch) carries no recoverable alpha signal -- see the
        # class docstring. `visible` is dilated by 3x3 so a thin visible
        # rim around an otherwise-invisible mark still counts.
        diff_mean = np.abs(obs_c - clean_c).mean(axis=2)
        visible = _dilate_bool_mask(diff_mean > (4.0 / 255.0), 3)

        w_alpha_c = weight_c * np.maximum((alpha_c <= 0.01).astype(np.float32), visible)
        # alpha > alpha_max is architecturally unrecoverable by `recover`
        # (denominator floored at 1 - alpha_max) -- excluded from the
        # reconstruction losses so the network isn't penalised for a
        # ceiling it cannot exceed by construction.
        w_rec_c = weight_c * (alpha_c <= self.alpha_max).astype(np.float32)

        obs_t = torch.from_numpy(np.ascontiguousarray(obs_c.transpose(2, 0, 1)))
        clean_t = torch.from_numpy(np.ascontiguousarray(clean_c.transpose(2, 0, 1)))
        alpha_t = torch.from_numpy(np.ascontiguousarray(alpha_c[None, :, :]))
        w_alpha_t = torch.from_numpy(np.ascontiguousarray(w_alpha_c[None, :, :].astype(np.float32)))
        w_rec_t = torch.from_numpy(np.ascontiguousarray(w_rec_c[None, :, :].astype(np.float32)))
        return obs_t, clean_t, alpha_t, w_alpha_t, w_rec_t


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def alpha_loss(model_out, batch, alpha_max, lambdas=(5.0, 1.0, 0.5)):
    """model_out = (pred_alpha, pred_ink); batch = (obs, clean, alpha,
    w_alpha, w_rec). Returns (total_loss, {"L_alpha":..., "L_rec":...,
    "L_comp":...}) -- the parts dict holds plain Python floats (already
    .item()'d).

    The alpha target is clamped to `alpha_max` before computing L_alpha --
    the network's own output is bounded there (`alpha_max * sigmoid(...)`
    in `AlphaUNet.forward`), so an uncapped target above that ceiling
    would create an irreducible loss floor for no benefit; `w_rec` already
    excludes those pixels from L_rec/L_comp entirely (see
    `AlphaCropDataset`).
    """
    pred_alpha, pred_ink = model_out
    obs, clean, alpha, w_alpha, w_rec = batch

    alpha_clamped = torch.clamp(alpha, max=alpha_max)
    rec = alpha_net.recover(obs, pred_alpha, pred_ink, alpha_max)
    l_alpha = (w_alpha * (pred_alpha - alpha_clamped).abs()).mean()
    l_rec = (w_rec * (rec - clean).abs().mean(dim=1, keepdim=True)).mean()
    comp = pred_alpha * pred_ink + (1.0 - pred_alpha) * clean
    l_comp = (w_rec * (comp - obs).abs().mean(dim=1, keepdim=True)).mean()

    la, lr, lc = lambdas
    total = la * l_alpha + lr * l_rec + lc * l_comp
    parts = {"L_alpha": float(l_alpha.item()), "L_rec": float(l_rec.item()), "L_comp": float(l_comp.item())}
    return total, parts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, alpha_max) -> dict:
    """Pixel-accurate validation metrics over the WHOLE val set (accumulated
    across batches, not batch-averaged), split into watermark (alpha_gt >
    0.01) and background regions. Pixels with alpha_gt > alpha_max are
    excluded from every metric here (they are architecturally
    unrecoverable -- see `alpha_loss`); `frac_alpha_gt_max` reports how
    much of the val set that was. `val_score` is lower-is-better and
    penalises any damage `recover` does to genuinely clean pixels."""
    model.eval()
    sums = dict(alpha_err_wm=0.0, n_wm=0, alpha_bg=0.0, n_bg=0,
                rec_err_wm=0.0, id_err_wm=0.0, rec_err_bg=0.0, id_err_bg=0.0,
                n_excluded=0, n_total=0)

    with torch.no_grad():
        for obs, clean, alpha, _w_alpha, _w_rec in loader:
            obs = obs.to(device)
            clean = clean.to(device)
            alpha = alpha.to(device)

            pred_alpha, pred_ink = model(obs)
            rec = alpha_net.recover(obs, pred_alpha, pred_ink, alpha_max)

            recoverable = alpha <= alpha_max
            wm_mask = (alpha > 0.01) & recoverable
            bg_mask = (~(alpha > 0.01)) & recoverable

            alpha_err = (pred_alpha - torch.clamp(alpha, max=alpha_max)).abs()
            rec_err = (rec - clean).abs().mean(dim=1, keepdim=True) * 255.0
            id_err = (obs - clean).abs().mean(dim=1, keepdim=True) * 255.0

            n_wm = int(wm_mask.sum().item())
            n_bg = int(bg_mask.sum().item())
            if n_wm:
                sums["alpha_err_wm"] += float(alpha_err[wm_mask].sum().item())
                sums["rec_err_wm"] += float(rec_err[wm_mask].sum().item())
                sums["id_err_wm"] += float(id_err[wm_mask].sum().item())
                sums["n_wm"] += n_wm
            if n_bg:
                sums["alpha_bg"] += float(pred_alpha[bg_mask].sum().item())
                sums["rec_err_bg"] += float(rec_err[bg_mask].sum().item())
                sums["id_err_bg"] += float(id_err[bg_mask].sum().item())
                sums["n_bg"] += n_bg
            sums["n_excluded"] += int((~recoverable).sum().item())
            sums["n_total"] += int(alpha.numel())

    n_wm = max(sums["n_wm"], 1)
    n_bg = max(sums["n_bg"], 1)

    alpha_mae_wm = sums["alpha_err_wm"] / n_wm
    alpha_mean_bg = sums["alpha_bg"] / n_bg
    rec_mae_wm_255 = sums["rec_err_wm"] / n_wm
    identity_mae_wm_255 = sums["id_err_wm"] / n_wm
    rec_mae_bg_255 = sums["rec_err_bg"] / n_bg
    identity_mae_bg_255 = sums["id_err_bg"] / n_bg
    frac_alpha_gt_max = sums["n_excluded"] / max(sums["n_total"], 1)

    improvement_wm = (1.0 - rec_mae_wm_255 / identity_mae_wm_255) if identity_mae_wm_255 > 1e-8 else 0.0
    val_score = rec_mae_wm_255 + 2.0 * max(0.0, rec_mae_bg_255 - identity_mae_bg_255)

    return {
        "alpha_mae_wm": alpha_mae_wm,
        "alpha_mean_bg": alpha_mean_bg,
        "rec_mae_wm_255": rec_mae_wm_255,
        "identity_mae_wm_255": identity_mae_wm_255,
        "improvement_wm": improvement_wm,
        "rec_mae_bg_255": rec_mae_bg_255,
        "identity_mae_bg_255": identity_mae_bg_255,
        "val_score": val_score,
        "frac_alpha_gt_max": frac_alpha_gt_max,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_metrics_csv(path, epoch, train_parts, val_metrics, lr, seconds):
    path = pathlib.Path(path)
    is_new = not path.exists()
    fieldnames = (["epoch", "train_loss", "train_L_alpha", "train_L_rec", "train_L_comp", "lr", "seconds"]
                  + list(val_metrics.keys()))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        row = {
            "epoch": epoch,
            "train_loss": train_parts["total"],
            "train_L_alpha": train_parts["L_alpha"],
            "train_L_rec": train_parts["L_rec"],
            "train_L_comp": train_parts["L_comp"],
            "lr": lr,
            "seconds": seconds,
        }
        row.update(val_metrics)
        w.writerow(row)


def _make_loaders(config, log=print):
    images_root = config["images_root"]
    targets_root = config["targets_root"]
    crop = config["crop"]
    seed = config.get("seed", 0)
    nw = int(config.get("num_workers", 0))
    alpha_max = config["alpha_max"]
    is_cuda = torch.cuda.is_available()

    train_ds = AlphaCropDataset(
        images_root, targets_root, "train", crop, samples_per_epoch=config.get("samples_per_epoch"),
        deterministic=False, seed=seed, alpha_max=alpha_max, log=log,
    )
    val_ds = AlphaCropDataset(
        images_root, targets_root, "val", crop, samples_per_epoch=None, deterministic=True, seed=seed,
        alpha_max=alpha_max, log=log,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=nw,
        pin_memory=is_cuda, drop_last=True, persistent_workers=(nw > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=nw,
        pin_memory=is_cuda, persistent_workers=(nw > 0),
    )
    return train_loader, val_loader


def train(config: dict, on_epoch_end=None, log=print) -> dict:
    """Trains an AlphaUNet per `config` (see module/notebook docs for the
    full key list -- notably `images_root` and `targets_root`, replacing
    the old single `data_root`). Returns a summary dict: {best_epoch,
    best_metrics, best_path, last_path, stopped_early, epochs_ran}.
    """
    seed = int(config.get("seed", 0))
    _seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _make_loaders(config, log=log)
    iters_per_epoch = len(train_loader)
    if iters_per_epoch == 0:
        raise ValueError(
            "Training loader produced zero batches -- reduce batch_size or "
            "raise samples_per_epoch."
        )

    model = alpha_net.AlphaUNet(widths=config["widths"], alpha_max=config["alpha_max"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_epochs = int(config["epochs"])
    warmup_iters = iters_per_epoch  # 1-epoch linear warmup
    total_iters = max(1, total_epochs * iters_per_epoch)

    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / max(1, warmup_iters)
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = bool(config.get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.csv"

    start_epoch = 1
    best_score = float("inf")
    epochs_no_improve = 0
    global_step = 0

    if config.get("resume", True) and last_path.exists():
        # last.pt bundles optimizer/scheduler/scaler state -- arbitrary
        # objects, not just tensors/primitives -- so weights_only=True (as
        # used by alpha_net.load_model for the "clean" best.pt format)
        # cannot load it. weights_only=False is acceptable here ONLY
        # because last.pt is written exclusively by this same trusted
        # training loop and never loaded from an untrusted source.
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_score = ckpt.get("best_score", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = ckpt.get("global_step", (start_epoch - 1) * iters_per_epoch)
        log(f"[alpha_train] resuming from {last_path} at epoch {start_epoch}")

    summary = {
        "best_epoch": None, "best_metrics": None,
        "best_path": str(best_path), "last_path": str(last_path),
        "stopped_early": False, "epochs_ran": 0,
    }

    patience = int(config.get("patience", 12))
    snapshot_every = int(config.get("snapshot_every", 10))

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        t_epoch0 = time.time()
        part_sums = {"total": 0.0, "L_alpha": 0.0, "L_rec": 0.0, "L_comp": 0.0}
        n_batches = 0

        for obs, clean, alpha, w_alpha, w_rec in train_loader:
            obs = obs.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            alpha = alpha.to(device, non_blocking=True)
            w_alpha = w_alpha.to(device, non_blocking=True)
            w_rec = w_rec.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred_alpha, pred_ink = model(obs)
                    loss, parts = alpha_loss(
                        (pred_alpha, pred_ink), (obs, clean, alpha, w_alpha, w_rec), config["alpha_max"]
                    )
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(
                        f"CUDA out of memory during a training step "
                        f"(batch_size={config['batch_size']}, crop={config['crop']}). "
                        f"Try a smaller batch_size (e.g. 4) and/or crop (e.g. 384)."
                    ) from e
                raise

            scheduler.step()
            global_step += 1

            part_sums["total"] += float(loss.item())
            for k in ("L_alpha", "L_rec", "L_comp"):
                part_sums[k] += parts[k]
            n_batches += 1

        n_batches = max(1, n_batches)
        train_parts = {k: v / n_batches for k, v in part_sums.items()}

        val_metrics = evaluate(model, val_loader, device, config["alpha_max"])
        epoch_seconds = time.time() - t_epoch0
        lr_now = scheduler.get_last_lr()[0]

        log(
            f"[alpha_train] epoch {epoch}/{total_epochs} "
            f"train_loss={train_parts['total']:.4f} "
            f"(alpha={train_parts['L_alpha']:.4f} rec={train_parts['L_rec']:.4f} "
            f"comp={train_parts['L_comp']:.4f}) val_score={val_metrics['val_score']:.4f} "
            f"lr={lr_now:.2e} time={epoch_seconds:.1f}s"
        )
        _append_metrics_csv(metrics_path, epoch, train_parts, val_metrics, lr_now, epoch_seconds)

        improved = val_metrics["val_score"] < best_score
        if improved:
            best_score = val_metrics["val_score"]
            epochs_no_improve = 0
            torch.save(alpha_net.checkpoint_dict(model, epoch, val_metrics, config), best_path)
            summary["best_epoch"] = epoch
            summary["best_metrics"] = val_metrics
        else:
            epochs_no_improve += 1

        last_ckpt = alpha_net.checkpoint_dict(model, epoch, val_metrics, config)
        last_ckpt.update({
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "epochs_no_improve": epochs_no_improve,
            "global_step": global_step,
        })
        torch.save(last_ckpt, last_path)

        summary["epochs_ran"] += 1

        if epoch % snapshot_every == 0:
            snap_path = None
            if best_path.exists():
                snap_dir = out_dir.parent
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_path = snap_dir / f"alpha_net_best_epoch_{epoch:03d}.pt"
                shutil.copy2(best_path, snap_path)
            if on_epoch_end is not None:
                on_epoch_end(epoch, str(snap_path) if snap_path is not None else None, val_metrics)

        if epochs_no_improve >= patience:
            log(f"[alpha_train] early stopping at epoch {epoch} "
                f"({epochs_no_improve} epochs without improvement)")
            summary["stopped_early"] = True
            break

    return summary
