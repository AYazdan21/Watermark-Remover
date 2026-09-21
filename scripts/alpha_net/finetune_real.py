"""Fine-tune an alpha-net checkpoint on REAL pages, with Stamp Fit targets.

Why: every synthetic-trained checkpoint (v1-v3) puts false opacity on ~5% of
a real page's pixels (UI bars, table headers, ornaments) -- synthetic
training never shows the network real page UI -- and real pages render the
mark slightly differently from the synthetic composites. Real pages have no
ground-truth clean version, but for AriaTender marks Stamp Fit
(watermark_remover/stamp_fit.py) produces a physically consistent one: its
fitted per-pixel opacity and its exactly-inverted page. A page with no mark
is a negative: opacity 0 everywhere, which directly teaches "real UI is not
watermark".

Targets come from ``wm_realtune/alpha_targets/`` ({stem}_alpha.png as
uint16 opacity, {stem}_clean.png), made from the wm_realtune pages.

Training mixes real crops with synthetic composites of the calibrated stamps
(assets/stamps) on clean pages (wm_backgrounds_v2) so the network doesn't
forget everything else while fitting a handful of pages; low LR with its own
warmup + cosine (NOT the from-scratch schedule -- restarting a converged
model at peak LR is what degraded v3); lambda_ink = 1 (at 3 the ink term was
over half of v3's loss); EMA weights are what gets saved and evaluated; the
best checkpoint is chosen on HELD-OUT REAL pages.

Usage (from the repo root, in the app's venv):
    ../.venv/Scripts/python.exe scripts/alpha_net/finetune_real.py \
        --init weights/alpha_net_v3_best_final.pt --out weights/alpha_net_v4_realft.pt
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "watermark_remover"))
sys.path.insert(0, str(REPO / "scripts" / "alpha_net"))
import alpha_net  # noqa: E402
import alpha_train  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def dilate(mask, k):
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def make_weights(alpha, obs, clean, alpha_max):
    """Same definitions as the notebook's 11.1d / 10.6 (AlphaCropDataset)."""
    weight = 1.0 + 4.0 * dilate(alpha > 0.01, 7)
    visible = dilate(np.abs(obs - clean).mean(2) > 4.0 / 255.0, 3)
    w_alpha = weight * np.maximum((alpha <= 0.01).astype(np.float32), visible)
    w_rec = weight * (alpha <= alpha_max).astype(np.float32)
    return np.clip(alpha, 0, alpha_max), w_alpha.astype(np.float32), w_rec.astype(np.float32)


def to_tensors(obs, clean, alpha, alpha_max):
    a, wa, wr = make_weights(alpha, obs, clean, alpha_max)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
    u = lambda x: torch.from_numpy(np.ascontiguousarray(x[None]))
    return t(obs), t(clean), u(a), u(wa), u(wr)


def random_crop(rng, arrays, crop, centre_on=None):
    H, W = arrays[0].shape[:2]
    if H < crop or W < crop:
        pad = lambda x: np.pad(x, ((0, max(0, crop - H)), (0, max(0, crop - W))) + ((0, 0),) * (x.ndim - 2), mode="reflect")
        arrays = [pad(x) for x in arrays]
        H, W = arrays[0].shape[:2]
    if centre_on is not None and centre_on.any() and rng.random() < 0.7:
        ys, xs = np.nonzero(centre_on[:H, :W])
        i = rng.integers(len(ys))
        y0 = int(np.clip(ys[i] - crop // 2 + rng.integers(-crop // 4, crop // 4 + 1), 0, H - crop))
        x0 = int(np.clip(xs[i] - crop // 2 + rng.integers(-crop // 4, crop // 4 + 1), 0, W - crop))
    else:
        y0, x0 = int(rng.integers(0, H - crop + 1)), int(rng.integers(0, W - crop + 1))
    return [x[y0:y0 + crop, x0:x0 + crop] for x in arrays]


class RealPages:
    def __init__(self, stems, targets_dir, images_dir):
        self.pages = []
        for stem in stems:
            img = next(images_dir.glob(stem + ".*"))
            obs = np.asarray(Image.open(img).convert("RGB"), np.float32) / 255.0
            clean = np.asarray(Image.open(targets_dir / f"{stem}_clean.png").convert("RGB"), np.float32) / 255.0
            alpha = np.asarray(Image.open(targets_dir / f"{stem}_alpha.png"), np.float32) / 65535.0
            self.pages.append((stem, obs, clean, alpha))

    def sample(self, rng, crop):
        _, obs, clean, alpha = self.pages[int(rng.integers(len(self.pages)))]
        s = float(rng.uniform(0.8, 1.25))
        if abs(s - 1) > 0.02:
            H, W = alpha.shape
            size = (max(8, int(W * s)), max(8, int(H * s)))
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            obs = np.clip(cv2.resize(obs, size, interpolation=interp), 0, 1)
            clean = np.clip(cv2.resize(clean, size, interpolation=interp), 0, 1)
            alpha = np.clip(cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR), 0, 1)
        return random_crop(rng, [obs, clean, alpha], crop, centre_on=alpha > 0.02)


class SyntheticComposites:
    """Calibrated stamps on clean pages, exact targets, random scale/position/
    strength and (half the time) a random ink tint -- the same idea as the
    notebook's colour-augmented crisp composites, kept small."""

    def __init__(self, stamps_dir, backgrounds_dir):
        self.stamps = []
        for p in sorted(stamps_dir.glob("ariatender_*.png")):
            rgba = np.asarray(Image.open(p).convert("RGBA"), np.float32) / 255.0
            self.stamps.append((rgba[..., :3].copy(), rgba[..., 3].copy()))
        self.backgrounds = sorted(p for p in backgrounds_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        self._bg_cache = {}

    def _bg(self, i):
        if i not in self._bg_cache:
            self._bg_cache[i] = cv2.cvtColor(cv2.imread(str(self.backgrounds[i])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if len(self._bg_cache) > 24:
                self._bg_cache.pop(next(iter(self._bg_cache)))
        return self._bg_cache[i]

    def sample(self, rng, crop):
        bg = self._bg(int(rng.integers(len(self.backgrounds))))
        H, W = bg.shape[:2]
        if H < crop or W < crop:
            bg = cv2.resize(bg, (max(W, crop), max(H, crop)))
            H, W = bg.shape[:2]
        y0, x0 = int(rng.integers(0, H - crop + 1)), int(rng.integers(0, W - crop + 1))
        clean = bg[y0:y0 + crop, x0:x0 + crop].copy()
        ink, a = self.stamps[int(rng.integers(len(self.stamps)))]
        width = float(rng.uniform(0.6, 2.2)) * crop
        s = width / a.shape[1]
        tw, th = max(2, int(a.shape[1] * s)), max(2, int(a.shape[0] * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        a_s = np.clip(cv2.resize(a, (tw, th), interpolation=interp), 0, 1)
        ink_s = np.clip(cv2.resize(ink * a[..., None], (tw, th), interpolation=interp) / np.maximum(a_s, 1e-4)[..., None], 0, 1)
        if rng.random() < 0.5:
            hsv = cv2.cvtColor(ink_s.astype(np.float32), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(0, 360)) % 360
            hsv[..., 1] = np.clip(0.2 * hsv[..., 1] + rng.uniform(0.3, 0.9), 0, 1)
            ink_s = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)
        a_s = np.clip(a_s * float(rng.uniform(0.6, 1.5)), 0, 0.9)
        px, py = int(rng.integers(-tw // 2, crop - tw // 2 + 1)), int(rng.integers(-th // 2, crop - th // 2 + 1))
        A = np.zeros((crop, crop), np.float32)
        INK = np.zeros((crop, crop, 3), np.float32)
        sx, sy, dx, dy = max(0, -px), max(0, -py), max(0, px), max(0, py)
        ex, ey = min(crop, px + tw), min(crop, py + th)
        if ex > dx and ey > dy:
            A[dy:ey, dx:ex] = a_s[sy:sy + ey - dy, sx:sx + ex - dx]
            INK[dy:ey, dx:ex] = ink_s[sy:sy + ey - dy, sx:sx + ex - dx]
        obs = A[..., None] * INK + (1 - A[..., None]) * clean
        return [obs.astype(np.float32), clean, A]


# ---------------------------------------------------------------------------
# Evaluation on held-out real pages
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, pages, device):
    """Per page: mark error (mean |recovered - target clean| on mark pixels,
    grey levels), off-mark damage (% of pixels well away from the mark
    changed by > 6 levels) and false-positive opacity (% of those pixels with
    predicted alpha > 0.05)."""
    model.eval()
    out = {}
    for stem, obs, clean, alpha in pages:
        img = np.round(obs * 255).astype(np.uint8)
        a, _ink, rec = alpha_net.predict_image(model, img, device=device)
        mark = alpha > 0.02
        far = ~dilate(alpha > 0.005, 15)
        err = float(np.abs(rec.astype(np.float32) / 255 - clean).mean(2)[mark].mean() * 255) if mark.any() else float("nan")
        dmg = float((np.abs(rec.astype(np.int16) - img.astype(np.int16)).max(2) > 6)[far].mean() * 100)
        fp = float((a > 0.05)[far].mean() * 100)
        out[stem] = dict(mark_err=err, damage_pct=dmg, fp_pct=fp)
    model.train()
    return out


def summary_score(metrics):
    """Lower is better: mark error plus damage to real content (weighted so
    one percent of the page damaged costs as much as 5 grey levels of
    leftover mark)."""
    errs = [m["mark_err"] for m in metrics.values() if not math.isnan(m["mark_err"])]
    return (float(np.mean(errs)) if errs else 0.0) + 5.0 * float(np.mean([m["damage_pct"] for m in metrics.values()]))


def load_init(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    # full-state snapshots keep the RAW weights in model_state and the EMA
    # weights (the ones worth starting from) in ema_state
    key = "ema_state" if "ema_state" in ck else "model_state"
    m = alpha_net.AlphaUNet(widths=ck["widths"], alpha_max=ck["alpha_max"])
    m.load_state_dict({k.replace("module.", ""): v for k, v in ck[key].items() if k != "n_averaged"})
    return m, key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=str(REPO / "weights" / "alpha_net_v3_best_final.pt"))
    ap.add_argument("--out", default=str(REPO / "weights" / "alpha_net_v4_realft.pt"))
    ap.add_argument("--targets", default=str(REPO / "wm_realtune" / "alpha_targets"))
    ap.add_argument("--holdout", nargs="*", default=["0_f711f9d448", "70_original"],
                    help="page stems kept out of training and used to pick the best checkpoint")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--p-real", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambdas", type=float, nargs=4, default=[5.0, 2.0, 0.5, 1.0])
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    targets = Path(args.targets)
    stems = sorted(p.name[:-len("_alpha.png")] for p in targets.glob("*_alpha.png"))
    train_stems = [s for s in stems if s not in args.holdout]
    hold_stems = [s for s in stems if s in args.holdout]
    real = RealPages(train_stems, targets, REPO / "wm_realtune" / "images")
    held = RealPages(hold_stems, targets, REPO / "wm_realtune" / "images").pages
    synth = SyntheticComposites(REPO / "assets" / "stamps", REPO / "wm_backgrounds_v2")
    print(f"train pages: {train_stems}\nheld-out pages: {hold_stems}\ndevice: {device}")

    model, key = load_init(args.init)
    model.to(device).train()
    alpha_max = model.alpha_max
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (it - warm) / max(1, args.steps - warm)))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    base = evaluate(ema, held, device)
    best = (summary_score(base), 0, copy.deepcopy(ema.state_dict()), base)
    print(f"step 0 (init from {Path(args.init).name} [{key}]): score {best[0]:.2f} | "
          + " | ".join(f"{s}: err {m['mark_err']:.1f} dmg {m['damage_pct']:.2f}% fp {m['fp_pct']:.2f}%" for s, m in base.items()))
    history = [dict(step=0, score=best[0], metrics=base)]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        items = [(real if rng.random() < args.p_real else synth).sample(rng, args.crop) for _ in range(args.batch)]
        batch = [torch.stack(x) for x in zip(*[to_tensors(o, c, a, alpha_max) for o, c, a in items])]
        batch = [b.to(device, non_blocking=True) for b in batch]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pa, pi = model(batch[0])
        loss, parts = alpha_train.alpha_loss((pa.float(), pi.float()), batch, alpha_max, lambdas=tuple(args.lambdas))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
        if step % args.eval_every == 0 or step == args.steps:
            met = evaluate(ema, held, device)
            sc = summary_score(met)
            history.append(dict(step=step, score=sc, metrics=met, loss=float(loss.item()), parts=parts))
            flag = ""
            if sc < best[0]:
                best = (sc, step, copy.deepcopy(ema.state_dict()), met)
                flag = "  <- best"
            print(f"step {step} ({time.time() - t0:.0f}s) loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.1e} | score {sc:.2f} | "
                  + " | ".join(f"{s}: err {m['mark_err']:.1f} dmg {m['damage_pct']:.2f}% fp {m['fp_pct']:.2f}%" for s, m in met.items()) + flag)

    final = alpha_net.AlphaUNet(widths=model.widths, alpha_max=alpha_max)
    final.load_state_dict({k: v.cpu() for k, v in best[2].items()})
    cfg = dict(init=str(args.init), init_key=key, train_pages=train_stems, holdout_pages=hold_stems, steps=args.steps,
               lr=args.lr, lambdas=args.lambdas, crop=args.crop, batch=args.batch, p_real=args.p_real, ema=args.ema,
               best_step=best[1])
    torch.save(alpha_net.checkpoint_dict(final, best[1], {"heldout": best[3], "score": best[0], "init_score": history[0]["score"]}, cfg), args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(history, indent=1, default=float))
    print(f"\nbest step {best[1]} score {best[0]:.2f} (init {history[0]['score']:.2f}) -> saved {args.out}")


if __name__ == "__main__":
    main()
