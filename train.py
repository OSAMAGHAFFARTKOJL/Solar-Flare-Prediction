"""Training and evaluation pipeline that produced rgfusion_checkpoint.pt.

Reproduce the reported numbers:

    python train.py --variant full --seeds 5

Model selection and the decision threshold use a validation split carved out of
training. The test set is scored once per run, at that threshold.
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data as D
from metrics import best_threshold, flare_metrics
from model import RGFusion, enable_dropout

MODES = ("mag", "euv", "concat", "full")
CORRUPTION_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


# Every knob for a run in one place, so a result can be traced back to the settings
# that produced it. The defaults here are the ones used for the reported numbers.
@dataclass
class Config:
    variant: str = "full"
    val_frac: float = 0.15
    img_size: int = 64
    flux_threshold: float = 1e-6
    euv_channels: tuple = D.EUV_CHANNELS
    max_train: int = 0
    neg_per_pos: float = 0.0
    width: int = 64
    dropout: float = 0.3
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    corruption_aware: bool = True
    corrupt_prob: float = 0.5
    mc_passes: int = 30
    num_workers: int = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_corruption(mag, euv, prob):
    """Degrade one modality per sample toward uniform noise by a random amount."""
    b, dev = mag.size(0), mag.device
    hit = torch.rand(b, device=dev) < prob
    pick_euv = torch.rand(b, device=dev) < 0.5
    level = torch.rand(b, device=dev)

    w_euv = ((hit & pick_euv).float() * level).view(b, 1, 1, 1)
    w_mag = ((hit & ~pick_euv).float() * level).view(b, 1, 1, 1)
    euv = (1 - w_euv) * euv + w_euv * torch.rand_like(euv)
    mag = (1 - w_mag) * mag + w_mag * torch.rand_like(mag)
    return mag, euv


@torch.no_grad()
def predict(model, loader, dev, corrupt_euv=0.0, seed=42):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ys, ps = [], []
    for mag, euv, y in loader:
        mag, euv = mag.to(dev), euv.to(dev)
        if corrupt_euv > 0:
            noise = torch.rand(euv.shape, generator=gen).to(dev)
            euv = (1 - corrupt_euv) * euv + corrupt_euv * noise
        ps.append(torch.sigmoid(model(mag, euv)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def mc_predict(model, loader, dev, passes):
    model.eval()
    enable_dropout(model)
    draws, ys = [], None
    for _ in range(passes):
        ps, yy = [], []
        for mag, euv, y in loader:
            ps.append(torch.sigmoid(model(mag.to(dev), euv.to(dev))).cpu().numpy())
            yy.append(y.numpy())
        draws.append(np.concatenate(ps))
        ys = np.concatenate(yy)
    stacked = np.stack(draws)
    return ys, stacked.mean(0), stacked.std(0)


def train_model(mode, cfg, train_loader, val_loader, pos_weight, dev, verbose=True):
    model = RGFusion(cfg.width, len(cfg.euv_channels), cfg.dropout).set_mode(mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    corrupt = cfg.corruption_aware and mode in ("concat", "full")

    best_tss, best_state = -1.0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for mag, euv, y in train_loader:
            mag, euv, y = mag.to(dev), euv.to(dev), y.to(dev)
            if corrupt:
                mag, euv = apply_corruption(mag, euv, cfg.corrupt_prob)
            opt.zero_grad()
            loss = criterion(model(mag, euv), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)

        y_val, p_val = predict(model, val_loader, dev)
        _, tss = best_threshold(y_val, p_val)
        if tss >= best_tss:
            best_tss = tss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % max(cfg.epochs // 5, 1) == 0 or epoch == 1):
            n = len(train_loader.dataset)
            print(f"  [{mode:6s}] epoch {epoch:2d}/{cfg.epochs} "
                  f"loss {total / n:.4f}  val TSS {tss:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def build_loaders(train_samples, val_samples, test_samples, cfg):
    mk = lambda s, tr: D.SDOSet(s, cfg.img_size, cfg.euv_channels, train=tr)
    train_loader = DataLoader(mk(train_samples, True), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(mk(val_samples, False), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(mk(test_samples, False), batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


def run(cfg, seeds, out_path="rgfusion_checkpoint.pt"):
    dev = device()
    print("device:", dev)

    data_dir = D.download(cfg.variant)
    print("data:", data_dir)

    pool = D.subsample(D.load_split(data_dir, "training", cfg.flux_threshold),
                       cfg.max_train, cfg.neg_per_pos)
    test_samples = D.load_split(data_dir, "test", cfg.flux_threshold)
    print("pool:", D.class_balance(pool), "test:", D.class_balance(test_samples))

    clean = {m: {"TSS": [], "HSS": [], "AUC": []} for m in MODES}
    robust = {m: [] for m in ("concat", "full")}
    keep_models, keep_results = None, None

    start = time.time()
    for seed in seeds:
        set_seed(seed)
        train_samples, val_samples = D.stratified_split(pool, cfg.val_frac, seed=seed)
        train_loader, val_loader, test_loader = build_loaders(
            train_samples, val_samples, test_samples, cfg)
        pw = D.pos_weight(train_samples)

        models, results = {}, {}
        for mode in MODES:
            model = train_model(mode, cfg, train_loader, val_loader, pw, dev,
                                verbose=(seed == seeds[0]))
            y_val, p_val = predict(model, val_loader, dev)
            threshold, _ = best_threshold(y_val, p_val)
            y_test, p_test = predict(model, test_loader, dev)
            scored = flare_metrics(y_test, p_test, threshold)
            scored["threshold"] = threshold

            for k in ("TSS", "HSS", "AUC"):
                clean[mode][k].append(scored[k])
            models[mode], results[mode] = model, scored

            if mode in robust:
                robust[mode].append([
                    flare_metrics(*predict(model, test_loader, dev, corrupt_euv=lv),
                                  threshold)["TSS"]
                    for lv in CORRUPTION_LEVELS
                ])

        print(f"seed {seed}: " + "  ".join(
            f"{m}={results[m]['TSS']:.3f}" for m in MODES) +
            f"   ({time.time() - start:.0f}s)")

        if keep_models is None:
            keep_models, keep_results = models, results

    print("\nclean test TSS, mean +/- std")
    for mode in MODES:
        v = np.array(clean[mode]["TSS"])
        print(f"  {mode:7s} {v.mean():.3f} +/- {v.std():.3f}   worst {v.min():.3f}")

    print("\nTSS vs EUV corruption")
    for mode in ("concat", "full"):
        arr = np.array(robust[mode])
        print(f"  {mode:7s} " + "  ".join(f"{x:.3f}" for x in arr.mean(0)))

    payload = {
        "modes": list(keep_models.keys()),
        "state_dicts": {k: {n: v.detach().cpu() for n, v in m.state_dict().items()}
                        for k, m in keep_models.items()},
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
        "n_euv": len(cfg.euv_channels),
        "single_run_results": {m: {k: float(v) for k, v in r.items()}
                               for m, r in keep_results.items()},
        "multiseed_clean": clean,
        "multiseed_robust": robust,
        "corruption_levels": list(CORRUPTION_LEVELS),
        "seeds": list(seeds),
    }
    torch.save(payload, out_path)
    print("\nsaved", out_path)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "example"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--no-corruption", action="store_true")
    ap.add_argument("--out", default="rgfusion_checkpoint.pt")
    args = ap.parse_args()

    cfg = Config(variant=args.variant, epochs=args.epochs,
                 corruption_aware=not args.no_corruption)
    run(cfg, list(range(args.seeds)), args.out)


if __name__ == "__main__":
    main()
