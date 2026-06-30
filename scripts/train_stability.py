"""
scripts/train_stability.py
==========================
Train the Siamese ΔΔG head on MegaScale pairs with the antisymmetric loss, on a
leakage-controlled split. Reports Pearson r, MAE, RMSE, and mean antisymmetry
violation on the held-out test split.

Ablation: pass --no-mech-track to disable E_mech (single-track) and measure the
contribution of the mechanistic edges (the difference between the two runs).

Example
-------
python -m scripts.train_stability --splits data/splits_megascale.parquet \
    --cache cache/emb_150M --esm-model facebook/esm2_t30_150M_UR50D \
    --epochs 8 --batch 16 --out runs/stability_150M
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import StabilityPairDataset, EmbeddingStore, collate_pairs
from src.full_model import HoloGNN
from src.loss import AntisymmetricLoss
from src.backbone import ESM_DIMS, DEFAULT_ESM


def move(g, dev):
    return {k: v.to(dev) for k, v in g.items()}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    preds, gts, viol = [], [], []
    for g_wt, g_mt, y in loader:
        f, r = model.stability(move(g_wt, dev), move(g_mt, dev))
        f = f.squeeze(-1).cpu().numpy(); r = r.squeeze(-1).cpu().numpy()
        preds.append(f); gts.append(y.numpy()); viol.append(np.abs(f + r))
    p = np.concatenate(preds); t = np.concatenate(gts); v = np.concatenate(viol)
    pear = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else float("nan")
    mae = float(np.mean(np.abs(p - t))); rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return {"pearson_r": pear, "mae": mae, "rmse": rmse,
            "mean_antisymmetry_violation": float(np.mean(v)), "n_test": int(len(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--esm-model", default=DEFAULT_ESM)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="antisymmetry weight")
    ap.add_argument("--no-mech-track", action="store_true", help="ablation: single-track")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="runs/stability")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.splits)
    if args.max_rows:
        df = df.groupby("split", group_keys=False).head(args.max_rows)
    store = EmbeddingStore(args.cache)
    cfg = {"geo_threshold": 0.5, "eps": 0.10, "delta": 0.10, "mech_max_degree": 8}

    tr = df[df.split == "train"]; te = df[df.split == "test"]
    train_ds = StabilityPairDataset(tr, store, cfg)
    test_ds = StabilityPairDataset(te, store, cfg)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_pairs)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pairs)

    esm_dim = ESM_DIMS[args.esm_model]
    model = HoloGNN(esm_dim=esm_dim, use_mech_track=not args.no_mech_track, use_idr=False).to(args.device)
    crit = AntisymmetricLoss(alpha=args.alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"train={len(train_ds)} test={len(test_ds)} | esm_dim={esm_dim} | "
          f"mech_track={not args.no_mech_track} | device={args.device}")

    best = None
    for ep in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for g_wt, g_mt, y in tqdm(train_ld, desc=f"epoch {ep}/{args.epochs}"):
            f, r = model.stability(move(g_wt, args.device), move(g_mt, args.device))
            loss, _ = crit(f, r, y.to(args.device))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        metrics = evaluate(model, test_ld, args.device)
        metrics["epoch"] = ep; metrics["train_loss"] = running / max(len(train_ld), 1)
        print(f"  ep{ep}: r={metrics['pearson_r']:.4f} mae={metrics['mae']:.3f} "
              f"rmse={metrics['rmse']:.3f} viol={metrics['mean_antisymmetry_violation']:.3f}")
        if best is None or metrics["pearson_r"] > best["pearson_r"]:
            best = metrics
            torch.save(model.state_dict(), os.path.join(args.out, "stability_best.pt"))

    json.dump({"best": best, "config": vars(args)},
              open(os.path.join(args.out, "stability_metrics.json"), "w"), indent=2)
    print(f"\nBEST: {best}\nSaved to {args.out}")


if __name__ == "__main__":
    main()
