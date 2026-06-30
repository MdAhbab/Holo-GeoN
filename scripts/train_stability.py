"""
scripts/train_stability.py
==========================
Train the Siamese ΔΔG head on MegaScale pairs with the antisymmetric loss, on a
leakage-controlled split. Reports Pearson r, MAE, RMSE, and mean antisymmetry
violation on the held-out test split.

Ablation: pass --no-mech-track to disable E_mech (single-track) and measure the
contribution of the mechanistic edges (the difference between the two runs).

Mechanistic edges: --eps / --delta set the |dCAI| / |dmRNA| thresholds. Pass
--sweep to retrain briefly across a small grid of (eps=delta) values and record
the test r for each (a sensitivity probe; written to eps_delta_sweep.json).

Back-translation: --back-translate generates a usage-weighted proxy transcript
for sequences with no coding-DNA column, so the mechanistic track is defined even
when the stability data is protein-only (requires ViennaRNA for real folding).

Example
-------
python -m scripts.train_stability --splits data/splits_megascale.parquet \
    --cache cache/emb_150M --esm-model facebook/esm2_t30_150M_UR50D \
    --epochs 8 --batch 16 --out runs/stability_150M
"""
import os, sys, argparse, json
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


def train_eval(tr, te, store, edge_cfg, esm_dim, use_mech, epochs, batch, lr,
               alpha, device, save_to=None, desc="stability"):
    """One training run. Returns the best-by-r metrics dict. Saves a checkpoint
    if ``save_to`` (a directory) is given."""
    train_ds = StabilityPairDataset(tr, store, edge_cfg)
    test_ds = StabilityPairDataset(te, store, edge_cfg)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_pairs)
    test_ld = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate_pairs)

    model = HoloGNN(esm_dim=esm_dim, use_mech_track=use_mech, use_idr=False).to(device)
    crit = AntisymmetricLoss(alpha=alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best = None
    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for g_wt, g_mt, y in tqdm(train_ld, desc=f"{desc} ep {ep}/{epochs}"):
            f, r = model.stability(move(g_wt, device), move(g_mt, device))
            loss, _ = crit(f, r, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        metrics = evaluate(model, test_ld, device)
        metrics["epoch"] = ep; metrics["train_loss"] = running / max(len(train_ld), 1)
        print(f"  ep{ep}: r={metrics['pearson_r']:.4f} mae={metrics['mae']:.3f} "
              f"rmse={metrics['rmse']:.3f} viol={metrics['mean_antisymmetry_violation']:.3f}")
        if best is None or metrics["pearson_r"] > best["pearson_r"]:
            best = metrics
            if save_to:
                os.makedirs(save_to, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_to, "stability_best.pt"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--esm-model", default=DEFAULT_ESM)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="antisymmetry weight")
    ap.add_argument("--eps", type=float, default=0.10, help="|dCAI| threshold for E_mech")
    ap.add_argument("--delta", type=float, default=0.10, help="|dmRNA| threshold for E_mech")
    ap.add_argument("--no-mech-track", action="store_true", help="ablation: single-track")
    ap.add_argument("--back-translate", action="store_true",
                    help="use a usage-weighted proxy transcript when no coding DNA is present")
    ap.add_argument("--sweep", action="store_true",
                    help="after the main run, probe eps/delta sensitivity (Q9)")
    ap.add_argument("--sweep-values", default="0.05,0.10,0.20",
                    help="comma-separated eps=delta values for --sweep")
    ap.add_argument("--sweep-epochs", type=int, default=3, help="epochs per sweep point")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="runs/stability")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.splits)
    if args.max_rows:
        df = df.groupby("split", group_keys=False).head(args.max_rows)
    store = EmbeddingStore(args.cache)
    tr = df[df.split == "train"]; te = df[df.split == "test"]
    esm_dim = ESM_DIMS[args.esm_model]
    use_mech = not args.no_mech_track

    def edge_cfg(eps, delta):
        return {"geo_threshold": 0.5, "eps": eps, "delta": delta,
                "mech_max_degree": 8, "back_translate": args.back_translate}

    print(f"train={len(tr)} test={len(te)} | esm_dim={esm_dim} | mech_track={use_mech} | "
          f"eps={args.eps} delta={args.delta} back_translate={args.back_translate} | device={args.device}")

    best = train_eval(tr, te, store, edge_cfg(args.eps, args.delta), esm_dim, use_mech,
                      args.epochs, args.batch, args.lr, args.alpha, args.device,
                      save_to=args.out, desc="stability")
    json.dump({"best": best, "config": vars(args)},
              open(os.path.join(args.out, "stability_metrics.json"), "w"), indent=2)
    print(f"\nBEST: {best}\nSaved to {args.out}")

    # ----- eps/delta sensitivity sweep (Q9) -----
    if args.sweep and use_mech:
        vals = [float(x) for x in args.sweep_values.split(",") if x.strip()]
        print(f"\n[sweep] eps=delta in {vals} ({args.sweep_epochs} epochs each)")
        sweep = []
        for v in vals:
            m = train_eval(tr, te, store, edge_cfg(v, v), esm_dim, True,
                           args.sweep_epochs, args.batch, args.lr, args.alpha,
                           args.device, save_to=None, desc=f"sweep eps=delta={v}")
            sweep.append({"eps": v, "delta": v, "pearson_r": m["pearson_r"],
                          "mae": m["mae"], "n_test": m["n_test"]})
        out = {"sweep_epochs": args.sweep_epochs, "points": sweep}
        json.dump(out, open(os.path.join(args.out, "eps_delta_sweep.json"), "w"), indent=2)
        print("[sweep] " + " | ".join(f"({p['eps']}): r={p['pearson_r']:.4f}" for p in sweep))
        print(f"[sweep] saved {os.path.join(args.out, 'eps_delta_sweep.json')}")


if __name__ == "__main__":
    main()
