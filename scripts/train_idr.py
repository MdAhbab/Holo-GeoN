"""
scripts/train_idr.py
====================
Train the IDR ensemble head to predict the radius of gyration R_g.

The target is a polymer-physics prior (src/idr_prior.py; Flory scaling with a
sequence-dependent exponent) — a learnability check that the head can reproduce a
cited physics estimate from the learned representation. Substitute experimental /
simulation labels (e.g. PED, ALBATROSS) for an experimentally-grounded version.

Example
-------
python -m scripts.train_idr --data data/clinvar_protein_pairs.parquet \
    --cache cache/emb_150M --esm-model facebook/esm2_t30_150M_UR50D \
    --epochs 5 --out runs/idr_150M
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data import detect_columns, EmbeddingStore, build_graph, collate_graphs
from src.full_model import HoloGNN
from src.loss import gaussian_nll
from src.idr_prior import rg_prior
from src.backbone import ESM_DIMS, DEFAULT_ESM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--esm-model", default=DEFAULT_ESM)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-seqs", type=int, default=20000)
    ap.add_argument("--out", default="runs/idr")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    df = pd.read_parquet(args.data)
    col = detect_columns(df)
    seqs = set()
    for k in ("wt_seq", "mt_seq"):
        if col[k]:
            seqs.update(df[col[k]].astype(str).tolist())
    store = EmbeddingStore(args.cache)
    seqs = [s for s in seqs if store.has(s)][: args.max_seqs]
    targets = np.array([rg_prior(s) for s in seqs], dtype=np.float32)
    print(f"IDR training sequences: {len(seqs)} | Rg prior range "
          f"[{targets.min():.1f}, {targets.max():.1f}] Å")

    cfg = {"geo_threshold": 0.5, "eps": 0.10, "delta": 0.10, "mech_max_degree": 8}
    esm_dim = ESM_DIMS[args.esm_model]
    model = HoloGNN(esm_dim=esm_dim, use_idr=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    idx = np.arange(len(seqs)); n_tr = int(0.9 * len(idx))
    rng = np.random.default_rng(0); rng.shuffle(idx)
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

    def batches(order):
        for i in range(0, len(order), args.batch):
            sel = order[i:i + args.batch]
            graphs = []
            for j in sel:
                emb, cm, sq = store.get(seqs[j])
                graphs.append(build_graph(sq, emb, cm, None, cfg))
            g = collate_graphs(graphs)
            y = torch.tensor(targets[sel], dtype=torch.float32)
            yield g, y

    start_ep = 1; mae = float("nan")
    ckpt = os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt):  # resume from a per-epoch checkpoint after an interruption
        try:
            st = torch.load(ckpt, map_location=dev)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            start_ep = int(st["epoch"]) + 1; mae = float(st.get("mae", float("nan")))
            print(f"  [resume] idr: continuing from epoch {st['epoch']} (Rg MAE={mae:.2f} Å)")
        except Exception as e:
            print(f"  [resume] idr: checkpoint load failed ({e}); training fresh")
            start_ep = 1
    for ep in range(start_ep, args.epochs + 1):
        model.train(); rng.shuffle(tr_idx); tot = 0.0
        for g, y in tqdm(list(batches(tr_idx)), desc=f"epoch {ep}/{args.epochs}"):
            g = {k: v.to(dev) for k, v in g.items()}
            mu, sigma = model.idr(g)
            loss = gaussian_nll(mu, sigma, y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        # eval MAE vs prior
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for g, y in batches(te_idx):
                g = {k: v.to(dev) for k, v in g.items()}
                mu, _ = model.idr(g)
                preds.append(mu.squeeze(-1).cpu().numpy()); gts.append(y.numpy())
        p = np.concatenate(preds); t = np.concatenate(gts)
        mae = float(np.mean(np.abs(p - t)))
        print(f"  ep{ep}: nll/batch={tot/max(len(tr_idx)//args.batch,1):.3f} | Rg MAE vs prior={mae:.2f} Å")
        try:  # per-epoch resume checkpoint (guarded)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep, "mae": mae}, ckpt)
        except Exception as e:
            print(f"  [ckpt] idr: save failed at ep{ep} ({e})")

    torch.save(model.state_dict(), os.path.join(args.out, "idr_best.pt"))
    json.dump({"rg_mae_vs_prior": mae, "note": "target = disclosed polymer-physics prior, not experimental",
               "config": vars(args)}, open(os.path.join(args.out, "idr_metrics.json"), "w"), indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
