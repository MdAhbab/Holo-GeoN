"""
scripts/train_pathogenicity.py
==============================
Train the pathogenicity head on ClinVar protein pairs and report AUROC under both
regimes:
    - random split        (the same gene can appear in train and test)
    - gene-disjoint split  (whole genes held out)
plus GroupKFold (by gene) cross-validation for a confidence interval, the full
ROC curve, and class balance.

Example
-------
python -m scripts.train_pathogenicity --data data/clinvar_protein_pairs.parquet \
    --ddg data/clinvar_protein_pairs.parquet.ddg_w49.npy \
    --cache cache/emb_150M --esm-model facebook/esm2_t30_150M_UR50D \
    --epochs 6 --out runs/pathogenicity_150M
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.data import load_pathogenicity_pairs, VariantDataset, EmbeddingStore, collate_variants
from src.full_model import HoloGNN
from src.backbone import ESM_DIMS, DEFAULT_ESM
from src import splits as S


def move(g, dev):
    return {k: v.to(dev) for k, v in g.items()}


def train_eval(train_df, test_df, store, cfg, esm_dim, args):
    dev = args.device
    tr = DataLoader(VariantDataset(train_df, store, cfg), batch_size=args.batch,
                    shuffle=True, collate_fn=collate_variants)
    te = DataLoader(VariantDataset(test_df, store, cfg), batch_size=args.batch,
                    shuffle=False, collate_fn=collate_variants)
    model = HoloGNN(esm_dim=esm_dim, use_mech_track=not args.no_mech_track, use_idr=False).to(dev)
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        model.train()
        for g, y, _ in tr:
            logit = model.pathogenicity(move(g, dev)).squeeze(-1)
            loss = crit(logit, y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    scores, ys = [], []
    with torch.no_grad():
        for g, y, _ in te:
            s = torch.sigmoid(model.pathogenicity(move(g, dev)).squeeze(-1))
            scores.append(s.cpu().numpy()); ys.append(y.numpy())
    s = np.concatenate(scores); y = np.concatenate(ys)
    auroc = float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")
    auprc = float(average_precision_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y, s) if len(set(y.tolist())) > 1 else (np.array([]),)*3
    return {"auroc": auroc, "auprc": auprc, "n_test": int(len(y)),
            "pos_frac_test": float(np.mean(y))}, (fpr.tolist(), tpr.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ddg", default=None, help="optional cached ddg .npy (hybrid context)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--esm-model", default=DEFAULT_ESM)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-mech-track", action="store_true")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="runs/pathogenicity")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_pathogenicity_pairs(args.data, ddg_npy=args.ddg, max_rows=args.max_rows)
    store = EmbeddingStore(args.cache)
    cfg = {"geo_threshold": 0.5, "eps": 0.10, "delta": 0.10, "mech_max_degree": 8}
    esm_dim = ESM_DIMS[args.esm_model]
    print(f"variants={len(df)} | pos_frac={df.label.mean():.3f} | genes={df.group.nunique()}")

    results = {}

    # --- random split ---
    rnd = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n_tr = int(0.8 * len(rnd))
    m_rand, roc_rand = train_eval(rnd.iloc[:n_tr], rnd.iloc[n_tr:], store, cfg, esm_dim, args)
    results["random_split"] = m_rand
    print(f"RANDOM split   AUROC={m_rand['auroc']:.3f}  AUPRC={m_rand['auprc']:.3f}")

    # --- gene-disjoint split ---
    row_split = S.row_splits_from_groups(df.group.tolist(), ratios=(0.8, 0.0, 0.2), seed=42)
    df2 = df.copy(); df2["split"] = row_split
    m_gene, roc_gene = train_eval(df2[df2.split == "train"], df2[df2.split == "test"],
                                  store, cfg, esm_dim, args)
    results["gene_disjoint_split"] = m_gene
    print(f"GENE-DISJOINT  AUROC={m_gene['auroc']:.3f}  AUPRC={m_gene['auprc']:.3f}")

    # --- GroupKFold CV (by gene) for a confidence interval ---
    if args.cv_folds > 1 and df.group.nunique() >= args.cv_folds:
        gkf = GroupKFold(n_splits=args.cv_folds)
        aurocs = []
        for k, (tri, tei) in enumerate(gkf.split(df, df.label, groups=df.group)):
            m, _ = train_eval(df.iloc[tri], df.iloc[tei], store, cfg, esm_dim, args)
            aurocs.append(m["auroc"]); print(f"  fold {k+1}: AUROC={m['auroc']:.3f}")
        results["gene_disjoint_cv"] = {
            "folds": aurocs, "mean": float(np.nanmean(aurocs)), "std": float(np.nanstd(aurocs))}
        print(f"GENE-DISJOINT CV AUROC = {np.nanmean(aurocs):.3f} ± {np.nanstd(aurocs):.3f}")

    json.dump({"results": results, "roc_gene_disjoint": roc_gene, "config": vars(args)},
              open(os.path.join(args.out, "pathogenicity_metrics.json"), "w"), indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
