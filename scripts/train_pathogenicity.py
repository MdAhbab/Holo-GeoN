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

from src.data import (load_pathogenicity_pairs, VariantDataset,
                      EmbeddingStore, collate_variants)
from src.full_model import HoloGNN
from src.backbone import ESM_DIMS, DEFAULT_ESM
from src import splits as S


def move(g, dev):
    return {k: v.to(dev) for k, v in g.items()}


def train_eval(train_df, test_df, store, cfg, esm_dim, args, tag="run"):
    dev = args.device
    tr = DataLoader(VariantDataset(train_df, store, cfg), batch_size=args.batch,
                    shuffle=True, collate_fn=collate_variants)
    te = DataLoader(VariantDataset(test_df, store, cfg), batch_size=args.batch,
                    shuffle=False, collate_fn=collate_variants)
    model = HoloGNN(esm_dim=esm_dim, use_mech_track=not args.no_mech_track, use_idr=False).to(dev)
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # per-epoch checkpoint so a crash mid sub-run resumes WITHIN this sub-run
    start_ep = 0
    ckpt = os.path.join(args.out, f"last_{tag}.pt")
    if os.path.exists(ckpt):
        try:
            st = torch.load(ckpt, map_location=dev)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start_ep = int(st["epoch"])
            print(f"    [resume] pathogenicity[{tag}]: from epoch {start_ep}/{args.epochs}")
        except Exception as e:
            print(f"    [resume] pathogenicity[{tag}]: load failed ({e}); training fresh"); start_ep = 0
    for ep in range(start_ep, args.epochs):
        model.train()
        for g, y, _ in tr:
            logit = model.pathogenicity(move(g, dev)).squeeze(-1)
            loss = crit(logit, y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        try:  # guarded; must never break training
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1}, ckpt)
        except Exception as e:
            print(f"    [ckpt] pathogenicity[{tag}] save failed at ep{ep+1} ({e})")
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
    try:  # sub-run finished successfully — drop its per-epoch checkpoint
        if os.path.exists(ckpt):
            os.remove(ckpt)
    except Exception:
        pass
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
    # Drop variants whose embedding was never cached (proteins >5000 aa that
    # precompute skips) — the dataset would otherwise crash on a missing .pt.
    before = len(df)
    df = df[df["mt_seq"].map(store.has)].reset_index(drop=True)
    if len(df) < before:
        print(f"[filter] dropped {before - len(df)} / {before} variants with no cached embedding; {len(df)} remain")
    cfg = {"geo_threshold": 0.5, "eps": 0.10, "delta": 0.10, "mech_max_degree": 8}
    esm_dim = ESM_DIMS[args.esm_model]
    print(f"variants={len(df)} | pos_frac={df.label.mean():.3f} | genes={df.group.nunique()}")

    # Resumable partial state: survives an interruption between sub-runs so a
    # re-run continues at the next unfinished split/fold instead of from scratch.
    partial = os.path.join(args.out, "pathogenicity_partial.json")
    state = {}
    if os.path.exists(partial):
        try:
            state = json.load(open(partial))
            print(f"[resume] partial pathogenicity state: done={list(state.get('results', {}).keys())} "
                  f"cv_folds_done={len(state.get('cv_aurocs', []))}")
        except Exception as e:
            print(f"[resume] partial load failed ({e}); starting fresh"); state = {}
    results = state.get("results", {})
    roc_gene = tuple(state["roc_gene"]) if state.get("roc_gene") else None
    cv_aurocs = list(state.get("cv_aurocs", []))

    def save_partial():
        try:
            json.dump({"results": results, "roc_gene": list(roc_gene) if roc_gene else None,
                       "cv_aurocs": cv_aurocs}, open(partial, "w"), indent=2)
        except Exception as e:
            print(f"[ckpt] partial save failed: {e}")

    # --- random split ---
    if "random_split" not in results:
        rnd = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        n_tr = int(0.8 * len(rnd))
        m_rand, _ = train_eval(rnd.iloc[:n_tr], rnd.iloc[n_tr:], store, cfg, esm_dim, args, tag="random")
        results["random_split"] = m_rand; save_partial()
        print(f"RANDOM split   AUROC={m_rand['auroc']:.3f}  AUPRC={m_rand['auprc']:.3f}")
    else:
        print(f"[skip] random_split (done): AUROC={results['random_split']['auroc']:.3f}")

    # --- gene-disjoint split ---
    if "gene_disjoint_split" not in results or roc_gene is None:
        row_split = S.row_splits_from_groups(df.group.tolist(), ratios=(0.8, 0.0, 0.2), seed=42)
        df2 = df.copy(); df2["split"] = row_split
        m_gene, roc_gene = train_eval(df2[df2.split == "train"], df2[df2.split == "test"],
                                      store, cfg, esm_dim, args, tag="gene")
        results["gene_disjoint_split"] = m_gene; save_partial()
        print(f"GENE-DISJOINT  AUROC={m_gene['auroc']:.3f}  AUPRC={m_gene['auprc']:.3f}")
    else:
        print(f"[skip] gene_disjoint_split (done): AUROC={results['gene_disjoint_split']['auroc']:.3f}")

    # --- GroupKFold CV (by gene) for a confidence interval; resumable fold-by-fold ---
    if args.cv_folds > 1 and df.group.nunique() >= args.cv_folds:
        gkf = GroupKFold(n_splits=args.cv_folds)
        folds = list(gkf.split(df, df.label, groups=df.group))
        for k, (tri, tei) in enumerate(folds):
            if k < len(cv_aurocs):
                print(f"  [skip] fold {k+1}: AUROC={cv_aurocs[k]:.3f}"); continue
            m, _ = train_eval(df.iloc[tri], df.iloc[tei], store, cfg, esm_dim, args, tag=f"cv{k}")
            cv_aurocs.append(m["auroc"]); save_partial()
            print(f"  fold {k+1}: AUROC={m['auroc']:.3f}")
        results["gene_disjoint_cv"] = {
            "folds": cv_aurocs, "mean": float(np.nanmean(cv_aurocs)), "std": float(np.nanstd(cv_aurocs))}
        print(f"GENE-DISJOINT CV AUROC = {np.nanmean(cv_aurocs):.3f} ± {np.nanstd(cv_aurocs):.3f}")

    json.dump({"results": results, "roc_gene_disjoint": roc_gene, "config": vars(args)},
              open(os.path.join(args.out, "pathogenicity_metrics.json"), "w"), indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
