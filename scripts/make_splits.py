"""
scripts/make_splits.py
======================
Write leakage-controlled train/val/test labels.

Stability  : group by parent protein (wt_seq), optionally refined to family level
             with MMseqs2 clustering at --identity (default 0.30).
Pathogenicity: group by gene (gene-disjoint) by default — the gene-disjoint
             evaluation regime (vs. the random split).

Example
-------
python -m scripts.make_splits --data data/megascale_ddg_pairs.parquet \
    --kind stability --method mmseqs --identity 0.30 --out data/splits_megascale.parquet
python -m scripts.make_splits --data data/clinvar_protein_pairs.parquet \
    --kind pathogenicity --method group --out data/splits_clinvar.parquet
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_stability_pairs, load_pathogenicity_pairs
from src import splits as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", choices=["stability", "pathogenicity"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["mmseqs", "group"], default="group")
    ap.add_argument("--identity", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.kind == "stability":
        df = load_stability_pairs(args.data)
    else:
        df = load_pathogenicity_pairs(args.data)

    groups = df["group"].tolist()

    if args.method == "mmseqs":
        # cluster the unique PARENT sequences (wt_seq) at strict identity
        uniq = {S.group_key_for_sequence(s): s for s in df["wt_seq"].unique()}
        if not S.mmseqs_available():
            print("⚠️  mmseqs not on PATH — falling back to parent-sequence grouping.")
        else:
            clusters = S.cluster_with_mmseqs(uniq, min_seq_id=args.identity, coverage=args.coverage)
            # map each row's wt_seq -> its cluster representative
            wt_to_rep = {seq: clusters.get(S.group_key_for_sequence(seq), S.group_key_for_sequence(seq))
                         for seq in df["wt_seq"].unique()}
            groups = [wt_to_rep[s] for s in df["wt_seq"]]

    row_splits = S.row_splits_from_groups(groups, ratios=tuple(args.ratios), seed=args.seed)
    assert S.verify_no_leakage(groups, row_splits), "leakage detected!"
    df = df.copy()
    df["split"] = row_splits
    df["group_used"] = groups
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    counts = df["split"].value_counts().to_dict()
    print(f"Wrote {args.out} | rows={len(df)} | splits={counts} | "
          f"unique groups={len(set(groups))} | method={args.method}")


if __name__ == "__main__":
    main()
