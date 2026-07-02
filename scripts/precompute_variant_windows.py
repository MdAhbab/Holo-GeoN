"""
scripts/precompute_variant_windows.py
=====================================
Cache FROZEN ESM-2 embeddings + contact maps for MUTATION-CENTERED WINDOWS of
variant (ClinVar) sequences. Unlike precompute_embeddings.py (which head-truncates
each sequence to the first --max-length residues and therefore drops the mutation
for any variant beyond that point), this embeds a --window-length slice centered
on the mutated residue, so the variant is always inside the ESM context — at the
same embedding/cache size. Fixes the gene-disjoint pathogenicity collapse caused
by ~64% of ClinVar mutations sitting past residue 256.

The window is keyed by its subsequence (md5), matching src.data.WindowedVariant-
Dataset / add_window_column, so training loads the same tensors deterministically.

Example
-------
python -m scripts.precompute_variant_windows \
    --data data/clinvar_protein_pairs.parquet \
    --esm-model facebook/esm2_t30_150M_UR50D \
    --cache cache/emb_win256_150M --window 256 --device cuda
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm

from src.data import detect_columns, seq_key, mutation_position, variant_window
from src.backbone import EsmEncoder, ESM_DIMS, DEFAULT_ESM


def collect_windows(paths, window):
    """Unique mutation-centered windows across the given variant parquet files."""
    wins = set()
    for p in paths:
        df = pd.read_parquet(p)
        col = detect_columns(df)
        wt_c, mt_c = col["wt_seq"], col["mt_seq"]
        if not wt_c or not mt_c:
            print(f"  ! {p}: could not detect wt_seq/mt_seq; skipping")
            continue
        for wt, mt in zip(df[wt_c].astype(str), df[mt_c].astype(str)):
            pos = mutation_position(wt, mt)
            if pos is None or pos < 0 or pos >= len(mt):
                pos = 0  # fallback (multi-substitution / length mismatch) -> head window
            win = variant_window(mt, pos, window)[0]
            if 1 <= len(win) <= 5000:
                wins.add(win)
    return sorted(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--esm-model", default=DEFAULT_ESM)
    ap.add_argument("--cache", default="cache/emb_windows")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-seqs", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    wins = collect_windows(args.data, args.window)
    if args.max_seqs:
        wins = wins[: args.max_seqs]
    todo = [w for w in wins if not os.path.exists(os.path.join(args.cache, seq_key(w) + ".pt"))]
    print(f"Unique windows: {len(wins)} | to embed: {len(todo)} | window: {args.window} | model: {args.esm_model}")
    print(f"ESM dim: {ESM_DIMS.get(args.esm_model, '?')} | device: {args.device}")

    enc = EsmEncoder(args.esm_model, device=args.device)
    for w in tqdm(todo, desc="embedding windows"):
        try:
            emb = enc.embed(w, max_length=args.window)
            cm = enc.contacts(w, max_length=args.window)
            L = min(emb.shape[0], cm.shape[0])
            dst = os.path.join(args.cache, seq_key(w) + ".pt")
            tmp = dst + ".tmp"
            torch.save(
                {"emb": emb[:L].half(), "contacts": cm[:L, :L].half(), "seq": w[:L]},
                tmp,
            )
            os.replace(tmp, dst)  # atomic
        except Exception as e:
            print(f"  ! failed ({type(e).__name__}) len={len(w)}: {e}")
    print(f"Done. Cache: {args.cache}")


if __name__ == "__main__":
    main()
