"""
scripts/precompute_embeddings.py
================================
Cache FROZEN ESM-2 per-residue embeddings + contact maps for every unique
sequence in the given parquet file(s). This is the one heavy GPU pass; after it,
training the GNN reads cached tensors and is light (fits a 16 GB GPU easily).

Example
-------
python -m scripts.precompute_embeddings \
    --data data/megascale_ddg_pairs.parquet data/clinvar_protein_pairs.parquet \
    --esm-model facebook/esm2_t30_150M_UR50D \
    --cache cache/emb_150M --max-length 256 --device cuda
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm

from src.data import detect_columns, seq_key
from src.backbone import EsmEncoder, ESM_DIMS, DEFAULT_ESM


def collect_sequences(paths):
    seqs = set()
    for p in paths:
        df = pd.read_parquet(p)
        col = detect_columns(df)
        for key in ("wt_seq", "mt_seq"):
            if col[key]:
                seqs.update(df[col[key]].astype(str).tolist())
    # drop empties / NaNs
    return sorted(s for s in seqs if isinstance(s, str) and 1 <= len(s) <= 5000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--esm-model", default=DEFAULT_ESM)
    ap.add_argument("--cache", default="cache/emb")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-seqs", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    seqs = collect_sequences(args.data)
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    todo = [s for s in seqs if not os.path.exists(os.path.join(args.cache, seq_key(s) + ".pt"))]
    print(f"Unique sequences: {len(seqs)} | to embed: {len(todo)} | model: {args.esm_model}")
    print(f"ESM dim: {ESM_DIMS.get(args.esm_model, '?')} | device: {args.device}")

    enc = EsmEncoder(args.esm_model, device=args.device)
    for s in tqdm(todo, desc="embedding"):
        try:
            emb = enc.embed(s, max_length=args.max_length)        # (L, d)
            cm = enc.contacts(s, max_length=args.max_length)      # (L, L)
            L = min(emb.shape[0], cm.shape[0])
            torch.save(
                {"emb": emb[:L].half(), "contacts": cm[:L, :L].half(), "seq": s[:L]},
                os.path.join(args.cache, seq_key(s) + ".pt"),
            )
        except Exception as e:  # keep going; report at end
            print(f"  ! failed ({type(e).__name__}) len={len(s)}: {e}")
    print(f"Done. Cache: {args.cache}")


if __name__ == "__main__":
    main()
