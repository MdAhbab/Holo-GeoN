"""
scripts/baselines.py
====================
Training-free baseline: ESM zero-shot ΔΔG scoring (wt-marginal, Meier et al. 2021).

For each single-point mutation we score
    s = log p(mt_aa | wt context) - log p(wt_aa | wt context)
at the mutated position, using the ESM masked-LM head. We then correlate s with
the experimental ΔΔG on the SAME held-out test split used for HoloGNN, so the two
numbers are directly comparable.

Note on sign: a higher ESM score means "more tolerated", which corresponds to a
LOWER destabilising ΔΔG, so the expected Pearson correlation is negative; we
report both r and |r|.

Example
-------
python -m scripts.baselines --splits data/splits_megascale.parquet \
    --esm-model facebook/esm2_t30_150M_UR50D --split test --device cuda
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def mutated_position(wt: str, mt: str):
    if len(wt) != len(mt):
        return None
    diffs = [i for i in range(len(wt)) if wt[i] != mt[i]]
    return diffs[0] if len(diffs) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--esm-model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/baseline_esm_zeroshot.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer, EsmForMaskedLM
    tok = AutoTokenizer.from_pretrained(args.esm_model)
    mlm = EsmForMaskedLM.from_pretrained(args.esm_model).to(args.device).eval()

    df = pd.read_parquet(args.splits)
    if "split" in df.columns:
        df = df[df.split == args.split]
    if args.max_rows:
        df = df.iloc[: args.max_rows]
    df = df.reset_index(drop=True)

    # Incremental checkpoint: scoring 11k+ rows one at a time is slow, so save
    # progress periodically and resume from the last row after an interruption.
    ckpt = args.out + ".partial.npz"
    start_i = 0
    scores, ddgs = [], []
    if os.path.exists(ckpt):
        try:
            z = np.load(ckpt)
            scores = z["scores"].tolist(); ddgs = z["ddgs"].tolist(); start_i = int(z["next_i"])
            print(f"[resume] baseline: from row {start_i}/{len(df)} ({len(scores)} scored)")
        except Exception as e:
            print(f"[resume] baseline: checkpoint load failed ({e}); starting fresh")
            start_i = 0; scores, ddgs = [], []

    def save_ckpt(next_i):
        try:
            tmp = ckpt + ".tmp.npz"  # keep a .npz suffix so np.savez doesn't rename it
            np.savez(tmp, scores=np.array(scores), ddgs=np.array(ddgs), next_i=next_i)
            os.replace(tmp, ckpt)  # atomic
        except Exception as e:
            print(f"[ckpt] baseline save failed: {e}")

    with torch.no_grad():
        for i in tqdm(range(start_i, len(df)), total=len(df) - start_i, desc="esm zero-shot"):
            r = df.iloc[i]
            wt, mt = str(r["wt_seq"]), str(r["mt_seq"])
            pos = mutated_position(wt, mt)
            if pos is not None:
                enc = tok(wt, return_tensors="pt", truncation=True, max_length=1024).to(args.device)
                logits = mlm(**enc).logits[0]                      # (L+2, V)
                lp = torch.log_softmax(logits[pos + 1], dim=-1)    # +1 for CLS
                wt_id = tok.convert_tokens_to_ids(wt[pos])
                mt_id = tok.convert_tokens_to_ids(mt[pos])
                scores.append(float(lp[mt_id] - lp[wt_id]))
                ddgs.append(float(r["target"]))
            if (i + 1) % 1000 == 0:
                save_ckpt(i + 1)

    s = np.array(scores); d = np.array(ddgs)
    r = float(np.corrcoef(s, d)[0, 1]) if len(s) > 1 else float("nan")
    out = {"baseline": "esm_zeroshot_wt_marginal", "esm_model": args.esm_model,
           "n": int(len(s)), "pearson_r": r, "abs_pearson_r": abs(r)}
    json.dump(out, open(args.out, "w"), indent=2)
    try:  # done — drop the partial checkpoint
        if os.path.exists(ckpt):
            os.remove(ckpt)
    except Exception:
        pass
    print(out)


if __name__ == "__main__":
    main()
