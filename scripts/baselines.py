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

    scores, ddgs = [], []
    with torch.no_grad():
        for _, r in tqdm(df.iterrows(), total=len(df), desc="esm zero-shot"):
            wt, mt = str(r["wt_seq"]), str(r["mt_seq"])
            pos = mutated_position(wt, mt)
            if pos is None:
                continue
            enc = tok(wt, return_tensors="pt", truncation=True, max_length=1024).to(args.device)
            logits = mlm(**enc).logits[0]                      # (L+2, V)
            lp = torch.log_softmax(logits[pos + 1], dim=-1)    # +1 for CLS
            wt_id = tok.convert_tokens_to_ids(wt[pos])
            mt_id = tok.convert_tokens_to_ids(mt[pos])
            scores.append(float(lp[mt_id] - lp[wt_id]))
            ddgs.append(float(r["target"]))

    s = np.array(scores); d = np.array(ddgs)
    r = float(np.corrcoef(s, d)[0, 1]) if len(s) > 1 else float("nan")
    out = {"baseline": "esm_zeroshot_wt_marginal", "esm_model": args.esm_model,
           "n": int(len(s)), "pearson_r": r, "abs_pearson_r": abs(r)}
    json.dump(out, open(args.out, "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
