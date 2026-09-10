"""
scripts/build_megascale_table.py
================================
Build ``data/megascale_ddg_pairs.parquet`` from the raw Tsuboyama et al. (2023)
mega-scale cDNA proteolysis release.

The repository's pipeline reads normalised parquet tables; the raw release under
``dataset/mega_scale_cdna/`` is a per-measurement CSV. This is the join between
the two, and it is the only place the raw-column semantics are interpreted.

Source
------
``Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`` — the
curated table, the one carrying ``ddG_ML``. ``Dataset1`` is deliberately not
used: it has no ``aa_seq`` / ``mut_type`` / ``ddG_ML`` columns, so it cannot be
turned into (wild-type, mutant, ddG) triples without refitting the model the
authors already fitted.

What one output row is
----------------------
One **single amino-acid substitution** of one wild-type domain:

    wt_seq   the WT domain's protein sequence      (this is the parent encoded)
    mt_seq   the same sequence with one residue changed
    ddG      ddG_ML, in kcal/mol
    wt_dna   the WT domain's real coding sequence, in frame with wt_seq
    parent   WT_name — the construct the row is a variant of

Decisions that are not mechanical, and why
------------------------------------------
**Sign.** Holo-GeoN's internal convention is ``ddG = dG_mut - dG_wt``, so
POSITIVE is STABILISING (``src/data.DDG_CONVENTION``). The source's convention
is not documented in the CSV, so it is not assumed: this script checks it
against the release's own ``Stabilizing_mut`` boolean and refuses to write the
table if the check fails.

The check is an IMPLICATION, not an equality. ``Stabilizing_mut`` is a
confidence threshold rather than a sign bit — the smallest ``ddG_ML`` it flags
on the shipped file is +1.00 kcal/mol, so only 2.4% of positive-ddG rows carry
it — but every flagged row must have ``ddG_ML > 0`` if positive means
stabilising. All 2112 of them do, and the mean over all substitutions is
negative (-0.758), as it must be when random substitutions are destabilising on
average. So the default ``ddg_sign: stabilizing_positive`` in
``configs/data.yaml`` is correct and no override is needed. Correlation metrics
are invariant to this flip and would never reveal it, which is why it is
asserted here rather than eyeballed.

**Coding sequence, and its reading frame.** ``dna_seq`` encodes ``aa_seq_full``
— the whole expressed construct, including the constant ``SAGGSAGG...`` linkers
— while the domain actually scored is ``aa_seq``, which sits at some offset
inside it. ``src/features.py`` reads codon ``i`` as belonging to residue ``i``,
so handing it the unsliced CDS would silently shift every codon by the linker
length and make the CAI and mRNA-folding channels wrong rather than absent. The
CDS is therefore sliced to the domain's own frame and **verified by translating
it back** to ``aa_seq``; a row whose CDS does not translate to its protein is
emitted with no CDS, so ``mech_valid`` is 0 for it. Silent misalignment is worse
than an inactive stream.

**Only single substitutions.** Insertions, deletions and multi-mutants are
dropped: ``src/windowing.mutation_position`` accepts one substituted residue,
and everything downstream — the site-anchored readout, the substitution head —
is defined for exactly that.

**Per-measurement uncertainty.** ``ddg_std`` is propagated from the release's
own 95% confidence intervals on dG. ``deltaG_95CI`` is the FULL width
(``high - low``; verified, matches on 100% of rows), so a row's dG standard
error is ``width / (2 * 1.96)``, and because ``ddG = dG_mut - dG_wt`` the two
combine in quadrature.

This matters more than it looks. ``src/heads.py`` labels a mutation neutral when
``|ddG| <= sigma``, and with no ``sigma`` column that band falls back to the
``three_state_sigma: 0.5`` constant in ``configs/run.yaml``. On this data the
constant sets the class of **46.0%** of all rows - not the "roughly a fifth"
that config comment estimates - and squeezes the stabilising class to 4.2%,
which is the class the swap-equivariant head's only evidence rests on. The
propagated interval gives 74.1% / 7.3% / 18.7%.

Caveat, stated because the number is easy to over-trust: this is the confidence
interval of the authors' ML FIT, so it measures fit precision, not
reproducibility across experimental replicates. Real measurement error is
probably larger, which makes this band, if anything, too narrow. It is used
because it is measured per row rather than invented; ``metrics.json`` records
the band and a sensitivity table alongside it. Dropping the ``ddg_std`` column
reverts to the constant.

**Duplicate measurements** of the same (construct, substitution) are averaged,
and the count is reported.

Usage
-----
python -m scripts.build_megascale_table \
    --source dataset/mega_scale_cdna/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --out data/megascale_ddg_pairs.parquet
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

SUB_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

DEFAULT_SOURCE = os.path.join(
    "dataset", "mega_scale_cdna", "Processed_K50_dG_datasets",
    "Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
)

USECOLS = ["aa_seq", "aa_seq_full", "dna_seq", "mut_type", "WT_name",
           "ddG_ML", "Stabilizing_mut", "deltaG_95CI"]

#: Half-width of a 95% interval, in standard deviations.
Z95 = 1.959963985

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S", "CCT": "P", "CCC": "P",
    "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R",
    "CGG": "R", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G",
    "GGG": "G", "TAA": "*", "TAG": "*", "TGA": "*",
}


def translate(dna: str) -> str:
    dna = dna.upper().replace("U", "T")
    return "".join(CODON_TABLE.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3))


def domain_cds(aa_seq: str, aa_seq_full: str, dna_seq: str) -> str | None:
    """
    Slice the construct's CDS down to the codons that encode ``aa_seq``.

    Returns None when the slice does not translate back to the domain, so the
    mechanistic channels stay masked rather than being fed a shifted frame.
    """
    if not isinstance(dna_seq, str) or not isinstance(aa_seq_full, str):
        return None
    offset = aa_seq_full.find(aa_seq)
    if offset < 0:
        return None
    cds = dna_seq[offset * 3: (offset + len(aa_seq)) * 3]
    if len(cds) != 3 * len(aa_seq) or translate(cds) != aa_seq:
        return None
    return cds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=os.path.join("data", "megascale_ddg_pairs.parquet"))
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    wt: dict[str, dict] = {}
    muts: list[tuple[str, str, float, bool, float]] = []
    rows_in = 0
    dropped: dict[str, int] = {}

    def drop(reason: str, n: int = 1) -> None:
        if n:
            dropped[reason] = dropped.get(reason, 0) + n

    print(f"[build] reading {args.source}")
    reader = pd.read_csv(args.source, usecols=USECOLS, chunksize=args.chunk,
                         low_memory=False)
    for k, chunk in enumerate(reader):
        rows_in += len(chunk)
        mut_type = chunk["mut_type"].astype(str)

        # Wild types: one record per construct, carrying the sequence and CDS.
        for r in chunk[mut_type == "wt"].itertuples(index=False):
            name = str(r.WT_name)
            if name in wt or not isinstance(r.aa_seq, str) or not r.aa_seq:
                continue
            wt[name] = {"aa_seq": r.aa_seq.upper(),
                        "ci": pd.to_numeric(r.deltaG_95CI, errors="coerce"),
                        "cds": domain_cds(r.aa_seq.upper(),
                                          str(r.aa_seq_full).upper(),
                                          str(r.dna_seq))}

        # Single substitutions.
        sub = chunk[mut_type.str.fullmatch(SUB_RE.pattern, na=False)].copy()
        drop("not_single_substitution", len(chunk) - len(sub)
             - int((mut_type == "wt").sum()))
        sub["ddg"] = pd.to_numeric(sub["ddG_ML"], errors="coerce")
        sub["ci"] = pd.to_numeric(sub["deltaG_95CI"], errors="coerce")
        bad = ~np.isfinite(sub["ddg"].to_numpy(dtype=float, na_value=np.nan))
        drop("non_finite_ddg", int(bad.sum()))
        sub = sub[~bad]
        for r in sub.itertuples(index=False):
            muts.append((str(r.WT_name), str(r.mut_type), float(r.ddg),
                         str(r.Stabilizing_mut).lower() == "true", float(r.ci)))
        print(f"  chunk {k}: {rows_in} rows read, {len(wt)} constructs, "
              f"{len(muts)} substitutions kept", flush=True)

    if not muts:
        raise SystemExit("[build] no single substitutions found — wrong source file?")

    mt = pd.DataFrame(muts, columns=["parent", "mut_type", "ddg",
                                     "stabilizing", "ci"])

    # --- sign check, before anything is written -----------------------------
    # Stabilizing_mut is a CONFIDENCE THRESHOLD, not a sign bit: it flags only
    # clearly stabilising mutations (the smallest flagged ddG_ML on the shipped
    # file is +1.00), so most positive-ddG rows are not flagged and requiring
    # equality would fail on a correctly-signed table. The implication is what
    # must hold.
    flagged = mt["stabilizing"].to_numpy()
    n_flagged = int(flagged.sum())
    mean_ddg = float(mt["ddg"].mean())
    if not n_flagged:
        raise SystemExit(
            "[build] REFUSING to write: no Stabilizing_mut rows, so the sign "
            "convention cannot be checked against the source."
        )
    implied = float((mt.loc[flagged, "ddg"] > 0).mean())
    print(f"\n[build] sign check: {implied:.2%} of the {n_flagged} rows the "
          f"release flags Stabilizing_mut have ddG_ML > 0 (smallest flagged ddG "
          f"{mt.loc[flagged, 'ddg'].min():+.3f}); mean ddG over all "
          f"substitutions = {mean_ddg:+.3f}")
    if implied < 0.99:
        raise SystemExit(
            f"[build] REFUSING to write: only {implied:.2%} of the rows the "
            f"release calls stabilising have ddG_ML > 0. The source convention is "
            f"not what this script assumes, and a wrong sign is invisible to "
            f"correlation metrics."
        )
    if mean_ddg > 0:
        raise SystemExit(
            f"[build] REFUSING to write: mean ddG is {mean_ddg:+.3f}. Random "
            f"substitutions are destabilising on average, so under "
            f"'positive is stabilising' this mean must be negative."
        )

    # --- average duplicate measurements -------------------------------------
    n_before = len(mt)
    mt = mt.groupby(["parent", "mut_type"], as_index=False).agg(
        ddg=("ddg", "mean"), ci=("ci", "mean"))
    print(f"[build] {n_before - len(mt)} duplicate (construct, substitution) "
          f"measurements averaged")

    # --- build the pairs ----------------------------------------------------
    records = []
    for r in mt.itertuples(index=False):
        parent = wt.get(r.parent)
        if parent is None:
            drop("no_wildtype_row_for_construct")
            continue
        m = SUB_RE.match(r.mut_type)
        wt_aa, pos, mt_aa = m.group(1), int(m.group(2)), m.group(3)
        seq = parent["aa_seq"]
        if pos < 1 or pos > len(seq):
            drop("position_outside_sequence")
            continue
        if seq[pos - 1] != wt_aa:
            drop("wildtype_residue_mismatch")
            continue
        # ddG = dG_mut - dG_wt, so the two dG standard errors add in quadrature.
        wt_ci, mt_ci = parent["ci"], r.ci
        if np.isfinite(wt_ci) and np.isfinite(mt_ci):
            sigma = float(np.hypot(mt_ci / (2 * Z95), wt_ci / (2 * Z95)))
        else:
            sigma = float("nan")
        records.append({
            "wt_seq": seq,
            "mt_seq": seq[:pos - 1] + mt_aa + seq[pos:],
            "ddG": float(r.ddg),
            "ddg_std": sigma,
            "wt_dna": parent["cds"],
            "parent": r.parent,
            "mut_type": r.mut_type,
        })

    out = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)

    with_cds = float(out["wt_dna"].notna().mean())
    lengths = out["wt_seq"].str.len()
    print(f"\n[build] wrote {args.out}")
    print(f"  rows in source          : {rows_in}")
    print(f"  rows written            : {len(out)}")
    print(f"  distinct constructs     : {out['parent'].nunique()}")
    print(f"  rows carrying a real CDS: {with_cds:.1%} "
          f"({'E_mech active for those rows' if with_cds else 'E_mech empty'})")
    print(f"  protein length          : median {int(lengths.median())}, "
          f"min {int(lengths.min())}, max {int(lengths.max())}")
    print(f"  ddG                     : mean {out['ddG'].mean():+.3f}, "
          f"sd {out['ddG'].std():.3f}, "
          f"range [{out['ddG'].min():+.2f}, {out['ddG'].max():+.2f}]")

    # The three-state neutral band is |ddG| <= sigma (src/heads.py, n_sigma=1).
    # Print what the measured band does to the class balance next to what the
    # three_state_sigma constant would have done, because that constant decides
    # a large share of the labels and the difference is not small.
    sig = out["ddg_std"]
    print(f"  ddg_std (propagated)    : finite {sig.notna().mean():.1%}, "
          f"median {sig.median():.4f}, p90 {sig.quantile(0.90):.4f}")
    for label, band in (("measured ddg_std", sig),
                        ("constant 0.5", pd.Series(0.5, index=out.index))):
        d = out["ddG"]
        print(f"  three-state, {label:<16}: "
              f"destabilising {(d < -band).mean():5.1%}  "
              f"neutral {(d.abs() <= band).mean():5.1%}  "
              f"stabilising {(d > band).mean():5.1%}")
    print(f"  dropped                 : "
          f"{', '.join(f'{k}={v}' for k, v in sorted(dropped.items())) or 'none'}")


if __name__ == "__main__":
    main()
