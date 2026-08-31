"""
scripts/make_splits.py
======================
Leakage-controlled train/val/test assignment, for **every** task.

v1 applied identity clustering to stability only; the pathogenicity stage used
gene grouping, and the HoloGNN training path used ``random_split``, which puts
mutations of the same protein on both sides of the evaluation. Here one routine
serves both tasks and the grouping is the union of two relations:

  * **Sequence homology.** Parent proteins are clustered with MMseqs2 at
    ``--identity`` (default 30%). Whole clusters go to one split, so a test
    protein has no close homolog in training.
  * **Gene identity.** For pathogenicity, variants of the same gene are also
    forced together — two isoforms of one gene can fall below 30% identity over
    a short alignment yet still leak.

The two relations are merged with union-find, so a group is a connected
component under "same cluster OR same gene". Without MMseqs2 on PATH the
clustering relation degrades to exact parent-sequence identity, which is still
protein-disjoint, and the run records which of the two happened.

The relation lives in ``src/splits.build_groups`` so the whole-sequence tasks in
``scripts/train.py`` use the identical one rather than a second implementation
that drifted (round-2 review, R2-04). Whichever relation is used, the split is
then AUDITED by comparing held-out sequences against training sequences
directly - ``verify_no_leakage`` can only restate how the groups were built,
whereas the audit can fail.

Example
-------
python -m scripts.make_splits --data data/megascale_ddg_pairs.parquet \
    --kind stability --out data/splits_megascale.parquet
python -m scripts.make_splits --data data/clinvar_protein_pairs.parquet \
    --kind pathogenicity --out data/splits_clinvar.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import splits as S
from src.config import load_config
from src.data import load_mutation_table
from src.provenance import provenance

#: Fraction of held-out proteins allowed to have a near-identical training
#: protein. Zero: the grouping relation is supposed to make this impossible.
NEAR_DUPLICATE_TOLERANCE = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", choices=["stability", "pathogenicity"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--identity", type=float, default=None)
    ap.add_argument("--coverage", type=float, default=None)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        overrides={"identity": args.identity, "coverage": args.coverage,
                   "seed": args.seed, "max_rows": args.max_rows},
    )

    dataset_key = os.path.splitext(os.path.basename(args.data))[0]
    df, report = load_mutation_table(
        args.data,
        kind=args.kind,
        overrides=cfg.column_overrides(dataset_key),
        max_rows=cfg.max_rows,
        max_protein_length=cfg.max_protein_length,
    )
    print(report.summary())
    if df.empty:
        raise SystemExit(f"{args.data}: no usable rows after normalisation.")

    has_gene = "group" in df.columns and not df["group"].equals(df["parent_seq"])
    groups, info = S.build_groups(
        df["parent_seq"].tolist(),
        declared=df["group"].tolist() if has_gene else None,
        identity=cfg.identity, coverage=cfg.coverage,
    )
    row_splits = S.row_splits_from_groups(
        groups, ratios=tuple(args.ratios), seed=cfg.seed
    )
    # Round-2 review, R2-17: the report used to write `no_group_spans_splits`
    # as the literal True. It was true - the raise below guarantees it - but a
    # constant asserting its own validation is the pattern R1-02 deleted
    # `validated_by_real_run` for. Write what the check returned.
    no_group_spans = S.verify_no_leakage(groups, row_splits)
    if not no_group_spans:
        raise SystemExit("leakage detected: a group spans more than one split")

    df = df.copy()
    df["split"] = row_splits
    df["group_used"] = groups

    # verify_no_leakage only re-states how the groups were built. This measures
    # the sequences across the boundary, so it can fail where the group key is
    # blind (R2-04).
    audit = S.near_duplicate_audit(
        df.loc[df["split"] == "train", "parent_seq"].tolist(),
        df.loc[df["split"] != "train", "parent_seq"].tolist(),
    )
    leaked = audit["fraction_with_near_train_neighbour"]
    if leaked > NEAR_DUPLICATE_TOLERANCE:
        raise SystemExit(
            f"{args.data}: {leaked:.1%} of held-out proteins have a "
            f"near-identical protein in train (estimated k-mer Jaccard >= "
            f"{audit['jaccard_threshold']}). Whole groups went to one side, so "
            f"the grouping relation itself is too fine: declare the parent "
            f"protein or gene under this dataset's key in configs/data.yaml, "
            f"and install MMseqs2 to group homologs. Overlap examples: "
            f"{audit['examples']}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)

    counts = df["split"].value_counts().to_dict()
    # Target-variance per split travels with the split: a test split with
    # compressed variance mechanically depresses Pearson r, and reporting r
    # without it invites a comparison the number cannot support.
    variance = {}
    if "target" in df.columns:
        for name, part in df.groupby("split"):
            variance[name] = {
                "n": int(len(part)),
                "target_mean": float(part["target"].mean()),
                "target_std": float(part["target"].std()),
            }
    elif "label" in df.columns:
        for name, part in df.groupby("split"):
            variance[name] = {
                "n": int(len(part)),
                "positive_fraction": float(part["label"].mean()),
            }

    report_path = os.path.join(
        os.path.dirname(args.out) or ".", f"leakage_report_{args.kind}.json"
    )
    payload = {
        "provenance": provenance(),
        "data": args.data,
        "kind": args.kind,
        "grouping": info,
        "ratios": list(args.ratios),
        "seed": cfg.seed,
        "split_counts": {k: int(v) for k, v in counts.items()},
        "per_split": variance,
        "no_group_spans_splits": no_group_spans,
        "near_duplicate_audit": audit,
        "load_report": report.as_dict(),
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"[splits] wrote {args.out} | rows={len(df)} | splits={counts}")
    print(f"[splits] {info['clustering']} | groups={info['n_groups']} "
          f"| gene relation merged: {info['declared_group_relation_used']}")
    print(f"[splits] near-duplicate audit: "
          f"{audit['fraction_with_near_train_neighbour']:.1%} of "
          f"{audit['n_held_checked']} held-out proteins have a near-identical "
          f"training protein")
    for name, stats in sorted(variance.items()):
        print(f"[splits]   {name}: {stats}")
    print(f"[splits] leakage report -> {report_path}")


if __name__ == "__main__":
    main()
