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

import numpy as np

from src import splits as S
from src.config import load_config
from src.data import load_mutation_table


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(df, identity: float, coverage: float) -> tuple[list[str], dict]:
    """Connected components under 'same identity cluster OR same gene'."""
    parents = sorted(df["parent_seq"].unique())
    info: dict = {"n_parent_sequences": len(parents)}

    seq_to_cluster: dict[str, str] = {}
    if S.mmseqs_available():
        keyed = {S.group_key_for_sequence(s): s for s in parents}
        clusters = S.cluster_with_mmseqs(
            keyed, min_seq_id=identity, coverage=coverage
        )
        for s in parents:
            k = S.group_key_for_sequence(s)
            seq_to_cluster[s] = clusters.get(k, k)
        info["clustering"] = f"mmseqs2 @ {identity:.0%} identity, {coverage:.0%} coverage"
    else:
        print(
            "[splits] mmseqs not found on PATH — falling back to exact "
            "parent-sequence grouping. This is still protein-disjoint, but a "
            "close homolog can appear on both sides. Install MMseqs2 "
            "(conda install -c bioconda mmseqs2) for identity-clustered splits."
        )
        for s in parents:
            seq_to_cluster[s] = S.group_key_for_sequence(s)
        info["clustering"] = "exact parent-sequence identity (MMseqs2 unavailable)"

    uf = _UnionFind()
    for s in parents:
        uf.union(f"c:{seq_to_cluster[s]}", f"s:{s}")

    has_gene = "group" in df.columns and not df["group"].equals(df["parent_seq"])
    if has_gene:
        for gene, seq in zip(df["group"], df["parent_seq"]):
            uf.union(f"g:{gene}", f"s:{seq}")
    info["gene_relation_used"] = bool(has_gene)

    groups = [uf.find(f"s:{s}") for s in df["parent_seq"]]
    info["n_groups"] = len(set(groups))
    return groups, info


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

    groups, info = build_groups(df, cfg.identity, cfg.coverage)
    row_splits = S.row_splits_from_groups(
        groups, ratios=tuple(args.ratios), seed=cfg.seed
    )
    if not S.verify_no_leakage(groups, row_splits):
        raise SystemExit("leakage detected: a group spans more than one split")

    df = df.copy()
    df["split"] = row_splits
    df["group_used"] = groups

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
        "data": args.data,
        "kind": args.kind,
        "grouping": info,
        "ratios": list(args.ratios),
        "seed": cfg.seed,
        "split_counts": {k: int(v) for k, v in counts.items()},
        "per_split": variance,
        "no_group_spans_splits": True,
        "load_report": report.as_dict(),
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"[splits] wrote {args.out} | rows={len(df)} | splits={counts}")
    print(f"[splits] {info['clustering']} | groups={info['n_groups']} "
          f"| gene relation merged: {info['gene_relation_used']}")
    for name, stats in sorted(variance.items()):
        print(f"[splits]   {name}: {stats}")
    print(f"[splits] leakage report -> {report_path}")


if __name__ == "__main__":
    main()
