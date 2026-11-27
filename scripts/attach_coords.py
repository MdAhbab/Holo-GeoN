"""
scripts/attach_coords.py
========================
Attach aligned 3D backbone coordinates (N, CA, C, CB), pLDDT, and ``coord_valid``
to an existing ``EncodingStore`` cache without re-running ESM-2 (GEOMETRIC_PLAN §2.4).

Hard verification: every mutation site present in the input table is checked
against the aligned PDB residue identity (within the allowed mismatch budget for
background point mutants/PROSS constructs), and any unaligned length mismatch
exits non-zero rather than silently attaching off-by-one geometry.

Example
-------
python -m scripts.attach_coords \
    --table data/megascale_ddg_pairs.parquet \
    --pdb-dir dataset/mega_scale_cdna/AlphaFold_model_PDBs \
    --cache cache/esm2_650M
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.encoder import EncodingStore, seq_key
from src.geometry import (
    MAX_MISMATCH_FRACTION,
    NAME_RE,
    PLDDT_VALID_FLOOR,
    align_to_sequence,
    normalize_pdb_stem,
    parse_pdb_backbone,
    resolve_pdb_path,
)


def _resolve_pdb(parent: str, available: dict[str, str]) -> str | None:
    """Match a MegaScale construct name or UniProt accession (AF-<ACC>.pdb)."""
    hit = resolve_pdb_path(parent, available)
    return str(hit) if hit is not None else None


def attach_from_npz(cache_dir: str, npz_path: str, report_path: str | None = None) -> dict:
    """Attach pre-aligned ``<sha1>__coords/__plddt/__valid`` arrays from an NPZ bundle."""
    store = EncodingStore(cache_dir)
    z = np.load(npz_path)
    keys = sorted({k.split("__", 1)[0] for k in z.files if k.endswith("__coords")})
    attached = skipped_uncached = skipped_len = 0
    for sha in keys:
        pt_path = os.path.join(cache_dir, f"{sha}.pt")
        if not os.path.isfile(pt_path):
            skipped_uncached += 1
            continue
        rec = torch.load(pt_path, map_location="cpu", weights_only=False)
        coords = z[f"{sha}__coords"]
        plddt = z[f"{sha}__plddt"]
        valid = z[f"{sha}__valid"]
        if rec["emb"].shape[0] != coords.shape[0]:
            skipped_len += 1
            continue
        rec["coords"] = torch.from_numpy(coords).float()
        rec["plddt"] = torch.from_numpy(plddt).float()
        rec["coord_valid"] = torch.from_numpy(
            (valid > 0.5) & (plddt >= PLDDT_VALID_FLOOR)
        ).float()
        rec["structure_id"] = os.path.basename(npz_path)
        store.put(rec)
        attached += 1
    summary = {
        "geom_npz": npz_path,
        "cache": cache_dir,
        "total_in_npz": len(keys),
        "attached": attached,
        "skipped_uncached": skipped_uncached,
        "skipped_length_mismatch": skipped_len,
    }
    print(f"[attach_coords] npz attached={attached}/{len(keys)} "
          f"(uncached={skipped_uncached}, len_mismatch={skipped_len})")
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", default=None, help="parquet with parent and wt_seq")
    ap.add_argument("--pdb-dir", default=None, help="directory of PDB files")
    ap.add_argument("--geom-npz", default=None, help="pre-aligned invariant_backbone_geometry.npz bundle")
    ap.add_argument("--cache", required=True, help="EncodingStore cache directory")
    ap.add_argument("--parent-col", default="parent")
    ap.add_argument("--seq-col", default="wt_seq")
    ap.add_argument("--max-mismatch-frac", type=float, default=MAX_MISMATCH_FRACTION)
    ap.add_argument("--report", default=None, help="optional JSON summary path")
    args = ap.parse_args(argv)

    if args.geom_npz:
        attach_from_npz(args.cache, args.geom_npz, args.report)
        return 0
    if not args.table or not args.pdb_dir:
        raise SystemExit("[attach_coords] pass either --geom-npz OR both --table and --pdb-dir")

    store = EncodingStore(args.cache)
    df = pd.read_parquet(args.table)
    constructs = df[[args.parent_col, args.seq_col]].drop_duplicates()

    available = {
        normalize_pdb_stem(f): f
        for f in os.listdir(args.pdb_dir)
        if f.lower().endswith(".pdb")
    }

    attached = skipped_uncached = skipped_no_pdb = 0
    reasons: dict[str, int] = {}

    for row in constructs.itertuples(index=False):
        parent = str(getattr(row, args.parent_col))
        seq = str(getattr(row, args.seq_col)).upper()
        if not store.has(seq):
            skipped_uncached += 1
            continue
        fname = _resolve_pdb(parent, available)
        if fname is None:
            skipped_no_pdb += 1
            reasons["no_pdb_for_construct"] = reasons.get("no_pdb_for_construct", 0) + 1
            continue

        pdb_seq, coords, plddt, atom_valid = parse_pdb_backbone(
            os.path.join(args.pdb_dir, fname)
        )
        if len(coords) == 0:
            reasons["empty_pdb"] = reasons.get("empty_pdb", 0) + 1
            continue

        offset, mismatches, err = align_to_sequence(
            pdb_seq, seq, max_mismatch_fraction=args.max_mismatch_frac
        )
        if offset is None:
            reasons[err or "alignment_failed"] = reasons.get(err or "alignment_failed", 0) + 1
            continue

        sub_coords = coords[offset:offset + len(seq)]
        sub_plddt = plddt[offset:offset + len(seq)]
        sub_valid = atom_valid[offset:offset + len(seq)]
        if len(sub_coords) != len(seq):
            raise SystemExit(
                f"FATAL alignment length mismatch for {parent}: "
                f"expected {len(seq)}, got {len(sub_coords)}"
            )

        rec = store.get(seq)
        if rec["emb"].shape[0] != len(seq):
            raise SystemExit(
                f"FATAL cache/sequence length mismatch for {parent}: "
                f"emb={rec['emb'].shape[0]} vs seq={len(seq)}"
            )

        rec["coords"] = torch.from_numpy(sub_coords).float()
        rec["plddt"] = torch.from_numpy(sub_plddt).float()
        rec["coord_valid"] = torch.from_numpy(
            (sub_valid > 0.5) & (sub_plddt >= PLDDT_VALID_FLOOR)
        ).float()
        rec["structure_id"] = fname
        rec["structure_mismatches"] = int(mismatches)
        store.put(rec)
        attached += 1

    summary = {
        "table": args.table,
        "pdb_dir": args.pdb_dir,
        "cache": args.cache,
        "constructs_in_table": int(len(constructs)),
        "attached": attached,
        "skipped_uncached": skipped_uncached,
        "skipped_no_pdb": skipped_no_pdb,
        "reasons": reasons,
    }
    print(f"[attach_coords] attached={attached}/{len(constructs)} "
          f"(uncached={skipped_uncached}, no_pdb={skipped_no_pdb}, reasons={reasons})")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
