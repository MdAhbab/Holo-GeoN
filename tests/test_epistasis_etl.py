"""
tests/test_epistasis_etl.py
===========================
Unit & integration tests for:
  1. ``scripts/build_epistasis_table.py`` (streaming double-mutant ETL, canonical
     ``site_i < site_j`` ordering, exact ``dddg``, exact 3D ``dist_cb`` from
     aligned PDBs, sign convention & ``--singles`` parity, CDS frame slicing,
     split mapping, and ``scripts/train_epistasis.py`` compatibility).
  2. Upgraded ``scripts/zenodo_pack.py`` (``--local-processed-dir`` staging,
     deterministic ``true_contacts_megascale.tar.gz``, and ``DATASHEET.md``
     provenance + claims-discipline checks).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tests.helpers import check, section

AA_CODON = {
    "A": "GCT", "C": "TGC", "D": "GAC", "E": "GAA", "F": "TTC",
    "G": "GGT", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
    "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGT",
    "S": "TCT", "T": "ACC", "V": "GTT", "W": "TGG", "Y": "TAC",
}
ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def _encode_dna(seq: str) -> str:
    return "".join(AA_CODON[a] for a in seq)


def _write_synthetic_pdb(path: Path, seq: str, cb_coords: list[tuple[float, float, float]]) -> None:
    lines = []
    atom_id = 1
    for res_idx, (aa, (x, y, z)) in enumerate(zip(seq, cb_coords), 1):
        res3 = ONE_TO_THREE[aa]
        # Place CA nearby, and CB (omitted for GLY so parse_pdb_cb exercises CA fallback).
        ca_x, ca_y, ca_z = (x, y, z) if aa == "G" else (x - 1.0, y, z)
        lines.append(
            f"ATOM  {atom_id:5d}  CA  {res3} A{res_idx:4d}    "
            f"{ca_x:8.3f}{ca_y:8.3f}{ca_z:8.3f}  1.00 90.00           C"
        )
        atom_id += 1
        if aa != "G":
            lines.append(
                f"ATOM  {atom_id:5d}  CB  {res3} A{res_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C"
            )
            atom_id += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_synthetic_fixtures(tmp: Path) -> dict:
    # Domain sequences (length 7..8):
    # 1ABC.pdb has PDB offset +2 (linker "GS" in PDB before domain seq "MAKLVKY").
    seq_1abc = "MAKLVKY"
    seq_1abc_l4f = "MAKFVKY"
    seq_design = "ACDEFGH"
    seq_unsplit = "QRSTVWY"

    pdb_dir = tmp / "AlphaFold_model_PDBs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    # PDB sequence = "GS" + "MAKLVKY" (length 9); domain sits at offset 2.
    # Domain idx 1 ('A', PDB res 4): CB = (0, 0, 0)
    # Domain idx 5 ('K', PDB res 8): CB = (3, 4, 0) -> dist(1, 5) = 5.0 A (< 8 A)
    # Domain idx 6 ('Y', PDB res 9): CB = (0, 12, 0) -> dist(1, 6) = 12.0 A (>= 8 A)
    pdb_seq = "GS" + seq_1abc
    coords = [
        (50.0, 50.0, 50.0),   # G linker (idx 0)
        (40.0, 40.0, 40.0),   # S linker (idx 1)
        (10.0, 0.0, 0.0),     # M (domain idx 0)
        (0.0, 0.0, 0.0),      # A (domain idx 1) -> site 1
        (20.0, 0.0, 0.0),     # K (domain idx 2)
        (30.0, 0.0, 0.0),     # L (domain idx 3)
        (40.0, 0.0, 0.0),     # V (domain idx 4)
        (3.0, 4.0, 0.0),      # K (domain idx 5) -> site 5, dist from 1 == 5.0
        (0.0, 12.0, 0.0),     # Y (domain idx 6) -> site 6, dist from 1 == 12.0
    ]
    _write_synthetic_pdb(pdb_dir / "1ABC.pdb", pdb_seq, coords)

    linker = "SAGG"
    rows = []

    def add_row(wt_name, mut_type, aa_seq, ddg_ml, dg_ml=None, stab="False", aa_full="", dna=""):
        if dg_ml is None and ddg_ml != "-":
            dg_ml = f"{3.0 + float(ddg_ml):.6f}"
        rows.append({
            "WT_name": wt_name,
            "mut_type": mut_type,
            "aa_seq": aa_seq,
            "aa_seq_full": aa_full,
            "dna_seq": dna,
            "dG_ML": "-" if dg_ml is None else str(dg_ml),
            "ddG_ML": str(ddg_ml),
            "Stabilizing_mut": stab,
        })

    # WT rows (test 1ABC with valid sliced CDS, DESIGNED with shifted bad CDS, UNSPLIT & 1ABC_L4F after some mutants).
    full_1abc = linker + seq_1abc + linker
    dna_1abc = _encode_dna(full_1abc)
    add_row("1ABC.pdb", "wt", seq_1abc, 0.0, dg_ml=3.0, aa_full=full_1abc, dna=dna_1abc)

    full_design = linker + seq_design + linker
    dna_design_bad = "A" + _encode_dna(full_design)  # frame shift by 1 nt -> fails translation check
    add_row("DESIGNED.pdb", "wt", seq_design, 0.0, dg_ml=2.5, aa_full=full_design, dna=dna_design_bad)

    # Single mutants for 1ABC.pdb (including stabilizing positive, negative mean, and duplicate averaging).
    add_row("1ABC.pdb", "A2V", "MVKLVKY", -0.80, stab="False")
    add_row("1ABC.pdb", "K6E", "MAKLVEY", -0.60, stab="False")
    add_row("1ABC.pdb", "Y7W", "MAKLVKW", 1.20, stab="True")    # stabilizing_positive check
    add_row("1ABC.pdb", "M1A", "AAKLVKY", -1.10, stab="False")
    add_row("1ABC.pdb", "M1A", "AAKLVKY", -0.90, stab="False")   # duplicate -> mean -1.00
    add_row("1ABC.pdb", "V5I", "MAKLIKY", "-")                   # non-finite single

    # Single mutants for 1ABC.pdb_L4F, DESIGNED.pdb, UNSPLIT.pdb.
    add_row("1ABC.pdb_L4F", "A2G", "MGKFVKY", -0.50, stab="False")
    add_row("1ABC.pdb_L4F", "K6R", "MAKFVRY", -0.40, stab="False")
    add_row("DESIGNED.pdb", "A1V", "VCDEFGH", -0.70, stab="False")
    add_row("DESIGNED.pdb", "D3E", "ACEEFGH", -0.30, stab="False")
    add_row("UNSPLIT.pdb", "Q1A", "ARSTVWY", -0.50, stab="False")
    add_row("UNSPLIT.pdb", "R2A", "QASTVWY", -0.20, stab="False")

    # Valid double mutants (including reversed order + duplicates!).
    # 1) 1ABC.pdb A2V:K6E listed twice: once forward (-0.30), once reversed K6E:A2V (-0.50) -> mean ddg_ab = -0.40.
    #    site_i=1 (A2V, ddg_a=-0.80), site_j=5 (K6E, ddg_b=-0.60), dddg = -0.40 - (-0.80) - (-0.60) = +1.00, dist_cb = 5.0
    add_row("1ABC.pdb", "A2V:K6E", "MVKLVEY", -0.30)
    add_row("1ABC.pdb", "K6E:A2V", "MVKLVEY", -0.50)

    # 2) 1ABC.pdb Y7W:A2V (reversed! site 6 vs site 1) -> canonical site_i=1 (A2V, -0.80), site_j=6 (Y7W, +1.20)
    #    ddg_ab = +0.10 -> dddg = 0.10 - (-0.80) - 1.20 = -0.30, dist_cb = 12.0
    add_row("1ABC.pdb", "Y7W:A2V", "MVKLVKW", 0.10)

    # 3) 1ABC.pdb_L4F K6R:A2G (tests NAME_RE + 1 mismatch alignment to 1ABC.pdb -> dist_cb = 5.0)
    #    ddg_ab = -0.75 -> dddg = -0.75 - (-0.50) - (-0.40) = +0.15
    add_row("1ABC.pdb_L4F", "K6R:A2G", "MGKFVRY", -0.75)

    # 4) DESIGNED.pdb A1V:D3E (no PDB -> dist_cb = NaN; bad CDS -> wt_dna = None)
    #    ddg_ab = -1.20 -> dddg = -1.20 - (-0.70) - (-0.30) = -0.20
    add_row("DESIGNED.pdb", "A1V:D3E", "VCEEFGH", -1.20)

    # 5) UNSPLIT.pdb Q1A:R2A (parent absent from splits.parquet)
    add_row("UNSPLIT.pdb", "Q1A:R2A", "AASTVWY", -0.60)

    # Invalid / filtered doubles:
    add_row("1ABC.pdb", "insG1:A2V", "GMVKLVKY", -0.50)       # indel
    add_row("1ABC.pdb", "A2V:L4I:K6E", "MVKIVEY", -0.50)      # triple
    add_row("1ABC.pdb", "A2V:K6K", "MVKLVKY", -0.80)          # synonymous residue in double
    add_row("1ABC.pdb", "A2V:A2G", "MGKLVKY", -0.50)          # same-site collision
    add_row("1ABC.pdb", "A2V:V5I", "MVKLIKY", -0.90)          # missing single ddG (V5I was '-')
    add_row("1ABC.pdb", "G2V:K6E", "MVKLVEY", -0.40)          # wt_residue_mismatch
    add_row("1ABC.pdb", "A2V:K6E", "MVKLVEY", "-", dg_ml="-") # non-finite dG/ddG

    # Put WT rows for 1ABC.pdb_L4F and UNSPLIT.pdb AFTER mutants to verify order independence.
    add_row("1ABC.pdb_L4F", "wt", seq_1abc_l4f, 0.0, dg_ml=2.8)
    add_row("UNSPLIT.pdb", "wt", seq_unsplit, 0.0, dg_ml=2.2)

    csv_path = tmp / "tsuboyama_synth.csv"
    fieldnames = ["WT_name", "mut_type", "aa_seq", "aa_seq_full", "dna_seq", "dG_ML", "ddG_ML", "Stabilizing_mut"]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Splits parquet matching splits_megascale_*.parquet schema (parent_seq keyed).
    splits_path = tmp / "splits_synth.parquet"
    pd.DataFrame([
        {"parent_seq": seq_1abc, "site": 1, "wt_aa": "A", "mt_aa": "V", "target": -0.8,
         "sigma": 0.05, "group": seq_1abc, "cds": None, "split": "train", "group_used": "c:1"},
        {"parent_seq": seq_1abc_l4f, "site": 1, "wt_aa": "A", "mt_aa": "G", "target": -0.5,
         "sigma": 0.05, "group": seq_1abc_l4f, "cds": None, "split": "test", "group_used": "c:2"},
        {"parent_seq": seq_design, "site": 0, "wt_aa": "A", "mt_aa": "V", "target": -0.7,
         "sigma": 0.05, "group": seq_design, "cds": None, "split": "val", "group_used": "c:3"},
    ]).to_parquet(splits_path, index=False)

    # Matching singles parquet (positive = stabilizing) + inverted singles parquet for negative test.
    singles_ok = tmp / "singles_ok.parquet"
    pd.DataFrame([
        {"parent": "1ABC.pdb", "mut_type": "A2V", "ddG": -0.80},
        {"parent": "1ABC.pdb", "mut_type": "K6E", "ddG": -0.60},
        {"parent": "1ABC.pdb", "mut_type": "Y7W", "ddG": 1.20},
        {"parent": "1ABC.pdb", "mut_type": "M1A", "ddG": -1.00},
    ]).to_parquet(singles_ok, index=False)

    singles_inv = tmp / "singles_inv.parquet"
    pd.DataFrame([
        {"parent": "1ABC.pdb", "mut_type": "A2V", "ddG": 0.80},
        {"parent": "1ABC.pdb", "mut_type": "K6E", "ddG": 0.60},
        {"parent": "1ABC.pdb", "mut_type": "Y7W", "ddG": -1.20},
        {"parent": "1ABC.pdb", "mut_type": "M1A", "ddG": 1.00},
    ]).to_parquet(singles_inv, index=False)

    return {
        "csv": csv_path,
        "pdb_dir": pdb_dir,
        "splits": splits_path,
        "singles_ok": singles_ok,
        "singles_inv": singles_inv,
        "seqs": {
            "1ABC.pdb": seq_1abc,
            "1ABC.pdb_L4F": seq_1abc_l4f,
            "DESIGNED.pdb": seq_design,
            "UNSPLIT.pdb": seq_unsplit,
        },
        "expected_dna_1abc": _encode_dna(seq_1abc),
    }


def test_build_epistasis_table_end_to_end() -> None:
    section("epistasis ETL - streaming parse, canonical ordering, dddg, dist_cb, sign & splits")
    from scripts.build_epistasis_table import build_epistasis_table, main as etl_main
    from scripts.train_epistasis import _build_double_groups
    from src.encoder import EncodingStore, N_TAP_LAYERS

    with tempfile.TemporaryDirectory(prefix="holo_epi_etl_") as td:
        tmp = Path(td)
        fx = _build_synthetic_fixtures(tmp)
        out_pq = tmp / "megascale_double_mutants.parquet"
        sum_json = tmp / "epistasis_etl_summary.json"

        rc = etl_main([
            "--source", str(fx["csv"]),
            "--pdb-dir", str(fx["pdb_dir"]),
            "--splits", str(fx["splits"]),
            "--singles", str(fx["singles_ok"]),
            "--out", str(out_pq),
            "--summary", str(sum_json),
            "--chunk", "5",
        ])
        check(rc == 0 and out_pq.is_file() and sum_json.is_file(),
              "CLI ran cleanly and wrote output parquet + epistasis_etl_summary.json")

        df = pd.read_parquet(out_pq)
        summary = json.loads(sum_json.read_text(encoding="utf-8"))

        expected_cols = [
            "parent", "parent_seq", "wt_dna", "site_i", "wt_i", "mt_i",
            "site_j", "wt_j", "mt_j", "ddg_a", "ddg_b", "ddg_ab", "dddg",
            "dist_cb", "split", "group_used",
        ]
        check(list(df.columns) == expected_cols, f"output columns match schema ({len(expected_cols)} cols)")
        check(len(df) == 4, f"kept exactly 4 valid split double mutants after dedup (got {len(df)})")

        # Canonical ordering & WT residue identity on every row.
        check(bool((df["site_i"] < df["site_j"]).all()), "all rows enforce canonical site_i < site_j")
        check(
            all(s[i] == w for s, i, w in zip(df["parent_seq"], df["site_i"], df["wt_i"]))
            and all(s[j] == w for s, j, w in zip(df["parent_seq"], df["site_j"], df["wt_j"])),
            "parent_seq[site_i]==wt_i and parent_seq[site_j]==wt_j hold on 100% of rows",
        )

        # Row 1: 1ABC.pdb A2V:K6E (dedup of forward -0.30 and reversed -0.50 -> -0.40)
        r1 = df[(df["parent"] == "1ABC.pdb") & (df["site_i"] == 1) & (df["site_j"] == 5)].iloc[0]
        check(
            r1["wt_i"] == "A" and r1["mt_i"] == "V" and r1["wt_j"] == "K" and r1["mt_j"] == "E",
            "A2V:K6E / K6E:A2V canonicalized cleanly",
        )
        check(
            abs(r1["ddg_a"] - (-0.80)) < 1e-9
            and abs(r1["ddg_b"] - (-0.60)) < 1e-9
            and abs(r1["ddg_ab"] - (-0.40)) < 1e-9
            and abs(r1["dddg"] - 1.00) < 1e-9,
            f"A2V:K6E exact dedup & dddg=+1.00 (got ddg_ab={r1['ddg_ab']:.3f}, dddg={r1['dddg']:.3f})",
        )
        check(abs(r1["dist_cb"] - 5.0) < 1e-4, f"1ABC.pdb (1,5) aligned dist_cb == 5.0 A (got {r1['dist_cb']:.4f})")
        check(r1["wt_dna"] == fx["expected_dna_1abc"], "1ABC.pdb wt_dna sliced cleanly to domain frame")
        check(r1["split"] == "train" and r1["group_used"] == "c:1", "1ABC.pdb mapped to train / c:1")

        # Row 2: 1ABC.pdb Y7W:A2V (reversed in CSV! site 6 vs site 1 -> swapped to 1, 6; ddg_a=-0.80, ddg_b=+1.20)
        r2 = df[(df["parent"] == "1ABC.pdb") & (df["site_i"] == 1) & (df["site_j"] == 6)].iloc[0]
        check(
            r2["wt_i"] == "A" and r2["mt_i"] == "V" and r2["wt_j"] == "Y" and r2["mt_j"] == "W"
            and abs(r2["ddg_a"] - (-0.80)) < 1e-9 and abs(r2["ddg_b"] - 1.20) < 1e-9
            and abs(r2["dddg"] - (-0.30)) < 1e-9,
            "reversed Y7W:A2V swapped ddg_a/ddg_b properly and computed exact dddg=-0.30",
        )
        check(abs(r2["dist_cb"] - 12.0) < 1e-4, f"1ABC.pdb (1,6) aligned dist_cb == 12.0 A (got {r2['dist_cb']:.4f})")

        # Row 3: 1ABC.pdb_L4F K6R:A2G (NAME_RE + 1 mismatch alignment -> test / c:2)
        r3 = df[df["parent"] == "1ABC.pdb_L4F"].iloc[0]
        check(
            r3["site_i"] == 1 and r3["site_j"] == 5
            and abs(r3["ddg_a"] - (-0.50)) < 1e-9 and abs(r3["ddg_b"] - (-0.40)) < 1e-9
            and abs(r3["dddg"] - 0.15) < 1e-9
            and abs(r3["dist_cb"] - 5.0) < 1e-4
            and r3["split"] == "test" and r3["group_used"] == "c:2",
            "1ABC.pdb_L4F matched via NAME_RE with 1 mismatch, dist_cb=5.0 A, split=test",
        )

        # Row 4: DESIGNED.pdb A1V:D3E (no PDB -> NaN dist_cb; bad CDS -> None wt_dna; val / c:3)
        r4 = df[df["parent"] == "DESIGNED.pdb"].iloc[0]
        check(
            not math.isfinite(r4["dist_cb"]) and pd.isna(r4["wt_dna"])
            and r4["split"] == "val" and r4["group_used"] == "c:3",
            "DESIGNED.pdb keeps NaN dist_cb, drops bad-frame wt_dna as None, maps split=val",
        )

        # Summary accounting checks.
        check(summary["n_duplicate_doubles_averaged"] == 1, "summary reports 1 duplicate double averaged")
        check(summary["n_duplicate_singles_averaged"] == 1, "summary reports 1 duplicate single averaged")
        check(summary["absent_from_splits"]["n_parents"] == 1 and summary["absent_from_splits"]["n_rows"] == 1,
              "summary counts UNSPLIT.pdb absent from splits")
        check(summary["absent_from_splits"]["parents"] == ["UNSPLIT.pdb"],
              "summary names absent parent UNSPLIT.pdb")
        check(summary["pdb_alignment"]["rows_contact_lt_8A"] == 2
              and summary["pdb_alignment"]["rows_distant_ge_8A"] == 1,
              "summary distance stratification counts 2 <8A and 1 >=8A")
        check(summary["sign_parity_vs_singles"]["sign_agreement"] == 1.0,
              "sign parity against singles parquet == 1.0")

        # Keep-unsplit mode tags UNSPLIT.pdb rather than dropping.
        out_keep = tmp / "doubles_keep_unsplit.parquet"
        sum_keep = build_epistasis_table(
            fx["csv"], out_keep, pdb_dir=fx["pdb_dir"], splits_path=fx["splits"], keep_unsplit=True
        )
        df_keep = pd.read_parquet(out_keep)
        check(len(df_keep) == 5 and int((df_keep["split"] == "unsplit").sum()) == 1,
              "keep_unsplit=True retains UNSPLIT.pdb tagged as split='unsplit'")
        check(sum_keep["absent_from_splits"]["action"] == "tagged_unsplit",
              "summary records action='tagged_unsplit'")

        # Inverted singles parquet MUST fail loudly.
        refused = False
        try:
            build_epistasis_table(
                fx["csv"], tmp / "should_fail.parquet",
                pdb_dir=fx["pdb_dir"], splits_path=fx["splits"], singles_path=fx["singles_inv"],
            )
        except SystemExit:
            refused = True
        check(refused, "inverted-sign --singles parquet is refused loudly via SystemExit")

        # End-to-end schema check with train_epistasis._build_double_groups.
        store = EncodingStore(str(tmp / "enc_cache"))
        rng = np.random.default_rng(11)
        for seq in fx["seqs"].values():
            L = len(seq)
            store.put({
                "seq": seq,
                "emb": torch.from_numpy(rng.normal(0, 0.2, (L, 16)).astype(np.float32)),
                "contact_edges": torch.tensor([[1, 5], [5, 1]], dtype=torch.long),
                "contact_probs": torch.tensor([0.9, 0.9], dtype=torch.float32),
                "layer_embs": torch.from_numpy(rng.normal(0, 0.2, (N_TAP_LAYERS, L, 16)).astype(np.float32)),
                "wt_logprobs": torch.log_softmax(torch.zeros((L, 20), dtype=torch.float32), dim=-1),
                "site_entropy": torch.full((L,), 0.4, dtype=torch.float32),
                "dim": 16,
            })
        cfg = {"window": 32, "graph_top_k": 4, "geo_threshold": 0.3, "neighborhood_hops": 2}
        tr_g = _build_double_groups(df[df["split"] == "train"], store, cfg, use_true_contact=True)
        te_g = _build_double_groups(df[df["split"] == "test"], store, cfg, use_true_contact=True)
        check(len(tr_g) == 1 and len(te_g) == 1 and tr_g[0]["si"].shape[0] == 2,
              "output parquet loads directly into scripts/train_epistasis._build_double_groups")


def test_zenodo_pack_local_processed_and_datasheet() -> None:
    section("zenodo_pack upgrade - --local-processed-dir, tar.gz, DATASHEET provenance & claims discipline")
    from scripts.zenodo_pack import main as zenodo_main, sha256_file

    with tempfile.TemporaryDirectory(prefix="holo_zenodo_up_") as td:
        tmp = Path(td)
        local_data = tmp / "local_data"
        local_data.mkdir()
        (local_data / "true_contacts").mkdir()

        pd.DataFrame({
            "ref_protein": ["MKTAY", "ACDEF"],
            "alt_protein": ["MKTAV", "ACDEG"],
            "clinical_significance": ["Pathogenic", "Benign"],
            "wt_dna": ["ATGAAAACCGCTTAC", "GCTTGCGACGAATTC"],
        }).to_parquet(local_data / "clinvar_protein_pairs.parquet", index=False)

        pd.DataFrame({
            "parent_seq": ["MKTAY", "ACDEF"],
            "split": ["train", "test"],
            "group_used": ["c1", "c2"],
        }).to_parquet(local_data / "splits_clinvar.parquet", index=False)

        pd.DataFrame({
            "parent": ["1ABC.pdb"],
            "parent_seq": ["MAKLVKY"],
            "site_i": [1], "wt_i": ["A"], "mt_i": ["V"],
            "site_j": [5], "wt_j": ["K"], "mt_j": ["E"],
            "ddg_a": [-0.8], "ddg_b": [-0.6], "ddg_ab": [-0.4], "dddg": [1.0],
            "dist_cb": [5.0], "split": ["train"], "group_used": ["c:1"],
        }).to_parquet(local_data / "megascale_double_mutants.parquet", index=False)

        (local_data / "leakage_report_pathogenicity.json").write_text('{"n_groups": 10099}', encoding="utf-8")
        (local_data / "leakage_report_stability.json").write_text('{"n_groups": 190}', encoding="utf-8")

        for k in ("aaa111", "bbb222"):
            np.save(local_data / "true_contacts" / f"{k}.npy", np.eye(4, dtype=np.uint8))

        stage = tmp / "Holo-StabBench-v1.0"
        rc = zenodo_main([
            "--stage", str(stage),
            "--version", "1.0.0-test",
            "--local-processed-dir", str(local_data),
        ])
        check(rc == 0, "zenodo_pack CLI with --local-processed-dir exited 0")

        expected_staged = [
            stage / "tables" / "clinvar_protein_pairs.parquet",
            stage / "tables" / "megascale_double_mutants.parquet",
            stage / "splits" / "splits_clinvar.parquet",
            stage / "splits" / "leakage_report_pathogenicity.json",
            stage / "splits" / "leakage_report_stability.json",
            stage / "features" / "true_contacts_megascale.tar.gz",
        ]
        check(all(p.is_file() for p in expected_staged),
              "all 6 local processed targets staged into tables/, splits/, features/")

        tar_path = stage / "features" / "true_contacts_megascale.tar.gz"
        with tarfile.open(tar_path, "r:gz") as tf:
            names = sorted(tf.getnames())
        check(names == ["aaa111.npy", "bbb222.npy"],
              f"true_contacts_megascale.tar.gz holds sorted .npy maps ({names})")

        sha_lines = [
            ln.strip() for ln in (stage / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        check(len(sha_lines) == 6, f"SHA256SUMS.txt covers all 6 staged files (got {len(sha_lines)})")
        all_match = True
        for ln in sha_lines:
            digest, rel = ln.split("  ", 1)
            if sha256_file(stage / rel) != digest:
                all_match = False
        check(all_match, "every entry in SHA256SUMS.txt matches actual file digest")

        man = json.loads((stage / "release_manifest.json").read_text(encoding="utf-8"))
        check(man["total_files"] == 6, "release_manifest.json total_files == 6")

        ds = (stage / "DATASHEET.md").read_text(encoding="utf-8")
        check("Included vs. Excluded Provenance Table" in ds,
              "DATASHEET.md contains Included vs. Excluded Provenance Table")
        for needle in (
            "190,389", "10,099", "69,378", "373", "412", "538", "166",
            "massive_kb", "uniref50",
            "fold-clustered, NOT fold-disjoint",
            "0.5956", "37/39", "0.82", "0.72", "0.62",
            "76 fold clusters",
        ):
            check(needle in ds, f"DATASHEET.md documents '{needle}'")


def test_pdb_stem_normalization_and_resolution() -> None:
    section("PDB stem normalization — EA|run2_0325_0005.pdb vs colon/underscore/AFDB variants")
    from scripts.attach_coords import _resolve_pdb
    from scripts.build_epistasis_table import _align_parent_pdbs
    from scripts.competitors.rasp import _match_pdb as rasp_match, _resolve_pdb_map as rasp_map
    from scripts.competitors.spurs import _match_pdb as spurs_match, _resolve_pdb_map as spurs_map
    from scripts.competitors.thermompnn import _match_pdb as thermo_match, _resolve_pdb_map as thermo_map
    from src.geometry import normalize_pdb_stem, resolve_pdb_path

    # 1. normalize_pdb_stem canonicalizes '|', ':', '_', and trailing '.pdb'
    for variant in (
        "EA|run2_0325_0005.pdb",
        "EA:run2_0325_0005.pdb",
        "EA_run2_0325_0005.pdb",
        "ea|run2_0325_0005.PDB",
        "EA:run2_0325_0005",
    ):
        check(
            normalize_pdb_stem(variant) == "EA_RUN2_0325_0005",
            f"normalize_pdb_stem('{variant}') == 'EA_RUN2_0325_0005'",
        )

    # 2. resolve_pdb_path matches EA|run2_0325_0005.pdb against Windows '_', Linux ':', and '|' PDBs
    for disk_fname in (
        "EA_run2_0325_0005.pdb",
        "EA:run2_0325_0005.pdb",
        "EA|run2_0325_0005.pdb",
    ):
        avail_norm = {normalize_pdb_stem(disk_fname): disk_fname}
        avail_raw = {disk_fname[:-4].upper(): disk_fname}
        check(
            resolve_pdb_path("EA|run2_0325_0005.pdb", avail_norm) == disk_fname,
            f"resolve_pdb_path matches EA|run2_0325_0005.pdb -> {disk_fname} (normalized keys)",
        )
        check(
            resolve_pdb_path("EA|run2_0325_0005.pdb", avail_raw) == disk_fname,
            f"resolve_pdb_path matches EA|run2_0325_0005.pdb -> {disk_fname} (raw uppercase keys)",
        )
        check(
            resolve_pdb_path("EA|run2_0325_0005.pdb_A12V", avail_norm) == disk_fname,
            f"resolve_pdb_path matches background mutant EA|run2_0325_0005.pdb_A12V -> {disk_fname}",
        )
        check(
            _resolve_pdb("EA|run2_0325_0005.pdb", avail_norm) == disk_fname,
            f"attach_coords._resolve_pdb matches EA|run2_0325_0005.pdb -> {disk_fname}",
        )

    # 3. AlphaFold DB naming: AF-P12345-F1-model_v4.pdb
    af_fname = "AF-P12345-F1-model_v4.pdb"
    af_avail = {normalize_pdb_stem(af_fname): af_fname}
    check(
        resolve_pdb_path("P12345", af_avail) == af_fname,
        "resolve_pdb_path matches UniProt accession 'P12345' -> AF-P12345-F1-model_v4.pdb",
    )
    check(
        resolve_pdb_path("P12345.pdb", af_avail) == af_fname,
        "resolve_pdb_path matches 'P12345.pdb' -> AF-P12345-F1-model_v4.pdb",
    )

    # 4. End-to-end disk resolution for epistasis ETL and competitor adapters
    with tempfile.TemporaryDirectory(prefix="holo_pdb_norm_") as td:
        tmp = Path(td)
        seq = "MAKLVKY"
        coords = [(float(i), 0.0, 0.0) for i in range(len(seq))]
        pdb_file = tmp / "EA_run2_0325_0005.pdb"
        _write_synthetic_pdb(pdb_file, seq, coords)

        aligned, skips, hist = _align_parent_pdbs(
            ["EA|run2_0325_0005.pdb"],
            {"EA|run2_0325_0005.pdb": {"aa_seq": seq}},
            tmp,
        )
        check(
            "EA|run2_0325_0005.pdb" in aligned and not skips and hist == {0: 1},
            "build_epistasis_table._align_parent_pdbs aligns EA|run2_0325_0005.pdb to EA_run2_0325_0005.pdb",
        )
        check(
            thermo_match("EA|run2_0325_0005.pdb", thermo_map(str(tmp))) == pdb_file
            and spurs_match("EA|run2_0325_0005.pdb", spurs_map(str(tmp))) == pdb_file
            and rasp_match("EA|run2_0325_0005.pdb", rasp_map(str(tmp))) == pdb_file,
            "ThermoMPNN, SPURS, and RaSP adapters all match EA|run2_0325_0005.pdb -> EA_run2_0325_0005.pdb",
        )


TESTS = [
    test_build_epistasis_table_end_to_end,
    test_zenodo_pack_local_processed_and_datasheet,
    test_pdb_stem_normalization_and_resolution,
]


if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
