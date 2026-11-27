"""
tests/test_a100_suite_smoke.py
==============================
Comprehensive CPU end-to-end smoke test for the Holo-GeoN v2 A100 benchmark
suite, exercising every stage and bug fix on lightweight synthetic data:

  1. ``scripts.train`` (v2-Lean + v2-Lean with ``--true-contact-edges``)
  2. ``scripts.export_predictions`` + ``scripts.eval_external`` (asserting
     ``max_abs_diff < 1e-5`` between batched per-protein fast-path and
     row-wise ``MutationDataset`` on mixed short <= window and long > window
     proteins!)
  3. ``scripts.calibration``
  4. ``scripts.competitors.{spurs,rasp,prostata,ddgemb}`` (including
     ``spurs --mode upstream --official-external-only`` clean non-zero +
     ``official_spurs_status.json``, plus ``--extra-externals``)
  5. ``stage_block2_prostata_ft`` helpers (``_group_by_parent``,
     ``_eval_groups``, ``_measure_contact_drift``)
  6. ``holo.py`` (``doctor``, ``predict``, ``scan``, ``epistasis``)
  7. FastAPI ``TestClient`` against ``app/backend/app.py`` (``/api/health``,
     ``/api/ddg`` with ``wt_sequence`` + ``compare_modes`` + ``structure_pdb``,
     ``/api/scan``, ``/api/epistasis``)
  8. ``stage_block4_dataset_zip`` + legacy ``g4_report.json`` resume +
     ``g4_unified_suite.py`` compatibility shim

    python -m tests.test_a100_suite_smoke
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.encoder import EncodingStore, N_TAP_LAYERS, seq_key
from src.features import KD_HYDROPATHY
from src.model import HoloGeoN
from tests.helpers import check, random_protein, section, stub_encoding

STUB_DIM = 320  # matches facebook/esm2_t6_8M_UR50D in ESM_DIMS
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _run(cmd: list[str], label: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    --- {label} (rc={proc.returncode}) stdout ---\n{proc.stdout[-1500:]}")
        print(f"    --- {label} (rc={proc.returncode}) stderr ---\n{proc.stderr[-1500:]}")
    return proc


def _write_synthetic_pdb(path: Path, seq: str, chain: str = "A", seed: int = 0) -> str:
    """Write a minimal valid backbone PDB (N, CA, C, O, CB) with pLDDT in B-factor."""
    rng = np.random.default_rng(seed)
    lines = []
    atom_id = 1
    for i, aa in enumerate(seq, start=1):
        res3 = AA1_TO_3.get(aa, "ALA")
        cx = i * 3.8
        cy = float(np.sin(i * 0.6) * 4.0 + rng.normal(0, 0.2))
        cz = float(np.cos(i * 0.6) * 4.0 + rng.normal(0, 0.2))
        bfac = 92.0 if i > 3 else 62.0
        for name, dx, dy, dz, elem in (
            ("N", -1.2, 0.0, 0.0, "N"),
            ("CA", 0.0, 0.0, 0.0, "C"),
            ("C", 1.2, 0.3, 0.0, "C"),
            ("O", 1.8, 1.3, 0.0, "O"),
            ("CB", 0.0, -1.2, 0.6, "C"),
        ):
            if aa == "G" and name == "CB":
                continue
            lines.append(
                f"ATOM  {atom_id:5d}  {name:<3s} {res3:>3s} {chain}{i:4d}    "
                f"{cx + dx:8.3f}{cy + dy:8.3f}{cz + dz:8.3f}  1.00{bfac:6.2f}           {elem:>2s}"
            )
            atom_id += 1
    lines.append("END")
    txt = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")
    return txt


def _make_stub_record(seq: str, dim: int = STUB_DIM, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    rec = stub_encoding(seq, dim=dim, top_k=8)
    L = len(seq)
    rec["layer_embs"] = torch.randn(N_TAP_LAYERS, L, dim, generator=g).half()
    rec["wt_logprobs"] = torch.log_softmax(torch.randn(L, 20, generator=g), dim=-1).half()
    rec["site_entropy"] = torch.rand(L, generator=g).half()
    return rec


def _write_tiny_v2_config(path: Path, cache_dir: Path, window: int = 48) -> None:
    path.write_text(
        f"""
esm_model: facebook/esm2_t6_8M_UR50D
cache: {cache_dir.as_posix()}
device: cpu
window: {window}
contact_top_k: 8
contact_min_sep: 3
out_root: runs
max_protein_length: 5000
seed: 42
strict_determinism: false
graph_top_k: 8
geo_threshold: 0.30
eps: 0.10
delta: 0.10
mech_max_degree: 8
neighborhood_hops: 2
arch: v2
hidden: 32
out_dim: 32
gat_heads: 2
aa_dim: 8
dropout: 0.1
feat_dropout: 0.1
drop_edge: 0.0
use_ssm: false
use_mech_stream: false
use_geo_stream: true
use_str_stream: false
true_contact_edges: false
use_multi_layer: false
use_contact_burial: true
use_esm_logits_prior: true
use_contact_gate: true
site_anchored_readout: true
epochs: 2
batch_size: 16
protein_batch: true
proteins_per_batch: 4
muts_per_protein: 16
lr: 0.002
weight_decay: 0.001
warmup_frac: 0.1
early_stop_patience: 5
grad_clip: 1.0
amp: false
num_workers: 0
nll_beta: 0.5
cds_dropout: 0.0
""".lstrip(),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Train (v2-Lean + true_contacts) -> Export -> External -> Calibration
# ---------------------------------------------------------------------------
def test_train_export_external_calibration() -> None:
    section("smoke 1: train (v2-Lean & true_contacts) + export + eval_external + calibration")
    with tempfile.TemporaryDirectory(prefix="holo_a100_smoke1_") as tmp:
        td = Path(tmp)
        cache_dir = td / "cache"
        tc_dir = td / "true_contacts"
        tc_dir.mkdir(parents=True, exist_ok=True)
        store = EncodingStore(str(cache_dir))

        rng = np.random.default_rng(101)
        split_rows = []
        split_plan = [("train", 6), ("val", 3), ("test", 3)]
        p_idx = 0
        for sp, count in split_plan:
            for _ in range(count):
                L = int(rng.integers(32, 44))
                seq = random_protein(L, seed=300 + p_idx)
                store.put(_make_stub_record(seq, dim=STUB_DIM, seed=300 + p_idx))
                # Synthetic symmetric 8A contact map
                cmap = np.zeros((L, L), dtype=np.float32)
                for i in range(L):
                    for j in range(i + 4, min(L, i + 10), 3):
                        cmap[i, j] = cmap[j, i] = 1.0
                np.save(tc_dir / f"{seq_key(seq)}.npy", cmap)

                for _m in range(12):
                    site = int(rng.integers(0, L))
                    wt = seq[site]
                    mt = AA20[int(rng.integers(0, 20))]
                    if mt == wt:
                        mt = "A" if wt != "A" else "C"
                    ddg = float(0.35 * (KD_HYDROPATHY[mt] - KD_HYDROPATHY[wt]) + rng.normal(0, 0.05))
                    split_rows.append({
                        "parent": f"P{p_idx:03d}_A",
                        "parent_seq": seq,
                        "site": site,
                        "wt_aa": wt,
                        "mt_aa": mt,
                        "target": ddg,
                        "sigma": 0.3,
                        "split": sp,
                        "group_used": f"C{p_idx}",
                    })
                p_idx += 1

        splits_pq = td / "splits_megascale_id25.parquet"
        pd.DataFrame(split_rows).to_parquet(splits_pq, index=False)

        # External holdout mixing short (L=38 <= window=48) and long (L=72 > window=48) proteins
        ext_rows = []
        for ei, L in enumerate((38, 42, 72)):
            seq = random_protein(L, seed=900 + ei)
            store.put(_make_stub_record(seq, dim=STUB_DIM, seed=900 + ei))
            for _m in range(15):
                site = int(rng.integers(0, L))
                wt = seq[site]
                mt = AA20[int(rng.integers(0, 20))]
                if mt == wt:
                    mt = "A" if wt != "A" else "C"
                mut_seq = seq[:site] + mt + seq[site + 1:]
                ddg = float(0.3 * (KD_HYDROPATHY[mt] - KD_HYDROPATHY[wt]))
                ext_rows.append({
                    "parent": f"EXT{ei}_A",
                    "wt_seq": seq,
                    "mt_seq": mut_seq,
                    "ddg": ddg,
                })
        ext_pq = td / "s669_holdout.parquet"
        pd.DataFrame(ext_rows).to_parquet(ext_pq, index=False)

        cfg_path = td / "v2_tiny.yaml"
        _write_tiny_v2_config(cfg_path, cache_dir, window=48)

        # 1a. Train v2-Lean
        out_lean = td / "v2_lean"
        p_tr1 = _run([
            sys.executable, "-m", "scripts.train",
            "--config", str(cfg_path),
            "--stability-splits", str(splits_pq),
            "--pathogenicity-splits", "__none__",
            "--cache", str(cache_dir),
            "--out", str(out_lean),
            "--seed", "42",
            "--no-mech-stream", "--no-ssm", "--no-multi-layer",
            "--epochs", "2", "--num-workers", "0",
        ], "train v2-Lean")
        ckpt_lean = out_lean / "model_best.pt"
        check(p_tr1.returncode == 0 and ckpt_lean.is_file() and (out_lean / "metrics.json").is_file(),
              "scripts.train (v2-Lean) succeeded and saved model_best.pt + metrics.json")

        # 1b. Train v2-Lean + true_contacts
        out_tc = td / "v2_lean_tc"
        p_tr2 = _run([
            sys.executable, "-m", "scripts.train",
            "--config", str(cfg_path),
            "--stability-splits", str(splits_pq),
            "--pathogenicity-splits", "__none__",
            "--cache", str(cache_dir),
            "--out", str(out_tc),
            "--seed", "42",
            "--no-mech-stream", "--no-ssm", "--no-multi-layer",
            "--true-contact-edges", "--true-contacts-dir", str(tc_dir),
            "--epochs", "2", "--num-workers", "0",
        ], "train v2-Lean true_contacts")
        check(p_tr2.returncode == 0 and (out_tc / "model_best.pt").is_file(),
              "scripts.train (v2-Lean + true_contacts) succeeded")

        # 2. export_predictions fast-path vs --rowwise exact equivalence (< 1e-5)
        exp_fast = out_lean / "pred_test_fast.parquet"
        exp_row = out_lean / "pred_test_rowwise.parquet"
        p_ef = _run([
            sys.executable, "-m", "scripts.export_predictions",
            "--checkpoint", str(ckpt_lean), "--splits", str(splits_pq), "--split", "test",
            "--config", str(cfg_path), "--cache", str(cache_dir), "--out", str(exp_fast),
        ], "export fast")
        p_er = _run([
            sys.executable, "-m", "scripts.export_predictions",
            "--checkpoint", str(ckpt_lean), "--splits", str(splits_pq), "--split", "test",
            "--config", str(cfg_path), "--cache", str(cache_dir), "--out", str(exp_row), "--rowwise",
        ], "export rowwise")
        check(p_ef.returncode == 0 and p_er.returncode == 0, "export_predictions succeeded in both modes")
        df_ef = pd.read_parquet(exp_fast)
        df_er = pd.read_parquet(exp_row)
        d_exp_mu = float(np.max(np.abs(df_ef["y_pred"].to_numpy() - df_er["y_pred"].to_numpy())))
        d_exp_sg = float(np.max(np.abs(df_ef["y_sigma"].to_numpy() - df_er["y_sigma"].to_numpy())))
        check(np.array_equal(df_ef["row_id"].to_numpy(), df_er["row_id"].to_numpy())
              and d_exp_mu < 1e-5 and d_exp_sg < 1e-5,
              f"export_predictions fast == rowwise (max|dmu|={d_exp_mu:.2e}, max|dsg|={d_exp_sg:.2e} < 1e-5)")

        # 3. eval_external fast-path (short + >window fallback) vs --rowwise exact equivalence (< 1e-5)
        ext_fast_dir = out_lean / "ext_fast"
        ext_row_dir = out_lean / "ext_row"
        p_xf = _run([
            sys.executable, "-m", "scripts.eval_external",
            "--checkpoint", str(ckpt_lean), "--data", str(ext_pq),
            "--config", str(cfg_path), "--cache", str(cache_dir), "--out", str(ext_fast_dir),
        ], "eval_external fast")
        p_xr = _run([
            sys.executable, "-m", "scripts.eval_external",
            "--checkpoint", str(ckpt_lean), "--data", str(ext_pq),
            "--config", str(cfg_path), "--cache", str(cache_dir), "--out", str(ext_row_dir), "--rowwise",
        ], "eval_external rowwise")
        check(p_xf.returncode == 0 and p_xr.returncode == 0, "eval_external succeeded in both modes")
        pf_df = pd.read_parquet(ext_fast_dir / "pred_s669_holdout.parquet")
        pr_df = pd.read_parquet(ext_row_dir / "pred_s669_holdout.parquet")
        d_ext_mu = float(np.max(np.abs(pf_df["y_pred"].to_numpy() - pr_df["y_pred"].to_numpy())))
        d_ext_sg = float(np.max(np.abs(pf_df["y_sigma"].to_numpy() - pr_df["y_sigma"].to_numpy())))
        check(np.array_equal(pf_df["row_id"].to_numpy(), pr_df["row_id"].to_numpy())
              and d_ext_mu < 1e-5 and d_ext_sg < 1e-5,
              f"eval_external fast == rowwise on mixed <=win/>win (max|dmu|={d_ext_mu:.2e}, max|dsg|={d_ext_sg:.2e} < 1e-5)")

        # 4. scripts.calibration
        p_cal = _run([
            sys.executable, "-m", "scripts.calibration",
            "--checkpoint", str(ckpt_lean), "--splits", str(splits_pq),
            "--config", str(cfg_path), "--cache", str(cache_dir), "--out", str(out_lean),
        ], "calibration")
        cal_json = out_lean / "calibration.json"
        check(p_cal.returncode == 0 and cal_json.is_file(), "scripts.calibration wrote calibration.json")
        if cal_json.is_file():
            cal = json.loads(cal_json.read_text(encoding="utf-8"))
            sc = (cal.get("variance_scale") or {}).get("scale")
            check(sc is not None and np.isfinite(sc) and sc > 0,
                  f"calibration fitted positive finite variance scale ({sc})")


# ---------------------------------------------------------------------------
# 2. Competitors smoke (spurs, rasp, prostata, ddgemb + upstream guard)
# ---------------------------------------------------------------------------
def test_competitors_suite_smoke() -> None:
    section("smoke 2: competitors (spurs, rasp, prostata, ddgemb + spurs official-external-only guard)")
    with tempfile.TemporaryDirectory(prefix="holo_a100_smoke2_") as tmp:
        td = Path(tmp)
        cache_dir = td / "cache"
        pdb_ms = td / "pdbs_ms"
        pdb_ext1 = td / "pdbs_ext1"
        pdb_ext2 = td / "pdbs_ext2"
        store = EncodingStore(str(cache_dir))

        rng = np.random.default_rng(202)
        split_rows, pair_rows = [], []
        for pi, (sp, chain) in enumerate([
            ("train", "A"), ("train", "A"), ("train", "A"),
            ("val", "A"), ("test", "B"),
        ]):
            L = 32
            seq = random_protein(L, seed=700 + pi)
            parent = f"1M{pi:02d}_{chain}"
            store.put(_make_stub_record(seq, dim=32, seed=700 + pi))
            _write_synthetic_pdb(pdb_ms / f"{parent}.pdb", seq, chain=chain, seed=700 + pi)
            for _m in range(10):
                site = int(rng.integers(0, L))
                wt = seq[site]
                mt = AA20[int(rng.integers(0, 20))]
                if mt == wt:
                    mt = "A" if wt != "A" else "C"
                ddg = float(0.3 * (KD_HYDROPATHY[mt] - KD_HYDROPATHY[wt]))
                split_rows.append({
                    "parent": parent, "parent_seq": seq, "site": site,
                    "wt_aa": wt, "mt_aa": mt, "target": ddg, "split": sp,
                })
                pair_rows.append({
                    "parent": parent, "wt_seq": seq,
                    "mt_seq": seq[:site] + mt + seq[site + 1:], "ddg": ddg,
                })

        splits_pq = td / "splits.parquet"
        pairs_pq = td / "pairs.parquet"
        pd.DataFrame(split_rows).to_parquet(splits_pq, index=False)
        pd.DataFrame(pair_rows).to_parquet(pairs_pq, index=False)

        # Two externals: ext1 (primary) + ext2 (--extra-externals)
        def _make_ext(pq: Path, pdir: Path, tag: str, chain: str, seed_base: int):
            rows = []
            for k in range(2):
                L = 30
                seq = random_protein(L, seed=seed_base + k)
                par = f"{tag}{k}_{chain}"
                store.put(_make_stub_record(seq, dim=32, seed=seed_base + k))
                _write_synthetic_pdb(pdir / f"{par}.pdb", seq, chain=chain, seed=seed_base + k)
                for _m in range(8):
                    site = int(rng.integers(0, L))
                    wt = seq[site]
                    mt = AA20[int(rng.integers(0, 20))]
                    if mt == wt:
                        mt = "A" if wt != "A" else "C"
                    rows.append({
                        "parent": par, "parent_seq": seq, "wt_seq": seq,
                        "mt_seq": seq[:site] + mt + seq[site + 1:],
                        "site": site, "wt_aa": wt, "mt_aa": mt,
                        "target": float(0.25 * (KD_HYDROPATHY[mt] - KD_HYDROPATHY[wt])),
                    })
            pd.DataFrame(rows).to_parquet(pq, index=False)

        ext1_pq = td / "fireprotdb_hf25.parquet"
        ext2_pq = td / "s669_holdout.parquet"
        _make_ext(ext1_pq, pdb_ext1, "1FP", "A", 810)
        _make_ext(ext2_pq, pdb_ext2, "1S6", "B", 820)

        # 2a. SPURS --mode upstream --official-external-only clean non-zero + status JSON (Bug G!)
        out_spurs_off = td / "spurs_official"
        fake_spurs_dir = td / "fake_spurs_repo"
        (fake_spurs_dir / ".git").mkdir(parents=True, exist_ok=True)
        p_so = _run([
            sys.executable, "-m", "scripts.competitors.spurs",
            "--cache", str(cache_dir),
            "--mode", "upstream", "--official-external-only",
            "--work-dir", str(fake_spurs_dir),
            "--external", str(ext1_pq), "--external-pdb-dir", str(pdb_ext1),
            "--extra-externals", f"s669={ext2_pq}::{pdb_ext2}",
            "--out", str(out_spurs_off),
        ], "spurs official-external-only")
        combined_so = (p_so.stdout or "") + (p_so.stderr or "")
        check(p_so.returncode != 0, "spurs --mode upstream --official-external-only exits non-zero cleanly when upstream unavailable")
        check("FileNotFoundError" not in combined_so and "splits_megascale" not in combined_so,
              "spurs official-external-only never falls through to MegaScale retrain")
        check((out_spurs_off / "official_spurs_status.json").is_file(),
              "spurs writes official_spurs_status.json with diagnostics")

        # 2b. Retrained competitors with --extra-externals (1 epoch CPU)
        for name, cmd, extra_pred in [
            ("spurs_local", [
                sys.executable, "-m", "scripts.competitors.spurs",
                "--splits", str(splits_pq), "--pairs", str(pairs_pq),
                "--cache", str(cache_dir), "--pdb-dir", str(pdb_ms),
                "--external", str(ext1_pq), "--external-pdb-dir", str(pdb_ext1),
                "--extra-externals", f"s669={ext2_pq}::{pdb_ext2}",
                "--epochs", "1", "--seed", "42", "--out", str(td / "spurs_local"),
            ], td / "spurs_local" / "pred_s669.parquet"),
            ("rasp", [
                sys.executable, "-m", "scripts.competitors.rasp",
                "--splits", str(splits_pq), "--pairs", str(pairs_pq),
                "--pdb-dir", str(pdb_ms),
                "--external", str(ext1_pq), "--external-pdb-dir", str(pdb_ext1),
                "--extra-externals", f"s669={ext2_pq}::{pdb_ext2}",
                "--epochs", "1", "--seed", "42", "--out", str(td / "rasp"),
            ], td / "rasp" / "pred_s669.parquet"),
            ("prostata", [
                sys.executable, "-m", "scripts.competitors.prostata",
                "--splits", str(splits_pq), "--cache", str(cache_dir),
                "--external", str(ext1_pq),
                "--extra-externals", f"s669={ext2_pq}",
                "--epochs", "1", "--seed", "42", "--out", str(td / "prostata"),
            ], td / "prostata" / "pred_s669.parquet"),
            ("ddgemb", [
                sys.executable, "-m", "scripts.competitors.ddgemb",
                "--splits", str(splits_pq), "--cache", str(cache_dir),
                "--external", str(ext1_pq),
                "--extra-externals", f"s669={ext2_pq}",
                "--epochs", "1", "--seed", "42", "--out", str(td / "ddgemb"),
            ], td / "ddgemb" / "pred_s669.parquet"),
        ]:
            proc = _run(cmd, f"competitor {name}")
            out_dir = td / name
            check(
                proc.returncode == 0
                and (out_dir / "pred_test.parquet").is_file()
                and extra_pred.is_file(),
                f"competitor {name} ran 1 epoch + wrote pred_test & pred_s669",
            )


# ---------------------------------------------------------------------------
# 3. stage_block2_prostata_ft helpers (_group_by_parent, _eval_groups, drift)
# ---------------------------------------------------------------------------
def test_prostata_ft_helpers() -> None:
    section("smoke 3: stage_block2_prostata_ft helpers & edge-case safety")
    from remote.gpu_jobs.a100_unified_suite import (
        _eval_groups,
        _group_by_parent,
        _measure_contact_drift,
        _safe_pearson,
    )
    from scripts.competitors.prostata import ProstataFiveHead

    seq1 = random_protein(30, seed=11)
    seq2 = random_protein(35, seed=12)
    df = pd.DataFrame([
        {"parent_seq": seq1, "site": 2, "wt_aa": seq1[2], "mt_aa": "A", "target": -1.2},
        {"parent_seq": seq1, "site": 999, "wt_aa": seq1[5], "mt_aa": "V", "target": 0.4},
        # Non-canonical AA and NaN rows must be safely filtered out, never crash!
        {"parent_seq": seq1, "site": 4, "wt_aa": "X", "mt_aa": "A", "target": -0.5},
        {"parent_seq": seq1, "site": 4, "wt_aa": "A", "mt_aa": "U", "target": -0.5},
        {"parent_seq": seq1, "site": 4, "wt_aa": None, "mt_aa": "A", "target": -0.5},
        {"parent_seq": "", "site": 0, "wt_aa": "A", "mt_aa": "C", "target": -0.5},
        {"parent_seq": seq2, "site": 10, "wt_aa": seq2[10], "mt_aa": "L", "target": 0.8},
    ])
    groups = _group_by_parent(df, drop_nan_target=True)
    check(len(groups) == 2 and len(groups[0]["sites"]) == 2 and len(groups[1]["sites"]) == 1,
          "_group_by_parent filters non-canonical AAs/None/empty seq and clips sites")
    check(int(groups[0]["sites"][1]) == len(seq1) - 1, "_group_by_parent clamps out-of-range site")

    # Mock tiny tokenizer & ESM / MLM
    def _mock_tok(seq: str, return_tensors="pt", add_special_tokens=True):
        ids = torch.arange(len(seq) + 2, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class _MockContactHead(nn.Module):
        def forward(self, input_ids, attns):
            L = input_ids.shape[1] - 2
            # Return symmetric scores with strongest signal on (i, i+6) and (i, i+24)
            m = torch.zeros(1, L, L)
            for i in range(L):
                if i + 6 < L:
                    m[0, i, i + 6] = m[0, i + 6, i] = 0.95
                if i + 24 < L:
                    m[0, i, i + 24] = m[0, i + 24, i] = 0.99
            return m

    class _MockEsm(nn.Module):
        def __init__(self, dim: int = 32):
            super().__init__()
            self.dim = dim
            self.contact_head = _MockContactHead()
            self.w = nn.Parameter(torch.ones(1))

        def forward(self, input_ids, attention_mask=None, output_attentions=False):
            B, T = input_ids.shape
            h = torch.linspace(-1, 1, B * T * self.dim).reshape(B, T, self.dim) * self.w
            attns = (torch.ones(B, 2, T, T),) if output_attentions else None
            return types.SimpleNamespace(last_hidden_state=h, attentions=attns)

    class _MockMlm(nn.Module):
        def __init__(self, dim: int = 32):
            super().__init__()
            self.esm = _MockEsm(dim)

    mlm = _MockMlm(32)
    head = ProstataFiveHead(32, hidden=16)
    dev = torch.device("cpu")

    rids, preds, ys = _eval_groups(mlm.esm, head, groups, _mock_tok, dev)
    check(len(rids) == 3 and len(preds) == 3 and bool(np.isfinite(preds).all()),
          "_eval_groups runs cleanly on CPU and returns aligned finite predictions")
    e_r, e_p, e_y = _eval_groups(mlm.esm, head, [], _mock_tok, dev)
    check(len(e_r) == 0 and len(e_p) == 0 and len(e_y) == 0,
          "_eval_groups handles empty groups without crashing")
    check(np.isnan(_safe_pearson(np.array([1.0, 1.0]), np.array([2.0, 3.0]))),
          "_safe_pearson returns NaN cleanly on constant arrays")

    with tempfile.TemporaryDirectory(prefix="holo_drift_") as tmp:
        tc_dir = Path(tmp)
        for s in (seq1, seq2):
            L = len(s)
            cmap = np.zeros((L, L), dtype=np.float32)
            for i in range(L):
                if i + 6 < L:
                    cmap[i, i + 6] = cmap[i + 6, i] = 1.0
                if i + 24 < L:
                    cmap[i, i + 24] = cmap[i + 24, i] = 1.0
            np.save(tc_dir / f"{seq_key(s)}.npy", cmap)
        drift = _measure_contact_drift(mlm, groups, _mock_tok, dev, tc_dir)
        check(
            drift["n_proteins_scored"] == 2
            and drift["contact_p_at_L_sep6"] is not None
            and 0.0 <= drift["contact_p_at_L_sep6"] <= 1.0
            and drift["contact_p_at_L5_sep24"] == 1.0,
            f"_measure_contact_drift scored 2 proteins accurately ({drift})",
        )


# ---------------------------------------------------------------------------
# 4. holo.py CLI + FastAPI TestClient
# ---------------------------------------------------------------------------
def test_holo_cli_and_fastapi_testclient() -> None:
    section("smoke 4: holo.py CLI + FastAPI TestClient (/health, /ddg, /scan, /epistasis)")
    import holo
    import src.encoder as _enc

    ubq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    gb1 = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

    class _StubEnc:
        def __init__(self, *a, **k):
            self.dim = 32

        def encode(self, seq, top_k=24, min_sep=3):
            return _make_stub_record(seq, dim=32, seed=len(seq))

    def _save_ckpt(path: Path, *, epistasis: bool = False, str_stream: bool = False):
        torch.manual_seed(0)
        kw = {
            "arch": "v2", "esm_dim": 32, "hidden": 32, "out_dim": 32,
            "aa_dim": 8, "gat_heads": 4, "dropout": 0.1,
            "use_ssm": False, "use_mech_stream": False, "use_geo_stream": True,
            "use_multi_layer": False, "use_contact_burial": True, "use_esm_logits_prior": True,
        }
        cfg = {
            "arch": "v2", "esm_model": "stub", "window": 96, "contact_top_k": 8,
            "contact_min_sep": 3, "graph_top_k": 8, "geo_threshold": 0.3,
            "eps": 0.1, "delta": 0.1, "mech_max_degree": 8, "neighborhood_hops": 2,
        }
        if str_stream:
            kw["use_str_stream"] = True
            cfg["use_str_stream"] = True
        m = HoloGeoN(**kw)
        torch.save({
            "model": m.state_dict(), "model_kwargs": kw, "esm_dim": 32, "config": cfg,
            "trained_tasks": ["stability"] + (["epistasis"] if epistasis else []),
            "untrained_heads": ["idr", "mfi", "proteomics"],
        }, path)

    orig_enc = _enc.EsmEncoder
    _enc.EsmEncoder = _StubEnc
    try:
        with tempfile.TemporaryDirectory(prefix="holo_cli_api_", ignore_cleanup_errors=True) as tmp:
            td = Path(tmp)
            ckpt_main = td / "v2_lean.pt"
            ckpt_3d = td / "v2_3d.pt"
            _save_ckpt(ckpt_main, epistasis=True, str_stream=False)
            _save_ckpt(ckpt_3d, epistasis=False, str_stream=True)

            # holo.py CLI checks
            fa = td / "ubq.fasta"
            fa.write_text(f">Ubq\n{ubq}\n", encoding="utf-8")
            check(holo.main(["doctor", "--weights", str(ckpt_main)]) == 0, "holo.py doctor rc=0")
            pred_j = td / "pred.json"
            check(
                holo.main(["predict", "--weights", str(ckpt_main), "--device", "cpu",
                           "--fasta", str(fa), "--muts", "L8P,V5A", "--out", str(pred_j)]) == 0
                and pred_j.is_file(),
                "holo.py predict wrote JSON",
            )
            scan_csv = td / "scan.csv"
            scan_html = td / "scan.html"
            check(
                holo.main(["scan", "--weights", str(ckpt_main), "--device", "cpu",
                           "--fasta", str(fa), "--start", "1", "--end", "6",
                           "--out", str(scan_csv), "--html", str(scan_html)]) == 0
                and scan_csv.is_file() and scan_html.is_file(),
                "holo.py scan wrote CSV + HTML",
            )
            epi_csv = td / "epi.csv"
            check(
                holo.main(["epistasis", "--weights", str(ckpt_main), "--device", "cpu",
                           "--seq", gb1, "--pairs", "V39A+D40A,L5A+V54A", "--out", str(epi_csv)]) == 0
                and epi_csv.is_file(),
                "holo.py epistasis wrote CSV",
            )

            # FastAPI TestClient checks
            from fastapi.testclient import TestClient

            from tests.test_a100_ui_capture import load_backend_app

            backend_app = load_backend_app()
            import db as backend_db  # type: ignore
            import inference as backend_inf  # type: ignore

            orig_db = backend_db.DB_PATH
            orig_w = backend_inf.WEIGHTS_PATH
            orig_w3 = backend_inf.WEIGHTS_3D_PATH
            try:
                backend_db.DB_PATH = td / "test_hist.db"
                backend_db.init_db()
                backend_inf.WEIGHTS_PATH = ckpt_main
                backend_inf.WEIGHTS_3D_PATH = str(ckpt_3d)
                backend_inf.get_engine.cache_clear()

                client = TestClient(backend_app.app)
                r_h = client.get("/api/health")
                check(r_h.status_code == 200 and r_h.json().get("ready") is True,
                      "TestClient GET /api/health ready=True")

                # Bug A regression: wrong key `sequence` returns 422; `wt_sequence` returns 200
                r_bad = client.post("/api/ddg", json={"sequence": ubq, "mutation": "L8P"})
                check(r_bad.status_code == 422, "POST /api/ddg rejects wrong key `sequence` (HTTP 422)")

                ubq_pdb = _write_synthetic_pdb(td / "1UBQ_A.pdb", ubq, chain="A", seed=1)
                r_ddg = client.post("/api/ddg", json={"wt_sequence": ubq, "mutation": "L8P"})
                check(r_ddg.status_code == 200 and np.isfinite(r_ddg.json()["ddg"]),
                      "POST /api/ddg with wt_sequence returns 200 + finite ddG")

                r_cmp = client.post("/api/ddg", json={
                    "wt_sequence": ubq, "mutation": "L8P",
                    "compare_modes": True, "structure_pdb": ubq_pdb,
                })
                check(
                    r_cmp.status_code == 200
                    and "comparison" in r_cmp.json()
                    and "invariant_3d" in r_cmp.json()["comparison"],
                    "POST /api/ddg compare_modes=True + structure_pdb returns both modes",
                )

                r_scan = client.post("/api/scan", json={"sequence": ubq, "start": 1, "end": 5})
                check(r_scan.status_code == 200 and len(r_scan.json()["positions"]) == 5,
                      "POST /api/scan returns 200 with 5 positions")

                r_epi = client.post("/api/epistasis", json={
                    "wt_sequence": gb1, "pairs": ["V39A+D40A", "L5A+V54A"],
                })
                check(
                    r_epi.status_code == 200
                    and len(r_epi.json()["pairs"]) == 2
                    and np.isfinite(r_epi.json()["pairs"][0]["epistasis_dddg"]),
                    "POST /api/epistasis returns 200 with 2 finite pairs",
                )
            finally:
                try:
                    client.close()
                except Exception:
                    pass
                backend_db.DB_PATH = orig_db
                backend_inf.WEIGHTS_PATH = orig_w
                backend_inf.WEIGHTS_3D_PATH = orig_w3
                backend_inf.get_engine.cache_clear()
                gc.collect()
    finally:
        _enc.EsmEncoder = orig_enc


# ---------------------------------------------------------------------------
# 5. stage_block4_dataset_zip + g4_report.json resume + g4 shim
# ---------------------------------------------------------------------------
def test_block4_zip_and_resume_shim() -> None:
    section("smoke 5: stage_block4_dataset_zip (streaming SHA-256 + dynamic GPU + g4 resume/shim)")
    from remote.gpu_jobs.a100_unified_suite import A100Suite
    from remote.gpu_jobs.g4_unified_suite import G4Suite

    check(G4Suite is A100Suite, "g4_unified_suite.py cleanly aliases G4Suite -> A100Suite")

    with tempfile.TemporaryDirectory(prefix="holo_zip_smoke_") as tmp:
        td = Path(tmp)
        drive_root = td / "drive"
        work_root = td / "work"
        drive_out = drive_root / "runs" / "g4_suite"
        drive_out.mkdir(parents=True, exist_ok=True)

        # Simulate pre-existing legacy g4_report.json with already-completed Drive items
        legacy_report = {
            "created_utc": "2026-11-27T00:00:00Z",
            "stages": {"prepare_inputs": {"manifest_files_ok": 27}},
            "items": {
                "block1_speed": {"status": "ok"},
                "block1_parity/official_thermompnn": {"status": "ok"},
                "block1_parity/id25/v2_lean_s42": {"status": "ok"},
            },
            "errors": [],
            "environment": {"gpu_name": "NVIDIA A100-SXM4-40GB", "gpu_vram_gb": 42.47},
        }
        (drive_out / "g4_report.json").write_text(json.dumps(legacy_report), encoding="utf-8")

        # Populate representative outputs on Drive
        (drive_out / "block1_speed").mkdir(parents=True, exist_ok=True)
        (drive_out / "block1_speed" / "speed_vram_microbench.json").write_text('{"ok": true}', encoding="utf-8")
        (drive_out / "block1_parity" / "id25" / "v2_lean_s42").mkdir(parents=True, exist_ok=True)
        (drive_out / "block1_parity" / "id25" / "v2_lean_s42" / "model_best.pt").write_bytes(b"SKIP_ME")
        (drive_out / "block1_parity" / "id25" / "v2_lean_s42" / "last.pt").write_bytes(b"SKIP_LAST")
        (drive_out / "block1_parity" / "id25" / "v2_lean_s42" / "pred_test.parquet").write_bytes(b"PARQ_OK")
        (drive_out / "block3_demo" / "release_bundle").mkdir(parents=True, exist_ok=True)
        (drive_out / "block3_demo" / "release_bundle" / "hologe_v2_lean_best.pt").write_bytes(b"KEEP_RELEASE_PT")

        suite = A100Suite(drive_root, work_root, run_name="g4_suite")
        check(
            {"block1_speed", "block1_parity/official_thermompnn", "block1_parity/id25/v2_lean_s42"}
            <= set(suite.report["items"].keys()),
            "A100Suite resumes all completed items from legacy g4_report.json",
        )

        suite.stage_block4_dataset_zip()

        zip_path = drive_out / "holo_geon_a100_dataset_release.zip"
        check(zip_path.is_file(), "stage_block4_dataset_zip wrote release ZIP to Drive")
        check((drive_out / "a100_report.json").is_file(), "a100_report.json synced to Drive")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            readme = zf.read("Holo-GeoN-A100-Results/README_DATASET.md").decode("utf-8")
            shas = zf.read("Holo-GeoN-A100-Results/SHA256SUMS.txt").decode("utf-8").strip().splitlines()
            check("NVIDIA A100-SXM4-40GB" in readme and "A100 80 GB" not in readme,
                  "README_DATASET.md uses dynamic GPU name (no hardcoded A100 80 GB)")
            check("Holo-GeoN-A100-Results/block3_demo/release_bundle/hologe_v2_lean_best.pt" in names,
                  "release_bundle .pt included in ZIP")
            check("Holo-GeoN-A100-Results/block1_parity/id25/v2_lean_s42/model_best.pt" not in names
                  and "Holo-GeoN-A100-Results/block1_parity/id25/v2_lean_s42/last.pt" not in names,
                  "non-release .pt and last.pt excluded from ZIP")
            sha_ok = True
            for line in shas:
                digest, arc = line.split("  ", 1)
                actual = hashlib.sha256(zf.read(arc)).hexdigest()
                if digest != actual:
                    sha_ok = False
            check(len(shas) >= 3 and sha_ok, f"every entry in SHA256SUMS.txt ({len(shas)} files) verified exact")


TESTS = [
    test_train_export_external_calibration,
    test_competitors_suite_smoke,
    test_prostata_ft_helpers,
    test_holo_cli_and_fastapi_testclient,
    test_block4_zip_and_resume_shim,
]

if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
