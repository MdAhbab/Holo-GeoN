"""
tests/test_competitors_smoke.py
===============================
End-to-end CPU smoke tests & edge-case regressions for every competitor adapter
in ``scripts/competitors/`` (SPURS, RaSP, PROSTATA, DDGemb, ThermoMPNN, common):

  * BUG G: ``spurs --mode upstream --official-external-only`` NEVER falls
    through to MegaScale retrain; writes ``official_spurs_status.json`` with
    exact diagnostics and exits cleanly non-zero on failure.
  * SPURS upstream helpers: ``_torch_load_weights_only_false`` context manager
    restoration, real chain ID extraction (``1A0F_B -> B``, ``1UBQ_A -> A``,
    and first-ATOM detection), defensive ``model(pdb, ...)`` & matrix parsing,
    and end-to-end mock upstream inference with ``--extra-externals``.
  * End-to-end 1-epoch CPU training + inference for ``spurs`` (local_fusion),
    ``rasp``, ``prostata``, ``ddgemb``, and offline mock ``thermompnn``,
    covering ``--extra-externals`` (``tag=parquet`` and ``tag=parquet::pdb_dir``),
    ``domainome_apca_hf25`` schema + ``uniprot_id`` PDB fallback, missing
    columns, non-canonical AAs (``X``, ``U``, ``B``, ``Z``), sites outside
    sequence bounds, missing PDBs, constant predictions, ``bfloat16``/``float32``
    dtypes, and empty subsets (empty val, empty test, empty external).

Runnable standalone via ``python -m tests.test_competitors_smoke`` and
compatible with ``tests/run_all.py`` (exposes ``TESTS``).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tests.helpers import check, section

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _write_synthetic_pdb(path: Path, seq: str, chain: str = "A", bfactor: float = 92.0) -> Path:
    """Write a minimal valid backbone (N, CA, C, CB) PDB file for ``seq`` on ``chain``."""
    lines: list[str] = []
    atom_id = 1
    for i, aa in enumerate(seq):
        resname = ONE_TO_THREE.get(aa.upper(), "ALA")
        t = i * 0.9
        cx = 2.3 * np.cos(t)
        cy = 2.3 * np.sin(t)
        cz = 1.5 * i
        atoms = [
            ("N", cx - 0.8, cy, cz),
            ("CA", cx, cy, cz),
            ("C", cx + 0.8, cy, cz),
        ]
        if resname != "GLY":
            atoms.append(("CB", cx, cy + 0.9, cz + 0.2))
        for aname, x, y, z in atoms:
            lines.append(
                f"ATOM  {atom_id:5d}  {aname:<3s} {resname:>3s} {chain}{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           {aname[0]:>2s}\n"
            )
            atom_id += 1
    lines.append("END\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _make_tiny_dataset(td: Path, esm_dim: int = 16) -> dict[str, Path]:
    """Build tiny synthetic MegaScale splits, cache, PDBs, and external holdouts."""
    from src.encoder import EncodingStore

    seq_tr1 = "ACDEFGHIKLMNPQRS"  # L=16
    seq_tr2 = "MKTAYIAKQRQISFVK"  # L=16
    seq_va = "TVWYACDEFGHIKLMN"   # L=16
    seq_te = "PQRSACDEFGHIKLMN"   # L=16
    seq_ext = "GHIKLMNPQRSTVWYA"  # L=16
    seq_dom = "VCKVCGKGFRQASTLC"  # L=16 (domainome_apca_hf25 style)

    # EncodingStore cache (mix float16, float32, and bfloat16 to test dtype handling)
    cache_dir = td / "cache"
    store = EncodingStore(str(cache_dir))
    rng = np.random.default_rng(123)
    for idx, seq in enumerate((seq_tr1, seq_tr2, seq_va, seq_te, seq_ext, seq_dom)):
        L = len(seq)
        emb = torch.from_numpy(rng.normal(0.0, 0.5, (L, esm_dim)).astype(np.float32))
        if idx == 0:
            emb_store = emb.half()
        elif idx == 1:
            emb_store = emb.to(torch.bfloat16)
        else:
            emb_store = emb
        store.put({
            "seq": seq,
            "emb": emb_store,
            "contact_edges": torch.zeros((2, 0), dtype=torch.int32),
            "contact_probs": torch.zeros((0,), dtype=torch.float16),
            "dim": esm_dim,
        })

    # PDB directories
    ms_pdb_dir = td / "pdbs_ms"
    _write_synthetic_pdb(ms_pdb_dir / "1TR1.pdb", seq_tr1, chain="A")
    _write_synthetic_pdb(ms_pdb_dir / "1TR2.pdb", seq_tr2, chain="A")
    _write_synthetic_pdb(ms_pdb_dir / "1VAL.pdb", seq_va, chain="A")
    _write_synthetic_pdb(ms_pdb_dir / "1TST.pdb", seq_te, chain="A")
    # Note: deliberately omit a PDB for missing-parent rows in splits!

    ext_pdb_dir = td / "pdbs_ext"
    _write_synthetic_pdb(ext_pdb_dir / "1EXT.pdb", seq_ext, chain="B")
    _write_synthetic_pdb(ext_pdb_dir / "AF-A0PJY2-F1-model_v4.pdb", seq_dom, chain="A")

    # Splits parquet (includes non-canonical AA X/U/B/Z + out-of-bounds sites + missing-PDB parent)
    split_rows = []
    for s_idx, (seq, par, sp) in enumerate((
        (seq_tr1, "1TR1.pdb", "train"),
        (seq_tr2, "1TR2.pdb", "train"),
        (seq_va, "1VAL.pdb", "val"),
        (seq_te, "1TST.pdb", "test"),
    )):
        for pos in range(6):
            wt = seq[pos]
            mt = "G" if wt != "G" else "A"
            split_rows.append({
                "parent": par,
                "parent_seq": seq,
                "site": pos,
                "wt_aa": wt,
                "mt_aa": mt,
                "target": float(-0.3 * (pos + 1) + 0.1 * s_idx),
                "split": sp,
                "group_used": f"grp_{s_idx}",
            })
    # Add edge-case rows to train & test: non-canonical AAs + out-of-bounds sites + missing PDB parent
    split_rows.extend([
        {"parent": "1TR1.pdb", "parent_seq": seq_tr1, "site": 2, "wt_aa": "D", "mt_aa": "X", "target": -0.5, "split": "train", "group_used": "grp_0"},
        {"parent": "1TR1.pdb", "parent_seq": seq_tr1, "site": 3, "wt_aa": "E", "mt_aa": "U", "target": -0.4, "split": "train", "group_used": "grp_0"},
        {"parent": "1TST.pdb", "parent_seq": seq_te, "site": 4, "wt_aa": "B", "mt_aa": "Z", "target": -0.2, "split": "test", "group_used": "grp_3"},
        # Out-of-bounds sites (must be filtered cleanly, never clipped!)
        {"parent": "1TST.pdb", "parent_seq": seq_te, "site": -1, "wt_aa": "P", "mt_aa": "A", "target": -1.0, "split": "test", "group_used": "grp_3"},
        {"parent": "1TST.pdb", "parent_seq": seq_te, "site": 999, "wt_aa": "P", "mt_aa": "A", "target": -1.0, "split": "test", "group_used": "grp_3"},
        # Missing PDB parent (still in cache for seq models; skipped cleanly by PDB models)
        {"parent": "9MISSING.pdb", "parent_seq": seq_te, "site": 1, "wt_aa": seq_te[1], "mt_aa": "A", "target": -0.8, "split": "test", "group_used": "grp_4"},
    ])
    splits_pq = td / "splits.parquet"
    pd.DataFrame(split_rows).to_parquet(splits_pq, index=False)

    # Splits with EMPTY val split & constant-target val split
    splits_empty_val_pq = td / "splits_empty_val.parquet"
    df_ev = pd.DataFrame(split_rows)
    df_ev[df_ev["split"] != "val"].to_parquet(splits_empty_val_pq, index=False)

    # External holdout (FireProtDB-like with wt_seq + mt_seq, NO target/ddg column for inference-only check!)
    ext_pq = td / "ext_fireprot_like.parquet"
    pd.DataFrame([
        {"parent": "1EXT_B", "wt_seq": seq_ext, "mt_seq": "AHIKLMNPQRSTVWYA"},  # site=0 G->A
        {"parent": "1EXT_B", "wt_seq": seq_ext, "mt_seq": "GAIKLMNPQRSTVWYA"},  # site=1 H->A
        {"parent": "1EXT_B", "wt_seq": seq_ext, "mt_seq": seq_ext},             # synonymous -> dropped
        {"parent": "9NOPDB", "wt_seq": seq_ext, "mt_seq": "GHAKLMNPQRSTVWYA"},  # missing PDB
    ]).to_parquet(ext_pq, index=False)

    # Domainome-HF25-like external table (parent=domain_ID, uniprot_id=A0PJY2, target without ddg)
    dom_pq = td / "domainome_apca_hf25.parquet"
    pd.DataFrame([
        {
            "parent": "A0PJY2_PF00096_289", "uniprot_id": "A0PJY2",
            "wt_seq": seq_dom, "parent_seq": seq_dom,
            "site": 0, "wt_aa": "V", "mt_aa": "A", "mutation": "V1A", "target": -0.25,
        },
        {
            "parent": "A0PJY2_PF00096_289", "uniprot_id": "A0PJY2",
            "wt_seq": seq_dom, "parent_seq": seq_dom,
            "site": 1, "wt_aa": "C", "mt_aa": "X", "mutation": "C2X", "target": -0.80,
        },
        {
            "parent": "A0PJY2_PF00096_289", "uniprot_id": "A0PJY2",
            "wt_seq": seq_dom, "parent_seq": seq_dom,
            "site": 500, "wt_aa": "C", "mt_aa": "A", "mutation": "C501A", "target": -0.10,  # out-of-bounds
        },
    ]).to_parquet(dom_pq, index=False)

    # Empty external table (0 rows)
    empty_pq = td / "empty_ext.parquet"
    pd.DataFrame({
        "parent": pd.Series([], dtype="object"),
        "wt_seq": pd.Series([], dtype="object"),
        "site": pd.Series([], dtype="int64"),
        "wt_aa": pd.Series([], dtype="object"),
        "mt_aa": pd.Series([], dtype="object"),
        "target": pd.Series([], dtype="float32"),
    }).to_parquet(empty_pq, index=False)

    return {
        "cache": cache_dir,
        "ms_pdbs": ms_pdb_dir,
        "ext_pdbs": ext_pdb_dir,
        "splits": splits_pq,
        "splits_empty_val": splits_empty_val_pq,
        "ext": ext_pq,
        "dom": dom_pq,
        "empty": empty_pq,
        "seq_ext": seq_ext,
        "seq_dom": seq_dom,
    }


def test_common_edge_cases() -> None:
    section("competitors_smoke - common helpers & edge cases")
    from scripts.competitors.common import (
        load_external_table,
        parse_extra_external_spec,
        safe_aa_index,
        safe_pearson,
        valid_site_mask,
    )
    from src.substitution import UNKNOWN_AA_IDX

    # parse_extra_external_spec
    check(parse_extra_external_spec("dom=a.parquet") == ("dom", "a.parquet", None),
          "parse_extra_external_spec handles tag=parquet")
    check(parse_extra_external_spec("s669=s.parquet::/tmp/pdbs") == ("s669", "s.parquet", "/tmp/pdbs"),
          "parse_extra_external_spec handles tag=parquet::pdb_dir")
    bad_raised = False
    try:
        parse_extra_external_spec("missing_equals_sign")
    except ValueError:
        bad_raised = True
    check(bad_raised, "parse_extra_external_spec rejects malformed spec without '='")

    # safe_aa_index on non-canonical AAs and nulls
    for aa in ("X", "U", "B", "Z", "O", None, float("nan"), ""):
        check(safe_aa_index(aa) == UNKNOWN_AA_IDX, f"safe_aa_index({aa!r}) maps cleanly to UNKNOWN_AA_IDX")

    # valid_site_mask
    mask, sites = valid_site_mask(pd.Series([-1, 0, 5, 10, 99, float("nan"), 2.5]), seq_len=10)
    check(mask.tolist() == [False, True, True, False, False, False, False] and sites.tolist() == [0, 5],
          "valid_site_mask keeps only finite integer sites in [0, L)")

    # safe_pearson: warning-free on empty, short, constant, NaN
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        r_empty = safe_pearson(np.array([]), np.array([]))
        r_one = safe_pearson(np.array([1.0]), np.array([2.0]))
        r_const = safe_pearson(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]))
        r_good = safe_pearson(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0]))
    runtime_warns = [w for w in wrec if issubclass(w.category, RuntimeWarning)]
    check(len(runtime_warns) == 0, "safe_pearson emits zero RuntimeWarnings on empty/short/constant inputs")
    check(np.isnan(r_empty) and np.isnan(r_one) and np.isnan(r_const),
          "safe_pearson returns nan for empty/short/constant arrays")
    check(abs(r_good - 1.0) < 1e-12, f"safe_pearson exact on valid linear inputs (got {r_good})")

    # load_external_table variants: mutation-code fallback, missing target, null seq, empty table
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        mut_pq = p / "mut_code.parquet"
        pd.DataFrame({
            "wt_seq": ["ACDEFG", "ACDEFG", None],
            "mutation": ["A1G", "Z2A", "C2A"],
            "ddG": [-1.2, -0.5, -0.3],
        }).to_parquet(mut_pq, index=False)
        df_m = load_external_table(str(mut_pq))
        check(len(df_m) == 1 and int(df_m["row_id"].iloc[0]) == 0 and int(df_m["site"].iloc[0]) == 0,
              "load_external_table parses mutation codes, handles ddG, and drops null seqs preserving row_id")

        notgt_pq = p / "no_tgt.parquet"
        pd.DataFrame({
            "parent_seq": ["ACDEFG"], "site": [1], "wt_aa": ["c"], "mt_aa": ["a"],
        }).to_parquet(notgt_pq, index=False)
        df_nt = load_external_table(str(notgt_pq))
        check("target" in df_nt.columns and "wt_seq" in df_nt.columns
              and df_nt["wt_aa"].iloc[0] == "C" and df_nt["mt_aa"].iloc[0] == "A",
              "load_external_table normalizes missing target/wt_seq and uppercases AA columns")


def test_spurs_upstream_bug_g_and_helpers() -> None:
    section("competitors_smoke - SPURS BUG G, chain extraction, torch.load patch & mock upstream")
    from scripts.competitors import spurs

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        ds = _make_tiny_dataset(tdp)

        # 1. Context manager restores torch.load on normal exit AND on exception
        orig_load = torch.load
        captured_kw: dict = {}

        def _fake_load(*a, **kw):
            captured_kw.update(kw)
            return "ok"

        torch.load = _fake_load  # type: ignore[assignment]
        try:
            with spurs._torch_load_weights_only_false():
                torch.load("dummy.ckpt")
            check(captured_kw.get("weights_only") is False,
                  "_torch_load_weights_only_false defaults weights_only=False inside block")
            check(torch.load is _fake_load,
                  "_torch_load_weights_only_false restores torch.load on normal exit")
            try:
                with spurs._torch_load_weights_only_false():
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            check(torch.load is _fake_load,
                  "_torch_load_weights_only_false restores torch.load even on exception")
        finally:
            torch.load = orig_load

        # 2. Chain ID extraction
        pdb_chain_d = _write_synthetic_pdb(tdp / "chain_d.pdb", "ACDEFG", chain="D")
        check(spurs._extract_chain_id("1A0F_B", pdb_chain_d) == "B", "chain extraction: 1A0F_B -> 'B'")
        check(spurs._extract_chain_id("1UBQ_A", pdb_chain_d) == "A", "chain extraction: 1UBQ_A -> 'A'")
        check(spurs._extract_chain_id("2CI2_I", pdb_chain_d) == "I", "chain extraction: 2CI2_I -> 'I'")
        check(spurs._extract_chain_id("P04637", pdb_chain_d) == "D",
              "chain extraction: name without chain detects first ATOM chain ('D')")
        check(spurs._extract_chain_id("1A0N.pdb_L7S", pdb_chain_d) == "D",
              "chain extraction: MegaScale mutation suffix does not spoof chain ID")
        check(spurs._extract_chain_id("P04637", pdb_chain_d, explicit_chain="C") == "C",
              "chain extraction: explicit chain column takes priority")

        # 3. Defensive model(pdb, ...) calling & (L, 20) output matrix parsing
        class _ModLogist(torch.nn.Module):
            def forward(self, pdb, return_logist: bool = False):
                return torch.ones((len(pdb["seq"]), 20), dtype=torch.float32) * (2.0 if return_logist else -1.0)

        class _ModLogitsBf16(torch.nn.Module):
            def forward(self, pdb, return_logits: bool = False):
                return torch.full((1, len(pdb["seq"]), 20), 3.0, dtype=torch.bfloat16)

        class _ModPlainDictTransposed(torch.nn.Module):
            def forward(self, pdb):
                return {"ddg": torch.full((20, len(pdb["seq"])), 4.0, dtype=torch.float32)}

        m1 = spurs._call_upstream_spurs_model(_ModLogist(), {"seq": "ACDEFG"})
        m2 = spurs._call_upstream_spurs_model(_ModLogitsBf16(), {"seq": "ACDEFG"})
        m3 = spurs._call_upstream_spurs_model(_ModPlainDictTransposed(), {"seq": "ACDEFG"})
        check(m1.shape == (6, 20) and np.allclose(m1, 2.0), "SPURS caller uses return_logist=True")
        check(m2.shape == (6, 20) and np.allclose(m2, 3.0), "SPURS caller handles return_logits + (1,L,20) bf16")
        check(m3.shape == (6, 20) and np.allclose(m3, 4.0), "SPURS caller handles dict + transposed (20,L)")

        # 4. BUG G: --mode upstream --official-external-only when spurs.inference cannot be imported
        # Pre-create empty .git dir inside work-dir so clone_repo does not touch network
        empty_repo = tdp / "empty_spurs_repo"
        (empty_repo / ".git").mkdir(parents=True)
        out_fail = tdp / "out_bug_g_fail"
        rc = spurs.main([
            "--mode", "upstream",
            "--official-external-only",
            "--splits", str(tdp / "DOES_NOT_EXIST_splits.parquet"),
            "--external", str(ds["ext"]),
            "--external-pdb-dir", str(ds["ext_pdbs"]),
            "--work-dir", str(empty_repo),
            "--out", str(out_fail),
        ])
        status_file = out_fail / "official_spurs_status.json"
        check(rc != 0, f"BUG G: upstream --official-external-only returns clean non-zero exit (rc={rc})")
        check(status_file.is_file(), "BUG G: writes official_spurs_status.json on import failure")
        st = json.loads(status_file.read_text(encoding="utf-8"))
        for key in ("exception_type", "message", "traceback_tail", "python_version",
                    "torch_version", "spurs_importable", "repo_path_searched"):
            check(key in st, f"BUG G: official_spurs_status.json contains {key!r}")
        check(st["status"] == "failed" and st["spurs_importable"] is False,
              "BUG G: diagnostics report status='failed' and spurs_importable=False (no FileNotFoundError fallback)")

        # Also check missing --external with --mode upstream --official-external-only
        out_noext = tdp / "out_bug_g_noext"
        rc_noext = spurs.main([
            "--mode", "upstream", "--official-external-only",
            "--splits", str(tdp / "DOES_NOT_EXIST_splits.parquet"),
            "--work-dir", str(empty_repo),
            "--out", str(out_noext),
        ])
        check(rc_noext != 0 and (out_noext / "official_spurs_status.json").is_file(),
              "BUG G: missing --external with --official-external-only exits non-zero cleanly without retrain")

        # 5. End-to-end mock upstream SPURS (verifying chain='B', weights_only patch, extra-externals)
        mock_repo = tdp / "mock_spurs_repo"
        (mock_repo / "spurs").mkdir(parents=True)
        (mock_repo / "spurs" / "__init__.py").write_text("", encoding="utf-8")
        (mock_repo / "spurs" / "inference.py").write_text(
            "import torch\n"
            "from src.geometry import parse_pdb_backbone\n"
            "SEEN_CHAINS = []\n"
            "class _FakeSpurs(torch.nn.Module):\n"
            "    def forward(self, pdb, return_logist=True):\n"
            "        L = len(pdb['seq'])\n"
            "        return torch.arange(L * 20, dtype=torch.float32).reshape(L, 20) * 0.01\n"
            "def get_SPURS_from_hub():\n"
            "    # Verify torch.load is patched to default weights_only=False\n"
            "    return _FakeSpurs(), {'mock': True}\n"
            "def parse_pdb(pdb_path, pdb_name, chain, cfg):\n"
            "    SEEN_CHAINS.append((pdb_name, chain))\n"
            "    seq, _, _, _ = parse_pdb_backbone(pdb_path)\n"
            "    return {'seq': seq, 'chain': chain}\n",
            encoding="utf-8",
        )
        out_ok = tdp / "out_upstream_ok"
        old_sys_path = list(sys.path)
        try:
            rc_ok = spurs.main([
                "--mode", "upstream", "--official-external-only",
                "--external", str(ds["ext"]),
                "--external-pdb-dir", str(ds["ext_pdbs"]),
                "--extra-externals", f"domainome={ds['dom']}::{ds['ext_pdbs']}",
                "--work-dir", str(mock_repo),
                "--out", str(out_ok),
            ])
            import spurs.inference as fake_spurs_inf  # type: ignore

            check(rc_ok == 0, f"mock upstream SPURS exited 0 (rc={rc_ok})")
            check(("1EXT_B", "B") in fake_spurs_inf.SEEN_CHAINS,
                  f"mock upstream SPURS passed real chain 'B' for 1EXT_B (seen={fake_spurs_inf.SEEN_CHAINS})")
            p_ext = pd.read_parquet(out_ok / "pred_external_official.parquet")
            p_dom = pd.read_parquet(out_ok / "pred_domainome_official.parquet")
            check(len(p_ext) == 2 and list(p_ext["row_id"]) == [0, 1],
                  "mock upstream SPURS wrote pred_external_official.parquet with exact row_ids")
            check(len(p_dom) == 1 and int(p_dom["row_id"].iloc[0]) == 0,
                  "mock upstream SPURS matched domainome via uniprot_id and skipped non-canonical/OOB cleanly")
            st_ok = json.loads((out_ok / "official_spurs_status.json").read_text(encoding="utf-8"))
            check(st_ok["status"] == "ok", "official_spurs_status.json records status='ok' on success")
        finally:
            sys.modules.pop("spurs.inference", None)
            sys.modules.pop("spurs", None)
            sys.path[:] = old_sys_path


def test_competitors_end_to_end_cpu() -> None:
    section("competitors_smoke - end-to-end 1-epoch CPU runs (SPURS, RaSP, PROSTATA, DDGemb)")
    from scripts.competitors import ddgemb, prostata, rasp, spurs

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        ds = _make_tiny_dataset(tdp, esm_dim=16)

        # 1. SPURS local_fusion (1 epoch CPU, extra-externals + empty external + empty val)
        out_spurs = tdp / "out_spurs"
        rc = spurs.main([
            "--mode", "local_fusion",
            "--splits", str(ds["splits"]),
            "--cache", str(ds["cache"]),
            "--pdb-dir", str(ds["ms_pdbs"]),
            "--external", str(ds["ext"]),
            "--external-pdb-dir", str(ds["ext_pdbs"]),
            "--extra-externals",
            f"domainome={ds['dom']}::{ds['ext_pdbs']}",
            f"empty={ds['empty']}",
            "--epochs", "1", "--batch-size", "4", "--seed", "7",
            "--out", str(out_spurs),
        ])
        check(rc == 0, "SPURS local_fusion 1-epoch CPU exited 0")
        sp_te = pd.read_parquet(out_spurs / "pred_test.parquet")
        sp_ex = pd.read_parquet(out_spurs / "pred_external.parquet")
        sp_dm = pd.read_parquet(out_spurs / "pred_domainome.parquet")
        sp_em = pd.read_parquet(out_spurs / "pred_empty.parquet")
        # test has 6 valid + 1 non-canonical (B->Z) + 1 missing-PDB (still cached) = 8; 2 OOB dropped!
        check(len(sp_te) == 8 and np.isfinite(sp_te["y_pred"]).all(),
              f"SPURS pred_test has 8 finite rows (OOB dropped, non-canonical kept; got {len(sp_te)})")
        check(len(sp_ex) == 3 and np.isfinite(sp_ex["y_pred"]).all(),
              f"SPURS pred_external handles inference-only table without target (got {len(sp_ex)})")
        check(len(sp_dm) == 2 and np.isfinite(sp_dm["y_pred"]).all(),
              f"SPURS pred_domainome drops OOB and scores non-canonical cleanly (got {len(sp_dm)})")
        check(len(sp_em) == 0, "SPURS pred_empty writes valid 0-row parquet cleanly")

        # Empty validation split test for SPURS
        out_spurs_ev = tdp / "out_spurs_empty_val"
        rc_ev = spurs.main([
            "--mode", "local_fusion",
            "--splits", str(ds["splits_empty_val"]),
            "--cache", str(ds["cache"]),
            "--epochs", "1", "--batch-size", "4",
            "--out", str(out_spurs_ev),
        ])
        meta_ev = json.loads((out_spurs_ev / "pred_test.meta.json").read_text(encoding="utf-8"))
        check(rc_ev == 0 and meta_ev["best_val_pearson"] is None,
              "SPURS handles empty val split cleanly with best_val_pearson=None")

        # 2. RaSP (1 epoch CPU, missing PDBs, NaN coord cavity check, extra-externals with & without pdb_dir)
        coords_nan = np.ones((6, 4, 3), dtype=np.float32) * np.arange(6)[:, None, None]
        coords_nan[0, :, :] = np.nan
        plddt_nan = np.array([np.nan, 90.0, 90.0, 90.0, 90.0, 90.0], dtype=np.float32)
        cav_nan = rasp.extract_cavity_features("AXUBZG", coords_nan, plddt_nan)
        check(cav_nan.shape == (6, rasp.CAVITY_DIM) and np.isfinite(cav_nan).all(),
              "RaSP extract_cavity_features stays finite on NaN coords + non-canonical AAs")

        out_rasp = tdp / "out_rasp"
        rc_r = rasp.main([
            "--splits", str(ds["splits"]),
            "--pdb-dir", str(ds["ms_pdbs"]),
            "--external", str(ds["ext"]),
            "--external-pdb-dir", str(ds["ext_pdbs"]),
            "--extra-externals",
            f"domainome={ds['dom']}::{ds['ext_pdbs']}",
            f"empty={ds['empty']}::{ds['ext_pdbs']}",
            f"nopdb={ds['dom']}",  # missing ::pdb_dir -> must skip cleanly, not crash!
            "--epochs", "1", "--batch-size", "4", "--seed", "7",
            "--out", str(out_rasp),
        ])
        check(rc_r == 0, "RaSP 1-epoch CPU exited 0")
        r_te = pd.read_parquet(out_rasp / "pred_test.parquet")
        r_ex = pd.read_parquet(out_rasp / "pred_external.parquet")
        r_dm = pd.read_parquet(out_rasp / "pred_domainome.parquet")
        r_em = pd.read_parquet(out_rasp / "pred_empty.parquet")
        # test has 6 valid + 1 non-canonical (B->Z) with PDB = 7; 2 OOB + 1 missing PDB dropped
        check(len(r_te) == 7 and np.isfinite(r_te["y_pred"]).all(),
              f"RaSP pred_test has 7 finite rows (OOB & missing PDB dropped; got {len(r_te)})")
        check(len(r_ex) == 2 and np.isfinite(r_ex["y_pred"]).all(),
              f"RaSP pred_external scored 2 PDB-matched rows without target (got {len(r_ex)})")
        check(len(r_dm) == 2 and np.isfinite(r_dm["y_pred"]).all(),
              f"RaSP matched domainome via uniprot_id fallback (got {len(r_dm)})")
        check(len(r_em) == 0 and not (out_rasp / "pred_nopdb.parquet").exists(),
              "RaSP handles empty external and cleanly skips tag=parquet lacking ::pdb_dir")

        # 3. PROSTATA (1 epoch CPU, tag=parquet::pdb_dir stripping, empty val, empty external, bf16 autocast)
        out_pros = tdp / "out_prostata"
        rc_p = prostata.main([
            "--splits", str(ds["splits_empty_val"]),
            "--cache", str(ds["cache"]),
            "--external", str(ds["ext"]),
            "--extra-externals",
            f"domainome={ds['dom']}::{ds['ext_pdbs']}",  # must strip ::pdb_dir!
            f"empty={ds['empty']}",
            "--epochs", "1", "--batch-size", "4", "--seed", "7",
            "--out", str(out_pros),
        ])
        check(rc_p == 0, "PROSTATA 1-epoch CPU (empty val + ::pdb_dir spec + empty ext) exited 0")
        p_te = pd.read_parquet(out_pros / "pred_test.parquet")
        p_dm = pd.read_parquet(out_pros / "pred_domainome.parquet")
        p_em = pd.read_parquet(out_pros / "pred_empty.parquet")
        check(len(p_te) == 8 and np.isfinite(p_te["y_pred"]).all(),
              f"PROSTATA pred_test has 8 finite rows (got {len(p_te)})")
        check(len(p_dm) == 2 and np.isfinite(p_dm["y_pred"]).all(),
              f"PROSTATA pred_domainome parsed tag=parquet::pdb_dir cleanly (got {len(p_dm)})")
        check(len(p_em) == 0, "PROSTATA pred_empty writes valid 0-row parquet")

        p_mod = prostata.ProstataFiveHead(esm_dim=16, hidden=16, dropout=0.0).eval()
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            p_bf = p_mod(torch.randn(4, 16), torch.randn(4, 16),
                         torch.tensor([0, 1, 2, 20]), torch.tensor([20, 3, 4, 5]))
        check(p_bf.shape == (4, 1) and torch.isfinite(p_bf.float()).all(),
              "PROSTATA forward runs cleanly under CPU bfloat16 autocast")

        # 4. DDGemb (1 epoch CPU, radius=3, tag=parquet::pdb_dir stripping, empty val, empty external, bf16)
        out_ddg = tdp / "out_ddgemb"
        rc_d = ddgemb.main([
            "--splits", str(ds["splits_empty_val"]),
            "--cache", str(ds["cache"]),
            "--external", str(ds["ext"]),
            "--extra-externals",
            f"domainome={ds['dom']}::{ds['ext_pdbs']}",
            f"empty={ds['empty']}",
            "--radius", "3", "--epochs", "1", "--batch-size", "4", "--seed", "7",
            "--out", str(out_ddg),
        ])
        check(rc_d == 0, "DDGemb 1-epoch CPU (empty val + ::pdb_dir spec + empty ext) exited 0")
        d_te = pd.read_parquet(out_ddg / "pred_test.parquet")
        d_dm = pd.read_parquet(out_ddg / "pred_domainome.parquet")
        d_em = pd.read_parquet(out_ddg / "pred_empty.parquet")
        check(len(d_te) == 8 and np.isfinite(d_te["y_pred"]).all(),
              f"DDGemb pred_test has 8 finite rows (got {len(d_te)})")
        check(len(d_dm) == 2 and np.isfinite(d_dm["y_pred"]).all(),
              f"DDGemb pred_domainome parsed tag=parquet::pdb_dir cleanly (got {len(d_dm)})")
        check(len(d_em) == 0, "DDGemb pred_empty writes valid 0-row parquet without RuntimeError")

        d_mod = ddgemb.DdgEmbTransformer(esm_dim=16, d_model=16, n_heads=2, n_layers=1, radius=3, dropout=0.0).eval()
        d_empty = d_mod(torch.empty((0, 7, 16)), torch.empty((0, 7), dtype=torch.bool),
                        torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long))
        check(d_empty.shape == (0, 1), "DDGemb forward handles empty B=0 batch cleanly")
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            d_bf = d_mod(torch.randn(4, 7, 16), torch.ones(4, 7, dtype=torch.bool),
                         torch.tensor([0, 1, 2, 20]), torch.tensor([20, 3, 4, 5]))
        check(d_bf.shape == (4, 1) and torch.isfinite(d_bf.float()).all(),
              "DDGemb forward runs cleanly under CPU bfloat16 autocast")


def test_thermompnn_offline_helpers_and_cli() -> None:
    section("competitors_smoke - ThermoMPNN offline helpers, CLI validation & mock end-to-end")
    from scripts.competitors import thermompnn

    check(thermompnn._resolve_pdb_map(None) == {}, "ThermoMPNN _resolve_pdb_map(None) -> {}")
    check(thermompnn._resolve_pdb_map("/nonexistent_dir_xyz") == {}, "ThermoMPNN _resolve_pdb_map(missing) -> {}")

    # CLI arg validation: --pdb-dir required when NOT --official-external-only
    arg_failed = False
    try:
        thermompnn.main(["--out", "/tmp/unused_tm_out"])
    except SystemExit as exc:
        arg_failed = (exc.code == 2)
    check(arg_failed, "ThermoMPNN CLI enforces --pdb-dir when --official-external-only is not set")

    # Offline mock ThermoMPNN repo to test both --official-external-only and 1-epoch CPU retrain
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        ds = _make_tiny_dataset(tdp, esm_dim=16)
        mock_repo = tdp / "mock_thermompnn"
        (mock_repo / "models").mkdir(parents=True)

        (mock_repo / "protein_mpnn_utils.py").write_text(
            "import torch\n"
            "from src.geometry import parse_pdb_backbone\n"
            "def alt_parse_PDB(path):\n"
            "    seq, _, _, _ = parse_pdb_backbone(path)\n"
            "    return [{'seq': seq}]\n"
            "def tied_featurize(pdb_batch, device, *args, **kwargs):\n"
            "    L = len(pdb_batch[0]['seq'])\n"
            "    X = torch.zeros((1, L, 4, 3), device=device)\n"
            "    S = torch.zeros((1, L), dtype=torch.long, device=device)\n"
            "    mask = torch.ones((1, L), device=device)\n"
            "    return (X, S, mask, [L], mask, mask, None, None, None, None, None, None, torch.arange(L, device=device).unsqueeze(0))\n",
            encoding="utf-8",
        )
        (mock_repo / "transfer_model.py").write_text(
            "import torch\n"
            "import torch.nn as nn\n"
            "ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'\n"
            "class _MockProtMPNN(nn.Module):\n"
            "    def __init__(self, d=8):\n"
            "        super().__init__()\n"
            "        self.dummy = nn.Linear(d, d)\n"
            "    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, extra):\n"
            "        L = X.shape[1]\n"
            "        h = torch.linspace(-0.5, 0.5, L * 8, device=X.device).reshape(1, L, 8)\n"
            "        return [h, h], h, None\n"
            "class _MockLA(nn.Module):\n"
            "    def __init__(self, d=24):\n"
            "        super().__init__()\n"
            "        self.feature_convolution = nn.Conv1d(d, d, 1)\n"
            "        self.attention_convolution = nn.Conv1d(d, d, 1)\n"
            "        self.softmax = nn.Softmax(dim=-1)\n"
            "        self.dropout = nn.Dropout(0.0)\n"
            "class TransferModel(nn.Module):\n"
            "    def __init__(self, cfg):\n"
            "        super().__init__()\n"
            "        self.lightattn = True\n"
            "        self.subtract_mut = True\n"
            "        self.num_final_layers = 2\n"
            "        self.prot_mpnn = _MockProtMPNN(8)\n"
            "        self.light_attention = _MockLA(24)\n"
            "        self.both_out = nn.Sequential(nn.ReLU(), nn.Linear(24, 21))\n"
            "        self.ddg_out = nn.Linear(1, 1)\n"
            "    def forward(self, pdb_list, mut_list):\n"
            "        out = [{'ddG': torch.tensor(-0.25 * (i + 1))} for i, _ in enumerate(mut_list)]\n"
            "        return out, None\n",
            encoding="utf-8",
        )

        old_sys_path = list(sys.path)
        try:
            sys.path.append(str(mock_repo))
            import transfer_model as tm_mod  # type: ignore

            m_init = tm_mod.TransferModel(None)
            sd = {f"model.{k}": v for k, v in m_init.state_dict().items()}
            sd["metrics.val.r2"] = torch.tensor(0.9)
            torch.save({"state_dict": sd}, str(mock_repo / "models" / "thermoMPNN_default.pt"))

            # Official external only WITHOUT --pdb-dir!
            out_off = tdp / "out_tm_official"
            rc_off = thermompnn.main([
                "--official-external-only",
                "--external", str(ds["ext"]),
                "--external-pdb-dir", str(ds["ext_pdbs"]),
                "--extra-externals",
                f"domainome={ds['dom']}::{ds['ext_pdbs']}",
                f"empty={ds['empty']}::{ds['ext_pdbs']}",
                f"nopdb={ds['dom']}",
                "--work-dir", str(mock_repo),
                "--out", str(out_off),
            ])
            check(rc_off == 0, "ThermoMPNN --official-external-only works without --pdb-dir")
            tm_ext = pd.read_parquet(out_off / "pred_external_official.parquet")
            tm_dom = pd.read_parquet(out_off / "pred_domainome_official.parquet")
            tm_emp = pd.read_parquet(out_off / "pred_empty_official.parquet")
            check(len(tm_ext) == 2 and len(tm_dom) == 2 and len(tm_emp) == 0,
                  f"ThermoMPNN official outputs exact (ext={len(tm_ext)}, dom={len(tm_dom)}, empty={len(tm_emp)})")

            # 1-epoch CPU retrain (splits already has 'parent' and --pairs does not exist!)
            out_ret = tdp / "out_tm_retrain"
            rc_ret = thermompnn.main([
                "--splits", str(ds["splits"]),
                "--pairs", str(tdp / "NO_PAIRS_NEEDED.parquet"),
                "--pdb-dir", str(ds["ms_pdbs"]),
                "--external", str(ds["ext"]),
                "--external-pdb-dir", str(ds["ext_pdbs"]),
                "--extra-externals", f"domainome={ds['dom']}::{ds['ext_pdbs']}",
                "--work-dir", str(mock_repo),
                "--epochs", "1", "--batch-size", "4", "--seed", "7",
                "--out", str(out_ret),
            ])
            check(rc_ret == 0, "ThermoMPNN 1-epoch CPU retrain exited 0 (without --pairs when parent exists)")
            tm_te = pd.read_parquet(out_ret / "pred_test.parquet")
            # 6 canonical test rows (B->Z skipped cleanly as non-TM-alphabet, 2 OOB skipped, 1 missing PDB skipped)
            check(len(tm_te) == 6 and np.isfinite(tm_te["y_pred"]).all(),
                  f"ThermoMPNN retrain pred_test has 6 finite rows (got {len(tm_te)})")

            # Empty B=0 head check
            empty_h = thermompnn._forward_thermompnn_head(
                m_init, torch.empty((0, 24)), torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
            )
            check(empty_h.shape == (0,), "ThermoMPNN _forward_thermompnn_head handles empty B=0 batch")
        finally:
            sys.modules.pop("transfer_model", None)
            sys.modules.pop("protein_mpnn_utils", None)
            sys.path[:] = old_sys_path


TESTS = [
    test_common_edge_cases,
    test_spurs_upstream_bug_g_and_helpers,
    test_competitors_end_to_end_cpu,
    test_thermompnn_offline_helpers_and_cli,
]


if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"\n{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
