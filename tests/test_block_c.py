"""
tests/test_block_c.py
=====================
Lightweight CPU unit & integration tests for Block C deliverables:
  * Contact-Gated Potts Epistasis training & distance-stratified evaluation
  * Orthogonal between-protein / between-site / within-site error decomposition
  * Competitor adapter architectures (PROSTATA, SPURS, RaSP, DDGemb) + common I/O
  * v2 Kaggle / G4 experiment manifests (plan, sharding, budget reserve)
  * Zenodo Holo-StabBench packager (SHA256SUMS, DATASHEET, manifest)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml

from tests.helpers import check, section

ROOT = Path(__file__).resolve().parent.parent


def _make_synthetic_store(tmp_dir: Path, seqs: list[str], esm_dim: int = 32):
    from src.encoder import EncodingStore, N_TAP_LAYERS

    store = EncodingStore(str(tmp_dir / "cache"))
    rng = np.random.default_rng(7)
    for idx, seq in enumerate(seqs):
        L = len(seq)
        emb = torch.from_numpy(rng.normal(0, 0.5, (L, esm_dim)).astype(np.float32))
        layer_embs = torch.from_numpy(rng.normal(0, 0.5, (N_TAP_LAYERS, L, esm_dim)).astype(np.float32))
        wt_lp = torch.log_softmax(torch.from_numpy(rng.normal(0, 1.0, (L, 20)).astype(np.float32)), dim=-1)
        ent = torch.full((L,), 0.4, dtype=torch.float32)
        # Include both a direct long-range contact (2 <-> 10) and standard sequence backbone
        edges = torch.tensor([[2, 10, 4, 12], [10, 2, 12, 4]], dtype=torch.long)
        probs = torch.tensor([0.92, 0.92, 0.78, 0.78], dtype=torch.float32)
        store.put({
            "seq": seq,
            "emb": emb,
            "contact_edges": edges,
            "contact_probs": probs,
            "layer_embs": layer_embs,
            "wt_logprobs": wt_lp,
            "site_entropy": ent,
            "dim": esm_dim,
        })
    return store


def test_train_epistasis_synthetic() -> None:
    section("block_c - Contact-Gated Epistasis training & distance-stratified eval")
    from scripts.train_epistasis import _build_double_groups, evaluate_groups, train_epistasis_model
    from src.model import HoloGeoN

    seqs = [
        "MKTAYIAKQRQISFVKSHFSRQ",
        "ACDEFGHIKLMNPQRSTVWYAC",
    ]
    rows = []
    rng = np.random.default_rng(123)
    for split, s_idx in (("train", 0), ("val", 1), ("test", 1)):
        seq = seqs[s_idx]
        for k in range(12):
            i = 2 if k < 6 else 1
            j = 10 if k < 6 else 18
            wt_i, wt_j = seq[i], seq[j]
            mt_i = "A" if wt_i != "A" else "G"
            mt_j = "V" if wt_j != "V" else "L"
            dist = 5.5 if k < 6 else 14.0
            ddg_a = float(rng.normal(-0.8, 0.2))
            ddg_b = float(rng.normal(-0.5, 0.2))
            dddg = float(0.9 if dist < 8.0 else 0.05) + float(rng.normal(0, 0.05))
            rows.append({
                "parent_seq": seq, "site_i": i, "wt_i": wt_i, "mt_i": mt_i,
                "site_j": j, "wt_j": wt_j, "mt_j": mt_j,
                "ddg_a": ddg_a, "ddg_b": ddg_b, "dddg": dddg,
                "ddg_ab": ddg_a + ddg_b + dddg, "dist_cb": dist, "split": split,
                "row_id": len(rows),
            })
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as td:
        store = _make_synthetic_store(Path(td), seqs, esm_dim=32)
        cfg = {"window": 64, "graph_top_k": 8, "geo_threshold": 0.3, "neighborhood_hops": 2}
        tr_g = _build_double_groups(df[df["split"] == "train"], store, cfg)
        va_g = _build_double_groups(df[df["split"] == "val"], store, cfg)
        te_g = _build_double_groups(df[df["split"] == "test"], store, cfg)
        check(len(tr_g) == 1 and len(va_g) == 1, "double mutants grouped cleanly by parent")

        model = HoloGeoN(arch="v2", esm_dim=32, hidden=32, out_dim=32, gat_heads=2, dropout=0.0, feat_dropout=0.0)
        info = train_epistasis_model(model, tr_g, va_g, torch.device("cpu"), epochs=6, lr=5e-3, freeze_backbone=True)
        check(len(info["history"]) == 6, "epistasis training ran 6 epochs")
        check(info["history"][-1]["train_loss"] < info["history"][0]["train_loss"],
              "epistasis training loss decreased")

        ev = evaluate_groups(model, te_g, torch.device("cpu"))
        check("ddg_ab" in ev and "dddg_epistasis" in ev, "overall ddg_ab and dddg_epistasis reported")
        check("dddg_contact_lt8A" in ev and "dddg_distant_ge8A" in ev,
              "distance-stratified <8A and >=8A epistasis metrics reported")
        ev_add = evaluate_groups(model, te_g, torch.device("cpu"), additive_only=True)
        check(np.allclose(ev_add["preds"]["pred_epi"].to_numpy(), 0.0),
              "additive_only arm outputs identically zero epistasis")


def test_error_decomp_anova_identity() -> None:
    section("block_c - orthogonal ANOVA error decomposition identity")
    from scripts.error_decomp import decompose_errors

    rng = np.random.default_rng(0)
    rows = []
    for p in range(6):
        b_p = rng.normal(0, 0.8)
        for s in range(10):
            s_pi = rng.normal(0, 1.0)
            for m in range(5):
                w = rng.normal(0, 0.6)
                y = -1.0 + b_p + s_pi + w
                pred = -0.8 + 0.9 * b_p + 0.8 * s_pi + 0.7 * w + rng.normal(0, 0.25)
                rows.append({"parent_seq": f"P{p}", "site": s, "target": y, "y_pred": pred})
    res = decompose_errors(pd.DataFrame(rows))
    dec = res["mse_decomposition"]
    diff = abs(res["total"]["mse"] - dec["sum_check_mse"])
    check(diff < 1e-4, f"orthogonal MSE components sum exactly to total MSE (diff={diff:.2e})")
    frac_sum = dec["frac_global_bias"] + dec["frac_between_protein"] + dec["frac_between_site"] + dec["frac_within_site"]
    check(abs(frac_sum - 1.0) < 1e-2, f"MSE fractions sum to 1.0 (got {frac_sum:.4f})")
    cc = res["component_correlations"]
    check(cc["between_protein_pearson"] > 0.85 and cc["between_site_all_pearson"] > 0.85,
          "high signal recovery detected in between-protein and between-site correlations")
    check(cc["within_site_median_pearson"] > 0.75, "within-site median Pearson computed cleanly")


def test_competitor_models_and_common_io() -> None:
    section("block_c - competitor adapters (PROSTATA, SPURS, RaSP, DDGemb) & common I/O")
    from scripts.competitors.common import write_predictions
    from scripts.competitors.ddgemb import DdgEmbTransformer
    from scripts.competitors.prostata import ProstataFiveHead
    from scripts.competitors.rasp import CAVITY_DIM, RaspDownstreamMLP, extract_cavity_features
    from scripts.competitors.spurs import SpursFusionHead

    B = 8
    wt = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    mt = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])

    # PROSTATA
    p_mod = ProstataFiveHead(esm_dim=32, hidden=32, dropout=0.0).eval()
    h = torch.randn(B, 32)
    out_p = p_mod(h, h, wt, mt)
    check(out_p.shape == (B, 1) and torch.isfinite(out_p).all(), "PROSTATA 5-head output shape & finite")

    # SPURS fusion (exact antisymmetry)
    s_mod = SpursFusionHead(esm_dim=32, struct_dim=10, hidden=32, dropout=0.0).eval()
    hs = torch.randn(B, 10)
    f_s = s_mod(h, hs, wt, mt)
    r_s = s_mod(h, hs, mt, wt)
    check(torch.allclose(f_s, -r_s, atol=1e-5), "SPURS fusion head is exactly antisymmetric")

    # RaSP cavity features + MLP (exact antisymmetry)
    L = 16
    t = np.linspace(0, 4 * np.pi, L)
    ca = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(L)], axis=-1).astype(np.float32)
    coords = np.stack([ca - [0.8, 0, 0], ca, ca + [0.8, 0, 0], ca + [0, 0.9, 0]], axis=1)
    cav = extract_cavity_features("ACDEFGHIKLMNPQRS", coords, np.full(L, 90.0, dtype=np.float32))
    check(cav.shape == (L, CAVITY_DIM) and np.isfinite(cav).all(),
          f"RaSP cavity descriptor shape {(L, CAVITY_DIM)} and finite")
    r_mod = RaspDownstreamMLP(cavity_dim=CAVITY_DIM, hidden=32, dropout=0.0).eval()
    cav_t = torch.from_numpy(cav[:B])
    check(torch.allclose(r_mod(cav_t, wt, mt), -r_mod(cav_t, mt, wt), atol=1e-5),
          "RaSP downstream MLP is exactly antisymmetric")

    # DDGemb Transformer (exact antisymmetry)
    d_mod = DdgEmbTransformer(esm_dim=32, d_model=32, n_heads=2, n_layers=1, radius=4, dropout=0.0).eval()
    w_emb = torch.randn(B, 9, 32)
    w_mask = torch.ones(B, 9, dtype=torch.bool)
    check(torch.allclose(d_mod(w_emb, w_mask, wt, mt), -d_mod(w_emb, w_mask, mt, wt), atol=1e-5),
          "DDGemb Transformer is exactly antisymmetric")

    # Common prediction writer (drops NaN cleanly)
    with tempfile.TemporaryDirectory() as td:
        p_out = Path(td) / "pred_test.parquet"
        write_predictions(p_out, np.array([0, 1, 2]), np.array([0.5, np.nan, -1.2]), {"model": "smoke"})
        df = pd.read_parquet(p_out)
        check(len(df) == 2 and list(df["row_id"]) == [0, 2], "write_predictions filters NaN & keeps exact row_id")
        check(p_out.with_suffix(".meta.json").is_file(), "companion .meta.json written")

        # load_external_table derives site/wt_aa/mt_aa from (wt_seq, mt_seq)
        from scripts.competitors.common import load_external_table
        ext_p = Path(td) / "ext_raw.parquet"
        pd.DataFrame({
            "wt_seq": ["ACDEFG", "ACDEFG", "ACDEFG"],
            "mt_seq": ["ACKEFG", "ACDEFG", "ACDEFA"],  # middle row is synonymous -> dropped
            "ddg": [-1.2, 0.0, -0.4],
            "parent": ["P1", "P1", "P1"],
        }).to_parquet(ext_p, index=False)
        ext_df = load_external_table(str(ext_p))
        check(len(ext_df) == 2 and list(ext_df["row_id"]) == [0, 2],
              "load_external_table drops non-single-mut while preserving raw row_id")
        check(list(ext_df["site"]) == [2, 5] and list(ext_df["wt_aa"]) == ["D", "G"]
              and list(ext_df["mt_aa"]) == ["K", "A"],
              "load_external_table derives exact 0-based site/wt_aa/mt_aa")

        # DDGemb compact parent-indexed _extract_windows matches exact window padding
        from scripts.competitors.ddgemb import _extract_windows
        from src.encoder import EncodingStore
        store = EncodingStore(str(Path(td) / "cache"))
        seq = "ACDEFGH"
        emb = torch.arange(len(seq) * 8, dtype=torch.float32).reshape(len(seq), 8)
        store.put({"seq": seq, "emb": emb.half(), "contact_edges": torch.zeros((2, 0), dtype=torch.long),
                   "contact_probs": torch.zeros((0,), dtype=torch.float32)})
        mini = pd.DataFrame({
            "parent_seq": [seq, seq], "site": [0, 6], "wt_aa": ["A", "H"],
            "mt_aa": ["G", "A"], "target": [-1.0, -2.0], "row_id": [10, 20],
        })
        widx = _extract_windows(mini, store, radius=2)
        we, wm, wi, mi, yb = widx.gather_batch(slice(0, 2), torch.device("cpu"))
        check(we.shape == (2, 5, 8) and wm.shape == (2, 5), "DDGemb gather_batch shape matches (B, 2R+1, D)")
        check(wm[0].tolist() == [False, False, True, True, True]
              and wm[1].tolist() == [True, True, True, False, False],
              "DDGemb boundary window masks match left/right zero padding")
        check(torch.allclose(we[0, 2], emb[0]) and torch.allclose(we[1, 2], emb[6]),
              "DDGemb center token matches exact mutated residue embedding")

        # load_mutation_table accepts pre-normalised tables lacking mt_seq (e.g. s669)
        from src.data import load_mutation_table
        s669_p = Path(td) / "s669_like.parquet"
        pd.DataFrame({
            "parent": ["P1", "P1", "P1"],
            "wt_seq": ["ACDEFG", "ACDEFG", "ACDEFG"],
            "site": [1, 2, 3],
            "wt_aa": ["C", "X", "E"],  # row 1 mismatches wt_seq[2]=='D' -> dropped
            "mt_aa": ["A", "A", "K"],
            "ddg": [-1.1, -0.5, -2.0],
        }).to_parquet(s669_p, index=False)
        norm, rep = load_mutation_table(str(s669_p), kind="stability")
        check(len(norm) == 2 and rep.dropped.get("wt_aa_mismatch") == 1,
              "load_mutation_table accepts site/wt_aa/mt_aa without mt_seq and catches mismatch")
        check(list(norm["source_row_id"]) == [0, 2],
              "load_mutation_table records original source_row_id")

        # apply_checkpoint_graph_config restores 3D / true-contact flags onto default Config
        from src.config import apply_checkpoint_graph_config, load_config
        cfg = load_config("configs/benchmark_v2.yaml")
        check(cfg.use_str_stream is False and cfg.true_contact_edges is False,
              "benchmark_v2 default is structure-free")
        apply_checkpoint_graph_config(cfg, {"config": {
            "use_str_stream": True, "str_topology_only": True,
            "true_contact_edges": True, "true_contacts_dir": "data/true_contacts",
        }})
        check(cfg.use_str_stream is True and cfg.str_topology_only is True
              and cfg.true_contact_edges is True and cfg.true_contacts_dir == "data/true_contacts",
              "apply_checkpoint_graph_config propagates 3D and true-contact settings")

        # ThermoMPNN Lightning checkpoint stripping & batched head exactness
        from scripts.competitors.thermompnn import (
            _forward_thermompnn_head,
            _strip_lightning_state_dict,
        )
        sd_clean = _strip_lightning_state_dict({
            "state_dict": {
                "model.both_out.0.weight": torch.zeros(2, 2),
                "model.ddg_out.bias": torch.zeros(1),
                "metrics.val_metrics.ddG.r2": torch.tensor(0.5),
            }
        })
        check(set(sd_clean.keys()) == {"both_out.0.weight", "ddg_out.bias"},
              "ThermoMPNN _strip_lightning_state_dict strips model. prefix and drops metrics.*")

        class _MockLA(torch.nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.feature_convolution = torch.nn.Conv1d(d, d, 3, padding=1)
                self.attention_convolution = torch.nn.Conv1d(d, d, 3, padding=1)
                self.softmax = torch.nn.Softmax(dim=-1)
                self.dropout = torch.nn.Dropout(0.0)

        class _MockTM(torch.nn.Module):
            def __init__(self, d: int = 8):
                super().__init__()
                self.lightattn = True
                self.subtract_mut = True
                self.light_attention = _MockLA(d)
                self.both_out = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(d, 21))
                self.ddg_out = torch.nn.Linear(1, 1)

        tm = _MockTM(8).eval()
        rf = torch.randn(4, 8)
        wi_t = torch.tensor([0, 1, 2, 3])
        mi_t = torch.tensor([4, 5, 6, 7])
        y_batch = _forward_thermompnn_head(tm, rf, wi_t, mi_t)
        y_single = torch.stack([
            _forward_thermompnn_head(tm, rf[i:i + 1], wi_t[i:i + 1], mi_t[i:i + 1])[0]
            for i in range(4)
        ])
        check(y_batch.shape == (4,) and torch.allclose(y_batch, y_single, atol=1e-6),
              "ThermoMPNN batched head matches per-mutation evaluation (including B=1)")

        # _load_true_contact_map loads <sha1>.npy cleanly (catches missing `import os`)
        from src.data import _load_true_contact_map
        from src.encoder import seq_key
        tc_dir = Path(td) / "true_contacts"
        tc_dir.mkdir()
        seq_t = "ACDEFGHIK"
        mat = np.eye(len(seq_t), dtype=np.float32)
        np.save(tc_dir / f"{seq_key(seq_t)}.npy", mat)
        loaded = _load_true_contact_map(seq_t, str(tc_dir))
        check(loaded is not None and tuple(loaded.shape) == (len(seq_t), len(seq_t)),
              "_load_true_contact_map loads <sha1>.npy without NameError")

        # HoloBackboneV2._mix_esm normalizes each tapped layer before scalar mixing
        from src.backbone import HoloBackboneV2
        bb = HoloBackboneV2(esm_dim=16, hidden=16, out_dim=16, heads=2, n_tap_layers=4, use_multi_layer=True)
        torch.manual_seed(0)
        base_layers = torch.randn(1, 4, 6, 16)
        scaled_layers = base_layers.clone()
        scaled_layers[:, 0] *= 50.0  # huge intermediate-layer scale drift
        scaled_layers[:, 1] *= 25.0
        out_base = bb._mix_esm(base_layers[:, -1], base_layers)
        out_scaled = bb._mix_esm(base_layers[:, -1], scaled_layers)
        check(torch.allclose(out_base, out_scaled, atol=1e-5),
              "HoloBackboneV2._mix_esm is invariant to per-layer scale mismatch via LayerNorm")


def test_v2_manifests_and_zenodo_pack() -> None:
    section("block_c - v2 experiment manifests & Zenodo packager")
    import io
    from contextlib import redirect_stdout
    from scripts.remote_job import expand_items, print_plan, take_shard
    from scripts.zenodo_pack import build_release_bundle

    for fname, expected_arms, n_shards in (
        ("kaggle_v2_ablations.yaml", 11, 5),
        ("kaggle_v2_geometry.yaml", 5, 5),
        ("kaggle_v2_clean_abl_geom.yaml", 14, 5),
        ("g4_v2_headline.yaml", 1, 1),
    ):
        m = yaml.safe_load((ROOT / "configs" / "experiments" / fname).read_text(encoding="utf-8"))
        items = expand_items(m)
        trains = [i for i in items if i["kind"] == "train"]
        arms = {i["arm"] for i in trains}
        check(len(arms) == expected_arms, f"{fname}: {len(arms)}/{expected_arms} distinct arms")
        if n_shards > 1:
            part = take_shard(items, f"0/{n_shards}")
            check(len([i for i in part if i["kind"] == "train"]) == expected_arms,
                  f"{fname}: shard 0/{n_shards} holds complete paired seed block")
        buf = io.StringIO()
        with redirect_stdout(buf):
            total_h = print_plan(m, items, float(m["sizing"]["sec_per_step"]), "ASSUMED")
        if m.get("cost", {}).get("budget_cu"):
            usable = m["cost"]["budget_cu"] - m["cost"].get("reserve_cu", 0)
            check(total_h * m["cost"]["cu_per_hour"] <= usable,
                  f"{fname}: fits inside usable CU budget ({total_h * m['cost']['cu_per_hour']:.1f} <= {usable})")

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "Holo-StabBench-v1.0"
        (stage / "tables").mkdir(parents=True)
        pd.DataFrame({"parent": ["P1"], "wt_seq": ["ACDE"], "ddg": [-1.0]}).to_parquet(
            stage / "tables" / "sample.parquet", index=False
        )
        man = build_release_bundle(stage, version="1.0.0-test")
        check(man["total_files"] == 1, "zenodo_pack indexed 1 staged file")
        check((stage / "SHA256SUMS.txt").is_file() and (stage / "DATASHEET.md").is_file(),
              "SHA256SUMS.txt and DATASHEET.md generated")


TESTS = [
    test_train_epistasis_synthetic,
    test_error_decomp_anova_identity,
    test_competitor_models_and_common_io,
    test_v2_manifests_and_zenodo_pack,
]


if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
