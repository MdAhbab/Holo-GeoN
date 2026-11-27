"""
tests/test_fast_eval.py
=======================
The per-protein batched scorer (``src.fast_eval.predict_table``, used by
``scripts.export_predictions`` and ``scripts.eval_external``) must reproduce the
legacy row-by-row ``MutationDataset`` path to float rounding, return rows in the
caller's order, and fall back to the row-wise path for proteins longer than the
window. Also pins the data-layer caching changes (shared parent-graph cache,
all-site k-hop tables, ``prewarm``) to the uncached maths.

    python -m tests.test_fast_eval
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import (
    MutationDataset,
    ProteinBatchMutationDataset,
    clear_shared_graph_caches,
    collate_graphs,
    collate_protein_batch,
    move_batch,
)
from src.encoder import EncodingStore
from src.fast_eval import predict_table
from src.graphs import khop_neighborhood
from src.model import HoloGeoN
from tests.helpers import check, random_protein, section, stub_encoding

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 32
AA = "ACDEFGHIKLMNPQRSTVWY"


def build_world(tmp: str, lens: list[int], per_protein: int = 40, seed: int = 0,
                dim: int = DIM) -> tuple[EncodingStore, pd.DataFrame]:
    """Synthetic cache (with WT log-probs + entropy) and an interleaved mutation table."""
    store = EncodingStore(os.path.join(tmp, "cache"))
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for p, L in enumerate(lens):
        seq = random_protein(L, seed=500 + p)
        rec = stub_encoding(seq, dim=dim, top_k=8)
        rec["wt_logprobs"] = torch.log_softmax(torch.randn(L, 20, generator=g), dim=-1)
        rec["site_entropy"] = torch.rand(L, generator=g)
        store.put(rec)
        for _ in range(per_protein):
            s = int(rng.integers(0, L))
            mt = AA[int(rng.integers(0, 20))]
            if mt == seq[s]:
                mt = "A" if seq[s] != "A" else "C"
            rows.append({"parent_seq": seq, "site": s, "wt_aa": seq[s], "mt_aa": mt,
                         "target": float(rng.normal()), "sigma": 0.3})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return store, df


def tiny_cfg(window: int) -> dict:
    cfg = load_config(os.path.join(ROOT, "configs", "benchmark_v2.yaml"),
                      overrides={"device": "cpu"}).as_dict()
    cfg.update({"window": window, "batch_size": 16, "num_workers": 0})
    return cfg


def tiny_model(kind: str, seed: int = 3) -> HoloGeoN:
    torch.manual_seed(seed)
    kw = dict(arch="v2", esm_dim=DIM, hidden=32, out_dim=32, use_geo_stream=True,
              use_multi_layer=False, use_contact_burial=True, use_esm_logits_prior=True)
    if kind == "lean":
        kw.update(use_ssm=False, use_mech_stream=False)
    else:
        kw.update(use_ssm=True, use_mech_stream=True)
    return HoloGeoN(**kw).eval()


@torch.no_grad()
def legacy_scores(model, df, store, cfg):
    """The exact pre-optimisation scoring loop (MutationDataset + collate_graphs)."""
    loader = DataLoader(MutationDataset(df, store, cfg, target="target"), batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=0, collate_fn=collate_graphs)
    mus, sigs = [], []
    for b in loader:
        ddg, logvar = model.ddg(move_batch(b, torch.device("cpu")))
        mus.append(ddg.reshape(-1).float().numpy())
        sigs.append(torch.exp(0.5 * logvar).reshape(-1).float().numpy())
    return np.concatenate(mus), np.concatenate(sigs)


def test_fast_path_matches_rowwise() -> None:
    section("batched per-protein scoring == row-wise MutationDataset (max |diff| < 1e-5)")
    clear_shared_graph_caches()
    with tempfile.TemporaryDirectory(prefix="holo_fast_") as tmp:
        # four proteins fit the window (100), two are longer -> row-wise fallback
        store, df = build_world(tmp, lens=[34, 52, 71, 99, 140, 163], per_protein=30)
        cfg = tiny_cfg(window=100)
        for kind in ("lean", "full"):
            model = tiny_model(kind)
            mu_ref, sg_ref = legacy_scores(model, df, store, cfg)
            mu, sg, y, info = predict_table(model, df, store, cfg, torch.device("cpu"))
            d_mu = float(np.max(np.abs(mu - mu_ref)))
            d_sg = float(np.max(np.abs(sg - sg_ref)))
            print(f"    [{kind}] rows={len(df)} fast={info['fast_rows']} rowwise={info['rowwise_rows']} "
                  f"max|dmu|={d_mu:.3e} max|dsigma|={d_sg:.3e}")
            check(info["fast_rows"] > 0 and info["rowwise_rows"] > 0,
                  f"[{kind}] both the batched path and the >window fallback were exercised")
            check(info["fast_rows"] + info["rowwise_rows"] == len(df), f"[{kind}] every row scored once")
            check(d_mu < 1e-5, f"[{kind}] mu equivalent (max abs diff {d_mu:.2e} < 1e-5)")
            check(d_sg < 1e-5, f"[{kind}] sigma equivalent (max abs diff {d_sg:.2e} < 1e-5)")
            check(np.allclose(y, df["target"].to_numpy(np.float32)), f"[{kind}] y returned in table order")
            mu_rw, sg_rw, _, info_rw = predict_table(model, df, store, cfg, torch.device("cpu"), rowwise=True)
            check(info_rw["fast_rows"] == 0 and np.array_equal(mu_rw, mu_ref) and np.array_equal(sg_rw, sg_ref),
                  f"[{kind}] rowwise=True reproduces the legacy loop bit-for-bit")
        # a second call reuses the shared caches and must not change anything
        mu2, sg2, _, _ = predict_table(model, df, store, cfg, torch.device("cpu"))
        check(np.array_equal(mu2, mu) and np.array_equal(sg2, sg), "warm-cache call is deterministic")


def test_all_short_and_empty() -> None:
    section("fast path: all-short table, single-row table, empty table")
    clear_shared_graph_caches()
    with tempfile.TemporaryDirectory(prefix="holo_fast_") as tmp:
        store, df = build_world(tmp, lens=[40, 60], per_protein=25, seed=2)
        cfg = tiny_cfg(window=256)
        model = tiny_model("lean")
        mu_ref, sg_ref = legacy_scores(model, df, store, cfg)
        mu, sg, _, info = predict_table(model, df, store, cfg, torch.device("cpu"))
        check(info["rowwise_rows"] == 0 and float(np.max(np.abs(mu - mu_ref))) < 1e-5
              and float(np.max(np.abs(sg - sg_ref))) < 1e-5, "all-short table equivalent")
        one = df.iloc[:1]
        m1, s1, _, _ = predict_table(model, one, store, cfg, torch.device("cpu"))
        check(len(m1) == 1 and abs(float(m1[0]) - float(mu_ref[0])) < 1e-5, "single-row table equivalent")
        e_mu, e_sg, e_y, e_info = predict_table(model, df.iloc[:0], store, cfg, torch.device("cpu"))
        check(len(e_mu) == 0 and len(e_sg) == 0 and e_info["rows"] == 0, "empty table returns empty arrays")
        # out-of-range site is clamped identically by both paths
        bad = df.iloc[:5].copy()
        bad["site"] = [10_000, 0, 39, 3, 1]
        mu_b_ref, _ = legacy_scores(model, bad, store, cfg)
        mu_b, _, _, _ = predict_table(model, bad, store, cfg, torch.device("cpu"))
        check(float(np.max(np.abs(mu_b - mu_b_ref))) < 1e-5, "out-of-range site clamped identically")


def test_cache_exactness() -> None:
    section("shared graph cache, k-hop tables and prewarm are pure caching")
    clear_shared_graph_caches()
    with tempfile.TemporaryDirectory(prefix="holo_fast_") as tmp:
        store, df = build_world(tmp, lens=[36, 58, 80], per_protein=30, seed=4)
        cfg = tiny_cfg(window=256)
        ds_a = ProteinBatchMutationDataset(df, store, cfg, muts_per_protein=16, train=False)
        n_pre = ds_a.prewarm()
        check(n_pre == 3, "prewarm cached every parent graph")
        ds_b = ProteinBatchMutationDataset(df, store, cfg, muts_per_protein=16, train=False)
        check(ds_a._graph_cache is ds_b._graph_cache, "datasets with the same graph config share one cache")
        cfg_other = dict(cfg, graph_top_k=4)
        ds_c = ProteinBatchMutationDataset(df, store, cfg_other, muts_per_protein=16, train=False)
        check(ds_a._graph_cache is not ds_c._graph_cache, "a different graph config gets its own cache")
        item = ds_b[0]
        seq, idxs = ds_b._chunks[0]
        g = ds_b._graph_cache.get_whole(seq, store.get(seq))
        sites = torch.from_numpy(np.clip(ds_b._site[idxs], 0, g["n"] - 1))
        ref1 = torch.stack([khop_neighborhood(g["edge_geo"], int(s), g["n"], hops=1) for s in sites])
        ref2 = torch.stack([khop_neighborhood(g["edge_geo"], int(s), g["n"], hops=2) for s in sites])
        check(torch.equal(item["neigh_1hop"], ref1) and torch.equal(item["neigh_2hop"], ref2),
              "indexed k-hop tables equal per-site khop_neighborhood()")
        # the protein-batch collate over the whole table covers each row exactly once
        loader = DataLoader(ds_b, batch_size=2, shuffle=False, collate_fn=collate_protein_batch)
        n_rows = sum(int(b["site"].numel()) for b in loader)
        check(n_rows == len(df), "protein-batch loader visits every row exactly once")


TESTS = [test_fast_path_matches_rowwise, test_all_short_and_empty, test_cache_exactness]

if __name__ == "__main__":
    from tests import helpers
    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
