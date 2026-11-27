"""
tests/test_v2.py
================
Regression and mathematical-invariant tests for Holo-GeoN v2:

  * Parameter compression (v2 ~2.5M vs v1 20.6M at esm_dim=1280)
  * Exact antisymmetry, swap-equivariance, scan parity & padding invariance
    with WT-marginal log-odds prior and multi-layer scalar-mix
  * Strict SE(3) rigid-transform invariance of src/geometry.py and the
    invariant 3D coordinate stream (Arms B/C/D)
  * Exact Contact-Gated Potts Epistasis invariants (zero boundary epistasis,
    site-order symmetry, Potts cycle parity)
  * Protein-Batch multi-mutation forward parity with row-by-row MutationDataset
  * beta-NLL gradient damping vs standard Gaussian NLL
  * tap_layer_indices and multi-mutation parsing
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.data import (
    MutationDataset,
    ProteinBatchMutationDataset,
    build_sample_graph,
    collate_graphs,
    collate_protein_batch,
)
from src.encoder import EncodingStore, tap_layer_indices
from src.features import CH_CAI, CH_MRNA, CH_VALID, mechanistic_features
from src.geometry import (
    align_to_sequence,
    edge_geometry,
    local_frames,
    node_geometry,
    parse_pdb_backbone,
    parse_pdb_cb,
)
from src.graphs import (
    build_geometric_graph,
    build_structure_graph,
    build_true_contact_graph,
    contact_topology_features,
    pair_contact_features,
)
from src.losses import beta_gaussian_nll, gaussian_nll
from src.model import HoloGeoN
from src.substitution import AA_BIOPHYS, MutationParseError, parse_multi_mutation
from tests.helpers import check, make_batch, random_protein, section, stub_encoding


def _random_so3(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(M)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.float()


def _make_v2_batch(dim: int = 32, batch_size: int = 3, length: int = 24,
                   pad_to: int | None = None, seed: int = 11,
                   include_str: bool = False) -> dict:
    rng = torch.Generator().manual_seed(seed)
    base = make_batch(dim=dim, batch_size=batch_size, length=length, pad_to=pad_to, seed=seed)
    L = base["esm"].shape[1]
    base["topo"] = torch.zeros(batch_size, L, 6)
    base["topo"][:, :length] = torch.rand(batch_size, length, 6, generator=rng)
    base["neighborhood_1hop"] = base["neighborhood_mask"].clone()
    base["layer_embs"] = torch.zeros(batch_size, 4, L, dim)
    base["layer_embs"][:, :, :length] = torch.randn(batch_size, 4, length, dim, generator=rng)
    base["wt_logprobs"] = torch.zeros(batch_size, L, 20)
    raw_lp = torch.log_softmax(torch.randn(batch_size, length, 20, generator=rng), dim=-1)
    base["wt_logprobs"][:, :length] = raw_lp
    if include_str:
        # Synthetic smooth 3D backbone coordinates for real positions [0..length)
        coords = torch.zeros(batch_size, length, 4, 3)
        for b in range(batch_size):
            t = torch.linspace(0, 6.0, length)
            ca = torch.stack([3.8 * t, 2.0 * torch.sin(t), 2.0 * torch.cos(t)], dim=-1)
            n_ = ca + torch.tensor([-0.9, 0.5, 0.2])
            c_ = ca + torch.tensor([0.9, -0.4, 0.3])
            cb = ca + torch.tensor([0.1, 0.8, -0.9])
            coords[b] = torch.stack([n_, ca, c_, cb], dim=1)
        plddt = torch.full((length,), 90.0)
        cvalid = torch.ones(length)
        es, as_ = [], []
        sn = torch.zeros(batch_size, L, 9)
        for b in range(batch_size):
            ei, ea, node_g = build_structure_graph(coords[b], plddt, cvalid, length)
            es.append(ei + b * L)
            as_.append(ea)
            sn[b, :length] = node_g
        base["edge_str"] = torch.cat(es, dim=1)
        base["attr_str"] = torch.cat(as_, dim=0)
        base["str_node"] = sn
        base["_coords"] = coords
    return base


def test_v2_parameter_compression() -> None:
    section("Holo-GeoN v2 — parameter compression vs v1")
    v1 = HoloGeoN(arch="v1", esm_dim=1280, hidden=256, out_dim=256)
    v2 = HoloGeoN(arch="v2", esm_dim=1280, hidden=256, out_dim=256)
    p1 = v1.parameter_count()["total"]
    p2 = v2.parameter_count()["total"]
    ratio = p1 / p2
    check(p1 > 20_000_000, f"v1 parameter count intact ({p1:,})")
    check(p2 < 3_200_000, f"v2 compressed below 3.2M trainable params ({p2:,})")
    check(ratio > 6.5, f"v2 achieves >6.5x compression vs v1 ({ratio:.2f}x)")


def test_v2_invariants_and_padding() -> None:
    section("Holo-GeoN v2 — antisymmetry, swap-equivariance, scan parity & padding invariance")
    torch.manual_seed(0)
    dim = 32
    model = HoloGeoN(arch="v2", esm_dim=dim, hidden=32, out_dim=32, aa_dim=8, gat_heads=4)
    batch = _make_v2_batch(dim=dim, batch_size=3, length=22, seed=5)
    rev = dict(batch)
    rev["wt_idx"], rev["mt_idx"] = batch["mt_idx"], batch["wt_idx"]

    for mode in ("train", "eval"):
        model.train(mode == "train")
        with torch.no_grad():
            z = model.context(batch)
            ddg, lv = model.ddg(batch, z)
            ddg_r, _ = model.ddg(rev, z)
            viol = float((ddg + ddg_r).abs().max())
            check(viol < 1e-6, f"[{mode}] v2 ddG exact antisymmetry with ESM log-odds prior ({viol:.2e})")

            ts = model.three_state(batch, z)
            ts_r = model.three_state(rev, z)
            check(float((ts_r - ts.flip(-1)).abs().max()) < 1e-6,
                  f"[{mode}] v2 three-state swap-equivariance")

            scan = model.scan(batch, z)
            picked = scan.gather(1, batch["mt_idx"].view(-1, 1))
            check(float((picked - ddg).abs().max()) < 1e-5,
                  f"[{mode}] v2 scan matches per-mutation ddG")
            wt_col = scan.gather(1, batch["wt_idx"].view(-1, 1))
            check(float(wt_col.abs().max()) < 1e-5,
                  f"[{mode}] v2 scan wild-type column is strictly zero")
            check(bool(torch.isfinite(lv).all()), f"[{mode}] v2 logvar is finite")

    # Padding invariance across BidirectionalMixer, FiLM, GATv2, and MultiScaleSiteReadout
    model.eval()
    with torch.no_grad():
        small = _make_v2_batch(dim=dim, batch_size=1, length=20, seed=9)
        padded = _make_v2_batch(dim=dim, batch_size=1, length=20, pad_to=44, seed=9)
        a, _ = model.ddg(small)
        b, _ = model.ddg(padded)
        delta = float((a - b).abs().max())
        check(delta < 1e-4, f"v2 prediction is strictly invariant to right-padding (delta={delta:.2e})")


def test_se3_geometry_invariance() -> None:
    section("SE(3) rigid-transform invariance of src/geometry.py and v2 3D stream")
    torch.manual_seed(42)
    L = 24
    t = torch.linspace(0, 8.0, L)
    ca = torch.stack([3.8 * t, 2.5 * torch.sin(t), 2.5 * torch.cos(t)], dim=-1)
    n_ = ca + torch.tensor([-0.85, 0.55, 0.25])
    c_ = ca + torch.tensor([0.92, -0.42, 0.31])
    cb = ca + torch.tensor([0.12, 0.78, -0.88])
    coords = torch.stack([n_, ca, c_, cb], dim=1)
    plddt = torch.full((L,), 92.0)
    cvalid = torch.ones(L)

    Q = _random_so3(seed=123)
    shift = torch.tensor([17.3, -42.1, 8.9])
    coords_rot = coords @ Q.T + shift

    ei1, ea1, ng1 = build_structure_graph(coords, plddt, cvalid, L)
    ei2, ea2, ng2 = build_structure_graph(coords_rot, plddt, cvalid, L)

    check(torch.equal(ei1, ei2), "k-NN edge set is identical after random SO(3) rotation + translation")
    check(float((ea1 - ea2).abs().max()) < 1e-4,
          f"all 26 edge geometry channels are SE(3)-invariant (max diff={float((ea1-ea2).abs().max()):.2e})")
    check(float((ng1 - ng2).abs().max()) < 1e-4,
          f"all 9 node geometry channels are SE(3)-invariant (max diff={float((ng1-ng2).abs().max()):.2e})")

    # End-to-end HoloGeoN v2 with use_str_stream=True must emit identical ddG
    model = HoloGeoN(arch="v2", esm_dim=32, hidden=32, out_dim=32, aa_dim=8,
                     use_geo_stream=False, use_str_stream=True).eval()
    b1 = _make_v2_batch(dim=32, batch_size=1, length=L, seed=7)
    b1["edge_str"], b1["attr_str"], b1["str_node"] = ei1, ea1, ng1.unsqueeze(0)
    b2 = dict(b1)
    b2["edge_str"], b2["attr_str"], b2["str_node"] = ei2, ea2, ng2.unsqueeze(0)
    with torch.no_grad():
        y1, _ = model.ddg(b1)
        y2, _ = model.ddg(b2)
    diff = float((y1 - y2).abs().max())
    check(diff < 1e-4, f"end-to-end 3D-stream HoloGeoN ddG is SE(3)-invariant (diff={diff:.2e})")

    # Low pLDDT (<50) must zero coord_valid (GEOMETRIC_PLAN §6.3)
    low_plddt = torch.full((L,), 35.0)
    _, _, ng_low = build_structure_graph(coords, low_plddt, cvalid, L)
    check(float(ng_low[:, 8].max()) == 0.0, "pLDDT < 50 zeroes effective coord_valid")


def test_contact_gated_epistasis_invariants() -> None:
    section("Contact-Gated Potts Epistasis Head — exact physical invariants")
    torch.manual_seed(0)
    model = HoloGeoN(arch="v2", esm_dim=32, hidden=32, out_dim=32, aa_dim=8)
    head = model.epistasis_head
    assert head is not None

    B = 6
    zi = torch.randn(B, 32)
    zj = torch.randn(B, 32)
    wt_i = torch.tensor([0, 3, 5, 7, 10, 12])
    mt_i = torch.tensor([1, 4, 8, 9, 11, 15])
    wt_j = torch.tensor([2, 6, 9, 11, 14, 18])
    mt_j = torch.tensor([5, 1, 2, 4, 19, 3])
    pg = torch.rand(B, 4)

    for mode in ("train", "eval"):
        head.train(mode == "train")
        with torch.no_grad():
            e_ij = head(zi, zj, wt_i, mt_i, wt_j, mt_j, pg)
            # 1. Exact site-order symmetry: (i, j) == (j, i)
            e_ji = head(zj, zi, wt_j, mt_j, wt_i, mt_i, pg)
            check(float((e_ij - e_ji).abs().max()) < 1e-6,
                  f"[{mode}] epistasis is exactly symmetric under site swap (i <-> j)")

            # 2. Exact zero boundary epistasis when either site is unmutated
            e_zero1 = head(zi, zj, wt_i, wt_i, wt_j, mt_j, pg)
            e_zero2 = head(zi, zj, wt_i, mt_i, wt_j, wt_j, pg)
            check(float(e_zero1.abs().max()) == 0.0 and float(e_zero2.abs().max()) == 0.0,
                  f"[{mode}] single-mutant boundary epistasis is identically 0")

            # 3. Exact Potts cycle parity: EVEN under double reversal, ODD under single flip
            e_both_rev = head(zi, zj, mt_i, wt_i, mt_j, wt_j, pg)
            e_one_rev = head(zi, zj, mt_i, wt_i, wt_j, mt_j, pg)
            check(float((e_both_rev - e_ij).abs().max()) < 1e-6,
                  f"[{mode}] epistasis is even under simultaneous double reversal")
            check(float((e_one_rev + e_ij).abs().max()) < 1e-6,
                  f"[{mode}] epistasis is odd under single-site reversal")

    # pair_contact_features symmetry
    n = 30
    ei, ea = build_geometric_graph(n, torch.tensor([[3, 5], [18, 25]], dtype=torch.int32),
                                   torch.tensor([0.9, 0.7]))
    f_ab = pair_contact_features(ei, ea, 3, 18, n)
    f_ba = pair_contact_features(ei, ea, 18, 3, n)
    check(torch.allclose(f_ab, f_ba), "pair_contact_features is symmetric in (i, j)")
    check(float(f_ab[0]) > 0.8 and float(f_ab[3]) == 1.0,
          "direct contact pair reports high probability and is_contact=1")


def test_protein_batch_parity_and_cds_dropout() -> None:
    section("Protein-Batch multi-mutation parity & CDS modality dropout")
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        store = EncodingStore(tmp, cache_records=16)
        rows = []
        for p_idx in range(3):
            seq = random_protein(36 + p_idx * 4, seed=100 + p_idx)
            rec = stub_encoding(seq, dim=32, top_k=8)
            rec["layer_embs"] = torch.randn(4, len(seq), 32).half()
            rec["wt_logprobs"] = torch.log_softmax(torch.randn(len(seq), 20), dim=-1).half()
            rec["site_entropy"] = torch.rand(len(seq)).half()
            # Populate synthetic real CDS channels (mech_valid = 1)
            mech = mechanistic_features(seq, None, len(seq))
            mech[:, CH_MRNA] = -0.35
            mech[:, CH_CAI] = 0.72
            mech[:, CH_VALID] = 1.0
            rec["mech"] = mech.half()
            store.put(rec)
            for s in (2, 7, 15, 22, 30):
                wt = seq[s]
                mt = "A" if wt != "A" else "G"
                rows.append({
                    "parent_seq": seq, "site": s, "wt_aa": wt, "mt_aa": mt,
                    "target": -0.5 * ((s % 5) - 2), "sigma": 0.2,
                })
        df = pd.DataFrame(rows)
        cfg = {"window": 256, "graph_top_k": 8, "geo_threshold": 0.3,
               "eps": 0.1, "delta": 0.1, "mech_max_degree": 8, "neighborhood_hops": 2}

        ds_row = MutationDataset(df, store, cfg, target="target")
        ds_pb = ProteinBatchMutationDataset(df, store, cfg, target="target",
                                            muts_per_protein=64, train=False)

        b_row = collate_graphs([ds_row[i] for i in range(len(ds_row))])
        b_pb = collate_protein_batch([ds_pb[i] for i in range(len(ds_pb))])

        model = HoloGeoN(arch="v2", esm_dim=32, hidden=32, out_dim=32, aa_dim=8).eval()
        with torch.no_grad():
            y_row, lv_row = model.ddg(b_row)
            y_pb, lv_pb = model.ddg(b_pb)
        diff_y = float((y_row - y_pb).abs().max())
        diff_lv = float((lv_row - lv_pb).abs().max())
        check(diff_y < 1e-4 and diff_lv < 1e-4,
              f"Protein-Batch forward matches row-by-row MutationDataset exactly (d_ddg={diff_y:.2e}, d_lv={diff_lv:.2e})")

        # CDS modality dropout at p=1.0 must mask all CDS channels & empty E_mech
        ds_drop = MutationDataset(df, store, cfg, target="target", cds_dropout=1.0)
        sample_d = ds_drop[0]
        check(float(sample_d["mech"][:, CH_VALID].max()) == 0.0,
              "cds_dropout=1.0 zeroes mech_valid")
        non_loop = sample_d["edge_mech"][:, sample_d["edge_mech"][0] != sample_d["edge_mech"][1]]
        check(non_loop.numel() == 0, "cds_dropout=1.0 empties E_mech non-loop edges")


def test_beta_nll_and_helpers() -> None:
    section("beta-NLL damping, tap_layer_indices, multi-mutation parsing & Arm E")
    mu = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    lv = torch.tensor([-4.0, -4.0, -4.0])        # tiny sigma -> huge 1/sigma^2
    y = torch.tensor([0.0, 0.0, 0.0])

    # beta=0.0 matches gaussian_nll identically
    l0 = beta_gaussian_nll(mu, lv, y, beta=0.0)
    lg = gaussian_nll(mu, lv, y)
    check(float((l0 - lg).detach().abs()) < 1e-6, "beta_gaussian_nll(beta=0) == gaussian_nll")

    # beta=0.5 damps mean gradient magnitude when sigma << 1
    g_std = torch.autograd.grad(gaussian_nll(mu, lv, y), mu, retain_graph=True)[0]
    g_beta = torch.autograd.grad(beta_gaussian_nll(mu, lv, y, beta=0.5), mu)[0]
    ratio = float(g_beta.norm() / g_std.norm())
    expected_ratio = float(torch.exp(0.5 * lv[0]))
    check(abs(ratio - expected_ratio) < 1e-5,
          f"beta=0.5 scales small-sigma mean gradient by sigma ({ratio:.4f} vs {expected_ratio:.4f})")

    # tap_layer_indices
    check(tap_layer_indices(33) == (18, 24, 30, 33), "33-layer ESM-2 taps (18,24,30,33)")
    check(tap_layer_indices(6) == (3, 4, 5, 6), "6-layer ESM-2 taps (3,4,5,6)")

    # Multi-mutation parsing
    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ"
    mt, muts = parse_multi_mutation(seq, "T7A:L8P")
    check(len(muts) == 2 and mt[6] == "A" and mt[7] == "P", "parse_multi_mutation applies double mutant")
    try:
        parse_multi_mutation(seq, "L8P:L8A")
        check(False, "rejects duplicate site in multi-mutation")
    except MutationParseError:
        check(True, "rejects duplicate site in multi-mutation")

    # Arm E: build_true_contact_graph
    cm = torch.zeros(20, 20, dtype=torch.uint8)
    cm[2, 15] = cm[15, 2] = 1
    ei_e, ea_e = build_true_contact_graph(20, cm)
    hit = (ei_e[0] == 2) & (ei_e[1] == 15)
    check(bool(hit.any()) and ea_e.shape[1] == 3, "Arm E true contact map injects real C-beta edges")


TESTS = [
    test_v2_parameter_compression,
    test_v2_invariants_and_padding,
    test_se3_geometry_invariance,
    test_contact_gated_epistasis_invariants,
    test_protein_batch_parity_and_cds_dropout,
    test_beta_nll_and_helpers,
]


if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
