"""End-to-end plumbing smoke test on synthetic data (CPU, real GATv2 path).
Run:  python -m tests.test_model_smoke
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data import build_graph, collate_pairs, collate_graphs
from src.full_model import HoloGNN
from src.loss import AntisymmetricLoss, gaussian_nll
from src.backbone import PYG_AVAILABLE

AAS = "ACDEFGHIKLMNPQRSTVWY"
ESM_DIM = 64


def fake_graph(L, with_dna):
    seq = "".join(random.choice(AAS) for _ in range(L))
    dna = "".join(random.choice("ACGT") for _ in range(L * 3)) if with_dna else None
    emb = torch.randn(L, ESM_DIM)
    cm = torch.rand(L, L); cm = (cm + cm.t()) / 2          # symmetric contact map
    return build_graph(seq, emb, cm, dna, {"geo_threshold": 0.6, "eps": 0.2, "delta": 0.5})


def test_dual_track_stability_step():
    print(f"    PyG available: {PYG_AVAILABLE} (real GATv2 path)")
    pairs = []
    for _ in range(4):
        L = random.randint(20, 40)
        pairs.append((fake_graph(L, True), fake_graph(L, True), torch.tensor(random.uniform(-3, 3))))
    g_wt, g_mt, y = collate_pairs(pairs)

    model = HoloGNN(esm_dim=ESM_DIM, hidden=32, out_dim=32, use_mech_track=True, use_idr=True)
    crit = AntisymmetricLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    fwd, rev = model.stability(g_wt, g_mt)
    assert fwd.shape == (4, 1) and rev.shape == (4, 1)
    loss, comp = crit(fwd, rev, y)
    opt.zero_grad(); loss.backward(); opt.step()
    # gradients flowed?
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"ok  dual-track stability step (loss={loss.item():.3f}, comps={comp})")


def test_single_track_ablation():
    g = collate_graphs([fake_graph(25, True) for _ in range(3)])
    m = HoloGNN(esm_dim=ESM_DIM, hidden=32, out_dim=32, use_mech_track=False, use_idr=False)
    z = m.encode(g)
    assert z.shape == (3, 32)
    print("ok  single-track ablation encode (use_mech_track=False)")


def test_pathogenicity_and_idr():
    g = collate_graphs([fake_graph(30, False) for _ in range(5)])  # protein-only -> empty E_mech
    m = HoloGNN(esm_dim=ESM_DIM, hidden=32, out_dim=32, use_idr=True)
    logit = m.pathogenicity(g)
    mu, sigma = m.idr(g)
    assert logit.shape == (5, 1) and mu.shape == (5, 1) and (sigma > 0).all()
    nll = gaussian_nll(mu, sigma, torch.rand(5) * 20 + 10)
    nll.backward()
    print(f"ok  pathogenicity logit + IDR (mu,sigma) on protein-only graphs (nll={nll.item():.3f})")


if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    test_dual_track_stability_step()
    test_single_track_ablation()
    test_pathogenicity_and_idr()
    print("\nMODEL SMOKE TEST PASSED")
