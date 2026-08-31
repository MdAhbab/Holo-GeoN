"""
tests/helpers.py
================
Shared test utilities: assertions, synthetic proteins, and synthetic batches.

The synthetic encoder here exists **only for tests**. It produces deterministic
pseudo-embeddings so the pipeline can be exercised end to end without loading
ESM-2, and it lives in ``tests/`` rather than ``src/`` precisely so no
production code path can reach it. Anything it produces is a plumbing artifact —
shapes, batching, checkpointing, metric computation — and never a statement
about model performance.
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.features import mechanistic_features
from src.graphs import build_geometric_graph, build_mechanistic_graph, khop_neighborhood
from src.substitution import AA_ORDER

_PASSED = 0
_FAILED: list[str] = []


def section(name: str) -> None:
    print(f"\n--- {name} ---")


def check(condition: bool, message: str) -> None:
    """Assert and report, accumulating failures so one run shows them all."""
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  ok   {message}")
    else:
        _FAILED.append(message)
        print(f"  FAIL {message}")


def results() -> tuple[int, list[str]]:
    return _PASSED, list(_FAILED)


def reset() -> None:
    global _PASSED
    _PASSED = 0
    _FAILED.clear()


# ---------------------------------------------------------------------------
def random_protein(length: int, seed: int = 0) -> str:
    """Deterministic pseudo-random amino-acid sequence."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(AA_ORDER), (length,), generator=g)
    return "".join(AA_ORDER[i] for i in idx.tolist())


def stub_encoding(seq: str, dim: int = 32, top_k: int = 8) -> dict:
    """
    A deterministic stand-in for an :class:`src.encoder.EsmEncoder` record.

    Embeddings are seeded from the sequence hash, so the same sequence always
    yields the same tensor and cache behaviour is realistic. Contacts are a
    deterministic pseudo-random sparse set with a real separation floor.

    TEST ONLY. These are not protein representations and nothing measured on
    them means anything about the model's accuracy.
    """
    length = len(seq)
    seed = int(hashlib.sha1(seq.encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(length, dim, generator=g)

    src_nodes, dst_nodes = [], []
    for i in range(length):
        for j in range(i + 3, min(i + 3 + top_k, length), 3):
            src_nodes.append(i)
            dst_nodes.append(j)
    if src_nodes:
        edges = torch.tensor([src_nodes, dst_nodes], dtype=torch.int32)
        probs = torch.rand(edges.shape[1], generator=g).clamp(min=0.31)
    else:
        edges = torch.zeros((2, 0), dtype=torch.int32)
        probs = torch.zeros(0)

    return {
        "seq": seq,
        "emb": emb.half(),
        "contact_edges": edges,
        "contact_probs": probs.half(),
        "mech": mechanistic_features(seq, None, length).half(),
        "esm_model": "TEST-STUB-NOT-A-REAL-ENCODER",
        "dim": dim,
        "has_cds": False,
    }


def make_batch(
    dim: int = 32,
    batch_size: int = 4,
    length: int = 30,
    seed: int = 0,
    pad_to: int | None = None,
) -> dict:
    """Build a model-ready batch directly, without touching the data layer."""
    L = pad_to or length
    esm = torch.zeros(batch_size, L, dim)
    mech = torch.zeros(batch_size, L, 7)
    mask = torch.zeros(batch_size, L, dtype=torch.bool)
    neighborhood = torch.zeros(batch_size, L, dtype=torch.bool)
    eg, ag, em, am, sites = [], [], [], [], []

    g = torch.Generator().manual_seed(seed)
    for b in range(batch_size):
        seq = random_protein(length, seed=seed + b)
        rec = stub_encoding(seq, dim=dim)
        n = length
        esm[b, :n] = rec["emb"].float()
        mech[b, :n] = rec["mech"].float()
        mask[b, :n] = True

        edge_geo, attr_geo = build_geometric_graph(
            n, rec["contact_edges"].long(), rec["contact_probs"].float(),
            top_k=8, threshold=0.30,
        )
        edge_mech, attr_mech = build_mechanistic_graph(
            rec["mech"][:, 1].float(), rec["mech"][:, 0].float(), rec["mech"][:, 6].float()
        )
        site = int(torch.randint(0, n, (1,), generator=g).item())
        sites.append(site)
        neighborhood[b, :n] = khop_neighborhood(edge_geo, site, n, hops=2)

        eg.append(edge_geo + b * L)
        ag.append(attr_geo)
        em.append(edge_mech + b * L)
        am.append(attr_mech)

    wt = torch.randint(0, 20, (batch_size,), generator=g)
    mt = (wt + 1 + torch.randint(0, 19, (batch_size,), generator=g)) % 20
    return {
        "esm": esm,
        "mech": mech,
        "mask": mask,
        "edge_geo": torch.cat(eg, dim=1),
        "attr_geo": torch.cat(ag, dim=0),
        "edge_mech": torch.cat(em, dim=1),
        "attr_mech": torch.cat(am, dim=0),
        "site": torch.tensor(sites, dtype=torch.long),
        "neighborhood_mask": neighborhood,
        "wt_idx": wt,
        "mt_idx": mt,
        "y": torch.randn(batch_size, generator=g),
        "sigma": torch.full((batch_size,), float("nan")),
    }


__all__ = [
    "section", "check", "results", "reset",
    "random_protein", "stub_encoding", "make_batch",
]
