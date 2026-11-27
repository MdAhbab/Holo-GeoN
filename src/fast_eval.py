"""
src/fast_eval.py
================
Per-protein batched scoring of a mutation table (bit-for-bit the same maths as
the row-by-row :class:`src.data.MutationDataset` path, ~10-50x faster).

Why
---
``MutationDataset`` yields ONE mutation per item. Scoring a table of N rows
therefore collates N graph copies and runs the encoder N times, even when the
table is 464 tiny 50-residue domains with ~700 mutations each (Human Domainome:
332,854 rows -> ~6 min of CPU-bound graph collation per checkpoint).

``ProteinBatchMutationDataset`` + ``collate_protein_batch`` is the layout the
model was *trained* with: each protein's graph is built ONCE and the backbone
runs ONCE per protein; every mutation of that protein is then gathered in
parallel by ``MultiScaleSiteReadout.gather_protein_batch``.

``predict_table`` uses that path for every protein with ``len(seq) <= window``
(for which ``build_sample_graph`` yields the whole protein regardless of the
site, so the two paths see identical graphs) and falls back to the unchanged
row-wise ``MutationDataset`` path for longer proteins, whose crop depends on the
mutated site. Results are returned in the input table's row order, so callers
need not care which path scored a row.

The escape hatch ``rowwise=True`` (CLI ``--rowwise``, env ``HOLO_EVAL_ROWWISE=1``)
forces the legacy path for every row.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    MutationDataset,
    ProteinBatchMutationDataset,
    collate_graphs,
    collate_protein_batch,
    move_batch,
)

# Upper bound on (mutations in a batch) x (nodes): the readout materialises an
# (M, L, D) gather, so this bounds peak memory independently of the table.
_MUT_NODE_BUDGET = 524_288
_CHUNK_NODE_BUDGET = 262_144


def rowwise_forced() -> bool:
    return os.environ.get("HOLO_EVAL_ROWWISE", "").strip().lower() in ("1", "true", "yes")


def _budget_batches(chunks: list[tuple[str, np.ndarray]], budget: int) -> list[list[int]]:
    """Group consecutive protein chunks so that sum(M_i * L_i) stays <= budget."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cost = 0
    for ci, (seq, idxs) in enumerate(chunks):
        c = len(idxs) * max(len(seq), 1)
        if cur and cost + c > budget:
            batches.append(cur)
            cur, cost = [], 0
        cur.append(ci)
        cost += c
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def _run(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    mus, sigs = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        ddg, logvar = model.ddg(batch)
        mus.append(ddg.reshape(-1).float().cpu().numpy())
        sigs.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
    if not mus:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    return np.concatenate(mus), np.concatenate(sigs)


@torch.no_grad()
def predict_table(
    model,
    df: pd.DataFrame,
    store,
    cfg,
    device,
    *,
    target: str = "target",
    rowwise: bool | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Score every row of ``df`` and return ``(mu, sigma, y, info)`` in ``df`` order.

    ``df`` needs the normalised columns ``parent_seq, site, wt_aa, mt_aa`` and
    ``target`` (``y`` is read back from it). Every parent must have a cached
    encoding in ``store`` (callers run :func:`src.data.filter_to_cache` first).
    """
    model.eval()
    cfgd = cfg if isinstance(cfg, dict) else cfg.as_dict()
    window = int(cfgd.get("window", 256))
    bs = int(batch_size or cfgd.get("batch_size", 64))
    nw = int(cfgd.get("num_workers", 0) if num_workers is None else num_workers)
    if rowwise is None:
        rowwise = rowwise_forced()

    df = df.reset_index(drop=True)
    n = len(df)
    mu = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    y = (df[target].to_numpy(dtype=np.float32) if (n and target in df.columns)
         else np.zeros(n, dtype=np.float32))
    info: dict = {"rows": n, "fast_rows": 0, "rowwise_rows": 0, "fast_proteins": 0,
                  "window": window, "rowwise_forced": bool(rowwise)}
    if n == 0:
        return mu, sigma, y, info
    t0 = time.time()

    lens = df["parent_seq"].astype(str).str.len().to_numpy()
    fast = np.zeros(n, dtype=bool) if rowwise else (lens <= window)

    # ---- fast path: one graph + one backbone pass per protein --------------
    if fast.any():
        pos = np.flatnonzero(fast)
        sub = df.iloc[pos].reset_index(drop=True)
        max_len = int(lens[pos].max())
        chunk = int(min(2048, max(128, _CHUNK_NODE_BUDGET // max(max_len, 1))))
        ds = ProteinBatchMutationDataset(
            sub, store, cfgd, target=target, muts_per_protein=chunk, train=False,
        )
        loader = DataLoader(
            ds, batch_sampler=_budget_batches(ds._chunks, _MUT_NODE_BUDGET),
            collate_fn=collate_protein_batch, num_workers=0,
        )
        m, s = _run(model, loader, device)
        order = np.concatenate([idxs for _, idxs in ds._chunks])
        if len(m) != len(order):
            raise RuntimeError(f"fast path scored {len(m)} rows for {len(order)} inputs")
        mu[pos[order]] = m
        sigma[pos[order]] = s
        info["fast_rows"] = int(len(pos))
        info["fast_proteins"] = int(len(ds._groups))

    # ---- legacy path: site-centred crop per row (proteins longer than window)
    slow_pos = np.flatnonzero(~fast)
    if len(slow_pos):
        rest = df.iloc[slow_pos].reset_index(drop=True)
        loader = DataLoader(
            MutationDataset(rest, store, cfgd, target=target),
            batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_graphs,
        )
        m, s = _run(model, loader, device)
        if len(m) != len(slow_pos):
            raise RuntimeError(f"row-wise path scored {len(m)} rows for {len(slow_pos)} inputs")
        mu[slow_pos] = m
        sigma[slow_pos] = s
        info["rowwise_rows"] = int(len(slow_pos))

    info["seconds"] = round(time.time() - t0, 3)
    return mu, sigma, y, info


__all__ = ["predict_table", "rowwise_forced"]
