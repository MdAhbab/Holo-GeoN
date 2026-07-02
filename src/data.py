"""
src/data.py
===========
Data layer that consumes the PREPROCESSED parquet files already on the training
machine (see guide.md). No raw downloads, no ETL — just read, assemble dual-track
graphs from cached ESM features, and batch.

Expected files (column names auto-detected; override in configs/data.yaml):
    megascale_ddg_pairs.parquet        -> stability  (wt_seq, mt_seq, ddG[, dna])
    clinvar_protein_pairs.parquet      -> pathogenicity (wt_seq, mt_seq, label, gene)
    clinvar_protein_pairs.parquet.ddg_w49.npy  -> optional cached ΔΔG feature

Graphs are built from a per-sequence embedding cache produced by
scripts/precompute_embeddings.py:  cache/<md5>.pt = {emb (L,d), contacts (L,L), seq}.
"""

from __future__ import annotations

import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features import mechanistic_features
from src.utils import graph_builder as G


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------
_CANDIDATES = {
    "wt_seq":  ["wt_seq", "wildtype_seq", "seq_wt", "ref_seq", "wild_seq", "wildtype", "reference_seq"],
    "mt_seq":  ["mt_seq", "mutant_seq", "seq_mt", "alt_seq", "mut_seq", "mutant", "alternate_seq"],
    "target":  ["ddg", "ddG", "dg", "delta_delta_g", "target", "ddg_exp", "score"],
    "label":   ["label", "y", "pathogenic", "is_pathogenic", "clinical_significance", "clnsig"],
    "gene":    ["gene", "gene_symbol", "symbol", "geneinfo"],
    "wt_dna":  ["wt_dna", "wt_cds", "ref_cds", "wt_cdna", "dna_wt", "cds_wt"],
    "mt_dna":  ["mt_dna", "mt_cds", "alt_cds", "mt_cdna", "dna_mt", "cds_mt"],
}


def detect_columns(df: pd.DataFrame, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    lower = {c.lower(): c for c in df.columns}
    out: dict[str, str | None] = {}
    for key, cands in _CANDIDATES.items():
        if key in overrides and overrides[key] in df.columns:
            out[key] = overrides[key]
            continue
        out[key] = next((lower[c.lower()] for c in cands if c.lower() in lower), None)
    return out


def seq_key(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Normalised loaders
# ---------------------------------------------------------------------------
def load_stability_pairs(path: str, overrides: dict | None = None,
                         max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if max_rows:
        df = df.iloc[:max_rows].copy()
    col = detect_columns(df, overrides)
    missing = [k for k in ("wt_seq", "mt_seq", "target") if col[k] is None]
    if missing:
        raise ValueError(f"{path}: could not find columns for {missing}. "
                         f"Columns present: {list(df.columns)}. Set them in configs/data.yaml.")
    out = pd.DataFrame({
        "wt_seq": df[col["wt_seq"]].astype(str),
        "mt_seq": df[col["mt_seq"]].astype(str),
        "target": df[col["target"]].astype(float),
    })
    out["wt_dna"] = df[col["wt_dna"]].astype(str) if col["wt_dna"] else None
    out["mt_dna"] = df[col["mt_dna"]].astype(str) if col["mt_dna"] else None
    out["group"] = out["wt_seq"].map(seq_key)          # parent-protein group
    return out.reset_index(drop=True)


def load_pathogenicity_pairs(path: str, ddg_npy: str | None = None,
                             overrides: dict | None = None,
                             max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if max_rows:
        df = df.iloc[:max_rows].copy()
    col = detect_columns(df, overrides)
    missing = [k for k in ("wt_seq", "mt_seq", "label") if col[k] is None]
    if missing:
        raise ValueError(f"{path}: missing columns for {missing}. Columns: {list(df.columns)}")
    lab = df[col["label"]]
    if lab.dtype == object:
        lab = lab.str.lower().map(
            lambda s: 1 if any(k in str(s) for k in ("pathogenic", "deleterious", "1")) and "benign" not in str(s)
            else 0)
    out = pd.DataFrame({
        "wt_seq": df[col["wt_seq"]].astype(str),
        "mt_seq": df[col["mt_seq"]].astype(str),
        "label":  lab.astype(int),
    })
    out["gene"] = df[col["gene"]].astype(str) if col["gene"] else out["wt_seq"].map(seq_key)
    out["group"] = out["gene"]                          # gene-disjoint grouping
    if ddg_npy and os.path.exists(ddg_npy):
        cached = np.load(ddg_npy)
        if len(cached) == len(out):
            out["ddg_cached"] = cached.astype(float)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
class EmbeddingStore:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def path(self, seq: str) -> str:
        return os.path.join(self.cache_dir, seq_key(seq) + ".pt")

    def has(self, seq: str) -> bool:
        return os.path.exists(self.path(seq))

    def get(self, seq: str):
        rec = torch.load(self.path(seq), map_location="cpu")
        return rec["emb"].float(), rec["contacts"].float(), rec["seq"]


# ---------------------------------------------------------------------------
# Graph assembly (dual-track)
# ---------------------------------------------------------------------------
def build_graph(seq: str, emb: torch.Tensor, contacts: torch.Tensor,
                dna: str | None, cfg: dict, back_translate: bool = False) -> dict:
    L = emb.shape[0]
    mech = mechanistic_features(seq[:L], dna, max_length=L,
                                back_translate_fallback=back_translate)   # (L, 4)
    edge_geo = G.build_geometric_edges(contacts[:L, :L], threshold=cfg.get("geo_threshold", 0.5))
    if edge_geo.numel() == 0:
        edge_geo = G.simple_linear_graph(L)                              # never leave it edgeless
    edge_mech = G.build_mechanistic_edges(
        mech[:, 1], mech[:, 0], mech[:, 3],
        eps=cfg.get("eps", 0.10), delta=cfg.get("delta", 0.10),
        max_degree=cfg.get("mech_max_degree", 8),
    )
    return {"esm": emb, "mech": mech, "edge_geo": edge_geo, "edge_mech": edge_mech}


def collate_graphs(graphs: list[dict]) -> dict:
    """Batch a list of single-graph dicts into one big graph with a batch vector."""
    esm, mech, eg, em, batch = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = g["esm"].shape[0]
        esm.append(g["esm"]); mech.append(g["mech"])
        if g["edge_geo"].numel():
            eg.append(g["edge_geo"] + offset)
        if g["edge_mech"].numel():
            em.append(g["edge_mech"] + offset)
        batch.append(torch.full((n,), i, dtype=torch.long))
        offset += n
    return {
        "esm": torch.cat(esm, 0),
        "mech": torch.cat(mech, 0),
        "edge_geo": torch.cat(eg, 1) if eg else torch.zeros((2, 0), dtype=torch.long),
        "edge_mech": torch.cat(em, 1) if em else torch.zeros((2, 0), dtype=torch.long),
        "batch": torch.cat(batch, 0),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class StabilityPairDataset(Dataset):
    """Yields (g_wt, g_mt, ddG) for the Siamese stability task."""
    def __init__(self, df: pd.DataFrame, store: EmbeddingStore, cfg: dict):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        bt = self.cfg.get("back_translate", False)
        emb_w, cm_w, sw = self.store.get(r["wt_seq"])
        emb_m, cm_m, sm = self.store.get(r["mt_seq"])
        g_wt = build_graph(sw, emb_w, cm_w, r.get("wt_dna"), self.cfg, back_translate=bt)
        g_mt = build_graph(sm, emb_m, cm_m, r.get("mt_dna"), self.cfg, back_translate=bt)
        return g_wt, g_mt, torch.tensor(float(r["target"]), dtype=torch.float32)


def collate_pairs(batch):
    g_wt = collate_graphs([b[0] for b in batch])
    g_mt = collate_graphs([b[1] for b in batch])
    y = torch.stack([b[2] for b in batch])
    return g_wt, g_mt, y


class VariantDataset(Dataset):
    """Yields (g_mt, label[, ddg_cached]) for pathogenicity (uses the mutant graph)."""
    def __init__(self, df: pd.DataFrame, store: EmbeddingStore, cfg: dict, use_cached_ddg=False):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.cfg = cfg
        self.use_cached_ddg = use_cached_ddg and ("ddg_cached" in df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        emb_m, cm_m, sm = self.store.get(r["mt_seq"])
        g = build_graph(sm, emb_m, cm_m, None, self.cfg)
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        extra = torch.tensor([float(r["ddg_cached"])], dtype=torch.float32) if self.use_cached_ddg else torch.zeros(1)
        return g, y, extra


def collate_variants(batch):
    g = collate_graphs([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    extra = torch.stack([b[2] for b in batch])
    return g, y, extra


# ---------------------------------------------------------------------------
# Mutation-centered windows
# ---------------------------------------------------------------------------
# Fix for variant pathogenicity on long proteins: a fixed head-truncation to the
# first `max_length` residues drops the mutation entirely when it sits beyond the
# window (the majority of ClinVar variants). Embedding a window CENTERED on the
# mutated residue keeps the variant in the ESM context at the same embedding size.
def mutation_position(wt_seq, mt_seq):
    """0-based index of the single differing residue between equal-length wt/mt,
    or None if lengths differ or there isn't exactly one substitution."""
    if not wt_seq or not mt_seq or len(wt_seq) != len(mt_seq):
        return None
    diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mt_seq)) if a != b]
    return diffs[0] if len(diffs) == 1 else None


def variant_window(mt_seq: str, pos: int, window: int = 256):
    """A `window`-length slice of ``mt_seq`` centered on ``pos`` (the mutated
    residue), clamped to sequence bounds so the mutation is always inside it.
    Returns (window_subsequence, mutation_index_within_window)."""
    L = len(mt_seq)
    if window >= L:
        return mt_seq, pos
    half = window // 2
    start = max(0, min(pos - half, L - window))
    return mt_seq[start:start + window], pos - start


def add_window_column(df: pd.DataFrame, window: int = 256) -> pd.DataFrame:
    """Add a 'win_seq' column: the mutation-centered window of each mt_seq. The
    mutation position is derived from the wt/mt difference (deterministic, so it
    matches scripts.precompute_variant_windows exactly). Multi-substitution or
    length-mismatched rows fall back to the head window (pos=0)."""
    df = df.copy()
    wins = []
    for wt, mt in zip(df["wt_seq"].astype(str), df["mt_seq"].astype(str)):
        pos = mutation_position(wt, mt)
        if pos is None or pos < 0 or pos >= len(mt):
            pos = 0
        wins.append(variant_window(mt, pos, window)[0])
    df["win_seq"] = wins
    return df


class WindowedVariantDataset(Dataset):
    """Like VariantDataset, but builds each variant graph from a mutation-centered
    window embedding (cache keyed by the window subsequence). Expects a 'win_seq'
    column (add it once via add_window_column)."""
    def __init__(self, df: pd.DataFrame, store: EmbeddingStore, cfg: dict, use_cached_ddg=False):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.cfg = cfg
        self.use_cached_ddg = use_cached_ddg and ("ddg_cached" in df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        emb_m, cm_m, sm = self.store.get(r["win_seq"])
        g = build_graph(sm, emb_m, cm_m, None, self.cfg)
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        extra = torch.tensor([float(r["ddg_cached"])], dtype=torch.float32) if self.use_cached_ddg else torch.zeros(1)
        return g, y, extra
