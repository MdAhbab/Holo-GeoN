"""
src/data.py
===========
Data layer: preprocessed tables in, dense padded graph batches out.

Every mutation row is normalised to the same shape regardless of source:

    parent_seq, site (0-based), wt_aa, mt_aa, target/label, group[, cds, sigma]

The **parent** sequence is what gets encoded and cached. A mutant is not a
separate sequence to embed — it is a residue identity handed to the head — so
the cache is keyed by unique proteins (order 10^3-10^4) rather than by unique
mutants (order 10^5-10^6).

Exclusion accounting
--------------------
Rows that cannot be represented are dropped *loudly*: every loader returns a
:class:`LoadReport` counting what went and why (indels, multi-substitution
rows, missing sequences, absent cache entries, over-length proteins). v1 dropped
2.81% of ClinVar for having no cached embedding and 64% of ClinVar mutations
fell outside the truncation window, neither of which reached the results files.
An exclusion that is not counted is a silent change to the evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.encoder import EncodingStore
from src.features import CH_CAI, CH_MRNA, CH_VALID, N_MECH_CHANNELS, mechanistic_features
from src.graphs import (
    build_geometric_graph,
    build_mechanistic_graph,
    khop_neighborhood,
    window_edges,
)
from src.substitution import aa_index
from src.windowing import MutationParseError, mutation_position, window_bounds

# Column-name candidates, in priority order. configs/data.yaml can override any
# of these per dataset; see src/config.py.
_CANDIDATES: dict[str, list[str]] = {
    "wt_seq": ["wt_seq", "wildtype_seq", "seq_wt", "ref_seq", "wild_seq",
               "wildtype", "reference_seq", "source_sequence", "ref_protein"],
    "mt_seq": ["mt_seq", "mutant_seq", "seq_mt", "alt_seq", "mut_seq",
               "mutant", "alternate_seq", "target_sequence", "alt_protein"],
    "target": ["ddg", "ddG", "delta_delta_g", "ddg_exp", "DDG_mean", "target", "score"],
    "sigma": ["ddg_std", "ddG_std", "DDG_std", "ddg_err", "sigma", "std_err"],
    "label": ["label", "y", "pathogenic", "is_pathogenic",
              "clinical_significance", "clnsig"],
    "gene": ["gene", "gene_symbol", "symbol", "geneinfo"],
    "cds": ["wt_dna", "wt_cds", "ref_cds", "wt_cdna", "dna_wt", "cds_wt", "dna_seq"],
    "rg": ["rg", "radius_of_gyration", "rg_exp", "rg_angstrom"],
    "value": ["value", "mfi", "intensity", "retention_time", "rt", "expression"],
    "species": ["species", "species_id", "organism"],
}


def detect_columns(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> dict:
    """Map canonical names to actual columns, honouring config overrides."""
    overrides = overrides or {}
    lower = {c.lower(): c for c in df.columns}
    out: dict[str, str | None] = {}
    for key, candidates in _CANDIDATES.items():
        if key in overrides and overrides[key] in df.columns:
            out[key] = overrides[key]
            continue
        out[key] = next((lower[c.lower()] for c in candidates if c.lower() in lower), None)
    return out


@dataclass
class LoadReport:
    """What a loader kept, what it dropped, and why."""

    source: str = ""
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    #: Free-form provenance that must travel with the data: the sign convention
    #: applied, the exact label terms that were dropped, and similar. Anything a
    #: reader would need to interpret the kept rows correctly.
    notes: dict = field(default_factory=dict)

    def drop(self, reason: str, n: int = 1) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + n

    def note_value(self, key: str, value) -> None:
        """Record a distinct value that was seen and rejected, with a count."""
        bucket = self.notes.setdefault(key, {})
        k = str(value)
        bucket[k] = bucket.get(k, 0) + 1

    def as_dict(self) -> dict:
        total_dropped = sum(self.dropped.values())
        return {
            "source": self.source,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": total_dropped,
            "dropped_fraction": (total_dropped / self.rows_in) if self.rows_in else 0.0,
            "dropped_by_reason": dict(self.dropped),
            "notes": dict(self.notes),
        }

    def summary(self) -> str:
        d = self.as_dict()
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items())) or "none"
        return (f"[load] {self.source}: kept {d['rows_out']}/{d['rows_in']} "
                f"({d['dropped_fraction']:.2%} dropped) | reasons: {parts}")


# ---------------------------------------------------------------------------
# Normalised loading
# ---------------------------------------------------------------------------
#: Internal sign convention, fixed for the whole project.
#:
#: Round-1 review, R1-19: the convention was stated once, in a comment in the
#: web backend, contradicted in the baselines docstring, and never enforced.
#: Pearson r and Spearman rho are invariant to a global sign flip, so a table
#: ingested with the opposite convention produces an apparently excellent
#: correlation while the three-state labels, the interface's
#: stabilising/destabilising verdict and every qualitative claim are inverted -
#: with nothing in any artifact revealing it.
DDG_CONVENTION = "ddG = dG_mut - dG_wt; positive is STABILISING"
_SIGN_KEYS = {"stabilizing_positive": 1.0, "destabilizing_positive": -1.0}


def _resolve_ddg_sign(overrides: dict[str, str] | None) -> tuple[float, str]:
    """Sign multiplier that maps a dataset's convention onto the internal one."""
    key = (overrides or {}).get("ddg_sign", "stabilizing_positive")
    if key not in _SIGN_KEYS:
        raise ValueError(
            f"ddg_sign must be one of {sorted(_SIGN_KEYS)}, got {key!r}. "
            f"Set it under this dataset's key in configs/data.yaml. "
            f"Internally, {DDG_CONVENTION}."
        )
    return _SIGN_KEYS[key], key


def load_mutation_table(
    path: str,
    kind: str,
    overrides: dict[str, str] | None = None,
    max_rows: int | None = None,
    max_protein_length: int = 5000,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Load a mutation table into the normalised schema.

    Args:
        kind : ``"stability"`` (needs a numeric target) or ``"pathogenicity"``
               (needs a binary label).

    Targets are normalised to the internal sign convention
    (:data:`DDG_CONVENTION`) using the dataset's ``ddg_sign`` key, and the
    applied sign is recorded in the load report. A sanity check follows: over a
    large mutation set the mean ddG must be negative, because random
    substitutions are destabilising on average. A positive mean is reported
    loudly, since the most likely cause is an unlabelled convention flip.

    Grouping for leakage control: stability groups by parent sequence, and
    pathogenicity groups by gene where a gene column exists, falling back to the
    parent sequence. ``scripts/make_splits.py`` refines these into
    identity-clustered groups.
    """
    df = pd.read_parquet(path)
    report = LoadReport(source=f"{path} ({kind})", rows_in=len(df))
    if max_rows:
        df = df.iloc[:max_rows].copy()
        report.rows_in = len(df)

    col = detect_columns(df, overrides)
    needed = ["wt_seq", "mt_seq"] + (["target"] if kind == "stability" else ["label"])
    missing = [k for k in needed if col[k] is None]
    if missing:
        raise ValueError(
            f"{path}: could not find columns for {missing}. Present: {list(df.columns)}. "
            f"Set the real names under this dataset's key in configs/data.yaml."
        )

    sign, sign_key = (_resolve_ddg_sign(overrides) if kind == "stability" else (1.0, "n/a"))
    report.notes["ddg_sign"] = sign_key
    report.notes["convention"] = DDG_CONVENTION if kind == "stability" else "n/a"

    if kind != "stability":
        _reject_legacy_integer_labels(df[col["label"]], path)

    wt = df[col["wt_seq"]].astype(str).str.upper()
    mt = df[col["mt_seq"]].astype(str).str.upper()

    records: list[dict] = []
    for i in range(len(df)):
        w, m = wt.iat[i], mt.iat[i]
        if not w or not m or w == "NAN" or m == "NAN":
            report.drop("missing_sequence")
            continue
        if len(w) > max_protein_length:
            report.drop("protein_too_long")
            continue
        try:
            site = mutation_position(w, m)
        except MutationParseError as exc:
            reason = "length_mismatch_indel" if "length mismatch" in str(exc) else "not_single_substitution"
            report.drop(reason)
            continue

        rec: dict = {
            "parent_seq": w,
            "site": site,
            "wt_aa": w[site],
            "mt_aa": m[site],
        }
        if kind == "stability":
            try:
                rec["target"] = sign * float(df[col["target"]].iat[i])
            except (TypeError, ValueError):
                report.drop("unparseable_target")
                continue
            if not np.isfinite(rec["target"]):
                report.drop("non_finite_target")
                continue
            rec["sigma"] = (
                float(df[col["sigma"]].iat[i]) if col["sigma"] else float("nan")
            )
            rec["group"] = w
        else:
            raw_label = df[col["label"]].iat[i]
            rec["label"] = _binary_label(raw_label)
            if rec["label"] is None:
                report.drop("unmappable_label")
                # Which terms were rejected, so an unrecognised vocabulary shows
                # up in the leakage report rather than vanishing.
                report.note_value("rejected_label_terms", raw_label)
                continue
            rec["group"] = str(df[col["gene"]].iat[i]) if col["gene"] else w

        rec["cds"] = str(df[col["cds"]].iat[i]) if col["cds"] else None
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    report.rows_out = len(out)

    if kind == "stability" and len(out) >= 100:
        mean_ddg = float(out["target"].mean())
        report.notes["mean_ddg_after_sign_normalisation"] = mean_ddg
        if mean_ddg > 0:
            # Random substitutions are destabilising on average, so a positive
            # mean under the internal convention almost always means the source
            # table uses the opposite sign and ddg_sign was left at its default.
            print(
                f"[load] WARNING: mean ddG is {mean_ddg:+.3f} after applying "
                f"ddg_sign={sign_key!r}. Under {DDG_CONVENTION}, the mean over a "
                f"large mutation set should be NEGATIVE. This table may use the "
                f"opposite convention - set ddg_sign: destabilizing_positive "
                f"under its key in configs/data.yaml. Correlation metrics are "
                f"invariant to this flip and will NOT reveal it."
            )
            report.notes["sign_warning"] = (
                "mean ddG is positive under the internal convention; the source "
                "table may use the opposite sign"
            )

    return out.reset_index(drop=True), report


def _reject_legacy_integer_labels(series: pd.Series, path: str) -> None:
    """
    Refuse a numeric label column that looks like the legacy integer CLNSIG scale.

    That encoding is 0 Uncertain, 2 Benign, 3 Likely benign, 4 Likely pathogenic,
    5 Pathogenic. Reading it as "any nonzero is pathogenic" silently maps Benign
    and Likely-benign to the positive class, which is not recoverable from the
    run artifacts - so it fails here instead.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return
    values = set(pd.unique(series.dropna()).tolist())
    if values and not values <= {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(
            f"{path}: the label column holds numeric values {sorted(values)[:8]}, "
            f"which is not a 0/1 encoding. If this is the legacy integer CLNSIG "
            f"scale (2=Benign, 3=Likely benign, 4=Likely pathogenic, 5=Pathogenic), "
            f"map it to the CLNSIG text vocabulary or to explicit 0/1 before "
            f"loading. Holo-GeoN will not guess: reading 'any nonzero' as "
            f"pathogenic silently labels benign variants pathogenic."
        )


# ClinVar CLNSIG vocabulary, matched exactly rather than by substring.
#
# Round-1 review, R1-04: substring matching mapped
# `Conflicting_interpretations_of_pathogenicity` to pathogenic, because the
# string contains "pathogenic". Conflicting-classification variants are one of
# the largest CLNSIG classes and are explicitly NOT a pathogenic call - ACMG
# treats them as unresolved. They are also the hardest, most ambiguous variants,
# so assigning them all to the positive class corrupts the labels and inflates
# the positive rate. Because those rows were *kept*, nothing in the exclusion
# accounting revealed it.
_BENIGN_TERMS = {
    "benign",
    "likely_benign",
    "benign/likely_benign",
}
_PATHOGENIC_TERMS = {
    "pathogenic",
    "likely_pathogenic",
    "pathogenic/likely_pathogenic",
}
# Present in ClinVar and deliberately not a binary call. Listed explicitly so a
# new vocabulary term is dropped-and-counted rather than guessed at.
_UNRESOLVED_TERMS = {
    "conflicting_interpretations_of_pathogenicity",
    "conflicting_classifications_of_pathogenicity",
    "uncertain_significance",
    "uncertain_risk_allele",
    "not_provided",
    "no_classification_for_the_single_variant",
    "no_classification_from_unflagged_records",
    "drug_response",
    "association",
    "association_not_found",
    "risk_factor",
    "protective",
    "affects",
    "confers_sensitivity",
    "other",
}


def _normalise_clnsig(value: str) -> str:
    """Canonical form of a CLNSIG term: lowercase, underscores, no modifiers."""
    s = str(value).strip().lower().replace(" ", "_")
    # ClinVar appends secondary terms after '|' and penetrance modifiers after
    # ',': `Likely_pathogenic|other`, `Pathogenic,_low_penetrance`. The primary
    # classification is the first field.
    s = s.split("|", 1)[0]
    s = s.split(",", 1)[0]
    return s.strip("_")


def _binary_label(value) -> int | None:
    """
    Map a clinical-significance value to 1 (pathogenic) / 0 (benign), or None.

    None means "not a binary call" and the caller counts it as an exclusion.
    Anything outside the known vocabulary returns None rather than being
    guessed at, so an unrecognised term shows up in the leakage report instead
    of silently joining a class.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        # Only an explicit 0/1 encoding is accepted. The legacy integer CLNSIG
        # scale (0 Uncertain, 2 Benign, 3 Likely benign, 4 Likely pathogenic,
        # 5 Pathogenic) is NOT interpreted here: reading "any nonzero" as
        # pathogenic silently mapped Benign and Likely-benign to the positive
        # class, and that is not recoverable from the run artifacts.
        v = float(value)
        if v == 0.0:
            return 0
        if v == 1.0:
            return 1
        return None

    s = _normalise_clnsig(value)
    if not s or s == "nan":
        return None
    if s in _UNRESOLVED_TERMS:
        return None
    if s in _BENIGN_TERMS:
        return 0
    if s in _PATHOGENIC_TERMS:
        return 1
    if s in {"true", "yes"}:
        return 1
    if s in {"false", "no"}:
        return 0
    return None


def load_sequence_table(
    path: str,
    value_key: str,
    overrides: dict[str, str] | None = None,
    max_rows: int | None = None,
    max_protein_length: int = 5000,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Load a whole-sequence table (IDR radius of gyration, MFI, retention time).

    ``value_key`` is the canonical column to read as the regression target
    (``"rg"`` or ``"value"``).
    """
    df = pd.read_parquet(path)
    report = LoadReport(source=f"{path} ({value_key})", rows_in=len(df))
    if max_rows:
        df = df.iloc[:max_rows].copy()
        report.rows_in = len(df)

    col = detect_columns(df, overrides)
    seq_col = col["wt_seq"]
    if seq_col is None or col[value_key] is None:
        raise ValueError(
            f"{path}: need a sequence column and a {value_key!r} column. "
            f"Present: {list(df.columns)}. Set them in configs/data.yaml."
        )

    records = []
    for i in range(len(df)):
        s = str(df[seq_col].iat[i]).upper()
        if not s or s == "NAN":
            report.drop("missing_sequence")
            continue
        if len(s) > max_protein_length:
            report.drop("protein_too_long")
            continue
        try:
            v = float(df[col[value_key]].iat[i])
        except (TypeError, ValueError):
            report.drop("unparseable_value")
            continue
        if not np.isfinite(v):
            report.drop("non_finite_value")
            continue
        rec = {"parent_seq": s, "target": v, "group": s}
        rec["cds"] = str(df[col["cds"]].iat[i]) if col["cds"] else None
        if col["species"] is not None:
            try:
                rec["species_id"] = int(df[col["species"]].iat[i])
            except (TypeError, ValueError):
                rec["species_id"] = 0
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    report.rows_out = len(out)
    return out.reset_index(drop=True), report


def filter_to_cache(
    df: pd.DataFrame, store: EncodingStore, report: LoadReport
) -> pd.DataFrame:
    """
    Keep only rows whose parent protein has a cached encoding, counting the rest.

    Silently loading an uncached row raised FileNotFoundError in v1; silently
    dropping it changes the evaluation set. Neither is acceptable without a
    number attached.
    """
    keep = df["parent_seq"].map(store.has)
    report.drop("no_cached_encoding", int((~keep).sum()))
    out = df[keep].reset_index(drop=True)
    report.rows_out = len(out)
    return out


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_sample_graph(
    record: dict,
    site: int | None,
    window: int,
    cfg: dict,
    max_nodes: int | None = None,
) -> dict:
    """
    Build both edge sets for one cached protein.

    ``record`` is an :class:`EncodingStore` entry. ``site`` is the mutation
    position in whole-protein coordinates.

    With a ``site``, the graph covers a ``window``-long slice centred on it, so
    the mutation is always present and always in context.

    With ``site is None`` — the whole-sequence tasks — the graph covers the
    **entire protein**, not a window. Round-1 review, R1-07: anchoring at 0 and
    windowing gave those heads residues [0, window) only, which is precisely the
    head truncation this project exists to remove. It is worst for the radius of
    gyration, which is dominated by chain length (R_g = R_0 N^v): a 900-residue
    and a 300-residue chain sharing an N-terminus were indistinguishable, and
    the mask could not even recover the length. ``max_nodes`` caps pathological
    inputs; when it bites, the caller is told via ``truncated`` rather than
    finding out from a silently short mask.
    """
    emb = record["emb"]
    length = emb.shape[0]
    truncated = False

    if site is None:
        n = length if max_nodes is None else min(length, max_nodes)
        truncated = n < length
        start, end, anchor = 0, n, 0
    else:
        anchor = min(site, length - 1)
        start, end = window_bounds(length, anchor, window)
        n = end - start

    esm = emb[start:end].float()

    mech = record.get("mech")
    if mech is None:
        # No precomputed mechanistic channels: protein-only, which is exactly
        # what mech_valid == 0 encodes. E_mech is then empty by construction.
        mech = mechanistic_features(record["seq"][start:end], None, n)
    else:
        mech = mech[start:end].float()

    ce, cp = window_edges(record["contact_edges"], record["contact_probs"], start, end)
    edge_geo, attr_geo = build_geometric_graph(
        n, ce, cp,
        top_k=cfg.get("graph_top_k", 16),
        threshold=cfg.get("geo_threshold", 0.30),
    )
    edge_mech, attr_mech = build_mechanistic_graph(
        mech[:, CH_CAI], mech[:, CH_MRNA], mech[:, CH_VALID],
        eps=cfg.get("eps", 0.10),
        delta=cfg.get("delta", 0.10),
        max_degree=cfg.get("mech_max_degree", 8),
    )

    local_site = anchor - start
    neighborhood = khop_neighborhood(
        edge_geo, local_site, n, hops=cfg.get("neighborhood_hops", 2)
    )
    return {
        "esm": esm,
        "mech": mech,
        "edge_geo": edge_geo,
        "attr_geo": attr_geo,
        "edge_mech": edge_mech,
        "attr_mech": attr_mech,
        "site": local_site,
        "neighborhood": neighborhood,
        "n": n,
        # True chain length, independent of how many residues this graph holds.
        # The whole-sequence heads need it explicitly: pooling cannot recover a
        # length the graph does not span.
        "length": length,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class MutationDataset(Dataset):
    """
    Mutation rows (stability or pathogenicity) over cached parent encodings.

    Args:
        df     : normalised table from :func:`load_mutation_table`.
        store  : encoding cache.
        cfg    : graph/window settings (a :class:`src.config.Config` or a dict).
        target : ``"target"`` for regression, ``"label"`` for classification.
    """

    def __init__(self, df: pd.DataFrame, store: EncodingStore, cfg, target: str = "target"):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.cfg = cfg if isinstance(cfg, dict) else cfg.as_dict()
        self.window = int(self.cfg.get("window", 256))
        self.target = target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        rec = self.store.get(row["parent_seq"])
        g = build_sample_graph(rec, int(row["site"]), self.window, self.cfg)
        g["wt_idx"] = aa_index(row["wt_aa"])
        g["mt_idx"] = aa_index(row["mt_aa"])
        g["y"] = float(row[self.target])
        g["sigma"] = float(row["sigma"]) if "sigma" in row and pd.notna(row["sigma"]) else float("nan")
        return g


class SequenceDataset(Dataset):
    """Whole-sequence rows (IDR, MFI, retention time) over cached encodings."""

    def __init__(self, df: pd.DataFrame, store: EncodingStore, cfg):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.cfg = cfg if isinstance(cfg, dict) else cfg.as_dict()
        self.window = int(self.cfg.get("window", 256))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        rec = self.store.get(row["parent_seq"])
        # site=None -> the whole protein, not a window (see build_sample_graph).
        g = build_sample_graph(
            rec, None, self.window, self.cfg,
            max_nodes=int(self.cfg.get("max_sequence_nodes", 2048)),
        )
        g["wt_idx"] = 0
        g["mt_idx"] = 0
        g["y"] = float(row["target"])
        g["sigma"] = float("nan")
        g["species_id"] = int(row["species_id"]) if "species_id" in row else 0
        return g


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def collate_graphs(samples: list[dict]) -> dict:
    """
    Pad a list of windowed graphs into one dense batch.

    Node indices are offset by ``b * L`` so a single flattened ``(B*L, C)``
    tensor feeds the GATv2 layers. Padding positions carry no edges and are
    masked out of every pool.
    """
    b = len(samples)
    lengths = [s["n"] for s in samples]
    L = max(lengths)
    d = samples[0]["esm"].shape[-1]

    esm = torch.zeros(b, L, d)
    mech = torch.zeros(b, L, N_MECH_CHANNELS)
    mask = torch.zeros(b, L, dtype=torch.bool)
    neighborhood = torch.zeros(b, L, dtype=torch.bool)

    eg, ag, em, am = [], [], [], []
    for i, s in enumerate(samples):
        n = s["n"]
        esm[i, :n] = s["esm"]
        mech[i, :n] = s["mech"]
        mask[i, :n] = True
        neighborhood[i, :n] = s["neighborhood"]
        off = i * L
        if s["edge_geo"].numel():
            eg.append(s["edge_geo"] + off)
            ag.append(s["attr_geo"])
        if s["edge_mech"].numel():
            em.append(s["edge_mech"] + off)
            am.append(s["attr_mech"])

    def _cat(edges, attrs):
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3))
        return torch.cat(edges, dim=1), torch.cat(attrs, dim=0)

    edge_geo, attr_geo = _cat(eg, ag)
    edge_mech, attr_mech = _cat(em, am)

    batch = {
        "esm": esm,
        "mech": mech,
        "mask": mask,
        "edge_geo": edge_geo,
        "attr_geo": attr_geo,
        "edge_mech": edge_mech,
        "attr_mech": attr_mech,
        "site": torch.tensor([s["site"] for s in samples], dtype=torch.long),
        "neighborhood_mask": neighborhood,
        "wt_idx": torch.tensor([s["wt_idx"] for s in samples], dtype=torch.long),
        "mt_idx": torch.tensor([s["mt_idx"] for s in samples], dtype=torch.long),
        "y": torch.tensor([s["y"] for s in samples], dtype=torch.float32),
        "sigma": torch.tensor([s["sigma"] for s in samples], dtype=torch.float32),
        # True chain length per sample. The whole-sequence heads consume it
        # directly; a pool over the graph cannot recover it.
        "seq_length": torch.tensor(
            [float(s.get("length", s["n"])) for s in samples], dtype=torch.float32
        ),
    }
    if "species_id" in samples[0]:
        batch["species_id"] = torch.tensor(
            [s["species_id"] for s in samples], dtype=torch.long
        )
    return batch


def move_batch(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


__all__ = [
    "LoadReport",
    "detect_columns",
    "load_mutation_table",
    "load_sequence_table",
    "filter_to_cache",
    "build_sample_graph",
    "MutationDataset",
    "SequenceDataset",
    "collate_graphs",
    "move_batch",
]
