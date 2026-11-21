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

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.encoder import EncodingStore, seq_key
from src.features import CH_CAI, CH_MRNA, CH_VALID, N_MECH_CHANNELS, mechanistic_features
from src.geometry import STR_NODE_DIM
from src.graphs import (
    EDGE_DIM,
    N_TOPO_CHANNELS,
    STR_EDGE_DIM,
    build_geometric_graph,
    build_mechanistic_graph,
    build_structure_graph,
    build_true_contact_graph,
    contact_topology_features,
    khop_neighborhood,
    window_edges,
)
from src.substitution import aa_index
from src.windowing import MutationParseError, mutation_position, window_bounds

# Column-name candidates, in priority order. configs/data.yaml can override any
# of these per dataset; see src/config.py.
_CANDIDATES: dict[str, list[str]] = {
    "wt_seq": ["wt_seq", "wildtype_seq", "seq_wt", "ref_seq", "wild_seq",
               "wildtype", "reference_seq", "source_sequence", "ref_protein",
               "parent_seq"],
    "mt_seq": ["mt_seq", "mutant_seq", "seq_mt", "alt_seq", "mut_seq",
               "mutant", "alternate_seq", "target_sequence", "alt_protein"],
    "site": ["site"],
    "wt_aa": ["wt_aa", "ref_aa"],
    "mt_aa": ["mt_aa", "mut_aa", "alt_aa"],
    "target": ["ddg", "ddG", "delta_delta_g", "ddg_exp", "DDG_mean", "target", "score"],
    "sigma": ["ddg_std", "ddG_std", "DDG_std", "ddg_err", "sigma", "std_err"],
    "label": ["label", "y", "pathogenic", "is_pathogenic",
              "clinical_significance", "clnsig"],
    "gene": ["gene", "gene_symbol", "symbol", "geneinfo"],
    "cds": ["wt_dna", "wt_cds", "ref_cds", "wt_cdna", "dna_wt", "cds_wt", "dna_seq"],
    "rg": ["rg", "radius_of_gyration", "rg_exp", "rg_angstrom"],
    "value": ["value", "mfi", "intensity", "retention_time", "rt", "expression"],
    "species": ["species", "species_id", "organism"],
    # The protein a row is a variant OF. For a deep mutational scan every row is
    # a distinct sequence, so grouping by the sequence puts 400 variants of one
    # construct into 400 groups and any split of them leaks (round-2 review,
    # R2-04). This column is what makes such a table splittable; set it in
    # configs/data.yaml when auto-detection misses it.
    "parent": ["parent", "parent_id", "parent_seq_id", "construct", "construct_id",
               "protein", "protein_id", "uniprot", "uniprot_id", "accession",
               "wt_name", "wt_id", "variant_of"],
}


def clean_cds(value) -> str | None:
    """
    Normalise a coding-sequence cell to a real sequence or None.

    ``str(value)`` on a missing cell yields the literal ``"nan"`` (or
    ``"None"``), and that string is TRUTHY. ``src/features.mechanistic_features``
    treats any truthy ``dna_seq`` as a real coding sequence: it would stamp
    ``mech_valid = 1`` and fill the CAI and mRNA-folding channels from a
    three-character non-sequence, for a row that has no CDS at all. The run
    would then report mechanistic coverage it does not have, and the
    ``no_mech_stream`` ablation would be measured against partly fabricated
    channels.

    A row with no coding sequence must read as having no coding sequence.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):      # arrays and other non-scalars
        pass
    text = str(value).strip()
    return text if text and text.lower() not in ("nan", "none", "<na>") else None


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
    has_mt_seq = col["mt_seq"] is not None
    has_site_cols = all(col[k] is not None for k in ("site", "wt_aa", "mt_aa"))
    needed = ["wt_seq"] + (["target"] if kind == "stability" else ["label"])
    if not has_mt_seq and not has_site_cols:
        needed.append("mt_seq")
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
    mt = df[col["mt_seq"]].astype(str).str.upper() if has_mt_seq else None
    site_ser = df[col["site"]] if (not has_mt_seq and has_site_cols) else None
    wt_aa_ser = df[col["wt_aa"]].astype(str).str.strip().str.upper() if (not has_mt_seq and has_site_cols) else None
    mt_aa_ser = df[col["mt_aa"]].astype(str).str.strip().str.upper() if (not has_mt_seq and has_site_cols) else None

    records: list[dict] = []
    for i in range(len(df)):
        w = wt.iat[i]
        if not w or w == "NAN":
            report.drop("missing_sequence")
            continue
        if len(w) > max_protein_length:
            report.drop("protein_too_long")
            continue

        if has_mt_seq:
            m = mt.iat[i]
            if not m or m == "NAN":
                report.drop("missing_sequence")
                continue
            try:
                site = mutation_position(w, m)
            except MutationParseError as exc:
                reason = "length_mismatch_indel" if "length mismatch" in str(exc) else "not_single_substitution"
                report.drop(reason)
                continue
            wa = w[site]
            ma = m[site]
        else:
            try:
                raw_site = site_ser.iat[i]
                if pd.isna(raw_site):
                    raise ValueError("nan site")
                site = int(raw_site)
            except (TypeError, ValueError):
                report.drop("unparseable_site")
                continue
            wa = wt_aa_ser.iat[i]
            ma = mt_aa_ser.iat[i]
            if not (0 <= site < len(w)):
                report.drop("site_out_of_bounds")
                continue
            if len(wa) != 1 or len(ma) != 1 or wa == ma:
                report.drop("not_single_substitution")
                continue
            if w[site] != wa:
                report.drop("wt_aa_mismatch")
                continue

        rec: dict = {
            "source_row_id": i,
            "parent_seq": w,
            "site": site,
            "wt_aa": wa,
            "mt_aa": ma,
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

        rec["cds"] = clean_cds(df[col["cds"]].iat[i]) if col["cds"] else None
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
        # Round-2 review, R2-04: `group` used to be the row's own sequence
        # unconditionally, which is the identity relation dressed up as a
        # grouping. A declared parent/construct column wins; a gene column is
        # the next best; only then does it fall back to the sequence, and the
        # report says so rather than leaving the caller to assume.
        declared = None
        for key in ("parent", "gene"):
            if col.get(key) is not None:
                value = df[col[key]].iat[i]
                if value is not None and str(value).lower() not in ("", "nan", "none"):
                    declared = f"{key}:{value}"
                    break
        rec = {"parent_seq": s, "target": v, "group": declared or s}
        rec["cds"] = clean_cds(df[col["cds"]].iat[i]) if col["cds"] else None
        if col["species"] is not None:
            try:
                rec["species_id"] = int(df[col["species"]].iat[i])
            except (TypeError, ValueError):
                rec["species_id"] = 0
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    report.rows_out = len(out)
    group_col = next((k for k in ("parent", "gene") if col.get(k) is not None), None)
    report.notes["group_source"] = (
        f"declared column {col[group_col]!r} (canonical {group_col!r})" if group_col
        else "the row's own sequence - NO grouping column was found, so a deep "
             "mutational scan of one construct would be split across the "
             "boundary. Declare a parent/construct column in configs/data.yaml"
    )
    report.notes["group_declared"] = bool(group_col)
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
def _load_true_contact_map(seq: str, true_contacts_dir: str | None) -> torch.Tensor | None:
    if not true_contacts_dir:
        return None
    path = os.path.join(true_contacts_dir, seq_key(seq) + ".npy")
    if not os.path.isfile(path):
        return None
    return torch.from_numpy(np.load(path))


def _apply_cds_dropout(
    mech: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero CDS-derived channels (mRNA/CAI/valid) and replace E_mech with self-loops."""
    mech_d = mech.clone()
    mech_d[:, CH_MRNA] = 0.0
    mech_d[:, CH_CAI] = 0.0
    mech_d[:, CH_VALID] = 0.0
    loops = torch.arange(n, dtype=torch.long)
    ei = torch.stack([loops, loops], dim=0)
    ea = torch.zeros((n, EDGE_DIM), dtype=torch.float32)
    ea[:, 0] = 1.0
    return mech_d, ei, ea


def build_sample_graph(
    record: dict,
    site: int | None,
    window: int,
    cfg: dict,
    max_nodes: int | None = None,
    drop_cds: bool = False,
) -> dict:
    """
    Build all graph streams and v2 node features for one cached protein.

    Supports:
      * ``E_geo`` (predicted ESM-2 contacts, or real C-beta contacts when
        ``cfg["true_contact_edges"]`` is True — Arm E in ``GEOMETRIC_PLAN.md``)
      * ``E_mech`` (translational coupling; optionally masked via ``drop_cds``
        during training for CDS modality dropout)
      * ``E_str`` (optional SE(3)-invariant 3D coordinate stream when
        ``cfg["use_str_stream"]`` is True — Arms B/C/D)
      * ``topo`` (6-channel structure-free contact-burial/packing + entropy)
      * ``layer_embs`` (optional multi-layer ESM-2 taps)
      * ``wt_logprobs`` (optional 20-AA WT-marginal log-probabilities)
      * ``neighborhood_1hop`` and ``neighborhood`` (1-hop vs k-hop site masks)
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
        mech = mechanistic_features(record["seq"][start:end], None, n)
    else:
        mech = mech[start:end].float()

    # E_geo: predicted ESM-2 contacts OR real C-beta contact map (Arm E)
    if cfg.get("true_contact_edges"):
        tmap = _load_true_contact_map(record["seq"], cfg.get("true_contacts_dir"))
        if tmap is not None and tmap.shape[0] >= end:
            edge_geo, attr_geo = build_true_contact_graph(
                n, tmap[start:end, start:end],
                top_k=cfg.get("graph_top_k", 16),
                min_sep=cfg.get("contact_min_sep", 3),
            )
        else:
            ce, cp = window_edges(record["contact_edges"], record["contact_probs"], start, end)
            edge_geo, attr_geo = build_geometric_graph(
                n, ce, cp,
                top_k=cfg.get("graph_top_k", 16),
                threshold=cfg.get("geo_threshold", 0.30),
            )
    else:
        ce, cp = window_edges(record["contact_edges"], record["contact_probs"], start, end)
        edge_geo, attr_geo = build_geometric_graph(
            n, ce, cp,
            top_k=cfg.get("graph_top_k", 16),
            threshold=cfg.get("geo_threshold", 0.30),
        )

    if drop_cds:
        mech, edge_mech, attr_mech = _apply_cds_dropout(mech, n)
    else:
        edge_mech, attr_mech = build_mechanistic_graph(
            mech[:, CH_CAI], mech[:, CH_MRNA], mech[:, CH_VALID],
            eps=cfg.get("eps", 0.10),
            delta=cfg.get("delta", 0.10),
            max_degree=cfg.get("mech_max_degree", 8),
        )

    ent = record.get("site_entropy")
    ent_slice = ent[start:end].float() if ent is not None else None
    topo = contact_topology_features(n, edge_geo, attr_geo, mech, site_entropy=ent_slice)

    local_site = anchor - start
    neighborhood_1hop = khop_neighborhood(edge_geo, local_site, n, hops=1)
    neighborhood = khop_neighborhood(
        edge_geo, local_site, n, hops=cfg.get("neighborhood_hops", 2)
    )

    sample = {
        "esm": esm,
        "mech": mech,
        "topo": topo,
        "edge_geo": edge_geo,
        "attr_geo": attr_geo,
        "edge_mech": edge_mech,
        "attr_mech": attr_mech,
        "site": local_site,
        "neighborhood_1hop": neighborhood_1hop,
        "neighborhood": neighborhood,
        "n": n,
        "length": length,
        "truncated": truncated,
    }

    if record.get("layer_embs") is not None:
        sample["layer_embs"] = record["layer_embs"][:, start:end].float()
    if record.get("wt_logprobs") is not None:
        lp_win = record["wt_logprobs"][start:end].float()
        sample["wt_logprobs"] = lp_win
        sample["wt_logprobs_site"] = lp_win[local_site]

    if cfg.get("use_str_stream"):
        coords = record.get("coords")
        plddt = record.get("plddt")
        cvalid = record.get("coord_valid")
        c_slice = coords[start:end] if coords is not None else None
        p_slice = plddt[start:end] if plddt is not None else None
        v_slice = cvalid[start:end] if cvalid is not None else None
        e_str, a_str, s_node = build_structure_graph(
            c_slice, p_slice, v_slice, n,
            topology_only=bool(cfg.get("str_topology_only", False)),
        )
        sample["edge_str"] = e_str
        sample["attr_str"] = a_str
        sample["str_node"] = s_node
    return sample


# ---------------------------------------------------------------------------
# Parent-level graph cache (avoids rebuilding identical graphs 740x/epoch)
# ---------------------------------------------------------------------------
class _ParentGraphCache:
    """
    Caches whole-protein graph structures for parents with ``L <= window``.

    On MegaScale (412 parents, L in [32, 72], window 256), every mutation of a
    parent shares the exact same ``E_geo``, ``E_mech``, ``topo``, and optional
    ``E_str``. Caching once per parent cuts CPU graph construction by ~740x.
    """

    def __init__(self, cfg: dict, window: int, max_entries: int = 1024):
        self.cfg = cfg
        self.window = int(window)
        self.max_entries = int(max_entries)
        self._base: dict[str, dict] = {}
        self._khops: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_whole(self, seq: str, record: dict) -> dict | None:
        length = record["emb"].shape[0]
        if length > self.window:
            return None
        cached = self._base.get(seq)
        if cached is not None:
            return cached
        g = build_sample_graph(record, 0, self.window, self.cfg, drop_cds=False)
        if len(self._base) < self.max_entries:
            self._base[seq] = g
        return g

    def get_khops(self, seq: str, edge_geo: torch.Tensor, site: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq, site)
        hit = self._khops.get(key)
        if hit is not None:
            return hit
        m1 = khop_neighborhood(edge_geo, site, n, hops=1)
        m2 = khop_neighborhood(edge_geo, site, n, hops=self.cfg.get("neighborhood_hops", 2))
        if len(self._khops) < self.max_entries * 64:
            self._khops[key] = (m1, m2)
        return m1, m2


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
        noise_std : standard deviation of Gaussian noise added to the cached
            embeddings. TRAINING SPLIT ONLY — pass 0.0 for val and test.

    On ``noise_std``
    ----------------
    The encoder is frozen and its output is cached, so row *i* presents the model
    with a byte-identical input at every epoch. The readout then anchors on the
    mutated site, which makes (parent, site) a stable key: 17,071 distinct values
    over 305,995 training rows. A network with 20.6M parameters can store a
    response per key, and measurement says it does - train Pearson reaches 0.954
    against 0.758 on held-out proteins, and the fitted sigma is honest on train
    (1.12x) while being 2.5x overconfident on test.

    Raising weight decay 100x did not touch this, and could not: weight decay
    constrains the MAP, not the keys. Perturbing the embedding is the one cheap
    change that attacks the precondition instead - row *i* is a different input
    each epoch, so there is no fixed table to memorise.

    The noise is drawn ONCE per sample, before the backbone, so the context
    vector ``z`` is a single draw shared by both ``s()`` calls in the
    substitution head. ``ddG = s(z, e_mt) - s(z, e_wt)`` therefore stays exactly
    antisymmetric.
    """

    def __init__(self, df: pd.DataFrame, store: EncodingStore, cfg,
                 target: str = "target", noise_std: float = 0.0,
                 cds_dropout: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.cfg = cfg if isinstance(cfg, dict) else cfg.as_dict()
        self.window = int(self.cfg.get("window", 256))
        self.target = target
        self.noise_std = float(noise_std)
        self.cds_dropout = float(cds_dropout)
        self._graph_cache = _ParentGraphCache(self.cfg, self.window)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        seq = row["parent_seq"]
        site = int(row["site"])
        rec = self.store.get(seq)
        drop_cds = self.cds_dropout > 0.0 and float(torch.rand(())) < self.cds_dropout

        base = self._graph_cache.get_whole(seq, rec)
        if base is not None:
            g = dict(base)
            s_clamped = min(site, g["n"] - 1)
            m1, m2 = self._graph_cache.get_khops(seq, g["edge_geo"], s_clamped, g["n"])
            g["site"] = s_clamped
            g["neighborhood_1hop"] = m1
            g["neighborhood"] = m2
            if "wt_logprobs" in base:
                g["wt_logprobs_site"] = base["wt_logprobs"][s_clamped]
            if drop_cds:
                g["mech"], g["edge_mech"], g["attr_mech"] = _apply_cds_dropout(g["mech"], g["n"])
        else:
            g = build_sample_graph(rec, site, self.window, self.cfg, drop_cds=drop_cds)

        if self.noise_std > 0.0:
            g["esm"] = g["esm"] + self.noise_std * torch.randn_like(g["esm"])
        g["wt_idx"] = aa_index(row["wt_aa"])
        g["mt_idx"] = aa_index(row["mt_aa"])
        g["y"] = float(row[self.target])
        g["sigma"] = float(row["sigma"]) if "sigma" in row and pd.notna(row["sigma"]) else float("nan")
        return g


class ProteinBatchMutationDataset(Dataset):
    """
    Parent-grouped mutation dataset for 15-40x faster MegaScale training.

    Each item represents one parent protein (with ``L <= window``) paired with
    a chunk of ``muts_per_protein`` mutations. Running the backbone ONCE per
    parent protein and gathering all mutation contexts in parallel via
    :meth:`MultiScaleSiteReadout.gather_protein_batch` eliminates ~98% of
    redundant backbone forward passes while balancing gradient weight across
    proteins of different mutation counts.

    During training, each epoch iterates over ``visits_per_parent`` disjoint
    shuffled chunks per parent (covering the entire dataset once per epoch).
    During evaluation (``train=False``), every row is visited deterministically
    in exact order without duplication.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        store: EncodingStore,
        cfg,
        target: str = "target",
        muts_per_protein: int = 64,
        train: bool = True,
        noise_std: float = 0.0,
        cds_dropout: float = 0.0,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.cfg = cfg if isinstance(cfg, dict) else cfg.as_dict()
        self.window = int(self.cfg.get("window", 256))
        self.target = target
        self.muts_per_protein = max(int(muts_per_protein), 1)
        self.train = bool(train)
        self.noise_std = float(noise_std)
        self.cds_dropout = float(cds_dropout)
        self.seed = int(seed)
        self.epoch = 0
        self._graph_cache = _ParentGraphCache(self.cfg, self.window)

        # Group row indices by parent_seq
        self._groups: list[tuple[str, np.ndarray]] = []
        for seq, grp in self.df.groupby("parent_seq", sort=False):
            self._groups.append((str(seq), grp.index.to_numpy(dtype=np.int64)))

        # Pre-extract fast numpy arrays for O(1) mutation slicing
        self._site = self.df["site"].to_numpy(dtype=np.int64)
        self._wt_idx = np.array([aa_index(a) for a in self.df["wt_aa"]], dtype=np.int64)
        self._mt_idx = np.array([aa_index(a) for a in self.df["mt_aa"]], dtype=np.int64)
        self._y = self.df[self.target].to_numpy(dtype=np.float32)
        if "sigma" in self.df.columns:
            self._sigma = pd.to_numeric(self.df["sigma"], errors="coerce").to_numpy(dtype=np.float32)
        else:
            self._sigma = np.full(len(self.df), np.nan, dtype=np.float32)

        self._chunks: list[tuple[str, np.ndarray]] = []
        self._rebuild_chunks(epoch=0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rebuild_chunks(self.epoch)

    def _rebuild_chunks(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed * 1_000_003 + epoch * 7_919)
        chunks: list[tuple[str, np.ndarray]] = []
        m = self.muts_per_protein
        for seq, idxs in self._groups:
            arr = rng.permutation(idxs) if self.train else idxs
            for start in range(0, len(arr), m):
                chunks.append((seq, arr[start:start + m]))
        self._chunks = chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, i: int) -> dict:
        seq, idxs = self._chunks[i]
        rec = self.store.get(seq)
        drop_cds = self.train and self.cds_dropout > 0.0 and float(torch.rand(())) < self.cds_dropout

        base = self._graph_cache.get_whole(seq, rec)
        if base is None:
            # Fallback if a protein exceeds window: center on the first mutation
            base = build_sample_graph(rec, int(self._site[idxs[0]]), self.window, self.cfg, drop_cds=drop_cds)
            start_off = max(0, min(int(self._site[idxs[0]]), base["length"] - 1) - base["site"])
        else:
            base = dict(base)
            start_off = 0
            if drop_cds:
                base["mech"], base["edge_mech"], base["attr_mech"] = _apply_cds_dropout(
                    base["mech"], base["n"]
                )

        if self.train and self.noise_std > 0.0:
            base["esm"] = base["esm"] + self.noise_std * torch.randn_like(base["esm"])

        n = base["n"]
        local_sites = np.clip(self._site[idxs] - start_off, 0, n - 1)
        m1_list, m2_list = [], []
        for s in local_sites:
            m1, m2 = self._graph_cache.get_khops(seq, base["edge_geo"], int(s), n)
            m1_list.append(m1)
            m2_list.append(m2)

        item = {
            "esm": base["esm"],
            "mech": base["mech"],
            "topo": base["topo"],
            "edge_geo": base["edge_geo"],
            "attr_geo": base["attr_geo"],
            "edge_mech": base["edge_mech"],
            "attr_mech": base["attr_mech"],
            "n": n,
            "length": base["length"],
            "truncated": base["truncated"],
            "sites": torch.from_numpy(local_sites.astype(np.int64)),
            "neigh_1hop": torch.stack(m1_list, dim=0),      # (K, n)
            "neigh_2hop": torch.stack(m2_list, dim=0),      # (K, n)
            "wt_idx": torch.from_numpy(self._wt_idx[idxs]),
            "mt_idx": torch.from_numpy(self._mt_idx[idxs]),
            "y": torch.from_numpy(self._y[idxs]),
            "sigma": torch.from_numpy(self._sigma[idxs]),
        }
        if "layer_embs" in base:
            item["layer_embs"] = base["layer_embs"]
        if "wt_logprobs" in base:
            item["wt_logprobs_site"] = base["wt_logprobs"][item["sites"]]
        if "edge_str" in base:
            item["edge_str"] = base["edge_str"]
            item["attr_str"] = base["attr_str"]
            item["str_node"] = base["str_node"]
        return item


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
# Batching by node budget
# ---------------------------------------------------------------------------
class NodeBudgetBatchSampler(torch.utils.data.Sampler):
    """
    Batches capped by **padded node count**, not by sample count.

    Round-2 review, R2-02: the R1-07 fix grew a whole-sequence graph from a
    256-node window to the entire protein (up to ``max_sequence_nodes``, shipped
    at 2048) while every task went on sharing one ``batch_size: 64``. Activation
    memory here is linear in nodes - edges are ``graph_top_k`` per node plus
    backbone plus self-loops, the GATv2 message tensors are per-edge, and the
    SSM scan buffers are per-node - so a whole-sequence batch cost 8x the ddG
    batch the budget script models, putting the analytic lower bound above the
    documented 16 GB card. Three of the six heads could not be trained at the
    shipped configuration, and nothing said so until the OOM.

    Capping ``len(batch) * max_nodes_in_batch`` instead of ``len(batch)`` makes
    the memory a batch can demand a declared constant rather than a consequence
    of how long the proteins happen to be. A 2048-node protein yields 8 samples
    per batch at the shipped budget; a 200-residue one yields ``max_batch_size``.

    Padding is the other half: :func:`collate_graphs` pads to the longest sample,
    so batching long and short proteins together wastes the budget on zeros.
    Indices are shuffled, chunked into megabatches, and sorted by node count
    within each, which keeps a batch's members close in length without making
    the epoch order deterministic.

    Args:
        node_counts    : nodes per sample, in dataset index order.
        node_budget    : maximum ``len(batch) * max_nodes_in_batch``.
        max_batch_size : cap on samples per batch regardless of size.
        shuffle        : shuffle between epochs (train) or keep dataset order.
        seed           : base seed; combined with the epoch set by
                         :meth:`set_epoch` so the order is a pure function of
                         ``(seed, epoch)`` and a resume is exact (R2-12).
    """

    def __init__(self, node_counts, node_budget: int, max_batch_size: int,
                 shuffle: bool = True, seed: int = 42, megabatch_factor: int = 8):
        self.node_counts = [max(int(n), 1) for n in node_counts]
        self.node_budget = max(int(node_budget), 1)
        self.max_batch_size = max(int(max_batch_size), 1)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        # A small whole-sequence table is exhausted several times inside one
        # epoch (scripts/train.endless re-creates the iterator), and its
        # docstring promises a reshuffle on each pass - `itertools.cycle`
        # replaying identical batches is what R1-03 removed. So the order is a
        # function of (seed, epoch, pass), not of (seed, epoch) alone.
        self.pass_index = 0
        self.megabatch_factor = max(int(megabatch_factor), 1)
        self._cached: tuple[int, int, list[list[int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.pass_index = 0
        self._cached = None

    def _build(self, pass_index: int) -> list[list[int]]:
        n = len(self.node_counts)
        order = list(range(n))
        if self.shuffle:
            g = np.random.default_rng(
                self.seed * 1_000_003 + self.epoch * 7_919 + pass_index)
            order = [int(i) for i in g.permutation(n)]

        span = self.max_batch_size * self.megabatch_factor
        batches: list[list[int]] = []
        for chunk_start in range(0, n, span):
            chunk = order[chunk_start:chunk_start + span]
            # Longest first: the first member sets the padded width, so the
            # budget is spent on real residues rather than on padding.
            chunk.sort(key=lambda i: -self.node_counts[i])
            batch: list[int] = []
            widest = 0
            for i in chunk:
                trial_width = max(widest, self.node_counts[i])
                if batch and (
                    len(batch) + 1 > self.max_batch_size
                    or (len(batch) + 1) * trial_width > self.node_budget
                ):
                    batches.append(batch)
                    batch, widest = [i], self.node_counts[i]
                else:
                    batch.append(i)
                    widest = trial_width
            if batch:
                batches.append(batch)
        if self.shuffle:
            g = np.random.default_rng(
                self.seed * 104_729 + self.epoch * 7_919 + pass_index)
            batches = [batches[int(i)] for i in g.permutation(len(batches))]
        return batches

    def _batches(self, pass_index: int) -> list[list[int]]:
        key = (self.epoch, pass_index)
        if self._cached is None or self._cached[:2] != key:
            self._cached = (self.epoch, pass_index, self._build(pass_index))
        return self._cached[2]

    def __iter__(self):
        batches = self._batches(self.pass_index)
        # Advance only for a shuffling loader: an evaluation loader must yield
        # the same order every time it is read.
        if self.shuffle:
            self.pass_index += 1
        yield from batches

    def __len__(self) -> int:
        # Does NOT advance the pass counter: len() is asked for at setup and
        # between epochs, and consuming a pass there would desynchronise the
        # order from (seed, epoch, pass).
        return len(self._batches(self.pass_index))


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
    topo = torch.zeros(b, L, N_TOPO_CHANNELS)
    mask = torch.zeros(b, L, dtype=torch.bool)
    neighborhood = torch.zeros(b, L, dtype=torch.bool)
    neighborhood_1hop = torch.zeros(b, L, dtype=torch.bool)

    has_layers = all("layer_embs" in s for s in samples)
    layer_embs = (
        torch.zeros(b, samples[0]["layer_embs"].shape[0], L, d)
        if has_layers else None
    )
    has_lp = all("wt_logprobs" in s for s in samples)
    wt_logprobs = (
        torch.zeros(b, L, samples[0]["wt_logprobs"].shape[-1])
        if has_lp else None
    )
    has_str = all("edge_str" in s for s in samples)
    str_node = torch.zeros(b, L, STR_NODE_DIM) if has_str else None
    str_edge_dim = samples[0]["attr_str"].shape[-1] if has_str else STR_EDGE_DIM

    eg, ag, em, am, es, as_ = [], [], [], [], [], []
    for i, s in enumerate(samples):
        n = s["n"]
        esm[i, :n] = s["esm"]
        mech[i, :n] = s["mech"]
        if "topo" in s:
            topo[i, :n] = s["topo"]
        mask[i, :n] = True
        neighborhood[i, :n] = s["neighborhood"]
        neighborhood_1hop[i, :n] = s.get("neighborhood_1hop", s["neighborhood"])
        if layer_embs is not None:
            layer_embs[i, :, :n] = s["layer_embs"]
        if wt_logprobs is not None:
            wt_logprobs[i, :n] = s["wt_logprobs"]
        off = i * L
        if s["edge_geo"].numel():
            eg.append(s["edge_geo"] + off)
            ag.append(s["attr_geo"])
        if s["edge_mech"].numel():
            em.append(s["edge_mech"] + off)
            am.append(s["attr_mech"])
        if has_str:
            str_node[i, :n] = s["str_node"]
            if s["edge_str"].numel():
                es.append(s["edge_str"] + off)
                as_.append(s["attr_str"])

    def _cat(edges, attrs, edim=EDGE_DIM):
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, edim))
        return torch.cat(edges, dim=1), torch.cat(attrs, dim=0)

    edge_geo, attr_geo = _cat(eg, ag, EDGE_DIM)
    edge_mech, attr_mech = _cat(em, am, EDGE_DIM)

    batch = {
        "esm": esm,
        "mech": mech,
        "topo": topo,
        "mask": mask,
        "edge_geo": edge_geo,
        "attr_geo": attr_geo,
        "edge_mech": edge_mech,
        "attr_mech": attr_mech,
        "site": torch.tensor([s["site"] for s in samples], dtype=torch.long),
        "truncated": torch.tensor([bool(s.get("truncated", False))
                                   for s in samples], dtype=torch.bool),
        "neighborhood_mask": neighborhood,
        "neighborhood_1hop": neighborhood_1hop,
        "wt_idx": torch.tensor([s["wt_idx"] for s in samples], dtype=torch.long),
        "mt_idx": torch.tensor([s["mt_idx"] for s in samples], dtype=torch.long),
        "y": torch.tensor([s["y"] for s in samples], dtype=torch.float32),
        "sigma": torch.tensor([s["sigma"] for s in samples], dtype=torch.float32),
        "seq_length": torch.tensor(
            [float(s.get("length", s["n"])) for s in samples], dtype=torch.float32
        ),
    }
    if layer_embs is not None:
        batch["layer_embs"] = layer_embs
    if wt_logprobs is not None:
        batch["wt_logprobs"] = wt_logprobs
    if has_str:
        edge_str, attr_str = _cat(es, as_, str_edge_dim)
        batch["edge_str"] = edge_str
        batch["attr_str"] = attr_str
        batch["str_node"] = str_node
    if "species_id" in samples[0]:
        batch["species_id"] = torch.tensor(
            [s["species_id"] for s in samples], dtype=torch.long
        )
    return batch


def collate_protein_batch(items: list[dict]) -> dict:
    """
    Collate ``P`` parent protein graphs from :class:`ProteinBatchMutationDataset`
    alongside all ``M`` flattened mutations pointing into ``prot_idx in [0, P)``.
    """
    p = len(items)
    L = max(it["n"] for it in items)
    d = items[0]["esm"].shape[-1]

    esm = torch.zeros(p, L, d)
    mech = torch.zeros(p, L, N_MECH_CHANNELS)
    topo = torch.zeros(p, L, N_TOPO_CHANNELS)
    mask = torch.zeros(p, L, dtype=torch.bool)

    has_layers = all("layer_embs" in it for it in items)
    layer_embs = (
        torch.zeros(p, items[0]["layer_embs"].shape[0], L, d)
        if has_layers else None
    )
    has_lp_site = all("wt_logprobs_site" in it for it in items)
    has_str = all("edge_str" in it for it in items)
    str_node = torch.zeros(p, L, STR_NODE_DIM) if has_str else None
    str_edge_dim = items[0]["attr_str"].shape[-1] if has_str else STR_EDGE_DIM

    eg, ag, em, am, es, as_ = [], [], [], [], [], []
    prot_idxs, sites, wt_l, mt_l, y_l, sig_l, lp_l = [], [], [], [], [], [], []
    n1_rows, n2_rows = [], []

    for i, it in enumerate(items):
        n = it["n"]
        k = int(it["sites"].numel())
        esm[i, :n] = it["esm"]
        mech[i, :n] = it["mech"]
        topo[i, :n] = it["topo"]
        mask[i, :n] = True
        if layer_embs is not None:
            layer_embs[i, :, :n] = it["layer_embs"]
        off = i * L
        if it["edge_geo"].numel():
            eg.append(it["edge_geo"] + off)
            ag.append(it["attr_geo"])
        if it["edge_mech"].numel():
            em.append(it["edge_mech"] + off)
            am.append(it["attr_mech"])
        if has_str:
            str_node[i, :n] = it["str_node"]
            if it["edge_str"].numel():
                es.append(it["edge_str"] + off)
                as_.append(it["attr_str"])

        prot_idxs.append(torch.full((k,), i, dtype=torch.long))
        sites.append(it["sites"])
        wt_l.append(it["wt_idx"])
        mt_l.append(it["mt_idx"])
        y_l.append(it["y"])
        sig_l.append(it["sigma"])
        if has_lp_site:
            lp_l.append(it["wt_logprobs_site"])
        m1_pad = torch.zeros((k, L), dtype=torch.bool)
        m2_pad = torch.zeros((k, L), dtype=torch.bool)
        m1_pad[:, :n] = it["neigh_1hop"]
        m2_pad[:, :n] = it["neigh_2hop"]
        n1_rows.append(m1_pad)
        n2_rows.append(m2_pad)

    def _cat(edges, attrs, edim=EDGE_DIM):
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, edim))
        return torch.cat(edges, dim=1), torch.cat(attrs, dim=0)

    edge_geo, attr_geo = _cat(eg, ag, EDGE_DIM)
    edge_mech, attr_mech = _cat(em, am, EDGE_DIM)

    batch = {
        "esm": esm,
        "mech": mech,
        "topo": topo,
        "mask": mask,
        "edge_geo": edge_geo,
        "attr_geo": attr_geo,
        "edge_mech": edge_mech,
        "attr_mech": attr_mech,
        "prot_idx": torch.cat(prot_idxs, dim=0),
        "site": torch.cat(sites, dim=0),
        "neighborhood_1hop": torch.cat(n1_rows, dim=0),
        "neighborhood_mask": torch.cat(n2_rows, dim=0),
        "wt_idx": torch.cat(wt_l, dim=0),
        "mt_idx": torch.cat(mt_l, dim=0),
        "y": torch.cat(y_l, dim=0),
        "sigma": torch.cat(sig_l, dim=0),
        "truncated": torch.tensor([bool(it.get("truncated", False)) for it in items], dtype=torch.bool),
        "seq_length": torch.tensor([float(it["length"]) for it in items], dtype=torch.float32),
    }
    if layer_embs is not None:
        batch["layer_embs"] = layer_embs
    if has_lp_site:
        batch["wt_logprobs_site"] = torch.cat(lp_l, dim=0)
    if has_str:
        edge_str, attr_str = _cat(es, as_, str_edge_dim)
        batch["edge_str"] = edge_str
        batch["attr_str"] = attr_str
        batch["str_node"] = str_node
    return batch


def move_batch(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


__all__ = [
    "LoadReport",
    "NodeBudgetBatchSampler",
    "clean_cds",
    "detect_columns",
    "load_mutation_table",
    "load_sequence_table",
    "filter_to_cache",
    "build_sample_graph",
    "MutationDataset",
    "ProteinBatchMutationDataset",
    "SequenceDataset",
    "collate_graphs",
    "collate_protein_batch",
    "move_batch",
]
