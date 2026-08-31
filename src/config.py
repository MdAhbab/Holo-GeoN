"""
src/config.py
=============
The single place configuration is loaded from.

Holo-GeoN has two config files and, before v2, only one of them was ever read:
``configs/data.yaml`` was documented in the guide ("set the real column name
here if auto-detection fails") but no script imported it, so editing it did
nothing. Every loader now goes through :func:`load_config`, so both files are
live.

Precedence, lowest to highest:

    DEFAULTS  <  configs/run.yaml  <  configs/data.yaml  <  CLI overrides

``data.yaml`` carries per-dataset column overrides and the graph/feature
hyperparameters; ``run.yaml`` carries paths, sizes and schedules. They are
merged into one flat :class:`Config` so callers never have to know which file a
key came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RUN_YAML = os.path.join("configs", "run.yaml")
DEFAULT_DATA_YAML = os.path.join("configs", "data.yaml")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    # --- encoder ---
    "esm_model": "facebook/esm2_t33_650M_UR50D",
    "cache": "cache/esm2_650M",
    "device": "cuda",
    "window": 256,          # mutation-centered window length (residues)
    "contact_top_k": 24,    # contacts cached per residue
    "contact_min_sep": 3,   # |i-j| below this is backbone context, not a contact

    # --- data ---
    "out_root": "runs",
    "stability_data": "data/megascale_ddg_pairs.parquet",
    "clinvar_data": "data/clinvar_protein_pairs.parquet",
    "idr_data": None,          # experimental Rg table; prior-only when absent
    "mfi_data": None,          # fluorescence table; head untrained when absent
    "proteomics_data": None,   # retention-time table; head untrained when absent
    "max_protein_length": 5000,

    # --- splits ---
    "split_method": "mmseqs",
    "identity": 0.30,
    "coverage": 0.80,
    "cv_folds": 5,
    "seed": 42,
    # Deterministic algorithms + cuDNN determinism. Slower, but makes a rerun
    # bit-reproducible; worth it for the ablation arms.
    "strict_determinism": False,

    # --- graph ---
    "graph_top_k": 16,       # neighbours per node in the built graph
    "geo_threshold": 0.30,   # contact-probability floor for E_geo
    "eps": 0.10,             # |dCAI|  threshold for E_mech
    "delta": 0.10,           # |dmRNA| threshold for E_mech
    "mech_max_degree": 8,
    "neighborhood_hops": 2,  # k-hop site neighbourhood for the readout

    # --- model ---
    "hidden": 256,
    "out_dim": 256,
    "gat_heads": 4,
    "aa_dim": 32,
    "dropout": 0.1,
    "use_ssm": True,
    "use_mech_stream": True,   # ablation axis; the shipped model is dual-stream
    "use_geo_stream": True,    # ablation axis
    "site_anchored_readout": True,  # ablation axis; False is the v1 formulation

    # --- training ---
    "epochs": 60,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_frac": 0.05,
    "early_stop_patience": 8,
    "grad_clip": 1.0,
    "amp": True,
    "num_workers": 4,
    "max_rows": None,
    # Task sampling exponent: 1 is strictly proportional to loader size, 0.5 a
    # square-root schedule, 0 the old equal-steps-per-task behaviour.
    "task_sampling_temperature": 0.5,
    # Cap on nodes for a whole-sequence graph (IDR / MFI / proteomics).
    "max_sequence_nodes": 2048,
    # Whole-sequence batches are capped by padded NODE count, not sample count
    # (R2-02). None means "the same node budget the windowed loaders spend",
    # i.e. batch_size * window - so no task can cost more per step than the ddG
    # task the memory budget is computed for.
    "sequence_node_budget": None,
    # The ONE metric per task the ablation grid compares. Pre-registered here so
    # the comparison family is fixed before any number is seen; comparing every
    # numeric key instead gave 80 uncorrected comparisons and a 0.999 chance of
    # a spurious "component matters" flag under the global null (R2-06).
    # Three-state neutral band for stability rows with no reported measurement
    # error, in kcal/mol. A label is "neutral" when |ddG| <= n_sigma * sigma;
    # with no `sigma` column this constant decides the class of ~21% of a
    # realistic ddG distribution, so it is declared here and recorded in
    # metrics.json with a sensitivity table (R2-11).
    "three_state_sigma": 0.5,
    # In-memory cached encoder records per DataLoader worker. 0 disables it.
    # A mutation dataset reads one record per ROW from a cache keyed by PARENT,
    # so the same file is deserialised many times per epoch; this trades roughly
    # `record_cache_size * L * dim * 2` bytes per worker for those reads. Off by
    # default because the RAM cost should be chosen, not inherited (R1-17).
    "record_cache_size": 0,
    "ablation_metrics": {
        "stability": "pearson",
        "pathogenicity": "auroc",
        "idr": "pearson",
        "mfi": "pearson",
        "proteomics": "pearson",
    },
}


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Flat, attribute-and-key accessible configuration."""

    values: dict[str, Any] = field(default_factory=dict)
    columns: dict[str, dict[str, str]] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__["values"][name]
        except KeyError as exc:
            raise AttributeError(
                f"unknown config key {name!r}; known keys: "
                f"{sorted(self.__dict__['values'])}"
            ) from exc

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def column_overrides(self, dataset: str) -> dict[str, str]:
        """Column-name overrides for one dataset key from ``configs/data.yaml``."""
        return dict(self.columns.get(dataset, {}) or {})

    def resolve(self, path: str | None) -> str | None:
        """Absolute path for a config value, relative to the repository root."""
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(ROOT, path)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return {}
    with open(full, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    run_yaml: str | None = DEFAULT_RUN_YAML,
    data_yaml: str | None = DEFAULT_DATA_YAML,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """
    Build the effective configuration.

    ``configs/data.yaml`` contributes two things: a ``graph:`` block that is
    merged into the flat namespace, and per-dataset column-name maps that are
    kept aside under :meth:`Config.column_overrides`. Anything else at its top
    level is merged flat too, so a value can live in either file.

    ``overrides`` (CLI) wins over both. ``None`` values in ``overrides`` are
    ignored, so an unset argparse flag never clobbers a file setting.
    """
    values = dict(DEFAULTS)
    sources: list[str] = []

    run_cfg = _read_yaml(run_yaml)
    if run_cfg:
        sources.append(run_yaml or "")
        values.update(run_cfg)

    columns: dict[str, dict[str, str]] = {}
    data_cfg = _read_yaml(data_yaml)
    if data_cfg:
        sources.append(data_yaml or "")
        for key, block in data_cfg.items():
            if key == "graph" and isinstance(block, dict):
                values.update(block)
            elif isinstance(block, dict) or block is None:
                # A per-dataset column map, e.g. megascale_ddg_pairs: {wt_seq: ...}.
                # A block whose entries are all commented out parses as None; that
                # is an empty override map, not a config value called
                # "megascale_ddg_pairs".
                columns[key] = {k: v for k, v in (block or {}).items() if v is not None}
            else:
                values[key] = block

    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    unknown = set(values) - set(DEFAULTS)
    if unknown:
        # Not fatal: a config may legitimately carry extra keys for a custom
        # script. Surfacing them catches typos that would otherwise be silent.
        print(f"[config] note: keys not in DEFAULTS: {sorted(unknown)}")

    return Config(values=values, columns=columns, sources=sources)


__all__ = ["Config", "load_config", "DEFAULTS", "ROOT"]
