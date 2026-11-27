"""
scripts/eval_external.py
========================
Score a trained checkpoint on an INDEPENDENT stability table it never saw.

Why this stage exists
---------------------
The shipped training set is 190 identity clusters of 32-72-residue designed and
natural domains, all measured by one proteolysis assay in one publication. Every
in-run number - the test split, the baselines, the ablation grid - is computed
inside that world. A held-out split of it answers "does the model generalise to
held-out domains of this kind"; it cannot answer "does any of this transfer to
real, full-length proteins measured by classical biophysics", because no such
protein is anywhere in the run.

``data/fireprotdb_holdout.parquet`` is that second question: FireProtDB with the
604,920 MegaScale-sourced rows removed, leaving ~4.5k mutations over 182
proteins from 67 organisms at a median length of 228. It shares no measurement
with the training table by construction.

Three tables are scored with this script, and each gets its own result file,
named after the data file's stem (the same ``key`` used for column overrides in
``configs/data.yaml`` and for ``pred_{key}.parquet``):

* ``fireprotdb_holdout.parquet`` - FireProtDB with all MegaScale-sourced rows
  removed (no homology filter).
* ``fireprotdb_hf25.parquet``    - the subset of that table whose proteins are
  under 25% identity to MegaScale (MMseqs2).
* ``s669_holdout.parquet``       - S669, a separate 669-mutation literature
  holdout whose wild-type residues are verified against the WT PDB chains.

They used to share one output name, ``external_fireprotdb.json``, so evaluating
several into one directory silently kept only the last. The result is now
``external_{key}.json``. The old name is still written, for ``fireprotdb*``
tables only, because ``run.py``, ``scripts/final_scores.py``,
``scripts/remote_job.py`` and earlier run artifacts read it; the files differ
only in name, so with several ``fireprotdb*`` tables in one directory the
legacy file is whichever of them ran last. Read ``external_{key}.json``.

Two things this reports that a plain metric would hide
------------------------------------------------------
* **Rows dropped for want of a cached encoding.** An external table's proteins
  are not in the run's cache unless they were deliberately encoded, and quietly
  scoring the subset that happens to be cached would report a metric for a set
  nobody chose. The count is in the output.
* **The sign convention actually applied.** This table is
  destabilizing-positive while the training table is stabilizing-positive
  (see ``scripts/build_fireprot_table.py``); the flip comes from
  ``configs/data.yaml`` via the loader, and the applied sign is echoed into the
  report so the number cannot be read without it.

Usage
-----
python -m scripts.eval_external --checkpoint runs/main/model_best.pt
python -m scripts.eval_external --checkpoint runs/main/model_best.pt \\
    --data data/s669_holdout.parquet          # -> runs/main/external_s669_holdout.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import apply_checkpoint_graph_config, load_config
from src.data import (
    MutationDataset,
    collate_graphs,
    filter_to_cache,
    load_mutation_table,
    move_batch,
)
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore
from src.fast_eval import predict_table
from src.metrics import format_report, regression_metrics
from src.model import HoloGeoN
from src.provenance import provenance
from src.seeding import seed_everything

try:                                    # same fallback calibration.py uses
    from scripts.train import model_kwargs_from_config
except Exception:                       # pragma: no cover
    model_kwargs_from_config = None


# The result file's name before results were keyed by table. Still written for
# ``fireprotdb*`` tables (see the module docstring); readers should prefer
# ``external_{key}.json``.
LEGACY_RESULT_JSON = "external_fireprotdb.json"

# What each known table measures, keyed by the data file's stem. ``title`` heads
# the printed report; ``what`` goes into the JSON as ``what_this_measures``.
TABLE_INFO: dict[str, dict[str, str]] = {
    "fireprotdb_holdout": {
        "title": "FireProtDB (no MegaScale rows)",
        "what": (
            "transfer to proteins the run never saw: FireProtDB with all "
            "MegaScale-sourced rows removed, so no measurement is shared with "
            "the training table. No homology filter is applied, so a protein "
            "can still be similar to a MegaScale training domain"
        ),
    },
    "fireprotdb_hf25": {
        "title": "FireProtDB homology-filtered (<25% identity to MegaScale)",
        "what": (
            "transfer to proteins dissimilar to the training table: the "
            "homology-filtered subset of the MegaScale-free FireProtDB holdout, "
            "keeping only proteins under 25% sequence identity to MegaScale "
            "(MMseqs2 search). Shares no measurement with the training table, "
            "and no kept protein reached 25% identity to a MegaScale sequence"
        ),
    },
    "s669_holdout": {
        "title": "S669 (669-mutation literature holdout)",
        "what": (
            "transfer to a separate literature holdout: S669 (Pancotti et al. "
            "2022), 669 mutations whose wild-type residues were verified "
            "against the WT PDB chains. It is a different table from both "
            "FireProtDB and the MegaScale training table, not a FireProtDB "
            "subset"
        ),
    },
    "s1724_thermomutdb_holdout": {
        "title": "S1724 ThermoMutDB curated holdout (Rocklin 2026)",
        "what": (
            "transfer to the curated ThermoMutDB classical stability holdout "
            "(1,433 verified single mutants across 35 PDBs / 33 unique WT "
            "proteins, stabilizing-positive). No homology filter applied"
        ),
    },
    "s1724_thermomutdb_hf25": {
        "title": "S1724 ThermoMutDB homology-filtered (<25% identity to MegaScale)",
        "what": (
            "transfer to the homology-filtered (<25% MMseqs2 identity to "
            "MegaScale) subset of S1724 ThermoMutDB (565 verified single "
            "mutants across 22 parents / 21 unique WT proteins), excluding "
            "the 12/33 proteins sharing >=25% homology with MegaScale"
        ),
    },
    "domainome_apca_hf25": {
        "title": "Human Domainome 1.0 aPCA (<25% identity to MegaScale)",
        "what": (
            "zero-shot rank transfer to Human Domainome 1.0 aPCA abundance "
            "fitness (Beltran et al. 2025) filtered to <25% MMseqs2 identity "
            "to MegaScale (evaluate strictly by per-domain Spearman rank)"
        ),
    },
}


def describe_table(key: str) -> dict[str, str]:
    """Title and ``what_this_measures`` text for the table with file stem ``key``."""
    info = TABLE_INFO.get(key)
    if info is not None:
        return dict(info)
    # An unregistered table must not inherit another table's description.
    return {
        "title": key,
        "what": (
            f"transfer to the external table {key!r}, which the run never "
            f"saw. No description is registered for this table in "
            f"scripts/eval_external.py (TABLE_INFO), so check how it was built "
            f"before reading the number"
        ),
    }


def result_paths(out_dir: str, key: str) -> list[str]:
    """JSON files to write for table ``key``: the keyed one, then the legacy one.

    The legacy ``external_fireprotdb.json`` is added when, and only when, the key
    starts with ``fireprotdb``; other tables never write it.
    """
    paths = [os.path.join(out_dir, f"external_{key}.json")]
    legacy = os.path.join(out_dir, LEGACY_RESULT_JSON)
    if key.startswith("fireprotdb") and legacy not in paths:
        paths.append(legacy)
    return paths


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus, sigmas, ys = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        ddg, logvar = model.ddg(batch)
        mus.append(ddg.reshape(-1).float().cpu().numpy())
        sigmas.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
        ys.append(batch["y"].reshape(-1).float().cpu().numpy())
    if not mus:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(mus), np.concatenate(sigmas), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.path.join("runs", "main", "model_best.pt"))
    ap.add_argument("--data", default=os.path.join("data", "fireprotdb_holdout.parquet"))
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join("runs", "main"))
    ap.add_argument("--rowwise", action="store_true",
                    help="score row-by-row with MutationDataset (legacy, slow) instead of "
                         "the per-protein batched fast path; same numbers either way")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={"cache": args.cache, "device": args.device})
    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"[external] no checkpoint at {args.checkpoint}. Train first: "
            f"python run.py train"
        )
    os.makedirs(args.out, exist_ok=True)
    seed_everything(cfg.seed)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    key = os.path.splitext(os.path.basename(args.data))[0]
    overrides = cfg.column_overrides(key)
    norm, report = load_mutation_table(
        args.data, kind="stability", overrides=overrides,
        max_protein_length=cfg.max_protein_length,
    )
    if "source_row_id" in norm.columns:
        norm["row_id"] = norm["source_row_id"].to_numpy(dtype=np.int64)
    else:
        norm["row_id"] = np.arange(len(norm), dtype=np.int64)
    print(report.summary())
    sign_applied = report.notes.get("ddg_sign", "stabilizing_positive")
    print(f"[external] ddg_sign applied: {sign_applied}  "
          f"(mean ddG after normalisation: {norm['target'].mean():+.3f})")
    if sign_applied == "stabilizing_positive" and norm["target"].mean() > 0:
        print("[external] WARNING: mean ddG is positive under the internal "
              "convention. Random substitutions are destabilising on average, "
              "so this table may need ddg_sign: destabilizing_positive under "
              f"the key {key!r} in configs/data.yaml.")

    n_before = len(norm)
    norm = filter_to_cache(norm, store, report)
    n_uncached = n_before - len(norm)
    if n_uncached:
        print(f"[external] {n_uncached}/{n_before} rows have no cached encoding "
              f"and are NOT scored. Encode them first:\n"
              f"    python -m scripts.precompute_embeddings --data {args.data} "
              f"--cache {cfg.cache} --device {cfg.device}")
    if norm.empty:
        raise SystemExit(
            "[external] no rows have a cached encoding; nothing to score.")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    apply_checkpoint_graph_config(cfg, state)
    esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.esm_model, 1280))
    kwargs = state.get("model_kwargs")
    if kwargs is None and model_kwargs_from_config is not None:
        kwargs = model_kwargs_from_config(state.get("config", {}), esm_dim)
    model = HoloGeoN(**kwargs).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Per-protein batched scoring (graph built once per protein, all of its
    # mutations in batched forward passes; proteins longer than cfg.window use
    # the row-wise MutationDataset path). Rows come back in ``norm`` order and
    # match the legacy loader to float rounding - see tests/test_fast_eval.py.
    mu, sigma, y, info = predict_table(model, norm, store, cfg, device, target="target",
                                       rowwise=True if args.rowwise else None)
    print(f"[external] scored {info['rows']:,} rows: {info['fast_rows']:,} batched over "
          f"{info['fast_proteins']:,} proteins, {info['rowwise_rows']:,} row-wise "
          f"({info.get('seconds', 0)} s)", flush=True)
    metrics = regression_metrics(y, mu)

    # Also write common-format per-row predictions so scripts/score_predictions.py
    # can compute cluster-bootstrap CIs and paired competitor comparisons directly.
    if "row_id" in norm.columns:
        row_ids = norm["row_id"].to_numpy(dtype=np.int64)
    else:
        row_ids = np.arange(len(norm), dtype=np.int64)
    pred_path = os.path.join(args.out, f"pred_{key}.parquet")
    pd.DataFrame({
        "row_id": row_ids,
        "y_true": y.astype(np.float32),
        "y_pred": mu.astype(np.float32),
        "y_sigma": sigma.astype(np.float32),
    }).to_parquet(pred_path, index=False)

    table = describe_table(key)
    out = {
        "provenance": provenance(),
        "checkpoint": args.checkpoint,
        "data": args.data,
        "table_key": key,
        "predictions_parquet": pred_path,
        "what_this_measures": table["what"],
        "ddg_sign_applied": sign_applied,
        "rows_scored": int(len(y)),
        "rows_without_cached_encoding": int(n_uncached),
        "distinct_proteins": int(norm["parent_seq"].nunique()),
        "load_report": report.as_dict(),
        "metrics": metrics,
        "predicted_sigma": {
            "mean": float(np.mean(sigma)) if len(sigma) else float("nan"),
            "median": float(np.median(sigma)) if len(sigma) else float("nan"),
        },
    }
    paths = result_paths(args.out, key)
    for p in paths:
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    print("\n" + format_report(metrics, title=f"External test — {table['title']}"))
    print(f"\n[external] {len(y)} rows over {out['distinct_proteins']} proteins")
    print(f"[external] wrote {paths[0]}")
    for legacy in paths[1:]:
        print(f"[external] wrote {legacy}  (legacy name, same content; the last "
              f"fireprotdb* table written to {args.out} owns it)")


if __name__ == "__main__":
    main()
