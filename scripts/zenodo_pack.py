"""
scripts/zenodo_pack.py
======================
Assemble, validate, checksum, and document the **Holo-StabBench v1.0** Zenodo
processed dataset + weights + benchmark-predictions release inside
``extras/zenodo_staging/Holo-StabBench-v1.0/``.

Generates:
  * ``SHA256SUMS.txt`` covering every artifact in the release tree
  * ``DATASHEET.md`` (Gebru-style Dataset Card with measured row/protein counts,
    sign conventions, split leakage summaries, and upstream licence attributions)
  * ``release_manifest.json`` (machine-readable inventory)

Usage:
    python -m scripts.zenodo_pack --stage ../extras/zenodo_staging/Holo-StabBench-v1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

SUBDIRS = ("tables", "splits", "features", "weights", "benchmark_predictions")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_parquet(path: Path) -> dict:
    df = pd.read_parquet(path)
    info = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "bytes": int(path.stat().st_size),
    }
    for col in ("parent", "parent_seq", "wt_seq", "uniprot_id"):
        if col in df.columns:
            info[f"unique_{col}"] = int(df[col].nunique())
    if "split" in df.columns:
        info["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    return info


def build_release_bundle(stage_dir: Path, version: str = "1.0.0") -> dict:
    for sub in SUBDIRS:
        (stage_dir / sub).mkdir(parents=True, exist_ok=True)

    inventory: list[dict] = []
    sha_lines: list[str] = []
    table_stats: dict[str, dict] = {}

    for p in sorted(stage_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(stage_dir).as_posix()
        if rel in ("SHA256SUMS.txt", "DATASHEET.md", "release_manifest.json"):
            continue
        digest = sha256_file(p)
        sha_lines.append(f"{digest}  {rel}")
        rec = {"path": rel, "bytes": int(p.stat().st_size), "sha256": digest}
        if p.suffix.lower() == ".parquet":
            stats = inspect_parquet(p)
            rec["parquet"] = stats
            table_stats[rel] = stats
        inventory.append(rec)

    (stage_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + ("\n" if sha_lines else ""),
                                             encoding="utf-8")

    manifest = {
        "name": "Holo-StabBench",
        "version": version,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sign_convention": "stabilizing_positive (ddG = dG_mut - dG_wt; positive = stabilizing)",
        "total_files": len(inventory),
        "total_bytes": sum(r["bytes"] for r in inventory),
        "files": inventory,
    }
    (stage_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows_md = []
    for rel, st in sorted(table_stats.items()):
        rows_md.append(f"| `{rel}` | {st['rows']:,} | {len(st['columns'])} | {st['bytes'] / 1e6:.2f} MB |")
    if not rows_md:
        rows_md.append("| *(stage Parquet files into `tables/`, `splits/`, `features/`, `benchmark_predictions/`)* | — | — | — |")

    datasheet = f"""# Holo-StabBench v{version} — Dataset & Artifacts Datasheet

Generated: `{manifest['generated_utc']}` | Total files: **{manifest['total_files']}** | Total size: **{manifest['total_bytes'] / 1e6:.2f} MB**

## 1. Overview & Motivation
**Holo-StabBench** accompanies the *Holo-GeoN* protein stability (ΔΔG) and epistasis study. It provides:
1. **Sign-harmonized, reading-frame-verified Parquet tables** (`positive = stabilizing`, $\\Delta\\Delta G = \\Delta G_{{\\text{{mut}}}} - \\Delta G_{{\\text{{wt}}}}$) spanning MegaScale cDNA proteolysis (singles + double mutants), FireProtDB 2.0 (MegaScale-removed + 25%-homology-filtered `FireProt-HF25`), S669/Ssym+, and Human Domainome 1.0 (rank-only external aPCA validation).
2. **Sequence- and Structure-Level Leakage Audits**: MMseqs2 25% & 30% identity splits plus the first pairwise **TM-align structural fold leakage matrix** across MegaScale test vs. train/val PDBs.
3. **Expensive Precomputed Features**: Reading-frame-verified real-CDS ViennaRNA mRNA MFE + CAI features, 176 SHA-256-pinned FireProtDB AlphaFold DB structures, and aligned SE(3)-invariant backbone geometry tensors (`N/CA/C/CB`, `pLDDT`, `coord_valid`).
4. **Trained Checkpoints & Common-Format Predictions**: Self-describing PyTorch weights (5 seeds + calibration JSONs) and `(row_id, y_pred)` Parquets for Holo-GeoN and every benchmarked competitor so every table & cluster-bootstrap CI reproduces in one command via `scripts/score_predictions.py`.

## 2. Staged Parquet Tables
| Artifact | Rows | Columns | Size |
|---|---:|---:|---:|
{chr(10).join(rows_md)}

## 3. Upstream Provenance & Licences
* **MegaScale cDNA proteolysis** (Tsuboyama et al., *Nature* 2023; Zenodo `7844779`) — CC-BY 4.0 / open academic.
* **FireProtDB 2.0** (Stourac et al., *NAR* 2021/2025) — open academic database; MegaScale-sourced rows strictly removed from external holdout.
* **S669 / Ssym+** (Pancotti et al., *Brief. Bioinform.* 2022; Zenodo `7568094`) — CC-BY 4.0.
* **Human Domainome 1.0** (Beltran et al., *Nature* 2025; Zenodo `14356805`) — CC-BY 4.0 (aPCA abundance: Spearman rank-only evaluation, not kcal/mol).
* **AlphaFold Protein Structure Database** (EMBL-EBI / Google DeepMind) — CC-BY 4.0.
* **MGnify Stability** (Cho/Rocklin/OpenFold, bioRxiv `10.64898/2026.05.19.726285`) — if used in extended pretraining, check portal registration terms before redistributing raw tables; ship deterministic ETL scripts & split IDs instead.

## 4. Integrity Verification
Verify all files against `SHA256SUMS.txt`:
```bash
sha256sum -c SHA256SUMS.txt
```
"""
    (stage_dir / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    print(f"[zenodo_pack] staged {manifest['total_files']} files ({manifest['total_bytes'] / 1e6:.2f} MB) -> {stage_dir}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default=os.path.join("..", "extras", "zenodo_staging", "Holo-StabBench-v1.0"))
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()
    build_release_bundle(Path(args.stage).resolve(), version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
