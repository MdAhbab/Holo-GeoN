"""
scripts/zenodo_pack.py
======================
Assemble, validate, checksum, and document the **Holo-StabBench v1.0** Zenodo
processed dataset + weights + benchmark-predictions release inside
``extras/zenodo_staging/Holo-StabBench-v1.0/``.

Generates:
  * ``SHA256SUMS.txt`` covering every artifact in the release tree
  * ``DATASHEET.md`` (Gebru-style Dataset Card with measured row/protein counts,
    sign conventions, Included vs. Excluded provenance table, strict split
    leakage claims discipline, and upstream licence attributions)
  * ``release_manifest.json`` (machine-readable inventory)

Usage:
    python -m scripts.zenodo_pack --stage ../extras/zenodo_staging/Holo-StabBench-v1.0
    python -m scripts.zenodo_pack --stage ../extras/zenodo_staging/Holo-StabBench-v1.0 \\
        --local-processed-dir DATASETS/DATASETS/data
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SUBDIRS = ("tables", "splits", "features", "weights", "benchmark_predictions")

_META_INSPECT_COLS = ("parent", "parent_seq", "wt_seq", "uniprot_id", "split")


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file using 1 MiB streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_parquet(path: Path) -> dict:
    """Inspect a Parquet file's row count, schema, and split/parent cardinalities without loading heavy sequence columns."""
    pf = pq.ParquetFile(path)
    columns = list(pf.schema_arrow.names)
    info = {
        "rows": int(pf.metadata.num_rows),
        "columns": columns,
        "bytes": int(path.stat().st_size),
    }
    needed = [c for c in _META_INSPECT_COLS if c in columns]
    if needed:
        df = pd.read_parquet(path, columns=needed)
        for col in ("parent", "parent_seq", "wt_seq", "uniprot_id"):
            if col in df.columns:
                key = f"unique_{col}"
                info[key] = int(df[col].nunique())
        if "split" in df.columns:
            info["split_counts"] = {
                str(k): int(v) for k, v in df["split"].value_counts().items()
            }
    return info


def pack_true_contacts_tar(contacts_dir: Path, out_tar: Path) -> int:
    """Deterministically pack ``*.npy`` contact maps from ``contacts_dir`` into ``out_tar`` (``.tar.gz``)."""
    npy_files = sorted(contacts_dir.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy contact map files found in {contacts_dir}")
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tar, "wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz_fh:
            with tarfile.open(fileobj=gz_fh, mode="w") as tf:
                for p in npy_files:
                    ti = tf.gettarinfo(str(p), arcname=p.name)
                    ti.mtime = 0
                    ti.uid = 0
                    ti.gid = 0
                    ti.uname = ""
                    ti.gname = ""
                    with open(p, "rb") as src_fh:
                        tf.addfile(ti, src_fh)
    return len(npy_files)


def stage_local_processed(local_dir: Path, stage_dir: Path) -> dict:
    """
    Stage genuinely engineered local processed artifacts from ``local_dir``
    (e.g. ``DATASETS/DATASETS/data/``) into ``stage_dir``.
    """
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local processed directory not found: {local_dir}")

    for sub in SUBDIRS:
        (stage_dir / sub).mkdir(parents=True, exist_ok=True)

    file_mappings = (
        ("clinvar_protein_pairs.parquet", Path("tables") / "clinvar_protein_pairs.parquet"),
        ("megascale_double_mutants.parquet", Path("tables") / "megascale_double_mutants.parquet"),
        ("splits_clinvar.parquet", Path("splits") / "splits_clinvar.parquet"),
        ("leakage_report_pathogenicity.json", Path("splits") / "leakage_report_pathogenicity.json"),
        ("leakage_report_stability.json", Path("splits") / "leakage_report_stability.json"),
    )

    staged: list[str] = []
    skipped_missing: list[str] = []

    for src_name, rel_dst in file_mappings:
        src = local_dir / src_name
        dst = stage_dir / rel_dst
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rel_str = rel_dst.as_posix()
            staged.append(rel_str)
            print(f"[zenodo_pack] staged {src.name} -> {rel_str}")
        else:
            skipped_missing.append(src_name)
            print(f"[zenodo_pack] skipped missing optional artifact: {src}")

    contacts_dir = local_dir / "true_contacts"
    if contacts_dir.is_dir():
        tar_dst = stage_dir / "features" / "true_contacts_megascale.tar.gz"
        n_maps = pack_true_contacts_tar(contacts_dir, tar_dst)
        staged.append("features/true_contacts_megascale.tar.gz")
        print(f"[zenodo_pack] packed {n_maps} contact maps -> features/true_contacts_megascale.tar.gz")
    else:
        skipped_missing.append("true_contacts/")
        print(f"[zenodo_pack] skipped missing optional directory: {contacts_dir}")

    return {"staged": staged, "skipped_missing": skipped_missing}


def _render_datasheet(
    version: str,
    manifest: dict,
    table_stats: dict[str, dict],
    inventory: list[dict],
) -> str:
    rows_md: list[str] = []
    for rel, st in sorted(table_stats.items()):
        size_mb = st["bytes"] / 1e6
        rows_md.append(
            f"| `{rel}` | {st['rows']:,} | {len(st['columns'])} | {size_mb:.2f} MB |"
        )
    if not rows_md:
        rows_md.append(
            "| *(stage Parquet files into `tables/`, `splits/`, `features/`, `benchmark_predictions/`)* | — | — | — |"
        )

    other_md: list[str] = []
    for rec in inventory:
        if "parquet" in rec:
            continue
        rel = rec["path"]
        size_mb = rec["bytes"] / 1e6
        digest_prefix = rec["sha256"][:12]
        other_md.append(f"| `{rel}` | {size_mb:.2f} MB | `{digest_prefix}...` |")
    if not other_md:
        other_md.append("| *(no non-Parquet artifacts staged yet)* | — | — |")

    gen_utc = manifest["generated_utc"]
    total_files = manifest["total_files"]
    total_mb = manifest["total_bytes"] / 1e6

    return f"""# Holo-StabBench v{version} — Dataset & Artifacts Datasheet

Generated: `{gen_utc}` | Total files: **{total_files}** | Total size: **{total_mb:.2f} MB**

## 1. Overview & Motivation
**Holo-StabBench** accompanies the *Holo-GeoN* protein stability (ΔΔG), epistasis (ΔΔΔG), and variant pathogenicity study. It provides:
1. **Sign-harmonized, reading-frame-verified Parquet tables** (`positive = stabilizing`, $\\Delta\\Delta G = \\Delta G_{{\\text{{mut}}}} - \\Delta G_{{\\text{{wt}}}}$) spanning MegaScale cDNA proteolysis (singles + double-mutant epistasis), ClinVar + MANE Select GRCh38 v1.5 (pathogenicity with real `wt_dna`), FireProtDB 2.0 (MegaScale-removed + 25%-homology-filtered `FireProt-HF25`), S669/Ssym+, and Human Domainome 1.0 (`Domainome-HF25`, rank-only external aPCA validation).
2. **Sequence- and Structure-Level Leakage Audits**: MMseqs2 25% & 30% identity splits (`10,099` ClinVar clusters; MegaScale `id30`, `id25`, and `fold_tm60`), plus the `69,378`-pair all-vs-all **TM-align structural fold leakage audit** across MegaScale PDBs.
3. **Expensive Precomputed Features**: `412` frame-verified real-CDS ViennaRNA mRNA MFE + CAI profiles, `373` uniquely aligned C-beta (`<= 8 A`, CA for Gly) true contact maps (`features/true_contacts_megascale.tar.gz`), `166` SHA-256-pinned FireProtDB AlphaFold DB structures (`166/176` resolved), and `538` aligned SE(3)-invariant backbone geometry tensors (`373` MegaScale + `165` FireProtDB: `N/CA/C/CB`, `pLDDT`, `coord_valid`).
4. **Trained Checkpoints & Common-Format Predictions**: Self-describing PyTorch weights (5 seeds + calibration JSONs) and `(row_id, y_pred)` Parquets for Holo-GeoN and benchmarked competitors so every table & cluster-bootstrap CI reproduces in one command via `scripts/score_predictions.py`.

## 2. Staged Artifacts

### 2.1 Staged Parquet Tables
| Artifact | Rows | Columns | Size |
|---|---:|---:|---:|
{chr(10).join(rows_md)}

### 2.2 Staged Archives, Tensors & Audit Reports
| Artifact | Size | SHA-256 Prefix |
|---|---:|---|
{chr(10).join(other_md)}

## 3. Included vs. Excluded Provenance Table

| Dataset / Artifact | Status | Scale / Composition | Rationale & Verification |
|---|---|---|---|
| **ClinVar + MANE Select (GRCh38 v1.5)** (`tables/clinvar_protein_pairs.parquet`, `splits/splits_clinvar.parquet`, `splits/leakage_report_pathogenicity.json`) | **Included** | 190,389 missense rows; 15,177 parent proteins; 10,099 MMseqs2-30% clusters (`154,917` train / `15,037` val / `20,435` test) | Engineered via `build_clinvar_table.py`; HGVS ref-residue verified against MANE NP protein; translation-verified real `wt_dna` CDS; zero 8-mer near-duplicate train/heldout leakage |
| **MegaScale Singles & 3-Tier Splits** (`tables/megascale_ddg_pairs.parquet`, `splits/splits_megascale_{{id30,id25,fold_tm60}}.parquet`, `splits/leakage_report_stability.json`, `splits/leakage_report_megascale_id25.json`) | **Included** | 389,068 single substitutions across 412 constructs; 190 (`id30`) / 168 (`id25`) / 76 (`fold_tm60`) clusters | Sign-checked (`Stabilizing_mut` positive, `ddG = dG_mut - dG_wt`), propagated `ddg_std`, translation-verified `wt_dna`, zero 8-mer overlap in `id25` |
| **MegaScale Double Mutants / Epistasis** (`tables/megascale_double_mutants.parquet`) | **Included** | Exact 2-point substitutions from Tsuboyama Zenodo `7844779` matched to same-parent `ddg_a` & `ddg_b` | Canonicalized `site_i < site_j`, `wt_seq[site] == wt_aa`, exact `dddg = ddg_ab - ddg_a - ddg_b`, aligned 3D `dist_cb` from `AlphaFold_model_PDBs/` for `<8 A` vs `>=8 A` stratification |
| **FireProtDB 2.0 & FireProt-HF25** (`tables/fireprotdb_holdout.parquet`, `tables/fireprotdb_hf25.parquet`, `splits/fireprotdb_homology_audit.parquet`) | **Included** | 4,233 non-MegaScale rows / 176 proteins; `FireProt-HF25`: 1,894 rows / 108 proteins (`<=25%` seq id to MegaScale) | MegaScale rows and substring-contaminated proteins removed; classical rows sign-flipped to `stabilizing_positive`; 68/176 homologous proteins removed in `HF25` |
| **S669 / Ssym+** (`tables/s669_holdout.parquet`) | **Included** | 669 rows / 94 PDBs (93 unique sequences) | 100% WT-chain residue verified against `S669/pdbs/`; stabilizing-positive (`Experimental_DDG_dir`) |
| **Human Domainome 1.0 & Domainome-HF25** (`tables/domainome_apca_holdout.parquet`, `tables/domainome_apca_hf25.parquet`, `splits/domainome_homology_audit.parquet`) | **Included** | 536,164 aPCA rows / 522 domains; `Domainome-HF25`: 332,854 rows / 327 domains (`<=25%` seq id) | 100% WT-sequence verified; 195/522 homologous domains removed in `HF25`; evaluated strictly by per-domain Spearman rank (not kcal/mol) |
| **TM-align Structural Leakage Audit** (`splits/megascale_tm_all_vs_all.parquet`, `splits/megascale_tm_leakage_matrix.parquet`, `splits/megascale_tm_test_max.parquet`) | **Included** | 69,378 all-vs-all pairs (373 structured domains) + 12,432 test-vs-train/val pairs | Establishes exact fold leakage across `id30`, `id25`, and `fold_tm60` |
| **Aligned True C-beta Contact Maps** (`features/true_contacts_megascale.tar.gz`) | **Included** | 373 uniquely aligned binary `.npy` contact maps (`dist_CB < 8 A`, CA for Gly) | Keyed by `sha1(parent_seq)` after unambiguous PDB-to-construct block alignment (`build_contact_maps.py`) |
| **Mechanistic CDS Profiles** (`features/cds_mechanistic_vienna_cai.parquet`) | **Included** | 412 frame-verified real-CDS ViennaRNA mRNA MFE + CAI profiles | 412/412 MegaScale constructs frame-verified by codon translation |
| **SE(3)-Invariant Backbone Tensors** (`features/invariant_backbone_geometry.npz`) | **Included** | 538 aligned tensors (373 MegaScale + 165 FireProtDB) | Aligned `[N, CA, C, CB]`, `pLDDT`, and `coord_valid` arrays for the 3D geometry ablation arms |
| **FireProtDB AlphaFold Structures** (`features/alphafold_fireprot_pdbs.tar.gz`, `features/alphafold_fireprot_manifest.json`) | **Included** | 166 SHA-256-pinned AFDB PDB files (`166/176` resolved) | 10 FireProtDB accessions unavailable in AFDB are documented in manifest |
| **Raw `massive_kb` spectral dump** (`LIBRARY_TO_SPTXT...sptxt`) | **Excluded** | 27.4 GB (28.76 GB decimal / 26.78 GiB) raw `.sptxt` | Unused third-party proteomics spectral library outside ΔΔG/pathogenicity scope; bloats archive with zero benchmark utility |
| **Raw `uniref50` FASTA** (`uniref50.fasta.gz`) | **Excluded** | 11.8 GB (12.39 GB decimal / 11.53 GiB) raw FASTA | Unmodified third-party clustering background database; freely downloadable from UniProt |
| **Raw ClinVar VCF & TSV** (`clinvar.vcf`, `clinvar.vcf.gz`, `variant_summary.txt.gz`) | **Excluded** | ~2.35 GiB (2.52 GB) raw dumps | Raw VCF carries genomic coordinates only (risking silent nucleotide-as-protein errors); all verified missense info is captured in `clinvar_protein_pairs.parquet` |
| **Raw FireProtDB CSV** (`fireprotdb_20251015-164116.csv`) | **Excluded** | 1.65 GiB (1.78 GB) raw export | 98.6% (`604,920/613,208` ddG rows) redundant with MegaScale and carries inverted classical sign; superseded by `fireprotdb_holdout.parquet` + `fireprotdb_hf25.parquet` |
| **Raw MegaScale CSVs & PDB Archive** (`Tsuboyama2023_Dataset1/2_3.csv`, `K50_dG_tables/`, `AlphaFold_model_PDBs/`) | **Excluded** | ~3.40 GiB (3.65 GB) raw release (`665 MiB` `Dataset2_Dataset3.csv`, `862` PDBs) | Upstream Zenodo `7844779` dump; superseded by sign-checked, CDS-verified Parquets, `true_contacts_megascale.tar.gz`, and `invariant_backbone_geometry.npz`, reproducible via committed ETL scripts |
| **MGnify Stability** (Cho/Rocklin/OpenFold, bioRxiv `10.64898/2026.05.19.726285`) | **Excluded** | Optional external pretraining corpus | Portal-registration-gated terms restrict raw redistribution; ship deterministic ETL scripts & split IDs instead |

## 4. Claims Discipline & Split Leakage Audit
* **Model Geometry Scope:** Default Holo-GeoN is a **structure-free contact-graph** model (ESM-2 `E_geo` + `E_mech`). It is **not** SE(3)- or E(3)-equivariant. Only the 3D coordinate ablation arm (`invariant_backbone_geometry.npz`) is **SE(3)-invariant**. Exact algebraic symmetries in the default architecture are wild-type/mutant swap equivariance and Potts site-swap / double-reversal parity in `ContactGatedEpistasisHead`.
* **`fold_tm60` is fold-clustered, NOT fold-disjoint:** The published 76 fold clusters are `TM_min >= 0.60` links united with the `id30` sequence clusters (`TM_min >= 0.60` alone gives 78). Across `fold_tm60`, `max_cross_tm_min = 0.5956` (no test-train pair reaches `TM_min >= 0.60`), yet **37/39** structured test domains still reach `TM_max >= 0.5` (max `0.6874`; `25/39` at `TM_min >= 0.5`). At `TM_max >= 0.50`, `97.05%` (`362/373`) of structured MegaScale domains collapse into a single connected component, so no split disjoint at `TM_max >= 0.5` exists. Never write "fold-disjoint" without the `TM_min >= 0.60` qualifier.
* **Nearest-train median `TM_max` by MegaScale split:**
  * `id30`: **0.82** (`0.8243`; `37/37` test domains `TM_max >= 0.5`, `27/37` `TM_max >= 0.7`)
  * `id25`: **0.72** (`0.7176`; `40/40` test domains `TM_max >= 0.5`, `22/40` `TM_max >= 0.7`, zero 8-mer overlap `>= 0.30`)
  * `fold_tm60`: **0.62** (`0.6244`; `37/39` test domains `TM_max >= 0.5`, `0/39` `TM_max >= 0.7`, `max_cross_tm_min = 0.5956`)
* **Metric Separation:** Pooled and per-protein-median metrics are always reported in separate columns with cluster-bootstrap 95% CIs (`group_used`). `Domainome 1.0` aPCA abundance is evaluated strictly by per-domain Spearman rank, never kcal/mol RMSE.

## 5. Upstream Provenance & Licences
* **MegaScale cDNA proteolysis** (Tsuboyama et al., *Nature* 2023; Zenodo `7844779`) — CC-BY 4.0 / open academic.
* **ClinVar** (NCBI / NIH) & **MANE Select GRCh38 v1.5** (NCBI / EMBL-EBI) — public domain / CC0.
* **FireProtDB 2.0** (Stourac et al., *NAR* 2021/2025) — open academic database; MegaScale-sourced rows strictly removed from external holdout.
* **S669 / Ssym+** (Pancotti et al., *Brief. Bioinform.* 2022; Zenodo `7568094`) — CC-BY 4.0.
* **Human Domainome 1.0** (Beltran et al., *Nature* 2025; Zenodo `14356805`) — CC-BY 4.0 (aPCA abundance: Spearman rank-only evaluation, not kcal/mol).
* **AlphaFold Protein Structure Database** (EMBL-EBI / Google DeepMind) — CC-BY 4.0.
* **MGnify Stability** (Cho/Rocklin/OpenFold, bioRxiv `10.64898/2026.05.19.726285`) — if used in extended pretraining, check portal registration terms before redistributing raw tables; ship deterministic ETL scripts & split IDs instead.

## 6. Integrity Verification
Verify all files against `SHA256SUMS.txt`:
```bash
sha256sum -c SHA256SUMS.txt
```
"""


def build_release_bundle(
    stage_dir: Path,
    version: str = "1.0.0",
    local_processed_dir: Path | None = None,
) -> dict:
    """
    Validate, checksum, and document the Holo-StabBench release directory.

    When ``local_processed_dir`` is provided, genuinely engineered local artifacts
    (ClinVar + MANE pairs/splits, leakage reports, MegaScale double mutants if
    present, and ``true_contacts/`` packed as ``true_contacts_megascale.tar.gz``)
    are staged prior to hashing and datasheet generation.
    """
    stage_dir = Path(stage_dir)
    for sub in SUBDIRS:
        (stage_dir / sub).mkdir(parents=True, exist_ok=True)

    staging_report = None
    if local_processed_dir is not None:
        staging_report = stage_local_processed(Path(local_processed_dir), stage_dir)

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

    (stage_dir / "SHA256SUMS.txt").write_text(
        "\n".join(sha_lines) + ("\n" if sha_lines else ""),
        encoding="utf-8",
    )

    manifest = {
        "name": "Holo-StabBench",
        "version": version,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sign_convention": "stabilizing_positive (ddG = dG_mut - dG_wt; positive = stabilizing)",
        "total_files": len(inventory),
        "total_bytes": sum(r["bytes"] for r in inventory),
        "files": inventory,
    }
    if staging_report is not None:
        manifest["local_processed_staging"] = staging_report
    (stage_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    datasheet = _render_datasheet(version, manifest, table_stats, inventory)
    (stage_dir / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    total_mb = manifest["total_bytes"] / 1e6
    print(
        f"[zenodo_pack] staged {manifest['total_files']} files "
        f"({total_mb:.2f} MB) -> {stage_dir}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--stage",
        default=os.path.join("..", "extras", "zenodo_staging", "Holo-StabBench-v1.0"),
    )
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument(
        "--local-processed-dir",
        default=None,
        help="Optional directory (e.g. DATASETS/DATASETS/data) from which to stage "
        "local processed tables, splits, leakage reports, and true_contacts/",
    )
    args = ap.parse_args(argv)
    local_dir = Path(args.local_processed_dir).resolve() if args.local_processed_dir else None
    build_release_bundle(
        Path(args.stage).resolve(),
        version=args.version,
        local_processed_dir=local_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
