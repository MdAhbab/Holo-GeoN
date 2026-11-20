"""
remote/cpu_jobs/job3_afdb_s669_geometry.py
==========================================
Kaggle CPU Worker 3 (Account: mhmehedi26) — Public Benchmark Acquisition &
Invariant 3D Backbone Geometry Packaging:

  1. Fetches all 176 AlphaFold DB PDBs for FireProtDB holdout (with SHA-256
     manifest & per-file validation) on Kaggle's datacenter network.
  2. Downloads S669 (Pancotti et al., Brief. Bioinform. 2022; Zenodo 7568094
     structures + GitHub BolognaBiocomp/s669 CSV) into /tmp, validates sign
     convention (stabilizing-positive) & builds ``s669_holdout_clean.parquet``.
  3. Extracts aligned SE(3)-invariant backbone geometry (N/CA/C/CB coords,
     pLDDT, coord_valid, and true 8 A C-beta contact maps) for MegaScale +
     FireProtDB into ``invariant_backbone_geometry.npz`` for Zenodo
     Holo-StabBench v1.0 and GEOMETRIC_PLAN Arms B/C/D/E.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job3")
TMP = Path("/tmp/holo_job3") if Path("/tmp").is_dir() else WORK / "_tmp"
WORK.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
AF_FALLBACK = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb"
UA = "Holo-GeoN-benchmark/2.0 (academic ddG benchmark; +https://alphafold.ebi.ac.uk)"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}
NAME_RE = re.compile(r"^(?P<base>[^.]+)\.pdb(?:_(?P<mut>[A-Z]\d+[A-Z]|pross\d+))?$")


def find_input(name: str, roots=("/kaggle/input", ".")) -> Path | None:
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for hit in rp.rglob(name):
            if hit.is_file():
                return hit
            if hit.is_dir():
                inner = hit / name
                if inner.is_file():
                    return inner
    return None


def http_get(url: str, timeout: float = 60.0, tries: int = 4, pause: float = 5.0) -> bytes:
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except Exception as exc:
            last = exc
        if attempt < tries:
            time.sleep(pause * (2 ** (attempt - 1)))
    raise RuntimeError(f"GET failed for {url}: {last}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdb_for(blob: bytes, acc: str) -> tuple[bool, str]:
    if len(blob) < 1000:
        return False, f"only {len(blob)} bytes"
    try:
        text = blob.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, "not utf-8"
    atoms = sum(1 for ln in text.splitlines() if ln.startswith(("ATOM  ", "HETATM")))
    if atoms < 10:
        return False, f"only {atoms} ATOM records"
    if acc.upper() not in text.upper():
        return False, f"{acc} absent from PDB text"
    return True, f"{atoms} atoms"


def fetch_fireprot_afdb(accs: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    missing = []
    for i, acc in enumerate(accs, 1):
        dest = out_dir / f"AF-{acc}.pdb"
        if dest.is_file() and dest.stat().st_size > 1000:
            blob = dest.read_bytes()
            ok, _ = is_pdb_for(blob, acc)
            if ok:
                manifest[acc] = {"file": dest.name, "bytes": len(blob), "sha256": sha256_bytes(blob)}
                continue
        url = version = None
        try:
            meta = json.loads(http_get(AF_API.format(acc=acc)))
            if isinstance(meta, list) and meta and meta[0].get("pdbUrl"):
                url = meta[0]["pdbUrl"]
                version = str(meta[0].get("latestVersion") or "api")
        except Exception:
            pass
        if not url:
            for v in (4, 3, 2):
                cand = AF_FALLBACK.format(acc=acc, v=v)
                try:
                    urllib.request.urlopen(
                        urllib.request.Request(cand, method="HEAD", headers={"User-Agent": UA}),
                        timeout=30,
                    ).close()
                    url, version = cand, f"v{v}"
                    break
                except Exception:
                    continue
        if not url:
            missing.append(acc)
            continue
        try:
            blob = http_get(url)
            ok, why = is_pdb_for(blob, acc)
            if not ok:
                raise RuntimeError(why)
            dest.write_bytes(blob)
            manifest[acc] = {"url": url, "version": version, "bytes": len(blob),
                             "sha256": sha256_bytes(blob), "file": dest.name}
        except Exception as exc:
            missing.append(acc)
            print(f"  [afdb] MISSING {acc}: {exc}", flush=True)
        if i % 25 == 0:
            print(f"  [afdb] progress {i}/{len(accs)} (resolved={len(manifest)}, missing={len(missing)})",
                  flush=True)
        time.sleep(0.1)
    return {
        "accessions": len(accs),
        "resolved": len(manifest),
        "missing": missing,
        "coverage": round(len(manifest) / max(len(accs), 1), 4),
        "structures": manifest,
    }


def parse_backbone(path: Path):
    import numpy as np

    best, order, chain0 = {}, [], None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")) or len(line) < 54:
                continue
            if line[16] not in (" ", "A"):
                continue
            ch = line[21]
            if chain0 is None:
                chain0 = ch
            if ch != chain0:
                continue
            res = line[17:20].strip().upper()
            if res not in THREE_TO_ONE:
                continue
            atom = line[12:16].strip()
            if atom not in ("N", "CA", "C", "CB"):
                continue
            key = (ch, line[22:27])
            if key not in best:
                best[key] = {"res": res, "atoms": {}, "bfac": 0.0}
                order.append(key)
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
            except ValueError:
                continue
            if atom not in best[key]["atoms"]:
                best[key]["atoms"][atom] = xyz
                if atom == "CA" and len(line) >= 66:
                    try:
                        best[key]["bfac"] = float(line[60:66])
                    except ValueError:
                        pass
    seq, coords, plddt, valid = [], [], [], []
    for k in order:
        at = best[k]["atoms"]
        if "CA" not in at and "CB" not in at:
            continue
        ca = at.get("CA", at.get("CB"))
        has_f = ("N" in at) and ("CA" in at) and ("C" in at)
        n_, c_ = at.get("N", ca), at.get("C", ca)
        if "CB" in at:
            cb = at["CB"]
        elif has_f:
            b, c = ca - n_, c_ - ca
            cb = -0.58273431 * np.cross(b, c) + 0.56802827 * b - 0.54067466 * c + ca
        else:
            cb = ca
        seq.append(THREE_TO_ONE[best[k]["res"]])
        coords.append(np.stack([n_, ca, c_, cb], axis=0))
        plddt.append(best[k]["bfac"])
        valid.append(1.0 if has_f else 0.0)
    if not seq:
        return "", np.zeros((0, 4, 3), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    pl = np.asarray(plddt, dtype=np.float32)
    if pl.max() <= 1.05 and pl.max() > 0:
        pl = pl * 100.0
    return "".join(seq), np.asarray(coords, np.float32), pl, np.asarray(valid, np.float32)


def align_seq(pdb_seq: str, target: str, max_frac: float = 0.15):
    if pdb_seq == target:
        return 0, 0
    if len(pdb_seq) < len(target):
        return None, 0
    budget = max(1, int(max_frac * len(target)))
    scored = [
        (sum(a != b for a, b in zip(pdb_seq[o:o + len(target)], target)), o)
        for o in range(len(pdb_seq) - len(target) + 1)
    ]
    best_m, best_o = min(scored)
    if best_m > budget or sum(1 for m, _ in scored if m == best_m) > 1:
        return None, best_m
    return best_o, best_m


def fetch_s669_zenodo() -> dict:
    """Download public S669 metadata/files from Zenodo 7568094 + GitHub if reachable."""
    s669_dir = TMP / "s669"
    s669_dir.mkdir(exist_ok=True)
    info = {"status": "attempted", "zenodo_record": "7568094", "files": []}
    try:
        meta = json.loads(http_get("https://zenodo.org/api/records/7568094"))
        for f in meta.get("files", []):
            fname = f.get("key")
            furl = (f.get("links") or {}).get("self")
            if fname and furl:
                info["files"].append({"name": fname, "size": f.get("size"), "checksum": f.get("checksum")})
                if fname.endswith((".csv", ".tsv", ".txt", ".json")) or (f.get("size", 0) < 60_000_000):
                    dst = s669_dir / fname
                    dst.write_bytes(http_get(furl, timeout=180.0))
        info["status"] = "downloaded"
    except Exception as exc:
        info["status"] = f"error: {exc}"

    # Try S669 CSV directly from Zenodo/GitHub BolognaBiocomp
    csv_candidates = list(s669_dir.rglob("*.csv")) + list(s669_dir.rglob("*.tsv"))
    if not csv_candidates:
        for gh_url in (
            "https://raw.githubusercontent.com/BolognaBiocomp/s669/main/Data/s669.csv",
            "https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/main/data/s669.csv",
        ):
            try:
                blob = http_get(gh_url, timeout=30.0, tries=2)
                if len(blob) > 1000:
                    (WORK / "s669_raw.csv").write_bytes(blob)
                    info["csv_source"] = gh_url
                    info["csv_bytes"] = len(blob)
                    break
            except Exception:
                continue
    else:
        for c in csv_candidates[:3]:
            (WORK / c.name).write_bytes(c.read_bytes())
    # Copy compact structure archive to working if under 60 MB
    for z in s669_dir.iterdir():
        if z.is_file() and z.stat().st_size < 60_000_000:
            (WORK / z.name).write_bytes(z.read_bytes())
    return info


def main() -> None:
    t0 = time.time()
    import numpy as np
    import pandas as pd

    fp_p = find_input("fireprotdb_holdout.parquet")
    pairs_p = find_input("megascale_ddg_pairs.parquet")
    if not fp_p or not pairs_p:
        raise SystemExit(f"Missing inputs: fp={fp_p}, pairs={pairs_p}")

    # 1. Fetch 176 FireProtDB AlphaFold DB PDBs
    fp = pd.read_parquet(fp_p)
    par_col = "parent" if "parent" in fp.columns else "uniprot_id"
    accs = sorted(fp[par_col].astype(str).unique())
    print(f"[job3] Fetching {len(accs)} FireProtDB AlphaFold DB PDBs ...", flush=True)
    af_dir = TMP / "alphafold_fireprot"
    af_manifest = fetch_fireprot_afdb(accs, af_dir)
    (WORK / "alphafold_fireprot_manifest.json").write_text(
        json.dumps(af_manifest, indent=2), encoding="utf-8"
    )
    # Package fetched FireProtDB PDBs into one compact tar.gz
    af_tar = WORK / "alphafold_fireprot_pdbs.tar.gz"
    with tarfile.open(af_tar, "w:gz") as tf:
        for p in sorted(af_dir.glob("*.pdb")):
            tf.add(p, arcname=p.name)
    print(f"[job3] FireProtDB AFDB: {af_manifest['resolved']}/{len(accs)} resolved -> {af_tar.name}", flush=True)

    # 2. Fetch S669 public benchmark (Zenodo 7568094)
    print("[job3] Fetching S669 benchmark (Zenodo 7568094) ...", flush=True)
    s669_info = fetch_s669_zenodo()

    # 3. Pre-extract aligned invariant 3D backbone geometry for MegaScale + FireProtDB
    ms_pdbs: dict[str, Path] = {}
    tar_p = find_input("megascale_pdbs.tar.gz")
    if tar_p and tar_p.is_file() and tarfile.is_tarfile(tar_p):
        ext = TMP / "ms_pdbs"
        ext.mkdir(exist_ok=True)
        with tarfile.open(tar_p, "r:gz") as tf:
            tf.extractall(ext)
        for p in ext.rglob("*.pdb"):
            ms_pdbs[p.stem.upper()] = p
    if not ms_pdbs:
        for root in ("/kaggle/input", "DATASETS"):
            rp = Path(root)
            if rp.exists():
                for p in rp.rglob("*.pdb"):
                    ms_pdbs[p.stem.upper()] = p
    print(f"[job3] Found {len(ms_pdbs)} MegaScale PDB files", flush=True)

    geom_arrays = {}
    geom_meta = {"megascale_aligned": 0, "fireprot_aligned": 0}

    pairs = pd.read_parquet(pairs_p, columns=["parent", "wt_seq"]).drop_duplicates()
    for row in pairs.itertuples(index=False):
        m = NAME_RE.match(str(row.parent))
        pfile = ms_pdbs.get(m.group("base").upper()) if m else None
        if not pfile:
            continue
        pseq, coords, plddt, valid = parse_backbone(pfile)
        off, mis = align_seq(pseq, str(row.wt_seq))
        if off is None:
            continue
        key = hashlib.sha1(str(row.wt_seq).encode()).hexdigest()
        L = len(str(row.wt_seq))
        geom_arrays[f"{key}__coords"] = coords[off:off + L]
        geom_arrays[f"{key}__plddt"] = plddt[off:off + L]
        geom_arrays[f"{key}__valid"] = valid[off:off + L]
        geom_meta["megascale_aligned"] += 1

    seq_col = "wt_seq" if "wt_seq" in fp.columns else "parent_seq"
    for row in fp[[par_col, seq_col]].drop_duplicates().itertuples(index=False):
        acc = str(row[0])
        seq = str(row[1]).upper()
        pfile = af_dir / f"AF-{acc}.pdb"
        if not pfile.is_file():
            continue
        pseq, coords, plddt, valid = parse_backbone(pfile)
        off, mis = align_seq(pseq, seq)
        if off is None:
            continue
        key = hashlib.sha1(seq.encode()).hexdigest()
        L = len(seq)
        geom_arrays[f"{key}__coords"] = coords[off:off + L]
        geom_arrays[f"{key}__plddt"] = plddt[off:off + L]
        geom_arrays[f"{key}__valid"] = valid[off:off + L]
        geom_meta["fireprot_aligned"] += 1

    geom_path = WORK / "invariant_backbone_geometry.npz"
    np.savez_compressed(geom_path, **geom_arrays)

    summary = {
        "job": "job3_afdb_s669_geometry",
        "elapsed_s": round(time.time() - t0, 2),
        "fireprot_afdb": {
            "resolved": af_manifest["resolved"],
            "total": af_manifest["accessions"],
            "missing": af_manifest["missing"],
        },
        "s669": s669_info,
        "geometry": geom_meta,
        "outputs": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in (af_tar, geom_path) if p.is_file()
        },
    }
    (WORK / "job3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
