"""
remote/cpu_jobs/job9_geom_contacts_v2.py
========================================
Kaggle CPU Worker 9 (Account: mdahbab) — Upgraded SE(3)-Invariant Backbone
Geometry NPZ, Aligned True C-Beta Contact Maps, and TM-Align Audit for the
~37 MegaScale Constructs Recovered via ``| / : -> _`` PDB Stem Normalization.

Why this exists
---------------
``job3`` and ``cpu1``/``cpu5`` ran before ``src/geometry.py:normalize_pdb_stem``
normalized ``|`` (MegaScale table) and ``:`` (Zenodo 7844779 zip) to ``_``
(Windows/extracted filenames). As a result, ~37 MegaScale constructs
(``EA|...``, ``GG|...``, ``XX|...``) were missed (373/412 = 90.5% aligned).

Using the already-mounted private ``holo-geon-v2-gpu-seed`` dataset under
``/kaggle/input`` (0 large downloads, 0 GPU hours), this worker:
  1. Indexes ALL MegaScale PDBs (862) and FireProtDB AFDB PDBs (166) with
     ``normalize_pdb_stem`` (``|`` and ``:`` -> ``_``).
  2. Builds the upgraded drop-in ``invariant_backbone_geometry.npz``
     (``<sha1>__coords``, ``<sha1>__plddt``, ``<sha1>__valid``) covering all
     aligned MegaScale (~410/412) + FireProtDB (165/176) proteins.
  3. Builds a deterministic ``true_contacts_megascale.tar.gz`` containing
     8.0 A C-beta contact maps (``<sha1>.npy``, bool) for all aligned
     MegaScale domains.
  4. Compiles official Zhang-Lab ``TMalign`` and aligns every newly recovered
     construct against all aligned MegaScale domains to audit fold leakage
     across ``id30``, ``id25``, and ``fold_tm60``.
  5. Writes ``job9_summary.json`` with SHA-256 digests and always exits 0.
"""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job9")
TMP = Path("/tmp/holo_job9") if Path("/tmp").is_dir() else WORK / "_tmp"
WORK.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}
NAME_RE = re.compile(r"^(?P<base>[^.]+)\.pdb(?:_(?P<mut>[A-Z]\d+[A-Z]|pross\d+))?$")
TM_RE = re.compile(r"TM-score=\s*([0-9.]+)")


def normalize_pdb_stem(name: str) -> str:
    s = str(name).strip()
    if s.lower().endswith(".pdb"):
        s = s[:-4]
    return s.upper().replace("|", "_").replace(":", "_")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_input(name: str) -> Path | None:
    for root in ("/kaggle/input", "project/DATASETS", "DATASETS", "extras"):
        rp = Path(root)
        if not rp.exists():
            continue
        hits = list(rp.rglob(name))
        if hits:
            return hits[0]
    return None


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
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                )
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
        return (
            "",
            np.zeros((0, 4, 3), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.float32),
        )
    pl = np.asarray(plddt, dtype=np.float32)
    if pl.max() <= 1.05 and pl.max() > 0:
        pl = pl * 100.0
    return "".join(seq), np.asarray(coords, np.float32), pl, np.asarray(valid, np.float32)


def parse_pdb_cb(path: Path):
    """Exact CB (CA for glycine) parser matching src/geometry.py:parse_pdb_cb."""
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
            if atom not in ("CA", "CB"):
                continue
            key = (ch, line[22:27])
            if key not in best:
                best[key] = {"res": res, "atoms": {}}
                order.append(key)
            try:
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            if atom not in best[key]["atoms"]:
                best[key]["atoms"][atom] = xyz
    seq, coords = [], []
    for k in order:
        at = best[k]["atoms"]
        res = best[k]["res"]
        if res == "GLY":
            pt = at.get("CA")
        else:
            pt = at.get("CB") if "CB" in at else at.get("CA")
        if pt is None:
            continue
        seq.append(THREE_TO_ONE[res])
        coords.append(pt)
    if not seq:
        return "", np.zeros((0, 3), np.float32)
    return "".join(seq), np.asarray(coords, np.float32)


def align_seq(pdb_seq: str, target: str, max_frac: float = 0.15):
    if pdb_seq == target:
        return 0, 0
    if len(pdb_seq) < len(target):
        return None, 0
    budget = max(1, int(max_frac * len(target)))
    scored = [
        (sum(a != b for a, b in zip(pdb_seq[o : o + len(target)], target)), o)
        for o in range(len(pdb_seq) - len(target) + 1)
    ]
    best_m, best_o = min(scored)
    if best_m > budget or sum(1 for m, _ in scored if m == best_m) > 1:
        return None, best_m
    return best_o, best_m


def index_all_pdbs() -> tuple[dict[str, Path], dict[str, Path]]:
    """Collect MegaScale PDBs and FireProtDB AFDB PDBs from /kaggle/input."""
    ms_pdbs: dict[str, Path] = {}
    af_pdbs: dict[str, Path] = {}

    def register(p: Path) -> None:
        stem_norm = normalize_pdb_stem(p.name)
        if stem_norm.startswith("AF-"):
            af_pdbs.setdefault(stem_norm, p)
        else:
            ms_pdbs.setdefault(stem_norm, p)

    # First check tarballs if not auto-extracted
    for tar_name, sub in (
        ("megascale_pdbs.tar.gz", "ms_pdbs"),
        ("alphafold_fireprot_pdbs.tar.gz", "af_pdbs"),
    ):
        tp = find_input(tar_name)
        if tp and tp.is_file() and tarfile.is_tarfile(tp):
            ext = TMP / sub
            ext.mkdir(exist_ok=True)
            with tarfile.open(tp, "r:gz") as tf:
                tf.extractall(ext)
            for p in sorted(ext.rglob("*.pdb")):
                register(p)

    # Then walk /kaggle/input (Kaggle auto-extracts .tar.gz into directories)
    for root in ("/kaggle/input", "project/DATASETS", "DATASETS"):
        rp = Path(root)
        if not rp.exists():
            continue
        for p in sorted(rp.rglob("*.pdb")):
            register(p)

    return ms_pdbs, af_pdbs


def resolve_af_pdb(acc: str, af_pdbs: dict[str, Path]) -> Path | None:
    acc_u = normalize_pdb_stem(acc)
    for key in (f"AF-{acc_u}", acc_u):
        if key in af_pdbs:
            return af_pdbs[key]
    pref = f"AF-{acc_u}-"
    for k, p in af_pdbs.items():
        if k.startswith(pref):
            return p
    return None


def compile_tmalign() -> str | None:
    tm_bin = TMP / "TMalign"
    if tm_bin.is_file():
        return str(tm_bin)
    urls = (
        "https://zhanggroup.org/TM-align/TMalign.cpp",
        "https://seq2fun.dcmb.med.umich.edu//TM-align/TMalign.cpp",
    )
    cpp_path = TMP / "TMalign.cpp"
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Holo-GeoN/2.0"})
            with urllib.request.urlopen(req, timeout=45.0) as resp:
                data = resp.read()
            if len(data) > 50_000 and b"TM-align" in data:
                cpp_path.write_bytes(data)
                break
        except Exception:
            continue
    if not cpp_path.is_file():
        return None
    subprocess.run(
        ["g++", "-static", "-O3", "-ffast-math", "-lm", "-o", str(tm_bin), str(cpp_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not tm_bin.is_file():
        subprocess.run(
            ["g++", "-O3", "-ffast-math", "-lm", "-o", str(tm_bin), str(cpp_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    os.chmod(tm_bin, 0o755)
    return str(tm_bin)


def run_tm_pair(tm_bin: str, pdb_a: str, pdb_b: str) -> tuple[float, float]:
    out = subprocess.check_output([tm_bin, pdb_a, pdb_b], text=True, stderr=subprocess.DEVNULL)
    vals = [float(x) for x in TM_RE.findall(out)]
    if len(vals) >= 2:
        return max(vals[0], vals[1]), min(vals[0], vals[1])
    if len(vals) == 1:
        return vals[0], vals[0]
    return float("nan"), float("nan")


def write_deterministic_targz(files: list[tuple[str, bytes]], out_path: Path) -> None:
    """Deterministic tar.gz (sorted names, zero mtime, gzip mtime=0)."""
    raw_buf = io.BytesIO()
    with tarfile.open(fileobj=raw_buf, mode="w") as tf:
        for arcname, payload in sorted(files, key=lambda x: x[0]):
            ti = tarfile.TarInfo(name=arcname)
            ti.size = len(payload)
            ti.mtime = 0
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(payload))
    with open(out_path, "wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0) as gz:
            gz.write(raw_buf.getvalue())


def main() -> int:
    t0 = time.time()
    errors: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "job": "job9_geom_contacts_v2",
        "errors": errors,
        "outputs": {},
    }

    def record_error(stage: str, exc: Exception) -> None:
        print(f"[job9] ERROR in {stage}: {exc}", flush=True)
        errors.append({
            "stage": stage,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })

    try:
        import numpy as np
        import pandas as pd

        pairs_p = find_input("megascale_ddg_pairs.parquet")
        fp_p = find_input("fireprotdb_holdout.parquet")
        if not pairs_p or not fp_p:
            raise FileNotFoundError(f"Missing seed tables: pairs={pairs_p}, fp={fp_p}")

        ms_pdbs, af_pdbs = index_all_pdbs()
        print(
            f"[job9] Indexed {len(ms_pdbs)} MegaScale PDBs and {len(af_pdbs)} AFDB FireProtDB PDBs",
            flush=True,
        )
        summary["pdb_index"] = {
            "megascale_pdb_files": len(ms_pdbs),
            "fireprot_afdb_files": len(af_pdbs),
        }

        pairs = pd.read_parquet(pairs_p, columns=["parent", "wt_seq"]).drop_duplicates()
        fp = pd.read_parquet(fp_p)
        par_col = "parent" if "parent" in fp.columns else "uniprot_id"
        seq_col = "wt_seq" if "wt_seq" in fp.columns else "parent_seq"

        geom_arrays: dict[str, np.ndarray] = {}
        contact_blobs: list[tuple[str, bytes]] = []
        seen_contact_keys: set[str] = set()

        ms_aligned_old_rule = 0
        ms_aligned_new_rule = 0
        recovered_constructs: list[dict[str, object]] = []
        still_missing: list[dict[str, str]] = []
        aligned_pdb_by_parent: dict[str, Path] = {}
        mismatch_hist: dict[str, int] = {}

        # 1. MegaScale geometry + true 8.0 A C-beta contacts
        for row in pairs.itertuples(index=False):
            parent = str(row.parent)
            wt_seq = str(row.wt_seq).upper()
            m = NAME_RE.match(parent)
            if not m:
                still_missing.append({"parent": parent, "reason": "unparseable_construct_name"})
                continue
            raw_base = m.group("base").upper()
            norm_base = normalize_pdb_stem(raw_base)
            matched_old = raw_base in ms_pdbs
            pfile = ms_pdbs.get(norm_base)
            if pfile is None:
                still_missing.append({"parent": parent, "reason": "no_pdb_for_construct"})
                continue

            pseq, coords, plddt, valid = parse_backbone(pfile)
            off, mis = align_seq(pseq, wt_seq)
            if off is None:
                still_missing.append({"parent": parent, "reason": "alignment_failed"})
                continue

            L = len(wt_seq)
            key = hashlib.sha1(wt_seq.encode()).hexdigest()
            geom_arrays[f"{key}__coords"] = coords[off : off + L]
            geom_arrays[f"{key}__plddt"] = plddt[off : off + L]
            geom_arrays[f"{key}__valid"] = valid[off : off + L]
            aligned_pdb_by_parent[parent] = pfile

            ms_aligned_new_rule += 1
            mismatch_hist[str(mis)] = mismatch_hist.get(str(mis), 0) + 1
            if matched_old:
                ms_aligned_old_rule += 1
            else:
                recovered_constructs.append({
                    "parent": parent,
                    "pdb_file": pfile.name,
                    "seq_len": L,
                    "mismatches": int(mis),
                    "sha1": key,
                })

            # True C-beta 8.0 A contact map (exact parse_pdb_cb contract)
            if key not in seen_contact_keys:
                cb_seq, cb_coords = parse_pdb_cb(pfile)
                cb_off, _ = align_seq(cb_seq, wt_seq)
                if cb_off is not None and cb_off + L <= len(cb_coords):
                    sub = cb_coords[cb_off : cb_off + L]
                    diff = sub[:, None, :] - sub[None, :, :]
                    dist = np.linalg.norm(diff, axis=-1)
                    cmap = (dist < 8.0)
                    np.fill_diagonal(cmap, False)
                    buf = io.BytesIO()
                    np.save(buf, cmap.astype(np.bool_))
                    contact_blobs.append((f"{key}.npy", buf.getvalue()))
                    seen_contact_keys.add(key)

        # 2. FireProtDB geometry
        fp_aligned = 0
        fp_missing = 0
        for row in fp[[par_col, seq_col]].drop_duplicates().itertuples(index=False):
            acc = str(row[0])
            seq = str(row[1]).upper()
            pfile = resolve_af_pdb(acc, af_pdbs)
            if pfile is None:
                fp_missing += 1
                continue
            pseq, coords, plddt, valid = parse_backbone(pfile)
            off, _ = align_seq(pseq, seq)
            if off is None:
                fp_missing += 1
                continue
            key = hashlib.sha1(seq.encode()).hexdigest()
            L = len(seq)
            geom_arrays[f"{key}__coords"] = coords[off : off + L]
            geom_arrays[f"{key}__plddt"] = plddt[off : off + L]
            geom_arrays[f"{key}__valid"] = valid[off : off + L]
            fp_aligned += 1

        geom_path = WORK / "invariant_backbone_geometry.npz"
        np.savez_compressed(geom_path, **geom_arrays)

        contacts_tar = WORK / "true_contacts_megascale.tar.gz"
        write_deterministic_targz(contact_blobs, contacts_tar)

        summary["geometry_upgrade"] = {
            "megascale_total_constructs": int(len(pairs)),
            "megascale_aligned_before_fix": int(ms_aligned_old_rule),
            "megascale_aligned_after_fix": int(ms_aligned_new_rule),
            "megascale_recovered_count": len(recovered_constructs),
            "megascale_coverage_pct": round(100.0 * ms_aligned_new_rule / max(len(pairs), 1), 2),
            "megascale_still_missing": still_missing,
            "megascale_mismatch_hist": mismatch_hist,
            "fireprot_aligned": int(fp_aligned),
            "fireprot_missing": int(fp_missing),
            "total_unique_sha1_in_npz": len({k.split("__", 1)[0] for k in geom_arrays}),
            "true_contacts_written": len(contact_blobs),
            "recovered_constructs": recovered_constructs,
        }
        summary["outputs"][geom_path.name] = {
            "bytes": geom_path.stat().st_size,
            "sha256": sha256_file(geom_path),
        }
        summary["outputs"][contacts_tar.name] = {
            "bytes": contacts_tar.stat().st_size,
            "sha256": sha256_file(contacts_tar),
        }
        print(
            f"[job9] Geometry upgraded: MegaScale {ms_aligned_old_rule} -> {ms_aligned_new_rule}/"
            f"{len(pairs)} (+{len(recovered_constructs)} recovered), FireProtDB={fp_aligned}, "
            f"contacts={len(contact_blobs)}",
            flush=True,
        )

        # 3. TM-align audit of recovered constructs across all 3 split tiers
        try:
            tm_bin = compile_tmalign()
            if not tm_bin:
                raise RuntimeError("Could not download/compile TMalign binary")
            rec_parents = [r["parent"] for r in recovered_constructs]
            all_parents = sorted(aligned_pdb_by_parent.keys())
            if rec_parents:
                tasks = [
                    (rp, ap, str(aligned_pdb_by_parent[rp]), str(aligned_pdb_by_parent[ap]))
                    for rp in rec_parents
                    for ap in all_parents
                    if rp != ap
                ]
                print(
                    f"[job9] Running TM-align for {len(rec_parents)} recovered x "
                    f"{len(all_parents) - 1} targets = {len(tasks)} pairs ...",
                    flush=True,
                )
                tm_records: list[dict[str, object]] = []

                def _worker(item):
                    rp, ap, pa, pb = item
                    tmax, tmin = run_tm_pair(tm_bin, pa, pb)
                    return {"query": rp, "target": ap, "tm_max": tmax, "tm_min": tmin}

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    for res in ex.map(_worker, tasks):
                        tm_records.append(res)

                tm_df = pd.DataFrame(tm_records)
                tm_out = WORK / "tm_align_recovered_pairs.parquet"
                tm_df.to_parquet(tm_out, index=False)
                summary["outputs"][tm_out.name] = {
                    "bytes": tm_out.stat().st_size,
                    "sha256": sha256_file(tm_out),
                }

                # Summarize cross-split nearest-train TM_max / TM_min for recovered domains
                split_audit = {}
                for sname, fname in (
                    ("id30", "splits_megascale.parquet"),
                    ("id25", "splits_megascale_id25.parquet"),
                    ("fold_tm60", "splits_megascale_fold_tm60.parquet"),
                ):
                    sp_p = find_input(fname)
                    if not sp_p:
                        continue
                    sp_df = pd.read_parquet(sp_p)
                    if "parent" in sp_df.columns:
                        p2s = dict(zip(sp_df["parent"].astype(str), sp_df["split"].astype(str)))
                    else:
                        seq2s = dict(zip(sp_df["parent_seq"].astype(str).str.upper(), sp_df["split"].astype(str)))
                        p2s = {
                            str(r.parent): seq2s.get(str(r.wt_seq).upper(), "")
                            for r in pairs.itertuples(index=False)
                        }
                    rec_split_counts = {}
                    for rp in rec_parents:
                        s = p2s.get(rp, "unknown")
                        rec_split_counts[s] = rec_split_counts.get(s, 0) + 1
                    test_recs = [rp for rp in rec_parents if p2s.get(rp) == "test"]
                    train_set = {p for p, s in p2s.items() if s == "train"}
                    sub_tm = tm_df[tm_df["query"].isin(test_recs) & tm_df["target"].isin(train_set)]
                    if len(sub_tm) > 0:
                        per_q = sub_tm.groupby("query").agg(
                            max_tm_max=("tm_max", "max"),
                            max_tm_min=("tm_min", "max"),
                        )
                        split_audit[sname] = {
                            "recovered_by_split": rec_split_counts,
                            "recovered_in_test": len(test_recs),
                            "max_cross_tm_max": round(float(per_q["max_tm_max"].max()), 4),
                            "max_cross_tm_min": round(float(per_q["max_tm_min"].max()), 4),
                            "n_test_tm_max_ge_05": int((per_q["max_tm_max"] >= 0.5).sum()),
                            "n_test_tm_min_ge_06": int((per_q["max_tm_min"] >= 0.6).sum()),
                        }
                    else:
                        split_audit[sname] = {
                            "recovered_by_split": rec_split_counts,
                            "recovered_in_test": len(test_recs),
                        }
                summary["tm_align_recovered_audit"] = {
                    "pairs_evaluated": len(tm_records),
                    "split_audit": split_audit,
                }
        except Exception as exc:
            record_error("tm_align_recovered_audit", exc)

    except Exception as exc:
        record_error("geometry_and_contacts", exc)

    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["status"] = "ok" if not errors else "completed_with_errors"
    (WORK / "job9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
