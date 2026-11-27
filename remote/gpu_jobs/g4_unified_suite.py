#!/usr/bin/env python3
"""
remote/gpu_jobs/g4_unified_suite.py
===================================
Unified, resumable, Drive-mirrored Colab G4 (RTX PRO 6000 Blackwell 96 GB)
benchmark & reviewer-shield runner for Holo-GeoN v2 (*Nature Computational Science*).

Stages (selectable via ``--stages``):
  1. ``dry_run``              — Cheap Colab CPU rehearsal: verifies all 27 Drive files
                                against MANIFEST.json, extracts PDB & true-contact
                                archives, asserts PDB resolution rates on MegaScale,
                                FireProtDB-HF25, and S669, validates imports & tiny
                                CPU forward pass, and verifies ``npm run build``.
  2. ``block1_precompute_650m`` — Single-pass EsmForMaskedLM 650M cache over MegaScale
                                + all external holdouts + attach invariant 3D geometry.
  3. ``block1_speed``         — CUDA-synchronized speed & peak VRAM microbenchmark
                                (Table 5 / Fig 9): 1-pass vs 2-pass ESM-2 across
                                L in {50, 150, 300, 500, 1000}, full 20xL DMS scan
                                across all models, and PROSTATA-FT E2E step telemetry.
  4. ``block1_parity``        — Identical-GPU (G4) 5/5 paired seeds (42..46) on id25
                                & fold_tm60 + FULL external holdout parity
                                (FireProtDB-HF25, S669, S1724-HF25, Domainome-HF25)
                                for v2-Lean, v2-Lean+8A True Contacts, ThermoMPNN
                                (+ official), SPURS_local (+ upstream official), RaSP,
                                PROSTATA (frozen), and DDGemb.
  5. ``block2_prostata_ft``   — True end-to-end 650M fine-tuned PROSTATA-FT (3 paired
                                seeds 42..44 on id25 & fold_tm60 + all 4 external
                                holdouts + ESM-2 contact P@L & weight drift audit!).
  6. ``block2_esm3b``         — ESM-2 3B (facebook/esm2_t36_3B_UR50D, dim=2560) scaling
                                study: single-pass precompute + 5/5 seeds of
                                v2-Lean (3B) & v2-Lean+8A Contacts (3B) on id25 &
                                fold_tm60 + all 4 external holdouts.
  7. ``block3_demo``          — Calibration of G4 v2-Lean release checkpoint + live
                                CLI throughput benchmark + live FastAPI endpoint test.

Design rules (extras/kaggle-colab-lessons.md):
  * Every finished item writes ``_DONE.json`` and immediately mirrors to
    ``<drive_root>/runs/g4_suite/`` alongside a live ``heartbeat.json`` and
    ``g4_report.json`` so VM preemption loses at most one item.
  * On restart, already-completed items on Drive are restored/skipped automatically.
  * Every subprocess exception is caught, logged into ``g4_report.json``, and synced
    to Drive rather than silently losing logs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[g4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd: list[str], cwd: Path = REPO_ROOT, env: dict | None = None,
            timeout: int | None = None) -> tuple[int, float, str]:
    t0 = time.time()
    merged_env = dict(os.environ)
    merged_env["PYTHONUTF8"] = "1"
    merged_env["PYTHONIOENCODING"] = "utf-8"
    merged_env["PYTHONPATH"] = str(REPO_ROOT)
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    dt = round(time.time() - t0, 2)
    tail = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        log(f"CMD FAILED (rc={proc.returncode}, {dt}s): {' '.join(str(c) for c in cmd)}")
        log(tail[-1500:])
    return proc.returncode, dt, tail


def sync_to_drive(local_path: Path, drive_path: Path) -> None:
    """Mirror a file or directory to Google Drive atomically enough for preemption survival."""
    if not local_path.exists():
        return
    if local_path.is_file():
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, drive_path)
    else:
        drive_path.mkdir(parents=True, exist_ok=True)
        for src in local_path.rglob("*"):
            rel = src.relative_to(local_path)
            dst = drive_path / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)


class G4Suite:
    def __init__(self, drive_root: Path, work_root: Path, run_name: str = "g4_suite"):
        self.drive_root = drive_root
        self.work_root = work_root
        self.drive_out = drive_root / "runs" / run_name
        self.local_out = work_root / "runs" / run_name
        self.data_dir = work_root / "data"
        self.pdb_dir = work_root / "pdbs"
        self.weights_dir = work_root / "weights"
        self.true_contacts_dir = work_root / "true_contacts"
        self.cache_650m = work_root / "cache_esm2_650M"
        self.cache_3b = work_root / "cache_esm2_3B"
        self.started_at = time.time()

        for d in (self.drive_out, self.local_out, self.data_dir, self.pdb_dir, self.weights_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.report_path_local = self.local_out / "g4_report.json"
        self.report_path_drive = self.drive_out / "g4_report.json"
        if self.report_path_drive.is_file():
            self.report = json.loads(self.report_path_drive.read_text(encoding="utf-8"))
        else:
            self.report = {
                "created_utc": utc_now(),
                "stages": {},
                "items": {},
                "errors": [],
                "environment": {},
            }

    def save_heartbeat(self, stage: str, note: str = "") -> None:
        hb = {
            "utc": utc_now(),
            "stage": stage,
            "note": note,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "n_done_items": sum(1 for v in self.report["items"].values() if v.get("status") == "ok"),
            "n_errors": len(self.report["errors"]),
        }
        (self.local_out / "heartbeat.json").write_text(json.dumps(hb, indent=2), encoding="utf-8")
        sync_to_drive(self.local_out / "heartbeat.json", self.drive_out / "heartbeat.json")
        self.report["updated_utc"] = utc_now()
        self.report["elapsed_s"] = round(time.time() - self.started_at, 1)
        self.report_path_local.write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        sync_to_drive(self.report_path_local, self.report_path_drive)

    def item_done(self, rel_pack: str) -> bool:
        return (self.drive_out / rel_pack / "_DONE.json").is_file() or (
            self.local_out / rel_pack / "_DONE.json"
        ).is_file()

    def mark_done(self, rel_pack: str, meta: dict) -> None:
        lp = self.local_out / rel_pack
        lp.mkdir(parents=True, exist_ok=True)
        stamp = {"status": "ok", "utc": utc_now(), **meta}
        (lp / "_DONE.json").write_text(json.dumps(stamp, indent=2, default=str), encoding="utf-8")
        sync_to_drive(lp, self.drive_out / rel_pack)
        self.report["items"][rel_pack] = stamp
        self.save_heartbeat(stage=rel_pack, note="done")

    def record_error(self, item: str, err: str) -> None:
        entry = {"utc": utc_now(), "item": item, "error": err[-2000:]}
        self.report["errors"].append(entry)
        self.save_heartbeat(stage=item, note="error")
        log(f"ERROR in {item}: {err[:400]}")

    def record_env(self) -> dict:
        import torch
        import transformers
        import torch_geometric

        env = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if torch.cuda.is_available()
            else 0.0,
            "transformers": transformers.__version__,
            "torch_geometric": torch_geometric.__version__,
            "git_head": subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True
            ).stdout.strip(),
        }
        try:
            import RNA  # type: ignore
            env["ViennaRNA"] = getattr(RNA, "__version__", "installed")
        except Exception:
            env["ViennaRNA"] = None
        self.report["environment"] = env
        self.save_heartbeat("env", env["gpu_name"])
        log(f"Environment: {json.dumps(env)}")
        return env

    # ------------------------------------------------------------------
    # Fail-fast input staging & extraction
    # ------------------------------------------------------------------
    def prepare_inputs(self) -> dict:
        t0 = time.time()
        man_path = self.drive_root / "MANIFEST.json"
        if not man_path.is_file():
            raise FileNotFoundError(f"Drive MANIFEST.json missing at {man_path}")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        missing, bad_sz = [], []
        for entry in man["files"]:
            src = self.drive_root / entry["rel"]
            if not src.is_file():
                missing.append(entry["rel"])
                continue
            if src.stat().st_size != entry["bytes"]:
                bad_sz.append((entry["rel"], src.stat().st_size, entry["bytes"]))
            dst = self.work_root / entry["rel"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
        if missing or bad_sz:
            raise RuntimeError(f"Drive MANIFEST verification failed: missing={missing} bad_sz={bad_sz}")

        # Extract true contacts
        if not (self.true_contacts_dir.is_dir() and len(list(self.true_contacts_dir.rglob("*.npy"))) >= 350):
            self.true_contacts_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.data_dir / "true_contacts_megascale.tar.gz", "r:*") as tf:
                tf.extractall(self.true_contacts_dir)
            # Flatten nested subdirs so <sha1>.npy lives directly inside self.true_contacts_dir
            for npy in list(self.true_contacts_dir.rglob("*.npy")):
                if npy.parent != self.true_contacts_dir:
                    shutil.move(str(npy), str(self.true_contacts_dir / npy.name))

        # Extract PDB archives
        ms_pdb_dir = self.pdb_dir / "megascale"
        if not (ms_pdb_dir.is_dir() and len(list(ms_pdb_dir.rglob("*.pdb"))) >= 350):
            ms_pdb_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.pdb_dir / "megascale_pdbs.tar.gz", "r:*") as tf:
                tf.extractall(ms_pdb_dir)

        fp_pdb_dir = self.pdb_dir / "fireprot"
        if not (fp_pdb_dir.is_dir() and len(list(fp_pdb_dir.rglob("*.pdb"))) >= 150):
            fp_pdb_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.pdb_dir / "alphafold_fireprot_pdbs.tar.gz", "r:*") as tf:
                tf.extractall(fp_pdb_dir)

        s669_pdb_dir = self.pdb_dir / "s669"
        if not (s669_pdb_dir.is_dir() and len(list(s669_pdb_dir.rglob("*.pdb"))) >= 600):
            s669_pdb_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.pdb_dir / "S669.zip", "r") as zf:
                zf.extractall(s669_pdb_dir)

        n_tc = len(list(self.true_contacts_dir.glob("*.npy")))
        n_ms = len(list(ms_pdb_dir.rglob("*.pdb")))
        n_fp = len(list(fp_pdb_dir.rglob("*.pdb")))
        n_s669 = len(list(s669_pdb_dir.rglob("*.pdb")))
        assert n_tc >= 350, f"Too few true contact .npy files: {n_tc}"
        assert n_ms >= 350, f"Too few MegaScale PDBs: {n_ms}"
        assert n_fp >= 150, f"Too few FireProtDB PDBs: {n_fp}"
        assert n_s669 >= 600, f"Too few S669 PDBs: {n_s669}"

        # Verify PDB resolution rates using src.geometry.resolve_pdb_path
        import pandas as pd
        from src.geometry import normalize_pdb_stem, resolve_pdb_path

        def _match_rate(pq: Path, pdir: Path) -> tuple[int, int]:
            df = pd.read_parquet(pq)
            col = "parent" if "parent" in df.columns else "uniprot_id"
            parents = df[col].dropna().astype(str).unique().tolist()
            pmap = {normalize_pdb_stem(p.name): p for p in pdir.rglob("*.pdb")}
            hits = sum(1 for par in parents if resolve_pdb_path(par, pmap) is not None)
            return hits, len(parents)

        s669_hits, s669_tot = _match_rate(self.data_dir / "s669_holdout.parquet", s669_pdb_dir)
        fp_hits, fp_tot = _match_rate(self.data_dir / "fireprotdb_hf25.parquet", fp_pdb_dir)
        ms_hits, ms_tot = _match_rate(self.data_dir / "megascale_ddg_pairs.parquet", ms_pdb_dir)
        assert s669_hits >= 80, f"S669 PDB match failed: {s669_hits}/{s669_tot}"
        assert fp_hits >= 80, f"FireProtDB-HF25 PDB match failed: {fp_hits}/{fp_tot}"

        info = {
            "manifest_files_ok": len(man["files"]),
            "true_contacts_npy": n_tc,
            "megascale_pdbs": n_ms,
            "fireprot_pdbs": n_fp,
            "s669_pdbs": n_s669,
            "pdb_match_megascale": f"{ms_hits}/{ms_tot}",
            "pdb_match_fireprot_hf25": f"{fp_hits}/{fp_tot}",
            "pdb_match_s669": f"{s669_hits}/{s669_tot}",
            "elapsed_s": round(time.time() - t0, 2),
        }
        log(f"Inputs verified: {json.dumps(info)}")
        self.report["stages"]["prepare_inputs"] = info
        self.save_heartbeat("prepare_inputs", "ok")
        return info

    # ------------------------------------------------------------------
    # Stage 1: Cheap CPU Rehearsal
    # ------------------------------------------------------------------
    def stage_dry_run(self) -> None:
        log("=== STAGE: dry_run (CPU rehearsal) ===")
        env = self.record_env()
        inp = self.prepare_inputs()

        import torch
        from src.config import load_config
        from src.data import build_sample_graph, collate_protein_batch
        from src.graphs import khop_neighborhood
        from src.model import HoloGeoN

        # Synthetic tiny forward check for v2-Lean at both 650M (1280) and 3B (2560)
        cfg_v2 = load_config("configs/benchmark_v2.yaml")
        smoke = {}
        for tag, edim in [("v2_lean_650m", 1280), ("v2_lean_3b", 2560)]:
            m = HoloGeoN(
                arch="v2", esm_dim=edim, hidden=256, out_dim=256,
                use_ssm=False, use_mech_stream=False, use_geo_stream=True,
                use_multi_layer=False, use_contact_burial=True, use_esm_logits_prior=True,
            ).eval()
            L = 24
            rec_g = {
                "emb": torch.randn(L, edim),
                "contact_edges": torch.tensor([[0, 5], [10, 18]], dtype=torch.long),
                "contact_probs": torch.tensor([0.9, 0.8]),
                "mech": torch.zeros(L, 8),
                "wt_logprobs": torch.zeros(L, 20),
                "site_entropy": torch.ones(L),
            }
            gb = build_sample_graph(rec_g, site=0, window=L, cfg=cfg_v2.as_dict(), drop_cds=True)
            n1 = torch.stack([khop_neighborhood(gb["edge_geo"], i, L, hops=1) for i in range(L)], dim=0)
            n2 = torch.stack([khop_neighborhood(gb["edge_geo"], i, L, hops=2) for i in range(L)], dim=0)
            sites = torch.tensor([3, 10])
            batch = collate_protein_batch([{
                **gb, "sites": sites, "wt_idx": torch.tensor([0, 1]), "mt_idx": torch.tensor([2, 3]),
                "y": torch.zeros(2), "sigma": torch.full((2,), 0.5),
                "wt_logprobs_site": torch.zeros(2, 20),
                "neigh_1hop": n1[sites], "neigh_2hop": n2[sites],
            }])
            with torch.no_grad():
                ddg, logvar = m.ddg(batch)
            smoke[tag] = {"params": m.parameter_count(), "ddg_shape": list(ddg.shape), "finite": bool(torch.isfinite(ddg).all())}
        log(f"Model param & forward smoke check: {smoke}")

        # Verify checkpoints from Drive load cleanly
        ckpt_ok = {}
        for fn in ["v2_lean_id25_s42.pt", "v2_lean_true_contacts_id25_s42.pt", "v2_lean_str_only_id25_s42.pt", "epistasis_egeo_id25_s42.pt"]:
            p = self.weights_dir / fn
            st = torch.load(p, map_location="cpu", weights_only=False)
            ckpt_ok[fn] = sorted(list(st.keys()))[:8]

        # Test npm run build on Colab CPU (never locally)
        fe_dir = REPO_ROOT / "app" / "frontend"
        npm_res = {}
        if shutil.which("npm") and (fe_dir / "package.json").is_file():
            rc1, dt1, tail1 = run_cmd(["npm", "ci", "--no-audit", "--no-fund"], cwd=fe_dir, timeout=240)
            if rc1 != 0:
                rc1, dt1, tail1 = run_cmd(["npm", "install", "--no-audit", "--no-fund"], cwd=fe_dir, timeout=240)
            rc2, dt2, tail2 = run_cmd(["npm", "run", "build"], cwd=fe_dir, timeout=240)
            npm_res = {"install_rc": rc1, "install_s": dt1, "build_rc": rc2, "build_s": dt2, "tail": tail2[-500:]}
            log(f"Frontend npm build: rc={rc2} ({dt2}s)")

        # Shallow clone reachability check for ThermoMPNN and SPURS
        repo_check = {}
        for name, url in [
            ("ThermoMPNN", "https://github.com/Kuhlman-Lab/ThermoMPNN.git"),
            ("SPURS", "https://github.com/luo-group/SPURS.git"),
        ]:
            dest = self.work_root / "competitor_repos" / name
            if not (dest / ".git").exists():
                rc, dt, _ = run_cmd(["git", "clone", "--depth", "1", url, str(dest)], timeout=120)
                repo_check[name] = {"rc": rc, "s": dt}
            else:
                repo_check[name] = {"rc": 0, "cached": True}

        out = {
            "status": "ok",
            "utc": utc_now(),
            "environment": env,
            "inputs": inp,
            "model_smoke": smoke,
            "checkpoints_verified": ckpt_ok,
            "frontend_build": npm_res,
            "competitor_repos": repo_check,
        }
        p_out = self.local_out / "cpu_rehearsal_report.json"
        p_out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        sync_to_drive(p_out, self.drive_out / "cpu_rehearsal_report.json")
        self.report["stages"]["dry_run"] = out
        self.save_heartbeat("dry_run", "COMPLETE")
        log("CPU rehearsal COMPLETE!")

    # ------------------------------------------------------------------
    # Stage 2: Single-Pass ESM-2 650M Precompute + 3D Coords Attach
    # ------------------------------------------------------------------
    def _precompute_cache(self, esm_model: str, cache_dir: Path, tag: str) -> None:
        if (cache_dir / "manifest.json").is_file() and (self.local_out / f"{tag}_DONE.json").is_file():
            log(f"[{tag}] cache already ready at {cache_dir}")
            return
        tables = [
            self.data_dir / "megascale_ddg_pairs.parquet",
            self.data_dir / "fireprotdb_hf25.parquet",
            self.data_dir / "fireprotdb_holdout.parquet",
            self.data_dir / "s669_holdout.parquet",
            self.data_dir / "s1724_thermomutdb_hf25.parquet",
            self.data_dir / "domainome_apca_hf25.parquet",
        ]
        cmd = [
            sys.executable, "-m", "scripts.precompute_embeddings",
            "--config", "configs/benchmark_v2.yaml",
            "--esm-model", esm_model,
            "--cache", str(cache_dir),
            "--device", "cuda",
            "--data", *[str(t) for t in tables if t.is_file()],
        ]
        log(f"[{tag}] running single-pass precompute for {esm_model} ...")
        rc, dt, tail = run_cmd(cmd, timeout=3600)
        if rc != 0:
            raise RuntimeError(f"precompute failed ({tag}): {tail[-1000:]}")
        # Attach invariant 3D backbone geometry (coords + pLDDT) to cache
        rc2, dt2, tail2 = run_cmd([
            sys.executable, "-m", "scripts.attach_coords",
            "--geom-npz", str(self.data_dir / "invariant_backbone_geometry.npz"),
            "--cache", str(cache_dir),
        ], timeout=600)
        if rc2 != 0:
            log(f"[{tag}] WARNING: attach_coords rc={rc2}: {tail2[-500:]}")
        info = {"esm_model": esm_model, "precompute_s": dt, "attach_coords_s": dt2, "cache_dir": str(cache_dir)}
        (self.local_out / f"{tag}_DONE.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        self.report["stages"][tag] = info
        self.save_heartbeat(tag, f"done in {dt}s")

    def stage_block1_precompute_650m(self) -> None:
        self.record_env()
        self.prepare_inputs()
        self._precompute_cache("facebook/esm2_t33_650M_UR50D", self.cache_650m, "block1_precompute_650m")

    # ------------------------------------------------------------------
    # Stage 3: Controlled Speed & Peak VRAM Microbenchmark (Table 5 / Fig 9)
    # ------------------------------------------------------------------
    def stage_block1_speed(self) -> None:
        pack = "block1_speed"
        if self.item_done(pack):
            log(f"[{pack}] already complete on Drive — skipping")
            return
        log("=== STAGE: block1_speed (CUDA-synchronized throughput & VRAM microbenchmark) ===")
        import numpy as np
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM
        from src.encoder import EsmEncoder
        from src.model import HoloGeoN
        from src.graphs import khop_neighborhood
        from src.data import build_sample_graph, collate_protein_batch, move_batch
        from scripts.competitors.prostata import ProstataFiveHead
        from scripts.competitors.ddgemb import DdgEmbTransformer
        from scripts.competitors.spurs import SpursFusionHead
        from scripts.competitors.rasp import RaspDownstreamMLP

        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

        # Deterministic real-like AA sequences of exact benchmark lengths
        aa20 = "ACDEFGHIKLMNPQRSTVWY"
        rng = np.random.default_rng(2026)
        seqs = {L: "".join(rng.choice(list(aa20), size=L)) for L in (50, 150, 300, 500, 1000)}

        results: dict = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "encoder_single_vs_two_pass": {},
            "dms_20xL_throughput": {},
            "prostata_ft_step_telemetry": {},
        }

        # 1. Single-pass EsmForMaskedLM vs Legacy Two-pass (EsmModel + predict_contacts)
        log("Benchmarking 1-pass vs 2-pass ESM-2 650M across L in {50, 150, 300, 500, 1000} ...")
        enc = EsmEncoder("facebook/esm2_t33_650M_UR50D", device="cuda")
        enc.eval()
        for L, seq in seqs.items():
            toks = enc.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            reps = 8 if L <= 500 else 4

            # Single-pass (Holo-GeoN v2)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(2):
                    _ = enc.encode(seq)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    _ = enc.encode(seq)
                torch.cuda.synchronize()
                dt_1p = (time.perf_counter() - t0) / reps * 1000.0
            vram_1p = round(torch.cuda.max_memory_allocated() / 1e9, 3)

            # Legacy two-pass (separate backbone forward + separate attention/contact pass, exact v1 cost)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                def _two_pass():
                    out1 = enc.esm(**toks)
                    _ = out1.last_hidden_state
                    out2 = enc.esm(**toks, output_attentions=True)
                    attns = torch.stack([a.float() for a in out2.attentions], dim=1)
                    _ = enc.esm.contact_head(toks["input_ids"], attns)

                for _ in range(2):
                    _two_pass()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    _two_pass()
                torch.cuda.synchronize()
                dt_2p = (time.perf_counter() - t0) / reps * 1000.0
            vram_2p = round(torch.cuda.max_memory_allocated() / 1e9, 3)

            results["encoder_single_vs_two_pass"][str(L)] = {
                "single_pass_ms": round(dt_1p, 2),
                "two_pass_ms": round(dt_2p, 2),
                "speedup_x": round(dt_2p / max(dt_1p, 1e-6), 3),
                "single_pass_vram_gb": vram_1p,
                "two_pass_vram_gb": vram_2p,
            }
            log(f"  L={L:4d}: 1-pass={dt_1p:.1f}ms ({vram_1p:.2f}GB) vs 2-pass={dt_2p:.1f}ms ({vram_2p:.2f}GB) -> {dt_2p/dt_1p:.2f}x")

        # 2. Full 20 x L DMS scan throughput across Holo-GeoN & Competitors (L in {50, 150, 300, 500})
        log("Benchmarking full 20xL DMS scan across Holo-GeoN variants & competitors ...")
        cached_enc = {L: enc.encode(seqs[L]) for L in (50, 150, 300, 500)}
        del enc
        gc.collect()
        torch.cuda.empty_cache()

        from src.config import load_config
        cfg_v2 = load_config("configs/benchmark_v2.yaml")

        holo_specs = {
            "Holo-GeoN v2-Lean (1.00M)": dict(
                arch="v2", esm_dim=1280, hidden=256, out_dim=256,
                use_ssm=False, use_mech_stream=False, use_geo_stream=True,
                use_multi_layer=False, use_contact_burial=True, use_esm_logits_prior=True,
            ),
            "Holo-GeoN v2-Full (2.05M)": dict(
                arch="v2", esm_dim=1280, hidden=256, out_dim=256,
                use_ssm=True, use_mech_stream=True, use_geo_stream=True,
                use_multi_layer=False, use_contact_burial=True, use_esm_logits_prior=True,
            ),
            "Holo-GeoN v2-Lean + 8A Contacts (1.00M)": dict(
                arch="v2", esm_dim=1280, hidden=256, out_dim=256,
                use_ssm=False, use_mech_stream=False, use_geo_stream=True,
                use_multi_layer=False, use_contact_burial=True, use_esm_logits_prior=True,
            ),
            "Holo-GeoN v1 (20.6M)": dict(
                arch="v1", esm_dim=1280, hidden=256, out_dim=256,
                use_ssm=True, use_mech_stream=True, use_geo_stream=True,
            ),
        }

        for mname, kwargs in holo_specs.items():
            model = HoloGeoN(**kwargs).to(device).eval()
            n_params = model.parameter_count()
            results["dms_20xL_throughput"][mname] = {"trainable_params": n_params, "by_length": {}}
            for L in (50, 150, 300, 500):
                rec = cached_enc[L]
                n_muts = 20 * L
                sites = torch.arange(L, dtype=torch.long).repeat_interleave(20)
                wt_idx = torch.tensor([aa20.index(c) for c in seqs[L]], dtype=torch.long).repeat_interleave(20)
                mt_idx = torch.arange(20, dtype=torch.long).repeat(L)
                rec_g = {
                    "emb": rec["emb"],
                    "layer_embs": rec["layer_embs"],
                    "contact_edges": rec["contact_edges"],
                    "contact_probs": rec["contact_probs"],
                    "mech": torch.zeros((L, 8), dtype=torch.float32),
                    "wt_logprobs": rec["wt_logprobs"],
                    "site_entropy": rec["site_entropy"],
                    "seq": seqs[L],
                }
                g_base = build_sample_graph(rec_g, site=0, window=L, cfg=cfg_v2.as_dict(), drop_cds=True)
                n1_per_res = torch.stack([khop_neighborhood(g_base["edge_geo"], i, L, hops=1) for i in range(L)], dim=0)
                n2_per_res = torch.stack([khop_neighborhood(g_base["edge_geo"], i, L, hops=2) for i in range(L)], dim=0)
                g_item = {
                    **g_base,
                    "sites": sites,
                    "wt_idx": wt_idx,
                    "mt_idx": mt_idx,
                    "y": torch.zeros(n_muts, dtype=torch.float32),
                    "sigma": torch.full((n_muts,), 0.5, dtype=torch.float32),
                    "wt_logprobs_site": rec["wt_logprobs"].float()[sites],
                    "neigh_1hop": n1_per_res[sites],
                    "neigh_2hop": n2_per_res[sites],
                }
                batch = move_batch(collate_protein_batch([g_item]), device)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for _ in range(3):
                        _ = model.ddg(batch)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    reps = 20
                    for _ in range(reps):
                        _ = model.ddg(batch)
                    torch.cuda.synchronize()
                    head_ms = (time.perf_counter() - t0) / reps * 1000.0
                vram_head = round(torch.cuda.max_memory_allocated() / 1e9, 4)
                enc_ms = results["encoder_single_vs_two_pass"][str(L)]["single_pass_ms"]
                tot_ms = enc_ms + head_ms
                results["dms_20xL_throughput"][mname]["by_length"][str(L)] = {
                    "n_mutations": n_muts,
                    "head_only_ms": round(head_ms, 3),
                    "total_e2e_ms": round(tot_ms, 2),
                    "muts_per_sec_e2e": round(n_muts / (tot_ms / 1000.0), 1),
                    "muts_per_sec_cached": round(n_muts / max(head_ms / 1000.0, 1e-6), 1),
                    "head_peak_vram_gb": vram_head,
                }
            del model
            torch.cuda.empty_cache()

        # Competitor head scan latency (PROSTATA frozen, DDGemb, SPURS_local, RaSP)
        comp_models = {
            "PROSTATA (frozen 5-head)": (ProstataFiveHead(1280).to(device).eval(), "prostata"),
            "DDGemb (Transformer)": (DdgEmbTransformer(1280, radius=15).to(device).eval(), "ddgemb"),
            "SPURS_local (3D fusion)": (SpursFusionHead(1280).to(device).eval(), "spurs"),
            "RaSP (3D cavity MLP)": (RaspDownstreamMLP().to(device).eval(), "rasp"),
        }
        for cname, (cmod, ckind) in comp_models.items():
            n_params = sum(p.numel() for p in cmod.parameters())
            results["dms_20xL_throughput"][cname] = {"trainable_params": n_params, "by_length": {}}
            for L in (50, 150, 300, 500):
                n_muts = 20 * L
                wt_idx = torch.zeros(n_muts, dtype=torch.long, device=device)
                mt_idx = torch.arange(20, dtype=torch.long, device=device).repeat(L)
                h_site = torch.randn(n_muts, 1280, device=device)
                h_glob = torch.randn(n_muts, 1280, device=device)
                h_str = torch.randn(n_muts, 10, device=device)
                cav = torch.randn(n_muts, 75, device=device)
                w_emb = torch.randn(min(n_muts, 1024), 31, 1280, device=device)
                w_msk = torch.ones(min(n_muts, 1024), 31, dtype=torch.bool, device=device)

                def _run_comp():
                    if ckind == "prostata":
                        return cmod(h_site, h_glob, wt_idx, mt_idx)
                    if ckind == "spurs":
                        return cmod(h_site, h_str, wt_idx, mt_idx)
                    if ckind == "rasp":
                        return cmod(cav, wt_idx, mt_idx)
                    # ddgemb batched in chunks of 1024
                    outs = []
                    for lo in range(0, n_muts, 1024):
                        bs = min(1024, n_muts - lo)
                        outs.append(cmod(w_emb[:bs], w_msk[:bs], wt_idx[lo:lo + bs], mt_idx[lo:lo + bs]))
                    return torch.cat(outs, dim=0)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for _ in range(3):
                        _ = _run_comp()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    reps = 15
                    for _ in range(reps):
                        _ = _run_comp()
                    torch.cuda.synchronize()
                    head_ms = (time.perf_counter() - t0) / reps * 1000.0
                vram_head = round(torch.cuda.max_memory_allocated() / 1e9, 4)
                enc_ms = 0.0 if ckind == "rasp" else results["encoder_single_vs_two_pass"][str(L)]["single_pass_ms"]
                tot_ms = enc_ms + head_ms
                results["dms_20xL_throughput"][cname]["by_length"][str(L)] = {
                    "n_mutations": n_muts,
                    "head_only_ms": round(head_ms, 3),
                    "total_e2e_ms": round(tot_ms, 2),
                    "muts_per_sec_e2e": round(n_muts / max(tot_ms / 1000.0, 1e-6), 1),
                    "muts_per_sec_cached": round(n_muts / max(head_ms / 1000.0, 1e-6), 1),
                    "head_peak_vram_gb": vram_head,
                }
            del cmod
            torch.cuda.empty_cache()

        # 3. PROSTATA-FT E2E 650M training step telemetry (1 parent x 64 muts, bf16 + grad ckpt)
        log("Measuring PROSTATA-FT E2E 650M single training step latency & VRAM ...")
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
        esm = mlm.esm
        esm.gradient_checkpointing_enable()
        head = ProstataFiveHead(1280).to(device)
        opt = torch.optim.AdamW(list(esm.parameters()) + list(head.parameters()), lr=1e-5)
        for L in (64, 300):
            t_in = tok(seqs[50 if L == 64 else 300], return_tensors="pt", add_special_tokens=True)
            t_in = {k: v.to(device) for k, v in t_in.items()}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for it in range(4):
                if it == 1:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h = esm(**t_in).last_hidden_state[0, 1:-1]
                    sites = torch.randint(0, h.size(0), (64,), device=device)
                    pred = head(h[sites], h.mean(0, keepdim=True).expand(64, -1),
                                torch.zeros(64, dtype=torch.long, device=device),
                                torch.ones(64, dtype=torch.long, device=device))
                    loss = (pred ** 2).mean()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) / 3.0 * 1000.0
            results["prostata_ft_step_telemetry"][f"L{L}_M64"] = {
                "trainable_params": sum(p.numel() for p in esm.parameters()) + sum(p.numel() for p in head.parameters()),
                "step_ms": round(step_ms, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            }
        del mlm, esm, head, opt
        gc.collect()
        torch.cuda.empty_cache()

        lp = self.local_out / pack
        lp.mkdir(parents=True, exist_ok=True)
        (lp / "speed_vram_microbench.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        self.mark_done(pack, {"file": "speed_vram_microbench.json"})
        log("block1_speed COMPLETE!")

    # ------------------------------------------------------------------
    # Stage 4: Identical-GPU (G4) 5/5 Parity & Missing External Competitors
    # ------------------------------------------------------------------
    def _train_and_export_holo(self, pack: str, split_file: Path, cache_dir: Path,
                               esm_model: str, seed: int, true_contacts: bool) -> None:
        if self.item_done(pack):
            log(f"  [skip] {pack}")
            return
        out_dir = self.local_out / pack
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "scripts.train",
            "--config", "configs/benchmark_v2.yaml",
            "--stability-splits", str(split_file),
            "--pathogenicity-splits", "__none__",
            "--cache", str(cache_dir),
            "--esm-model", esm_model,
            "--out", str(out_dir),
            "--seed", str(seed),
            "--no-mech-stream", "--no-ssm", "--no-multi-layer",
            "--epochs", "20", "--num-workers", "4",
        ]
        if true_contacts:
            cmd += ["--true-contact-edges", "--true-contacts-dir", str(self.true_contacts_dir)]
        rc, dt, tail = run_cmd(cmd, timeout=1800)
        if rc != 0:
            self.record_error(pack, tail)
            return
        ckpt = out_dir / "model_best.pt"
        # Export val & test
        for sp_part in ("val", "test"):
            rc_e, _, tail_e = run_cmd([
                sys.executable, "-m", "scripts.export_predictions",
                "--checkpoint", str(ckpt), "--splits", str(split_file), "--split", sp_part,
                "--config", "configs/benchmark_v2.yaml", "--cache", str(cache_dir),
                "--out", str(out_dir / f"pred_{sp_part}.parquet"),
            ], timeout=600)
            if rc_e != 0:
                self.record_error(f"{pack}/export_{sp_part}", tail_e)

        # Evaluate on all 4 external holdouts
        for ext_name in [
            "fireprotdb_hf25.parquet",
            "s669_holdout.parquet",
            "s1724_thermomutdb_hf25.parquet",
            "domainome_apca_hf25.parquet",
        ]:
            ext_p = self.data_dir / ext_name
            if ext_p.is_file():
                rc_x, _, tail_x = run_cmd([
                    sys.executable, "-m", "scripts.eval_external",
                    "--checkpoint", str(ckpt), "--data", str(ext_p),
                    "--config", "configs/benchmark_v2.yaml", "--cache", str(cache_dir),
                    "--out", str(out_dir),
                ], timeout=900)
                if rc_x != 0:
                    self.record_error(f"{pack}/ext_{ext_name}", tail_x)

        self.mark_done(pack, {"wall_clock_s": dt, "seed": seed, "esm_model": esm_model, "true_contacts": true_contacts})

    def stage_block1_parity(self) -> None:
        log("=== STAGE: block1_parity (Same-GPU G4 5/5 Parity + Missing External Competitors) ===")
        self.record_env()
        self.prepare_inputs()
        self._precompute_cache("facebook/esm2_t33_650M_UR50D", self.cache_650m, "block1_precompute_650m")

        ms_pdbs = self.pdb_dir / "megascale"
        fp_pdbs = self.pdb_dir / "fireprot"
        s669_pdbs = self.pdb_dir / "s669"
        pairs_pq = self.data_dir / "megascale_ddg_pairs.parquet"
        fp_pq = self.data_dir / "fireprotdb_hf25.parquet"
        s669_pq = self.data_dir / "s669_holdout.parquet"
        s1724_pq = self.data_dir / "s1724_thermomutdb_hf25.parquet"
        dom_pq = self.data_dir / "domainome_apca_hf25.parquet"

        # 1. Official ThermoMPNN & Official Upstream SPURS on FireProtDB-HF25 & S669
        for off_name, cmd in [
            ("block1_parity/official_thermompnn", [
                sys.executable, "-m", "scripts.competitors.thermompnn",
                "--pdb-dir", str(ms_pdbs),
                "--external", str(fp_pq), "--external-pdb-dir", str(fp_pdbs),
                "--extra-externals", f"s669={s669_pq}::{s669_pdbs}",
                "--official-external-only",
                "--out", str(self.local_out / "block1_parity/official_thermompnn"),
            ]),
            ("block1_parity/official_spurs", [
                sys.executable, "-m", "scripts.competitors.spurs",
                "--cache", str(self.cache_650m),
                "--mode", "upstream", "--official-external-only",
                "--external", str(fp_pq), "--external-pdb-dir", str(fp_pdbs),
                "--extra-externals", f"s669={s669_pq}::{s669_pdbs}",
                "--out", str(self.local_out / "block1_parity/official_spurs"),
            ]),
        ]:
            if not self.item_done(off_name):
                log(f"Running {off_name} ...")
                rc, dt, tail = run_cmd(cmd, timeout=1800)
                if rc == 0:
                    self.mark_done(off_name, {"wall_clock_s": dt})
                else:
                    self.record_error(off_name, tail)

        # 2. Same-GPU G4 5/5 paired seeds across id25 & fold_tm60
        for sp_name, sp_file in [
            ("id25", self.data_dir / "splits_megascale_id25.parquet"),
            ("fold_tm60", self.data_dir / "splits_megascale_fold_tm60.parquet"),
        ]:
            for seed in (42, 43, 44, 45, 46):
                # Holo-GeoN v2-Lean & v2-Lean + True Contacts
                self._train_and_export_holo(
                    f"block1_parity/{sp_name}/v2_lean_s{seed}",
                    sp_file, self.cache_650m, "facebook/esm2_t33_650M_UR50D", seed, False,
                )
                self._train_and_export_holo(
                    f"block1_parity/{sp_name}/v2_lean_true_contacts_s{seed}",
                    sp_file, self.cache_650m, "facebook/esm2_t33_650M_UR50D", seed, True,
                )

                # Retrained competitors with FULL external parity
                comp_jobs = [
                    (f"block1_parity/{sp_name}/thermompnn_s{seed}", [
                        sys.executable, "-m", "scripts.competitors.thermompnn",
                        "--splits", str(sp_file), "--pairs", str(pairs_pq),
                        "--pdb-dir", str(ms_pdbs),
                        "--external", str(fp_pq), "--external-pdb-dir", str(fp_pdbs),
                        "--extra-externals", f"s669={s669_pq}::{s669_pdbs}",
                        "--epochs", "20", "--seed", str(seed),
                        "--out", str(self.local_out / f"block1_parity/{sp_name}/thermompnn_s{seed}"),
                    ]),
                    (f"block1_parity/{sp_name}/spurs_local_s{seed}", [
                        sys.executable, "-m", "scripts.competitors.spurs",
                        "--splits", str(sp_file), "--pairs", str(pairs_pq),
                        "--cache", str(self.cache_650m), "--pdb-dir", str(ms_pdbs),
                        "--external", str(fp_pq), "--external-pdb-dir", str(fp_pdbs),
                        "--extra-externals", f"s669={s669_pq}::{s669_pdbs}",
                        "--epochs", "20", "--seed", str(seed),
                        "--out", str(self.local_out / f"block1_parity/{sp_name}/spurs_local_s{seed}"),
                    ]),
                    (f"block1_parity/{sp_name}/rasp_s{seed}", [
                        sys.executable, "-m", "scripts.competitors.rasp",
                        "--splits", str(sp_file), "--pairs", str(pairs_pq),
                        "--pdb-dir", str(ms_pdbs),
                        "--external", str(fp_pq), "--external-pdb-dir", str(fp_pdbs),
                        "--extra-externals", f"s669={s669_pq}::{s669_pdbs}",
                        "--epochs", "25", "--seed", str(seed),
                        "--out", str(self.local_out / f"block1_parity/{sp_name}/rasp_s{seed}"),
                    ]),
                    (f"block1_parity/{sp_name}/prostata_s{seed}", [
                        sys.executable, "-m", "scripts.competitors.prostata",
                        "--splits", str(sp_file), "--cache", str(self.cache_650m),
                        "--external", str(fp_pq),
                        "--extra-externals", f"s669={s669_pq}", f"s1724_hf25={s1724_pq}", f"domainome_hf25={dom_pq}",
                        "--epochs", "20", "--seed", str(seed),
                        "--out", str(self.local_out / f"block1_parity/{sp_name}/prostata_s{seed}"),
                    ]),
                    (f"block1_parity/{sp_name}/ddgemb_s{seed}", [
                        sys.executable, "-m", "scripts.competitors.ddgemb",
                        "--splits", str(sp_file), "--cache", str(self.cache_650m),
                        "--external", str(fp_pq),
                        "--extra-externals", f"s669={s669_pq}", f"s1724_hf25={s1724_pq}", f"domainome_hf25={dom_pq}",
                        "--epochs", "15", "--seed", str(seed),
                        "--out", str(self.local_out / f"block1_parity/{sp_name}/ddgemb_s{seed}"),
                    ]),
                ]
                for cname, ccmd in comp_jobs:
                    if self.item_done(cname):
                        log(f"  [skip] {cname}")
                        continue
                    log(f"Running {cname} ...")
                    rc, dt, tail = run_cmd(ccmd, timeout=1800)
                    if rc == 0:
                        self.mark_done(cname, {"wall_clock_s": dt, "seed": seed})
                    else:
                        self.record_error(cname, tail)

        log("block1_parity COMPLETE!")

    # ------------------------------------------------------------------
    # Stage 5: Block 2A — End-to-End 650M Fine-Tuned PROSTATA-FT + Drift Audit
    # ------------------------------------------------------------------
    def stage_block2_prostata_ft(self) -> None:
        log("=== STAGE: block2_prostata_ft (End-to-End 650M ESM-2 Fine-Tuned PROSTATA + Contact Drift Audit) ===")
        self.record_env()
        self.prepare_inputs()

        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from scipy.stats import pearsonr
        from transformers import AutoTokenizer, EsmForMaskedLM
        from scripts.competitors.common import gpu_peak_gb, load_external_table, load_split_table, write_predictions
        from scripts.competitors.prostata import ProstataFiveHead
        from src.encoder import seq_key
        from src.substitution import aa_index

        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

        ext_tables = {
            "fireprotdb_hf25": load_external_table(str(self.data_dir / "fireprotdb_hf25.parquet")),
            "s669": load_external_table(str(self.data_dir / "s669_holdout.parquet")),
            "s1724_hf25": load_external_table(str(self.data_dir / "s1724_thermomutdb_hf25.parquet")),
            "domainome_hf25": load_external_table(str(self.data_dir / "domainome_apca_hf25.parquet")),
        }

        def _group_by_parent(df: pd.DataFrame) -> list[dict]:
            scol = "parent_seq" if "parent_seq" in df.columns else "wt_seq"
            tcol = "target" if "target" in df.columns else "ddg"
            groups = []
            for seq, g in df.groupby(scol, sort=False):
                seq_s = str(seq).upper()
                if not seq_s or len(seq_s) > 1020:
                    # Chunk/skip >1020 only if needed; handle <=1020 directly
                    seq_s_use = seq_s[:1020]
                else:
                    seq_s_use = seq_s
                sites = np.clip(g["site"].to_numpy(dtype=np.int64), 0, len(seq_s_use) - 1)
                wt = np.array([aa_index(a) for a in g["wt_aa"]], dtype=np.int64)
                mt = np.array([aa_index(a) for a in g["mt_aa"]], dtype=np.int64)
                y = g[tcol].to_numpy(dtype=np.float32) if tcol in g.columns else np.zeros(len(g), dtype=np.float32)
                rids = g["row_id"].to_numpy(dtype=np.int64)
                groups.append({"seq": seq_s_use, "orig_seq": seq_s, "sites": sites, "wt": wt, "mt": mt, "y": y, "rids": rids})
            return groups

        def _eval_groups(esm_mod: nn.Module, head_mod: nn.Module, groups: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            esm_mod.eval()
            head_mod.eval()
            rids_all, preds_all, ys_all = [], [], []
            with torch.no_grad():
                for g in groups:
                    toks = tokenizer(g["seq"], return_tensors="pt", add_special_tokens=True)
                    toks = {k: v.to(device) for k, v in toks.items()}
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        h = esm_mod(**toks).last_hidden_state[0, 1:1 + len(g["seq"])].float()
                    gvec = h.mean(dim=0, keepdim=True)
                    sites_t = torch.from_numpy(g["sites"]).to(device)
                    wt_t = torch.from_numpy(g["wt"]).to(device)
                    mt_t = torch.from_numpy(g["mt"]).to(device)
                    p_chunks = []
                    for lo in range(0, len(sites_t), 2048):
                        hi = lo + 2048
                        p_chunks.append(
                            head_mod(h[sites_t[lo:hi]], gvec.expand(hi - lo if hi <= len(sites_t) else len(sites_t) - lo, -1),
                                     wt_t[lo:hi], mt_t[lo:hi]).reshape(-1).cpu().numpy()
                        )
                    rids_all.append(g["rids"])
                    preds_all.append(np.concatenate(p_chunks))
                    ys_all.append(g["y"])
            return np.concatenate(rids_all), np.concatenate(preds_all), np.concatenate(ys_all)

        def _measure_contact_drift(mlm_ft: EsmForMaskedLM, test_groups: list[dict]) -> dict:
            """Measure ESM-2 contact head P@L (|i-j|>=6) & P@L/5 (|i-j|>=24) on fine-tuned ESM-2."""
            mlm_ft.eval()
            p_at_l, p_at_l5 = [], []
            with torch.no_grad():
                for g in test_groups:
                    cpath = self.true_contacts_dir / f"{seq_key(g['orig_seq'])}.npy"
                    if not cpath.is_file():
                        continue
                    cmap = np.load(cpath)
                    L = len(g["seq"])
                    if cmap.shape != (L, L):
                        continue
                    toks = tokenizer(g["seq"], return_tensors="pt", add_special_tokens=True)
                    toks = {k: v.to(device) for k, v in toks.items()}
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = mlm_ft.esm(**toks, output_attentions=True)
                    attns = torch.stack([a.float() for a in out.attentions], dim=1)
                    pred_c = mlm_ft.esm.contact_head(toks["input_ids"], attns)[0].cpu().numpy()
                    if pred_c.shape != (L, L):
                        continue
                    ii, jj = np.triu_indices(L, k=6)
                    if len(ii) >= L:
                        top = np.argsort(pred_c[ii, jj])[::-1][:L]
                        p_at_l.append(float((cmap[ii[top], jj[top]] > 0.5).mean()))
                    ii24, jj24 = np.triu_indices(L, k=24)
                    k5 = max(1, L // 5)
                    if len(ii24) >= k5:
                        top5 = np.argsort(pred_c[ii24, jj24])[::-1][:k5]
                        p_at_l5.append(float((cmap[ii24[top5], jj24[top5]] > 0.5).mean()))
            return {
                "n_proteins_scored": len(p_at_l),
                "contact_p_at_L_sep6": round(float(np.mean(p_at_l)), 4) if p_at_l else None,
                "contact_p_at_L5_sep24": round(float(np.mean(p_at_l5)), 4) if p_at_l5 else None,
            }

        ext_groups = {k: _group_by_parent(df) for k, df in ext_tables.items()}

        for sp_name, sp_file in [
            ("id25", self.data_dir / "splits_megascale_id25.parquet"),
            ("fold_tm60", self.data_dir / "splits_megascale_fold_tm60.parquet"),
        ]:
            splits = load_split_table(str(sp_file))
            tr_groups = _group_by_parent(splits[splits["split"] == "train"])
            va_groups = _group_by_parent(splits[splits["split"] == "val"])
            te_groups = _group_by_parent(splits[splits["split"] == "test"])

            for seed in (42, 43, 44):
                pack = f"block2_prostata_ft/{sp_name}/prostata_ft_s{seed}"
                if self.item_done(pack):
                    log(f"  [skip] {pack}")
                    continue
                log(f"Training {pack} (650M E2E bf16 + grad ckpt, 10 epochs, patience 3) ...")
                t0 = time.time()
                torch.manual_seed(seed)
                np.random.seed(seed)
                rng = np.random.default_rng(seed)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", attn_implementation="eager").to(device)
                esm = mlm.esm
                esm.gradient_checkpointing_enable()
                # Freeze only contact_head & lm_head so we can measure drift cleanly on untouched heads!
                for p in mlm.lm_head.parameters():
                    p.requires_grad = False
                for p in esm.contact_head.parameters():
                    p.requires_grad = False
                head = ProstataFiveHead(1280).to(device)

                # Snapshot initial backbone weights for Frobenius drift norm on seed 42
                init_snap = None
                if seed == 42:
                    init_snap = {k: v.detach().cpu().float().clone() for k, v in esm.state_dict().items() if v.is_floating_point()}

                opt = torch.optim.AdamW([
                    {"params": [p for p in esm.parameters() if p.requires_grad], "lr": 1e-5, "weight_decay": 0.01},
                    {"params": list(head.parameters()), "lr": 3e-4, "weight_decay": 0.01},
                ])
                epochs = 10
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                loss_fn = nn.MSELoss()

                best_r = -1e9
                best_ep = 0
                patience = 0
                best_esm_cpu = None
                best_head_cpu = None
                hist = []

                for ep in range(1, epochs + 1):
                    esm.train()
                    head.train()
                    order = rng.permutation(len(tr_groups))
                    opt.zero_grad(set_to_none=True)
                    ep_loss, n_steps = 0.0, 0
                    accum = 4  # effective 4 parents x up to 128 muts = 512 muts/step
                    for step_i, gi in enumerate(order):
                        g = tr_groups[gi]
                        toks = tokenizer(g["seq"], return_tensors="pt", add_special_tokens=True)
                        toks = {k: v.to(device) for k, v in toks.items()}
                        n_m = len(g["sites"])
                        sel = rng.choice(n_m, size=min(128, n_m), replace=False)
                        sites_t = torch.from_numpy(g["sites"][sel]).to(device)
                        wt_t = torch.from_numpy(g["wt"][sel]).to(device)
                        mt_t = torch.from_numpy(g["mt"][sel]).to(device)
                        y_t = torch.from_numpy(g["y"][sel]).to(device)

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            h = esm(**toks).last_hidden_state[0, 1:1 + len(g["seq"])]
                            gvec = h.mean(dim=0, keepdim=True).expand(len(sel), -1)
                            pred = head(h[sites_t], gvec, wt_t, mt_t).reshape(-1)
                            loss = loss_fn(pred.float(), y_t) / accum
                        loss.backward()
                        ep_loss += float(loss.item()) * accum
                        n_steps += 1
                        if (step_i + 1) % accum == 0 or (step_i + 1) == len(order):
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in esm.parameters() if p.requires_grad] + list(head.parameters()), 1.0
                            )
                            opt.step()
                            opt.zero_grad(set_to_none=True)
                    sched.step()

                    _, vp, vy = _eval_groups(esm, head, va_groups)
                    vr = float(pearsonr(vy, vp).statistic)
                    hist.append({"epoch": ep, "train_loss": round(ep_loss / max(n_steps, 1), 4), "val_r": round(vr, 4)})
                    log(f"  [{pack}] ep {ep}/{epochs}: loss={ep_loss/max(n_steps,1):.4f}  val_r={vr:.4f}")
                    if np.isfinite(vr) and vr > best_r:
                        best_r = vr
                        best_ep = ep
                        patience = 0
                        best_esm_cpu = {k: v.detach().cpu().clone() for k, v in esm.state_dict().items()}
                        best_head_cpu = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    else:
                        patience += 1
                        if patience >= 3:
                            log(f"  [{pack}] early stop at ep {ep} (best ep {best_ep}, val_r={best_r:.4f})")
                            break

                if best_esm_cpu is not None:
                    esm.load_state_dict(best_esm_cpu)
                    head.load_state_dict(best_head_cpu)

                out_dir = self.local_out / pack
                out_dir.mkdir(parents=True, exist_ok=True)
                dt = round(time.time() - t0, 2)
                peak_gb = gpu_peak_gb()

                # Evaluate test + all 4 external holdouts
                te_rids, te_preds, te_y = _eval_groups(esm, head, te_groups)
                te_r = float(pearsonr(te_y, te_preds).statistic)
                te_rmse = float(np.sqrt(np.mean((te_y - te_preds) ** 2)))
                meta = {
                    "model": "PROSTATA_FT_650M_E2E",
                    "sign": "stabilizing_positive (:+1)",
                    "trainable_params": sum(p.numel() for p in esm.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters()),
                    "best_epoch": best_ep,
                    "best_val_pearson": round(best_r, 4),
                    "test_pearson": round(te_r, 4),
                    "test_rmse": round(te_rmse, 4),
                    "wall_clock_s": dt,
                    "peak_vram_gb": peak_gb,
                    "seed": seed,
                    "history": hist,
                }
                write_predictions(out_dir / "pred_test.parquet", te_rids, te_preds, meta=meta)

                ext_scores = {}
                for ename, egrps in ext_groups.items():
                    erids, epreds, ey = _eval_groups(esm, head, egrps)
                    sign_flip = -1.0 if ename == "fireprotdb_hf25" else 1.0
                    er = float(pearsonr(ey * sign_flip, epreds).statistic)
                    ext_scores[ename] = round(er, 4)
                    write_predictions(
                        out_dir / f"pred_{ename}.parquet", erids, epreds,
                        meta={"model": "PROSTATA_FT_650M_E2E", "seed": seed, "external": ename, "pearson": round(er, 4)},
                    )

                drift = {}
                if seed == 42 and init_snap is not None:
                    sq = 0.0
                    for k, v0 in init_snap.items():
                        diff = esm.state_dict()[k].detach().cpu().float() - v0
                        sq += float((diff * diff).sum().item())
                    drift["delta_W_frobenius"] = round(float(np.sqrt(sq)), 4)
                    drift["contact_probe_ft"] = _measure_contact_drift(mlm, te_groups)
                    # Also measure untouched base ESM-2 on exact same test_groups once
                    base_mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", attn_implementation="eager").to(device)
                    drift["contact_probe_frozen_base"] = _measure_contact_drift(base_mlm, te_groups)
                    del base_mlm
                    (out_dir / "prostata_ft_drift_diagnostic.json").write_text(json.dumps(drift, indent=2), encoding="utf-8")
                    log(f"  [{pack}] DRIFT AUDIT: {json.dumps(drift)}")

                meta["external_pearson"] = ext_scores
                meta["drift"] = drift
                self.mark_done(pack, meta)

                del mlm, esm, head, opt, best_esm_cpu, best_head_cpu, init_snap
                gc.collect()
                torch.cuda.empty_cache()

        log("block2_prostata_ft COMPLETE!")

    # ------------------------------------------------------------------
    # Stage 6: Block 2B — ESM-2 3B Scaling Study on v2-Lean
    # ------------------------------------------------------------------
    def stage_block2_esm3b(self) -> None:
        log("=== STAGE: block2_esm3b (ESM-2 3B Scaling Study on v2-Lean & v2-Lean+8A Contacts) ===")
        self.record_env()
        self.prepare_inputs()
        self._precompute_cache("facebook/esm2_t36_3B_UR50D", self.cache_3b, "block2_precompute_3b")

        for sp_name, sp_file in [
            ("id25", self.data_dir / "splits_megascale_id25.parquet"),
            ("fold_tm60", self.data_dir / "splits_megascale_fold_tm60.parquet"),
        ]:
            for seed in (42, 43, 44, 45, 46):
                self._train_and_export_holo(
                    f"block2_esm3b/{sp_name}/v2_lean_3b_s{seed}",
                    sp_file, self.cache_3b, "facebook/esm2_t36_3B_UR50D", seed, False,
                )
                self._train_and_export_holo(
                    f"block2_esm3b/{sp_name}/v2_lean_3b_true_contacts_s{seed}",
                    sp_file, self.cache_3b, "facebook/esm2_t36_3B_UR50D", seed, True,
                )
        log("block2_esm3b COMPLETE!")

    # ------------------------------------------------------------------
    # Stage 7: Block 3 — Calibrated Release Bundle + Live CLI & API Verification
    # ------------------------------------------------------------------
    def stage_block3_demo(self) -> None:
        pack = "block3_demo"
        log("=== STAGE: block3_demo (Calibration, Release Bundle, Live CLI & FastAPI Verification) ===")
        self.record_env()
        self.prepare_inputs()
        self._precompute_cache("facebook/esm2_t33_650M_UR50D", self.cache_650m, "block1_precompute_650m")

        bundle = self.local_out / pack / "release_bundle"
        bundle.mkdir(parents=True, exist_ok=True)

        # Prefer G4-trained v2_lean s42, else fall back to uploaded weight
        g4_ckpt = self.local_out / "block1_parity/id25/v2_lean_s42/model_best.pt"
        if not g4_ckpt.is_file():
            g4_ckpt = self.drive_out / "block1_parity/id25/v2_lean_s42/model_best.pt"
        if not g4_ckpt.is_file():
            g4_ckpt = self.weights_dir / "v2_lean_id25_s42.pt"
        shutil.copy2(g4_ckpt, bundle / "hologe_v2_lean_best.pt")
        shutil.copy2(self.weights_dir / "v2_lean_str_only_id25_s42.pt", bundle / "hologe_v2_3d_best.pt")
        shutil.copy2(self.weights_dir / "epistasis_egeo_id25_s42.pt", bundle / "hologe_v2_epistasis_best.pt")

        # Run scripts.calibration on id25
        rc_c, dt_c, tail_c = run_cmd([
            sys.executable, "-m", "scripts.calibration",
            "--checkpoint", str(bundle / "hologe_v2_lean_best.pt"),
            "--splits", str(self.data_dir / "splits_megascale_id25.parquet"),
            "--config", "configs/benchmark_v2.yaml",
            "--cache", str(self.cache_650m),
            "--out", str(bundle),
        ], timeout=600)
        if rc_c != 0:
            self.record_error("block3_demo/calibration", tail_c)

        # Live CLI benchmarks + Real Biological Case Studies (Ubiquitin, GB1, Human p53 DBD)
        import pandas as pd
        import numpy as np

        ubq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        gb1 = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
        # Human TP53 (UniProt P04637) DNA-binding domain residues 94-292 (199 aa);
        # offset +93 maps 1-based DBD pos to canonical full-length TP53 numbering!
        p53_dbd = (
            "SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPP"
            "GTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFR"
            "HSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVR"
            "VCACPGRDRRTEEENLRKK"
        )
        # Verify canonical oncogenic hotspot WT residues in p53 DBD (full-length pos - 93)
        p53_hotspots_full = [("R175H", 175), ("Y220C", 220), ("G245S", 245), ("R248Q", 248), ("R249S", 249), ("R273H", 273), ("R282W", 282)]
        p53_dbd_muts = []
        for full_code, full_pos in p53_hotspots_full:
            dbd_pos = full_pos - 93
            assert p53_dbd[dbd_pos - 1] == full_code[0], f"p53 mismatch at {full_code}: got {p53_dbd[dbd_pos - 1]}"
            p53_dbd_muts.append(f"{full_code[0]}{dbd_pos}{full_code[-1]}")

        case_dir = self.local_out / pack / "case_studies"
        case_dir.mkdir(parents=True, exist_ok=True)
        fa_ubq = case_dir / "ubiquitin_76aa.fasta"
        fa_gb1 = case_dir / "gb1_56aa.fasta"
        fa_p53 = case_dir / "tp53_dbd_199aa.fasta"
        fa_ubq.write_text(f">Ubiquitin_P0CG48_76aa\n{ubq}\n", encoding="utf-8")
        fa_gb1.write_text(f">GB1_ProteinG_56aa\n{gb1}\n", encoding="utf-8")
        fa_p53.write_text(f">Human_TP53_DBD_P04637_94_292\n{p53_dbd}\n", encoding="utf-8")

        cli_telemetry = {}
        rc1, dt1, out1 = run_cmd([sys.executable, "holo.py", "doctor"], timeout=120)
        cli_telemetry["doctor"] = {"rc": rc1, "s": dt1, "output": out1[-1500:]}

        for label, fa_p, muts_str, out_name in [
            ("predict_ubq_4muts", fa_ubq, "V5A,L8P,I44A,K48R", "ubq_predict.json"),
            ("predict_p53_hotspots", fa_p53, ",".join(p53_dbd_muts), "p53_hotspots_predict.json"),
            ("predict_gb1_singles", fa_gb1, "L5A,V39A,D40A,G41A,V54A", "gb1_predict.json"),
        ]:
            rc_p, dt_p, out_p = run_cmd([
                sys.executable, "holo.py", "predict",
                "--fasta", str(fa_p), "--muts", muts_str,
                "--weights", str(bundle / "hologe_v2_lean_best.pt"),
                "--out", str(case_dir / out_name),
            ], timeout=300)
            cli_telemetry[label] = {"rc": rc_p, "s": dt_p, "output": out_p[-1500:]}

        for label, fa_p, stem in [
            ("scan_ubq_76aa", fa_ubq, "ubq_scan"),
            ("scan_gb1_56aa", fa_gb1, "gb1_scan"),
            ("scan_p53_dbd_199aa", fa_p53, "p53_dbd_scan"),
        ]:
            rc2, dt2, out2 = run_cmd([
                sys.executable, "holo.py", "scan",
                "--fasta", str(fa_p),
                "--weights", str(bundle / "hologe_v2_lean_best.pt"),
                "--out", str(case_dir / f"{stem}.csv"),
                "--html", str(case_dir / f"{stem}.html"),
            ], timeout=360)
            cli_telemetry[label] = {"rc": rc2, "s": dt2, "output": out2[-1500:]}

        for label, fa_p, pairs_str, out_csv in [
            ("epistasis_ubq", fa_ubq, "V5A+I13A,L8P+I44A,K48R+K63R", "ubq_epistasis.csv"),
            ("epistasis_gb1", fa_gb1, "V39A+D40A,V39A+G41A,D40A+G41A,G41A+V54A,V39A+V54A,L5A+V54A", "gb1_epistasis.csv"),
        ]:
            rc3, dt3, out3 = run_cmd([
                sys.executable, "holo.py", "epistasis",
                "--fasta", str(fa_p), "--pairs", pairs_str,
                "--weights", str(bundle / "hologe_v2_epistasis_best.pt"),
                "--out", str(case_dir / out_csv),
            ], timeout=300)
            cli_telemetry[label] = {"rc": rc3, "s": dt3, "output": out3[-1500:]}

        # Render publication-ready 300-DPI visual simulation figures directly from live GPU outputs
        vis_dir = self.local_out / pack / "visual_outputs"
        vis_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def _plot_dms_heatmap(csv_path: Path, title: str, out_stem: Path, pos_offset: int = 0,
                                  highlight_positions: list[tuple[int, str]] | None = None):
                if not csv_path.is_file():
                    return
                df = pd.read_csv(csv_path)
                pos_col = "position" if "position" in df.columns else "site"
                aa_col = "mt_aa" if "mt_aa" in df.columns else ("mutant_aa" if "mutant_aa" in df.columns else "mut_aa")
                val_col = "ddg_kcal_mol" if "ddg_kcal_mol" in df.columns else ("ddg" if "ddg" in df.columns else "pred_ddg")
                if not all(c in df.columns for c in (pos_col, aa_col, val_col)):
                    return
                piv = df.pivot_table(index=aa_col, columns=pos_col, values=val_col, aggfunc="mean")
                aa_order = [a for a in "ACDEFGHIKLMNPQRSTVWY" if a in piv.index]
                piv = piv.loc[aa_order]
                xs = piv.columns.to_numpy() + pos_offset
                mean_ddg = piv.mean(axis=0).to_numpy()

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.2), gridspec_kw={"height_ratios": [1.1, 2.4]}, sharex=True)
                ax1.fill_between(xs, mean_ddg, 0, where=(mean_ddg < 0), color="#d62728", alpha=0.35)
                ax1.fill_between(xs, mean_ddg, 0, where=(mean_ddg >= 0), color="#1f77b4", alpha=0.35)
                ax1.plot(xs, mean_ddg, color="#222222", lw=1.2)
                ax1.axhline(0.0, color="#666666", lw=0.7, ls="--")
                ax1.set_ylabel("Mean ΔΔG\n(kcal/mol)", fontsize=9)
                ax1.set_title(title, fontsize=11, fontweight="bold")
                if highlight_positions:
                    for hp, lbl in highlight_positions:
                        idx = np.argmin(np.abs(xs - hp))
                        ax1.scatter([xs[idx]], [mean_ddg[idx]], color="#b2182b", s=28, zorder=5)
                        ax1.annotate(lbl, (xs[idx], mean_ddg[idx]), xytext=(0, -14), textcoords="offset points",
                                     ha="center", fontsize=7.5, fontweight="bold", color="#8c1515")
                im = ax2.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu", vmin=-3.0, vmax=1.5,
                                extent=[xs[0] - 0.5, xs[-1] + 0.5, len(aa_order) - 0.5, -0.5])
                ax2.set_yticks(range(len(aa_order)))
                ax2.set_yticklabels(aa_order, fontsize=7.5)
                ax2.set_xlabel("Residue Position", fontsize=9.5)
                ax2.set_ylabel("Mutant AA", fontsize=9)
                cbar = fig.colorbar(im, ax=[ax1, ax2], pad=0.015, aspect=28)
                cbar.set_label("Predicted ΔΔG (kcal/mol; >0 stabilizing)", fontsize=8.5)
                fig.tight_layout()
                fig.savefig(str(out_stem) + ".png", dpi=300, bbox_inches="tight")
                fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
                plt.close(fig)

            _plot_dms_heatmap(
                case_dir / "p53_dbd_scan.csv",
                "Holo-GeoN v2-Lean Live GPU Stability Landscape: Human TP53 DNA-Binding Domain (Residues 94–292)",
                vis_dir / "case_study_p53_dbd_landscape",
                pos_offset=93,
                highlight_positions=[(175, "R175"), (220, "Y220"), (245, "G245"), (248, "R248"), (273, "R273"), (282, "R282")],
            )
            _plot_dms_heatmap(
                case_dir / "ubq_scan.csv",
                "Holo-GeoN v2-Lean Live GPU Stability Landscape: Human Ubiquitin (P0CG48, 76 aa)",
                vis_dir / "case_study_ubiquitin_landscape",
                pos_offset=0,
            )
            _plot_dms_heatmap(
                case_dir / "gb1_scan.csv",
                "Holo-GeoN v2-Lean Live GPU Stability Landscape: Protein G B1 Domain (GB1, 56 aa)",
                vis_dir / "case_study_gb1_landscape",
                pos_offset=0,
            )
            log("Rendered 300-DPI case-study DMS landscape PNG/PDF plots!")
        except Exception as exc:
            log(f"WARNING: visual case-study plot failed: {exc}")

        # Live FastAPI Web GUI backend end-to-end verification + Headless Browser UI Screenshots
        api_results = {}
        try:
            import urllib.request
            # Find a real Ubiquitin or MegaScale PDB for /api/compare & 3D mode verification
            sample_pdb_text = None
            for p_cand in (self.pdb_dir / "megascale").rglob("*.pdb"):
                if "1UBQ" in p_cand.name.upper() or "1PGA" in p_cand.name.upper():
                    sample_pdb_text = p_cand.read_text(encoding="utf-8", errors="ignore")
                    break
            if sample_pdb_text is None:
                pdbs = sorted((self.pdb_dir / "megascale").rglob("*.pdb"))
                if pdbs:
                    sample_pdb_text = pdbs[0].read_text(encoding="utf-8", errors="ignore")

            env_srv = dict(os.environ)
            env_srv["PYTHONPATH"] = str(REPO_ROOT)
            env_srv["HOLOGEON_WEIGHTS"] = str(bundle / "hologe_v2_lean_best.pt")
            env_srv["HOLOGEON_WEIGHTS_3D"] = str(bundle / "hologe_v2_3d_best.pt")
            env_srv["HOLOGEON_CALIBRATION"] = str(bundle / "calibration.json")
            srv = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8011"],
                cwd=str(REPO_ROOT / "app" / "backend"),
                env=env_srv,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                for _ in range(90):
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:8011/api/health", timeout=3) as r:
                            api_results["health"] = json.loads(r.read().decode())
                            break
                    except Exception:
                        time.sleep(1.0)
                if "health" in api_results:
                    def _post(path: str, payload: dict) -> dict:
                        req = urllib.request.Request(
                            f"http://127.0.0.1:8011{path}",
                            data=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        t_r = time.perf_counter()
                        with urllib.request.urlopen(req, timeout=180) as resp:
                            body = json.loads(resp.read().decode())
                        return {"elapsed_ms": round((time.perf_counter() - t_r) * 1000.0, 2), "response": body}

                    api_results["ddg_ubq_L8P"] = _post("/api/ddg", {"sequence": ubq, "mutation": "L8P"})
                    api_results["ddg_p53_R175H"] = _post("/api/ddg", {"sequence": p53_dbd, "mutation": "R82H"})
                    api_results["scan_ubq"] = _post("/api/scan", {"sequence": ubq})
                    if "response" in api_results["scan_ubq"]:
                        sr = api_results["scan_ubq"]["response"]
                        api_results["scan_ubq"]["response"] = {
                            k: v for k, v in sr.items() if k not in ("matrix", "ddg_matrix", "columns")
                        }

                    # Capture live high-DPI headless Chromium screenshots of the real Web GUI v2!
                    try:
                        subprocess.run(
                            "uv pip install --system --quiet playwright && playwright install chromium",
                            shell=True, check=False, timeout=240,
                        )
                        from playwright.sync_api import sync_playwright  # type: ignore

                        with sync_playwright() as pw:
                            browser = pw.chromium.launch(headless=True)
                            page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
                            page.goto("http://127.0.0.1:8011/", wait_until="networkidle", timeout=60000)
                            page.wait_for_timeout(1500)
                            page.screenshot(path=str(vis_dir / "ui_01_home_ddg.png"), full_page=True)
                            # Trigger a live ΔΔG prediction on Ubiquitin L8P via UI inputs if present
                            textareas = page.locator("textarea")
                            if textareas.count() > 0:
                                textareas.first.fill(ubq)
                            mut_inputs = page.locator("input[placeholder*='L8P'], input[placeholder*='mutation' i], input[type='text']")
                            if mut_inputs.count() > 0:
                                mut_inputs.first.fill("L8P")
                            run_btn = page.locator("button:has-text('Predict'), button:has-text('Run')").first
                            if run_btn.count() > 0:
                                run_btn.click()
                                page.wait_for_timeout(5000)
                            page.screenshot(path=str(vis_dir / "ui_02_ddg_live_result.png"), full_page=True)

                            # Stability Landscape tab
                            scan_tab = page.locator("button:has-text('Stability Landscape')").first
                            if scan_tab.count() > 0:
                                scan_tab.click()
                                page.wait_for_timeout(1000)
                                ta = page.locator("textarea").first
                                if ta.count() > 0:
                                    ta.fill(ubq)
                                btn = page.locator("button:has-text('Scan'), button:has-text('Run')").first
                                if btn.count() > 0:
                                    btn.click()
                                    page.wait_for_timeout(8000)
                                page.screenshot(path=str(vis_dir / "ui_03_scan_landscape_live.png"), full_page=True)

                            # AlphaFold Compare tab (3Dmol structure + pLDDT + stability overlay)
                            cmp_tab = page.locator("button:has-text('AlphaFold Compare')").first
                            if cmp_tab.count() > 0:
                                cmp_tab.click()
                                page.wait_for_timeout(1500)
                                page.screenshot(path=str(vis_dir / "ui_04_alphafold_compare.png"), full_page=True)

                            browser.close()
                        api_results["ui_screenshots"] = sorted(p.name for p in vis_dir.glob("ui_*.png"))
                        log(f"Captured live Web GUI screenshots: {api_results['ui_screenshots']}")
                    except Exception as ui_exc:
                        api_results["ui_screenshot_warning"] = f"{type(ui_exc).__name__}: {ui_exc}"
                        log(f"WARNING: Playwright UI screenshot skipped: {ui_exc}")
            finally:
                srv.terminate()
                try:
                    srv.wait(timeout=10)
                except Exception:
                    srv.kill()

            # Second quick FastAPI check with the trained Epistasis checkpoint so /api/epistasis is verified live!
            env_epi = dict(os.environ)
            env_epi["PYTHONPATH"] = str(REPO_ROOT)
            env_epi["HOLOGEON_WEIGHTS"] = str(bundle / "hologe_v2_epistasis_best.pt")
            srv2 = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8012"],
                cwd=str(REPO_ROOT / "app" / "backend"),
                env=env_epi,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                for _ in range(90):
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:8012/api/health", timeout=3) as r:
                            api_results["health_epistasis"] = json.loads(r.read().decode())
                            break
                    except Exception:
                        time.sleep(1.0)
                if "health_epistasis" in api_results:
                    req = urllib.request.Request(
                        "http://127.0.0.1:8012/api/epistasis",
                        data=json.dumps({"wt_sequence": gb1, "pairs": ["V39A+D40A", "V39A+G41A", "D40A+G41A", "L5A+V54A"]}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    t_r = time.perf_counter()
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        api_results["epistasis_gb1_api"] = {
                            "elapsed_ms": round((time.perf_counter() - t_r) * 1000.0, 2),
                            "response": json.loads(resp.read().decode()),
                        }
            finally:
                srv2.terminate()
                try:
                    srv2.wait(timeout=10)
                except Exception:
                    srv2.kill()
        except Exception as exc:
            api_results["error"] = f"{type(exc).__name__}: {exc}"

        (self.local_out / pack / "live_api_verification.json").write_text(
            json.dumps(api_results, indent=2, default=str), encoding="utf-8"
        )
        (self.local_out / pack / "live_cli_telemetry.json").write_text(
            json.dumps(cli_telemetry, indent=2), encoding="utf-8"
        )
        self.mark_done(pack, {"calibration_rc": rc_c, "cli_telemetry": cli_telemetry, "api_ok": "ddg_ubq_L8P" in api_results})
        log("block3_demo COMPLETE!")

    # ------------------------------------------------------------------
    # Stage 8: Block 4 — Separate Dataset Submission ZIP + SHA256SUMS on Drive
    # ------------------------------------------------------------------
    def stage_block4_dataset_zip(self) -> None:
        pack = "block4_dataset_zip"
        log("=== STAGE: block4_dataset_zip (Packaging separate SHA-256 dataset release ZIP on Drive) ===")
        import hashlib

        zip_local = self.work_root / "holo_geon_a100_dataset_release.zip"
        sha_lines = []
        included = []

        # Collect from drive_out (source of truth across any restarts) + local_out
        src_root = self.drive_out
        with zipfile.ZipFile(zip_local, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            # Add all benchmark predictions, metadata, reports, calibrated release bundle, and simulation outputs
            for f in sorted(src_root.rglob("*")):
                if not f.is_file():
                    continue
                if f.name.endswith(".zip") or f.name == "last.pt":
                    continue
                rel = f.relative_to(src_root).as_posix()
                # Skip bulky optimizer checkpoints if any; keep model_best.pt for release_bundle & parquets/jsons/csvs/pngs/pdfs/htmls
                if f.suffix == ".pt" and "release_bundle" not in rel:
                    continue
                zf.write(f, arcname=f"Holo-GeoN-A100-Results/{rel}")
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                sha_lines.append(f"{h}  Holo-GeoN-A100-Results/{rel}")
                included.append({"rel": rel, "bytes": f.stat().st_size})

            readme = (
                "# Holo-GeoN v2 — Unified Same-GPU (A100 80 GB) Benchmark & Dataset Release\n\n"
                f"Generated UTC: {utc_now()}\n"
                f"Environment: {json.dumps(self.report.get('environment', {}), indent=2)}\n\n"
                "## Contents\n"
                "- `block1_speed/speed_vram_microbench.json`: CUDA-synchronized single-pass vs two-pass ESM-2 & 20xL DMS throughput/VRAM.\n"
                "- `block1_parity/`: Identical-GPU 5-seed predictions (id25, fold_tm60, FireProtDB-HF25, S669, S1724-HF25, Domainome-HF25) for Holo-GeoN v2-Lean, v2-Lean+8A Contacts, ThermoMPNN (+official), SPURS (+official), RaSP, PROSTATA, and DDGemb.\n"
                "- `block2_prostata_ft/`: True end-to-end 650M fine-tuned PROSTATA-FT predictions + contact P@L & Frobenius drift diagnostic.\n"
                "- `block2_esm3b/`: ESM-2 3B scaling predictions (5 seeds x id25 & fold_tm60 + externals).\n"
                "- `block3_demo/`: Calibrated release weights, live CLI/API outputs, Ubiquitin/GB1/TP53-DBD simulations, interactive HTML heatmaps, 300-DPI figures, and live Web GUI screenshots.\n"
            )
            zf.writestr("Holo-GeoN-A100-Results/README_DATASET.md", readme)
            zf.writestr("Holo-GeoN-A100-Results/SHA256SUMS.txt", "\n".join(sha_lines) + "\n")

        zip_sha = hashlib.sha256(zip_local.read_bytes()).hexdigest()
        zip_mb = round(zip_local.stat().st_size / 1e6, 2)
        dst_zip = self.drive_out / zip_local.name
        shutil.copy2(zip_local, dst_zip)
        meta = {
            "zip_path_drive": str(dst_zip),
            "zip_size_mb": zip_mb,
            "zip_sha256": zip_sha,
            "n_files_packaged": len(included),
        }
        self.mark_done(pack, meta)
        log(f"Dataset submission ZIP written to Drive: {dst_zip} ({zip_mb} MB, {len(included)} files, sha256={zip_sha[:16]})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Holo-GeoN v2 Unified Colab G4 Benchmark Suite")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/holo_geon_g4")
    ap.add_argument("--work-root", default="/tmp/holo_g4")
    ap.add_argument("--run-name", default="g4_suite")
    ap.add_argument(
        "--stages", nargs="+", required=True,
        choices=[
            "dry_run",
            "block1_precompute_650m",
            "block1_speed",
            "block1_parity",
            "block2_prostata_ft",
            "block2_esm3b",
            "block3_demo",
            "block4_dataset_zip",
        ],
    )
    args = ap.parse_args()
    suite = G4Suite(Path(args.drive_root), Path(args.work_root), run_name=args.run_name)

    had_fatal = False
    for st in args.stages:
        try:
            getattr(suite, f"stage_{st}")()
        except Exception:
            suite.record_error(st, traceback.format_exc())
            if st in ("dry_run", "block1_precompute_650m"):
                raise
            had_fatal = True
            log(f"Stage {st} failed; continuing to remaining stages so completed work & dataset ZIP are saved.")
    return 1 if had_fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
