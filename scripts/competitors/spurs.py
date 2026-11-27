"""
scripts/competitors/spurs.py
============================
Competitor adapter for **SPURS** (Li & Luo, *Nature Communications* 17, 891,
2026; ``luo-group/SPURS``, MIT licence): Structure-informed Protein stability
predictor fusing ESM-2 650M and ProteinMPNN structural representations.

Two explicit, provenance-tagged execution modes:
  1. ``--mode upstream`` (default when running on GPU with internet): clones
     ``luo-group/SPURS`` (appended to ``sys.path``, never at index 0), runs
     SPURS's official inference on external holdouts (FireProtDB-HF25 / S669 /
     Domainome) when ``--official-external-only`` is set, or retrains SPURS's
     regression head from scratch on OUR exact MegaScale train split so ZERO
     split leakage enters MegaScale test numbers.
  2. ``--mode local_fusion``: clean, self-contained SPURS-style frozen ESM-2
     650M + SE(3)-invariant 3D backbone/structure fusion head trained directly
     on our exact MegaScale train split (useful for fast deterministic T4 runs
     or offline testing without downloading external weights).

Sign convention:
  This repository uses **stabilizing-positive (:+1)** ``ddG = dG_mut - dG_wt``
  (matching Tsuboyama et al. MegaScale and SPURS's MegaScale-trained checkpoint;
  positive values indicate stabilizing mutations, verified by
  ``scripts/score_predictions.py``).

Outputs common-format ``(row_id, y_pred)`` Parquets + ``.meta.json`` for
``scripts/score_predictions.py``.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.competitors.common import (
    clone_repo,
    gpu_peak_gb,
    load_external_table,
    load_split_table,
    lookup_store_seq,
    match_group_pdb,
    parse_extra_external_spec,
    row_id_array,
    safe_aa_index,
    safe_pearson,
    target_array,
    valid_site_mask,
    write_predictions,
)
from src.encoder import EncodingStore
from src.geometry import (
    NAME_RE,
    STR_NODE_DIM,
    align_to_sequence,
    node_geometry,
    normalize_pdb_stem,
    parse_pdb_backbone,
    resolve_pdb_path,
)
from src.substitution import AA_ORDER, aa_index

SPURS_URL = "https://github.com/luo-group/SPURS.git"
_PDB_CHAIN_SUFFIX_RE = re.compile(r"^[A-Za-z0-9]{4}[_\-\.]([A-Za-z0-9])$")


class SpursFusionHead(nn.Module):
    """
    SPURS-style low-rank ESM-2 + 3D structure fusion regressor with explicit
    wild-type / mutant substitution tokens and antisymmetric prediction.
    """

    def __init__(self, esm_dim: int = 1280, struct_dim: int = STR_NODE_DIM + 1,
                 hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.esm_proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden), nn.GELU())
        self.str_proj = nn.Sequential(nn.LayerNorm(struct_dim), nn.Linear(struct_dim, hidden // 2), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aa = nn.Embedding(len(AA_ORDER) + 1, 64)
        self.scorer = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_esm: torch.Tensor, h_str: torch.Tensor,
                wt_idx: torch.Tensor, mt_idx: torch.Tensor) -> torch.Tensor:
        e_p = self.esm_proj(h_esm)
        s_p = self.str_proj(h_str).to(e_p.dtype)
        z = self.fuse(torch.cat([e_p, s_p], dim=-1))
        s_mt = self.scorer(torch.cat([z, self.aa(mt_idx).to(z.dtype)], dim=-1))
        s_wt = self.scorer(torch.cat([z, self.aa(wt_idx).to(z.dtype)], dim=-1))
        return s_mt - s_wt


@contextlib.contextmanager
def _torch_load_weights_only_false():
    """
    Temporarily default ``torch.load(..., weights_only=False)`` so legacy
    Lightning ``.ckpt`` files (such as upstream SPURS) load under PyTorch >= 2.6,
    restoring ``torch.load`` unconditionally on exit.
    """
    orig_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _compat_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig_load


def _check_module_importable(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except Exception:
        return False


def _first_pdb_chain_id(pdb_path: Path) -> str:
    """Detect the first ATOM (fallback HETATM) chain ID in a PDB file; defaults to 'A'."""
    first_het: str | None = None
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if len(line) > 21 and line.startswith("ATOM"):
                    ch = line[21].strip()
                    if ch:
                        return ch
                elif first_het is None and len(line) > 21 and line.startswith("HETATM"):
                    ch = line[21].strip()
                    if ch:
                        first_het = ch
    except OSError:
        pass
    return first_het or "A"


def _extract_chain_id(parent: str, pdb_path: Path, explicit_chain: str | None = None) -> str:
    """
    Determine the real PDB chain ID for ``parent``:
      1. Explicit single-char chain column value (when present and valid).
      2. Crystal PDB chain suffix on ``parent`` (e.g. ``1A0F_B -> "B"``, ``1UBQ_A -> "A"``).
      3. First ``ATOM`` chain ID found in ``pdb_path`` (fallback ``"A"``).
    """
    if explicit_chain is not None and not (isinstance(explicit_chain, float) and not np.isfinite(explicit_chain)):
        try:
            is_na = bool(pd.isna(explicit_chain))
        except (TypeError, ValueError):
            is_na = False
        if not is_na:
            ch = str(explicit_chain).strip()
            if len(ch) == 1 and ch.isalnum():
                return ch
    s = str(parent or "").strip()
    if s.lower().endswith(".pdb"):
        s = s[:-4]
    m = _PDB_CHAIN_SUFFIX_RE.match(s)
    if m:
        return m.group(1)
    return _first_pdb_chain_id(pdb_path)


def _parse_spurs_output_matrix(raw, seq_len: int) -> np.ndarray:
    """
    Normalize upstream SPURS model output into a finite float32 ``(L, 20)``
    ddG matrix aligned with ``AA_ORDER`` (``ACDEFGHIKLMNPQRSTVWY``).
    """
    val = raw
    if isinstance(val, dict):
        for key in ("ddg", "ddG", "logist", "logits", "pred", "preds"):
            if key in val and val[key] is not None:
                val = val[key]
                break
        else:
            for cand in val.values():
                if isinstance(cand, (torch.Tensor, np.ndarray)):
                    val = cand
                    break
    elif isinstance(val, (tuple, list)):
        if not val:
            raise ValueError("Upstream SPURS model returned an empty sequence")
        val = val[0]

    if isinstance(val, torch.Tensor):
        arr = val.detach().cpu().float().numpy()
    else:
        arr = np.asarray(val, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (L, 20) SPURS output, got shape {arr.shape}")

    n_aa = len(AA_ORDER)
    if arr.shape[1] == n_aa:
        return arr.astype(np.float32, copy=False)
    if arr.shape[0] == n_aa and (seq_len == 0 or arr.shape[1] == seq_len):
        return arr.T.astype(np.float32, copy=False)
    if arr.shape[1] > n_aa and (seq_len == 0 or arr.shape[0] == seq_len):
        return arr[:, :n_aa].astype(np.float32, copy=False)
    raise ValueError(
        f"Cannot align SPURS output matrix of shape {arr.shape} to (L={seq_len}, {n_aa})"
    )


def _call_upstream_spurs_model(model: nn.Module, pdb: dict) -> np.ndarray:
    """
    Call upstream SPURS defensively across API revisions
    (``return_logist=True`` -> ``return_logits=True`` -> positional ``model(pdb)``)
    and return a ``(L, 20)`` float32 array in stabilizing-positive (:+1) convention.
    """
    seq_len = len(str(pdb.get("seq") or pdb.get("sequence") or ""))
    raw = None
    last_type_err: TypeError | None = None
    for kwargs in ({"return_logist": True}, {"return_logits": True}, {}):
        try:
            raw = model(pdb, **kwargs)
            last_type_err = None
            break
        except TypeError as exc:
            last_type_err = exc
            continue
    if last_type_err is not None:
        raise last_type_err
    return _parse_spurs_output_matrix(raw, seq_len=seq_len)


def _write_official_spurs_status(
    out_dir: Path,
    *,
    status: str,
    stage: str,
    repo_dir: Path,
    exc: BaseException | None = None,
    message: str | None = None,
    extra: dict | None = None,
) -> Path:
    tb_tail: list[str] = []
    if exc is not None:
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_tail = "".join(tb_lines).splitlines()[-30:]
    payload = {
        "status": status,
        "stage": stage,
        "exception_type": type(exc).__name__ if exc is not None else None,
        "message": str(exc) if exc is not None else (message or ""),
        "traceback_tail": tb_tail,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "spurs_importable": _check_module_importable("spurs"),
        "spurs_inference_importable": _check_module_importable("spurs.inference"),
        "repo_path_searched": str(repo_dir),
        "repo_exists": bool(repo_dir.exists()),
        "sign": "stabilizing_positive (:+1)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "official_spurs_status.json"
    status_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return status_path


def _resolve_pdb_map(pdb_dir: str | None) -> dict[str, Path]:
    if not pdb_dir or not os.path.isdir(pdb_dir):
        return {}
    return {normalize_pdb_stem(p.name): p for p in Path(pdb_dir).rglob("*.pdb")}


def _match_pdb(parent: str, pdb_map: dict[str, Path]) -> Path | None:
    if not pdb_map:
        return None
    hit = resolve_pdb_path(str(parent), pdb_map)
    return Path(hit) if hit is not None else None


def _build_local_fusion_arrays(df: pd.DataFrame, store: EncodingStore,
                               pdb_map: dict[str, Path],
                               par_col: str = "parent",
                               esm_dim: int = 0) -> tuple:
    he_l, hs_l, wt_l, mt_l, y_l, rid_l = [], [], [], [], [], []
    seq_col = "parent_seq" if "parent_seq" in df.columns else ("wt_seq" if "wt_seq" in df.columns else None)

    if len(df) > 0 and seq_col is not None and all(c in df.columns for c in ("site", "wt_aa", "mt_aa")):
        for seq, grp in df.groupby(seq_col, sort=False):
            seq_key_str = lookup_store_seq(store, str(seq))
            if seq_key_str is None:
                continue
            rec = store.get(seq_key_str)
            emb = rec["emb"].float()
            L = int(emb.shape[0])
            if L <= 0:
                continue
            esm_dim = int(emb.shape[-1])
            seq_s = str(seq).strip().upper()

            in_bounds, sites = valid_site_mask(grp["site"], L)
            if sites.size == 0:
                continue
            grp_ok = grp if bool(in_bounds.all()) else grp.loc[in_bounds]

            # Structural features: prefer cache coords, else aligned PDB, else zero (masked)
            s_feat = torch.zeros((L, STR_NODE_DIM + 1), dtype=torch.float32)
            coords_rec = rec.get("coords")
            cvalid_rec = rec.get("coord_valid")
            if (
                coords_rec is not None
                and cvalid_rec is not None
                and int(coords_rec.shape[0]) == L
                and int(cvalid_rec.shape[0]) == L
            ):
                plddt_rec = rec.get("plddt")
                if plddt_rec is None or int(plddt_rec.shape[0]) != L:
                    plddt_rec = torch.full((L,), 100.0, dtype=torch.float32)
                ng = node_geometry(coords_rec.float(), plddt_rec.float(), cvalid_rec.float())
                s_feat[:, :STR_NODE_DIM] = ng
                s_feat[:, -1] = cvalid_rec.float()
            elif pdb_map:
                p_val = grp_ok[par_col].iloc[0] if par_col in grp_ok.columns else None
                pfile = match_group_pdb(p_val, grp_ok, pdb_map, _match_pdb)
                if pfile is not None:
                    pseq, coords, plddt, atom_valid = parse_pdb_backbone(str(pfile))
                    off, _, _ = align_to_sequence(pseq, seq_s)
                    if off is not None and off + L <= len(pseq):
                        c_sl = torch.from_numpy(coords[off:off + L]).float()
                        p_sl = torch.from_numpy(plddt[off:off + L]).float()
                        valid = torch.from_numpy(atom_valid[off:off + L]).float()
                        s_feat[:, :STR_NODE_DIM] = node_geometry(c_sl, p_sl, valid)
                        s_feat[:, -1] = valid

            he_l.append(emb[sites])
            hs_l.append(s_feat[sites])
            wt_l.append(torch.tensor([safe_aa_index(a) for a in grp_ok["wt_aa"]], dtype=torch.long))
            mt_l.append(torch.tensor([safe_aa_index(a) for a in grp_ok["mt_aa"]], dtype=torch.long))
            y_l.append(torch.from_numpy(target_array(grp_ok)))
            rid_l.append(row_id_array(grp_ok))

    if not he_l:
        return (
            torch.empty((0, esm_dim), dtype=torch.float32),
            torch.empty((0, STR_NODE_DIM + 1), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        torch.cat(he_l, dim=0),
        torch.cat(hs_l, dim=0),
        torch.cat(wt_l, dim=0),
        torch.cat(mt_l, dim=0),
        torch.cat(y_l, dim=0),
        np.concatenate(rid_l),
    )


def _score_upstream_spurs_table(
    model: nn.Module,
    cfg,
    parse_pdb,
    ext_df: pd.DataFrame,
    pdb_map: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score one external table with a loaded upstream SPURS ``(model, cfg, parse_pdb)``
    bundle. Returns ``(row_ids, y_pred)`` in stabilizing-positive (:+1) convention.
    """
    if len(ext_df) == 0 or not pdb_map:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    par_col = "parent" if "parent" in ext_df.columns else ("uniprot_id" if "uniprot_id" in ext_df.columns else None)
    seq_col = "wt_seq" if "wt_seq" in ext_df.columns else ("parent_seq" if "parent_seq" in ext_df.columns else None)
    if par_col is None or seq_col is None or any(c not in ext_df.columns for c in ("site", "mt_aa")):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    aa_to_col = {aa: idx for idx, aa in enumerate(AA_ORDER)}
    rids: list[int] = []
    preds: list[float] = []
    n_attempted = 0
    last_err: Exception | None = None

    for parent, grp in ext_df.groupby(par_col, sort=False):
        pfile = match_group_pdb(parent, grp, pdb_map, _match_pdb)
        if pfile is None:
            continue
        explicit_chain = grp["chain"].iloc[0] if "chain" in grp.columns else None
        chain = _extract_chain_id(str(parent), pfile, explicit_chain=explicit_chain)
        n_attempted += 1
        try:
            pdb = parse_pdb(str(pfile), str(parent), chain, cfg)
            if not isinstance(pdb, dict):
                continue
            pseq = str(pdb.get("seq") or pdb.get("sequence") or "").strip().upper()
            if not pseq:
                continue
            with torch.no_grad():
                ddg_mat = _call_upstream_spurs_model(model, pdb)  # (L, 20)
        except Exception as exc:
            last_err = exc
            print(f"[spurs] WARNING: skipping parent {parent!r} (chain={chain}): {exc}", flush=True)
            continue

        wt_seq = str(grp[seq_col].iloc[0]).strip().upper()
        off, _, _ = align_to_sequence(pseq, wt_seq)
        if off is None:
            continue
        grp_rids = row_id_array(grp)
        for rid_val, r in zip(grp_rids, grp.itertuples(index=False)):
            raw_site = getattr(r, "site", None)
            if raw_site is None or pd.isna(raw_site):
                continue
            pos = int(raw_site) + off
            raw_mt = getattr(r, "mt_aa", None)
            if raw_mt is None or pd.isna(raw_mt):
                continue
            m_aa = str(raw_mt).strip().upper()
            if 0 <= pos < ddg_mat.shape[0] and m_aa in aa_to_col:
                rids.append(int(rid_val))
                preds.append(float(ddg_mat[pos, aa_to_col[m_aa]]))

    if not rids and n_attempted > 0 and last_err is not None:
        raise last_err
    return np.asarray(rids, np.int64), np.asarray(preds, np.float32)


def _try_upstream_spurs_external(
    repo_dir: Path,
    ext_df: pd.DataFrame,
    pdb_map: dict[str, Path],
    device: torch.device,
    *,
    raise_on_error: bool = False,
    loaded_bundle: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Attempt official ``luo-group/SPURS`` checkpoint inference on an external table.

    Sign convention: SPURS upstream predicts MegaScale-calibrated stabilizing-positive
    ``ddG`` (``:+1``, matching this repository's convention).

    Returns ``(row_ids, y_pred)`` on success, or ``None`` if upstream API/weights
    are unavailable (unless ``raise_on_error=True``, in which case the exact
    exception propagates for structured diagnostics in ``official_spurs_status.json``).
    """
    if repo_dir.is_dir() and str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))

    if loaded_bundle is None:
        try:
            from spurs.inference import get_SPURS_from_hub, parse_pdb  # type: ignore
        except Exception as exc:
            if raise_on_error:
                raise
            print(
                f"[spurs] upstream spurs.inference import unavailable ({exc}); "
                f"falling back to local_fusion retrain.",
                flush=True,
            )
            return None

        try:
            with _torch_load_weights_only_false():
                model, cfg = get_SPURS_from_hub()
            model = model.to(device).eval()
        except Exception as exc:
            if raise_on_error:
                raise
            print(
                f"[spurs] upstream SPURS hub load failed ({exc}); "
                f"falling back to local_fusion retrain.",
                flush=True,
            )
            return None
    else:
        model, cfg, parse_pdb = loaded_bundle

    try:
        rids, preds = _score_upstream_spurs_table(model, cfg, parse_pdb, ext_df, pdb_map)
    except Exception as exc:
        if raise_on_error:
            raise
        print(
            f"[spurs] upstream SPURS inference failed ({exc}); "
            f"falling back to local_fusion retrain.",
            flush=True,
        )
        return None
    if len(rids) == 0:
        return None
    return rids, preds


def _run_official_upstream_only(args, device: torch.device, out_dir: Path) -> int:
    """
    Execute ``--mode upstream --official-external-only`` strictly without ever
    falling through to MegaScale local_fusion retraining. Always writes
    ``official_spurs_status.json`` with full diagnostics and returns a clean
    non-zero exit code on failure.
    """
    repo_dir = Path(args.work_dir)
    if not args.external or not args.external_pdb_dir:
        msg = "--mode upstream --official-external-only requires --external and --external-pdb-dir"
        sp = _write_official_spurs_status(
            out_dir, status="failed", stage="args_validation", repo_dir=repo_dir, message=msg
        )
        print(f"[spurs] ERROR: {msg} (diagnostics -> {sp})", file=sys.stderr, flush=True)
        return 1
    if not os.path.isfile(args.external) or not os.path.isdir(args.external_pdb_dir):
        msg = f"Missing external table ({args.external}) or PDB dir ({args.external_pdb_dir})"
        sp = _write_official_spurs_status(
            out_dir, status="failed", stage="args_validation", repo_dir=repo_dir, message=msg
        )
        print(f"[spurs] ERROR: {msg} (diagnostics -> {sp})", file=sys.stderr, flush=True)
        return 1

    if not (repo_dir / "spurs").is_dir() and not (repo_dir / ".git").exists():
        try:
            repo_dir = clone_repo(SPURS_URL, repo_dir)
        except Exception as exc:
            sp = _write_official_spurs_status(
                out_dir, status="failed", stage="clone_repo", repo_dir=repo_dir, exc=exc
            )
            print(
                f"[spurs] ERROR: failed to clone upstream SPURS ({type(exc).__name__}: {exc}); "
                f"diagnostics -> {sp}",
                file=sys.stderr,
                flush=True,
            )
            return 1

    if repo_dir.is_dir() and str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))

    try:
        from spurs.inference import get_SPURS_from_hub, parse_pdb  # type: ignore
    except Exception as exc:
        sp = _write_official_spurs_status(
            out_dir, status="failed", stage="import_spurs", repo_dir=repo_dir, exc=exc
        )
        print(
            f"[spurs] ERROR: cannot import spurs.inference ({type(exc).__name__}: {exc}); "
            f"diagnostics -> {sp}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    try:
        with _torch_load_weights_only_false():
            model, cfg = get_SPURS_from_hub()
        model = model.to(device).eval()
    except Exception as exc:
        sp = _write_official_spurs_status(
            out_dir, status="failed", stage="load_model", repo_dir=repo_dir, exc=exc
        )
        print(
            f"[spurs] ERROR: get_SPURS_from_hub() failed ({type(exc).__name__}: {exc}); "
            f"diagnostics -> {sp}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    bundle = (model, cfg, parse_pdb)
    written: dict[str, int] = {}
    try:
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        t_inf = time.time()
        res = _try_upstream_spurs_external(
            repo_dir, ext_df, ext_pdbs, device, raise_on_error=True, loaded_bundle=bundle
        )
        if res is None or len(res[0]) == 0:
            msg = f"Upstream SPURS produced 0 predictions on {args.external}"
            sp = _write_official_spurs_status(
                out_dir, status="failed", stage="inference", repo_dir=repo_dir, message=msg
            )
            print(f"[spurs] ERROR: {msg} (diagnostics -> {sp})", file=sys.stderr, flush=True)
            return 1
        write_predictions(
            out_dir / "pred_external_official.parquet",
            res[0], res[1],
            meta={
                "model": "SPURS_official",
                "implementation": "upstream_luo_group_SPURS",
                "sign": "stabilizing_positive (:+1)",
                "inference_s": round(time.time() - t_inf, 2),
                "peak_vram_gb": gpu_peak_gb(),
            },
        )
        written["pred_external_official.parquet"] = int(len(res[0]))

        for spec in args.extra_externals:
            tag, epq, epdb = parse_extra_external_spec(spec)
            if os.path.isfile(epq) and epdb and os.path.isdir(epdb):
                edf = load_external_table(epq)
                emap = _resolve_pdb_map(epdb)
                t_i = time.time()
                r2 = _try_upstream_spurs_external(
                    repo_dir, edf, emap, device, raise_on_error=True, loaded_bundle=bundle
                )
                r2_ids = r2[0] if r2 is not None else np.empty((0,), dtype=np.int64)
                r2_preds = r2[1] if r2 is not None else np.empty((0,), dtype=np.float32)
                fname = f"pred_{tag}_official.parquet"
                write_predictions(
                    out_dir / fname,
                    r2_ids, r2_preds,
                    meta={
                        "model": "SPURS_official",
                        "implementation": "upstream_luo_group_SPURS",
                        "sign": "stabilizing_positive (:+1)",
                        "inference_s": round(time.time() - t_i, 2),
                        "peak_vram_gb": gpu_peak_gb(),
                    },
                )
                written[fname] = int(len(r2_ids))
            else:
                print(
                    f"[spurs] WARNING: skipping extra external {tag!r} in upstream official mode "
                    f"(missing parquet {epq!r} or pdb_dir {epdb!r})",
                    flush=True,
                )
    except Exception as exc:
        sp = _write_official_spurs_status(
            out_dir, status="failed", stage="inference", repo_dir=repo_dir, exc=exc,
            extra={"written": written},
        )
        print(
            f"[spurs] ERROR: upstream SPURS inference failed ({type(exc).__name__}: {exc}); "
            f"diagnostics -> {sp}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    _write_official_spurs_status(
        out_dir, status="ok", stage="complete", repo_dir=repo_dir,
        message="Upstream SPURS official external inference succeeded",
        extra={"written": written},
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SPURS competitor adapter (Nature Comms 2026)")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--pairs", default="data/megascale_ddg_pairs.parquet")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--external", default=None)
    ap.add_argument("--external-pdb-dir", default=None)
    ap.add_argument("--extra-externals", nargs="*", default=[],
                    help="Additional external holdouts as tag=parquet or tag=parquet::pdb_dir")
    ap.add_argument("--mode", choices=["local_fusion", "upstream"], default="local_fusion")
    ap.add_argument("--work-dir", default="/tmp/competitors/SPURS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--official-external-only", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # BUG G guard: when --mode upstream --official-external-only is requested,
    # NEVER fall through to MegaScale local_fusion retrain on error.
    if args.mode == "upstream" and args.official_external_only:
        return _run_official_upstream_only(args, device, out_dir)

    if args.mode == "upstream" and args.external and args.external_pdb_dir:
        repo_dir = Path(args.work_dir)
        try:
            if not (repo_dir / "spurs").is_dir() and not (repo_dir / ".git").exists():
                repo_dir = clone_repo(SPURS_URL, repo_dir)
            ext_df = load_external_table(args.external)
            ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
            t_inf = time.time()
            res = _try_upstream_spurs_external(repo_dir, ext_df, ext_pdbs, device)
            if res is not None:
                write_predictions(
                    out_dir / "pred_external_official.parquet",
                    res[0], res[1],
                    meta={
                        "model": "SPURS_official",
                        "implementation": "upstream_luo_group_SPURS",
                        "sign": "stabilizing_positive (:+1)",
                        "inference_s": round(time.time() - t_inf, 2),
                        "peak_vram_gb": gpu_peak_gb(),
                    },
                )
        except Exception as exc:
            print(
                f"[spurs] upstream external attempt failed ({exc}); "
                f"continuing with local_fusion retrain.",
                flush=True,
            )

    if not args.cache:
        ap.error("--cache is required unless --mode upstream --official-external-only is set")

    store = EncodingStore(args.cache, cache_records=512)
    splits = load_split_table(args.splits)
    if args.pairs and os.path.isfile(args.pairs) and "parent" not in splits.columns:
        pairs = pd.read_parquet(args.pairs, columns=["parent", "wt_seq"]).drop_duplicates("wt_seq")
        pairs["wt_seq"] = pairs["wt_seq"].astype(str).str.strip().str.upper()
        splits = splits.merge(pairs, left_on="parent_seq", right_on="wt_seq", how="left")
    ms_pdbs = _resolve_pdb_map(args.pdb_dir)

    tr = _build_local_fusion_arrays(splits[splits["split"] == "train"], store, ms_pdbs)
    if len(tr[4]) == 0:
        raise RuntimeError("SPURS: no training rows matched cached parent sequences")
    esm_dim = int(tr[0].shape[-1])
    va = _build_local_fusion_arrays(splits[splits["split"] == "val"], store, ms_pdbs, esm_dim=esm_dim)
    te = _build_local_fusion_arrays(splits[splits["split"] == "test"], store, ms_pdbs, esm_dim=esm_dim)

    model = SpursFusionHead(esm_dim=esm_dim, struct_dim=tr[1].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    best_r = -1e9
    best_state = None
    rng = np.random.default_rng(args.seed)
    n_tr = len(tr[4])
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for lo in range(0, n_tr, args.batch_size):
            idx = torch.from_numpy(perm[lo:lo + args.batch_size]).long()
            opt.zero_grad(set_to_none=True)
            pred = model(tr[0][idx].to(device), tr[1][idx].to(device),
                         tr[2][idx].to(device), tr[3][idx].to(device)).reshape(-1)
            loss = loss_fn(pred, tr[4][idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp = []
        with torch.no_grad():
            for lo in range(0, len(va[4]), 1024):
                vp.append(model(va[0][lo:lo + 1024].to(device), va[1][lo:lo + 1024].to(device),
                                va[2][lo:lo + 1024].to(device), va[3][lo:lo + 1024].to(device))
                          .reshape(-1).detach().cpu().float().numpy())
        vp_arr = np.concatenate(vp) if vp else np.empty((0,), dtype=np.float32)
        r = safe_pearson(va[4].numpy(), vp_arr)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def _predict_tuple(tup):
        preds = []
        with torch.no_grad():
            for lo in range(0, len(tup[4]), 1024):
                preds.append(model(tup[0][lo:lo + 1024].to(device), tup[1][lo:lo + 1024].to(device),
                                   tup[2][lo:lo + 1024].to(device), tup[3][lo:lo + 1024].to(device))
                             .reshape(-1).detach().cpu().float().numpy())
        return np.concatenate(preds).astype(np.float32) if preds else np.empty((0,), dtype=np.float32)

    write_predictions(
        out_dir / "pred_test.parquet",
        te[5], _predict_tuple(te),
        meta={
            "model": "SPURS_retrained_fusion",
            "implementation": "retrained_spurs_architecture_esm2_plus_backbone3d",
            "sign": "stabilizing_positive (:+1)",
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "best_val_pearson": round(best_r, 4) if best_state is not None and np.isfinite(best_r) else None,
            "wall_clock_s": round(time.time() - t0, 2),
            "peak_vram_gb": gpu_peak_gb(),
            "seed": args.seed,
        },
    )

    if args.external and os.path.isfile(args.external):
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        par_col = "parent" if "parent" in ext_df.columns else "uniprot_id"
        ex = _build_local_fusion_arrays(ext_df, store, ext_pdbs, par_col=par_col, esm_dim=esm_dim)
        write_predictions(
            out_dir / "pred_external.parquet",
            ex[5], _predict_tuple(ex),
            meta={
                "model": "SPURS_retrained_fusion",
                "implementation": "retrained_spurs_architecture_esm2_plus_backbone3d",
                "sign": "stabilizing_positive (:+1)",
                "seed": args.seed,
            },
        )
    for spec in args.extra_externals:
        tag, epq, epdb = parse_extra_external_spec(spec)
        if os.path.isfile(epq):
            edf = load_external_table(epq)
            emap = _resolve_pdb_map(epdb)
            pc = "parent" if "parent" in edf.columns else "uniprot_id"
            ex = _build_local_fusion_arrays(edf, store, emap, par_col=pc, esm_dim=esm_dim)
            write_predictions(
                out_dir / f"pred_{tag}.parquet",
                ex[5], _predict_tuple(ex),
                meta={"model": "SPURS_retrained_fusion", "implementation": "retrained_spurs_architecture_esm2_plus_backbone3d",
                      "sign": "stabilizing_positive (:+1)", "seed": args.seed},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
