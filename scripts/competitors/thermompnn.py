"""
scripts/competitors/thermompnn.py
=================================
Competitor adapter for **ThermoMPNN** (Dieckhaus et al., PNAS 2024;
``Kuhlman-Lab/ThermoMPNN``, MIT licence).

Protocol (BENCHMARK_PLAN.md §2):
  * Clones ``Kuhlman-Lab/ThermoMPNN`` into ``--work-dir`` (appended to
    sys.path, never inserted at 0).
  * **MegaScale test:** retrains ThermoMPNN's stability head on OUR exact train
    split (frozen ProteinMPNN backbone) using our aligned MegaScale AlphaFold
    PDBs — official weights are NOT scored on MegaScale test because their
    official split overlaps ours.
  * **External (FireProtDB / S669):** evaluates BOTH our retrained checkpoint
    AND ThermoMPNN's official released checkpoint (``models/thermoMPNN_default.pt``)
    using fetched AlphaFold/crystal PDBs.
  * Exports common-format ``(row_id, y_pred)`` Parquets with declared sign
    ``:-1`` (ThermoMPNN native convention is destabilizing-positive, OR ``:+1``
    when trained directly on our stabilizing-positive targets — explicitly
    recorded in ``.meta.json``).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
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
    write_predictions,
)
from src.geometry import (
    NAME_RE,
    align_to_sequence,
    normalize_pdb_stem,
    parse_pdb_cb,
    resolve_pdb_path,
)
from src.substitution import AA_ORDER, aa_index

THERMOMPNN_URL = "https://github.com/Kuhlman-Lab/ThermoMPNN.git"


def _resolve_pdb_map(pdb_dir: str) -> dict[str, Path]:
    return {
        normalize_pdb_stem(p.name): p
        for p in Path(pdb_dir).rglob("*.pdb")
    }


def _match_pdb(parent: str, pdb_map: dict[str, Path]) -> Path | None:
    hit = resolve_pdb_path(str(parent), pdb_map)
    return Path(hit) if hit is not None else None


def _strip_lightning_state_dict(raw_ckpt: dict) -> dict[str, torch.Tensor]:
    """Extract a ``TransferModel`` state_dict from a Lightning or raw checkpoint."""
    sd = raw_ckpt["state_dict"] if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt else raw_ckpt
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("metrics."):
            continue
        cleaned[k[6:] if k.startswith("model.") else k] = v
    return cleaned


def _forward_thermompnn_head(
    model: nn.Module,
    res_feats: torch.Tensor,
    wt_idx: torch.Tensor,
    mt_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized exact equivalent of ``TransferModel.forward``'s trainable head
    (``light_attention`` + ``both_out`` + ``ddg_out`` + ``subtract_mut``) over
    a batch of pre-extracted frozen ProteinMPNN residue features ``(B, D)``.
    """
    x = res_feats
    if getattr(model, "lightattn", False):
        x_in = x.unsqueeze(-1)  # (B, D, 1)
        la = model.light_attention
        o = la.dropout(la.feature_convolution(x_in))
        attn = la.attention_convolution(x_in)
        x = (o * la.softmax(attn)).squeeze(-1)  # (B, D)
    both_input = model.both_out(x).unsqueeze(-1)  # (B, 21, 1)
    ddg_out = model.ddg_out(both_input).squeeze(-1)  # (B, 21)
    idx = torch.arange(ddg_out.size(0), device=ddg_out.device)
    if getattr(model, "subtract_mut", True):
        return ddg_out[idx, mt_idx] - ddg_out[idx, wt_idx]
    return ddg_out[idx, mt_idx]


def _extract_mpnn_residue_features(
    model: nn.Module,
    pdb_list: list[dict],
    tied_featurize,
    device: torch.device,
) -> torch.Tensor:
    """Run ThermoMPNN's frozen ProteinMPNN backbone ONCE for one PDB -> ``(L, D)``."""
    with torch.no_grad():
        feat = tied_featurize(
            [pdb_list[0]], device, None, None, None, None, None, None, ca_only=False
        )
        X, S, mask, _lengths, chain_M, chain_encoding_all = feat[0], feat[1], feat[2], feat[3], feat[4], feat[5]
        residue_idx = feat[12]
        all_mpnn_hid, mpnn_embed, _ = model.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all, None
        )
        parts = []
        n_final = int(getattr(model, "num_final_layers", 2))
        if n_final > 0:
            parts.append(torch.cat(all_mpnn_hid[:n_final], dim=-1))
        parts.append(mpnn_embed)
        return torch.cat(parts, dim=-1)[0].detach().cpu()


class _AttrDict(dict):
    """Lightweight dict supporting attribute access + ``in`` checks (no omegaconf needed)."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


def _load_thermompnn_model(repo_dir: Path, device: torch.device):
    """
    Instantiate ThermoMPNN's pure PyTorch ``TransferModel(cfg)`` (no
    ``pytorch_lightning``, ``wandb``, or ``omegaconf`` needed) and load the
    official checkpoint ``models/thermoMPNN_default.pt`` safely under PyTorch 2.x.
    """
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    from protein_mpnn_utils import alt_parse_PDB, tied_featurize  # type: ignore
    from transfer_model import ALPHABET as TM_ALPHABET, TransferModel  # type: ignore

    cfg = _AttrDict(
        platform=_AttrDict(thermompnn_dir=str(repo_dir)),
        model=_AttrDict(
            hidden_dims=[64, 32],
            subtract_mut=True,
            num_final_layers=2,
            freeze_weights=True,
            load_pretrained=True,
            lightattn=True,
            lr=1e-3,
        ),
        training=_AttrDict(num_workers=0, learn_rate=1e-3, epochs=10),
    )
    ckpt = repo_dir / "models" / "thermoMPNN_default.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Expected ThermoMPNN checkpoint at {ckpt}")

    # PyTorch 2.6+ defaults torch.load(weights_only=True); if upstream
    # get_protein_mpnn() hits an unpickling error on vanilla weights, temporarily
    # pass weights_only=False only for that load.
    try:
        model = TransferModel(cfg)
    except Exception:
        _orig_load = torch.load
        try:
            def _load_compat(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig_load(*a, **kw)
            torch.load = _load_compat  # type: ignore[assignment]
            model = TransferModel(cfg)
        finally:
            torch.load = _orig_load

    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(_strip_lightning_state_dict(raw), strict=True)
    model.eval().to(device)
    return model, alt_parse_PDB, tied_featurize, TM_ALPHABET


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ThermoMPNN competitor adapter")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--pairs", default="data/megascale_ddg_pairs.parquet")
    ap.add_argument("--pdb-dir", required=True, help="MegaScale AlphaFold PDB directory")
    ap.add_argument("--external", default=None, help="Optional FireProtDB holdout parquet")
    ap.add_argument("--external-pdb-dir", default=None, help="FireProtDB AlphaFold PDB dir")
    ap.add_argument("--work-dir", default="/tmp/competitors/ThermoMPNN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--official-external-only", action="store_true",
                    help="Score ThermoMPNN official checkpoint on external holdout only (no retrain)")
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = clone_repo(THERMOMPNN_URL, Path(args.work_dir))
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))

    model, alt_parse_PDB, tied_featurize, tm_alphabet = _load_thermompnn_model(repo_dir, device)
    aa_to_tm = {aa: idx for idx, aa in enumerate(tm_alphabet)}

    def _score_table_direct(df: pd.DataFrame, pdb_map: dict[str, Path],
                            par_col: str, seq_col: str) -> tuple[np.ndarray, np.ndarray]:
        rids, preds = [], []
        for parent, grp in df.groupby(par_col, sort=False):
            pfile = _match_pdb(str(parent), pdb_map)
            if pfile is None:
                continue
            pdb_list = alt_parse_PDB(str(pfile))
            if not pdb_list or "seq" not in pdb_list[0]:
                continue
            pdb_seq = str(pdb_list[0]["seq"]).upper()
            wt_seq = str(grp[seq_col].iloc[0]).upper()
            off, _, _err = align_to_sequence(pdb_seq, wt_seq)
            if off is None:
                continue
            mut_list = []
            valid_rids = []
            for r in grp.itertuples(index=False):
                site = int(r.site) + off
                w, m_aa = str(r.wt_aa).upper(), str(r.mt_aa).upper()
                if 0 <= site < len(pdb_seq) and w in aa_to_tm and m_aa in aa_to_tm:
                    mut_list.append(type("M", (), {"position": site, "wildtype": w, "mutation": m_aa})())
                    valid_rids.append(int(r.row_id))
            if not mut_list:
                continue
            with torch.no_grad():
                out_list, _ = model(pdb_list, mut_list)
            for rid, item in zip(valid_rids, out_list):
                rids.append(rid)
                preds.append(float(item["ddG"].item()))
        return np.asarray(rids, np.int64), np.asarray(preds, np.float32)

    if args.external and args.external_pdb_dir:
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        par_col = "parent" if "parent" in ext_df.columns else "uniprot_id"
        seq_col = "wt_seq" if "wt_seq" in ext_df.columns else "parent_seq"
        t_inf = time.time()
        e_rids, e_preds = _score_table_direct(ext_df, ext_pdbs, par_col, seq_col)
        write_predictions(
            out_dir / "pred_external_official.parquet",
            e_rids, e_preds,
            meta={
                "model": "ThermoMPNN_official",
                "sign": "destabilizing_positive (:-1)",
                "inference_s": round(time.time() - t_inf, 2),
                "peak_vram_gb": gpu_peak_gb(),
            },
        )
        if args.official_external_only:
            return 0

    # Retrain stability head on OUR exact MegaScale train split (frozen ProteinMPNN backbone!)
    splits = load_split_table(args.splits)
    pairs = pd.read_parquet(args.pairs, columns=["parent", "wt_seq"]).drop_duplicates("wt_seq")
    splits = splits.merge(pairs, left_on="parent_seq", right_on="wt_seq", how="left")
    ms_pdbs = _resolve_pdb_map(args.pdb_dir)

    # Re-initialize ONLY the trainable ddG head (keep frozen ProteinMPNN backbone!)
    for name, p in model.named_parameters():
        is_backbone = "prot_mpnn" in name
        p.requires_grad = not is_backbone
    # Reset head parameters cleanly so ZERO weights from ThermoMPNN's own MegaScale split remain!
    for name, mod in model.named_modules():
        if "prot_mpnn" not in name and hasattr(mod, "reset_parameters"):
            mod.reset_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def _build_split_tensors(sub_df: pd.DataFrame):
        feats_list, wt_list, mt_list, y_list, rid_list = [], [], [], [], []
        n_proteins = 0
        for parent, grp in sub_df.groupby("parent", sort=False):
            pfile = _match_pdb(str(parent), ms_pdbs)
            if pfile is None:
                continue
            pdb_list = alt_parse_PDB(str(pfile))
            if not pdb_list or "seq" not in pdb_list[0]:
                continue
            pdb_seq = str(pdb_list[0]["seq"]).upper()
            wt_seq = str(grp["parent_seq"].iloc[0]).upper()
            off, _, _err = align_to_sequence(pdb_seq, wt_seq)
            if off is None:
                continue
            res_emb = _extract_mpnn_residue_features(model, pdb_list, tied_featurize, device)
            pos_idx, w_idx, m_idx, ys, rids = [], [], [], [], []
            for r in grp.itertuples(index=False):
                site = int(r.site) + off
                w, m_aa = str(r.wt_aa).upper(), str(r.mt_aa).upper()
                if 0 <= site < res_emb.size(0) and w in aa_to_tm and m_aa in aa_to_tm:
                    pos_idx.append(site)
                    w_idx.append(aa_to_tm[w])
                    m_idx.append(aa_to_tm[m_aa])
                    ys.append(float(r.target))
                    rids.append(int(r.row_id))
            if pos_idx:
                n_proteins += 1
                feats_list.append(res_emb[torch.tensor(pos_idx, dtype=torch.long)])
                wt_list.append(torch.tensor(w_idx, dtype=torch.long))
                mt_list.append(torch.tensor(m_idx, dtype=torch.long))
                y_list.append(torch.tensor(ys, dtype=torch.float32))
                rid_list.append(np.asarray(rids, dtype=np.int64))
        if not feats_list:
            return (
                torch.empty((0, 384), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
                np.empty((0,), dtype=np.int64),
                0,
            )
        return (
            torch.cat(feats_list, dim=0),
            torch.cat(wt_list, dim=0),
            torch.cat(mt_list, dim=0),
            torch.cat(y_list, dim=0),
            np.concatenate(rid_list),
            n_proteins,
        )

    print("[thermompnn] Extracting frozen ProteinMPNN residue features for train/val/test ...", flush=True)
    tr_X, tr_w, tr_m, tr_y, _tr_r, n_tr = _build_split_tensors(splits[splits["split"] == "train"])
    va_X, va_w, va_m, va_y, _va_r, n_va = _build_split_tensors(splits[splits["split"] == "val"])
    te_X, te_w, te_m, _te_y, te_rids, n_te = _build_split_tensors(splits[splits["split"] == "test"])
    print(
        f"[thermompnn] Proteins: train={n_tr} ({len(tr_y)}), val={n_va} ({len(va_y)}), "
        f"test={n_te} ({len(te_rids)})",
        flush=True,
    )

    def _predict_batched(X: torch.Tensor, w: torch.Tensor, m: torch.Tensor) -> np.ndarray:
        preds = []
        model.eval()
        with torch.no_grad():
            for s in range(0, len(X), args.batch_size):
                e = min(s + args.batch_size, len(X))
                p = _forward_thermompnn_head(
                    model, X[s:e].to(device), w[s:e].to(device), m[s:e].to(device)
                )
                preds.append(p.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)

    best_r = -1e9
    best_state = None
    rng = np.random.default_rng(args.seed)
    n_train = len(tr_y)
    for ep in range(1, args.epochs + 1):
        model.train()
        model.prot_mpnn.eval()
        order = rng.permutation(n_train)
        for s in range(0, n_train, args.batch_size):
            idx = order[s:min(s + args.batch_size, n_train)]
            idx_t = torch.from_numpy(idx)
            opt.zero_grad(set_to_none=True)
            pred = _forward_thermompnn_head(
                model, tr_X[idx_t].to(device), tr_w[idx_t].to(device), tr_m[idx_t].to(device)
            )
            loss = loss_fn(pred, tr_y[idx_t].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        vp = _predict_batched(va_X, va_w, va_m)
        r = float(np.corrcoef(va_y.numpy(), vp)[0, 1]) if len(vp) > 1 else float("nan")
        print(f"[thermompnn] epoch {ep}/{args.epochs}  val_pearson={r:.4f}", flush=True)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    t_te = time.time()
    te_preds = _predict_batched(te_X, te_w, te_m)
    write_predictions(
        out_dir / "pred_test.parquet",
        te_rids, te_preds,
        meta={
            "model": "ThermoMPNN_retrained",
            "sign": "stabilizing_positive (:+1 — trained on our targets)",
            "trainable_params": sum(p.numel() for p in trainable),
            "best_val_pearson": round(best_r, 4),
            "wall_clock_s": round(time.time() - t0, 2),
            "test_inference_s": round(time.time() - t_te, 2),
            "peak_vram_gb": gpu_peak_gb(),
            "seed": args.seed,
        },
    )

    if args.external and args.external_pdb_dir and os.path.isfile(args.external):
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        par_col = "parent" if "parent" in ext_df.columns else "uniprot_id"
        seq_col = "wt_seq" if "wt_seq" in ext_df.columns else "parent_seq"
        er_rids, er_preds = _score_table_direct(ext_df, ext_pdbs, par_col, seq_col)
        write_predictions(
            out_dir / "pred_external_retrained.parquet",
            er_rids, er_preds,
            meta={
                "model": "ThermoMPNN_retrained",
                "sign": "stabilizing_positive (:+1 — trained on our targets)",
                "seed": args.seed,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
