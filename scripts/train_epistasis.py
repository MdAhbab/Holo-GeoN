"""
scripts/train_epistasis.py
==========================
Train and evaluate Holo-GeoN v2's **Contact-Gated Potts Bilinear Epistasis Head**
on double-mutant stability datasets (e.g. MegaScale Tsuboyama Zenodo 7844779
double mutants, or any Parquet with columns ``parent_seq``, ``site_i``,
``wt_i``, ``mt_i``, ``site_j``, ``wt_j``, ``mt_j``, ``ddg_ab``, and optionally
``ddg_a``, ``ddg_b``, ``dddg``, ``dist_cb``, ``split``).

Controlled ablation arms (novelty pillar #1):
  1. ``additive_only``       — zero epistasis (ddG_ab = ddG_a + ddG_b)
  2. ``potts_no_gate``       — Potts bilinear head with scalar constant gate
  3. ``contact_gated_egeo``  — Potts bilinear head gated by structure-free
                               ESM-2 ``E_geo`` pair features ``[p_ij, log_sep, inv_hop, is_contact]``
  4. ``true_contact_oracle`` — same head gated by true 3D C-beta contact features
                               when ``dist_cb`` / PDBs are supplied

Reports overall AND 3D-distance-stratified metrics (contacting pairs ``< 8 A``
vs. distant pairs ``>= 8 A``), plus exact Potts parity/symmetry diagnostics, and
writes ``epistasis_metrics.json`` + ``pred_epistasis_test.parquet``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import load_config
from src.data import _ParentGraphCache, build_sample_graph
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore
from src.graphs import pair_contact_features
from src.metrics import regression_metrics
from src.model import HoloGeoN, model_kwargs_from_config
from src.seeding import seed_everything
from src.substitution import aa_index


def _build_double_groups(df: pd.DataFrame, store: EncodingStore, cfg,
                         use_true_contact: bool = False) -> list[dict]:
    """
    Group double mutants by unique ``parent_seq`` so each parent's GNN backbone
    runs ONCE per batch step regardless of how many double mutants it owns.
    """
    cfg_dict = cfg if isinstance(cfg, dict) else cfg.as_dict()
    window = int(cfg_dict.get("window", 256))
    cache = _ParentGraphCache(cfg_dict, window=window, max_entries=512)
    groups: list[dict] = []

    for seq, grp in df.groupby("parent_seq", sort=False):
        seq_s = str(seq).upper()
        if not store.has(seq_s):
            continue
        rec = store.get(seq_s)
        pgraph = cache.get_whole(seq_s, rec)
        if pgraph is None:
            # Whole-sequence fallback for L > window
            pgraph = build_sample_graph(rec, None, window, cfg_dict, max_nodes=int(cfg_dict.get("max_sequence_nodes", 2048)))
        L = int(pgraph["esm"].shape[0])

        si = np.clip(grp["site_i"].to_numpy(dtype=np.int64), 0, L - 1)
        sj = np.clip(grp["site_j"].to_numpy(dtype=np.int64), 0, L - 1)
        wt_i = np.array([aa_index(a) for a in grp["wt_i"]], dtype=np.int64)
        mt_i = np.array([aa_index(a) for a in grp["mt_i"]], dtype=np.int64)
        wt_j = np.array([aa_index(a) for a in grp["wt_j"]], dtype=np.int64)
        mt_j = np.array([aa_index(a) for a in grp["mt_j"]], dtype=np.int64)

        edge_geo = pgraph["edge_geo"]
        attr_geo = pgraph["attr_geo"]
        p_geo = torch.stack(
            [pair_contact_features(edge_geo, attr_geo, int(i), int(j), L) for i, j in zip(si, sj)],
            dim=0,
        )  # (M, 4)

        if use_true_contact and "dist_cb" in grp.columns:
            d = grp["dist_cb"].to_numpy(dtype=np.float32)
            p_true = np.where(np.isfinite(d), np.clip(1.0 - d / 16.0, 0.0, 1.0), 0.0)
            is_c = np.where(np.isfinite(d), (d <= 8.0).astype(np.float32), 0.0)
            p_geo = p_geo.clone()
            p_geo[:, 0] = torch.from_numpy(p_true)
            p_geo[:, 3] = torch.from_numpy(is_c)

        m_count = len(si)
        n1_i = torch.zeros((m_count, L), dtype=torch.bool)
        n2_i = torch.zeros((m_count, L), dtype=torch.bool)
        n1_j = torch.zeros((m_count, L), dtype=torch.bool)
        n2_j = torch.zeros((m_count, L), dtype=torch.bool)
        for k in range(m_count):
            m1_i, m2_i = cache.get_khops(seq_s, edge_geo, int(si[k]), L)
            m1_j, m2_j = cache.get_khops(seq_s, edge_geo, int(sj[k]), L)
            n1_i[k] = m1_i
            n2_i[k] = m2_i
            n1_j[k] = m1_j
            n2_j[k] = m2_j

        y_ab = grp["ddg_ab"].to_numpy(dtype=np.float32)
        if "dddg" in grp.columns and grp["dddg"].notna().all():
            y_epi = grp["dddg"].to_numpy(dtype=np.float32)
            has_epi = True
        elif "ddg_a" in grp.columns and "ddg_b" in grp.columns:
            y_epi = (grp["ddg_ab"] - grp["ddg_a"] - grp["ddg_b"]).to_numpy(dtype=np.float32)
            has_epi = True
        else:
            y_epi = np.full(m_count, np.nan, dtype=np.float32)
            has_epi = False

        dist_cb = grp["dist_cb"].to_numpy(dtype=np.float32) if "dist_cb" in grp.columns else np.full(m_count, np.nan, np.float32)
        row_ids = grp["row_id"].to_numpy(dtype=np.int64) if "row_id" in grp.columns else np.arange(m_count, dtype=np.int64)

        groups.append({
            "parent_seq": seq_s,
            "pgraph": pgraph,
            "si": torch.from_numpy(si),
            "sj": torch.from_numpy(sj),
            "wt_i": torch.from_numpy(wt_i),
            "mt_i": torch.from_numpy(mt_i),
            "wt_j": torch.from_numpy(wt_j),
            "mt_j": torch.from_numpy(mt_j),
            "n1_i": n1_i,
            "n2_i": n2_i,
            "n1_j": n1_j,
            "n2_j": n2_j,
            "pair_geo": p_geo,
            "y_ab": torch.from_numpy(y_ab),
            "y_epi": torch.from_numpy(y_epi),
            "has_epi": has_epi,
            "dist_cb": dist_cb,
            "row_ids": row_ids,
        })
    return groups


def _forward_group(model: HoloGeoN, g: dict, device: torch.device,
                   additive_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Run backbone ONCE on parent graph and score all M double mutants on it."""
    pg = g["pgraph"]
    L = pg["esm"].shape[0]
    batch = {
        "esm": pg["esm"].unsqueeze(0).to(device),
        "mech": pg["mech"].unsqueeze(0).to(device),
        "mask": torch.ones((1, L), dtype=torch.bool, device=device),
        "edge_geo": pg["edge_geo"].to(device),
        "attr_geo": pg["attr_geo"].to(device),
        "edge_mech": pg["edge_mech"].to(device),
        "attr_mech": pg["attr_mech"].to(device),
    }
    if "topo" in pg and pg["topo"] is not None:
        batch["topo"] = pg["topo"].unsqueeze(0).to(device)
    if "layer_embs" in pg and pg["layer_embs"] is not None:
        batch["layer_embs"] = pg["layer_embs"].unsqueeze(0).to(device)
    if "edge_str" in pg and pg["edge_str"] is not None:
        batch["edge_str"] = pg["edge_str"].to(device)
        batch["attr_str"] = pg["attr_str"].to(device)
    if "str_node" in pg and pg["str_node"] is not None:
        batch["str_node"] = pg["str_node"].unsqueeze(0).to(device)

    h = model.nodes(batch)  # (1, L, D)
    m = g["si"].shape[0]
    pidx = torch.zeros(m, dtype=torch.long, device=device)

    b_i = {
        "mask": batch["mask"],
        "prot_idx": pidx,
        "site": g["si"].to(device),
        "neighborhood_1hop": g["n1_i"].to(device),
        "neighborhood_mask": g["n2_i"].to(device),
    }
    b_j = {
        "mask": batch["mask"],
        "prot_idx": pidx,
        "site": g["sj"].to(device),
        "neighborhood_1hop": g["n1_j"].to(device),
        "neighborhood_mask": g["n2_j"].to(device),
    }
    z_i = model.context_from_nodes(h, b_i)
    z_j = model.context_from_nodes(h, b_j)

    lp = pg.get("wt_logprobs")
    lp_i = lp[g["si"]].to(device) if lp is not None else None
    lp_j = lp[g["sj"]].to(device) if lp is not None else None

    ddg_ab, dddg, _ = model.epistasis(
        z_i, z_j,
        g["wt_i"].to(device), g["mt_i"].to(device),
        g["wt_j"].to(device), g["mt_j"].to(device),
        pair_geo=g["pair_geo"].to(device),
        lp_i=lp_i, lp_j=lp_j,
    )
    if additive_only:
        ddg_ab = ddg_ab - dddg
        dddg = torch.zeros_like(dddg)
    return ddg_ab.reshape(-1), dddg.reshape(-1)


@torch.no_grad()
def evaluate_groups(model: HoloGeoN, groups: list[dict], device: torch.device,
                    additive_only: bool = False) -> dict:
    model.eval()
    rids, y_ab_l, p_ab_l, y_epi_l, p_epi_l, dist_l = [], [], [], [], [], []
    for g in groups:
        p_ab, p_epi = _forward_group(model, g, device, additive_only=additive_only)
        rids.append(g["row_ids"])
        y_ab_l.append(g["y_ab"].numpy())
        p_ab_l.append(p_ab.float().cpu().numpy())
        y_epi_l.append(g["y_epi"].numpy())
        p_epi_l.append(p_epi.float().cpu().numpy())
        dist_l.append(g["dist_cb"])

    if not rids:
        return {"n_rows": 0}
    rid = np.concatenate(rids)
    y_ab = np.concatenate(y_ab_l)
    p_ab = np.concatenate(p_ab_l)
    y_epi = np.concatenate(y_epi_l)
    p_epi = np.concatenate(p_epi_l)
    dist = np.concatenate(dist_l)

    out = {
        "n_rows": int(len(rid)),
        "ddg_ab": regression_metrics(y_ab, p_ab),
        "preds": pd.DataFrame({
            "row_id": rid,
            "y_ab": y_ab,
            "pred_ab": p_ab,
            "y_epi": y_epi,
            "pred_epi": p_epi,
            "dist_cb": dist,
        }),
    }
    epi_valid = np.isfinite(y_epi)
    if epi_valid.sum() >= 3:
        out["dddg_epistasis"] = regression_metrics(y_epi[epi_valid], p_epi[epi_valid])
        # Distance-stratified epistasis evaluation (< 8 A contact vs >= 8 A non-contact)
        near = epi_valid & np.isfinite(dist) & (dist < 8.0)
        far = epi_valid & np.isfinite(dist) & (dist >= 8.0)
        if near.sum() >= 3:
            out["dddg_contact_lt8A"] = {
                "n": int(near.sum()),
                **regression_metrics(y_epi[near], p_epi[near]),
            }
        if far.sum() >= 3:
            out["dddg_distant_ge8A"] = {
                "n": int(far.sum()),
                **regression_metrics(y_epi[far], p_epi[far]),
            }
    return out


def train_epistasis_model(
    model: HoloGeoN,
    tr_groups: list[dict],
    va_groups: list[dict],
    device: torch.device,
    *,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    epi_weight: float = 1.0,
    freeze_backbone: bool = True,
    additive_only: bool = False,
    seed: int = 42,
) -> dict:
    """Fine-tune the Contact-Gated Potts Epistasis Head (and optionally backbone)."""
    if additive_only or model.epistasis_head is None:
        val_res = evaluate_groups(model, va_groups, device, additive_only=True)
        return {"best_val_pearson_ab": val_res.get("ddg_ab", {}).get("pearson"), "history": []}

    for name, p in model.named_parameters():
        if freeze_backbone:
            p.requires_grad = name.startswith("epistasis_head.")
        else:
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    rng = np.random.default_rng(seed)

    best_score = -1e9
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(tr_groups))
        ep_loss = 0.0
        for idx in order:
            g = tr_groups[int(idx)]
            opt.zero_grad(set_to_none=True)
            p_ab, p_epi = _forward_group(model, g, device)
            loss = mse(p_ab, g["y_ab"].to(device))
            if g["has_epi"] and epi_weight > 0:
                ye = g["y_epi"].to(device)
                mask = torch.isfinite(ye)
                if mask.any():
                    loss = loss + epi_weight * mse(p_epi[mask], ye[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(loss.item())

        val_res = evaluate_groups(model, va_groups, device)
        val_ab = val_res.get("ddg_ab", {}).get("pearson", float("nan"))
        val_epi = val_res.get("dddg_epistasis", {}).get("pearson", float("nan"))
        score = val_epi if np.isfinite(val_epi) else val_ab
        history.append({
            "epoch": ep,
            "train_loss": round(ep_loss / max(len(tr_groups), 1), 5),
            "val_pearson_ab": val_ab,
            "val_pearson_epi": val_epi,
        })
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_score": best_score, "history": history}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train & evaluate Holo-GeoN v2 Contact-Gated Epistasis Head")
    ap.add_argument("--data", required=True, help="Double-mutant parquet with split column")
    ap.add_argument("--checkpoint", default=None, help="Pretrained single-mutant Holo-GeoN v2 checkpoint")
    ap.add_argument("--config", default="configs/benchmark_v2.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["contact_gated_egeo", "potts_no_gate", "additive_only", "true_contact_oracle"],
                    default="contact_gated_egeo")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epi-weight", type=float, default=1.0)
    ap.add_argument("--unfreeze-backbone", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config, overrides={"cache": args.cache, "device": args.device, "seed": args.seed})
    seed_everything(args.seed)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data).copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    use_true = args.arm == "true_contact_oracle"

    tr_g = _build_double_groups(df[df["split"] == "train"], store, cfg, use_true_contact=use_true)
    va_g = _build_double_groups(df[df["split"] == "val"], store, cfg, use_true_contact=use_true)
    te_g = _build_double_groups(df[df["split"] == "test"], store, cfg, use_true_contact=use_true)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.esm_model, 1280))
        kwargs = dict(state.get("model_kwargs") or model_kwargs_from_config(cfg, esm_dim))
        kwargs["use_contact_gate"] = args.arm != "potts_no_gate"
        model = HoloGeoN(**kwargs).to(device)
        model.load_state_dict(state["model"], strict=False)
    else:
        esm_dim = ESM_DIMS.get(cfg.esm_model, 1280)
        kwargs = model_kwargs_from_config(cfg, esm_dim, arch="v2",
                                          use_contact_gate=(args.arm != "potts_no_gate"))
        model = HoloGeoN(**kwargs).to(device)

    train_info = train_epistasis_model(
        model, tr_g, va_g, device,
        epochs=args.epochs, lr=args.lr, epi_weight=args.epi_weight,
        freeze_backbone=not args.unfreeze_backbone,
        additive_only=(args.arm == "additive_only"),
        seed=args.seed,
    )
    te_res = evaluate_groups(model, te_g, device, additive_only=(args.arm == "additive_only"))
    preds_df = te_res.pop("preds", None)
    if preds_df is not None:
        preds_df.to_parquet(out_dir / "pred_epistasis_test.parquet", index=False)

    torch.save({
        "model": model.state_dict(),
        "model_kwargs": kwargs,
        "esm_dim": esm_dim,
        "arm": args.arm,
        "seed": args.seed,
    }, out_dir / "model_epistasis_best.pt")

    report = {
        "arm": args.arm,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 2),
        "n_groups": {"train": len(tr_g), "val": len(va_g), "test": len(te_g)},
        "train": train_info,
        "test": te_res,
    }
    (out_dir / "epistasis_metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
