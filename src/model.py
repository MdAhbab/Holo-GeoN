"""
src/model.py
============
Holo-GeoN — one model, one forward path, two versioned architectures:

* ``arch="v1"`` — legacy 20.6M dual-stream architecture preserved verbatim so
  existing checkpoints load without drift and the Kaggle ablation suite can
  directly benchmark v2 against v1 on identical splits and seeds.
* ``arch="v2"`` — compressed ~2.5M Holo-GeoN v2 architecture:
    multi-layer ESM-2 ScalarMix -> shared 1280->256 bottleneck -> FiLM
    (mechanistic + contact-burial + optional 3D node geometry) -> mask-aware
    BidirectionalMixer -> compact Residual Pre-LN GATv2 streams (E_geo, E_mech,
    optional E_str) with DropEdge -> MultiScaleSiteReadout (site + 1-hop +
    2-hop + global, supporting Protein-Batch multi-mutation gather) ->
    EvolutionarySubstitutionHead (exact antisymmetry + WT-marginal log-odds
    anchor) + ContactGatedEpistasisHead (Potts bilinear double-mutant head).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.backbone import DualStreamGNN, HoloBackboneV2
from src.encoder import DEFAULT_ESM, ESM_DIMS, N_TAP_LAYERS
from src.features import N_MECH_CHANNELS
from src.graphs import N_TOPO_CHANNELS
from src.heads import (
    ContactGatedEpistasisHead,
    EvolutionarySubstitutionHead,
    IdrEnsembleHead,
    MfiHead,
    PathogenicityHead,
    ProteomicsHead,
    SubstitutionScoreHead,
    ThreeStateHead,
)
from src.readout import MultiScaleSiteReadout, SiteAnchoredReadout
from src.substitution import AminoAcidEmbedding, N_AMINO_ACIDS


class HoloGeoN(nn.Module):
    """
    Unified Holo-GeoN model supporting both ``arch="v1"`` (legacy 20.6M) and
    ``arch="v2"`` (compressed ~2.5M evolutionary-anchored architecture).
    """

    def __init__(
        self,
        esm_dim: int = ESM_DIMS[DEFAULT_ESM],
        hidden: int = 256,
        out_dim: int = 256,
        ctx_dim: int | None = None,
        aa_dim: int = 32,
        gat_heads: int = 4,
        dropout: float = 0.1,
        use_ssm: bool = True,
        use_mech_stream: bool = True,
        use_geo_stream: bool = True,
        site_anchored_readout: bool = True,
        num_species: int = 0,
        # --- Holo-GeoN v2 / geometric ablation options ---
        arch: str = "v1",
        feat_dropout: float = 0.25,
        drop_edge: float = 0.20,
        n_tap_layers: int = N_TAP_LAYERS,
        use_multi_layer: bool = True,
        use_contact_burial: bool = True,
        use_esm_logits_prior: bool = True,
        use_str_stream: bool = False,
        str_topology_only: bool = False,
        use_contact_gate: bool = True,
    ):
        super().__init__()
        if arch not in ("v1", "v2"):
            raise ValueError(f"unknown arch {arch!r}; expected 'v1' or 'v2'")
        ctx_dim = ctx_dim or out_dim
        self.arch = arch
        self.esm_dim = esm_dim
        self.ctx_dim = ctx_dim

        if arch == "v1":
            self.backbone = DualStreamGNN(
                esm_dim=esm_dim,
                mech_dim=N_MECH_CHANNELS,
                hidden=hidden,
                out_dim=out_dim,
                heads=gat_heads,
                use_ssm=use_ssm,
                use_mech_stream=use_mech_stream,
                use_geo_stream=use_geo_stream,
                dropout=dropout,
            )
            self.readout = SiteAnchoredReadout(
                out_dim, ctx_dim, dropout=dropout, site_anchored=site_anchored_readout
            )
        else:
            self.backbone = HoloBackboneV2(
                esm_dim=esm_dim,
                mech_dim=N_MECH_CHANNELS,
                topo_dim=N_TOPO_CHANNELS,
                hidden=hidden,
                out_dim=out_dim,
                heads=gat_heads,
                n_tap_layers=n_tap_layers,
                use_multi_layer=use_multi_layer,
                use_contact_burial=use_contact_burial,
                use_ssm=use_ssm,
                use_mech_stream=use_mech_stream,
                use_geo_stream=use_geo_stream,
                use_str_stream=use_str_stream,
                str_topology_only=str_topology_only,
                dropout=dropout,
                feat_dropout=feat_dropout,
                drop_edge=drop_edge,
            )
            self.readout = MultiScaleSiteReadout(
                out_dim, ctx_dim, dropout=dropout, site_anchored=site_anchored_readout
            )

        # One shared residue vocabulary across every mutation-conditioned head.
        self.aa_embedding = AminoAcidEmbedding(aa_dim)

        if arch == "v1":
            self.ddg_head = SubstitutionScoreHead(
                ctx_dim, aa_dim, hidden, dropout, aa_embedding=self.aa_embedding
            )
            self.epistasis_head = None
        else:
            self.ddg_head = EvolutionarySubstitutionHead(
                ctx_dim, aa_dim, hidden, dropout,
                aa_embedding=self.aa_embedding,
                use_esm_logits_prior=use_esm_logits_prior,
            )
            self.epistasis_head = ContactGatedEpistasisHead(
                ctx_dim, aa_dim, potts_dim=min(64, max(ctx_dim // 4, 16)),
                aa_embedding=self.aa_embedding,
                use_contact_gate=use_contact_gate,
            )

        self.three_state_head = ThreeStateHead(
            ctx_dim, aa_dim, hidden, dropout, aa_embedding=self.aa_embedding
        )
        self.pathogenicity_head = PathogenicityHead(
            ctx_dim, aa_dim, hidden // 2, dropout, aa_embedding=self.aa_embedding
        )
        self.idr_head = IdrEnsembleHead(ctx_dim, hidden // 2)
        self.mfi_head = MfiHead(ctx_dim, hidden // 2, dropout)
        self.proteomics_head = ProteomicsHead(
            ctx_dim, hidden, num_species=num_species, dropout=dropout
        )

    # ------------------------------------------------------------------
    # Shared encoding
    # ------------------------------------------------------------------
    def nodes(self, batch: dict) -> torch.Tensor:
        """Run the backbone. Returns ``(B, L, out_dim)`` node embeddings."""
        if self.arch == "v1":
            return self.backbone(
                esm=batch["esm"],
                mech=batch["mech"],
                mask=batch["mask"],
                edge_geo=batch["edge_geo"],
                attr_geo=batch["attr_geo"],
                edge_mech=batch["edge_mech"],
                attr_mech=batch["attr_mech"],
            )
        return self.backbone(
            esm=batch["esm"],
            mech=batch["mech"],
            mask=batch["mask"],
            edge_geo=batch["edge_geo"],
            attr_geo=batch["attr_geo"],
            edge_mech=batch["edge_mech"],
            attr_mech=batch["attr_mech"],
            topo=batch.get("topo"),
            layer_embs=batch.get("layer_embs"),
            edge_str=batch.get("edge_str"),
            attr_str=batch.get("attr_str"),
            str_node=batch.get("str_node"),
        )

    def context_from_nodes(self, h: torch.Tensor, batch: dict) -> torch.Tensor:
        """Read out per-mutation context ``z`` from precomputed node embeddings ``h``."""
        if "prot_idx" in batch:
            prot_idx = batch["prot_idx"]
            if isinstance(self.readout, MultiScaleSiteReadout):
                neigh1 = batch.get("neighborhood_1hop", batch["neighborhood_mask"])
                return self.readout.gather_protein_batch(
                    h, batch["mask"], prot_idx, batch["site"],
                    neigh1, batch["neighborhood_mask"],
                )
            # v1 fallback for Protein-Batch mode
            return self.readout(
                h[prot_idx], batch["mask"][prot_idx],
                batch["site"], batch["neighborhood_mask"],
            )
        if isinstance(self.readout, MultiScaleSiteReadout):
            return self.readout(
                h, batch["mask"], batch["site"],
                batch["neighborhood_mask"],
                batch.get("neighborhood_1hop"),
            )
        return self.readout(h, batch["mask"], batch["site"], batch["neighborhood_mask"])

    def context(self, batch: dict) -> torch.Tensor:
        """Site-anchored context ``z`` of shape ``(M, ctx_dim)``."""
        h = self.nodes(batch)
        return self.context_from_nodes(h, batch)

    def global_context(self, batch: dict) -> torch.Tensor:
        """Whole-sequence context for tasks with no mutation site."""
        h = self.nodes(batch)
        return self.readout.global_only(h, batch["mask"])

    # ------------------------------------------------------------------
    # Mutation-conditioned tasks
    # ------------------------------------------------------------------
    def _site_wt_logprobs(self, batch: dict) -> torch.Tensor | None:
        """Extract ``(M, 20)`` WT-marginal log-probabilities at the mutated sites."""
        lp = batch.get("wt_logprobs_site")
        if lp is not None:
            return lp
        full_lp = batch.get("wt_logprobs")
        if full_lp is None:
            return None
        if "prot_idx" in batch:
            p = batch["prot_idx"]
            s = batch["site"].clamp(min=0, max=full_lp.size(1) - 1)
            return full_lp[p, s]
        b = torch.arange(full_lp.size(0), device=full_lp.device)
        s = batch["site"].clamp(min=0, max=full_lp.size(1) - 1)
        return full_lp[b, s]

    def ddg(self, batch: dict, z: torch.Tensor | None = None):
        """``(ddg (M,1), logvar (M,1))`` — exactly antisymmetric in wt/mt."""
        z = self.context(batch) if z is None else z
        if isinstance(self.ddg_head, EvolutionarySubstitutionHead):
            return self.ddg_head(
                z, batch["wt_idx"], batch["mt_idx"],
                wt_logprobs=self._site_wt_logprobs(batch),
            )
        return self.ddg_head(z, batch["wt_idx"], batch["mt_idx"])

    def epistasis(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        wt_i: torch.Tensor,
        mt_i: torch.Tensor,
        wt_j: torch.Tensor,
        mt_j: torch.Tensor,
        pair_geo: torch.Tensor | None = None,
        lp_i: torch.Tensor | None = None,
        lp_j: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Double-mutant stability prediction: returns ``(ddg_ab, dddg_epistasis, logvar_ab)``
        where ``ddg_ab = ddg_i + ddg_j + dddg_epistasis``.
        """
        if isinstance(self.ddg_head, EvolutionarySubstitutionHead):
            ddg_i, lv_i = self.ddg_head(z_i, wt_i, mt_i, wt_logprobs=lp_i)
            ddg_j, lv_j = self.ddg_head(z_j, wt_j, mt_j, wt_logprobs=lp_j)
        else:
            ddg_i, lv_i = self.ddg_head(z_i, wt_i, mt_i)
            ddg_j, lv_j = self.ddg_head(z_j, wt_j, mt_j)
        if self.epistasis_head is not None:
            dddg = self.epistasis_head(z_i, z_j, wt_i, mt_i, wt_j, mt_j, pair_geo=pair_geo)
        else:
            dddg = torch.zeros_like(ddg_i)
        ddg_ab = ddg_i + ddg_j + dddg
        lv_ab = torch.logaddexp(lv_i.float(), lv_j.float()).to(ddg_ab.dtype)
        return ddg_ab, dddg, lv_ab

    def three_state(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        """``(M, 3)`` class logits, equivariant under wt/mt swap."""
        z = self.context(batch) if z is None else z
        return self.three_state_head(z, batch["wt_idx"], batch["mt_idx"])

    def pathogenicity(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        """``(M, 1)`` raw logit."""
        z = self.context(batch) if z is None else z
        return self.pathogenicity_head(z, batch["wt_idx"], batch["mt_idx"])

    def scan(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        """``(M, 21)`` ddG for every substitution at the batch's sites."""
        z = self.context(batch) if z is None else z
        if isinstance(self.ddg_head, EvolutionarySubstitutionHead):
            return self.ddg_head.scan(
                z, batch["wt_idx"], n_aa=N_AMINO_ACIDS + 1,
                wt_logprobs=self._site_wt_logprobs(batch),
            )
        return self.ddg_head.scan(z, batch["wt_idx"], n_aa=N_AMINO_ACIDS + 1)

    # ------------------------------------------------------------------
    # Whole-sequence tasks
    # ------------------------------------------------------------------
    def idr(self, batch: dict, z: torch.Tensor | None = None):
        z = self.global_context(batch) if z is None else z
        return self.idr_head(z, batch.get("seq_length"))

    def mfi(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        z = self.global_context(batch) if z is None else z
        return self.mfi_head(z, batch.get("seq_length"))

    def proteomics(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        z = self.global_context(batch) if z is None else z
        return self.proteomics_head(z, batch.get("species_id"), batch.get("seq_length"))

    # ------------------------------------------------------------------
    def parameter_count(self) -> dict[str, int]:
        """Trainable parameters by component, for the model card."""
        def count(m: nn.Module | None) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad) if m is not None else 0

        heads_total = sum(
            count(h)
            for h in (
                self.ddg_head, self.epistasis_head, self.three_state_head,
                self.pathogenicity_head, self.idr_head, self.mfi_head, self.proteomics_head,
            )
        )
        return {
            "arch": self.arch,
            "backbone": count(self.backbone),
            "readout": count(self.readout),
            "heads": heads_total,
            "total": count(self),
        }


def model_kwargs_from_config(cfg, esm_dim: int, **overrides) -> dict:
    """
    The single place a configuration becomes model constructor arguments.
    Checkpoints store the result verbatim so ``HoloGeoN(**state["model_kwargs"])``
    always reconstructs the exact trained architecture.
    """
    get = cfg.get if hasattr(cfg, "get") else cfg.__getitem__
    arch = overrides.get("arch") or get("arch", "v1")
    kwargs: dict = {
        "arch": arch,
        "esm_dim": esm_dim,
        "hidden": get("hidden", 256),
        "out_dim": get("out_dim", 256),
        "aa_dim": get("aa_dim", 32),
        "gat_heads": get("gat_heads", 4),
        "dropout": get("dropout", 0.1),
        "use_ssm": get("use_ssm", True),
        "use_mech_stream": get("use_mech_stream", True),
        "use_geo_stream": get("use_geo_stream", True),
        "site_anchored_readout": get("site_anchored_readout", True),
        "num_species": get("num_species", 0),
    }
    if arch == "v2":
        kwargs.update({
            "feat_dropout": get("feat_dropout", 0.25),
            "drop_edge": get("drop_edge", 0.20),
            "use_multi_layer": get("use_multi_layer", True),
            "use_contact_burial": get("use_contact_burial", True),
            "use_esm_logits_prior": get("use_esm_logits_prior", True),
            "use_str_stream": get("use_str_stream", False),
            "str_topology_only": get("str_topology_only", False),
            "use_contact_gate": get("use_contact_gate", True),
        })
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


__all__ = ["HoloGeoN", "model_kwargs_from_config"]
