"""
src/model.py
============
Holo-GeoN — one model, one forward path.

A frozen ESM-2 encoder (cached, see ``src/encoder.py``) supplies per-residue
embeddings and contact edges. On top of that sits a single trainable stack:

    dual-stream GATv2 backbone  ->  site-anchored readout  ->  task heads

All mutation-conditioned tasks share the same context vector ``z``, and all
residue-identity reasoning goes through one shared amino-acid embedding, so the
heads agree on what a leucine is.

The model has no modes. There is no demo path, no heuristic fallback, and no
alternative "legacy" forward. ``use_mech_stream`` and ``use_ssm`` exist so the
ablation runner can hold everything else fixed and measure one component; the
shipped configuration has both on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.backbone import DualStreamGNN
from src.encoder import DEFAULT_ESM, ESM_DIMS
from src.features import N_MECH_CHANNELS
from src.heads import (
    IdrEnsembleHead,
    MfiHead,
    PathogenicityHead,
    ProteomicsHead,
    SubstitutionScoreHead,
    ThreeStateHead,
)
from src.readout import SiteAnchoredReadout
from src.substitution import AminoAcidEmbedding, N_AMINO_ACIDS


class HoloGeoN(nn.Module):
    """
    Args:
        esm_dim         : encoder hidden size; defaults to the 650M checkpoint.
        hidden, out_dim : backbone widths.
        ctx_dim         : readout context width (defaults to ``out_dim``).
        aa_dim          : amino-acid embedding width.
        gat_heads       : GATv2 attention heads.
        dropout         : dropout across backbone, readout and heads.
        use_ssm         : selective state-space residue mixer.
        use_mech_stream : E_mech (translational coupling) message-passing stream.
        use_geo_stream  : E_geo (spatial contact) message-passing stream.
        site_anchored_readout : anchor the readout at the mutated residue.
                          False reproduces the v1 global-pool formulation and
                          exists only as an ablation axis.
        num_species     : >0 conditions the proteomics head on a species id.
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
    ):
        super().__init__()
        ctx_dim = ctx_dim or out_dim
        self.esm_dim = esm_dim
        self.ctx_dim = ctx_dim

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

        # One shared residue vocabulary across every mutation-conditioned head.
        self.aa_embedding = AminoAcidEmbedding(aa_dim)

        self.ddg_head = SubstitutionScoreHead(
            ctx_dim, aa_dim, hidden, dropout, aa_embedding=self.aa_embedding
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
        return self.backbone(
            esm=batch["esm"],
            mech=batch["mech"],
            mask=batch["mask"],
            edge_geo=batch["edge_geo"],
            attr_geo=batch["attr_geo"],
            edge_mech=batch["edge_mech"],
            attr_mech=batch["attr_mech"],
        )

    def context(self, batch: dict) -> torch.Tensor:
        """Site-anchored context ``z`` of shape ``(B, ctx_dim)``."""
        h = self.nodes(batch)
        return self.readout(h, batch["mask"], batch["site"], batch["neighborhood_mask"])

    def global_context(self, batch: dict) -> torch.Tensor:
        """Whole-sequence context for tasks with no mutation site."""
        h = self.nodes(batch)
        return self.readout.global_only(h, batch["mask"])

    # ------------------------------------------------------------------
    # Mutation-conditioned tasks
    # ------------------------------------------------------------------
    def ddg(self, batch: dict, z: torch.Tensor | None = None):
        """``(ddg (B,1), logvar (B,1))`` — exactly antisymmetric in wt/mt."""
        z = self.context(batch) if z is None else z
        return self.ddg_head(z, batch["wt_idx"], batch["mt_idx"])

    def three_state(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, 3)`` class logits, equivariant under wt/mt swap."""
        z = self.context(batch) if z is None else z
        return self.three_state_head(z, batch["wt_idx"], batch["mt_idx"])

    def pathogenicity(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, 1)`` raw logit."""
        z = self.context(batch) if z is None else z
        return self.pathogenicity_head(z, batch["wt_idx"], batch["mt_idx"])

    def scan(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        """
        ``(B, 21)`` ddG for every substitution at the batch's sites.

        One backbone pass covers every candidate, which is what makes a full
        deep mutational scan tractable in the interface. The last column is the
        unknown-residue slot; it keeps the wild-type column exactly zero even
        when the parent residue is non-standard.
        """
        z = self.context(batch) if z is None else z
        return self.ddg_head.scan(z, batch["wt_idx"], n_aa=N_AMINO_ACIDS + 1)

    # ------------------------------------------------------------------
    # Whole-sequence tasks
    # ------------------------------------------------------------------
    def idr(self, batch: dict, z: torch.Tensor | None = None):
        """``(mu, sigma)`` of the radius of gyration, in Angstrom."""
        z = self.global_context(batch) if z is None else z
        return self.idr_head(z, batch.get("seq_length"))

    def mfi(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        z = self.context(batch) if z is None else z
        return self.mfi_head(z, batch.get("seq_length"))

    def proteomics(self, batch: dict, z: torch.Tensor | None = None) -> torch.Tensor:
        z = self.global_context(batch) if z is None else z
        return self.proteomics_head(z, batch.get("species_id"), batch.get("seq_length"))

    # ------------------------------------------------------------------
    def parameter_count(self) -> dict[str, int]:
        """Trainable parameters by component, for the model card."""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "backbone": count(self.backbone),
            "readout": count(self.readout),
            "heads": sum(
                count(h)
                for h in (
                    self.ddg_head, self.three_state_head, self.pathogenicity_head,
                    self.idr_head, self.mfi_head, self.proteomics_head,
                )
            ),
            "total": count(self),
        }


def model_kwargs_from_config(cfg, esm_dim: int, **overrides) -> dict:
    """
    The single place a configuration becomes model constructor arguments.

    Checkpoints store the result verbatim, so loading is
    ``HoloGeoN(**state["model_kwargs"])`` and cannot drift from how the model was
    built. Reconstructing from loose config keys is how a checkpoint becomes
    unloadable when a constructor argument is added but not recorded.
    """
    get = cfg.get if hasattr(cfg, "get") else cfg.__getitem__
    kwargs = {
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
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


__all__ = ["HoloGeoN", "model_kwargs_from_config"]
