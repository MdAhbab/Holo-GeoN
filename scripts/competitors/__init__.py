"""
scripts/competitors
===================
Standardized competitor adapters for fair, leak-free head-to-head benchmarking
against Holo-GeoN (all scored by ``scripts/score_predictions.py``):

  * ``thermompnn`` — ThermoMPNN (Dieckhaus et al., PNAS 2024; ProteinMPNN + lightattn)
  * ``spurs``      — SPURS (Li & Luo, Nature Comms 2026; ESM-2 + 3D structure fusion)
  * ``rasp``       — RaSP (Blaabjerg et al., eLife 2023; 3D cavity microenvironment MLP)
  * ``prostata``   — PROSTATA (Umerenkov et al., Bioinformatics 2023; 5-head ESM-2)
  * ``ddgemb``     — DDGemb (Savojardo et al., Bioinformatics 2024; seq-only Transformer)
"""

ADAPTERS = ("thermompnn", "spurs", "rasp", "prostata", "ddgemb")

__all__ = ["ADAPTERS"]
