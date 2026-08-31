"""
app/backend/inference.py
========================
Inference engine for the web interface.

There is no demo mode.

The predecessor shipped a deterministic biophysical heuristic
(``src/heuristics.py``) that silently stood in for the model whenever weights
were absent, so the interface always returned plausible numbers whether or not a
model existed. Worse, in *full* mode it still drew its confidence interval from
``heuristic_confidence(ddg) = ddg +/- (0.4 + 0.12*|ddg|)`` — a formula with no
connection to the model, wrapped around a genuine prediction.

Both are gone. Without a checkpoint and an encoder cache the engine reports that
it is not ready and every prediction endpoint fails cleanly. Confidence
intervals come from the model's own predicted sigma, which is trained under
Gaussian NLL and whose coverage is measured by ``scripts/calibration.py``.

Sign convention (dG_mut - dG_wt): ddG > 0 stabilising, ddG < 0 destabilising.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

# The model package sits two levels up from app/backend/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

WEIGHTS_PATH = Path(os.environ.get("HOLOGEON_WEIGHTS", _REPO_ROOT / "runs" / "main" / "model_best.pt"))
CALIBRATION_PATH = Path(
    os.environ.get("HOLOGEON_CALIBRATION", _REPO_ROOT / "runs" / "main" / "calibration.json")
)
MAX_SEQ_LENGTH = int(os.environ.get("HOLOGEON_MAX_LENGTH", "5000"))
MAX_SCAN_POSITIONS = int(os.environ.get("HOLOGEON_MAX_SCAN", "600"))

# 95% interval multiplier for a Gaussian predictive distribution.
Z95 = 1.959964


class ModelUnavailable(RuntimeError):
    """Raised when a prediction is requested and no usable model is loaded."""


class HoloGeoNEngine:
    """
    Lazy, cached wrapper around the trained model.

    Loading is attempted once. On failure the engine stays unavailable and says
    why; it never substitutes an approximation.
    """

    def __init__(self) -> None:
        self._tried = False
        self._ready = False
        self._error: Optional[str] = None
        self._model = None
        self._encoder = None
        self._torch = None
        self._device = None
        self._config: dict = {}
        self._calibration: Optional[dict] = None
        self._untrained_heads: list[str] = []

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True

        if not WEIGHTS_PATH.exists():
            self._error = (
                f"no trained checkpoint at {WEIGHTS_PATH}. Train one with "
                f"`python run.py train`, or point HOLOGEON_WEIGHTS at an existing "
                f"checkpoint. The interface does not approximate predictions."
            )
            return
        try:
            import json

            import torch

            from src.encoder import ESM_DIMS, EsmEncoder
            from src.model import HoloGeoN, model_kwargs_from_config

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
            cfg = state.get("config", {})
            esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.get("esm_model"), 1280))

            # Build from what the checkpoint says it is, not from what the
            # current config happens to hold.
            kwargs = state.get("model_kwargs") or model_kwargs_from_config(cfg, esm_dim)
            model = HoloGeoN(**kwargs)
            model.load_state_dict(state["model"])
            model.to(device).eval()

            self._torch = torch
            self._model = model
            self._device = device
            self._config = cfg
            self._untrained_heads = state.get("untrained_heads", [])
            # The encoder runs live here: the interface accepts arbitrary
            # sequences, which by definition are not in the training cache.
            self._encoder = EsmEncoder(
                cfg.get("esm_model", "facebook/esm2_t33_650M_UR50D"), device=str(device)
            )
            if CALIBRATION_PATH.exists():
                with open(CALIBRATION_PATH, encoding="utf-8") as fh:
                    self._calibration = json.load(fh).get("calibration")
            self._ready = True
        except Exception as exc:  # noqa: BLE001
            self._error = f"{exc.__class__.__name__}: {exc}"
            self._ready = False

    @property
    def ready(self) -> bool:
        self._load()
        return self._ready

    def _require(self) -> None:
        if not self.ready:
            raise ModelUnavailable(self._error or "model unavailable")

    # ------------------------------------------------------------------
    def health(self) -> dict:
        self._load()
        cal = self._calibration or {}
        return {
            "ready": self._ready,
            "error": None if self._ready else self._error,
            "weights_path": str(WEIGHTS_PATH),
            "esm_model": self._config.get("esm_model"),
            "window": self._config.get("window"),
            "untrained_heads": self._untrained_heads,
            # The interface may only present an interval as calibrated if a
            # calibration run measured it. Absent that, it says so.
            "uncertainty_calibrated": bool(cal),
            "calibration": {
                "expected_calibration_error": cal.get("expected_calibration_error"),
                "observed_coverage": cal.get("coverage_observed"),
                "measured_on": str(CALIBRATION_PATH),
            } if cal else None,
        }

    # ------------------------------------------------------------------
    def _context_for(self, seq: str, site: int):
        """Encode a sequence and build the model context at one site."""
        from src.data import build_sample_graph, collate_graphs, move_batch

        torch = self._torch
        rec = self._encoder.encode(
            seq,
            top_k=self._config.get("contact_top_k", 24),
            min_sep=self._config.get("contact_min_sep", 3),
        )
        from src.features import mechanistic_features

        # A sequence typed into the interface carries no coding sequence, so the
        # mechanistic channels are masked (mech_valid = 0) and E_mech is empty
        # for it. That is the same handling the training pipeline gives a row
        # without a CDS.
        rec["mech"] = mechanistic_features(seq, None, len(seq)).half()

        graph = build_sample_graph(rec, site, self._config.get("window", 256), self._config)
        graph.update({"wt_idx": 0, "mt_idx": 0, "y": 0.0, "sigma": float("nan")})
        batch = move_batch(collate_graphs([graph]), self._device)
        with torch.no_grad():
            z = self._model.context(batch)
        return z, batch

    # ------------------------------------------------------------------
    def predict_ddg(self, wt_sequence: str, mutation: str) -> dict:
        """ddG for one substitution, with the model's own confidence interval."""
        from src.substitution import aa_index, parse_mutation

        self._require()
        seq = clean_sequence(wt_sequence)
        _, site, wt_aa, mut_aa = parse_mutation(seq, mutation)

        torch = self._torch
        z, batch = self._context_for(seq, site)
        batch["wt_idx"] = torch.tensor([aa_index(wt_aa)], device=self._device)
        batch["mt_idx"] = torch.tensor([aa_index(mut_aa)], device=self._device)
        with torch.no_grad():
            ddg_t, logvar_t = self._model.ddg(batch, z)
            three = torch.softmax(self._model.three_state(batch, z), dim=-1)[0]

        ddg = float(ddg_t.item())
        sigma = float(torch.exp(0.5 * logvar_t).item())
        return {
            "mutation": f"{wt_aa}{site + 1}{mut_aa}",
            "position": site + 1,
            "wt_residue": wt_aa,
            "mut_residue": mut_aa,
            "ddg": round(ddg, 4),
            "sigma": round(sigma, 4),
            "ci_low": round(ddg - Z95 * sigma, 4),
            "ci_high": round(ddg + Z95 * sigma, 4),
            "ci_source": "model predicted sigma (Gaussian, 95%)",
            "uncertainty_calibrated": bool(self._calibration),
            "three_state": {
                "destabilizing": round(float(three[0]), 4),
                "neutral": round(float(three[1]), 4),
                "stabilizing": round(float(three[2]), 4),
            },
            "verdict": "stabilizing" if ddg > 0 else "destabilizing",
        }

    # ------------------------------------------------------------------
    def mutation_scan(self, sequence: str, start: int = 1, end: Optional[int] = None) -> dict:
        """
        Deep mutational scan: ddG for every substitution over a window.

        One encoder pass per position covers all 20 candidate residues, because
        the head is conditioned on residue identity rather than re-encoding a
        mutant sequence. The predecessor called full inference once per cell —
        20 x L x 2 encoder passes, or roughly 16,000 for a 400-residue protein,
        which made this endpoint unusable with a real model loaded.
        """
        from src.substitution import AA_ORDER, aa_index

        self._require()
        seq = clean_sequence(sequence)
        start = max(1, int(start))
        end = int(end) if end else len(seq)
        end = min(end, len(seq))
        if end < start:
            raise ValueError("end must be >= start")
        if end - start + 1 > MAX_SCAN_POSITIONS:
            raise ValueError(
                f"scan window too wide ({end - start + 1} > {MAX_SCAN_POSITIONS} positions)"
            )

        torch = self._torch
        positions = list(range(start, end + 1))
        columns: list[list[float]] = []
        for pos in positions:
            site = pos - 1
            z, batch = self._context_for(seq, site)
            batch["wt_idx"] = torch.tensor([aa_index(seq[site])], device=self._device)
            with torch.no_grad():
                row = self._model.scan(batch, z)[0]
            columns.append([round(float(v), 4) for v in row])

        # Transpose to rows = amino acids, columns = positions.
        matrix = [[columns[c][r] for c in range(len(positions))]
                  for r in range(len(AA_ORDER))]
        return {
            "positions": positions,
            "wt_residues": [seq[p - 1] for p in positions],
            "aa_order": list(AA_ORDER),
            "matrix": matrix,
        }

    # ------------------------------------------------------------------
    def idr_ensemble(self, sequence: str) -> dict:
        """
        Radius-of-gyration ensemble from the IDR head.

        Refuses when that head was never trained: a forward pass through
        randomly-initialised weights returns a number, and a number the
        interface displays is a claim.
        """
        self._require()
        if "idr" in self._untrained_heads:
            raise ModelUnavailable(
                "the IDR head was not trained in this checkpoint (no experimental "
                "radius-of-gyration table was supplied), so it has no predictions "
                "to give. Configure idr_data and retrain to enable it."
            )
        torch = self._torch
        seq = clean_sequence(sequence)
        # Whole-sequence task: the global readout, not the site-anchored one.
        _, batch = self._context_for(seq, 0)
        with torch.no_grad():
            mu_t, sigma_t = self._model.idr(batch)
        mu, sigma = float(mu_t.item()), float(sigma_t.item())
        return {
            "length": len(seq),
            "mu": round(mu, 4),
            "sigma": round(max(sigma, 1e-3), 4),
            "ci_low": round(mu - Z95 * sigma, 4),
            "ci_high": round(mu + Z95 * sigma, 4),
        }


# ---------------------------------------------------------------------------
def clean_sequence(seq: str) -> str:
    """Uppercase, strip FASTA headers and whitespace, validate the alphabet."""
    from src.substitution import AA_ORDER

    if not seq:
        raise ValueError("empty sequence")
    lines = [ln for ln in seq.splitlines() if not ln.startswith(">")]
    seq = "".join("".join(lines).split()).upper()
    if not seq:
        raise ValueError("sequence contains no residues")
    bad = sorted(set(seq) - set(AA_ORDER))
    if bad:
        raise ValueError(
            f"invalid amino-acid symbol(s): {''.join(bad)}. This field takes a "
            f"PROTEIN sequence, not nucleotides."
        )
    if len(seq) > MAX_SEQ_LENGTH:
        raise ValueError(f"sequence too long ({len(seq)} > {MAX_SEQ_LENGTH})")
    return seq


@lru_cache(maxsize=1)
def get_engine() -> HoloGeoNEngine:
    """Process-wide singleton."""
    return HoloGeoNEngine()
