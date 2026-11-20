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
import threading
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
# Sites per backbone forward pass during a scan. Each site is a `window`-node
# graph, so this times the window is the padded node count of one batch - the
# same quantity src/data.NodeBudgetBatchSampler caps during training (R2-02).
SCAN_SITE_BATCH = int(os.environ.get("HOLOGEON_SCAN_BATCH", "32"))

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
        self._lock = threading.Lock()
        self._tried = False
        self._ready = False
        self._error: Optional[str] = None
        self._model = None
        self._encoder = None
        self._torch = None
        self._device = None
        self._config: dict = {}
        self._calibration: Optional[dict] = None
        #: Which sigma `self._calibration` describes. Serving one interval and
        #: reporting another's coverage is R2-09.
        self._calibration_of: str = "no calibration file"
        self._sigma_scale: float = 1.0
        self._untrained_heads: list[str] = []

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._tried:
            return
        with self._lock:
            if self._tried:
                return
            try:
                self._load_locked()
            finally:
                # Set _tried ONLY after loading completes (or fails), inside the
                # lock, so concurrent startup requests (e.g. /api/health polling
                # while torch imports) wait cleanly instead of seeing ready=False
                # with error=None.
                self._tried = True

    def _load_locked(self) -> None:
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
                    payload = json.load(fh)
                # A scale fitted on the validation split, applied to the ddG
                # sigma the interface reports. Without it a measured
                # miscalibration had no remedy anywhere in the codebase.
                variance_scale = payload.get("variance_scale") or {}
                scale = variance_scale.get("scale")
                self._sigma_scale = float(scale) if scale else 1.0
                # Round-2 review, R2-09: the interface reported the calibration
                # of the RAW sigma while serving `sigma * scale`. The reviewer
                # constructed a case where the flag read True at ECE 0.0053
                # while the band actually drawn had 99.8% coverage for a nominal
                # 95% and an ECE of 0.070 - above the repository's own 0.05
                # tolerance. scripts/calibration.py already computed
                # `calibration_after_scaling` and nothing read it. Whichever
                # sigma is served is the one whose numbers get published.
                after = variance_scale.get("calibration_after_scaling")
                if self._sigma_scale != 1.0 and after:
                    self._calibration = after
                    self._calibration_of = (
                        f"the served sigma (raw x {self._sigma_scale:.3f})"
                    )
                else:
                    self._calibration = payload.get("calibration")
                    self._calibration_of = "the served sigma (unscaled)"
                    if self._sigma_scale != 1.0:
                        # A scale is applied but the run recorded no post-scaling
                        # measurement, so nothing here describes what is served.
                        self._calibration = None
                        self._calibration_of = (
                            f"NOT MEASURED: a scale of {self._sigma_scale:.3f} is "
                            f"applied but {CALIBRATION_PATH.name} carries no "
                            f"calibration_after_scaling, so no measurement "
                            f"describes the interval this interface draws. "
                            f"Re-run scripts.calibration."
                        )
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
            # Round-1 review, R1-14: this used to be `bool(cal)` - the existence
            # of a file. A calibration run measuring 11% coverage for a 95% band
            # flipped the interface to "calibrated". It now reports the VERDICT
            # the calibration run computed from its own numbers, and surfaces
            # those numbers so a reader can check it.
            "uncertainty_calibrated": bool(cal.get("well_calibrated", False)),
            "sigma_scale": self._sigma_scale,
            "calibration": {
                "expected_calibration_error": cal.get("expected_calibration_error"),
                "max_coverage_gap": cal.get("max_coverage_gap"),
                "observed_coverage": cal.get("coverage_observed"),
                "well_calibrated": cal.get("well_calibrated"),
                "measured_on": str(CALIBRATION_PATH),
                # Which sigma these numbers describe. When a variance scale is
                # applied, they describe the SCALED sigma - the one the ddG
                # endpoint actually draws its interval from (R2-09).
                "describes": self._calibration_of,
            } if cal else None,
            "calibration_note": None if cal else self._calibration_of,
        }

    # ------------------------------------------------------------------
    def _encode_once(self, seq: str) -> dict:
        """
        Encode a sequence. The result does not depend on any mutation site, so
        a scan reuses it across every position.

        Round-1 review, R1-13: the scan called this per position on the
        identical, unchanged sequence, so a 600-position heatmap ran 600
        identical ESM-2 forward passes. The docstrings claimed "one encoder
        pass"; the implementation did L.
        """
        from src.features import mechanistic_features

        rec = self._encoder.encode(
            seq,
            top_k=self._config.get("contact_top_k", 24),
            min_sep=self._config.get("contact_min_sep", 3),
        )
        # A sequence typed into the interface carries no coding sequence, so the
        # mechanistic channels are masked (mech_valid = 0) and E_mech is empty
        # for it. That is the same handling the training pipeline gives a row
        # without a CDS.
        rec["mech"] = mechanistic_features(seq, None, len(seq)).half()
        return rec

    def _batch_for_sites(self, rec: dict, sites: list[int]):
        """Build one batch of per-site graphs from an already-encoded protein."""
        from src.data import build_sample_graph, collate_graphs, move_batch

        window = self._config.get("window", 256)
        graphs = []
        for site in sites:
            g = build_sample_graph(rec, site, window, self._config)
            g.update({"wt_idx": 0, "mt_idx": 0, "y": 0.0, "sigma": float("nan")})
            graphs.append(g)
        return move_batch(collate_graphs(graphs), self._device)

    def _context_for(self, seq: str, site: int):
        """Encode a sequence and build the model context at one site."""
        torch = self._torch
        batch = self._batch_for_sites(self._encode_once(seq), [site])
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
        sigma = float(torch.exp(0.5 * logvar_t).item()) * self._sigma_scale
        classes = ["destabilizing", "neutral", "stabilizing"]
        # Round-1 review, R1-25: the verdict was a strict sign test with no
        # neutral case, in a payload that simultaneously reported a three-class
        # probability including "neutral" - so a ddG of +0.006 was rendered
        # "stabilizing" while the head's own argmax said neutral. It now comes
        # from the head that has a defined neutral class, and the payload says
        # so.
        verdict = classes[int(torch.argmax(three).item())]
        cal = self._calibration or {}
        return {
            "mutation": f"{wt_aa}{site + 1}{mut_aa}",
            "position": site + 1,
            "wt_residue": wt_aa,
            "mut_residue": mut_aa,
            "ddg": round(ddg, 4),
            "sigma": round(sigma, 4),
            "ci_low": round(ddg - Z95 * sigma, 4),
            "ci_high": round(ddg + Z95 * sigma, 4),
            "ci_source": (
                "model predicted sigma (Gaussian, 95%)"
                + (f", rescaled by {self._sigma_scale:.3f} from calibration"
                   if self._sigma_scale != 1.0 else "")
            ),
            "uncertainty_calibrated": bool(cal.get("well_calibrated", False)),
            "three_state": {
                "destabilizing": round(float(three[0]), 4),
                "neutral": round(float(three[1]), 4),
                "stabilizing": round(float(three[2]), 4),
            },
            "verdict": verdict,
            "verdict_source": "three-state head argmax",
        }

    # ------------------------------------------------------------------
    def mutation_scan(self, sequence: str, start: int = 1, end: Optional[int] = None) -> dict:
        """
        Deep mutational scan: ddG for every substitution over a window.

        **One encoder pass per protein**, not per position. The encoder output
        does not depend on the mutation site, and the head is conditioned on
        residue identity rather than on a re-encoded mutant sequence, so the
        whole scan reuses a single ESM-2 forward pass and then runs the
        trainable backbone in batches of sites.

        Round-2 review, R2-05: the round-1 fix split ``_context_for`` into
        ``_encode_once`` + ``_batch_for_sites`` - the two pieces needed to hoist
        the encoder call - and then left this loop calling ``_context_for`` per
        position. The reviewer counted 60 encoder calls for a 60-position scan,
        all on one identical sequence, while three docstrings claimed one. At
        ``MAX_SCAN_POSITIONS = 600`` that was 600 identical ESM-2 650M forward
        passes for one heatmap, and ``/api/compare`` pays it for up to 400
        positions on every AlphaFold comparison.
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
        sites = [p - 1 for p in positions]

        rec = self._encode_once(seq)          # the one encoder pass
        columns: list[list[float]] = []
        for lo in range(0, len(sites), SCAN_SITE_BATCH):
            chunk = sites[lo:lo + SCAN_SITE_BATCH]
            batch = self._batch_for_sites(rec, chunk)
            batch["wt_idx"] = torch.tensor(
                [aa_index(seq[site]) for site in chunk], device=self._device
            )
            with torch.no_grad():
                z = self._model.context(batch)
                rows = self._model.scan(batch, z)
            for row in rows:
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

        The ddG variance scale is deliberately NOT applied here. Round-2 review,
        R2-09: it was, and ``scripts/calibration.py`` fits that scalar from
        ``model.ddg()`` alone - so a correction in kcal/mol was multiplying a
        radius of gyration in angstrom, moving the reported 95% interval from
        1.78 A to 4.44 A for the same model and sequence. A per-head scale would
        have to be fitted on a per-head calibration set; until one exists this
        head serves the sigma it predicted.
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
            # Round-2 review, R2-03: the interface displayed a "Character:
            # Globular / Extended" chip computed in the browser from
            # `mu / L**0.33 < 2.5` - two constants that appear nowhere else in
            # the repository, with no citation, no config key and no model
            # involved. A categorical biophysical conclusion the model never
            # produced, presented next to numbers it did. It is gone; what the
            # head actually emits says where it came from, as the ddG payload
            # already does.
            "value_source": "IDR head, radius of gyration in Angstrom",
            "ci_source": "model predicted sigma (Gaussian, 95%)",
        }


# ---------------------------------------------------------------------------
def clean_sequence(seq: str) -> str:
    """Uppercase, strip FASTA headers and whitespace, validate the alphabet."""
    from src.substitution import AA_ORDER, NON_STANDARD_AA

    if not seq:
        raise ValueError("empty sequence")
    lines = [ln for ln in seq.splitlines() if not ln.startswith(">")]
    seq = "".join("".join(lines).split()).upper()
    if not seq:
        raise ValueError("sequence contains no residues")
    # Round-1 review, R1-24: rejecting the whole sequence for a single B/Z/X/U/O
    # meant the interface refused UniProt entries the training path accepts.
    # Non-standard residues are allowed and flagged by the aa_known channel; the
    # nucleotide check below is what actually needs to fail loudly.
    bad = sorted(set(seq) - set(AA_ORDER) - set(NON_STANDARD_AA))
    if bad:
        raise ValueError(
            f"invalid amino-acid symbol(s): {''.join(bad)}. This field takes a "
            f"PROTEIN sequence."
        )
    if len(seq) >= 8 and set(seq) <= set("ACGTU"):
        raise ValueError(
            "this looks like a nucleotide sequence. A, C, G, T and U are all "
            "valid amino-acid letters, so a protein language model would accept "
            "it silently and return meaningless predictions. Paste the PROTEIN "
            "sequence."
        )
    if len(seq) > MAX_SEQ_LENGTH:
        raise ValueError(f"sequence too long ({len(seq)} > {MAX_SEQ_LENGTH})")
    return seq


@lru_cache(maxsize=1)
def get_engine() -> HoloGeoNEngine:
    """Process-wide singleton."""
    return HoloGeoNEngine()
