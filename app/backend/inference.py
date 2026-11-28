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

Holo-GeoN v2 capabilities, each of which refuses rather than approximates
------------------------------------------------------------------------
* **Double-mutant epistasis** (``predict_epistasis``): served only when the loaded
  checkpoint's contact-gated epistasis head was actually trained
  (``scripts.train_epistasis``). A single-mutant checkpoint carries that head at
  its near-zero initialisation, so it is refused, as the untrained IDR head is.
* **Optional coding sequence** (``cds=``): validated as a real reading frame that
  encodes the very protein supplied, then fed to the mechanistic channels. The model
  is trained with CDS dropout, so both paths are in-distribution; ``predict_ddg``
  returns them side by side. If ViennaRNA is missing the CDS cannot be used, and the
  request is refused rather than silently answered without it.
* **Structure-free vs SE(3)-invariant 3D** (``mode=``): the default model is a
  structure-free predicted-contact graph and reads no coordinates; its only symmetry is
  wild-type/mutant swap equivariance, and it is NOT SE(3)- or E(3)-equivariant. The 3D
  mode needs a second checkpoint (``HOLOGEON_WEIGHTS_3D``) trained with the coordinate
  stream, plus a supplied structure; features are distances and local frames, so the
  output is SE(3)-INVARIANT.
* **Live progress**: ``mutation_scan_events`` / ``epistasis_events`` yield events as
  they go; the blocking ``mutation_scan`` / ``predict_epistasis`` just consume them.
"""

from __future__ import annotations

import collections
import os
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

# The model package sits two levels up from app/backend/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

WEIGHTS_PATH = Path(os.environ.get("HOLOGEON_WEIGHTS", _REPO_ROOT / "runs" / "main" / "model_best.pt"))
CALIBRATION_PATH = Path(
    os.environ.get("HOLOGEON_CALIBRATION", _REPO_ROOT / "runs" / "main" / "calibration.json")
)
#: Optional second checkpoint trained WITH the SE(3)-invariant coordinate stream
#: (Arms B/C/D of the geometry suite). Without it the 3D mode is unavailable.
WEIGHTS_3D_PATH = os.environ.get("HOLOGEON_WEIGHTS_3D") or None
MAX_SEQ_LENGTH = int(os.environ.get("HOLOGEON_MAX_LENGTH", "5000"))
MAX_SCAN_POSITIONS = int(os.environ.get("HOLOGEON_MAX_SCAN", "600"))
MAX_EPISTASIS_PAIRS = int(os.environ.get("HOLOGEON_MAX_PAIRS", "200"))
MAX_PDB_CHARS = int(os.environ.get("HOLOGEON_MAX_PDB_CHARS", str(8_000_000)))
#: Smallest fraction of the sequence a supplied structure must cover when it is SHORTER than
#: the sequence (a crystal structure or design construct missing a terminal tail). Residues it
#: does not cover get no coordinates and are masked out of the coordinate stream.
MIN_STRUCTURE_COVERAGE = float(os.environ.get("HOLOGEON_MIN_STRUCTURE_COVERAGE", "0.8"))
# Sites per backbone forward pass during a scan. Each site is a `window`-node
# graph, so this times the window is the padded node count of one batch - the
# same quantity src/data.NodeBudgetBatchSampler caps during training (R2-02).
SCAN_SITE_BATCH = int(os.environ.get("HOLOGEON_SCAN_BATCH", "32"))
#: Encoder outputs kept per sequence (a scan, a CDS comparison and an epistasis
#: request on one protein share one ESM-2 forward pass).
ENCODER_CACHE_SIZE = 4

# 95% interval multiplier for a Gaussian predictive distribution.
Z95 = 1.959964
#: AlphaFold's own boundary between "confident" and "low confidence".
LOW_PLDDT = 70.0

MODES = ("structure_free", "invariant_3d")
SYMMETRY = {
    "structure_free": (
        "structure-free predicted-contact graph; reads no coordinates. Its only symmetry is "
        "wild-type/mutant swap equivariance: it is NOT SE(3)- or E(3)-equivariant."
    ),
    "invariant_3d": (
        "SE(3)-invariant 3D backbone features (distances and local frames) from the supplied "
        "structure. Invariant, not equivariant: the output is a scalar."
    ),
}


class ModelUnavailable(RuntimeError):
    """Raised when a prediction is requested and no usable model is loaded."""


def _vienna_available() -> bool:
    """Whether ViennaRNA's ``RNA`` module is importable, without importing src.features (and torch) to ask."""
    mod = sys.modules.get("src.features")
    if mod is not None:
        return bool(getattr(mod, "VIENNA_AVAILABLE", False))
    import importlib.util

    return importlib.util.find_spec("RNA") is not None


def _epistasis_status(model, state: dict) -> dict:
    """Whether this checkpoint's epistasis head can honestly serve predictions."""
    if getattr(model, "epistasis_head", None) is None:
        return {"available": False, "source": None,
                "reason": "this checkpoint has no epistasis head (arch v1 has none; train a v2 model)"}
    arm = state.get("arm")
    if arm is not None:
        if arm == "additive_only":
            return {"available": False, "source": f"scripts.train_epistasis arm={arm}",
                    "reason": "this checkpoint is the additive-only epistasis arm: its pair term is "
                              "switched off by design, so it has no epistasis prediction to give"}
        return {"available": True, "reason": None, "source": f"scripts.train_epistasis arm={arm}"}
    if "epistasis" in (state.get("trained_tasks") or []):
        return {"available": True, "reason": None, "source": "trained_tasks"}
    return {"available": False, "source": None,
            "reason": "the epistasis head was never trained in this checkpoint: it is trained only on "
                      "double mutants by scripts.train_epistasis, and a single-mutant checkpoint carries "
                      "the head at its near-zero initialisation, whose output would be noise"}


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
        self._calibration_split: Optional[str] = None
        self._untrained_heads: list[str] = []
        self._epistasis: dict = {"available": False, "reason": "model not loaded", "source": None}
        # SE(3)-invariant 3D checkpoint (optional, loaded on first need)
        self._lock3d = threading.Lock()
        self._tried3d = False
        self._bundle3d: Optional[dict] = None
        self._error3d: Optional[str] = None
        # one ESM-2 pass per distinct sequence
        self._enc_cache: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self._cache_lock = threading.Lock()

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

    @staticmethod
    def _read_checkpoint(path: Path, device) -> dict:
        """Rebuild a model from what the checkpoint says it is, not from the current config."""
        import torch

        from src.encoder import ESM_DIMS
        from src.model import HoloGeoN, model_kwargs_from_config

        state = torch.load(path, map_location=device, weights_only=False)
        cfg = state.get("config", {})
        esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.get("esm_model"), 1280))
        kwargs = state.get("model_kwargs") or model_kwargs_from_config(cfg, esm_dim)
        model = HoloGeoN(**kwargs)
        model.load_state_dict(state["model"])
        model.to(device).eval()
        untrained = state.get("untrained_heads")
        if untrained is None:
            # A scripts.train_epistasis checkpoint records no head list; only the ddG and
            # epistasis heads were trained there, so the rest are not to be offered.
            untrained = ["idr", "mfi", "proteomics"] if "arm" in state else []
        return {
            "model": model, "config": cfg, "untrained": list(untrained), "path": str(path),
            "epistasis": _epistasis_status(model, state), "arm": state.get("arm"),
            "uses_structure": bool(kwargs.get("use_str_stream")),
            "str_topology_only": bool(kwargs.get("str_topology_only")),
        }

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

            from src.encoder import EsmEncoder

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bundle = self._read_checkpoint(WEIGHTS_PATH, device)
            cfg = bundle["config"]

            self._torch = torch
            self._model = bundle["model"]
            self._device = device
            self._config = cfg
            self._untrained_heads = bundle["untrained"]
            self._epistasis = bundle["epistasis"]
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
                self._calibration_split = payload.get("split")
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
    # optional SE(3)-invariant 3D checkpoint
    # ------------------------------------------------------------------
    def _load_3d(self) -> None:
        if self._tried3d:
            return
        with self._lock3d:
            if self._tried3d:
                return
            try:
                self._load_3d_locked()
            finally:
                self._tried3d = True

    def _load_3d_locked(self) -> None:
        if not WEIGHTS_3D_PATH:
            self._error3d = (
                "no SE(3)-invariant 3D checkpoint is configured (set HOLOGEON_WEIGHTS_3D to the weights "
                "of a model trained with the coordinate stream). The interface does not approximate a "
                "3D prediction with the structure-free model."
            )
            return
        path = Path(WEIGHTS_3D_PATH)
        if not path.exists():
            self._error3d = f"no 3D checkpoint at {path}"
            return
        if not self.ready:
            self._error3d = "the primary model is not loaded, so the 3D checkpoint cannot share its encoder"
            return
        try:
            bundle = self._read_checkpoint(path, self._device)
            if not bundle["uses_structure"]:
                self._error3d = (f"{path} was not trained with the 3D coordinate stream "
                                 f"(use_str_stream is false), so it cannot serve the SE(3)-invariant mode")
                return
            if bundle["config"].get("esm_model") != self._config.get("esm_model"):
                self._error3d = ("the 3D checkpoint was trained on a different ESM-2 encoder than the "
                                 "primary model, so it cannot share the loaded encoder")
                return
            cfg = dict(bundle["config"])
            cfg["use_str_stream"] = True
            cfg["str_topology_only"] = bundle["str_topology_only"]
            bundle["config"] = cfg
            self._bundle3d = bundle
        except Exception as exc:  # noqa: BLE001
            self._error3d = f"{exc.__class__.__name__}: {exc}"

    def _unavailable_3d(self) -> Optional[str]:
        self._load_3d()
        return None if self._bundle3d is not None else self._error3d

    # ------------------------------------------------------------------
    def health(self) -> dict:
        self._load()
        cal = self._calibration or {}
        if self._ready:
            err3d = self._unavailable_3d()          # loads the optional 3D checkpoint once
        else:
            err3d = self._error or "the primary model is not loaded"
        vienna = _vienna_available()
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
                # the thresholds the verdict above was computed against (ECE and max coverage gap)
                "tolerances": cal.get("tolerances"),
                "split": self._calibration_split,
                "measured_on": str(CALIBRATION_PATH),
                # Which sigma these numbers describe. When a variance scale is
                # applied, they describe the SCALED sigma - the one the ddG
                # endpoint actually draws its interval from (R2-09).
                "describes": self._calibration_of,
            } if cal else None,
            "calibration_note": None if cal else self._calibration_of,
            # What this server can and cannot do, each with the reason when it cannot.
            "capabilities": {
                "epistasis": {"available": bool(self._ready and self._epistasis["available"]),
                              "reason": None if (self._ready and self._epistasis["available"])
                              else (self._epistasis["reason"] if self._ready else self._error),
                              "source": self._epistasis.get("source"),
                              "uncertainty": "calibration was measured on single mutants only; the "
                                             "epistasis term itself has no uncertainty estimate"},
                "cds": {"available": bool(vienna),
                        "reason": None if vienna else
                        "ViennaRNA is not installed on this server, so a coding sequence cannot be "
                        "used; such requests are refused rather than answered without it"},
                "invariant_3d": {"available": bool(self._ready and err3d is None),
                                 "reason": err3d if self._ready else self._error,
                                 "weights_path": WEIGHTS_3D_PATH},
                "streaming": {"available": True,
                              "endpoints": ["/api/scan/stream", "/api/epistasis/stream"]},
                "modes": {m: SYMMETRY[m] for m in MODES},
            },
        }

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def _esm_record(self, seq: str) -> dict:
        """The ESM-2 record of a sequence: it does not depend on any site or CDS."""
        with self._cache_lock:
            rec = self._enc_cache.get(seq)
            if rec is not None:
                self._enc_cache.move_to_end(seq)
                return rec
        rec = self._encoder.encode(
            seq,
            top_k=self._config.get("contact_top_k", 24),
            min_sep=self._config.get("contact_min_sep", 3),
        )
        with self._cache_lock:
            self._enc_cache[seq] = rec
            while len(self._enc_cache) > ENCODER_CACHE_SIZE:
                self._enc_cache.popitem(last=False)
        return rec

    def _encode_once(self, seq: str, cds_dna: Optional[str] = None) -> dict:
        """
        Encode a sequence. The result does not depend on any mutation site, so
        a scan reuses it across every position.

        Round-1 review, R1-13: the scan called this per position on the
        identical, unchanged sequence, so a 600-position heatmap ran 600
        identical ESM-2 forward passes. The docstrings claimed "one encoder
        pass"; the implementation did L.

        A sequence typed into the interface carries no coding sequence unless the
        caller supplies one, so by default the mechanistic channels are masked
        (mech_valid = 0) and E_mech is empty for it - the same handling the
        training pipeline gives a row without a CDS, and the one CDS dropout
        trains for. A validated ``cds_dna`` populates them instead.
        """
        from src.features import mechanistic_features

        rec = dict(self._esm_record(seq))
        rec["mech"] = mechanistic_features(seq, cds_dna, len(seq)).half()
        return rec

    def _prepare_cds(self, seq: str, cds_text: Optional[str]) -> tuple[Optional[str], dict]:
        """(validated DNA or None, info). ValueError for a bad CDS, ModelUnavailable without ViennaRNA."""
        if not cds_text or not str(cds_text).strip():
            return None, {"supplied": False}
        from src.features import VIENNA_AVAILABLE

        if not VIENNA_AVAILABLE:
            raise ModelUnavailable(
                "a coding sequence was supplied but ViennaRNA is not installed on this server, so the "
                "mRNA-folding and codon-usage features cannot be computed. The request is refused rather "
                "than answered with the CDS silently ignored."
            )
        parsed = clean_cds(cds_text, seq)
        return parsed["dna"], {
            "supplied": True, "frame_verified": True, "n_codons": parsed["n_codons"],
            "stop_codon_stripped": parsed["stop_codon_stripped"], "organism": "human",
        }

    def _batch_for_sites(self, rec: dict, sites: list[int], cfg: Optional[dict] = None):
        """Build one batch of per-site graphs from an already-encoded protein."""
        from src.data import build_sample_graph, collate_graphs, move_batch

        cfg = self._config if cfg is None else cfg
        window = cfg.get("window", 256)
        graphs = []
        for site in sites:
            g = build_sample_graph(rec, site, window, cfg)
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
    # structure-free vs SE(3)-invariant 3D
    # ------------------------------------------------------------------
    def _attach_structure(self, rec: dict, seq: str, pdb_text: str) -> tuple[dict, dict]:
        """A copy of ``rec`` carrying aligned backbone coordinates, plus what was aligned."""
        import numpy as np

        from src.geometry import PLDDT_VALID_FLOOR, align_to_sequence, parse_pdb_backbone

        if len(pdb_text) > MAX_PDB_CHARS:
            raise ValueError(f"structure too large ({len(pdb_text)} > {MAX_PDB_CHARS} characters)")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structure.pdb")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(pdb_text)
            pdb_seq, coords, plddt, valid = parse_pdb_backbone(path)
        if not pdb_seq:
            raise ValueError("no protein backbone atoms (N, CA, C) were found in the supplied PDB text")
        L = len(seq)
        offset, mismatches, err = align_to_sequence(pdb_seq, seq)
        if offset is not None:
            # The structure covers the whole sequence (it may carry extra residues around it).
            first, n_cov = 0, L
            sl = slice(offset, offset + L)
            c_full, p_full, v_full = coords[sl], plddt[sl], valid[sl]
        elif err == "structure_shorter_than_sequence":
            # The structure covers only part of the sequence: typically a crystal structure or a
            # design construct missing a terminal tail (the MegaScale 1UBQ model has residues
            # 1-72 of ubiquitin's 76). It is placed as ONE contiguous block inside the sequence,
            # under the same mismatch budget, and every residue it does not cover is masked out of
            # the coordinate stream through coord_valid - the channel that already masks residues
            # below the pLDDT floor. No coordinates are invented for the uncovered residues.
            first, mismatches, err_in = align_to_sequence(seq, pdb_seq)
            if first is None:
                raise ValueError(
                    f"the structure could not be aligned to the sequence ({err_in}; structure has "
                    f"{len(pdb_seq)} residues, sequence has {L}). The 3D mode needs the structure of "
                    f"this protein."
                )
            n_cov = len(pdb_seq)
            if n_cov / L < MIN_STRUCTURE_COVERAGE:
                raise ValueError(
                    f"the structure aligned to residues {first + 1}-{first + n_cov} but covers only "
                    f"{n_cov} of {L} residues ({n_cov / L:.0%}); the 3D mode needs coordinates for at "
                    f"least {MIN_STRUCTURE_COVERAGE:.0%} of the sequence"
                )
            c_full = np.zeros((L,) + tuple(coords.shape[1:]), dtype=np.float32)
            p_full = np.zeros((L,), dtype=np.float32)
            v_full = np.zeros((L,), dtype=np.float32)
            c_full[first:first + n_cov] = coords
            p_full[first:first + n_cov] = plddt
            v_full[first:first + n_cov] = valid
        else:
            raise ValueError(
                f"the structure could not be aligned to the sequence ({err}; structure has {len(pdb_seq)} "
                f"residues, sequence has {L}). The 3D mode needs the structure of this protein."
            )
        torch = self._torch
        covered = np.zeros((L,), dtype=bool)
        covered[first:first + n_cov] = True
        usable = covered & (v_full > 0.5) & (p_full >= PLDDT_VALID_FLOOR)
        rec3 = dict(rec)
        rec3["coords"] = torch.from_numpy(np.ascontiguousarray(c_full)).float()
        rec3["plddt"] = torch.from_numpy(np.ascontiguousarray(p_full)).float()
        rec3["coord_valid"] = torch.from_numpy(usable.astype("float32"))
        low = [i + 1 for i in range(L) if covered[i] and p_full[i] < LOW_PLDDT]
        uncovered = [i + 1 for i in range(L) if not covered[i]]
        info = {
            "source": "supplied PDB", "residues_in_pdb": int(len(pdb_seq)),
            "aligned_offset": int(offset) if offset is not None else 0,
            "mismatches": int(mismatches),
            "mean_plddt": round(float(np.mean(p_full[covered])), 2),
            "fraction_below_70": round(len(low) / max(L, 1), 4),
            "usable_residue_fraction": round(float(usable.mean()), 4),
            "low_confidence_positions": low,
            # which residues of the SEQUENCE carry coordinates (1-based, inclusive)
            "sequence_coverage": {"first": int(first + 1), "last": int(first + n_cov), "covered": int(n_cov),
                                  "length": int(L), "fraction": round(n_cov / max(L, 1), 4)},
            "residues_without_coordinates": uncovered,
            "masked_positions": [i + 1 for i in range(L) if not usable[i]],
            "note": ("the B-factor column is read as pLDDT (the 3D checkpoint was trained on AlphaFold "
                     "models); residues below the pLDDT floor are masked out of the coordinate stream"
                     + (", and so are the residues the structure does not cover, which get no coordinates"
                        if uncovered else "")),
        }
        return rec3, info

    def _resolve_mode(self, mode: str, structure_pdb: Optional[str], seq: str, rec: dict):
        """(model, cfg, rec_for_mode, info) for the requested mode; refuses rather than substitutes."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; use one of {', '.join(MODES)}")
        if mode == "structure_free":
            return self._model, self._config, rec, {
                "mode": mode, "uses_coordinates": False, "symmetry": SYMMETRY[mode]}
        err = self._unavailable_3d()
        if err:
            raise ModelUnavailable(err)
        if not structure_pdb or not str(structure_pdb).strip():
            raise ValueError("mode 'invariant_3d' needs structure_pdb: the structure (PDB text) of this protein")
        rec3, sinfo = self._attach_structure(rec, seq, str(structure_pdb))
        b = self._bundle3d
        return b["model"], b["config"], rec3, {
            "mode": mode, "uses_coordinates": True, "symmetry": SYMMETRY[mode], "structure": sinfo,
            "checkpoint": b["path"]}

    # ------------------------------------------------------------------
    def _ddg_numbers(self, model, cfg: dict, rec: dict, site: int, wt_aa: str, mut_aa: str,
                     calibrated: bool) -> dict:
        from src.substitution import aa_index

        torch = self._torch
        batch = self._batch_for_sites(rec, [site], cfg)
        batch["wt_idx"] = torch.tensor([aa_index(wt_aa)], device=self._device)
        batch["mt_idx"] = torch.tensor([aa_index(mut_aa)], device=self._device)
        with torch.no_grad():
            z = model.context(batch)
            ddg_t, logvar_t = model.ddg(batch, z)
            three = torch.softmax(model.three_state(batch, z), dim=-1)[0]
        ddg = float(ddg_t.item())
        scale = self._sigma_scale if calibrated else 1.0
        sigma = float(torch.exp(0.5 * logvar_t).item()) * scale
        classes = ["destabilizing", "neutral", "stabilizing"]
        # Round-1 review, R1-25: the verdict was a strict sign test with no
        # neutral case, in a payload that simultaneously reported a three-class
        # probability including "neutral" - so a ddG of +0.006 was rendered
        # "stabilizing" while the head's own argmax said neutral. It now comes
        # from the head that has a defined neutral class, and the payload says
        # so.
        verdict = classes[int(torch.argmax(three).item())]
        return {
            "ddg": round(ddg, 4), "sigma": round(sigma, 4),
            "ci_low": round(ddg - Z95 * sigma, 4), "ci_high": round(ddg + Z95 * sigma, 4),
            "three_state": {"destabilizing": round(float(three[0]), 4), "neutral": round(float(three[1]), 4),
                            "stabilizing": round(float(three[2]), 4)},
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    def predict_ddg(self, wt_sequence: str, mutation: str, cds: Optional[str] = None,
                    mode: str = "structure_free", structure_pdb: Optional[str] = None,
                    compare_modes: bool = False) -> dict:
        """
        ddG for one substitution, with the model's own confidence interval.

        With ``cds`` the headline numbers use the real coding sequence and the response
        carries both paths (with and without it) side by side. With ``compare_modes`` the
        other mode (structure-free vs SE(3)-invariant 3D) is computed too.
        """
        from src.substitution import parse_mutation

        self._require()
        seq = clean_sequence(wt_sequence)
        _, site, wt_aa, mut_aa = parse_mutation(seq, mutation)
        cds_dna, cds_info = self._prepare_cds(seq, cds)

        def run(mode_: str, with_cds: bool) -> tuple[dict, dict]:
            rec = self._encode_once(seq, cds_dna if with_cds else None)
            model, cfg, rec_m, minfo = self._resolve_mode(mode_, structure_pdb, seq, rec)
            if "structure" in minfo:
                # whether the mutated residue itself had usable coordinates in the 3D pass
                minfo["structure"]["mutation_site_masked"] = (site + 1) in minfo["structure"]["masked_positions"]
            return self._ddg_numbers(model, cfg, rec_m, site, wt_aa, mut_aa,
                                     calibrated=(mode_ == "structure_free")), minfo

        main, minfo = run(mode, cds_dna is not None)
        cal = self._calibration or {}
        calibrated = bool(cal.get("well_calibrated", False)) and mode == "structure_free"
        out = {
            "mutation": f"{wt_aa}{site + 1}{mut_aa}",
            "position": site + 1,
            "wt_residue": wt_aa,
            "mut_residue": mut_aa,
            **{k: main[k] for k in ("ddg", "sigma", "ci_low", "ci_high")},
            "ci_source": (
                "model predicted sigma (Gaussian, 95%)"
                + (f", rescaled by {self._sigma_scale:.3f} from calibration"
                   if (self._sigma_scale != 1.0 and mode == "structure_free") else "")
                + ("" if mode == "structure_free" else ", not calibrated for the 3D checkpoint")
            ),
            "uncertainty_calibrated": calibrated,
            "three_state": main["three_state"],
            "verdict": main["verdict"],
            "verdict_source": "three-state head argmax",
            **minfo,
            "cds": dict(cds_info),
        }
        if cds_dna is not None:
            without, _ = run(mode, False)
            out["cds"].update({
                "used_in_headline": True,
                "with_cds": {k: main[k] for k in ("ddg", "sigma", "ci_low", "ci_high", "verdict")},
                "without_cds": {k: without[k] for k in ("ddg", "sigma", "ci_low", "ci_high", "verdict")},
                "delta_ddg": round(main["ddg"] - without["ddg"], 4),
                "note": "the model is trained with CDS dropout, so both paths are in-distribution",
            })
        if compare_modes:
            other = "invariant_3d" if mode == "structure_free" else "structure_free"
            second, ominfo = run(other, cds_dna is not None)
            pair = {mode: {k: main[k] for k in ("ddg", "sigma", "ci_low", "ci_high", "verdict")},
                    other: {k: second[k] for k in ("ddg", "sigma", "ci_low", "ci_high", "verdict")}}
            out["comparison"] = {
                **pair, "symmetry": {mode: SYMMETRY[mode], other: SYMMETRY[other]},
                "delta_ddg_3d_minus_structure_free": round(
                    pair["invariant_3d"]["ddg"] - pair["structure_free"]["ddg"], 4),
            }
            if "structure" in ominfo:
                out["comparison"]["structure"] = ominfo["structure"]
        return out

    # ------------------------------------------------------------------
    def mutation_scan_events(self, sequence: str, start: int = 1, end: Optional[int] = None,
                             cds: Optional[str] = None, mode: str = "structure_free",
                             structure_pdb: Optional[str] = None) -> Iterator[dict]:
        """
        Deep mutational scan as a stream of events: ``start``, one ``chunk`` per batch of
        sites (carrying ``done`` / ``total`` and the new columns), then ``result``.

        Everything that can be refused is checked HERE, before the first event, so a bad
        request is an HTTP error and not a half-written stream.

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
        cds_dna, cds_info = self._prepare_cds(seq, cds)
        t_enc = time.perf_counter()
        rec = self._encode_once(seq, cds_dna)          # the one encoder pass
        model, cfg, rec, minfo = self._resolve_mode(mode, structure_pdb, seq, rec)
        encode_ms = (time.perf_counter() - t_enc) * 1000.0

        torch = self._torch
        positions = list(range(start, end + 1))
        sites = [p - 1 for p in positions]
        total = len(positions)

        def gen() -> Iterator[dict]:
            yield {"event": "start", "data": {
                "length": len(seq), "positions": [positions[0], positions[-1]], "total": total,
                "aa_order": list(AA_ORDER), **minfo, "cds": cds_info}}
            columns: list[list[float]] = []
            t0 = time.perf_counter()
            for lo in range(0, total, SCAN_SITE_BATCH):
                chunk = sites[lo:lo + SCAN_SITE_BATCH]
                batch = self._batch_for_sites(rec, chunk, cfg)
                batch["wt_idx"] = torch.tensor(
                    [aa_index(seq[site]) for site in chunk], device=self._device
                )
                with torch.no_grad():
                    z = model.context(batch)
                    rows = model.scan(batch, z)
                new = [[round(float(v), 4) for v in row[:len(AA_ORDER)]] for row in rows]
                columns.extend(new)
                done = len(columns)
                elapsed = time.perf_counter() - t0
                yield {"event": "chunk", "data": {
                    "done": done, "total": total, "positions": positions[done - len(new):done],
                    "columns": new, "elapsed_s": round(elapsed, 3),
                    "eta_s": round(elapsed * (total - done) / max(done, 1), 3)}}
            scan_ms = (time.perf_counter() - t0) * 1000.0
            # Transpose to rows = amino acids, columns = positions.
            matrix = [[columns[c][r] for c in range(total)] for r in range(len(AA_ORDER))]
            n_mut = total * len(AA_ORDER)
            yield {"event": "result", "data": {
                "positions": positions,
                "wt_residues": [seq[p - 1] for p in positions],
                "aa_order": list(AA_ORDER),
                "matrix": matrix,
                **minfo,
                "cds": cds_info,
                "timing": {"encode_ms": round(encode_ms, 1), "scan_ms": round(scan_ms, 1),
                           "total_mutations": n_mut,
                           "muts_per_sec": round(n_mut / max(scan_ms / 1000.0, 1e-6), 1)},
            }}

        return gen()

    def mutation_scan(self, sequence: str, start: int = 1, end: Optional[int] = None,
                      cds: Optional[str] = None, mode: str = "structure_free",
                      structure_pdb: Optional[str] = None) -> dict:
        """Deep mutational scan: ddG for every substitution over a window (blocking form)."""
        result = None
        for ev in self.mutation_scan_events(sequence, start, end, cds, mode, structure_pdb):
            if ev["event"] == "result":
                result = ev["data"]
        return result

    # ------------------------------------------------------------------
    def epistasis_events(self, wt_sequence: str, pairs: list[str], cds: Optional[str] = None
                         ) -> Iterator[dict]:
        """
        Double-mutant epistasis as a stream of events (``start``, ``chunk``, ``result``).

        ``ddg_total = ddg_additive + epistasis_dddg``; the epistasis term is the contact-gated
        Potts head, gated by the PREDICTED contact graph (no coordinates). Served only when the
        checkpoint's epistasis head was trained; otherwise refused with the reason. Pairs are
        reported in ascending site order, whatever order they were typed in, because the head is
        exactly symmetric under swapping the two sites.
        """
        from src.graphs import build_geometric_graph, pair_contact_features
        from src.substitution import aa_index, parse_multi_mutation

        self._require()
        if not self._epistasis["available"]:
            raise ModelUnavailable(self._epistasis["reason"])
        seq = clean_sequence(wt_sequence)
        specs = [s.strip() for s in (pairs or []) if s and s.strip()]
        if not specs:
            raise ValueError("no mutation pairs supplied; use e.g. L8P+V5A")
        if len(specs) > MAX_EPISTASIS_PAIRS:
            raise ValueError(f"too many pairs ({len(specs)} > {MAX_EPISTASIS_PAIRS})")
        parsed = []
        for spec in specs:
            _, muts = parse_multi_mutation(seq, spec)
            if len(muts) != 2:
                raise ValueError(f"epistasis needs exactly two substitutions at two different sites, got {spec!r}")
            parsed.append((spec, muts[0], muts[1]))
        cds_dna, cds_info = self._prepare_cds(seq, cds)

        torch = self._torch
        model, cfg = self._model, self._config
        t_enc = time.perf_counter()
        rec = self._encode_once(seq, cds_dna)
        encode_ms = (time.perf_counter() - t_enc) * 1000.0
        L = len(seq)
        edge_geo, attr_geo = build_geometric_graph(
            L, rec["contact_edges"].long(), rec["contact_probs"].float(),
            top_k=cfg.get("graph_top_k", 16), threshold=cfg.get("geo_threshold", 0.30),
        )
        uniq = sorted({m[0] for _, a, b in parsed for m in (a, b)})
        where = {s: i for i, s in enumerate(uniq)}
        gated = bool(getattr(model.epistasis_head, "use_contact_gate", True))
        head = {"trained": True, "source": self._epistasis.get("source"), "contact_gated": gated,
                "gate_input": ("predicted ESM-2 contact graph E_geo (no coordinates)" if gated
                               else "none: ungated pairwise term")}

        def gen() -> Iterator[dict]:
            yield {"event": "start", "data": {"length": L, "total": len(parsed), "cds": cds_info,
                                              "epistasis_head": head}}
            t0 = time.perf_counter()
            zs = []
            for lo in range(0, len(uniq), SCAN_SITE_BATCH):
                batch = self._batch_for_sites(rec, uniq[lo:lo + SCAN_SITE_BATCH], cfg)
                with torch.no_grad():
                    zs.append(model.context(batch))
            z_all = torch.cat(zs, dim=0)
            lp_full = rec.get("wt_logprobs")
            rows, fresh = [], []
            every = max(1, len(parsed) // 20)
            for k, (spec, (s1, w1, m1), (s2, w2, m2)) in enumerate(parsed):
                zi = z_all[where[s1]:where[s1] + 1]
                zj = z_all[where[s2]:where[s2] + 1]
                idx = lambda a: torch.tensor([aa_index(a)], device=self._device)  # noqa: E731
                pg = pair_contact_features(edge_geo, attr_geo, s1, s2, L).unsqueeze(0).to(self._device)
                lp_i = lp_full[s1:s1 + 1].float().to(self._device) if lp_full is not None else None
                lp_j = lp_full[s2:s2 + 1].float().to(self._device) if lp_full is not None else None
                with torch.no_grad():
                    ddg_ab, dddg, lv_ab = model.epistasis(
                        zi, zj, idx(w1), idx(m1), idx(w2), idx(m2), pair_geo=pg, lp_i=lp_i, lp_j=lp_j)
                total = float(ddg_ab.item())
                epi = float(dddg.item())
                sigma = float(torch.exp(0.5 * lv_ab).item()) * self._sigma_scale
                geo = [float(v) for v in pg[0].tolist()]
                row = {
                    "pair": f"{w1}{s1 + 1}{m1}+{w2}{s2 + 1}{m2}", "sites": [s1 + 1, s2 + 1],
                    "ddg_total": round(total, 4), "ddg_additive": round(total - epi, 4),
                    "epistasis_dddg": round(epi, 4), "sigma": round(sigma, 4),
                    "ci_low": round(total - Z95 * sigma, 4), "ci_high": round(total + Z95 * sigma, 4),
                    "contact_prob": round(geo[0], 4), "seq_separation": abs(s2 - s1),
                    "is_contact": bool(geo[3] > 0.5),
                    "pair_geo": {"direct_contact_prob": round(geo[0], 4), "log_seq_sep": round(geo[1], 4),
                                 "inv_hop_dist": round(geo[2], 4), "is_direct_contact": round(geo[3], 4)},
                }
                rows.append(row)
                fresh.append(row)
                if (k + 1) % every == 0 or k + 1 == len(parsed):
                    elapsed = time.perf_counter() - t0
                    yield {"event": "chunk", "data": {
                        "done": k + 1, "total": len(parsed), "pairs": fresh, "elapsed_s": round(elapsed, 3),
                        "eta_s": round(elapsed * (len(parsed) - k - 1) / (k + 1), 3)}}
                    fresh = []
            pred_ms = (time.perf_counter() - t0) * 1000.0
            yield {"event": "result", "data": {
                "length": L, "n_pairs": len(rows), "pairs": rows,
                "mode": "structure_free", "uses_coordinates": False, "symmetry": SYMMETRY["structure_free"],
                "epistasis_head": head, "cds": cds_info,
                "ci_source": "model predicted sigma (Gaussian, 95%) of the additive single-mutant parts"
                             + (f", rescaled by {self._sigma_scale:.3f}" if self._sigma_scale != 1.0 else ""),
                "uncertainty_calibrated": False,
                "uncertainty_note": ("calibration was measured on single mutants only, and the epistasis "
                                     "term itself has no uncertainty estimate: the interval is for ddg_total "
                                     "and is not a calibrated claim about the epistasis term"),
                "timing": {"encode_ms": round(encode_ms, 1), "predict_ms": round(pred_ms, 1),
                           "pairs_per_sec": round(len(rows) / max(pred_ms / 1000.0, 1e-6), 1)},
            }}

        return gen()

    def predict_epistasis(self, wt_sequence: str, pairs: list[str], cds: Optional[str] = None) -> dict:
        """Double-mutant epistasis (blocking form)."""
        result = None
        for ev in self.epistasis_events(wt_sequence, pairs, cds):
            if ev["event"] == "result":
                result = ev["data"]
        return result

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


def clean_cds(cds: str, protein: str) -> dict:
    """
    Validate a coding sequence as a real reading frame that encodes ``protein``.

    The mechanistic channels are only meaningful for the CDS that actually encodes the
    protein, so a frame error or a single mismatching codon is refused with where it is,
    not tolerated: the training tables are 412/412 frame-verified for the same reason.
    Accepts DNA or RNA (U is read as T) and a FASTA header; one terminal stop codon is
    stripped. Non-standard residues in ``protein`` (X, B, Z, ...) are not compared.
    """
    from src.codon_tables import GENETIC_CODE
    from src.substitution import AA_ORDER

    lines = [ln for ln in str(cds).splitlines() if not ln.startswith(">")]
    dna = "".join("".join(lines).split()).upper().replace("U", "T")
    if not dna:
        raise ValueError("empty coding sequence")
    bad = sorted(set(dna) - set("ACGT"))
    if bad:
        raise ValueError(f"invalid nucleotide symbol(s) in the coding sequence: {''.join(bad)}. "
                         f"Use A, C, G and T (or U).")
    if len(dna) % 3:
        raise ValueError(f"the coding sequence length ({len(dna)}) is not a multiple of 3, so it has "
                         f"no reading frame")
    codons = [dna[i:i + 3] for i in range(0, len(dna), 3)]
    stop_stripped = False
    if GENETIC_CODE.get(codons[-1]) == "*":
        codons = codons[:-1]
        stop_stripped = True
    aa = []
    for k, codon in enumerate(codons):
        a = GENETIC_CODE.get(codon)
        if a is None or a == "*":
            raise ValueError(f"internal stop codon {codon} at codon {k + 1}: not a coding sequence "
                             f"for this protein")
        aa.append(a)
    if len(aa) != len(protein):
        raise ValueError(f"the coding sequence encodes {len(aa)} residues but the protein has "
                         f"{len(protein)}")
    standard = set(AA_ORDER)
    mismatches = [(k, a, p) for k, (a, p) in enumerate(zip(aa, protein)) if p in standard and a != p]
    if mismatches:
        k, a, p = mismatches[0]
        raise ValueError(f"the coding sequence does not encode this protein: codon {k + 1} "
                         f"({codons[k]}) is {a} but the protein has {p} "
                         f"({len(mismatches)} mismatching codon(s) in total)")
    return {"dna": "".join(codons), "n_codons": len(codons), "stop_codon_stripped": stop_stripped}


@lru_cache(maxsize=1)
def get_engine() -> HoloGeoNEngine:
    """Process-wide singleton."""
    return HoloGeoNEngine()
