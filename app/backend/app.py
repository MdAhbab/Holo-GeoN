"""
app/backend/app.py
==================
FastAPI HTTP layer for the Holo-GeoN web interface.

Inference lives entirely in inference.py; this file is HTTP only.

Every prediction endpoint requires a loaded model. There is no demo mode and no
heuristic fallback, so a request made without a trained checkpoint returns 503
with the reason rather than a plausible-looking number. `demo_mode` has been
removed from the API contract, the database schema and the frontend types.

v2 additions (see app/API_V2.md): /api/epistasis (+ /stream), optional `cds` on
/api/ddg and /api/scan with both paths side by side, a structure-free vs
SE(3)-invariant-3D `mode`, /api/scan/stream for live progress, and pLDDT bands on
/api/compare. Each refuses with 503/400 and the reason when it cannot be served
honestly (untrained epistasis head, no ViennaRNA, no 3D checkpoint, bad CDS frame).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as _requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import ModelUnavailable, get_engine
from db import (
    init_db,
    insert_prediction,
    list_predictions,
    get_prediction,
    delete_prediction,
    clear_history,
)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Holo-GeoN API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction history is persisted in a local SQLite file (see db.py) so results
# survive page refreshes and server restarts.
init_db()

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class DdgRequest(BaseModel):
    wt_sequence: str
    mutation: str
    # Optional real coding sequence: validated as a reading frame that encodes this very
    # protein; the response then carries the with-CDS and without-CDS paths side by side.
    cds: Optional[str] = None
    # "structure_free" (default; reads no coordinates) or "invariant_3d" (needs the optional
    # 3D checkpoint and structure_pdb). compare_modes returns both, with the delta.
    mode: str = "structure_free"
    structure_pdb: Optional[str] = None
    compare_modes: bool = False


class ScanRequest(BaseModel):
    sequence: str
    start: int = 1
    end: Optional[int] = None
    cds: Optional[str] = None
    mode: str = "structure_free"
    structure_pdb: Optional[str] = None


class EpistasisRequest(BaseModel):
    wt_sequence: str
    pairs: List[str]               # e.g. ["L8P+V5A", "I13A+L15V"]
    cds: Optional[str] = None


class IdrRequest(BaseModel):
    sequence: str


class CompareRequest(BaseModel):
    uniprot_id: str


class ExportRequest(BaseModel):
    format: str          # "csv" | "json" | "pdb"
    filename: Optional[str] = None
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _error(status: int, detail: str) -> Response:
    """Return a JSON error response."""
    return Response(
        content=json.dumps({"detail": detail}),
        status_code=status,
        media_type="application/json",
    )


def _as_dict(model: BaseModel) -> Dict[str, Any]:
    """Pydantic v1/v2-safe dict() for persisting the request payload."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _unavailable(exc: ModelUnavailable) -> Response:
    """503 with the reason. Never a substitute prediction."""
    return _error(503, str(exc))


def _loggable(model: BaseModel) -> Dict[str, Any]:
    """The request as stored in history: a supplied structure is replaced by its size."""
    d = _as_dict(model)
    if d.get("structure_pdb"):
        d["structure_pdb"] = f"<{len(d['structure_pdb'])} characters of PDB text>"
    return d


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream(events, kind: str, req: BaseModel, summarize) -> StreamingResponse:
    """
    Server-sent events for a long request: ``start``, ``chunk`` (progress + the new columns /
    pairs), then ``result``, or ``error`` carrying the status the blocking endpoint would have
    returned. Anything that can be refused was refused before this was called, so a bad
    request is an HTTP error, not a half-written stream. The final result is stored in
    history exactly as the blocking endpoint stores it.
    """
    def gen():
        final = None
        try:
            for ev in events:
                if ev["event"] == "result":
                    final = ev["data"]
                yield _sse(ev["event"], ev["data"])
        except ModelUnavailable as exc:
            yield _sse("error", {"status": 503, "detail": str(exc)})
            return
        except ValueError as exc:
            yield _sse("error", {"status": 400, "detail": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"status": 500, "detail": f"{exc.__class__.__name__}: {exc}"})
            return
        if final is not None:
            insert_prediction(kind, summarize(final), _loggable(req), final)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _scan_summary(result: Dict[str, Any]) -> str:
    pos = result["positions"]
    tags = []
    if (result.get("cds") or {}).get("supplied"):
        tags.append("with CDS")
    if result.get("mode") == "invariant_3d":
        tags.append("3D")
    suffix = f" · {', '.join(tags)}" if tags else ""
    return (f"Scan {len(pos)} positions (residues {pos[0]}–{pos[-1]}){suffix}"
            if pos else "Scan (empty window)")


def _epistasis_summary(result: Dict[str, Any]) -> str:
    return f"Epistasis · {result['n_pairs']} pair(s) · L={result['length']}"


def _plddt_summary(plddt: List[float]) -> Dict[str, Any]:
    """AlphaFold's own confidence bands over a per-residue pLDDT list."""
    if not plddt:
        return {}
    bands = {
        "very_high_gt_90": sum(1 for v in plddt if v > 90),
        "confident_70_90": sum(1 for v in plddt if 70 < v <= 90),
        "low_50_70": sum(1 for v in plddt if 50 < v <= 70),
        "very_low_le_50": sum(1 for v in plddt if v <= 50),
    }
    return {
        "plddt_mean": round(sum(plddt) / len(plddt), 2),
        "plddt_bands": bands,
        "low_confidence_positions": [i + 1 for i, v in enumerate(plddt) if v < 70],
        "plddt_note": (
            "pLDDT describes the AlphaFold model drawn here, not the prediction: the structure-free model "
            "reads no coordinates. Residues below 70 are usually disordered or poorly modelled, where a "
            "folded-state ddG is less meaningful, so read the heatmap there with the same caution."
        ),
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict:
    eng = get_engine()
    return eng.health()


@app.post("/api/ddg")
def ddg(req: DdgRequest):
    eng = get_engine()
    try:
        result = eng.predict_ddg(
            req.wt_sequence, req.mutation, cds=req.cds, mode=req.mode,
            structure_pdb=req.structure_pdb, compare_modes=req.compare_modes,
        )
    except ModelUnavailable as exc:
        return _unavailable(exc)
    except ValueError as exc:
        return _error(400, str(exc))
    summary = (f"ΔΔG {result['mutation']} = {result['ddg']:+.2f} kcal/mol "
               f"({result['verdict']})")
    insert_prediction("ddg", summary, _loggable(req), result)
    return result


@app.post("/api/scan")
def scan(req: ScanRequest):
    eng = get_engine()
    try:
        result = eng.mutation_scan(req.sequence, req.start, req.end, cds=req.cds, mode=req.mode,
                                   structure_pdb=req.structure_pdb)
    except ModelUnavailable as exc:
        return _unavailable(exc)
    except ValueError as exc:
        return _error(400, str(exc))
    insert_prediction("scan", _scan_summary(result), _loggable(req), result)
    return result


@app.post("/api/scan/stream")
def scan_stream(req: ScanRequest):
    """The same scan as /api/scan, streamed per batch of sites so a long scan shows progress."""
    eng = get_engine()
    try:
        events = eng.mutation_scan_events(req.sequence, req.start, req.end, cds=req.cds,
                                          mode=req.mode, structure_pdb=req.structure_pdb)
    except ModelUnavailable as exc:
        return _unavailable(exc)
    except ValueError as exc:
        return _error(400, str(exc))
    return _stream(events, "scan", req, _scan_summary)


@app.post("/api/epistasis")
def epistasis(req: EpistasisRequest):
    """Double-mutant epistasis from the contact-gated Potts head; refused if that head is untrained."""
    eng = get_engine()
    try:
        result = eng.predict_epistasis(req.wt_sequence, req.pairs, cds=req.cds)
    except ModelUnavailable as exc:
        return _unavailable(exc)
    except ValueError as exc:
        return _error(400, str(exc))
    insert_prediction("epistasis", _epistasis_summary(result), _loggable(req), result)
    return result


@app.post("/api/epistasis/stream")
def epistasis_stream(req: EpistasisRequest):
    """The same request as /api/epistasis, streamed in batches of pairs."""
    eng = get_engine()
    try:
        events = eng.epistasis_events(req.wt_sequence, req.pairs, cds=req.cds)
    except ModelUnavailable as exc:
        return _unavailable(exc)
    except ValueError as exc:
        return _error(400, str(exc))
    return _stream(events, "epistasis", req, _epistasis_summary)


@app.post("/api/idr")
def idr(req: IdrRequest):
    eng = get_engine()
    try:
        result = eng.idr_ensemble(req.sequence)
    except ModelUnavailable as exc:
        return _unavailable(exc)
    except ValueError as exc:
        return _error(400, str(exc))
    summary = (f"IDR ensemble · L={result['length']} · "
               f"Rg μ={result['mu']:.1f} σ={result['sigma']:.2f}")
    insert_prediction("idr", summary, _as_dict(req), result)
    return result


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Fetch AlphaFold2 structure + PAE, run the Holo-GeoN scan, return combined payload."""
    eng = get_engine()
    if not eng.ready:
        return _error(503, eng.health()["error"] or "model unavailable")
    uid = req.uniprot_id.strip().upper()

    try:
        # 1. AlphaFold metadata
        af_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
        meta_resp = _requests.get(af_url, timeout=20)
        if meta_resp.status_code == 404:
            return _error(502, f"AlphaFold entry not found for {uid}")
        meta_resp.raise_for_status()
        meta = meta_resp.json()[0]

        pdb_url = meta.get("pdbUrl", "")
        pae_doc_url = meta.get("paeDocUrl", meta.get("paeImageUrl", ""))
        sequence = meta.get("uniprotSequence", "")

        if not pdb_url:
            return _error(502, f"AlphaFold entry for {uid} has no downloadable PDB URL.")

        # 2. Fetch PDB structure text
        pdb_resp = _requests.get(pdb_url, timeout=20)
        pdb_resp.raise_for_status()
        structure_pdb: str = pdb_resp.text

        # 3. Parse pLDDT from B-factor column of CA atoms
        plddt: List[float] = []
        for line in structure_pdb.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    plddt.append(float(line[60:66]))
                except (ValueError, IndexError):
                    pass

        # 4. Fetch PAE document
        pae_matrix: List[List[float]] = []
        pae_downsampled = False
        # Residues per displayed PAE cell. Cell k covers residue 1 + k * pae_stride, so the
        # interface can label the axes in residue numbers rather than in cell indices.
        pae_stride = 1
        if pae_doc_url and not pae_doc_url.endswith(".png"):
            pae_resp = _requests.get(pae_doc_url, timeout=20)
            pae_resp.raise_for_status()
            pae_json = pae_resp.json()
            # Handle both flat {"predicted_aligned_error": ...}
            # and list format [{"predicted_aligned_error": ...}]
            if isinstance(pae_json, list):
                raw_pae = pae_json[0]["predicted_aligned_error"]
            else:
                raw_pae = pae_json["predicted_aligned_error"]

            n = len(raw_pae)
            if n > 200:
                # Ceiling division: `n // 200` is 1 for 200 < n < 400, which kept every cell
                # while still reporting the matrix as downsampled.
                pae_stride = -(-n // 200)
                raw_pae = [row[::pae_stride] for row in raw_pae[::pae_stride]]
                pae_downsampled = True
            pae_matrix = [[float(v) for v in row] for row in raw_pae]

        # 5. Deep mutational scan -> per-position most-destabilising ddG
        min_ddg_per_position: List[float] = []
        if sequence:
            scan_end = min(len(sequence), 400)
            try:
                scan_result = eng.mutation_scan(sequence, 1, scan_end)
                matrix = scan_result["matrix"]  # 20 rows × N cols
                n_cols = len(scan_result["positions"])
                for col in range(n_cols):
                    col_vals = [matrix[row][col] for row in range(len(matrix))]
                    min_ddg_per_position.append(round(min(col_vals), 4))
            except (ValueError, ModelUnavailable):
                min_ddg_per_position = []

        result = {
            "uniprot_id": uid,
            "sequence": sequence,
            "structure_pdb": structure_pdb,
            "plddt": plddt,
            **_plddt_summary(plddt),
            # aligned with min_ddg_per_position: True where the AlphaFold model is low confidence
            "low_confidence_in_scan": [
                bool(plddt[i] < 70) if i < len(plddt) else False
                for i in range(len(min_ddg_per_position))
            ],
            "pae": pae_matrix,
            "pae_downsampled": pae_downsampled,
            "pae_stride": pae_stride,
            "min_ddg_per_position": min_ddg_per_position,
        }
        summary = f"AlphaFold compare · {uid} · {len(sequence)} residues"
        insert_prediction("compare", summary, _as_dict(req), result)
        return result

    except _requests.exceptions.Timeout:
        return _error(502, f"Request to AlphaFold timed out for {uid}")
    except _requests.exceptions.ConnectionError:
        return _error(502, f"Could not connect to AlphaFold API for {uid}")
    except (_requests.exceptions.HTTPError, KeyError, IndexError, ValueError) as exc:
        return _error(502, f"AlphaFold entry not found or parse error for {uid}: {exc}")


@app.post("/api/export")
def export(req: ExportRequest):
    """Return data as a downloadable file (CSV, JSON, or PDB)."""
    fmt = req.format.lower()
    data = req.data

    # ------------------------------------------------------------------ JSON
    if fmt == "json":
        fname = req.filename or "holo_geon_results.json"
        content = json.dumps(data, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    # ------------------------------------------------------------------ CSV
    elif fmt == "csv":
        fname = req.filename or "holo_geon_results.csv"
        buf = io.StringIO()

        is_scan = ("matrix" in data and "positions" in data and "aa_order" in data)
        if is_scan:
            positions: List[int] = data["positions"]
            wt_residues: List[str] = data.get("wt_residues", ["?"] * len(positions))
            aa_order: List[str] = data["aa_order"]
            matrix: List[List[float]] = data["matrix"]  # 20 × N

            # Header: position, wt, then one col per amino acid
            header = ["position", "wt"] + aa_order
            buf.write(",".join(header) + "\n")

            for col_idx, pos in enumerate(positions):
                wt = wt_residues[col_idx] if col_idx < len(wt_residues) else "?"
                row_vals = [str(matrix[row_idx][col_idx]) for row_idx in range(len(aa_order))]
                buf.write(",".join([str(pos), wt] + row_vals) + "\n")
        else:
            # Flatten top-level scalar keys into key/value CSV
            buf.write("key,value\n")
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    buf.write(f"{k},{v}\n")

        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    # ------------------------------------------------------------------ PDB
    elif fmt == "pdb":
        fname = req.filename or "holo_geon_annotated.pdb"

        # Determine sequence
        sequence: str = ""
        if "sequence" in data and isinstance(data["sequence"], str):
            sequence = data["sequence"].upper()
        elif "wt_residues" in data:
            sequence = "".join(data["wt_residues"])

        # Determine B-factor annotation
        if "bfactor" in data and isinstance(data["bfactor"], list):
            bfactors: List[float] = [float(v) for v in data["bfactor"]]
        elif "matrix" in data and "positions" in data:
            # per-position min ΔΔG from scan matrix
            matrix = data["matrix"]
            n_cols = len(data["positions"])
            bfactors = []
            for col in range(n_cols):
                col_vals = [matrix[row][col] for row in range(len(matrix))]
                bfactors.append(min(col_vals))
        else:
            bfactors = [0.0] * len(sequence)

        # Pad / trim bfactors to match sequence length
        n = len(sequence)
        bfactors = (bfactors + [0.0] * n)[:n]

        pdb_lines: List[str] = []
        # Round-2 review, R2-17: this writes CA atoms on a boustrophedon grid -
        # invented coordinates, carrying no structural information whatsoever -
        # into a file format whose entire purpose is to carry structure. A .pdb
        # downloads, opens in PyMOL, and looks like a prediction. The REMARK
        # records below travel inside the file, so the disclaimer cannot be
        # separated from it the way a note on a web page can.
        pdb_lines += [
            "REMARK 999 GENERATED BY HOLO-GEON. THE COORDINATES IN THIS FILE ARE",
            "REMARK 999 NOT A STRUCTURE PREDICTION AND CARRY NO STRUCTURAL",
            "REMARK 999 INFORMATION. CA ATOMS ARE PLACED ON A REGULAR 2D GRID",
            "REMARK 999 PURELY SO PER-RESIDUE VALUES CAN BE CARRIED IN THE",
            "REMARK 999 B-FACTOR COLUMN FOR COLOURING. THE MODEL PREDICTS",
            "REMARK 999 STABILITY EFFECTS, NOT COORDINATES. DO NOT INTERPRET",
            "REMARK 999 GEOMETRY, DISTANCES, CONTACTS OR FOLD FROM THIS FILE.",
            "REMARK 999 B-FACTOR COLUMN = MINIMUM PREDICTED DDG PER POSITION.",
        ]
        ca_spacing = 3.8  # Å
        # Lay residues in a boustrophedon grid so the x-coordinate field (%8.3f,
        # max 8 chars) never overflows for long chains (e.g. AlphaFold inputs).
        per_row = 250                      # 250 * 3.8 = 950 Å < 9999.999

        for i, aa1 in enumerate(sequence):
            resname = _AA1_TO_3.get(aa1, "GLY")
            serial = (i + 1) % 100000       # ATOM serial field is 5 chars
            resseq = (i + 1) % 10000        # resSeq field is 4 chars
            bf = bfactors[i] if i < len(bfactors) else 0.0
            row, col = divmod(i, per_row)
            x = col * ca_spacing
            y = row * ca_spacing
            # Standard 80-column PDB ATOM record.
            line = (
                f"ATOM  {serial:>5}  CA  {resname} A{resseq:>4}    "
                f"{x:>8.3f}{y:>8.3f}{0.0:>8.3f}"
                f"{1.00:>6.2f}{bf:>6.2f}          "
                f" C  "
            )
            pdb_lines.append(line)

        pdb_lines.append("END")
        content = "\n".join(pdb_lines) + "\n"

        return Response(
            content=content,
            media_type="chemical/x-pdb",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    else:
        return _error(400, f"Unknown export format '{req.format}'. Use csv, json, or pdb.")


# ---------------------------------------------------------------------------
# Prediction history (SQLite-backed)
# ---------------------------------------------------------------------------

@app.get("/api/history")
def history(kind: Optional[str] = None, limit: int = 50):
    """Most-recent-first prediction history (metadata only). Optional ?kind=."""
    return {"items": list_predictions(kind, limit)}


@app.get("/api/history/{pred_id}")
def history_item(pred_id: int):
    """Full record (request + response payload) for one prediction."""
    item = get_prediction(pred_id)
    if item is None:
        return _error(404, f"No prediction with id {pred_id}.")
    return item


@app.delete("/api/history/{pred_id}")
def history_delete(pred_id: int):
    """Delete a single history entry."""
    if not delete_prediction(pred_id):
        return _error(404, f"No prediction with id {pred_id}.")
    return {"deleted": pred_id}


@app.delete("/api/history")
def history_clear(kind: Optional[str] = None):
    """Clear all history (or just one ?kind=)."""
    return {"cleared": clear_history(kind)}


# ---------------------------------------------------------------------------
# Frontend static files (mount LAST so /api routes are not shadowed)
# ---------------------------------------------------------------------------
_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "note": (
                "Frontend not built yet. "
                "Run `npm run build` inside frontend/ to generate the dist/ directory, "
                "then restart the server."
            )
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
