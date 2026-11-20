#!/usr/bin/env python3
"""
holo.py — User-friendly cross-platform CLI for Holo-GeoN
========================================================

Designed for biologists, Windows users, and live GPU/G4 real-time runs without
requiring Linux shell syntax, PYTHONPATH exports, or bash scripts.

Quick examples (PowerShell, Command Prompt, macOS, Linux, or Colab G4):

    python holo.py                                # Interactive guided menu
    python holo.py doctor                         # Verify GPU, packages & weights
    python holo.py predict --uniprot P0CG48 --mut L8P
    python holo.py predict --fasta prot.fasta --muts L8P,V5A --cds mrna.fasta
    python holo.py scan --fasta prot.fasta --out scan.csv --html heatmap.html
    python holo.py epistasis --fasta prot.fasta --pairs L8P+V5A,I13A+L15V
    python holo.py batch --input mutations.csv --output scored.csv
    python holo.py gui --weights runs/main/model_best.pt --port 8000

Sign convention (internal Holo-GeoN convention):
    ddG = dG_mut - dG_wt   (positive = stabilizing, negative = destabilizing)
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Z95 = 1.959964
DEFAULT_WEIGHTS = ROOT / "runs" / "main" / "model_best.pt"
DEFAULT_CALIBRATION = ROOT / "runs" / "main" / "calibration.json"


def _configure_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sequence & FASTA / UniProt loading
# ---------------------------------------------------------------------------
def read_fasta_first(path: str) -> tuple[str, str]:
    """Read the first record of a FASTA or plain sequence file -> (header, seq)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    header = p.stem
    parts: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                break
            header = line[1:].strip() or header
        else:
            parts.append("".join(line.split()))
    return header, "".join(parts).upper()


def fetch_uniprot_sequence(accession: str, timeout: float = 30.0) -> tuple[str, str]:
    """Fetch a canonical protein sequence from UniProt REST API."""
    acc = accession.strip().upper()
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "Holo-GeoN-CLI/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"UniProt lookup failed for {acc}: HTTP {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not reach UniProt for {acc}: {exc}") from exc
    header = acc
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            header = line[1:].strip() or acc
        elif line:
            parts.append(line)
    if not parts:
        raise RuntimeError(f"UniProt returned an empty sequence for {acc}")
    return header, "".join(parts).upper()


def resolve_sequence(args) -> tuple[str, str]:
    if getattr(args, "seq", None):
        return "inline_sequence", "".join(args.seq.split()).upper()
    if getattr(args, "fasta", None):
        return read_fasta_first(args.fasta)
    if getattr(args, "uniprot", None):
        print(f"[holo] Fetching UniProt {args.uniprot} ...")
        return fetch_uniprot_sequence(args.uniprot)
    raise SystemExit("Provide one of --seq, --fasta <file>, or --uniprot <ACC>.")


def resolve_cds(cds_arg: str | None) -> str | None:
    if not cds_arg:
        return None
    if os.path.isfile(cds_arg):
        _, dna = read_fasta_first(cds_arg)
        return dna
    return "".join(cds_arg.split()).upper()


# ---------------------------------------------------------------------------
# Lightweight inference engine (works with v1 & v2 checkpoints)
# ---------------------------------------------------------------------------
class CLIEngine:
    """Direct single-pass inference engine for CLI predictions, DMS & epistasis."""

    def __init__(self, weights_path: Path, calibration_path: Path | None = None,
                 device_req: str | None = None):
        import torch

        from src.encoder import ESM_DIMS, EsmEncoder
        from src.model import HoloGeoN, model_kwargs_from_config

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"No trained checkpoint found at {weights_path}.\n"
                "Pass --weights <path/to/model_best.pt> or download pre-trained "
                "weights from our Zenodo release."
            )

        self.torch = torch
        if device_req:
            self.device = torch.device(device_req)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.perf_counter()
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.cfg = state.get("config", {})
        esm_dim = state.get("esm_dim", ESM_DIMS.get(self.cfg.get("esm_model"), 1280))
        kwargs = state.get("model_kwargs") or model_kwargs_from_config(self.cfg, esm_dim)
        self.model = HoloGeoN(**kwargs).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.weights_path = weights_path

        # Load optional calibration variance scale
        self.sigma_scale = 1.0
        self.calibrated = False
        cal_path = calibration_path or (weights_path.parent / "calibration.json")
        if cal_path.is_file():
            try:
                payload = json.loads(cal_path.read_text(encoding="utf-8"))
                vs = payload.get("variance_scale") or {}
                if vs.get("scale"):
                    self.sigma_scale = float(vs["scale"])
                after = vs.get("calibration_after_scaling") or payload.get("calibration") or {}
                self.calibrated = bool(after.get("well_calibrated", False))
            except Exception:
                pass

        esm_name = self.cfg.get("esm_model", "facebook/esm2_t33_650M_UR50D")
        print(f"[holo] Loaded Holo-GeoN ({self.model.arch}, "
              f"{self.model.parameter_count()['total']:,} params) on {self.device} "
              f"in {time.perf_counter() - t0:.2f}s")
        print(f"[holo] Loading frozen encoder {esm_name} ...")
        t1 = time.perf_counter()
        self.encoder = EsmEncoder(esm_name, device=str(self.device))
        print(f"[holo] Encoder ready in {time.perf_counter() - t1:.2f}s")
        self._enc_cache: dict[tuple[str, str | None], dict] = {}

    def encode_protein(self, seq: str, cds: str | None = None) -> tuple[dict, float]:
        from src.features import mechanistic_features

        key = (seq, cds)
        if key in self._enc_cache:
            return self._enc_cache[key], 0.0
        t0 = time.perf_counter()
        rec = self.encoder.encode(
            seq,
            top_k=self.cfg.get("contact_top_k", 24),
            min_sep=self.cfg.get("contact_min_sep", 3),
        )
        rec["mech"] = mechanistic_features(seq, cds, len(seq)).half()
        self._enc_cache[key] = rec
        return rec, (time.perf_counter() - t0) * 1000.0

    def _batch_sites(self, rec: dict, sites: list[int]):
        from src.data import build_sample_graph, collate_graphs, move_batch

        window = int(self.cfg.get("window", 256))
        graphs = []
        for s in sites:
            g = build_sample_graph(rec, s, window, self.cfg)
            g.update({"wt_idx": 0, "mt_idx": 0, "y": 0.0, "sigma": float("nan")})
            graphs.append(g)
        return move_batch(collate_graphs(graphs), self.device)

    def predict_mutations(self, seq: str, mutations: list[str],
                          cds: str | None = None) -> tuple[list[dict], dict]:
        from src.substitution import aa_index, parse_mutation

        rec, enc_ms = self.encode_protein(seq, cds)
        parsed = [parse_mutation(seq, m) for m in mutations]
        sites = [p[1] for p in parsed]

        t0 = time.perf_counter()
        batch = self._batch_sites(rec, sites)
        batch["wt_idx"] = self.torch.tensor([aa_index(p[2]) for p in parsed], device=self.device)
        batch["mt_idx"] = self.torch.tensor([aa_index(p[3]) for p in parsed], device=self.device)
        with self.torch.no_grad():
            z = self.model.context(batch)
            ddg_t, lv_t = self.model.ddg(batch, z)
            three_t = self.torch.softmax(self.model.three_state(batch, z), dim=-1)
        head_ms = (time.perf_counter() - t0) * 1000.0

        classes = ("destabilizing", "neutral", "stabilizing")
        results = []
        for i, (_, site, wt_aa, mt_aa) in enumerate(parsed):
            ddg = float(ddg_t[i].item())
            sigma = float(self.torch.exp(0.5 * lv_t[i]).item()) * self.sigma_scale
            probs = three_t[i].tolist()
            verdict = classes[int(self.torch.argmax(three_t[i]).item())]
            results.append({
                "mutation": f"{wt_aa}{site + 1}{mt_aa}",
                "position": site + 1,
                "wt_aa": wt_aa,
                "mt_aa": mt_aa,
                "ddg": round(ddg, 4),
                "sigma": round(sigma, 4),
                "ci95_low": round(ddg - Z95 * sigma, 4),
                "ci95_high": round(ddg + Z95 * sigma, 4),
                "verdict": verdict,
                "p_destabilizing": round(probs[0], 4),
                "p_neutral": round(probs[1], 4),
                "p_stabilizing": round(probs[2], 4),
                "calibrated": self.calibrated,
            })
        timing = {
            "encode_ms": round(enc_ms, 1),
            "predict_ms": round(head_ms, 1),
            "muts_per_sec": round(len(results) / max(head_ms / 1000.0, 1e-6), 1),
        }
        return results, timing

    def scan_protein(self, seq: str, start: int = 1, end: int | None = None,
                     cds: str | None = None, batch_sites: int = 32) -> tuple[dict, dict]:
        from src.substitution import AA_ORDER, aa_index

        rec, enc_ms = self.encode_protein(seq, cds)
        start = max(1, int(start))
        end = min(len(seq), int(end) if end else len(seq))
        positions = list(range(start, end + 1))
        sites = [p - 1 for p in positions]

        t0 = time.perf_counter()
        cols: list[list[float]] = []
        for lo in range(0, len(sites), batch_sites):
            chunk = sites[lo:lo + batch_sites]
            batch = self._batch_sites(rec, chunk)
            batch["wt_idx"] = self.torch.tensor(
                [aa_index(seq[s]) for s in chunk], device=self.device
            )
            with self.torch.no_grad():
                z = self.model.context(batch)
                rows = self.model.scan(batch, z)
            for r in rows:
                cols.append([round(float(v), 4) for v in r[:len(AA_ORDER)]])
        scan_ms = (time.perf_counter() - t0) * 1000.0

        matrix = [[cols[c][r] for c in range(len(positions))] for r in range(len(AA_ORDER))]
        payload = {
            "length": len(seq),
            "positions": positions,
            "wt_residues": [seq[p - 1] for p in positions],
            "aa_order": list(AA_ORDER),
            "matrix": matrix,
        }
        total_muts = len(positions) * len(AA_ORDER)
        timing = {
            "encode_ms": round(enc_ms, 1),
            "scan_ms": round(scan_ms, 1),
            "total_mutations": total_muts,
            "muts_per_sec": round(total_muts / max(scan_ms / 1000.0, 1e-6), 1),
        }
        return payload, timing

    def predict_epistasis(self, seq: str, pairs: list[str],
                          cds: str | None = None) -> tuple[list[dict], dict]:
        from src.graphs import build_geometric_graph, pair_contact_features
        from src.substitution import aa_index, parse_multi_mutation

        rec, enc_ms = self.encode_protein(seq, cds)
        parsed_pairs = []
        for spec in pairs:
            _, muts = parse_multi_mutation(seq, spec)
            if len(muts) != 2:
                raise ValueError(f"epistasis expects double mutants (2 sites), got {spec!r}")
            parsed_pairs.append((spec, muts[0], muts[1]))

        t0 = time.perf_counter()
        # Whole-protein contact graph for pair contact features
        edge_geo, attr_geo = build_geometric_graph(
            len(seq), rec["contact_edges"], rec["contact_probs"],
            top_k=self.cfg.get("graph_top_k", 16),
            threshold=self.cfg.get("geo_threshold", 0.30),
        )
        uniq_sites = sorted({m[0] for _, m1, m2 in parsed_pairs for m in (m1, m2)})
        site_to_pos = {s: i for i, s in enumerate(uniq_sites)}
        batch = self._batch_sites(rec, uniq_sites)
        with self.torch.no_grad():
            z_all = self.model.context(batch)
        lp_full = rec.get("wt_logprobs")

        results = []
        for spec, (s1, w1, m1), (s2, w2, m2) in parsed_pairs:
            zi = z_all[site_to_pos[s1]:site_to_pos[s1] + 1]
            zj = z_all[site_to_pos[s2]:site_to_pos[s2] + 1]
            wt_i = self.torch.tensor([aa_index(w1)], device=self.device)
            mt_i = self.torch.tensor([aa_index(m1)], device=self.device)
            wt_j = self.torch.tensor([aa_index(w2)], device=self.device)
            mt_j = self.torch.tensor([aa_index(m2)], device=self.device)
            pg = pair_contact_features(edge_geo, attr_geo, s1, s2, len(seq)).unsqueeze(0).to(self.device)
            lp_i = lp_full[s1:s1 + 1].float().to(self.device) if lp_full is not None else None
            lp_j = lp_full[s2:s2 + 1].float().to(self.device) if lp_full is not None else None
            with self.torch.no_grad():
                ddg_ab, dddg, lv_ab = self.model.epistasis(
                    zi, zj, wt_i, mt_i, wt_j, mt_j, pair_geo=pg, lp_i=lp_i, lp_j=lp_j
                )
            ddg_total = float(ddg_ab.item())
            epi = float(dddg.item())
            sigma = float(self.torch.exp(0.5 * lv_ab).item()) * self.sigma_scale
            results.append({
                "pair": f"{w1}{s1 + 1}{m1}+{w2}{s2 + 1}{m2}",
                "ddg_total": round(ddg_total, 4),
                "ddg_additive": round(ddg_total - epi, 4),
                "epistasis_dddg": round(epi, 4),
                "sigma": round(sigma, 4),
                "contact_prob": round(float(pg[0, 0].item()), 4),
                "seq_separation": abs(s2 - s1),
            })
        pred_ms = (time.perf_counter() - t0) * 1000.0
        return results, {"encode_ms": round(enc_ms, 1), "predict_ms": round(pred_ms, 1)}


# ---------------------------------------------------------------------------
# Standalone interactive HTML heatmap exporter (zero server needed!)
# ---------------------------------------------------------------------------
def write_scan_html(scan: dict, title: str, out_path: str) -> None:
    """Write a self-contained browser-viewable interactive DMS heatmap HTML."""
    positions = scan["positions"]
    wt = scan["wt_residues"]
    aas = scan["aa_order"]
    mat = scan["matrix"]

    # Find top 10 stabilizing & destabilizing substitutions
    flat = []
    for r, aa in enumerate(aas):
        for c, pos in enumerate(positions):
            if aa == wt[c]:
                continue
            flat.append((mat[r][c], f"{wt[c]}{pos}{aa}"))
    flat.sort()
    top_destab = flat[:10]
    top_stab = list(reversed(flat[-10:]))

    def _cell_color(v: float) -> str:
        # Negative = destabilizing (red), 0 = white, Positive = stabilizing (blue/teal)
        clamped = max(-3.0, min(3.0, v)) / 3.0
        if clamped < 0:
            t = -clamped
            r, g, b = 235, int(242 * (1 - 0.72 * t)), int(242 * (1 - 0.75 * t))
        else:
            t = clamped
            r, g, b = int(242 * (1 - 0.72 * t)), int(242 * (1 - 0.28 * t)), 235
        return f"rgb({r},{g},{b})"

    header_cells = "".join(
        f"<th title='{wt[i]}{positions[i]}'>{wt[i]}<br><span>{positions[i]}</span></th>"
        for i in range(len(positions))
    )
    body_rows = []
    for r, aa in enumerate(aas):
        tds = [f"<td class='aa'>{aa}</td>"]
        for c, pos in enumerate(positions):
            v = mat[r][c]
            is_wt = aa == wt[c]
            cls = "wt" if is_wt else ""
            bg = "#1e293b" if is_wt else _cell_color(v)
            tip = f"{wt[c]}{pos}{aa}: {v:+.3f} kcal/mol"
            tds.append(f"<td class='{cls}' style='background:{bg}' title='{tip}'>"
                       f"{'•' if is_wt else f'{v:+.1f}'}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    stab_rows = "".join(f"<tr><td>{m}</td><td>{v:+.3f}</td></tr>" for v, m in top_stab)
    destab_rows = "".join(f"<tr><td>{m}</td><td>{v:+.3f}</td></tr>" for v, m in top_destab)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Holo-GeoN DMS Heatmap — {html.escape(title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  .sub {{ color: #94a3b8; font-size: 13px; margin-bottom: 18px; }}
  .wrap {{ overflow-x: auto; border: 1px solid #334155; border-radius: 10px; background: #0b1120; padding: 12px; }}
  table.hm {{ border-collapse: collapse; font-size: 11px; }}
  table.hm th, table.hm td {{ width: 28px; height: 22px; text-align: center; border: 1px solid #1e293b; padding: 2px; color: #0f172a; font-weight: 600; }}
  table.hm th {{ background: #1e293b; color: #cbd5e1; position: sticky; top: 0; }}
  table.hm th span {{ font-size: 9px; color: #94a3b8; font-weight: 400; }}
  table.hm td.aa {{ background: #1e293b; color: #f8fafc; position: sticky; left: 0; }}
  table.hm td.wt {{ color: #f8fafc; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 20px; max-width: 720px; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 14px; }}
  .card h2 {{ font-size: 14px; margin: 0 0 10px 0; }}
  .card table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .card td {{ padding: 4px 6px; border-bottom: 1px solid #334155; }}
</style></head><body>
<h1>Holo-GeoN Deep Mutational Scan — {html.escape(title)}</h1>
<div class="sub">Sign convention: ΔΔG = ΔG<sub>mut</sub> − ΔG<sub>wt</sub> (positive / blue = stabilizing, negative / red = destabilizing; hover any cell for exact kcal/mol).</div>
<div class="wrap"><table class="hm"><thead><tr><th>AA</th>{header_cells}</tr></thead>
<tbody>{"".join(body_rows)}</tbody></table></div>
<div class="grid">
  <div class="card"><h2>Top Stabilizing Mutations</h2><table>{stab_rows}</table></div>
  <div class="card"><h2>Top Destabilizing Mutations</h2><table>{destab_rows}</table></div>
</div></body></html>"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def cmd_doctor(args) -> int:
    print("=" * 68)
    print("Holo-GeoN System & Environment Check")
    print("=" * 68)
    print(f"Python       : {sys.version.split()[0]} ({sys.executable})")
    print(f"OS           : {os.name} / {sys.platform}")
    print(f"Project Root : {ROOT}")

    pkgs = ["torch", "torch_geometric", "transformers", "pandas", "scipy", "yaml", "RNA", "fastapi", "uvicorn"]
    for mod in pkgs:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            note = ""
            if mod == "transformers" and str(ver).startswith("5"):
                note = "  [note: pin transformers>=4.40,<5 on GPU training VMs]"
            print(f"  {mod:<16}: {ver}{note}")
        except Exception:
            hint = " (optional; needed for real CDS E_mech)" if mod == "RNA" else " (MISSING)"
            print(f"  {mod:<16}: NOT INSTALLED{hint}")

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU          : {name} (sm_{cap[0]}{cap[1]}, {vram:.1f} GB VRAM, CUDA {torch.version.cuda})")
        else:
            print("GPU          : None detected (CPU mode — fine for light tests; use Kaggle T4 or Colab G4 for training & fast DMS)")
    except Exception as exc:
        print(f"GPU          : could not probe ({exc})")

    weights = Path(args.weights) if args.weights else DEFAULT_WEIGHTS
    if weights.is_file():
        print(f"Weights      : FOUND ({weights})")
    else:
        print(f"Weights      : NOT FOUND at {weights}")
        print("               Pass --weights <path/to/model_best.pt> or train with `python run.py train`.")
    print("=" * 68)
    return 0


def cmd_predict(args) -> int:
    header, seq = resolve_sequence(args)
    cds = resolve_cds(args.cds)
    muts: list[str] = []
    if args.mut:
        muts.append(args.mut)
    if args.muts:
        muts.extend([m.strip() for m in args.muts.split(",") if m.strip()])
    if not muts:
        raise SystemExit("Provide --mut L8P or --muts L8P,V5A")

    eng = CLIEngine(Path(args.weights), device_req=args.device)
    rows, timing = eng.predict_mutations(seq, muts, cds=cds)

    print(f"\nProtein: {header} (L={len(seq)}, CDS={'yes' if cds else 'no'})")
    print("Sign   : positive = stabilizing, negative = destabilizing (kcal/mol)")
    print(f"{'Mutation':<12}{'ddG':>9}{'95% CI':>21}{'Verdict':>16}{'P(dest/neu/stab)':>22}")
    print("-" * 80)
    for r in rows:
        ci = f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]"
        p3 = f"{r['p_destabilizing']:.2f}/{r['p_neutral']:.2f}/{r['p_stabilizing']:.2f}"
        print(f"{r['mutation']:<12}{r['ddg']:>+9.3f}{ci:>21}{r['verdict']:>16}{p3:>22}")
    print(f"\nTiming : encode={timing['encode_ms']} ms | predict={timing['predict_ms']} ms "
          f"({timing['muts_per_sec']} muts/s)")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".json":
            out_p.write_text(json.dumps({"protein": header, "timing": timing, "predictions": rows}, indent=2),
                             encoding="utf-8")
        else:
            with open(out_p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"Saved  : {out_p}")
    return 0


def cmd_scan(args) -> int:
    header, seq = resolve_sequence(args)
    cds = resolve_cds(args.cds)
    eng = CLIEngine(Path(args.weights), device_req=args.device)
    scan, timing = eng.scan_protein(seq, start=args.start, end=args.end, cds=cds)
    print(f"\nDeep Mutational Scan complete: {header} ({len(scan['positions'])} positions, "
          f"{timing['total_mutations']:,} variants)")
    print(f"Timing : encode={timing['encode_ms']} ms | scan={timing['scan_ms']} ms "
          f"({timing['muts_per_sec']:,.1f} muts/s)")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["mutation", "position", "wt_aa", "mt_aa", "ddg_kcal_mol"])
            for c, pos in enumerate(scan["positions"]):
                wt_aa = scan["wt_residues"][c]
                for r, aa in enumerate(scan["aa_order"]):
                    w.writerow([f"{wt_aa}{pos}{aa}", pos, wt_aa, aa, scan["matrix"][r][c]])
        print(f"CSV    : {out_p}")

    if args.html:
        write_scan_html(scan, header, args.html)
        print(f"HTML   : {args.html}  (open directly in Chrome/Edge — no server needed!)")
    return 0


def cmd_epistasis(args) -> int:
    header, seq = resolve_sequence(args)
    cds = resolve_cds(args.cds)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    eng = CLIEngine(Path(args.weights), device_req=args.device)
    rows, timing = eng.predict_epistasis(seq, pairs, cds=cds)

    print(f"\nDouble-Mutant Epistasis: {header} (L={len(seq)}, structure-free E_geo gate)")
    print(f"{'Pair':<18}{'ddG_total':>11}{'Additive':>11}{'Epistasis':>11}{'p(contact)':>12}{'Sep':>6}")
    print("-" * 70)
    for r in rows:
        print(f"{r['pair']:<18}{r['ddg_total']:>+11.3f}{r['ddg_additive']:>+11.3f}"
              f"{r['epistasis_dddg']:>+11.3f}{r['contact_prob']:>12.3f}{r['seq_separation']:>6d}")
    print(f"\nTiming : encode={timing['encode_ms']} ms | predict={timing['predict_ms']} ms")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved  : {args.out}")
    return 0


def cmd_batch(args) -> int:
    import pandas as pd

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input table not found: {in_path}")
    sep = "\t" if in_path.suffix.lower() == ".tsv" else ","
    df = pd.read_parquet(in_path) if in_path.suffix.lower() == ".parquet" else pd.read_csv(in_path, sep=sep)

    default_seq = None
    if args.fasta:
        _, default_seq = read_fasta_first(args.fasta)

    seq_col = args.seq_col if args.seq_col in df.columns else None
    mut_col = args.mut_col
    if mut_col not in df.columns:
        raise SystemExit(f"Column {mut_col!r} not found in {in_path}; available: {list(df.columns)}")
    if seq_col is None and default_seq is None:
        raise SystemExit(f"No {args.seq_col!r} column found and no --fasta given.")

    eng = CLIEngine(Path(args.weights), device_req=args.device)
    t0 = time.perf_counter()
    out_rows = []
    # Group by sequence so each unique parent encodes ONCE
    seqs = df[seq_col].astype(str).str.upper() if seq_col else pd.Series([default_seq] * len(df))
    df = df.copy()
    df["_seq"] = seqs
    for seq, grp in df.groupby("_seq", sort=False):
        muts = grp[mut_col].astype(str).tolist()
        preds, _ = eng.predict_mutations(seq, muts)
        for orig_idx, pred in zip(grp.index, preds):
            rec = dict(df.loc[orig_idx].drop("_seq"))
            rec.update(pred)
            out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.suffix.lower() == ".parquet":
        out_df.to_parquet(out_p, index=False)
    else:
        out_df.to_csv(out_p, index=False)
    elapsed = time.perf_counter() - t0
    print(f"[holo] Scored {len(out_df):,} mutations across {df['_seq'].nunique()} protein(s) "
          f"in {elapsed:.2f}s ({len(out_df) / max(elapsed, 1e-6):.1f} muts/s) -> {out_p}")
    return 0


def cmd_gui(args) -> int:
    cmd = [sys.executable, str(ROOT / "runapp.py"), "--use-current-env",
           "--port", str(args.port), "--host", args.host]
    if args.weights:
        cmd += ["--weights", args.weights]
    if args.no_browser:
        cmd.append("--no-browser")
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_interactive() -> int:
    print("\n" + "=" * 60)
    print("  Holo-GeoN Interactive Mode (No Linux flags needed!)")
    print("=" * 60)
    print("  [1] System & GPU Doctor")
    print("  [2] Predict single / multiple mutation ddG")
    print("  [3] Full 20xL Deep Mutational Scan + HTML Heatmap")
    print("  [4] Double-Mutant Epistasis Prediction")
    print("  [5] Launch Web GUI")
    print("  [q] Quit")
    choice = input("\nChoose an option [1-5, q]: ").strip().lower()
    if choice in ("q", "quit", "exit", ""):
        return 0
    if choice == "1":
        return cmd_doctor(argparse.Namespace(weights=str(DEFAULT_WEIGHTS)))
    if choice == "5":
        w = input(f"Weights path [{DEFAULT_WEIGHTS}]: ").strip() or str(DEFAULT_WEIGHTS)
        return cmd_gui(argparse.Namespace(weights=w, port=8000, host="127.0.0.1", no_browser=False))

    w = input(f"Weights path [{DEFAULT_WEIGHTS}]: ").strip() or str(DEFAULT_WEIGHTS)
    src_mode = input("Protein input — [f]asta file, [u]niProt ID, or [s]equence paste? [u]: ").strip().lower() or "u"
    seq = fasta = uniprot = None
    if src_mode.startswith("f"):
        fasta = input("Path to FASTA file: ").strip()
    elif src_mode.startswith("s"):
        seq = input("Paste protein sequence: ").strip()
    else:
        uniprot = input("UniProt accession (e.g. P0CG48): ").strip()
    cds = input("Optional CDS file/DNA (press Enter to skip): ").strip() or None

    if choice == "2":
        muts = input("Mutation(s), comma-separated (e.g. L8P,V5A): ").strip()
        out = input("Optional output CSV/JSON path (Enter to skip): ").strip() or None
        return cmd_predict(argparse.Namespace(
            weights=w, device=None, seq=seq, fasta=fasta, uniprot=uniprot,
            cds=cds, mut=None, muts=muts, out=out,
        ))
    if choice == "3":
        csv_out = input("Output CSV path [scan_results.csv]: ").strip() or "scan_results.csv"
        html_out = input("Interactive HTML heatmap path [scan_heatmap.html]: ").strip() or "scan_heatmap.html"
        return cmd_scan(argparse.Namespace(
            weights=w, device=None, seq=seq, fasta=fasta, uniprot=uniprot,
            cds=cds, start=1, end=None, out=csv_out, html=html_out,
        ))
    if choice == "4":
        pairs = input("Double-mutant pair(s), comma-separated (e.g. L8P+V5A): ").strip()
        out = input("Optional output CSV path (Enter to skip): ").strip() or None
        return cmd_epistasis(argparse.Namespace(
            weights=w, device=None, seq=seq, fasta=fasta, uniprot=uniprot,
            cds=cds, pairs=pairs, out=out,
        ))
    print("Unknown choice.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hologe",
        description="Holo-GeoN — Cross-platform protein stability (ddG) & epistasis CLI.",
    )
    sub = ap.add_subparsers(dest="cmd")

    def _add_common(p):
        p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained .pt checkpoint")
        p.add_argument("--device", default=None, help="cuda or cpu (auto-detected by default)")
        p.add_argument("--seq", default=None, help="Protein amino-acid sequence")
        p.add_argument("--fasta", default=None, help="Path to FASTA file")
        p.add_argument("--uniprot", default=None, help="UniProt accession (auto-fetched)")
        p.add_argument("--cds", default=None, help="Optional real coding DNA sequence or FASTA")

    p_doc = sub.add_parser("doctor", help="Check Python, GPU, dependencies, and weights")
    p_doc.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_doc.set_defaults(func=cmd_doctor)

    p_pred = sub.add_parser("predict", help="Predict ddG, 95% CI, and 3-state verdict")
    _add_common(p_pred)
    p_pred.add_argument("--mut", default=None, help="Single mutation code (e.g. L8P)")
    p_pred.add_argument("--muts", default=None, help="Comma-separated mutations (e.g. L8P,V5A)")
    p_pred.add_argument("--out", default=None, help="Optional CSV or JSON output path")
    p_pred.set_defaults(func=cmd_predict)

    p_scan = sub.add_parser("scan", help="Full 20xL deep mutational scan + HTML heatmap")
    _add_common(p_scan)
    p_scan.add_argument("--start", type=int, default=1, help="1-based start residue")
    p_scan.add_argument("--end", type=int, default=None, help="1-based end residue")
    p_scan.add_argument("--out", default=None, help="Output CSV path")
    p_scan.add_argument("--html", default=None, help="Standalone interactive HTML heatmap path")
    p_scan.set_defaults(func=cmd_scan)

    p_epi = sub.add_parser("epistasis", help="Double-mutant ddG and contact-gated epistasis")
    _add_common(p_epi)
    p_epi.add_argument("--pairs", required=True, help="Comma-separated double mutants (e.g. L8P+V5A,I13A+L15V)")
    p_epi.add_argument("--out", default=None, help="Optional CSV output path")
    p_epi.set_defaults(func=cmd_epistasis)

    p_bat = sub.add_parser("batch", help="Score a CSV/TSV/Parquet table of mutations")
    p_bat.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_bat.add_argument("--device", default=None)
    p_bat.add_argument("--input", required=True, help="Input CSV, TSV, or Parquet file")
    p_bat.add_argument("--output", required=True, help="Output CSV or Parquet file")
    p_bat.add_argument("--fasta", default=None, help="Optional single FASTA if input has no sequence column")
    p_bat.add_argument("--seq-col", default="sequence", help="Sequence column name (default: sequence)")
    p_bat.add_argument("--mut-col", default="mutation", help="Mutation column name (default: mutation)")
    p_bat.set_defaults(func=cmd_batch)

    p_gui = sub.add_parser("gui", help="Launch the Holo-GeoN web interface")
    p_gui.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p_gui.add_argument("--port", type=int, default=8000)
    p_gui.add_argument("--host", default="127.0.0.1")
    p_gui.add_argument("--no-browser", action="store_true")
    p_gui.set_defaults(func=cmd_gui)
    return ap


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    ap = build_parser()
    if argv is None and len(sys.argv) == 1:
        return cmd_interactive()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        return cmd_interactive()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
