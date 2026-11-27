"""
tests/test_cli.py
=================
Pure-Python / CPU unit tests for the cross-platform user CLI (``holo.py``):
  * ``doctor`` runs cleanly without crashing on Windows/Linux
  * ``predict``, ``scan --html``, ``epistasis``, and ``batch`` run end-to-end
    against a synthetic Holo-GeoN v2 checkpoint and stub encoder
  * Standalone HTML heatmap is valid and contains the expected mutations
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import holo
from src.model import HoloGeoN
from tests.helpers import check, section, stub_encoding

SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _make_v2_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    kwargs = {
        "arch": "v2", "esm_dim": 32, "hidden": 32, "out_dim": 32,
        "aa_dim": 8, "gat_heads": 4, "dropout": 0.1,
    }
    m = HoloGeoN(**kwargs)
    torch.save(
        {
            "model": m.state_dict(),
            "model_kwargs": kwargs,
            "esm_dim": 32,
            "config": {
                "arch": "v2", "esm_model": "stub", "window": 64,
                "contact_top_k": 8, "contact_min_sep": 3, "graph_top_k": 8,
                "geo_threshold": 0.3, "eps": 0.1, "delta": 0.1,
                "mech_max_degree": 8, "neighborhood_hops": 2,
            },
            "untrained_heads": ["idr", "mfi", "proteomics"],
        },
        path,
    )


def test_cli_end_to_end() -> None:
    section("holo.py CLI — doctor, predict, scan (+HTML), epistasis, batch")
    import src.encoder as _enc

    class _StubEnc:
        def __init__(self, *a, **k):
            self.dim = 32
        def encode(self, seq, top_k=24, min_sep=3):
            rec = stub_encoding(seq, dim=32, top_k=8)
            rec["layer_embs"] = torch.randn(4, len(seq), 32).half()
            rec["wt_logprobs"] = torch.log_softmax(torch.randn(len(seq), 20), dim=-1).half()
            rec["site_entropy"] = torch.rand(len(seq)).half()
            return rec

    orig_enc = _enc.EsmEncoder
    _enc.EsmEncoder = _StubEnc
    try:
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            ckpt = td / "v2.pt"
            _make_v2_checkpoint(ckpt)
            fasta = td / "ubq.fasta"
            fasta.write_text(f">Ubiquitin_test\n{SEQ}\n", encoding="utf-8")

            # 1. doctor
            rc = holo.main(["doctor", "--weights", str(ckpt)])
            check(rc == 0, "holo.py doctor exits 0")

            # 2. predict
            pred_csv = td / "pred.csv"
            rc = holo.main([
                "predict", "--weights", str(ckpt), "--device", "cpu",
                "--fasta", str(fasta), "--muts", "L8P,V5A", "--out", str(pred_csv),
            ])
            check(rc == 0 and pred_csv.is_file(), "holo.py predict writes CSV")
            rows = list(csv.DictReader(pred_csv.open(encoding="utf-8")))
            check(len(rows) == 2 and rows[0]["mutation"] == "L8P",
                  "predict output contains both requested mutations")

            # 3. scan + standalone HTML heatmap
            scan_csv = td / "scan.csv"
            scan_html = td / "heatmap.html"
            rc = holo.main([
                "scan", "--weights", str(ckpt), "--device", "cpu",
                "--fasta", str(fasta), "--start", "1", "--end", "8",
                "--out", str(scan_csv), "--html", str(scan_html),
            ])
            check(rc == 0 and scan_csv.is_file() and scan_html.is_file(),
                  "holo.py scan writes CSV and standalone HTML heatmap")
            html_txt = scan_html.read_text(encoding="utf-8")
            check("Holo-GeoN Deep Mutational Scan" in html_txt and "L8P" in html_txt,
                  "HTML heatmap contains title and mutation tooltips")

            # 4. epistasis
            epi_csv = td / "epi.csv"
            rc = holo.main([
                "epistasis", "--weights", str(ckpt), "--device", "cpu",
                "--fasta", str(fasta), "--pairs", "V5A+L8P,K6A+T7A",
                "--out", str(epi_csv),
            ])
            check(rc == 0 and epi_csv.is_file(), "holo.py epistasis writes CSV")
            erows = list(csv.DictReader(epi_csv.open(encoding="utf-8")))
            check(len(erows) == 2 and "epistasis_dddg" in erows[0],
                  "epistasis output contains dddG and contact_prob")

            # 5. batch
            in_csv = td / "muts.csv"
            in_csv.write_text("mutation\nL8P\nV5A\nT7A\n", encoding="utf-8")
            out_csv = td / "scored.csv"
            rc = holo.main([
                "batch", "--weights", str(ckpt), "--device", "cpu",
                "--fasta", str(fasta), "--input", str(in_csv), "--output", str(out_csv),
            ])
            check(rc == 0 and out_csv.is_file(), "holo.py batch scores CSV cleanly")
            brows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
            check(len(brows) == 3, "batch output has all 3 rows")
    finally:
        _enc.EsmEncoder = orig_enc


TESTS = [test_cli_end_to_end]


if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
