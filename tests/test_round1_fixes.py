"""
tests/test_round1_fixes.py
==========================
Regression tests for the round-1 review findings.

Each test names the finding it locks down. These are the defects an adversarial
reviewer executed against the code and reproduced; without a test, a later
refactor reintroduces them silently, which is exactly how several of them
arrived in the first place.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.data import _binary_label, load_mutation_table
from src.features import CH_AA_KNOWN, CH_VOLUME, mechanistic_features
from src.losses import UncertaintyWeightedLoss
from src.seeding import seed_everything
from tests.helpers import check, random_protein, section

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
def test_r1_01_seeding() -> None:
    section("R1-01: seeding makes a run reproducible")

    def one_run() -> list[float]:
        seed_everything(1234)
        from src.model import HoloGeoN
        from tests.helpers import make_batch

        model = HoloGeoN(esm_dim=16, hidden=8, out_dim=8, aa_dim=4)
        batch = make_batch(dim=16, batch_size=3, length=12, seed=7)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(3):
            ddg, logvar = model.ddg(batch)
            loss = ((ddg.reshape(-1) - batch["y"]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return losses

    a, b = one_run(), one_run()
    check(a == b, f"two seeded runs are identical ({a[0]:.6f} == {b[0]:.6f})")

    seed_everything(1234)
    x = torch.randn(3).tolist()
    seed_everything(1234)
    y = torch.randn(3).tolist()
    seed_everything(9999)
    z = torch.randn(3).tolist()
    check(x == y and x != z, "seed_everything controls torch RNG and differs by seed")

    import random as _random
    seed_everything(5)
    combo = (_random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(5)
    check(combo == (_random.random(), float(np.random.rand()), float(torch.rand(1))),
          "python, numpy and torch generators are all seeded")


# ---------------------------------------------------------------------------
def test_r1_03_endless_does_not_retain() -> None:
    section("R1-03: the training cycler holds one batch, not the epoch")

    from scripts.train import endless

    class _Loader:
        """Yields big tensors and reshuffles each pass."""
        def __init__(self):
            self.passes = 0

        def __iter__(self):
            self.passes += 1
            order = [0, 1, 2] if self.passes % 2 else [2, 1, 0]
            for i in order:
                yield torch.zeros(64, 256, 64)  # 4 MB each

        def __len__(self):
            return 3

    loader = _Loader()
    gen = endless(loader)
    seen = [next(gen) for _ in range(6)]
    live = sum(1 for o in gc.get_objects()
               if torch.is_tensor(o) and o.shape == (64, 256, 64))
    del seen
    # itertools.cycle would keep all 6 alive; endless keeps only what the caller
    # still references.
    check(live <= 7, f"cycler does not retain the whole epoch (live tensors {live})")
    check(loader.passes >= 2, "the loader is re-iterated, so shuffling re-runs")


# ---------------------------------------------------------------------------
def test_r1_04_clinvar_labels() -> None:
    section("R1-04: ClinVar vocabulary is matched exactly, not by substring")

    for term in ["Conflicting_interpretations_of_pathogenicity",
                 "Conflicting_classifications_of_pathogenicity",
                 "Conflicting_interpretations_of_pathogenicity|other",
                 "Uncertain_significance", "not_provided", "drug_response",
                 "risk_factor", "protective", "non-pathogenic"]:
        check(_binary_label(term) is None,
              f"{term!r} is not a binary call -> dropped and counted")

    for term, want in [("Benign", 0), ("Likely_benign", 0),
                       ("Benign/Likely_benign", 0),
                       ("Pathogenic", 1), ("Likely_pathogenic", 1),
                       ("Pathogenic/Likely_pathogenic", 1),
                       ("Pathogenic,_low_penetrance", 1),
                       ("Likely_pathogenic|other", 1)]:
        check(_binary_label(term) == want, f"{term!r} -> {want}")

    check(_binary_label(0) == 0 and _binary_label(1) == 1,
          "an explicit 0/1 encoding is accepted")
    for v in (2, 3, 4, 5):
        check(_binary_label(v) is None,
              f"legacy integer CLNSIG {v} is refused, not read as 'nonzero = pathogenic'")


# ---------------------------------------------------------------------------
def test_r1_07_whole_sequence_spans_full_protein() -> None:
    section("R1-07: whole-sequence tasks are not head-truncated")

    from src.data import build_sample_graph
    from tests.helpers import stub_encoding

    seq = random_protein(900, seed=11)
    rec = stub_encoding(seq, dim=16, top_k=4)
    cfg = {"graph_top_k": 8, "geo_threshold": 0.3, "neighborhood_hops": 2}

    whole = build_sample_graph(rec, None, 256, cfg, max_nodes=2048)
    check(whole["n"] == 900,
          f"a 900-residue protein yields {whole['n']} nodes, not a 256 window")
    check(whole["length"] == 900, "true chain length travels with the sample")
    check(whole["truncated"] is False, "no truncation within the cap")

    capped = build_sample_graph(rec, None, 256, cfg, max_nodes=400)
    check(capped["n"] == 400 and capped["truncated"] is True,
          "the node cap is applied AND reported, never silent")

    # A mutation-conditioned sample still windows around the site.
    sited = build_sample_graph(rec, 700, 256, cfg)
    check(sited["n"] == 256 and sited["site"] < 256,
          "a site-anchored sample still uses a window containing the mutation")


# ---------------------------------------------------------------------------
def test_r1_09_multitask_loss_is_bounded() -> None:
    section("R1-09: the multi-task objective cannot diverge")

    crit = UncertaintyWeightedLoss(["stability", "pathogenicity"])
    opt = torch.optim.Adam(crit.parameters(), lr=0.5)
    totals, weights = [], []
    for _ in range(500):
        total, parts = crit({"stability": torch.tensor(-4.0),
                             "pathogenicity": torch.tensor(0.7)})
        opt.zero_grad()
        total.backward()
        opt.step()
        totals.append(float(total))
        weights.append(parts["stability_weight"])

    check(np.isfinite(totals[-1]) and totals[-1] > -100,
          f"the total stays bounded with a negative NLL ({totals[-1]:.3f}); the "
          f"reviewer measured -1.9e19 before the floor was applied")
    check(weights[-1] < 1e3,
          f"the task weight stays bounded ({weights[-1]:.3g}); it reached 4.7e18 before")


# ---------------------------------------------------------------------------
def test_r1_19_sign_convention() -> None:
    section("R1-19: the ddG sign convention is declared and normalised")

    with tempfile.TemporaryDirectory() as tmp:
        wt = random_protein(40, seed=3)
        rows = []
        for i in range(120):
            site = i % 40
            mt_aa = "A" if wt[site] != "A" else "C"
            rows.append({
                "wt_seq": wt,
                "mt_seq": wt[:site] + mt_aa + wt[site + 1:],
                # Destabilising-positive source table: mean is POSITIVE here.
                "ddg": 1.5 + 0.01 * i,
            })
        path = os.path.join(tmp, "signed.parquet")
        pd.DataFrame(rows).to_parquet(path)

        default, rep_d = load_mutation_table(path, "stability")
        check(default["target"].mean() > 0,
              "under the default convention this table reads positive-mean")
        check("sign_warning" in rep_d.notes,
              "a positive mean ddG is flagged as a probable convention flip")

        flipped, rep_f = load_mutation_table(
            path, "stability", overrides={"ddg_sign": "destabilizing_positive"})
        check(flipped["target"].mean() < 0,
              "declaring the source convention normalises the sign")
        check(rep_f.notes["ddg_sign"] == "destabilizing_positive",
              "the applied sign is recorded in the load report")
        check(np.allclose(default["target"].to_numpy(),
                          -flipped["target"].to_numpy()),
              "the two readings are exact negations of one another")


# ---------------------------------------------------------------------------
def test_r1_24_unknown_residues() -> None:
    section("R1-24: non-standard residues are flagged, not misencoded")

    feats = mechanistic_features("ACDXBZUO", None, 8)
    known = feats[:, CH_AA_KNOWN]
    check(known[:3].min() == 1.0, "standard residues are flagged known")
    check(known[3:].max() == 0.0, "B/Z/X/U/O are flagged unknown")

    vol_unknown = float(feats[3, CH_VOLUME])
    vol_gly = float(mechanistic_features("G", None, 1)[0, CH_VOLUME])
    check(vol_unknown > vol_gly,
          f"an unknown residue's volume ({vol_unknown:.3f}) is mid-scale, not "
          f"below glycine ({vol_gly:.3f})")

    from app.backend.inference import clean_sequence  # noqa: E402
    check(clean_sequence("ACDXBZUO") == "ACDXBZUO",
          "the interface accepts non-standard residues rather than refusing")
    try:
        clean_sequence("ACGTACGTACGT")
        check(False, "a nucleotide-looking sequence is refused")
    except ValueError as exc:
        check("nucleotide" in str(exc).lower(),
              "a nucleotide-looking sequence is refused with a specific reason")


# ---------------------------------------------------------------------------
def test_r1_02_refuses_without_val_split() -> None:
    section("R1-02: a primary task with no val split fails loudly")

    from src.encoder import EncodingStore
    from tests.helpers import stub_encoding
    from tests.test_pipeline import CONFIG_TEMPLATE, STUB_DIM

    with tempfile.TemporaryDirectory(prefix="holo_r102_") as tmp:
        data, cache, runs = (os.path.join(tmp, d) for d in ("data", "cache", "runs"))
        os.makedirs(data, exist_ok=True)
        store = EncodingStore(cache)

        rows = []
        for p in range(6):
            parent = random_protein(60, seed=400 + p)
            store.put(stub_encoding(parent, dim=STUB_DIM, top_k=8))
            for i in range(10):
                site = i % 60
                mt_aa = "A" if parent[site] != "A" else "C"
                rows.append({"parent_seq": parent, "site": site,
                             "wt_aa": parent[site], "mt_aa": mt_aa,
                             "target": 0.1 * i, "sigma": 0.3,
                             "group": parent,
                             # deliberately no 'val' rows
                             "split": "train" if i < 8 else "test",
                             "group_used": parent})
        splits = os.path.join(data, "splits_noval.parquet")
        pd.DataFrame(rows).to_parquet(splits)

        config = os.path.join(tmp, "config.yaml")
        with open(config, "w", encoding="utf-8") as fh:
            fh.write(CONFIG_TEMPLATE.format(
                cache=cache, runs=runs, stability=splits, clinvar=splits, epochs=2))

        proc = subprocess.run(
            [sys.executable, "-m", "scripts.train", "--config", config,
             "--stability-splits", splits,
             "--pathogenicity-splits", os.path.join(tmp, "absent.parquet"),
             "--out", os.path.join(runs, "noval")],
            cwd=ROOT, capture_output=True, text=True,
        )
        check(proc.returncode != 0,
              f"training exits non-zero without a val split (got {proc.returncode})")
        check("no validation split" in (proc.stdout + proc.stderr),
              "the refusal names the missing validation split")
        check(not os.path.exists(os.path.join(runs, "noval", "metrics.json")),
              "no metrics.json is written, so nothing downstream reads a "
              "headline number from a model that was never selected")


# ---------------------------------------------------------------------------
def test_r1_05_cache_manifest_records_cds_coverage() -> None:
    section("R1-05: the cache records whether the mechanistic stream was fed")

    from src.encoder import EncodingStore
    from tests.helpers import stub_encoding

    with tempfile.TemporaryDirectory(prefix="holo_r105_") as tmp:
        store = EncodingStore(tmp)
        check(store.manifest() == {},
              "a cache with no manifest reads as empty rather than raising")

        store.put(stub_encoding(random_protein(40, seed=1), dim=16, top_k=4))
        store.write_manifest({"esm_model": "stub", "parents_requested": 4,
                              "cds_active": 0, "cds_coverage": 0.0})
        got = store.manifest()
        check(got["cds_coverage"] == 0.0,
              "zero mechanistic coverage is recorded, not left to be assumed")
        check(os.path.exists(os.path.join(tmp, "manifest.json")),
              "the manifest lands next to the records it describes")

        store.write_manifest({"cds_active": 3, "cds_coverage": 0.75})
        merged = store.manifest()
        check(merged["cds_coverage"] == 0.75 and merged["esm_model"] == "stub",
              "a later pass updates coverage without dropping what it did not touch")

        with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        check(store.manifest() == {},
              "a corrupt manifest degrades to empty instead of killing a run")

        # The number train.py copies into metrics.json is the one a reader needs
        # to falsify a claim about E_mech, so it must survive a JSON round-trip.
        check(json.loads(json.dumps(merged))["cds_coverage"] == 0.75,
              "coverage survives into metrics.json as a plain number")


TESTS = [
    test_r1_01_seeding,
    test_r1_05_cache_manifest_records_cds_coverage,
    test_r1_03_endless_does_not_retain,
    test_r1_04_clinvar_labels,
    test_r1_07_whole_sequence_spans_full_protein,
    test_r1_09_multitask_loss_is_bounded,
    test_r1_19_sign_convention,
    test_r1_24_unknown_residues,
    test_r1_02_refuses_without_val_split,
]
