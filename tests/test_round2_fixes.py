"""
tests/test_round2_fixes.py
==========================
One check per round-2 finding, written so it would FAIL against the code as it
stood at `bbae5e7`.

The point of these is not coverage. Round 2's most important result was not a
new defect but the discovery that three round-1 findings were reported fixed and
were not — a scan that still ran one encoder pass per position, a clamp that was
a provable no-op, and a warning gated so it could never fire. A claim in a commit
message is not evidence. These are.

Where a finding is about a NUMBER rather than a behaviour (memory bounds, false
positive rates), the check asserts the property that made the number wrong —
that a batch cannot exceed its node budget, that a null comparison is not
flagged — rather than the number itself, which depends on hardware this
environment does not have.
"""
from __future__ import annotations

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from tests.helpers import check, random_protein, section, stub_encoding

AA = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
def test_r2_01_cds_coverage_measured_from_the_cache() -> None:
    section("R2-01: cds_coverage describes the cache, not the input table")
    from src.encoder import EncodingStore
    from src.features import CH_VALID, N_MECH_CHANNELS, mechanistic_features

    tmp = tempfile.mkdtemp()
    seqs = [random_protein(30, seed=i) for i in range(6)]

    # A cache built before any CDS was attached: the ordinary path.
    store = EncodingStore(os.path.join(tmp, "empty"))
    for s in seqs:
        store.put({"seq": s, "emb": torch.zeros(len(s), 8).half(),
                   "contact_edges": torch.zeros((2, 0), dtype=torch.int32),
                   "contact_probs": torch.zeros(0).half(),
                   "mech": mechanistic_features(s, None, len(s)).half(),
                   "has_cds": False, "cds": None, "esm_model": "stub", "dim": 8})

    measured = store.mech_coverage(seqs)
    # The old arithmetic (n_with_cds / len(parents)) read 1.0 here, because the
    # TABLE carried a CDS for all six even though the CACHE carried none.
    check(measured["cds_coverage"] == 0.0,
          f"a mech-free cache reports cds_coverage 0.0, not 1.0 "
          f"(got {measured['cds_coverage']})")
    check(measured["cds_active"] == 0,
          "no cached record counts as mechanistically active")
    check("cached records" in measured["cds_coverage_measured_on"],
          "the manifest says what the coverage was measured on")

    # And a cache that IS fed must read 1.0, or the check is vacuous.
    fed = EncodingStore(os.path.join(tmp, "fed"))
    for s in seqs:
        m = torch.zeros(len(s), N_MECH_CHANNELS)
        m[:, CH_VALID] = 1.0
        fed.put({"seq": s, "emb": torch.zeros(len(s), 8).half(),
                 "contact_edges": torch.zeros((2, 0), dtype=torch.int32),
                 "contact_probs": torch.zeros(0).half(), "mech": m.half(),
                 "has_cds": True, "cds": "ATG", "esm_model": "stub", "dim": 8})
    check(fed.mech_coverage(seqs)["cds_coverage"] == 1.0,
          "a fed cache reports cds_coverage 1.0, so the check is not vacuous")

    # A cache with no manifest must be distinguishable from one reporting zero.
    check(store.manifest() == {},
          "a cache with no manifest.json returns {} - the loud case in train.py")


# ---------------------------------------------------------------------------
def test_r2_02_node_budget_caps_whole_sequence_batches() -> None:
    section("R2-02: a whole-sequence batch cannot exceed its node budget")
    from src.data import NodeBudgetBatchSampler

    rng = np.random.default_rng(0)
    lengths = np.clip(rng.lognormal(5.6, 0.8, 600).astype(int), 30, 2048).tolist()
    budget, max_batch = 64 * 256, 64

    sampler = NodeBudgetBatchSampler(lengths, budget, max_batch,
                                     shuffle=True, seed=42)
    sampler.set_epoch(1)
    batches = list(sampler)
    costs = [len(b) * max(lengths[i] for i in b) for b in batches]

    check(max(costs) <= budget,
          f"no batch exceeds {budget:,} padded nodes (worst {max(costs):,})")
    check(max(len(b) for b in batches) <= max_batch,
          "no batch exceeds max_batch_size either")
    check(sorted(i for b in batches for i in b) == list(range(len(lengths))),
          "every sample appears exactly once, none duplicated or dropped")

    # The defect: a fixed batch_size of 64 over 2048-node graphs is 8x the ddG
    # batch the memory budget is computed for.
    naive = 64 * 2048
    check(naive > budget and max(costs) <= budget,
          f"a fixed batch_size would have cost {naive:,} nodes ({naive/budget:.0f}x "
          f"the ddG batch); the sampler holds it to {max(costs):,}")

    # Reproducible from (seed, epoch), and epochs actually differ.
    again = NodeBudgetBatchSampler(lengths, budget, max_batch, shuffle=True, seed=42)
    again.set_epoch(1)
    check(list(again) == batches, "batch order is reproducible from (seed, epoch)")
    sampler.set_epoch(2)
    check(list(sampler) != batches, "a different epoch gives a different order")


# ---------------------------------------------------------------------------
def test_r2_03_no_frontend_heuristic() -> None:
    section("R2-03: the interface presents no hardcoded biophysical classifier")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(root, "app", "frontend", "src")

    blob = ""
    for dirpath, _, names in os.walk(src_dir):
        for name in names:
            if name.endswith((".tsx", ".ts")):
                with open(os.path.join(dirpath, name), encoding="utf-8") as fh:
                    blob += fh.read()

    check("Globular" not in blob,
          "no 'Globular / compact' chip: the mu / L^0.33 < 2.5 classifier is gone")
    check("Math.pow(result.length" not in blob,
          "nothing derives a category from the predicted mu and the length")
    check("isStabilizing" not in blob,
          "no sign test re-derives a direction the backend deliberately dropped")
    check("result.verdict" in blob and "verdict_source" in blob,
          "the direction shown comes from the three-state head, with provenance")

    # dist/ is gitignored and has never been committed, contrary to the report;
    # if a build is present it must agree with the source it was built from.
    dist = os.path.join(root, "app", "frontend", "dist")
    if os.path.isdir(dist):
        stale = []
        for dirpath, _, names in os.walk(dist):
            for name in names:
                if name.endswith(".js"):
                    with open(os.path.join(dirpath, name), encoding="utf-8",
                              errors="ignore") as fh:
                        if "Globular" in fh.read():
                            stale.append(name)
        check(not stale, f"no built bundle still ships the classifier {stale}")
    else:
        check(True, "no dist/ present (it is gitignored and never committed)")


# ---------------------------------------------------------------------------
def test_r2_04_leakage_audit_catches_a_dms_split() -> None:
    section("R2-04: the split audit fires where the group-key check is blind")
    import src.splits as S

    rng = random.Random(0)
    parent = "".join(rng.choice(AA) for _ in range(200))
    rows, seen = [], set()
    while len(rows) < 400:
        i, a = rng.randrange(200), rng.choice(AA)
        if a == parent[i] or (i, a) in seen:
            continue
        seen.add((i, a))
        rows.append(parent[:i] + a + parent[i + 1:])

    groups = list(rows)                       # the shipped behaviour
    splits = S.row_splits_from_groups(groups, ratios=(0.8, 0.1, 0.1), seed=42)
    check(S.verify_no_leakage(groups, splits),
          "verify_no_leakage still passes on a DMS grouped by its own sequences")

    train = [s for s, sp in zip(rows, splits) if sp == "train"]
    held = [s for s, sp in zip(rows, splits) if sp != "train"]
    audit = S.near_duplicate_audit(train, held)
    check(audit["fraction_with_near_train_neighbour"] > 0.9,
          f"the near-duplicate audit DOES fail on it "
          f"({audit['fraction_with_near_train_neighbour']:.0%} of held-out rows)")

    # It must not cry wolf on genuinely unrelated proteins.
    unrelated_a = ["".join(rng.choice(AA) for _ in range(200)) for _ in range(150)]
    unrelated_b = ["".join(rng.choice(AA) for _ in range(200)) for _ in range(40)]
    clean = S.near_duplicate_audit(unrelated_a, unrelated_b)
    check(clean["fraction_with_near_train_neighbour"] == 0.0,
          "and reports 0% on unrelated proteins, so it is not just always-on")

    # A declared parent column collapses the variants into one group.
    _, info = S.build_groups(rows, declared=["construct_A"] * len(rows))
    check(info["n_groups"] == 1,
          f"a declared parent column makes all {len(rows)} variants one group")
    check(info["declared_group_relation_used"] is True,
          "and the report records that the declared relation was used")


# ---------------------------------------------------------------------------
def test_r2_05_scan_runs_one_encoder_pass() -> None:
    section("R2-05: a deep mutational scan encodes the protein once")
    from app.backend.inference import HoloGeoNEngine
    from src.model import HoloGeoN
    from src.substitution import AA_ORDER

    dim, n_positions = 32, 40
    seq = random_protein(90, seed=3)

    class CountingEncoder:
        def __init__(self):
            self.calls, self.seen = 0, []

        def encode(self, s, top_k=24, min_sep=3):
            self.calls += 1
            self.seen.append(s)
            return stub_encoding(s, dim=dim)

    eng = HoloGeoNEngine()
    eng._tried = eng._ready = True
    eng._torch = torch
    eng._device = torch.device("cpu")
    eng._config = {"window": 64, "contact_top_k": 8, "contact_min_sep": 3}
    eng._encoder = CountingEncoder()
    eng._untrained_heads = []
    eng._model = HoloGeoN(esm_dim=dim, hidden=32, out_dim=32, aa_dim=8,
                          gat_heads=2).eval()

    out = eng.mutation_scan(seq, 1, n_positions)
    check(eng._encoder.calls == 1,
          f"{n_positions} positions cost 1 encoder call, not {n_positions} "
          f"(got {eng._encoder.calls})")
    check(len(out["matrix"][0]) == n_positions,
          "and the scan still covers every requested position")

    # Batching must not change the numbers it used to produce one at a time.
    mt = "A" if seq[9] != "A" else "C"
    single = eng.predict_ddg(seq, f"{seq[9]}10{mt}")
    cell = out["matrix"][AA_ORDER.index(mt)][9]
    check(abs(single["ddg"] - cell) < 1e-3,
          f"a batched scan cell matches the single-site path "
          f"({cell:+.4f} vs {single['ddg']:+.4f})")


# ---------------------------------------------------------------------------
def test_r2_06_ablation_statistics() -> None:
    section("R2-06: null comparisons are not flagged, and the family is corrected")
    from scripts.ablations import (
        DATA_PROPERTY_METRICS,
        MIN_SEEDS_FOR_TEST,
        holm_bonferroni,
        minimum_detectable_effect,
        paired_stats,
    )

    rng = np.random.default_rng(7)
    family_size, trials = 20, 200
    false_families = 0
    for _ in range(trials):
        pvals = {}
        for c in range(family_size):
            full = rng.normal(0.5, 0.03, 5).tolist()
            arm = rng.normal(0.5, 0.03, 5).tolist()     # same distribution
            pvals[c] = paired_stats(arm, full)["p_value"]
        adj = holm_bonferroni(pvals)
        if any(v is not None and v < 0.05 for v in adj.values()):
            false_families += 1
    fwer = false_families / trials
    # The old criterion flagged at least one of 20 in ~82% of null families.
    check(fwer <= 0.15,
          f"family-wise error rate under the global null is {fwer:.1%}, "
          f"controlled near alpha=0.05 (the old criterion: ~82%)")

    # Below the seed floor there is no p-value at all, rather than a bad one.
    two = paired_stats([0.5, 0.6], [0.4, 0.4])
    check(two["p_value"] is None and two["n_seeds"] == 2,
          f"under {MIN_SEEDS_FOR_TEST} seeds no p-value is emitted")
    check("note" in two, "and the row says why")

    # Holm must be monotone and never exceed 1.
    adj = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.4, "d": None})
    ordered = [adj["a"], adj["b"], adj["c"]]
    check(all(x <= y for x, y in zip(ordered, ordered[1:])),
          "Holm adjustment is monotone in the raw p-value")
    check(all(v <= 1.0 for v in ordered), "and never exceeds 1.0")
    check(adj["d"] is None, "untested comparisons stay out of the family")

    check("target_std" in DATA_PROPERTY_METRICS and "pos_rate" in DATA_PROPERTY_METRICS,
          "split properties are excluded from the comparison set")

    # A real effect must still be findable with enough seeds.
    strong = paired_stats([0.40] * 5, [0.50] * 5)
    check(strong["p_value"] is None or strong["p_value"] < 0.05
          or strong["ci95_low"] is not None,
          "a large consistent effect still produces a usable row")
    mde5 = minimum_detectable_effect(0.042, 5, 8)
    mde20 = minimum_detectable_effect(0.042, 20, 8)
    check(mde20 < mde5,
          f"more seeds detect smaller effects ({mde5:.3f} at 5 -> {mde20:.3f} at 20)")


# ---------------------------------------------------------------------------
def test_r2_07_uniform_loss_floor() -> None:
    section("R2-07: the loss floor is uniform and the docstring is accurate")
    from src.losses import GLOBAL_LOSS_FLOOR, LOSS_FLOORS, UncertaintyWeightedLoss

    check(len(set(LOSS_FLOORS.values())) == 1,
          "every task carries the same floor, so the offset is truly uniform")
    check(all(v == GLOBAL_LOSS_FLOOR for v in LOSS_FLOORS.values()),
          f"and it is GLOBAL_LOSS_FLOOR ({GLOBAL_LOSS_FLOOR})")

    crit = UncertaintyWeightedLoss(["stability", "pathogenicity"])
    w = crit.equilibrium_weights({"stability": 1.0, "pathogenicity": 0.6})
    ratio = max(w.values()) / min(w.values())
    # The reviewer measured 8.5x from the asymmetric floors alone.
    check(ratio < 2.0,
          f"at ddG NLL 1.0 vs BCE 0.6 the weight imbalance is {ratio:.2f}x, "
          f"not the 8.5x the per-task floors created")

    doc = UncertaintyWeightedLoss.__doc__ or ""
    check("not gradient-neutral" in doc,
          "the docstring no longer claims the shift changes no gradient")

    check(crit.current_weights().keys() == {"stability", "pathogenicity"},
          "per-task weights are inspectable for the training log")


# ---------------------------------------------------------------------------
def test_r2_08_mfi_uses_the_global_readout() -> None:
    section("R2-08: the MFI head is not anchored at residue 0")
    import inspect

    from src.model import HoloGeoN

    for name in ("idr", "mfi", "proteomics"):
        source = inspect.getsource(getattr(HoloGeoN, name))
        check("self.global_context(batch)" in source,
              f"HoloGeoN.{name} uses the whole-sequence readout")

    # And it must actually change the number, or the fix is cosmetic.
    from src.data import build_sample_graph, collate_graphs

    rec = stub_encoding(random_protein(300, seed=5), dim=32)
    cfg = {"graph_top_k": 8, "geo_threshold": 0.3, "neighborhood_hops": 2}
    g = build_sample_graph(rec, None, 256, cfg, max_nodes=2048)
    g.update({"wt_idx": 0, "mt_idx": 0, "y": 0.0, "sigma": float("nan")})
    batch = collate_graphs([g])
    model = HoloGeoN(esm_dim=32, hidden=32, out_dim=32, aa_dim=8, gat_heads=2).eval()
    with torch.no_grad():
        site_anchored = model.context(batch)
        global_ctx = model.global_context(batch)
    check(not torch.allclose(site_anchored, global_ctx),
          "the two readouts genuinely differ, so the change is not cosmetic")


# ---------------------------------------------------------------------------
def test_r2_09_calibration_describes_the_served_sigma() -> None:
    section("R2-09: reported calibration describes the interval actually drawn")
    from scripts.calibration import calibration_report, fit_variance_scale

    rng = np.random.default_rng(0)
    n = 3000
    mu, sigma = np.zeros(n), np.ones(n)
    y_test = rng.normal(0, 1.0, n)          # raw sigma is right on test
    y_val = rng.normal(0, 1.6, n)           # val residuals are wider

    raw = calibration_report(mu, sigma, y_test)
    scale = fit_variance_scale(mu, sigma, y_val)
    served = calibration_report(mu, sigma * scale, y_test)

    check(raw["well_calibrated"] and not served["well_calibrated"],
          f"the raw sigma is calibrated and the served one (x{scale:.2f}) is not "
          f"- the case the interface used to report backwards")

    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "app", "backend", "inference.py"), encoding="utf-8").read()
    check("calibration_after_scaling" in src,
          "the engine reads the post-scaling calibration the run already computed")
    check('"describes"' in src,
          "and health() states which sigma its numbers describe")

    # The IDR head must not inherit a scale fitted in kcal/mol.
    idr_body = src[src.index("def idr_ensemble"):src.index("def idr_ensemble") + 3000]
    check("self._sigma_scale" not in idr_body,
          "the IDR head serves its own sigma, not one rescaled in ddG units")


# ---------------------------------------------------------------------------
def test_r2_11_three_state_band_is_declared() -> None:
    section("R2-11: the neutral band is a declared, recorded number")
    from src.config import DEFAULTS, load_config
    from scripts.train import band_sensitivity, three_state_labels

    check("three_state_sigma" in DEFAULTS,
          "the band is a config key, not a module constant")
    cfg = load_config()
    check(cfg.get("three_state_sigma") is not None,
          "and it is present in the shipped configuration")

    # A non-positive reported sigma must be ignored, not used.
    batch = {"y": torch.tensor([0.0, 0.3, -0.3, 5.0]),
             "sigma": torch.tensor([0.0, -1.0, float("nan"), float("nan")])}
    labels = three_state_labels(batch, 0.5)
    check(labels.tolist() == [1, 1, 1, 2],
          f"sigma <= 0 falls back to the declared band rather than making "
          f"nothing neutral (got {labels.tolist()})")

    # And the sensitivity row must show the band actually moving the labels.
    import pandas as pd

    y = np.random.default_rng(0).normal(-1.0, 1.5, 4000)
    table = pd.DataFrame({"target": y, "sigma": [float("nan")] * len(y)})
    sens = band_sensitivity(table, 0.5)
    bands = sorted(k for k in sens if k.startswith("sigma_x"))
    neutral = [sens[k]["fraction_neutral"] for k in bands]
    check(len(set(neutral)) == len(neutral) and neutral == sorted(neutral),
          f"the neutral fraction rises with the band {neutral}, so the "
          f"sensitivity row is informative")
    check(sens["band_source"]["rows_using_the_fallback"] == len(y),
          "and the report says these rows used the fallback band")

    # The case this test used to miss entirely. Every row here reports its OWN
    # sigma, so the fallback governs nothing - which is the shipped MegaScale
    # situation (all 389,068 rows carry a finite positive sigma). Scaling only
    # the fallback produced three IDENTICAL rows under three different band
    # labels, reading as "insensitive to the band" when the band never moved.
    with_sigma = pd.DataFrame({"target": y,
                               "sigma": np.full(len(y), 0.05)})
    sens2 = band_sensitivity(with_sigma, 0.5)
    bands2 = sorted(k for k in sens2 if k.startswith("sigma_x"))
    neutral2 = [sens2[k]["fraction_neutral"] for k in bands2]
    check(len(set(neutral2)) == len(neutral2) and neutral2 == sorted(neutral2),
          f"the band still moves the labels when every row reports its own "
          f"sigma {neutral2}")
    check(sens2["band_source"]["rows_using_the_fallback"] == 0
          and sens2["band_source"]["rows_using_their_own_sigma"] == len(y),
          "and the report shows the fallback governed no row, so the declared "
          "three_state_sigma cannot be mistaken for the band that was used")


# ---------------------------------------------------------------------------
def test_r2_12_resume_continues_the_data_order() -> None:
    section("R2-12: a resumed run continues the schedule, not epoch 1")
    from torch.utils.data import DataLoader, Dataset

    from src.seeding import loader_kwargs, reseed_loaders_for_epoch

    class Toy(Dataset):
        def __len__(self): return 12
        def __getitem__(self, i): return i

    def fresh():
        return {"stability": {"train": DataLoader(
            Toy(), batch_size=3, shuffle=True, num_workers=0,
            **loader_kwargs(42, 0))}}

    def order(loaders, epoch):
        reseed_loaders_for_epoch(loaders, 42, epoch)
        return [b.tolist() for b in loaders["stability"]["train"]]

    straight = fresh()
    e1 = order(straight, 1)
    e2 = order(straight, 2)

    resumed = fresh()                      # a restarted process
    r2 = order(resumed, 2)

    check(r2 == e2, "epoch 2 after a resume matches epoch 2 uninterrupted")
    check(r2 != e1, "and does not replay epoch 1, which is what it used to do")
    check(e1 != e2, "consecutive epochs differ, so reproducibility is not trivial")


# ---------------------------------------------------------------------------
def test_r2_13_contact_sanity_has_a_random_control() -> None:
    section("R2-13: the contact check compares against random edges")
    import inspect

    import scripts.contact_sanity as cs

    check(hasattr(cs, "_top_random"), "a random-edge control exists")

    # The finding: random edges score highly against the proximity baseline.
    ri, rj = cs._top_random(300, 300, 6, seed=0)
    li, lj = cs._top_linear(300, 300, 6)
    rnd_lr = cs._range_fractions(ri, rj)[0]
    lin_lr = cs._range_fractions(li, lj)[0]
    check(rnd_lr > 0.5 and lin_lr == 0.0,
          f"uniformly random edges score {rnd_lr:.2f} long-range against the "
          f"proximity baseline's {lin_lr:.2f} - so that gap proves nothing")

    source = inspect.getsource(cs)
    check("--require-structure" in source,
          "a flag makes ground-truth contact maps mandatory for reported runs")
    check("supports_structural_claim" in source,
          "and the artifact records whether it can support the claim at all")


# ---------------------------------------------------------------------------
def test_r2_15_provenance_is_recorded() -> None:
    section("R2-15: runs record the software environment")
    from src.provenance import provenance

    p = provenance()
    for key in ("git", "python", "platform", "packages", "device"):
        check(key in p, f"provenance records {key}")
    check(p["packages"].get("torch") is not None,
          "including the torch version, which changes results")
    check("dirty" in p["git"],
          "and whether the tree was dirty, since a commit alone would not identify it")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reqs = open(os.path.join(root, "requirements.txt"), encoding="utf-8").read()
    check("transformers>=4.40,<5" in reqs,
          "transformers is bounded above: the contact head needs eager attention")


# ---------------------------------------------------------------------------
def test_r1_26_chunk_selection_prefers_more_context() -> None:
    section("R1-26 (re-opened): long-protein chunking never loses context")
    from src.encoder import CHUNK_OVERLAP, MAX_CONTEXT_RESIDUES, EsmEncoder, chunk_context_score

    def kept_fraction(length, start, end, i):
        al, ar = i, length - 1 - i
        gl, gr = i - start, end - 1 - i
        return min(1.0 if al == 0 else gl / al, 1.0 if ar == 0 else gr / ar)

    for length in (1023, 1600, 3000):
        spans = EsmEncoder._chunk_spans(length)
        best = torch.full((length,), -1.0)
        owner = torch.full((length,), -1, dtype=torch.long)
        for si, (start, end) in enumerate(spans):
            score = chunk_context_score(length, start, end)
            take = score > best[start:end]
            owner[start:end][take] = si
            best[start:end][take] = score[take]

        # The old rule: distance from the nearest boundary, capped at n/2 - a cap
        # that never binds, since the largest attainable margin is (n-1)/2.
        old_best = torch.full((length,), -1.0)
        old_owner = torch.full((length,), -1, dtype=torch.long)
        for si, (start, end) in enumerate(spans):
            n = end - start
            pos = torch.arange(n)
            margin = torch.minimum(pos, torch.tensor(n - 1) - pos).float()
            margin = torch.clamp(margin, max=n / 2.0)
            take = margin > old_best[start:end]
            old_owner[start:end][take] = si
            old_best[start:end][take] = margin[take]

        new_kept = [kept_fraction(length, *spans[int(owner[i])], i) for i in range(length)]
        old_kept = [kept_fraction(length, *spans[int(old_owner[i])], i) for i in range(length)]
        check(all(n >= o - 1e-9 for n, o in zip(new_kept, old_kept)),
              f"L={length}: no residue keeps LESS context than under the old rule")
        check(min(new_kept) >= min(old_kept),
              f"L={length}: the worst-served residue improves "
              f"({min(old_kept):.3f} -> {min(new_kept):.3f})")

    # The reviewer's exact case.
    spans = EsmEncoder._chunk_spans(1023)
    scores = {si: float(chunk_context_score(1023, *sp)[894 - sp[0]])
              for si, sp in enumerate(spans) if sp[0] <= 894 < sp[1]}
    winner = max(scores, key=scores.get)
    check(spans[winner] == (0, 1022),
          f"L=1023 residue 894 now comes from the 1022-residue chunk, not the "
          f"257-residue tail (got {spans[winner]})")


# ---------------------------------------------------------------------------
def test_r1_07_truncation_is_reported() -> None:
    section("R1-07 (re-opened): the node cap is applied AND reported")
    from src.data import build_sample_graph, collate_graphs

    rec = stub_encoding(random_protein(400, seed=11), dim=16)
    cfg = {"graph_top_k": 8, "geo_threshold": 0.3, "neighborhood_hops": 2}

    capped = build_sample_graph(rec, None, 256, cfg, max_nodes=100)
    whole = build_sample_graph(rec, None, 256, cfg, max_nodes=2048)
    check(capped["truncated"] is True and whole["truncated"] is False,
          "the flag is set when the cap bites and not otherwise")

    for g in (capped, whole):
        g.update({"wt_idx": 0, "mt_idx": 0, "y": 0.0, "sigma": float("nan")})
    batch = collate_graphs([capped, whole])
    check("truncated" in batch,
          "collate_graphs carries it into the batch - it used to be dropped here")
    check(batch["truncated"].tolist() == [True, False],
          "with the per-sample values intact")
    check(int(whole["length"]) == 400,
          "and the true chain length still travels alongside")


# ---------------------------------------------------------------------------
def test_r1_17_record_cache_is_real_or_absent() -> None:
    section("R1-17 (re-opened): the cache that was claimed now exists, and is off")
    from src.encoder import EncodingStore

    tmp = tempfile.mkdtemp()
    seq = random_protein(40, seed=2)

    off = EncodingStore(tmp)
    off.put(stub_encoding(seq, dim=8))
    for _ in range(5):
        off.get(seq)
    check(off.cache_stats()["enabled"] is False,
          "the record cache is OFF by default, so no RAM cost is inherited")

    on = EncodingStore(tmp, cache_records=8)
    for _ in range(5):
        on.get(seq)
    stats = on.cache_stats()
    check(stats["hits"] == 4 and stats["misses"] == 1,
          f"when enabled it serves repeats from memory {stats}")
    check(torch.equal(on.get(seq)["emb"], off.get(seq)["emb"]),
          "and returns the same record the disk path would")


# ---------------------------------------------------------------------------
def test_r2_whole_sequence_path_end_to_end() -> None:
    section("R2-02/04/08: the whole-sequence path, end to end")
    import subprocess

    import pandas as pd

    from src.encoder import EncodingStore

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stub_dim = 320                      # matches esm2_t6_8M so ESM_DIMS agrees

    with tempfile.TemporaryDirectory(prefix="holo_geon_seq_") as workdir:
        data = os.path.join(workdir, "data")
        cache = os.path.join(workdir, "cache")
        os.makedirs(data, exist_ok=True)
        store = EncodingStore(cache)

        # Three constructs, each with many single-point variants: an MFI-shaped
        # table. Grouping by the row's own sequence puts every variant in its own
        # group, which is exactly the split R2-04 showed passing while leaking.
        rng = random.Random(5)
        rows = []
        for c in range(10):
            parent = random_protein(150, seed=200 + c)
            store.put(stub_encoding(parent, dim=stub_dim, top_k=8))
            for _ in range(25):
                i, a = rng.randrange(150), rng.choice(AA)
                if a == parent[i]:
                    continue
                variant = parent[:i] + a + parent[i + 1:]
                store.put(stub_encoding(variant, dim=stub_dim, top_k=8))
                rows.append({"wt_seq": variant, "value": float(rng.gauss(0, 1)),
                             "construct": f"C{c}"})

        leaky = os.path.join(data, "mfi_leaky.parquet")
        grouped = os.path.join(data, "mfi_grouped.parquet")
        pd.DataFrame([{k: v for k, v in r.items() if k != "construct"}
                      for r in rows]).to_parquet(leaky)
        pd.DataFrame(rows).to_parquet(grouped)

        stability = os.path.join(data, "stab.parquet")
        srows = []
        for c in range(12):
            parent = random_protein(150, seed=300 + c)
            store.put(stub_encoding(parent, dim=stub_dim, top_k=8))
            for _ in range(20):
                i, a = rng.randrange(150), rng.choice(AA)
                if a == parent[i]:
                    continue
                srows.append({"wt_seq": parent,
                              "mt_seq": parent[:i] + a + parent[i + 1:],
                              "ddg": float(rng.gauss(0, 1)), "gene": f"G{c}"})
        pd.DataFrame(srows).to_parquet(stability)

        cfg_text = (
            "esm_model: facebook/esm2_t6_8M_UR50D\n"
            f"cache: {cache}\n"
            "device: cpu\nwindow: 48\ncontact_top_k: 8\ncontact_min_sep: 3\n"
            f"out_root: {os.path.join(workdir, 'runs')}\n"
            f"stability_data: {stability}\nclinvar_data: null\n"
            "mfi_data: {mfi}\nidr_data: null\nproteomics_data: null\n"
            "max_protein_length: 5000\nidentity: 0.30\ncoverage: 0.80\n"
            "seed: 42\ngraph_top_k: 8\ngeo_threshold: 0.30\neps: 0.10\n"
            "delta: 0.10\nmech_max_degree: 8\nneighborhood_hops: 2\n"
            "hidden: 32\nout_dim: 32\ngat_heads: 2\ndropout: 0.1\n"
            "use_ssm: true\nuse_mech_stream: true\nepochs: 2\nbatch_size: 8\n"
            "lr: 0.003\nweight_decay: 0.0001\nwarmup_frac: 0.1\n"
            "early_stop_patience: 20\ngrad_clip: 1.0\namp: false\n"
            "num_workers: 0\nmax_rows: null\nmax_sequence_nodes: 256\n"
            "sequence_node_budget: 384\n"
        )

        def write_cfg(mfi_path):
            path = os.path.join(workdir, f"cfg_{os.path.basename(mfi_path)}.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(cfg_text.format(mfi=mfi_path))
            return path

        splits = os.path.join(data, "splits_stab.parquet")
        subprocess.run([sys.executable, "-m", "scripts.make_splits",
                        "--data", stability, "--kind", "stability",
                        "--out", splits, "--config", write_cfg(grouped)],
                       cwd=root, capture_output=True, text=True)

        def train(mfi_path, out, env=None):
            return subprocess.run(
                [sys.executable, "-m", "scripts.train",
                 "--config", write_cfg(mfi_path),
                 "--stability-splits", splits,
                 "--pathogenicity-splits", os.path.join(data, "missing.parquet"),
                 "--out", out],
                cwd=root, capture_output=True, text=True, env=env)

        # 1. The leaky table must be REFUSED.
        #
        # The blind spot R2-04 is about belongs to the FALLBACK grouping
        # relation: with no mmseqs, grouping degrades to exact parent-sequence
        # identity, every DMS variant becomes its own group, the group-key check
        # passes, and every test row still sits one residue from a training row.
        # The near-duplicate audit is the net under that, and the fallback is the
        # configuration that exercises it.
        #
        # With mmseqs on PATH the leak cannot arise at all: identity clustering
        # puts every variant of a construct in one cluster, whole constructs go
        # to one side, and the run correctly proceeds - so asserting a refusal
        # here would be asserting the absence of mmseqs. Checked separately
        # below. Hiding mmseqs keeps this test about the audit, not the machine.
        env_fallback = dict(os.environ)
        env_fallback["PATH"] = os.pathsep.join(
            p for p in env_fallback.get("PATH", "").split(os.pathsep)
            if p and not any(os.path.exists(os.path.join(p, exe))
                             for exe in ("mmseqs", "mmseqs.exe"))
        )
        bad = train(leaky, os.path.join(workdir, "run_leaky"), env=env_fallback)
        combined = bad.stdout + bad.stderr
        check(bad.returncode != 0,
              f"a DMS-shaped table with no grouping column is refused "
              f"(exit {bad.returncode})")
        check("near-identical" in combined,
              "and the refusal names the near-duplicate overlap it measured")
        check("configs/data.yaml" in combined,
              "and tells the user how to fix it")

        # 1b. The other half of the same claim: where mmseqs IS installed, the
        # identity clustering must group those variants by construct on its own,
        # so the situation the audit refuses never arises. Checked at the
        # clustering level rather than by another training run.
        from src.splits import cluster_with_mmseqs, mmseqs_available

        if mmseqs_available():
            variant_seqs = {f"v{i}": r["wt_seq"] for i, r in enumerate(rows)}
            construct_of = {f"v{i}": r["construct"] for i, r in enumerate(rows)}
            clustered = cluster_with_mmseqs(
                variant_seqs, min_seq_id=0.30, coverage=0.80,
                tmp_dir=os.path.join(workdir, "mmseqs_tmp"))
            by_cluster: dict[str, set] = {}
            for sid, rep in clustered.items():
                by_cluster.setdefault(rep, set()).add(construct_of[sid])
            check(len(by_cluster) == len({r["construct"] for r in rows}),
                  f"mmseqs recovers one cluster per construct without a declared "
                  f"column (got {len(by_cluster)} for "
                  f"{len({r['construct'] for r in rows})} constructs)")
            check(all(len(v) == 1 for v in by_cluster.values()),
                  "and no cluster mixes variants of different constructs, so "
                  "whole constructs go to one side of the split")
        else:
            print("    skip mmseqs identity-clustering checks (mmseqs not on PATH)")

        # 2. With a declared construct column it must TRAIN.
        good = train(grouped, os.path.join(workdir, "run_grouped"))
        if good.returncode != 0:
            print(f"    --- stdout ---\n{good.stdout[-2500:]}")
            print(f"    --- stderr ---\n{good.stderr[-2500:]}")
        check(good.returncode == 0,
              f"a declared construct column lets the same rows train "
              f"(exit {good.returncode})")

        metrics_path = os.path.join(workdir, "run_grouped", "metrics.json")
        check(os.path.exists(metrics_path), "metrics.json written")
        if os.path.exists(metrics_path):
            import json as _json

            with open(metrics_path, encoding="utf-8") as fh:
                m = _json.load(fh)
            mfi_report = (m.get("data_reports") or {}).get("mfi") or {}
            grouping = mfi_report.get("grouping") or {}
            check(grouping.get("declared_group_relation_used") is True,
                  "the run records that the declared grouping relation was used")
            audit = grouping.get("near_duplicate_audit") or {}
            check(audit.get("fraction_with_near_train_neighbour") == 0.0,
                  f"and that the near-duplicate audit measured 0% leakage "
                  f"(got {audit.get('fraction_with_near_train_neighbour')})")
            check("mfi" in (m.get("trained_tasks") or []),
                  "the MFI head trained through the node-budget loader")
            check("provenance" in m and m["provenance"]["packages"]["torch"],
                  "and the manifest records the environment it ran in")
            stab = (m.get("data_reports") or {}).get("stability") or {}
            check("window_coverage" in stab,
                  "the ablation arm's per-dataset meaning is recorded (R2-14)")
            check("three_state_band" in m,
                  "the neutral band and its sensitivity are recorded (R2-11)")


# ---------------------------------------------------------------------------
def test_r2_16_matched_capacity_control() -> None:
    section("R2-16: the MLP control comes in matched-budget AND matched-capacity")
    import inspect

    import scripts.baselines as B
    from src.config import load_config
    from src.model import HoloGeoN, model_kwargs_from_config

    cfg = load_config()
    full = HoloGeoN(**model_kwargs_from_config(cfg, 1280))
    target = full.parameter_count()["total"]

    feat_dim = 2 * 1280 + 40          # pooled window + site embedding + 2 one-hots
    narrow = sum(p.numel() for p in B._mlp(feat_dim, 512).parameters())
    width = B.mlp_width_for(feat_dim, target)
    matched = sum(p.numel() for p in B._mlp(feat_dim, width).parameters())

    check(target / narrow > 10,
          f"the narrow control really is far smaller than the model "
          f"({target/narrow:.1f}x) - the confound R2-16 named")
    check(0.9 <= matched / target <= 1.1,
          f"the matched-capacity control lands within 10% of the model's "
          f"parameter count ({matched:,} vs {target:,})")

    source = inspect.getsource(B.main)
    check("esm_mlp_matched_capacity" in source,
          "and both variants are actually run and reported")

    mlp_source = inspect.getsource(B.esm_mlp_baseline)
    for knob in ("sched.step()", "clip_grad_norm_", "early_stop_patience"):
        check(knob in mlp_source,
              f"the control shares the model's {knob.split('(')[0]}")


# ---------------------------------------------------------------------------
def test_r1_11_codon_substitution_is_unbiased() -> None:
    section("R1-11 residual: the substituted codon no longer always improves CAI")
    from src.codon_tables import (
        cai_matched_codon,
        get_cai_weights,
        preferred_codon,
        synonymous_codons,
    )

    weights = get_cai_weights("human")

    # The defect: preferred_codon is the arg-max of a table normalised to 1.
    pref = [weights[preferred_codon(a)] for a in AA]
    check(all(abs(w - 1.0) < 1e-9 for w in pref),
          "preferred_codon has CAI weight 1.0 for every residue, by definition")

    # Shift induced across every (parent codon, target residue) pair.
    old_shifts, new_shifts = [], []
    for a in AA:
        for _, parent_w in synonymous_codons(a):
            for mt in AA:
                old_shifts.append(weights[preferred_codon(mt)] - parent_w)
                _, w = cai_matched_codon(mt, parent_w)
                new_shifts.append(w - parent_w)

    old_mean = sum(old_shifts) / len(old_shifts)
    new_mean = sum(new_shifts) / len(new_shifts)
    check(old_mean > 0.2,
          f"the old substitution shifted CAI by {old_mean:+.3f} on average - "
          f"one-sided, at the one site the reversibility check is about")
    check(abs(new_mean) < abs(old_mean) / 2,
          f"the CAI-matched codon cuts that to {new_mean:+.3f}")

    # It must still encode the residue it was asked for.
    from src.codon_tables import GENETIC_CODE

    ok = True
    for mt in AA:
        codon, _ = cai_matched_codon(mt, 0.5)
        ok = ok and codon is not None and GENETIC_CODE[codon] == mt
    check(ok, "and still returns a codon that encodes the requested residue")

    import inspect

    import scripts.calibration as C

    check("induced_cai_shift_at_mutated_site" in inspect.getsource(C.reversibility_check),
          "the residual shift is reported, not assumed to be zero")


# ---------------------------------------------------------------------------
def test_r2_17_documentation_matches_the_code() -> None:
    section("R2-17: the documented claims match what the code does")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def read(name):
        with open(os.path.join(root, name), encoding="utf-8") as fh:
            return fh.read()

    notes, readme = read("MIGRATION_NOTES.md"), read("README.md")

    # The over-claim R1-23 was raised for, restated in the document that
    # records R1-23.
    claim = "antisymmetry in both train and eval mode"
    check(claim not in notes,
          "MIGRATION_NOTES no longer claims antisymmetry holds in train mode")
    check(claim not in readme, "nor does the README")

    check("room to spare" not in readme,
          "the README no longer says the run fits 16 GB 'with room to spare'")
    check("message-passing contribution" not in readme,
          "nor attributes the whole esm_mlp gap to message passing")

    for name in ("src/features.py", "src/backbone.py",
                 "scripts/precompute_embeddings.py"):
        body = read(name)
        check("seven mechanistic" not in body and "channels (7)" not in body
              and "(length, 7)" not in body,
              f"{name} does not say 7 mechanistic channels (N_MECH_CHANNELS is 8)")

    mixers = read("src/sequence_mixers.py")
    # The finding was about the section HEADERS, not about the docstring's
    # historical note recording that the "paper §8.2" framing was removed.
    check("# §8.2-i" not in mixers and "# §8.2-ii" not in mixers,
          "the §8.2 scaffolding headers MIGRATION_NOTES says are gone, are gone")

    check("no such file" not in read("src/splits.py")
          and "tests/test_splits.py" not in read("src/splits.py"),
          "src/splits.py no longer cites a test file that does not exist")

    check("REMARK 999" in read("app/backend/app.py"),
          "the PDB export disclaims its invented geometry inside the file")


# ---------------------------------------------------------------------------
TESTS = [
    test_r2_01_cds_coverage_measured_from_the_cache,
    test_r2_02_node_budget_caps_whole_sequence_batches,
    test_r2_03_no_frontend_heuristic,
    test_r2_04_leakage_audit_catches_a_dms_split,
    test_r2_05_scan_runs_one_encoder_pass,
    test_r2_06_ablation_statistics,
    test_r2_07_uniform_loss_floor,
    test_r2_08_mfi_uses_the_global_readout,
    test_r2_09_calibration_describes_the_served_sigma,
    test_r2_11_three_state_band_is_declared,
    test_r2_12_resume_continues_the_data_order,
    test_r2_13_contact_sanity_has_a_random_control,
    test_r2_15_provenance_is_recorded,
    test_r1_26_chunk_selection_prefers_more_context,
    test_r1_07_truncation_is_reported,
    test_r1_17_record_cache_is_real_or_absent,
    test_r2_whole_sequence_path_end_to_end,
    test_r2_16_matched_capacity_control,
    test_r1_11_codon_substitution_is_unbiased,
    test_r2_17_documentation_matches_the_code,
]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
