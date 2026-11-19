"""
tests/test_remote_job.py
========================
Regressions for scripts/remote_job.py, the Kaggle/Colab runner.

Pure-Python: no GPU, no data, no subprocess is launched. These cover the parts
whose failure would waste a paid session silently - item order, sharding,
resume, input discovery on Kaggle's extracted-archive layout, and the cost plan.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from tests.helpers import check, section

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _manifest(name: str) -> dict:
    with open(os.path.join(ROOT, "configs", "experiments", name), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_items_are_seed_major_and_paired() -> None:
    section("remote_job - seed-major order keeps paired observations complete")
    from scripts.remote_job import expand_items

    items = expand_items(_manifest("kaggle_ablation.yaml"))
    trains = [i for i in items if i["kind"] == "train"]
    check(len(trains) == 12, f"4 arms x 3 seeds = 12 train items (got {len(trains)})")
    seeds = [t["seed"] for t in trains]
    check(seeds == sorted(seeds), "every arm at seed 42 runs before any arm at seed 43")
    check(all(t["needs"] == ["precompute"] for t in trains),
          "every train item depends on the encoder cache being built")
    ext = [i for i in items if i["kind"] == "external"]
    check(all(e["needs"][0] in {t["id"] for t in trains} for e in ext),
          "each external item follows its own train item")
    check(items[0]["id"] == "precompute" and items[-1]["kind"] == "aggregate",
          "precompute first, aggregate last")


def test_shards_partition_the_grid() -> None:
    section("remote_job - shards are whole seed blocks, covering each run once")
    from scripts.remote_job import expand_items, take_shard

    items = expand_items(_manifest("kaggle_ablation.yaml"))
    all_arms = {i["arm"] for i in items if i["kind"] == "train"}
    seen: list[str] = []
    for k in range(3):
        part = take_shard(items, f"{k}/3")
        trains = [i for i in part if i["kind"] == "train"]
        t = [i["id"] for i in trains]
        check(len(t) == 4, f"shard {k}/3 has 4 train items (got {len(t)})")
        # The point of a shard: one lost session must not cost the paired test.
        shard_seeds = {i["seed"] for i in trains}
        check(len(shard_seeds) == 1,
              f"shard {k}/3 holds exactly one seed (got {sorted(shard_seeds)})")
        check({i["arm"] for i in trains} == all_arms,
              f"shard {k}/3 holds every arm, so it is a complete paired block")
        for e in (i for i in part if i["kind"] == "external"):
            check(e["needs"][0] in t, f"shard {k}: {e['id']} sits with its train item")
        check(any(i["kind"] == "precompute" for i in part),
              f"shard {k} builds its own encoder cache (each session is its own VM)")
        check(not any(i["kind"] == "aggregate" for i in part),
              f"shard {k} does not aggregate a partial grid")
        seen += t
    check(len(seen) == len(set(seen)) == 12, "no train item is run twice or dropped")

    # A shard cannot split a seed, so more shards than seeds must be refused
    # rather than silently producing a shard with no paired partner.
    try:
        take_shard(items, "0/4")
    except SystemExit as exc:
        check("only 3 seeds" in str(exc), "4 shards over 3 seeds is refused with a usable message")
    else:
        check(False, "4 shards over 3 seeds must be refused")


def test_input_discovery_flattens_kaggle_folders() -> None:
    section("remote_job - inputs are found by name, X.parquet/X.parquet flattened")
    from scripts.remote_job import find_inputs

    with tempfile.TemporaryDirectory() as tmp:
        deep = os.path.join(tmp, "datasets", "user", "slug")
        os.makedirs(os.path.join(deep, "a.parquet"))
        with open(os.path.join(deep, "a.parquet", "a.parquet"), "w") as fh:
            fh.write("x")
        with open(os.path.join(deep, "b.parquet"), "w") as fh:
            fh.write("y")
        got = find_inputs([tmp], ("a.parquet", "b.parquet", "c.parquet"))
        check(os.path.isfile(got["a.parquet"]), "a folder named a.parquet resolves to the file inside")
        check(got["b.parquet"].endswith("b.parquet") and os.path.isfile(got["b.parquet"]),
              "a plain file resolves directly")
        check("c.parquet" not in got, "a missing file is reported missing, not guessed")


def test_resume_restores_finished_items() -> None:
    section("remote_job - a restarted VM restores finished packs and skips them")
    from scripts.remote_job import Store, _write_json, expand_items

    items = expand_items(_manifest("colab_g4_benchmark.yaml"))
    train = next(i for i in items if i["kind"] == "train")
    with tempfile.TemporaryDirectory() as persist, tempfile.TemporaryDirectory() as run1, \
            tempfile.TemporaryDirectory() as run2:
        s1 = Store(run1, persist, os.path.join(run1, "cache"))
        pack = s1.pack_path(train)
        os.makedirs(pack)
        for name in ("metrics.json", "model_best.pt", "last.pt"):
            with open(os.path.join(pack, name), "w") as fh:
                fh.write("{}")
        s1.save(train, {"id": train["id"], "returncode": 0})
        check(s1.is_done(train["id"]), "DONE marker written")
        check(not os.path.exists(os.path.join(persist, "packs", train["pack"], "last.pt")),
              "last.pt (resume state, not a result) is not copied to persistent storage")

        s2 = Store(run2, persist, os.path.join(run2, "cache"))   # a fresh VM
        check(not s2.is_done(train["id"]), "fresh VM knows nothing yet")
        restored = s2.restore(items)
        check(train["id"] in restored and s2.is_done(train["id"]),
              "restore brings the DONE marker back")
        check(os.path.isfile(os.path.join(s2.pack_path(train), "model_best.pt")),
              "and the checkpoint, so the external item can run without retraining")


def test_plan_reprojects_and_cuts_in_order() -> None:
    section("remote_job - the plan fits the budget, and cuts in the declared order")
    import io
    from contextlib import redirect_stdout

    from scripts.remote_job import est_hours, expand_items, print_plan

    m = _manifest("colab_g4_benchmark.yaml")
    items = expand_items(m)
    sz = m["sizing"]
    train = next(i for i in items if i["kind"] == "train")
    fast, slow = est_hours(train, sz, 0.030), est_hours(train, sz, 0.120)
    check(abs(slow / fast - 4.0) < 1e-9, "train hours scale linearly with seconds per step")
    one_epoch = sz["steps_per_epoch"] * 0.040 / 3600
    check(abs(est_hours(dict(train, epochs=1), sz, 0.040) - one_epoch * 1.10) < 1e-9,
          "one epoch = steps_per_epoch x s/step x (1 + eval overhead)")

    buf = io.StringIO()
    with redirect_stdout(buf):
        total_ok = print_plan(m, items, 0.040, "ASSUMED")
    check("OVER BUDGET" not in buf.getvalue(), "the shipped plan fits 100 CU at the assumed speed")
    check(total_ok * m["cost"]["cu_per_hour"] <= m["cost"]["budget_cu"] - m["cost"]["reserve_cu"],
          "total CU stays under budget minus the reserve")

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_plan(m, items, 0.120, "from --sec-per-step")   # 3x slower than assumed
    out = buf.getvalue()
    check("OVER BUDGET" in out, "a 3x slower measured step is flagged over budget")
    first_cut = out.index("after cutting")
    check("competitor_retrain_*" in out[first_cut:first_cut + 80],
          "the first thing cut is the declared first entry, not the headline arm")


def test_scorer_guards_sign_and_common_rows() -> None:
    section("score_predictions - wrong sign refused, scoring restricted to common rows")
    import io
    from contextlib import redirect_stdout

    import numpy as np
    import pandas as pd

    from scripts import score_predictions as sp

    rng = np.random.default_rng(0)
    n = 600
    splits = pd.DataFrame({
        "parent_seq": np.repeat([f"SEQ{i}" for i in range(12)], n // 12),
        "target": rng.normal(size=n),
        "split": "test",
        "group_used": np.repeat([f"g{i % 4}" for i in range(12)], n // 12),
    })
    y = splits["target"].to_numpy()
    with tempfile.TemporaryDirectory() as tmp:
        sp_path = os.path.join(tmp, "splits.parquet")
        splits.to_parquet(sp_path)
        good = pd.DataFrame({"row_id": np.arange(n), "y_pred": y + rng.normal(0, 0.5, n)})
        weak = pd.DataFrame({"row_id": np.arange(n), "y_pred": y + rng.normal(0, 2.0, n)})
        flipped = pd.DataFrame({"row_id": np.arange(n), "y_pred": -(y + rng.normal(0, 0.5, n))})
        partial = good.iloc[: n // 2]
        for name, frame in (("good", good), ("weak", weak), ("flipped", flipped), ("partial", partial)):
            frame.to_parquet(os.path.join(tmp, f"{name}.parquet"))

        def run(*models: str):
            argv = ["score", "--splits", sp_path, "--reps", "50"]
            for m in models:
                argv += ["--model", m]
            old, sys.argv = sys.argv, argv
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    sp.main()
            finally:
                sys.argv = old
            return buf.getvalue()

        out = run(f"good={tmp}/good.parquet", f"weak={tmp}/weak.parquet")
        check("good" in out and "weak" in out, "two models are scored side by side")
        try:
            run(f"flipped={tmp}/flipped.parquet:+1")
            refused = False
        except SystemExit as exc:
            refused = "sign-convention" in str(exc)
        check(refused, "a model whose sign is wrong is refused, not reported as a bad model")
        out = run(f"flipped={tmp}/flipped.parquet:-1")
        check("flipped" in out, "declaring the model's sign (:-1) fixes it")
        out = run(f"good={tmp}/good.parquet", f"partial={tmp}/partial.parquet")
        check(f"rows scored by all models: {n // 2}" in out,
              "scoring uses only the rows EVERY model scored (a model that skipped hard rows cannot look better)")


TESTS = [
    test_items_are_seed_major_and_paired,
    test_scorer_guards_sign_and_common_rows,
    test_shards_partition_the_grid,
    test_input_discovery_flattens_kaggle_folders,
    test_resume_restores_finished_items,
    test_plan_reprojects_and_cuts_in_order,
]


if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
