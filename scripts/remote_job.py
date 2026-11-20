"""
scripts/remote_job.py
=====================
Resumable experiment runner for Kaggle and Google Colab.

One manifest (``configs/experiments/*.yaml``) lists the items of a job: encode
the proteins, train arm A at seed s, score it on the external table, run the
baselines, aggregate. Each item is a subprocess of an existing pipeline script;
this file adds only what a remote VM needs around them
(``~/.claude/kaggle-colab-lessons.md``):

  * every finished item is a pack copied to persistent storage (Drive or
    /kaggle/working) together with a DONE marker, so a restarted VM restores
    finished packs and continues with the next item;
  * a heartbeat file (job stamp + time) and a live copy of the running item's
    log every ~10 minutes;
  * a report.json with the facts needed later (torch / CUDA / transformers
    versions, GPU, git commit, items run, measured seconds per step);
  * inputs are found BY NAME anywhere under the mount roots, because Kaggle
    extracts uploads (X.parquet may arrive as a folder X.parquet/ holding
    X.parquet);
  * each stage checks its inputs before it starts and fails fast;
  * a cost plan printed before launch and re-projected from measured steady
    state, with a pre-agreed order in which arms are cut if over budget.

    python -m scripts.remote_job --manifest configs/experiments/colab_g4_benchmark.yaml --plan
    python -m scripts.remote_job --manifest ... --persist /content/drive/MyDrive/holo_geon/run1
    python -m scripts.remote_job --manifest ... --shard 0/2      # Kaggle: one of two sessions

``--plan`` needs neither torch nor the data and is safe to run locally.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED_INPUTS = ("megascale_ddg_pairs.parquet", "splits_megascale.parquet",
                   "fireprotdb_holdout.parquet")
HEARTBEAT_S = 60
LIVE_COPY_S = 600


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Manifest -> items
# ---------------------------------------------------------------------------
def expand_items(manifest: dict) -> list[dict]:
    """
    Expand the manifest into a flat, ordered item list.

    Training specs carry ``arm``, ``seeds`` and ``flags``. Order is seed-major
    (every arm at seed 42, then every arm at seed 43, ...), so an interrupted or
    budget-cut job still holds COMPLETE paired observations for the seeds it
    reached instead of five seeds of one arm and none of another.
    """
    items: list[dict] = []
    train_specs = [s for s in manifest.get("items", []) if s.get("kind") == "train"]
    seeds: list[int] = []
    for spec in train_specs:
        for s in spec.get("seeds", []):
            if s not in seeds:
                seeds.append(s)

    for spec in manifest.get("items", []):
        if spec.get("kind") == "train":
            continue
        if spec.get("phase", "pre") == "pre":
            items.append(_plain_item(spec))

    for seed in seeds:
        for spec in train_specs:
            if seed not in spec.get("seeds", []):
                continue
            tid = f"{spec['arm']}_s{seed}"
            items.append({
                "id": tid, "kind": "train", "arm": spec["arm"], "seed": seed,
                "flags": list(spec.get("flags", [])), "epochs": spec.get("epochs"),
                "strict": bool(spec.get("strict", False)), "pack": tid,
                "needs": ["precompute"] if _has_id(manifest, "precompute") else [],
            })
            if spec.get("export", False):
                items.append({
                    "id": f"exp_{tid}", "kind": "export", "arm": spec["arm"],
                    "seed": seed, "needs": [tid], "pack": tid,
                })
            if spec.get("external", True):
                items.append({
                    "id": f"ext_{tid}", "kind": "external", "arm": spec["arm"],
                    "seed": seed, "needs": [tid], "pack": tid,
                })

    for spec in manifest.get("items", []):
        if spec.get("kind") != "train" and spec.get("phase", "pre") == "post":
            items.append(_plain_item(spec))
    ids = [i["id"] for i in items]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"manifest has duplicate item ids: {sorted(dup)}")
    return items


def _has_id(manifest: dict, item_id: str) -> bool:
    return any(s.get("id") == item_id for s in manifest.get("items", []))


def _plain_item(spec: dict) -> dict:
    item = dict(spec)
    item.setdefault("id", spec["kind"])
    item.setdefault("pack", item["id"])
    item.setdefault("needs", [])
    item.pop("phase", None)
    return item


def take_shard(items: list[dict], spec: str | None) -> list[dict]:
    """
    ``K/N`` keeps whole **seed blocks**: every arm trained at a given seed goes
    to the same shard, so a shard that finishes yields a complete set of paired
    observations (one value per arm at that seed) and ``scripts/ablations.py``
    can run its paired test on it alone.

    Round-robin over seeds, not over runs. A round-robin over runs (the first
    version of this function) spread the arms of one seed across every shard,
    so a single lost session left no seed complete and no arm comparable -
    exactly the failure a shard is supposed to contain.

    Shared items (precompute, baselines) are kept by every shard because each
    session has its own VM and cache; ``aggregate`` is dropped from a shard and
    run once over the merged packs. An external item follows its train item.

    A shard cannot split a seed, so ``N`` may not exceed the number of distinct
    seeds. To fit a slower GPU, cut epochs or seeds instead of adding shards.
    """
    if not spec:
        return items
    k, n = (int(x) for x in spec.split("/"))
    if not (0 <= k < n):
        raise SystemExit(f"--shard {spec}: need 0 <= K < N")

    seed_order: list = []
    for it in items:
        if it["kind"] == "train" and it.get("seed") not in seed_order:
            seed_order.append(it.get("seed"))
    if seed_order and n > len(seed_order):
        raise SystemExit(
            f"--shard {spec}: {n} shards but only {len(seed_order)} seeds "
            f"({seed_order}); a shard holds whole seeds so that each one yields "
            f"a complete paired block. Use at most {len(seed_order)} shards and "
            f"cut epochs or seeds if a shard is still too long."
        )
    my_seeds = {s for j, s in enumerate(seed_order) if j % n == k}
    mine = {i["id"] for i in items
            if i["kind"] == "train" and i.get("seed") in my_seeds}
    keep = []
    for it in items:
        if it["kind"] == "train" and it["id"] not in mine:
            continue
        if it["kind"] in ("external", "export") and it["needs"][0] not in mine:
            continue
        if it["kind"] == "aggregate" and n > 1:
            continue   # aggregation runs once, after the shards are merged
        keep.append(it)
    return keep


# ---------------------------------------------------------------------------
# Cost plan
# ---------------------------------------------------------------------------
def est_hours(item: dict, sz: dict, sec_per_step: float) -> float:
    kind = item["kind"]
    if kind == "train":
        epochs = item.get("epochs") or sz["epochs"]
        hours = sz["steps_per_epoch"] * epochs * sec_per_step \
            * (1.0 + sz.get("eval_overhead", 0.10)) / 3600.0
        if item.get("strict"):
            hours *= sz.get("strict_factor", 1.3)
        return hours
    if "est_hours" in item:
        return float(item["est_hours"])
    return float(sz.get(f"{kind}_h", 0.0 if kind == "aggregate" else 0.05))


def measured_sec_per_step(done_dir: str) -> float | None:
    vals = []
    if os.path.isdir(done_dir):
        for name in os.listdir(done_dir):
            d = _read_json(os.path.join(done_dir, name))
            if d and d.get("sec_per_step"):
                vals.append(float(d["sec_per_step"]))
    return statistics.median(vals) if vals else None


def print_plan(manifest: dict, items: list[dict], sec_per_step: float,
               source: str) -> float:
    sz = manifest["sizing"]
    cost = manifest.get("cost", {})
    rate = float(cost.get("cu_per_hour", 0.0))
    budget = cost.get("budget_cu")
    reserve = float(cost.get("reserve_cu", 0.0))
    overhead_h = float(sz.get("vm_overhead_h", 0.0))

    print(f"\nPLAN  {manifest.get('name')}  ({manifest.get('platform')})")
    print(f"seconds per step: {sec_per_step * 1000:.1f} ms  [{source}]")
    if source.startswith("ASSUMED"):
        print("  ^ NOT MEASURED on the target GPU. Run the first train item, then re-run "
              "--plan: it re-projects from steady state.")
    print(f"{'item':<30}{'kind':<11}{'hours':>7}{'cum h':>8}{'cum CU':>8}")
    cum = overhead_h
    for it in items:
        h = est_hours(it, sz, sec_per_step)
        cum += h
        print(f"{it['id']:<30}{it['kind']:<11}{h:>7.2f}{cum:>8.2f}{cum * rate:>8.1f}")
    print(f"{'(VM start-up, installs, idle)':<41}{overhead_h:>7.2f}")
    total_cu = cum * rate
    print(f"TOTAL {cum:.2f} h" + (f"  =  {total_cu:.1f} CU" if rate else ""))
    if budget is not None and rate:
        usable = float(budget) - reserve
        print(f"budget {budget} CU, reserve {reserve:g} CU -> usable {usable:g} CU; "
              f"margin {usable - total_cu:+.1f} CU")
        if total_cu > usable:
            order = manifest.get("cut_order", [])
            print(f"OVER BUDGET. Pre-agreed cut order: {order or '(none declared)'}")
            kept = list(items)
            for arm in order:
                # An entry is an arm name or an item-id glob ("*_s46" also drops
                # the matching ext_* item).
                kept = [i for i in kept if not (i.get("arm") == arm
                                                or fnmatch.fnmatch(i["id"], arm))]
                h = overhead_h + sum(est_hours(i, sz, sec_per_step) for i in kept)
                print(f"  after cutting {arm:<22}: {h:6.2f} h = {h * rate:6.1f} CU")
                if h * rate <= usable:
                    break
    return cum


# ---------------------------------------------------------------------------
# Inputs, environment
# ---------------------------------------------------------------------------
def detect_platform() -> str:
    if os.path.isdir("/kaggle"):
        return "kaggle"
    if os.path.isdir("/content") or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def find_inputs(roots: list[str], names: tuple[str, ...], max_depth: int = 7) -> dict:
    """Locate files by NAME under the mount roots, flattening one-file folders."""
    found: dict[str, str] = {}
    wanted = set(names)
    for root in roots:
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
            # Kaggle: X.parquet arrives as a directory X.parquet/ holding X.parquet.
            for name in list(dirnames) + list(filenames):
                if name in wanted and name not in found:
                    path = os.path.join(dirpath, name)
                    if os.path.isdir(path):
                        inner = os.path.join(path, name)
                        if os.path.isfile(inner):
                            path = inner
                        else:
                            continue
                    found[name] = path
            if wanted <= set(found):
                return found
    return found


def stage_inputs(roots: list[str], extra_files: tuple[str, ...] = ()) -> dict:
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    extra = (
        "leakage_report_stability.json",
        "splits_megascale_id25.parquet",
        "splits_megascale_fold_tm60.parquet",
        "fireprotdb_hf25.parquet",
        "s669_holdout.parquet",
        "domainome_apca_hf25.parquet",
        "invariant_backbone_geometry.npz",
    ) + tuple(extra_files)
    all_names = tuple(dict.fromkeys(REQUIRED_INPUTS + extra))
    present = {n: os.path.join(data_dir, n) for n in all_names
               if os.path.isfile(os.path.join(data_dir, n))}
    missing = [n for n in all_names if n not in present]
    if missing:
        located = find_inputs(roots, tuple(missing))
        for name, src in located.items():
            dst = os.path.join(data_dir, name)
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            present[name] = dst
    absent = [n for n in REQUIRED_INPUTS if n not in present]
    if absent:
        raise SystemExit(
            f"[remote_job] required input(s) not found under {roots}: {absent}. "
            f"Mount/attach the data first; nothing was started.")
    return present


def env_report(device: str) -> dict:
    code = (
        "import json,sys,platform\n"
        "o={'python':sys.version.split()[0],'platform':platform.platform()}\n"
        "def v(m):\n"
        "    try:\n"
        "        x=__import__(m); return getattr(x,'__version__','?')\n"
        "    except Exception as e:\n"
        "        return 'MISSING: '+type(e).__name__\n"
        "for m in ('torch','torch_geometric','transformers','numpy','pandas','scipy','sklearn','yaml','RNA'):\n"
        "    o[m]=v(m)\n"
        "try:\n"
        "    import torch\n"
        "    o['cuda_available']=bool(torch.cuda.is_available())\n"
        "    o['torch_cuda']=torch.version.cuda\n"
        "    if torch.cuda.is_available():\n"
        "        o['gpu']=torch.cuda.get_device_name(0)\n"
        "        o['capability']=list(torch.cuda.get_device_capability(0))\n"
        "        o['vram_gb']=round(torch.cuda.get_device_properties(0).total_memory/2**30,1)\n"
        "except Exception as e:\n"
        "    o['torch_error']=repr(e)\n"
        "print(json.dumps(o))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        rep = json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        rep = {"error": "environment probe failed", "stderr": out.stderr[-500:]}
    try:
        rep["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        rep["git_commit"] = None
    rep["platform_detected"] = detect_platform()
    rep["problems"] = problems = []
    tf = str(rep.get("transformers", ""))
    if tf.startswith("5"):
        problems.append("transformers 5 breaks ESM contact extraction; need >=4.40,<5")
    if "MISSING" in str(rep.get("torch_geometric", "")):
        problems.append("torch_geometric is required (no non-graph fallback)")
    if "MISSING" in str(rep.get("RNA", "")):
        problems.append("ViennaRNA missing: E_mech channels stay masked (mech_valid=0)")
    if device == "cuda":
        if not rep.get("cuda_available"):
            problems.append("device=cuda but CUDA is not available")
        elif rep.get("capability", [9])[0] < 7:
            problems.append("GPU older than sm_70 (P100?): new torch builds lack sm_60")
    return rep


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class Store:
    """Local out_root plus optional persistent mirror."""

    def __init__(self, out_root: str, persist: str | None, cache: str):
        self.out_root, self.persist, self.cache = out_root, persist, cache
        self.done_dir = os.path.join(out_root, "_done")
        os.makedirs(self.done_dir, exist_ok=True)

    def pack_path(self, item: dict) -> str:
        return self.cache if item["kind"] == "precompute" else os.path.join(self.out_root, item["pack"])

    def _persist_pack(self, item: dict) -> str | None:
        if not self.persist:
            return None
        return os.path.join(self.persist, "packs", "cache" if item["kind"] == "precompute" else item["pack"])

    def is_done(self, item_id: str) -> bool:
        return os.path.isfile(os.path.join(self.done_dir, item_id + ".json"))

    def restore(self, items: list[dict]) -> list[str]:
        """Copy finished packs and DONE markers back from persistent storage."""
        restored = []
        if not self.persist:
            return restored
        src_done = os.path.join(self.persist, "_done")
        if not os.path.isdir(src_done):
            return restored
        by_id = {i["id"]: i for i in items}
        for name in sorted(os.listdir(src_done)):
            item_id = name[:-5]
            if item_id not in by_id or self.is_done(item_id):
                continue
            item = by_id[item_id]
            src = self._persist_pack(item)
            if src and os.path.isdir(src):
                shutil.copytree(src, self.pack_path(item), dirs_exist_ok=True)
            shutil.copy2(os.path.join(src_done, name), os.path.join(self.done_dir, name))
            restored.append(item_id)
        return restored

    def save(self, item: dict, done: dict) -> None:
        _write_json(os.path.join(self.done_dir, item["id"] + ".json"), done)
        dst = self._persist_pack(item)
        if not dst:
            return
        src = self.pack_path(item)
        if os.path.isdir(src):
            # last.pt is resume state for an unfinished run, not a result.
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("last.pt", "*.tmp"))
        _write_json(os.path.join(self.persist, "_done", item["id"] + ".json"), done)

    def mirror_file(self, local: str, rel: str) -> None:
        if self.persist and os.path.isfile(local):
            dst = os.path.join(self.persist, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(local, dst)


class Heartbeat(threading.Thread):
    def __init__(self, store: Store, job: str):
        super().__init__(daemon=True)
        self.store, self.job = store, job
        self.state = {"item": None, "log": None}
        # Not `_stop`: threading.Thread owns that name internally.
        self._halt = threading.Event()
        self._last_live = 0.0

    def run(self) -> None:
        while not self._halt.wait(HEARTBEAT_S):
            self.beat()

    def beat(self) -> None:
        now = time.time()
        beat = {"job": self.job, "stamp": utc(), "epoch_s": now,
                "item": self.state["item"]}
        local = os.path.join(self.store.out_root, "heartbeat.json")
        _write_json(local, beat)
        self.store.mirror_file(local, "heartbeat.json")
        log = self.state["log"]
        if log and now - self._last_live >= LIVE_COPY_S:
            self.store.mirror_file(log, f"live/{self.state['item']}.log")
            self._last_live = now

    def stop(self) -> None:
        self._halt.set()


# ---------------------------------------------------------------------------
# Item execution
# ---------------------------------------------------------------------------
def build_cmd(item: dict, ctx: dict) -> list[str]:
    py = sys.executable
    cfg, cache, dev = ctx["config"], ctx["cache"], ctx["device"]
    splits_file = ctx.get("splits_file", "data/splits_megascale.parquet")
    external_file = ctx.get("external_file", "data/fireprotdb_holdout.parquet")
    out = os.path.join(ctx["out_root"], item["pack"])
    kind = item["kind"]
    if kind == "precompute":
        data_args = ["data/megascale_ddg_pairs.parquet", external_file]
        for opt_ext in ("data/fireprotdb_holdout.parquet", "data/s669_holdout.parquet"):
            if opt_ext not in data_args and os.path.isfile(os.path.join(ROOT, opt_ext)):
                data_args.append(opt_ext)
        return [py, "-m", "scripts.precompute_embeddings", "--data", *data_args,
                "--config", cfg, "--cache", cache, "--device", dev]
    if kind == "train":
        cmd = [py, "-m", "scripts.train", "--config", cfg,
               "--stability-splits", splits_file,
               # Benchmark is DDG only; a missing path drops the second task.
               "--pathogenicity-splits", "none",
               "--cache", cache, "--device", dev, "--out", out,
               "--tag", item["id"], "--seed", str(item["seed"])]
        if ctx.get("num_workers"):
            cmd += ["--num-workers", str(ctx["num_workers"])]
        if item.get("epochs"):
            cmd += ["--epochs", str(item["epochs"])]
        if item.get("strict"):
            cmd.append("--strict-determinism")
        return cmd + list(item.get("flags", []))
    if kind == "export":
        return [py, "-m", "scripts.export_predictions",
                "--checkpoint", os.path.join(out, "model_best.pt"),
                "--splits", splits_file, "--split", "test",
                "--config", cfg, "--cache", cache, "--device", dev,
                "--out", os.path.join(out, "pred_test.parquet")]
    if kind == "external":
        return [py, "-m", "scripts.eval_external",
                "--checkpoint", os.path.join(out, "model_best.pt"),
                "--data", external_file, "--config", cfg,
                "--cache", cache, "--device", dev, "--out", out]
    if kind == "baselines":
        return [py, "-m", "scripts.baselines", "--splits",
                splits_file, "--out", out, "--config", cfg,
                "--cache", cache, "--device", dev] + list(item.get("flags", []))
    if kind == "cmd":
        sub = {"out": out, "cache": cache, "cfg": cfg, "device": dev, "root": ROOT,
               "out_root": ctx["out_root"], "py": py,
               "splits": splits_file, "external": external_file}
        return [str(a).format(**sub) for a in item["argv"]]
    raise SystemExit(f"unknown item kind {kind!r} (item {item['id']})")


def preflight(item: dict, ctx: dict, store: Store) -> None:
    """Fail fast: every input the stage needs must exist BEFORE it starts."""
    need: list[str] = []
    kind = item["kind"]
    if kind in ("precompute", "baselines", "train", "external", "export"):
        need += [os.path.join(ROOT, "data", n) for n in REQUIRED_INPUTS]
    if kind in ("train", "external", "export", "baselines"):
        if not (os.path.isdir(ctx["cache"]) and os.listdir(ctx["cache"])):
            need.append(ctx["cache"] + "  (encoder cache is empty: precompute not done)")
    if kind in ("external", "export"):
        need.append(os.path.join(store.pack_path(item), "model_best.pt"))
    for dep in item.get("needs", []):
        if not store.is_done(dep):
            need.append(f"dependency item {dep!r} (not finished)")
    # Entries that are already a message (not a path) are failures by construction.
    bad = [p for p in need if "  (" in p or "dependency item" in p or not os.path.exists(p)]
    if bad:
        raise RuntimeError(f"{item['id']}: missing input(s): {bad}")


def run_item(item: dict, ctx: dict, store: Store, hb: Heartbeat) -> dict:
    out = store.pack_path(item)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(ctx["out_root"], "logs", item["id"] + ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = build_cmd(item, ctx)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    hb.state.update(item=item["id"], log=log_path)
    hb.beat()
    print(f"\n[remote_job] === {item['id']} ({item['kind']}) ===\n$ {' '.join(cmd)}", flush=True)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# {utc()}  $ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line.replace("\r", "\n"))
        rc = proc.wait()
    seconds = time.time() - started
    rec = {"id": item["id"], "kind": item["kind"], "arm": item.get("arm"),
           "seed": item.get("seed"), "returncode": rc, "seconds": round(seconds, 1),
           "finished_utc": utc()}
    if rc == 0 and item["kind"] == "train":
        rec.update(train_facts(os.path.join(out, "metrics.json"), ctx, seconds))
    hb.state.update(item=None, log=None)
    return rec


def train_facts(metrics_path: str, ctx: dict, seconds: float) -> dict:
    m = _read_json(metrics_path) or {}
    epochs_run = len(m.get("history", []))
    facts = {"epochs_run": epochs_run, "best_epoch": m.get("best_epoch"),
             "wall_clock_hours": m.get("wall_clock_hours")}
    spe = ctx.get("steps_per_epoch")
    if epochs_run and spe:
        wall = (m.get("wall_clock_hours") or 0) * 3600 or seconds
        facts["sec_per_step"] = wall / (epochs_run * spe)
    test = (m.get("test") or {}).get("stability") or {}
    facts["test_pearson"] = test.get("pearson")
    facts["test_rmse"] = test.get("rmse")
    if facts["test_pearson"] is None or facts["test_pearson"] != facts["test_pearson"]:
        facts["warning"] = "test pearson missing or NaN: no finite-validation checkpoint"
    return facts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _read_external_result(pack: str, ctx: dict) -> dict | None:
    """The external-eval JSON of one pack, for the manifest's external table.

    scripts/eval_external writes ``external_{key}.json`` (key = the data file's
    stem). Packs made before that hold only ``external_fireprotdb.json``; it is
    read as a fallback for a ``fireprotdb*`` table only, so a pack never has
    another table's numbers passed off as this one's.
    """
    ext_file = ctx.get("external_file", "data/fireprotdb_holdout.parquet")
    key = os.path.splitext(os.path.basename(ext_file))[0]
    found = _read_json(os.path.join(pack, f"external_{key}.json"))
    if found is None and key.startswith("fireprotdb"):
        found = _read_json(os.path.join(pack, "external_fireprotdb.json"))
    return found


def aggregate(manifest: dict, items: list[dict], ctx: dict) -> dict:
    rows: dict[str, dict[int, dict]] = {}
    for it in items:
        if it["kind"] != "train":
            continue
        pack = os.path.join(ctx["out_root"], it["pack"])
        m = _read_json(os.path.join(pack, "metrics.json"))
        if not m:
            continue
        test = (m.get("test") or {}).get("stability") or {}
        ext = (_read_external_result(pack, ctx) or {}).get("metrics") or {}
        rows.setdefault(it["arm"], {})[it["seed"]] = {
            "test_pearson": test.get("pearson"), "test_rmse": test.get("rmse"),
            "ext_pearson": ext.get("pearson"), "ext_rmse": ext.get("rmse"),
            "best_epoch": m.get("best_epoch"),
        }

    def col(arm: str, key: str) -> dict[int, float]:
        return {s: float(r[key]) for s, r in rows.get(arm, {}).items()
                if isinstance(r.get(key), (int, float)) and r[key] == r[key]}

    summary: dict = {"generated_utc": utc(), "arms": {}, "paired_vs_full": {}}
    for arm in rows:
        entry = {"n_seeds": len(rows[arm]), "per_seed": rows[arm]}
        for key in ("test_pearson", "test_rmse", "ext_pearson", "ext_rmse"):
            vals = list(col(arm, key).values())
            entry[key] = {"mean": statistics.mean(vals) if vals else None,
                          "sd": statistics.stdev(vals) if len(vals) > 1 else None}
        summary["arms"][arm] = entry

    try:
        from scripts.ablations import holm_bonferroni, paired_stats
    except Exception as exc:   # keep the table even if scipy/torch are unavailable
        summary["paired_vs_full"] = {"unavailable": repr(exc)}
        paired_stats = holm_bonferroni = None
    if paired_stats and "full" in rows:
        p_values = {}
        for arm in rows:
            if arm == "full":
                continue
            for key in ("test_pearson", "ext_pearson"):
                a, f = col(arm, key), col("full", key)
                shared = sorted(set(a) & set(f))
                if not shared:
                    continue
                stats = paired_stats([a[s] for s in shared], [f[s] for s in shared])
                summary["paired_vs_full"].setdefault(arm, {})[key] = stats
                p_values[f"{arm}/{key}"] = stats.get("p_value")
        adj = holm_bonferroni(p_values)
        for k, p in adj.items():
            arm, key = k.split("/")
            summary["paired_vs_full"][arm][key]["p_value_holm"] = p
            summary["paired_vs_full"][arm][key]["family_size"] = len(
                [v for v in p_values.values() if v is not None])
    b = _read_json(os.path.join(ctx["out_root"], "baselines", "baselines.json"))
    if b:
        summary["baselines"] = b
    return summary


def write_summary(summary: dict, out_root: str, store: Store) -> None:
    path = os.path.join(out_root, "summary.json")
    _write_json(path, summary)
    lines = ["| arm | seeds | test r | test RMSE | external r | external RMSE |",
             "|---|---|---|---|---|---|"]

    def fmt(d):
        if not d or d["mean"] is None:
            return "n/a"
        return f"{d['mean']:.4f}" + (f" ± {d['sd']:.4f}" if d["sd"] is not None else "")
    for arm, e in summary["arms"].items():
        lines.append(f"| {arm} | {e['n_seeds']} | {fmt(e['test_pearson'])} | "
                     f"{fmt(e['test_rmse'])} | {fmt(e['ext_pearson'])} | {fmt(e['ext_rmse'])} |")
    md = os.path.join(out_root, "summary.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    store.mirror_file(path, "summary.json")
    store.mirror_file(md, "summary.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--plan", action="store_true", help="print the cost plan and exit")
    ap.add_argument("--sec-per-step", type=float, default=None,
                    help="override the seconds-per-step used by --plan")
    ap.add_argument("--shard", default=None, help="K/N: this session's share of train items")
    ap.add_argument("--persist", default=None, help="persistent dir (Drive / /kaggle/working)")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", nargs="*", default=None,
                    help="mount roots searched by file name")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop launching items that would not finish inside this wall-clock")
    ap.add_argument("--only", nargs="*", default=None, help="run only these item ids (glob ok)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="restore finished packs from --persist and write summary.*")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    items = take_shard(expand_items(manifest), args.shard)
    if args.only:
        items = [i for i in items if any(fnmatch.fnmatch(i["id"], p) for p in args.only)]
    sz = manifest["sizing"]

    plat = detect_platform()
    out_root = args.out_root or manifest.get("out_root") or os.path.join(ROOT, "runs", manifest["name"])
    cache = args.cache or manifest.get("cache") or "/tmp/holo/cache/esm2_650M"
    done_dir = os.path.join(out_root, "_done")
    measured = measured_sec_per_step(done_dir)
    if args.persist and not measured:
        measured = measured_sec_per_step(os.path.join(args.persist, "_done"))
    if args.sec_per_step:
        sps, src = args.sec_per_step, "from --sec-per-step"
    elif measured:
        sps, src = measured, "MEASURED steady state on this GPU (median of finished train items)"
    else:
        sps, src = float(sz["sec_per_step"]), "ASSUMED"

    if args.plan:
        print_plan(manifest, items, sps, src)
        return

    ctx = {"config": manifest["config"], "cache": cache, "device": args.device,
           "out_root": out_root, "steps_per_epoch": sz["steps_per_epoch"],
           "num_workers": args.num_workers or manifest.get("num_workers"),
           "splits_file": manifest.get("splits_file", "data/splits_megascale.parquet"),
           "external_file": manifest.get("external_file", "data/fireprotdb_holdout.parquet")}
    os.makedirs(out_root, exist_ok=True)
    store = Store(out_root, args.persist, cache)
    job = f"{manifest['name']}:{args.shard or 'all'}:{int(time.time())}"

    if args.aggregate_only:
        # Merge step for sharded sessions: copy every shard's packs into one
        # persist dir first, then point --persist at it.
        full = expand_items(manifest)
        restored = store.restore(full)
        print(f"[remote_job] restored {len(restored)} finished item(s) from {args.persist}")
        summary = aggregate(manifest, full, ctx)
        write_summary(summary, out_root, store)
        print(open(os.path.join(out_root, "summary.md"), encoding="utf-8").read())
        return

    roots = args.data_root or manifest.get("data_roots") or (
        ["/kaggle/input"] if plat == "kaggle"
        else ["/content/drive/MyDrive/holo_geon/data", "/content/data"] if plat == "colab"
        else [os.path.join(ROOT, "DATASETS")])
    stage_inputs(roots)

    report = env_report(args.device)
    report.update(job=job, manifest=manifest["name"], shard=args.shard, started_utc=utc(),
                  cache=cache, out_root=out_root, assumed_or_measured_sec_per_step=[sps, src])
    _write_json(os.path.join(out_root, "report.json"), report)
    store.mirror_file(os.path.join(out_root, "report.json"), "report.json")
    print(json.dumps(report, indent=2))
    if report["problems"]:
        hard = [p for p in report["problems"] if "ViennaRNA" not in p]
        if hard:
            raise SystemExit(f"[remote_job] environment problems, nothing started: {hard}")
        print(f"[remote_job] WARNING: {report['problems']}")

    restored = store.restore(items)
    if restored:
        print(f"[remote_job] restored finished items from {args.persist}: {restored}")

    t0 = time.time()
    hb = Heartbeat(store, job)
    hb.start()
    log_path = os.path.join(out_root, "items.jsonl")
    failures: list[str] = []
    for it in items:
        if it["kind"] == "aggregate":
            continue
        if store.is_done(it["id"]):
            print(f"[remote_job] {it['id']}: already done, skipping")
            continue
        est = est_hours(it, sz, sps)
        if args.max_hours and (time.time() - t0) / 3600.0 + est > args.max_hours:
            print(f"[remote_job] {it['id']}: est {est:.2f} h does not fit inside "
                  f"--max-hours {args.max_hours}; stopping cleanly (resumable)")
            break
        try:
            preflight(it, ctx, store)
        except RuntimeError as exc:
            print(f"[remote_job] BLOCKED {exc}")
            failures.append(it["id"])
            if args.fail_fast:
                break
            continue
        rec = run_item(it, ctx, store, hb)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        store.mirror_file(log_path, "items.jsonl")
        if rec["returncode"] == 0:
            store.save(it, rec)
            sps_new = measured_sec_per_step(done_dir)
            if sps_new:
                sps = sps_new
        else:
            failures.append(it["id"])
            print(f"[remote_job] {it['id']} FAILED (exit {rec['returncode']}); see "
                  f"{os.path.join(out_root, 'logs', it['id'] + '.log')}")
            store.mirror_file(os.path.join(out_root, "logs", it["id"] + ".log"),
                              f"failed/{it['id']}.log")
            if args.fail_fast:
                break

    if any(i["kind"] == "aggregate" for i in items) or not args.shard:
        summary = aggregate(manifest, items, ctx)
        write_summary(summary, out_root, store)
        print("\n" + open(os.path.join(out_root, "summary.md"), encoding="utf-8").read())
    hb.stop()
    report.update(finished_utc=utc(), failures=failures,
                  elapsed_h=round((time.time() - t0) / 3600.0, 3),
                  sec_per_step_final=sps, python=platform.python_version())
    _write_json(os.path.join(out_root, "report.json"), report)
    store.mirror_file(os.path.join(out_root, "report.json"), "report.json")
    print(f"[remote_job] finished; failures: {failures or 'none'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
