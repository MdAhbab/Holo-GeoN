"""
scripts/fetch_alphafold.py
==========================
Fetch one AlphaFold DB structure per FireProtDB holdout protein. This is the
**only network step** in the benchmark pipeline, and it exists because the
structure-input competitors (ThermoMPNN, SPURS, RaSP) cannot be scored on the
external set without coordinates, while Holo-GeoN itself needs none.

    python -m scripts.fetch_alphafold \
        --holdout data/fireprotdb_holdout.parquet --out data/alphafold_fireprot

Run it on a laptop, not on a paid GPU VM: 176 small files over HTTP, no GPU and
no large local load. Copy the finished folder to Drive (or into the Kaggle
dataset) along with the parquet files.

Design follows kaggle-colab-lessons.md:

* **Resumable.** A file that is already present and parses as a PDB is skipped,
  so a rerun after a server outage fetches only what is missing.
* **Pinned.** ``manifest.json`` records accession, resolved URL, AlphaFold DB
  version, byte count and sha256 for every file, so a later rerun can be checked
  against it and the paper can state exactly which structures were used.
* **Per-item log.** ``fetch_log.jsonl`` grows as the job runs; a killed process
  leaves a readable record of what succeeded.
* **Validated.** A download is only accepted if it contains ATOM records for the
  expected accession. Downloaded files are untrusted input: nothing here
  executes or imports them, and the only parsing is a line-prefix check.
* **Fail-fast.** Exits non-zero if any accession is unresolved, unless
  ``--allow-missing N`` permits up to N (the paper must then report coverage).

The version is resolved per accession through the AlphaFold DB API rather than
hard-coding ``-model_v4.pdb``, because the file version has changed between
releases and a hard-coded one silently 404s after an update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
# Used only if the API itself is unreachable; newest first.
FALLBACK = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb"
FALLBACK_VERSIONS = (4, 3, 2)
UA = "Holo-GeoN-benchmark/1.0 (academic ddG benchmark; +https://alphafold.ebi.ac.uk)"


def _get(url: str, timeout: float, tries: int, pause: float) -> bytes:
    """GET with long patience: public EBI endpoints go down for hours."""
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise                      # a real miss, not an outage
            last = exc
        except Exception as exc:           # timeouts, resets, DNS
            last = exc
        if attempt < tries:
            wait = pause * (2 ** (attempt - 1))
            print(f"    retry {attempt}/{tries - 1} in {wait:.0f}s ({last})")
            time.sleep(wait)
    raise RuntimeError(f"{url}: {tries} attempts failed, last error: {last}")


def _is_pdb_for(blob: bytes, acc: str) -> tuple[bool, str]:
    """A downloaded blob is accepted only as a PDB holding this accession."""
    if len(blob) < 1000:
        return False, f"only {len(blob)} bytes"
    try:
        text = blob.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, "not utf-8 text"
    atoms = sum(1 for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM")))
    if atoms < 10:
        return False, f"only {atoms} ATOM records"
    if acc.upper() not in text.upper():
        return False, f"accession {acc} does not appear in the file"
    return True, f"{atoms} atoms"


def _resolve(acc: str, args) -> tuple[str, str]:
    """(url, version) for one accession, via the API, else the file pattern."""
    try:
        meta = json.loads(_get(API.format(acc=acc), args.timeout, args.tries, args.pause))
        if isinstance(meta, list) and meta:
            entry = meta[0]
            url = entry.get("pdbUrl")
            if url:
                return url, str(entry.get("latestVersion") or entry.get("modelCreatedDate") or "api")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise
        print(f"    API unavailable ({exc}); falling back to the file pattern")
    except Exception as exc:
        print(f"    API unusable ({exc}); falling back to the file pattern")
    for v in FALLBACK_VERSIONS:
        url = FALLBACK.format(acc=acc, v=v)
        try:
            urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA}),
                timeout=args.timeout,
            ).close()
            return url, f"v{v}"
        except Exception:
            continue
    raise RuntimeError(f"{acc}: neither the API nor any known file version resolved")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", default="data/fireprotdb_holdout.parquet",
                    help="parquet with a `parent` column of UniProt accessions")
    ap.add_argument("--out", default="data/alphafold_fireprot",
                    help="output directory (created; holds only downloaded files)")
    ap.add_argument("--column", default="parent")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--pause", type=float, default=10.0, help="first backoff, doubling")
    ap.add_argument("--sleep", type=float, default=0.2, help="pause between accessions")
    ap.add_argument("--allow-missing", type=int, default=0,
                    help="tolerate up to N unresolved accessions (report coverage!)")
    ap.add_argument("--limit", type=int, default=None, help="first N accessions only (smoke test)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.holdout):
        print(f"holdout table not found: {args.holdout}", file=sys.stderr)
        return 2
    accs = sorted(pd.read_parquet(args.holdout)[args.column].astype(str).unique())
    if args.limit:
        accs = accs[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "fetch_log.jsonl")

    manifest: dict[str, dict] = {}
    man_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as fh:
            manifest = json.load(fh).get("structures", {})

    print(f"{len(accs)} accessions -> {args.out}")
    fetched = skipped = 0
    missing: list[str] = []

    for i, acc in enumerate(accs, 1):
        dest = os.path.join(args.out, f"AF-{acc}.pdb")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            with open(dest, "rb") as fh:
                ok, _ = _is_pdb_for(fh.read(), acc)
            if ok:
                skipped += 1
                continue
            print(f"[{i}/{len(accs)}] {acc}: present but invalid, refetching")

        print(f"[{i}/{len(accs)}] {acc}")
        record = {"accession": acc, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            url, version = _resolve(acc, args)
            blob = _get(url, args.timeout, args.tries, args.pause)
            ok, why = _is_pdb_for(blob, acc)
            if not ok:
                raise RuntimeError(f"rejected the download: {why}")
            with open(dest, "wb") as fh:
                fh.write(blob)
            digest = hashlib.sha256(blob).hexdigest()
            record.update(status="ok", url=url, version=version, bytes=len(blob),
                          sha256=digest, note=why)
            manifest[acc] = {"url": url, "version": version, "bytes": len(blob),
                             "sha256": digest, "file": os.path.basename(dest)}
            fetched += 1
        except Exception as exc:
            record.update(status="missing", error=f"{type(exc).__name__}: {exc}")
            missing.append(acc)
            print(f"    MISSING {acc}: {exc}")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        time.sleep(args.sleep)

    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump({"holdout": args.holdout, "column": args.column,
                   "accessions": len(accs), "resolved": len(manifest),
                   "missing": missing, "structures": manifest}, fh, indent=2)

    print(f"\nfetched {fetched}, already present {skipped}, missing {len(missing)}")
    print(f"coverage {len(manifest)}/{len(accs)} = {100.0 * len(manifest) / max(len(accs), 1):.1f}%")
    print(f"manifest: {man_path}")
    if missing:
        print("missing accessions: " + ", ".join(missing))
        if len(missing) > args.allow_missing:
            print(f"\n{len(missing)} missing exceeds --allow-missing {args.allow_missing}.",
                  file=sys.stderr)
            print("Rerun to retry only the missing ones, or raise the limit and "
                  "report structure coverage in the paper.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
