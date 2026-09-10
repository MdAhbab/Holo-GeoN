"""
scripts/build_fireprot_table.py
===============================
Build ``data/fireprotdb_holdout.parquet`` — an INDEPENDENT stability test set of
real, full-length proteins — from the FireProtDB export.

Why a separate table rather than more training data
---------------------------------------------------
FireProtDB's 613,208 rows carrying a ddG are **98.6% MegaScale**: 604,920 of them
come from the same Tsuboyama release ``data/megascale_ddg_pairs.parquet`` is
built from. Merging the export into training would duplicate the training set
and put the same measurements on both sides of a split, which is the leakage the
whole splitting machinery exists to prevent.

What is left after removing them is the valuable part, and it is valuable
precisely because it is *different*: ~5.4k unique mutations over 171 proteins
from 72 organisms, median length 164 rather than 56, measured by classical
biophysics rather than by proteolysis. The shipped model's training set is 191
identity clusters of 32-72-residue designed and natural domains; this is the set
that says whether anything learned there transfers.

The sign, and how it was established
------------------------------------
This table is NOT in the same convention as the training table, and nothing in
the file says so.

* Rows sourced from MegaScale match ``megascale_ddg_pairs.parquet`` at
  r = +0.9995 over 346,489 shared (PDB, substitution) pairs, with 100% sign
  agreement — so those follow the internal convention, positive = stabilising.
* The **non-MegaScale rows are inverted**. Of the mutations measured by both
  sources, the two disagree: r = -0.77 over 51 shared mutations, with sign
  agreement of 0.255. Independently, the non-MegaScale mean ddG is **+0.864**,
  and a set of classical folding mutations whose mean is strongly *stabilising*
  is not a credible dataset.

So this table is written with ``ddg_sign: destabilizing_positive`` declared under
its key in ``configs/data.yaml``, and ``src/data.load_mutation_table`` performs
the flip and records it in the load report. The evidence is written down here
because Pearson r and Spearman rho are invariant to a global sign flip: had this
gone unnoticed, the external validation would have reported an excellent
correlation while every qualitative claim it supported was backwards.

Independence, and the check that enforces it
--------------------------------------------
Removing the MegaScale-sourced ROWS makes this table share no *measurement* with
the training table. It does NOT make it independent at the sequence level: a
MegaScale construct is a 32-72-residue domain, and several FireProt entries are
the full-length proteins those domains were cut from, so a trained construct can
sit inside a "held-out" protein as a literal substring. On the shipped tables
that was 6 of 182 proteins and 302 of 4535 rows, 162 of which mutate a residue
that was inside the trained construct - and none contained a held-out construct,
so the bias ran one way, towards inflating the external score. Those proteins are
now dropped, against the training split, and the count is reported.

Sequences
---------
The export carries ``SEQUENCE_ID`` (an internal integer) and ``SEQUENCE_LENGTH``
but no sequence, so the protein sequences are fetched once from UniProt by
accession and cached in ``dataset/fireprotdb/uniprot_sequences.fasta``. Every row
is then checked: the wild-type residue named in ``SUBSTITUTION`` must equal the
residue at that position in the fetched sequence. Rows failing that are dropped
and counted — the same check that makes the ClinVar table trustworthy.

Usage
-----
python -m scripts.build_fireprot_table
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

FIREPROT_DIR = os.path.join("dataset", "fireprotdb")
DEFAULT_SOURCE = os.path.join(FIREPROT_DIR, "fireprotdb_20251015-164116.csv")
SEQ_CACHE = os.path.join(FIREPROT_DIR, "uniprot_sequences.fasta")

SUB_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

#: A UniProtKB accession. Some export cells list several ("P0A3D9, P0A3E0");
#: the first is taken. Anything not matching is dropped rather than sent to the
#: API, which is what turned one malformed cell into a failed request for the
#: whole 100-accession batch it happened to sit in - silently costing half the
#: table.
ACC_RE = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")


def normalise_accession(value) -> str | None:
    m = ACC_RE.match(str(value).strip().split(",")[0].strip())
    return m.group(0) if m else None


def fetch_uniprot(accessions: list[str], cache: str, batch: int = 100) -> dict[str, str]:
    """Fetch sequences by accession, caching the FASTA so this runs once."""
    seqs: dict[str, str] = {}
    if os.path.exists(cache):
        acc, buf = None, []
        with open(cache, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(">"):
                    if acc:
                        seqs[acc] = "".join(buf)
                    # >sp|P00698|LYSC_CHICK ...
                    parts = line[1:].split("|")
                    acc = parts[1] if len(parts) > 2 else line[1:].split()[0]
                    buf = []
                else:
                    buf.append(line.strip())
        if acc:
            seqs[acc] = "".join(buf)
        print(f"[fireprot] {len(seqs)} sequences from cache {cache}")

    missing = [a for a in accessions if a not in seqs]
    if not missing:
        return seqs

    print(f"[fireprot] fetching {len(missing)} sequences from UniProt")
    fetched: list[str] = []
    for i in range(0, len(missing), batch):
        chunk = missing[i:i + batch]
        query = "+OR+".join(f"accession:{a}" for a in chunk)
        url = (f"https://rest.uniprot.org/uniprotkb/stream?query={query}"
               f"&format=fasta&includeIsoform=false")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=180) as resp:
                    fetched.append(resp.read().decode("utf-8"))
                break
            except Exception as exc:                       # network hiccup
                if attempt == 3:
                    print(f"[fireprot] giving up on batch {i}: {exc}")
                else:
                    time.sleep(2 * (attempt + 1))
        print(f"  {min(i + batch, len(missing))}/{len(missing)}", flush=True)

    text = "".join(fetched)
    if text:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "a", encoding="utf-8") as fh:
            fh.write(text)
        acc, buf = None, []
        for line in text.splitlines():
            if line.startswith(">"):
                if acc:
                    seqs.setdefault(acc, "".join(buf))
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) > 2 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if acc:
            seqs.setdefault(acc, "".join(buf))
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=os.path.join("data", "fireprotdb_holdout.parquet"))
    ap.add_argument("--seq-cache", default=SEQ_CACHE)
    ap.add_argument("--max-protein-length", type=int, default=5000)
    ap.add_argument("--training-splits",
                    default=os.path.join("data", "splits_megascale.parquet"),
                    help="stability split table; FireProt proteins that CONTAIN "
                         "a training construct are dropped (see below). Pass '' "
                         "to skip the check.")
    args = ap.parse_args()

    print(f"[fireprot] reading {args.source}")
    df = pd.read_csv(args.source, low_memory=False)
    print(f"[fireprot] {len(df)} rows")

    stats: dict[str, int] = {}

    def drop(reason: str, n: int = 1) -> None:
        if n:
            stats[reason] = stats.get(reason, 0) + n

    n0 = len(df)
    df = df[df["DDG"].notna()]
    drop("no_ddg", n0 - len(df))

    n0 = len(df)
    # The whole point: keep only what is NOT already the training set.
    df = df[df["SOURCE_DATASET"] != "MegaScale"]
    drop("megascale_already_in_training_set", n0 - len(df))

    n0 = len(df)
    df = df[df["SUBSTITUTION"].notna()
            & df["INSERTION"].isna() & df["DELETION"].isna()]
    drop("not_a_single_substitution", n0 - len(df))

    n0 = len(df)
    df = df[df["UNIPROTKB"].notna()].copy()
    drop("no_uniprot_accession", n0 - len(df))

    n0 = len(df)
    df["accession"] = df["UNIPROTKB"].map(normalise_accession)
    df = df[df["accession"].notna()]
    drop("unparseable_uniprot_accession", n0 - len(df))

    df = df.groupby(["accession", "SUBSTITUTION"], as_index=False).agg(
        DDG=("DDG", "mean"), PROTEIN=("PROTEIN", "first"),
        ORGANISM=("ORGANISM", "first"))
    print(f"[fireprot] {len(df)} unique (protein, substitution) measurements")

    accessions = sorted(df["accession"].astype(str).unique())
    seqs = fetch_uniprot(accessions, args.seq_cache)
    print(f"[fireprot] have sequences for {len(seqs)}/{len(accessions)} accessions")

    records = []
    for r in df.itertuples(index=False):
        seq = seqs.get(str(r.accession))
        if not seq:
            drop("no_sequence_for_accession")
            continue
        m = SUB_RE.match(str(r.SUBSTITUTION).upper())
        if not m:
            drop("unparseable_substitution")
            continue
        wt_aa, pos, mt_aa = m.group(1), int(m.group(2)), m.group(3)
        if pos < 1 or pos > len(seq):
            drop("position_outside_sequence")
            continue
        if seq[pos - 1] != wt_aa:
            # Usually a numbering offset (mature chain vs precursor). Not
            # guessed at: an offset applied blindly would move every mutation.
            drop("wildtype_residue_mismatch")
            continue
        if wt_aa == mt_aa:
            drop("synonymous")
            continue
        if len(seq) > args.max_protein_length:
            drop("protein_too_long")
            continue
        records.append({
            "wt_seq": seq,
            "mt_seq": seq[:pos - 1] + mt_aa + seq[pos:],
            "ddg": float(r.DDG),
            "parent": str(r.accession),
            "protein_name": str(r.PROTEIN),
            "organism": str(r.ORGANISM),
            "substitution": str(r.SUBSTITUTION).upper(),
        })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise SystemExit("[fireprot] no rows survived")

    # SEQUENCE contamination, which removing the MegaScale-sourced ROWS does not
    # catch. Dropping rows whose measurement came from MegaScale makes this table
    # share no measurement with the training table - and this file claimed that
    # made it independent. It does not. A MegaScale construct is a 32-72-residue
    # DOMAIN, and several FireProt proteins are the full-length parents those
    # domains were cut from, so the trained construct sits inside the "held-out"
    # protein as a literal substring. Measured on the shipped tables: 6 of 182
    # proteins, 302 of 4535 rows (6.7%), of which 162 (3.6%) mutate a residue
    # that was inside the trained construct. Zero contain a HELD-OUT construct,
    # so the contamination is one-sided towards train and can only inflate the
    # external number - the one figure in the run that is supposed to say
    # whether anything transfers off the training distribution.
    if args.training_splits and os.path.exists(args.training_splits):
        sp = pd.read_parquet(args.training_splits, columns=["parent_seq", "split"])
        train_constructs = set(sp.loc[sp.split == "train", "parent_seq"].unique())
        contaminated = {
            r.parent for r in out[["parent", "wt_seq"]].drop_duplicates("parent")
                                  .itertuples(index=False)
            if any(c in r.wt_seq for c in train_constructs)
        }
        if contaminated:
            n_rows = int(out["parent"].isin(contaminated).sum())
            drop("contains_a_training_construct_as_substring", n_rows)
            out = out[~out["parent"].isin(contaminated)].reset_index(drop=True)
            print(f"[fireprot] dropped {len(contaminated)} proteins ({n_rows} rows) "
                  f"that contain a MegaScale TRAINING construct as a substring")
    else:
        print(f"[fireprot] WARNING: no split table at {args.training_splits}, so "
              f"sequence contamination against the training constructs was NOT "
              f"checked. This table's independence is unverified.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)

    lengths = out["wt_seq"].str.len()
    print(f"\n[fireprot] wrote {args.out}")
    print(f"  rows written      : {len(out)}")
    print(f"  distinct proteins : {out['parent'].nunique()}")
    print(f"  organisms         : {out['organism'].nunique()}")
    print(f"  protein length    : median {int(lengths.median())}, "
          f"min {int(lengths.min())}, max {int(lengths.max())}")
    print(f"  ddg (RAW, source convention): mean {out['ddg'].mean():+.3f}, "
          f"sd {out['ddg'].std():.3f}")
    print(f"  NOTE: this table is destabilizing-positive. configs/data.yaml "
          f"declares ddg_sign for it; after the loader's flip the mean becomes "
          f"{-out['ddg'].mean():+.3f}, i.e. destabilising on average as it must be.")
    print(f"  dropped:")
    for reason, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"      {reason:<38} {n}")


if __name__ == "__main__":
    main()
