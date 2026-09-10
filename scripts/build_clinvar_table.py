"""
scripts/build_clinvar_table.py
==============================
Build ``data/clinvar_protein_pairs.parquet`` — reference/alternate PROTEIN pairs
for the pathogenicity head — from ClinVar's tab-delimited release plus MANE.

Why not the VCF
---------------
``dataset/clinvar/clinvar.vcf`` carries genomic coordinates only
(``CLNHGVS=NC_000001.11:g.69134A>G``). Turning that into a protein pair needs a
reference genome and transcript models, and getting it wrong is not loud: A, C,
G and T are all valid amino-acid letters, so a protein language model accepts a
nucleotide string silently and learns nothing. ``configs/data.yaml`` records
that this exact mistake produced a meaningless ClinVar task in the predecessor
repository.

``variant_summary.txt.gz`` avoids the problem entirely: its ``Name`` field
already carries the curated protein consequence,

    NM_000059.4(BRCA2):c.7397C>T (p.Ala2466Val)

so the transcript, the reference residue, the position and the alternate
residue are read rather than recomputed. MANE Select supplies the NM -> NP
mapping and the protein sequences.

The check that makes this trustworthy
-------------------------------------
For every variant the reference residue named in the HGVS must equal the residue
actually at that position in the MANE protein. A row that fails is dropped and
counted, never coerced. That single check is what separates a real ClinVar task
from the silent-garbage failure mode above: it catches transcript-version drift,
off-by-one indexing, and any NM->NP mismatch, because all three break the
identity.

What is kept
------------
* single-nucleotide, missense consequences only (``p.Xxx###Yyy``) — the model
  predicts one substitution, and nonsense/frameshift/synonymous are different
  tasks;
* classifications in ClinVar's benign/pathogenic vocabulary, passed through as
  the raw text so ``src/data._binary_label`` owns the mapping and its own
  drop accounting;
* at least one review star by default, so ``no assertion criteria provided``
  submissions do not become labels.

Usage
-----
python -m scripts.build_clinvar_table
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

CLINVAR_DIR = os.path.join("dataset", "clinvar")

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Sec": "U", "Pyl": "O",
}

#: NM_000059.4(BRCA2):c.7397C>T (p.Ala2466Val)
NAME_RE = re.compile(
    r"^(?P<nm>N[MR]_\d+)\.\d+\([^)]*\):c\.[^ ]+ "
    r"\(p\.(?P<ref>[A-Z][a-z]{2})(?P<pos>\d+)(?P<alt>[A-Z][a-z]{2})\)$"
)

#: Review status -> ClinVar's own star rating.
STARS = {
    "no assertion provided": 0,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no classification for the single variant": 0,
    "no classifications from unflagged records": 0,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, multiple submitters": 2,
    "reviewed by expert panel": 3,
    "practice guideline": 4,
}

BENIGN = {"benign", "likely benign", "benign/likely benign"}
PATHOGENIC = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}


CODONS = {}
for _i, _b1 in enumerate("TCAG"):
    for _b2 in "TCAG":
        for _b3 in "TCAG":
            CODONS[_b1 + _b2 + _b3] = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"[
                len(CODONS)]


def translate(dna: str) -> str:
    return "".join(CODONS.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3))


def read_fasta(path_gz: str) -> dict[str, str]:
    """Parse a gzipped FASTA into {accession without version: sequence}."""
    out: dict[str, str] = {}
    acc, buf = None, []
    with gzip.open(path_gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc:
                    out[acc] = "".join(buf)
                acc = line[1:].split()[0].split(".")[0]
                buf = []
            else:
                buf.append(line.strip())
    if acc:
        out[acc] = "".join(buf)
    return out


def find_cds(rna: str, protein: str) -> str | None:
    """
    Return the coding subsequence of ``rna`` that translates to ``protein``.

    A MANE transcript carries UTRs, so the CDS starts at some offset. That
    offset is FOUND, by translating each of the three reading frames once and
    locating the protein in it, then VERIFIED by translating the slice back.
    ``src/features.py`` reads codon i as belonging to residue i, so an
    unverified offset would not error - it would shift every codon and make the
    CAI and mRNA-folding channels wrong rather than absent, which is the same
    silent failure the mega-scale reading frame had.
    """
    rna = rna.upper().replace("U", "T")
    for frame in (0, 1, 2):
        idx = translate(rna[frame:]).find(protein)
        if idx < 0:
            continue
        start = frame + 3 * idx
        cds = rna[start:start + 3 * len(protein)]
        if len(cds) == 3 * len(protein) and translate(cds) == protein:
            return cds
    return None


def read_mane(summary_gz: str, faa_gz: str) -> tuple[dict, dict]:
    """Return (NM base accession -> NP accession, NP base accession -> sequence)."""
    nm_to_np: dict[str, str] = {}
    with gzip.open(summary_gz, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            nuc, prot = row.get("RefSeq_nuc"), row.get("RefSeq_prot")
            if nuc and prot:
                nm_to_np[nuc.split(".")[0]] = prot.split(".")[0]

    seqs: dict[str, str] = {}
    acc, buf = None, []
    with gzip.open(faa_gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                acc = line[1:].split()[0].split(".")[0]
                buf = []
            else:
                buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return nm_to_np, seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant-summary",
                    default=os.path.join(CLINVAR_DIR, "variant_summary.txt.gz"))
    ap.add_argument("--mane-summary",
                    default=os.path.join(CLINVAR_DIR, "MANE.GRCh38.v1.5.summary.txt.gz"))
    ap.add_argument("--mane-protein",
                    default=os.path.join(CLINVAR_DIR, "MANE.GRCh38.v1.5.refseq_protein.faa.gz"))
    ap.add_argument("--mane-rna",
                    default=os.path.join(CLINVAR_DIR, "MANE.GRCh38.v1.5.refseq_rna.fna.gz"),
                    help="MANE RefSeq transcripts; supplies the real coding "
                         "sequence that activates the mechanistic stream")
    ap.add_argument("--out", default=os.path.join("data", "clinvar_protein_pairs.parquet"))
    ap.add_argument("--min-stars", type=int, default=1)
    ap.add_argument("--max-protein-length", type=int, default=5000)
    args = ap.parse_args()

    print(f"[clinvar] reading MANE from {args.mane_summary}")
    nm_to_np, seqs = read_mane(args.mane_summary, args.mane_protein)
    print(f"[clinvar] {len(nm_to_np)} MANE transcripts, {len(seqs)} protein sequences")

    # Real coding sequences. Without these mech_valid is 0 for every ClinVar row
    # and E_mech is empty, which would leave the mechanistic stream - the
    # paper's distinguishing contribution - measured on the stability task only.
    cds_by_np: dict[str, str] = {}
    if args.mane_rna and os.path.exists(args.mane_rna):
        rna_by_nm = read_fasta(args.mane_rna)
        n_frame_fail = 0
        for nm, np_acc in nm_to_np.items():
            rna, prot = rna_by_nm.get(nm), seqs.get(np_acc)
            if not rna or not prot:
                continue
            cds = find_cds(rna, prot)
            if cds is None:
                n_frame_fail += 1
            else:
                cds_by_np[np_acc] = cds
        print(f"[clinvar] coding sequence recovered and translation-verified for "
              f"{len(cds_by_np)}/{len(nm_to_np)} transcripts "
              f"({n_frame_fail} failed the frame check and are left without a CDS)")
    else:
        print(f"[clinvar] no MANE RNA at {args.mane_rna}; rows will carry no CDS "
              f"and E_mech will be empty for this task")

    stats: dict[str, int] = {}

    def drop(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    n_rows = 0

    print(f"[clinvar] streaming {args.variant_summary}")
    with gzip.open(args.variant_summary, "rt", encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  {n_rows} rows scanned, {len(records)} kept", flush=True)

            sig = (row.get("ClinicalSignificance") or "").strip()
            low = sig.lower()
            if low not in BENIGN and low not in PATHOGENIC:
                drop("classification_not_binary")
                continue

            status = (row.get("ReviewStatus") or "").strip().lower()
            if STARS.get(status, 0) < args.min_stars:
                drop("below_min_review_stars")
                continue

            m = NAME_RE.match((row.get("Name") or "").strip())
            if not m:
                drop("not_a_missense_protein_change")
                continue

            ref3, alt3 = m.group("ref"), m.group("alt")
            if ref3 not in THREE_TO_ONE or alt3 not in THREE_TO_ONE:
                drop("non_standard_residue")          # Ter, Xaa, synonymous '='
                continue
            ref, alt = THREE_TO_ONE[ref3], THREE_TO_ONE[alt3]
            if ref == alt:
                drop("synonymous")
                continue

            np_acc = nm_to_np.get(m.group("nm"))
            if np_acc is None:
                drop("transcript_not_in_mane_select")
                continue
            seq = seqs.get(np_acc)
            if seq is None:
                drop("no_sequence_for_protein")
                continue

            pos = int(m.group("pos"))
            if pos < 1 or pos > len(seq):
                drop("position_outside_protein")
                continue
            # The check the whole table rests on.
            if seq[pos - 1] != ref:
                drop("reference_residue_mismatch")
                continue
            if len(seq) > args.max_protein_length:
                drop("protein_too_long")
                continue

            key = (np_acc, f"{ref}{pos}{alt}")
            if key in seen:
                drop("duplicate_variant")
                continue
            seen.add(key)

            records.append({
                "ref_protein": seq,
                "alt_protein": seq[:pos - 1] + alt + seq[pos:],
                "clinical_significance": sig,
                "gene_symbol": (row.get("GeneSymbol") or "").strip(),
                "wt_dna": cds_by_np.get(np_acc),
                "protein_accession": np_acc,
                "protein_change": f"{ref}{pos}{alt}",
                "review_stars": STARS.get(status, 0),
            })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise SystemExit("[clinvar] no rows survived; check the input files")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)

    n_path = int(out["clinical_significance"].str.lower().isin(PATHOGENIC).sum())
    lengths = out["ref_protein"].str.len()
    print(f"\n[clinvar] wrote {args.out}")
    print(f"  rows scanned      : {n_rows}")
    print(f"  rows written      : {len(out)}")
    print(f"  pathogenic        : {n_path} ({n_path / len(out):.1%})")
    print(f"  benign            : {len(out) - n_path} "
          f"({1 - n_path / len(out):.1%})")
    cds_cov = float(out["wt_dna"].notna().mean())
    print(f"  rows with real CDS: {cds_cov:.1%} "
          f"({'E_mech active for those rows' if cds_cov else 'E_mech empty'})")
    print(f"  distinct genes    : {out['gene_symbol'].nunique()}")
    print(f"  distinct proteins : {out['protein_accession'].nunique()}")
    print(f"  protein length    : median {int(lengths.median())}, "
          f"max {int(lengths.max())}")
    positions = out["protein_change"].str.slice(1, -1).astype(int)
    print(f"  mutation position : median {int(positions.median())}, "
          f"max {int(positions.max())}")
    print(f"  review stars      : "
          f"{dict(out['review_stars'].value_counts().sort_index())}")
    print(f"  dropped           :")
    for reason, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"      {reason:<34} {n}")


if __name__ == "__main__":
    main()
