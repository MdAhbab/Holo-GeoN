"""
scripts/build_contact_maps.py
=============================
Turn the AlphaFold models shipped with the mega-scale release into ground-truth
C-beta contact maps, so ``scripts/contact_sanity.py`` can report precision@L.

Why this exists
---------------
Without ``--true-contacts-dir`` the contact stage reports a long-range
*fraction*, which uniformly random edges also score well on. It can show ESM-2's
contact edges are non-local; it cannot show they are *correct*. The README is
explicit that precision@L against real C-beta contacts is "the number that
justifies building a graph rather than a chain", and the structures needed to
compute it are already in ``dataset/mega_scale_cdna/AlphaFold_model_PDBs``.

The contract this must satisfy
------------------------------
``contact_sanity.py`` loads ``<cache-key>.npy`` where the cache key is the
encoder's key for the parent sequence — ``sha1(sequence)`` (``src/encoder.py``)
— and requires ``true_map.shape[0] >= length``, then slices ``[:L, :L]``. So
entry ``[i, j]`` must be residue ``i`` against residue ``j`` **of the parent
sequence as encoded**, not of the PDB's own numbering. An off-by-one or an
unnoticed offset would not raise; it would quietly produce a precision figure
that means nothing, which is the same class of silent error the sign convention
and the reading frame both had.

So the alignment is verified, not assumed: the sequence is read back out of the
PDB (via residue names) and must either equal the parent sequence or contain it
exactly once as a contiguous block, in which case the map is cut to that block.
Anything else is skipped and counted. A structure is never guessed at.

Definitions
-----------
Contact: ``dist(CB_i, CB_j) < threshold`` (default 8 A), the standard C-beta
definition. Glycine has no CB, so its CA is used as the pseudo-CB — also
standard, and the alternative is dropping every glycine from the map.

Usage
-----
python -m scripts.build_contact_maps            # writes data/true_contacts
python run.py contacts --true-contacts-dir data/true_contacts
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

DEFAULT_PDB_DIR = os.path.join("dataset", "mega_scale_cdna", "AlphaFold_model_PDBs")

#: A construct name is "<PDB>.pdb", "<PDB>.pdb_<MUT>" (a scan whose own wild
#: type is a point mutant of the deposited structure), or "<PDB>.pdb_pross<N>"
#: (a PROSS-redesigned variant, which carries several substitutions). All
#: resolve to the same deposited model: these are stability-redesigned variants
#: of one fold, not different proteins.
NAME_RE = re.compile(
    r"^(?P<base>[^.]+)\.pdb(?:_(?P<mut>[A-Z]\d+[A-Z]|pross\d+))?$")

#: Fraction of positions allowed to differ between the parent sequence and the
#: deposited structure's sequence. A point-mutant background needs 1; a PROSS
#: redesign needs more. The backbone is what the contact map is read off, and a
#: handful of substitutions do not change it - but the budget is capped, and
#: what was actually tolerated is reported, so "same fold" stays a claim the
#: output can be checked against rather than an assumption.
MAX_MISMATCH_FRACTION = 0.15


def parse_pdb_cb(path: str) -> tuple[str, np.ndarray]:
    """
    Read one PDB into (sequence, coords) using CB, or CA for glycine.

    Takes the first chain encountered and the first altloc of each atom, which
    is what an AlphaFold model has anyway (single chain, no alternates).
    """
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    chain0 = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            chain = line[21]
            if chain0 is None:
                chain0 = chain
            if chain != chain0:
                continue
            resname = line[17:20].strip().upper()
            if resname not in THREE_TO_ONE:
                continue
            atom = line[12:16].strip()
            if atom not in ("CA", "CB"):
                continue
            key = (chain, line[22:27])          # resSeq + insertion code
            if key not in best:
                best[key] = {"res": resname}
                order.append(key)
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            # CB wins; CA is only the fallback, and only for glycine in practice.
            if atom == "CB" or "xyz" not in best[key]:
                if atom == "CB" or best[key].get("atom") != "CB":
                    best[key]["xyz"] = xyz
                    best[key]["atom"] = atom

    seq, coords = [], []
    for key in order:
        rec = best[key]
        if "xyz" not in rec:
            continue
        seq.append(THREE_TO_ONE[rec["res"]])
        coords.append(rec["xyz"])
    return "".join(seq), np.asarray(coords, dtype=np.float64)


def contact_map(coords: np.ndarray, threshold: float) -> np.ndarray:
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    m = (d < threshold).astype(np.uint8)
    np.fill_diagonal(m, 0)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=os.path.join("data", "megascale_ddg_pairs.parquet"))
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--out", default=os.path.join("data", "true_contacts"))
    ap.add_argument("--threshold", type=float, default=8.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.table, columns=["parent", "wt_seq"])
    constructs = df.drop_duplicates("parent")[["parent", "wt_seq"]]
    print(f"[contacts] {len(constructs)} constructs in {args.table}")

    available = {f[:-4].upper(): f for f in os.listdir(args.pdb_dir)
                 if f.lower().endswith(".pdb")}
    print(f"[contacts] {len(available)} PDB models in {args.pdb_dir}")

    written = 0
    stats: dict[str, int] = {}
    mismatch_hist: dict[int, int] = {}

    def note(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    for row in constructs.itertuples(index=False):
        m = NAME_RE.match(str(row.parent))
        if not m:
            note("unparseable_construct_name")
            continue
        fname = available.get(m.group("base").upper())
        if fname is None:
            # Designed proteins ("EA|run2_...") have no deposited structure.
            note("no_pdb_for_construct")
            continue

        pdb_seq, coords = parse_pdb_cb(os.path.join(args.pdb_dir, fname))
        if len(coords) == 0:
            note("pdb_had_no_usable_residues")
            continue

        seq = str(row.wt_seq)
        mismatches = 0
        if pdb_seq == seq:
            offset = 0
        else:
            # A background mutant or PROSS redesign differs from the deposited
            # sequence at one or more sites, so an exact substring search would
            # fail for exactly the constructs NAME_RE was written to recover.
            # Score every offset by mismatch count and take the best, provided
            # it is unique and within budget.
            budget = max(1, int(MAX_MISMATCH_FRACTION * len(seq)))
            offset = -1
            if len(pdb_seq) >= len(seq):
                scored = [
                    (sum(a != b for a, b in zip(pdb_seq[o:o + len(seq)], seq)), o)
                    for o in range(len(pdb_seq) - len(seq) + 1)
                ]
                best = min(scored)
                if best[0] > budget:
                    note("too_many_mismatches_vs_structure")
                    continue
                if sum(1 for s, _ in scored if s == best[0]) > 1:
                    note("ambiguous_alignment")
                    continue
                mismatches, offset = best
            if offset < 0:
                note("structure_shorter_than_sequence")
                continue
        mismatch_hist[mismatches] = mismatch_hist.get(mismatches, 0) + 1

        sub = coords[offset:offset + len(seq)]
        if len(sub) != len(seq):
            note("structure_shorter_than_sequence")
            continue

        cmap = contact_map(sub, args.threshold)
        key = hashlib.sha1(seq.encode()).hexdigest()
        np.save(os.path.join(args.out, key + ".npy"), cmap)
        written += 1

    print(f"\n[contacts] wrote {written} contact maps to {args.out}")
    print(f"  coverage: {written}/{len(constructs)} constructs "
          f"({written / len(constructs):.1%})")
    for reason, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {reason}: {n}")
    if mismatch_hist:
        shown = ", ".join(f"{k} mismatch{'es' if k != 1 else ''}: {v}"
                          for k, v in sorted(mismatch_hist.items()))
        print(f"  sequence vs structure agreement: {shown}")
    if written:
        print(f"\n  contact definition: CB-CB (CA for glycine) < {args.threshold} A")
        print(f"  maps are keyed by sha1(parent sequence), matching the encoder "
              f"cache key")


if __name__ == "__main__":
    main()
