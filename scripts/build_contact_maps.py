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

from src.geometry import (
    MAX_MISMATCH_FRACTION,
    NAME_RE,
    THREE_TO_ONE,
    align_to_sequence,
    contact_map,
    parse_pdb_cb,
)

DEFAULT_PDB_DIR = os.path.join("dataset", "mega_scale_cdna", "AlphaFold_model_PDBs")


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
        offset, mismatches, err = align_to_sequence(pdb_seq, seq, MAX_MISMATCH_FRACTION)
        if offset is None:
            note(err or "alignment_failed")
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
