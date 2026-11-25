"""
src/geometry.py
===============
SE(3)-invariant backbone geometry extracted from AlphaFold or crystal PDBs.

Implements ``GEOMETRIC_PLAN.md`` §2.1:

* Factors the validated PDB parser and unique-best-offset alignment out of
  ``scripts/build_contact_maps.py`` so contact maps and 3D coordinate features
  share one exact alignment contract.
* Generalizes parsing from ``{CA, CB}`` to ``{N, CA, C, CB}`` plus per-residue
  ``pLDDT`` from the B-factor column.
* Produces strictly SE(3)-invariant edge features (23 channels: 16 Gaussian RBF
  bins of CB-CB distance over 2-22 A, 3 local-frame CA->CA unit vector
  components, 4 canonical unit quaternion components of ``R_i^T R_j``) and
  SE(3)-invariant node features (9 channels: 6 backbone dihedral sin/cos,
  normalized spatial neighbor count, pLDDT/100, and ``coord_valid``).

Because ddG is an E(3)-invariant scalar, distances plus local frames form a
complete invariant basis (Villar et al., NeurIPS 2021) without the order-of-
magnitude overhead of tensor-product equivariant layers.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import torch

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

#: Construct name convention in MegaScale ("1A0N.pdb", "1A0N.pdb_L7S", "1A0N.pdb_pross1").
NAME_RE = re.compile(
    r"^(?P<base>[^.]+)\.pdb(?:_(?P<mut>[A-Z]\d+[A-Z]|pross\d+))?$"
)


def normalize_pdb_stem(name: str) -> str:
    """
    Canonicalize a PDB filename or construct stem for cross-platform lookup.

    Strips a trailing ``.pdb`` (case-insensitively) if present, uppercases, and
    normalizes ``|`` (MegaScale ``WT_name`` convention, e.g.
    ``EA|run2_0325_0005.pdb``) and ``:`` (Zenodo 7844779 zip member naming,
    e.g. ``EA:run2_0325_0005.pdb``, which Windows extracts as
    ``EA_run2_0325_0005.pdb``) to ``_``.
    """
    s = str(name).strip()
    if s.lower().endswith(".pdb"):
        s = s[:-4]
    return s.upper().replace("|", "_").replace(":", "_")


def resolve_pdb_path(
    parent: str,
    available: dict[str, Path | str],
) -> Path | str | None:
    """
    Resolve a MegaScale construct name (e.g. ``1A0N.pdb``, ``1A0N.pdb_L7S``,
    ``EA|run2_0325_0005.pdb``) or UniProt accession (``P12345``,
    ``AF-P12345-F1-model_v4.pdb``) against a mapping of available PDB stems.

    Checks exact ``NAME_RE`` base, normalized base (via :func:`normalize_pdb_stem`),
    and direct / AlphaFold DB ``AF-<KEY>`` / ``AF-<KEY>-*`` lookups while
    normalizing keys in ``available`` so ``|``, ``:``, and ``_`` variants match
    identically on Windows and Linux.
    """
    if not available or not parent:
        return None
    parent_s = str(parent).strip()
    if not parent_s:
        return None

    m = NAME_RE.match(parent_s)
    base_str = m.group("base") if m else parent_s
    raw_base = base_str.upper()
    norm_base = normalize_pdb_stem(base_str)

    if m:
        hit = available.get(raw_base)
        if hit is not None:
            return hit
        hit = available.get(norm_base)
        if hit is not None:
            return hit

    norm_available: dict[str, Path | str] = {}
    for k, v in available.items():
        nk = normalize_pdb_stem(k)
        if nk not in norm_available:
            norm_available[nk] = v

    if m and norm_base in norm_available:
        return norm_available[norm_base]

    keys: list[str] = []
    for k in (raw_base, norm_base, parent_s.upper(), normalize_pdb_stem(parent_s)):
        if k and k not in keys:
            keys.append(k)

    for key in keys:
        for cand in (key, f"AF-{key}"):
            if cand in available:
                return available[cand]
            norm_cand = normalize_pdb_stem(cand)
            if norm_cand in norm_available:
                return norm_available[norm_cand]

    for key in keys:
        prefix = f"AF-{key}-"
        for stem, val in available.items():
            if stem.upper().startswith(prefix) or normalize_pdb_stem(stem).startswith(prefix):
                return val

    # Crystal PDB with chain suffix (e.g. S669 parent '1A0F_A' -> '1A0F.pdb').
    for key in keys:
        if len(key) >= 6 and key[4] in ("_", "-", ".") and key[:4].isalnum():
            pdb4 = key[:4]
            if pdb4 in available:
                return available[pdb4]
            if pdb4 in norm_available:
                return norm_available[pdb4]
    return None

MAX_MISMATCH_FRACTION = 0.15

#: Atom slot indices in ``coords_4`` arrays of shape ``(L, 4, 3)``.
ATOM_N, ATOM_CA, ATOM_C, ATOM_CB = 0, 1, 2, 3

#: Geometry feature widths (GEOMETRIC_PLAN.md §2.1).
N_RBF_BINS = 16
STR_GEOM_EDGE_DIM = 23      # 16 RBF + 3 unit vector in local frame + 4 quaternion
STR_NODE_DIM = 9            # 6 dihedral sin/cos + 1 neighbor count + 1 pLDDT + 1 valid
PLDDT_VALID_FLOOR = 50.0    # below this, coord_valid is zeroed (GEOMETRIC_PLAN §6.3)


# ---------------------------------------------------------------------------
# PDB parsing & sequence alignment
# ---------------------------------------------------------------------------
def parse_pdb_cb(path: str) -> tuple[str, np.ndarray]:
    """
    Read one PDB into ``(sequence, coords)`` using CB, or CA for glycine.

    Byte-identical contract to the original ``scripts/build_contact_maps.py``.
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
            key = (chain, line[22:27])
            if key not in best:
                best[key] = {"res": resname}
                order.append(key)
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
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


def parse_pdb_backbone(
    path: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse first chain of a PDB into ``(seq, coords_4, plddt, atom_valid)``.

    * ``coords_4`` : ``(L, 4, 3)`` float32 for ``[N, CA, C, CB]``. For glycine
      or residues lacking CB, a virtual CB is placed from ``(N, CA, C)`` via
      the standard ideal tetrahedral formula (falling back to CA if N/C are
      missing).
    * ``plddt``    : ``(L,)`` float32 read from CA B-factor (0-100 scale; if
      values appear to be in [0, 1], rescaled x100).
    * ``atom_valid``: ``(L,)`` float32, 1.0 when N, CA, and C are all present.
    """
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    chain0 = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            if len(line) < 54:
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
            if atom not in ("N", "CA", "C", "CB"):
                continue
            key = (chain, line[22:27])
            if key not in best:
                best[key] = {"res": resname, "atoms": {}, "bfac": 0.0}
                order.append(key)
            try:
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                )
            except ValueError:
                continue
            if atom not in best[key]["atoms"]:
                best[key]["atoms"][atom] = xyz
                if atom == "CA" and len(line) >= 66:
                    try:
                        best[key]["bfac"] = float(line[60:66])
                    except ValueError:
                        best[key]["bfac"] = 0.0

    seq: list[str] = []
    coords_list: list[np.ndarray] = []
    plddt_list: list[float] = []
    valid_list: list[float] = []

    for key in order:
        rec = best[key]
        atoms = rec["atoms"]
        if "CA" not in atoms and "CB" not in atoms:
            continue
        ca = atoms.get("CA", atoms.get("CB"))
        has_frame = ("N" in atoms) and ("CA" in atoms) and ("C" in atoms)
        n_xyz = atoms.get("N", ca)
        c_xyz = atoms.get("C", ca)
        if "CB" in atoms:
            cb_xyz = atoms["CB"]
        elif has_frame:
            # Ideal virtual CB from (N, CA, C) (ProteinMPNN / AlphaFold convention).
            b = ca - n_xyz
            c = c_xyz - ca
            a = np.cross(b, c)
            cb_xyz = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca
        else:
            cb_xyz = ca

        seq.append(THREE_TO_ONE[rec["res"]])
        coords_list.append(np.stack([n_xyz, ca, c_xyz, cb_xyz], axis=0))
        plddt_list.append(rec["bfac"])
        valid_list.append(1.0 if has_frame else 0.0)

    if not seq:
        return (
            "",
            np.zeros((0, 4, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    plddt = np.asarray(plddt_list, dtype=np.float32)
    if plddt.max() <= 1.05 and plddt.max() > 0.0:
        plddt = plddt * 100.0
    return (
        "".join(seq),
        np.asarray(coords_list, dtype=np.float32),
        plddt,
        np.asarray(valid_list, dtype=np.float32),
    )


def align_to_sequence(
    pdb_seq: str,
    target_seq: str,
    max_mismatch_fraction: float = MAX_MISMATCH_FRACTION,
) -> tuple[int | None, int, str | None]:
    """
    Align ``target_seq`` onto ``pdb_seq`` as a contiguous block with a unique
    best offset within the mismatch budget.

    Returns ``(offset, mismatches, error_reason)``. On failure ``offset`` is
    ``None`` and ``error_reason`` names why; never guesses.
    """
    if pdb_seq == target_seq:
        return 0, 0, None
    if len(pdb_seq) < len(target_seq):
        return None, 0, "structure_shorter_than_sequence"

    budget = max(1, int(max_mismatch_fraction * len(target_seq)))
    scored = [
        (sum(a != b for a, b in zip(pdb_seq[o:o + len(target_seq)], target_seq)), o)
        for o in range(len(pdb_seq) - len(target_seq) + 1)
    ]
    best_mismatches, best_offset = min(scored)
    if best_mismatches > budget:
        return None, best_mismatches, "too_many_mismatches_vs_structure"
    if sum(1 for m, _ in scored if m == best_mismatches) > 1:
        return None, best_mismatches, "ambiguous_alignment"
    return best_offset, best_mismatches, None


def contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Binary ``(L, L)`` contact map for ``dist(i, j) < threshold`` (0 diagonal)."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    m = (d < threshold).astype(np.uint8)
    np.fill_diagonal(m, 0)
    return m


# ---------------------------------------------------------------------------
# SE(3)-invariant differential & frame geometry (PyTorch)
# ---------------------------------------------------------------------------
def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def local_frames(coords_4: torch.Tensor) -> torch.Tensor:
    """
    Build right-handed local orthonormal frames ``R`` of shape ``(L, 3, 3)``
    from backbone ``(N, CA, C)`` coordinates.

    Columns of ``R[i]`` are ``[e1, e2, e3]`` where ``e1 = unit(C - CA)``,
    ``e2 = unit(orth(N - CA, e1))``, ``e3 = e1 x e2``. Under any global rigid
    transform ``x' = Q x + t`` (``Q in SO(3)``), ``R'[i] = Q R[i]``, so
    ``R_i^T (x_j - x_i)`` and ``R_i^T R_j`` are strictly SE(3)-invariant.
    """
    n_xyz = coords_4[:, ATOM_N]
    ca_xyz = coords_4[:, ATOM_CA]
    c_xyz = coords_4[:, ATOM_C]

    u1 = c_xyz - ca_xyz
    u2 = n_xyz - ca_xyz
    e1 = _safe_normalize(u1)
    u2_orth = u2 - (u2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = _safe_normalize(u2_orth)
    e3 = torch.cross(e1, e2, dim=-1)
    # Columns [e1, e2, e3] -> (L, 3, 3)
    return torch.stack([e1, e2, e3], dim=-1)


def _dihedral_angle(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> torch.Tensor:
    """Signed torsion angle in radians for quadruples ``(p0, p1, p2, p3)``."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_unit = _safe_normalize(b1)
    v = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_unit, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x + 1e-8)


def dihedrals(coords_4: torch.Tensor) -> torch.Tensor:
    """
    Backbone torsion features ``(L, 6)`` = ``[sin/cos(phi), sin/cos(psi), sin/cos(omega)]``.
    Strictly SE(3)-invariant (and zero at undefined termini).
    """
    L = coords_4.shape[0]
    out = torch.zeros((L, 6), dtype=coords_4.dtype, device=coords_4.device)
    if L < 2:
        return out
    N = coords_4[:, ATOM_N]
    CA = coords_4[:, ATOM_CA]
    C = coords_4[:, ATOM_C]

    # phi[i] = dihedral(C[i-1], N[i], CA[i], C[i]) for i >= 1
    phi = _dihedral_angle(C[:-1], N[1:], CA[1:], C[1:])
    out[1:, 0] = torch.sin(phi)
    out[1:, 1] = torch.cos(phi)

    # psi[i] = dihedral(N[i], CA[i], C[i], N[i+1]) for i <= L-2
    psi = _dihedral_angle(N[:-1], CA[:-1], C[:-1], N[1:])
    out[:-1, 2] = torch.sin(psi)
    out[:-1, 3] = torch.cos(psi)

    # omega[i] = dihedral(CA[i-1], C[i-1], N[i], CA[i]) for i >= 1
    omega = _dihedral_angle(CA[:-1], C[:-1], N[1:], CA[1:])
    out[1:, 4] = torch.sin(omega)
    out[1:, 5] = torch.cos(omega)
    return out


def rbf(
    distances: torch.Tensor,
    d_min: float = 2.0,
    d_max: float = 22.0,
    num_bins: int = N_RBF_BINS,
) -> torch.Tensor:
    """Gaussian radial basis expansion of scalar distances into ``(..., num_bins)``."""
    centers = torch.linspace(d_min, d_max, num_bins, device=distances.device, dtype=distances.dtype)
    sigma = (d_max - d_min) / max(num_bins - 1, 1)
    diff = distances.unsqueeze(-1) - centers
    return torch.exp(-0.5 * (diff / sigma) ** 2)


def rotation_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices ``R`` of shape ``(..., 3, 3)`` to canonical unit
    quaternions ``(w, x, y, z)`` of shape ``(..., 4)``.

    Canonicalization enforces a consistent hemisphere (first entry exceeding
    1e-4 is non-negative) so equivalent representations ``q`` and ``-q`` never
    produce sign flips across numerical noise.
    """
    # Compute in float64 internally for numerical stability near trace = -1.
    orig_dtype = R.dtype
    R64 = R.to(torch.float64)
    r00, r01, r02 = R64[..., 0, 0], R64[..., 0, 1], R64[..., 0, 2]
    r10, r11, r12 = R64[..., 1, 0], R64[..., 1, 1], R64[..., 1, 2]
    r20, r21, r22 = R64[..., 2, 0], R64[..., 2, 1], R64[..., 2, 2]

    # Shepperd's 4-branch method (singularity-free over all SO(3)).
    t0 = 1.0 + r00 + r11 + r22
    t1 = 1.0 + r00 - r11 - r22
    t2 = 1.0 - r00 + r11 - r22
    t3 = 1.0 - r00 - r11 + r22

    q0 = torch.stack([t0, r21 - r12, r02 - r20, r10 - r01], dim=-1)
    q1 = torch.stack([r21 - r12, t1, r01 + r10, r02 + r20], dim=-1)
    q2 = torch.stack([r02 - r20, r01 + r10, t2, r12 + r21], dim=-1)
    q3 = torch.stack([r10 - r01, r02 + r20, r12 + r21, t3], dim=-1)

    cands = torch.stack([q0, q1, q2, q3], dim=-2)           # (..., 4, 4)
    traces = torch.stack([t0, t1, t2, t3], dim=-1)          # (..., 4)
    best = traces.argmax(dim=-1)                            # (...)
    idx = best.unsqueeze(-1).unsqueeze(-1).expand(*best.shape, 1, 4)
    q = cands.gather(-2, idx).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # Canonical hemisphere: first entry with |q_k| > 1e-4 made positive.
    sign = torch.ones(q.shape[:-1], dtype=torch.float64, device=q.device)
    decided = torch.zeros(q.shape[:-1], dtype=torch.bool, device=q.device)
    for k in range(4):
        comp = q[..., k]
        use = (~decided) & (comp.abs() > 1e-4)
        sign = torch.where(use, torch.where(comp < 0, -1.0, 1.0), sign)
        decided = decided | use
    q = q * sign.unsqueeze(-1)
    return q.to(orig_dtype)


def knn_edges(
    cb_coords: torch.Tensor,
    coord_valid: torch.Tensor | None = None,
    k: int = 30,
    max_dist: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build directed k-NN edges on CB coordinates within ``max_dist`` Angstroms,
    excluding self-loops and residues with ``coord_valid <= 0.5``.

    Returns ``(edge_index (2, E), cb_dist (E,))``.
    """
    L = cb_coords.shape[0]
    if L < 2:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=cb_coords.device),
            torch.zeros((0,), dtype=cb_coords.dtype, device=cb_coords.device),
        )
    diff = cb_coords.unsqueeze(1) - cb_coords.unsqueeze(0)   # (L, L, 3)
    dist = diff.norm(dim=-1)                                 # (L, L)

    # Mask self-loops and invalid residues to +inf.
    inf = torch.tensor(float("inf"), dtype=dist.dtype, device=dist.device)
    eye = torch.eye(L, dtype=torch.bool, device=dist.device)
    invalid = eye
    if coord_valid is not None:
        bad = coord_valid <= 0.5
        invalid = invalid | bad.unsqueeze(0) | bad.unsqueeze(1)
    dist_masked = dist.masked_fill(invalid, inf)

    k_eff = min(k, L - 1)
    vals, nbrs = torch.topk(dist_masked, k=k_eff, dim=1, largest=False)
    keep = vals <= max_dist
    src = torch.arange(L, device=cb_coords.device).unsqueeze(1).expand_as(nbrs)[keep]
    dst = nbrs[keep]
    d_keep = vals[keep]
    # Canonical (src, dst) ordering so tie-breaking under rigid transforms never permutes edges.
    order = torch.argsort(src * L + dst)
    return torch.stack([src[order], dst[order]], dim=0).long(), d_keep[order]


def edge_geometry(coords_4: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    23-channel strictly SE(3)-invariant edge features for directed edges ``i -> j``:

    * ``[0:16]``  Gaussian RBF of ``||CB_j - CB_i||`` over 2-22 A
    * ``[16:19]`` unit vector ``R_i^T (CA_j - CA_i) / ||CA_j - CA_i||`` in residue i's frame
    * ``[19:23]`` canonical unit quaternion of relative frame ``R_i^T R_j``
    """
    E = edge_index.shape[1]
    if E == 0:
        return torch.zeros((0, STR_GEOM_EDGE_DIM), dtype=coords_4.dtype, device=coords_4.device)

    src, dst = edge_index[0], edge_index[1]
    cb = coords_4[:, ATOM_CB]
    ca = coords_4[:, ATOM_CA]
    R = local_frames(coords_4)                               # (L, 3, 3)

    d_cb = (cb[dst] - cb[src]).norm(dim=-1)                  # (E,)
    rbf_feat = rbf(d_cb)                                     # (E, 16)

    delta_ca = ca[dst] - ca[src]                             # (E, 3)
    dir_ca = _safe_normalize(delta_ca)                       # (E, 3)
    # R_i^T @ dir_ca_ij
    local_dir = torch.bmm(R[src].transpose(-1, -2), dir_ca.unsqueeze(-1)).squeeze(-1)
    # Self-loops have delta_ca = 0 -> zero local_dir explicitly
    is_loop = src == dst
    local_dir = local_dir.masked_fill(is_loop.unsqueeze(-1), 0.0)

    rel_R = torch.bmm(R[src].transpose(-1, -2), R[dst])      # (E, 3, 3)
    quat = rotation_to_quaternion(rel_R)                     # (E, 4)

    return torch.cat([rbf_feat, local_dir, quat], dim=-1)


def node_geometry(
    coords_4: torch.Tensor,
    plddt: torch.Tensor,
    coord_valid: torch.Tensor,
    neighbor_radius: float = 10.0,
) -> torch.Tensor:
    """
    9-channel strictly SE(3)-invariant node features:

    * ``[0:6]`` backbone dihedral sin/cos (phi, psi, omega)
    * ``[6]``   normalized CB neighbor count within ``neighbor_radius`` A (/ 20.0)
    * ``[7]``   pLDDT / 100.0
    * ``[8]``   effective ``coord_valid`` (zeroed where pLDDT < 50 per GEOMETRIC_PLAN §6.3)
    """
    L = coords_4.shape[0]
    if L == 0:
        return torch.zeros((0, STR_NODE_DIM), dtype=coords_4.dtype, device=coords_4.device)

    eff_valid = ((coord_valid > 0.5) & (plddt >= PLDDT_VALID_FLOOR)).to(coords_4.dtype)
    dih = dihedrals(coords_4) * eff_valid.unsqueeze(-1)

    cb = coords_4[:, ATOM_CB]
    d = (cb.unsqueeze(1) - cb.unsqueeze(0)).norm(dim=-1)
    valid_pair = (eff_valid.unsqueeze(0) > 0.5) & (eff_valid.unsqueeze(1) > 0.5)
    nbr_mask = (d < neighbor_radius) & (~torch.eye(L, dtype=torch.bool, device=coords_4.device)) & valid_pair
    nbr_count = (nbr_mask.sum(dim=-1).to(coords_4.dtype) / 20.0).clamp(0.0, 2.0) * eff_valid

    plddt_norm = (plddt.to(coords_4.dtype) / 100.0).clamp(0.0, 1.0)
    return torch.cat(
        [dih, nbr_count.unsqueeze(-1), plddt_norm.unsqueeze(-1), eff_valid.unsqueeze(-1)],
        dim=-1,
    )


__all__ = [
    "THREE_TO_ONE",
    "NAME_RE",
    "normalize_pdb_stem",
    "resolve_pdb_path",
    "MAX_MISMATCH_FRACTION",
    "ATOM_N", "ATOM_CA", "ATOM_C", "ATOM_CB",
    "N_RBF_BINS", "STR_GEOM_EDGE_DIM", "STR_NODE_DIM", "PLDDT_VALID_FLOOR",
    "parse_pdb_cb",
    "parse_pdb_backbone",
    "align_to_sequence",
    "contact_map",
    "local_frames",
    "dihedrals",
    "rbf",
    "rotation_to_quaternion",
    "knn_edges",
    "edge_geometry",
    "node_geometry",
]
