"""
src/encoder.py
==============
Frozen ESM-2 encoder and its on-disk cache.

This is the one heavy GPU pass in the pipeline. It runs once per **unique
protein** and writes two things per sequence:

  * ``emb``  — per-residue embeddings, fp16, full length.
  * a **sparse** contact edge list from ESM-2's contact-prediction head.

Two decisions worth stating.

**Contact head, not attention map.** Edges come from ``predict_contacts``,
whose output is supervised against real structural contacts, rather than from
raw attention. Whether that actually carries non-local signal on your data is
measured, not asserted: run ``scripts/contact_sanity.py`` after precompute and
read ``results/contacts/contact_sanity.json``.

Read it carefully. Its default mode reports the long-range fraction (|i-j| >= 24)
among top-L predicted pairs, against a sequence-proximity baseline whose
long-range fraction is 0 by construction AND against a random-edge control.
Round-2 review, R2-13: with only the proximity baseline, uniformly random edges
scored 0.880 against a baseline defined to score 0.000, so that comparison could
not tell the contact head from noise. Only ``--true-contacts-dir`` - precision@L
against real C-beta contact maps - can show the edges are correct rather than
merely non-local, and that is the number to cite for the graph.

**Sparse, full-length storage.** A dense ``L x L`` map costs O(L^2) on disk and
is thrown away immediately after thresholding; storing the top-k neighbours per
residue costs O(L k) and loses nothing the graph builder would have used.
Caching at full length (rather than pre-windowed) means a window sliced later
carries whole-protein context, and one cached protein serves every mutation in
it — which is what keeps the cache keyed by ~10^3-10^4 proteins instead of
~10^5-10^6 mutants.

Proteins longer than the encoder's context are embedded in overlapping chunks;
see :func:`EsmEncoder.encode` for what that does and does not preserve.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from src.substitution import AA_ORDER

# Hidden size per ESM-2 checkpoint, so projections can be sized without a
# forward pass.
ESM_DIMS: dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}
DEFAULT_ESM = "facebook/esm2_t33_650M_UR50D"
N_TAP_LAYERS = 4

# ESM-2 positional limit is 1024 tokens including CLS/EOS.
MAX_CONTEXT_RESIDUES = 1022
CHUNK_OVERLAP = 256


def tap_layer_indices(num_layers: int) -> tuple[int, ...]:
    """
    Choose 4 hidden-state indices spanning middle-late to final layers for any
    ESM-2 depth (e.g. ``(18, 24, 30, 33)`` for 33-layer 650M, ``(3, 4, 5, 6)``
    for 6-layer 8M). Middle-late PLM layers retain stronger solvent-accessibility
    and secondary-structure geometry than the final token-prediction layer alone.
    """
    if num_layers <= 0:
        return (0,)
    if num_layers == 33:
        return (18, 24, 30, 33)
    if num_layers == 30:
        return (16, 21, 26, 30)
    if num_layers <= N_TAP_LAYERS:
        return tuple(range(1, num_layers + 1))
    raw = [max(1, min(num_layers, round(num_layers * f))) for f in (0.55, 0.73, 0.91, 1.0)]
    # Ensure 4 strictly increasing indices ending at num_layers.
    uniq: list[int] = []
    for x in raw:
        if x not in uniq:
            uniq.append(x)
    while len(uniq) < N_TAP_LAYERS:
        cand = max(1, uniq[0] - 1)
        while cand in uniq and cand < num_layers:
            cand += 1
        uniq.append(cand)
        uniq.sort()
    return tuple(sorted(uniq[-N_TAP_LAYERS:]))


def seq_key(seq: str) -> str:
    """Stable cache key for a sequence."""
    return hashlib.sha1(seq.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Sparse contact extraction
# ---------------------------------------------------------------------------
def sparse_topk_contacts(
    contact_map: torch.Tensor,
    k: int,
    min_sep: int,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce a dense ``(L, L)`` contact-probability map to its top-k neighbours
    per residue, as a deduplicated upper-triangular edge list.

    Pairs closer than ``min_sep`` in sequence are excluded: they are backbone
    context, which the graph builder adds explicitly, and letting them win the
    top-k would crowd out the long-range pairs that justify the graph.

    Returns ``(edges (2, E) int32, probs (E,) float32)`` with ``edges[0] <
    edges[1]``, both shifted by ``offset`` (used when the map covers a chunk of
    a longer protein).
    """
    L = contact_map.shape[0]
    if L < 2:
        return torch.zeros((2, 0), dtype=torch.int32), torch.zeros(0)

    cm = contact_map.float().clone()
    idx = torch.arange(L, device=cm.device)
    sep = (idx[:, None] - idx[None, :]).abs()
    cm[sep < min_sep] = -1.0

    kk = int(min(k, L - 1))
    vals, cols = torch.topk(cm, kk, dim=1)          # (L, kk)
    rows = idx.unsqueeze(1).expand(-1, kk)
    keep = vals > 0

    r = rows[keep].reshape(-1)
    c = cols[keep].reshape(-1)
    p = vals[keep].reshape(-1)
    if r.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.int32), torch.zeros(0)

    # Canonicalise to i < j, then deduplicate (top-k is not symmetric).
    lo = torch.minimum(r, c)
    hi = torch.maximum(r, c)
    flat = lo.to(torch.int64) * L + hi.to(torch.int64)
    uniq, inverse = torch.unique(flat, return_inverse=True)
    # Keep the highest probability seen for each undirected pair.
    best = torch.zeros(uniq.numel(), dtype=p.dtype, device=p.device)
    best.scatter_reduce_(0, inverse, p, reduce="amax", include_self=False)
    u_lo = (uniq // L).to(torch.int32) + offset
    u_hi = (uniq % L).to(torch.int32) + offset
    return torch.stack([u_lo, u_hi], dim=0), best


def chunk_context_score(length: int, start: int, end: int) -> torch.Tensor:
    """
    Per-residue score for "how good is this chunk for this residue", over
    ``[start, end)`` of a protein of ``length`` residues.

    For residue ``i`` the score is the fraction of its **real** neighbouring
    context that the chunk retains, taking the worse of the two sides::

        kept_left  = (i - start) / i                  (1.0 when i is the N terminus)
        kept_right = (end - 1 - i) / (length - 1 - i) (1.0 when i is the C terminus)
        score      = min(kept_left, kept_right)

    A residue is therefore assigned to the chunk that discards the least of what
    actually surrounds it. The naive alternative - distance from the nearest
    chunk boundary - is blind to context the chunk never covered, which is how a
    short tail chunk could win a residue away from a chunk holding eight times
    more of its neighbourhood (R1-26).

    Reaching a true sequence terminus is not truncation, so that side scores a
    full 1.0 rather than being penalised for context that does not exist.
    """
    idx = torch.arange(start, end, dtype=torch.float32)
    avail_left = idx
    avail_right = (length - 1) - idx
    kept_left = idx - start
    kept_right = (end - 1) - idx
    ones = torch.ones_like(idx)
    ratio_left = torch.where(avail_left > 0, kept_left / avail_left.clamp(min=1.0), ones)
    ratio_right = torch.where(avail_right > 0, kept_right / avail_right.clamp(min=1.0), ones)
    return torch.minimum(ratio_left, ratio_right)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EsmEncoder(nn.Module):
    """
    Frozen ESM-2 wrapper: sequence -> (per-residue embeddings, sparse contacts).

    The encoder is never trained. It is instantiated only by
    ``scripts/precompute_embeddings.py`` and by the inference layer; the
    training loop reads the cache and never holds ESM in memory, which is what
    lets the trainable model use large batches on a 16 GB card.
    """

    def __init__(self, esm_model_name: str = DEFAULT_ESM, device: str = "cpu"):
        super().__init__()
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.name = esm_model_name
        self.dim = ESM_DIMS.get(esm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        # Single EsmForMaskedLM gives hidden states, LM logits AND contact_head
        # in ONE forward pass (previously EsmModel + predict_contacts ran the
        # entire 33-layer 650M transformer twice per protein). Eager attention
        # is required so output_attentions=True emits attention maps for
        # contact_head across transformers 4.x and 5.x.
        self.model = EsmForMaskedLM.from_pretrained(
            esm_model_name, attn_implementation="eager"
        )
        self.esm = self.model.esm
        self._mlm = self.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = torch.device(device)
        self.to(self.device)
        if self.dim is None:
            self.dim = int(self.esm.config.hidden_size)
        self.tap_layers = tap_layer_indices(int(self.esm.config.num_hidden_layers))
        self._aa_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_ORDER],
            dtype=torch.long,
            device=self.device,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_spans(length: int) -> list[tuple[int, int]]:
        """Overlapping [start, end) spans covering a sequence of ``length``."""
        if length <= MAX_CONTEXT_RESIDUES:
            return [(0, length)]
        stride = MAX_CONTEXT_RESIDUES - CHUNK_OVERLAP
        spans = []
        start = 0
        while start < length:
            end = min(start + MAX_CONTEXT_RESIDUES, length)
            spans.append((start, end))
            if end == length:
                break
            start += stride
        return spans

    @torch.no_grad()
    def _forward_chunk(
        self, seq: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-pass ESM-2 chunk forward returning:
          ``(emb (L,d), layer_embs (K,L,d), wt_logprobs (L,20), site_entropy (L,), cm (L,L))``
        """
        toks = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.model(
            **toks,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        emb = out.hidden_states[-1][0, 1:-1]
        layer_embs = torch.stack(
            [out.hidden_states[i][0, 1:-1] for i in self.tap_layers], dim=0
        )

        # Wild-type marginal log-probabilities over the 20 standard amino acids
        # plus normalized site Shannon entropy H_i / log(20) in [0, 1].
        logits_aa = out.logits[0, 1:-1].index_select(-1, self._aa_token_ids).float()
        wt_logprobs = torch.log_softmax(logits_aa, dim=-1)
        probs = wt_logprobs.exp()
        site_entropy = -(probs * wt_logprobs).sum(dim=-1) / math.log(len(AA_ORDER))

        # Exact EsmModel.predict_contacts math from the ALREADY computed attentions.
        attns = torch.stack(out.attentions, dim=1)
        amask = toks["attention_mask"]
        attns = attns * amask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns = attns * amask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        cm = self.esm.contact_head(toks["input_ids"], attns)[0]

        return (
            emb.float().cpu(),
            layer_embs.float().cpu(),
            wt_logprobs.float().cpu(),
            site_entropy.float().cpu(),
            cm.float().cpu(),
        )

    @torch.no_grad()
    def encode(self, seq: str, top_k: int = 24, min_sep: int = 3) -> dict:
        """
        Encode one protein in a single ESM-2 pass per chunk.

        Returns ``{"seq", "emb" (L, d) fp16, "layer_embs" (K, L, d) fp16,
        "wt_logprobs" (L, 20) fp16, "site_entropy" (L,) fp16,
        "contact_edges" (2, E) int32, "contact_probs" (E,) fp16,
        "esm_model", "dim", "tap_layers"}``.
        """
        L = len(seq)
        if L == 0:
            raise ValueError("cannot encode an empty sequence")

        spans = self._chunk_spans(L)
        if len(spans) == 1:
            emb, layer_embs, wt_logprobs, site_entropy, cm = self._forward_chunk(seq)
            edges, probs = sparse_topk_contacts(cm, top_k, min_sep)
        else:
            k_layers = len(self.tap_layers)
            emb = torch.zeros((L, self.dim), dtype=torch.float32)
            layer_embs = torch.zeros((k_layers, L, self.dim), dtype=torch.float32)
            wt_logprobs = torch.zeros((L, len(AA_ORDER)), dtype=torch.float32)
            site_entropy = torch.zeros((L,), dtype=torch.float32)
            best_score = torch.full((L,), -1.0)
            all_edges, all_probs = [], []
            for start, end in spans:
                c_emb, c_layers, c_lp, c_ent, c_cm = self._forward_chunk(seq[start:end])
                score = chunk_context_score(L, start, end)
                take = score > best_score[start:end]
                emb[start:end][take] = c_emb[take]
                layer_embs[:, start:end][:, take] = c_layers[:, take]
                wt_logprobs[start:end][take] = c_lp[take]
                site_entropy[start:end][take] = c_ent[take]
                best_score[start:end][take] = score[take]

                e, p = sparse_topk_contacts(c_cm, top_k, min_sep, offset=start)
                if e.numel():
                    all_edges.append(e)
                    all_probs.append(p)
            if all_edges:
                edges = torch.cat(all_edges, dim=1)
                probs = torch.cat(all_probs)
                flat = edges[0].to(torch.int64) * L + edges[1].to(torch.int64)
                uniq, inverse = torch.unique(flat, return_inverse=True)
                best = torch.zeros(uniq.numel(), dtype=probs.dtype)
                best.scatter_reduce_(0, inverse, probs, reduce="amax", include_self=False)
                edges = torch.stack(
                    [(uniq // L).to(torch.int32), (uniq % L).to(torch.int32)], dim=0
                )
                probs = best
            else:
                edges = torch.zeros((2, 0), dtype=torch.int32)
                probs = torch.zeros(0)

        return {
            "seq": seq,
            "emb": emb.half(),
            "layer_embs": layer_embs.half(),
            "wt_logprobs": wt_logprobs.half(),
            "site_entropy": site_entropy.half(),
            "contact_edges": edges.to(torch.int32),
            "contact_probs": probs.half(),
            "esm_model": self.name,
            "dim": int(self.dim),
            "tap_layers": list(self.tap_layers),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def masked_marginals(self, seq: str, positions: list[int]) -> dict[int, torch.Tensor]:
        """
        Log-softmax over the amino-acid vocabulary at each requested position,
        with that position masked. Reuses ``self.model`` rather than loading a
        second copy of ESM-2.
        """
        out: dict[int, torch.Tensor] = {}
        mask_id = self.tokenizer.mask_token_id
        for pos in positions:
            toks = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            ids = toks["input_ids"].to(self.device)
            ids[0, pos + 1] = mask_id                     # +1 for CLS
            logits = self._mlm(input_ids=ids,
                               attention_mask=toks["attention_mask"].to(self.device)).logits
            out[pos] = torch.log_softmax(logits[0, pos + 1], dim=-1).cpu()
        return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class EncodingStore:
    """
    Read/write access to the precomputed encoder cache.

    An optional in-memory LRU sits in front of :meth:`get`. A mutation dataset
    calls it once per ROW while the cache is keyed by PARENT, so on MegaScale the
    same file is re-read from disk and re-deserialised on the order of 100 times
    per epoch per protein. Round-2 review, R1-17: the round-1 commit claimed
    "built graphs cached" and no cache of any kind existed - a repo-wide grep
    found no ``lru_cache`` in ``src/`` or ``scripts/``. This is the cache that
    was worth having; graph construction itself is 2.81 ms and was already
    made cheap by the vectorised ``_cap_degree``, which is the half of that
    commit that was real.

    It is **off by default**, because it trades a bounded amount of RAM per
    DataLoader worker for the saving and that trade should be made deliberately:
    one record is roughly ``L * dim * 2`` bytes (about 1.5 MB for a 600-residue
    protein at dim 1280), so ``cache_records=32`` costs ~50 MB per worker. Set
    ``record_cache_size`` in the config to enable it.
    """

    def __init__(self, cache_dir: str, cache_records: int = 0):
        self.cache_dir = cache_dir
        self.cache_records = int(cache_records)
        self._memo: "OrderedDict[str, dict]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def path(self, seq: str) -> str:
        return os.path.join(self.cache_dir, seq_key(seq) + ".pt")

    def has(self, seq: str) -> bool:
        return os.path.exists(self.path(seq))

    #: Whether ``torch.load(mmap=True)`` works for this cache. Probed once, on
    #: the first read, and cached: a file written by an older torch without the
    #: zipfile serialisation cannot be mapped and must be read whole.
    _mmap_ok: bool | None = None

    def _load(self, seq: str) -> dict:
        """
        Read one cached record, memory-mapped where the format allows it.

        A record holds the WHOLE protein - ``emb`` is ``L x 1280`` fp16, 1.7 MB
        at L=659 and up to 12 MB for the longest ClinVar proteins - while a
        mutation sample uses only a ``window``-long slice around its site.
        Reading the file whole to take 256 rows of it cost 42.6 ms per record
        here, which at batch 64 is 2.7 s of pure deserialisation per batch: the
        GPU sat idle at 0% between bursts, and a ClinVar batch took 1.8 s
        against a stability batch's 31 ms.

        Mapping the file instead faults in only the pages the window touches:
        0.4 ms per record on the same data, a 113x reduction. Every consumer in
        ``build_sample_graph`` copies out of the record (``.float()`` on a
        slice, ``.long()`` and boolean masks in ``window_edges``), so nothing
        writes through the mapping.
        """
        path = self.path(seq)
        if self._mmap_ok is not False:
            try:
                rec = torch.load(path, map_location="cpu", weights_only=False,
                                 mmap=True)
                EncodingStore._mmap_ok = True
                return rec
            except Exception:
                # Legacy pickle format, or a filesystem that will not map.
                EncodingStore._mmap_ok = False
        return torch.load(path, map_location="cpu", weights_only=False)

    def get(self, seq: str) -> dict:
        if self.cache_records <= 0:
            return self._load(seq)
        key = seq_key(seq)
        hit = self._memo.get(key)
        if hit is not None:
            self.hits += 1
            self._memo.move_to_end(key)
            return hit
        self.misses += 1
        rec = self._load(seq)
        self._memo[key] = rec
        while len(self._memo) > self.cache_records:
            self._memo.popitem(last=False)
        return rec

    def cache_stats(self) -> dict:
        """Hit rate of the in-memory record cache, for a run manifest."""
        total = self.hits + self.misses
        return {
            "enabled": self.cache_records > 0,
            "capacity_records": self.cache_records,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def put(self, rec: dict) -> str:
        """Write one record atomically, so a crash never leaves a partial file."""
        os.makedirs(self.cache_dir, exist_ok=True)
        dst = self.path(rec["seq"])
        tmp = dst + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, dst)
        return dst

    # -- manifest ----------------------------------------------------------
    # A cache directory is a pile of opaque .pt files, so nothing downstream can
    # tell what produced it: which encoder, at what contact settings, and - the
    # one that silently changes what the model is - how many parents carried a
    # real coding sequence. Round-1 review R1-05 turned on exactly that
    # invisibility: E_mech was empty for a whole run and no artifact said so.
    # The manifest makes coverage a recorded number that train.py copies into
    # metrics.json, so a claim about the mechanistic stream can be checked
    # against the cache the run actually read.
    MANIFEST = "manifest.json"

    def manifest_path(self) -> str:
        return os.path.join(self.cache_dir, self.MANIFEST)

    def manifest(self) -> dict:
        """The cache manifest, or ``{}`` if the cache predates one."""
        try:
            with open(self.manifest_path(), encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def write_manifest(self, entries: dict) -> str:
        """Merge ``entries`` into the manifest and write it atomically."""
        merged = self.manifest()
        merged.update(entries)
        os.makedirs(self.cache_dir, exist_ok=True)
        tmp = self.manifest_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path())
        return self.manifest_path()

    def mech_coverage(self, seqs, sample: int | None = None) -> dict:
        """
        Measure the mechanistic stream's coverage **by reading the cache**.

        Round-2 review, R2-01: ``cds_coverage`` was derived from the input
        table - how many rows carried a CDS column - and written into the
        manifest as though it described the cache. Those are different numbers
        whenever precompute skips an already-cached parent, which is the
        ordinary path (``--cds-source`` is optional, so the first cache is
        usually built without one). The reviewer made the manifest report
        ``cds_coverage: 1.0`` for a cache in which every record had
        ``mech_valid = 0``. A number that reaches a methods section must be
        measured on the artifact it describes, so this opens the records that
        will actually be read and counts them.

        Returns counts plus ``cds_coverage``, the fraction of cached records
        whose mechanistic channels are actually populated. A record is counted
        as active only when its stored ``mech`` tensor carries a set
        ``mech_valid`` channel - not when it merely remembers a CDS string.

        ``sample`` bounds how many records are opened: a 20k-parent cache would
        otherwise cost 20k deserialisations just to describe itself. The result
        says whether it is a sample, so the number is never read as a census.
        """
        from src.features import CH_VALID

        seqs = list(seqs)
        sampled = sample is not None and 0 < sample < len(seqs)
        if sampled:
            # Spread across the cache rather than the first N, which would
            # over-represent whatever the input table happened to list first.
            step = max(1, len(seqs) // sample)
            seqs = seqs[::step][:sample]
        present = active = has_cds_flag = unreadable = 0
        for seq in seqs:
            if not self.has(seq):
                continue
            present += 1
            try:
                rec = self.get(seq)
            except Exception:
                unreadable += 1
                continue
            if rec.get("has_cds"):
                has_cds_flag += 1
            mech = rec.get("mech")
            if mech is not None and mech.numel() and float(mech[:, CH_VALID].max()) > 0:
                active += 1
        return {
            "parents_examined": len(seqs),
            "parents_present_in_cache": present,
            "parents_unreadable": unreadable,
            "cds_flagged_in_cache": has_cds_flag,
            "cds_active": active,
            "cds_coverage": round(active / present, 4) if present else 0.0,
            "cds_coverage_measured_on": "cached records (mech[:, CH_VALID].max() > 0)",
            "cds_coverage_is_a_sample": bool(sampled),
            "cds_coverage_sample_limit": sample if sampled else None,
        }

    def coord_coverage(self, seqs, sample: int | None = None) -> dict:
        """
        Measure 3D coordinate coverage on the cached records themselves
        (GEOMETRIC_PLAN.md §2.4, mirroring :meth:`mech_coverage`).
        """
        seqs = list(seqs)
        sampled = sample is not None and 0 < sample < len(seqs)
        if sampled:
            step = max(1, len(seqs) // sample)
            seqs = seqs[::step][:sample]
        present = active = 0
        for seq in seqs:
            if not self.has(seq):
                continue
            present += 1
            try:
                rec = self.get(seq)
            except Exception:
                continue
            cv = rec.get("coord_valid")
            if rec.get("coords") is not None and cv is not None and float(cv.max()) > 0:
                active += 1
        return {
            "parents_examined": len(seqs),
            "parents_present_in_cache": present,
            "coords_active": active,
            "coord_coverage": round(active / present, 4) if present else 0.0,
            "coord_coverage_is_a_sample": bool(sampled),
        }

    def v2_coverage(self, seqs, sample: int | None = None) -> dict:
        """Measure presence of v2 single-pass features (``layer_embs``, ``wt_logprobs``)."""
        seqs = list(seqs)
        sampled = sample is not None and 0 < sample < len(seqs)
        if sampled:
            step = max(1, len(seqs) // sample)
            seqs = seqs[::step][:sample]
        present = has_layers = has_logprobs = 0
        for seq in seqs:
            if not self.has(seq):
                continue
            present += 1
            try:
                rec = self.get(seq)
            except Exception:
                continue
            if rec.get("layer_embs") is not None:
                has_layers += 1
            if rec.get("wt_logprobs") is not None:
                has_logprobs += 1
        return {
            "parents_present_in_cache": present,
            "layer_embs_coverage": round(has_layers / present, 4) if present else 0.0,
            "wt_logprobs_coverage": round(has_logprobs / present, 4) if present else 0.0,
        }

    @staticmethod
    def projected_bytes(total_residues: int, dim: int, top_k: int, include_v2: bool = False) -> int:
        """
        Disk footprint for a planned cache.

        Embeddings dominate: ``residues x dim x 2`` bytes at fp16. Edges add
        ``residues x top_k x (8 + 2)`` for the int32 pair plus the fp16
        probability. With ``include_v2=True``, adds 4 tapped layers plus the
        20-AA WT log-probabilities and site entropy.
        """
        base = total_residues * (dim * 2 + top_k * 10)
        if include_v2:
            base += total_residues * (N_TAP_LAYERS * dim * 2 + 21 * 2)
        return base


__all__ = [
    "ESM_DIMS", "DEFAULT_ESM", "MAX_CONTEXT_RESIDUES", "N_TAP_LAYERS",
    "tap_layer_indices",
    "EsmEncoder", "EncodingStore", "seq_key", "sparse_topk_contacts",
    "chunk_context_score",
]
