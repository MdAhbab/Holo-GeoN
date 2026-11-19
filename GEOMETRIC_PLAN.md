# GEOMETRIC_PLAN.md - using the coordinates we already have

Prepared 2026-11-04 on `main` (`bbd620e`). Companion to `BENCHMARK_PLAN.md`; read that first.

The question this answers: **`DATASETS/DATASETS/dataset/mega_scale_cdna/AlphaFold_model_PDBs/`
holds 862 backbone models, so can Holo-GeoN use real structure properly rather than ESM-2's
predicted contacts?** Yes, and the data is better placed for it than expected. But the right
change is smaller than "add an equivariant network", and the recommendation is explicitly
**not** to add one.

Numbers marked MEASURED were read off the files on 2026-11-04 and independently re-checked.
Numbers marked PROJECTED hang off the unmeasured 40 ms/step G4 assumption.

---

## 0. What the data actually looks like (MEASURED)

| fact | value | consequence |
|---|---|---|
| MegaScale parent length | 32-72 residues, mean 52.6, median 51 | **`window: 256` is never binding** for the stability-only benchmark. Zero of 412 parents exceed it |
| structured parents | 373 of 412 (matches the 373 maps in `data/true_contacts/`) | the contact maps and the PDBs cover the same set |
| **test split** | **37/37 parents, 33,226/33,226 rows (100%)** | a coordinate arm is evaluable on the *entire* test set with no coverage caveat |
| train split | 299/322 parents, 284,898/305,995 rows (93.1%) | |
| val split | 37/53 parents, 35,674/49,847 rows (71.6%) | early stopping runs on a 28.4% smaller val set in a structured-subset arm |
| parents with no structure | 39: `EA*` 10, `GG*` 13, `XX*` 14 (all 47 aa) and 2 `v*` (66 and 70 aa); **23 train, 16 val, 0 test** | one nameable class of designed constructs, not scattered noise |

**The length finding changes the cost plan.** The 33.8 ms/step local measurement came from a
ClinVar-inclusive config whose proteins do reach 256 residues. At L<=72 the stability-only step
should be materially cheaper, so 40 ms on the G4 is probably pessimistic. Wave 0 may re-project
the budget **downward**; `--plan` must be re-run before concluding anything is unaffordable.

---

## 1. Recommendation: an invariant coordinate path, not an equivariant one

**Do not add EGNN, GVP, e3nn or Equiformer.** The reasoning, in order of weight:

1. **DDG is a scalar, i.e. an E(3)-invariant target, and for that target class the question is
   settled theory.** Villar et al. (NeurIPS 2021, arXiv:2106.06610) show from the first
   fundamental theorem of invariant theory for O(d) that a function of vector inputs is invariant
   **iff** it factors through their pairwise inner products. Distances plus local frames are
   therefore a *complete* basis for an invariant scalar, not an approximation to an equivariant
   one. Invariant architectures that are provably E(3)-complete exist (DimeNet, SphereNet,
   GemNet, `k`-DisGNN for `k>=2`), and ComENet/SphereNet outperform equivariant GNNs on
   molecular tasks (arXiv:2402.04836, arXiv:2302.05743).
2. **What equivariance empirically buys is sample and compute efficiency, not accuracy ceiling.**
   Brehmer et al., "Does equivariance matter at scale?" (TMLR 2025, arXiv:2410.23179): the
   advantage appears at low data and augmentation closes it given enough epochs. AlphaFold3
   dropped AF2's equivariant IPA for a non-equivariant diffusion module that learns the symmetry
   from augmentation. We have 305,995 training rows, which is not the low-data regime.
3. **The DDG state of the art is invariant.** ThermoMPNN is ProteinMPNN plus a stability head,
   and ProteinMPNN's geometry is RBF-encoded pairwise backbone distances to 48 neighbours
   expressed in local frames. SPURS is ESM-2 plus a ProteinMPNN adapter. RaSP is a 3D CNN and is
   not even invariant, and still reaches 0.71/0.67.
4. **It would be spent in the wrong place in this model.** Node features are 1280-dim frozen
   ESM-2 scalars with a residual adding them back; the backbone is **two** GATv2 layers over
   <=72 nodes; the output is a scalar. Tensor products would cost an order of magnitude to impose
   a symmetry bias on the thinnest part of the network.

**Adding coordinates does not earn the word "equivariant".** An invariant model is not an
equivariant one. `README.md` and `BENCHMARK_PLAN.md` section 2 stay as written: never claim
SE(3) for Holo-GeoN. What the paper may claim is *invariance*, and section 2.4 below makes that
claim mechanically checkable instead of asserted.

**Correction to `BENCHMARK_PLAN.md` section 2 (now applied).** The earlier line "a survey found
no prominent open DDG baseline built on EGNN, GVP, e3nn or Equiformer" was too strong. One
exists: Boyer, Money-Kyrle and Bent, *Predicting protein stability changes under multiple amino
acid substitutions using equivariant graph neural networks*, arXiv:2305.19801 (30 May 2023),
E(3)-equivariant, trained on MegaScale. Verified on arXiv: no code, no released weights, no
stated venue, and the abstract itself reports only "promising initial results". The conclusion
survives - no equivariant comparator can be run off the shelf - but the claim had to be weakened
to what is true.

---

## 2. Implementation, file by file

### 2.1 New `src/geometry.py`

Factor the PDB parsing **out of `scripts/build_contact_maps.py`**, which already holds a
validated PDB-to-sequence path (15% mismatch budget, unique-best-offset, refuses rather than
guesses). Reuse it; do not rewrite it.

* `parse_backbone` - generalize from {CA, CB} to {N, CA, C, CB}; read pLDDT from the B-factor column
* `align_to_sequence` - lifted verbatim
* `local_frames`, `dihedrals`, `rbf`, `knn_edges`, `edge_geometry`

Edge channels (23): 16 RBF bins of CB-CB over 2-22 A, the CA->CA unit vector **in residue i's
local frame** (3), the quaternion of `R_i^T R_j` (4).
Node channels (9): dihedral sin/cos (6), neighbour count (1), pLDDT (1), `coord_valid` (1).

### 2.2 `src/graphs.py` - a third builder, not an extension

`build_structure_graph` returns `3 + 23 = 26`-wide edge attrs, keeping the same first three
channels as the other two streams. **Do not fold coordinates into `E_geo` in place**: it is the
smaller diff, but it destroys the only ablation worth running (section 3).

### 2.3 `src/backbone.py`, `src/model.py`

`DualStreamGNN` -> `MultiStreamGNN` (keep the old name as an alias); `_Stream` takes `edge_dim`
as a parameter instead of reading a module constant; `self.fuse` is sized from the **count** of
enabled streams, so an existing checkpoint's `model_kwargs` (geo + mech) still reproduces
`2*out_dim` exactly. `HoloGeoN` gains `use_str_stream: bool = False`, which keeps every existing
checkpoint loadable.

### 2.4 Caching: `scripts/attach_coords.py`, a separate CPU script

Not a flag on `precompute_embeddings.py`: its staleness rule re-encodes any parent whose record
lacks a feature, so adding `--structure-dir` there would trigger a full ESM-2 re-encode of 588
proteins to attach data that costs nothing. The new script opens each cached record, writes
`coords` / `plddt` / `coord_valid` / `structure_id` / `structure_mismatches`, and rewrites
atomically through `store.put`. About 600 records: seconds, local, no GPU. Cached records already
carry `mech`, `has_cds` and `cds`, so coordinates belong in that record.

Add `EncodingStore.coord_coverage()` mirroring `mech_coverage`, measured on the records training
actually reads (the R2-01 lesson: measure coverage on the artifact, not the source).

**The one assertion that must be a hard failure:** `attach_coords.py` verifies that the amino acid
at the mutation site in the coordinate record matches the table's `wt_aa`, and exits non-zero on
mismatch. An off-by-one yields no error and meaningless geometry - the same silent class as the
sign convention, the reading frame and the contact alignment, all three of which have already
bitten this repo.

**The load-bearing new test:** a random rotation and translation of the coordinates must leave
every edge feature, every node channel and the DDG output identical to fp32 tolerance. That is
what makes "invariant" checkable in the paper rather than asserted.

### 2.5 Missing structures, and two files that break silently

Reuse the existing `mech_valid` convention as `str_valid` rather than inventing a second one:
self-loops only, zeroed features, counted in the load report. FireProtDB coordinate numbers are
**not reported at all** until `scripts/fetch_alphafold.py` has run locally - not reported as a
degraded number.

* **`scripts/budget.py`** computes an analytic memory bound from real tensor shapes. A third edge
  set changes them, and the README's 5.35 GB figure becomes wrong if this is skipped.
* **`scripts/build_contact_maps.py`** must import the refactored parser, with a test that an
  existing `true_contacts/*.npy` reproduces byte for byte. Those 373 maps are in use for
  precision@L.

`--no-geo-stream` keeps its exact current meaning and becomes *more* informative: with the
coordinate stream on it reads "replace predicted contacts with real coordinates" instead of
"lose all topology".

---

## 3. The ablation that answers the real question

State the confound once, in the methods: ESM-2's contact head is supervised against real
contacts, so the axis is **explicit coordinates at inference** versus **coordinates distilled
into frozen encoder weights**. Neither arm is structure-naive in the training-history sense.
Anything stronger will be attacked.

Arms - same split, rows, epochs, seeds and readout; only the edge set and features vary:

| arm | edges | features |
|---|---|---|
| **A** `full` | ESM-2 predicted contacts | current 3-channel |
| **B** `str_only` | coordinate k-NN | full 26-channel geometry |
| **C** `both_streams` | both | both |
| **D** `str_topology_only` | coordinate k-NN | 3-channel only |
| **E** `true_contact_edges` | **real C-beta contact maps** | current 3-channel |

Read it as: **B-A** = what coordinates buy end to end. **D-A** = how much is just a better edge
set. **B-D** = how much is continuous geometry a contact graph cannot express. **E-A** = how much
of A's deficit is ESM-2 contact *error*, measurable with no coordinates in the model at all.

**Arm E is the highest value per GPU-hour and the arm a reviewer will ask for.** It is a loader
switch: 373 true maps are already on disk under the same key, covering 37/37 test parents. For
~50-residue single-domain proteins an 8 A contact map nearly determines the fold, so `E ~ B` is a
live possibility and would be the most interesting result in the paper.

**Subset rule, stated twice because someone will try to save 3 CU:** every grid arm runs on the
structured subset (284,898 / 35,674 / 33,226). The Wave 1 headline `full` model runs on the
complete split and **cannot be substituted for arm A**, or `B-A` mixes an architecture change
with a 7% smaller train set and a differently composed early-stopping val set.

Free diagnostics: precision@L per test parent (the script exists), arm A's error stratified by
it, the pLDDT distribution, and arm B stratified by pLDDT.

---

## 4. Positioning: keep both arms, ship one model

**Holo-GeoN stays the structure-free model.** The coordinate path ships as a controlled ablation
arm, not a second product, and not in the abstract's main claim. The second contribution is one
nobody in this field has delivered cleanly:

> holding architecture, split, budget and readout fixed, and varying only whether coordinates
> enter at inference, coordinates are worth X.

Every existing structure-versus-sequence comparison in DDG also changes the architecture, the
training set and the split at once. This one does not.

Why not pick one side. Structure-free only invites "you had 862 AlphaFold models in the dataset
directory, why not use them?", and the answer now sounds evasive. Structure-based only makes
Holo-GeoN a weaker ThermoMPNN that has to beat 0.754/0.725 to publish at all - a two-layer GATv2
on a 30% identity split against a tuned ProteinMPNN transfer model is not a bet worth taking, and
losing leaves nothing.

The decisive property: *"we vary only whether coordinates enter at inference"* is publishable in
**both** directions. Coordinates buy little -> the strongest version of the structure-free claim,
with the mechanism (`E-A`) attached. Coordinates buy a lot -> the first clean measurement of how
much, and the structure-free number still stands at a lower input requirement. **Pre-commit the
wording before the run**; that is how a null avoids being spun afterwards.

---

## 5. Cost inside the 90 usable CU (PROJECTED off 40 ms/step)

The coordinate stream is projected at **+25% to +60% per step**. On a 50-residue protein a 12 A
k-NN graph is near-complete, so this stream behaves more like a small geometric transformer than a
sparse GNN: cheap in absolute terms, roughly 2-3x the contact stream's edge count. Planned at
+30%. **Measure it in Wave 0.**

* **Wave 0 (~11 CU, unchanged):** precompute + `full_s42` + external, **plus a ~300-step
  three-stream timing probe (~0.05 h)**. Then `--plan` again.
* **Wave 1 (untouched):** the existing headline matrix, 4.5 h / 40 CU cumulative.
* **Wave 2 as designed does not fit.** Six arms x 3 seeds x 6 epochs = 6.75 h / 60 CU, which is
  100 CU against 90. Dropping C (-11 CU, least informative per hour) and the `no_ssm` probe
  (-9 CU, already in the deferred Kaggle grid and not about coordinates) leaves A, B, D, E x 3
  seeds = 40 CU, total 80 of 90. **The competitor retrains then do not fit at all.**

**Recommended allocation, competitors first, because the competitor table *is* the paper:**

| item | CU |
|---|---|
| Waves 0 + 1 | 40 |
| arms A, B, E x **2 seeds** | 19 |
| competitor retrains | ~29 |
| slack | 2 |
| reserve | 10 |

At two seeds `scripts/ablations.py` refuses a paired test, so report those arms as a single-split
comparison with the cluster bootstrap `score_predictions.py` already computes: honest, still
publishable as a measurement with an interval. **If Wave 0 measures <=30 ms/step - plausible at
L~52, see section 0 - the full three-seed grid and the competitors both fit.**

Cut order for >=60 ms/step, to be written into the manifest so `--plan` enforces it:
StabilityArc -> arm D -> seed 44 -> arm C -> seed 43 -> arm B last. **Prefer dropping a seed over
dropping arm E.** If the budget collapses, keep E at one seed and drop the coordinate stream from
this paper entirely.

**Engineering: about 3 focused days** (geometry module 0.5-1 d; graph/backbone/model 0.5 d;
`src/data.py` plumbing 0.5 d, where the bugs will be; `attach_coords` 0.5 d; arm E 0.25 d; tests
0.5 d; `budget.py` and manifests 0.25 d).

**Sequencing, and this is the one scheduling rule that matters: do not start the geometry work
until one competitor adapter is written, timed on a GPU, and producing a scored prediction file.**
The four adapters are not started and are realistically 4-8 days with the project's largest
variance. The coordinate arm is a good measurement; the competitor table is the paper.

---

## 6. Risks, in order of how much they matter

1. **These are predictions, not crystals** - including for the 33 deposited test parents, because
   the release shipped AlphaFold models. Write "predicted coordinates, explicit at inference",
   never "experimental structures". Real mitigation: we use backbone N/CA/C/CB only, exactly as
   ProteinMPNN and ThermoMPNN do. Free sensitivity check: fetch the 33 RCSB depositions locally
   and report arm B on crystal coordinates for the test set.
2. **30% vs the field's 25% identity gets *worse* with coordinates, not neutral.** The split is
   sequence-identity based; fold similarity survives below 30% identity, and a coordinate model
   can exploit it where a sequence model cannot. The existing MinHash 8-mer audit **cannot see
   fold similarity at all, so coordinates open a leakage route the current audit does not
   detect.** Two fixes, both free and both before Wave 2: a Foldseek or TM-align audit of the 37
   test structures against the 299 structured train parents, reported beside the MinHash number;
   and arm B's advantage stratified by max TM-score. If the advantage concentrates in high-TM
   proteins, "coordinates help" is partly fold memorization. **This is the most important new
   check the coordinate path creates.**
3. **pLDDT**: feed it as a node channel *and* zero `coord_valid` below 50, and say it plays both
   roles. **Do not filter out low-pLDDT proteins** - on 47-72 residue constructs pLDDT tracks
   foldedness, which tracks DDG magnitude, so that filter selects on something close to the label.
   One-seed diagnostic with the pLDDT channel zeroed; if performance moves a lot, the model is
   reading confidence as a stability prior and that must be reported.
4. **AlphaFold2 saw the PDB**, so arm B consumes near-memorized coordinates for the test set. It
   hits every structure-input model in the table equally, so it does not bias the comparison, but
   it makes absolute structure-based numbers optimistic for novel folds. Say so. Conversely, arm B
   as designed introduces **no third-party pretrained weights at all**, which is worth one
   sentence as a methodological advantage. **Explicit recommendation against:** do not initialize
   the coordinate stream from ProteinMPNN features; that imports its PDB training set and makes
   the ThermoMPNN comparison circular.
5. **Val shrinks 28.4%** in the structured subset, so early stopping runs on a different val set.
   The comparison stays clean but the absolute numbers are not cross-quotable with the headline
   arm - the same trap as the ClinVar incomparability in `BENCHMARK_PLAN.md` section 5.
6. **The most likely outcome is a null** (`B ~ A` within seed noise), given that the contact head
   is distilled from structure and a contact map nearly determines a 50-residue fold. That is a
   *good* result for the framing, but pre-commit the wording: "we can exclude an advantage larger
   than X Pearson", X taken from the measured seed spread. At three seeds the detectable effect is
   about 0.02; at two seeds there is no test, only a bootstrap interval.

**Do not do:** e3nn / Equiformer / EGNN / GVP; any renaming toward "SE(3)" or "equivariant";
making the coordinate stream default-on; folding coordinates into `E_geo` in place; starting
before one competitor adapter works.

---

## 7. The smallest defensible version

If only one thing happens, do **arm E**: substitute real C-beta contact edges for ESM-2's
predicted ones, change nothing else, report `E-A`. About 0.25 day and ~2.9 CU per seed, with no
new module, no new dependency, no pLDDT question and no new leakage route. It answers the paper's
actual question - *is ESM-2's contact error what separates structure-free from structure-based?* -
without the model ever taking a coordinate, and it leaves the README's "No 3D coordinates
anywhere" true.

---

## References

* Villar et al., *Scalars are universal*, NeurIPS 2021 - arXiv:2106.06610
* GeoNGNN / `k`-DisGNN completeness - arXiv:2402.04836
* Invariant vs equivariant, ComENet / SphereNet - arXiv:2302.05743
* Brehmer et al., *Does equivariance matter at scale?*, TMLR 2025 - arXiv:2410.23179
* Dieckhaus et al., ThermoMPNN, PNAS 2024 - `10.1073/pnas.2314853121` (ProteinMPNN RBF distance features)
* Boyer, Money-Kyrle, Bent, equivariant DDG - arXiv:2305.19801 (no code, no weights, no stated venue)
