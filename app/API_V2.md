# Holo-GeoN web API, v2 additions

Backend only (`app/backend/`); the React frontend has not been updated to use any of this yet.
Every addition **refuses with the reason** (`503` capability missing, `400` bad input) when it
cannot be served honestly. Tested live in `tests/test_app_v2.py` (random weights, stub encoder:
the contract and the invariants, never accuracy).

## Honesty rules the UI must keep

* No checkpoint => `503` with the reason; there is no approximation and no demo mode.
* The default model is a **structure-free predicted-contact graph**. Its only symmetry is
  wild-type/mutant swap equivariance; it is **not** SE(3)- or E(3)-equivariant. Only the optional
  3D checkpoint is **SE(3)-invariant**. Every payload carries `mode`, `uses_coordinates`, `symmetry`.
* Untrained heads are refused (`/api/idr`, and now `/api/epistasis`), never run.
* `uncertainty_calibrated` is the calibration run's verdict. It is `false` for the 3D checkpoint and
  for epistasis (calibration was measured on single mutants only).

## `GET /api/health` -> new `capabilities`

```
capabilities.epistasis    {available, reason, source, uncertainty}
capabilities.cds          {available, reason}            # false without ViennaRNA
capabilities.invariant_3d {available, reason, weights_path}   # env HOLOGEON_WEIGHTS_3D
capabilities.streaming    {available, endpoints}
capabilities.modes        {structure_free: "<symmetry statement>", invariant_3d: "<...>"}
```
Existing keys are unchanged. Gate every new control on these flags and show `reason` when false.

## `POST /api/ddg`, `POST /api/scan` -> new optional fields

| field | meaning |
|---|---|
| `cds` | real coding sequence (DNA/RNA, FASTA ok, one stop codon ok). Must be a reading frame that encodes the protein exactly (`400` says which codon). `503` if ViennaRNA is missing |
| `mode` | `"structure_free"` (default) or `"invariant_3d"` (needs `structure_pdb` and the 3D checkpoint) |
| `structure_pdb` | PDB text of this protein; B-factor read as pLDDT. Ignored by the default mode |
| `compare_modes` | `/api/ddg` only: also run the other mode and return `comparison` |

`/api/ddg` response adds `mode`, `uses_coordinates`, `symmetry`, `structure` (3D only: alignment,
`mean_plddt`, `low_confidence_positions`) and:

```
cds: {supplied: false}
cds: {supplied: true, frame_verified: true, n_codons, stop_codon_stripped,
      with_cds: {ddg, sigma, ci_low, ci_high, verdict}, without_cds: {...}, delta_ddg}
comparison: {structure_free: {...}, invariant_3d: {...}, delta_ddg_3d_minus_structure_free, structure}
```
With a CDS the headline `ddg` is the with-CDS value; `without_cds` equals the plain request. The model
is trained with CDS dropout, so both paths are in-distribution: show them side by side.

`/api/scan` response adds `mode`, `uses_coordinates`, `symmetry`, `cds`, and
`timing: {encode_ms, scan_ms, total_mutations, muts_per_sec}` (for the live A100 demo).

## `POST /api/epistasis` (new)

```
request   {wt_sequence, pairs: ["V5A+L8P", ...], cds?}      # exactly two sites per pair, <= 200 pairs
response  {length, n_pairs, mode, uses_coordinates, symmetry, cds,
           epistasis_head: {trained, source, contact_gated, gate_input},
           pairs: [{pair, sites, ddg_total, ddg_additive, epistasis_dddg, sigma, ci_low, ci_high,
                    contact_prob, is_contact, seq_separation, pair_geo:{...}}],
           uncertainty_calibrated: false, uncertainty_note, ci_source, timing}
```
`ddg_total = ddg_additive + epistasis_dddg`. Pairs are reported in ascending site order (the head is
exactly symmetric under swapping the two sites). The interval is for `ddg_total` only. `503` unless the
checkpoint's epistasis head was trained (`scripts.train_epistasis`; a single-mutant checkpoint carries it
at its near-zero initialisation). Structure-free only: the gate is the predicted ESM-2 contact graph.

## Streaming: `POST /api/scan/stream`, `POST /api/epistasis/stream`

Same request bodies as the blocking endpoints; `Content-Type: text/event-stream`. Use `fetch` +
`ReadableStream` (the browser `EventSource` cannot POST). Anything refusable is refused **before** the
stream starts as an ordinary HTTP error.

```
event: start    data: {length, total, positions:[first,last]?, aa_order?, mode, cds, ...}
event: chunk    data: {done, total, elapsed_s, eta_s, positions, columns}   # scan: new 20-value columns
                data: {done, total, elapsed_s, eta_s, pairs}                # epistasis: new pair rows
event: result   data: <exactly the blocking endpoint's response>
event: error    data: {status, detail}                                      # failure after the start
```
A chunk is one batch of `HOLOGEON_SCAN_BATCH` sites (default 32). The streamed result equals the blocking one.
Finished streams are stored in `/api/history` (kind `scan` / `epistasis`); a supplied PDB is stored as its size.

## `POST /api/compare` -> new pLDDT fields

`plddt_mean`, `plddt_bands` (`very_high_gt_90`, `confident_70_90`, `low_50_70`, `very_low_le_50`),
`low_confidence_positions` (1-based, pLDDT < 70), `low_confidence_in_scan` (aligned with
`min_ddg_per_position`) and `plddt_note`. pLDDT describes the AlphaFold model drawn, not the prediction:
the structure-free model reads no coordinates.

## Frontend to do (not started)

1. Epistasis tab on `/api/epistasis/stream` (predicted vs additive, contact probability, per-pair table).
2. Optional CDS box on the ddG and scan tabs, showing `with_cds` / `without_cds` / `delta_ddg`.
3. Mode toggle (structure-free / 3D) driven by `capabilities`, with the `symmetry` line and `comparison` delta.
4. Progress bar and live heatmap fill from the `chunk` events.
5. 3Dmol: colour by `plddt` (AlphaFold bands), mark `low_confidence_positions`, show `plddt_note`.
6. Keep the existing banners: model-not-ready, uncalibrated uncertainty, untrained heads.

Verify the frontend on a Kaggle CPU session or the A100 VM (`npm run build` is too heavy for the local PC).
