# Web interface: HCD / perception audit (v2.2)

Scope: the React interface in `app/frontend/`, two backend fixes it needed, and the
screenshot capture (`remote/gpu_jobs/a100_ui_capture.py`, `remote/gpu_jobs/l4_ui_recapture.py`).
Model weights, predictions and training code are unchanged.

## Defects found in the A100 screenshots and what changed

| # | Defect (screenshot) | Principle | Fix |
|---|---|---|---|
| 0 | No axis or colorbar titles on ANY chart (all) | Figure legibility | The bundle resolves plotly.js 3.6, which silently drops `title: 'text'`, `titlefont` and `titleside`. All charts now use `{text, font, side}` via `src/plotTheme.ts`; a test forbids the old form. |
| 1 | Two full-width amber banners on every tab, one claiming no calibration run existed (ui_01-06) | Cognitive load; alert fatigue; accuracy | One calm status strip (`components/EpistemicStrip.tsx`): default symmetry, calibration status, untrained heads, plus a Details drawer with the coverage table. Calibration is shown as **measured, not certified**: ECE 0.0265 and 95% coverage 95.9% pass; the 50% band covers 57.4% (gap 0.074 > 0.05 tolerance), so the server's own `well_calibrated` verdict stays false. The run-calibration instruction appears only when no calibration exists. |
| 1b | Per-result amber banner said "not calibrated" next to an interval that was rescaled by 2.122 | Information at the point of decision | `components/IntervalNote.tsx` sits beside the interval and states the measured coverage, the σ scale and the exact failed tolerance. 3D-mode and epistasis intervals keep explicit "not calibrated for this checkpoint / epistasis term" notes. |
| 2 | Ubiquitin landscape clipped at G53 (ui_03) | Figure-ground; completeness | The heatmap is sized to its card (ResizeObserver), with tick stride chosen so labels never collide (all 76 shown at 1440 px), explicit category ranges, ColorBrewer RdBu (red = destabilizing), a titled colorbar, a mean-ΔΔG track and wild-type markers. It scrolls only below 7 px per column. The capture script verifies "fits, M1…G76". |
| 3 | TP53 DBD mutation shown as R82H (ui_02) | Match to the user's mental model | "First residue no." field: type R175H with 94; the API still receives R82H (contract unchanged) and both labels are shown. Live check of the wild-type residue. Same relabelling on the landscape axis. |
| 4 | Black 3Dmol canvas on a white card; legend misaligned; "Holo-GNN" label (ui_04/05) | Figure-ground harmony; naming | Studio-light canvas with an ink outline, framed caption and in-canvas legend overlay (continuous ΔΔG ramp instead of three mislabelled dots), equal-height viewer/PAE row, full-width overlay chart. PAE uses AlphaFold-DB greens, residue-number axes (`pae_stride`), and the stride bug (`n // 200` = 1 while flagged "downsampled") is fixed. Holo-GNN renamed everywhere. |
| 5 | `/api/ddg compare_modes` HTTP 400 on 1UBQ.pdb | Error prevention | Root cause: the MegaScale 1UBQ model has 72 residues, ubiquitin has 76, and the aligner refused any structure shorter than the sequence. A structure covering one contiguous ≥80% span is now placed inside the sequence; uncovered residues get no coordinates and are masked through `coord_valid` (the channel already used for pLDDT < 50). The response reports `sequence_coverage`, `residues_without_coordinates` and `mutation_site_masked`. New 7th screenshot `ui_07_compare_modes_3d.png`. |
| 6 | Epistasis plots loose, colorbar detached, pair list cut at 5 rows, "Distal" for adjacent residues (ui_06) | Gestalt proximity; accuracy | Square parity plot (y = x at 45°), symmetric ΔΔΔG colour scale, legends above plots, unclipped markers, textarea sized to the pairs, denser table. "Distal" → "Non-contact (no E_geo edge)": adjacent residues are not distal. |

## Information-density pass (design critique)
Rule: each fact appears once, at the place it is used. Detail moves to tooltips or disclosures; it is never deleted.
- Status strip: 2 rows → 1 (`Default · structure-free, swap-equivariant only (not SE(3)/E(3)-equivariant)`, `Uncertainty · measured, not certified · 95% CI covers 95.9%`, `4 untrained heads disabled`). ECE, σ × 2.122, per-band gaps and head names sit in the tooltip and the Details drawer.
- ΔΔG result: the five-column row (mutation, σ, CI low, CI high, direction) repeated the headline and is gone. σ joins the headline, and the long symmetry box becomes one muted provenance line. The interval note shrinks from 3 lines to 1.
- Forms: always-on hint paragraphs are replaced by ⓘ tooltips. A hint appears only when it carries information (a numbering translation such as "sent as R82H", or a wild-type mismatch).
- Landscape: six stat chips → one meta line (`1,520 mutations · scan 0.40 s · 3,761 mut/s`). CDS and structure facts appear only when they apply.
- AlphaFold: four pLDDT band chips duplicated the band bar, which now carries its own counts. Explanatory paragraphs are folded into tooltips.
- Epistasis: results (KPIs) come first; the five provenance tiles and the symmetry statement move into a "Provenance & symmetry" disclosure. The interval-scope note stays visible as one line.
- Visual system: bundled IBM Plex Sans + JetBrains Mono (both carry Greek for ΔΔG/σ, so text renders identically on Colab), hairline borders with diffused shadow, pill controls, eased transitions. Deliberately not adopted: scroll-entry animation, `py-24`-scale whitespace and glass navbars. They slow scanning in a data tool and make screenshot timing nondeterministic.

## Cross-cutting rules now in force
- Direction colours are one system: destabilizing `#b2182b`, stabilizing `#2166ac` (ColorBrewer RdBu, colour-vision-deficiency safe), always paired with ▼/▲ glyphs and words.
- Tabular numerals everywhere; a monospace stack with Consolas / DejaVu Sans Mono fallbacks (no thin Courier).
- Symmetry wording: the default model is "structure-free, wild-type/mutant swap-equivariant only, not SE(3)/E(3)-equivariant"; the optional 3D mode is "SE(3)-invariant". A test fails if any UI string calls the model SE(3)/E(3)-equivariant without negation.
- Optional inputs (CDS) sit behind disclosures; the primary action stays in a fixed place on each form.
