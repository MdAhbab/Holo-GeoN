import { useMemo, useState } from 'react'
// @ts-ignore - react-plotly.js types resolved via @types/react-plotly.js
import Plot from 'react-plotly.js'
import { postDdg, downloadExport } from '../api'
import { setDdg } from '../store'
import type { DdgPathNumbers, DdgResponse, PredictionMode, TabProps } from '../types'
import IntervalNote from '../components/IntervalNote'
import { parseMutation, toNative, toSequenceIndex } from '../epistemic'
import { BASE_LAYOUT, COLORS, PLOT_CONFIG, axisTitle, baseAxis } from '../plotTheme'

const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

function verdictStyle(verdict: string | undefined) {
  if (verdict === 'stabilizing') {
    return { cls: 'verdict-stabilizing', color: COLORS.stabilizing, arrow: '▲' }
  }
  if (verdict === 'destabilizing') {
    return { cls: 'verdict-destabilizing', color: COLORS.destabilizing, arrow: '▼' }
  }
  return { cls: 'verdict-neutral', color: COLORS.neutral, arrow: '●' }
}

function formatSigned(v: number, digits = 3): string {
  return `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`
}

function cleanSeq(raw: string): string {
  return raw
    .split('\n')
    .filter((l) => !l.trimStart().startsWith('>'))
    .join('')
    .replace(/\s+/g, '')
    .toUpperCase()
}

function formatPositions(positions: number[], offset = 0): string {
  if (positions.length === 0) return 'None'
  const spans: string[] = []
  let start = positions[0]
  let prev = positions[0]
  const fmt = (a: number, b: number) => (a === b ? `${a + offset}` : `${a + offset}–${b + offset}`)
  for (let i = 1; i < positions.length; i++) {
    if (positions[i] === prev + 1) {
      prev = positions[i]
    } else {
      spans.push(fmt(start, prev))
      start = positions[i]
      prev = positions[i]
    }
  }
  spans.push(fmt(start, prev))
  return spans.join(', ')
}

function PathCard({
  title,
  subtitle,
  tooltip,
  data,
}: {
  title: string
  subtitle?: string
  tooltip?: string
  data: DdgPathNumbers
}) {
  const vs = verdictStyle(data.verdict)
  return (
    <div className="compare-card" title={tooltip}>
      <div className="compare-card-header">
        <span className="compare-card-title">{title}</span>
        <span className={`verdict-pill ${vs.cls}`}>
          {vs.arrow} {data.verdict}
        </span>
      </div>
      <div className={`path-value ${vs.cls}`}>{formatSigned(data.ddg, 3)} <span className="unit">kcal/mol</span></div>
      <div className="path-ci">
        σ {data.sigma.toFixed(3)} · 95% CI [{formatSigned(data.ci_low, 2)}, {formatSigned(data.ci_high, 2)}]
      </div>
      {subtitle && <div className="path-sub">{subtitle}</div>}
    </div>
  )
}

/** Stacked three-state probability bar: read in one glance, numbers kept for precision. */
function ThreeStateBar({ p }: { p: DdgResponse['three_state'] }) {
  const parts = [
    { key: 'destabilizing', v: p.destabilizing, color: COLORS.destabilizing, label: 'Destabilizing' },
    { key: 'neutral', v: p.neutral, color: '#a3acb9', label: 'Neutral' },
    { key: 'stabilizing', v: p.stabilizing, color: COLORS.stabilizing, label: 'Stabilizing' },
  ]
  return (
    <div className="three-state" aria-label="Three-state head probabilities">
      <div className="three-state-bar">
        {parts.map((s) => (
          <div
            key={s.key}
            className="three-state-seg"
            style={{ width: `${Math.max(0, s.v * 100)}%`, background: s.color }}
            title={`${s.label}: ${(s.v * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      <div className="three-state-legend">
        {parts.map((s) => (
          <span key={s.key} className="three-state-item">
            <span className="legend-swatch" style={{ background: s.color }} aria-hidden="true" />
            P({s.key}) <strong>{(s.v * 100).toFixed(1)}%</strong>
          </span>
        ))}
      </div>
    </div>
  )
}

export default function DdgTab({ health = null }: Partial<TabProps>) {
  const [sequence, setSequence] = useState(UBIQUITIN)
  const [mutation, setMutation] = useState('L8P')
  const [numberingStart, setNumberingStart] = useState(1)
  const [cds, setCds] = useState('')
  const [mode, setMode] = useState<PredictionMode>('structure_free')
  const [compareModes, setCompareModes] = useState(false)
  const [structurePdb, setStructurePdb] = useState('')
  const [pdbFilename, setPdbFilename] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DdgResponse | null>(null)
  // numbering in force when the shown result was requested (offset = first residue number - 1)
  const [resultNumbering, setResultNumbering] = useState(1)
  const [exporting, setExporting] = useState(false)

  const cdsCap = health?.capabilities?.cds
  const cdsAvailable = Boolean(cdsCap?.available)
  const mode3dCap = health?.capabilities?.invariant_3d
  const mode3dAvailable = Boolean(mode3dCap?.available)

  const effectiveMode: PredictionMode = mode === 'invariant_3d' && mode3dAvailable ? 'invariant_3d' : 'structure_free'
  const effectiveCompare = compareModes && mode3dAvailable
  const needsStructure = effectiveMode === 'invariant_3d' || effectiveCompare

  const seqClean = useMemo(() => cleanSeq(sequence), [sequence])
  const firstNo = Number.isFinite(numberingStart) && numberingStart >= 1 ? Math.floor(numberingStart) : 1

  // Mutations are typed in the numbering the user chose; the API always receives the
  // 1-based index into the sequence as supplied (its contract is unchanged).
  const mutCheck = useMemo(() => {
    const parsed = parseMutation(mutation)
    if (!parsed) return { ok: false, apiMutation: mutation.trim(), message: null as string | null, idx: null as number | null }
    const idx = toSequenceIndex(parsed.pos, firstNo)
    if (idx < 1 || idx > seqClean.length) {
      return {
        ok: false,
        apiMutation: mutation.trim(),
        idx: null,
        message: `Position ${parsed.pos} is outside this sequence (residues ${firstNo}–${toNative(seqClean.length, firstNo)}).`,
      }
    }
    const actual = seqClean[idx - 1]
    const apiMutation = `${parsed.wt}${idx}${parsed.mut}`
    const wtMatch = actual === parsed.wt
    let message: string | null = null
    if (!wtMatch) {
      message = `Residue ${parsed.pos} is ${actual}, not ${parsed.wt}.`
    } else if (firstNo !== 1) {
      message = `Sent to the model as ${apiMutation} (sequence index ${idx}); residue ${parsed.pos} is ${actual} ✓`
    }
    return { ok: wtMatch, apiMutation, message, idx }
  }, [mutation, firstNo, seqClean])

  async function handlePdbFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      setStructurePdb(text)
      setPdbFilename(file.name)
    } catch {
      setError(`Could not read PDB file "${file.name}"`)
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await postDdg({
        wt_sequence: sequence.trim(),
        mutation: mutCheck.apiMutation,
        cds: cdsAvailable && cds.trim() ? cds.trim() : undefined,
        mode: effectiveMode,
        structure_pdb: needsStructure && structurePdb.trim() ? structurePdb : undefined,
        compare_modes: effectiveCompare || undefined,
      })
      setResult(res)
      setResultNumbering(firstNo)
      setDdg(res)
    } catch (err: unknown) {
      const msg = err as { response?: { data?: { detail?: string } }; message?: string }
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  const offset = resultNumbering - 1
  const displayMutation = result
    ? `${result.wt_residue}${result.position + offset}${result.mut_residue}`
    : ''

  async function handleExportJSON() {
    if (!result) return
    setExporting(true)
    try {
      const payload = offset
        ? { ...result, numbering: { first_residue_number: resultNumbering, mutation_in_that_numbering: displayMutation } }
        : result
      await downloadExport('json', payload, `ddg_${displayMutation || result.mutation}.json`)
    } catch {
      setError('Export failed')
    } finally {
      setExporting(false)
    }
  }

  // The direction shown comes from the three-state head (verdict_source), never from the
  // sign of ddg: a +0.006 kcal/mol prediction whose argmax is neutral is shown as neutral.
  const mainVs = verdictStyle(result?.verdict)

  const plotEntries: Array<{ label: string; path: DdgPathNumbers }> = []
  if (result) {
    const modeTag = result.mode === 'invariant_3d' ? 'SE(3)-invariant 3D' : 'structure-free'
    const cdsTag = result.cds?.supplied ? ' + CDS' : ''
    plotEntries.push({
      label: `${displayMutation} · ${modeTag}${cdsTag}`,
      path: { ddg: result.ddg, sigma: result.sigma, ci_low: result.ci_low, ci_high: result.ci_high, verdict: result.verdict },
    })
    if (result.cds?.supplied && result.cds.without_cds) {
      plotEntries.push({ label: `${displayMutation} · ${modeTag}, no CDS`, path: result.cds.without_cds })
    }
    if (result.comparison) {
      const otherMode: PredictionMode = result.mode === 'structure_free' ? 'invariant_3d' : 'structure_free'
      const otherLabel = otherMode === 'invariant_3d' ? 'SE(3)-invariant 3D' : 'structure-free'
      plotEntries.push({ label: `${displayMutation} · ${otherLabel}`, path: result.comparison[otherMode] })
    }
  }

  const xs = plotEntries.flatMap((e) => [e.path.ci_low, e.path.ci_high, 0])
  const xMin = xs.length ? Math.min(...xs) : -1
  const xMax = xs.length ? Math.max(...xs) : 1
  const xPad = Math.max(0.25, (xMax - xMin) * 0.08)

  const plotData = plotEntries.map((entry) => {
    const vs = verdictStyle(entry.path.verdict)
    return {
      type: 'scatter',
      x: [entry.path.ddg],
      y: [entry.label],
      mode: 'markers',
      error_x: {
        type: 'data',
        symmetric: false,
        array: [entry.path.ci_high - entry.path.ddg],
        arrayminus: [entry.path.ddg - entry.path.ci_low],
        color: vs.color,
        thickness: 2,
        width: 7,
      },
      marker: { size: 12, color: vs.color, line: { color: '#ffffff', width: 1.5 } },
      name: entry.label,
      hovertemplate: `${entry.label}<br>ΔΔG %{x:.3f} kcal/mol<br>95% CI [${entry.path.ci_low.toFixed(3)}, ${entry.path.ci_high.toFixed(3)}]<extra></extra>`,
    }
  })

  const plotLayout = {
    ...BASE_LAYOUT,
    height: 150 + plotEntries.length * 44,
    margin: { l: 12, r: 24, t: 34, b: 48 },
    xaxis: baseAxis({
      title: axisTitle('ΔΔG (kcal/mol) with 95% interval'),
      range: [xMin - xPad, xMax + xPad],
      zeroline: true,
      zerolinecolor: COLORS.zero,
      zerolinewidth: 1.5,
      tickformat: '.1f',
    }),
    yaxis: baseAxis({ showline: false, ticks: '', tickfont: { size: 12, color: COLORS.ink }, autorange: 'reversed' }),
    showlegend: false,
    annotations: [
      {
        xref: 'paper', yref: 'paper', x: 0, y: 1.0, yanchor: 'bottom', xanchor: 'left', showarrow: false,
        text: '◀ destabilizing (ΔΔG < 0)', font: { size: 11, color: COLORS.destabilizing },
      },
      {
        xref: 'paper', yref: 'paper', x: 1, y: 1.0, yanchor: 'bottom', xanchor: 'right', showarrow: false,
        text: 'stabilizing (ΔΔG > 0) ▶', font: { size: 11, color: COLORS.stabilizing },
      },
    ],
  }

  const structInfo = result?.structure ?? result?.comparison?.structure
  const submitBlocked =
    loading || !sequence.trim() || !mutation.trim() || (needsStructure && !structurePdb.trim()) ||
    (parseMutation(mutation) !== null && !mutCheck.ok)

  return (
    <div>
      <div className="card">
        <div className="card-title">ΔΔG Predictor — Single Mutation Stability</div>
        <form onSubmit={handleSubmit}>
          <div className="form-field">
            <div className="label-row">
              <label htmlFor="ddg-seq">Wild-type Protein Sequence</label>
              <span className="field-meta">{seqClean.length} residues{firstNo !== 1 ? ` · numbered ${firstNo}–${toNative(seqClean.length, firstNo)}` : ''}</span>
            </div>
            <textarea
              id="ddg-seq"
              rows={3}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Paste single-letter amino acid sequence..."
              className="seq-input"
            />
          </div>

          <div className="input-row form-field">
            <div className="field">
              <label htmlFor="ddg-mut">Mutation (e.g. L8P)</label>
              <input
                id="ddg-mut"
                type="text"
                value={mutation}
                onChange={(e) => setMutation(e.target.value)}
                placeholder="L8P"
                aria-describedby={mutCheck.message ? 'ddg-mut-hint' : undefined}
              />
            </div>
            <div className="field-sm">
              <label
                htmlFor="ddg-numbering"
                title="Residue number of the first residue above. For a domain construct (e.g. TP53 DBD, residues 94–292) enter 94 and type the mutation in native numbering (R175H)."
              >
                First residue no.<span className="info-tip" aria-hidden="true">i</span>
              </label>
              <input
                id="ddg-numbering"
                type="number"
                min={1}
                value={numberingStart}
                onChange={(e) => setNumberingStart(Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: '2 1 300px' }}>
              <label>Inference Mode</label>
              <div className="mode-toggle" role="group" aria-label="Inference mode">
                <button
                  type="button"
                  className={`mode-btn${effectiveMode === 'structure_free' ? ' active' : ''}`}
                  onClick={() => setMode('structure_free')}
                  title={health?.capabilities?.modes?.structure_free ?? 'Reads no coordinates; WT/mutant swap-equivariant only (not SE(3)/E(3)-equivariant).'}
                >
                  Structure-free (default)
                </button>
                <button
                  type="button"
                  className={`mode-btn${effectiveMode === 'invariant_3d' ? ' active' : ''}`}
                  onClick={() => mode3dAvailable && setMode('invariant_3d')}
                  disabled={!mode3dAvailable}
                  title={!mode3dAvailable ? mode3dCap?.reason ?? undefined : health?.capabilities?.modes?.invariant_3d ?? undefined}
                >
                  SE(3)-invariant 3D
                </button>
              </div>
            </div>
          </div>
          {/* shown only when it carries information: a numbering translation or a mismatch */}
          {mutCheck.message && (
            <div id="ddg-mut-hint" className={`field-hint${!mutCheck.ok ? ' hint-error' : ''}`}>
              {mutCheck.message}
            </div>
          )}

          {!mode3dAvailable && (
            <div className="capability-notice capability-disabled" style={{ marginBottom: 12 }}>
              <strong>SE(3)-invariant 3D mode unavailable:</strong>{' '}
              {mode3dCap?.reason ?? 'Waiting for server capability status…'}
            </div>
          )}

          <div className="form-field">
            <label className="check-label" style={{ cursor: mode3dAvailable ? 'pointer' : 'not-allowed' }}>
              <input
                id="ddg-compare-modes"
                type="checkbox"
                checked={effectiveCompare}
                disabled={!mode3dAvailable}
                onChange={(e) => setCompareModes(e.target.checked)}
              />
              Compare structure-free and SE(3)-invariant 3D modes side by side
            </label>
          </div>

          {needsStructure && (
            <div className="structure-box form-field">
              <div className="label-row" style={{ marginBottom: 8 }}>
                <label htmlFor="ddg-pdb" style={{ marginBottom: 0 }}>
                  Structure PDB (required for 3D mode — B-factor read as pLDDT)
                </label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    id="ddg-pdb-file"
                    type="file"
                    accept=".pdb,.ent,.brk,text/plain"
                    onChange={handlePdbFile}
                    style={{ fontSize: '0.8rem' }}
                  />
                  {structurePdb && (
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        setStructurePdb('')
                        setPdbFilename(null)
                      }}
                    >
                      Clear PDB
                    </button>
                  )}
                </div>
              </div>
              <textarea
                id="ddg-pdb"
                rows={3}
                value={structurePdb}
                onChange={(e) => {
                  setStructurePdb(e.target.value)
                  if (!e.target.value) setPdbFilename(null)
                }}
                placeholder="Paste PDB text (ATOM records) or load a .pdb file above..."
                className="seq-input"
                style={{ background: '#fff' }}
              />
              <div className="field-hint">
                {pdbFilename
                  ? `Loaded ${pdbFilename} (${structurePdb.length.toLocaleString()} chars)`
                  : structurePdb.trim()
                    ? `${structurePdb.length.toLocaleString()} chars of PDB text supplied`
                    : 'Supply a structure of this protein. A structure missing a terminal tail is accepted; uncovered residues get no coordinates.'}
              </div>
            </div>
          )}

          <details className="form-disclosure" open={Boolean(cds)}>
            <summary>
              Optional coding sequence (CDS) <span className="summary-meta">mRNA folding &amp; codon-usage channels</span>
            </summary>
            <div style={{ marginTop: 8 }}>
              <div className="label-row">
                <label htmlFor="ddg-cds">Coding Sequence (DNA/RNA, FASTA ok, one stop codon ok)</label>
                {cds && cdsAvailable && (
                  <button type="button" className="btn-link" onClick={() => setCds('')}>
                    Clear CDS
                  </button>
                )}
              </div>
              <textarea
                id="ddg-cds"
                rows={2}
                value={cds}
                disabled={!cdsAvailable}
                onChange={(e) => setCds(e.target.value)}
                placeholder={
                  cdsAvailable
                    ? 'Optional: coding sequence encoding this exact protein...'
                    : 'CDS input disabled (unavailable on this server)'
                }
                className="seq-input"
              />
              {!cdsAvailable && (
                <div className="capability-notice capability-disabled" style={{ marginTop: 6 }}>
                  <strong>CDS features unavailable:</strong> {cdsCap?.reason ?? 'Waiting for server capability status…'}
                </div>
              )}
            </div>
          </details>

          <div className="btn-group">
            <button type="submit" className="btn btn-primary" disabled={submitBlocked}>
              {loading && <span className="spinner" />}
              {loading ? 'Predicting…' : effectiveCompare ? 'Predict & Compare Modes' : 'Predict ΔΔG'}
            </button>
          </div>
        </form>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <div className="card-header-row">
            <div className="card-title card-title-flat">Prediction Result</div>
            <span className="symmetry-badge">
              {result.mode === 'invariant_3d' ? 'SE(3)-Invariant 3D · uses coordinates' : 'Structure-Free · Swap-Equivariant Only'}
            </span>
          </div>

          <div className="headline-grid">
            <div className="result-block">
              <div className="result-label">
                ΔΔG prediction · <span className="mono">{displayMutation}</span>
                {result.cds?.supplied ? ' · headline uses the CDS' : ''}
              </div>
              <div className={`result-value ${mainVs.cls}`}>
                {formatSigned(result.ddg, 3)} <span className="unit">kcal/mol</span>
              </div>
              <div className="headline-meta">
                <span className={`verdict-pill ${mainVs.cls}`} title={`Direction: ${result.verdict_source}`}>
                  {mainVs.arrow} {result.verdict}
                </span>
                <span className="headline-ci">
                  95% CI [{formatSigned(result.ci_low, 2)}, {formatSigned(result.ci_high, 2)}]
                </span>
                <span className="headline-sigma" title={result.ci_source}>σ {result.sigma.toFixed(3)}</span>
              </div>
              {offset !== 0 && (
                <div className="result-sub">
                  Native numbering from residue {resultNumbering}; sent to the model as{' '}
                  <span className="mono">{result.mutation}</span>.
                </div>
              )}
            </div>
            <div className="headline-side">
              <ThreeStateBar p={result.three_state} />
              <IntervalNote
                health={health}
                calibrated={result.uncertainty_calibrated}
                ciSource={result.ci_source}
                mode={result.mode}
              />
            </div>
          </div>

          {result.cds?.supplied && result.cds.with_cds && result.cds.without_cds && (
            <div className="section-gap">
              <div className="section-title">Coding sequence: with vs without CDS</div>
              <div className="meta-line">
                <span>Frame verified <strong>{result.cds.frame_verified ? 'yes' : 'no'}</strong></span>
                <span><strong>{result.cds.n_codons ?? '—'}</strong> codons</span>
                {result.cds.delta_ddg != null && (
                  <div className={`delta-chip ${result.cds.delta_ddg > 0 ? 'delta-positive' : result.cds.delta_ddg < 0 ? 'delta-negative' : 'delta-neutral'}`}>
                    Δ (with − without CDS) <strong>{formatSigned(result.cds.delta_ddg, 3)} kcal/mol</strong>
                  </div>
                )}
              </div>
              <div className="compare-grid">
                <PathCard title="With CDS (headline)" subtitle="Sequence + mRNA folding & codon usage" data={result.cds.with_cds} />
                <PathCard title="Without CDS (dropout path)" subtitle="Mechanistic channels masked" data={result.cds.without_cds} />
              </div>
              {result.cds.note && <div className="provenance-line">{result.cds.note}</div>}
            </div>
          )}

          {result.comparison && (
            <div className="section-gap mode-comparison">
              <div className="section-title-row">
                <div className="section-title">Mode comparison · structure-free vs SE(3)-invariant 3D</div>
                <span
                  className={`delta-chip ${
                    result.comparison.delta_ddg_3d_minus_structure_free > 0
                      ? 'delta-positive'
                      : result.comparison.delta_ddg_3d_minus_structure_free < 0
                        ? 'delta-negative'
                        : 'delta-neutral'
                  }`}
                >
                  Δ (3D − structure-free): <strong>{formatSigned(result.comparison.delta_ddg_3d_minus_structure_free, 3)} kcal/mol</strong>
                </span>
              </div>
              <div className="compare-grid compare-grid-2">
                <PathCard
                  title="Structure-free (default)"
                  subtitle="Reads no coordinates · WT/mutant swap-equivariant only"
                  tooltip={result.comparison.symmetry.structure_free}
                  data={result.comparison.structure_free}
                />
                <PathCard
                  title="SE(3)-invariant 3D"
                  subtitle="Supplied backbone coordinates · SE(3)-invariant (scalar output)"
                  tooltip={result.comparison.symmetry.invariant_3d}
                  data={result.comparison.invariant_3d}
                />
              </div>
            </div>
          )}

          {structInfo && (
            <div className="section-gap structure-summary">
              <div className="section-title">Supplied structure · alignment &amp; pLDDT</div>
              <div className="meta-line">
                {structInfo.sequence_coverage && (
                  <span>
                    Coordinates for residues{' '}
                    <strong>{structInfo.sequence_coverage.first + offset}–{structInfo.sequence_coverage.last + offset}</strong> of{' '}
                    {structInfo.sequence_coverage.length}
                  </span>
                )}
                {(structInfo.residues_without_coordinates?.length ?? 0) > 0 && (
                  <span className="meta-warn">
                    masked (no coordinates) <strong>{formatPositions(structInfo.residues_without_coordinates ?? [], offset)}</strong>
                  </span>
                )}
                {structInfo.mutation_site_masked !== undefined && (
                  <span className={structInfo.mutation_site_masked ? 'meta-warn' : ''}>
                    mutation site <strong>{structInfo.mutation_site_masked ? 'masked' : 'has coordinates'}</strong>
                  </span>
                )}
                <span>mean pLDDT <strong>{structInfo.mean_plddt.toFixed(1)}</strong></span>
                <span><strong>{structInfo.mismatches}</strong> mismatches</span>
                {structInfo.low_confidence_positions.length > 0 && (
                  <span className="meta-warn">
                    pLDDT &lt; 70 <strong>{formatPositions(structInfo.low_confidence_positions, offset)}</strong>
                  </span>
                )}
              </div>
              <div className="provenance-line">{structInfo.note}.</div>
            </div>
          )}

          <div className="plot-wrap section-gap">
            <Plot
              data={plotData}
              layout={plotLayout}
              config={PLOT_CONFIG}
              style={{ width: '100%' }}
              useResizeHandler
            />
          </div>

          <div className="provenance-line" title={result.symmetry}>
            {result.uses_coordinates ? 'SE(3)-invariant 3D model, supplied coordinates' : 'Structure-free model, reads no coordinates'}
            {' · '}CDS {result.cds?.supplied ? `verified (${result.cds.n_codons ?? '?'} codons)` : 'not supplied'}
            {' · '}direction: {result.verdict_source}
            {' · '}interval: {result.ci_source}
            {result.checkpoint ? <> · checkpoint <code>{result.checkpoint}</code></> : null}
          </div>

          <div className="btn-group">
            <button className="btn btn-secondary" onClick={handleExportJSON} disabled={exporting}>
              {exporting && <span className="spinner" />}
              Export JSON
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
