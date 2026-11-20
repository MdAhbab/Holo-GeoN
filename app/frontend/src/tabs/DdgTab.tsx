import { useState } from 'react'
// @ts-ignore - react-plotly.js types resolved via @types/react-plotly.js
import Plot from 'react-plotly.js'
import { postDdg, downloadExport } from '../api'
import { setDdg } from '../store'
import type { DdgPathNumbers, DdgResponse, PredictionMode, TabProps } from '../types'

const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

function verdictStyle(verdict: string | undefined) {
  if (verdict === 'stabilizing') {
    return { cls: 'verdict-stabilizing', color: '#2563eb', arrow: '▲' }
  }
  if (verdict === 'destabilizing') {
    return { cls: 'verdict-destabilizing', color: '#dc2626', arrow: '▼' }
  }
  return { cls: 'verdict-neutral', color: '#64748b', arrow: '●' }
}

function formatSigned(v: number, digits = 3): string {
  return `${v > 0 ? '+' : ''}${v.toFixed(digits)}`
}

function formatPositions(positions: number[]): string {
  if (positions.length === 0) return 'None (all residues ≥ 70 pLDDT)'
  const spans: string[] = []
  let start = positions[0]
  let prev = positions[0]
  for (let i = 1; i < positions.length; i++) {
    if (positions[i] === prev + 1) {
      prev = positions[i]
    } else {
      spans.push(start === prev ? `${start}` : `${start}–${prev}`)
      start = positions[i]
      prev = positions[i]
    }
  }
  spans.push(start === prev ? `${start}` : `${start}–${prev}`)
  return spans.join(', ')
}

function PathCard({
  title,
  subtitle,
  data,
}: {
  title: string
  subtitle?: string
  data: DdgPathNumbers
}) {
  const vs = verdictStyle(data.verdict)
  return (
    <div className="compare-card">
      <div className="compare-card-header">
        <span className="compare-card-title">{title}</span>
        <span className={vs.cls} style={{ fontSize: '0.82rem', fontWeight: 700 }}>
          {vs.arrow} {data.verdict}
        </span>
      </div>
      {subtitle && (
        <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginBottom: 6 }}>
          {subtitle}
        </div>
      )}
      <div style={{ fontSize: '1.45rem', fontWeight: 700, fontFamily: 'monospace' }} className={vs.cls}>
        {formatSigned(data.ddg, 4)} kcal/mol
      </div>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6, fontFamily: 'monospace' }}>
        σ = {data.sigma.toFixed(4)} · 95% CI [{data.ci_low.toFixed(3)}, {data.ci_high.toFixed(3)}]
      </div>
    </div>
  )
}

export default function DdgTab({ health = null }: Partial<TabProps>) {
  const [sequence, setSequence] = useState(UBIQUITIN)
  const [mutation, setMutation] = useState('L8P')
  const [cds, setCds] = useState('')
  const [mode, setMode] = useState<PredictionMode>('structure_free')
  const [compareModes, setCompareModes] = useState(false)
  const [structurePdb, setStructurePdb] = useState('')
  const [pdbFilename, setPdbFilename] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DdgResponse | null>(null)
  const [exporting, setExporting] = useState(false)

  const cdsCap = health?.capabilities?.cds
  const cdsAvailable = Boolean(cdsCap?.available)
  const mode3dCap = health?.capabilities?.invariant_3d
  const mode3dAvailable = Boolean(mode3dCap?.available)

  const effectiveMode: PredictionMode = mode === 'invariant_3d' && mode3dAvailable ? 'invariant_3d' : 'structure_free'
  const effectiveCompare = compareModes && mode3dAvailable
  const needsStructure = effectiveMode === 'invariant_3d' || effectiveCompare

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
        mutation: mutation.trim(),
        cds: cdsAvailable && cds.trim() ? cds.trim() : undefined,
        mode: effectiveMode,
        structure_pdb: needsStructure && structurePdb.trim() ? structurePdb : undefined,
        compare_modes: effectiveCompare || undefined,
      })
      setResult(res)
      setDdg(res)
    } catch (err: unknown) {
      const msg = err as { response?: { data?: { detail?: string } }; message?: string }
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleExportJSON() {
    if (!result) return
    setExporting(true)
    try {
      await downloadExport('json', result, `ddg_${result.mutation}.json`)
    } catch {
      setError('Export failed')
    } finally {
      setExporting(false)
    }
  }

  // Round-2 review, R2-03 (R1-25 re-opened): this re-derived the direction from
  // `result.ddg > 0` - the exact sign test R1-25 removed from the backend for
  // having no neutral case - and then coloured the backend's three-state verdict
  // with it. A +0.006 kcal/mol prediction whose argmax is `neutral` rendered as
  // a green "Stabilizing". The verdict now comes from the one head that has a
  // defined neutral class, which is what `verdict_source` in the payload names.
  const mainVs = verdictStyle(result?.verdict)
  const verdictClass = mainVs.cls
  const markerColor = mainVs.color
  const verdictArrow = mainVs.arrow

  const plotEntries: Array<{ label: string; path: DdgPathNumbers }> = []
  if (result) {
    const modeTag = result.mode === 'invariant_3d' ? 'SE(3)-invariant 3D' : 'structure-free'
    const cdsTag = result.cds?.supplied ? ' + CDS' : ''
    plotEntries.push({
      label: `${result.mutation} (${modeTag}${cdsTag})`,
      path: {
        ddg: result.ddg,
        sigma: result.sigma,
        ci_low: result.ci_low,
        ci_high: result.ci_high,
        verdict: result.verdict,
      },
    })
    if (result.cds?.supplied && result.cds.without_cds) {
      plotEntries.push({
        label: `${result.mutation} (${modeTag}, without CDS)`,
        path: result.cds.without_cds,
      })
    }
    if (result.comparison) {
      const otherMode: PredictionMode = result.mode === 'structure_free' ? 'invariant_3d' : 'structure_free'
      const otherLabel = otherMode === 'invariant_3d' ? 'SE(3)-invariant 3D' : 'structure-free'
      plotEntries.push({
        label: `${result.mutation} (${otherLabel})`,
        path: result.comparison[otherMode],
      })
    }
  }

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
        thickness: 2.5,
        width: 8,
      },
      marker: { size: 14, color: vs.color, symbol: 'circle' },
      name: entry.label,
    }
  })

  const plotLayout = {
    height: Math.max(200, 145 + plotEntries.length * 46),
    margin: { l: 210, r: 30, t: 30, b: 50 },
    xaxis: {
      title: 'ΔΔG (kcal/mol)',
      zeroline: true,
      zerolinecolor: '#94a3b8',
      zerolinewidth: 2,
      gridcolor: '#f1f5f9',
    },
    yaxis: { showticklabels: true, automargin: true },
    showlegend: false,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    shapes: [
      {
        type: 'line',
        x0: 0,
        x1: 0,
        y0: -0.5,
        y1: Math.max(1.5, plotEntries.length - 0.5),
        line: { color: '#64748b', width: 1.5, dash: 'dot' },
      },
    ],
    font: { family: 'Inter, system-ui, sans-serif', size: 12 },
  }

  const structInfo = result?.structure ?? result?.comparison?.structure

  return (
    <div>
      <div className="card">
        <div className="card-title">ΔΔG Predictor — Single Mutation Stability</div>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 14 }}>
            <label htmlFor="ddg-seq">Wild-type Protein Sequence</label>
            <textarea
              id="ddg-seq"
              rows={3}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Paste single-letter amino acid sequence..."
              style={{ fontFamily: 'monospace' }}
            />
          </div>

          <div className="input-row" style={{ marginBottom: 14 }}>
            <div className="field">
              <label htmlFor="ddg-mut">Mutation (e.g. L8P)</label>
              <input
                id="ddg-mut"
                type="text"
                value={mutation}
                onChange={(e) => setMutation(e.target.value)}
                placeholder="L8P"
              />
            </div>

            <div className="field" style={{ flex: '2 1 280px' }}>
              <label>Inference Mode</label>
              <div className="mode-toggle" role="group" aria-label="Inference mode">
                <button
                  type="button"
                  className={`mode-btn${effectiveMode === 'structure_free' ? ' active' : ''}`}
                  onClick={() => setMode('structure_free')}
                >
                  Structure-free (default)
                </button>
                <button
                  type="button"
                  className={`mode-btn${effectiveMode === 'invariant_3d' ? ' active' : ''}`}
                  onClick={() => mode3dAvailable && setMode('invariant_3d')}
                  disabled={!mode3dAvailable}
                >
                  SE(3)-invariant 3D
                </button>
              </div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 4 }}>
                {health?.capabilities?.modes?.[effectiveMode] ??
                  (effectiveMode === 'structure_free'
                    ? 'Structure-free predicted-contact graph; wild-type/mutant swap equivariant only (NOT SE(3)- or E(3)-equivariant).'
                    : 'SE(3)-invariant 3D backbone features from supplied structure.')}
              </div>
            </div>
          </div>

          {!mode3dAvailable && (
            <div className="capability-notice capability-disabled" style={{ marginBottom: 12 }}>
              <strong>SE(3)-invariant 3D mode unavailable:</strong>{' '}
              {mode3dCap?.reason ?? 'Waiting for server capability status…'}
            </div>
          )}

          <div style={{ marginBottom: 14 }}>
            <label
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                textTransform: 'none',
                letterSpacing: 'normal',
                fontSize: '0.88rem',
                color: mode3dAvailable ? 'var(--text)' : 'var(--text-muted)',
                cursor: mode3dAvailable ? 'pointer' : 'not-allowed',
              }}
            >
              <input
                type="checkbox"
                checked={effectiveCompare}
                disabled={!mode3dAvailable}
                onChange={(e) => setCompareModes(e.target.checked)}
              />
              Compare structure-free and SE(3)-invariant 3D modes side by side (<code>compare_modes</code>)
            </label>
          </div>

          {needsStructure && (
            <div
              style={{
                marginBottom: 14,
                padding: '12px 14px',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--teal-mid)',
                background: 'var(--teal-light)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginBottom: 8 }}>
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
                      className="btn btn-secondary"
                      style={{ padding: '4px 10px', fontSize: '0.78rem' }}
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
                rows={4}
                value={structurePdb}
                onChange={(e) => {
                  setStructurePdb(e.target.value)
                  if (!e.target.value) setPdbFilename(null)
                }}
                placeholder="Paste PDB text (ATOM records) or load a .pdb file above..."
                style={{ fontFamily: 'monospace', background: '#fff' }}
              />
              <div style={{ fontSize: '0.76rem', color: 'var(--teal-dark)', marginTop: 4 }}>
                {pdbFilename
                  ? `Loaded ${pdbFilename} (${structurePdb.length.toLocaleString()} chars)`
                  : structurePdb.trim()
                    ? `${structurePdb.length.toLocaleString()} chars of PDB text supplied`
                    : 'Supply a PDB file or paste ATOM records for this exact protein.'}
              </div>
            </div>
          )}

          <div style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label htmlFor="ddg-cds">Optional Coding Sequence (CDS — DNA/RNA, FASTA ok, one stop codon ok)</label>
              {cds && cdsAvailable && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: '2px 8px', fontSize: '0.75rem', marginBottom: 4 }}
                  onClick={() => setCds('')}
                >
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
                  ? 'Optional: paste coding sequence encoding this exact protein for mechanistic mRNA folding + codon usage channels...'
                  : 'CDS input disabled (unavailable on this server)'
              }
              style={{ fontFamily: 'monospace', opacity: cdsAvailable ? 1 : 0.6 }}
            />
            {!cdsAvailable && (
              <div className="capability-notice capability-disabled" style={{ marginTop: 6 }}>
                <strong>CDS features unavailable:</strong>{' '}
                {cdsCap?.reason ?? 'Waiting for server capability status…'}
              </div>
            )}
          </div>

          <div className="input-row">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !sequence.trim() || !mutation.trim() || (needsStructure && !structurePdb.trim())}
            >
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
            <div className="card-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>
              Prediction Result
            </div>
            <span className="symmetry-badge">
              {result.mode === 'invariant_3d' ? 'SE(3)-Invariant 3D' : 'Structure-Free · Swap-Equivariant Only'}
            </span>
          </div>

          <div className="symmetry-statement">
            <strong>Coordinates:</strong>{' '}
            {result.uses_coordinates ? 'Uses aligned 3D backbone coordinates' : 'Reads no coordinates'} ·{' '}
            <strong>Symmetry:</strong> {result.symmetry}
            {result.checkpoint && (
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 2 }}>
                Checkpoint: <code>{result.checkpoint}</code>
              </div>
            )}
          </div>

          {!result.uncertainty_calibrated && (
            <div className="warn-banner" style={{ borderRadius: 'var(--radius)', marginBottom: 12 }}>
              <span>⚠</span>
              <span>
                <strong>Uncertainty not calibrated for this prediction.</strong>{' '}
                Interval source: {result.ci_source}.
              </span>
            </div>
          )}

          <div className="result-block">
            <div className="result-label">
              ΔΔG Prediction {result.cds?.supplied ? '(Headline — With CDS)' : ''}
            </div>
            <div className={`result-value ${verdictClass}`}>
              {result.ddg > 0 ? '+' : ''}{result.ddg.toFixed(3)} kcal/mol
            </div>
            <div style={{ marginTop: 6, fontSize: '0.95rem', fontWeight: 600 }} className={verdictClass}>
              {result.verdict}
            </div>
          </div>

          <div className="result-row">
            <div className="result-item">
              <div className="result-label">Mutation</div>
              <div style={{ fontWeight: 700, fontFamily: 'monospace', fontSize: '1.1rem' }}>
                {result.wt_residue}{result.position}{result.mut_residue}
              </div>
            </div>
            <div className="result-item">
              <div className="result-label">Predicted σ</div>
              <div style={{ fontFamily: 'monospace' }}>{result.sigma.toFixed(3)}</div>
              <div className="result-sub">{result.ci_source}</div>
            </div>
            <div className="result-item">
              <div className="result-label">95% CI (low)</div>
              <div style={{ fontFamily: 'monospace' }}>{result.ci_low.toFixed(3)}</div>
            </div>
            <div className="result-item">
              <div className="result-label">95% CI (high)</div>
              <div style={{ fontFamily: 'monospace' }}>{result.ci_high.toFixed(3)}</div>
            </div>
            <div className="result-item">
              <div className="result-label">Direction</div>
              <div style={{ fontWeight: 600 }} className={verdictClass}>
                {verdictArrow} {result.verdict}
              </div>
              <div className="result-sub">{result.verdict_source}</div>
            </div>
          </div>

          <div className="stat-chips" style={{ marginTop: 12 }}>
            <div className="stat-chip">
              P(destabilizing): <strong>{(result.three_state.destabilizing * 100).toFixed(1)}%</strong>
            </div>
            <div className="stat-chip">
              P(neutral): <strong>{(result.three_state.neutral * 100).toFixed(1)}%</strong>
            </div>
            <div className="stat-chip">
              P(stabilizing): <strong>{(result.three_state.stabilizing * 100).toFixed(1)}%</strong>
            </div>
            <div className="stat-chip">
              CDS:{' '}
              <strong>
                {result.cds?.supplied
                  ? `verified (${result.cds.n_codons ?? '?'} codons${result.cds.stop_codon_stripped ? ', stop stripped' : ''})`
                  : 'not supplied (CDS-dropout path)'}
              </strong>
            </div>
          </div>

          {result.cds?.supplied && result.cds.with_cds && result.cds.without_cds && (
            <div className="section-gap">
              <div style={{ fontWeight: 700, fontSize: '0.92rem', color: 'var(--teal-dark)', marginBottom: 8 }}>
                Coding Sequence Comparison (With CDS vs Without CDS)
              </div>
              <div className="stat-chips" style={{ marginBottom: 10 }}>
                <div className="stat-chip">
                  Frame verified: <strong>{result.cds.frame_verified ? 'Yes' : 'No'}</strong>
                </div>
                <div className="stat-chip">
                  Codons: <strong>{result.cds.n_codons ?? '—'}</strong>
                </div>
                <div className="stat-chip">
                  Stop codon stripped: <strong>{result.cds.stop_codon_stripped ? 'Yes' : 'No'}</strong>
                </div>
                {result.cds.delta_ddg != null && (
                  <div
                    className={`delta-chip ${
                      result.cds.delta_ddg > 0
                        ? 'delta-positive'
                        : result.cds.delta_ddg < 0
                          ? 'delta-negative'
                          : 'delta-neutral'
                    }`}
                  >
                    ΔΔG (with − without CDS):{' '}
                    <strong>{formatSigned(result.cds.delta_ddg, 4)} kcal/mol</strong>
                  </div>
                )}
              </div>
              <div className="compare-grid">
                <PathCard
                  title="With CDS (Headline)"
                  subtitle="Sequence + mechanistic mRNA folding & codon usage"
                  data={result.cds.with_cds}
                />
                <PathCard
                  title="Without CDS (Dropout Path)"
                  subtitle="Sequence-only baseline (mechanistic channels masked)"
                  data={result.cds.without_cds}
                />
              </div>
              {result.cds.note && (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6 }}>
                  {result.cds.note}
                </div>
              )}
            </div>
          )}

          {result.comparison && (
            <div className="section-gap">
              <div style={{ fontWeight: 700, fontSize: '0.92rem', color: 'var(--teal-dark)', marginBottom: 8 }}>
                Mode Comparison (Structure-free vs SE(3)-invariant 3D)
              </div>
              <div style={{ marginBottom: 10 }}>
                <span
                  className={`delta-chip ${
                    result.comparison.delta_ddg_3d_minus_structure_free > 0
                      ? 'delta-positive'
                      : result.comparison.delta_ddg_3d_minus_structure_free < 0
                        ? 'delta-negative'
                        : 'delta-neutral'
                  }`}
                >
                  ΔΔG (3D − structure-free):{' '}
                  <strong>{formatSigned(result.comparison.delta_ddg_3d_minus_structure_free, 4)} kcal/mol</strong>
                </span>
              </div>
              <div className="compare-grid">
                <PathCard
                  title="Structure-free Mode"
                  subtitle={result.comparison.symmetry.structure_free}
                  data={result.comparison.structure_free}
                />
                <PathCard
                  title="SE(3)-invariant 3D Mode"
                  subtitle={result.comparison.symmetry.invariant_3d}
                  data={result.comparison.invariant_3d}
                />
              </div>
            </div>
          )}

          {structInfo && (
            <div
              className="section-gap"
              style={{
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                padding: '12px 16px',
                background: '#fafafa',
              }}
            >
              <div style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--teal-dark)', marginBottom: 6 }}>
                Supplied Structure Alignment &amp; pLDDT
              </div>
              <div className="stat-chips" style={{ marginBottom: 6 }}>
                <div className="stat-chip">Mean pLDDT: <strong>{structInfo.mean_plddt.toFixed(2)}</strong></div>
                <div className="stat-chip">Aligned offset: <strong>{structInfo.aligned_offset}</strong></div>
                <div className="stat-chip">PDB residues: <strong>{structInfo.residues_in_pdb}</strong></div>
                <div className="stat-chip">Mismatches: <strong>{structInfo.mismatches}</strong></div>
                <div className="stat-chip">
                  Usable fraction: <strong>{(structInfo.usable_residue_fraction * 100).toFixed(1)}%</strong>
                </div>
                <div className="stat-chip">
                  Below 70 pLDDT: <strong>{(structInfo.fraction_below_70 * 100).toFixed(1)}%</strong>
                </div>
              </div>
              <div style={{ fontSize: '0.82rem', marginBottom: 4 }}>
                <strong>Low-confidence positions (1-based, pLDDT &lt; 70):</strong>{' '}
                {formatPositions(structInfo.low_confidence_positions)}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{structInfo.note}</div>
            </div>
          )}

          <div className="plot-wrap section-gap">
            <Plot
              data={plotData}
              layout={plotLayout}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
              useResizeHandler
            />
          </div>

          <div className="btn-group">
            <button
              className="btn btn-secondary"
              onClick={handleExportJSON}
              disabled={exporting}
            >
              {exporting && <span className="spinner" />}
              Export JSON
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
