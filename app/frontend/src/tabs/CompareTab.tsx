import { useState, useRef, useEffect, useMemo } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import * as $3Dmol from '3dmol'
import { postCompare } from '../api'
import { setCompare } from '../store'
import type { CompareResponse, PlddtBands, TabProps } from '../types'

/** Map AlphaFold pLDDT (B-factor column) to standard AlphaFold confidence colors. */
function plddt2color(bfactor: number): string {
  if (bfactor > 90) return '#0053d6'
  if (bfactor >= 70) return '#65cbf3'
  if (bfactor > 50) return '#ffdb13'
  return '#ff7d45'
}

/** Map per-position min ΔΔG (<0 destabilizing red, 0 neutral light gray, >0 stabilizing blue). */
function ddg2color(ddg: number): string {
  const clamp = Math.max(-3.5, Math.min(3.5, ddg))
  if (clamp < 0) {
    const t = Math.abs(clamp) / 3.5
    const r = Math.round(248 + (220 - 248) * t)
    const g = Math.round(250 + (38 - 250) * t)
    const b = Math.round(252 + (38 - 252) * t)
    return `rgb(${r},${g},${b})`
  }
  if (clamp > 0) {
    const t = clamp / 3.5
    const r = Math.round(248 + (37 - 248) * t)
    const g = Math.round(250 + (99 - 250) * t)
    const b = Math.round(252 + (235 - 252) * t)
    return `rgb(${r},${g},${b})`
  }
  return '#f8fafc'
}

function computeBandsFallback(plddt: number[]): PlddtBands {
  let very_high_gt_90 = 0
  let confident_70_90 = 0
  let low_50_70 = 0
  let very_low_le_50 = 0
  for (const v of plddt) {
    if (v > 90) very_high_gt_90 += 1
    else if (v > 70) confident_70_90 += 1
    else if (v > 50) low_50_70 += 1
    else very_low_le_50 += 1
  }
  return { very_high_gt_90, confident_70_90, low_50_70, very_low_le_50 }
}

function formatPositionRanges(positions: number[]): string {
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

function lowConfidenceSpans(flags: boolean[]): Array<[number, number]> {
  const spans: Array<[number, number]> = []
  let start: number | null = null
  for (let i = 0; i < flags.length; i++) {
    if (flags[i] && start === null) {
      start = i + 1
    } else if (!flags[i] && start !== null) {
      spans.push([start, i])
      start = null
    }
  }
  if (start !== null) {
    spans.push([start, flags.length])
  }
  return spans
}

export default function CompareTab({ health: _health = null }: Partial<TabProps>) {
  const [uniprotId, setUniprotId] = useState('P69905')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CompareResponse | null>(null)
  const [colorMode, setColorMode] = useState<'plddt' | 'ddg'>('plddt')
  const [markLowConf, setMarkLowConf] = useState(true)
  const [showAllLowPos, setShowAllLowPos] = useState(false)

  const viewerRef = useRef<HTMLDivElement>(null)
  const viewerElementRef = useRef<HTMLDivElement | null>(null)
  const viewerInstance = useRef<$3Dmol.GLViewer | null>(null)
  const loadedPdbRef = useRef<string | null>(null)

  const plddtMean = result
    ? (result.plddt_mean ??
        (result.plddt.length > 0
          ? Number((result.plddt.reduce((a, b) => a + b, 0) / result.plddt.length).toFixed(2))
          : 0))
    : 0

  const plddtBands: PlddtBands | null = result
    ? (result.plddt_bands ?? computeBandsFallback(result.plddt))
    : null

  const lowConfPositions: number[] = useMemo(
    () =>
      result
        ? (result.low_confidence_positions ??
            result.plddt
              .map((v, i) => (v < 70 ? i + 1 : -1))
              .filter((p) => p > 0))
        : [],
    [result]
  )

  const lowConfInScan: boolean[] = useMemo(
    () =>
      result
        ? (result.low_confidence_in_scan ??
            result.min_ddg_per_position.map((_, i) => (result.plddt[i] ?? 100) < 70))
        : [],
    [result]
  )

  const plddtNote = result
    ? (result.plddt_note ??
        'pLDDT describes the AlphaFold model drawn here, not the prediction: the structure-free model reads no coordinates.')
    : ''

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    setResult(null)
    setShowAllLowPos(false)
    viewerInstance.current = null
    viewerElementRef.current = null
    loadedPdbRef.current = null
    try {
      const res = await postCompare({ uniprot_id: uniprotId.trim() })
      setResult(res)
      setCompare(res)
    } catch (err: unknown) {
      const axiosErr = err as {
        response?: { status?: number; data?: { detail?: string } }
        message?: string
      }
      if (axiosErr.response?.status === 502) {
        setError(
          'Could not reach AlphaFold / UniProt servers (502 Bad Gateway). ' +
          'Check your internet connection or verify the UniProt accession is valid.'
        )
      } else {
        setError(
          axiosErr.response?.data?.detail ??
          axiosErr.message ??
          'Request failed'
        )
      }
    } finally {
      setLoading(false)
    }
  }

  // Initialize / reuse the 3Dmol viewer whenever the mounted container or PDB changes,
  // and update colors/markers cleanly on coloring toggle without resetting camera zoom.
  useEffect(() => {
    if (!result?.structure_pdb || !viewerRef.current) return

    let viewer = viewerInstance.current
    if (!viewer || viewerElementRef.current !== viewerRef.current) {
      viewer = $3Dmol.createViewer(viewerRef.current, {
        backgroundColor: '#111111',
        antialias: true,
      })
      viewerInstance.current = viewer
      viewerElementRef.current = viewerRef.current
      loadedPdbRef.current = null
    }

    if (loadedPdbRef.current !== result.structure_pdb) {
      viewer.clear()
      viewer.addModel(result.structure_pdb, 'pdb')
      viewer.zoomTo()
      loadedPdbRef.current = result.structure_pdb
    }

    const colorfunc =
      colorMode === 'plddt'
        ? (atom: $3Dmol.AtomSpec) => plddt2color(atom.b ?? 0)
        : (atom: $3Dmol.AtomSpec) => {
            const idx = (atom.resi ?? 0) - 1
            if (idx < 0 || idx >= result.min_ddg_per_position.length) {
              return '#94a3b8'
            }
            return ddg2color(result.min_ddg_per_position[idx])
          }

    viewer.setStyle({}, { cartoon: { colorfunc } })

    if (markLowConf && lowConfPositions.length > 0) {
      viewer.addStyle(
        { resi: lowConfPositions, atom: 'CA' },
        { sphere: { radius: 0.55, color: '#ff7d45', opacity: 0.85 } }
      )
    }

    viewer.render()
  }, [result, colorMode, markLowConf, lowConfPositions])

  const paeData = result
    ? [
        {
          type: 'heatmap',
          z: result.pae,
          colorscale: 'Viridis',
          reversescale: true,
          colorbar: {
            title: 'PAE (Å)',
            titleside: 'right',
          },
          hovertemplate:
            'Residue i: %{x}<br>Residue j: %{y}<br>PAE: %{z:.2f} Å<extra></extra>',
        },
      ]
    : []

  const paeLayout = {
    height: 280,
    margin: { l: 50, r: 90, t: 30, b: 50 },
    xaxis: { title: 'Residue j', gridcolor: '#f1f5f9' },
    yaxis: { title: 'Residue i', autorange: 'reversed', gridcolor: '#f1f5f9' },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', size: 11 },
  }

  const lowScanX: number[] = []
  const lowScanY: number[] = []
  if (result) {
    for (let i = 0; i < result.min_ddg_per_position.length; i++) {
      if (lowConfInScan[i]) {
        lowScanX.push(i + 1)
        lowScanY.push(result.min_ddg_per_position[i])
      }
    }
  }

  const overlayData = result
    ? [
        {
          type: 'scatter',
          x: Array.from({ length: result.min_ddg_per_position.length }, (_, i) => i + 1),
          y: result.min_ddg_per_position,
          mode: 'lines',
          line: { color: '#dc2626', width: 2 },
          name: 'Holo-GNN min ΔΔG',
          yaxis: 'y',
          hovertemplate: 'Pos %{x}<br>Min ΔΔG: %{y:.3f}<extra></extra>',
        },
        ...(lowScanX.length > 0
          ? [
              {
                type: 'scatter',
                x: lowScanX,
                y: lowScanY,
                mode: 'markers',
                marker: { color: '#ff7d45', size: 6, symbol: 'diamond-open', line: { width: 1.5, color: '#b45309' } },
                name: 'Low-conf AlphaFold (pLDDT < 70)',
                yaxis: 'y',
                hovertemplate: 'Pos %{x} (pLDDT &lt; 70)<br>Min ΔΔG: %{y:.3f}<extra></extra>',
              },
            ]
          : []),
        {
          type: 'scatter',
          x: Array.from({ length: result.plddt.length }, (_, i) => i + 1),
          y: result.plddt,
          mode: 'lines',
          line: { color: '#2563eb', width: 2, dash: 'dot' },
          name: 'AlphaFold pLDDT',
          yaxis: 'y2',
          hovertemplate: 'Pos %{x}<br>pLDDT: %{y:.1f}<extra></extra>',
        },
      ]
    : []

  const spanShapes = lowConfidenceSpans(lowConfInScan).map(([s, e]) => ({
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: s - 0.5,
    x1: e + 0.5,
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(255, 219, 19, 0.22)',
    line: { width: 0 },
    layer: 'below',
  }))

  const overlayLayout = {
    height: 270,
    margin: { l: 55, r: 70, t: 30, b: 60 },
    xaxis: { title: 'Residue position', gridcolor: '#f1f5f9' },
    yaxis: {
      title: 'Min ΔΔG (kcal/mol)',
      titlefont: { color: '#dc2626' },
      tickfont: { color: '#dc2626' },
      gridcolor: '#f1f5f9',
    },
    yaxis2: {
      title: 'pLDDT',
      titlefont: { color: '#2563eb' },
      tickfont: { color: '#2563eb' },
      overlaying: 'y',
      side: 'right',
      range: [0, 100],
    },
    shapes: [
      ...spanShapes,
      {
        type: 'line',
        xref: 'paper',
        yref: 'y2',
        x0: 0,
        x1: 1,
        y0: 70,
        y1: 70,
        line: { color: '#f59e0b', width: 1, dash: 'dash' },
      },
    ],
    legend: { orientation: 'h', y: -0.32 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', size: 11 },
  }

  const totalPlddtCount = result ? Math.max(result.plddt.length, 1) : 1

  return (
    <div>
      <div className="card">
        <div className="card-title">AlphaFold Compare — Structure Confidence vs Predicted Fragility</div>
        <form onSubmit={handleSubmit}>
          <div className="input-row">
            <div className="field">
              <label htmlFor="af-uniprot">UniProt Accession</label>
              <input
                id="af-uniprot"
                type="text"
                value={uniprotId}
                onChange={(e) => setUniprotId(e.target.value)}
                placeholder="e.g. P69905"
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !uniprotId.trim()}
            >
              {loading && <span className="spinner" />}
              {loading ? 'Fetching…' : 'Compare'}
            </button>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6 }}>
            Fetches AlphaFold structure + PAE from EBI, then runs structure-free Holo-GeoN scan on the sequence.
          </div>
        </form>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <>
          <div className="plddt-note" role="note">
            <strong>Model / Structure Provenance:</strong> {plddtNote}
          </div>

          {result.pae_downsampled && (
            <div
              style={{
                background: '#eff6ff',
                border: '1px solid #bfdbfe',
                borderRadius: 'var(--radius)',
                padding: '8px 14px',
                fontSize: '0.82rem',
                color: '#1e40af',
                marginBottom: 14,
              }}
            >
              Note: PAE matrix was downsampled for display performance.
            </div>
          )}

          <div className="card">
            <div className="card-title">AlphaFold pLDDT Confidence Summary — {result.uniprot_id}</div>
            <div className="stat-chips">
              <div className="stat-chip">
                Mean pLDDT: <strong>{plddtMean.toFixed(2)}</strong>
              </div>
              <div className="stat-chip">
                Sequence length: <strong>{result.sequence.length} aa</strong>
              </div>
              <div className="stat-chip">
                Low confidence (&lt;70): <strong>{lowConfPositions.length} residues</strong>
              </div>
              {plddtBands && (
                <>
                  <div className="stat-chip">
                    <span className="legend-dot plddt-very-high" style={{ background: '#0053d6', marginRight: 6 }} />
                    &gt;90 Very high: <strong>{plddtBands.very_high_gt_90}</strong>
                  </div>
                  <div className="stat-chip">
                    <span className="legend-dot plddt-confident" style={{ background: '#65cbf3', marginRight: 6 }} />
                    70–90 Confident: <strong>{plddtBands.confident_70_90}</strong>
                  </div>
                  <div className="stat-chip">
                    <span className="legend-dot plddt-low" style={{ background: '#ffdb13', marginRight: 6 }} />
                    50–70 Low: <strong>{plddtBands.low_50_70}</strong>
                  </div>
                  <div className="stat-chip">
                    <span className="legend-dot plddt-very-low" style={{ background: '#ff7d45', marginRight: 6 }} />
                    ≤50 Very low: <strong>{plddtBands.very_low_le_50}</strong>
                  </div>
                </>
              )}
            </div>

            {plddtBands && (
              <div
                aria-label="AlphaFold pLDDT confidence band breakdown"
                style={{
                  display: 'flex',
                  height: 10,
                  borderRadius: 999,
                  overflow: 'hidden',
                  background: '#e2e8f0',
                  marginTop: 8,
                  marginBottom: 10,
                }}
              >
                <div
                  className="plddt-very-high"
                  style={{
                    width: `${(plddtBands.very_high_gt_90 / totalPlddtCount) * 100}%`,
                    background: '#0053d6',
                  }}
                  title={`Very high (>90): ${plddtBands.very_high_gt_90}`}
                />
                <div
                  className="plddt-confident"
                  style={{
                    width: `${(plddtBands.confident_70_90 / totalPlddtCount) * 100}%`,
                    background: '#65cbf3',
                  }}
                  title={`Confident (70–90): ${plddtBands.confident_70_90}`}
                />
                <div
                  className="plddt-low"
                  style={{
                    width: `${(plddtBands.low_50_70 / totalPlddtCount) * 100}%`,
                    background: '#ffdb13',
                  }}
                  title={`Low (50–70): ${plddtBands.low_50_70}`}
                />
                <div
                  className="plddt-very-low"
                  style={{
                    width: `${(plddtBands.very_low_le_50 / totalPlddtCount) * 100}%`,
                    background: '#ff7d45',
                  }}
                  title={`Very low (≤50): ${plddtBands.very_low_le_50}`}
                />
              </div>
            )}

            <div style={{ fontSize: '0.83rem', color: 'var(--text)' }}>
              <strong>Low-confidence positions (1-based, pLDDT &lt; 70):</strong>{' '}
              <span style={{ fontFamily: 'monospace' }}>
                {formatPositionRanges(lowConfPositions)}
              </span>
              {lowConfPositions.length > 0 && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: '2px 8px', fontSize: '0.74rem', marginLeft: 8 }}
                  onClick={() => setShowAllLowPos((v) => !v)}
                >
                  {showAllLowPos ? 'Hide list' : `Show all ${lowConfPositions.length} indices`}
                </button>
              )}
              {showAllLowPos && lowConfPositions.length > 0 && (
                <div
                  style={{
                    marginTop: 6,
                    padding: '8px 10px',
                    background: '#f8fafc',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius)',
                    fontFamily: 'monospace',
                    fontSize: '0.78rem',
                    maxHeight: 90,
                    overflowY: 'auto',
                  }}
                >
                  {lowConfPositions.join(', ')}
                </div>
              )}
            </div>
          </div>

          <div className="compare-layout">
            {/* Left: 3Dmol viewer */}
            <div>
              <div className="card" style={{ marginBottom: 0 }}>
                <div
                  className="card-title"
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: 8,
                  }}
                >
                  <span>
                    AlphaFold Structure —{' '}
                    {colorMode === 'plddt' ? 'colored by pLDDT' : 'colored by min ΔΔG stability'}
                  </span>
                  <div className="mode-toggle" role="group" aria-label="3D structure coloring mode">
                    <button
                      type="button"
                      className={`mode-btn${colorMode === 'plddt' ? ' active' : ''}`}
                      onClick={() => setColorMode('plddt')}
                    >
                      AlphaFold pLDDT
                    </button>
                    <button
                      type="button"
                      className={`mode-btn${colorMode === 'ddg' ? ' active' : ''}`}
                      onClick={() => setColorMode('ddg')}
                      disabled={result.min_ddg_per_position.length === 0}
                    >
                      ΔΔG stability
                    </button>
                  </div>
                </div>

                {lowConfPositions.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <label
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                        textTransform: 'none',
                        letterSpacing: 'normal',
                        fontSize: '0.82rem',
                        color: 'var(--text)',
                        cursor: 'pointer',
                        marginBottom: 0,
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={markLowConf}
                        onChange={(e) => setMarkLowConf(e.target.checked)}
                      />
                      Mark low-confidence positions (pLDDT &lt; 70, {lowConfPositions.length} CA spheres)
                    </label>
                  </div>
                )}

                <div className="viewer-container" ref={viewerRef} />

                {colorMode === 'plddt' ? (
                  <div className="viewer-legend">
                    <div className="legend-item">
                      <span className="legend-dot plddt-very-high" style={{ background: '#0053d6' }} />
                      pLDDT &gt; 90 (very high)
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot plddt-confident" style={{ background: '#65cbf3' }} />
                      70–90 (confident)
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot plddt-low" style={{ background: '#ffdb13' }} />
                      50–70 (low)
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot plddt-very-low" style={{ background: '#ff7d45' }} />
                      ≤ 50 (very low)
                    </div>
                  </div>
                ) : (
                  <div className="viewer-legend">
                    <div className="legend-item">
                      <span className="legend-dot" style={{ background: '#dc2626' }} />
                      Destabilizing (min ΔΔG &lt; −2)
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot" style={{ background: '#f8fafc', border: '1px solid #cbd5e1' }} />
                      Neutral (≈ 0)
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot" style={{ background: '#2563eb' }} />
                      Stabilizing (&gt; +1)
                    </div>
                    {result.sequence.length > result.min_ddg_per_position.length && (
                      <div className="legend-item">
                        <span className="legend-dot" style={{ background: '#94a3b8' }} />
                        Unscanned (&gt;{result.min_ddg_per_position.length})
                      </div>
                    )}
                    {markLowConf && lowConfPositions.length > 0 && (
                      <div className="legend-item">
                        <span className="legend-dot" style={{ background: '#ff7d45' }} />
                        CA marker: pLDDT &lt; 70
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Right: PAE + overlay */}
            <div>
              <div className="card">
                <div className="card-title">AlphaFold PAE (Å)</div>
                <div className="plot-wrap">
                  <Plot
                    data={paeData}
                    layout={paeLayout}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler
                  />
                </div>
              </div>
              <div className="card">
                <div className="card-title">Holo-GNN min ΔΔG vs AlphaFold pLDDT</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 6 }}>
                  Red (left axis): most destabilizing predicted mutation per position.
                  Blue dashed (right): AlphaFold pLDDT.
                  Shaded amber regions &amp; markers indicate pLDDT &lt; 70 (<code>low_confidence_in_scan</code>).
                </div>
                {result.sequence.length > result.min_ddg_per_position.length && (
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                    Scan curve covers positions 1–{result.min_ddg_per_position.length} of {result.sequence.length}.
                  </div>
                )}
                <div className="plot-wrap">
                  <Plot
                    data={overlayData}
                    layout={overlayLayout}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler
                  />
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
