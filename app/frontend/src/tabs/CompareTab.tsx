import { useState, useRef, useEffect, useMemo } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import * as $3Dmol from '3dmol'
import { postCompare } from '../api'
import { setCompare } from '../store'
import type { CompareResponse, PlddtBands, TabProps } from '../types'
import {
  BASE_LAYOUT,
  COLORS,
  PAE_SCALE,
  PLOT_CONFIG,
  axisTitle,
  baseAxis,
  colorbarTitle,
} from '../plotTheme'

/** AlphaFold's own pLDDT confidence colours (kept identical to AlphaFold DB for recognisability). */
const PLDDT_COLORS = {
  veryHigh: '#0053d6',
  confident: '#65cbf3',
  low: '#ffdb13',
  veryLow: '#ff7d45',
}

/** Studio-light canvas: harmonises with the card surface instead of a black hole in the page. */
const VIEWER_BG = '#f3f5f8'

function plddt2color(bfactor: number): string {
  if (bfactor > 90) return PLDDT_COLORS.veryHigh
  if (bfactor >= 70) return PLDDT_COLORS.confident
  if (bfactor > 50) return PLDDT_COLORS.low
  return PLDDT_COLORS.veryLow
}

/**
 * Diverging ribbon colour for per-position min ΔΔG (ColorBrewer RdBu). The midpoint is a
 * light neutral grey rather than white, so near-zero residues stay visible on the light canvas.
 */
const DDG_RIBBON_STOPS: Array<[number, [number, number, number]]> = [
  [-3.5, [103, 0, 31]],
  [-2.5, [178, 24, 43]],
  [-1.5, [214, 96, 77]],
  [-0.6, [244, 165, 130]],
  [0.0, [214, 219, 226]],
  [0.6, [146, 197, 222]],
  [1.5, [67, 147, 195]],
  [3.5, [33, 102, 172]],
]
const DDG_CLAMP = 3.5

function ddg2color(ddg: number): string {
  const v = Math.max(-DDG_CLAMP, Math.min(DDG_CLAMP, ddg))
  for (let i = 0; i < DDG_RIBBON_STOPS.length - 1; i++) {
    const [x0, c0] = DDG_RIBBON_STOPS[i]
    const [x1, c1] = DDG_RIBBON_STOPS[i + 1]
    if (v >= x0 && v <= x1) {
      const t = x1 === x0 ? 0 : (v - x0) / (x1 - x0)
      const c = c0.map((a, k) => Math.round(a + (c1[k] - a) * t))
      return `rgb(${c[0]},${c[1]},${c[2]})`
    }
  }
  return 'rgb(214,219,226)'
}

const DDG_GRADIENT_CSS = `linear-gradient(90deg, ${DDG_RIBBON_STOPS.map(
  ([x, c]) => `rgb(${c.join(',')}) ${(((x + DDG_CLAMP) / (2 * DDG_CLAMP)) * 100).toFixed(1)}%`,
).join(', ')})`

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

  const plddtBands: PlddtBands | null = result ? (result.plddt_bands ?? computeBandsFallback(result.plddt)) : null

  const lowConfPositions: number[] = useMemo(
    () =>
      result
        ? (result.low_confidence_positions ?? result.plddt.map((v, i) => (v < 70 ? i + 1 : -1)).filter((p) => p > 0))
        : [],
    [result]
  )

  const lowConfInScan: boolean[] = useMemo(
    () =>
      result
        ? (result.low_confidence_in_scan ?? result.min_ddg_per_position.map((_, i) => (result.plddt[i] ?? 100) < 70))
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
      const axiosErr = err as { response?: { status?: number; data?: { detail?: string } }; message?: string }
      if (axiosErr.response?.status === 502) {
        setError(
          'Could not reach AlphaFold / UniProt servers (502 Bad Gateway). ' +
            'Check your internet connection or verify the UniProt accession is valid.'
        )
      } else {
        setError(axiosErr.response?.data?.detail ?? axiosErr.message ?? 'Request failed')
      }
    } finally {
      setLoading(false)
    }
  }

  // Initialise / reuse the 3Dmol viewer whenever the mounted container or PDB changes, and
  // update colours/markers on toggle without resetting the camera.
  useEffect(() => {
    if (!result?.structure_pdb || !viewerRef.current) return

    let viewer = viewerInstance.current
    if (!viewer || viewerElementRef.current !== viewerRef.current) {
      viewer = $3Dmol.createViewer(viewerRef.current, { backgroundColor: VIEWER_BG, antialias: true })
      try {
        // thin ink outline keeps pale ribbons (pLDDT yellow, near-zero ΔΔG) legible on a light canvas
        viewer.setViewStyle({ style: 'outline', color: '#3b4554', width: 0.035 })
      } catch {
        /* outline is cosmetic; older 3Dmol builds may not support it */
      }
      viewerInstance.current = viewer
      viewerElementRef.current = viewerRef.current
      loadedPdbRef.current = null
    }

    if (loadedPdbRef.current !== result.structure_pdb) {
      viewer.clear()
      viewer.addModel(result.structure_pdb, 'pdb')
      viewer.zoomTo()
      viewer.zoom(0.9) // headroom so the caption and legend overlays never cover the model
      loadedPdbRef.current = result.structure_pdb
    }

    const colorfunc =
      colorMode === 'plddt'
        ? (atom: $3Dmol.AtomSpec) => plddt2color(atom.b ?? 0)
        : (atom: $3Dmol.AtomSpec) => {
            const idx = (atom.resi ?? 0) - 1
            if (idx < 0 || idx >= result.min_ddg_per_position.length) return '#a7b0bd'
            return ddg2color(result.min_ddg_per_position[idx])
          }

    viewer.setStyle({}, { cartoon: { colorfunc } })

    if (markLowConf && lowConfPositions.length > 0) {
      viewer.addStyle(
        { resi: lowConfPositions, atom: 'CA' },
        { sphere: { radius: 0.5, color: PLDDT_COLORS.veryLow, opacity: 0.9 } }
      )
    }

    viewer.render()
  }, [result, colorMode, markLowConf, lowConfPositions])

  // ---- PAE: axes in residue numbers, not in (possibly downsampled) cell indices ----
  const paeStride = result?.pae_stride ?? 1
  const paeAxis = result ? result.pae.map((_, k) => 1 + k * paeStride) : []
  const paeMax = result ? Math.max(30, ...result.pae.map((row) => Math.max(...row))) : 30

  const paeData = result
    ? [
        {
          type: 'heatmap',
          x: paeAxis,
          y: paeAxis,
          z: result.pae,
          colorscale: PAE_SCALE,
          zmin: 0,
          zmax: Math.ceil(paeMax),
          colorbar: {
            title: colorbarTitle('Expected position error (Å)'),
            thickness: 12,
            len: 1,
            outlinewidth: 0,
            ticks: 'outside',
            ticklen: 3,
            tickfont: { size: 10.5, color: COLORS.inkMuted },
          },
          hovertemplate: 'Scored residue %{x}<br>Aligned residue %{y}<br>PAE %{z:.1f} Å<extra></extra>',
        },
      ]
    : []

  const paeLayout = {
    ...BASE_LAYOUT,
    height: 420,
    margin: { l: 58, r: 20, t: 10, b: 50 },
    xaxis: baseAxis({
      title: axisTitle('Scored residue'),
      showgrid: false,
      constrain: 'domain',
      range: result ? [0.5, (paeAxis[paeAxis.length - 1] ?? 1) + paeStride - 0.5] : undefined,
    }),
    yaxis: baseAxis({
      title: axisTitle('Aligned residue'),
      range: result ? [(paeAxis[paeAxis.length - 1] ?? 1) + paeStride - 0.5, 0.5] : undefined,
      autorange: result ? false : 'reversed',
      showgrid: false,
      scaleanchor: 'x',
      constrain: 'domain',
    }),
  }

  // ---- min ΔΔG vs pLDDT overlay ----
  const nScan = result?.min_ddg_per_position.length ?? 0
  const overlayData = result
    ? [
        {
          type: 'scatter',
          x: Array.from({ length: nScan }, (_, i) => i + 1),
          y: result.min_ddg_per_position,
          mode: 'lines',
          line: { color: COLORS.destabilizing, width: 1.4 },
          name: 'Holo-GeoN min ΔΔG per residue (left)',
          yaxis: 'y',
          hovertemplate: 'Residue %{x}<br>Min ΔΔG %{y:.2f} kcal/mol<extra></extra>',
        },
        {
          type: 'scatter',
          x: Array.from({ length: result.plddt.length }, (_, i) => i + 1),
          y: result.plddt,
          mode: 'lines',
          line: { color: COLORS.plddt, width: 1.6, dash: 'dot' },
          name: 'AlphaFold pLDDT (right)',
          yaxis: 'y2',
          hovertemplate: 'Residue %{x}<br>pLDDT %{y:.1f}<extra></extra>',
        },
        {
          // legend key for the shaded spans (shapes have no legend entry of their own)
          type: 'scatter',
          x: [null],
          y: [null],
          mode: 'markers',
          marker: { symbol: 'square', size: 12, color: COLORS.lowConfidence, line: { color: COLORS.lowConfidenceLine, width: 1 } },
          name: 'pLDDT < 70',
          hoverinfo: 'skip',
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
    fillcolor: COLORS.lowConfidence,
    line: { width: 0 },
    layer: 'below',
  }))

  const overlayLayout = {
    ...BASE_LAYOUT,
    height: 320,
    margin: { l: 64, r: 64, t: 44, b: 50 },
    xaxis: baseAxis({
      title: axisTitle('Residue position'),
      range: [0.5, Math.max(nScan, result?.plddt.length ?? 1) + 0.5],
      showgrid: false,
    }),
    yaxis: baseAxis({
      title: axisTitle('Min ΔΔG (kcal/mol)', { color: COLORS.destabilizing }),
      tickfont: { size: 11, color: COLORS.destabilizing },
      zeroline: true,
      zerolinecolor: COLORS.zero,
    }),
    yaxis2: baseAxis({
      title: axisTitle('pLDDT', { color: COLORS.plddt }),
      tickfont: { size: 11, color: COLORS.plddt },
      overlaying: 'y',
      side: 'right',
      range: [0, 100],
      showgrid: false,
      tickvals: [0, 50, 70, 90, 100],
    }),
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
        line: { color: COLORS.lowConfidenceLine, width: 1, dash: 'dash' },
      },
    ],
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.02,
      xanchor: 'left',
      yanchor: 'bottom',
      font: { size: 11.5, color: COLORS.ink },
      bgcolor: 'rgba(0,0,0,0)',
    },
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
            <button type="submit" className="btn btn-primary" disabled={loading || !uniprotId.trim()}>
              {loading && <span className="spinner" />}
              {loading ? 'Fetching…' : 'Compare'}
            </button>
          </div>
          <div className="field-hint">AlphaFold DB model + PAE from EBI; structure-free Holo-GeoN scan of the first 400 residues.</div>
        </form>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <>
          <div className="card">
            <div className="card-header-row">
              <div className="card-title card-title-flat">AlphaFold pLDDT Confidence — {result.uniprot_id}</div>
              <span className="symmetry-badge">Scan: Structure-Free · Reads No Coordinates</span>
            </div>
            <div className="stat-chips">
              <div className="stat-chip">Mean pLDDT <strong>{plddtMean.toFixed(1)}</strong></div>
              <div className="stat-chip">Length <strong>{result.sequence.length} aa</strong></div>
              <div className="stat-chip">pLDDT &lt; 70 <strong>{lowConfPositions.length} residues</strong></div>
            </div>

            {plddtBands && (
              <div className="band-bar" aria-label="AlphaFold pLDDT confidence band breakdown">
                <div className="plddt-very-high" style={{ width: `${(plddtBands.very_high_gt_90 / totalPlddtCount) * 100}%`, background: PLDDT_COLORS.veryHigh }} title={`Very high (>90): ${plddtBands.very_high_gt_90}`} />
                <div className="plddt-confident" style={{ width: `${(plddtBands.confident_70_90 / totalPlddtCount) * 100}%`, background: PLDDT_COLORS.confident }} title={`Confident (70–90): ${plddtBands.confident_70_90}`} />
                <div className="plddt-low" style={{ width: `${(plddtBands.low_50_70 / totalPlddtCount) * 100}%`, background: PLDDT_COLORS.low }} title={`Low (50–70): ${plddtBands.low_50_70}`} />
                <div className="plddt-very-low" style={{ width: `${(plddtBands.very_low_le_50 / totalPlddtCount) * 100}%`, background: PLDDT_COLORS.veryLow }} title={`Very low (≤50): ${plddtBands.very_low_le_50}`} />
              </div>
            )}
            {plddtBands && (
              <div className="band-legend" aria-hidden="true">
                <span><span className="legend-dot" style={{ background: PLDDT_COLORS.veryHigh }} />&gt;90 <strong>{plddtBands.very_high_gt_90}</strong></span>
                <span><span className="legend-dot" style={{ background: PLDDT_COLORS.confident }} />70–90 <strong>{plddtBands.confident_70_90}</strong></span>
                <span><span className="legend-dot" style={{ background: PLDDT_COLORS.low }} />50–70 <strong>{plddtBands.low_50_70}</strong></span>
                <span><span className="legend-dot" style={{ background: PLDDT_COLORS.veryLow }} />≤50 <strong>{plddtBands.very_low_le_50}</strong></span>
              </div>
            )}

            <div className="low-conf-row">
              <span className="low-conf-label">Low-confidence positions (pLDDT &lt; 70):</span>{' '}
              <span className="mono">{formatPositionRanges(lowConfPositions)}</span>
              {lowConfPositions.length > 0 && (
                <button type="button" className="btn-link" onClick={() => setShowAllLowPos((v) => !v)}>
                  {showAllLowPos ? 'Hide list' : `Show all ${lowConfPositions.length} indices`}
                </button>
              )}
              {showAllLowPos && lowConfPositions.length > 0 && (
                <div className="index-list">{lowConfPositions.join(', ')}</div>
              )}
            </div>

            <div className="context-note" role="note" title={plddtNote}>
              <span className="context-glyph" aria-hidden="true">ⓘ</span>
              <span>
                pLDDT describes the AlphaFold model, not the prediction (the scan reads no coordinates); below 70, read
                ΔΔG with caution.
                {result.pae_downsampled && <> PAE drawn every {paeStride} residues.</>}
              </span>
            </div>
          </div>

          <div className="compare-layout">
            {/* Left: 3Dmol viewer */}
            <div className="card compare-panel">
              <div className="card-header-row">
                <div className="card-title card-title-flat">
                  AlphaFold Structure — {colorMode === 'plddt' ? 'colored by pLDDT' : 'colored by min ΔΔG stability'}
                </div>
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

              <div className="viewer-stage">
                <div className="viewer-container" ref={viewerRef} />
                <div className="viewer-caption" aria-hidden="true">
                  {result.uniprot_id} · AlphaFold DB model
                  {lowConfPositions.length > 0 && markLowConf ? ' · spheres: pLDDT < 70' : ''}
                </div>
                <div className="viewer-legend-overlay">
                  {colorMode === 'plddt' ? (
                    <div className="viewer-legend">
                      <div className="legend-item"><span className="legend-dot" style={{ background: PLDDT_COLORS.veryHigh }} />&gt; 90 very high</div>
                      <div className="legend-item"><span className="legend-dot" style={{ background: PLDDT_COLORS.confident }} />70–90 confident</div>
                      <div className="legend-item"><span className="legend-dot" style={{ background: PLDDT_COLORS.low }} />50–70 low</div>
                      <div className="legend-item"><span className="legend-dot" style={{ background: PLDDT_COLORS.veryLow }} />≤ 50 very low</div>
                    </div>
                  ) : (
                    <div className="ramp-legend">
                      <div className="ramp-title">Min ΔΔG per residue (kcal/mol)</div>
                      <div className="ramp-bar" style={{ background: DDG_GRADIENT_CSS }} />
                      <div className="ramp-ticks">
                        <span>≤ −{DDG_CLAMP}</span>
                        <span>0</span>
                        <span>≥ +{DDG_CLAMP}</span>
                      </div>
                      <div className="ramp-ticks ramp-words">
                        <span style={{ color: COLORS.destabilizing }}>destabilizing</span>
                        <span style={{ color: COLORS.stabilizing }}>stabilizing</span>
                      </div>
                      {result.sequence.length > result.min_ddg_per_position.length && (
                        <div className="ramp-note">
                          <span className="legend-dot" style={{ background: '#a7b0bd' }} /> unscanned (&gt; {result.min_ddg_per_position.length})
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {lowConfPositions.length > 0 && (
                <label className="check-label" style={{ marginTop: 10 }}>
                  <input type="checkbox" checked={markLowConf} onChange={(e) => setMarkLowConf(e.target.checked)} />
                  Mark low-confidence positions (pLDDT &lt; 70, {lowConfPositions.length} Cα spheres)
                </label>
              )}
            </div>

            {/* Right: PAE */}
            <div className="card compare-panel">
              <div className="card-header-row">
                <div className="card-title card-title-flat">AlphaFold Predicted Aligned Error</div>
              </div>
              <div className="plot-wrap">
                <Plot data={paeData} layout={paeLayout} config={PLOT_CONFIG} style={{ width: '100%' }} useResizeHandler />
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header-row">
              <div
                className="card-title card-title-flat"
                title="Min ΔΔG: the most destabilizing predicted substitution at each residue. Shaded: pLDDT < 70, where a folded-state ΔΔG is less meaningful."
              >
                Holo-GeoN min ΔΔG vs AlphaFold pLDDT
              </div>
              {result.sequence.length > nScan && (
                <span className="title-meta">scan covers residues 1–{nScan} of {result.sequence.length}</span>
              )}
            </div>
            <div className="plot-wrap">
              <Plot data={overlayData} layout={overlayLayout} config={PLOT_CONFIG} style={{ width: '100%' }} useResizeHandler />
            </div>
          </div>
        </>
      )}
    </div>
  )
}
