import { useState, useRef, useEffect, useMemo } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import { streamScan, downloadExport } from '../api'
import { setScan } from '../store'
import type { PredictionMode, ScanResponse, ScanStreamStart, TabProps } from '../types'
import { useMeasuredWidth } from '../hooks'
import {
  BASE_LAYOUT,
  COLORS,
  DDG_DIVERGING,
  MONO_FAMILY,
  PLOT_CONFIG,
  axisTitle,
  baseAxis,
  colorbarTitle,
  niceStride,
  robustAbsLimit,
} from '../plotTheme'

const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

/** Pixel geometry of the landscape figure. Margins are fixed so the cell grid is predictable. */
const HM = {
  marginL: 58,
  marginR: 104,   // colorbar + its title
  marginT: 10,
  marginB: 64,    // rotated residue labels + axis title
  rowPx: 19,      // per substituted amino acid
  trackPx: 74,    // mean-ΔΔG track above the heatmap
  gapPx: 10,
  minColPx: 7,    // below this the figure scrolls horizontally instead of squeezing
  minLabelPx: 13, // rotated 10.5 px labels need this much room each
}

function cleanSeqText(raw: string): string {
  return raw
    .split('\n')
    .filter((l) => !l.trimStart().startsWith('>'))
    .join('')
    .replace(/\s+/g, '')
    .toUpperCase()
}

interface ScanProgress {
  done: number
  total: number
  elapsed_s: number
  eta_s: number
}

export default function ScanTab({ health = null }: Partial<TabProps>) {
  const [sequence, setSequence] = useState(UBIQUITIN)
  const [start, setStart] = useState(1)
  const [end, setEnd] = useState(Math.min(UBIQUITIN.length, 60))
  const [numberingStart, setNumberingStart] = useState(1)
  const [cds, setCds] = useState('')
  const [mode, setMode] = useState<PredictionMode>('structure_free')
  const [structurePdb, setStructurePdb] = useState('')
  const [pdbFilename, setPdbFilename] = useState<string | null>(null)

  const [loading, setLoading] = useState(false)
  const [cancelled, setCancelled] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<(ScanResponse & { sequence?: string }) | null>(null)
  const [streamStart, setStreamStart] = useState<ScanStreamStart | null>(null)
  const [progress, setProgress] = useState<ScanProgress | null>(null)
  const [liveColumns, setLiveColumns] = useState<(number[] | null)[]>([])
  const [submittedSeq, setSubmittedSeq] = useState(UBIQUITIN)
  const [submittedOffset, setSubmittedOffset] = useState(0)
  const [exporting, setExporting] = useState<string | null>(null)

  const abortCtrlRef = useRef<AbortController | null>(null)
  const [plotBoxRef, plotBoxWidth] = useMeasuredWidth<HTMLDivElement>()

  useEffect(() => {
    return () => {
      abortCtrlRef.current?.abort()
    }
  }, [])

  const cdsCap = health?.capabilities?.cds
  const cdsAvailable = Boolean(cdsCap?.available)
  const mode3dCap = health?.capabilities?.invariant_3d
  const mode3dAvailable = Boolean(mode3dCap?.available)

  const effectiveMode: PredictionMode = mode === 'invariant_3d' && mode3dAvailable ? 'invariant_3d' : 'structure_free'
  const needsStructure = effectiveMode === 'invariant_3d'
  const cleanLen = cleanSeqText(sequence).length
  const firstNo = Number.isFinite(numberingStart) && numberingStart >= 1 ? Math.floor(numberingStart) : 1

  function onSeqChange(val: string) {
    setSequence(val)
    const len = cleanSeqText(val).length
    setEnd(Math.min(len, 60))
  }

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

  function handleCancel() {
    abortCtrlRef.current?.abort()
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    abortCtrlRef.current?.abort()
    const controller = new AbortController()
    abortCtrlRef.current = controller

    const seqClean = cleanSeqText(sequence)
    setSubmittedSeq(seqClean)
    setSubmittedOffset(firstNo - 1)
    setError(null)
    setCancelled(false)
    setResult(null)
    setStreamStart(null)
    setProgress(null)
    setLiveColumns([])
    setLoading(true)

    let firstPos = Math.max(1, start)

    try {
      await streamScan(
        {
          sequence: sequence.trim(),
          start,
          end,
          cds: cdsAvailable && cds.trim() ? cds.trim() : undefined,
          mode: effectiveMode,
          structure_pdb: needsStructure && structurePdb.trim() ? structurePdb : undefined,
        },
        {
          onStart(info) {
            firstPos = info.positions[0]
            setStreamStart(info)
            setProgress({ done: 0, total: info.total, elapsed_s: 0, eta_s: 0 })
            setLiveColumns(new Array<number[] | null>(info.total).fill(null))
          },
          onChunk(chunk) {
            setProgress({ done: chunk.done, total: chunk.total, elapsed_s: chunk.elapsed_s, eta_s: chunk.eta_s })
            setLiveColumns((prev) => {
              const next = prev.slice()
              for (let i = 0; i < chunk.positions.length; i++) {
                const idx = chunk.positions[i] - firstPos
                if (idx >= 0 && idx < next.length) {
                  next[idx] = chunk.columns[i]
                }
              }
              return next
            })
          },
          onResult(res) {
            const withSeq = { ...res, sequence: seqClean }
            setResult(withSeq)
            setScan(withSeq)
            setProgress((prev) =>
              prev
                ? { ...prev, done: prev.total }
                : { done: res.positions.length, total: res.positions.length, elapsed_s: res.timing.scan_ms / 1000, eta_s: 0 }
            )
          },
        },
        controller.signal
      )
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setCancelled(true)
      } else if ((err as { name?: string })?.name === 'AbortError') {
        setCancelled(true)
      } else {
        const msg = err as { response?: { data?: { detail?: string } }; message?: string }
        setError(msg?.response?.data?.detail ?? msg?.message ?? 'Scan request failed')
      }
    } finally {
      setLoading(false)
      if (abortCtrlRef.current === controller) {
        abortCtrlRef.current = null
      }
    }
  }

  async function handleExport(format: 'csv' | 'pdb') {
    if (!result) return
    setExporting(format)
    try {
      const payload = format === 'pdb' ? { ...result, sequence: result.sequence ?? cleanSeqText(sequence) } : result
      await downloadExport(format, payload, `scan_${start}-${end}.${format}`)
    } catch (err: unknown) {
      setError((err as Error)?.message ?? 'Export failed')
    } finally {
      setExporting(null)
    }
  }

  // ---- matrix (final, or partially filled while streaming) ----
  const grid = useMemo(() => {
    let positions: number[] = []
    let wt: string[] = []
    let aa: string[] = []
    let z: (number | null)[][] = []
    if (result) {
      positions = result.positions
      wt = result.positions.map((_, i) => result.wt_residues[i] ?? '?')
      aa = result.aa_order
      z = result.matrix
    } else if (streamStart && liveColumns.some((c) => c !== null)) {
      const first = streamStart.positions[0]
      positions = Array.from({ length: streamStart.total }, (_, i) => first + i)
      wt = positions.map((p) => submittedSeq[p - 1] ?? '?')
      aa = streamStart.aa_order
      z = aa.map((_, r) => positions.map((__, c) => (liveColumns[c] ? liveColumns[c]![r] : null)))
    }
    const labels = positions.map((p, i) => `${wt[i]}${p + submittedOffset}`)
    const means = positions.map((_, c) => {
      const vals = z.map((row) => row[c]).filter((v): v is number => typeof v === 'number')
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
    })
    return { positions, wt, aa, z, labels, means }
  }, [result, streamStart, liveColumns, submittedSeq, submittedOffset])

  const nCols = grid.positions.length
  const hasHeatmap = nCols > 0 && grid.aa.length > 0

  // ---- geometry: fit the card; scroll only when columns would drop below minColPx ----
  const boxW = Math.max(plotBoxWidth, 480)
  const fitArea = boxW - HM.marginL - HM.marginR
  const colPx = nCols > 0 ? Math.max(HM.minColPx, fitArea / nCols) : 0
  const plotWidth = nCols > 0 ? Math.max(boxW, Math.ceil(nCols * colPx + HM.marginL + HM.marginR)) : boxW
  const scrolls = plotWidth > boxW + 1
  const stride = niceStride(colPx, HM.minLabelPx)
  const tickIdx = grid.positions
    .map((p, i) => ({ p: p + submittedOffset, i }))
    .filter(({ p, i }) => stride === 1 || p % stride === 0 || i === 0)
    .map(({ i }) => i)
  const heatH = grid.aa.length * HM.rowPx
  const plotH = HM.marginT + HM.trackPx + HM.gapPx + heatH + HM.marginB
  const innerH = HM.trackPx + HM.gapPx + heatH
  const heatTop = heatH / innerH

  const zLim = useMemo(() => robustAbsLimit(grid.z.flat(), 0.98, 1), [grid.z])
  const meanLim = useMemo(() => {
    const m = grid.means.filter((v): v is number => v !== null).map(Math.abs)
    return Math.max(0.5, ...(m.length ? m : [0.5])) * 1.08
  }, [grid.means])
  const meanTickStep = Math.max(1, Math.floor(meanLim))
  const meanTicks = [-meanTickStep, 0, meanTickStep]

  const wtX: string[] = []
  const wtY: string[] = []
  grid.labels.forEach((lab, i) => {
    if (grid.aa.includes(grid.wt[i])) {
      wtX.push(lab)
      wtY.push(grid.wt[i])
    }
  })

  const plotData = hasHeatmap
    ? [
        {
          type: 'bar',
          x: grid.labels,
          y: grid.means.map((v) => (v === null ? 0 : v)),
          xaxis: 'x',
          yaxis: 'y2',
          marker: {
            color: grid.means.map((v) => (v === null ? 'rgba(0,0,0,0)' : v < 0 ? '#d6604d' : '#4393c3')),
            line: { width: 0 },
          },
          hovertemplate: '<b>%{x}</b><br>mean ΔΔG over 20 substitutions: %{y:.2f} kcal/mol<extra></extra>',
          showlegend: false,
        },
        {
          type: 'heatmap',
          x: grid.labels,
          y: grid.aa,
          z: grid.z,
          xaxis: 'x',
          yaxis: 'y',
          colorscale: DDG_DIVERGING,
          zmin: -zLim,
          zmax: zLim,
          zmid: 0,
          xgap: colPx >= 9 ? 1 : 0,
          ygap: 1,
          hoverongaps: false,
          colorbar: {
            title: colorbarTitle('ΔΔG (kcal/mol)'),
            thickness: 12,
            len: heatTop,
            lenmode: 'fraction',
            y: 0,
            yanchor: 'bottom',
            x: 1,
            xpad: 12,
            xanchor: 'left',
            outlinewidth: 0,
            ticks: 'outside',
            ticklen: 3,
            tickfont: { size: 10.5, color: COLORS.inkMuted },
          },
          hovertemplate: '<b>%{x} → %{y}</b><br>ΔΔG %{z:.3f} kcal/mol<extra></extra>',
        },
        {
          type: 'scatter',
          mode: 'markers',
          x: wtX,
          y: wtY,
          xaxis: 'x',
          yaxis: 'y',
          marker: { size: Math.max(3, Math.min(6, colPx * 0.32)), color: '#1f2937', symbol: 'circle' },
          hovertemplate: '<b>%{x}</b>: wild-type residue %{y}<extra></extra>',
          showlegend: false,
        },
      ]
    : []

  const plotLayout = {
    ...BASE_LAYOUT,
    width: plotWidth,
    height: plotH,
    margin: { l: HM.marginL, r: HM.marginR, t: HM.marginT, b: HM.marginB },
    xaxis: baseAxis({
      type: 'category',
      anchor: 'y',
      // explicit range: the wild-type marker trace would otherwise pad the category axis
      autorange: false,
      range: [-0.5, nCols - 0.5],
      tickmode: 'array',
      tickvals: tickIdx.map((i) => grid.labels[i]),
      ticktext: tickIdx.map((i) => grid.labels[i]),
      tickangle: -90,
      tickfont: { family: MONO_FAMILY, size: 10.5, color: COLORS.inkMuted },
      showgrid: false,
      showline: false,
      automargin: false,
      title: axisTitle(
        submittedOffset ? `Residue (wild type + native number, first residue ${submittedOffset + 1})` : 'Residue (wild type + position)',
        { standoff: 6 }),
    }),
    yaxis: baseAxis({
      domain: [0, heatTop],
      type: 'category',
      categoryorder: 'array',
      categoryarray: grid.aa,
      autorange: false,
      range: [grid.aa.length - 0.5, -0.5], // first amino acid on top, no marker padding
      tickfont: { family: MONO_FAMILY, size: 11, color: COLORS.ink },
      showgrid: false,
      showline: false,
      ticks: '',
      automargin: false,
      title: axisTitle('Substituted residue', { standoff: 4 }),
    }),
    yaxis2: baseAxis({
      domain: [heatTop + HM.gapPx / innerH, 1],
      anchor: 'x',
      range: [-meanLim, meanLim],
      zeroline: true,
      zerolinecolor: COLORS.zero,
      showline: false,
      tickmode: 'array',
      tickvals: meanTicks,
      tickformat: '.0f',
      automargin: false,
      title: axisTitle('Mean', { size: 11, standoff: 4 }),
    }),
    bargap: 0.12,
    showlegend: false,
  }

  const activeMeta = result ?? streamStart
  const pct = progress ? Math.min(100, Math.round((progress.done / Math.max(progress.total, 1)) * 100)) : 0
  const firstLabel = grid.labels[0]
  const lastLabel = grid.labels[grid.labels.length - 1]

  return (
    <div>
      <div className="card">
        <div className="card-title">Stability Landscape — Saturation Scan</div>
        <form onSubmit={handleSubmit}>
          <div className="form-field">
            <div className="label-row">
              <label htmlFor="scan-seq">Protein Sequence</label>
              <span className="field-meta">{cleanLen} residues</span>
            </div>
            <textarea
              id="scan-seq"
              rows={3}
              value={sequence}
              onChange={(e) => onSeqChange(e.target.value)}
              placeholder="Paste single-letter amino acid sequence..."
              className="seq-input"
            />
          </div>

          <div className="input-row form-field">
            <div className="field-sm">
              <label htmlFor="scan-start" title="Start and end index the sequence as pasted (1-based)">Start</label>
              <input
                id="scan-start"
                type="number"
                min={1}
                max={Math.max(1, cleanLen)}
                value={start}
                onChange={(e) => setStart(Number(e.target.value))}
              />
            </div>
            <div className="field-sm">
              <label htmlFor="scan-end" title="Inclusive, 1-based index into the sequence as pasted">End</label>
              <input
                id="scan-end"
                type="number"
                min={start}
                max={Math.max(start, cleanLen)}
                value={end}
                onChange={(e) => setEnd(Number(e.target.value))}
              />
            </div>
            <div className="field-sm">
              <label
                htmlFor="scan-numbering"
                title="Relabels the axis only, e.g. 94 for a TP53 DBD construct (residues 94–292). Start/End still index the pasted sequence."
              >
                First residue no.<span className="info-tip" aria-hidden="true">i</span>
              </label>
              <input
                id="scan-numbering"
                type="number"
                min={1}
                value={numberingStart}
                onChange={(e) => setNumberingStart(Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: '2 1 260px' }}>
              <label>Inference Mode</label>
              <div className="mode-toggle" role="group" aria-label="Scan inference mode">
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
                  title={!mode3dAvailable ? mode3dCap?.reason ?? undefined : undefined}
                >
                  SE(3)-invariant 3D
                </button>
              </div>
            </div>
          </div>
          {!mode3dAvailable && (
            <div className="capability-notice capability-disabled" style={{ marginBottom: 12 }}>
              <strong>SE(3)-invariant 3D mode unavailable:</strong>{' '}
              {mode3dCap?.reason ?? 'Waiting for server capability status…'}
            </div>
          )}

          {needsStructure && (
            <div className="structure-box form-field">
              <div className="label-row" style={{ marginBottom: 8 }}>
                <label htmlFor="scan-pdb" style={{ marginBottom: 0 }}>
                  Structure PDB (required for 3D mode — B-factor read as pLDDT)
                </label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    id="scan-pdb-file"
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
                id="scan-pdb"
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
                    : 'Supply a structure of this exact protein.'}
              </div>
            </div>
          )}

          <details className="form-disclosure" open={Boolean(cds)}>
            <summary>
              Optional coding sequence (CDS) <span className="summary-meta">mRNA folding &amp; codon-usage channels</span>
            </summary>
            <div style={{ marginTop: 8 }}>
              <div className="label-row">
                <label htmlFor="scan-cds">Coding Sequence (DNA/RNA, FASTA ok, one stop codon ok)</label>
                {cds && cdsAvailable && (
                  <button type="button" className="btn-link" onClick={() => setCds('')}>
                    Clear CDS
                  </button>
                )}
              </div>
              <textarea
                id="scan-cds"
                rows={2}
                value={cds}
                disabled={!cdsAvailable}
                onChange={(e) => setCds(e.target.value)}
                placeholder={cdsAvailable ? 'Optional: coding sequence encoding this exact protein...' : 'CDS input disabled (unavailable on this server)'}
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
            <button type="submit" className="btn btn-primary" disabled={loading || !cleanLen || (needsStructure && !structurePdb.trim())}>
              {loading && <span className="spinner" />}
              {loading ? 'Streaming Scan…' : 'Run Scan'}
            </button>
            {loading && (
              <button type="button" className="btn btn-danger" onClick={handleCancel}>
                Cancel Scan
              </button>
            )}
            <span className="btn-aside">{Math.max(0, end - start + 1)} positions × 20 substitutions</span>
          </div>
        </form>

        {error && <div className="error-msg">{error}</div>}
        {cancelled && (
          <div className="capability-notice capability-disabled" style={{ marginTop: 12 }}>
            Scan cancelled{progress ? ` (${progress.done} / ${progress.total} sites received)` : ''}.
          </div>
        )}
      </div>

      {(loading || progress || activeMeta) && (
        <div className="card">
          <div className="card-header-row">
            <div className="card-title card-title-flat">
              {result ? 'ΔΔG Landscape' : 'Live Scan Progress & Incremental Heatmap'}
              {hasHeatmap && (
                <span className="title-meta">
                  {' '}· {firstLabel}–{lastLabel} · {nCols} positions × {grid.aa.length} substitutions
                </span>
              )}
            </div>
            {activeMeta && (
              <span className="symmetry-badge">
                {activeMeta.mode === 'invariant_3d' ? 'SE(3)-Invariant 3D' : 'Structure-Free · Swap-Equivariant Only'}
              </span>
            )}
          </div>

          {activeMeta && (
            <div className="meta-line" title={activeMeta.symmetry}>
              {result?.timing && (
                <>
                  <span><strong>{result.timing.total_mutations.toLocaleString()}</strong> mutations</span>
                  <span>scan <strong>{(result.timing.scan_ms / 1000).toFixed(2)} s</strong></span>
                  <span><strong>{Math.round(result.timing.muts_per_sec).toLocaleString()}</strong> mut/s</span>
                </>
              )}
              {activeMeta.cds?.supplied && (
                <span>CDS <strong>verified ({activeMeta.cds.n_codons ?? '?'} codons)</strong></span>
              )}
              {activeMeta.structure && (
                <>
                  <span>mean pLDDT <strong>{activeMeta.structure.mean_plddt.toFixed(1)}</strong></span>
                  {activeMeta.structure.low_confidence_positions.length > 0 && (
                    <span className="meta-warn">pLDDT &lt; 70 <strong>{activeMeta.structure.low_confidence_positions.length} sites</strong></span>
                  )}
                  {(activeMeta.structure.residues_without_coordinates?.length ?? 0) > 0 && (
                    <span className="meta-warn">
                      <strong>{activeMeta.structure.residues_without_coordinates?.length} sites</strong> without coordinates (masked)
                    </span>
                  )}
                </>
              )}
            </div>
          )}

          {progress && (loading || !result) && (
            <div className="progress-wrap" style={{ marginBottom: 14 }} aria-live="off">
              <div className="progress-header">
                <span>
                  Scanning residues: <strong>{progress.done}</strong> / <strong>{progress.total}</strong> sites ({pct}%)
                </span>
                <span className="progress-meta">
                  Elapsed <strong>{progress.elapsed_s.toFixed(2)}s</strong> · ETA <strong>{progress.eta_s.toFixed(2)}s</strong>
                </span>
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-label="Scan stream progress"
                aria-valuenow={progress.done}
                aria-valuemin={0}
                aria-valuemax={progress.total}
              >
                <div className="progress-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          )}

          <div ref={plotBoxRef} className={`heatmap-box${scrolls ? ' heatmap-scroll' : ''}`}>
            {hasHeatmap ? (
              <Plot
                data={plotData}
                layout={plotLayout}
                config={{ ...PLOT_CONFIG, responsive: false }}
                style={{ width: `${plotWidth}px` }}
              />
            ) : (
              loading && (
                <div className="loading-msg" style={{ padding: '20px 0' }}>
                  <span className="spinner" />Encoding sequence and waiting for first site batch…
                </div>
              )
            )}
          </div>

          {hasHeatmap && (
            <div className="figure-caption">
              <span className="legend-ramp" aria-hidden="true" />
              <span>
                <strong style={{ color: COLORS.destabilizing }}>destabilizing (ΔΔG &lt; 0)</strong> ·{' '}
                <strong style={{ color: COLORS.stabilizing }}>stabilizing (ΔΔG &gt; 0)</strong> · colour clipped at ±
                {zLim.toFixed(1)} kcal/mol (98th pct.) · top: mean of 20 substitutions · dots: wild type
                {stride > 1 ? ` · every ${stride}th residue labelled` : ''}
              </span>
            </div>
          )}

          {result && (
            <div className="btn-group">
              <button className="btn btn-secondary" onClick={() => handleExport('csv')} disabled={exporting === 'csv'}>
                {exporting === 'csv' && <span className="spinner" />}
                Export CSV
              </button>
              <button className="btn btn-secondary" onClick={() => handleExport('pdb')} disabled={exporting === 'pdb'}>
                {exporting === 'pdb' && <span className="spinner" />}
                Export PDB (B-factor)
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
