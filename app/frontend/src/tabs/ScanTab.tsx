import { useState, useRef, useEffect } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import { streamScan, downloadExport } from '../api'
import { setScan } from '../store'
import type {
  PredictionMode,
  ScanResponse,
  ScanStreamStart,
  TabProps,
} from '../types'

const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

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
  const [exporting, setExporting] = useState<string | null>(null)

  const abortCtrlRef = useRef<AbortController | null>(null)

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
            setProgress({
              done: chunk.done,
              total: chunk.total,
              elapsed_s: chunk.elapsed_s,
              eta_s: chunk.eta_s,
            })
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
              prev ? { ...prev, done: prev.total } : { done: res.positions.length, total: res.positions.length, elapsed_s: res.timing.scan_ms / 1000, eta_s: 0 }
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
      const payload = format === 'pdb'
        ? { ...result, sequence: result.sequence ?? cleanSeqText(sequence) }
        : result
      await downloadExport(format, payload, `scan_${start}-${end}.${format}`)
    } catch (err: unknown) {
      setError((err as Error)?.message ?? 'Export failed')
    } finally {
      setExporting(null)
    }
  }

  // Derive live or final heatmap matrix
  let xLabels: string[] = []
  let yLabels: string[] = []
  let zMatrix: (number | null)[][] = []
  let nCols = 0

  if (result) {
    xLabels = result.positions.map((pos, i) => `${result.wt_residues[i] ?? '?'}${pos}`)
    yLabels = result.aa_order
    zMatrix = result.matrix
    nCols = result.positions.length
  } else if (streamStart && liveColumns.some((c) => c !== null)) {
    const first = streamStart.positions[0]
    const total = streamStart.total
    const positions = Array.from({ length: total }, (_, i) => first + i)
    xLabels = positions.map((pos) => `${submittedSeq[pos - 1] ?? '?'}${pos}`)
    yLabels = streamStart.aa_order
    zMatrix = yLabels.map((_, rIdx) =>
      positions.map((__, cIdx) => {
        const col = liveColumns[cIdx]
        return col ? col[rIdx] : null
      })
    )
    nCols = positions.length
  }

  const hasHeatmap = nCols > 0 && yLabels.length > 0

  const plotData = hasHeatmap
    ? [
        {
          type: 'heatmap',
          x: xLabels,
          y: yLabels,
          z: zMatrix,
          colorscale: 'RdBu',
          zmid: 0,
          colorbar: {
            title: 'ΔΔG (kcal/mol)',
            titleside: 'right',
            len: 0.9,
          },
          hovertemplate:
            '<b>%{y}%{x}</b><br>ΔΔG: %{z:.3f} kcal/mol<extra></extra>',
          reversescale: true,
        },
      ]
    : []

  const nRows = 20
  const plotHeight = Math.max(300, nRows * 18 + 80)
  const plotWidth = Math.max(400, nCols * 24 + 120)

  const plotLayout = {
    height: plotHeight,
    margin: { l: 45, r: 110, t: 30, b: 90 },
    xaxis: {
      title: 'Position',
      tickangle: -45,
      automargin: true,
      gridcolor: '#f1f5f9',
    },
    yaxis: {
      title: 'Substituted AA',
      automargin: true,
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', size: 11 },
  }

  const activeMeta = result ?? streamStart
  const pct = progress ? Math.min(100, Math.round((progress.done / Math.max(progress.total, 1)) * 100)) : 0

  return (
    <div>
      <div className="card">
        <div className="card-title">Stability Landscape — Saturation Scan</div>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 14 }}>
            <label htmlFor="scan-seq">Protein Sequence</label>
            <textarea
              id="scan-seq"
              rows={3}
              value={sequence}
              onChange={(e) => onSeqChange(e.target.value)}
              placeholder="Paste single-letter amino acid sequence..."
            />
          </div>

          <div className="input-row" style={{ marginBottom: 14 }}>
            <div className="field-sm">
              <label htmlFor="scan-start">Start (1-based)</label>
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
              <label htmlFor="scan-end">End (inclusive)</label>
              <input
                id="scan-end"
                type="number"
                min={start}
                max={Math.max(start, cleanLen)}
                value={end}
                onChange={(e) => setEnd(Number(e.target.value))}
              />
            </div>
            <div className="field" style={{ flex: '2 1 260px' }}>
              <label>Inference Mode</label>
              <div className="mode-toggle" role="group" aria-label="Scan inference mode">
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
                id="scan-pdb"
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
              <label htmlFor="scan-cds">Optional Coding Sequence (CDS — DNA/RNA, FASTA ok, one stop codon ok)</label>
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
              id="scan-cds"
              rows={2}
              value={cds}
              disabled={!cdsAvailable}
              onChange={(e) => setCds(e.target.value)}
              placeholder={
                cdsAvailable
                  ? 'Optional: paste coding sequence encoding this exact protein...'
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

          <div className="btn-group" style={{ marginTop: 0 }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !cleanLen || (needsStructure && !structurePdb.trim())}
            >
              {loading && <span className="spinner" />}
              {loading ? 'Streaming Scan…' : 'Run Scan'}
            </button>
            {loading && (
              <button
                type="button"
                className="btn btn-danger"
                onClick={handleCancel}
              >
                Cancel Scan
              </button>
            )}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6 }}>
            Scanning {Math.max(0, end - start + 1)} positions × 20 amino acids over SSE stream.
            ESM-2 runs once per protein; columns fill in live by batch.
          </div>
        </form>

        {error && <div className="error-msg">{error}</div>}
        {cancelled && (
          <div className="warn-banner" style={{ borderRadius: 'var(--radius)', marginTop: 12 }}>
            <span>⚠</span>
            <span>
              Scan cancelled by user
              {progress ? ` (${progress.done} / ${progress.total} sites received)` : ''}.
            </span>
          </div>
        )}
      </div>

      {(loading || progress || activeMeta) && (
        <div className="card">
          <div className="card-header-row">
            <div className="card-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>
              {result
                ? 'ΔΔG Heatmap (ΔΔG > 0 = stabilizing ▲, < 0 = destabilizing ▼)'
                : 'Live Scan Progress & Incremental Heatmap'}
            </div>
            {activeMeta && (
              <span className="symmetry-badge">
                {activeMeta.mode === 'invariant_3d' ? 'SE(3)-Invariant 3D' : 'Structure-Free · Swap-Equivariant Only'}
              </span>
            )}
          </div>

          {activeMeta && (
            <div className="symmetry-statement">
              <strong>Coordinates:</strong>{' '}
              {activeMeta.uses_coordinates ? 'Uses aligned 3D backbone coordinates' : 'Reads no coordinates'} ·{' '}
              <strong>Symmetry:</strong> {activeMeta.symmetry}
            </div>
          )}

          {activeMeta && (
            <div className="stat-chips" style={{ marginBottom: 10 }}>
              <div className="stat-chip">
                CDS:{' '}
                <strong>
                  {activeMeta.cds?.supplied
                    ? `verified (${activeMeta.cds.n_codons ?? '?'} codons${activeMeta.cds.stop_codon_stripped ? ', stop stripped' : ''})`
                    : 'not supplied (CDS-dropout path)'}
                </strong>
              </div>
              {activeMeta.structure && (
                <>
                  <div className="stat-chip">
                    Mean pLDDT: <strong>{activeMeta.structure.mean_plddt.toFixed(2)}</strong>
                  </div>
                  <div className="stat-chip">
                    Low-conf (&lt;70): <strong>{activeMeta.structure.low_confidence_positions.length} sites</strong>
                  </div>
                </>
              )}
              {result?.timing && (
                <>
                  <div className="stat-chip">
                    Encode: <strong>{result.timing.encode_ms.toFixed(1)} ms</strong>
                  </div>
                  <div className="stat-chip">
                    Scan: <strong>{result.timing.scan_ms.toFixed(1)} ms</strong>
                  </div>
                  <div className="stat-chip">
                    Mutations: <strong>{result.timing.total_mutations.toLocaleString()}</strong>
                  </div>
                  <div className="stat-chip">
                    Throughput: <strong>{result.timing.muts_per_sec.toFixed(1)} mut/s</strong>
                  </div>
                </>
              )}
            </div>
          )}

          {progress && (
            <div className="progress-wrap" style={{ marginBottom: 14 }} aria-live="off">
              <div className="progress-header">
                <span>
                  Scanning residues: <strong>{progress.done}</strong> / <strong>{progress.total}</strong> sites ({pct}%)
                </span>
                <span className="progress-meta">
                  Elapsed: <strong>{progress.elapsed_s.toFixed(2)}s</strong> · ETA:{' '}
                  <strong>{progress.eta_s.toFixed(2)}s</strong>
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

          {hasHeatmap ? (
            <div style={{ overflowX: 'auto' }} className="plot-wrap">
              <Plot
                data={plotData}
                layout={{ ...plotLayout, width: plotWidth }}
                config={{ displayModeBar: true, responsive: false }}
                style={{ minWidth: `${plotWidth}px` }}
              />
            </div>
          ) : (
            loading && (
              <div className="loading-msg" style={{ padding: '20px 0' }}>
                <span className="spinner" />Encoding sequence and waiting for first site batch…
              </div>
            )
          )}

          {result && (
            <div className="btn-group">
              <button
                className="btn btn-secondary"
                onClick={() => handleExport('csv')}
                disabled={exporting === 'csv'}
              >
                {exporting === 'csv' && <span className="spinner" />}
                Export CSV
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => handleExport('pdb')}
                disabled={exporting === 'pdb'}
              >
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
