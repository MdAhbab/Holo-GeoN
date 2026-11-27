import { useState, useRef, useMemo } from 'react'
// @ts-ignore - react-plotly.js types resolved via @types/react-plotly.js
import Plot from 'react-plotly.js'
import { streamEpistasis, downloadExport } from '../api'
import { setEpistasis } from '../store'
import type {
  TabProps,
  EpistasisResponse,
  EpistasisPairResult,
  EpistasisStreamStart,
} from '../types'

const UBIQUITIN =
  'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

const DEFAULT_PAIRS = [
  'V5A+L8P',
  'K6A+T7A',
  'I13A+L15V',
  'V5A+I13A',
  'L8P+T14A',
  'E18A+D21A',
].join('\n')

const MAX_PAIRS = 200

type SortKey =
  | 'pair'
  | 'sites'
  | 'ddg_total'
  | 'ddg_additive'
  | 'epistasis_dddg'
  | 'sigma'
  | 'contact_prob'
  | 'is_contact'
  | 'seq_separation'

type SortDir = 'asc' | 'desc'

function cleanSeqLength(raw: string): number {
  return raw
    .split('\n')
    .filter((line) => !line.startsWith('>'))
    .join('')
    .replace(/\s/g, '').length
}

function parsePairsList(raw: string): string[] {
  return raw
    .split(/[\n,;]+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

function formatSigned(val: number, digits = 3): string {
  return `${val > 0 ? '+' : ''}${val.toFixed(digits)}`
}

export default function EpistasisTab({ health }: TabProps) {
  const [sequence, setSequence] = useState(UBIQUITIN)
  const [pairsText, setPairsText] = useState(DEFAULT_PAIRS)
  const [cds, setCds] = useState('')
  const [showCds, setShowCds] = useState(false)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cancelledMsg, setCancelledMsg] = useState<string | null>(null)

  const [streamStart, setStreamStart] = useState<EpistasisStreamStart | null>(null)
  const [progress, setProgress] = useState<{
    done: number
    total: number
    elapsed_s: number
    eta_s: number
  } | null>(null)
  const [livePairs, setLivePairs] = useState<EpistasisPairResult[]>([])
  const [result, setResult] = useState<EpistasisResponse | null>(null)
  const [exporting, setExporting] = useState<'json' | 'csv' | null>(null)

  const [sortKey, setSortKey] = useState<SortKey>('sites')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const abortRef = useRef<AbortController | null>(null)

  const epiCap = health?.capabilities?.epistasis
  const cdsCap = health?.capabilities?.cds

  const epiUnavailable = Boolean(
    health && (!health.ready || (epiCap && !epiCap.available))
  )
  const epiReason =
    epiCap?.reason ??
    health?.error ??
    'The epistasis head was not trained in the active checkpoint.'

  const cdsUnavailable = Boolean(cdsCap && !cdsCap.available)
  const cdsReason =
    cdsCap?.reason ??
    'ViennaRNA is not installed on this server, so coding-sequence features are unavailable.'

  const parsedPairs = useMemo(() => parsePairsList(pairsText), [pairsText])
  const seqLen = useMemo(() => cleanSeqLength(sequence), [sequence])
  const pairsCountInvalid =
    parsedPairs.length === 0 || parsedPairs.length > MAX_PAIRS

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(
        key === 'pair' || key === 'sites' || key === 'seq_separation'
          ? 'asc'
          : 'desc'
      )
    }
  }

  function handleCancel() {
    abortRef.current?.abort()
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (epiUnavailable || pairsCountInvalid) return

    setError(null)
    setCancelledMsg(null)
    setResult(null)
    setLivePairs([])
    setStreamStart(null)
    setProgress({
      done: 0,
      total: parsedPairs.length,
      elapsed_s: 0,
      eta_s: 0,
    })
    setLoading(true)

    const controller = new AbortController()
    abortRef.current = controller

    const trimmedCds = !cdsUnavailable && cds.trim() ? cds.trim() : undefined

    try {
      const finalRes = await streamEpistasis(
        {
          wt_sequence: sequence.trim(),
          pairs: parsedPairs,
          ...(trimmedCds ? { cds: trimmedCds } : {}),
        },
        {
          onStart: (startData) => {
            setStreamStart(startData)
            setProgress({
              done: 0,
              total: startData.total,
              elapsed_s: 0,
              eta_s: 0,
            })
          },
          onChunk: (chunkData) => {
            setProgress({
              done: chunkData.done,
              total: chunkData.total,
              elapsed_s: chunkData.elapsed_s,
              eta_s: chunkData.eta_s,
            })
            setLivePairs((prev) => [...prev, ...chunkData.pairs])
          },
          onResult: (resData) => {
            setResult(resData)
            setLivePairs(resData.pairs)
            setEpistasis({ ...resData, wt_sequence: sequence.trim() })
          },
        },
        controller.signal
      )
      setResult(finalRes)
      setLivePairs(finalRes.pairs)
      setEpistasis({ ...finalRes, wt_sequence: sequence.trim() })
    } catch (err: unknown) {
      if (
        (err instanceof DOMException && err.name === 'AbortError') ||
        (err as { name?: string })?.name === 'AbortError'
      ) {
        setCancelledMsg('Epistasis stream cancelled.')
      } else {
        setError((err as Error)?.message ?? 'Epistasis prediction failed')
      }
    } finally {
      setLoading(false)
      abortRef.current = null
    }
  }

  async function handleExportJson() {
    if (!result) return
    setExporting('json')
    try {
      await downloadExport(
        'json',
        result,
        `holo_geon_epistasis_${result.n_pairs}pairs.json`
      )
    } catch (err: unknown) {
      setError((err as Error)?.message ?? 'JSON export failed')
    } finally {
      setExporting(null)
    }
  }

  function handleExportCsv() {
    const rowsToExport = result?.pairs ?? livePairs
    if (rowsToExport.length === 0) return
    setExporting('csv')
    try {
      const header = [
        'pair',
        'site_i',
        'site_j',
        'ddg_total',
        'ddg_additive',
        'epistasis_dddg',
        'sigma_total',
        'ci_low_total',
        'ci_high_total',
        'contact_prob',
        'is_contact',
        'seq_separation',
        'inv_hop_dist',
      ]
      const lines = [
        header.join(','),
        ...rowsToExport.map((r) =>
          [
            r.pair,
            r.sites[0],
            r.sites[1],
            r.ddg_total,
            r.ddg_additive,
            r.epistasis_dddg,
            r.sigma,
            r.ci_low,
            r.ci_high,
            r.contact_prob,
            r.is_contact,
            r.seq_separation,
            r.pair_geo.inv_hop_dist,
          ].join(',')
        ),
      ]
      const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `holo_geon_epistasis_${rowsToExport.length}pairs.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } finally {
      setExporting(null)
    }
  }

  const activePairs = result?.pairs ?? livePairs

  const sortedPairs = useMemo(() => {
    const copy = activePairs.slice()
    const factor = sortDir === 'asc' ? 1 : -1
    copy.sort((a, b) => {
      switch (sortKey) {
        case 'pair':
          return factor * a.pair.localeCompare(b.pair)
        case 'sites':
          return factor * (a.sites[0] - b.sites[0] || a.sites[1] - b.sites[1])
        case 'ddg_total':
          return factor * (a.ddg_total - b.ddg_total)
        case 'ddg_additive':
          return factor * (a.ddg_additive - b.ddg_additive)
        case 'epistasis_dddg':
          return factor * (Math.abs(a.epistasis_dddg) - Math.abs(b.epistasis_dddg))
        case 'sigma':
          return factor * (a.sigma - b.sigma)
        case 'contact_prob':
          return factor * (a.contact_prob - b.contact_prob)
        case 'is_contact':
          return factor * (Number(a.is_contact) - Number(b.is_contact))
        case 'seq_separation':
          return factor * (a.seq_separation - b.seq_separation)
        default:
          return 0
      }
    })
    return copy
  }, [activePairs, sortKey, sortDir])

  // Summary statistics over activePairs
  const summaryStats = useMemo(() => {
    if (activePairs.length === 0) return null
    const contactPairs = activePairs.filter((p) => p.is_contact)
    const distalPairs = activePairs.filter((p) => !p.is_contact)
    const strongest = activePairs.reduce((best, cur) =>
      Math.abs(cur.epistasis_dddg) > Math.abs(best.epistasis_dddg) ? cur : best
    )
    const meanAbsContact =
      contactPairs.length > 0
        ? contactPairs.reduce((acc, p) => acc + Math.abs(p.epistasis_dddg), 0) /
          contactPairs.length
        : null
    const meanAbsDistal =
      distalPairs.length > 0
        ? distalPairs.reduce((acc, p) => acc + Math.abs(p.epistasis_dddg), 0) /
          distalPairs.length
        : null
    const meanContactProb =
      activePairs.reduce((acc, p) => acc + p.contact_prob, 0) /
      activePairs.length

    return {
      nTotal: activePairs.length,
      nContacts: contactPairs.length,
      meanContactProb,
      strongest,
      meanAbsContact,
      meanAbsDistal,
    }
  }, [activePairs])

  // Plotly chart 1: Predicted Total ΔΔG vs Additive ΔΔG
  const totalVsAdditivePlot = useMemo(() => {
    if (activePairs.length === 0) return { data: [], layout: {} }
    const allVals = activePairs.flatMap((p) => [
      p.ddg_additive,
      p.ddg_total,
      p.ci_low,
      p.ci_high,
    ])
    const minVal = Math.min(...allVals, -1)
    const maxVal = Math.max(...allVals, 1)
    const pad = (maxVal - minVal) * 0.1 || 0.5
    const lo = minVal - pad
    const hi = maxVal + pad

    const data = [
      {
        type: 'scatter',
        mode: 'lines',
        x: [lo, hi],
        y: [lo, hi],
        line: { color: '#94a3b8', width: 1.5, dash: 'dash' },
        name: 'Additive null (ΔΔG_total = ΔΔG_add)',
        hoverinfo: 'skip',
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: activePairs.map((p) => p.ddg_additive),
        y: activePairs.map((p) => p.ddg_total),
        text: activePairs.map(
          (p) =>
            `<b>${p.pair}</b> (sites ${p.sites[0]}, ${p.sites[1]})` +
            `<br>ΔΔG_total: ${formatSigned(p.ddg_total, 4)} kcal/mol` +
            `<br>ΔΔG_additive: ${formatSigned(p.ddg_additive, 4)} kcal/mol` +
            `<br>ΔΔΔG (epistasis): ${formatSigned(p.epistasis_dddg, 4)} kcal/mol` +
            `<br>Contact prob: ${p.contact_prob.toFixed(3)} (${p.is_contact ? 'contact' : 'distal'})`
        ),
        hovertemplate: '%{text}<extra></extra>',
        error_y: {
          type: 'data',
          symmetric: false,
          array: activePairs.map((p) => p.ci_high - p.ddg_total),
          arrayminus: activePairs.map((p) => p.ddg_total - p.ci_low),
          color: 'rgba(100,116,139,0.38)',
          thickness: 1.5,
          width: 4,
        },
        marker: {
          size: 11,
          color: activePairs.map((p) => p.epistasis_dddg),
          colorscale: 'RdBu',
          cmid: 0,
          reversescale: true,
          line: { width: 1, color: '#0f172a' },
          colorbar: {
            title: 'ΔΔΔG (kcal/mol)',
            titleside: 'right',
            thickness: 14,
            len: 0.85,
          },
        },
        name: 'Double mutants',
      },
    ]

    const layout = {
      height: 320,
      margin: { l: 60, r: 85, t: 28, b: 55 },
      xaxis: {
        title: 'Additive ΔΔG_a + ΔΔG_b (kcal/mol)',
        range: [lo, hi],
        gridcolor: '#f1f5f9',
        zeroline: true,
        zerolinecolor: '#cbd5e1',
      },
      yaxis: {
        title: 'Predicted Total ΔΔG_ab (kcal/mol)',
        range: [lo, hi],
        gridcolor: '#f1f5f9',
        zeroline: true,
        zerolinecolor: '#cbd5e1',
      },
      showlegend: true,
      legend: { orientation: 'h', y: 1.13, x: 0 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: 'Inter, system-ui, sans-serif', size: 11 },
    }
    return { data, layout }
  }, [activePairs])

  // Plotly chart 2: Epistasis ΔΔΔG vs Predicted Contact Probability
  const epistasisVsContactPlot = useMemo(() => {
    if (activePairs.length === 0) return { data: [], layout: {} }
    const contacts = activePairs.filter((p) => p.is_contact)
    const distals = activePairs.filter((p) => !p.is_contact)

    const makeTrace = (
      subset: EpistasisPairResult[],
      label: string,
      color: string,
      symbol: string
    ) => ({
      type: 'scatter',
      mode: 'markers',
      name: label,
      x: subset.map((p) => p.contact_prob),
      y: subset.map((p) => p.epistasis_dddg),
      text: subset.map(
        (p) =>
          `<b>${p.pair}</b> (|j − i| = ${p.seq_separation})` +
          `<br>ΔΔΔG: ${formatSigned(p.epistasis_dddg, 4)} kcal/mol` +
          `<br>Contact prob: ${p.contact_prob.toFixed(4)}` +
          `<br>1/hop_dist: ${p.pair_geo.inv_hop_dist.toFixed(3)}`
      ),
      hovertemplate: '%{text}<extra></extra>',
      marker: {
        size: 11,
        color,
        symbol,
        line: { width: 1, color: '#0f172a' },
      },
    })

    const data = [
      makeTrace(contacts, 'Direct contact (E_geo)', '#008080', 'circle'),
      makeTrace(distals, 'Non-contact / distal', '#64748b', 'diamond'),
    ]

    const layout = {
      height: 320,
      margin: { l: 60, r: 28, t: 28, b: 55 },
      xaxis: {
        title: 'Predicted Contact Probability P(contact)',
        range: [-0.04, 1.04],
        gridcolor: '#f1f5f9',
      },
      yaxis: {
        title: 'Epistasis Coupling ΔΔΔG (kcal/mol)',
        gridcolor: '#f1f5f9',
        zeroline: true,
        zerolinecolor: '#94a3b8',
        zerolinewidth: 1.5,
      },
      showlegend: true,
      legend: { orientation: 'h', y: 1.13, x: 0 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: 'Inter, system-ui, sans-serif', size: 11 },
    }
    return { data, layout }
  }, [activePairs])

  const progressPct =
    progress && progress.total > 0
      ? Math.min(100, Math.round((progress.done / progress.total) * 100))
      : 0

  const headInfo = result?.epistasis_head ?? streamStart?.epistasis_head
  const cdsInfo = result?.cds ?? streamStart?.cds

  return (
    <div>
      {epiUnavailable && (
        <div className="capability-notice capability-refused" role="alert">
          <div className="capability-notice-title">
            ⚠ Double-Mutant Epistasis Unavailable (HTTP 503 Refusal)
          </div>
          <p className="capability-notice-body">{epiReason}</p>
          <p className="capability-notice-sub">
            Holo-GeoN never substitutes random-weight or additive-only outputs when the
            contact-gated Potts epistasis head has not been trained with{' '}
            <code>python -m scripts.train_epistasis</code>.
          </p>
        </div>
      )}

      <div className="card">
        <div className="card-header-row">
          <h2 className="card-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>
            Double-Mutant Epistasis — Contact-Gated Potts Coupling (ΔΔΔG)
          </h2>
          <span className="symmetry-badge" title="Default mode symmetry statement">
            Structure-Free · Swap-Equivariant · Not SE(3)-Equivariant
          </span>
        </div>
        <p className="card-lead">
          Predicts non-additive pairwise thermodynamic coupling{' '}
          <code>ΔΔG_total = ΔΔG_additive + ΔΔΔG</code> using the contact-gated Potts head over the
          predicted ESM-2 geometric contact graph <code>E_geo</code> (reads no 3D coordinates).
          Pairs are canonicalized in ascending site order (<code>i &lt; j</code>) and are strictly
          symmetric under site swap.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-field">
            <div className="label-row">
              <label htmlFor="epi-seq">Wild-type Protein Sequence</label>
              <span className="field-meta">{seqLen} residues</span>
            </div>
            <textarea
              id="epi-seq"
              rows={3}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Paste single-letter amino-acid sequence (FASTA header allowed)..."
            />
          </div>

          <div className="form-field">
            <div className="label-row">
              <label htmlFor="epi-pairs">
                Double-Mutant Pairs (one per line or comma-separated, e.g. <code>V5A+L8P</code>)
              </label>
              <span
                className={`field-meta ${
                  parsedPairs.length > MAX_PAIRS ? 'text-danger' : ''
                }`}
              >
                {parsedPairs.length} / {MAX_PAIRS} pairs
              </span>
            </div>
            <textarea
              id="epi-pairs"
              rows={4}
              value={pairsText}
              onChange={(e) => setPairsText(e.target.value)}
              placeholder="V5A+L8P&#10;K6A+T7A&#10;I13A+L15V"
              aria-describedby="epi-pairs-hint"
            />
            <div id="epi-pairs-hint" className="field-hint-row">
              <span>
                Each pair must specify two distinct 1-based sites (e.g. <code>V5A+L8P</code> or{' '}
                <code>L8P+V5A</code> — both yield the identical canonical prediction).
              </span>
              <div className="inline-actions">
                <button
                  type="button"
                  className="btn-link"
                  onClick={() => {
                    setSequence(UBIQUITIN)
                    setPairsText(DEFAULT_PAIRS)
                  }}
                >
                  Load Ubiquitin Preset (6 pairs)
                </button>
                <button
                  type="button"
                  className="btn-link"
                  onClick={() => setPairsText('')}
                >
                  Clear pairs
                </button>
              </div>
            </div>
          </div>

          {/* Optional Coding Sequence (CDS) section gated on capabilities.cds */}
          <div className="cds-box">
            <div className="cds-box-header">
              <button
                type="button"
                className="btn-link cds-toggle-btn"
                onClick={() => setShowCds((v) => !v)}
                aria-expanded={showCds || cdsUnavailable}
              >
                {showCds || cdsUnavailable ? '▾' : '▸'} Optional Coding Sequence (CDS — mRNA folding &amp; codon usage channels)
              </button>
              <span className={`status-pill ${cdsUnavailable ? 'status-off' : 'status-on'}`}>
                {cdsUnavailable ? 'ViennaRNA Unavailable' : 'ViennaRNA Ready'}
              </span>
            </div>

            {cdsUnavailable && (
              <div className="capability-notice capability-disabled" role="note">
                <strong>CDS input disabled:</strong> {cdsReason}
              </div>
            )}

            {(showCds || cdsUnavailable) && (
              <div style={{ marginTop: 8 }}>
                <label htmlFor="epi-cds">
                  Coding Sequence (DNA/RNA reading frame encoding the exact wild-type protein)
                </label>
                <textarea
                  id="epi-cds"
                  rows={2}
                  value={cds}
                  onChange={(e) => setCds(e.target.value)}
                  disabled={cdsUnavailable}
                  placeholder={
                    cdsUnavailable
                      ? cdsReason
                      : 'Optional: paste frame-verified DNA/RNA CDS (3N or 3N+3 nucleotides with terminal stop codon)...'
                  }
                />
              </div>
            )}
          </div>

          <div className="btn-group" style={{ alignItems: 'center' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={
                loading ||
                epiUnavailable ||
                !sequence.trim() ||
                pairsCountInvalid
              }
              title={epiUnavailable ? epiReason : undefined}
            >
              {loading && <span className="spinner" />}
              {loading
                ? 'Streaming Epistasis…'
                : `Predict Epistasis (${parsedPairs.length} pair${
                    parsedPairs.length === 1 ? '' : 's'
                  })`}
            </button>

            {loading && (
              <button
                type="button"
                className="btn btn-danger"
                onClick={handleCancel}
              >
                Cancel Stream
              </button>
            )}
          </div>
        </form>

        {loading && progress && (
          <div className="progress-wrap" aria-live="off">
            <div className="progress-header">
              <span>
                Scoring double mutants: <strong>{progress.done}</strong> /{' '}
                <strong>{progress.total}</strong> pairs ({progressPct}%)
              </span>
              <span className="progress-meta">
                Elapsed: <strong>{progress.elapsed_s.toFixed(2)}s</strong> · ETA:{' '}
                <strong>{progress.eta_s.toFixed(2)}s</strong>
              </span>
            </div>
            <div
              className="progress-track"
              role="progressbar"
              aria-label="Epistasis stream progress"
              aria-valuenow={progress.done}
              aria-valuemin={0}
              aria-valuemax={progress.total}
            >
              <div
                className="progress-fill"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}

        {cancelledMsg && <div className="capability-notice">{cancelledMsg}</div>}
        {error && (
          <div className="error-msg" role="alert">
            {error}
          </div>
        )}
      </div>

      {activePairs.length > 0 && (
        <>
          {/* Provenance & Honesty Notices */}
          <div className="card">
            <div className="card-title">Model Symmetry, Head Provenance &amp; Uncertainty Scope</div>

            <div className="provenance-bar">
              <div className="provenance-item">
                <span className="provenance-label">Mode &amp; Coordinates</span>
                <span className="provenance-val">
                  <code>{result?.mode ?? 'structure_free'}</code> ·{' '}
                  {result?.uses_coordinates ? 'Uses 3D coordinates' : 'Reads no coordinates'}
                </span>
              </div>
              <div className="provenance-item">
                <span className="provenance-label">Epistasis Head</span>
                <span className="provenance-val">
                  {headInfo?.contact_gated ? 'Contact-gated Potts' : 'Ungated Potts'} (
                  <code>{headInfo?.source ?? 'trained'}</code>)
                </span>
              </div>
              <div className="provenance-item">
                <span className="provenance-label">Contact Gate Input</span>
                <span className="provenance-val">
                  {headInfo?.gate_input ?? 'predicted ESM-2 contact graph E_geo (no coordinates)'}
                </span>
              </div>
              <div className="provenance-item">
                <span className="provenance-label">Coding Sequence (CDS)</span>
                <span className="provenance-val">
                  {cdsInfo?.supplied
                    ? `Supplied & frame-verified (${cdsInfo.n_codons} codons${
                        cdsInfo.stop_codon_stripped ? ', stop stripped' : ''
                      })`
                    : 'Not supplied (in-distribution CDS-dropout path)'}
                </span>
              </div>
              {result?.timing && (
                <div className="provenance-item">
                  <span className="provenance-label">Throughput</span>
                  <span className="provenance-val">
                    Encode {result.timing.encode_ms.toFixed(1)} ms · Predict{' '}
                    {result.timing.predict_ms.toFixed(1)} ms ({result.timing.pairs_per_sec.toFixed(1)}{' '}
                    pairs/s)
                  </span>
                </div>
              )}
            </div>

            <div className="symmetry-statement">
              <strong>Symmetry:</strong>{' '}
              {result?.symmetry ??
                'structure-free predicted-contact graph; reads no coordinates. Its only symmetry is wild-type/mutant swap equivariance: it is NOT SE(3)- or E(3)-equivariant.'}
            </div>

            <div className="uncertainty-notice" role="note">
              <strong>Uncalibrated Total-ΔΔG Interval Notice:</strong>{' '}
              {result?.uncertainty_note ??
                'Calibration was measured on single mutants only, and the epistasis term itself has no uncertainty estimate: the interval is for ddg_total and is not a calibrated claim about the epistasis term.'}{' '}
              {result?.ci_source && (
                <span className="result-sub" style={{ display: 'inline' }}>
                  (Source: <code>{result.ci_source}</code>)
                </span>
              )}
            </div>

            {/* KPI Summary Cards */}
            {summaryStats && (
              <div className="kpi-grid">
                <div className="kpi-card">
                  <div className="kpi-label">Pairs Scored</div>
                  <div className="kpi-value">{summaryStats.nTotal}</div>
                  <div className="kpi-sub">
                    Sequence length L = {result?.length ?? streamStart?.length ?? seqLen} aa
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-label">Predicted Direct Contacts</div>
                  <div className="kpi-value">
                    {summaryStats.nContacts} / {summaryStats.nTotal}
                  </div>
                  <div className="kpi-sub">
                    Mean P(contact) = {summaryStats.meanContactProb.toFixed(3)}
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-label">Strongest Coupling |ΔΔΔG|</div>
                  <div className="kpi-value">
                    {formatSigned(summaryStats.strongest.epistasis_dddg, 3)}{' '}
                    <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>kcal/mol</span>
                  </div>
                  <div className="kpi-sub">
                    Pair <code>{summaryStats.strongest.pair}</code> (|j − i| ={' '}
                    {summaryStats.strongest.seq_separation})
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-label">Mean |ΔΔΔG| (Contact vs Distal)</div>
                  <div className="kpi-value" style={{ fontSize: '1.25rem' }}>
                    {summaryStats.meanAbsContact !== null
                      ? `${summaryStats.meanAbsContact.toFixed(3)}`
                      : '—'}{' '}
                    vs{' '}
                    {summaryStats.meanAbsDistal !== null
                      ? `${summaryStats.meanAbsDistal.toFixed(3)}`
                      : '—'}
                  </div>
                  <div className="kpi-sub">kcal/mol (contact-gated vs non-contact)</div>
                </div>
              </div>
            )}
          </div>

          {/* Interactive Plotly Charts */}
          <div className="compare-grid">
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-title">
                Predicted Total ΔΔG vs. Additive Single-Mutant Sum
              </div>
              <div className="plot-wrap">
                <Plot
                  data={totalVsAdditivePlot.data}
                  layout={totalVsAdditivePlot.layout}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                  useResizeHandler
                />
              </div>
            </div>

            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-title">
                Epistasis Coupling ΔΔΔG vs. Predicted Contact Probability
              </div>
              <div className="plot-wrap">
                <Plot
                  data={epistasisVsContactPlot.data}
                  layout={epistasisVsContactPlot.layout}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                  useResizeHandler
                />
              </div>
            </div>
          </div>

          {/* Sortable Table of Double-Mutant Pairs */}
          <div className="card section-gap">
            <div className="card-header-row">
              <h3 className="card-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>
                Double-Mutant Epistasis Table ({sortedPairs.length} pairs)
              </h3>
              <div className="btn-group" style={{ marginTop: 0 }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleExportCsv}
                  disabled={exporting === 'csv'}
                >
                  {exporting === 'csv' && <span className="spinner" />}
                  Export CSV
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleExportJson}
                  disabled={!result || exporting === 'json'}
                >
                  {exporting === 'json' && <span className="spinner" />}
                  Export JSON
                </button>
              </div>
            </div>

            <div className="table-scroll">
              <table className="data-table epistasis-table">
                <caption className="visually-hidden">
                  Double-mutant epistasis predictions sorted by {sortKey} ({sortDir})
                </caption>
                <thead>
                  <tr>
                    {(
                      [
                        ['pair', 'Pair (Canonical)'],
                        ['sites', 'Sites (i, j)'],
                        ['ddg_total', 'ΔΔG_total'],
                        ['ddg_additive', 'ΔΔG_additive'],
                        ['epistasis_dddg', 'ΔΔΔG (Epistasis)'],
                        ['sigma', 'σ / 95% CI (Total)'],
                        ['contact_prob', 'P(contact)'],
                        ['is_contact', 'E_geo Contact'],
                        ['seq_separation', '|j − i|'],
                      ] as [SortKey, string][]
                    ).map(([key, label]) => {
                      const active = sortKey === key
                      const ariaSort = active
                        ? sortDir === 'asc'
                          ? 'ascending'
                          : 'descending'
                        : 'none'
                      return (
                        <th key={key} scope="col" aria-sort={ariaSort}>
                          <button
                            type="button"
                            className={`sort-btn${active ? ' active' : ''}`}
                            onClick={() => handleSort(key)}
                          >
                            <span>{label}</span>
                            <span className="sort-arrow" aria-hidden="true">
                              {active ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                            </span>
                          </button>
                        </th>
                      )
                    })}
                  </tr>
                </thead>
                <tbody>
                  {sortedPairs.map((row, idx) => {
                    const epiMag = Math.abs(row.epistasis_dddg)
                    const deltaClass =
                      epiMag < 0.05
                        ? 'delta-neutral'
                        : row.epistasis_dddg > 0
                          ? 'delta-positive'
                          : 'delta-negative'
                    return (
                      <tr key={`${row.pair}-${idx}`}>
                        <th scope="row" className="mono-cell">
                          {row.pair}
                        </th>
                        <td className="mono-cell">
                          {row.sites[0]}, {row.sites[1]}
                        </td>
                        <td className="num-cell">
                          <strong>{formatSigned(row.ddg_total, 4)}</strong>
                        </td>
                        <td className="num-cell">
                          {formatSigned(row.ddg_additive, 4)}
                        </td>
                        <td className="num-cell">
                          <span className={`delta-chip ${deltaClass}`}>
                            {formatSigned(row.epistasis_dddg, 4)}
                          </span>
                        </td>
                        <td className="num-cell ci-cell">
                          <span>±{row.sigma.toFixed(3)}</span>
                          <span className="ci-bounds">
                            [{row.ci_low.toFixed(2)}, {row.ci_high.toFixed(2)}]
                          </span>
                        </td>
                        <td className="num-cell">
                          <div className="prob-cell">
                            <div className="prob-bar" aria-hidden="true">
                              <div
                                className="prob-fill"
                                style={{
                                  width: `${Math.min(100, Math.round(row.contact_prob * 100))}%`,
                                }}
                              />
                            </div>
                            <span>{row.contact_prob.toFixed(3)}</span>
                          </div>
                        </td>
                        <td>
                          <span
                            className={`contact-badge ${
                              row.is_contact ? 'contact-yes' : 'contact-no'
                            }`}
                            title={`1/hop_dist = ${row.pair_geo.inv_hop_dist.toFixed(3)}, log_seq_sep = ${row.pair_geo.log_seq_sep.toFixed(3)}`}
                          >
                            {row.is_contact ? 'Contact' : 'Distal'}
                          </span>
                        </td>
                        <td className="num-cell">{row.seq_separation}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
