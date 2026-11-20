import { useState, useEffect } from 'react'
import { downloadExport } from '../api'
import { getEpistasis, getStore, subscribe } from '../store'
import type { EpistasisResponse, TabProps } from '../types'

type Format = 'csv' | 'json' | 'pdb'
type CsvSource = 'scan' | 'epistasis' | 'ddg' | 'idr'
type JsonScope = 'all' | 'epistasis'

function csvEscape(val: string | number | boolean | null | undefined): string {
  const s = val == null ? '' : String(val)
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return `"${s.replace(/"/g, '""')}"`
  }
  return s
}

function downloadEpistasisCsv(epistasis: EpistasisResponse, filename = 'holo_geon_epistasis_pairs.csv') {
  const headers = [
    'pair',
    'site_1',
    'site_2',
    'ddg_total',
    'ddg_additive',
    'epistasis_dddg',
    'sigma',
    'ci_low',
    'ci_high',
    'contact_prob',
    'seq_separation',
    'is_contact',
    'direct_contact_prob',
    'log_seq_sep',
    'inv_hop_dist',
    'is_direct_contact',
  ]
  const rows = epistasis.pairs.map((p) =>
    [
      p.pair,
      p.sites[0],
      p.sites[1],
      p.ddg_total,
      p.ddg_additive,
      p.epistasis_dddg,
      p.sigma,
      p.ci_low,
      p.ci_high,
      p.contact_prob,
      p.seq_separation,
      p.is_contact,
      p.pair_geo?.direct_contact_prob ?? '',
      p.pair_geo?.log_seq_sep ?? '',
      p.pair_geo?.inv_hop_dist ?? '',
      p.pair_geo?.is_direct_contact ?? '',
    ]
      .map(csvEscape)
      .join(',')
  )
  const content = [headers.join(','), ...rows].join('\n') + '\n'
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export default function ExportTab({ health: _health = null }: Partial<TabProps>) {
  const [, forceUpdate] = useState(0)
  const [busy, setBusy] = useState<Format | null>(null)
  const [csvSource, setCsvSource] = useState<CsvSource | 'auto'>('auto')
  const [jsonScope, setJsonScope] = useState<JsonScope>('all')
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Subscribe to store changes so the component re-renders when results arrive
  useEffect(() => {
    const unsub = subscribe(() => forceUpdate((n) => n + 1))
    return unsub
  }, [])

  const store = getStore()
  const epistasis = getEpistasis()
  const hasDdg = !!store.ddg
  const hasScan = !!store.scan
  const hasEpistasis = !!epistasis
  const hasIdr = !!store.idr
  const hasCompare = !!store.compare

  const availableCsvSources: Array<{ value: CsvSource; label: string }> = [
    ...(hasScan ? [{ value: 'scan' as const, label: `Scan matrix (${store.scan!.positions.length} pos)` }] : []),
    ...(hasEpistasis ? [{ value: 'epistasis' as const, label: `Epistasis pairs (${epistasis!.pairs.length})` }] : []),
    ...(hasDdg ? [{ value: 'ddg' as const, label: `ΔΔG (${store.ddg!.mutation})` }] : []),
    ...(hasIdr ? [{ value: 'idr' as const, label: `IDR (${store.idr!.length} aa)` }] : []),
  ]

  const resolvedCsvSource: CsvSource | null =
    csvSource !== 'auto' && availableCsvSources.some((s) => s.value === csvSource)
      ? csvSource
      : availableCsvSources[0]?.value ?? null

  const resolvedJsonScope: JsonScope =
    jsonScope === 'epistasis' && hasEpistasis ? 'epistasis' : 'all'

  async function doExport(format: Format) {
    setBusy(format)
    setError(null)
    setSuccess(null)
    try {
      if (format === 'csv') {
        if (!resolvedCsvSource) return
        if (resolvedCsvSource === 'epistasis' && epistasis) {
          const filename = 'holo_geon_epistasis_pairs.csv'
          downloadEpistasisCsv(epistasis, filename)
          setSuccess(`Downloaded ${filename}`)
          return
        }
        const data =
          resolvedCsvSource === 'scan'
            ? store.scan
            : resolvedCsvSource === 'ddg'
              ? store.ddg
              : store.idr
        const filename = `holo_geon_${resolvedCsvSource}.csv`
        await downloadExport('csv', data, filename)
        setSuccess(`Downloaded ${filename}`)
      } else if (format === 'json') {
        const data =
          resolvedJsonScope === 'epistasis' && epistasis
            ? epistasis
            : {
                ddg: store.ddg,
                scan: store.scan,
                epistasis,
                idr: store.idr,
                compare: store.compare,
              }
        const filename =
          resolvedJsonScope === 'epistasis'
            ? 'holo_geon_epistasis.json'
            : 'holo_geon_export.json'
        await downloadExport('json', data, filename)
        setSuccess(`Downloaded ${filename}`)
      } else {
        // pdb — needs scan data with sequence
        const filename = 'holo_geon_export.pdb'
        await downloadExport('pdb', store.scan, filename)
        setSuccess(`Downloaded ${filename}`)
      }
    } catch {
      setError('Export failed. Check that the backend is running.')
    } finally {
      setBusy(null)
    }
  }

  const anyResult = hasDdg || hasScan || hasEpistasis || hasIdr || hasCompare
  const hasCsvData = availableCsvSources.length > 0

  return (
    <div>
      <div className="card">
        <div className="card-title">Export Results</div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 16, lineHeight: 1.7 }}>
          Export your computed results in various formats. Run a prediction in one of the other tabs first,
          or hydrate a saved run from History, then download here.
        </p>

        {!anyResult && (
          <div style={{
            background: 'var(--teal-light)',
            border: '1px solid var(--teal-mid)',
            borderRadius: 'var(--radius)',
            padding: '14px 18px',
            color: 'var(--teal-dark)',
            fontSize: '0.88rem',
            marginBottom: 16,
          }}>
            No results yet. Run a prediction in the ΔΔG, Scan, Epistasis, IDR, or Compare tab first.
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}
        {success && (
          <div style={{
            background: '#f0fdf4',
            border: '1px solid #bbf7d0',
            borderRadius: 'var(--radius)',
            padding: '10px 16px',
            color: 'var(--green)',
            fontSize: '0.9rem',
            marginBottom: 12,
          }}>
            {success}
          </div>
        )}
      </div>

      <div className="format-grid">
        {/* CSV */}
        <div className="format-card">
          <h3>CSV</h3>
          <p>
            Comma-separated values for stability scan matrices, double-mutant epistasis pairs,
            or single-mutation summaries.
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10 }}>
            {hasCsvData
              ? `Uses: ✓ ${availableCsvSources.find((s) => s.value === resolvedCsvSource)?.label ?? ''}`
              : '— no CSV-exportable data'}
          </div>
          {availableCsvSources.length > 1 && (
            <div style={{ marginBottom: 10 }}>
              <select
                aria-label="Select dataset for CSV export"
                value={resolvedCsvSource ?? ''}
                onChange={(e) => setCsvSource(e.target.value as CsvSource)}
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                  fontSize: '0.82rem',
                  background: '#fafafa',
                }}
              >
                {availableCsvSources.map((src) => (
                  <option key={src.value} value={src.value}>
                    {src.label}
                  </option>
                ))}
              </select>
            </div>
          )}
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={() => doExport('csv')}
            disabled={!hasCsvData || busy === 'csv'}
          >
            {busy === 'csv' && <span className="spinner" />}
            Download CSV
          </button>
        </div>

        {/* JSON */}
        <div className="format-card">
          <h3>JSON</h3>
          <p>
            Full structured JSON containing all computed results (ΔΔG, scan, epistasis, IDR, compare)
            or standalone epistasis output.
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10 }}>
            Includes:{' '}
            {[
              hasDdg && 'DDG',
              hasScan && 'Scan',
              hasEpistasis && 'Epistasis',
              hasIdr && 'IDR',
              hasCompare && 'Compare',
            ].filter(Boolean).join(', ') || '— no data'}
          </div>
          {hasEpistasis && (
            <div style={{ marginBottom: 10 }}>
              <select
                aria-label="Select JSON export scope"
                value={resolvedJsonScope}
                onChange={(e) => setJsonScope(e.target.value as JsonScope)}
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                  fontSize: '0.82rem',
                  background: '#fafafa',
                }}
              >
                <option value="all">All session results</option>
                <option value="epistasis">Epistasis only ({epistasis!.n_pairs} pairs)</option>
              </select>
            </div>
          )}
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={() => doExport('json')}
            disabled={!anyResult || busy === 'json'}
          >
            {busy === 'json' && <span className="spinner" />}
            Download JSON
          </button>
        </div>

        {/* PDB */}
        <div className="format-card">
          <h3>PDB (B-factor annotated)</h3>
          <p>
            PDB file placing residues on a 2D reference grid with minimum predicted ΔΔG in the
            B-factor column for visualization (not a 3D fold prediction).
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12 }}>
            Requires: {hasScan ? '✓ Scan result (with sequence)' : '— run a scan first'}
          </div>
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={() => doExport('pdb')}
            disabled={!hasScan || busy === 'pdb'}
          >
            {busy === 'pdb' && <span className="spinner" />}
            Download PDB
          </button>
        </div>
      </div>

      {/* Status summary */}
      <div className="card" style={{ marginTop: 8 }}>
        <div className="card-title">Session Data Status</div>
        <div className="stat-chips">
          <div className="stat-chip">
            ΔΔG:{' '}
            {hasDdg
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.ddg!.mutation}</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
          <div className="stat-chip">
            Scan:{' '}
            {hasScan
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.scan!.positions.length} positions</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
          <div className="stat-chip">
            Epistasis:{' '}
            {hasEpistasis
              ? <strong style={{ color: 'var(--green)' }}>✓ {epistasis!.n_pairs} pair(s)</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
          <div className="stat-chip">
            IDR:{' '}
            {hasIdr
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.idr!.length} residues</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
          <div className="stat-chip">
            Compare:{' '}
            {hasCompare
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.compare!.uniprot_id}</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
        </div>
      </div>
    </div>
  )
}
