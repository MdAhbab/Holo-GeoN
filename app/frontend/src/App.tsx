import { useState, useEffect } from 'react'
import { getHealth, getHistory, getHistoryItem } from './api'
import type { HealthResponse } from './types'
import type { DdgResponse, ScanResponse, IdrResponse, CompareResponse } from './types'
import { setDdg, setScan, setIdr, setCompare } from './store'

import DdgTab from './tabs/DdgTab'
import ScanTab from './tabs/ScanTab'
import IdrTab from './tabs/IdrTab'
import CompareTab from './tabs/CompareTab'
import ExportTab from './tabs/ExportTab'
import HistoryTab from './tabs/HistoryTab'

import './styles.css'

const TABS = [
  { id: 'ddg',     label: 'ΔΔG Predictor' },
  { id: 'scan',    label: 'Stability Landscape' },
  { id: 'idr',     label: 'IDR Ensemble' },
  { id: 'compare', label: 'AlphaFold Compare' },
  { id: 'export',  label: 'Export' },
  { id: 'history', label: 'History' },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('ddg')
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => {/* silently ignore — backend may not be up in static preview */})
  }, [])

  // Hydrate the store from history so the Export tab is pre-populated after a page refresh
  useEffect(() => {
    async function hydrate() {
      try {
        const items = await getHistory()
        const kinds = ['ddg', 'scan', 'idr', 'compare'] as const
        await Promise.all(kinds.map(async (kind) => {
          const latest = items.find((it) => it.kind === kind)
          if (!latest) return
          try {
            const record = await getHistoryItem(latest.id)
            if (kind === 'ddg') setDdg(record.response as unknown as DdgResponse)
            else if (kind === 'scan') setScan(record.response as unknown as ScanResponse & { sequence?: string })
            else if (kind === 'idr') setIdr(record.response as unknown as IdrResponse)
            else if (kind === 'compare') setCompare(record.response as unknown as CompareResponse)
          } catch {
            // ignore individual record fetch failures
          }
        }))
      } catch {
        // silently ignore — backend may not be running in static preview
      }
    }
    hydrate()
  }, [])

  return (
    <>
      <header className="app-header">
        <h1>Holo-GeoN</h1>
        <p>Protein Stability, Interaction &amp; Expression — Graph Neural Network Predictions</p>
      </header>

      {health && !health.ready && (
        <div className="error-banner">
          <span style={{ fontSize: '1.1rem' }}>⚠</span>
          <span>
            <strong>No model loaded — predictions are unavailable.</strong>{' '}
            This interface does not approximate results when a checkpoint is
            missing; every prediction endpoint will return an error until one is
            present. Train with <code>python run.py train</code>, or set{' '}
            <code>HOLOGEON_WEIGHTS</code> to an existing checkpoint.
            {health.error && <> &nbsp;({health.error})</>}
          </span>
        </div>
      )}

      {health?.ready && !health.uncertainty_calibrated && (
        <div className="warn-banner">
          <span style={{ fontSize: '1.1rem' }}>⚠</span>
          <span>
            <strong>Uncertainty not calibrated.</strong> Confidence intervals come
            from the model's predicted sigma, but no calibration run has measured
            whether its coverage is correct. Run{' '}
            <code>python run.py calibration</code> before treating an interval as
            a 95% interval.
          </span>
        </div>
      )}

      {health?.ready && health.untrained_heads.length > 0 && (
        <div className="warn-banner">
          <span style={{ fontSize: '1.1rem' }}>⚠</span>
          <span>
            <strong>Untrained heads:</strong> {health.untrained_heads.join(', ')}.
            No data was supplied for these tasks, so they were never trained and
            their endpoints are disabled.
          </span>
        </div>
      )}

      <nav className="tab-bar" role="tablist" aria-label="Main navigation">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`tab-btn${activeTab === tab.id ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="tab-content" role="tabpanel">
        {activeTab === 'ddg'     && <DdgTab />}
        {activeTab === 'scan'    && <ScanTab />}
        {activeTab === 'idr'     && <IdrTab />}
        {activeTab === 'compare' && <CompareTab />}
        {activeTab === 'export'  && <ExportTab />}
        {activeTab === 'history' && <HistoryTab />}
      </main>
    </>
  )
}
