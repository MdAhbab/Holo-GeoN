import { useState, useEffect, type ComponentType } from 'react'
import { getHealth, getHistory, getHistoryItem } from './api'
import type {
  HealthResponse,
  TabProps,
  DdgResponse,
  ScanResponse,
  EpistasisResponse,
  IdrResponse,
  CompareResponse,
} from './types'
import { setDdg, setScan, setEpistasis, setIdr, setCompare } from './store'

import DdgTabBase from './tabs/DdgTab'
import ScanTabBase from './tabs/ScanTab'
import EpistasisTabBase from './tabs/EpistasisTab'
import IdrTabBase from './tabs/IdrTab'
import CompareTabBase from './tabs/CompareTab'
import ExportTabBase from './tabs/ExportTab'
import HistoryTabBase from './tabs/HistoryTab'
import EpistemicStrip from './components/EpistemicStrip'

import './styles.css'

const DdgTab = DdgTabBase as ComponentType<TabProps>
const ScanTab = ScanTabBase as ComponentType<TabProps>
const EpistasisTab = EpistasisTabBase as ComponentType<TabProps>
const IdrTab = IdrTabBase as ComponentType<TabProps>
const CompareTab = CompareTabBase as ComponentType<TabProps>
const ExportTab = ExportTabBase as ComponentType<TabProps>
const HistoryTab = HistoryTabBase as ComponentType<TabProps>

const TABS = [
  { id: 'ddg',       label: 'ΔΔG Predictor' },
  { id: 'scan',      label: 'Stability Landscape' },
  { id: 'epistasis', label: 'Double-Mutant Epistasis' },
  { id: 'idr',       label: 'IDR Ensemble' },
  { id: 'compare',   label: 'AlphaFold Compare' },
  { id: 'export',    label: 'Export' },
  { id: 'history',   label: 'History' },
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
        const kinds = ['ddg', 'scan', 'epistasis', 'idr', 'compare'] as const
        await Promise.all(kinds.map(async (kind) => {
          const latest = items.find((it) => it.kind === kind)
          if (!latest) return
          try {
            const record = await getHistoryItem(latest.id)
            if (kind === 'ddg') {
              setDdg(record.response as unknown as DdgResponse)
            } else if (kind === 'scan') {
              const seq = typeof record.request?.sequence === 'string' ? record.request.sequence : undefined
              setScan({
                ...(record.response as unknown as ScanResponse),
                ...(seq ? { sequence: seq } : {}),
              })
            } else if (kind === 'epistasis') {
              const wtSeq = typeof record.request?.wt_sequence === 'string' ? record.request.wt_sequence : undefined
              setEpistasis({
                ...(record.response as unknown as EpistasisResponse),
                ...(wtSeq ? { wt_sequence: wtSeq } : {}),
              })
            } else if (kind === 'idr') {
              setIdr(record.response as unknown as IdrResponse)
            } else if (kind === 'compare') {
              setCompare(record.response as unknown as CompareResponse)
            }
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

  const caps = health?.capabilities

  return (
    <>
      <header className="app-header">
        <div className="app-header-main">
          <div className="brand">
            <h1>Holo-GeoN</h1>
            <p>Protein stability, epistasis &amp; coding-sequence-aware graph neural network</p>
          </div>
          {caps && (
            <div className="header-cap-pills" aria-label="Server capabilities">
              <span
                className={`cap-pill ${caps.epistasis.available ? 'cap-on' : 'cap-off'}`}
                title={caps.epistasis.reason ?? (caps.epistasis.source ? `Source: ${caps.epistasis.source}` : 'Epistasis head trained')}
              >
                Epistasis: {caps.epistasis.available ? 'Ready' : 'Unavailable'}
              </span>
              <span
                className={`cap-pill ${caps.cds.available ? 'cap-on' : 'cap-off'}`}
                title={caps.cds.reason ?? 'ViennaRNA available for real coding sequences'}
              >
                CDS (ViennaRNA): {caps.cds.available ? 'Ready' : 'Unavailable'}
              </span>
              <span
                className={`cap-pill ${caps.invariant_3d.available ? 'cap-on' : 'cap-off'}`}
                title={caps.invariant_3d.reason ?? `3D SE(3)-invariant weights: ${caps.invariant_3d.weights_path ?? 'loaded'}`}
              >
                3D (SE(3)-invariant): {caps.invariant_3d.available ? 'Ready' : 'Unavailable'}
              </span>
            </div>
          )}
        </div>
      </header>

      {health && !health.ready && (
        <div className="error-banner" role="alert">
          <span aria-hidden="true" style={{ fontSize: '1.1rem' }}>⚠</span>
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

      {/* Calibration status, untrained heads and the default symmetry, as one status line
          (was two full-width warning banners on every tab). */}
      <EpistemicStrip health={health} />

      <nav className="tab-bar" role="tablist" aria-label="Main navigation">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            className={`tab-btn${activeTab === tab.id ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main
        id={`panel-${activeTab}`}
        className="tab-content"
        role="tabpanel"
        aria-labelledby={`tab-${activeTab}`}
      >
        {activeTab === 'ddg'       && <DdgTab health={health} />}
        {activeTab === 'scan'      && <ScanTab health={health} />}
        {activeTab === 'epistasis' && <EpistasisTab health={health} />}
        {activeTab === 'idr'       && <IdrTab health={health} />}
        {activeTab === 'compare'   && <CompareTab health={health} />}
        {activeTab === 'export'    && <ExportTab health={health} />}
        {activeTab === 'history'   && <HistoryTab health={health} />}
      </main>
    </>
  )
}
