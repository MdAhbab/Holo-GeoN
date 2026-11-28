import type { HealthResponse, PredictionMode } from '../types'
import { calibrationCaveat, calibrationView, pct } from '../epistemic'

/**
 * What a 95% interval means, stated where the interval is read (the point of decision),
 * in one line. The full reasoning sits in the tooltip and in the status strip's Details
 * drawer, so the limitation is never hidden but never shouts. Wording follows the
 * backend's own verdict.
 */
export default function IntervalNote({
  health,
  calibrated,
  ciSource,
  mode,
}: {
  health: HealthResponse | null
  calibrated: boolean
  ciSource: string
  mode: PredictionMode
}) {
  const view = calibrationView(health)

  let cls = 'interval-note'
  let glyph = 'ⓘ'
  let text: React.ReactNode = <>Interval source: {ciSource}.</>
  let title = `Interval source: ${ciSource}.`

  if (mode === 'invariant_3d') {
    cls += ' interval-caution'
    glyph = '◐'
    text = <><strong>Not calibrated</strong> for the 3D checkpoint (model σ).</>
    title = `Interval source: ${ciSource}. No calibration run describes the 3D checkpoint.`
  } else if (calibrated || view.status === 'certified') {
    cls += ' interval-ok'
    glyph = '✓'
    text = (
      <>
        <strong>Calibrated interval</strong>
        {view.coverage95 && <> · 95% coverage {pct(view.coverage95.observed)}</>}
      </>
    )
  } else if (view.status === 'measured') {
    cls += ' interval-caution'
    glyph = '◐'
    text = (
      <>
        {view.coverage95 && <>95% interval covered <strong>{pct(view.coverage95.observed)}</strong> of {view.split ?? 'held-out'} mutations</>}
        {view.sigmaScale !== 1 && <> (σ × {view.sigmaScale.toFixed(2)})</>} · <strong>not certified</strong>
      </>
    )
    title = [
      `Interval source: ${ciSource}.`,
      view.ece !== null ? `ECE ${view.ece.toFixed(4)} (tolerance ${view.eceTolerance}).` : '',
      `Not certified because ${calibrationCaveat(view)}.`,
    ].filter(Boolean).join(' ')
  } else if (view.status === 'not_measured') {
    cls += ' interval-warn'
    glyph = '!'
    text = <><strong>Not calibrated:</strong> model σ only; run <code>python run.py calibration</code>.</>
  }

  return (
    <div className={cls} role="note" title={title}>
      <span className="interval-glyph" aria-hidden="true">{glyph}</span>
      <span>{text}</span>
    </div>
  )
}
