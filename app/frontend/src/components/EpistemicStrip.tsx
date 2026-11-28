import type { HealthResponse } from '../types'
import { calibrationCaveat, calibrationHeadline, calibrationView, pct } from '../epistemic'

/**
 * One calm, scannable status line under the header that replaces two full-width warning
 * banners. Every limitation stays visible above the fold (progressive disclosure, not
 * concealment): the chips carry the numbers, the "Details" drawer carries the reasoning.
 */
export default function EpistemicStrip({ health }: { health: HealthResponse | null }) {
  if (!health || !health.ready) return null

  const view = calibrationView(health)
  const caveat = calibrationCaveat(view)
  const untrained = health.untrained_heads ?? []
  const modes = health.capabilities?.modes

  const calClass =
    view.status === 'certified' ? 'epi-ok' : view.status === 'measured' ? 'epi-caution' : 'epi-warn'
  const calGlyph = view.status === 'certified' ? '✓' : view.status === 'measured' ? '◐' : '!'
  // One short verdict on the strip; ECE, σ scale and per-band gaps live in the tooltip and drawer.
  const calText =
    view.status === 'certified'
      ? `calibrated${view.coverage95 ? ` · 95% CI covers ${pct(view.coverage95.observed)}` : ''}`
      : view.status === 'measured'
        ? `measured, not certified${view.coverage95 ? ` · 95% CI covers ${pct(view.coverage95.observed)}` : ''}`
        : view.status === 'not_measured'
          ? 'not calibrated (no calibration run)'
          : 'checking…'
  const calTitle = [calibrationHeadline(view), caveat ? `Not certified: ${caveat}` : null, view.describes]
    .filter(Boolean)
    .join('\n')

  return (
    <section className="epistemic-strip" aria-label="Model status and known limitations">
      <div className="epistemic-inner">
        <span className="epi-chip epi-info" title={modes?.structure_free ?? undefined}>
          <span className="epi-glyph" aria-hidden="true">◇</span>
          <span className="epi-key">Default</span>
          <span className="epi-val">structure-free · swap-equivariant only</span>
          <span className="epi-sub">(not SE(3)/E(3)-equivariant)</span>
        </span>

        <span className={`epi-chip ${calClass}`} title={calTitle}>
          <span className="epi-glyph" aria-hidden="true">{calGlyph}</span>
          <span className="epi-key">Uncertainty</span>
          <span className="epi-val">{calText}</span>
        </span>

        {untrained.length > 0 && (
          <span
            className="epi-chip epi-muted"
            title={`Untrained, endpoints disabled: ${untrained.join(', ')}. No training data was supplied for these heads.`}
          >
            <span className="epi-glyph" aria-hidden="true">○</span>
            <span className="epi-val">{untrained.length} untrained head{untrained.length === 1 ? '' : 's'} disabled</span>
          </span>
        )}

        <details className="epi-details">
          <summary>Details</summary>
          <div className="epi-drawer" role="note">
            <div className="epi-drawer-section">
              <div className="epi-drawer-title">Uncertainty calibration</div>
              {view.status === 'not_measured' ? (
                <p>
                  No calibration run describes the interval this server draws
                  {view.note ? <> ({view.note})</> : null}. Intervals are the model’s predicted σ
                  (Gaussian, 95%); run <code>python run.py calibration</code> before reading one as a 95% interval.
                </p>
              ) : (
                <>
                  <p>
                    {view.status === 'certified' ? (
                      <>Certified well-calibrated by <code>scripts.calibration</code>.</>
                    ) : (
                      <>
                        Measured but <strong>not certified</strong> well-calibrated: {caveat}. The verdict requires
                        ECE ≤ {view.eceTolerance} and every coverage gap ≤ {view.gapTolerance}.
                      </>
                    )}{' '}
                    {view.direction === 'conservative' &&
                      'Every measured interval over-covers (is wider than nominal), so the error is on the cautious side.'}
                    {view.direction === 'overconfident' &&
                      'Every measured interval under-covers (is narrower than nominal): treat intervals as optimistic.'}
                  </p>
                  {view.levels.length > 0 && (
                    <table className="epi-table">
                      <thead>
                        <tr>
                          <th scope="col">Nominal</th>
                          <th scope="col">Observed</th>
                          <th scope="col">Gap</th>
                          <th scope="col">Within ±{view.gapTolerance}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {view.levels.map((l) => (
                          <tr key={l.nominal}>
                            <td>{pct(l.nominal, 0)}</td>
                            <td>{pct(l.observed)}</td>
                            <td>{l.gap >= 0 ? '+' : '−'}{Math.abs(l.gap).toFixed(3)}</td>
                            <td>{l.withinTolerance ? 'yes' : 'no'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  <p className="epi-fine">
                    {view.ece !== null && <>ECE {view.ece.toFixed(4)} (tolerance {view.eceTolerance}). </>}
                    {view.sigmaScale !== 1 && <>Served σ = model σ × {view.sigmaScale.toFixed(3)} (scale fitted on the validation split). </>}
                    {view.split && <>Coverage measured on the <code>{view.split}</code> split. </>}
                    {view.describes && <>Numbers describe {view.describes}. </>}
                    Epistasis intervals are not covered: calibration was measured on single mutants only.
                  </p>
                </>
              )}
            </div>
            {untrained.length > 0 && (
              <div className="epi-drawer-section">
                <div className="epi-drawer-title">Untrained heads</div>
                <p>
                  {untrained.join(', ')}: no data was supplied for these tasks, so they were never trained and
                  their endpoints refuse requests instead of returning untrained outputs.
                </p>
              </div>
            )}
            {modes && (
              <div className="epi-drawer-section">
                <div className="epi-drawer-title">Symmetry of each mode</div>
                <p><strong>Structure-free (default):</strong> {modes.structure_free}</p>
                <p><strong>Optional 3D mode:</strong> {modes.invariant_3d}</p>
              </div>
            )}
          </div>
        </details>
      </div>
    </section>
  )
}
