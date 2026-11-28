import { useState } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import { postIdr } from '../api'
import { setIdr } from '../store'
import type { IdrResponse } from '../types'
import { BASE_LAYOUT, axisTitle, baseAxis } from '../plotTheme'

const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

/** Compute Gaussian PDF values over a range */
function gaussianCurve(mu: number, sigma: number, nPoints = 200) {
  const range = 4 * sigma
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < nPoints; i++) {
    const x = (mu - range) + (2 * range * i) / (nPoints - 1)
    const y = (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    xs.push(x)
    ys.push(y)
  }
  return { xs, ys }
}

export default function IdrTab() {
  const [sequence, setSequence] = useState(UBIQUITIN)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<IdrResponse | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await postIdr({ sequence: sequence.trim() })
      setResult(res)
      setIdr(res)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  const gaussian = result ? gaussianCurve(result.mu, result.sigma) : null

  const compactX = gaussian ? gaussian.xs.filter(x => x < result!.mu - result!.sigma) : []
  const compactY = compactX.map(x =>
    result
      ? (1 / (result.sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - result.mu) / result.sigma) ** 2)
      : 0
  )
  const expandX = gaussian ? gaussian.xs.filter(x => x > result!.mu + result!.sigma) : []
  const expandY = expandX.map(x =>
    result
      ? (1 / (result.sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - result.mu) / result.sigma) ** 2)
      : 0
  )

  const gaussianData = gaussian
    ? [
        {
          type: 'scatter',
          x: [...compactX, ...compactX.slice().reverse()],
          y: [...compactY, ...compactY.map(() => 0)],
          fill: 'toself',
          fillcolor: 'rgba(239,68,68,0.15)',
          line: { color: 'transparent' },
          name: 'Compact globule',
          hoverinfo: 'skip',
        },
        {
          type: 'scatter',
          x: [...expandX, ...expandX.slice().reverse()],
          y: [...expandY, ...expandY.map(() => 0)],
          fill: 'toself',
          fillcolor: 'rgba(37,99,235,0.12)',
          line: { color: 'transparent' },
          name: 'Expanded coil',
          hoverinfo: 'skip',
        },
        {
          type: 'scatter',
          x: gaussian.xs,
          y: gaussian.ys,
          mode: 'lines',
          line: { color: '#008080', width: 2.5 },
          name: 'Ensemble Rg distribution',
          hovertemplate: 'Rg = %{x:.2f} Å<br>p(Rg) = %{y:.4f}<extra></extra>',
        },
      ]
    : []

  const gaussianAnnotations = result
    ? [
        {
          x: result.mu - result.sigma,
          y: 0,
          text: 'compact',
          showarrow: false,
          yanchor: 'bottom',
          font: { size: 10, color: '#dc2626' },
        },
        {
          x: result.mu + result.sigma,
          y: 0,
          text: 'expanded',
          showarrow: false,
          yanchor: 'bottom',
          font: { size: 10, color: '#2563eb' },
        },
      ]
    : []

  const gaussianLayout = {
    ...BASE_LAYOUT,
    height: 300,
    margin: { l: 60, r: 20, t: 40, b: 50 },
    xaxis: baseAxis({ title: axisTitle('Radius of gyration Rg (Å)') }),
    yaxis: baseAxis({ title: axisTitle('Probability density') }),
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom' },
    annotations: gaussianAnnotations,
  }



  return (
    <div>
      <div className="card">
        <div className="card-title">IDR Ensemble — Intrinsic Disorder & Radius of Gyration</div>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 14 }}>
            <label htmlFor="idr-seq">Protein Sequence</label>
            <textarea
              id="idr-seq"
              rows={3}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Paste single-letter amino acid sequence..."
            />
          </div>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || !sequence.trim()}
          >
            {loading && <span className="spinner" />}
            {loading ? 'Analyzing…' : 'Analyze Ensemble'}
          </button>
        </form>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <>
          <div className="card">
            <div className="card-title">Ensemble Statistics</div>
            <div className="stat-chips">
              <div className="stat-chip">Sequence length: <strong>{result.length} residues</strong></div>
              <div className="stat-chip">μ(Rg): <strong>{result.mu.toFixed(3)} Å</strong></div>
              <div className="stat-chip">σ(Rg): <strong>{result.sigma.toFixed(3)} Å</strong></div>
              <div className="stat-chip">
                95% interval:{' '}
                <strong>{result.ci_low.toFixed(2)} – {result.ci_high.toFixed(2)} Å</strong>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Ensemble Rg Distribution (Gaussian approximation)</div>
            <div className="plot-wrap">
              <Plot
                data={gaussianData}
                layout={gaussianLayout}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
                useResizeHandler
              />
            </div>
          </div>

        </>
      )}
    </div>
  )
}
