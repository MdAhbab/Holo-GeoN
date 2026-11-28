/**
 * Shared Plotly styling for every chart in the interface.
 *
 * The bundle resolves plotly.js 3.x, which dropped the string shorthand `title: 'text'`,
 * `titlefont` and `colorbar.titleside`. Charts written that way render with NO axis or
 * colorbar titles at all, silently. Everything here uses the object form
 * `{ text, font, side }`, so build titles with `axisTitle()` / `colorbarTitle()`.
 */

export const FONT_FAMILY =
  "'IBM Plex Sans Variable', 'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"
export const MONO_FAMILY =
  "'JetBrains Mono Variable', 'JetBrains Mono', 'SFMono-Regular', Consolas, 'DejaVu Sans Mono', monospace"

/** Sign convention (dG_mut - dG_wt): ddG > 0 stabilizing, ddG < 0 destabilizing. */
export const COLORS = {
  destabilizing: '#b2182b',
  stabilizing: '#2166ac',
  neutral: '#5b6474',
  ink: '#1f2937',
  inkMuted: '#5b6474',
  grid: '#eef1f5',
  zero: '#9aa5b4',
  lowConfidence: 'rgba(230, 159, 0, 0.16)', // Okabe-Ito orange, translucent
  lowConfidenceLine: '#e69f00',
  plddt: '#0053d6',
}

/**
 * ColorBrewer RdBu (11-class), oriented so negative ddG (destabilizing) is red and positive
 * (stabilizing) is blue. Diverging, colour-vision-deficiency safe, with a near-white midpoint
 * so ddG near zero reads as "no effect" rather than as a third hue.
 */
export const DDG_DIVERGING: Array<[number, string]> = [
  [0.0, '#67001f'],
  [0.1, '#b2182b'],
  [0.2, '#d6604d'],
  [0.3, '#f4a582'],
  [0.4, '#fddbc7'],
  [0.5, '#f7f7f7'],
  [0.6, '#d1e5f0'],
  [0.7, '#92c5de'],
  [0.8, '#4393c3'],
  [0.9, '#2166ac'],
  [1.0, '#053061'],
]

/** AlphaFold-DB style PAE scale: dark green = low expected error, near-white = high. */
export const PAE_SCALE: Array<[number, string]> = [
  [0.0, '#0b4a1c'],
  [0.2, '#1f7a34'],
  [0.45, '#5fb06a'],
  [0.7, '#b7deb0'],
  [1.0, '#f4faf2'],
]

export function axisTitle(text: string, opts: { color?: string; size?: number; standoff?: number } = {}) {
  return {
    text,
    font: { family: FONT_FAMILY, size: opts.size ?? 12, color: opts.color ?? COLORS.inkMuted },
    standoff: opts.standoff ?? 8,
  }
}

export function colorbarTitle(text: string) {
  return { text, side: 'right', font: { family: FONT_FAMILY, size: 11.5, color: COLORS.inkMuted } }
}

export function baseAxis(extra: Record<string, unknown> = {}) {
  return {
    gridcolor: COLORS.grid,
    linecolor: '#cfd6df',
    showline: true,
    zeroline: false,
    ticks: 'outside',
    ticklen: 4,
    tickcolor: '#cfd6df',
    tickfont: { family: FONT_FAMILY, size: 11, color: COLORS.inkMuted },
    automargin: true,
    ...extra,
  }
}

export const BASE_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: FONT_FAMILY, size: 12, color: COLORS.ink },
  hoverlabel: { font: { family: FONT_FAMILY, size: 12 }, bgcolor: '#ffffff', bordercolor: '#cfd6df' },
}

/** Plotly config shared by static figures: no chrome in screenshots, high-DPI SVG/PNG export on hover. */
export const PLOT_CONFIG = {
  displayModeBar: 'hover',
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
  toImageButtonOptions: { format: 'svg', scale: 1 },
}

/** Smallest "nice" stride so tick labels are at least `minPx` apart. */
export function niceStride(pxPerItem: number, minPx: number): number {
  for (const s of [1, 2, 5, 10, 20, 25, 50, 100, 200, 500]) {
    if (s * pxPerItem >= minPx) return s
  }
  return 1000
}

/** Robust symmetric colour limit: the q-quantile of |values|, never below `floor`. */
export function robustAbsLimit(values: Array<number | null | undefined>, q = 0.98, floor = 1): number {
  const abs = values
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
    .map((v) => Math.abs(v))
    .sort((a, b) => a - b)
  if (abs.length === 0) return floor
  const idx = Math.min(abs.length - 1, Math.floor(q * (abs.length - 1)))
  const lim = abs[idx]
  return Math.max(floor, Math.ceil(lim * 2) / 2)
}
