/**
 * What the server has and has not established about its own outputs, derived ONLY from
 * /api/health. Nothing here is a judgement the backend did not make: the calibration
 * verdict is `calibration.well_calibrated` as computed by scripts/calibration.py, and the
 * numbers shown are the ones that verdict was computed from.
 */
import type { HealthResponse } from './types'

export interface CoverageLevel {
  nominal: number   // 0.5, 0.8, 0.9, 0.95
  observed: number  // empirical coverage of that central interval
  gap: number       // observed - nominal (positive = conservative, interval wider than needed)
  withinTolerance: boolean
}

export type CalibrationStatus = 'certified' | 'measured' | 'not_measured' | 'pending'

export interface CalibrationView {
  status: CalibrationStatus
  ece: number | null
  eceTolerance: number
  gapTolerance: number
  levels: CoverageLevel[]
  coverage95: CoverageLevel | null
  worst: CoverageLevel | null
  sigmaScale: number
  split: string | null
  describes: string | null
  note: string | null
  /** 'conservative' when every measured interval over-covers, 'overconfident' when every one under-covers. */
  direction: 'conservative' | 'overconfident' | 'mixed' | null
}

function parseLevel(key: string): number | null {
  const m = /^(\d+(?:\.\d+)?)\s*%?$/.exec(key.trim())
  if (!m) return null
  const v = Number(m[1])
  return v > 1 ? v / 100 : v
}

export function calibrationView(health: HealthResponse | null): CalibrationView {
  const empty: CalibrationView = {
    status: 'pending', ece: null, eceTolerance: 0.05, gapTolerance: 0.05, levels: [], coverage95: null,
    worst: null, sigmaScale: 1, split: null, describes: null, note: null, direction: null,
  }
  if (!health || !health.ready) return empty
  const cal = health.calibration
  if (!cal) {
    return { ...empty, status: 'not_measured', sigmaScale: health.sigma_scale ?? 1, note: health.calibration_note }
  }
  const eceTolerance = cal.tolerances?.ece ?? 0.05
  const gapTolerance = cal.tolerances?.coverage_gap ?? 0.05
  const levels: CoverageLevel[] = Object.entries(cal.observed_coverage ?? {})
    .map(([k, observed]) => {
      const nominal = parseLevel(k)
      if (nominal === null || typeof observed !== 'number') return null
      const gap = observed - nominal
      return { nominal, observed, gap, withinTolerance: Math.abs(gap) <= gapTolerance + 1e-9 }
    })
    .filter((x): x is CoverageLevel => x !== null)
    .sort((a, b) => a.nominal - b.nominal)
  const worst = levels.reduce<CoverageLevel | null>(
    (best, cur) => (best === null || Math.abs(cur.gap) > Math.abs(best.gap) ? cur : best), null)
  const coverage95 = levels.find((l) => Math.abs(l.nominal - 0.95) < 1e-6) ?? null
  const direction = levels.length === 0
    ? null
    : levels.every((l) => l.gap >= 0) ? 'conservative'
      : levels.every((l) => l.gap <= 0) ? 'overconfident' : 'mixed'
  return {
    status: cal.well_calibrated ? 'certified' : 'measured',
    ece: cal.expected_calibration_error,
    eceTolerance,
    gapTolerance,
    levels,
    coverage95,
    worst,
    sigmaScale: health.sigma_scale ?? 1,
    split: cal.split ?? null,
    describes: cal.describes ?? null,
    note: health.calibration_note,
    direction,
  }
}

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`

/** One-line, decision-point summary of what an interval does and does not mean. */
export function calibrationHeadline(view: CalibrationView): string {
  if (view.status === 'pending') return 'Checking calibration…'
  if (view.status === 'not_measured') return 'Not measured: intervals are the model’s raw σ'
  const parts: string[] = []
  if (view.ece !== null) parts.push(`ECE ${view.ece.toFixed(4)}`)
  if (view.coverage95) parts.push(`95% CI covers ${pct(view.coverage95.observed)}`)
  if (view.sigmaScale !== 1) parts.push(`σ × ${view.sigmaScale.toFixed(3)}`)
  return parts.join(' · ')
}

/** The single reason a measured calibration was not certified, in the verdict's own terms. */
export function calibrationCaveat(view: CalibrationView): string | null {
  if (view.status !== 'measured') return null
  const reasons: string[] = []
  if (view.ece !== null && view.ece > view.eceTolerance) {
    reasons.push(`ECE ${view.ece.toFixed(3)} > ${view.eceTolerance} tolerance`)
  }
  const off = view.levels.filter((l) => !l.withinTolerance)
  for (const l of off) {
    reasons.push(
      `${pct(l.nominal, 0)} band covers ${pct(l.observed)} (gap ${Math.abs(l.gap).toFixed(3)} > ${view.gapTolerance})`)
  }
  return reasons.length ? reasons.join('; ') : 'not certified by the calibration run'
}

// ---------------------------------------------------------------------------
// Residue numbering for constructs (e.g. TP53 DBD residues 94-292)
// ---------------------------------------------------------------------------

export interface ParsedMutation {
  wt: string
  pos: number
  mut: string
}

export function parseMutation(text: string): ParsedMutation | null {
  const m = /^\s*([A-Za-z])\s*(\d+)\s*([A-Za-z])\s*$/.exec(text)
  if (!m) return null
  return { wt: m[1].toUpperCase(), pos: Number(m[2]), mut: m[3].toUpperCase() }
}

/** offset = first residue number - 1; native = sequence index + offset. */
export function toSequenceIndex(nativePos: number, firstResidueNumber: number): number {
  return nativePos - (firstResidueNumber - 1)
}

export function toNative(seqIndex: number, firstResidueNumber: number): number {
  return seqIndex + (firstResidueNumber - 1)
}
