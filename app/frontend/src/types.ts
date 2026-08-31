// Health check.
// There is no demo mode: without a trained checkpoint the backend reports
// ready=false and every prediction endpoint returns 503 with a reason.
export interface CalibrationInfo {
  expected_calibration_error: number | null
  observed_coverage: Record<string, number> | null
  measured_on: string
}

export interface HealthResponse {
  ready: boolean
  error: string | null
  weights_path: string
  esm_model: string | null
  window: number | null
  untrained_heads: string[]
  uncertainty_calibrated: boolean
  calibration: CalibrationInfo | null
}

// DDG predictor
export interface DdgRequest {
  wt_sequence: string
  mutation: string
}

export interface DdgResponse {
  mutation: string
  position: number
  wt_residue: string
  mut_residue: string
  ddg: number
  // Predicted standard deviation from the model's own heteroscedastic head,
  // trained under Gaussian NLL. The interval below is derived from it, not
  // from a fixed-width formula.
  sigma: number
  ci_low: number
  ci_high: number
  ci_source: string
  uncertainty_calibrated: boolean
  three_state: {
    destabilizing: number
    neutral: number
    stabilizing: number
  }
  verdict: string
}

// Stability scan
export interface ScanRequest {
  sequence: string
  start: number
  end: number
}

export interface ScanResponse {
  positions: number[]
  wt_residues: string[]
  aa_order: string[]
  matrix: number[][]
}

// IDR ensemble
export interface IdrRequest {
  sequence: string
}

export interface IdrResponse {
  length: number
  mu: number
  sigma: number
  ci_low: number
  ci_high: number
}

// AlphaFold Compare
export interface CompareRequest {
  uniprot_id: string
}

export interface CompareResponse {
  uniprot_id: string
  sequence: string
  structure_pdb: string
  plddt: number[]
  pae: number[][]
  pae_downsampled: boolean
  min_ddg_per_position: number[]
}

// Export
export interface ExportRequest {
  format: 'csv' | 'json' | 'pdb'
  filename?: string
  data: unknown
}

// History
export interface HistoryItem {
  id: number
  kind: 'ddg' | 'scan' | 'idr' | 'compare'
  created_at: number
  summary: string
}

export interface HistoryRecord extends HistoryItem {
  request: Record<string, unknown>
  response: Record<string, unknown>
}
