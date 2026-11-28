// Health check.
// There is no demo mode: without a trained checkpoint the backend reports
// ready=false and every prediction endpoint returns 503 with a reason.
export interface CalibrationInfo {
  expected_calibration_error: number | null
  max_coverage_gap?: number | null
  observed_coverage: Record<string, number> | null
  well_calibrated?: boolean | null
  // Thresholds `well_calibrated` was computed against (scripts/calibration.py).
  tolerances?: { ece: number; coverage_gap: number } | null
  // Split the coverage was measured on (the variance scale is fitted on `val`).
  split?: string | null
  measured_on: string
  // Which sigma these numbers describe. The interface may serve `sigma * scale`,
  // and reporting the raw sigma's coverage against a scaled band is R2-09.
  describes: string
}

export type PredictionMode = 'structure_free' | 'invariant_3d'

export interface EpistasisCapability {
  available: boolean
  reason: string | null
  source: string | null
  uncertainty: string
}

export interface CdsCapability {
  available: boolean
  reason: string | null
}

export interface Invariant3dCapability {
  available: boolean
  reason: string | null
  weights_path: string | null
}

export interface StreamingCapability {
  available: boolean
  endpoints: string[]
}

export interface HealthCapabilities {
  epistasis: EpistasisCapability
  cds: CdsCapability
  invariant_3d: Invariant3dCapability
  streaming: StreamingCapability
  modes: Record<PredictionMode, string>
}

export interface HealthResponse {
  ready: boolean
  error: string | null
  weights_path: string
  esm_model: string | null
  window: number | null
  untrained_heads: string[]
  uncertainty_calibrated: boolean
  sigma_scale: number
  calibration: CalibrationInfo | null
  calibration_note: string | null
  capabilities: HealthCapabilities
}

export interface TabProps {
  health: HealthResponse | null
}

// Shared v2 metadata blocks (CDS & 3D structure alignment)
export interface DdgPathNumbers {
  ddg: number
  sigma: number
  ci_low: number
  ci_high: number
  verdict: string
}

export interface CdsInfo {
  supplied: boolean
  frame_verified?: boolean
  n_codons?: number
  stop_codon_stripped?: boolean
  organism?: string
  used_in_headline?: boolean
  with_cds?: DdgPathNumbers
  without_cds?: DdgPathNumbers
  delta_ddg?: number
  note?: string
}

export interface StructureAlignmentInfo {
  source: string
  residues_in_pdb: number
  aligned_offset: number
  mismatches: number
  mean_plddt: number
  fraction_below_70: number
  usable_residue_fraction: number
  low_confidence_positions: number[]
  // Which residues of the sequence carry coordinates (a structure may miss a terminal tail).
  sequence_coverage?: { first: number; last: number; covered: number; length: number; fraction: number }
  residues_without_coordinates?: number[]
  masked_positions?: number[]
  mutation_site_masked?: boolean
  note: string
}

export interface ModeComparison {
  structure_free: DdgPathNumbers
  invariant_3d: DdgPathNumbers
  symmetry: Record<PredictionMode, string>
  delta_ddg_3d_minus_structure_free: number
  structure?: StructureAlignmentInfo
}

// DDG predictor
export interface DdgRequest {
  wt_sequence: string
  mutation: string
  cds?: string | null
  mode?: PredictionMode
  structure_pdb?: string | null
  compare_modes?: boolean
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
  // Names the head the verdict came from, so the interface never has to guess a
  // direction from the sign of ddg (R1-25 / R2-03).
  verdict_source: string
  mode: PredictionMode
  uses_coordinates: boolean
  symmetry: string
  structure?: StructureAlignmentInfo
  checkpoint?: string
  cds: CdsInfo
  comparison?: ModeComparison
}

// Stability scan
export interface ScanRequest {
  sequence: string
  start?: number
  end?: number | null
  cds?: string | null
  mode?: PredictionMode
  structure_pdb?: string | null
}

export interface ScanTiming {
  encode_ms: number
  scan_ms: number
  total_mutations: number
  muts_per_sec: number
}

export interface ScanResponse {
  positions: number[]
  wt_residues: string[]
  aa_order: string[]
  matrix: number[][]
  mode: PredictionMode
  uses_coordinates: boolean
  symmetry: string
  structure?: StructureAlignmentInfo
  checkpoint?: string
  cds: CdsInfo
  timing: ScanTiming
}

export interface ScanStreamStart {
  length: number
  positions: [number, number]
  total: number
  aa_order: string[]
  mode: PredictionMode
  uses_coordinates: boolean
  symmetry: string
  structure?: StructureAlignmentInfo
  checkpoint?: string
  cds: CdsInfo
}

export interface ScanStreamChunk {
  done: number
  total: number
  positions: number[]
  columns: number[][]
  elapsed_s: number
  eta_s: number
}

// Double-Mutant Epistasis
export interface EpistasisRequest {
  wt_sequence: string
  pairs: string[]
  cds?: string | null
}

export interface EpistasisHeadInfo {
  trained: boolean
  source: string | null
  contact_gated: boolean
  gate_input: string
}

export interface EpistasisPairGeo {
  direct_contact_prob: number
  log_seq_sep: number
  inv_hop_dist: number
  is_direct_contact: number
}

export interface EpistasisPairResult {
  pair: string
  sites: [number, number]
  ddg_total: number
  ddg_additive: number
  epistasis_dddg: number
  sigma: number
  ci_low: number
  ci_high: number
  contact_prob: number
  seq_separation: number
  is_contact: boolean
  pair_geo: EpistasisPairGeo
}

export interface EpistasisTiming {
  encode_ms: number
  predict_ms: number
  pairs_per_sec: number
}

export interface EpistasisResponse {
  length: number
  n_pairs: number
  pairs: EpistasisPairResult[]
  mode: PredictionMode
  uses_coordinates: boolean
  symmetry: string
  epistasis_head: EpistasisHeadInfo
  cds: CdsInfo
  ci_source: string
  uncertainty_calibrated: boolean
  uncertainty_note: string
  timing: EpistasisTiming
}

export interface EpistasisStreamStart {
  length: number
  total: number
  cds: CdsInfo
  epistasis_head: EpistasisHeadInfo
}

export interface EpistasisStreamChunk {
  done: number
  total: number
  pairs: EpistasisPairResult[]
  elapsed_s: number
  eta_s: number
}

export interface StreamError {
  status: number
  detail: string
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
  value_source: string
  ci_source: string
}

// AlphaFold Compare
export interface CompareRequest {
  uniprot_id: string
}

export interface PlddtBands {
  very_high_gt_90: number
  confident_70_90: number
  low_50_70: number
  very_low_le_50: number
}

export interface CompareResponse {
  uniprot_id: string
  sequence: string
  structure_pdb: string
  plddt: number[]
  plddt_mean?: number
  plddt_bands?: PlddtBands
  low_confidence_positions?: number[]
  low_confidence_in_scan?: boolean[]
  plddt_note?: string
  pae: number[][]
  pae_downsampled: boolean
  // Residues per displayed PAE cell (cell k = residue 1 + k * pae_stride).
  pae_stride?: number
  min_ddg_per_position: number[]
}

// Export
export interface ExportRequest {
  format: 'csv' | 'json' | 'pdb'
  filename?: string
  data: unknown
}

// History
export type HistoryKind = 'ddg' | 'scan' | 'epistasis' | 'idr' | 'compare'

export interface HistoryItem {
  id: number
  kind: HistoryKind
  created_at: number
  summary: string
}

export interface HistoryRecord extends HistoryItem {
  request: Record<string, unknown>
  response: Record<string, unknown>
}
