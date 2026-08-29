export type BackendMode = "cpu" | "gpu" | "conda";

export interface AssetDownloadJob {
  id: string;
  filename: string;
  status: "queued" | "running" | "cancelling" | "cancelled" | "failed" | "completed";
  downloaded_bytes: number;
  total_bytes?: number | null;
  percent?: number | null;
  bytes_per_second?: number | null;
  error?: string | null;
  resumable: boolean;
}

export interface FileStatus {
  path: string | null;
  present: boolean;
  expected_sha256?: string | null;
  actual_sha256?: string | null;
  verified?: boolean | null;
  verification_source?: string | null;
}

export interface AssetStatus {
  ready: boolean;
  checkpoint: FileStatus;
  pca: FileStatus;
  esm: {
    license_confirmed: boolean;
    path: string | null;
    present: boolean;
    source: string | null;
    expected_sha256: string | null;
    actual_sha256: string | null;
    verified: boolean | null;
    filename: string;
  };
}

export interface DesktopStatus {
  paths: Record<string, string>;
  manifest: Record<string, unknown>;
  assets: AssetStatus;
  backend: {
    mode: BackendMode | null;
    python: string | null;
    python_present: boolean;
    sidecar_python?: string;
    runtime_matches_config?: boolean;
    backend_test_ok?: boolean | null;
    backend_tested_at?: string | null;
    backend_test_mode?: string | null;
    backend_test_python?: string | null;
    backend_test_package_version?: string | null;
    required_package_version?: string;
    proxy_url: string | null;
  };
  readiness?: {
    ready: boolean;
    issues: string[];
  };
  activity?: {
    batch_jobs: BatchJob[];
    asset_downloads: AssetDownloadJob[];
  };
}

export interface ChainInspection {
  chain_id: string;
  scorable_residue_count: number;
  standard_residues_missing_ca: number;
  modified_or_nonstandard_amino_acids: number;
  alternate_ca_residues: number;
  sequence_break_count: number;
  numbering_gap_count: number;
  exceeds_esm_context: boolean;
  residues_over_context_limit: number;
}

export interface StructureInspection {
  schema_version: string;
  input_structure: string;
  format: "PDB" | "mmCIF";
  model_count: number;
  available_chains: string[];
  selected_chains: string[];
  chain_summaries: ChainInspection[];
  scorable_residue_count: number;
  standard_residues_missing_ca: number;
  modified_or_nonstandard_amino_acids: number;
  alternate_ca_residues: number;
  sequence_break_count: number;
  numbering_gap_count: number;
  longest_chain_context: number;
  max_len: number;
  requires_truncation: boolean;
  warnings: string[];
  parser_warnings: string[];
  input_interpretation: Record<string, string>;
}

export interface PredictResponse {
  ok: boolean;
  summary: SummaryJson;
  pockets?: PocketJson;
  scores?: ResidueSummary[];
  top_pocket_residues?: ResidueSummary[];
  output_files: Record<string, string>;
}

export interface BatchResultResponse extends PredictResponse {
  input_structure: string;
  chain_id?: string | null;
}

export interface SummaryJson {
  schema_version: string;
  protcross_version?: string;
  asset_version?: string;
  geometry_backend?: string;
  input_structure?: string;
  input_file?: { sha256?: string; [key: string]: unknown };
  threshold?: number;
  cluster_cutoff?: number;
  top_pocket?: PocketSummary | null;
  aggregate_pocket?: PocketSummary | null;
  top_residues?: ResidueSummary[];
  output_files?: Record<string, string>;
  warnings?: string[];
  [key: string]: unknown;
}

export interface PocketSummary {
  cluster_id?: number;
  center: [number, number, number];
  residue_count: number;
  score_mean: number;
  score_max: number;
}

export interface ResidueSummary {
  residue_id: string;
  residue_key?: string;
  score?: number;
  probability: number;
  chain_id?: string;
  residue_number?: string | number;
  auth_asym_id?: string;
  label_asym_id?: string;
  auth_seq_id?: string | number;
  label_seq_id?: string | number;
  insertion_code?: string;
  cluster_id?: number | null;
  x?: number | null;
  y?: number | null;
  z?: number | null;
  rank_global?: number;
  rank_within_chain?: number;
  is_binding?: number | boolean;
  is_scored?: number | boolean;
  [key: string]: unknown;
}

export interface PocketJson {
  schema_version: string;
  threshold?: number;
  cluster_cutoff?: number;
  selected_residue_count?: number;
  aggregate_pocket?: PocketDetail | null;
  clustered_pockets: PocketDetail[];
  [key: string]: unknown;
}

export interface PocketDetail extends PocketSummary {
  center_unweighted?: [number, number, number];
  residues: ResidueSummary[];
}

export interface BatchJob {
  id: string;
  status: string;
  created_at?: number;
  completed_at?: number | null;
  retry_of?: string | null;
  settings?: {
    output_dir?: string | null;
    threshold: number;
    pocket_cluster_cutoff: number;
    allow_truncation: boolean;
    device?: string | null;
  };
  completed: number;
  failed: number;
  cancel_requested: boolean;
  item_count?: number;
  items_offset?: number;
  items_returned?: number;
  error?: string | null;
  items: BatchItem[];
}

export interface BatchItem {
  input_structure: string;
  chain_id?: string | null;
  status: string;
  output_dir?: string | null;
  output_files?: Record<string, string>;
  error?: string | null;
}
