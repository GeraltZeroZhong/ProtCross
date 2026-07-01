export type BackendMode = "cpu" | "gpu" | "conda";

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
    proxy_url: string | null;
  };
  readiness?: {
    ready: boolean;
    issues: string[];
  };
}

export interface PredictResponse {
  ok: boolean;
  summary: SummaryJson;
  pockets?: PocketJson;
  top_pocket_residues?: ResidueSummary[];
  output_files: Record<string, string>;
}

export interface BatchResultResponse extends PredictResponse {
  input_structure: string;
}

export interface SummaryJson {
  schema_version: string;
  input_structure?: string;
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
  probability: number;
  chain_id?: string;
  residue_number?: string | number;
  cluster_id?: number | null;
  [key: string]: unknown;
}

export interface PocketJson {
  schema_version: string;
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
  status: string;
  output_dir?: string | null;
  output_files?: Record<string, string>;
  error?: string | null;
}
