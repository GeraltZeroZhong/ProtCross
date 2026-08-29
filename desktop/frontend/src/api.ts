import type {
  AssetDownloadJob,
  BackendMode,
  BatchJob,
  BatchResultResponse,
  DesktopStatus,
  PredictResponse,
  StructureInspection
} from "./types";

let baseUrl = "";
let desktopToken = "";

export function configureDesktopApi(token: string, port: number): void {
  desktopToken = token;
  baseUrl = `http://127.0.0.1:${port}`;
}

export async function fetchDesktopFile(path: string, signal?: AbortSignal): Promise<Blob> {
  const response = await fetch(`${requireBaseUrl()}/file?path=${encodeURIComponent(path)}`, {
    headers: desktopToken
      ? {
          authorization: `Bearer ${desktopToken}`,
          "x-protcross-desktop-token": desktopToken
        }
      : {},
    signal
  });
  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { error: await response.text() };
    throw new Error(payload.error ?? `File request failed: ${response.status}`);
  }
  return response.blob();
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${requireBaseUrl()}${path}`, {
    ...options,
    headers: {
      "content-type": "application/json",
      ...(desktopToken
        ? {
            authorization: `Bearer ${desktopToken}`,
            "x-protcross-desktop-token": desktopToken
          }
        : {}),
      ...(options.headers ?? {})
    }
  });
  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

function requireBaseUrl(): string {
  if (!baseUrl) {
    throw new Error("Desktop backend API is not configured yet.");
  }
  return baseUrl;
}

export function getStatus(signal?: AbortSignal): Promise<DesktopStatus> {
  return request<DesktopStatus>("/status", { signal });
}

export function confirmLicense(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/license/confirm", {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function configureBackend(mode: BackendMode, condaPython?: string, proxyUrl?: string): Promise<DesktopStatus["backend"]> {
  return request<DesktopStatus["backend"]>("/backend/configure", {
    method: "POST",
    body: JSON.stringify({
      mode,
      conda_python: condaPython || undefined,
      proxy_url: proxyUrl || undefined
    })
  });
}

export function testBackend(mode?: BackendMode): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/backend/test", {
    method: "POST",
    body: JSON.stringify(mode ? { mode } : {})
  });
}

export function importEsm(path: string, copyToCache = true): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/assets/import-esm", {
    method: "POST",
    body: JSON.stringify({ path, copy_to_cache: copyToCache })
  });
}

export function importCheckpoint(path: string): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/assets/import-checkpoint", {
    method: "POST",
    body: JSON.stringify({ path })
  });
}

export function importPca(path: string): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/assets/import-pca", {
    method: "POST",
    body: JSON.stringify({ path })
  });
}

export function downloadEsm(force = false): Promise<AssetDownloadJob> {
  return request<AssetDownloadJob>("/assets/download-esm/start", {
    method: "POST",
    body: JSON.stringify({ force })
  });
}

export function getEsmDownload(jobId: string, signal?: AbortSignal): Promise<AssetDownloadJob> {
  return request<AssetDownloadJob>(`/asset-download/${jobId}`, { signal });
}

export function cancelEsmDownload(jobId: string): Promise<AssetDownloadJob> {
  return request<AssetDownloadJob>(`/asset-download/${jobId}/cancel`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function runPrediction(payload: {
  input_structure: string;
  output_dir?: string;
  threshold: number;
  pocket_cluster_cutoff: number;
  chain_id?: string;
  allow_truncation: boolean;
}): Promise<PredictResponse> {
  return request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function openResult(summaryJson: string): Promise<PredictResponse> {
  return request<PredictResponse>("/result/open", {
    method: "POST",
    body: JSON.stringify({ summary_json: summaryJson })
  });
}

export function inspectStructure(inputStructure: string, chainId?: string): Promise<StructureInspection> {
  return request<StructureInspection>("/inspect", {
    method: "POST",
    body: JSON.stringify({
      input_structure: inputStructure,
      chain_id: chainId === undefined ? undefined : chainId
    })
  });
}

export function submitBatch(payload: {
  items: Array<{
    input_structure: string;
    chain_id?: string;
  }>;
  output_dir?: string;
  threshold: number;
  pocket_cluster_cutoff: number;
  allow_truncation: boolean;
}): Promise<BatchJob> {
  return request<BatchJob>("/batch", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function getBatch(jobId: string, limit = 500, offset = 0, signal?: AbortSignal): Promise<BatchJob> {
  return request<BatchJob>(`/batch/${jobId}?limit=${limit}&offset=${offset}`, { signal });
}

export function getBatchResult(
  jobId: string,
  inputStructure: string,
  chainId?: string | null
): Promise<BatchResultResponse> {
  const chainQuery = chainId === undefined || chainId === null
    ? ""
    : `&chain_id=${encodeURIComponent(chainId)}`;
  return request<BatchResultResponse>(
    `/batch/${jobId}/result?input_structure=${encodeURIComponent(inputStructure)}${chainQuery}`
  );
}

export function cancelBatch(jobId: string): Promise<BatchJob> {
  return request<BatchJob>(`/batch/${jobId}/cancel`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function retryBatch(jobId: string): Promise<BatchJob> {
  return request<BatchJob>(`/batch/${jobId}/retry`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function exportDiagnostics(path?: string): Promise<{ path: string }> {
  return request<{ path: string }>("/diagnostics/export", {
    method: "POST",
    body: JSON.stringify(path ? { output_zip: path } : {})
  });
}
