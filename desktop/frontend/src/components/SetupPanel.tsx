import { useId } from "react";
import type { OpenDialogOptions } from "@tauri-apps/plugin-dialog";
import type { AssetDownloadJob, BackendMode, DesktopStatus } from "../types";
import { Icon } from "./Icon";
import { AssetInventory, backendIsHealthy, downloadPhaseLabel, EnvironmentStatus, isMacPlatform, runtimeName } from "./EnvironmentDetails";

interface SetupPanelProps {
  onBrowse: (options: OpenDialogOptions) => Promise<string | string[] | null>;
  status: DesktopStatus | null;
  backendMode: BackendMode;
  setBackendMode: (mode: BackendMode) => void;
  busy: boolean;
  pendingAction: string;
  onOpenLogs: () => void;
  assetDownload: AssetDownloadJob | null;
  setupIssues: string[];
  condaPython: string;
  setCondaPython: (value: string) => void;
  proxyUrl: string;
  setProxyUrl: (value: string) => void;
  onPrepare: () => void;
  onOpenSample: () => void;
  onOpenExisting: () => void;
  onInstallBackend: (mode: "cpu" | "gpu") => void;
  onImportEsm: () => void;
  onImportCheckpoint: () => void;
  onImportPca: () => void;
  onDownloadEsm: () => void;
  onRefreshEsm: () => void;
  onCancelEsm: () => void;
  runtimeLocked: boolean;
  backendConnectionLost: boolean;
  onRestartBackend: () => void;
  onContinue: () => void;
  onTestBackend: () => void;
}

export function SetupPanel(props: SetupPanelProps) {
  const condaPythonId = useId();
  const condaNeedsPath = props.backendMode === "conda" && !props.condaPython.trim();
  const downloadActive = ["queued", "running", "cancelling"].includes(props.assetDownload?.status ?? "");
  const backendReady = !props.backendConnectionLost && backendIsHealthy(props.status);
  const assetsReady = Boolean(props.status?.assets.ready);
  const overallReady = !props.backendConnectionLost && props.status?.readiness?.ready === true;
  const locked = props.busy || props.runtimeLocked;
  const connected = Boolean(props.status) && !props.backendConnectionLost;
  const accelerator = isMacPlatform() ? "Apple MPS" : "NVIDIA CUDA";
  const canRestart = !props.busy && (!props.runtimeLocked || props.backendConnectionLost);
  const needsAttention = props.backendConnectionLost || props.setupIssues.some((issue) => /fail|mismatch|restart|error/i.test(issue));

  return <div className="environment-page">
    <div className="workspace-toolbar">
      <EnvironmentStatus label={overallReady ? "Prediction ready" : props.backendConnectionLost ? "Runtime offline" : "Setup required"} tone={overallReady ? "success" : "warning"} />
      <div className="button-row">
        <button onClick={props.onOpenSample} title="Explore the bundled 1CRN result; no model download needed"><Icon name="results" /> Example</button>
        <button disabled={!connected || props.busy} onClick={props.onOpenExisting}><Icon name="folder" /> Open result</button>
        {overallReady
          ? <button className="primary-action" onClick={props.onContinue}>Predict <Icon name="arrow-right" /></button>
          : <button className="primary-action" disabled={locked || downloadActive} onClick={props.onPrepare}><Icon name="download" /> Prepare prediction</button>}
      </div>
    </div>

    {!overallReady ? <p className="environment-note">One-time setup: CPU runtime + 2.14 GiB model weights. Internet required.</p> : null}
    {props.busy ? <div className="environment-operation" role="status">
      <span className="spinner" aria-hidden="true" />
      <span>{props.pendingAction || "Working…"}</span>
      <button onClick={props.onOpenLogs}>Logs</button>
    </div> : null}
    {props.runtimeLocked ? <p className="environment-note" role="status">Runtime changes locked during active jobs. Finish the batch or pause the download.</p> : null}

    <section className="environment-section" aria-labelledby="setup-runtime-title">
      <div className="environment-section-heading">
        <h3 id="setup-runtime-title">Runtime</h3>
        <div className="button-row">
          <EnvironmentStatus label={backendReady ? `${runtimeName(props.status?.backend.mode)} · tested` : props.backendConnectionLost ? "Offline" : props.status?.backend.mode ? `${runtimeName(props.status.backend.mode)} · unverified` : "Not installed"} tone={backendReady ? "success" : "warning"} />
          {props.backendConnectionLost ? <button disabled={!canRestart} onClick={props.onRestartBackend}><Icon name="refresh" /> Reconnect</button> : null}
        </div>
      </div>
      {props.status?.backend.python ? <p className="environment-path"><code>{props.status.backend.python}</code></p> : null}
      <details className="disclosure runtime-options">
        <summary><Icon name="settings" /> Runtime options</summary>
        <div className="disclosure-content">
          <div className="segmented" role="group" aria-label="Runtime mode">
            {(["cpu", "gpu", "conda"] as BackendMode[]).map((mode) => <button
              aria-pressed={props.backendMode === mode}
              className={props.backendMode === mode ? "active" : ""}
              disabled={locked}
              key={mode}
              onClick={() => props.setBackendMode(mode)}
            >{runtimeName(mode)}</button>)}
          </div>
          <div className="environment-grid">
            {props.backendMode === "conda" ? <div className="field path-field">
              <label htmlFor={condaPythonId}>Conda Python</label>
              <div className="path-row">
                <input id={condaPythonId} disabled={locked} value={props.condaPython} onChange={(event) => props.setCondaPython(event.target.value)} placeholder={isMacPlatform() ? "/opt/conda/envs/protcross/bin/python" : "C:\\Miniconda3\\envs\\protcross\\python.exe"} />
                <button disabled={locked} onClick={async () => {
                  const selected = await props.onBrowse({ multiple: false });
                  if (typeof selected === "string") props.setCondaPython(selected);
                }}>Browse…</button>
              </div>
            </div> : null}
            <label className="field">
              <span>Proxy <small>Optional</small></span>
              <input disabled={locked} value={props.proxyUrl} onChange={(event) => props.setProxyUrl(event.target.value)} placeholder="http://proxy.example:8080" />
            </label>
          </div>
          <p className="field-help">{props.backendMode === "gpu" ? `${accelerator} requires compatible hardware${isMacPlatform() ? "." : " and drivers."} ` : ""}The selected runtime is tested before activation.</p>
          <div className="button-row">
            <button className="primary-action" disabled={locked || condaNeedsPath || !connected} onClick={props.onTestBackend}><Icon name="activity" /> Apply and test</button>
            {props.backendMode !== "conda" ? <button disabled={locked} onClick={() => props.onInstallBackend(props.backendMode as "cpu" | "gpu")}><Icon name="download" /> Install {runtimeName(props.backendMode)}</button> : null}
            <button disabled={!canRestart} onClick={props.onRestartBackend}><Icon name="refresh" /> Restart runtime</button>
          </div>
        </div>
      </details>
    </section>

    <section className="environment-section" aria-labelledby="setup-assets-title">
      <div className="environment-section-heading">
        <h3 id="setup-assets-title">Model assets</h3>
        {!assetsReady || downloadActive ? <div className="button-row">
          <button disabled={locked || !connected || downloadActive} onClick={props.onDownloadEsm}>
            <Icon name="download" />{["cancelled", "failed"].includes(props.assetDownload?.status ?? "") ? "Resume ESM-C" : "Download ESM-C · 2.14 GiB"}
          </button>
          {downloadActive ? <button disabled={props.assetDownload?.status === "cancelling"} onClick={props.onCancelEsm}><Icon name="pause" />{props.assetDownload?.status === "cancelling" ? "Pausing…" : "Pause"}</button> : null}
        </div> : <EnvironmentStatus label="Verified" tone="success" />}
      </div>
      <AssetInventory assets={props.status?.assets} />
      {props.assetDownload ? <AssetDownloadProgress job={props.assetDownload} /> : null}
      <details className="disclosure compact-disclosure">
        <summary><Icon name="more" /> Manual assets</summary>
        <div className="disclosure-content">
          <p className="field-help">Release assets only; SHA256 must match. Custom models: use the CLI.</p>
          <div className="button-row">
            <button disabled={locked || !connected} onClick={props.onImportCheckpoint}>Import checkpoint</button>
            <button disabled={locked || !connected} onClick={props.onImportPca}>Import PCA</button>
            <button disabled={locked || !connected} onClick={props.onImportEsm}>Import ESM-C</button>
            <button disabled={locked || !connected} onClick={props.onRefreshEsm}><Icon name="refresh" /> Redownload ESM-C</button>
            {assetsReady ? <button disabled={locked || !connected || downloadActive} onClick={props.onDownloadEsm}>Verify ESM-C</button> : null}
          </div>
        </div>
      </details>
    </section>

    {props.setupIssues.length ? <details className="disclosure readiness-details" open={needsAttention || undefined}>
      <summary><Icon name="warning" /> Readiness details <span>{props.setupIssues.length}</span></summary>
      <ul>{props.setupIssues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
    </details> : null}
    <div className="environment-links"><button onClick={props.onOpenLogs}><Icon name="folder" /> Open runtime logs</button><span>Available offline</span></div>
  </div>;
}

function AssetDownloadProgress({ job }: { job: AssetDownloadJob }) {
  const total = job.total_bytes ?? 0;
  const percent = Number.isFinite(job.percent) ? Number(job.percent) : total ? (100 * job.downloaded_bytes / total) : 0;
  const phase = downloadPhaseLabel(job);
  const transferring = job.status === "running" && phase !== "Verifying ESM-C";
  return <div className="download-progress">
    <span className="sr-only" role="status">{phase}</span>
    <div><strong>{phase}</strong><span>{formatBytes(job.downloaded_bytes)}{total ? ` / ${formatBytes(total)}` : ""}</span>{transferring && job.bytes_per_second ? <span>{formatBytes(job.bytes_per_second)}/s</span> : null}</div>
    <progress aria-label="ESM-C download progress" max={100} value={Math.max(0, Math.min(100, percent))} />
    <span>{percent.toFixed(1)}%{job.status !== "completed" && job.resumable ? " · resumable" : ""}</span>
    {job.error && job.status !== "cancelled" ? <div className="inline-error">{job.error}</div> : null}
  </div>;
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / (1024 ** exponent)).toFixed(exponent >= 3 ? 2 : 1)} ${units[exponent]}`;
}
