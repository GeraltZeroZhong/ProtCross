import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  cancelBatch,
  configureBackend,
  confirmLicense,
  downloadEsm,
  exportDiagnostics,
  getBatch,
  getBatchResult,
  getStatus,
  importCheckpoint,
  importEsm,
  importPca,
  runPrediction,
  configureDesktopApi,
  submitBatch,
  testBackend
} from "./api";
import type { BackendMode, BatchJob, DesktopStatus, PredictResponse, ResidueSummary } from "./types";

type Tab = "setup" | "predict" | "batch" | "results" | "diagnostics";
interface BackendStartResult {
  token: string;
  port: number;
}

const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_CLUSTER_CUTOFF = 8.0;
const ESM_LICENSE_URL = "https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement";
const MolstarViewer = lazy(() =>
  import("./components/MolstarViewer").then((module) => ({ default: module.MolstarViewer }))
);

export default function App() {
  const [tab, setTab] = useState<Tab>("setup");
  const [status, setStatus] = useState<DesktopStatus | null>(null);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [backendMode, setBackendMode] = useState<BackendMode>("cpu");
  const [condaPython, setCondaPython] = useState("");
  const [proxyUrl, setProxyUrl] = useState("");
  const [inputPath, setInputPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [clusterCutoff, setClusterCutoff] = useState(DEFAULT_CLUSTER_CUTOFF);
  const [allowTruncation, setAllowTruncation] = useState(false);
  const [batchInputs, setBatchInputs] = useState<string[]>([]);
  const [batchJob, setBatchJob] = useState<BatchJob | null>(null);
  const [batchResult, setBatchResult] = useState<PredictResponse | null>(null);
  const [selectedBatchInput, setSelectedBatchInput] = useState("");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [envTest, setEnvTest] = useState<Record<string, unknown> | null>(null);
  const [pendingAction, setPendingAction] = useState("");

  function applyStatus(next: DesktopStatus) {
    setStatus(next);
    if (next.backend.mode) {
      setBackendMode(next.backend.mode);
    }
    setProxyUrl(next.backend.proxy_url ?? "");
  }

  async function refresh() {
    const next = await getStatus();
    applyStatus(next);
  }

  async function waitForBackendStatus() {
    let lastError: unknown = null;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      try {
        applyStatus(await getStatus());
        return;
      } catch (exc) {
        lastError = exc;
        await new Promise((resolve) => window.setTimeout(resolve, 250));
      }
    }
    throw lastError;
  }

  async function runAction(action: () => Promise<unknown>, success: string) {
    if (pendingAction) {
      return;
    }
    setError("");
    setMessage("");
    setPendingAction(success);
    try {
      await action();
      setMessage(success);
      await refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setPendingAction("");
    }
  }

  async function restartBackend() {
    if (pendingAction) {
      return;
    }
    setError("");
    setMessage("");
    setPendingAction("Restarting backend...");
    try {
      await invoke("stop_backend");
      const backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
      configureDesktopApi(backend.token, backend.port);
      await waitForBackendStatus();
      setMessage("Desktop backend restarted.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setPendingAction("");
    }
  }

  useEffect(() => {
    async function start() {
      try {
        const backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
        configureDesktopApi(backend.token, backend.port);
        await waitForBackendStatus();
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    }
    void start();
  }, []);

  useEffect(() => {
    if (!batchJob || !["queued", "running"].includes(batchJob.status)) {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        setBatchJob(await getBatch(batchJob.id));
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [batchJob]);

  const setupIssues = useMemo(() => readinessIssues(status), [status]);
  const ready = setupIssues.length === 0;
  const batchActive = batchJob ? ["queued", "running"].includes(batchJob.status) : false;
  const resultBatchItem =
    batchJob?.items.find((item) => item.input_structure === selectedBatchInput)
    ?? batchJob?.items.find((item) => item.output_files?.structure);
  const resultStructure = prediction?.output_files.structure ?? batchResult?.output_files.structure ?? resultBatchItem?.output_files?.structure;
  const resultOutputFiles = prediction?.output_files ?? batchResult?.output_files ?? resultBatchItem?.output_files;
  const resultSummary = prediction?.summary ?? batchResult?.summary ?? null;
  const resultPockets = prediction?.pockets ?? batchResult?.pockets ?? null;
  const resultResidues = prediction?.top_pocket_residues ?? batchResult?.top_pocket_residues ?? [];
  const topResidues = useMemo(() => {
    const residues = resultResidues.length ? resultResidues : ((resultSummary?.top_residues ?? []) as ResidueSummary[]);
    return residues.slice(0, 100);
  }, [resultResidues, resultSummary]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">P</span>
          <div>
            <h1>ProtCross Desktop</h1>
            <p>Binding-site prediction</p>
          </div>
        </div>
        <nav>
          {(["setup", "predict", "batch", "results", "diagnostics"] as Tab[]).map((item) => (
            <button className={tab === item ? "active" : ""} key={item} onClick={() => setTab(item)}>
              {labelForTab(item)}
            </button>
          ))}
        </nav>
        <div className={ready ? "status-ok" : "status-warn"}>
          {ready ? "Predict ready" : "Predict not ready"}
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <strong>{labelForTab(tab)}</strong>
            <span>{status?.backend.mode ? `Backend: ${status.backend.mode}` : "Backend not selected"}</span>
          </div>
          <button onClick={() => refresh().catch((exc) => setError(String(exc)))}>Refresh</button>
        </header>

        {message ? <div className="banner success">{message}</div> : null}
        {error ? <div className="banner error">{error}</div> : null}

        {tab === "setup" ? (
          <SetupPanel
            status={status}
            backendMode={backendMode}
            setBackendMode={setBackendMode}
            busy={Boolean(pendingAction)}
            pendingAction={pendingAction}
            setupIssues={setupIssues}
            condaPython={condaPython}
            setCondaPython={setCondaPython}
            proxyUrl={proxyUrl}
            setProxyUrl={setProxyUrl}
            onConfirmLicense={() => runAction(() => confirmLicense(), "ESM-C license confirmation recorded.")}
            onOpenLicense={() =>
              invoke("open_url", { url: ESM_LICENSE_URL }).catch((exc) =>
                setError(exc instanceof Error ? exc.message : String(exc))
              )
            }
            onConfigureBackend={() =>
              runAction(
                () => configureBackend(backendMode, backendMode === "conda" ? condaPython : undefined, proxyUrl),
                "Backend configuration saved."
              )
            }
            onInstallBackend={(mode) =>
              runAction(
                () => invoke("install_backend", { mode, proxyUrl: proxyUrl || undefined }),
                `${mode.toUpperCase()} backend installed. Save, restart, and run the backend test.`
              )
            }
            onImportEsm={async () => {
              const selected = await open({ multiple: false, filters: [{ name: "ESM-C weights", extensions: ["pth"] }] });
              if (typeof selected === "string") {
                await runAction(() => importEsm(selected), "ESM-C weights imported.");
              }
            }}
            onImportCheckpoint={async () => {
              const selected = await open({ multiple: false, filters: [{ name: "ProtCross checkpoint", extensions: ["ckpt"] }] });
              if (typeof selected === "string") {
                await runAction(() => importCheckpoint(selected), "ProtCross checkpoint imported.");
              }
            }}
            onImportPca={async () => {
              const selected = await open({ multiple: false, filters: [{ name: "ProtCross PCA", extensions: ["pkl"] }] });
              if (typeof selected === "string") {
                await runAction(() => importPca(selected), "ProtCross PCA imported.");
              }
            }}
            onDownloadEsm={() => runAction(() => downloadEsm(false), "ESM-C weights downloaded.")}
            onRefreshEsm={() => runAction(() => downloadEsm(true), "ESM-C weights refreshed.")}
            onRestartBackend={restartBackend}
            onTestBackend={async () => {
              setError("");
              setMessage("");
              try {
                if (backendMode === "conda" && !condaPython) {
                  throw new Error("Choose a conda environment python before testing the conda backend.");
                }
                await configureBackend(backendMode, backendMode === "conda" ? condaPython : undefined, proxyUrl);
                setEnvTest(await testBackend());
                setMessage("Backend test finished.");
                await refresh();
                setTab("diagnostics");
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              }
            }}
          />
        ) : null}

        {tab === "predict" ? (
          <PredictPanel
            ready={ready}
            setupIssues={setupIssues}
            busy={Boolean(pendingAction)}
            inputPath={inputPath}
            setInputPath={setInputPath}
            outputDir={outputDir}
            setOutputDir={setOutputDir}
            threshold={threshold}
            setThreshold={setThreshold}
            clusterCutoff={clusterCutoff}
            setClusterCutoff={setClusterCutoff}
            allowTruncation={allowTruncation}
            setAllowTruncation={setAllowTruncation}
            onRun={async () => {
              if (pendingAction) {
                return;
              }
              setError("");
              setMessage("");
              setPendingAction("Running prediction...");
              try {
                validatePredictionInputs(inputPath, threshold, clusterCutoff);
                const result = await runPrediction({
                  input_structure: inputPath,
                  output_dir: outputDir || undefined,
                  threshold,
                  pocket_cluster_cutoff: clusterCutoff,
                  allow_truncation: allowTruncation
                });
                setPrediction(result);
                setBatchResult(null);
                setSelectedBatchInput("");
                setMessage("Prediction finished.");
                setTab("results");
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              } finally {
                setPendingAction("");
              }
            }}
          />
        ) : null}

        {tab === "batch" ? (
          <BatchPanel
            ready={ready}
            setupIssues={setupIssues}
            busy={Boolean(pendingAction)}
            batchInputs={batchInputs}
            setBatchInputs={setBatchInputs}
            outputDir={outputDir}
            setOutputDir={setOutputDir}
            threshold={threshold}
            setThreshold={setThreshold}
            clusterCutoff={clusterCutoff}
            setClusterCutoff={setClusterCutoff}
            allowTruncation={allowTruncation}
            setAllowTruncation={setAllowTruncation}
            batchJob={batchJob}
            batchActive={batchActive}
            onViewItem={async (item) => {
              if (!batchJob) {
                return;
              }
              setError("");
              try {
                const detail = await getBatchResult(batchJob.id, item.input_structure);
                setBatchResult(detail);
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
                return;
              }
              setSelectedBatchInput(item.input_structure);
              setPrediction(null);
              setTab("results");
            }}
            onSubmit={async () => {
              if (pendingAction || batchActive) {
                return;
              }
              setError("");
              try {
                if (batchInputs.length === 0) {
                  throw new Error("Select at least one structure for batch prediction.");
                }
                validatePredictionInputs(batchInputs[0], threshold, clusterCutoff);
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
                return;
              }
              setPendingAction("Starting batch...");
              try {
                const job = await submitBatch({
                  structures: batchInputs,
                  output_dir: outputDir || undefined,
                  threshold,
                  pocket_cluster_cutoff: clusterCutoff,
                  allow_truncation: allowTruncation
                });
                setBatchJob(job);
                setBatchResult(null);
                setSelectedBatchInput("");
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              } finally {
                setPendingAction("");
              }
            }}
            onCancel={() => batchJob && cancelBatch(batchJob.id).then(setBatchJob).catch((exc) => setError(String(exc)))}
          />
        ) : null}

        {tab === "results" ? (
          <ResultsPanel
            structurePath={resultStructure}
            outputFiles={resultOutputFiles}
            summary={resultSummary}
            pockets={resultPockets}
            residues={topResidues}
          />
        ) : null}

        {tab === "diagnostics" ? (
          <DiagnosticsPanel
            status={status}
            envTest={envTest}
            onTest={async () => {
              setError("");
              setMessage("");
              try {
                setEnvTest(await testBackend());
                setMessage("Environment test finished.");
                await refresh();
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              }
            }}
            onExport={async () => {
              setError("");
              setMessage("");
              try {
                const result = await exportDiagnostics();
                setMessage(`Diagnostic package written: ${result.path}`);
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              }
            }}
            onOpenReleases={() =>
              invoke("open_url", { url: "https://github.com/GeraltZeroZhong/ProtCross/releases" }).catch((exc) =>
                setError(exc instanceof Error ? exc.message : String(exc))
              )
            }
          />
        ) : null}
      </section>
    </main>
  );
}

function SetupPanel(props: {
  status: DesktopStatus | null;
  backendMode: BackendMode;
  setBackendMode: (mode: BackendMode) => void;
  busy: boolean;
  pendingAction: string;
  setupIssues: string[];
  condaPython: string;
  setCondaPython: (value: string) => void;
  proxyUrl: string;
  setProxyUrl: (value: string) => void;
  onConfirmLicense: () => void;
  onOpenLicense: () => void;
  onConfigureBackend: () => void;
  onInstallBackend: (mode: "cpu" | "gpu") => void;
  onImportEsm: () => void;
  onImportCheckpoint: () => void;
  onImportPca: () => void;
  onDownloadEsm: () => void;
  onRefreshEsm: () => void;
  onRestartBackend: () => void;
  onTestBackend: () => void;
}) {
  const licenseConfirmed = Boolean(props.status?.assets.esm.license_confirmed);
  const condaNeedsPath = props.backendMode === "conda" && !props.condaPython;
  const busyLabel = props.pendingAction || "Working...";
  return (
    <div className="grid two">
      {props.busy ? (
        <section className="panel span busy-panel" aria-live="polite">
          <div className="spinner" aria-hidden="true" />
          <div>
            <h2>{busyLabel}</h2>
            <p>Long backend installs can take several minutes. The app is still working.</p>
          </div>
        </section>
      ) : null}
      <section className="panel">
        <h2>ESM-C License</h2>
        <p>ESM-C weights are configured only after license confirmation.</p>
        <label className="checkbox-line">
          <input type="checkbox" checked={licenseConfirmed} readOnly />
          License confirmation recorded
        </label>
        <div className="button-row">
          <button disabled={props.busy} onClick={props.onOpenLicense}>Open ESM-C license</button>
        </div>
        <button disabled={props.busy || licenseConfirmed} onClick={props.onConfirmLicense}>
          I have reviewed and accept the ESM-C license terms
        </button>
      </section>

      <section className="panel">
        <h2>Backend</h2>
        <div className="segmented">
          {(["cpu", "gpu", "conda"] as BackendMode[]).map((mode) => (
            <button
              className={props.backendMode === mode ? "active" : ""}
              disabled={props.busy}
              key={mode}
              onClick={() => props.setBackendMode(mode)}
            >
              {mode === "gpu" ? "GPU / MPS" : mode.toUpperCase()}
            </button>
          ))}
        </div>
        {props.backendMode === "conda" ? (
          <input
            disabled={props.busy}
            value={props.condaPython}
            onChange={(event) => props.setCondaPython(event.target.value)}
            placeholder="Path to conda env python"
          />
        ) : null}
        <input
          disabled={props.busy}
          value={props.proxyUrl}
          onChange={(event) => props.setProxyUrl(event.target.value)}
          placeholder="Optional proxy URL"
        />
        <div className="button-row">
          <button disabled={props.busy} onClick={() => props.onInstallBackend("cpu")}>
            {props.pendingAction.includes("CPU backend") ? props.pendingAction : "Install CPU backend"}
          </button>
          <button disabled={props.busy} onClick={() => props.onInstallBackend("gpu")}>
            {props.pendingAction.includes("GPU") ? props.pendingAction : "Install GPU / MPS backend"}
          </button>
          <button disabled={props.busy || condaNeedsPath} onClick={props.onConfigureBackend}>Save backend</button>
          <button disabled={props.busy || condaNeedsPath} onClick={props.onTestBackend}>Save and test backend</button>
          <button disabled={props.busy} onClick={props.onRestartBackend}>Restart backend</button>
        </div>
      </section>

      <section className="panel span">
        <h2>Setup Status</h2>
        <ReadinessList issues={props.setupIssues} />
      </section>

      <section className="panel span">
        <h2>Assets</h2>
        <div className="asset-grid">
          <AssetLine label="Checkpoint" status={props.status?.assets.checkpoint} />
          <AssetLine label="PCA" status={props.status?.assets.pca} />
          <AssetLine label="ESM-C" status={props.status?.assets.esm} />
        </div>
        <div className="button-row">
          <button disabled={props.busy} onClick={props.onImportCheckpoint}>Import checkpoint</button>
          <button disabled={props.busy} onClick={props.onImportPca}>Import PCA</button>
          <button disabled={props.busy || !licenseConfirmed} onClick={props.onImportEsm}>Import ESM-C .pth</button>
          <button disabled={props.busy || !licenseConfirmed} onClick={props.onDownloadEsm}>Download ESM-C</button>
          <button disabled={props.busy || !licenseConfirmed} onClick={props.onRefreshEsm}>Re-download / verify ESM-C</button>
        </div>
      </section>
    </div>
  );
}

function PredictPanel(props: {
  ready: boolean;
  setupIssues: string[];
  busy: boolean;
  inputPath: string;
  setInputPath: (value: string) => void;
  outputDir: string;
  setOutputDir: (value: string) => void;
  threshold: number;
  setThreshold: (value: number) => void;
  clusterCutoff: number;
  setClusterCutoff: (value: number) => void;
  allowTruncation: boolean;
  setAllowTruncation: (value: boolean) => void;
  onRun: () => void;
}) {
  return (
    <section className="panel">
      <h2>Single Structure Prediction</h2>
      {!props.ready ? <ReadinessList issues={props.setupIssues} /> : null}
      <PathInput label="Input PDB/mmCIF" value={props.inputPath} setValue={props.setInputPath} kind="file" />
      <PathInput label="Output directory" value={props.outputDir} setValue={props.setOutputDir} kind="directory" />
      <NumberInput label="Threshold" value={props.threshold} setValue={props.setThreshold} min={0} max={1} step={0.01} />
      <NumberInput label="Cluster cutoff (A)" value={props.clusterCutoff} setValue={props.setClusterCutoff} min={0.1} max={40} step={0.5} />
      <label className="checkbox-line">
        <input type="checkbox" checked={props.allowTruncation} onChange={(event) => props.setAllowTruncation(event.target.checked)} />
        Allow truncation beyond ESM-C context length
      </label>
      <button disabled={props.busy || !props.ready || !props.inputPath} onClick={props.onRun}>
        {props.busy ? "Running..." : "Run prediction"}
      </button>
    </section>
  );
}

function BatchPanel(props: {
  ready: boolean;
  setupIssues: string[];
  busy: boolean;
  batchInputs: string[];
  setBatchInputs: (value: string[]) => void;
  outputDir: string;
  setOutputDir: (value: string) => void;
  threshold: number;
  setThreshold: (value: number) => void;
  clusterCutoff: number;
  setClusterCutoff: (value: number) => void;
  allowTruncation: boolean;
  setAllowTruncation: (value: boolean) => void;
  batchJob: BatchJob | null;
  batchActive: boolean;
  onViewItem: (item: BatchJob["items"][number]) => void | Promise<void>;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  const progressLabel = props.batchJob
    ? `${props.batchJob.completed}/${props.batchJob.item_count ?? props.batchJob.items.length} processed, ${props.batchJob.failed} failed`
    : "";
  const hiddenCount = props.batchJob ? Math.max(0, (props.batchJob.item_count ?? props.batchJob.items.length) - props.batchJob.items.length) : 0;
  return (
    <section className="panel">
      <h2>Batch Queue</h2>
      {!props.ready ? <ReadinessList issues={props.setupIssues} /> : null}
      <button
        onClick={async () => {
          const selected = await open({ multiple: true, filters: [{ name: "Structures", extensions: ["pdb", "cif", "mmcif"] }] });
          if (Array.isArray(selected)) {
            props.setBatchInputs(selected);
          }
        }}
      >
        Select structures
      </button>
      <p>{props.batchInputs.length} structures selected</p>
      <PathInput label="Output directory" value={props.outputDir} setValue={props.setOutputDir} kind="directory" />
      <NumberInput label="Threshold" value={props.threshold} setValue={props.setThreshold} min={0} max={1} step={0.01} />
      <NumberInput label="Cluster cutoff (A)" value={props.clusterCutoff} setValue={props.setClusterCutoff} min={0.1} max={40} step={0.5} />
      <label className="checkbox-line">
        <input type="checkbox" checked={props.allowTruncation} onChange={(event) => props.setAllowTruncation(event.target.checked)} />
        Allow truncation beyond ESM-C context length
      </label>
      <div className="button-row">
        <button disabled={props.busy || props.batchActive || !props.ready || props.batchInputs.length === 0} onClick={props.onSubmit}>
          {props.batchActive ? "Batch running" : props.busy ? "Starting..." : "Start batch"}
        </button>
        <button
          disabled={!props.batchJob || props.batchJob.cancel_requested || !["queued", "running"].includes(props.batchJob.status)}
          onClick={props.onCancel}
        >
          {props.batchJob?.cancel_requested ? "Stopping after current" : "Stop after current"}
        </button>
      </div>
      {props.batchJob ? (
        <div className="table-wrap">
          <div className="table-header">
            <h3>{props.batchJob.status}</h3>
            <span>{progressLabel}</span>
          </div>
          {props.batchJob.error ? (
            <div className="banner error batch-error">
              {String(props.batchJob.error).split("\n")[0]}
            </div>
          ) : null}
          {props.batchJob.cancel_requested && ["queued", "running"].includes(props.batchJob.status) ? (
            <div className="banner warning batch-error">
              Current prediction will finish; queued structures will not start.
            </div>
          ) : null}
          <table>
            <thead>
              <tr>
                <th scope="col">Status</th>
                <th scope="col">Input</th>
                <th scope="col">Result</th>
                <th scope="col">Actions</th>
              </tr>
            </thead>
            <tbody>
              {props.batchJob.items.map((item) => (
                <tr key={item.input_structure}>
                  <td>{item.status}</td>
                  <td>{item.input_structure}</td>
                  <td>{item.error ?? item.output_dir ?? item.output_files?.summary_json ?? ""}</td>
                  <td>
                    <button
                      disabled={item.status !== "completed" || !item.output_files?.summary_json || props.busy}
                      onClick={() => void props.onViewItem(item)}
                    >
                      View results
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {hiddenCount > 0 ? (
            <p className="muted-note">Showing the first {props.batchJob.items.length} rows; {hiddenCount} more are tracked by the backend.</p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function ResultsPanel(props: {
  structurePath?: string;
  outputFiles?: Record<string, string>;
  summary: any;
  pockets: any;
  residues: ResidueSummary[];
}) {
  const center = props.summary?.top_pocket?.center as number[] | undefined;
  const outputAnchor = props.outputFiles?.summary_json ?? props.outputFiles?.structure ?? props.summary?.output_files?.summary_json;
  const outputDir = outputAnchor
    ? String(outputAnchor).replace(/[\\/][^\\/]+$/, "")
    : undefined;
  return (
    <div className="results-layout">
      <Suspense fallback={<section className="viewer-panel viewer-loading">Loading structure viewer...</section>}>
        <MolstarViewer structurePath={props.structurePath} summary={props.summary} pockets={props.pockets} />
      </Suspense>
      <section className="panel result-panel">
        <h2>Prediction Results</h2>
        {props.summary?.warnings?.length ? (
          <div className="warning-list">
            {props.summary.warnings.map((warning: string) => <div key={warning}>{warning}</div>)}
          </div>
        ) : null}
        {props.summary?.top_pocket ? (
          <div className="metric-row">
            <Metric label="Pocket center" value={props.summary.top_pocket.center.map((v: number) => v.toFixed(3)).join(", ")} />
            <Metric label="Residues" value={String(props.summary.top_pocket.residue_count)} />
            <Metric label="Max probability" value={props.summary.top_pocket.score_max.toFixed(4)} />
          </div>
        ) : (
          <p>No pocket selected at the current threshold.</p>
        )}
        <div className="button-row">
          <button
            disabled={!center}
            onClick={() => center && navigator.clipboard.writeText(center.map((value) => value.toFixed(3)).join(", "))}
          >
            Copy pocket center
          </button>
          <button
            disabled={props.residues.length === 0}
            onClick={() => navigator.clipboard.writeText(formatResidueSelection(props.residues))}
          >
            Copy top pocket residues
          </button>
          <button disabled={!outputDir} onClick={() => outputDir && invoke("open_path", { path: outputDir })}>
            Open output folder
          </button>
        </div>
        {props.outputFiles ? (
          <div className="output-files">
            <h3>Output Files</h3>
            {Object.entries(props.outputFiles).map(([key, value]) => (
              <div className="output-file" key={key}>
                <span>{key}</span>
                <code>{value}</code>
                <button onClick={() => navigator.clipboard.writeText(value)}>Copy path</button>
              </div>
            ))}
          </div>
        ) : null}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Residue</th>
                <th>Probability</th>
                <th>Chain</th>
                <th>Cluster</th>
              </tr>
            </thead>
            <tbody>
              {props.residues.map((residue) => (
                <tr key={`${residue.residue_id}-${residue.cluster_id ?? ""}`}>
                  <td>{residue.residue_id}</td>
                  <td>{Number(residue.probability).toFixed(4)}</td>
                  <td>{String(residue.chain_id ?? "")}</td>
                  <td>{String(residue.cluster_id ?? "")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function DiagnosticsPanel(props: {
  status: DesktopStatus | null;
  envTest: Record<string, unknown> | null;
  onTest: () => void;
  onExport: () => void;
  onOpenReleases: () => void;
}) {
  return (
    <section className="panel">
      <h2>Diagnostics</h2>
      <div className="button-row">
        <button onClick={props.onTest}>Run environment test</button>
        <button onClick={props.onExport}>Export diagnostic package</button>
        <button onClick={props.onOpenReleases}>Check releases</button>
      </div>
      <pre>{JSON.stringify({ status: props.status, envTest: props.envTest }, null, 2)}</pre>
    </section>
  );
}

function AssetLine({ label, status }: { label: string; status?: { present?: boolean; path?: string | null; verified?: boolean | null } }) {
  const present = Boolean(status?.present);
  const verified = status?.verified;
  return (
    <div className={present ? "asset-line ok" : "asset-line missing"}>
      <strong>{label}</strong>
      <span>{assetStateLabel(present, verified)}</span>
      <small>{status?.path}</small>
    </div>
  );
}

function PathInput({ label, value, setValue, kind }: { label: string; value: string; setValue: (value: string) => void; kind: "file" | "directory" }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="path-row">
        <input value={value} onChange={(event) => setValue(event.target.value)} />
        <button
          onClick={async () => {
            const selected = await open({
              multiple: false,
              directory: kind === "directory",
              filters: kind === "file" ? [{ name: "Structures", extensions: ["pdb", "cif", "mmcif"] }] : undefined
            });
            if (typeof selected === "string") {
              setValue(selected);
            }
          }}
        >
          Browse
        </button>
      </div>
    </label>
  );
}

function NumberInput(props: { label: string; value: number; setValue: (value: number) => void; min: number; max: number; step: number }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        type="number"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(event) => props.setValue(Number(event.target.value))}
      />
    </label>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function labelForTab(tab: Tab): string {
  return {
    setup: "Setup",
    predict: "Predict",
    batch: "Batch Queue",
    results: "Results",
    diagnostics: "Diagnostics"
  }[tab];
}

function ReadinessList({ issues }: { issues: string[] }) {
  if (issues.length === 0) {
    return <div className="readiness ready">Ready for prediction.</div>;
  }
  return (
    <div className="readiness">
      <strong>Needs attention</strong>
      <ul>
        {issues.map((issue) => <li key={issue}>{issue}</li>)}
      </ul>
    </div>
  );
}

function readinessIssues(status: DesktopStatus | null): string[] {
  if (!status) {
    return ["Desktop backend is starting."];
  }
  if (status.readiness?.issues) {
    return status.readiness.issues;
  }
  const issues: string[] = [];
  if (!status.assets.esm.license_confirmed) {
    issues.push("Confirm the ESM-C license.");
  }
  if (!status.backend.mode) {
    issues.push("Select and save a backend.");
  } else if (!status.backend.python_present) {
    issues.push("Install the selected backend environment or choose a working conda Python.");
  }
  if (!status.assets.checkpoint.present) {
    issues.push("ProtCross checkpoint is missing from bundled assets.");
  }
  if (!status.assets.pca.present) {
    issues.push("ProtCross PCA asset is missing from bundled assets.");
  }
  if (!status.assets.esm.present) {
    issues.push("Download or import ESM-C weights.");
  } else if (status.assets.esm.verified === false) {
    issues.push("ESM-C weights failed SHA256 verification; repair or import the expected file.");
  }
  return issues;
}

function formatResidueSelection(residues: ResidueSummary[]): string {
  return residues
    .map((residue) => {
      const chain = residue.chain_id ? `${residue.chain_id}:` : "";
      const number = residue.residue_number ?? residue.residue_id;
      return `${chain}${number}`;
    })
    .join(",");
}

function assetStateLabel(present: boolean, verified?: boolean | null): string {
  if (!present) {
    return "Missing";
  }
  if (verified === false) {
    return "Hash mismatch";
  }
  if (verified === true) {
    return "Verified";
  }
  return "Present";
}

function validatePredictionInputs(inputPath: string, threshold: number, clusterCutoff: number) {
  if (!inputPath) {
    throw new Error("Select an input structure first.");
  }
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error("Threshold must be in [0, 1].");
  }
  if (!Number.isFinite(clusterCutoff) || clusterCutoff <= 0) {
    throw new Error("Cluster cutoff must be a positive number.");
  }
}
