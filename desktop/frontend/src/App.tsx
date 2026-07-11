import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import packageInfo from "../package.json";
import {
  cancelBatch,
  cancelEsmDownload,
  configureBackend,
  confirmLicense,
  downloadEsm,
  exportDiagnostics,
  getBatch,
  getBatchResult,
  getEsmDownload,
  getStatus,
  importCheckpoint,
  importEsm,
  importPca,
  inspectStructure,
  runPrediction,
  configureDesktopApi,
  submitBatch,
  testBackend
} from "./api";
import type {
  AssetDownloadJob,
  BackendMode,
  BatchJob,
  DesktopStatus,
  PredictResponse,
  PocketJson,
  ResidueSummary,
  StructureInspection
} from "./types";

type Tab = "setup" | "predict" | "batch" | "results" | "diagnostics";
interface BackendStartResult {
  token: string;
  port: number;
}

const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_CLUSTER_CUTOFF = 8.0;
const BATCH_PAGE_SIZE = 500;
const ESM_LICENSE_URL = "https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement";
const SCIENTIFIC_GUIDE_URL = "https://github.com/GeraltZeroZhong/ProtCross/blob/v0.2.1/README.md#scientific-scope";
const APP_VERSION = packageInfo.version;
const ALL_CHAINS = "__all_chains__";
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
  const [chainSelection, setChainSelection] = useState(ALL_CHAINS);
  const [inspection, setInspection] = useState<StructureInspection | null>(null);
  const [inspectionError, setInspectionError] = useState("");
  const [inspecting, setInspecting] = useState(false);
  const [outputDir, setOutputDir] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [clusterCutoff, setClusterCutoff] = useState(DEFAULT_CLUSTER_CUTOFF);
  const [allowTruncation, setAllowTruncation] = useState(false);
  const [batchInputs, setBatchInputs] = useState<string[]>([]);
  const [batchJob, setBatchJob] = useState<BatchJob | null>(null);
  const [batchPageOffset, setBatchPageOffset] = useState(0);
  const [batchResult, setBatchResult] = useState<PredictResponse | null>(null);
  const [selectedBatchInput, setSelectedBatchInput] = useState("");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [envTest, setEnvTest] = useState<Record<string, unknown> | null>(null);
  const [pendingAction, setPendingAction] = useState("");
  const [assetDownload, setAssetDownload] = useState<AssetDownloadJob | null>(null);

  function applyStatus(next: DesktopStatus) {
    setStatus(next);
    if (next.backend.mode) {
      setBackendMode(next.backend.mode);
    }
    setProxyUrl(next.backend.proxy_url ?? "");
    const activeDownload = [...(next.activity?.asset_downloads ?? [])]
      .reverse()
      .find((job) => ["queued", "running", "cancelling"].includes(job.status));
    if (!assetDownload && activeDownload) {
      setAssetDownload(activeDownload);
    }
    const activeBatch = [...(next.activity?.batch_jobs ?? [])]
      .reverse()
      .find((job) => ["queued", "running"].includes(job.status));
    if (!batchJob && activeBatch) {
      setBatchJob(activeBatch);
      setTab("batch");
    }
  }

  async function refresh() {
    const next = await getStatus();
    applyStatus(next);
  }

  async function waitForBackendStatus() {
    let lastError: unknown = null;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      try {
        const next = await getStatus();
        applyStatus(next);
        return next;
      } catch (exc) {
        lastError = exc;
        await new Promise((resolve) => window.setTimeout(resolve, 250));
      }
    }
    throw lastError;
  }

  async function installAndActivateBackend(mode: "cpu" | "gpu") {
    if (pendingAction) {
      return;
    }
    setError("");
    setMessage("");
    setPendingAction(`Installing ${mode.toUpperCase()} backend...`);
    try {
      await invoke("install_backend", { mode, proxyUrl: proxyUrl || undefined });
      await invoke("stop_backend");
      let backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
      configureDesktopApi(backend.token, backend.port);
      await waitForBackendStatus();
      await configureBackend(mode, undefined, proxyUrl);
      // Restart once more so the sidecar itself runs inside the selected environment.
      await invoke("stop_backend");
      backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
      configureDesktopApi(backend.token, backend.port);
      await waitForBackendStatus();
      const test = await testBackend(mode);
      setEnvTest(test);
      if (test.ok !== true) {
        throw new Error("The backend was installed but its environment test failed. Open Diagnostics for details.");
      }
      await refresh();
      setBackendMode(mode);
      setMessage(`${mode.toUpperCase()} backend installed, activated, and tested.`);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setPendingAction("");
    }
  }

  async function startEsmDownload(force: boolean) {
    if (pendingAction || ["queued", "running", "cancelling"].includes(assetDownload?.status ?? "")) {
      return;
    }
    setError("");
    setMessage("");
    try {
      const job = await downloadEsm(force);
      setAssetDownload(job);
      setMessage("ESM-C download started. Partial data is retained if you pause or lose the connection.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function runAction(action: () => Promise<unknown>, success: string, pending = success) {
    if (pendingAction) {
      return;
    }
    setError("");
    setMessage("");
    setPendingAction(pending);
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
        const next = await waitForBackendStatus();
        if (next?.readiness?.ready) {
          setTab("predict");
        }
      } catch (exc) {
        const detail = exc instanceof Error ? exc.message : String(exc);
        setError(
          "The prediction backend is not running yet. Start with ‘Install recommended CPU backend’ below; " +
          `ProtCross will then start and test it automatically. Details: ${detail}`
        );
      }
    }
    void start();
  }, []);

  useEffect(() => {
    if (!assetDownload || !["queued", "running", "cancelling"].includes(assetDownload.status)) {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const next = await getEsmDownload(assetDownload.id);
        setAssetDownload(next);
        if (next.status === "completed") {
          setMessage("ESM-C weights downloaded and verified.");
          await refresh();
        } else if (next.status === "failed") {
          setError(next.error || "ESM-C download failed. Start it again to resume the partial file.");
        }
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    }, 750);
    return () => window.clearInterval(timer);
  }, [assetDownload?.id, assetDownload?.status]);

  useEffect(() => {
    let cancelled = false;
    setInspectionError("");
    if (!inputPath || !status) {
      setInspecting(false);
      return undefined;
    }
    setInspecting(true);
    const timer = window.setTimeout(async () => {
      try {
        const next = await inspectStructure(
          inputPath,
          chainSelection === ALL_CHAINS ? undefined : chainSelection
        );
        if (!cancelled) {
          setInspection(next);
        }
      } catch (exc) {
        if (!cancelled) {
          setInspectionError(exc instanceof Error ? exc.message : String(exc));
        }
      } finally {
        if (!cancelled) {
          setInspecting(false);
        }
      }
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [inputPath, chainSelection, Boolean(status)]);

  useEffect(() => {
    if (!batchJob || !["queued", "running"].includes(batchJob.status)) {
      return;
    }
    const jobId = batchJob.id;
    const timer = window.setInterval(async () => {
      try {
        const next = await getBatch(jobId, BATCH_PAGE_SIZE, batchPageOffset);
        setBatchJob(next);
        setBatchPageOffset(next.items_offset ?? batchPageOffset);
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [batchJob?.id, batchJob?.status, batchPageOffset]);

  async function loadBatchPage(offset: number) {
    if (!batchJob) {
      return;
    }
    setError("");
    try {
      const next = await getBatch(batchJob.id, BATCH_PAGE_SIZE, Math.max(0, offset));
      setBatchJob(next);
      setBatchPageOffset(next.items_offset ?? Math.max(0, offset));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

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
    return residues;
  }, [resultResidues, resultSummary]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">P</span>
          <div>
            <h1>ProtCross Desktop</h1>
            <p>Binding-site prediction · v{APP_VERSION}</p>
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
            assetDownload={assetDownload}
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
            onInstallBackend={(mode) => void installAndActivateBackend(mode)}
            onImportEsm={async () => {
              const selected = await open({ multiple: false, filters: [{ name: "ESM-C weights", extensions: ["pth"] }] });
              if (typeof selected === "string") {
                await runAction(
                  async () => {
                    const imported = await importEsm(selected);
                    if (imported.verified !== true) {
                      throw new Error("The selected ESM-C file failed SHA256 verification and was not activated.");
                    }
                  },
                  "ESM-C weights imported and verified.",
                  "Copying and verifying the 2.14 GiB ESM-C file…"
                );
              }
            }}
            onImportCheckpoint={async () => {
              const selected = await open({ multiple: false, filters: [{ name: "ProtCross checkpoint", extensions: ["ckpt"] }] });
              if (typeof selected === "string") {
                await runAction(
                  async () => {
                    const imported = await importCheckpoint(selected);
                    if (imported.verified !== true) {
                      throw new Error("The selected checkpoint failed SHA256 verification and was not activated.");
                    }
                  },
                  "ProtCross checkpoint imported and verified.",
                  "Copying and verifying checkpoint…"
                );
              }
            }}
            onImportPca={async () => {
              const selected = await open({ multiple: false, filters: [{ name: "ProtCross PCA", extensions: ["pkl"] }] });
              if (typeof selected === "string") {
                await runAction(
                  async () => {
                    const imported = await importPca(selected);
                    if (imported.verified !== true) {
                      throw new Error("The selected PCA file failed SHA256 verification and was not activated.");
                    }
                  },
                  "ProtCross PCA imported and verified.",
                  "Copying and verifying PCA reducer…"
                );
              }
            }}
            onDownloadEsm={() => void startEsmDownload(false)}
            onRefreshEsm={() => void startEsmDownload(true)}
            onCancelEsm={() =>
              assetDownload && cancelEsmDownload(assetDownload.id).then(setAssetDownload).catch((exc) => setError(String(exc)))
            }
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
            setInputPath={(value) => {
              setInputPath(value);
              setChainSelection(ALL_CHAINS);
              setInspection(null);
              setInspectionError("");
            }}
            chainSelection={chainSelection}
            setChainSelection={setChainSelection}
            inspection={inspection}
            inspectionError={inspectionError}
            inspecting={inspecting}
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
                  chain_id: chainSelection === ALL_CHAINS ? undefined : chainSelection,
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
            batchPageSize={BATCH_PAGE_SIZE}
            batchPageOffset={batchPageOffset}
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
                setBatchPageOffset(job.items_offset ?? 0);
                setBatchResult(null);
                setSelectedBatchInput("");
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              } finally {
                setPendingAction("");
              }
            }}
            onCancel={() => batchJob && cancelBatch(batchJob.id).then(setBatchJob).catch((exc) => setError(String(exc)))}
            onPageChange={(offset) => void loadBatchPage(offset)}
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
            onOpenScientificGuide={() =>
              invoke("open_url", { url: SCIENTIFIC_GUIDE_URL }).catch((exc) =>
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
  assetDownload: AssetDownloadJob | null;
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
  onCancelEsm: () => void;
  onRestartBackend: () => void;
  onTestBackend: () => void;
}) {
  const licenseConfirmed = Boolean(props.status?.assets.esm.license_confirmed);
  const condaNeedsPath = props.backendMode === "conda" && !props.condaPython;
  const busyLabel = props.pendingAction || "Working...";
  const downloadActive = ["queued", "running", "cancelling"].includes(props.assetDownload?.status ?? "");
  return (
    <div className="grid two">
      <section className="panel span setup-guide">
        <h2>First-run setup</h2>
        <ol>
          <li><strong>Install the recommended CPU backend.</strong> It will be activated and tested automatically.</li>
          <li><strong>Review and confirm the Cambrian Non-Commercial License</strong> for ESM-C.</li>
          <li><strong>Download the 2.14 GiB ESM-C weights.</strong> Partial downloads can be resumed.</li>
        </ol>
        <p>Your structures and predictions stay on this computer.</p>
      </section>
      {props.busy ? (
        <section className="panel span busy-panel" aria-live="polite">
          <div className="spinner" aria-hidden="true" />
          <div>
            <h2>{busyLabel}</h2>
            <p>Long backend installs can take several minutes. The app is still working.</p>
          </div>
        </section>
      ) : null}
      <section className="panel setup-backend">
        <h2>1. Prediction backend</h2>
        <div className="segmented">
          {(["cpu", "gpu", "conda"] as BackendMode[]).map((mode) => (
            <button
              className={props.backendMode === mode ? "active" : ""}
              disabled={props.busy}
              key={mode}
              onClick={() => props.setBackendMode(mode)}
            >
              {mode === "gpu" ? "GPU / MPS (advanced)" : mode.toUpperCase()}
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
          <button className="primary-action" disabled={props.busy} onClick={() => props.onInstallBackend("cpu")}>
            {props.pendingAction.includes("CPU backend") ? props.pendingAction : "Install recommended CPU backend"}
          </button>
          <button disabled={props.busy} onClick={() => props.onInstallBackend("gpu")}>
            {props.pendingAction.includes("GPU") ? props.pendingAction : "Install GPU / MPS backend (advanced)"}
          </button>
          <button disabled={props.busy || condaNeedsPath || !props.status} onClick={props.onConfigureBackend}>Save backend</button>
          <button disabled={props.busy || condaNeedsPath || !props.status} onClick={props.onTestBackend}>Save and test backend</button>
          <button disabled={props.busy} onClick={props.onRestartBackend}>Restart backend</button>
        </div>
        <p className="field-help">CPU is the reproducible default. CUDA/MPS requires compatible hardware; Apple MPS is experimental and should be checked against CPU for scientific runs.</p>
      </section>

      <section className="panel setup-license">
        <h2>2. ESM-C License</h2>
        <p>Review the Cambrian Non-Commercial License before downloading or using ESM-C weights.</p>
        <label className="checkbox-line">
          <input type="checkbox" checked={licenseConfirmed} readOnly />
          License confirmation recorded
        </label>
        <div className="button-row">
          <button disabled={props.busy} onClick={props.onOpenLicense}>Open ESM-C license</button>
        </div>
        <button disabled={props.busy || licenseConfirmed || !props.status} onClick={props.onConfirmLicense}>
          I have reviewed and accept the ESM-C license terms
        </button>
        {!props.status ? <p className="muted-note">Install the CPU backend first so confirmation can be recorded.</p> : null}
      </section>

      <section className="panel span setup-status">
        <h2>Setup Status</h2>
        <ReadinessList issues={props.setupIssues} />
      </section>

      <section className="panel span setup-assets">
        <h2>3. Model assets</h2>
        <p>Checkpoint and PCA are bundled. ESM-C needs a separate 2.14 GiB download and about 2.4 GiB free space.</p>
        <div className="asset-grid">
          <AssetLine label="Checkpoint" status={props.status?.assets.checkpoint} />
          <AssetLine label="PCA" status={props.status?.assets.pca} />
          <AssetLine label="ESM-C" status={props.status?.assets.esm} />
        </div>
        <div className="button-row">
          <button disabled={props.busy || !props.status} onClick={props.onImportCheckpoint}>Import checkpoint</button>
          <button disabled={props.busy || !props.status} onClick={props.onImportPca}>Import PCA</button>
          <button disabled={props.busy || !licenseConfirmed || downloadActive} onClick={props.onImportEsm}>Import ESM-C .pth</button>
          <button className="primary-action" disabled={props.busy || !licenseConfirmed || downloadActive} onClick={props.onDownloadEsm}>
            {props.assetDownload?.status === "cancelled" ? "Resume ESM-C download" : "Download ESM-C (2.14 GiB)"}
          </button>
          <button disabled={props.busy || !licenseConfirmed || downloadActive} onClick={props.onRefreshEsm}>Restart download / verify</button>
          <button disabled={!downloadActive || props.assetDownload?.status === "cancelling"} onClick={props.onCancelEsm}>
            {props.assetDownload?.status === "cancelling" ? "Pausing..." : "Pause download"}
          </button>
        </div>
        {props.assetDownload ? <AssetDownloadProgress job={props.assetDownload} /> : null}
      </section>
    </div>
  );
}

function AssetDownloadProgress({ job }: { job: AssetDownloadJob }) {
  const total = job.total_bytes ?? 0;
  const percent = Number.isFinite(job.percent)
    ? Number(job.percent)
    : total ? (100 * job.downloaded_bytes / total) : 0;
  return (
    <div className="download-progress" aria-live="polite">
      <div>
        <strong>{downloadStatusLabel(job.status)}</strong>
        <span>{formatBytes(job.downloaded_bytes)} / {total ? formatBytes(total) : "unknown size"}</span>
        {job.bytes_per_second ? <span>{formatBytes(job.bytes_per_second)}/s</span> : null}
      </div>
      <progress max={100} value={Math.max(0, Math.min(100, percent))} />
      <span>{percent.toFixed(1)}% · partial data is retained for resume</span>
      {job.error && job.status !== "cancelled" ? <div className="inline-error">{job.error}</div> : null}
    </div>
  );
}

function PredictPanel(props: {
  ready: boolean;
  setupIssues: string[];
  busy: boolean;
  inputPath: string;
  setInputPath: (value: string) => void;
  chainSelection: string;
  setChainSelection: (value: string) => void;
  inspection: StructureInspection | null;
  inspectionError: string;
  inspecting: boolean;
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
  const truncationBlocked = Boolean(props.inspection?.requires_truncation && !props.allowTruncation);
  return (
    <section className="panel">
      <h2>Single Structure Prediction</h2>
      {!props.ready ? <ReadinessList issues={props.setupIssues} /> : null}
      <PathInput label="Input PDB/mmCIF" value={props.inputPath} setValue={props.setInputPath} kind="file" />
      <StructureInspectionCard
        inspection={props.inspection}
        error={props.inspectionError}
        inspecting={props.inspecting}
        chainSelection={props.chainSelection}
        setChainSelection={props.setChainSelection}
      />
      <PathInput label="Output directory" value={props.outputDir} setValue={props.setOutputDir} kind="directory" />
      <NumberInput label="Model-score cutoff" value={props.threshold} setValue={props.setThreshold} min={0} max={1} step={0.01} />
      <p className="field-help">Softmax class scores are not calibrated probabilities. The default 0.5 is a neutral class-decision cutoff.</p>
      <NumberInput label="Predicted-residue cluster cutoff (Å)" value={props.clusterCutoff} setValue={props.setClusterCutoff} min={0.1} max={40} step={0.5} />
      <label className="checkbox-line">
        <input type="checkbox" checked={props.allowTruncation} onChange={(event) => props.setAllowTruncation(event.target.checked)} />
        Allow truncation beyond ESM-C context length
      </label>
      {truncationBlocked ? (
        <div className="inline-error">The selected chain exceeds the ESM-C context. Enable truncation or choose a shorter chain.</div>
      ) : null}
      <button
        disabled={
          props.busy || !props.ready || !props.inputPath || props.inspecting ||
          Boolean(props.inspectionError) || !props.inspection || truncationBlocked
        }
        onClick={props.onRun}
      >
        {props.busy ? "Running..." : "Run prediction"}
      </button>
    </section>
  );
}

function StructureInspectionCard(props: {
  inspection: StructureInspection | null;
  error: string;
  inspecting: boolean;
  chainSelection: string;
  setChainSelection: (value: string) => void;
}) {
  if (props.inspecting) {
    return <div className="structure-check" aria-live="polite"><strong>Checking structure…</strong></div>;
  }
  if (props.error && !props.inspection) {
    return <div className="inline-error structure-check"><strong>Structure check failed:</strong> {props.error}</div>;
  }
  if (!props.inspection) {
    return <p className="field-help">Choose a structure to check chains and coordinate quality before model loading.</p>;
  }
  const report = props.inspection;
  return (
    <div className="structure-check" aria-live="polite">
      <div className="structure-check-heading">
        <div>
          <strong>Structure check passed</strong>
          <span>{report.format}; first of {report.model_count} coordinate model(s)</span>
        </div>
        <label>
          <span>Analyze</span>
          <select value={props.chainSelection} onChange={(event) => props.setChainSelection(event.target.value)}>
            <option value={ALL_CHAINS}>All scorable chains</option>
            {report.available_chains.map((chain) => (
              <option value={chain} key={chain || "blank-chain"}>Chain {displayChain(chain)}</option>
            ))}
          </select>
        </label>
      </div>
      {props.error ? <div className="inline-error"><strong>Chain check failed:</strong> {props.error}</div> : null}
      <div className="inspection-metrics">
        <Metric label="Scorable residues" value={String(report.scorable_residue_count)} />
        <Metric label="Missing Cα" value={String(report.standard_residues_missing_ca)} />
        <Metric label="Modified / non-standard" value={String(report.modified_or_nonstandard_amino_acids)} />
        <Metric label="Coordinate breaks" value={String(report.sequence_break_count)} />
        <Metric label="Numbering gaps" value={String(report.numbering_gap_count)} />
      </div>
      {report.warnings.length ? (
        <div className="warning-list compact">
          {report.warnings.map((warning) => <div key={warning}>{warning}</div>)}
        </div>
      ) : <p className="inspection-ok">No compatibility warning detected.</p>}
      <p className="field-help">
        ProtCross uses the supplied coordinates, first model only, and standard residues with Cα atoms. It does not generate a biological assembly; all chosen chains share one geometry graph.
      </p>
    </div>
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
  batchPageSize: number;
  batchPageOffset: number;
  onViewItem: (item: BatchJob["items"][number]) => void | Promise<void>;
  onSubmit: () => void;
  onCancel: () => void;
  onPageChange: (offset: number) => void;
}) {
  const progressLabel = props.batchJob
    ? `${props.batchJob.completed}/${props.batchJob.item_count ?? props.batchJob.items.length} processed, ${props.batchJob.failed} failed`
    : "";
  const pageOffset = props.batchJob?.items_offset ?? props.batchPageOffset;
  const pageReturned = props.batchJob?.items_returned ?? props.batchJob?.items.length ?? 0;
  const itemCount = props.batchJob?.item_count ?? props.batchJob?.items.length ?? 0;
  const pageStart = itemCount === 0 ? 0 : pageOffset + 1;
  const pageEnd = Math.min(itemCount, pageOffset + pageReturned);
  const canPrevious = Boolean(props.batchJob && pageOffset > 0);
  const canNext = Boolean(props.batchJob && pageOffset + pageReturned < itemCount);
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
            <span>{progressLabel}; showing {pageStart}-{pageEnd} of {itemCount}</span>
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
          <div className="pager">
            <button disabled={!canPrevious || props.busy} onClick={() => props.onPageChange(Math.max(0, pageOffset - props.batchPageSize))}>
              Previous
            </button>
            <span>{pageStart}-{pageEnd} / {itemCount}</span>
            <button disabled={!canNext || props.busy} onClick={() => props.onPageChange(pageOffset + props.batchPageSize)}>
              Next
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function ResultsPanel(props: {
  structurePath?: string;
  outputFiles?: Record<string, string>;
  summary: any;
  pockets: PocketJson | null;
  residues: ResidueSummary[];
}) {
  const [selectedClusterIndex, setSelectedClusterIndex] = useState(0);
  const clusters = props.pockets?.clustered_pockets ?? [];
  const selectedCluster = clusters[selectedClusterIndex] ?? null;
  const displayedPocket = selectedCluster ?? props.summary?.top_pocket ?? null;
  const displayedResidues = selectedCluster?.residues ?? props.residues;
  const center = displayedPocket?.center as number[] | undefined;
  useEffect(() => setSelectedClusterIndex(0), [props.pockets]);
  const outputAnchor = props.outputFiles?.summary_json ?? props.outputFiles?.structure ?? props.summary?.output_files?.summary_json;
  const outputDir = outputAnchor
    ? String(outputAnchor).replace(/[\\/][^\\/]+$/, "")
    : undefined;
  if (!props.summary && !props.pockets && !props.outputFiles) {
    return (
      <section className="panel empty-results">
        <h2>No prediction loaded</h2>
        <p>Run a single-structure prediction or open a completed batch item to view model scores and predicted-residue clusters.</p>
      </section>
    );
  }
  return (
    <div className="results-layout">
      <Suspense fallback={<section className="viewer-panel viewer-loading">Loading structure viewer...</section>}>
        <MolstarViewer
          structurePath={props.structurePath}
          summary={props.summary}
          pockets={props.pockets}
          selectedClusterIndex={selectedClusterIndex}
        />
      </Suspense>
      <section className="panel result-panel">
        <h2>Binding-site model scores</h2>
        <div className="science-note">
          Scores are uncalibrated softmax class scores, not binding probabilities, affinities, or confidence estimates.
          Predicted-residue clusters are geometric post-processing, not detected cavities.
          <button onClick={() => invoke("open_url", { url: SCIENTIFIC_GUIDE_URL })}>Read scientific guidance</button>
        </div>
        <p className="result-provenance">
          ProtCross {String(props.summary?.protcross_version ?? APP_VERSION)} · model assets {String(props.summary?.asset_version ?? "unknown")} · geometry {String(props.summary?.geometry_backend ?? "unknown")}
        </p>
        {props.summary?.warnings?.length ? (
          <div className="warning-list">
            {props.summary.warnings.map((warning: string) => <div key={warning}>{warning}</div>)}
          </div>
        ) : null}
        {displayedPocket ? (
          <div className="metric-row">
            <Metric label="Score-weighted Cα centroid (Å)" value={displayedPocket.center.map((v: number) => v.toFixed(3)).join(", ")} />
            <Metric label="Residues above cutoff" value={String(displayedPocket.residue_count)} />
            <Metric label="Maximum model score" value={displayedPocket.score_max.toFixed(4)} />
          </div>
        ) : (
          <p>No residue is above the current model-score cutoff; no cluster was formed.</p>
        )}
        {clusters.length ? (
          <label className="field">
            <span>Displayed predicted-residue cluster</span>
            <select value={selectedClusterIndex} onChange={(event) => setSelectedClusterIndex(Number(event.target.value))}>
              {clusters.map((cluster, index) => (
                <option value={index} key={cluster.cluster_id ?? index}>
                  Cluster {cluster.cluster_id ?? index + 1}: {cluster.residue_count} residues; max score {cluster.score_max.toFixed(3)}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <div className="button-row">
          <button
            disabled={!center}
            onClick={() => center && navigator.clipboard.writeText(center.map((value) => value.toFixed(3)).join(", "))}
          >
            Copy score-weighted Cα centroid
          </button>
          <button
            disabled={displayedResidues.length === 0}
            onClick={() => navigator.clipboard.writeText(formatResidueSelection(displayedResidues))}
          >
            Copy selected cluster residues
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
                <th>Model score</th>
                <th>Chain</th>
                <th>Cluster</th>
              </tr>
            </thead>
            <tbody>
              {displayedResidues.map((residue) => (
                <tr key={`${residue.residue_id}-${residue.cluster_id ?? ""}`}>
                  <td>{residue.residue_id}</td>
                  <td>{Number(residue.score ?? residue.probability).toFixed(4)}</td>
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
  onOpenScientificGuide: () => void;
}) {
  return (
    <section className="panel">
      <h2>Diagnostics</h2>
      <div className="button-row">
        <button onClick={props.onTest}>Run environment test</button>
        <button onClick={props.onExport}>Export diagnostic package</button>
        <button onClick={props.onOpenReleases}>Check releases</button>
        <button onClick={props.onOpenScientificGuide}>Open scientific guidance</button>
      </div>
      <p>Desktop v{APP_VERSION}; model asset version is shown in prediction summaries below.</p>
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
      const chainValue = residue.auth_asym_id ?? residue.chain_id ?? "";
      const chain = String(chainValue).trim() || "<blank>";
      const number = residue.auth_seq_id ?? residue.residue_number ?? residue.residue_id;
      const numberText = String(number);
      const insertionCode = String(residue.insertion_code ?? "").trim();
      const suffix = insertionCode && !numberText.endsWith(insertionCode) ? insertionCode : "";
      return `${chain}:${numberText}${suffix}`;
    })
    .join(",");
}

function displayChain(chainId: string): string {
  return chainId.trim() || "<blank>";
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

function downloadStatusLabel(status: AssetDownloadJob["status"]): string {
  return {
    queued: "Preparing download",
    running: "Downloading ESM-C",
    cancelling: "Pausing after the current chunk",
    cancelled: "Download paused",
    failed: "Download interrupted — start again to resume",
    completed: "Download complete and verified"
  }[status];
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KiB", "MiB", "GiB"];
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / (1024 ** exponent)).toFixed(exponent >= 3 ? 2 : 1)} ${units[exponent]}`;
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
