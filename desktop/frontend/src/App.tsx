import { Suspense, lazy, useEffect, useId, useMemo, useRef, useState } from "react";
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
  openResult,
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
import { Icon, type IconName } from "./components/Icon";

type Tab = "setup" | "predict" | "batch" | "results" | "diagnostics";
type ThemePreference = "system" | "light" | "dark";
interface BackendStartResult {
  token: string;
  port: number;
}

const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_CLUSTER_CUTOFF = 8.0;
const BATCH_PAGE_SIZE = 500;
const ESM_LICENSE_URL = "https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement";
const TECHNICAL_GUIDE_URL = "https://github.com/GeraltZeroZhong/ProtCross/blob/v0.2.2/README.md#model-and-inference-pipeline";
const APP_VERSION = packageInfo.version;
const ALL_CHAINS = "__all_chains__";
const MolstarViewer = lazy(() =>
  import("./components/MolstarViewer").then((module) => ({ default: module.MolstarViewer }))
);

const NAV_ITEMS: Array<{ id: Tab; label: string; description: string; icon: IconName }> = [
  { id: "setup", label: "Setup", description: "Runtime and assets", icon: "setup" },
  { id: "predict", label: "Predict", description: "One structure", icon: "predict" },
  { id: "batch", label: "Batch", description: "Multiple structures", icon: "batch" },
  { id: "results", label: "Results", description: "Inspect predictions", icon: "results" },
  { id: "diagnostics", label: "Diagnostics", description: "Health and support", icon: "activity" }
];

const PAGE_DESCRIPTIONS: Record<Tab, string> = {
  setup: "Prepare the local prediction runtime and model assets.",
  predict: "Inspect a structure, configure inference, and predict binding-site residues.",
  batch: "Run a managed queue with shared settings and isolated per-structure results.",
  results: "Explore predicted residue clusters, coordinates, and exported files.",
  diagnostics: "Review runtime health, paths, versions, and support information."
};
const UI_PREVIEW_TAB = import.meta.env.DEV
  ? parsePreviewTab(new URLSearchParams(window.location.search).get("preview"))
  : null;

export default function App() {
  const [tab, setTab] = useState<Tab>(UI_PREVIEW_TAB ?? "setup");
  const previousTab = useRef<Tab>(tab);
  const [themePreference, setThemePreference] = useState<ThemePreference>(() => {
    const saved = window.localStorage.getItem("protcross-theme");
    return saved === "light" || saved === "dark" ? saved : "system";
  });
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
  const [outputDir, setOutputDir] = useState(() => window.localStorage.getItem("protcross-output-dir") ?? "");
  const [threshold, setThreshold] = useState(() => storedNumber("protcross-threshold", DEFAULT_THRESHOLD));
  const [clusterCutoff, setClusterCutoff] = useState(() => storedNumber("protcross-cluster-cutoff", DEFAULT_CLUSTER_CUTOFF));
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
  const [backendConnectionLost, setBackendConnectionLost] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = themePreference;
    window.localStorage.setItem("protcross-theme", themePreference);
  }, [themePreference]);

  useEffect(() => {
    window.localStorage.setItem("protcross-output-dir", outputDir);
  }, [outputDir]);

  useEffect(() => {
    window.localStorage.setItem("protcross-threshold", String(threshold));
    window.localStorage.setItem("protcross-cluster-cutoff", String(clusterCutoff));
  }, [threshold, clusterCutoff]);

  useEffect(() => {
    if (previousTab.current !== tab) {
      window.requestAnimationFrame(() => document.getElementById("workspace-content")?.focus());
      previousTab.current = tab;
    }
  }, [tab]);

  function applyStatus(next: DesktopStatus) {
    setStatus(next);
    setBackendConnectionLost(false);
    if (next.backend.mode) {
      setBackendMode(next.backend.mode);
    }
    setProxyUrl(next.backend.proxy_url ?? "");
    const downloads = next.activity?.asset_downloads ?? [];
    const activeDownload = [...downloads]
      .reverse()
      .find((job) => ["queued", "running", "cancelling"].includes(job.status));
    setAssetDownload((current) => {
      if (activeDownload) {
        return activeDownload;
      }
      if (current && ["queued", "running", "cancelling"].includes(current.status)) {
        return downloads.find((job) => job.id === current.id) ?? {
          ...current,
          status: "failed",
          error: "The backend restarted before this download status could be recovered. Start again to resume retained partial data."
        };
      }
      return current;
    });
    const recentBatches = [...(next.activity?.batch_jobs ?? [])].reverse();
    const activeBatch = recentBatches.find((job) => ["queued", "running"].includes(job.status));
    const recentBatch = activeBatch ?? recentBatches[0];
    setBatchJob((current) => {
      if (activeBatch) {
        return activeBatch;
      }
      if (current && ["queued", "running"].includes(current.status)) {
        return recentBatches.find((job) => job.id === current.id) ?? {
          ...current,
          status: "interrupted",
          error: "The backend restarted and no longer has this in-memory batch job. Completed output files remain on disk."
        };
      }
      return current ?? recentBatch ?? null;
    });
    if (!batchJob && activeBatch) {
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
        const next = await withRequestDeadline((signal) => getStatus(signal), 2_000);
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

  async function openExistingResult() {
    if (pendingAction) {
      return;
    }
    const selected = await open({
      multiple: false,
      filters: [{ name: "ProtCross summary", extensions: ["json"] }]
    });
    if (typeof selected !== "string") {
      return;
    }
    setError("");
    setMessage("");
    setPendingAction("Opening result package…");
    try {
      const result = await openResult(selected);
      setPrediction(result);
      setBatchResult(null);
      setSelectedBatchInput("");
      setMessage(`Opened ${fileName(selected)}.`);
      setTab("results");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setPendingAction("");
    }
  }

  useEffect(() => {
    async function start() {
      if (UI_PREVIEW_TAB) {
        const preview = previewState(UI_PREVIEW_TAB);
        applyStatus(preview.status);
        if (preview.prediction) {
          setPrediction(preview.prediction);
        }
        if (preview.batchJob) {
          setBatchJob(preview.batchJob);
          setBatchInputs(preview.batchJob.items.map((item) => item.input_structure));
        }
        if (UI_PREVIEW_TAB === "diagnostics") {
          setEnvTest({ ok: true, device: "cpu", checks: { protcross: "0.2.2", torch: "2.3.1" } });
        }
        return;
      }
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
    let cancelled = false;
    let inFlight = false;
    let consecutiveFailures = 0;
    const jobId = assetDownload.id;
    const timer = window.setInterval(async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const next = await withRequestDeadline((signal) => getEsmDownload(jobId, signal));
        if (cancelled) {
          return;
        }
        consecutiveFailures = 0;
        setBackendConnectionLost(false);
        setAssetDownload(next);
        if (next.status === "completed") {
          setMessage("ESM-C weights downloaded and verified.");
          await refresh();
        } else if (next.status === "failed") {
          setError(next.error || "ESM-C download failed. Start it again to resume the partial file.");
        }
      } catch (exc) {
        if (cancelled) {
          return;
        }
        consecutiveFailures += 1;
        if (consecutiveFailures >= 3) {
          const detail = exc instanceof Error ? exc.message : String(exc);
          setBackendConnectionLost(true);
          setAssetDownload((current) => current?.id === jobId && ["queued", "running", "cancelling"].includes(current.status)
            ? {
                ...current,
                status: "failed",
                error: "Connection to the backend was lost. Restart the runtime, then start again to resume retained partial data."
              }
            : current);
          setError(`Lost connection to the desktop backend while monitoring the download: ${detail}`);
          window.clearInterval(timer);
        }
      } finally {
        inFlight = false;
      }
    }, 750);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
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
    let cancelled = false;
    let inFlight = false;
    let consecutiveFailures = 0;
    const timer = window.setInterval(async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const next = await withRequestDeadline(
          (signal) => getBatch(jobId, BATCH_PAGE_SIZE, batchPageOffset, signal)
        );
        if (cancelled) {
          return;
        }
        consecutiveFailures = 0;
        setBackendConnectionLost(false);
        setBatchJob(next);
        setBatchPageOffset(next.items_offset ?? batchPageOffset);
      } catch (exc) {
        if (cancelled) {
          return;
        }
        consecutiveFailures += 1;
        if (consecutiveFailures >= 3) {
          const detail = exc instanceof Error ? exc.message : String(exc);
          setBackendConnectionLost(true);
          setBatchJob((current) => current?.id === jobId && ["queued", "running"].includes(current.status)
            ? {
                ...current,
                status: "interrupted",
                error: "Connection to the backend was lost. Restart the runtime; completed output files remain on disk."
              }
            : current);
          setError(`Lost connection to the desktop backend while monitoring the batch: ${detail}`);
          window.clearInterval(timer);
        }
      } finally {
        inFlight = false;
      }
    }, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
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
  const downloadActive = ["queued", "running", "cancelling"].includes(assetDownload?.status ?? "");
  const activityLabel = pendingAction
    || (downloadActive && assetDownload ? downloadStatusLabel(assetDownload.status) : "")
    || (batchActive && batchJob ? `Batch prediction · ${batchJob.completed}/${batchJob.item_count ?? batchJob.items.length}` : "");

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace-content">Skip to content</a>
      <aside className="sidebar" aria-label="ProtCross workspace navigation">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <div>
            <h1>ProtCross</h1>
            <p>Desktop · v{APP_VERSION}</p>
          </div>
        </div>
        <nav className="primary-nav" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => {
            const selected = tab === item.id;
            const badge = item.id === "batch" && batchActive
              ? `${batchJob?.completed ?? 0}/${batchJob?.item_count ?? batchJob?.items.length ?? 0}`
              : item.id === "setup" && !ready ? String(setupIssues.length) : "";
            return (
              <button
                aria-current={selected ? "page" : undefined}
                className={selected ? "active" : ""}
                key={item.id}
                onClick={() => setTab(item.id)}
                title={item.label}
              >
                <Icon name={item.icon} />
                <span className="nav-copy"><strong>{item.label}</strong><small>{item.description}</small></span>
                <span className="nav-compact-label">{item.id === "diagnostics" ? "Health" : item.label}</span>
                {badge ? <span className="nav-badge">{badge}</span> : <Icon className="nav-chevron" name="chevron-right" size={15} />}
              </button>
            );
          })}
        </nav>
        <button
          className={`readiness-card ${ready ? "ready" : "attention"}`}
          onClick={() => setTab(ready ? "predict" : "setup")}
        >
          <span className="status-dot" aria-hidden="true" />
          <span>
            <strong>{ready ? "Ready to predict" : "Setup required"}</strong>
            <small>{ready ? backendDisplayName(status?.backend.mode) : `${setupIssues.length} item${setupIssues.length === 1 ? "" : "s"} need attention`}</small>
          </span>
          <Icon name="chevron-right" size={15} />
        </button>
        <div className="sidebar-footnote">Local inference · Your structures stay on this computer</div>
      </aside>

      <section className="workspace">
        <header className="workspace-header">
          <div className="page-heading">
            <span className="eyebrow">ProtCross workspace</span>
            <h2>{labelForTab(tab)}</h2>
            <p>{PAGE_DESCRIPTIONS[tab]}</p>
          </div>
          <div className="header-actions">
            <span className={`backend-chip ${ready ? "ready" : "idle"}`}>
              <span className="status-dot" aria-hidden="true" />
              {status?.backend.mode ? backendDisplayName(status.backend.mode) : "Backend unavailable"}
            </span>
            <label className="theme-control" title="Appearance">
              <Icon name={themePreference === "dark" ? "moon" : themePreference === "light" ? "sun" : "monitor"} />
              <select aria-label="Appearance" value={themePreference} onChange={(event) => setThemePreference(event.target.value as ThemePreference)}>
                <option value="system">System</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </label>
            <button
              aria-label="Refresh runtime status"
              className="icon-button"
              onClick={() => refresh().catch((exc) => setError(String(exc)))}
              title="Refresh runtime status"
            >
              <Icon name="refresh" />
            </button>
          </div>
        </header>

        {activityLabel ? (
          <div className="activity-strip" role="status" aria-live="polite">
            <div className="activity-indicator"><span /><span /><span /></div>
            <div><strong>{activityLabel}</strong><span>Processing continues while you move through the workspace.</span></div>
            {batchActive && tab !== "batch" ? <button onClick={() => setTab("batch")}>View batch</button> : null}
          </div>
        ) : null}

        <div className="notification-region">
          {message ? (
            <div className="banner success" role="status">
              <Icon name="check" />
              <span>{message}</span>
              <button aria-label="Dismiss message" className="banner-close" onClick={() => setMessage("")}><Icon name="close" size={16} /></button>
            </div>
          ) : null}
          {error ? (
            <div className="banner error" role="alert">
              <Icon name="warning" />
              <span>{error}</span>
              <button aria-label="Dismiss error" className="banner-close" onClick={() => setError("")}><Icon name="close" size={16} /></button>
            </div>
          ) : null}
        </div>

        <main className={`content content-${tab}`} id="workspace-content" tabIndex={-1} aria-busy={Boolean(pendingAction)}>

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
            runtimeLocked={batchActive || downloadActive}
            backendConnectionLost={backendConnectionLost}
            onRestartBackend={restartBackend}
            onContinue={() => setTab("predict")}
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
            defaultOutputRoot={status?.paths.outputs_dir}
            threshold={threshold}
            setThreshold={setThreshold}
            clusterCutoff={clusterCutoff}
            setClusterCutoff={setClusterCutoff}
            allowTruncation={allowTruncation}
            setAllowTruncation={setAllowTruncation}
            onOpenSetup={() => setTab("setup")}
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
            defaultOutputRoot={status?.paths.outputs_dir}
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
            darkMode={themePreference === "dark" || (themePreference === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)}
            onOpenExisting={() => void openExistingResult()}
            onNotify={setMessage}
            onError={setError}
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
              invoke("open_url", { url: TECHNICAL_GUIDE_URL }).catch((exc) =>
                setError(exc instanceof Error ? exc.message : String(exc))
              )
            }
          />
        ) : null}
      </main>
      </section>
    </div>
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
  runtimeLocked: boolean;
  backendConnectionLost: boolean;
  onRestartBackend: () => void;
  onContinue: () => void;
  onTestBackend: () => void;
}) {
  const condaPythonId = useId();
  const licenseConfirmed = Boolean(props.status?.assets.esm.license_confirmed);
  const condaNeedsPath = props.backendMode === "conda" && !props.condaPython;
  const busyLabel = props.pendingAction || "Working...";
  const downloadActive = ["queued", "running", "cancelling"].includes(props.assetDownload?.status ?? "");
  const backendReady = backendIsHealthy(props.status);
  const assetsReady = Boolean(props.status?.assets.ready);
  const completeCount = [backendReady, licenseConfirmed, assetsReady].filter(Boolean).length;
  const overallReady = props.status?.readiness?.ready === true;
  const locked = props.busy || props.runtimeLocked;
  const acceleratorLabel = isMacPlatform() ? "Apple silicon acceleration" : "NVIDIA CUDA acceleration";
  return (
    <div className="setup-layout">
      <section className={`setup-overview ${overallReady ? "complete" : ""}`}>
        <div className="setup-overview-copy">
          <span className="eyebrow">Environment readiness</span>
          <h3>{overallReady ? "ProtCross is ready" : `${completeCount} of 3 steps complete`}</h3>
          <p>{overallReady
            ? "The runtime and assets have passed their readiness checks."
            : "Complete the guided setup once. ProtCross reuses this local environment for future sessions."}</p>
          <div className="setup-progress" aria-label="Setup progress" aria-valuemax={3} aria-valuemin={0} aria-valuenow={completeCount} role="progressbar">
            {[0, 1, 2].map((step) => <span className={step < completeCount ? "complete" : ""} key={step} />)}
          </div>
        </div>
        {overallReady ? (
          <button className="primary-action" onClick={props.onContinue}>Start a prediction <Icon name="arrow-right" /></button>
        ) : (
          <div className="local-badge"><Icon name="setup" /><span><strong>Local workspace</strong><small>Structures remain on this computer</small></span></div>
        )}
      </section>
      {props.busy ? (
        <section className="panel busy-panel" aria-live="polite">
          <div className="spinner" aria-hidden="true" />
          <div>
            <h3>{busyLabel}</h3>
            <p>This operation can take several minutes. You can continue viewing other workspace pages.</p>
          </div>
        </section>
      ) : null}
      {props.runtimeLocked ? (
        <div className="callout warning"><Icon name="warning" /><div><strong>Environment controls are locked</strong><span>Finish the active batch or pause the asset download before changing its runtime.</span></div></div>
      ) : null}

      <section aria-labelledby="setup-runtime-title" className="panel setup-step setup-backend">
        <StepHeader id="setup-runtime-title" number={1} complete={backendReady} title="Prediction runtime" subtitle={backendReady ? `${backendDisplayName(props.status?.backend.mode)} is active and tested` : "Install the recommended local CPU runtime"} />
        {!backendReady ? (
          <button className="primary-action prominent-action" disabled={locked} onClick={() => props.onInstallBackend("cpu")}>
            <Icon name="download" />
            {props.pendingAction.includes("CPU backend") ? props.pendingAction : "Install recommended runtime"}
          </button>
        ) : (
          <div className="step-success"><Icon name="check" /><span>Runtime available</span></div>
        )}
        <details className="disclosure">
          <summary><Icon name="settings" /> Advanced runtime options</summary>
          <div className="disclosure-content">
            <div className="segmented" role="group" aria-label="Runtime mode">
              {(["cpu", "gpu", "conda"] as BackendMode[]).map((mode) => (
                <button
                  aria-pressed={props.backendMode === mode}
                  className={props.backendMode === mode ? "active" : ""}
                  disabled={locked}
                  key={mode}
                  onClick={() => props.setBackendMode(mode)}
                >
                  {mode === "gpu" ? (isMacPlatform() ? "Apple MPS" : "NVIDIA CUDA") : mode.toUpperCase()}
                </button>
              ))}
            </div>
            {props.backendMode === "conda" ? (
              <div className="field path-field">
                <label htmlFor={condaPythonId}>Conda environment Python</label>
                <div className="path-row">
                  <span className="path-leading" aria-hidden="true"><Icon name="file" /></span>
                  <input id={condaPythonId} disabled={locked} value={props.condaPython} onChange={(event) => props.setCondaPython(event.target.value)} placeholder={isMacPlatform() ? "/opt/conda/envs/protcross/bin/python" : "C:\\Miniconda3\\envs\\protcross\\python.exe"} />
                  <button disabled={locked} onClick={async () => {
                    const selected = await open({ multiple: false });
                    if (typeof selected === "string") {
                      props.setCondaPython(selected);
                    }
                  }}>Browse…</button>
                </div>
              </div>
            ) : null}
            <label className="field">
              <span>Network proxy <small>Optional</small></span>
              <input disabled={locked} value={props.proxyUrl} onChange={(event) => props.setProxyUrl(event.target.value)} placeholder="http://proxy.example:8080" />
            </label>
            <p className="field-help">{acceleratorLabel} requires compatible hardware and drivers.</p>
            <div className="button-row">
              <button disabled={locked} onClick={() => props.onInstallBackend("gpu")}><Icon name="download" /> Install {acceleratorLabel}</button>
              <button disabled={locked || condaNeedsPath || !props.status} onClick={props.onConfigureBackend}>Save selection</button>
              <button disabled={locked || condaNeedsPath || !props.status} onClick={props.onTestBackend}><Icon name="activity" /> Save and test</button>
              <button disabled={props.busy || (props.runtimeLocked && !props.backendConnectionLost)} onClick={props.onRestartBackend}><Icon name="refresh" /> Restart runtime</button>
            </div>
          </div>
        </details>
      </section>

      <section aria-labelledby="setup-license-title" className="panel setup-step setup-license">
        <StepHeader id="setup-license-title" number={2} complete={licenseConfirmed} title="ESM-C model terms" subtitle={licenseConfirmed ? "License confirmation recorded" : "Review the Cambrian Non-Commercial License"} />
        <p>ESM-C weights use EvolutionaryScale's Cambrian Non-Commercial License.</p>
        <div className="button-row">
          <button disabled={props.busy} onClick={props.onOpenLicense}>Read license <Icon name="external" /></button>
          <button className={!licenseConfirmed ? "primary-action" : ""} disabled={props.busy || licenseConfirmed || !props.status} onClick={props.onConfirmLicense}>
            {licenseConfirmed ? <><Icon name="check" /> Accepted</> : "I have reviewed and accept the terms"}
          </button>
        </div>
        {!props.status ? <p className="field-help">The local runtime must be available before this confirmation can be saved.</p> : null}
      </section>

      <section aria-labelledby="setup-assets-title" className="panel setup-step setup-assets">
        <StepHeader id="setup-assets-title" number={3} complete={assetsReady} title="Model assets" subtitle={assetsReady ? "All three assets are present and verified" : "Download the 2.14 GiB ESM-C weights"} />
        <div className="asset-grid">
          <AssetLine label="Checkpoint" status={props.status?.assets.checkpoint} />
          <AssetLine label="PCA" status={props.status?.assets.pca} />
          <AssetLine label="ESM-C" status={props.status?.assets.esm} />
        </div>
        <div className="button-row">
          <button className={!assetsReady ? "primary-action" : ""} disabled={props.busy || !licenseConfirmed || downloadActive} onClick={props.onDownloadEsm}>
            <Icon name="download" />
            {["cancelled", "failed"].includes(props.assetDownload?.status ?? "") ? "Resume ESM-C download" : assetsReady ? "Verify ESM-C again" : "Download ESM-C · 2.14 GiB"}
          </button>
          <button disabled={!downloadActive || props.assetDownload?.status === "cancelling"} onClick={props.onCancelEsm}>
            <Icon name="pause" /> {props.assetDownload?.status === "cancelling" ? "Pausing…" : "Pause"}
          </button>
        </div>
        {props.assetDownload ? <AssetDownloadProgress job={props.assetDownload} /> : null}
        <details className="disclosure compact-disclosure">
          <summary><Icon name="more" /> Manual asset options</summary>
          <div className="button-row disclosure-content">
            <button disabled={props.busy || !props.status} onClick={props.onImportCheckpoint}>Import checkpoint</button>
            <button disabled={props.busy || !props.status} onClick={props.onImportPca}>Import PCA</button>
            <button disabled={props.busy || !licenseConfirmed || downloadActive} onClick={props.onImportEsm}>Import ESM-C .pth</button>
            <button disabled={props.busy || !licenseConfirmed || downloadActive} onClick={props.onRefreshEsm}><Icon name="refresh" /> Redownload and verify</button>
          </div>
        </details>
      </section>

      {props.setupIssues.length ? (
        <section className="panel setup-summary">
          <h3>Items requiring attention</h3>
          <ReadinessList issues={props.setupIssues} />
        </section>
      ) : null}
    </div>
  );
}

function AssetDownloadProgress({ job }: { job: AssetDownloadJob }) {
  const total = job.total_bytes ?? 0;
  const percent = Number.isFinite(job.percent)
    ? Number(job.percent)
    : total ? (100 * job.downloaded_bytes / total) : 0;
  return (
    <div className="download-progress">
      <span className="sr-only" role="status">{downloadStatusLabel(job.status)}</span>
      <div>
        <strong>{downloadStatusLabel(job.status)}</strong>
        <span>{formatBytes(job.downloaded_bytes)} / {total ? formatBytes(total) : "unknown size"}</span>
        {job.bytes_per_second ? <span>{formatBytes(job.bytes_per_second)}/s</span> : null}
      </div>
      <progress aria-label="ESM-C download progress" max={100} value={Math.max(0, Math.min(100, percent))} />
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
  defaultOutputRoot?: string;
  threshold: number;
  setThreshold: (value: number) => void;
  clusterCutoff: number;
  setClusterCutoff: (value: number) => void;
  allowTruncation: boolean;
  setAllowTruncation: (value: boolean) => void;
  onOpenSetup: () => void;
  onRun: () => void;
}) {
  const truncationBlocked = Boolean(props.inspection?.requires_truncation && !props.allowTruncation);
  return (
    <div className="predict-layout">
      {!props.ready ? (
        <div className="callout warning span-all">
          <Icon name="warning" />
          <div><strong>Finish setup before running a prediction</strong><span>{props.setupIssues[0] ?? "The local runtime needs attention."}</span></div>
          <button onClick={props.onOpenSetup}>Open setup <Icon name="arrow-right" /></button>
        </div>
      ) : null}
      <section className="panel prediction-form">
        <div className="section-heading">
          <span className="step-kicker">Step 1</span>
          <h3>Choose a structure</h3>
          <p>Select a PDB or mmCIF coordinate file. ProtCross checks it before loading the model.</p>
        </div>
        <PathInput label="Structure file" value={props.inputPath} setValue={props.setInputPath} kind="file" prominent />

        <div className="section-divider" />
        <div className="section-heading compact">
          <span className="step-kicker">Step 2</span>
          <h3>Choose the destination</h3>
        </div>
        <PathInput label="Output directory" value={props.outputDir} setValue={props.setOutputDir} kind="directory" />
        {!props.outputDir ? <p className="field-help">Automatic location: <code>{props.defaultOutputRoot ? `${props.defaultOutputRoot}${pathSeparator()}<structure>` : "ProtCross application-data outputs"}</code>. Existing names receive a unique run suffix.</p> : null}

        <details className="disclosure settings-disclosure">
          <summary><Icon name="settings" /> Prediction settings <span>Defaults: {props.threshold.toFixed(2)} · {props.clusterCutoff.toFixed(1)} Å</span></summary>
          <div className="disclosure-content">
            <div className="inline-fields">
              <NumberInput label="Model-score cutoff" value={props.threshold} setValue={props.setThreshold} min={0} max={1} step={0.01} />
              <NumberInput label="Cluster distance (Å)" value={props.clusterCutoff} setValue={props.setClusterCutoff} min={0.1} max={40} step={0.5} />
            </div>
            <p className="field-help">Residues with scores above the cutoff are grouped by the Cα distance setting.</p>
            <label className="checkbox-line">
              <input type="checkbox" checked={props.allowTruncation} onChange={(event) => props.setAllowTruncation(event.target.checked)} />
              <span><strong>Allow long-chain truncation</strong><small>Keep the first 1,022 residues of an over-length ESM-C chain context.</small></span>
            </label>
          </div>
        </details>

        {truncationBlocked ? (
          <div className="inline-error" role="alert"><Icon name="warning" /> The selected chain exceeds the ESM-C context. Enable truncation or choose a shorter chain.</div>
        ) : null}
        <div className="form-action-bar">
          <div><strong>{props.inputPath ? fileName(props.inputPath) : "No structure selected"}</strong><span>{props.inspection ? `${props.inspection.scorable_residue_count} scorable residues` : "A structure check is required"}</span></div>
          <button
            className="primary-action run-action"
            disabled={
              props.busy || !props.ready || !props.inputPath || props.inspecting ||
              Boolean(props.inspectionError) || !props.inspection || truncationBlocked
            }
            onClick={props.onRun}
          >
            {props.busy ? <><span className="button-spinner" /> Running prediction</> : <><Icon name="play" /> Run prediction</>}
          </button>
        </div>
      </section>
      <section className="panel structure-preview">
        <div className="section-heading">
          <span className="step-kicker">Preflight</span>
          <h3>Structure check</h3>
          <p>Review chain selection and coordinate quality before inference.</p>
        </div>
        <StructureInspectionCard
          inspection={props.inspection}
          error={props.inspectionError}
          inspecting={props.inspecting}
          chainSelection={props.chainSelection}
          setChainSelection={props.setChainSelection}
        />
      </section>
    </div>
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
    return <div className="structure-empty checking" role="status"><span className="spinner" aria-hidden="true" /><strong>Checking structure…</strong><span>Reading coordinate models, chains, and scorable residues.</span></div>;
  }
  if (props.error && !props.inspection) {
    return <div className="inline-error structure-check" role="alert"><Icon name="warning" /><div><strong>Structure check failed</strong><span>{props.error}</span></div></div>;
  }
  if (!props.inspection) {
    return <div className="structure-empty"><div className="empty-icon"><Icon name="file" size={24} /></div><strong>No structure selected</strong><span>Choose a coordinate file to see its preflight summary here.</span></div>;
  }
  const report = props.inspection;
  const hasWarnings = report.warnings.length > 0;
  return (
    <div className={`structure-check ${hasWarnings ? "has-warnings" : "ready"}`}>
      <div className="structure-check-heading">
        <div>
          <span className={`state-icon ${hasWarnings ? "warning" : "success"}`}><Icon name={hasWarnings ? "warning" : "check"} /></span>
          <span><strong>{hasWarnings ? "Ready with warnings" : "Ready for prediction"}</strong><small>{report.format} · model 1 of {report.model_count}</small></span>
        </div>
        <label className="compact-field">
          <span>Chains to analyze</span>
          <select value={props.chainSelection} onChange={(event) => props.setChainSelection(event.target.value)}>
            <option value={ALL_CHAINS}>All scorable chains</option>
            {report.available_chains.map((chain) => (
              <option value={chain} key={chain || "blank-chain"}>Chain {displayChain(chain)}</option>
            ))}
          </select>
        </label>
      </div>
      <p className="chain-scope-help">All chains builds one complex-wide geometry graph. A single-chain choice limits both sequence context and scored geometry; prepare a subset coordinate file for a custom multi-chain scope.</p>
      {props.error ? <div className="inline-error"><strong>Chain check failed:</strong> {props.error}</div> : null}
      <div className="inspection-summary">
        <Metric label="Scorable residues" value={String(report.scorable_residue_count)} />
        <Metric label="Selected chains" value={String(report.selected_chains.length)} />
        <Metric label="Longest context" value={`${report.longest_chain_context} aa`} />
      </div>
      {report.warnings.length ? (
        <div className="warning-list compact">
          {report.warnings.map((warning) => <div key={warning}><Icon name="warning" />{warning}</div>)}
        </div>
      ) : null}
      <details className="disclosure compact-disclosure">
        <summary><Icon name="info" /> Coordinate details</summary>
        <div className="inspection-metrics disclosure-content">
          <Metric label="Missing Cα" value={String(report.standard_residues_missing_ca)} />
          <Metric label="Modified residues" value={String(report.modified_or_nonstandard_amino_acids)} />
          <Metric label="Coordinate breaks" value={String(report.sequence_break_count)} />
          <Metric label="Numbering gaps" value={String(report.numbering_gap_count)} />
        </div>
        <p className="field-help">ProtCross analyzes the supplied first coordinate model. Selected chains share one geometry graph.</p>
      </details>
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
  defaultOutputRoot?: string;
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
  const pageOffset = props.batchJob?.items_offset ?? props.batchPageOffset;
  const pageReturned = props.batchJob?.items_returned ?? props.batchJob?.items.length ?? 0;
  const itemCount = props.batchJob?.item_count ?? props.batchJob?.items.length ?? 0;
  const pageStart = itemCount === 0 ? 0 : pageOffset + 1;
  const pageEnd = Math.min(itemCount, pageOffset + pageReturned);
  const canPrevious = Boolean(props.batchJob && pageOffset > 0);
  const canNext = Boolean(props.batchJob && pageOffset + pageReturned < itemCount);
  const processed = props.batchJob?.completed ?? 0;
  const successful = Math.max(0, processed - (props.batchJob?.failed ?? 0));
  const progressLabel = props.batchJob ? `${processed}/${itemCount} processed · ${props.batchJob.failed} failed` : "";
  const progress = itemCount ? Math.min(100, (processed / itemCount) * 100) : 0;
  return (
    <div className="batch-layout">
      {!props.ready ? <div className="callout warning span-all"><Icon name="warning" /><div><strong>Batch prediction is unavailable</strong><span>{props.setupIssues[0] ?? "Complete environment setup first."}</span></div></div> : null}
      <section className="panel batch-staging">
        <div className="section-heading row-heading">
          <div><span className="step-kicker">Input queue</span><h3>Structures</h3><p>Add PDB and mmCIF files, then review the queue before starting.</p></div>
          <div className="button-row">
            {props.batchInputs.length ? <button disabled={props.batchActive} onClick={() => props.setBatchInputs([])}><Icon name="trash" /> Clear</button> : null}
            <button
              className="primary-action"
              disabled={props.batchActive}
              onClick={async () => {
                const selected = await open({ multiple: true, filters: [{ name: "Structures", extensions: ["pdb", "cif", "mmcif"] }] });
                if (Array.isArray(selected)) {
                  props.setBatchInputs(uniquePaths([...props.batchInputs, ...selected]));
                }
              }}
            >
              <Icon name="file" /> Add structures
            </button>
          </div>
        </div>
        {props.batchInputs.length ? (
          <div className="staging-list" aria-label={`${props.batchInputs.length} selected structures`}>
            <div className="staging-summary"><strong>{props.batchInputs.length} structure{props.batchInputs.length === 1 ? "" : "s"}</strong><span>Duplicates are removed automatically</span></div>
            {props.batchInputs.map((path) => (
              <div className="staging-item" key={path}>
                <span className="file-type">{fileExtension(path)}</span>
                <span className="file-copy"><strong>{fileName(path)}</strong><small title={path}>{parentPath(path)}</small></span>
                <button aria-label={`Remove ${fileName(path)}`} className="icon-button subtle" disabled={props.batchActive} onClick={() => props.setBatchInputs(props.batchInputs.filter((item) => item !== path))}><Icon name="close" /></button>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-dropzone"><div className="empty-icon"><Icon name="batch" size={26} /></div><strong>Your queue is empty</strong><span>Add one or more structures to begin a batch.</span></div>
        )}
      </section>

      <section className="panel batch-settings">
        <div className="section-heading"><span className="step-kicker">Shared configuration</span><h3>Batch settings</h3><p>These settings apply to every structure in this run.</p></div>
        <PathInput label="Output directory" value={props.outputDir} setValue={props.setOutputDir} kind="directory" />
        {!props.outputDir ? <p className="field-help">Automatic location: <code>{props.defaultOutputRoot ? `${props.defaultOutputRoot}${pathSeparator()}batch${pathSeparator()}<job-id>` : "ProtCross application-data outputs"}</code>.</p> : null}
        <details className="disclosure settings-disclosure">
          <summary><Icon name="settings" /> Advanced settings <span>{props.threshold.toFixed(2)} · {props.clusterCutoff.toFixed(1)} Å</span></summary>
          <div className="disclosure-content">
            <div className="callout neutral compact-callout"><Icon name="info" /><div><strong>Chain scope</strong><span>Each batch structure uses all scorable chains in one geometry graph. Use single prediction or a prepared subset file for chain-specific analysis.</span></div></div>
            <div className="inline-fields">
              <NumberInput label="Model-score cutoff" value={props.threshold} setValue={props.setThreshold} min={0} max={1} step={0.01} />
              <NumberInput label="Cluster distance (Å)" value={props.clusterCutoff} setValue={props.setClusterCutoff} min={0.1} max={40} step={0.5} />
            </div>
            <label className="checkbox-line"><input type="checkbox" checked={props.allowTruncation} onChange={(event) => props.setAllowTruncation(event.target.checked)} /><span><strong>Allow long-chain truncation</strong><small>Applied only where an ESM-C chain context exceeds 1,022 residues.</small></span></label>
          </div>
        </details>
        <div className="batch-submit-summary"><span>{props.batchInputs.length} queued</span><span>Bounded microbatch execution</span></div>
        <button className="primary-action run-action full-width" disabled={props.busy || props.batchActive || !props.ready || props.batchInputs.length === 0} onClick={props.onSubmit}>
          {props.batchActive ? <><span className="button-spinner" /> Batch running</> : props.busy ? "Starting batch…" : <><Icon name="play" /> Start batch prediction</>}
        </button>
      </section>

      {props.batchJob ? (
        <section className="panel batch-monitor span-all">
          <div className="batch-monitor-header">
            <div><span className="step-kicker">Current run</span><h3>{humanizeStatus(props.batchJob.status)}</h3><p>{progressLabel}</p></div>
            <button className="danger-action" disabled={props.batchJob.cancel_requested || !["queued", "running"].includes(props.batchJob.status)} onClick={props.onCancel}>
              <Icon name="pause" /> {props.batchJob.cancel_requested ? "Stopping after current group" : "Stop after current group"}
            </button>
          </div>
          <div className="batch-progress">
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            <div className="batch-stats">
              <Metric label="Processed" value={`${processed} / ${itemCount}`} />
              <Metric label="Succeeded" value={String(successful)} tone="success" />
              <Metric label="Failed" value={String(props.batchJob.failed)} tone={props.batchJob.failed ? "danger" : undefined} />
              <Metric label="Remaining" value={String(Math.max(0, itemCount - processed))} />
            </div>
          </div>
          {props.batchJob.error ? <div className="inline-error batch-error" role="alert"><Icon name="warning" />{String(props.batchJob.error).split("\n")[0]}</div> : null}
          {props.batchJob.cancel_requested && ["queued", "running"].includes(props.batchJob.status) ? <div className="callout warning compact-callout"><Icon name="info" /><div><strong>Stop requested</strong><span>The active microbatch will finish; queued structures will remain untouched.</span></div></div> : null}
          <div className="table-wrap" tabIndex={0} aria-label="Batch prediction results">
            <table>
              <caption className="sr-only">Batch structures and prediction status</caption>
              <thead><tr><th scope="col">Status</th><th scope="col">Structure</th><th scope="col">Output or error</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {props.batchJob.items.map((item) => (
                  <tr key={item.input_structure}>
                    <td><StatusPill status={item.status} /></td>
                    <td><span className="table-file"><strong>{fileName(item.input_structure)}</strong><small title={item.input_structure}>{parentPath(item.input_structure)}</small></span></td>
                    <td className={item.error ? "error-copy" : "path-copy"}>{item.error ? firstLine(item.error) : item.output_dir ?? parentPath(item.output_files?.summary_json ?? "")}</td>
                    <td><button disabled={item.status !== "completed" || !item.output_files?.summary_json || props.busy} onClick={() => void props.onViewItem(item)}>View <Icon name="arrow-right" /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="pager">
              <button disabled={!canPrevious || props.busy} onClick={() => props.onPageChange(Math.max(0, pageOffset - props.batchPageSize))}>Previous</button>
              <span>{pageStart}–{pageEnd} of {itemCount}</span>
              <button disabled={!canNext || props.busy} onClick={() => props.onPageChange(pageOffset + props.batchPageSize)}>Next</button>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}

function ResultsPanel(props: {
  structurePath?: string;
  outputFiles?: Record<string, string>;
  summary: any;
  pockets: PocketJson | null;
  residues: ResidueSummary[];
  darkMode: boolean;
  onOpenExisting: () => void;
  onNotify: (message: string) => void;
  onError: (message: string) => void;
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
  async function copyResult(value: string, label: string) {
    try {
      await navigator.clipboard.writeText(value);
      props.onNotify(`${label} copied to the clipboard.`);
    } catch (exc) {
      props.onError(`Could not copy ${label.toLowerCase()}: ${exc instanceof Error ? exc.message : String(exc)}`);
    }
  }
  if (!props.summary && !props.pockets && !props.outputFiles) {
    return (
      <section className="panel empty-results">
        <div className="empty-illustration"><Icon name="results" size={30} /></div>
        <span className="eyebrow">Results workspace</span>
        <h3>No prediction loaded</h3>
        <p>Run a prediction, select a completed batch item, or reopen a previous ProtCross result package.</p>
        <div className="button-row centered"><button className="primary-action" onClick={props.onOpenExisting}><Icon name="folder" /> Open existing result</button></div>
        <small>Choose a <code>*.protcross.summary.json</code> file.</small>
      </section>
    );
  }
  return (
    <div className="results-page">
      <section className="result-identity">
        <div><span className="eyebrow">Prediction result</span><h3>{fileName(String(props.summary?.input_structure ?? props.structurePath ?? "ProtCross result"))}</h3><p>{displayedResidues.length} residues in the displayed cluster · {clusters.length} predicted cluster{clusters.length === 1 ? "" : "s"}</p></div>
        <div className="button-row">
          <button onClick={props.onOpenExisting}><Icon name="folder" /> Open another result</button>
          <button className="primary-action" disabled={!outputDir} onClick={() => outputDir && invoke("open_path", { path: outputDir }).catch((exc) => props.onError(String(exc)))}><Icon name="external" /> Show in folder</button>
        </div>
      </section>
      <div className="results-layout">
      <Suspense fallback={<section className="viewer-panel viewer-loading">Loading structure viewer...</section>}>
        <MolstarViewer
          structurePath={props.structurePath}
          summary={props.summary}
          pockets={props.pockets}
          selectedClusterIndex={selectedClusterIndex}
          darkMode={props.darkMode}
        />
      </Suspense>
      <section className="panel result-panel">
        <div className="section-heading">
          <span className="step-kicker">Cluster inspector</span>
          <h3>Binding-site scores</h3>
          <p>Select a predicted-residue cluster to focus it in the structure viewer.</p>
        </div>
        {props.summary?.warnings?.length ? (
          <div className="warning-list">
            {props.summary.warnings.map((warning: string) => <div key={warning}><Icon name="warning" />{warning}</div>)}
          </div>
        ) : null}
        {clusters.length ? (
          <label className="field cluster-select">
            <span>Displayed predicted-residue cluster</span>
            <select value={selectedClusterIndex} onChange={(event) => setSelectedClusterIndex(Number(event.target.value))}>
              {clusters.map((cluster, index) => (
                <option value={index} key={cluster.cluster_id ?? index}>Cluster {cluster.cluster_id ?? index + 1} · {cluster.residue_count} residues · max {cluster.score_max.toFixed(3)}</option>
              ))}
            </select>
          </label>
        ) : null}
        {displayedPocket ? (
          <div className="metric-row">
            <Metric label="Residues in cluster" value={String(displayedPocket.residue_count)} />
            <Metric label="Maximum score" value={displayedPocket.score_max.toFixed(4)} tone="accent" />
            <Metric label="Mean score" value={Number(displayedPocket.score_mean ?? 0).toFixed(4)} />
          </div>
        ) : (
          <div className="callout neutral"><Icon name="info" /><div><strong>No cluster at this cutoff</strong><span>No residue group passed the current model-score and distance settings.</span></div></div>
        )}
        <div className="centroid-card">
          <div><span>Score-weighted Cα centroid</span><code>{center ? center.map((value) => value.toFixed(3)).join(", ") : "—"} Å</code></div>
          <button
            disabled={!center}
            aria-label="Copy score-weighted centroid"
            className="icon-button subtle"
            onClick={() => center && void copyResult(center.map((value) => value.toFixed(3)).join(", "), "Centroid")}
          >
            <Icon name="copy" />
          </button>
        </div>
        <div className="table-heading"><div><h4>Cluster residues</h4><span>Sorted by model score</span></div><button disabled={displayedResidues.length === 0} onClick={() => void copyResult(formatResidueSelection(displayedResidues), "Residue selection")}><Icon name="copy" /> Copy selection</button></div>
        <div className="table-wrap residue-table" tabIndex={0}>
          <table>
            <caption className="sr-only">Residues in the displayed predicted binding-site cluster</caption>
            <thead>
              <tr>
                <th scope="col">Residue</th>
                <th scope="col">Chain</th>
                <th scope="col">Model score</th>
              </tr>
            </thead>
            <tbody>
              {[...displayedResidues].sort((a, b) => Number(b.score ?? b.probability) - Number(a.score ?? a.probability)).map((residue) => (
                <tr key={`${residue.residue_id}-${residue.cluster_id ?? ""}`}>
                  <td>{residue.residue_id}</td>
                  <td>{String(residue.chain_id ?? "")}</td>
                  <td><ScoreBar value={Number(residue.score ?? residue.probability)} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <details className="disclosure result-details">
          <summary><Icon name="info" /> Result details and files</summary>
          <div className="disclosure-content">
            <p className="result-provenance">ProtCross {String(props.summary?.protcross_version ?? APP_VERSION)} · assets {String(props.summary?.asset_version ?? "unknown")} · geometry {String(props.summary?.geometry_backend ?? "unknown")}</p>
            <button onClick={() => invoke("open_url", { url: TECHNICAL_GUIDE_URL }).catch((exc) => props.onError(String(exc)))}>Open technical guide <Icon name="external" /></button>
            {props.outputFiles ? <div className="output-files">{Object.entries(props.outputFiles).map(([key, value]) => <div className="output-file" key={key}><span><Icon name="file" />{outputFileLabel(key)}</span><code title={value}>{value}</code><button aria-label={`Copy ${outputFileLabel(key)} path`} className="icon-button subtle" onClick={() => void copyResult(value, `${outputFileLabel(key)} path`)}><Icon name="copy" /></button></div>)}</div> : null}
          </div>
        </details>
      </section>
      </div>
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
  const backendReady = backendIsHealthy(props.status);
  const testOk = props.envTest?.ok === true || props.status?.backend.backend_test_ok === true;
  const assetsReady = Boolean(props.status?.assets.ready);
  return (
    <div className="diagnostics-layout">
      <section className="health-overview span-all">
        <div><span className="eyebrow">System health</span><h3>{backendReady && assetsReady ? "Environment is operational" : "Environment needs attention"}</h3><p>ProtCross Desktop {APP_VERSION} · {backendDisplayName(props.status?.backend.mode)}</p></div>
        <button className="primary-action" onClick={props.onTest}><Icon name="activity" /> Run environment test</button>
      </section>
      <section className="panel health-panel">
        <div className="section-heading"><span className="step-kicker">Runtime</span><h3>Backend health</h3></div>
        <div className="health-list">
          <HealthRow label="Python runtime" ok={backendReady} value={props.status?.backend.python ?? "Unavailable"} />
          <HealthRow label="Environment test" ok={testOk} value={testOk ? "Passed" : props.status?.backend.backend_test_ok === false ? "Failed" : "Not run"} />
          <HealthRow label="Runtime version" ok={Boolean(props.status?.backend.backend_test_package_version && props.status.backend.backend_test_package_version === props.status.backend.required_package_version)} value={props.status?.backend.backend_test_package_version ?? "Unknown"} />
        </div>
      </section>
      <section className="panel health-panel">
        <div className="section-heading"><span className="step-kicker">Model data</span><h3>Asset health</h3></div>
        <div className="asset-grid diagnostic-assets">
          <AssetLine label="Checkpoint" status={props.status?.assets.checkpoint} />
          <AssetLine label="PCA" status={props.status?.assets.pca} />
          <AssetLine label="ESM-C" status={props.status?.assets.esm} />
        </div>
      </section>
      <section className="panel support-panel">
        <div className="section-heading"><span className="step-kicker">Support</span><h3>Resolve an issue</h3><p>Run the health check first. Exported diagnostics include versions, configuration, and local logs.</p></div>
        <div className="support-actions">
          <button className="support-action" onClick={props.onExport}><span className="action-icon"><Icon name="download" /></span><span><strong>Export diagnostics</strong><small>Create a local ZIP for an issue report</small></span><Icon name="chevron-right" /></button>
          <button className="support-action" onClick={props.onOpenScientificGuide}><span className="action-icon"><Icon name="help" /></span><span><strong>Technical guide</strong><small>Inputs, outputs, and inference details</small></span><Icon name="external" /></button>
          <button className="support-action" onClick={props.onOpenReleases}><span className="action-icon"><Icon name="refresh" /></span><span><strong>Check releases</strong><small>View current Desktop downloads</small></span><Icon name="external" /></button>
        </div>
      </section>
      <section className="panel technical-panel">
        <div className="section-heading"><span className="step-kicker">Advanced</span><h3>Technical details</h3><p>Use this structured report when troubleshooting runtime or asset problems.</p></div>
        <details className="disclosure technical-disclosure">
          <summary><Icon name="diagnostics" /> Show runtime report</summary>
          <pre className="diagnostic-json">{JSON.stringify({ status: props.status, envTest: props.envTest }, null, 2)}</pre>
        </details>
      </section>
    </div>
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

function PathInput({ label, value, setValue, kind, prominent = false }: { label: string; value: string; setValue: (value: string) => void; kind: "file" | "directory"; prominent?: boolean }) {
  const id = useId();
  return (
    <div className={`field path-field ${prominent ? "prominent" : ""}`}>
      <label htmlFor={id}>{label}</label>
      <div className="path-row">
        <span className="path-leading" aria-hidden="true"><Icon name={kind === "file" ? "file" : "folder"} /></span>
        <input id={id} value={value} onChange={(event) => setValue(event.target.value)} placeholder={kind === "file" ? "Choose a .pdb, .cif, or .mmcif file" : "Use the automatic result location"} />
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
          Browse…
        </button>
      </div>
    </div>
  );
}

function NumberInput(props: { label: string; value: number; setValue: (value: number) => void; min: number; max: number; step: number }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{props.label}</label>
      <input
        id={id}
        type="number"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(event) => props.setValue(Number(event.target.value))}
      />
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "success" | "danger" | "accent" }) {
  return (
    <div className={`metric ${tone ? `metric-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StepHeader({ id, number, complete, title, subtitle }: { id: string; number: number; complete: boolean; title: string; subtitle: string }) {
  return (
    <div className="step-header">
      <span className={`step-number ${complete ? "complete" : ""}`}>{complete ? <Icon name="check" /> : number}</span>
      <span><h3 id={id}>{title}</h3><small>{subtitle}</small></span>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const tone = status === "completed" ? "success" : status === "failed" ? "danger" : status === "running" ? "active" : status === "cancelled" ? "neutral" : "queued";
  return <span className={`status-pill ${tone}`}><span className="status-dot" aria-hidden="true" />{humanizeStatus(status)}</span>;
}

function ScoreBar({ value }: { value: number }) {
  const score = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  return (
    <span className="score-cell">
      <span className="score-bar" aria-hidden="true"><span style={{ width: `${score * 100}%` }} /></span>
      <strong>{score.toFixed(4)}</strong>
    </span>
  );
}

function HealthRow({ label, ok, value }: { label: string; ok: boolean; value: string }) {
  return (
    <div className="health-row">
      <span className={`state-icon ${ok ? "success" : "warning"}`}><Icon name={ok ? "check" : "warning"} /></span>
      <span><strong>{label}</strong><small title={value}>{value}</small></span>
    </div>
  );
}

function labelForTab(tab: Tab): string {
  return {
    setup: "Setup",
    predict: "Predict",
    batch: "Batch prediction",
    results: "Results",
    diagnostics: "Diagnostics"
  }[tab];
}

function ReadinessList({ issues }: { issues: string[] }) {
  if (issues.length === 0) {
    return <div className="readiness ready"><Icon name="check" /> Ready for prediction.</div>;
  }
  return (
    <div className="readiness">
      <strong><Icon name="warning" /> Needs attention</strong>
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

function storedNumber(key: string, fallback: number): number {
  const stored = window.localStorage.getItem(key);
  if (stored === null) {
    return fallback;
  }
  const value = Number(stored);
  return Number.isFinite(value) ? value : fallback;
}

async function withRequestDeadline<T>(
  request: (signal: AbortSignal) => Promise<T>,
  timeoutMs = 10_000
): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await request(controller.signal);
  } catch (exc) {
    if (controller.signal.aborted) {
      throw new Error(`Desktop backend request timed out after ${Math.round(timeoutMs / 1000)} seconds.`);
    }
    throw exc;
  } finally {
    window.clearTimeout(timer);
  }
}

function isMacPlatform(): boolean {
  return /Mac|iPhone|iPad/.test(navigator.userAgent);
}

function backendDisplayName(mode?: BackendMode | null): string {
  if (mode === "cpu") {
    return "CPU runtime";
  }
  if (mode === "gpu") {
    return isMacPlatform() ? "Apple MPS runtime" : "NVIDIA CUDA runtime";
  }
  if (mode === "conda") {
    return "Conda runtime";
  }
  return "Runtime not selected";
}

function backendIsHealthy(status: DesktopStatus | null): boolean {
  const backend = status?.backend;
  return Boolean(
    backend?.mode
    && backend.python_present
    && backend.backend_test_ok === true
    && backend.backend_test_mode === backend.mode
    && backend.backend_test_python === backend.python
    && backend.backend_test_package_version === backend.required_package_version
    && backend.runtime_matches_config !== false
  );
}

function fileName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

function parentPath(path: string): string {
  const name = fileName(path);
  return path.slice(0, Math.max(0, path.length - name.length)).replace(/[\\/]$/, "") || ".";
}

function pathSeparator(): string {
  return /Windows/.test(navigator.userAgent) ? "\\" : "/";
}

function fileExtension(path: string): string {
  const match = /\.([^.\\/]+)$/.exec(path);
  return (match?.[1] ?? "file").toUpperCase();
}

function uniquePaths(paths: string[]): string[] {
  const seen = new Set<string>();
  return paths.filter((path) => {
    const key = isMacPlatform() ? path : path.toLocaleLowerCase();
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function firstLine(value: string): string {
  return value.split(/\r?\n/, 1)[0];
}

function humanizeStatus(status: string): string {
  return status.replace(/[_-]+/g, " ").replace(/^./, (letter) => letter.toUpperCase());
}

function outputFileLabel(key: string): string {
  return {
    structure: "Annotated structure",
    scores_tsv: "Residue score table",
    pockets_json: "Pocket clusters",
    summary_json: "Run summary"
  }[key] ?? humanizeStatus(key);
}

function parsePreviewTab(value: string | null): Tab | null {
  return NAV_ITEMS.some((item) => item.id === value) ? value as Tab : null;
}

function previewState(tab: Tab): { status: DesktopStatus; prediction?: PredictResponse; batchJob?: BatchJob } {
  const ready = tab !== "setup";
  const fileStatus = (name: string) => ({ path: `/data/protcross/assets/${name}`, present: true, verified: true });
  const batchJob: BatchJob | undefined = tab === "batch" ? {
    id: "preview-batch",
    status: "running",
    completed: 3,
    failed: 1,
    cancel_requested: false,
    item_count: 6,
    items_offset: 0,
    items_returned: 6,
    items: [
      { input_structure: "/data/proteins/6fhu.pdb", status: "completed", output_dir: "/data/results/6fhu", output_files: { summary_json: "/data/results/6fhu/6fhu.protcross.summary.json" } },
      { input_structure: "/data/proteins/7abc.cif", status: "completed", output_dir: "/data/results/7abc", output_files: { summary_json: "/data/results/7abc/7abc.protcross.summary.json" } },
      { input_structure: "/data/proteins/8xyz.pdb", status: "failed", error: "No scorable standard amino-acid residues with Cα coordinates were found." },
      { input_structure: "/data/proteins/complex_alpha.cif", status: "running" },
      { input_structure: "/data/proteins/complex_beta.pdb", status: "queued" },
      { input_structure: "/data/proteins/target_042.cif", status: "queued" }
    ]
  } : undefined;
  const status: DesktopStatus = {
    paths: { root: "/data/protcross", outputs: "/data/results" },
    manifest: { version: APP_VERSION },
    assets: {
      ready,
      checkpoint: fileStatus("protcross.ckpt"),
      pca: fileStatus("pca.pkl"),
      esm: {
        license_confirmed: ready,
        path: ready ? "/data/protcross/assets/esmc_600m.pth" : null,
        present: ready,
        source: ready ? "downloaded" : null,
        expected_sha256: "preview",
        actual_sha256: ready ? "preview" : null,
        verified: ready ? true : null,
        filename: "esmc_600m.pth"
      }
    },
    backend: {
      mode: ready ? "cpu" : null,
      python: ready ? "/data/protcross/runtime/python" : null,
      python_present: ready,
      backend_test_ok: ready,
      backend_test_mode: ready ? "cpu" : null,
      backend_test_python: ready ? "/data/protcross/runtime/python" : null,
      backend_test_package_version: ready ? APP_VERSION : null,
      required_package_version: APP_VERSION,
      runtime_matches_config: ready,
      proxy_url: null
    },
    readiness: ready ? { ready: true, issues: [] } : {
      ready: false,
      issues: ["Install the selected backend environment.", "Confirm the ESM-C license.", "Download or import ESM-C weights."]
    },
    activity: { batch_jobs: batchJob ? [batchJob] : [], asset_downloads: [] }
  };
  const prediction: PredictResponse | undefined = tab === "results" ? {
    ok: true,
    summary: {
      schema_version: "protcross-summary-v2",
      input_structure: "/data/proteins/6fhu.pdb",
      protcross_version: APP_VERSION,
      asset_version: "0.1.2",
      geometry_backend: "torch",
      top_pocket: { cluster_id: 1, center: [12.442, -4.102, 8.774], residue_count: 5, score_mean: 0.821, score_max: 0.962 }
    },
    pockets: {
      schema_version: "protcross-pocket-v2",
      clustered_pockets: [
        { cluster_id: 1, center: [12.442, -4.102, 8.774], residue_count: 5, score_mean: 0.821, score_max: 0.962, residues: previewResidues(1, [0.962, 0.901, 0.835, 0.744, 0.663]) },
        { cluster_id: 2, center: [-3.201, 14.702, 22.118], residue_count: 3, score_mean: 0.704, score_max: 0.817, residues: previewResidues(2, [0.817, 0.694, 0.601]) }
      ]
    },
    top_pocket_residues: previewResidues(1, [0.962, 0.901, 0.835, 0.744, 0.663]),
    output_files: {
      scores_tsv: "/data/results/6fhu/6fhu.protcross.scores.tsv",
      pockets_json: "/data/results/6fhu/6fhu.protcross.pockets.json",
      summary_json: "/data/results/6fhu/6fhu.protcross.summary.json"
    }
  } : undefined;
  return { status, prediction, batchJob };
}

function previewResidues(clusterId: number, scores: number[]): ResidueSummary[] {
  return scores.map((score, index) => ({
    residue_id: `A_${120 + clusterId * 10 + index}`,
    chain_id: "A",
    residue_number: 120 + clusterId * 10 + index,
    score,
    probability: score,
    cluster_id: clusterId
  }));
}
