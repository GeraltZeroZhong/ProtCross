import { useEffect, useId, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { confirm, open, type OpenDialogOptions } from "@tauri-apps/plugin-dialog";
import packageInfo from "../package.json";
import {
  cancelBatch,
  cancelEsmDownload,
  configureBackend,
  downloadEsm,
  exportDiagnostics,
  getBatch,
  getBatchResult,
  getEsmDownload,
  getPredictionProgress,
  getStatus,
  importCheckpoint,
  importEsm,
  importPca,
  inspectStructure,
  openResult,
  retryBatch,
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
  SummaryJson,
  StructureInspection
} from "./types";
import { Icon } from "./components/Icon";
import { ResultsPanel } from "./components/ResultsPanel";
import { SetupPanel } from "./components/SetupPanel";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { downloadPhaseLabel } from "./components/EnvironmentDetails";
import { AdvancedParameters } from "./components/AdvancedParameters";
import { DEFAULT_PREDICTION_PARAMETERS, parameterValidationMessage, type PredictionParameters } from "./parameterSettings";

type Tab = "setup" | "predict" | "batch" | "results" | "diagnostics";
type ThemePreference = "system" | "light" | "dark";
interface BackendStartResult {
  token: string;
  port: number;
}

type BatchPreflightStatus = "checking" | "ready" | "failed";

interface BatchPreflight {
  status: BatchPreflightStatus;
  chainId: string;
  inspection?: StructureInspection;
  error?: string;
}

const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_CLUSTER_CUTOFF = 8.0;
const BATCH_PAGE_SIZE = 500;
const TECHNICAL_GUIDE_URL = "https://github.com/GeraltZeroZhong/ProtCross/blob/v0.2.5/README.md#model-and-inference-pipeline";
const APP_VERSION = packageInfo.version;
const ALL_CHAINS = "__all_chains__";


const NAV_ITEMS: Array<{ id: Tab; label: string }> = [
  { id: "predict", label: "Predict" },
  { id: "batch", label: "Batch" },
  { id: "results", label: "Results" },
  { id: "setup", label: "Setup" },
  { id: "diagnostics", label: "Diagnostics" }
];
const UI_PREVIEW_TAB = import.meta.env.DEV
  ? parsePreviewTab(new URLSearchParams(window.location.search).get("preview"))
  : null;

type BrowseFiles = (options: OpenDialogOptions) => Promise<string | string[] | null>;

export default function App() {
  const [tab, setTab] = useState<Tab>(UI_PREVIEW_TAB ?? "setup");
  const previousTab = useRef<Tab>(tab);
  const workspaceOptionsRef = useRef<HTMLDetailsElement>(null);
  const navigationTouched = useRef(false);
  const navigationRevision = useRef(0);
  const resultLoadRevision = useRef(0);
  const runtimeDraftTouched = useRef(false);
  function navigate(next: Tab) { navigationTouched.current = true; navigationRevision.current += 1; setTab(next); }
  const [themePreference, setThemePreference] = useState<ThemePreference>(() => {
    const saved = window.localStorage.getItem("protcross-theme");
    return saved === "light" || saved === "dark" ? saved : "system";
  });
  const [systemDarkMode, setSystemDarkMode] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);
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
  const [inspectionRevision, setInspectionRevision] = useState(0);
  const [singleRunning, setSingleRunning] = useState(false);
  const [singleProgress, setSingleProgress] = useState("");
  const [singleElapsed, setSingleElapsed] = useState(0);
  const predictionRequest = useRef<{ controller: AbortController; cancelled: boolean } | null>(null);
  const [outputDir, setOutputDir] = useState(() => window.localStorage.getItem("protcross-output-dir") ?? "");
  const [parameters, setParameters] = useState<PredictionParameters>(() => ({
    ...DEFAULT_PREDICTION_PARAMETERS,
    threshold: storedNumber("protcross-threshold", DEFAULT_THRESHOLD),
    clusterCutoff: storedNumber("protcross-cluster-cutoff", DEFAULT_CLUSTER_CUTOFF)
  }));
  const { threshold, clusterCutoff, allowTruncation, device, batchSize } = parameters;
  const parameterError = parameterValidationMessage(parameters);
  const batchParameterError = parameterValidationMessage(parameters, true);
  const [batchInputs, setBatchInputs] = useState<string[]>([]);
  const [batchPreflights, setBatchPreflights] = useState<Record<string, BatchPreflight>>({});
  const batchInspectionRevision = useRef(0);
  const batchInspectionRequests = useRef(new Map<string, number>());
  const batchViewRevision = useRef(0);
  const batchPagePending = useRef(false);
  const [batchPageLoading, setBatchPageLoading] = useState(false);
  const batchCancelRequest = useRef<string | null>(null);
  const [batchCancelPending, setBatchCancelPending] = useState<string | null>(null);
  const [batchJob, setBatchJob] = useState<BatchJob | null>(null);
  const [batchHistory, setBatchHistory] = useState<BatchJob[]>([]);
  const [batchPageOffset, setBatchPageOffset] = useState(0);
  const [batchResult, setBatchResult] = useState<PredictResponse | null>(null);
  const [exampleStructureData, setExampleStructureData] = useState<string | undefined>();
  const [lastPrediction, setLastPrediction] = useState<PredictResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [envTest, setEnvTest] = useState<Record<string, unknown> | null>(null);
  const [pendingAction, setPendingAction] = useState("");
  const [assetDownload, setAssetDownload] = useState<AssetDownloadJob | null>(null);
  const assetPauseRequest = useRef<string | null>(null);
  const [backendConnectionLost, setBackendConnectionLost] = useState(false);

  useEffect(() => {
    const preference = window.matchMedia("(prefers-color-scheme: dark)");
    const changed = () => setSystemDarkMode(preference.matches);
    preference.addEventListener("change", changed);
    return () => preference.removeEventListener("change", changed);
  }, []);

  async function browseFiles(options: OpenDialogOptions): Promise<string | string[] | null> {
    setError("");
    try {
      return await open(options);
    } catch (exc) {
      setError(`Could not open file dialog: ${exc instanceof Error ? exc.message : String(exc)}`);
      return null;
    }
  }

  useEffect(() => {
    document.documentElement.dataset.theme = themePreference;
    window.localStorage.setItem("protcross-theme", themePreference);
  }, [themePreference]);

  useEffect(() => {
    function dismissOutside(event: PointerEvent) {
      const options = workspaceOptionsRef.current;
      if (options?.open && event.target instanceof Node && !options.contains(event.target)) {
        options.open = false;
      }
    }
    function dismissWithEscape(event: KeyboardEvent) {
      const options = workspaceOptionsRef.current;
      if (event.key === "Escape" && options?.open) {
        options.open = false;
        options.querySelector("summary")?.focus();
        event.preventDefault();
      }
    }
    document.addEventListener("pointerdown", dismissOutside);
    document.addEventListener("keydown", dismissWithEscape);
    return () => {
      document.removeEventListener("pointerdown", dismissOutside);
      document.removeEventListener("keydown", dismissWithEscape);
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem("protcross-output-dir", outputDir);
  }, [outputDir]);

  useEffect(() => {
    if (!parameterValidationMessage(parameters)) {
      window.localStorage.setItem("protcross-threshold", String(threshold));
      window.localStorage.setItem("protcross-cluster-cutoff", String(clusterCutoff));
    }
  }, [threshold, clusterCutoff, device]);

  useEffect(() => {
    if (previousTab.current !== tab) {
      window.requestAnimationFrame(() => {
        document.getElementById("workspace-content")?.focus({ preventScroll: true });
        window.scrollTo(0, 0);
      });
      previousTab.current = tab;
    }
  }, [tab]);

  function applyStatus(next: DesktopStatus, syncDraft = false) {
    setStatus(next);
    setBackendConnectionLost(false);
    if (syncDraft || !runtimeDraftTouched.current) {
      if (next.backend.mode) setBackendMode(next.backend.mode);
      setProxyUrl(next.backend.proxy_url ?? "");
      setCondaPython(next.backend.mode === "conda" ? next.backend.python ?? "" : "");
      runtimeDraftTouched.current = false;
    }
    const downloads = next.activity?.asset_downloads ?? [];
    const activeDownload = [...downloads]
      .reverse()
      .find((job) => ["queued", "running", "cancelling"].includes(job.status));
    setAssetDownload((current) => {
      if (activeDownload) {
        return activeDownload;
      }
      const reported = current ? downloads.find((job) => job.id === current.id) : undefined;
      if (reported) return reported;
      if (current && next.assets.esm.verified === true) {
        return { ...current, status: "completed", percent: 100,
          downloaded_bytes: current.total_bytes ?? current.downloaded_bytes,
          bytes_per_second: null, error: null };
      }
      if (current && ["queued", "running", "cancelling"].includes(current.status)) {
        return {
          ...current,
          status: "failed",
          error: "The backend restarted before this download status could be recovered. Start again to resume retained partial data."
        };
      }
      return current;
    });
    const recentBatches = [...(next.activity?.batch_jobs ?? [])].reverse();
    setBatchHistory(recentBatches);
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
  }

  function rememberBatchJob(job: BatchJob) {
    const summary = { ...job, items: [], items_offset: 0, items_returned: 0 };
    setBatchHistory((current) => current.some((item) => item.id === job.id)
      ? current.map((item) => item.id === job.id ? summary : item)
      : [summary, ...current]);
  }

  async function inspectBatchInputs(paths: string[]) {
    const requests = paths.map((path) => {
      const revision = ++batchInspectionRevision.current;
      batchInspectionRequests.current.set(path, revision);
      return { path, revision };
    });
    for (let start = 0; start < requests.length; start += 4) {
      const group = requests.slice(start, start + 4);
      await Promise.all(group.map(async ({ path, revision }) => {
        const currentRequest = () => batchInspectionRequests.current.get(path) === revision;
        if (!currentRequest()) return;
        try {
          const report = await inspectStructure(path);
          if (!report.chain_summaries.some((chain) => chain.scorable_residue_count > 0)) {
            throw new Error("No scorable chain with standard amino-acid Cα coordinates was found.");
          }
          setBatchPreflights((current) => {
            if (!currentRequest() || !current[path]) return current;
            const currentChain = current[path].chainId;
            const chainId = currentChain === ALL_CHAINS
              || report.chain_summaries.some((chain) => chain.chain_id === currentChain)
              ? currentChain : ALL_CHAINS;
            return { ...current, [path]: { status: "ready", chainId, inspection: report } };
          });
        } catch (exc) {
          setBatchPreflights((current) => !currentRequest() || !current[path] ? current : {
            ...current,
            [path]: { status: "failed", chainId: current[path].chainId,
              error: exc instanceof Error ? exc.message : String(exc) }
          });
        }
      }));
    }
  }

  function replaceBatchInputs(paths: string[]) {
    const next = uniquePaths(paths);
    for (const path of batchInspectionRequests.current.keys()) {
      if (!next.includes(path)) batchInspectionRequests.current.delete(path);
    }
    const added = next.filter((path) => !batchInputs.includes(path));
    setBatchInputs(next);
    setBatchPreflights((current) => {
      const retained = Object.fromEntries(
        Object.entries(current).filter(([path]) => next.includes(path))
      );
      for (const path of added) {
        retained[path] = { status: "checking", chainId: ALL_CHAINS };
      }
      return retained;
    });
    if (added.length) {
      void inspectBatchInputs(added);
    }
  }

  function recheckBatchInput(path: string) {
    setBatchPreflights((current) => ({
      ...current,
      [path]: { status: "checking", chainId: current[path]?.chainId ?? ALL_CHAINS }
    }));
    void inspectBatchInputs([path]);
  }

  async function refresh() {
    try {
      const next = await getStatus();
      applyStatus(next);
      return next;
    } catch (exc) {
      setBackendConnectionLost(true);
      throw exc;
    }
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

  function recordActiveRuntimeTest(report: Record<string, unknown>) {
    setEnvTest(report);
    if (report.ok === false) {
      setStatus((current) => current ? statusWithRuntimeTest(current, report) : current);
    }
  }

  async function runEnvironmentTest() {
    if (pendingAction || batchActive || downloadActive) return;
    setError(""); setMessage(""); setPendingAction("Testing runtime…");
    try {
      const report = await testBackend();
      recordActiveRuntimeTest(report);
      applyStatus(statusWithRuntimeTest(await refresh(), report));
      if (report.ok === true) setMessage("Environment test passed.");
      else setError("Runtime test failed. See checks and output below.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally { setPendingAction(""); }
  }

  async function applyAndTestBackend() {
    if (pendingAction || batchActive || downloadActive) return;
    setError(""); setMessage(""); setPendingAction("Testing selection…");
    try {
      const candidate = await testBackend(backendMode, backendMode === "conda" ? condaPython : undefined, false);
      setEnvTest(candidate);
      if (candidate.ok !== true) {
        setTab("diagnostics");
        throw new Error("Runtime test failed; saved configuration unchanged. See Diagnostics.");
      }
      await configureBackend(backendMode, backendMode === "conda" ? condaPython : undefined, proxyUrl);
      setPendingAction("Activating runtime…");
      setBackendConnectionLost(true);
      await invoke("stop_backend");
      const backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
      configureDesktopApi(backend.token, backend.port);
      await waitForBackendStatus();
      const report = await testBackend();
      recordActiveRuntimeTest(report);
      applyStatus(statusWithRuntimeTest(await getStatus(), report), true);
      if (report.ok !== true) throw new Error("The activated runtime needs attention. Review its full report in Diagnostics.");
      setMessage("Runtime activated.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally { setPendingAction(""); }
  }

  async function preparePrediction() {
    if (pendingAction || batchActive || downloadActive) return;
    if (!backendConnectionLost && backendIsHealthy(status)) {
      await startEsmDownload(false);
    } else {
      await installAndActivateBackend("cpu", true);
    }
  }

  async function installAndActivateBackend(mode: "cpu" | "gpu", prepareAssets = false) {
    if (pendingAction) {
      return;
    }
    setError("");
    setMessage("");
    setPendingAction(`Installing ${mode.toUpperCase()} backend...`);
    try {
      await invoke("install_backend", { mode, proxyUrl: proxyUrl || undefined });
      setPendingAction("Starting runtime…");
      setBackendConnectionLost(true);
      await invoke("stop_backend");
      const backend = await invoke<BackendStartResult>("start_backend", { port: 0, mode });
      configureDesktopApi(backend.token, backend.port);
      await waitForBackendStatus();
      await configureBackend(mode, undefined, proxyUrl);
      setPendingAction("Testing runtime…");
      const test = await testBackend(mode);
      recordActiveRuntimeTest(test);
      const next = statusWithRuntimeTest(await refresh(), test);
      applyStatus(next, true);
      if (test.ok !== true) {
        throw new Error("The backend was installed but its environment test failed. Open Diagnostics for details.");
      }
      if (prepareAssets && !next.assets.esm.verified) {
        setPendingAction("Starting download…");
        setAssetDownload(await downloadEsm(false));
        setMessage("Runtime ready · ESM-C download started.");
      } else {
        setMessage(`${mode.toUpperCase()} backend installed, activated, and tested.`);
      }
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
    setPendingAction("Starting model download...");
    try {
      const job = await downloadEsm(force);
      setAssetDownload(job);
      setMessage("ESM-C download started · resume supported.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally { setPendingAction(""); }
  }

  async function pauseAssetDownload() {
    const job = assetDownload;
    if (!job || !["queued", "running"].includes(job.status) || assetPauseRequest.current === job.id) return;
    assetPauseRequest.current = job.id;
    setAssetDownload((current) => current?.id === job.id ? { ...current, status: "cancelling" } : current);
    try {
      const paused = await withRequestDeadline((signal) => cancelEsmDownload(job.id, signal));
      setAssetDownload((current) => current?.id !== job.id || current.status === "completed" ? current : paused);
    } catch (exc) {
      setAssetDownload((current) => current?.id === job.id && current.status === "cancelling" ? job : current);
      setError(`Could not pause download. ${exc instanceof Error ? exc.message : String(exc)}`);
    } finally {
      if (assetPauseRequest.current === job.id) assetPauseRequest.current = null;
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
      setBackendConnectionLost(true);
      await invoke("stop_backend");
      const backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
      configureDesktopApi(backend.token, backend.port);
      await waitForBackendStatus();
      setMessage("Runtime restarted.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setPendingAction("");
    }
  }

  async function openLogs() {
    try {
      await invoke("open_logs");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function cancelSinglePrediction() {
    const operation = predictionRequest.current;
    if (!operation || operation.cancelled) return;
    if (!await confirm("Cancel this prediction and restart the runtime? Completed results are kept. Model loading will run again for the next prediction.", { title: "Cancel prediction", kind: "warning" })) return;
    // The prediction may have finished while the confirmation was open.
    if (predictionRequest.current !== operation) return;
    operation.cancelled = true;
    operation.controller.abort();
    setSingleRunning(false);
    setPendingAction("Cancelling prediction and restarting runtime...");
    setStatus(null);
    try {
      setBackendConnectionLost(true);
      await invoke("stop_backend");
      const backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
      configureDesktopApi(backend.token, backend.port);
      await waitForBackendStatus();
      setMessage("Prediction cancelled. Runtime ready.");
    } catch (exc) {
      setBackendConnectionLost(true);
      setError(`Prediction stopped. Restart or reinstall the runtime from Setup. ${String(exc)}`);
    } finally {
      predictionRequest.current = null;
      setPendingAction("");
    }
  }

  useEffect(() => {
    if (!singleRunning) return;
    let stopped = false;
    let inFlight = false;
    const started = Date.now();
    setSingleElapsed(0);
    const timer = window.setInterval(async () => {
      setSingleElapsed(Math.floor((Date.now() - started) / 1000));
      if (inFlight) return;
      inFlight = true;
      try {
        const progress = await withRequestDeadline((signal) => getPredictionProgress(signal));
        if (!stopped) setSingleProgress(progress.stage);
      } catch {
        if (!stopped) setSingleProgress("Waiting for the runtime. You can cancel and restart if it is unresponsive.");
      } finally {
        inFlight = false;
      }
    }, 1000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [singleRunning]);

  async function openExample() {
    const request = ++resultLoadRevision.current;
    const navigation = navigationRevision.current;
    navigationTouched.current = true;
    setError("");
    try {
      const example = await import("./exampleResult");
      if (request !== resultLoadRevision.current) return;
      setPrediction(example.exampleResult);
      setExampleStructureData(example.exampleStructureData);
      setBatchResult(null);
      if (navigation === navigationRevision.current) setTab("results");
      setMessage("Crambin example loaded.");
    } catch (exc) {
      if (request === resultLoadRevision.current) setError(`Could not open the bundled example: ${String(exc)}`);
    }
  }

  async function openExistingResult() {
    if (pendingAction) return;
    if (!status || backendConnectionLost) {
      setMessage("Start a runtime from Setup to open saved results. Model weights are not required for result exploration.");
      navigate("setup");
      return;
    }
    const request = ++resultLoadRevision.current;
    const navigation = navigationRevision.current;
    navigationTouched.current = true;
    setError(""); setMessage(""); setPendingAction("Opening result package...");
    try {
      const selected = await browseFiles({ multiple: false, filters: [{ name: "ProtCross summary", extensions: ["json"] }] });
      if (typeof selected !== "string" || request !== resultLoadRevision.current) return;
      const result = await openResult(selected);
      if (request !== resultLoadRevision.current) return;
      setExampleStructureData(undefined);
      setPrediction(result);
      setBatchResult(null);
      setMessage(`Opened ${fileName(selected)}.`);
      if (navigation === navigationRevision.current) setTab("results");
    } catch (exc) {
      if (request === resultLoadRevision.current) setError(exc instanceof Error ? exc.message : String(exc));
    } finally { setPendingAction(""); }
  }

  function openLatestPrediction() {
    if (!lastPrediction) return;
    resultLoadRevision.current += 1;
    setPrediction(lastPrediction);
    setBatchResult(null);
    setExampleStructureData(undefined);
    navigate("results");
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
          setEnvTest({ ok: true, device: "cpu", checks: { protcross: APP_VERSION, torch: "2.3.1" } });
        }
        return;
      }
      try {
        const backend = await invoke<BackendStartResult>("start_backend", { port: 0 });
        configureDesktopApi(backend.token, backend.port);
        const next = await waitForBackendStatus();
        const batchNeedsAttention = next.activity?.batch_jobs?.some((job) => (
          ["queued", "running", "interrupted"].includes(job.status)
        ));
        if (!navigationTouched.current && batchNeedsAttention) {
          setTab("batch");
        } else if (!navigationTouched.current && next?.readiness?.ready) {
          setTab("predict");
        }
      } catch (exc) {
        const detail = exc instanceof Error ? exc.message : String(exc);
        setBackendConnectionLost(true);
        setError(
          `Runtime unavailable. Open Setup to prepare or reconnect it. Details: ${detail}`
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
          setError(`Download connection lost: ${detail}`);
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
    if (!inputPath || !status || backendConnectionLost) {
      setInspection(null);
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
  }, [inputPath, chainSelection, Boolean(status), backendConnectionLost, inspectionRevision]);

  useEffect(() => {
    if (!batchJob || !["queued", "running"].includes(batchJob.status)) {
      return;
    }
    const jobId = batchJob.id;
    let cancelled = false;
    let inFlight = false;
    let consecutiveFailures = 0;
    const timer = window.setInterval(async () => {
      if (inFlight || batchPagePending.current) {
        return;
      }
      inFlight = true;
      const revision = batchViewRevision.current;
      try {
        const next = await withRequestDeadline(
          (signal) => getBatch(jobId, BATCH_PAGE_SIZE, batchPageOffset, signal)
        );
        if (cancelled || revision !== batchViewRevision.current) {
          return;
        }
        consecutiveFailures = 0;
        setBackendConnectionLost(false);
        setBatchJob(next);
        rememberBatchJob(next);
        setBatchPageOffset(next.items_offset ?? batchPageOffset);
      } catch (exc) {
        if (cancelled || revision !== batchViewRevision.current) {
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
          setError(`Batch connection lost: ${detail}`);
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
  }, [batchJob?.id, batchJob?.status, batchPageOffset, batchPageLoading]);

  useEffect(() => {
    if (
      !batchJob
      || ["queued", "running"].includes(batchJob.status)
      || (batchJob.item_count ?? 0) === 0
      || batchJob.items.length > 0
    ) {
      return;
    }
    const jobId = batchJob.id;
    let cancelled = false;
    getBatch(jobId, BATCH_PAGE_SIZE, 0)
      .then((job) => {
        if (!cancelled) {
          setBatchJob(job);
          rememberBatchJob(job);
          setBatchPageOffset(job.items_offset ?? 0);
        }
      })
      .catch((exc) => {
        if (!cancelled) {
          setError(exc instanceof Error ? exc.message : String(exc));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [batchJob?.id, batchJob?.status, batchJob?.item_count, batchJob?.items.length]);

  async function stopCurrentBatch() {
    const job = batchJob;
    if (!job || !["queued", "running"].includes(job.status) || job.cancel_requested || batchCancelRequest.current === job.id) return;
    const revision = batchViewRevision.current;
    const offset = job.items_offset ?? batchPageOffset;
    batchCancelRequest.current = job.id;
    setBatchCancelPending(job.id);
    try {
      await withRequestDeadline((signal) => cancelBatch(job.id, signal));
      // A cancellation acknowledgement may contain an older, first-page snapshot.
      // Read the requested page again rather than replacing the current view with it.
      const current = await withRequestDeadline((signal) => getBatch(job.id, BATCH_PAGE_SIZE, offset, signal));
      rememberBatchJob(current);
      if (revision === batchViewRevision.current) {
        setBatchJob((shown) => shown?.id === job.id ? current : shown);
      }
    } catch (exc) {
      if (revision === batchViewRevision.current) setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      if (batchCancelRequest.current === job.id) {
        batchCancelRequest.current = null;
        setBatchCancelPending(null);
      }
    }
  }

  function beginBatchView() {
    batchPagePending.current = false;
    setBatchPageLoading(false);
    return ++batchViewRevision.current;
  }

  async function loadBatchPage(offset: number) {
    if (!batchJob || pendingAction || batchPagePending.current) return;
    const revision = beginBatchView();
    const jobId = batchJob.id;
    batchPagePending.current = true;
    setBatchPageLoading(true);
    setError("");
    try {
      const next = await withRequestDeadline((signal) => getBatch(jobId, BATCH_PAGE_SIZE, Math.max(0, offset), signal));
      if (revision !== batchViewRevision.current) return;
      setBatchJob(next);
      rememberBatchJob(next);
      setBatchPageOffset(next.items_offset ?? Math.max(0, offset));
    } catch (exc) {
      if (revision === batchViewRevision.current) setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      if (revision === batchViewRevision.current) {
        batchPagePending.current = false;
        setBatchPageLoading(false);
      }
    }
  }

  const setupIssues = useMemo(
    () => backendConnectionLost ? ["The runtime is offline. Restart or reinstall it from Setup."] : readinessIssues(status),
    [status, backendConnectionLost]
  );
  const ready = setupIssues.length === 0;
  const batchActive = batchJob ? ["queued", "running"].includes(batchJob.status) : false;
  const resultStructure = prediction?.output_files.structure ?? batchResult?.output_files.structure;
  const resultOutputFiles = prediction?.output_files ?? batchResult?.output_files;
  const resultSummary = prediction?.summary ?? batchResult?.summary ?? null;
  const resultPockets = prediction?.pockets ?? batchResult?.pockets ?? null;
  const resultScores = prediction?.scores ?? batchResult?.scores ?? [];
  const resultResidues = prediction?.top_pocket_residues ?? batchResult?.top_pocket_residues ?? [];
  const topResidues = useMemo(() => {
    const residues = resultResidues.length ? resultResidues : ((resultSummary?.top_residues ?? []) as ResidueSummary[]);
    return residues;
  }, [resultResidues, resultSummary]);
  const downloadActive = ["queued", "running", "cancelling"].includes(assetDownload?.status ?? "");
  const activityLabel = pendingAction
    || (downloadActive && assetDownload ? downloadPhaseLabel(assetDownload) : "")
    || (batchActive && batchJob ? `Batch prediction · ${batchJob.completed}/${batchJob.item_count ?? batchJob.items.length}` : "");

  useEffect(() => {
    if (!UI_PREVIEW_TAB) {
      void invoke("set_activity", { active: Boolean(activityLabel) }).catch((exc) => console.error("Could not update task activity", exc));
    }
  }, [Boolean(activityLabel)]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace-content">Skip to content</a>
      <header className="app-header">
        <div className="brand"><h1>ProtCross</h1><span>v{APP_VERSION}</span></div>
        <nav className="primary-nav" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => (
            <button key={item.id} aria-current={tab === item.id ? "page" : undefined}
              className={`${tab === item.id ? "active" : ""} ${item.id === "setup" ? "nav-secondary" : ""}`}
              onClick={() => navigate(item.id)}>
              <span className="nav-compact-label">{item.label}</span>
              {item.id === "batch" && batchActive ? <span className="nav-badge">{batchJob?.completed}/{batchJob?.item_count ?? batchJob?.items.length}</span> : null}
            </button>
          ))}
        </nav>
        <div className="header-actions">
          <button className={`readiness-card ${ready ? "ready" : "attention"}`}
            title={ready ? `Ready to predict · ${backendDisplayName(status?.backend.mode)}` : setupIssues[0]}
            onClick={() => navigate("setup")}>
            <span className="status-dot" aria-hidden="true" />{ready ? "Ready" : "Setup required"}
          </button>
          <details className="app-options" ref={workspaceOptionsRef}>
            <summary aria-label="Workspace options" title="Workspace options"><Icon name="settings" size={17} /></summary>
            <div className="app-options-menu">
              <label className="field"><span>Theme</span><select aria-label="Appearance" value={themePreference} onChange={(event) => setThemePreference(event.target.value as ThemePreference)}>
                <option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option>
              </select></label>
              <button aria-label="Refresh runtime status" onClick={() => refresh().catch((exc) => setError(String(exc)))}><Icon name="refresh" size={16} /> Refresh status</button>
            </div>
          </details>
        </div>
      </header>
      <section className="workspace">
        {activityLabel ? (
          <div className="activity-strip" role="status" aria-live="polite">
            <span className="button-spinner" aria-hidden="true" />
            <strong>{activityLabel}</strong>
            {batchActive && tab !== "batch" ? <button onClick={() => navigate("batch")}>View batch</button> : null}
          </div>
        ) : null}

        <div className="notification-region">
          {message ? (
            <div className="banner success" role="status">
              <Icon name="check" />
              <span>{message}</span>
              {message === "Prediction finished." && lastPrediction && (prediction !== lastPrediction || tab !== "results") ? <button onClick={openLatestPrediction}>View prediction</button> : null}
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

        <main className={`content content-${tab}`} id="workspace-content" tabIndex={-1}>
          <h2 className="sr-only">{labelForTab(tab)}</h2>

        {tab === "setup" ? (
          <SetupPanel
            onBrowse={browseFiles}
            status={status}
            backendMode={backendMode}
            setBackendMode={(mode) => { runtimeDraftTouched.current = true; setBackendMode(mode); }}
            busy={Boolean(pendingAction)}
            pendingAction={pendingAction}
            onOpenLogs={openLogs}
            onPrepare={() => void preparePrediction()}
            onOpenSample={() => void openExample()}
            onOpenExisting={() => void openExistingResult()}
            assetDownload={assetDownload}
            setupIssues={setupIssues}
            condaPython={condaPython}
            setCondaPython={(value) => { runtimeDraftTouched.current = true; setCondaPython(value); }}
            proxyUrl={proxyUrl}
            setProxyUrl={(value) => { runtimeDraftTouched.current = true; setProxyUrl(value); }}
            onInstallBackend={(mode) => void installAndActivateBackend(mode)}
            onImportEsm={async () => {
              const selected = await browseFiles({ multiple: false, filters: [{ name: "ESM-C weights", extensions: ["pth"] }] });
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
              const selected = await browseFiles({ multiple: false, filters: [{ name: "ProtCross checkpoint", extensions: ["ckpt"] }] });
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
              const selected = await browseFiles({ multiple: false, filters: [{ name: "ProtCross PCA", extensions: ["pkl"] }] });
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
            onCancelEsm={() => void pauseAssetDownload()}
            runtimeLocked={batchActive || downloadActive}
            backendConnectionLost={backendConnectionLost}
            onRestartBackend={restartBackend}
            onContinue={() => navigate("predict")}
            onTestBackend={() => void applyAndTestBackend()}
          />
        ) : null}

        {tab === "predict" ? (
          <PredictPanel
            onBrowse={browseFiles}
            ready={ready}
            setupIssues={setupIssues}
            busy={Boolean(pendingAction)}
            inputPath={inputPath}
            setInputPath={(value) => {
              setInputPath(value);
              setChainSelection(ALL_CHAINS);
              setInspection(null);
              setInspectionError("");
              setInspectionRevision((revision) => revision + 1);
            }}
            onRecheck={() => {
              setInspection(null);
              setInspectionError("");
              setInspectionRevision((revision) => revision + 1);
            }}
            singleRunning={singleRunning}
            singleProgress={singleProgress}
            singleElapsed={singleElapsed}
            onCancel={cancelSinglePrediction}
            otherTaskActive={batchActive || downloadActive}
            chainSelection={chainSelection}
            setChainSelection={setChainSelection}
            inspection={inspection}
            inspectionError={inspectionError}
            inspecting={inspecting}
            outputDir={outputDir}
            setOutputDir={setOutputDir}
            defaultOutputRoot={status?.paths.outputs_dir}
            parameters={parameters}
            setParameters={setParameters}
            onResetParameters={() => setError("")}
            onOpenSetup={() => navigate("setup")}
            onRun={async () => {
              if (pendingAction || batchActive || downloadActive || parameterError || !ready) return;
              const resultRequest = ++resultLoadRevision.current;
              const navigation = navigationRevision.current;
              const operation = { controller: new AbortController(), cancelled: false };
              predictionRequest.current = operation;
              setError("");
              setMessage("");
              setPendingAction("Running prediction...");
              setSingleRunning(true);
              setSingleProgress("Preparing prediction...");
              try {
                validatePredictionInputs(inputPath, Number(threshold), Number(clusterCutoff));
                const result = await runPrediction({
                  input_structure: inputPath,
                  output_dir: outputDir || undefined,
                  threshold: Number(threshold),
                  pocket_cluster_cutoff: Number(clusterCutoff),
                  chain_id: chainSelection === ALL_CHAINS ? undefined : chainSelection,
                  allow_truncation: allowTruncation,
                  device: device || undefined
                }, operation.controller.signal);
                if (operation.cancelled) return;
                setLastPrediction(result);
                if (resultRequest === resultLoadRevision.current) {
                  setExampleStructureData(undefined);
                  setPrediction(result);
                  setBatchResult(null);
                  if (navigation === navigationRevision.current) setTab("results");
                }
                setMessage("Prediction finished.");
              } catch (exc) {
                if (!operation.cancelled) setError(exc instanceof Error ? exc.message : String(exc));
              } finally {
                if (!operation.cancelled) {
                  predictionRequest.current = null;
                  setSingleRunning(false);
                  setPendingAction("");
                }
              }
            }}
          />
        ) : null}

        {tab === "batch" ? (
          <BatchPanel
            onBrowse={browseFiles}
            ready={ready}
            setupIssues={setupIssues}
            busy={Boolean(pendingAction)}
            batchInputs={batchInputs}
            setBatchInputs={replaceBatchInputs}
            batchPreflights={batchPreflights}
            setBatchChain={(path, chainId) => setBatchPreflights((current) => ({
              ...current,
              [path]: { ...current[path], chainId }
            }))}
            onRecheckInput={recheckBatchInput}
            outputDir={outputDir}
            setOutputDir={setOutputDir}
            defaultOutputRoot={status?.paths.outputs_dir}
            parameters={parameters}
            setParameters={setParameters}
            onResetParameters={() => setError("")}
            batchJob={batchJob}
            batchHistory={batchHistory}
            batchActive={batchActive}
            otherTaskActive={singleRunning || downloadActive}
            onOpenSetup={() => navigate("setup")}
            batchPageSize={BATCH_PAGE_SIZE}
            batchPageOffset={batchPageOffset}
            pageLoading={batchPageLoading}
            cancelling={batchCancelPending === batchJob?.id}
            onViewItem={async (item) => {
              if (!batchJob || pendingAction) return;
              const request = ++resultLoadRevision.current;
              const navigation = navigationRevision.current;
              navigationTouched.current = true;
              setError(""); setPendingAction("Opening batch result...");
              try {
                const detail = await getBatchResult(batchJob.id, item.input_structure, item.chain_id);
                if (request !== resultLoadRevision.current) return;
                setExampleStructureData(undefined);
                setBatchResult(detail);
                setPrediction(null);
                if (navigation === navigationRevision.current) setTab("results");
              } catch (exc) {
                if (request === resultLoadRevision.current) setError(exc instanceof Error ? exc.message : String(exc));
              } finally { setPendingAction(""); }
            }}
            onSubmit={async () => {
              if (pendingAction || batchActive || downloadActive || batchParameterError || !ready) {
                return;
              }
              setError("");
              try {
                if (batchInputs.length === 0) {
                  throw new Error("Select at least one structure for batch prediction.");
                }
                const unchecked = batchInputs.filter((path) => batchPreflights[path]?.status !== "ready");
                if (unchecked.length) {
                  throw new Error("Finish the structure checks and resolve every failed precheck before starting the batch.");
                }
                validatePredictionInputs(batchInputs[0], Number(threshold), Number(clusterCutoff));
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
                return;
              }
              beginBatchView();
              setPendingAction("Starting batch...");
              try {
                const job = await submitBatch({
                  items: batchInputs.map((path) => ({
                    input_structure: path,
                    chain_id: batchPreflights[path].chainId === ALL_CHAINS
                      ? undefined
                      : batchPreflights[path].chainId
                  })),
                  output_dir: outputDir || undefined,
                  threshold: Number(threshold),
                  pocket_cluster_cutoff: Number(clusterCutoff),
                  allow_truncation: allowTruncation,
                  device: device || undefined,
                  batch_size: Number(batchSize)
                });
                setBatchJob(job);
                rememberBatchJob(job);
                setBatchPageOffset(job.items_offset ?? 0);
                setBatchResult(null);
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              } finally {
                setPendingAction("");
              }
            }}
            onCancel={() => void stopCurrentBatch()}
            onRetryFailed={async () => {
              if (!batchJob || pendingAction || batchActive || downloadActive || !ready) {
                return;
              }
              setError("");
              beginBatchView();
              setPendingAction("Starting unfinished structures...");
              try {
                const retry = await retryBatch(batchJob.id);
                setBatchJob(retry);
                setBatchPageOffset(retry.items_offset ?? 0);
                rememberBatchJob(retry);
                setMessage(`Started retry batch ${retry.id}.`);
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              } finally {
                setPendingAction("");
              }
            }}
            onSelectHistory={async (jobId) => {
              if (pendingAction || batchActive && batchJob?.id !== jobId) {
                return;
              }
              setError("");
              const revision = beginBatchView();
              setPendingAction("Loading batch history...");
              try {
                const selected = await withRequestDeadline((signal) => getBatch(jobId, BATCH_PAGE_SIZE, 0, signal));
                if (revision !== batchViewRevision.current) return;
                setBatchJob(selected);
                rememberBatchJob(selected);
                setBatchPageOffset(selected.items_offset ?? 0);
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : String(exc));
              } finally {
                setPendingAction("");
              }
            }}
            onPageChange={(offset) => void loadBatchPage(offset)}
          />
        ) : null}

        <div hidden={tab !== "results"}>
          <ResultsPanel
            onPrepareRuntime={() => void installAndActivateBackend("cpu", false)}
            active={tab === "results"}
            busy={Boolean(pendingAction)}
            onOpenLatest={lastPrediction && prediction !== lastPrediction ? openLatestPrediction : undefined}
            connected={Boolean(status) && !backendConnectionLost}
            onOpenSetup={() => navigate("setup")}
            onOpenSample={() => void openExample()}
            sample={exampleStructureData !== undefined}
            structureData={exampleStructureData}
            structurePath={resultStructure}
            outputFiles={resultOutputFiles}
            summary={resultSummary}
            pockets={resultPockets}
            scores={resultScores}
            residues={topResidues}
            darkMode={themePreference === "dark" || (themePreference === "system" && systemDarkMode)}
            onOpenExisting={() => void openExistingResult()}
            onNotify={setMessage}
            onError={setError}
          />
        </div>

        {tab === "diagnostics" ? (
          <DiagnosticsPanel
            status={status}
            envTest={envTest}
            connected={Boolean(status) && !backendConnectionLost}
            busy={Boolean(pendingAction) || batchActive || downloadActive}
            onOpenLogs={openLogs}
            onTest={() => void runEnvironmentTest()}
            onExport={() => void runAction(async () => {
              const result = await exportDiagnostics();
              await invoke("open_path", { path: parentPath(result.path) });
            }, "Diagnostic package saved and its folder opened.", "Exporting diagnostics...")}
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

function PredictPanel(props: {
  onBrowse: BrowseFiles;
  ready: boolean;
  setupIssues: string[];
  busy: boolean;
  onRecheck: () => void;
  singleRunning: boolean;
  singleProgress: string;
  singleElapsed: number;
  onCancel: () => void;
  otherTaskActive: boolean;
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
  parameters: PredictionParameters;
  setParameters: (value: PredictionParameters) => void;
  onResetParameters: () => void;
  onOpenSetup: () => void;
  onRun: () => void;
}) {
  const truncationBlocked = Boolean(props.inspection?.requires_truncation && !props.parameters.allowTruncation);
  const parameterError = parameterValidationMessage(props.parameters);
  return (
    <div className="predict-layout">
      <div className="workspace-toolbar"><h3>Single structure</h3><span className="format-label">PDB / mmCIF</span></div>
      {!props.ready ? <div className="callout warning"><Icon name="warning" /><span>{props.setupIssues[0]}</span><button onClick={props.onOpenSetup}>Open setup</button></div> : null}
      {props.otherTaskActive ? <p className="field-help">Runtime busy. Finish the batch or pause the download.</p> : null}
      {props.singleRunning ? <div className="prediction-progress"><span className="button-spinner" /><strong role="status">{props.singleProgress}</strong><span aria-live="off">{props.singleElapsed}s</span><button onClick={props.onCancel}>Cancel prediction</button></div> : null}
      <section className="prediction-form">
        <PathInput onBrowse={props.onBrowse} label="Structure file" value={props.inputPath} setValue={props.setInputPath} kind="file" prominent disabled={props.busy} />
        <div className="structure-preview">
          <StructureInspectionCard inspection={props.inspection} error={props.inspectionError} inspecting={props.inspecting}
            chainSelection={props.chainSelection} setChainSelection={props.setChainSelection} disabled={props.busy}
            onRecheck={props.onRecheck} canRecheck={Boolean(props.inputPath) && !props.inspecting && !props.busy} />
        </div>
        <AdvancedParameters values={props.parameters} onChange={props.setParameters} onReset={props.onResetParameters} disabled={props.busy} title="Prediction settings" />
        <details className="disclosure output-disclosure">
          <summary>Output <span>{props.outputDir ? fileName(props.outputDir) : "Automatic"}</span></summary>
          <div className="disclosure-content">
            <PathInput onBrowse={props.onBrowse} label="Output directory" value={props.outputDir} setValue={props.setOutputDir} kind="directory" disabled={props.busy} />
            <p className="field-help">Default: <code>{props.defaultOutputRoot ?? "application-data/outputs"}</code> · unique folder per run.</p>
            {props.outputDir ? <button disabled={props.busy} onClick={() => props.setOutputDir("")}>Use default</button> : null}
          </div>
        </details>
        {truncationBlocked ? <div className="inline-error" role="alert">Chain exceeds 1,022 residues. Enable truncation or select a shorter chain.</div> : null}
        {parameterError ? <p className="inline-error">{parameterError}</p> : null}
        <div className="form-action-bar">
          <span className="field-help">{props.inspection ? `${props.inspection.scorable_residue_count} residues · ${props.inspection.selected_chains.length} chain${props.inspection.selected_chains.length === 1 ? "" : "s"}` : "Select a structure to begin"}</span>
          <button className="primary-action run-action" disabled={props.busy || props.otherTaskActive || !props.ready || !props.inputPath || props.inspecting || Boolean(props.inspectionError) || !props.inspection || truncationBlocked || Boolean(parameterError)} onClick={props.onRun}>
            {props.singleRunning ? <><span className="button-spinner" /> Running</> : <><Icon name="play" size={16} /> Run prediction</>}
          </button>
        </div>
      </section>
    </div>
  );
}

function StructureInspectionCard(props: {
  disabled?: boolean;
  inspection: StructureInspection | null;
  error: string;
  inspecting: boolean;
  chainSelection: string;
  setChainSelection: (value: string) => void;
  onRecheck: () => void;
  canRecheck: boolean;
}) {
  if (props.inspecting) return <div className="structure-empty" role="status"><span className="button-spinner" /> Checking structure…</div>;
  if (!props.inspection) return props.error
    ? <div className="inline-error" role="alert"><span>{props.error}</span><button disabled={!props.canRecheck} onClick={props.onRecheck}>Check again</button></div>
    : null;
  const report = props.inspection;
  return (
    <div className="structure-check">
      <div className="structure-check-heading">
        <span className={`inspection-status ${report.warnings.length ? "warning" : "success"}`}><Icon name={report.warnings.length ? "warning" : "check"} size={16} />{report.warnings.length ? "Check warnings" : "Checked"}</span>
        <label className="compact-field"><span>Chains to analyze</span><select disabled={props.disabled} value={props.chainSelection} onChange={(event) => props.setChainSelection(event.target.value)}>
          <option value={ALL_CHAINS}>All scorable chains</option>
          {report.available_chains.map((chain) => <option value={chain} key={chain || "blank-chain"}>Chain {displayChain(chain)}</option>)}
        </select></label>
        <span className="structure-format">{report.format} · model 1/{report.model_count}</span>
        <button className="icon-button subtle" aria-label="Check again" title="Check again" disabled={!props.canRecheck} onClick={props.onRecheck}><Icon name="refresh" size={16} /></button>
      </div>
      {props.error ? <div className="inline-error" role="alert">{props.error}</div> : null}
      {report.warnings.length ? <div className="warning-list">{report.warnings.map((warning) => <div key={warning}><Icon name="warning" size={15} />{warning}</div>)}</div> : null}
      <details className="disclosure coordinate-details">
        <summary>Coordinate details <span>{report.longest_chain_context} aa max context</span></summary>
        <div className="inspection-metrics">
          <Metric label="Missing Cα" value={String(report.standard_residues_missing_ca)} />
          <Metric label="Modified residues" value={String(report.modified_or_nonstandard_amino_acids)} />
          <Metric label="Coordinate breaks" value={String(report.sequence_break_count)} />
          <Metric label="Numbering gaps" value={String(report.numbering_gap_count)} />
        </div>
        <p className="field-help">First coordinate model only. Selected chains share one geometry graph. A single-chain selection limits sequence context and geometry; use a subset file for custom multi-chain scope.</p>
      </details>
    </div>
  );
}

function BatchPanel(props: {
  onBrowse: BrowseFiles;
  ready: boolean;
  setupIssues: string[];
  busy: boolean;
  batchInputs: string[];
  setBatchInputs: (value: string[]) => void;
  batchPreflights: Record<string, BatchPreflight>;
  setBatchChain: (path: string, chainId: string) => void;
  onRecheckInput: (path: string) => void;
  outputDir: string;
  setOutputDir: (value: string) => void;
  defaultOutputRoot?: string;
  parameters: PredictionParameters;
  setParameters: (value: PredictionParameters) => void;
  onResetParameters: () => void;
  batchJob: BatchJob | null;
  batchHistory: BatchJob[];
  batchActive: boolean;
  otherTaskActive: boolean;
  onOpenSetup: () => void;
  batchPageSize: number;
  batchPageOffset: number;
  pageLoading: boolean;
  cancelling: boolean;
  onViewItem: (item: BatchJob["items"][number]) => void | Promise<void>;
  onSubmit: () => void;
  onCancel: () => void;
  onRetryFailed: () => void;
  onSelectHistory: (jobId: string) => void;
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
  const progress = itemCount ? Math.min(100, (processed / itemCount) * 100) : 0;
  const preflightsReady = props.batchInputs.length > 0
    && props.batchInputs.every((path) => props.batchPreflights[path]?.status === "ready");
  const truncationBlocked = !props.parameters.allowTruncation
    && props.batchInputs.some((path) => batchScopeRequiresTruncation(props.batchPreflights[path]));
  const retryable = Boolean(
    props.batchJob
    && (
      props.batchJob.failed > 0
      || (["interrupted", "cancelled"].includes(props.batchJob.status) && processed < itemCount)
    )
    && !["queued", "running"].includes(props.batchJob.status)
  );
  return (
    <div className="batch-layout">
      {!props.ready ? <div className="callout warning span-all"><Icon name="warning" /><div><strong>Setup required</strong><span>{props.setupIssues[0] ?? "Complete environment setup first."}</span></div><button onClick={props.onOpenSetup}>Open setup</button></div> : null}
      {props.otherTaskActive ? <p className="field-help span-all">Runtime busy. Finish the prediction or pause the download.</p> : null}
      {props.batchJob ? (
        <section className="panel batch-monitor span-all">
          <div className="batch-monitor-header">
            <div className="batch-run-title"><h3>Batch {shortJobId(props.batchJob.id)}</h3><StatusPill status={props.batchJob.status} /></div>
            <div className="button-row">
              {retryable ? <button className="primary-action" disabled={props.busy || props.otherTaskActive || !props.ready} onClick={props.onRetryFailed}><Icon name="refresh" /> {props.batchJob.status === "cancelled" ? "Continue remaining" : props.batchJob.status === "interrupted" ? "Retry unfinished" : "Retry failed"}</button> : null}
              <button className="danger-action" disabled={props.cancelling || props.batchJob.cancel_requested || !["queued", "running"].includes(props.batchJob.status)} onClick={props.onCancel}>
                <Icon name="pause" /> {props.cancelling || props.batchJob.cancel_requested ? "Stopping…" : "Stop batch"}
              </button>
            </div>
          </div>
          {props.batchJob.settings ? <details className="disclosure compact-disclosure">
            <summary><Icon name="settings" /> Run settings</summary>
            <p className="field-help">Retries retain these settings. For a different device or group size, start a new batch with unfinished inputs.</p>
            <pre className="diagnostic-json" tabIndex={0} aria-label="Batch run settings JSON">{JSON.stringify(props.batchJob.settings, null, 2)}</pre>
          </details> : null}
          <div className="batch-progress">
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            <div className="batch-stats">
              <Metric label="Processed" value={`${processed} / ${itemCount}`} />
              <Metric label="Succeeded" value={String(successful)} tone="success" />
              <Metric label="Failed" value={String(props.batchJob.failed)} tone={props.batchJob.failed ? "danger" : undefined} />
              <Metric label="Remaining" value={String(Math.max(0, itemCount - processed))} />
            </div>
          </div>
          {props.batchJob.status === "interrupted" ? <div className="callout warning compact-callout"><Icon name="warning" /><div><strong>Interrupted</strong><span>Completed outputs kept. Retry unfinished inputs.</span></div></div> : null}
          {props.batchJob.error ? <div className="inline-error batch-error" role="alert"><Icon name="warning" /><pre>{String(props.batchJob.error)}</pre></div> : null}
          {(props.cancelling || props.batchJob.cancel_requested) && ["queued", "running"].includes(props.batchJob.status) ? <div className="callout warning compact-callout"><Icon name="info" /><div><strong>Stopping after the current group</strong><span>Queued inputs remain untouched.</span></div></div> : null}
          <div className="table-wrap" tabIndex={0} aria-label="Batch prediction results">
            <table>
              <caption className="sr-only">Batch structures and prediction status</caption>
              <thead><tr><th scope="col">Status</th><th scope="col">Structure</th><th scope="col">Output or error</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {props.batchJob.items.map((item) => (
                  <tr key={`${item.input_structure}\u0000${item.chain_id ?? ALL_CHAINS}`}>
                    <td><StatusPill status={item.status} /></td>
                    <td><span className="table-file"><strong>{fileName(item.input_structure)}</strong><small title={item.input_structure}>{parentPath(item.input_structure)} · {item.chain_id === null || item.chain_id === undefined ? "all chains" : displayChain(item.chain_id)}</small></span></td>
                    <td className={item.error ? "error-copy" : "path-copy"}>{item.error ? <pre>{item.error}</pre> : item.output_dir ?? parentPath(item.output_files?.summary_json ?? "")}</td>
                    <td><button disabled={item.status !== "completed" || !item.output_files?.summary_json || props.busy} onClick={() => void props.onViewItem(item)}>View <Icon name="arrow-right" /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager" aria-busy={props.pageLoading}>
            <button disabled={!canPrevious || props.busy || props.pageLoading} onClick={() => props.onPageChange(Math.max(0, pageOffset - props.batchPageSize))}>Previous</button>
            <span role="status">{props.pageLoading ? "Loading page…" : `${pageStart}–${pageEnd} of ${itemCount}`}</span>
            <button disabled={!canNext || props.busy || props.pageLoading} onClick={() => props.onPageChange(pageOffset + props.batchPageSize)}>Next</button>
          </div>
        </section>
      ) : null}
      <details className="batch-composer" open={!props.batchJob}>
        <summary>New batch <span>{props.batchInputs.length} inputs</span></summary>
        <div className="batch-composer-body">
      <section className="panel batch-staging">
        <div className="section-heading row-heading">
          <h3>Inputs</h3>
          <div className="button-row">
            {props.batchInputs.length ? <button disabled={props.batchActive || props.busy} onClick={() => props.setBatchInputs([])}><Icon name="trash" /> Clear</button> : null}
            <button
              className="primary-action"
              disabled={props.batchActive || props.busy}
              onClick={async () => {
                const selected = await props.onBrowse({ multiple: true, filters: [{ name: "Structures", extensions: ["pdb", "cif", "mmcif"] }] });
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
          <div className="staging-list" aria-label={`${props.batchInputs.length} selected structures`} tabIndex={0}>
            <div className="staging-summary"><strong>{props.batchInputs.length} structure{props.batchInputs.length === 1 ? "" : "s"}</strong><span>Unique files</span></div>
            {props.batchInputs.map((path) => {
              const preflight = props.batchPreflights[path];
              const scorableChains = preflight?.inspection?.chain_summaries.filter(
                (chain) => chain.scorable_residue_count > 0
              ) ?? [];
              const requiresTruncation = batchScopeRequiresTruncation(preflight);
              return (
                <div className="staging-item" key={path}>
                  <span className="file-type">{fileExtension(path)}</span>
                  <span className="file-copy"><strong>{fileName(path)}</strong><small title={path}>{parentPath(path)}</small></span>
                  <div className="staging-preflight">
                    {preflight?.status === "checking" || !preflight ? (
                      <span className="preflight-state checking"><span className="button-spinner" /> Checking structure</span>
                    ) : null}
                    {preflight?.status === "ready" ? (
                      <label className="compact-field batch-chain-field">
                        <span>Scorable chain</span>
                        <select
                          aria-label={`Scorable chain for ${fileName(path)}`}
                          value={preflight.chainId}
                          disabled={props.batchActive || props.busy}
                          onChange={(event) => props.setBatchChain(path, event.target.value)}
                        >
                          <option value={ALL_CHAINS}>All scorable chains</option>
                          {scorableChains.map((chain) => (
                            <option value={chain.chain_id} key={chain.chain_id}>
                              {displayChain(chain.chain_id)} · {chain.scorable_residue_count} residues
                            </option>
                          ))}
                        </select>
                        {requiresTruncation ? <small>Over 1,022 residues · truncation required</small> : null}
                      </label>
                    ) : null}
                    {preflight?.status === "failed" ? (
                      <div className="preflight-failure" role="alert">
                        <span>{preflight.error}</span>
                        <button disabled={props.batchActive || props.busy} onClick={() => props.onRecheckInput(path)}>Check again</button>
                      </div>
                    ) : null}
                  </div>
                  <button aria-label={`Remove ${fileName(path)}`} className="icon-button subtle" disabled={props.batchActive || props.busy} onClick={() => props.setBatchInputs(props.batchInputs.filter((item) => item !== path))}><Icon name="close" /></button>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty-dropzone">No inputs · PDB / mmCIF</div>
        )}
      </section>

      <section className="panel batch-settings">
        <h3 className="sr-only">Shared settings</h3>
        <details className="disclosure output-disclosure"><summary>Output <span>{props.outputDir ? fileName(props.outputDir) : "Automatic"}</span></summary><div className="disclosure-content">
          <PathInput onBrowse={props.onBrowse} label="Output directory" value={props.outputDir} setValue={props.setOutputDir} kind="directory" disabled={props.busy || props.batchActive} />
          <p className="field-help">Default: <code>{props.defaultOutputRoot ?? "application-data/outputs"}/batch/&lt;job-id&gt;</code></p>
        </div></details>
        <AdvancedParameters
          values={props.parameters}
          onChange={props.setParameters}
          onReset={props.onResetParameters}
          disabled={props.busy || props.batchActive}
          showMicroBatchSize
        />
        {parameterValidationMessage(props.parameters, true) ? <p className="inline-error">{parameterValidationMessage(props.parameters, true)}</p> : null}
        <div className="batch-submit-summary"><span>{props.batchInputs.length} queued</span><span>{!props.batchInputs.length ? "No inputs" : !preflightsReady ? "Check inputs" : truncationBlocked ? "Truncation required" : "Checked"}</span></div>
        <button className="primary-action run-action full-width" disabled={props.busy || props.batchActive || props.otherTaskActive || !props.ready || !preflightsReady || truncationBlocked || Boolean(parameterValidationMessage(props.parameters, true))} onClick={props.onSubmit}>
          {props.batchActive ? <><span className="button-spinner" /> Batch running</> : props.busy ? "Busy…" : <><Icon name="play" /> Run batch</>}
        </button>
      </section>

        </div>
      </details>
      {props.batchHistory.length ? (
        <details className="disclosure batch-history"><summary>History <span>{props.batchHistory.length} runs</span></summary>
          <div className="batch-history-list">
            {props.batchHistory.map((job) => (
              <button
                className={job.id === props.batchJob?.id ? "selected" : ""}
                disabled={props.busy || props.batchActive && job.id !== props.batchJob?.id}
                key={job.id}
                onClick={() => props.onSelectHistory(job.id)}
              >
                <StatusPill status={job.status} />
                <span><strong>{formatBatchTime(job.created_at)}</strong><small>{job.completed}/{job.item_count ?? job.items.length} processed · {job.failed} failed</small></span>
                {job.retry_of ? <small>Retry of {shortJobId(job.retry_of)}</small> : <small>{shortJobId(job.id)}</small>}
              </button>
            ))}
          </div>
        </details>
      ) : null}

    </div>
  );
}

function batchScopeRequiresTruncation(preflight?: BatchPreflight): boolean {
  if (preflight?.status !== "ready" || !preflight.inspection) {
    return false;
  }
  if (preflight.chainId === ALL_CHAINS) {
    return preflight.inspection.requires_truncation;
  }
  return preflight.inspection.chain_summaries.some((chain) => (
    chain.chain_id === preflight.chainId && chain.exceeds_esm_context
  ));
}

function PathInput({ label, value, setValue, kind, onBrowse, prominent = false, disabled = false }: { onBrowse: BrowseFiles; label: string; value: string; setValue: (value: string) => void; kind: "file" | "directory"; prominent?: boolean; disabled?: boolean }) {
  const id = useId();
  return (
    <div className={`field path-field ${prominent ? "prominent" : ""}`}>
      <label htmlFor={id}>{label}</label>
      <div className="path-row">
        <span className="path-leading" aria-hidden="true"><Icon name={kind === "file" ? "file" : "folder"} /></span>
        <input id={id} disabled={disabled} value={value} onChange={(event) => setValue(event.target.value)} placeholder={kind === "file" ? "Select a PDB or mmCIF file" : "Automatic"} />
        <button
          disabled={disabled}
          onClick={async () => {
            const selected = await onBrowse({
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

function Metric({ label, value, tone }: { label: string; value: string; tone?: "success" | "danger" | "accent" }) {
  return (
    <div className={`metric ${tone ? `metric-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const tone = status === "completed"
    ? "success"
    : ["failed", "completed_with_errors", "interrupted"].includes(status)
      ? "danger"
      : status === "running"
        ? "active"
        : status === "cancelled"
          ? "neutral"
          : "queued";
  return <span className={`status-pill ${tone}`}><span className="status-dot" aria-hidden="true" />{humanizeStatus(status)}</span>;
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

function statusWithRuntimeTest(status: DesktopStatus, report: Record<string, unknown>): DesktopStatus {
  if (report.ok !== false) return status;
  const issue = "Runtime test failed. Open Diagnostics for details.";
  return { ...status, backend: { ...status.backend, backend_test_ok: false },
    readiness: { ready: false, issues: [...new Set([issue, ...(status.readiness?.issues ?? [])])] } };
}

function readinessIssues(status: DesktopStatus | null): string[] {
  if (!status) {
    return ["Desktop backend is starting."];
  }
  if (status.readiness?.issues) {
    return status.readiness.issues;
  }
  const issues: string[] = [];
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

function displayChain(chainId: string): string {
  return chainId.trim() || "<blank>";
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

function humanizeStatus(status: string): string {
  return status.replace(/[_-]+/g, " ").replace(/^./, (letter) => letter.toUpperCase());
}

function formatBatchTime(createdAt?: number): string {
  if (!createdAt || !Number.isFinite(createdAt)) {
    return "Batch run";
  }
  return new Date(createdAt * 1000).toLocaleString();
}

function shortJobId(jobId: string): string {
  return jobId.length > 12 ? `${jobId.slice(0, 8)}…` : jobId;
}

function parsePreviewTab(value: string | null): Tab | null {
  return NAV_ITEMS.some((item) => item.id === value) ? value as Tab : null;
}

function previewState(tab: Tab): { status: DesktopStatus; prediction?: PredictResponse; batchJob?: BatchJob } {
  const ready = tab !== "setup";
  const fileStatus = (name: string) => ({ path: `/data/protcross/assets/${name}`, present: true, verified: true });
  const batchJob: BatchJob | undefined = tab === "batch" ? {
    id: "preview-batch",
    created_at: 1_758_729_600,
    status: "interrupted",
    completed: 3,
    failed: 1,
    cancel_requested: false,
    item_count: 6,
    items_offset: 0,
    items_returned: 6,
    items: [
      { input_structure: "/data/proteins/6fhu.pdb", chain_id: "A", status: "completed", output_dir: "/data/results/6fhu", output_files: { summary_json: "/data/results/6fhu/6fhu.protcross.summary.json" } },
      { input_structure: "/data/proteins/7abc.cif", chain_id: null, status: "completed", output_dir: "/data/results/7abc", output_files: { summary_json: "/data/results/7abc/7abc.protcross.summary.json" } },
      { input_structure: "/data/proteins/8xyz.pdb", chain_id: "B", status: "failed", error: "StructureParserError: No scorable standard amino-acid residues were found.\nCheck that chain B contains Cα coordinates." },
      { input_structure: "/data/proteins/complex_alpha.cif", chain_id: "A", status: "interrupted" },
      { input_structure: "/data/proteins/complex_beta.pdb", chain_id: null, status: "interrupted" },
      { input_structure: "/data/proteins/target_042.cif", chain_id: "C", status: "interrupted" }
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
      issues: ["Install the selected backend environment.", "Download or import ESM-C weights."]
    },
    activity: { batch_jobs: batchJob ? [batchJob] : [], asset_downloads: [] }
  };
  const firstPreviewCluster = previewResidues(1, [0.962, 0.901, 0.835, 0.744, 0.663], 0, 1);
  const secondPreviewCluster = previewResidues(2, [0.817, 0.694, 0.601], 30, 6);
  const belowThreshold = previewResidues(0, [0.481, 0.302, 0.177, 0.052], 60, 9);
  const prediction: PredictResponse | undefined = tab === "results" ? {
    ok: true,
    summary: {
      schema_version: "protcross-summary-v2",
      input_structure: "/data/proteins/6fhu.pdb",
      protcross_version: APP_VERSION,
      asset_version: "0.1.2",
      geometry_backend: "torch",
      threshold: 0.5,
      cluster_cutoff: 8.0,
      top_pocket: { cluster_id: 1, center: [12.442, -4.102, 8.774], residue_count: 5, score_mean: 0.821, score_max: 0.962 }
    },
    pockets: {
      schema_version: "protcross-pocket-v2",
      threshold: 0.5,
      cluster_cutoff: 8.0,
      selected_residue_count: 8,
      clustered_pockets: [
        { cluster_id: 1, center: [12.442, -4.102, 8.774], residue_count: 5, score_mean: 0.821, score_max: 0.962, residues: firstPreviewCluster },
        { cluster_id: 2, center: [-3.201, 14.702, 22.118], residue_count: 3, score_mean: 0.704, score_max: 0.817, residues: secondPreviewCluster }
      ]
    },
    scores: [...firstPreviewCluster, ...secondPreviewCluster, ...belowThreshold],
    top_pocket_residues: firstPreviewCluster,
    output_files: {
      scores_tsv: "/data/results/6fhu/6fhu.protcross.scores.tsv",
      pockets_json: "/data/results/6fhu/6fhu.protcross.pockets.json",
      summary_json: "/data/results/6fhu/6fhu.protcross.summary.json"
    }
  } : undefined;
  return { status, prediction, batchJob };
}

function previewResidues(clusterId: number, scores: number[], xOrigin: number, rankStart: number): ResidueSummary[] {
  return scores.map((score, index) => ({
    residue_id: `A_${120 + clusterId * 10 + index}`,
    residue_key: `model:0|chain:A|het:ATOM|resseq:${120 + clusterId * 10 + index}|icode:|resname:ALA`,
    chain_id: "A",
    residue_number: 120 + clusterId * 10 + index,
    score,
    probability: score,
    cluster_id: clusterId || null,
    x: xOrigin + index * 2,
    y: clusterId * 2,
    z: 0,
    rank_global: rankStart + index,
    is_scored: 1
  }));
}
