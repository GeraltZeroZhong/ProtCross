import { Suspense, lazy, useId, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import packageInfo from "../../package.json";
import { Icon } from "./Icon";
import type { PocketJson, ResidueSummary, SummaryJson } from "../types";
import { recomputeLocalResult } from "../localResults";

const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_CLUSTER_CUTOFF = 8.0;
const APP_VERSION = packageInfo.version;
const TECHNICAL_GUIDE_URL = "https://github.com/GeraltZeroZhong/ProtCross/blob/v0.2.5/README.md#model-and-inference-pipeline";
const MolstarViewer = lazy(() => import("./MolstarViewer").then((module) => ({ default: module.MolstarViewer })));

interface ExplorationState {
  thresholdText: string;
  clusterCutoffText: string;
  threshold: number;
  clusterCutoff: number;
  selectedClusterIndex: number;
  rankingPage: number;
}

export function ResultsPanel(props: {
  structurePath?: string;
  structureData?: string;
  outputFiles?: Record<string, string>;
  summary: SummaryJson | null;
  pockets: PocketJson | null;
  scores: ResidueSummary[];
  residues: ResidueSummary[];
  darkMode: boolean;
  active?: boolean;
  busy?: boolean;
  onOpenLatest?: () => void;
  connected?: boolean;
  sample?: boolean;
  onOpenSetup?: () => void;
  onPrepareRuntime?: () => void;
  onOpenSample?: () => void;
  onOpenExisting: () => void;
  onNotify: (message: string) => void;
  onError: (message: string) => void;
}) {
  const originalThreshold = finiteNumber(props.summary?.threshold, finiteNumber(props.pockets?.threshold, DEFAULT_THRESHOLD));
  const originalClusterCutoff = finiteNumber(
    props.summary?.cluster_cutoff,
    finiteNumber(props.pockets?.cluster_cutoff, DEFAULT_CLUSTER_CUTOFF)
  );
  const sourceIdentity = {
    bundled_example: props.sample === true,
    summary_file: props.outputFiles?.summary_json ?? props.summary?.output_files?.summary_json ?? null,
    input_structure: props.summary?.input_structure ?? props.structurePath ?? null,
    input_sha256: props.summary?.input_file?.sha256 ?? null,
    protcross_version: props.summary?.protcross_version ?? null,
    asset_version: props.summary?.asset_version ?? null,
    chains_analyzed: props.summary?.chains_analyzed ?? null,
    assets: props.summary?.assets ?? null,
    device: props.summary?.device ?? null
  };
  // Include the recorded run so CLI overwrites at the same path cannot share view settings.
  const resultIdentity = JSON.stringify([sourceIdentity, props.summary, originalThreshold, originalClusterCutoff]);
  const initialView: ExplorationState = {
    thresholdText: String(originalThreshold),
    clusterCutoffText: String(originalClusterCutoff),
    threshold: originalThreshold,
    clusterCutoff: originalClusterCutoff,
    selectedClusterIndex: 0,
    rankingPage: 0
  };
  const [views, setViews] = useState<Record<string, ExplorationState>>({});
  const view = views[resultIdentity] ?? initialView;
  const displayThreshold = view.threshold;
  const displayClusterCutoff = view.clusterCutoff;
  const thresholdError = numericError(view.thresholdText, "score cutoff", 0, 1);
  const clusterCutoffError = numericError(view.clusterCutoffText, "cluster distance", 0);
  const invalidDraft = Boolean(thresholdError || clusterCutoffError);
  function updateView(patch: Partial<ExplorationState>) {
    setViews((current) => ({
      ...current,
      [resultIdentity]: { ...(current[resultIdentity] ?? initialView), ...patch }
    }));
  }
  function updateParameter(field: "thresholdText" | "clusterCutoffText", value: string) {
    setViews((current) => {
      const next = { ...(current[resultIdentity] ?? initialView), [field]: value };
      if (!numericError(next.thresholdText, "score cutoff", 0, 1)
        && !numericError(next.clusterCutoffText, "cluster distance", 0)) {
        next.threshold = Number(next.thresholdText);
        next.clusterCutoff = Number(next.clusterCutoffText);
        next.selectedClusterIndex = 0;
        next.rankingPage = 0;
      }
      return { ...current, [resultIdentity]: next };
    });
  }
  function resetDisplay() {
    updateView(initialView);
  }
  const localView = useMemo(
    () => recomputeLocalResult(props.scores, displayThreshold, displayClusterCutoff, props.pockets),
    [props.scores, props.pockets, displayThreshold, displayClusterCutoff]
  );
  const savedTopPocket = !localView.available && !props.pockets && props.summary?.top_pocket && props.residues.length
    ? { ...props.summary.top_pocket, residues: props.residues }
    : null;
  const displayedPockets = useMemo(() => localView.pockets ?? (savedTopPocket ? {
    schema_version: "protcross-pocket-v2",
    threshold: originalThreshold,
    cluster_cutoff: originalClusterCutoff,
    clustered_pockets: [savedTopPocket]
  } : null), [localView.pockets, props.summary?.top_pocket, props.residues, originalThreshold, originalClusterCutoff]);
  const clusters = displayedPockets?.clustered_pockets ?? [];
  const selectedClusterIndex = Math.min(view.selectedClusterIndex, Math.max(0, clusters.length - 1));
  const selectedCluster = clusters[selectedClusterIndex] ?? null;
  const displayedPocket = selectedCluster ?? (localView.available ? null : props.summary?.top_pocket) ?? null;
  const displayedResidues = selectedCluster?.residues ?? (localView.available ? [] : props.residues);
  const center = validCenter(displayedPocket?.center);
  const displayedSelectedCount = localView.available
    ? localView.selectedResidueCount
    : knownCount(props.pockets?.selected_residue_count) ?? knownCount(props.summary?.selected_residue_count);
  const displayedClusterCount = localView.available || Array.isArray(props.pockets?.clustered_pockets)
    ? clusters.length
    : knownCount(props.summary?.cluster_count);
  const displayedScoredCount = props.scores.length
    ? localView.records.length
    : knownCount(props.summary?.residues_scored);
  const rankingPageSize = 100;
  const rankingPageCount = Math.max(1, Math.ceil(localView.records.length / rankingPageSize));
  const boundedRankingPage = Math.min(view.rankingPage, rankingPageCount - 1);
  const rankingStart = boundedRankingPage * rankingPageSize;
  const rankingRows = localView.records.slice(rankingStart, rankingStart + rankingPageSize);
  const parametersChanged = displayThreshold !== originalThreshold
    || displayClusterCutoff !== originalClusterCutoff
    || invalidDraft;
  const scoredResidueKeys = useMemo(
    () => props.scores.some((residue) => Boolean(residue.residue_key))
      ? props.scores
        .filter((residue) => Number(residue.is_scored ?? 1) !== 0)
        .map((residue) => residue.residue_key)
        .filter((key): key is string => Boolean(key))
      : undefined,
    [props.scores]
  );
  const outputAnchor = props.outputFiles?.summary_json ?? props.outputFiles?.structure ?? props.summary?.output_files?.summary_json;
  const outputDir = outputAnchor
    ? String(outputAnchor).replace(/[\\/][^\\/]+$/, "")
    : undefined;
  async function copyResult(value: string, label: string) {
    try {
      await navigator.clipboard.writeText(value);
      props.onNotify(`${label} copied.`);
    } catch (exc) {
      props.onError(`Could not copy ${label.toLowerCase()}: ${exc instanceof Error ? exc.message : String(exc)}`);
    }
  }
  const originalRunRecord = {
    schema_version: "protcross-run-record-v1",
    source_identity: sourceIdentity,
    original_run: props.summary,
    original_pocket_settings: {
      threshold: props.pockets?.threshold ?? null,
      cluster_cutoff: props.pockets?.cluster_cutoff ?? null
    },
    output_files: props.outputFiles ?? props.summary?.output_files ?? null
  };
  const explorationRecord = {
    schema_version: "protcross-exploration-v1",
    source_identity: sourceIdentity,
    original_settings: { threshold: originalThreshold, cluster_cutoff: originalClusterCutoff },
    display_settings: {
      threshold: displayThreshold,
      threshold_operator: ">",
      cluster_cutoff: displayClusterCutoff,
      cluster_distance_unit: "angstrom",
      selected_cluster_id: selectedCluster?.cluster_id ?? null,
      ranking_page: boundedRankingPage + 1
    },
    selected_residue_count: displayedSelectedCount,
    displayed_cluster_count: localView.available || displayedPockets ? clusters.length : null,
    view_only: true,
    output_files_modified: false
  };
  function recordText(record: unknown) {
    return `${JSON.stringify(record, null, 2)}\n`;
  }
  const offlineNotice = props.connected === false ? (
    <div className="result-context" role="status">
      <Icon name="info" />
      <span>Saved files need the local runtime, without model weights.</span>
      {props.onPrepareRuntime ? <button className="primary-action" disabled={props.busy} title="Prepare the CPU runtime for saved results; no model download" onClick={props.onPrepareRuntime}>Prepare result viewer</button> : null}
      {props.onOpenSetup ? <button onClick={props.onOpenSetup}>Runtime setup</button> : null}
    </div>
  ) : null;
  if (!props.summary && !props.pockets && !props.outputFiles) {
    return (
      <section className="empty-results">
        <Icon name="results" size={24} />
        <h3>Explore a prediction</h3>
        <p>Open a saved run or try the Crambin example.</p>
        {offlineNotice}
        <div className="button-row centered">
          <button className="primary-action" title="Open a *.protcross.summary.json file" disabled={props.connected === false || props.busy} onClick={props.onOpenExisting}><Icon name="folder" /> Open result</button>
          {props.onOpenSample ? <button onClick={props.onOpenSample}>Try example</button> : null}
        </div>
        <small><code>*.protcross.summary.json</code></small>
      </section>
    );
  }
  return (
    <div className="results-page">
      <section className="result-identity">
        <div className="result-file-heading">
          <div><h3 title={String(props.summary?.input_structure ?? props.structurePath ?? "")}>{fileName(String(props.summary?.input_structure ?? props.structurePath ?? "Prediction"))}</h3>{props.sample ? <span className="result-example-tag" title="Bundled result from real model inference on Crambin, PDB 1CRN">Example · 1CRN</span> : null}</div>
          <p className="result-counts" role="status" aria-live={props.active === false ? "off" : "polite"}><strong>{displayedSelectedCount ?? "—"}</strong> selected · <strong>{displayedClusterCount ?? "—"}</strong> cluster{displayedClusterCount === 1 ? "" : "s"} · <strong>{displayedScoredCount ?? "—"}</strong> scored residues{!localView.available ? " · saved" : ""}</p>
        </div>
        <div className="button-row">
          {props.onOpenLatest ? <button onClick={props.onOpenLatest}>Latest run</button> : null}
          <button title="Open a *.protcross.summary.json file" disabled={props.connected === false || props.busy} onClick={props.onOpenExisting}><Icon name="folder" /> Open result</button>
          {!props.sample ? <button disabled={!outputDir} onClick={() => outputDir && invoke("open_path", { path: outputDir }).catch((exc) => props.onError(String(exc)))}><Icon name="external" /> Output folder</button> : null}
        </div>
      </section>
      {offlineNotice}
      <div className="results-layout">
        {props.active !== false ? <Suspense fallback={<section className="viewer-panel viewer-loading" role="status">Loading structure…</section>}>
          <MolstarViewer
            structurePath={props.structurePath}
            structureData={props.structureData}
            summary={props.summary}
            pockets={displayedPockets}
            selectedClusterIndex={selectedClusterIndex}
            scoredResidueKeys={scoredResidueKeys}
            darkMode={props.darkMode}
          />
        </Suspense> : null}
        <section className="result-panel">
          <div className="inspector-heading"><h3>Explore</h3><span title="Display changes do not rerun inference or modify saved output files">View only</span></div>
          <section className="result-parameters" aria-label="Displayed result parameters">
            <div className="inline-fields">
              <DisplayNumberInput label="Score cutoff" value={view.thresholdText} onChange={(value) => updateParameter("thresholdText", value)} min={0} max={1} error={thresholdError} disabled={localView.unavailableReason === "missing-data"} />
              <DisplayNumberInput label="Distance (Å)" value={view.clusterCutoffText} onChange={(value) => updateParameter("clusterCutoffText", value)} min={0} error={clusterCutoffError} disabled={localView.unavailableReason === "missing-data"} />
            </div>
            <div className="parameter-comparison">
              <dl className="parameter-comparison-row">
                <div><dt>Run</dt><dd>score &gt; {originalThreshold} · ≤ {originalClusterCutoff} Å</dd></div>
                <div><dt>View</dt><dd>score &gt; {displayThreshold} · ≤ {displayClusterCutoff} Å</dd></div>
              </dl>
              <button disabled={!parametersChanged} onClick={resetDisplay}>Reset</button>
            </div>
            {invalidDraft ? <p className="inline-error" role="status">Invalid value. Showing the last valid view.</p> : null}
            {!localView.available ? <p className="inline-error" role="status">{localView.unavailableReason === "invalid-parameters" ? "Use cutoff 0–1 and distance > 0 Å, or reset." : savedTopPocket ? "Regrouping unavailable. Only the saved top cluster is available." : "Regrouping unavailable: incomplete scores or Cα coordinates. Showing saved clusters."}</p> : null}
            <details className="scientific-note">
              <summary><Icon name="info" /> Score &amp; clustering</summary>
              <p>Model scores are uncalibrated, not binding probabilities. Residues above the cutoff form connected clusters at Cα distances ≤ the specified distance. The centroid is score-weighted. View changes do not alter saved files or rerun inference.</p>
            </details>
          </section>
          {props.summary?.warnings?.length ? <div className="warning-list">{props.summary.warnings.map((warning: string) => <div key={warning}><Icon name="warning" />{warning}</div>)}</div> : null}
          {clusters.length ? (
            <label className="field cluster-select">
              <span>Cluster</span>
              <select value={selectedClusterIndex} onChange={(event) => updateView({ selectedClusterIndex: Number(event.target.value) })}>
                {clusters.map((cluster, index) => <option value={index} key={cluster.cluster_id ?? index}>{cluster.cluster_id ?? index + 1} · {knownCount(cluster.residue_count) ?? "—"} residue{cluster.residue_count === 1 ? "" : "s"} · max {formatScore(cluster.score_max, 3)}</option>)}
              </select>
            </label>
          ) : null}
          {displayedPocket ? (
            <div className="metric-row">
              <Metric label="Residues" value={String(knownCount(displayedPocket.residue_count) ?? "—")} />
              <Metric label="Max score" value={formatScore(displayedPocket.score_max)} />
              <Metric label="Mean score" value={formatScore(displayedPocket.score_mean)} />
            </div>
          ) : (
            <div className="result-context"><Icon name="info" /><span>{localView.available ? "No cluster. Lower the score cutoff." : "No saved cluster."}</span></div>
          )}
          <div className="centroid-card">
            <div><span title="Score-weighted Cα centroid; coordinates only, no 3D marker">Cα centroid (Å)</span><code>{center ? center.map((value) => value.toFixed(3)).join(", ") : "—"}</code></div>
            <button disabled={!center} aria-label="Copy score-weighted centroid" className="icon-button subtle" onClick={() => center && void copyResult(center.map((value) => value.toFixed(3)).join(", "), "Centroid")}><Icon name="copy" /></button>
          </div>
          <div className="table-heading"><h4>Cluster residues</h4><button title="Copy chain:residue identifiers" disabled={displayedResidues.length === 0} onClick={() => void copyResult(formatResidueSelection(displayedResidues), "Residue selection")}><Icon name="copy" /> Selection</button></div>
          <div className="table-wrap residue-table" tabIndex={0}>
            <table>
              <caption className="sr-only">Residues in the displayed predicted binding-site cluster</caption>
              <thead><tr><th scope="col">Residue</th><th scope="col">Chain</th><th scope="col" aria-sort="descending" title="Uncalibrated model score">Score</th></tr></thead>
              <tbody>
                {[...displayedResidues].sort((a, b) => Number(b.score ?? b.probability) - Number(a.score ?? a.probability)).map((residue) => (
                  <tr key={`${residue.residue_key ?? `${residue.chain_id ?? ""}:${residue.residue_id}`}-${residue.cluster_id ?? ""}`}>
                    <td>{residue.residue_id}</td><td>{displayChain(String(residue.chain_id ?? ""))}</td><td><ScoreBar value={Number(residue.score ?? residue.probability)} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details className="disclosure result-rankings">
            <summary><Icon name="results" /> {localView.records.length ? "All residues" : "Residue ranking"} <span>{localView.records.length || "unavailable"}</span></summary>
            <div className="disclosure-content">
              {!localView.records.length ? <p className="field-help">The complete score table is unavailable.</p> : null}
              <div className="table-wrap all-residue-table" tabIndex={0}>
                <table>
                  <caption className="sr-only">All scored residues in global model-score rank order</caption>
                  <thead><tr><th scope="col">Rank</th><th scope="col">Residue</th><th scope="col">Chain</th><th scope="col">Cluster</th><th scope="col" title="Uncalibrated model score">Score</th></tr></thead>
                  <tbody>{rankingRows.map((residue, index) => (
                    <tr key={`${residue.residue_key ?? residue.residue_id}-${rankingStart + index}`}>
                      <td>{Number.isFinite(Number(residue.rank_global)) ? Number(residue.rank_global) : rankingStart + index + 1}</td>
                      <td>{residue.residue_id}</td><td>{displayChain(String(residue.chain_id ?? residue.auth_asym_id ?? ""))}</td><td>{residue.cluster_id ?? "—"}</td><td><ScoreBar value={Number(residue.score ?? residue.probability)} /></td>
                    </tr>
                  ))}</tbody>
                </table>
                <div className="pager">
                  <button disabled={boundedRankingPage === 0} onClick={() => updateView({ rankingPage: Math.max(0, boundedRankingPage - 1) })}>Previous</button>
                  <span>{localView.records.length ? `${rankingStart + 1}–${Math.min(localView.records.length, rankingStart + rankingRows.length)}` : "0"} of {localView.records.length}</span>
                  <button disabled={boundedRankingPage + 1 >= rankingPageCount} onClick={() => updateView({ rankingPage: Math.min(rankingPageCount - 1, boundedRankingPage + 1) })}>Next</button>
                </div>
              </div>
            </div>
          </details>
          <details className="disclosure result-records">
            <summary><Icon name="settings" /> Run record &amp; files</summary>
            <div className="disclosure-content">
              <p className="result-provenance">ProtCross {String(props.summary?.protcross_version ?? APP_VERSION)} · assets {String(props.summary?.asset_version ?? "unknown")} · {String(props.summary?.geometry_backend ?? "unknown")} geometry</p>
              <div className="button-row">
                <button aria-label="Copy run JSON" title="Copy recorded prediction settings and provenance" onClick={() => void copyResult(recordText(originalRunRecord), "Run record")}><Icon name="copy" /> Run JSON</button>
                <button aria-label="Copy view JSON" title="Copy source identity and current exploration settings" disabled={invalidDraft || !localView.available} onClick={() => void copyResult(recordText(explorationRecord), "View record")}><Icon name="copy" /> View JSON</button>
              </div>
              <details className="disclosure compact-disclosure">
                <summary>Saved run</summary>
                <pre className="diagnostic-json" tabIndex={0} aria-label="Original run metadata JSON">{recordText(originalRunRecord)}</pre>
              </details>
              <details className="disclosure compact-disclosure">
                <summary>Current view</summary>
                {invalidDraft ? <p className="field-help">Last valid view. Correct invalid values to copy.</p> : null}
                <pre className="diagnostic-json" tabIndex={0} aria-label="Current view settings JSON">{recordText(explorationRecord)}</pre>
              </details>
              <p className="field-help">View JSON records filters and cluster selection. Reopen the summary JSON to load a result.</p>
              {props.sample ? <p className="field-help">Bundled Crambin prediction · PDB 1CRN.</p> : null}
              {!props.sample && props.outputFiles ? <div className="output-files">{Object.entries(props.outputFiles).map(([key, value]) => <div className="output-file" key={key}><span><Icon name="file" />{outputFileLabel(key)}</span><code title={value}>{value}</code><button aria-label={`Copy ${outputFileLabel(key)} path`} className="icon-button subtle" onClick={() => void copyResult(value, `${outputFileLabel(key)} path`)}><Icon name="copy" /></button></div>)}</div> : null}
              <button onClick={() => invoke("open_url", { url: TECHNICAL_GUIDE_URL }).catch((exc) => props.onError(String(exc)))}>Technical guide <Icon name="external" /></button>
            </div>
          </details>
        </section>
      </div>
    </div>
  );
}

function numericError(value: string, label: string, min: number, max?: number): string {
  if (!value.trim()) return `Enter ${label}.`;
  const number = Number(value);
  if (!Number.isFinite(number)) return `Use a finite ${label}.`;
  if (max !== undefined && (number < min || number > max)) return `${label}: ${min}–${max}.`;
  if (max === undefined && number <= min) return `${label} must exceed ${min} Å.`;
  return "";
}

function DisplayNumberInput(props: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  min: number;
  max?: number;
  error: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{props.label}</label>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        min={props.min}
        max={props.max}
        step="any"
        value={props.value}
        disabled={props.disabled}
        aria-invalid={Boolean(props.error)}
        aria-describedby={props.error ? `${id}-error` : undefined}
        onChange={(event) => props.onChange(event.target.value)}
      />
      {props.error ? <span className="inline-error" id={`${id}-error`}>{props.error}</span> : null}
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

function ScoreBar({ value }: { value: number }) {
  if (!Number.isFinite(value) || value < 0 || value > 1) return <span className="score-cell" aria-label="Score unavailable">—</span>;
  const score = value;
  return (
    <span className="score-cell">
      <span className="score-bar" aria-hidden="true"><span style={{ width: `${score * 100}%` }} /></span>
      <strong>{score.toFixed(4)}</strong>
    </span>
  );
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

function fileName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

function knownCount(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function validCenter(value: unknown): number[] | undefined {
  return Array.isArray(value) && value.length === 3 && value.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate))
    ? value : undefined;
}

function formatScore(value: unknown, precision = 4): string {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed.toFixed(precision) : "—";
}

function finiteNumber(value: unknown, fallback: number): number {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function outputFileLabel(key: string): string {
  return {
    structure: "Annotated structure",
    scores_tsv: "Residue scores",
    pockets_json: "Clusters",
    summary_json: "Run summary"
  }[key] ?? key.replace(/[_-]+/g, " ").replace(/^./, (letter) => letter.toUpperCase());
}
