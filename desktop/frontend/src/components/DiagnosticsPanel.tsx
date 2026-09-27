import packageInfo from "../../package.json";
import type { DesktopStatus } from "../types";
import { Icon } from "./Icon";
import { AssetInventory, backendIsHealthy, EnvironmentStatus, runtimeName } from "./EnvironmentDetails";

interface DiagnosticsPanelProps {
  status: DesktopStatus | null;
  connected: boolean;
  busy: boolean;
  onOpenLogs: () => void;
  envTest: Record<string, unknown> | null;
  onTest: () => void;
  onExport: () => void;
  onOpenReleases: () => void;
  onOpenScientificGuide: () => void;
}

export function DiagnosticsPanel(props: DiagnosticsPanelProps) {
  const backend = props.status?.backend;
  const backendReady = props.connected && backendIsHealthy(props.status);
  const testOk = props.envTest ? props.envTest.ok === true : backend?.backend_test_ok === true;
  const testFailed = props.envTest ? props.envTest.ok === false : backend?.backend_test_ok === false;
  const checks = asRecord(props.envTest?.checks);
  const checksEntries = Object.entries(checks);
  const torch = asRecord(checks.torch);
  const accelerationAvailable = torch.cuda_available === true || torch.mps_available === true;
  const accelerationRequired = props.envTest?.backend === "gpu";
  const latestError = typeof props.envTest?.error === "string" ? props.envTest.error : null;
  const version = backend?.backend_test_package_version;
  const versionMatches = Boolean(version && version === backend?.required_package_version);
  const healthy = backendReady && props.status?.assets.ready && !testFailed;

  return <div className="diagnostics-page">
    <div className="workspace-toolbar">
      <EnvironmentStatus label={!props.connected ? "Runtime offline" : healthy ? "Operational" : "Needs attention"} tone={healthy ? "success" : "warning"} />
      <div className="button-row">
        <button onClick={props.onOpenLogs} title="Installation and runtime logs; available offline"><Icon name="folder" /> Open logs</button>
        <button disabled={!props.connected || props.busy} onClick={props.onExport} title="Export versions, configuration and local logs as ZIP"><Icon name="download" /> Export diagnostics</button>
        <button className="primary-action" disabled={!props.connected || props.busy} onClick={props.onTest}><Icon name="activity" /> Test runtime</button>
      </div>
    </div>
    {!props.connected ? <p className="environment-note" role="status">Tests and ZIP export require a running runtime. Local logs remain available.</p> : null}

    <section className="environment-section" aria-labelledby="diagnostic-runtime-title">
      <div className="environment-section-heading"><h3 id="diagnostic-runtime-title">Runtime</h3><span className="environment-note">Desktop {packageInfo.version}</span></div>
      <dl className="environment-list">
        <dt>Mode</dt><dd>{runtimeName(backend?.mode)} <EnvironmentStatus label={backendReady ? "Ready" : !props.connected ? "Offline" : "Unverified"} tone={backendReady ? "success" : "warning"} /></dd>
        <dt>Python</dt><dd><code>{backend?.python || "—"}</code></dd>
        {backend?.sidecar_python && backend.sidecar_python !== backend.python ? <><dt>Active Python</dt><dd><code>{backend.sidecar_python}</code></dd></> : null}
        <dt>ProtCross</dt><dd>{version || "Unknown"}{!versionMatches && backend?.required_package_version ? <span className="environment-note"> · requires {backend.required_package_version}</span> : null}</dd>
        <dt>Last test</dt><dd><EnvironmentStatus label={testOk ? "Passed" : testFailed ? "Failed" : "Not run"} tone={testOk ? "success" : testFailed ? "danger" : "neutral"} />{!props.envTest && backend?.backend_tested_at ? <time className="environment-note" dateTime={backend.backend_tested_at}>{formatTime(backend.backend_tested_at)}</time> : null}</dd>
      </dl>
    </section>

    {props.envTest ? <section className="environment-section" aria-labelledby="diagnostic-test-title">
      <div className="environment-section-heading"><h3 id="diagnostic-test-title">Latest test</h3><EnvironmentStatus label={testOk ? "Passed" : "Failed"} tone={testOk ? "success" : "danger"} /></div>
      {testFailed ? <p className="inline-error" role="status">{latestError || "Environment test failed. Inspect the checks and process output."}</p> : null}
      {props.envTest.python ? <p className="environment-path"><code>{String(props.envTest.python)}</code></p> : null}
      {checksEntries.length ? <div className="table-scroll"><table className="diagnostic-checks">
        <thead><tr><th scope="col">Check</th><th scope="col">State</th><th scope="col">Version / detail</th></tr></thead>
        <tbody>{checksEntries.map(([name, value]) => {
          const check = asRecord(value);
          const failed = check.ok === false || (typeof value === "string" && /error$/i.test(name));
          return <tr key={name}>
            <th scope="row"><code>{name}</code></th>
            <td><EnvironmentStatus label={check.ok === true ? "Passed" : failed ? "Failed" : "Reported"} tone={check.ok === true ? "success" : failed ? "danger" : "neutral"} /></td>
            <td>{String(check.error ?? check.distribution_version ?? check.version ?? (typeof value === "string" ? value : "—"))}</td>
          </tr>;
        })}
        {typeof torch.tensor_ok === "boolean" ? <tr>
          <th scope="row">Tensor operation</th>
          <td><EnvironmentStatus label={torch.tensor_ok ? "Passed" : "Failed"} tone={torch.tensor_ok ? "success" : "danger"} /></td>
          <td>PyTorch</td>
        </tr> : null}
        {typeof torch.cuda_available === "boolean" || typeof torch.mps_available === "boolean" ? <tr>
          <th scope="row">Acceleration</th>
          <td><EnvironmentStatus label={accelerationAvailable ? "Available" : "Unavailable"} tone={accelerationAvailable ? "success" : accelerationRequired ? "danger" : "neutral"} /></td>
          <td>{torch.gpu_name ? String(torch.gpu_name) : torch.mps_available ? "Apple MPS" : accelerationRequired ? "CUDA or MPS required" : "CPU runtime supported"}{torch.cuda_version ? ` · CUDA ${String(torch.cuda_version)}` : ""}</td>
        </tr> : null}</tbody>
      </table></div> : null}
      <details className="disclosure diagnostic-output" key={`output-${String(props.envTest.python)}-${String(props.envTest.ok)}`}>
        <summary>Process output{props.envTest.returncode != null ? ` · exit ${String(props.envTest.returncode)}` : ""}</summary>
        {props.envTest.stderr ? <><h4>stderr</h4><pre className="diagnostic-json" tabIndex={0} aria-label="Runtime stderr">{String(props.envTest.stderr)}</pre></> : null}
        {props.envTest.stdout ? <><h4>stdout</h4><pre className="diagnostic-json" tabIndex={0} aria-label="Runtime stdout">{String(props.envTest.stdout)}</pre></> : null}
        {!props.envTest.stderr && !props.envTest.stdout ? <p className="environment-note">No process output.</p> : null}
      </details>
    </section> : null}

    <section className="environment-section" aria-labelledby="diagnostic-assets-title">
      <div className="environment-section-heading"><h3 id="diagnostic-assets-title">Model assets</h3><EnvironmentStatus label={!props.status?.assets ? "Unknown" : props.status.assets.ready ? "Verified" : "Incomplete"} tone={!props.status?.assets ? "neutral" : props.status.assets.ready ? "success" : "warning"} /></div>
      <AssetInventory assets={props.status?.assets} />
    </section>

    <details className="disclosure technical-disclosure">
      <summary><Icon name="diagnostics" /> Full report</summary>
      <pre className="diagnostic-json" tabIndex={0} aria-label="Full diagnostic report JSON">{JSON.stringify({ status: props.status, envTest: props.envTest }, null, 2)}</pre>
    </details>
    <div className="environment-links">
      <button onClick={props.onOpenScientificGuide}>Technical guide <Icon name="external" size={14} /></button>
      <button onClick={props.onOpenReleases}>Releases <Icon name="external" size={14} /></button>
    </div>
  </div>;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}
