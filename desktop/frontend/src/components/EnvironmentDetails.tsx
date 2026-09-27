import type { AssetDownloadJob, AssetStatus, BackendMode, DesktopStatus, FileStatus } from "../types";
import { Icon } from "./Icon";

export function isMacPlatform(): boolean {
  return /Mac|iPhone|iPad/.test(navigator.userAgent);
}

export function runtimeName(mode?: BackendMode | null): string {
  if (mode === "cpu") return "CPU";
  if (mode === "gpu") return isMacPlatform() ? "Apple MPS" : "NVIDIA CUDA";
  if (mode === "conda") return "Conda";
  return "Not configured";
}

export function backendIsHealthy(status: DesktopStatus | null): boolean {
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

export function EnvironmentStatus({ label, tone = "neutral" }: {
  label: string;
  tone?: "success" | "warning" | "danger" | "neutral";
}) {
  return <span className={`environment-status ${tone}`}>
    <Icon name={tone === "success" ? "check" : tone === "warning" || tone === "danger" ? "warning" : "info"} size={14} />
    {label}
  </span>;
}

function assetState(asset?: FileStatus): { label: string; tone: "success" | "warning" | "danger" | "neutral" } {
  if (!asset) return { label: "Unknown", tone: "neutral" };
  if (!asset.present) return { label: "Missing", tone: "warning" };
  if (asset.verified === false) return { label: "Hash mismatch", tone: "danger" };
  if (asset.verified === true) return { label: "Verified", tone: "success" };
  return { label: "Present", tone: "neutral" };
}

export function AssetInventory({ assets }: { assets?: AssetStatus }) {
  const items: Array<{ name: string; asset?: FileStatus }> = [
    { name: "Checkpoint", asset: assets?.checkpoint },
    { name: "PCA", asset: assets?.pca },
    { name: "ESM-C 600M", asset: assets?.esm }
  ];
  return <>
    <div className="table-scroll">
      <table className="asset-table">
        <thead><tr><th scope="col">Asset</th><th scope="col">Verification</th><th scope="col">File</th></tr></thead>
        <tbody>{items.map(({ name, asset }) => <tr key={name}>
          <th scope="row">{name}</th>
          <td><EnvironmentStatus {...assetState(asset)} /></td>
          <td><code className="asset-path" title={asset?.path ?? undefined}>{asset?.path?.split(/[\\/]/).pop() || "—"}</code></td>
        </tr>)}</tbody>
      </table>
    </div>
    <details className="disclosure compact-disclosure asset-details">
      <summary>Paths &amp; SHA256</summary>
      <div className="disclosure-content">{items.map(({ name, asset }) => <div className="asset-detail" key={name}>
        <h4>{name}</h4>
        <dl className="environment-list">
          <dt>Path</dt><dd><code>{asset?.path || "—"}</code></dd>
          <dt>Expected SHA256</dt><dd><code>{asset?.expected_sha256 || "—"}</code></dd>
          <dt>Actual SHA256</dt><dd><code>{asset?.actual_sha256 || "—"}</code></dd>
        </dl>
      </div>)}</div>
    </details>
  </>;
}

export function downloadPhaseLabel(job: AssetDownloadJob): string {
  if (job.status === "running" && (job.total_bytes ?? 0) > 0 && job.downloaded_bytes >= Number(job.total_bytes)) {
    return "Verifying ESM-C";
  }
  return {
    queued: "Preparing download", running: "Downloading ESM-C", cancelling: "Pausing…",
    cancelled: "Download paused", failed: "Download interrupted", completed: "ESM-C verified"
  }[job.status];
}
