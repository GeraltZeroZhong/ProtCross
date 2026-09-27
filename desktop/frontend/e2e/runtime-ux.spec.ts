import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

const version = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
type Event = { kind: "ipc" | "http"; name: string; args?: Record<string, unknown> };
function gate() {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => { release = resolve; });
  return { promise, release };
}

/** IPC, backend reports and downloads are simulated; all React interactions are real. */
async function runtimeFixture(page: Page, options: {
  slowStart?: boolean;
  offline?: boolean;
  missingAssets?: boolean;
  failedReport?: boolean;
  holdCandidate?: boolean;
  holdInstall?: boolean;
  restartFails?: boolean;
  installFailures?: number;
  holdPause?: boolean;
  httpUnavailable?: boolean;
} = {}) {
  const events: Event[] = [];
  const startGate = gate();
  const candidateGate = gate();
  const installGate = gate();
  const pauseGate = gate();
  let downloadStarted = false;
  const saved = { mode: "cpu", python: "/saved/cpu/python", proxy_url: "http://saved.proxy:8080", tested: !options.offline };
  const download = { id: "fixture-download", filename: "esmc_600m_2024_12_v0.pth", status: "running", downloaded_bytes: 1024, total_bytes: 2299288000, percent: 0.01, bytes_per_second: 512, resumable: true };
  await page.exposeFunction("recordRuntimeUx", (event: Event) => { events.push(event); });
  await page.exposeFunction("waitForRuntimeStart", () => startGate.promise);
  await page.exposeFunction("waitForRuntimeInstall", () => installGate.promise);
  await page.addInitScript((flags) => {
    let repaired = !flags.offline;
    let installAttempts = 0;
    Object.assign(window, {
      __TAURI_INTERNALS__: {
        invoke: async (command: string, args: Record<string, unknown> = {}) => {
          await (window as any).recordRuntimeUx({ kind: "ipc", name: command, args });
          if (command === "install_backend") {
            installAttempts += 1;
            if (installAttempts <= (flags.installFailures ?? 0)) throw new Error("Runtime installation failed: connection reset.");
            if (flags.holdInstall) await (window as any).waitForRuntimeInstall();
            repaired = true;
            return null;
          }
          if (command === "stop_backend" && flags.restartFails) repaired = false;
          if (command === "start_backend") {
            if (flags.slowStart) await (window as any).waitForRuntimeStart();
            if (!repaired) throw new Error("The saved runtime is unavailable");
            return { token: "fixture", port: 54321 };
          }
          if (command === "plugin:dialog|message") return "Ok";
          if (command === "plugin:dialog|open") return "/data/previous.protcross.summary.json";
          return null;
        }
      }
    });
  }, options);
  await page.route("http://127.0.0.1:54321/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const headers = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET, POST, OPTIONS" };
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers });
      return;
    }
    const args = request.method() === "POST" ? request.postDataJSON() : {};
    events.push({ kind: "http", name: path, args });
    if (options.httpUnavailable) {
      await route.fulfill({ status: 503, json: { error: "Local runtime connection interrupted" }, headers });
      return;
    }
    let body: unknown = {};
    if (path === "/backend/configure") {
      saved.mode = args.mode;
      saved.python = args.mode === "conda" ? args.conda_python : "/saved/cpu/python";
      saved.proxy_url = args.proxy_url ?? "";
      saved.tested = false;
      body = saved;
    }
    if (path === "/backend/test") {
      if (args.persist === false && options.holdCandidate) await candidateGate.promise;
      if (args.persist !== false) saved.tested = !options.failedReport;
      body = {
        ok: !options.failedReport,
        python: args.conda_python ?? saved.python,
        backend: args.mode ?? saved.mode,
        checks: {
          torch: { ok: true, version: "2.3.1", tensor_ok: true, cuda_available: false, mps_available: false },
          esm: { ok: !options.failedReport, ...(options.failedReport ? { error: "missing" } : { version: "3.0" }) },
          protcross: { ok: true, distribution_version: version }
        },
        stdout: "diagnostic stdout marker\nTorch import completed.",
        stderr: options.failedReport ? "ModuleNotFoundError: esm\nInstall protcross[predict] in this Python." : ""
      };
    }
    if (path === "/status") {
      const asset = { path: "/data/model", present: true, verified: true };
      const assetsReady = !options.missingAssets;
      body = {
        paths: { outputs_dir: "/data/outputs" }, manifest: {},
        assets: { ready: assetsReady, checkpoint: asset, pca: asset, esm: { ...asset, present: assetsReady, verified: assetsReady } },
        backend: { mode: saved.mode, python: saved.python, python_present: true, runtime_matches_config: true, backend_test_ok: saved.tested, backend_test_mode: saved.mode, backend_test_python: saved.python, backend_test_package_version: version, required_package_version: version, proxy_url: saved.proxy_url },
        readiness: { ready: saved.tested && assetsReady, issues: [...(!saved.tested ? ["Run and pass the backend environment test."] : []), ...(!assetsReady ? ["Download ESM-C weights."] : [])] },
        activity: { batch_jobs: [], asset_downloads: downloadStarted ? [download] : [] }
      };
    }
    if (path === "/inspect") body = {
      schema_version: "test", input_structure: args.input_structure, format: "PDB", model_count: 1, available_chains: ["A"], selected_chains: ["A"],
      chain_summaries: [{ chain_id: "A", scorable_residue_count: 2, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0, alternate_ca_residues: 0, sequence_break_count: 0, numbering_gap_count: 0, exceeds_esm_context: false, residues_over_context_limit: 0 }],
      scorable_residue_count: 2, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0, alternate_ca_residues: 0, sequence_break_count: 0, numbering_gap_count: 0, longest_chain_context: 2, max_len: 1022, requires_truncation: false, warnings: [], parser_warnings: [], input_interpretation: {}
    };
    if (path === "/assets/download-esm/start") {
      downloadStarted = true;
      download.status = "running";
      body = download;
    }
    if (path === "/asset-download/fixture-download") body = download;
    if (path === "/asset-download/fixture-download/cancel") {
      download.status = "cancelling";
      if (options.holdPause) await pauseGate.promise;
      body = download;
    }
    await route.fulfill({ json: body, headers });
  });
  return {
    events, saved, download,
    releaseStart: startGate.release, releaseCandidate: candidateGate.release, releaseInstall: installGate.release,
    releasePause: pauseGate.release,
    setMissingAssets: (value: boolean) => { options.missingAssets = value; },
    setFailedReport: (value: boolean) => { options.failedReport = value; },
    setHttpUnavailable: (value: boolean) => { options.httpUnavailable = value; }
  };
}

async function openRuntimeOptions(page: Page) {
  await page.getByLabel("Structure file", { exact: true }).waitFor();
  await page.locator(".primary-nav").getByRole("button", { name: /Setup/ }).click();
  await page.getByText("Runtime options", { exact: false }).click();
}

function switchingEvents(events: Event[]) {
  return events.filter((event) => ["/backend/test", "/backend/configure", "stop_backend", "start_backend", "install_backend", "/assets/download-esm/start"].includes(event.name));
}

test("finishing a slow startup respects an explicit Results navigation", async ({ page }) => {
  const fixture = await runtimeFixture(page, { slowStart: true });
  await page.goto("/");
  await page.locator(".primary-nav").getByRole("button", { name: /Results/ }).click();
  await expect(page.getByRole("heading", { level: 2 })).toHaveText("Results");
  await expect(page.getByRole("button", { name: "Open result", exact: true })).toBeDisabled();
  fixture.releaseStart();
  await expect(page.getByRole("button", { name: "Open result", exact: true })).toBeEnabled();
  await expect(page.getByRole("heading", { level: 2 })).toHaveText("Results");
  expect(fixture.events.filter((event) => event.name === "/status").length).toBeGreaterThan(0);
});

test("an HTTP 200 failed environment report retains stdout stderr and failing checks", async ({ page }) => {
  const fixture = await runtimeFixture(page, { failedReport: true });
  await page.goto("/");
  await page.getByLabel("Structure file", { exact: true }).waitFor();
  await page.locator(".primary-nav").getByRole("button", { name: /Diagnostics/ }).click();
  await page.getByRole("button", { name: "Test runtime", exact: true }).click();
  const report = page.getByRole("region", { name: "Latest test", exact: true });
  await report.getByText(/^Process output/).click();
  await expect(report).toContainText("diagnostic stdout marker");
  await expect(report).toContainText("ModuleNotFoundError: esm");
  await expect(report.getByRole("row").filter({ has: page.getByRole("rowheader", { name: "esm", exact: true }) })).toContainText("Failed");
  await expect(report).toContainText("missing");
  await page.getByText("Full report", { exact: true }).click();
  const reportJson = page.getByLabel("Full diagnostic report JSON", { exact: true });
  await expect(reportJson).toHaveAttribute("tabindex", "0");
  await expect(report.getByLabel("Runtime stdout", { exact: true })).toHaveAttribute("tabindex", "0");
  await expect(report.getByLabel("Runtime stderr", { exact: true })).toHaveAttribute("tabindex", "0");
  await reportJson.focus();
  await expect(reportJson).toBeFocused();
  const fullReport = JSON.parse(await reportJson.innerText());
  expect(fullReport.envTest.checks.esm).toEqual({ ok: false, error: "missing" });
  await expect(report).toContainText("/saved/cpu/python");
  await expect(page.getByText(/Request failed: 200/)).toHaveCount(0);
  expect(fixture.events.find((event) => event.name === "/backend/test")?.args).toEqual({ persist: true });
});

test("a failing candidate Conda test leaves the saved runtime and process untouched", async ({ page }) => {
  const fixture = await runtimeFixture(page, { failedReport: true });
  await page.goto("/");
  await openRuntimeOptions(page);
  await page.getByRole("button", { name: "Conda", exact: true }).click();
  await page.getByLabel("Conda Python", { exact: true }).fill("/candidate/broken/python");
  const before = fixture.events.length;
  const previous = { ...fixture.saved };
  await page.getByRole("button", { name: "Apply and test", exact: true }).click();
  await expect(page.getByRole("heading", { level: 2 })).toHaveText("Diagnostics");
  await expect(page.getByText(/saved configuration unchanged/)).toBeVisible();
  const report = page.getByRole("region", { name: "Latest test", exact: true });
  await report.getByText(/^Process output/).click();
  await expect(report).toContainText("/candidate/broken/python");
  await expect(report).toContainText("ModuleNotFoundError: esm");
  expect(switchingEvents(fixture.events.slice(before))).toEqual([{ kind: "http", name: "/backend/test", args: { mode: "conda", conda_python: "/candidate/broken/python", persist: false } }]);
  expect(fixture.saved).toEqual(previous);
  await expect(page.getByRole("region", { name: "Runtime", exact: true })).toContainText("/saved/cpu/python");
});

test("successful Apply tests before configuration, restarts once, and prevents duplicate submissions", async ({ page }) => {
  const fixture = await runtimeFixture(page, { holdCandidate: true });
  await page.goto("/");
  await openRuntimeOptions(page);
  await page.getByRole("button", { name: "Conda", exact: true }).click();
  await page.getByLabel("Conda Python", { exact: true }).fill("/candidate/working/python");
  await page.getByLabel(/Proxy/).fill("http://candidate.proxy:3128");
  const before = fixture.events.length;
  const apply = page.getByRole("button", { name: "Apply and test", exact: true });
  await apply.click();
  await expect.poll(() => fixture.events.filter((event) => event.name === "/backend/test").length).toBe(1);
  await expect(apply).toBeDisabled();
  await expect(page.getByRole("button", { name: "Restart runtime", exact: true })).toBeDisabled();
  await expect(page.getByLabel("Conda Python", { exact: true })).toBeDisabled();
  await apply.evaluate((element: HTMLButtonElement) => element.click());
  expect(switchingEvents(fixture.events.slice(before)).map((event) => event.name)).toEqual(["/backend/test"]);
  fixture.releaseCandidate();
  await expect(page.getByText("Runtime activated.", { exact: true })).toBeVisible();
  const sequence = switchingEvents(fixture.events.slice(before));
  expect(sequence.map((event) => event.name)).toEqual(["/backend/test", "/backend/configure", "stop_backend", "start_backend", "/backend/test"]);
  expect(sequence[0].args).toEqual({ mode: "conda", conda_python: "/candidate/working/python", persist: false });
  expect(sequence[1].args).toEqual({ mode: "conda", conda_python: "/candidate/working/python", proxy_url: "http://candidate.proxy:3128" });
  expect(sequence[4].args).toEqual({ persist: true });
  expect(fixture.saved).toMatchObject({ mode: "conda", python: "/candidate/working/python", tested: true });
  await expect(apply).toBeEnabled();
});

test("refresh updates health without overwriting unsaved Conda and proxy edits", async ({ page }) => {
  const fixture = await runtimeFixture(page);
  await page.goto("/");
  await openRuntimeOptions(page);
  await page.getByRole("button", { name: "Conda", exact: true }).click();
  await page.getByLabel("Conda Python", { exact: true }).fill("/unsaved/python");
  await page.getByLabel(/Proxy/).fill("http://unsaved.proxy:9999");
  const before = fixture.events.filter((event) => event.name === "/status").length;
  await page.getByLabel("Workspace options", { exact: true }).click();
  await page.getByRole("button", { name: "Refresh runtime status", exact: true }).click();
  await expect.poll(() => fixture.events.filter((event) => event.name === "/status").length).toBeGreaterThan(before);
  await expect(page.getByLabel("Conda Python", { exact: true })).toHaveValue("/unsaved/python");
  await expect(page.getByLabel(/Proxy/)).toHaveValue("http://unsaved.proxy:9999");
  await expect(page.getByRole("button", { name: "Conda", exact: true })).toHaveAttribute("aria-pressed", "true");
  expect(fixture.saved).toMatchObject({ mode: "cpu", python: "/saved/cpu/python", proxy_url: "http://saved.proxy:8080" });
});

test("first launch can explore the bundled real result entirely offline", async ({ page }) => {
  const fixture = await runtimeFixture(page, { offline: true, missingAssets: true });
  const errors: string[] = [];
  const externalRequests: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (["http:", "https:"].includes(url.protocol) && !["127.0.0.1", "localhost"].includes(url.hostname)) externalRequests.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Runtime unavailable.");
  await page.locator(".primary-nav").getByRole("button", { name: /Results/ }).click();
  await expect(page.getByText("Saved files need the local runtime, without model weights.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open result", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Try example", exact: true }).click();
  await expect(page.getByText("Example · 1CRN", { exact: true })).toBeVisible();
  await expect(page.getByText(/46 scored residues/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Open result", exact: true })).toBeDisabled();
  await page.getByLabel("Score cutoff", { exact: true }).fill("1");
  await expect(page.getByText("No cluster. Lower the score cutoff.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(page.getByLabel("Score cutoff", { exact: true })).toHaveValue("0.5");
  expect(fixture.events.filter((event) => event.kind === "http")).toEqual([]);
  expect(fixture.events.filter((event) => ["install_backend", "plugin:dialog|open"].includes(event.name))).toEqual([]);
  expect(errors).toEqual([]);
  expect(externalRequests).toEqual([]);
});

test("Prepare prediction installs and tests once then starts the resumable model download", async ({ page }) => {
  const fixture = await runtimeFixture(page, { offline: true, missingAssets: true, holdInstall: true });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Runtime unavailable.");
  const before = fixture.events.length;
  const prepare = page.getByRole("button", { name: "Prepare prediction", exact: true });
  await prepare.click();
  await expect.poll(() => fixture.events.filter((event) => event.name === "install_backend").length).toBe(1);
  await expect(prepare).toBeDisabled();
  await prepare.evaluate((element: HTMLButtonElement) => element.click());
  expect(fixture.events.filter((event) => event.name === "install_backend")).toHaveLength(1);
  fixture.releaseInstall();
  await expect(page.getByText("Runtime ready · ESM-C download started.", { exact: true })).toBeVisible();
  await expect(page.getByRole("progressbar", { name: "ESM-C download progress" })).toBeVisible();
  const sequence = switchingEvents(fixture.events.slice(before));
  expect(sequence.map((event) => event.name)).toEqual(["install_backend", "stop_backend", "start_backend", "/backend/configure", "/backend/test", "/assets/download-esm/start"]);
  expect(sequence[0].args).toMatchObject({ mode: "cpu" });
  expect(sequence[2].args).toMatchObject({ mode: "cpu" });
  expect(sequence[4].args).toEqual({ mode: "cpu", persist: true });
  expect(sequence[5].args).toEqual({ force: false });
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Example", exact: true })).toBeEnabled();
});


for (const failure of ["restart", "refresh"] as const) {
  test(`a failed ${failure} invalidates earlier readiness and blocks an otherwise valid prediction`, async ({ page }) => {
    const fixture = await runtimeFixture(page, { restartFails: failure === "restart" });
    await page.goto("/");
    await page.getByLabel("Structure file", { exact: true }).fill("/data/valid.pdb");
    await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
    await expect(page.locator(".readiness-card")).toHaveText("Ready");
    if (failure === "restart") {
      await openRuntimeOptions(page);
      await page.getByRole("button", { name: "Restart runtime", exact: true }).click();
      await expect(page.getByRole("alert")).toContainText("The saved runtime is unavailable");
      expect(fixture.events.some((event) => event.name === "stop_backend")).toBe(true);
      await page.locator(".primary-nav").getByRole("button", { name: /Predict/ }).click();
    } else {
      await page.route("http://127.0.0.1:54321/status", (route) => route.fulfill({
        status: 503, json: { error: "Runtime status unavailable" }, headers: { "access-control-allow-origin": "*" }
      }));
      await page.getByLabel("Workspace options", { exact: true }).click();
      await page.getByRole("button", { name: "Refresh runtime status", exact: true }).click();
      await expect(page.getByRole("alert")).toContainText("Runtime status unavailable");
    }
    await expect(page.locator(".readiness-card")).toContainText("Setup required");
    await expect(page.getByText("The runtime is offline. Restart or reinstall it from Setup.", { exact: true })).toBeVisible();
    await expect(page.getByLabel("Structure file", { exact: true })).toHaveValue("/data/valid.pdb");
    await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeDisabled();
    expect(fixture.events.filter((event) => event.name === "/predict")).toEqual([]);
  });
}

test("a delayed bundled example load keeps a newer Predict navigation", async ({ page }) => {
  await runtimeFixture(page);
  const importGate = gate();
  let importRequested = false;
  await page.route("**/src/exampleResult.ts*", async (route) => {
    importRequested = true;
    await importGate.promise;
    await route.continue();
  });
  await page.goto("/");
  await page.getByLabel("Structure file", { exact: true }).waitFor();
  await page.locator(".primary-nav").getByRole("button", { name: /Results/ }).click();
  await page.getByRole("button", { name: "Try example", exact: true }).click();
  await expect.poll(() => importRequested).toBe(true);
  await page.locator(".primary-nav").getByRole("button", { name: /Predict/ }).click();
  importGate.release();
  await expect(page.getByText("Crambin example loaded.", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2 })).toHaveText("Predict");
  await page.locator(".primary-nav").getByRole("button", { name: /Results/ }).click();
  await expect(page.getByText("Example · 1CRN", { exact: true })).toBeVisible();
});

test("Results opening controls visibly lock during a runtime change and recover afterward", async ({ page }) => {
  const fixture = await runtimeFixture(page, { holdCandidate: true });
  await page.goto("/");
  await openRuntimeOptions(page);
  await page.getByRole("button", { name: "Apply and test", exact: true }).click();
  await expect.poll(() => fixture.events.some((event) => event.name === "/backend/test")).toBe(true);
  await page.locator(".primary-nav").getByRole("button", { name: /Results/ }).click();
  const openExisting = page.getByRole("button", { name: "Open result", exact: true });
  await expect(openExisting).toBeDisabled();
  await openExisting.evaluate((element: HTMLButtonElement) => element.click());
  await page.getByRole("button", { name: "Try example", exact: true }).click();
  await expect(page.getByText("Example · 1CRN", { exact: true })).toBeVisible();
  const openAnother = page.getByRole("button", { name: "Open result", exact: true });
  await expect(openAnother).toBeDisabled();
  await openAnother.evaluate((element: HTMLButtonElement) => element.click());
  expect(fixture.events.filter((event) => event.name === "plugin:dialog|open" || event.name === "/result/open")).toEqual([]);
  fixture.releaseCandidate();
  await expect(page.getByText("Runtime activated.", { exact: true })).toBeVisible();
  await expect(openAnother).toBeEnabled();
});


test("plain version strings in a passing diagnostic report are not marked failed", async ({ page }) => {
  await page.goto("/?preview=diagnostics");
  const latestTest = page.getByRole("region", { name: "Latest test", exact: true });
  await expect(latestTest.locator(".environment-section-heading")).toContainText("Passed");
  const torchRow = latestTest.getByRole("row").filter({ has: page.getByRole("rowheader", { name: "torch", exact: true }) });
  await expect(torchRow).toContainText("2.3.1");
  await expect(torchRow).toContainText("Reported");
  await expect(latestTest.getByText("Failed", { exact: true })).toHaveCount(0);
});


test("saved-result setup prepares only a viewer runtime and preserves a newer page", async ({ page }) => {
  const fixture = await runtimeFixture(page, { offline: true, missingAssets: true, holdInstall: true });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Runtime unavailable");
  await page.locator(".primary-nav").getByRole("button", { name: "Results", exact: true }).click();
  const prepare = page.getByRole("button", { name: "Prepare result viewer", exact: true });
  await prepare.click();
  await expect.poll(() => fixture.events.filter((event) => event.name === "install_backend").length).toBe(1);
  await expect(prepare).toBeDisabled();
  await prepare.evaluate((element: HTMLButtonElement) => element.click());
  await page.locator(".primary-nav").getByRole("button", { name: "Diagnostics", exact: true }).click();
  fixture.releaseInstall();
  await expect.poll(() => fixture.saved.tested).toBe(true);
  await expect(page.locator(".activity-strip")).toHaveCount(0);
  await expect(page.getByRole("heading", { level: 2 })).toHaveText("Diagnostics");
  await page.locator(".primary-nav").getByRole("button", { name: "Results", exact: true }).click();
  await expect(page.getByRole("button", { name: "Open result", exact: true })).toBeEnabled();
  expect(fixture.events.filter((event) => event.name === "install_backend")).toHaveLength(1);
  expect(fixture.events.filter((event) => event.name === "/assets/download-esm/start")).toEqual([]);
  expect(fixture.saved).toMatchObject({ mode: "cpu", tested: true });
});

test("first installation failure keeps logs and a clean retry available across pages", async ({ page }) => {
  const fixture = await runtimeFixture(page, { offline: true, missingAssets: true, installFailures: 1 });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Runtime unavailable");
  await page.getByRole("button", { name: "Prepare prediction", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("connection reset");
  await expect(page.getByRole("button", { name: "Prepare prediction", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Open runtime logs", exact: true }).click();
  expect(fixture.events.some((event) => event.name === "open_logs")).toBe(true);
  await page.locator(".primary-nav").getByRole("button", { name: "Results", exact: true }).click();
  await page.locator(".primary-nav").getByRole("button", { name: "Setup", exact: true }).click();
  await page.getByRole("button", { name: "Prepare prediction", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeEnabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(fixture.events.filter((event) => event.name === "install_backend")).toHaveLength(2);
  expect(fixture.events.filter((event) => event.name === "/backend/test")).toHaveLength(1);
  expect(fixture.events.filter((event) => event.name === "/assets/download-esm/start")).toHaveLength(1);
});

test("a failed reinstall test invalidates previously ready prediction controls", async ({ page }) => {
  const fixture = await runtimeFixture(page, { failedReport: true });
  await page.goto("/");
  await page.getByLabel("Structure file", { exact: true }).fill("/data/valid.pdb");
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  await openRuntimeOptions(page);
  await page.getByRole("button", { name: "Install CPU", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText(/environment test failed|runtime test failed/i);
  expect(fixture.saved.tested).toBe(false);
  await expect(page.locator(".readiness-card")).toHaveText("Setup required");
  await expect(page.locator(".workspace-toolbar .environment-status")).toHaveText("Setup required");
  await page.locator(".primary-nav").getByRole("button", { name: "Predict", exact: true }).click();
  await expect(page.getByLabel("Structure file", { exact: true })).toHaveValue("/data/valid.pdb");
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeDisabled();
  expect(fixture.events.filter((event) => event.name === "/predict")).toEqual([]);
});

test("pausing a download is single-flight across navigation and resumes retained bytes", async ({ page }) => {
  const fixture = await runtimeFixture(page, { missingAssets: true, holdPause: true });
  await page.goto("/");
  await page.getByRole("button", { name: "Prepare prediction", exact: true }).click();
  const pause = page.getByRole("button", { name: "Pause", exact: true });
  await expect(pause).toBeEnabled();
  await pause.evaluate((element: HTMLButtonElement) => { element.click(); element.click(); });
  await expect.poll(() => fixture.events.filter((event) => event.name === "/asset-download/fixture-download/cancel").length).toBe(1);
  await page.locator(".primary-nav").getByRole("button", { name: "Results", exact: true }).click();
  await page.locator(".primary-nav").getByRole("button", { name: "Setup", exact: true }).click();
  const pausing = page.getByRole("button", { name: "Pausing…", exact: true });
  await expect(pausing).toBeDisabled();
  await pausing.evaluate((element: HTMLButtonElement) => element.click());
  fixture.releasePause();
  fixture.download.status = "cancelled";
  await expect(page.getByRole("button", { name: "Resume ESM-C", exact: true })).toBeEnabled();
  await expect(page.locator(".download-progress")).not.toContainText("B/s");
  const retained = fixture.download.downloaded_bytes;
  await page.getByRole("button", { name: "Resume ESM-C", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeEnabled();
  expect(fixture.download.downloaded_bytes).toBe(retained);
  expect(fixture.events.filter((event) => event.name === "/asset-download/fixture-download/cancel")).toHaveLength(1);
  expect(fixture.events.filter((event) => event.name === "/assets/download-esm/start").map((event) => event.args)).toEqual([{ force: false }, { force: false }]);
});

test("full transfer reports verification and completion keeps Results navigation", async ({ page }) => {
  const fixture = await runtimeFixture(page, { missingAssets: true });
  await page.goto("/");
  await page.getByRole("button", { name: "Prepare prediction", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeEnabled();
  fixture.download.downloaded_bytes = fixture.download.total_bytes;
  fixture.download.percent = 100;
  await expect(page.locator(".download-progress strong")).toHaveText("Verifying ESM-C");
  await expect(page.locator(".activity-strip")).toContainText("Verifying ESM-C");
  await expect(page.locator(".download-progress")).not.toContainText("B/s");
  await expect(page.locator(".readiness-card")).toHaveText("Setup required");
  await page.locator(".primary-nav").getByRole("button", { name: "Results", exact: true }).click();
  fixture.download.status = "completed";
  fixture.setMissingAssets(false);
  await expect(page.locator(".readiness-card")).toHaveText("Ready");
  await expect(page.getByRole("heading", { level: 2 })).toHaveText("Results");
  await page.locator(".primary-nav").getByRole("button", { name: "Setup", exact: true }).click();
  await page.locator(".workspace-toolbar").getByRole("button", { name: "Predict", exact: true }).click();
  await expect(page.getByLabel("Structure file", { exact: true })).toBeEnabled();
});

test("reconnect clears a stale interrupted download once assets are verified", async ({ page }) => {
  const fixture = await runtimeFixture(page, { missingAssets: true });
  await page.goto("/");
  await page.getByRole("button", { name: "Prepare prediction", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeEnabled();
  fixture.setHttpUnavailable(true);
  await expect(page.getByRole("alert")).toContainText("Download connection lost", { timeout: 10000 });
  await expect(page.locator(".download-progress")).not.toContainText("B/s");
  fixture.download.status = "completed";
  fixture.download.downloaded_bytes = fixture.download.total_bytes;
  fixture.download.percent = 100;
  fixture.setMissingAssets(false);
  fixture.setHttpUnavailable(false);
  await page.getByRole("button", { name: "Reconnect", exact: true }).click();
  await expect(page.locator(".readiness-card")).toHaveText("Ready");
  await expect(page.getByText("Download interrupted", { exact: true })).toHaveCount(0);
  await expect(page.getByText(/Restart the runtime, then start again to resume retained partial data/)).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(fixture.events.filter((event) => event.name === "/assets/download-esm/start")).toHaveLength(1);
});


test("an unavailable runtime leaves uninspected assets unknown rather than missing", async ({ page }) => {
  const fixture = await runtimeFixture(page, { offline: true });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Runtime unavailable");
  await expect(page.locator(".asset-table").getByText("Unknown", { exact: true })).toHaveCount(3);
  await expect(page.locator(".asset-table").getByText("Missing", { exact: true })).toHaveCount(0);
  await page.locator(".primary-nav").getByRole("button", { name: "Diagnostics", exact: true }).click();
  await expect(page.locator(".asset-table").getByText("Unknown", { exact: true })).toHaveCount(3);
  await expect(page.getByRole("region", { name: "Model assets", exact: true }).locator(".environment-section-heading")).toContainText("Unknown");
  expect(fixture.events.filter((event) => event.kind === "http")).toEqual([]);
});
