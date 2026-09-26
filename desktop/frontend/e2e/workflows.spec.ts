import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
const packageInfo = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));

async function desktopFixture(page: Page, broken = false) {
  let mode = "conda";
  let inspections = 0;
  let releasePrediction: (() => void) | undefined;
  const version = packageInfo.version;
  await page.addInitScript(({ broken }) => {
    let repaired = !broken;
    const calls: Array<{ command: string; args: any }> = [];
    Object.assign(window, {
      ipcCalls: calls,
      __TAURI_INTERNALS__: {
        invoke: async (command: string, args: any = {}) => {
          calls.push({ command, args });
          if (command === "start_backend") {
            if (args.mode === "cpu") repaired = true;
            if (!repaired) throw new Error("The configured Conda runtime cannot start");
            return { token: "test", port: 54321 };
          }
          if (command === "plugin:dialog|open") return "/data/input.pdb";
          if (command === "plugin:dialog|message") return "Ok";
          return null;
        }
      }
    });
  }, { broken });
  await page.route("http://127.0.0.1:54321/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const file = { path: "/data/asset", present: true, verified: true };
    let body: unknown = {};
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET, POST, OPTIONS" } });
      return;
    }
    if (path === "/backend/configure") mode = request.postDataJSON().mode;
    if (path === "/backend/test") body = { ok: true };
    if (path === "/status") body = {
      paths: { outputs_dir: "/data/outputs" }, manifest: {},
      assets: { ready: true, checkpoint: file, pca: file, esm: file },
      backend: { mode, python: mode === "conda" ? "/conda/env/bin/python" : "/cpu/python", python_present: true, runtime_matches_config: true, backend_test_ok: true, backend_test_mode: mode, backend_test_package_version: version, required_package_version: version, proxy_url: null },
      readiness: { ready: true, issues: [] }, activity: { batch_jobs: [], asset_downloads: [] }
    };
    if (path === "/inspect") {
      inspections += 1;
      body = { schema_version: "test", input_structure: "/data/input.pdb", format: "PDB", model_count: 1, available_chains: ["A"], selected_chains: ["A"], chain_summaries: [{ chain_id: "A", scorable_residue_count: 2, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0, alternate_ca_residues: 0, sequence_break_count: 0, numbering_gap_count: 0, exceeds_esm_context: false, residues_over_context_limit: 0 }], scorable_residue_count: 2, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0, alternate_ca_residues: 0, sequence_break_count: 0, numbering_gap_count: 0, longest_chain_context: 2, max_len: 1022, requires_truncation: false, warnings: [], parser_warnings: [], input_interpretation: {} };
    }
    if (path === "/prediction/status") body = { stage: "Computing residue scores and writing results" };
    if (path === "/predict") {
      await new Promise<void>((resolve) => { releasePrediction = resolve; });
      await route.abort().catch(() => {});
      return;
    }
    await route.fulfill({ json: body, headers: { "access-control-allow-origin": "*" } });
  });
  return { inspections: () => inspections, releasePrediction: () => releasePrediction?.() };
}

test("choosing the same file and explicitly rechecking both leave prediction available", async ({ page }) => {
  const fixture = await desktopFixture(page);
  await page.goto("/");
  const browse = page.locator(".prediction-form .path-field").first().getByRole("button", { name: "Browse…" });
  await browse.click();
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  await browse.click();
  await expect.poll(fixture.inspections).toBe(2);
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Check again" }).click();
  await expect.poll(fixture.inspections).toBe(3);
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
});

test("saved Conda paths are shown and can be tested without reselecting Python", async ({ page }) => {
  await desktopFixture(page);
  await page.goto("/");
  await page.locator(".primary-nav").getByRole("button", { name: /Setup/ }).click();
  await page.getByText("Advanced runtime options", { exact: false }).click();
  await expect(page.getByLabel("Conda environment Python")).toHaveValue("/conda/env/bin/python");
  await expect(page.getByRole("button", { name: "Save and test" })).toBeEnabled();
});

test("offline diagnostics expose logs and CPU installation bypasses a broken Conda runtime", async ({ page }) => {
  await desktopFixture(page, true);
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("not running yet");
  await page.locator(".primary-nav").getByRole("button", { name: /Diagnostics/ }).click();
  await expect(page.getByRole("button", { name: "Run environment test" })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Export diagnostics/ })).toBeDisabled();
  await page.getByRole("button", { name: /Open logs/ }).click();
  await expect.poll(() => page.evaluate(() => (window as any).ipcCalls.some((call: any) => call.command === "open_logs"))).toBe(true);
  await page.locator(".primary-nav").getByRole("button", { name: /Setup/ }).click();
  await page.getByRole("button", { name: "Install recommended runtime" }).click();
  await expect(page.getByText("CPU backend installed, activated, and tested.")).toBeVisible();
  await expect.poll(() => page.evaluate(() => (window as any).ipcCalls.some((call: any) => call.command === "start_backend" && call.args.mode === "cpu"))).toBe(true);
});

test("single prediction shows a stage and can be cancelled by restarting its runtime", async ({ page }) => {
  const fixture = await desktopFixture(page);
  await page.goto("/");
  await page.getByLabel("Structure file").fill("/data/input.pdb");
  await page.getByRole("button", { name: "Run prediction", exact: true }).click();
  await expect(page.getByText("Computing residue scores and writing results")).toBeVisible();
  await expect.poll(() => page.evaluate(() => (window as any).ipcCalls.some((call: any) => call.command === "set_activity" && call.args.active))).toBe(true);
  await page.getByRole("button", { name: "Cancel prediction", exact: true }).click();
  fixture.releasePrediction();
  await expect(page.getByText("Prediction cancelled. The runtime is ready for another prediction.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  await expect.poll(() => page.evaluate(() => (window as any).ipcCalls.filter((call: any) => call.command === "set_activity").at(-1)?.args.active)).toBe(false);
});

test("filtering out every cluster also clears the viewer centroid", async ({ page }) => {
  await page.goto("/?preview=results");
  await expect(page.locator(".center-readout")).toBeVisible();
  await page.getByLabel("Displayed score cutoff").fill("1");
  await expect(page.getByText("No cluster at this cutoff")).toBeVisible();
  await expect(page.locator(".center-readout")).toHaveCount(0);
  await expect(page.locator(".centroid-card code")).toContainText("—");
});

test("saved prediction settings are labelled current and can be reset", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("protcross-threshold", "0.9"));
  await page.goto("/?preview=predict");
  await page.getByText("Prediction settings", { exact: false }).click();
  await expect(page.getByText(/Current settings: 0.90/)).toBeVisible();
  await page.getByRole("button", { name: "Restore defaults" }).click();
  await expect(page.getByText(/Current settings: 0.50/)).toBeVisible();
});
