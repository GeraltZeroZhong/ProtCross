import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import type { BatchJob } from "../src/types";
const packageInfo = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));

async function desktopFixture(page: Page, broken = false, batchJob?: BatchJob) {
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
      readiness: { ready: true, issues: [] }, activity: { batch_jobs: batchJob ? [batchJob] : [], asset_downloads: [] }
    };
    if (path.startsWith("/batch/") && batchJob) body = batchJob;
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
  await page.getByText("Runtime options", { exact: false }).click();
  await expect(page.getByLabel("Conda Python")).toHaveValue("/conda/env/bin/python");
  await expect(page.getByRole("button", { name: "Apply and test" })).toBeEnabled();
});

test("offline diagnostics expose logs and CPU installation bypasses a broken Conda runtime", async ({ page }) => {
  await desktopFixture(page, true);
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Runtime unavailable.");
  await page.locator(".primary-nav").getByRole("button", { name: /Diagnostics/ }).click();
  await expect(page.getByRole("button", { name: "Test runtime" })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Export diagnostics/ })).toBeDisabled();
  await page.getByRole("button", { name: /Open logs/ }).click();
  await expect.poll(() => page.evaluate(() => (window as any).ipcCalls.some((call: any) => call.command === "open_logs"))).toBe(true);
  await page.locator(".primary-nav").getByRole("button", { name: /Setup/ }).click();
  await page.getByText("Runtime options", { exact: false }).click();
  await page.getByRole("button", { name: "Install CPU", exact: true }).click();
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
  await expect(page.getByText("Prediction cancelled. Runtime ready.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  await expect.poll(() => page.evaluate(() => (window as any).ipcCalls.filter((call: any) => call.command === "set_activity").at(-1)?.args.active)).toBe(false);
});

test("filtering out every cluster clears the inspector centroid and disables copying", async ({ page }) => {
  await page.goto("/?preview=results");
  await expect(page.locator(".centroid-card code")).toHaveText(/^-?\d+\.\d{3}, -?\d+\.\d{3}, -?\d+\.\d{3}$/);
  await expect(page.getByRole("button", { name: "Copy score-weighted centroid", exact: true })).toBeEnabled();
  await page.getByLabel("Score cutoff").fill("1");
  await expect(page.getByText("No cluster. Lower the score cutoff.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Copy score-weighted centroid", exact: true })).toBeDisabled();
  await expect(page.locator(".centroid-card code")).toContainText("—");
});

test("saved prediction settings are labelled current and can be reset", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("protcross-threshold", "0.9"));
  await page.goto("/?preview=predict");
  await page.getByText("Prediction settings", { exact: false }).click();
  await expect(page.getByText(/Modified · Cutoff 0.90/)).toBeVisible();
  await page.getByRole("button", { name: "Restore defaults" }).click();
  await expect(page.locator(".settings-disclosure > summary")).toContainText("Default");
  await expect(page.getByLabel("Model-score cutoff", { exact: true })).toHaveValue("0.5");
});


test("500 batch rows keep new batches, history, and pagination within reach", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const batch: BatchJob = {
    id: "review-interrupted-500", status: "interrupted", completed: 3, failed: 0,
    item_count: 1000, items_offset: 0, items_returned: 500, cancel_requested: false,
    settings: { threshold: 0.5, pocket_cluster_cutoff: 8, allow_truncation: false, batch_size: 4 },
    items: Array.from({ length: 500 }, (_, index) => ({
      input_structure: `/data/structures/structure-${index + 1}.pdb`, chain_id: "A",
      status: index < 3 ? "completed" : "interrupted",
      ...(index < 3 ? { output_dir: `/data/outputs/run-${index + 1}`, output_files: { summary_json: `/data/outputs/run-${index + 1}/summary.json` } } : {})
    }))
  };
  await desktopFixture(page, false, batch);
  await page.goto("/");
  const monitor = page.locator(".batch-monitor");
  await expect(monitor.locator("tbody tr")).toHaveCount(500);
  await expect(monitor.getByRole("button", { name: "Retry unfinished", exact: true })).toBeEnabled();
  await expect(monitor.getByRole("button", { name: "Retry failed", exact: true })).toHaveCount(0);
  const tableViewport = monitor.locator(".table-wrap");
  const dimensions = await tableViewport.evaluate((element) => ({ height: element.clientHeight, scrollHeight: element.scrollHeight }));
  expect(dimensions.scrollHeight).toBeGreaterThan(dimensions.height);
  expect(dimensions.height).toBeLessThanOrEqual(520);
  const pager = monitor.locator(".pager");
  expect(await pager.evaluate((element) => element.closest(".table-wrap") === null)).toBe(true);
  await expect(pager).toBeInViewport({ ratio: 1 });
  await expect(pager.getByRole("button", { name: "Next", exact: true })).toBeEnabled();
  await expect(pager).toContainText("1–500 of 1000");
  await expect(page.locator(".batch-composer > summary")).toBeInViewport({ ratio: 1 });
  await expect(page.locator(".batch-history > summary")).toBeInViewport({ ratio: 1 });
  await tableViewport.focus();
  await page.keyboard.press("End");
  await expect.poll(() => tableViewport.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
  await expect(pager).toBeInViewport({ ratio: 1 });
});
