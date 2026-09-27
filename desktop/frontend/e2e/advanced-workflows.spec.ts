import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

const version = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;

/** Real frontend interactions with a controlled desktop API; no model or installer runs. */
async function runtimeFixture(page: Page, options: { predictionError?: string } = {}) {
  const predictions: Record<string, unknown>[] = [];
  const batches: Record<string, unknown>[] = [];
  let releasePrediction: (() => void) | undefined;
  let job: Record<string, unknown> | undefined;
  await page.addInitScript(() => {
    Object.assign(window, {
      ipcCalls: [] as Array<{ command: string; args: Record<string, unknown> }>,
      __TAURI_INTERNALS__: {
        invoke: async (command: string, args: Record<string, unknown> = {}) => {
          (window as any).ipcCalls.push({ command, args });
          if (command === "start_backend") return { token: "test", port: 54321 };
          if (command === "plugin:dialog|open") return (args.options as { multiple?: boolean } | undefined)?.multiple ? ["/data/first.pdb", "/data/second.pdb"] : "/data/first.pdb";
          if (command === "plugin:dialog|message") return "Ok";
          return null;
        }
      }
    });
  });
  await page.route("http://127.0.0.1:54321/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const headers = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET, POST, OPTIONS" };
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers });
      return;
    }
    const asset = { path: "/data/model", present: true, verified: true };
    let body: unknown = {};
    if (path === "/status") body = {
      paths: { outputs_dir: "/data/outputs" }, manifest: {},
      assets: { ready: true, checkpoint: asset, pca: asset, esm: asset },
      backend: { mode: "cpu", python: "/runtime/python", python_present: true, runtime_matches_config: true, backend_test_ok: true, backend_test_mode: "cpu", backend_test_python: "/runtime/python", backend_test_package_version: version, required_package_version: version, proxy_url: null },
      readiness: { ready: true, issues: [] }, activity: { batch_jobs: [], asset_downloads: [] }
    };
    if (path === "/inspect") body = {
      schema_version: "test", input_structure: request.postDataJSON().input_structure, format: "PDB", model_count: 1, available_chains: ["A"], selected_chains: ["A"],
      chain_summaries: [{ chain_id: "A", scorable_residue_count: 2, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0, alternate_ca_residues: 0, sequence_break_count: 0, numbering_gap_count: 0, exceeds_esm_context: false, residues_over_context_limit: 0 }],
      scorable_residue_count: 2, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0, alternate_ca_residues: 0, sequence_break_count: 0, numbering_gap_count: 0, longest_chain_context: 2, max_len: 1022, requires_truncation: false, warnings: [], parser_warnings: [], input_interpretation: {}
    };
    if (path === "/prediction/status") body = { stage: "Computing residue scores and writing results" };
    if (path === "/predict") {
      predictions.push(request.postDataJSON());
      if (!options.predictionError) await new Promise<void>((resolve) => { releasePrediction = resolve; });
      await route.fulfill({ status: 400, json: { error: options.predictionError ?? "Prediction fixture stopped" }, headers }).catch(() => {});
      return;
    }
    if (path === "/batch") {
      const payload = request.postDataJSON();
      batches.push(payload);
      job = { id: "fixture-batch", status: "running", created_at: Date.now() / 1000, completed: 0, failed: 0, cancel_requested: false, item_count: payload.items.length, items_offset: 0, items_returned: payload.items.length, settings: payload, items: payload.items.map((item: Record<string, unknown>) => ({ ...item, status: "running" })) };
      body = job;
    }
    if (path === "/batch/fixture-batch") body = job;
    await route.fulfill({ json: body, headers });
  });
  return { predictions, batches, release: () => releasePrediction?.() };
}

async function openReadyPrediction(page: Page) {
  await page.goto("/");
  await page.getByLabel("Structure file", { exact: true }).fill("/data/first.pdb");
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  await page.locator(".settings-disclosure > summary").click();
}

test("empty and invalid drafts survive workspace changes and cannot run or overwrite saved defaults", async ({ page }) => {
  const fixture = await runtimeFixture(page);
  await openReadyPrediction(page);
  await page.getByLabel("Model-score cutoff", { exact: true }).fill("");
  await page.getByLabel("Cluster distance (Å)", { exact: true }).fill("0");
  await expect(page.getByLabel("Model-score cutoff", { exact: true })).toHaveValue("");
  await expect(page.getByLabel("Model-score cutoff", { exact: true })).toHaveAttribute("aria-invalid", "true");
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeDisabled();
  await expect(page.getByText("Cluster distance must be greater than 0 Å.", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => [localStorage.getItem("protcross-threshold"), localStorage.getItem("protcross-cluster-cutoff")])).toEqual(["0.5", "8"]);

  await page.locator(".primary-nav").getByRole("button", { name: /Batch/ }).click();
  await page.locator(".settings-disclosure > summary").click();
  await expect(page.getByLabel("Model-score cutoff", { exact: true })).toHaveValue("");
  await expect(page.getByRole("button", { name: "Run batch", exact: true })).toBeDisabled();
  await page.locator(".primary-nav").getByRole("button", { name: /Predict/ }).click();
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeDisabled();
  await page.locator(".settings-disclosure > summary").click();
  await page.getByRole("button", { name: "Restore defaults", exact: true }).click();
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  await expect(page.locator(".settings-disclosure > summary > span")).toHaveText("Default");
  expect(fixture.predictions).toEqual([]);
});

test("expert parameters are sent numerically and all input controls freeze during a prediction", async ({ page }) => {
  const fixture = await runtimeFixture(page);
  try {
    await openReadyPrediction(page);
    await page.getByLabel("Model-score cutoff", { exact: true }).fill("0.7");
    await page.getByLabel("Cluster distance (Å)", { exact: true }).fill("100");
    await page.getByLabel("Allow long-chain truncation", { exact: true }).check();
    await page.getByLabel("Inference device", { exact: true }).selectOption("custom");
    await page.getByLabel("CUDA device index", { exact: true }).fill("");
    await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeDisabled();
    await page.getByLabel("CUDA device index", { exact: true }).fill("1");
    await page.locator(".output-disclosure > summary").click();
    await page.getByRole("button", { name: "Run prediction", exact: true }).click();
    await expect.poll(() => fixture.predictions.length).toBe(1);
    expect(fixture.predictions[0]).toMatchObject({ threshold: 0.7, pocket_cluster_cutoff: 100, allow_truncation: true, device: "cuda:1" });
    for (const label of ["Structure file", "Output directory", "Model-score cutoff", "Cluster distance (Å)", "Allow long-chain truncation", "Inference device", "CUDA device index"]) {
      await expect(page.getByLabel(label, { exact: true })).toBeDisabled();
    }
    await expect(page.getByRole("combobox", { name: "Chains to analyze", exact: true })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Restore defaults", exact: true })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Cancel prediction", exact: true })).toBeEnabled();
    await page.locator(".primary-nav").getByRole("button", { name: /Batch/ }).click();
    await page.locator(".settings-disclosure > summary").click();
    await expect(page.getByLabel("Structures per microbatch", { exact: true })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Add structures", exact: true })).toBeDisabled();
  } finally {
    fixture.release();
  }
});

test("device overrides keep their explicit meaning and restoring defaults clears an earlier runtime error", async ({ page }) => {
  const fixture = await runtimeFixture(page, { predictionError: "Previous runtime error" });
  await openReadyPrediction(page);
  for (const [index, device] of ["auto", "cpu", "cuda", "mps"].entries()) {
    await page.getByLabel("Inference device", { exact: true }).selectOption(device);
    await page.getByRole("button", { name: "Run prediction", exact: true }).click();
    await expect.poll(() => fixture.predictions.length).toBe(index + 1);
    expect(fixture.predictions[index].device).toBe(device);
    await expect(page.getByText("Previous runtime error", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
  }
  await page.getByRole("button", { name: "Restore defaults", exact: true }).click();
  await expect(page.getByText("Previous runtime error", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Inference device", { exact: true })).toHaveValue("");
  await page.getByRole("button", { name: "Run prediction", exact: true }).click();
  await expect.poll(() => fixture.predictions.length).toBe(5);
  expect(fixture.predictions[4]).not.toHaveProperty("device");
  expect(await page.evaluate(() => (window as any).ipcCalls.some((call: any) => call.command === "install_backend"))).toBe(false);
});

test("microbatch tuning validates whole numbers, reaches the batch API, and locks with the running queue", async ({ page }) => {
  const fixture = await runtimeFixture(page);
  await page.goto("/");
  await page.getByLabel("Structure file", { exact: true }).waitFor();
  await page.locator(".primary-nav").getByRole("button", { name: /Batch/ }).click();
  await page.getByRole("button", { name: "Add structures", exact: true }).click();
  const start = page.getByRole("button", { name: "Run batch", exact: true });
  await expect(start).toBeEnabled();
  await page.locator(".settings-disclosure > summary").click();
  const size = page.getByLabel("Structures per microbatch", { exact: true });
  await expect(size).toHaveValue("4");
  for (const invalid of ["", "0", "1.5", "5"]) {
    await size.fill(invalid);
    await expect(size).toHaveAttribute("aria-invalid", "true");
    await expect(start).toBeDisabled();
  }
  await size.fill("2");
  await page.getByLabel("Inference device", { exact: true }).selectOption("cpu");
  await start.click();
  await expect.poll(() => fixture.batches.length).toBe(1);
  expect(fixture.batches[0]).toMatchObject({ batch_size: 2, device: "cpu", threshold: 0.5, pocket_cluster_cutoff: 8 });
  await page.locator(".batch-composer > summary").click();
  await expect(size).toBeDisabled();
  await expect(page.getByLabel("Inference device", { exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Add structures", exact: true })).toBeDisabled();
  await page.getByText("Run settings", { exact: true }).click();
  await expect(page.locator(".diagnostic-json")).toContainText('"batch_size": 2');
});


test("default parameter summary is concise and scientific reference is keyboard accessible", async ({ page }) => {
  await page.goto("/?preview=predict");
  const summary = page.locator(".settings-disclosure > summary");
  await expect(summary.locator("span")).toHaveText("Default");
  await expect(page.getByLabel("Model-score cutoff", { exact: true })).not.toBeVisible();
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Model-score cutoff", { exact: true })).toBeVisible();
  await expect(page.getByText("Keep the first 1,022 residues per chain; remainder unscored.", { exact: true })).toBeVisible();

  const reference = page.locator(".parameter-reference > summary");
  await reference.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText(/not independently calibrated probabilities/)).toBeVisible();
  await expect(page.getByText(/connected components of selected Cα pairs/)).toBeVisible();
  await page.keyboard.press("Enter");
  await expect(page.getByText(/not independently calibrated probabilities/)).not.toBeVisible();

  await page.getByLabel("Allow long-chain truncation", { exact: true }).check();
  await page.getByLabel("Inference device", { exact: true }).selectOption("cpu");
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(summary.locator("span")).toHaveText("Modified · Truncation on · Device cpu");
});
