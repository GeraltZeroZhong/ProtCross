import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
const version = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
const recorded = JSON.parse(readFileSync(new URL("../src/example/response.json", import.meta.url), "utf8"));
const pdb = readFileSync(new URL("../src/example/1CRN.protcross.pdb", import.meta.url), "utf8");
const api = "http://127.0.0.1:54321";
const cors = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET,POST,OPTIONS" };
test.use({ launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] } });
function savedResult(revision = 1) {
  const result = structuredClone(recorded);
  result.output_files.structure = "/same/result.pdb";
  result.output_files.summary_json = "/same/result.protcross.summary.json";
  result.summary.output_files = result.output_files;
  result.summary.input_structure = "/same/1CRN.pdb";
  if (revision > 1) {
    result.summary.threshold = 0.55;
    result.scores = result.scores.map((residue: any) => ({ ...residue, score: 0.9, probability: 0.9 }));
  }
  return result;
}
async function fixture(page: Page) {
  const state = { opens: 0, files: 0, badFile: false, badSummary: false, predictionStarted: false, releasePrediction: () => {}, errors: [] as string[] };
  page.on("pageerror", (error) => state.errors.push(error.message));
  await page.addInitScript(() => Object.assign(window, { failDialog: false, __TAURI_INTERNALS__: { invoke: async (command: string, args: any = {}) => {
    if (command === "start_backend") return { token: "fixture", port: 54321 };
    if (command === "plugin:dialog|open") {
      if ((window as any).failDialog) throw new Error("Native file dialog failed: fixture denial");
      if (args.options?.multiple) return ["/same/1CRN.pdb"];
      return args.options?.filters?.[0]?.name === "ProtCross summary" ? "/same/result.protcross.summary.json" : "/same/1CRN.pdb";
    }
    if (command === "plugin:dialog|message") return "Ok";
    return null;
  } } }));
  const asset = { path: "/fixtures/asset", present: true, verified: true };
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === "OPTIONS") return route.fulfill({ status: 204, headers: cors });
    let body: any = {};
    if (path === "/status") body = {
      paths: { outputs_dir: "/fixtures/outputs" }, manifest: {}, assets: { ready: true, checkpoint: asset, pca: asset, esm: asset },
      backend: { mode: "cpu", python: "/runtime/python", python_present: true, runtime_matches_config: true, backend_test_ok: true,
        backend_test_mode: "cpu", backend_test_python: "/runtime/python", backend_test_package_version: version, required_package_version: version },
      readiness: { ready: true, issues: [] }, activity: { batch_jobs: [], asset_downloads: [] }
    };
    if (path === "/result/open") {
      if (state.badSummary) return route.fulfill({ status: 400, json: { error: "Saved summary is invalid JSON: fixture corruption" }, headers: cors });
      body = savedResult(++state.opens);
    }
    if (path === "/file") {
      state.files++;
      if (state.badFile) return route.fulfill({ status: 404, json: { error: "Annotated structure is no longer available" }, headers: cors });
      return route.fulfill({ body: state.opens > 1 ? pdb.split("\n").map((line) => line.startsWith("ATOM  ")
        ? `${line.slice(0, 60)}  0.90${line.slice(66)}` : line).join("\n") : pdb, contentType: "text/plain", headers: cors });
    }
    if (path === "/inspect") body = { input_structure: request.postDataJSON().input_structure, format: "PDB", model_count: 1,
      available_chains: ["A"], selected_chains: ["A"], scorable_residue_count: 46, requires_truncation: false,
      chain_summaries: [{ chain_id: "A", scorable_residue_count: 46, requires_truncation: false }], warnings: [], parser_warnings: [] };
    if (path === "/prediction/status") body = { stage: "Computing residue scores" };
    if (path === "/predict") { state.predictionStarted = true; await new Promise<void>((resolve) => { state.releasePrediction = resolve; }); body = savedResult(2); }
    await route.fulfill({ json: body, headers: cors }).catch(() => {});
  });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeVisible();
  return state;
}
async function scene(page: Page) {
  return page.evaluate(() => {
    const element = document.querySelector(".msp-plugin") as any;
    let fiber = element?.[Object.keys(element).find((key) => key.startsWith("__reactFiber$"))!];
    while (fiber) {
      const plugin = fiber.stateNode?.plugin;
      if (plugin?.canvas3d) {
        const structure = plugin.managers.structure.hierarchy.current.structures[0];
        return { background: plugin.canvas3d.props.renderer.backgroundColor,
          bFactor: structure?.cell.obj.data.models[0].atomicConformation.B_iso_or_equiv.value(0),
          structureCount: plugin.managers.structure.hierarchy.current.structures.length };
      }
      fiber = fiber.return;
    }
    throw new Error("Molecular canvas is not mounted");
  });
}
async function results(page: Page) { await page.locator(".primary-nav").getByRole("button", { name: "Results", exact: true }).click(); }

test("reopening an overwritten result reloads its actual annotated structure and recovers from missing files", async ({ page }) => {
  const state = await fixture(page); await results(page);
  const open = page.getByRole("button", { name: "Open result", exact: true }); await open.click();
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped"); const original = await scene(page);
  await page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true }).fill("0.85");
  await expect(page.locator(".result-counts")).toContainText("0 selected"); await open.click();
  await expect(page.locator(".result-counts")).toContainText("46 selected");
  await expect.poll(() => state.files).toBe(2);
  await expect.poll(async () => (await scene(page)).bFactor).toBeCloseTo(0.9);
  expect(original.bFactor).not.toBeCloseTo(0.9);
  await expect(page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true })).toHaveValue("0.55");
  state.badFile = true; await open.click();
  await expect(page.locator(".viewer-panel [role=alert]")).toHaveText("Annotated structure is no longer available");
  await expect.poll(async () => (await scene(page)).structureCount).toBe(0);
  await expect(page.getByRole("button", { name: "3D tools", exact: true })).toBeDisabled();
  await expect(page.locator(".residue-table tbody tr")).toHaveCount(46);
  state.badFile = false; await open.click();
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
  await expect(page.locator(".viewer-panel [role=alert]")).toHaveCount(0);
  await expect.poll(async () => (await scene(page)).bFactor).toBeCloseTo(0.9);
  state.badSummary = true; await open.click();
  await expect(page.locator(".banner.error")).toContainText("Saved summary is invalid JSON"); await expect(open).toBeEnabled();
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
  await expect(page.locator(".residue-table tbody tr")).toHaveCount(46); expect(state.errors).toEqual([]);
});

test("the molecular background follows live system appearance and respects an explicit override", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" }); await page.goto("/?preview=setup");
  await page.getByRole("button", { name: "Example", exact: true }).click();
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
  await expect.poll(async () => (await scene(page)).background).toBe(0xffffff);
  await page.emulateMedia({ colorScheme: "dark" }); await expect.poll(async () => (await scene(page)).background).toBe(0x101820);
  await page.getByLabel("Workspace options", { exact: true }).click(); await page.getByLabel("Appearance", { exact: true }).selectOption("light");
  await expect.poll(async () => (await scene(page)).background).toBe(0xffffff);
  await page.emulateMedia({ colorScheme: "light" }); await page.emulateMedia({ colorScheme: "dark" });
  await expect.poll(async () => (await scene(page)).background).toBe(0xffffff);
  await page.getByLabel("Appearance", { exact: true }).selectOption("system");
  await expect.poll(async () => (await scene(page)).background).toBe(0x101820);
});

test("native picker failures are announced across input output batch and runtime paths and can be retried", async ({ page }) => {
  const state = await fixture(page); await page.evaluate(() => { (window as any).failDialog = true; });
  const error = page.locator(".banner.error");
  await page.locator(".prediction-form .path-field").first().getByRole("button", { name: "Browse…", exact: true }).click();
  await expect(error).toContainText("Could not open file dialog: Native file dialog failed");
  await page.locator(".output-disclosure > summary").click();
  await page.locator(".output-disclosure").getByRole("button", { name: "Browse…", exact: true }).click(); await expect(error).toContainText("Could not open file dialog");
  await page.locator(".primary-nav").getByRole("button", { name: "Batch", exact: true }).click();
  await page.getByRole("button", { name: "Add structures", exact: true }).click(); await expect(error).toContainText("Could not open file dialog");
  await page.locator(".primary-nav").getByRole("button", { name: "Setup", exact: true }).click();
  await page.locator(".runtime-options > summary").click(); await page.getByRole("button", { name: "Conda", exact: true }).click();
  await page.locator(".runtime-options").getByRole("button", { name: "Browse…", exact: true }).click(); await expect(error).toContainText("Could not open file dialog");
  await page.getByText("Manual assets", { exact: true }).click(); await page.getByRole("button", { name: "Import PCA", exact: true }).click();
  await expect(error).toContainText("Could not open file dialog");
  await page.evaluate(() => { (window as any).failDialog = false; });
  await page.locator(".primary-nav").getByRole("button", { name: "Predict", exact: true }).click();
  await page.locator(".prediction-form .path-field").first().getByRole("button", { name: "Browse…", exact: true }).click();
  await expect(error).toHaveCount(0); await expect(page.getByLabel("Structure file", { exact: true })).toHaveValue("/same/1CRN.pdb");
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled(); expect(state.errors).toEqual([]);
});

test("a completed prediction remains available without replacing a newer example or developer workspace", async ({ page }) => {
  const state = await fixture(page);
  try {
    await page.getByLabel("Structure file", { exact: true }).fill("/same/1CRN.pdb");
    await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Run prediction", exact: true }).click(); await expect.poll(() => state.predictionStarted).toBe(true);
    await page.locator(".primary-nav").getByRole("button", { name: "Setup", exact: true }).click();
    await page.getByRole("button", { name: "Example", exact: true }).click(); await expect(page.getByText("Example · 1CRN", { exact: true })).toBeVisible();
    await page.locator(".primary-nav").getByRole("button", { name: "Diagnostics", exact: true }).click(); state.releasePrediction();
    await expect(page.getByText("Prediction finished.", { exact: true })).toBeVisible();
    await expect(page.locator(".primary-nav").getByRole("button", { name: "Diagnostics", exact: true })).toHaveAttribute("aria-current", "page");
    await results(page); await expect(page.getByText("Example · 1CRN", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Latest run", exact: true }).click(); await expect(page.getByText("Example · 1CRN", { exact: true })).toHaveCount(0);
    await expect(page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true })).toHaveValue("0.55"); expect(state.errors).toEqual([]);
  } finally { state.releasePrediction(); }
});
