import { expect, test, type BrowserContext, type Page, type Route } from "@playwright/test";
import { readFileSync } from "node:fs";

const version = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
const api = "http://127.0.0.1:54321";

function inspection(path: string, count = 46) {
  return { schema_version: "test", input_structure: path, format: "PDB", model_count: 1, available_chains: ["A"], selected_chains: ["A"],
    chain_summaries: [{ chain_id: "A", scorable_residue_count: count, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0,
      alternate_ca_residues: 0, sequence_break_count: 0, numbering_gap_count: 0, exceeds_esm_context: false, residues_over_context_limit: 0 }],
    scorable_residue_count: count, standard_residues_missing_ca: 0, modified_or_nonstandard_amino_acids: 0, alternate_ca_residues: 0,
    sequence_break_count: 0, numbering_gap_count: 0, longest_chain_context: count, max_len: 1022, requires_truncation: false,
    warnings: [], parser_warnings: [], input_interpretation: {} };
}
function batch(id: string, offset = 0, running = false) {
  const large = id === "newer-run";
  return { id, status: running ? "running" : "completed", created_at: large ? 2000 : 1000, completed: large ? 501 : 1,
    failed: 0, cancel_requested: false, item_count: large ? 501 : 1, items_offset: offset, items_returned: 1,
    settings: { threshold: 0.7, pocket_cluster_cutoff: 6, allow_truncation: false, device: "cpu", batch_size: 2 },
    items: [{ input_structure: large ? `/data/newer-${offset}.pdb` : "/data/older-only.pdb", status: "completed",
      output_dir: "/data/results", output_files: { summary_json: "/data/result.summary.json" } }] };
}
function status(jobs: ReturnType<typeof batch>[] = []) {
  const asset = { path: "/runtime/assets/model", present: true, verified: true };
  return { paths: { outputs_dir: "/data/outputs" }, manifest: {}, assets: { ready: true, checkpoint: asset, pca: asset, esm: asset },
    backend: { mode: "cpu", python: "/runtime/python", python_present: true, runtime_matches_config: true, backend_test_ok: true,
      backend_test_mode: "cpu", backend_test_python: "/runtime/python", backend_test_package_version: version, required_package_version: version, proxy_url: null },
    readiness: { ready: true, issues: [] }, activity: { batch_jobs: jobs, asset_downloads: [] } };
}
async function desktop(context: BrowserContext) {
  await context.addInitScript(() => Object.assign(window, { __TAURI_INTERNALS__: { invoke: async (command: string, args: Record<string, any> = {}) => {
    if (command === "start_backend") return { token: "fixture", port: 54321 };
    if (command === "plugin:dialog|open") return args.options?.multiple ? ["/data/edited.pdb"] : "/data/edited.pdb";
    if (command === "plugin:dialog|message") return "Ok";
    return null;
  } } }));
}
async function respond(route: Route, body: unknown, code = 200) {
  await route.fulfill({ status: code, json: body, headers: { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET,POST,OPTIONS" } }).catch(() => {});
}
async function releaseAndPaint(page: Page, release: () => void, urlPart: string) {
  const response = page.waitForResponse((response) => response.url().includes(urlPart));
  release();
  await (await response).finished();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}
async function goBatch(page: Page) {
  await page.goto("/");
  await page.locator(".primary-nav").getByRole("button", { name: /^Batch(?:\s|$)/ }).click();
}

test("a slow single-structure inspection cannot replace the current input check", async ({ page, context }) => {
  await desktop(context);
  let releaseOld: (() => void) | undefined;
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === "OPTIONS") return respond(route, {});
    if (path === "/status") return respond(route, status());
    if (path === "/inspect") {
      const input = request.postDataJSON().input_structure;
      if (input === "/data/old.pdb") {
        await new Promise<void>((resolve) => { releaseOld = resolve; });
        return respond(route, { error: "Old file check failed" }, 400);
      }
      return respond(route, inspection(input));
    }
    return respond(route, {});
  });
  try {
    await page.goto("/");
    await page.getByLabel("Structure file", { exact: true }).fill("/data/old.pdb");
    await expect.poll(() => Boolean(releaseOld)).toBe(true);
    await page.getByLabel("Structure file", { exact: true }).fill("/data/current.pdb");
    await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
    await releaseAndPaint(page, () => releaseOld?.(), "/inspect");
    await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
    await expect(page.getByText("Old file check failed", { exact: true })).toHaveCount(0);
  } finally { releaseOld?.(); }
});

test("removing and readding a batch input invalidates its earlier pending precheck", async ({ page, context }) => {
  await desktop(context);
  let checks = 0, releaseOld: (() => void) | undefined;
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === "OPTIONS") return respond(route, {});
    if (path === "/status") return respond(route, status());
    if (path === "/inspect") {
      if (++checks === 1) {
        await new Promise<void>((resolve) => { releaseOld = resolve; });
        return respond(route, { error: "Stale check: old file has no Cα" }, 400);
      }
      return respond(route, inspection(request.postDataJSON().input_structure, 122));
    }
    return respond(route, {});
  });
  try {
    await goBatch(page);
    await page.getByRole("button", { name: "Add structures", exact: true }).click();
    await expect.poll(() => Boolean(releaseOld)).toBe(true);
    await page.getByRole("button", { name: "Remove edited.pdb", exact: true }).click();
    await page.getByRole("button", { name: "Add structures", exact: true }).click();
    await expect(page.getByRole("button", { name: "Run batch", exact: true })).toBeEnabled();
    await releaseAndPaint(page, () => releaseOld?.(), "/inspect");
    await expect(page.getByRole("button", { name: "Run batch", exact: true })).toBeEnabled();
    await expect(page.getByText("Stale check: old file has no Cα", { exact: true })).toHaveCount(0);
    await expect(page.getByRole("combobox", { name: "Scorable chain for edited.pdb", exact: true })).toBeVisible();
  } finally { releaseOld?.(); }
});

test("a delayed next page cannot replace a newly selected historical batch", async ({ page, context }) => {
  await desktop(context);
  let releasePage: (() => void) | undefined;
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (request.method() === "OPTIONS") return respond(route, {});
    if (url.pathname === "/status") return respond(route, status([batch("older-run"), batch("newer-run")]));
    if (url.pathname === "/batch/newer-run") {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      if (offset === 500) await new Promise<void>((resolve) => { releasePage = resolve; });
      return respond(route, batch("newer-run", offset));
    }
    if (url.pathname === "/batch/older-run") return respond(route, batch("older-run"));
    return respond(route, {});
  });
  try {
    await goBatch(page);
    await expect(page.getByText("newer-0.pdb", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await expect.poll(() => Boolean(releasePage)).toBe(true);
    await page.locator(".batch-history > summary").click();
    await page.locator(".batch-history button").filter({ hasText: "older-run" }).click();
    await expect(page.getByText("older-only.pdb", { exact: true })).toBeVisible();
    await releaseAndPaint(page, () => releasePage?.(), "/batch/newer-run?");
    await expect(page.getByText("older-only.pdb", { exact: true })).toBeVisible();
    await expect(page.getByText("newer-500.pdb", { exact: true })).toHaveCount(0);
  } finally { releasePage?.(); }
});

test("an in-flight batch poll cannot move the user back after a page change", async ({ page, context }) => {
  await desktop(context);
  let releasePoll: (() => void) | undefined, held = false;
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (request.method() === "OPTIONS") return respond(route, {});
    if (url.pathname === "/status") return respond(route, status([batch("newer-run", 0, true)]));
    if (url.pathname === "/batch/newer-run") {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      if (offset === 0 && !held) { held = true; await new Promise<void>((resolve) => { releasePoll = resolve; }); }
      return respond(route, batch("newer-run", offset, true));
    }
    return respond(route, {});
  });
  try {
    await goBatch(page);
    await expect.poll(() => Boolean(releasePoll)).toBe(true);
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await expect(page.getByText("newer-500.pdb", { exact: true })).toBeVisible();
    await releaseAndPaint(page, () => releasePoll?.(), "/batch/newer-run?");
    await expect(page.getByText("newer-500.pdb", { exact: true })).toBeVisible();
    await expect(page.getByText("newer-0.pdb", { exact: true })).toHaveCount(0);
  } finally { releasePoll?.(); }
});

test("reconnecting rechecks the unchanged structure after an offline inspection", async ({ page, context }) => {
  await desktop(context);
  let offline = false, checks = 0;
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === "OPTIONS") return respond(route, {});
    if (path === "/status") return offline ? respond(route, { error: "Fixture runtime offline" }, 503) : respond(route, status());
    if (path === "/inspect") {
      checks += 1;
      return offline ? respond(route, { error: "Cannot check while offline" }, 503) : respond(route, inspection(request.postDataJSON().input_structure));
    }
    return respond(route, {});
  });
  await page.goto("/");
  await page.getByLabel("Structure file", { exact: true }).waitFor();
  offline = true;
  await page.getByLabel("Structure file", { exact: true }).fill("/data/current.pdb");
  await expect(page.getByText("Cannot check while offline", { exact: true })).toBeVisible();
  await page.getByLabel("Workspace options", { exact: true }).click();
  await page.getByRole("button", { name: "Refresh runtime status", exact: true }).click();
  await expect(page.getByRole("button", { name: "Setup required", exact: true })).toBeVisible();
  await page.locator(".primary-nav").getByRole("button", { name: "Setup", exact: true }).click();
  offline = false;
  await page.getByRole("button", { name: "Reconnect", exact: true }).click();
  await expect(page.getByRole("button", { name: "Ready", exact: true })).toBeVisible();
  await page.locator(".primary-nav").getByRole("button", { name: "Predict", exact: true }).click();
  await expect(page.getByLabel("Structure file", { exact: true })).toHaveValue("/data/current.pdb");
  await expect.poll(() => checks).toBe(2);
  await expect(page.getByRole("button", { name: "Run prediction", exact: true })).toBeEnabled();
});

test("an old stop response cannot replace a historical batch selected after completion", async ({ page, context }) => {
  await desktop(context);
  let releaseStop: (() => void) | undefined;
  let stopRequested = false;
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (request.method() === "OPTIONS") return respond(route, {});
    if (url.pathname === "/status") return respond(route, status([batch("older-run"), batch("newer-run", 0, true)]));
    if (url.pathname === "/batch/newer-run/cancel") {
      stopRequested = true;
      await new Promise<void>((resolve) => { releaseStop = resolve; });
      return respond(route, { ...batch("newer-run"), cancel_requested: true });
    }
    if (url.pathname === "/batch/newer-run") return respond(route, batch("newer-run", 0, !stopRequested));
    if (url.pathname === "/batch/older-run") return respond(route, batch("older-run"));
    return respond(route, {});
  });
  try {
    await goBatch(page);
    await page.getByRole("button", { name: "Stop batch", exact: true }).click();
    await expect.poll(() => Boolean(releaseStop)).toBe(true);
    expect(await page.locator(".batch-monitor-header .danger-action").isDisabled()).toBe(true);
    await expect(page.locator(".batch-run-title")).toContainText("Completed");
    await page.locator(".batch-history > summary").click();
    await page.locator(".batch-history button").filter({ hasText: "older-run" }).click();
    await expect(page.getByText("older-only.pdb", { exact: true })).toBeVisible();
    await releaseAndPaint(page, () => releaseStop?.(), "/batch/newer-run/cancel");
    await expect(page.getByText("older-only.pdb", { exact: true })).toBeVisible();
    await expect(page.getByText("newer-0.pdb", { exact: true })).toHaveCount(0);
  } finally { releaseStop?.(); }
});

test("stopping from a later batch page retains that page when the API returns its first page", async ({ page, context }) => {
  await desktop(context);
  let stopRequested = false;
  await page.route(`${api}/**`, async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (request.method() === "OPTIONS") return respond(route, {});
    if (url.pathname === "/status") return respond(route, status([batch("newer-run", 0, true)]));
    if (url.pathname === "/batch/newer-run/cancel") {
      stopRequested = true;
      return respond(route, { ...batch("newer-run", 0, true), cancel_requested: true });
    }
    if (url.pathname === "/batch/newer-run") return respond(route, {
      ...batch("newer-run", Number(url.searchParams.get("offset") ?? 0), true), cancel_requested: stopRequested
    });
    return respond(route, {});
  });
  await goBatch(page);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByText("newer-500.pdb", { exact: true })).toBeVisible();
  const response = page.waitForResponse((response) => response.url().endsWith("/batch/newer-run/cancel"));
  await page.getByRole("button", { name: "Stop batch", exact: true }).click();
  await (await response).finished();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  // Check the response frame, before a later status poll can hide a page jump.
  expect(await page.getByText("newer-500.pdb", { exact: true }).isVisible()).toBe(true);
  expect(await page.getByText("newer-0.pdb", { exact: true }).count()).toBe(0);
  await expect(page.locator(".batch-monitor .pager")).toContainText("501–501 of 501");
});
