import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const pages = ["setup", "predict", "batch", "results", "diagnostics"] as const;

for (const workspace of pages) {
  test(`${workspace} workspace has no serious accessibility violations`, async ({ page }) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.goto(`/?preview=${workspace}`);
    await expect(page.locator("#workspace-content")).toBeVisible();
    if (workspace === "results") {
      await page.waitForTimeout(500);
    }
    const builder = new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]);
    if (workspace === "results") {
      builder.exclude(".msp-plugin");
    }
    const results = await builder.analyze();
    expect(results.violations.filter((violation) => ["serious", "critical"].includes(violation.impact ?? ""))).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
}

test("workspace reflows to 320 CSS pixels", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 720 });
  await page.goto("/?preview=predict");
  await expect(page.locator(".predict-layout")).toBeVisible();
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  await expect(page.locator(".nav-compact-label", { hasText: "Predict" })).toBeVisible();
});

test("batch and results additions do not create narrow-screen overflow", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 720 });
  for (const workspace of ["batch", "results"]) {
    await page.goto(`/?preview=${workspace}`);
    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth
    }));
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  }
});

test("visible pointer targets meet the WCAG minimum", async ({ page }) => {
  await page.goto("/?preview=setup");
  const undersized = await page.evaluate(() => {
    const elements = [...document.querySelectorAll<HTMLElement>("button, select, summary, input:not([type=checkbox])")];
    return elements
      .filter((element) => element.offsetParent !== null && !(element as HTMLButtonElement).disabled)
      .map((element) => ({ label: element.getAttribute("aria-label") ?? element.textContent?.trim(), rect: element.getBoundingClientRect() }))
      .filter(({ rect }) => rect.width < 24 || rect.height < 24)
      .map(({ label, rect }) => ({ label, width: rect.width, height: rect.height }));
  });
  expect(undersized).toEqual([]);
});

test("keyboard navigation reaches content with a visible skip link", async ({ page }) => {
  await page.goto("/?preview=predict");
  await page.keyboard.press("Tab");
  await expect(page.locator(".skip-link")).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#workspace-content")).toBeFocused();
});

test("setup, appearance, result, and diagnostic disclosures are operable", async ({ page }) => {
  await page.goto("/?preview=setup");
  await page.getByText("Advanced runtime options").click();
  await expect(page.locator(".setup-backend .segmented")).toBeVisible();

  await page.getByLabel("Appearance").selectOption("dark");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  await page.goto("/?preview=results");
  await page.getByLabel("Displayed predicted-residue cluster").selectOption("1");
  await expect(page.getByText("0.8170").first()).toBeVisible();
  const fallback = page.getByText("3D rendering is unavailable");
  if (await fallback.isVisible()) {
    await expect(fallback).toBeVisible();
  } else {
    await expect(page.locator(".molstar-host")).toBeVisible();
  }

  await page.goto("/?preview=diagnostics");
  await page.getByText("Show runtime report").click();
  await expect(page.locator(".diagnostic-json")).toBeVisible();
});

test("results regroup locally and keep the complete residue ranking", async ({ page }) => {
  await page.goto("/?preview=results");
  await expect(page.getByText("12 ranked residues")).toBeVisible();
  await expect(page.getByText("12 scored", { exact: true })).toBeVisible();

  await page.getByLabel("Displayed score cutoff").fill("0.9");
  await expect(page.getByText("2 selected residues")).toBeVisible();
  await page.getByLabel("Displayed cluster distance (Å)").fill("1");
  await expect(page.getByText("2 displayed clusters")).toBeVisible();
  await expect(page.getByText("Original run")).toBeVisible();

  await page.getByRole("button", { name: "Reset display" }).click();
  await expect(page.getByText("8 selected residues")).toBeVisible();
  await expect(page.getByText("2 displayed clusters")).toBeVisible();

  const distanceInput = page.getByLabel("Displayed cluster distance (Å)");
  await distanceInput.fill("0");
  await expect(page.getByText("Display settings are invalid")).toBeVisible();
  await expect(distanceInput).toBeEnabled();
  await page.getByRole("button", { name: "Reset display" }).click();
  await expect(page.getByText("8 selected residues")).toBeVisible();
});

test("local regrouping matches prediction threshold, linkage, and ordering semantics", async ({ page }) => {
  await page.goto("/?preview=results");
  const result = await page.evaluate(async () => {
    const module = await import("/src/localResults.ts");
    const view = module.recomputeLocalResult([
      { residue_id: "threshold", probability: 0.5, x: -10, y: 0, z: 0 },
      { residue_id: "linked-high", probability: 0.9, x: 0, y: 0, z: 0 },
      { residue_id: "linked-low", probability: 0.7, x: 2, y: 0, z: 0 },
      { residue_id: "canonical-first", probability: 0.8, x: 20, y: 0, z: 0 },
      { residue_id: "canonical-second", probability: 0.8, x: 30, y: 0, z: 0 },
      { residue_id: "unscored", probability: 1, is_scored: 0, x: 0, y: 0, z: 0 }
    ], 0.5, 2, null);
    return {
      selectedResidueCount: view.selectedResidueCount,
      clusterResidues: view.pockets?.clustered_pockets.map((cluster) => (
        cluster.residues.map((residue) => residue.residue_id)
      )),
      firstCenter: view.pockets?.clustered_pockets[0]?.center,
      recordIds: view.records.map((record) => record.residue_id)
    };
  });
  expect(result.selectedResidueCount).toBe(4);
  expect(result.clusterResidues).toEqual([
    ["linked-high", "linked-low"],
    ["canonical-first"],
    ["canonical-second"]
  ]);
  expect(result.firstCenter?.[0]).toBeCloseTo(0.875, 10);
  expect(result.recordIds).not.toContain("unscored");
});

test("local connected-components regrouping stays within the interaction budget", async ({ page }) => {
  await page.goto("/?preview=results");
  await page.waitForTimeout(750);
  const result = await page.evaluate(async () => {
    const module = await import("/src/localResults.ts");
    const scores = Array.from({ length: 10_000 }, (_, index) => ({
      residue_id: `${String.fromCharCode(65 + index % 4)}_${index + 1}`,
      chain_id: String.fromCharCode(65 + index % 4),
      probability: 0.9,
      score: 0.9,
      x: index,
      y: 0,
      z: 0,
      rank_global: index + 1
    }));
    const started = performance.now();
    const view = module.recomputeLocalResult(scores, 0.5, 1, null);
    return {
      elapsed: performance.now() - started,
      clusterCount: view.pockets?.clustered_pockets.length,
      selectedResidueCount: view.selectedResidueCount
    };
  });
  expect(result.selectedResidueCount).toBe(10_000);
  expect(result.clusterCount).toBe(1);
  expect(result.elapsed).toBeLessThan(500);
});

test("score coverage treats an empty exact result as authoritative", async ({ page }) => {
  await page.goto("/?preview=results");
  const result = await page.evaluate(async () => {
    const module = await import("/src/components/ProtcrossScoreTheme.ts");
    return {
      exactEmpty: module.configureProtcrossScoreTheme([], []),
      unavailable: module.configureProtcrossScoreTheme([], undefined),
      insertionIdentity: module.normalizeProtcrossResidueKey(
        "model:0|chain:A|het:ATOM|resseq:10|icode:B|resname:GLY"
      )?.split("\u0000"),
      blankChainIdentity: module.normalizeProtcrossResidueKey(
        "model:0|chain: |het:ATOM|resseq:1|icode:|resname:ALA"
      )?.split("\u0000")
    };
  });
  expect(result.exactEmpty.source).toBe("result-keys");
  expect(result.exactEmpty.expectedScoredResidueCount).toBe(0);
  expect(result.unavailable.source).toBe("unavailable");
  expect(result.insertionIdentity).toEqual(["0", "A", "ATOM", "10", "B", "GLY"]);
  expect(result.blankChainIdentity).toEqual(["0", "", "ATOM", "1", "", "ALA"]);
});

test("batch monitor keeps multiline errors and exposes restored history", async ({ page }) => {
  await page.goto("/?preview=batch");
  await expect(page.getByRole("heading", { name: "Recent batches" })).toBeVisible();
  await expect(page.getByText("Recovered interrupted batch")).toBeVisible();
  await expect(page.getByRole("button", { name: "Retry failed" })).toBeVisible();
  await expect(page.getByText("Check that chain B contains Cα coordinates.")).toBeVisible();
});

test("results never present an unloaded batch item as a prediction", async ({ page }) => {
  await page.goto("/?preview=batch");
  await expect(page.getByRole("heading", { name: "Recent batches" })).toBeVisible();
  await page.getByRole("button", { name: /^Results/ }).click();
  await expect(page.getByRole("heading", { name: "No prediction loaded" })).toBeVisible();
  await expect(page.getByText(/0 selected residues/)).toHaveCount(0);
});
