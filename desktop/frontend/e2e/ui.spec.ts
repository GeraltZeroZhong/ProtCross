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
