import { expect, test, type Page } from "@playwright/test";

// Exercise the actual molecular controls with deterministic software WebGL.
test.use({ launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] } });

test("exploration settings and cluster selection survive navigation", async ({ page }) => {
  await page.goto("/?preview=results");
  await page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true }).fill("0.9");
  await page.locator(".result-parameters").getByLabel("Distance (Å)", { exact: true }).fill("1");
  await page.locator(".result-panel").getByRole("combobox", { name: "Cluster", exact: true }).selectOption("1");
  await page.getByRole("button", { name: /^Predict\b/ }).click();
  await expect(page.locator(".predict-layout")).toBeVisible();
  await page.getByRole("button", { name: /^Results\b/ }).click();
  await expect(page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true })).toHaveValue("0.9");
  await expect(page.locator(".result-parameters").getByLabel("Distance (Å)", { exact: true })).toHaveValue("1");
  await expect(page.locator(".result-panel").getByRole("combobox", { name: "Cluster", exact: true })).toHaveValue("1");
  await expect(page.locator(".result-counts")).toContainText("2 selected");
});

test("cleared and invalid display fields preserve the last valid view", async ({ page }) => {
  await page.goto("/?preview=results");
  const cutoff = page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true });
  await cutoff.fill("0.9");
  await expect(page.locator(".result-counts")).toContainText("2 selected");
  await cutoff.fill("");
  await expect(cutoff).toHaveValue("");
  await expect(cutoff).toHaveAttribute("aria-invalid", "true");
  await expect(page.getByText("Enter score cutoff.")).toBeVisible();
  await expect(page.locator(".result-counts")).toContainText("2 selected");
  await page.locator(".result-records > summary").click();
  await expect(page.getByRole("button", { name: "Copy view JSON" })).toBeDisabled();
  await cutoff.fill("1.1");
  await expect(page.getByText("score cutoff: 0–1.")).toBeVisible();
  await expect(page.locator(".result-counts")).toContainText("2 selected");
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(cutoff).toHaveValue("0.5");
  await expect(cutoff).toHaveAttribute("aria-invalid", "false");
  await expect(page.locator(".result-counts")).toContainText("8 selected");
  await expect(page.getByRole("button", { name: "Copy view JSON" })).toBeEnabled();
  const distance = page.locator(".result-parameters").getByLabel("Distance (Å)", { exact: true });
  await distance.fill("0");
  await expect(page.getByText("cluster distance must exceed 0 Å.")).toBeVisible();
  await expect(page.locator(".result-counts")).toContainText("8 selected");
});

test("an empty filtered result provides a direct route back to the original display", async ({ page }) => {
  await page.goto("/?preview=results");
  await page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true }).fill("1");
  await expect(page.locator(".result-counts")).toContainText("0 selected");
  await expect(page.locator(".result-panel").getByText("No cluster. Lower the score cutoff.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Copy score-weighted centroid" })).toBeDisabled();
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(page.locator(".result-counts")).toContainText("8 selected");
  await expect(page.getByRole("button", { name: "Copy score-weighted centroid" })).toBeEnabled();
});

test("copied view settings retain their source and never replace original run metadata", async ({ page }) => {
  await page.addInitScript(() => {
    (window as unknown as { copiedRecords: string[] }).copiedRecords = [];
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async (text: string) => {
        (window as unknown as { copiedRecords: string[] }).copiedRecords.push(text);
      } }
    });
  });
  await page.goto("/?preview=results");
  await page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true }).fill("0.9");
  await page.locator(".result-records > summary").click();
  await page.getByRole("button", { name: "Copy run JSON" }).click();
  await page.getByRole("button", { name: "Copy view JSON" }).click();
  const [run, view] = await page.evaluate(() => (
    (window as unknown as { copiedRecords: string[] }).copiedRecords.map((record) => JSON.parse(record))
  ));
  expect(run.original_run.threshold).toBe(0.5);
  expect(view.original_settings.threshold).toBe(0.5);
  expect(view.display_settings.threshold).toBe(0.9);
  expect(view.source_identity).toEqual(run.source_identity);
  expect(view.source_identity.input_structure).toBeTruthy();
  expect(view.view_only).toBe(true);
  expect(view.output_files_modified).toBe(false);
});

test("a blocked clipboard leaves the original and current JSON available for manual copying", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async () => { throw new Error("Clipboard unavailable"); } }
    });
  });
  await page.goto("/?preview=results");
  await page.locator(".result-records > summary").click();
  await page.getByRole("button", { name: "Copy run JSON" }).click();
  await expect(page.getByText(/Could not copy run record: Clipboard unavailable/)).toBeVisible();
  await page.locator(".result-records").getByText("Saved run", { exact: true }).click();
  const originalJson = page.getByLabel("Original run metadata JSON");
  await expect(originalJson).toBeVisible();
  await originalJson.focus();
  await expect(originalJson).toBeFocused();
  expect(JSON.parse(await originalJson.innerText()).original_run.threshold).toBe(0.5);
  await page.locator(".result-records").getByText("Current view", { exact: true }).click();
  await expect(page.getByLabel("Current view settings JSON")).toBeVisible();
});

test("expanded reproducibility records remain usable in a narrow window", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 720 });
  await page.goto("/?preview=results");
  await page.locator(".result-records > summary").click();
  await page.locator(".result-records").getByText("Saved run", { exact: true }).click();
  await page.locator(".result-records").getByText("Current view", { exact: true }).click();
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  await expect(page.getByLabel("Original run metadata JSON")).toBeVisible();
  await expect(page.getByRole("button", { name: "Copy view JSON" })).toBeVisible();
});

test("the bundled scientific example exposes exploration without local-output actions", async ({ page }) => {
  await page.goto("/?preview=setup");
  await page.getByRole("button", { name: "Example", exact: true }).click();
  await expect(page.getByText("Example · 1CRN", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Output folder" })).toHaveCount(0);
  const threshold = page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true });
  await threshold.fill("0.555");
  expect(await threshold.evaluate((element: HTMLInputElement) => element.validity.valid)).toBe(true);
  const distance = page.locator(".result-parameters").getByLabel("Distance (Å)", { exact: true });
  await distance.fill("8.25");
  expect(await distance.evaluate((element: HTMLInputElement) => element.validity.valid)).toBe(true);
  await page.locator(".result-records > summary").click();
  await expect(page.getByText("Bundled Crambin prediction · PDB 1CRN.")).toBeVisible();
  await expect(page.locator(".output-file")).toHaveCount(0);
  await page.locator(".result-records").getByText("Current view", { exact: true }).click();
  const view = JSON.parse(await page.getByLabel("Current view settings JSON").innerText());
  expect(view.source_identity.bundled_example).toBe(true);
  expect(view.source_identity.input_sha256).toBeTruthy();
  expect(view.display_settings.threshold).toBe(0.555);
  expect(view.display_settings.cluster_cutoff).toBe(8.25);
});


test.describe("molecular controls", () => {
  test("3D tools preserve score legend semantics through custom colors and restoration", async ({ page }) => {
    await page.goto("/?preview=setup");
    await page.getByRole("button", { name: "Example", exact: true }).click();
    await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
    const tools = page.getByRole("button", { name: "3D tools", exact: true });
    const panel = page.locator(".molstar-host .msp-layout-right");
    await expect(panel).toBeHidden();
    await expect(page.locator(".score-legend-custom")).toHaveCount(0);
    await expect(page.locator(".viewer-footer .score-gradient")).toBeVisible();
    await tools.click();
    await expect(page.getByRole("button", { name: "Hide tools", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(panel).toBeVisible();
    await panel.getByRole("button", { name: "Polymer Cartoon", exact: true })
      .locator("..").getByRole("button", { name: "Actions", exact: true }).click();
    await panel.getByRole("button", { name: "Set Coloring", exact: true }).click();
    await panel.getByRole("button", { name: "Uniform", exact: true }).click();
    await expect(page.locator(".score-legend-custom")).toContainText("Custom colors");
    await expect(page.locator(".viewer-footer .score-gradient")).toHaveCount(0);
    const clusterStatus = await page.locator(".viewer-status").innerText();
    await page.getByRole("button", { name: "Restore score colors", exact: true }).click();
    await expect(page.locator(".score-legend-custom")).toHaveCount(0);
    await expect(page.locator(".viewer-footer .score-gradient")).toBeVisible();
    await expect(page.locator(".viewer-status")).toHaveText(clusterStatus, { useInnerText: true });
    const selectedCluster = panel.getByRole("button", { name: "Selected cluster Ball & Stick", exact: true });
    await expect(selectedCluster).toHaveCount(1);
    await page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true }).fill("1");
    await expect(selectedCluster).toHaveCount(0);
    await expect(page.locator(".viewer-status")).toContainText("No cluster at this cutoff.");
    await page.locator(".result-parameters").getByRole("button", { name: "Reset", exact: true }).click();
    await expect(selectedCluster).toHaveCount(1);
    await expect(page.locator(".viewer-footer .score-gradient")).toBeVisible();
    await page.getByRole("button", { name: "Hide tools", exact: true }).click();
    await expect(panel).toBeHidden();
    await expect(page.getByRole("button", { name: "3D tools", exact: true })).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator(".viewer-footer .score-legend")).toBeVisible();
  });
});


// Mount the real component with explicit incomplete-file fixtures. Normal result
// opening requires pockets JSON; summary-only cases exercise defensive rendering.
async function mountSavedResult(page: Page) {
  await page.goto("/?preview=setup");
  await page.evaluate(async () => {
    const load = (path: string) => import(/* @vite-ignore */ path);
    const React = await load("/node_modules/.vite/deps/react.js");
    const ReactDOM = await load("/node_modules/.vite/deps/react-dom_client.js");
    const { ResultsPanel } = await load("/src/components/ResultsPanel.tsx");
    const { exampleResult, exampleStructureData } = await load("/src/exampleResult.ts");
    document.getElementById("root")!.hidden = true;
    const host = document.createElement("main");
    host.className = "content";
    document.body.append(host);
    const root = (ReactDOM.createRoot ?? ReactDOM.default.createRoot)(host);
    const fixture = {
      props: {
        summary: exampleResult.summary,
        pockets: exampleResult.pockets,
        scores: exampleResult.scores,
        residues: exampleResult.top_pocket_residues ?? [],
        outputFiles: exampleResult.output_files,
        structurePath: exampleResult.output_files.structure,
        structureData: exampleStructureData,
        darkMode: false,
        connected: true,
        sample: true,
        onOpenExisting: () => {},
        onOpenSetup: () => {},
        onPrepareRuntime: () => { fixture.prepareCalls++; },
        onNotify: () => {},
        onError: () => {}
      },
      prepareCalls: 0,
      render: (changes: Record<string, unknown>) => root.render(
        (React.createElement ?? React.default.createElement)(ResultsPanel, { ...fixture.props, ...changes })
      )
    };
    (window as any).savedResultFixture = fixture;
    fixture.render({});
  });
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
}

// Read actual Mol* state through its mounted React owner; never expose a debug API
// in the application. This checks canvas state as well as the user-facing legend.
async function molecularScene(page: Page) {
  return page.evaluate(() => {
    const element = document.querySelector(".msp-plugin") as any;
    let fiber = element?.[Object.keys(element).find((key) => key.startsWith("__reactFiber$"))!];
    while (fiber) {
      if (fiber.stateNode?.plugin?.canvas3d) {
        const plugin = fiber.stateNode.plugin;
        const components = plugin.managers.structure.hierarchy.current.structures.flatMap((structure: any) => structure.components);
        const representations = components.flatMap((component: any) => component.representations.map((representation: any) => ({
          selected: component.key === "structure-component-protcross-selected-predicted-cluster",
          paintedLayers: (representation.cell.obj?.data?.repr?.state?.overpaint?.layers ?? []).filter((layer: any) => !layer.clear).length
        })));
        return {
          structures: plugin.managers.structure.hierarchy.current.structures.length as number,
          camera: plugin.canvas3d.camera.getSnapshot(),
          basePaintedLayers: representations.filter((representation: any) => !representation.selected)
            .reduce((sum: number, representation: any) => sum + representation.paintedLayers, 0) as number,
          clusterPaintedLayers: representations.filter((representation: any) => representation.selected)
            .reduce((sum: number, representation: any) => sum + representation.paintedLayers, 0) as number
        };
      }
      fiber = fiber.return;
    }
    throw new Error("The molecular viewer is not mounted");
  });
}

test("a saved result without its score table retains recorded counts and available clusters", async ({ page }) => {
  await mountSavedResult(page);
  await page.evaluate(() => (window as any).savedResultFixture.render({ scores: [] }));
  await expect(page.locator(".result-counts")).toHaveText("27 selected · 1 cluster · 46 scored residues · saved");
  await expect(page.locator(".viewer-status")).toContainText("Residue identities missing: score colors unavailable.");
  await expect(page.locator(".score-legend-unavailable")).toHaveText("Score colors unavailable");
  await expect(page.locator(".viewer-footer .score-gradient")).toHaveCount(0);
  await expect(page.locator(".viewer-footer .unscored-key")).toHaveCount(0);
  await page.locator(".viewer-display-notes > summary").click();
  await expect(page.locator(".viewer-display-notes")).toContainText("gray does not indicate whether a residue was scored");
  await expect(page.locator(".result-parameters").getByLabel("Score cutoff", { exact: true })).toBeDisabled();
  await expect(page.locator(".residue-table tbody tr")).toHaveCount(27);
  await expect(page.getByRole("button", { name: "Copy score-weighted centroid", exact: true })).toBeEnabled();
  await page.getByText("Residue ranking unavailable", { exact: true }).click();
  await expect(page.getByText("The complete score table is unavailable.", { exact: true })).toBeVisible();
  await page.locator(".result-records > summary").click();
  await page.locator(".result-records").getByText("Current view", { exact: true }).click();
  const record = JSON.parse(await page.getByLabel("Current view settings JSON").innerText());
  expect(record.selected_residue_count).toBe(27);
  expect(record.displayed_cluster_count).toBe(1);
});

test("incomplete summaries show known top-cluster membership without fabricating unknown statistics", async ({ page }) => {
  await mountSavedResult(page);
  await page.evaluate(() => {
    const fixture = (window as any).savedResultFixture;
    fixture.render({ pockets: null, scores: [], summary: {
      ...fixture.props.summary,
      residues_scored: undefined,
      selected_residue_count: undefined,
      cluster_count: undefined,
      top_pocket: { ...fixture.props.summary.top_pocket, score_max: undefined, score_mean: undefined, center: null }
    } });
  });
  await expect(page.locator(".result-counts")).toHaveText("— selected · — clusters · — scored residues · saved");
  await expect(page.locator(".residue-table tbody tr")).toHaveCount(27);
  await expect(page.locator(".viewer-status")).toContainText("27 residues in selected cluster");
  await expect(page.locator(".metric-row")).toContainText("Max score—");
  await expect(page.locator(".metric-row")).toContainText("Mean score—");
  await expect(page.getByRole("button", { name: "Copy score-weighted centroid", exact: true })).toBeDisabled();
  await page.locator(".result-records > summary").click();
  await page.locator(".result-records").getByText("Current view", { exact: true }).click();
  let record = JSON.parse(await page.getByLabel("Current view settings JSON").innerText());
  expect(record.selected_residue_count).toBeNull();
  expect(record.display_settings.selected_cluster_id).toBe(await page.evaluate(
    () => (window as any).savedResultFixture.props.summary.top_pocket.cluster_id
  ));
  await page.evaluate(() => {
    const fixture = (window as any).savedResultFixture;
    fixture.render({ pockets: null, scores: [], residues: [], summary: {
      ...fixture.props.summary, residues_scored: undefined, selected_residue_count: undefined,
      cluster_count: undefined, top_pocket: null
    } });
  });
  await expect(page.locator(".residue-table tbody tr")).toHaveCount(0);
  await expect(page.locator(".viewer-status")).toContainText("No saved cluster.");
  record = JSON.parse(await page.getByLabel("Current view settings JSON").innerText());
  expect(record.displayed_cluster_count).toBeNull();
  expect(record.display_settings.selected_cluster_id).toBeNull();
});

test("switching from a rendered structure to missing structure clears the scene and can recover", async ({ page }) => {
  await mountSavedResult(page);
  for (let iteration = 0; iteration < 2; iteration++) {
    await page.evaluate(() => (window as any).savedResultFixture.render({ structurePath: undefined, structureData: undefined }));
    await expect(page.getByText("No structure file", { exact: true })).toBeVisible();
    await expect.poll(async () => (await molecularScene(page)).structures).toBe(0);
    await expect(page.getByRole("button", { name: "3D tools", exact: true })).toBeDisabled();
    await expect(page.locator(".score-legend")).toHaveCount(0);
    await expect(page.locator(".viewer-status")).toBeEmpty();
    await page.evaluate(() => (window as any).savedResultFixture.render({}));
    await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
    await expect(page.getByRole("button", { name: "3D tools", exact: true })).toBeEnabled();
    await expect.poll(async () => (await molecularScene(page)).structures).toBe(1);
  }
});

test("selection painting deactivates the score scale and restoration preserves cluster paint and camera", async ({ page }) => {
  await page.goto("/?preview=setup");
  await page.getByRole("button", { name: "Example", exact: true }).click();
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
  await page.getByRole("button", { name: "3D tools", exact: true }).click();
  await page.getByRole("button", { name: "Toggle Selection Mode", exact: true }).click();
  await page.getByRole("button", { name: "Apply Theme to Selection", exact: true }).click();
  await page.getByRole("button", { name: "Apply Theme", exact: true }).click();
  await expect(page.locator(".score-legend-custom")).toContainText("Custom colors");
  await expect.poll(async () => (await molecularScene(page)).basePaintedLayers).toBeGreaterThan(0);
  const before = await molecularScene(page);
  expect(before.clusterPaintedLayers).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Restore score colors", exact: true }).click();
  await expect(page.locator(".score-gradient")).toBeVisible();
  const after = await molecularScene(page);
  expect(after.basePaintedLayers).toBe(0);
  expect(after.clusterPaintedLayers).toBe(before.clusterPaintedLayers);
  expect(after.camera).toEqual(before.camera);
  await expect(page.locator(".viewer-status")).toContainText("27 residues in selected cluster");
});

test("hiding or removing the scored representation offers a working score-view recovery", async ({ page }) => {
  await page.goto("/?preview=setup");
  await page.getByRole("button", { name: "Example", exact: true }).click();
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped");
  await page.getByRole("button", { name: "3D tools", exact: true }).click();
  const panel = page.locator(".msp-layout-right");
  const polymer = panel.getByRole("button", { name: "Polymer Cartoon", exact: true }).locator("..");
  await polymer.getByRole("button", { name: "Hide component", exact: true }).click();
  await expect(page.locator(".score-legend-custom")).toContainText("Score view hidden");
  await expect(page.locator(".score-gradient")).toHaveCount(0);
  await page.getByRole("button", { name: "Show score view", exact: true }).click();
  await expect(polymer.getByRole("button", { name: "Hide component", exact: true })).toBeVisible();
  await expect(page.locator(".score-gradient")).toBeVisible();
  await polymer.getByRole("button", { name: "Remove", exact: true }).click();
  await expect(panel.getByRole("button", { name: "Polymer Cartoon", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Show score view", exact: true }).click();
  await expect(panel.getByRole("button", { name: "Polymer Cartoon", exact: true })).toHaveCount(1);
  await expect(panel.getByRole("button", { name: "Selected cluster Ball & Stick", exact: true })).toHaveCount(1);
  await expect(page.locator(".score-gradient")).toBeVisible();
});

test("offline results offer the runtime-only action and disable repeat preparation while busy", async ({ page }) => {
  await mountSavedResult(page);
  await page.setViewportSize({ width: 320, height: 720 });
  await page.evaluate(() => (window as any).savedResultFixture.render({ connected: false }));
  const prepare = page.getByRole("button", { name: "Prepare result viewer", exact: true });
  await expect(prepare).toBeVisible();
  await expect(prepare).toHaveAttribute("title", /CPU runtime.*no model download/);
  await expect(page.getByRole("button", { name: "Runtime setup", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open result", exact: true })).toBeDisabled();
  await prepare.click();
  expect(await page.evaluate(() => (window as any).savedResultFixture.prepareCalls)).toBe(1);
  await page.evaluate(() => (window as any).savedResultFixture.render({ connected: false, busy: true }));
  await expect(prepare).toBeDisabled();
  const dimensions = await page.evaluate(() => ({ client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.client);
});


test("zero scores, known empty domains and partial mappings retain the applicable score scale", async ({ page }) => {
  await mountSavedResult(page);
  await page.evaluate(() => {
    const fixture = (window as any).savedResultFixture;
    fixture.render({
      scores: fixture.props.scores.map((residue: any) => ({ ...residue, score: 0, probability: 0 })),
      structureData: fixture.props.structureData.split("\n").map((line: string) => (
        line.startsWith("ATOM  ") || line.startsWith("HETATM") ? `${line.slice(0, 60)}  0.00${line.slice(66)}` : line
      )).join("\n")
    });
  });
  await expect(page.locator(".result-counts")).toContainText("0 selected");
  await expect(page.locator(".viewer-status")).toContainText("46 scored residues mapped · 0 unscored");
  await expect(page.locator(".score-gradient")).toBeVisible();
  await expect(page.locator(".score-legend-unavailable")).toHaveCount(0);
  await page.evaluate(() => {
    const fixture = (window as any).savedResultFixture;
    fixture.render({ scores: fixture.props.scores.map((residue: any) => ({ ...residue, is_scored: 0 })) });
  });
  await expect(page.locator(".viewer-status")).toContainText("0 scored residues mapped · 46 unscored");
  await expect(page.locator(".score-gradient")).toBeVisible();
  await expect(page.locator(".unscored-key")).toHaveText("Not scored");
  await page.evaluate(() => {
    const fixture = (window as any).savedResultFixture;
    fixture.render({ scores: fixture.props.scores.map((residue: any, index: number) => index ? residue : {
      ...residue, residue_key: residue.residue_key.replace(/resseq:[^|]+/, "resseq:99999")
    }) });
  });
  await expect(page.locator(".viewer-status")).toContainText("45/46 scored residues mapped · 1 unmatched");
  await expect(page.locator(".score-gradient")).toBeVisible();
  await expect(page.locator(".score-legend-unavailable")).toHaveCount(0);
});
