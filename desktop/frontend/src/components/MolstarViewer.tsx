import { useEffect, useRef, useState } from "react";
import "molstar/build/viewer/molstar.css";
import { StructureSelection } from "molstar/lib/mol-model/structure";
import type { Expression } from "molstar/lib/mol-script/language/expression";
import { MolScriptBuilder as MS } from "molstar/lib/mol-script/language/builder";
import { Script } from "molstar/lib/mol-script/script";
import { fetchDesktopFile } from "../api";
import type { PocketJson, ResidueSummary, SummaryJson } from "../types";
import {
  configureProtcrossScoreTheme,
  PROTCROSS_UNSCORED_COLOR_CSS,
  ProtcrossScoreColorThemeProvider,
  type ProtcrossScoreCoverage
} from "./ProtcrossScoreTheme";
import { Icon } from "./Icon";

interface Props {
  structurePath?: string;
  structureData?: string;
  summary?: SummaryJson | null;
  pockets?: PocketJson | null;
  selectedClusterIndex?: number;
  scoredResidueKeys?: readonly string[];
  darkMode?: boolean;
}

export function MolstarViewer({
  structurePath,
  structureData,
  summary,
  pockets,
  selectedClusterIndex = 0,
  scoredResidueKeys,
  darkMode = false
}: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<any>(null);
  const operationQueueRef = useRef<Promise<void>>(Promise.resolve());
  const structureRequestRef = useRef(0);
  const selectionRequestRef = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const [selectionMessage, setSelectionMessage] = useState<string | null>(null);
  const [viewerReady, setViewerReady] = useState(false);
  const [showControls, setShowControls] = useState(false);
  const [scoreColorState, setScoreColorState] = useState<"score" | "custom" | "hidden">("score");
  const scoreColorsActive = scoreColorState === "score";
  const [restoringColors, setRestoringColors] = useState(false);
  const [webglAvailable, setWebglAvailable] = useState<boolean | null>(null);
  const [loadedStructureRequest, setLoadedStructureRequest] = useState(0);
  const [scoreCoverage, setScoreCoverage] = useState<ProtcrossScoreCoverage | null>(null);
  const scoreMappingUnavailable = scoreCoverage?.source === "unavailable";
  const selectedCluster = pockets?.clustered_pockets?.[selectedClusterIndex] ?? null;
  const hasClusterData = Array.isArray(pockets?.clustered_pockets);
  const scoredResidueKeySignature = stableScoredResidueKeySignature(scoredResidueKeys);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      if (!hostRef.current || viewerRef.current) {
        return;
      }
      try {
        const molstar = await import("molstar/build/viewer/molstar");
        // An inactive mount must not create (then dispose) a viewer in the live host.
        if (cancelled || !hostRef.current || viewerRef.current) return;
        const viewer = await molstar.Viewer.create(hostRef.current, {
          extensions: [],
          layoutIsExpanded: false,
          layoutShowControls: false,
          layoutShowRemoteState: false,
          layoutShowSequence: false,
          layoutShowLog: false,
          layoutShowLeftPanel: false,
          viewportShowExpand: true,
          viewportShowControls: false,
          viewportShowSelectionMode: true,
          viewportShowAnimation: false,
          viewportShowTrajectoryControls: false,
          volumeStreamingDisabled: true,
          backgroundColor: darkMode ? 0x101820 : 0xffffff
        });
        if (cancelled) {
          disposeViewer(viewer);
          return;
        }
        viewerRef.current = viewer;
        setWebglAvailable(Boolean(viewer.plugin.canvas3d));
        viewer.plugin.representation.structure.themes.colorThemeRegistry.add(ProtcrossScoreColorThemeProvider);
        viewer.plugin.managers.interactivity.setProps({ granularity: "residue" });
        if (!cancelled) {
          setViewerReady(true);
        }
      } catch (exc) {
        if (!cancelled) {
          setError(exc instanceof Error ? exc.message : String(exc));
        }
      }
    }
    void init();
    return () => {
      cancelled = true;
      structureRequestRef.current += 1;
      selectionRequestRef.current += 1;
      disposeViewer(viewerRef.current);
      viewerRef.current = null;
      setViewerReady(false);
      setWebglAvailable(null);
    };
  }, []);

  useEffect(() => {
    viewerRef.current?.plugin?.canvas3d?.setProps({
      renderer: { backgroundColor: darkMode ? 0x101820 : 0xffffff },
      transparentBackground: false
    });
  }, [darkMode, viewerReady]);

  useEffect(() => {
    viewerRef.current?.plugin?.layout?.updateProps({ showControls });
  }, [showControls, viewerReady]);

  // Reopening a saved run may overwrite the same path with the same residue keys.
  // A new summary object also invalidates the loaded coordinates and score colors.
  useEffect(() => {
    if (!viewerReady || !viewerRef.current) {
      setLoadedStructureRequest(0);
      setScoreCoverage(null);
      return;
    }
    const request = structureRequestRef.current + 1;
    const controller = new AbortController();
    structureRequestRef.current = request;
    selectionRequestRef.current += 1;
    setLoadedStructureRequest(0);
    setScoreCoverage(null);
    setError(null);
    setSelectionMessage(null);
    setScoreColorState("score");

    const operation = operationQueueRef.current.catch(() => undefined).then(async () => {
      if (request !== structureRequestRef.current || !viewerRef.current) {
        return;
      }
      try {
        const viewer = viewerRef.current;
        await viewer.plugin.clear();
        if (request !== structureRequestRef.current || !viewerRef.current || !structurePath) {
          return;
        }
        const blob = structureData !== undefined
          ? new Blob([structureData], { type: "text/plain" })
          : await fetchDesktopFile(structurePath, controller.signal);
        if (request !== structureRequestRef.current || !viewerRef.current) {
          return;
        }
        const url = URL.createObjectURL(blob);
        const format = structurePath.toLowerCase().endsWith(".cif") || structurePath.toLowerCase().endsWith(".mmcif")
          ? "mmcif"
          : "pdb";
        try {
          await viewer.loadStructureFromUrl(url, format, false);
        } finally {
          URL.revokeObjectURL(url);
        }
        const loadedStructures = (viewer.plugin.managers.structure.hierarchy.current.structures ?? [])
          .map((entry: any) => entry?.cell?.obj?.data)
          .filter(Boolean);
        const coverage = configureProtcrossScoreTheme(loadedStructures, scoredResidueKeys);
        await applyScoreTheme(viewer);
        if (request === structureRequestRef.current && viewerRef.current) {
          setScoreCoverage(coverage);
          setLoadedStructureRequest(request);
        }
      } catch (exc) {
        if (request === structureRequestRef.current) {
          setError(exc instanceof Error ? exc.message : String(exc));
        }
      }
    });
    operationQueueRef.current = operation;
    return () => {
      controller.abort();
      if (structureRequestRef.current === request) {
        structureRequestRef.current += 1;
      }
    };
  }, [structurePath, structureData, summary, viewerReady, scoredResidueKeySignature]);

  useEffect(() => {
    if (
      !viewerReady ||
      !viewerRef.current ||
      loadedStructureRequest === 0 ||
      loadedStructureRequest !== structureRequestRef.current
    ) {
      return;
    }
    const request = selectionRequestRef.current + 1;
    selectionRequestRef.current = request;
    const structureRequest = loadedStructureRequest;
    const residues = selectedCluster?.residues ?? [];
    const operation = operationQueueRef.current.catch(() => undefined).then(async () => {
      if (
        request !== selectionRequestRef.current ||
        structureRequest !== structureRequestRef.current ||
        !viewerRef.current
      ) {
        return;
      }
      try {
        const message = await selectPredictedCluster(viewerRef.current, residues);
        if (
          request === selectionRequestRef.current &&
          structureRequest === structureRequestRef.current
        ) {
          setSelectionMessage(hasClusterData ? message : "No saved cluster.");
        }
      } catch (exc) {
        if (request === selectionRequestRef.current) {
          setError(exc instanceof Error ? exc.message : String(exc));
        }
      }
    });
    operationQueueRef.current = operation;
    return () => {
      if (selectionRequestRef.current === request) {
        selectionRequestRef.current += 1;
      }
    };
  }, [loadedStructureRequest, selectedCluster, hasClusterData, viewerReady]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewerReady || !viewer || loadedStructureRequest === 0) return;
    const updateThemeState = (event?: { inTransaction?: boolean }) => {
      if (!event?.inTransaction) setScoreColorState(getScoreColorState(viewer));
    };
    const subscriptions = [
      viewer.plugin.state.data.events.changed.subscribe(updateThemeState),
      viewer.plugin.state.data.events.cell.stateUpdated.subscribe(updateThemeState)
    ];
    updateThemeState();
    return () => subscriptions.forEach((subscription) => subscription.unsubscribe());
  }, [viewerReady, loadedStructureRequest]);

  function restoreScoreColors() {
    const viewer = viewerRef.current;
    const request = loadedStructureRequest;
    if (!viewer || request === 0 || restoringColors) return;
    setRestoringColors(true);
    const operation = operationQueueRef.current.catch(() => undefined).then(async () => {
      try {
        if (viewerRef.current !== viewer || structureRequestRef.current !== request) return;
        await applyScoreTheme(viewer, scoreColorState === "hidden");
        if (viewerRef.current === viewer) setScoreColorState(getScoreColorState(viewer));
      } catch (exc) {
        if (viewerRef.current === viewer) setError(exc instanceof Error ? exc.message : String(exc));
      } finally {
        if (viewerRef.current === viewer) setRestoringColors(false);
      }
    });
    operationQueueRef.current = operation;
  }

  return (
    <section className="viewer-panel">
      <div className="viewer-toolbar">
        <h3>Structure</h3>
        <button aria-pressed={showControls} className="viewer-tools-button" disabled={!viewerReady || webglAvailable === false || !structurePath || loadedStructureRequest === 0} onClick={() => setShowControls((current) => !current)}>
          <Icon name="settings" size={15} /> {showControls ? "Hide tools" : "3D tools"}
        </button>
      </div>
      <div className={`molstar-frame ${webglAvailable === false ? "viewer-unavailable" : ""}`}>
        <div className="molstar-host" ref={hostRef} />
        {webglAvailable === false || !structurePath ? (
          <div className="viewer-fallback" role="status">
            <Icon name={structurePath ? "warning" : "file"} size={24} />
            <strong>{structurePath ? "3D unavailable" : "No structure file"}</strong>
            <p>{structurePath ? "Enable hardware acceleration and restart. Residue data remain available." : "Residue data remain available in the inspector."}</p>
          </div>
        ) : null}

      </div>
      {error ? <div className="inline-error" role="alert">{error}</div> : null}
      <div className="viewer-footer">
        {loadedStructureRequest > 0 && webglAvailable !== false ? scoreColorsActive ? scoreMappingUnavailable ? (
          <div className="score-legend score-legend-unavailable" role="status" aria-label="Model score colors unavailable">
            <div className="legend-heading"><span>Score colors unavailable</span></div>
          </div>
        ) : (
        <div className="score-legend" aria-label="ProtCross model score color scale from zero to one; residues not scored by the model are neutral gray">
          <div className="legend-heading"><span>Model score</span><span>uncalibrated</span></div>
          <div className="score-gradient" aria-hidden="true" />
          <div className="score-scale-ticks"><span>0.00</span><span>0.50</span><span>1.00</span></div>
          <div className="unscored-key"><span aria-hidden="true" style={{ background: PROTCROSS_UNSCORED_COLOR_CSS }} /><span>Not scored</span></div>
        </div>
        ) : (
          <div className="score-legend score-legend-custom" role="status" aria-label={scoreColorState === "hidden" ? "Score view hidden; model score scale is inactive" : "Custom 3D colors; model score scale is inactive"}>
            <div className="legend-heading"><span>{scoreColorState === "hidden" ? "Score view hidden" : "Custom colors"}</span></div>
            <button aria-label={scoreColorState === "hidden" ? "Show score view" : "Restore score colors"} title={scoreColorState === "hidden" ? "Show the structure with model score colors" : "Restore model score colors, including painted residues"} disabled={restoringColors} onClick={restoreScoreColors}><Icon name="refresh" size={14} />{restoringColors ? "Restoring…" : scoreColorState === "hidden" ? "Show scores" : "Score colors"}</button>
          </div>
        ) : null}
        <div className="viewer-status" role="status">
          {selectionMessage ? <span>{selectionMessage}</span> : null}
          {scoreCoverage ? <span>{scoreCoverageMessage(scoreCoverage)}</span> : null}
        </div>
        <details className="viewer-display-notes">
          <summary><Icon name="info" size={14} /> Display key</summary>
          <p>{scoreColorsActive ? scoreMappingUnavailable ? "Residue identities are unavailable; gray does not indicate whether a residue was scored." : "Unscored residues are gray; score 0 uses the 0.00 color." : scoreColorState === "hidden" ? "No score representation is visible. Show scores to restore it." : "Custom colors are active. Restore score colors for the model palette."} Default: ball-and-stick marks the cluster. Centroids have no 3D marker.</p>
        </details>
      </div>
    </section>
  );
}

function stableScoredResidueKeySignature(keys: readonly string[] | undefined): string {
  if (keys === undefined) {
    return "metadata-fallback";
  }
  return `result-keys\u0000${[...new Set(keys.map((key) => key.trim()).filter(Boolean))].sort().join("\u0000")}`;
}

function scoreCoverageMessage(coverage: ProtcrossScoreCoverage): string {
  if (coverage.source === "result-keys") {
    return `${coverage.scoredResidueCount} scored residues mapped · ${coverage.unscoredResidueCount} unscored`;
  }
  if (coverage.source === "result-keys-partial") {
    return `${coverage.scoredResidueCount}/${coverage.expectedScoredResidueCount ?? "?"} scored residues mapped · ${coverage.unmatchedScoredResidueCount} unmatched`;
  }
  return "Residue identities missing: score colors unavailable.";
}

function isSelectedClusterComponent(component: any): boolean {
  // Mol* prefixes component keys when storing them as transform tags.
  return (component.cell?.transform?.tags ?? []).includes("structure-component-protcross-selected-predicted-cluster");
}

function getScoreColorState(viewer: any): "score" | "custom" | "hidden" {
  const structures = viewer.plugin.managers.structure.hierarchy.current.structures ?? [];
  const representations = structures
    .filter((structure: any) => !structure.cell.state.isHidden)
    .flatMap((structure: any) => (structure.components ?? [])
      .filter((component: any) => !isSelectedClusterComponent(component) && !component.cell.state.isHidden)
      .flatMap((component: any) => component.representations ?? []))
    .filter((representation: any) => !representation.cell.state.isHidden);
  if (!representations.length) return "hidden";
  return representations.every((representation: any) => {
    const state = representation.cell.obj?.data?.repr?.state;
    const painted = (state?.themeStrength?.overpaint ?? 1) > 0
      && (state?.overpaint?.layers ?? []).some((layer: any) => !layer.clear);
    return representation.cell.transform.params?.colorTheme?.name === "protcross-score" && !painted;
  }) ? "score" : "custom";
}

async function applyScoreTheme(viewer: any, reveal = false): Promise<void> {
  const plugin = viewer.plugin;
  for (const structure of plugin.managers.structure.hierarchy.current.structures ?? []) {
    let components = (structure.components ?? []).filter((component: any) => !isSelectedClusterComponent(component));
    if (reveal && !components.some((component: any) => component.representations?.length)) {
      const polymer = await plugin.builders.structure.tryCreateComponentStatic(structure.cell, "polymer");
      if (polymer) {
        await plugin.builders.structure.representation.addRepresentation(polymer, { type: "cartoon", color: "protcross-score" });
      }
      const refreshed = plugin.managers.structure.hierarchy.current.structures.find((item: any) => item.cell.transform.ref === structure.cell.transform.ref);
      components = (refreshed?.components ?? []).filter((component: any) => !isSelectedClusterComponent(component));
    }
    if (reveal) {
      plugin.state.data.updateCellState(structure.cell.transform.ref, { isHidden: false });
      for (const component of components) {
        if (component.cell.state.isHidden) plugin.managers.structure.component.toggleVisibility([component]);
        for (const representation of component.representations ?? []) {
          if (representation.cell.state.isHidden) plugin.managers.structure.component.toggleVisibility([component], representation);
        }
      }
    }
    await plugin.managers.structure.component.updateRepresentationsTheme(components, { color: "protcross-score" as any });
    // Selection coloring is a child transform; changing the base theme does not remove it.
    const representationRefs = new Set(components.flatMap((component: any) => (component.representations ?? []).map((representation: any) => representation.cell.transform.ref)));
    const paintCells = [...plugin.state.data.cells.values()].filter((cell: any) => representationRefs.has(cell.transform.parent)
      && cell.transform.transformer.id.startsWith("ms-plugin.overpaint-structure-representation-3d-")) as any[];
    if (paintCells.length) {
      const update = plugin.state.data.build();
      for (const cell of paintCells) update.delete(cell.transform.ref);
      await update.commit({ canUndo: "Restore score colors" });
    }
  }
}

function disposeViewer(viewer: any): void {
  try {
    if (viewer?.dispose) {
      viewer.dispose();
      return;
    }
    viewer?.plugin?.dispose?.();
  } catch {
    // Best-effort cleanup for Mol* plugin/WebGL resources.
  }
}

interface AuthResidueSelector {
  authAsymId: string;
  authSeqId: number;
  insertionCode: string;
  hasInsertionCode: boolean;
}

async function selectPredictedCluster(viewer: any, residues: ResidueSummary[]): Promise<string> {
  const plugin = viewer.plugin;
  await clearClusterSelection(viewer);
  if (residues.length === 0) {
    return "No cluster at this cutoff.";
  }

  const selectors = uniqueAuthResidueSelectors(residues);
  if (selectors.length === 0) {
    return "3D selection unavailable: missing auth residue identifiers.";
  }

  const structureRef = plugin.managers.structure.hierarchy.current.structures?.[0];
  const structure = structureRef?.cell?.obj?.data;
  if (!structureRef || !structure) {
    return "No selectable molecular structure.";
  }

  const expression = clusterExpression(selectors);
  const selection = Script.getStructureSelection(expression, structure);
  if (StructureSelection.isEmpty(selection)) {
    return "3D selection unavailable: auth chain/residue/insertion-code identifiers do not match.";
  }

  const loci = StructureSelection.toLociWithSourceUnits(selection);
  const component = await plugin.builders.structure.tryCreateComponentFromExpression(
    structureRef.cell,
    expression,
    "protcross-selected-predicted-cluster",
    { label: "Selected cluster" }
  );
  if (component) {
    await plugin.builders.structure.representation.addRepresentation(component, {
      type: "ball-and-stick",
      color: "uniform",
      colorParams: { value: 0xe14f3d },
      size: "uniform",
      sizeParams: { value: 0.35 }
    });
  }

  plugin.managers.structure.selection.fromLoci("set", loci, false);
  plugin.managers.interactivity.lociHighlights.highlightOnly({ loci }, false);
  plugin.managers.camera.focusLoci(loci, { extraRadius: 8, minRadius: 8, durationMs: 250 });
  return `${selectors.length} residue${selectors.length === 1 ? "" : "s"} in selected cluster`;
}

async function clearClusterSelection(viewer: any): Promise<void> {
  const plugin = viewer?.plugin;
  if (!plugin) {
    return;
  }
  try {
    plugin.managers.interactivity.lociHighlights.clearHighlights();
    plugin.managers.structure.selection.clear();
    const components = (plugin.managers.structure.hierarchy.current.structures ?? [])
      .flatMap((structure: any) => structure.components ?? [])
      .filter((component: any) => isSelectedClusterComponent(component));
    if (components.length > 0) {
      await plugin.managers.structure.hierarchy.remove(components, false);
    }
  } catch {
    // Selection cleanup is best-effort while Mol* is loading or being disposed.
  }
}

function uniqueAuthResidueSelectors(residues: ResidueSummary[]): AuthResidueSelector[] {
  const selectors = new Map<string, AuthResidueSelector>();
  for (const residue of residues) {
    const selector = authResidueSelector(residue);
    if (!selector) {
      continue;
    }
    const key = `${selector.authAsymId}\u0000${selector.authSeqId}\u0000${selector.hasInsertionCode ? selector.insertionCode : "*"}`;
    selectors.set(key, selector);
  }
  return [...selectors.values()];
}

function authResidueSelector(residue: ResidueSummary): AuthResidueSelector | null {
  const chainValue = residue.auth_asym_id ?? residue.chain_id;
  if (chainValue === undefined || chainValue === null) {
    return null;
  }

  let sequenceValue = residue.auth_seq_id ?? residue.residue_number;
  let inferredInsertionCode: string | undefined;
  if (typeof sequenceValue === "string") {
    const compact = sequenceValue.trim();
    const match = /^(-?\d+)([A-Za-z]?)$/.exec(compact);
    if (match) {
      sequenceValue = Number(match[1]);
      inferredInsertionCode = match[2] || undefined;
    }
  }
  const authSeqId = Number(sequenceValue);
  if (!Number.isInteger(authSeqId)) {
    return null;
  }

  const hasInsertionCode = residue.insertion_code !== undefined || inferredInsertionCode !== undefined;
  const rawInsertionCode = inferredInsertionCode ?? residue.insertion_code ?? "";
  const insertionCode = [".", "?"].includes(String(rawInsertionCode).trim())
    ? ""
    : String(rawInsertionCode).trim();
  return {
    authAsymId: String(chainValue).trim(),
    authSeqId,
    insertionCode,
    hasInsertionCode
  };
}

function clusterExpression(selectors: AuthResidueSelector[]): Expression {
  const expressions = selectors.map((selector) => {
    const residueTests: Expression[] = [
      MS.core.rel.eq([MS.struct.atomProperty.macromolecular.auth_seq_id(), selector.authSeqId])
    ];
    if (selector.hasInsertionCode) {
      residueTests.push(
        MS.core.rel.eq([
          MS.struct.atomProperty.macromolecular.pdbx_PDB_ins_code(),
          selector.insertionCode
        ])
      );
    }
    return MS.struct.generator.atomGroups({
      "chain-test": MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.auth_asym_id(),
        selector.authAsymId
      ]),
      "residue-test": residueTests.length === 1 ? residueTests[0] : MS.core.logic.and(residueTests),
      "group-by": MS.struct.atomProperty.macromolecular.residueKey()
    });
  });
  if (expressions.length === 1) {
    return expressions[0];
  }
  return MS.struct.combinator.merge(
    expressions.map((expression) => MS.struct.modifier.union([expression]))
  );
}
