import { useEffect, useRef, useState } from "react";
import "molstar/build/viewer/molstar.css";
import { StructureSelection } from "molstar/lib/mol-model/structure";
import type { Expression } from "molstar/lib/mol-script/language/expression";
import { MolScriptBuilder as MS } from "molstar/lib/mol-script/language/builder";
import { Script } from "molstar/lib/mol-script/script";
import { desktopFileUrl } from "../api";
import type { PocketJson, ResidueSummary, SummaryJson } from "../types";

interface Props {
  structurePath?: string;
  summary?: SummaryJson | null;
  pockets?: PocketJson | null;
  selectedClusterIndex?: number;
}

export function MolstarViewer({ structurePath, summary, pockets, selectedClusterIndex = 0 }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<any>(null);
  const operationQueueRef = useRef<Promise<void>>(Promise.resolve());
  const structureRequestRef = useRef(0);
  const selectionRequestRef = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const [selectionMessage, setSelectionMessage] = useState<string | null>(null);
  const [viewerReady, setViewerReady] = useState(false);
  const [loadedStructureRequest, setLoadedStructureRequest] = useState(0);
  const selectedCluster = pockets?.clustered_pockets?.[selectedClusterIndex] ?? null;
  const displayedCluster = selectedCluster ?? summary?.top_pocket ?? null;

  useEffect(() => {
    let cancelled = false;
    async function init() {
      if (!hostRef.current || viewerRef.current) {
        return;
      }
      try {
        const molstar = await import("molstar/build/viewer/molstar");
        const viewer = await molstar.Viewer.create(hostRef.current, {
          layoutIsExpanded: false,
          layoutShowControls: true,
          layoutShowSequence: true,
          layoutShowLog: false,
          layoutShowLeftPanel: false,
          viewportShowExpand: true,
          viewportShowSelectionMode: true,
          backgroundColor: 0xffffff
        });
        if (cancelled) {
          disposeViewer(viewer);
          return;
        }
        viewerRef.current = viewer;
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
    };
  }, []);

  useEffect(() => {
    if (!viewerReady || !viewerRef.current || !structurePath) {
      setLoadedStructureRequest(0);
      return;
    }
    const request = structureRequestRef.current + 1;
    structureRequestRef.current = request;
    selectionRequestRef.current += 1;
    setLoadedStructureRequest(0);
    setError(null);
    setSelectionMessage(null);

    const operation = operationQueueRef.current.catch(() => undefined).then(async () => {
      if (request !== structureRequestRef.current || !viewerRef.current) {
        return;
      }
      try {
        const viewer = viewerRef.current;
        await viewer.plugin.clear();
        if (request !== structureRequestRef.current || !viewerRef.current) {
          return;
        }
        const url = desktopFileUrl(structurePath);
        const format = structurePath.toLowerCase().endsWith(".cif") || structurePath.toLowerCase().endsWith(".mmcif")
          ? "mmcif"
          : "pdb";
        await viewer.loadStructureFromUrl(url, format, false);
        if (request === structureRequestRef.current && viewerRef.current) {
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
      if (structureRequestRef.current === request) {
        structureRequestRef.current += 1;
      }
    };
  }, [structurePath, viewerReady]);

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
          setSelectionMessage(message);
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
  }, [loadedStructureRequest, selectedCluster, viewerReady]);

  return (
    <section className="viewer-panel">
      <div className="viewer-toolbar">
        <div>
          <strong>Structure Viewer</strong>
          <span>{structurePath || "No annotated structure loaded"}</span>
        </div>
        {displayedCluster ? (
          <div className="center-readout">
            <strong>Score-weighted Cα centroid</strong>
            <span>{displayedCluster.center.map((value) => value.toFixed(3)).join(", ")} Å</span>
            <small>Coordinates only; no 3D centroid marker is drawn.</small>
          </div>
        ) : null}
      </div>
      <div className="molstar-frame">
        <div className="molstar-host" ref={hostRef} />
      </div>
      {error ? <div className="inline-error">{error}</div> : null}
      <div className="viewer-note">
        {selectionMessage ? <span>{selectionMessage} </span> : null}
        Annotated structures store ProtCross model scores (0–1) in B-factor fields. The viewer uses its standard
        structure colors; model scores are not labeled as uncertainty. The selected ball-and-stick residues are the
        selected predicted-residue cluster. Its score-weighted Cα centroid is reported numerically above and is not shown
        as a 3D point.
      </div>
    </section>
  );
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
    return "No predicted-residue cluster exists at the current model-score threshold.";
  }

  const selectors = uniqueAuthResidueSelectors(residues);
  if (selectors.length === 0) {
    return "The predicted-residue cluster lacks auth chain/residue identifiers, so it could not be selected in 3D.";
  }

  const structureRef = plugin.managers.structure.hierarchy.current.structures?.[0];
  const structure = structureRef?.cell?.obj?.data;
  if (!structureRef || !structure) {
    return "The annotated structure loaded without a selectable molecular structure.";
  }

  const expression = clusterExpression(selectors);
  const selection = Script.getStructureSelection(expression, structure);
  if (StructureSelection.isEmpty(selection)) {
    return "No annotated-structure residues matched the cluster's auth chain/residue/insertion-code identifiers.";
  }

  const loci = StructureSelection.toLociWithSourceUnits(selection);
  const component = await plugin.builders.structure.tryCreateComponentFromExpression(
    structureRef.cell,
    expression,
    "protcross-selected-predicted-cluster",
    { label: "Selected predicted-residue cluster" }
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
  return `Selected and highlighted the predicted-residue cluster (${selectors.length} reported residues) by auth chain/residue/insertion code.`;
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
      .filter((component: any) => component.key === "protcross-selected-predicted-cluster");
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
