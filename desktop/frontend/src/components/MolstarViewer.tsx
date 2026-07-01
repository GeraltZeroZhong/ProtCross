import { useEffect, useRef, useState } from "react";
import "molstar/build/viewer/molstar.css";
import { desktopFileUrl } from "../api";
import type { PocketJson, SummaryJson } from "../types";

interface Props {
  structurePath?: string;
  summary?: SummaryJson | null;
  pockets?: PocketJson | null;
}

export function MolstarViewer({ structurePath, summary, pockets }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewerReady, setViewerReady] = useState(false);

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
      disposeViewer(viewerRef.current);
      viewerRef.current = null;
      setViewerReady(false);
    };
  }, []);

  useEffect(() => {
    async function load() {
      if (!viewerReady || !viewerRef.current || !structurePath) {
        return;
      }
      try {
        setError(null);
        await viewerRef.current.plugin.clear();
        const url = desktopFileUrl(structurePath);
        const format = structurePath.toLowerCase().endsWith(".cif") || structurePath.toLowerCase().endsWith(".mmcif")
          ? "mmcif"
          : "pdb";
        await viewerRef.current.loadStructureFromUrl(url, format, false, {
          representationParams: {
            theme: {
              globalName: "uncertainty",
              carbonColor: { name: "uncertainty", params: { domain: [0, 1] } }
            }
          }
        });
        await applyProbabilityTheme(viewerRef.current);
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    }
    void load();
  }, [structurePath, viewerReady]);

  const topPocket = summary?.top_pocket ?? pockets?.clustered_pockets?.[0] ?? null;
  const highlightedResidues = (pockets?.clustered_pockets?.[0]?.residues ?? []).slice(0, 8);

  return (
    <section className="viewer-panel">
      <div className="viewer-toolbar">
        <div>
          <strong>Structure Viewer</strong>
          <span>{structurePath || "No annotated structure loaded"}</span>
        </div>
        {topPocket ? (
          <div className="center-readout">
            Pocket center: {topPocket.center.map((value) => value.toFixed(3)).join(", ")}
          </div>
        ) : null}
      </div>
      <div className="molstar-frame">
        <div className="molstar-host" ref={hostRef} />
        {topPocket ? (
          <div className="viewer-overlay">
            <div className="pocket-marker">
              <span />
              Center {topPocket.center.map((value) => value.toFixed(2)).join(", ")}
            </div>
            <div className="viewer-residue-chips">
              {highlightedResidues.map((residue) => (
                <span key={`${residue.residue_id}-${residue.cluster_id ?? ""}`}>
                  {residue.residue_id} {Number(residue.probability).toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        ) : null}
      </div>
      {error ? <div className="inline-error">{error}</div> : null}
      <div className="viewer-note">
        Annotated structures use B-factors as continuous probabilities. Pocket centers and residues are read from
        ProtCross JSON outputs and shown in the result panels.
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

async function applyProbabilityTheme(viewer: any): Promise<void> {
  const plugin = viewer.plugin;
  const structures = plugin.managers.structure.hierarchy.current.structures ?? [];
  const components = structures.flatMap((structure: any) => structure.components ?? []);
  if (!components.length) {
    return;
  }
  await plugin.managers.structure.component.updateRepresentationsTheme(components, {
    color: "uncertainty",
    colorParams: { domain: [0, 1] }
  });
}
