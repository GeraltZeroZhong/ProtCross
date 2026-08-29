import { Bond, Model, StructureElement, Unit } from "molstar/lib/mol-model/structure";
import type { Structure } from "molstar/lib/mol-model/structure";
import type { ThemeDataContext } from "molstar/lib/mol-theme/theme";
import type { ColorTheme } from "molstar/lib/mol-theme/color";
import { Color, ColorScale } from "molstar/lib/mol-util/color";

export const PROTCROSS_UNSCORED_COLOR = Color(0x9aa0a6);
export const PROTCROSS_UNSCORED_COLOR_CSS = "#9aa0a6";

export interface ProtcrossScoreCoverage {
  source: "result-keys" | "result-keys-partial" | "unavailable";
  scoredResidueCount: number;
  expectedScoredResidueCount?: number;
  unscoredResidueCount: number;
  unmatchedScoredResidueCount: number;
}

interface AtomicResidueDescriptor {
  model: Model;
  residueIndex: number;
  identity: string;
}

const scoredResiduesByModel = new WeakMap<Model, ReadonlySet<number>>();

/** Register the exact scored-residue domain before applying the color theme. */
export function configureProtcrossScoreTheme(
  structures: readonly Structure[],
  scoredResidueKeys: readonly string[] | undefined
): ProtcrossScoreCoverage {
  const residues = atomicResidueDescriptors(structures);
  const scoredByModel = new Map<Model, Set<number>>();
  for (const residue of residues) {
    if (!scoredByModel.has(residue.model)) {
      scoredByModel.set(residue.model, new Set());
    }
  }

  if (scoredResidueKeys === undefined) {
    commitScoredResidues(scoredByModel);
    return {
      source: "unavailable",
      scoredResidueCount: 0,
      unscoredResidueCount: residues.length,
      unmatchedScoredResidueCount: 0
    };
  }

  const uniqueKeys = [...new Set(scoredResidueKeys.map((key) => key.trim()).filter(Boolean))];
  const expectedIdentities = new Set(
    uniqueKeys.map(normalizeProtcrossResidueKey).filter((identity): identity is string => identity !== null)
  );
  const matchedIdentities = new Set<string>();
  for (const residue of residues) {
    if (expectedIdentities.has(residue.identity)) {
      scoredByModel.get(residue.model)?.add(residue.residueIndex);
      matchedIdentities.add(residue.identity);
    }
  }
  commitScoredResidues(scoredByModel);
  const scoredResidueCount = scoredCount(scoredByModel);
  const unmatchedScoredResidueCount = uniqueKeys.length - matchedIdentities.size;
  return {
    source: unmatchedScoredResidueCount === 0 ? "result-keys" : "result-keys-partial",
    scoredResidueCount,
    expectedScoredResidueCount: uniqueKeys.length,
    unscoredResidueCount: Math.max(0, residues.length - scoredResidueCount),
    unmatchedScoredResidueCount
  };
}

/** Normalize a backend canonical residue_key into the viewer identity contract. */
export function normalizeProtcrossResidueKey(key: string): string | null {
  const fields = new Map<string, string>();
  for (const part of key.split("|")) {
    const separator = part.indexOf(":");
    if (separator >= 0) {
      fields.set(part.slice(0, separator), part.slice(separator + 1));
    }
  }
  const modelIndex = integerString(fields.get("model"));
  const authSeqId = integerString(fields.get("resseq"));
  const chainId = fields.get("chain");
  const residueName = fields.get("resname");
  if (modelIndex === null || authSeqId === null || chainId === undefined || residueName === undefined) {
    return null;
  }
  return residueIdentity(
    modelIndex,
    chainId,
    fields.get("het") || "ATOM",
    authSeqId,
    fields.get("icode") || "",
    residueName
  );
}

function scoreFor(unit: Unit, element: number): number | undefined {
  if (!Unit.isAtomic(unit)) {
    return undefined;
  }
  const residueIndex = unit.model.atomicHierarchy.residueAtomSegments.index[element];
  if (!scoredResiduesByModel.get(unit.model)?.has(residueIndex)) {
    return undefined;
  }
  const score = unit.model.atomicConformation.B_iso_or_equiv.value(element);
  return Number.isFinite(score) ? Math.min(1, Math.max(0, score)) : undefined;
}

export function ProtcrossScoreColorTheme(_ctx: ThemeDataContext, props: Record<string, never>): ColorTheme<{}> {
  const scale = ColorScale.create({
    domain: [0, 1],
    reverse: false,
    minLabel: "0.00 (scored)",
    maxLabel: "1.00",
    listOrName: [
      [Color(0x173b57), 0],
      [Color(0x2d708e), 0.3],
      [Color(0x5cb6a5), 0.55],
      [Color(0xf2c14e), 0.78],
      [Color(0xe4572e), 1]
    ]
  });

  function color(location: unknown) {
    if (StructureElement.Location.is(location)) {
      const score = scoreFor(location.unit, location.element);
      return score === undefined ? PROTCROSS_UNSCORED_COLOR : scale.color(score);
    }
    if (Bond.isLocation(location)) {
      const score = scoreFor(location.aUnit, location.aUnit.elements[location.aIndex]);
      return score === undefined ? PROTCROSS_UNSCORED_COLOR : scale.color(score);
    }
    return PROTCROSS_UNSCORED_COLOR;
  }

  return {
    factory: ProtcrossScoreColorTheme,
    granularity: "group",
    preferSmoothing: true,
    color,
    props,
    description: "ProtCross model score stored in annotated-structure B-factors. Unscored residues use neutral gray.",
    legend: scale.legend
  };
}

export const ProtcrossScoreColorThemeProvider: ColorTheme.Provider<{}, "protcross-score"> = {
  name: "protcross-score",
  label: "ProtCross score",
  category: "Miscellaneous",
  factory: ProtcrossScoreColorTheme,
  getParams: () => ({}),
  defaultValues: {},
  isApplicable: (ctx) => Boolean(ctx.structure?.models.some((model) => model.atomicConformation.B_iso_or_equiv.isDefined))
};

function atomicResidueDescriptors(structures: readonly Structure[]): AtomicResidueDescriptor[] {
  const descriptors: AtomicResidueDescriptor[] = [];
  const visitedModels = new Set<Model>();
  for (const structure of structures) {
    for (const model of structure.models) {
      if (visitedModels.has(model)) {
        continue;
      }
      visitedModels.add(model);
      const hierarchy = model.atomicHierarchy;
      const offsets = hierarchy.residueAtomSegments.offsets;
      const modelIndex = String(Model.TrajectoryInfo.get(model).index);
      for (let residueIndex = 0; residueIndex < hierarchy.residueAtomSegments.count; residueIndex += 1) {
        const atomIndex = offsets[residueIndex];
        const chainIndex = hierarchy.chainAtomSegments.index[atomIndex];
        const residueName = (
          hierarchy.atoms.auth_comp_id.value(atomIndex) || hierarchy.atoms.label_comp_id.value(atomIndex)
        ).trim().toUpperCase();
        descriptors.push({
          model,
          residueIndex,
          identity: residueIdentity(
            modelIndex,
            hierarchy.chains.auth_asym_id.value(chainIndex),
            hierarchy.residues.group_PDB.value(residueIndex),
            String(hierarchy.residues.auth_seq_id.value(residueIndex)),
            hierarchy.residues.pdbx_PDB_ins_code.value(residueIndex),
            residueName
          )
        });
      }
    }
  }
  return descriptors;
}

function residueIdentity(
  modelIndex: string,
  chainId: string,
  groupPdb: string,
  authSeqId: string,
  insertionCode: string,
  residueName: string
): string {
  return [
    integerString(modelIndex) ?? modelIndex.trim(),
    chainId.trim(),
    (groupPdb.trim() || "ATOM").toUpperCase(),
    integerString(authSeqId) ?? authSeqId.trim(),
    normalizeInsertionCode(insertionCode),
    residueName.trim().toUpperCase()
  ].join("\u0000");
}

function normalizeInsertionCode(value: string): string {
  const normalized = value.trim();
  return normalized === "." || normalized === "?" ? "" : normalized;
}

function integerString(value: unknown): string | null {
  if (value === undefined || value === null || String(value).trim() === "") {
    return null;
  }
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : null;
}

function commitScoredResidues(scoredByModel: ReadonlyMap<Model, ReadonlySet<number>>): void {
  for (const [model, residueIndices] of scoredByModel) {
    scoredResiduesByModel.set(model, residueIndices);
  }
}

function scoredCount(scoredByModel: ReadonlyMap<Model, ReadonlySet<number>>): number {
  let count = 0;
  for (const residues of scoredByModel.values()) {
    count += residues.size;
  }
  return count;
}
