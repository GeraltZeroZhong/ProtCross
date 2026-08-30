import type { PocketDetail, PocketJson, ResidueSummary } from "./types";

export interface LocalResultView {
  available: boolean;
  unavailableReason?: "missing-data" | "invalid-parameters";
  pockets: PocketJson | null;
  records: ResidueSummary[];
  selectedResidueCount: number;
}

interface IndexedResidue {
  index: number;
  record: ResidueSummary;
  score: number;
  coord: [number, number, number];
}

export function recomputeLocalResult(
  scores: ResidueSummary[],
  threshold: number,
  cutoff: number,
  sourcePockets: PocketJson | null
): LocalResultView {
  const scoredRecords = scores.filter((record) => Number(record.is_scored ?? 1) !== 0);
  if (!scoredRecords.length) {
    return {
      available: false,
      unavailableReason: "missing-data",
      pockets: sourcePockets,
      records: [],
      selectedResidueCount: Number(sourcePockets?.selected_residue_count ?? 0)
    };
  }
  if (
    !Number.isFinite(threshold)
    || threshold < 0
    || threshold > 1
    || !Number.isFinite(cutoff)
    || cutoff <= 0
  ) {
    return {
      available: false,
      unavailableReason: "invalid-parameters",
      pockets: sourcePockets,
      records: rankResidues(scoredRecords),
      selectedResidueCount: Number(sourcePockets?.selected_residue_count ?? 0)
    };
  }

  const indexed: IndexedResidue[] = [];
  for (let index = 0; index < scoredRecords.length; index += 1) {
    const record = scoredRecords[index];
    const score = residueScore(record);
    const coord = residueCoordinate(record);
    if (!Number.isFinite(score) || coord === null) {
      return {
        available: false,
        unavailableReason: "missing-data",
        pockets: sourcePockets,
        records: rankResidues(scoredRecords),
        selectedResidueCount: Number(sourcePockets?.selected_residue_count ?? 0)
      };
    }
    if (score > threshold) {
      indexed.push({ index, record, score, coord });
    }
  }

  const components = connectedComponents(indexed, cutoff);
  components.sort((left, right) => {
    if (left.length !== right.length) {
      return right.length - left.length;
    }
    const meanDelta = componentMean(right) - componentMean(left);
    if (meanDelta !== 0) {
      return meanDelta;
    }
    const maxDelta = componentMax(right) - componentMax(left);
    if (maxDelta !== 0) {
      return maxDelta;
    }
    return componentCanonicalIndex(left) - componentCanonicalIndex(right);
  });

  const clusterByIndex = new Map<number, number>();
  const clusteredPockets = components.map((component, clusterOffset) => {
    const clusterId = clusterOffset + 1;
    for (const residue of component) {
      clusterByIndex.set(residue.index, clusterId);
    }
    return pocketFromResidues(component, clusterId);
  });
  const records = scoredRecords.map((record, index) => ({
    ...record,
    cluster_id: clusterByIndex.get(index) ?? null,
    is_binding: residueScore(record) > threshold ? 1 : 0
  }));
  const selected = indexed.map((entry) => ({ ...entry, record: records[entry.index] }));
  const pockets: PocketJson = {
    ...(sourcePockets ?? { schema_version: "protcross-pocket-v2" }),
    threshold,
    cluster_cutoff: cutoff,
    selected_residue_count: selected.length,
    aggregate_pocket: selected.length ? pocketFromResidues(selected) : null,
    clustered_pockets: clusteredPockets
  };

  return {
    available: true,
    pockets,
    records: rankResidues(records),
    selectedResidueCount: selected.length
  };
}

export function rankResidues(scores: ResidueSummary[]): ResidueSummary[] {
  return scores
    .map((record, index) => ({ record, index }))
    .sort((left, right) => {
      const leftRank = Number(left.record.rank_global);
      const rightRank = Number(right.record.rank_global);
      if (Number.isFinite(leftRank) && Number.isFinite(rightRank) && leftRank !== rightRank) {
        return leftRank - rightRank;
      }
      const scoreDelta = residueScore(right.record) - residueScore(left.record);
      return scoreDelta || left.index - right.index;
    })
    .map(({ record }) => record);
}

function connectedComponents(residues: IndexedResidue[], cutoff: number): IndexedResidue[][] {
  if (!residues.length) {
    return [];
  }
  const parent = residues.map((_, index) => index);
  const size = residues.map(() => 1);
  const cells = new Map<string, number[]>();
  const cutoffSquared = cutoff * cutoff;

  function find(index: number): number {
    let root = index;
    while (parent[root] !== root) {
      root = parent[root];
    }
    while (parent[index] !== index) {
      const next = parent[index];
      parent[index] = root;
      index = next;
    }
    return root;
  }

  function union(left: number, right: number) {
    let leftRoot = find(left);
    let rightRoot = find(right);
    if (leftRoot === rightRoot) {
      return;
    }
    if (size[leftRoot] < size[rightRoot]) {
      [leftRoot, rightRoot] = [rightRoot, leftRoot];
    }
    parent[rightRoot] = leftRoot;
    size[leftRoot] += size[rightRoot];
  }

  for (let localIndex = 0; localIndex < residues.length; localIndex += 1) {
    const [x, y, z] = residues[localIndex].coord;
    const cellX = Math.floor(x / cutoff);
    const cellY = Math.floor(y / cutoff);
    const cellZ = Math.floor(z / cutoff);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const neighbors = cells.get(cellKey(cellX + dx, cellY + dy, cellZ + dz));
          if (!neighbors) {
            continue;
          }
          for (const neighborIndex of neighbors) {
            const [nx, ny, nz] = residues[neighborIndex].coord;
            const deltaX = x - nx;
            const deltaY = y - ny;
            const deltaZ = z - nz;
            if (deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ <= cutoffSquared) {
              union(localIndex, neighborIndex);
            }
          }
        }
      }
    }
    const key = cellKey(cellX, cellY, cellZ);
    const cell = cells.get(key);
    if (cell) {
      cell.push(localIndex);
    } else {
      cells.set(key, [localIndex]);
    }
  }

  const grouped = new Map<number, IndexedResidue[]>();
  for (let index = 0; index < residues.length; index += 1) {
    const root = find(index);
    const component = grouped.get(root);
    if (component) {
      component.push(residues[index]);
    } else {
      grouped.set(root, [residues[index]]);
    }
  }
  return [...grouped.values()];
}

function pocketFromResidues(residues: IndexedResidue[], clusterId?: number): PocketDetail {
  let scoreSum = 0;
  let weightedX = 0;
  let weightedY = 0;
  let weightedZ = 0;
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  let scoreMax = Number.NEGATIVE_INFINITY;
  for (const residue of residues) {
    const [x, y, z] = residue.coord;
    scoreSum += residue.score;
    weightedX += x * residue.score;
    weightedY += y * residue.score;
    weightedZ += z * residue.score;
    sumX += x;
    sumY += y;
    sumZ += z;
    scoreMax = Math.max(scoreMax, residue.score);
  }
  const count = residues.length;
  const center: [number, number, number] = scoreSum > 0
    ? [weightedX / scoreSum, weightedY / scoreSum, weightedZ / scoreSum]
    : [sumX / count, sumY / count, sumZ / count];
  const pocket: PocketDetail = {
    center,
    center_unweighted: [sumX / count, sumY / count, sumZ / count],
    residue_count: count,
    score_mean: scoreSum / count,
    score_max: scoreMax,
    residues: residues.map((residue) => ({
      ...residue.record,
      cluster_id: clusterId === undefined ? residue.record.cluster_id ?? null : clusterId
    }))
  };
  if (clusterId !== undefined) {
    pocket.cluster_id = clusterId;
  }
  return pocket;
}

function residueScore(record: ResidueSummary): number {
  return Number(record.score ?? record.probability);
}

function residueCoordinate(record: ResidueSummary): [number, number, number] | null {
  const values = [record.x, record.y, record.z];
  if (!values.every((value) => typeof value === "number" && Number.isFinite(value))) {
    return null;
  }
  return [values[0] as number, values[1] as number, values[2] as number];
}

function componentMean(component: IndexedResidue[]): number {
  return component.reduce((total, residue) => total + residue.score, 0) / component.length;
}

function componentMax(component: IndexedResidue[]): number {
  return component.reduce((highest, residue) => Math.max(highest, residue.score), Number.NEGATIVE_INFINITY);
}

function componentCanonicalIndex(component: IndexedResidue[]): number {
  return component.reduce((lowest, residue) => Math.min(lowest, residue.index), Number.POSITIVE_INFINITY);
}

function cellKey(x: number, y: number, z: number): string {
  return `${x}:${y}:${z}`;
}
