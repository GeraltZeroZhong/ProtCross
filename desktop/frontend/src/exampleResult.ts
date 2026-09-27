import response from "./example/response.json";
import provenance from "./example/provenance.json";
import structureData from "./example/1CRN.protcross.pdb?raw";
import type { PredictResponse } from "./types";

/** Precomputed with the release assets; available before installing a runtime. */
const recorded = response as unknown as PredictResponse;
export const exampleResult: PredictResponse = {
  ...recorded,
  summary: { ...recorded.summary, example_provenance: provenance }
};
export { structureData as exampleStructureData };
