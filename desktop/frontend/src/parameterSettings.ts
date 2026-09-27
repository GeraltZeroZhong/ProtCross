/** Editable values retain the user's draft, including an empty field. */
export interface PredictionParameterValues {
  threshold: number | string;
  clusterCutoff: number | string;
  allowTruncation: boolean;
  device: string;
  batchSize: number | string;
}

export type PredictionParameters = PredictionParameterValues;

export const DEFAULT_PREDICTION_PARAMETERS: Readonly<PredictionParameterValues> = {
  threshold: 0.5, clusterCutoff: 8, allowTruncation: false, device: "", batchSize: 4
};
export type ParameterField = "threshold" | "clusterCutoff" | "device" | "batchSize";
export type ParameterErrors = Partial<Record<ParameterField, string>>;

function enteredNumber(value: number | string): number | null {
  if (typeof value === "string" && value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function validatePredictionParameters(values: PredictionParameterValues, includeBatchSize = false): ParameterErrors {
  const errors: ParameterErrors = {};
  const threshold = enteredNumber(values.threshold);
  const distance = enteredNumber(values.clusterCutoff);
  if (threshold === null || threshold < 0 || threshold > 1) errors.threshold = "Score cutoff must be 0–1.";
  if (distance === null || distance <= 0) errors.clusterCutoff = "Cluster distance must be greater than 0 Å.";
  const device = values.device.trim().toLowerCase();
  if (device && !/^(?:auto|cpu|mps|cuda(?::\d+)?)$/.test(device)) {
    errors.device = "Use auto, cpu, mps, cuda, or cuda:N (N ≥ 0).";
  }
  if (includeBatchSize) {
    const size = enteredNumber(values.batchSize);
    if (size === null || !Number.isInteger(size) || size < 1 || size > 4) errors.batchSize = "Microbatch size must be an integer from 1 to 4.";
  }
  return errors;
}

export function parameterValidationMessage(values: PredictionParameterValues, includeBatchSize = false): string | null {
  return Object.values(validatePredictionParameters(values, includeBatchSize))[0] ?? null;
}

export function parametersAreDefault(values: PredictionParameterValues, includeBatchSize = false): boolean {
  return !parameterValidationMessage(values, includeBatchSize)
    && Number(values.threshold) === DEFAULT_PREDICTION_PARAMETERS.threshold
    && Number(values.clusterCutoff) === DEFAULT_PREDICTION_PARAMETERS.clusterCutoff
    && values.allowTruncation === DEFAULT_PREDICTION_PARAMETERS.allowTruncation
    && values.device.trim() === DEFAULT_PREDICTION_PARAMETERS.device
    && (!includeBatchSize || Number(values.batchSize) === DEFAULT_PREDICTION_PARAMETERS.batchSize);
}

function displayNumber(value: number | string, decimals: number): string {
  const parsed = enteredNumber(value);
  if (parsed === null) return "—";
  const rounded = parsed.toFixed(decimals);
  return Number(rounded) === parsed ? rounded : String(parsed);
}

export function predictionParameterSummary(values: PredictionParameterValues, includeBatchSize = false): string {
  if (parametersAreDefault(values, includeBatchSize)) return "Default";
  const errors = validatePredictionParameters(values, includeBatchSize);
  const parts = ["Modified"];
  if (Object.keys(errors).length) parts.push("Check values");
  if (errors.threshold || Number(values.threshold) !== DEFAULT_PREDICTION_PARAMETERS.threshold) parts.push(`Cutoff ${displayNumber(values.threshold, 2)}`);
  if (errors.clusterCutoff || Number(values.clusterCutoff) !== DEFAULT_PREDICTION_PARAMETERS.clusterCutoff) parts.push(`Distance ${displayNumber(values.clusterCutoff, 1)} Å`);
  if (values.allowTruncation) parts.push("Truncation on");
  if (values.device.trim()) parts.push(`Device ${values.device.trim().toLowerCase()}`);
  if (includeBatchSize && (errors.batchSize || Number(values.batchSize) !== DEFAULT_PREDICTION_PARAMETERS.batchSize)) parts.push(`Microbatch ${displayNumber(values.batchSize, 0)}`);
  return parts.join(" · ");
}
