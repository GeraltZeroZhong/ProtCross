import { useEffect, useId } from "react";
import { Icon } from "./Icon";
import {
  DEFAULT_PREDICTION_PARAMETERS,
  parameterValidationMessage,
  predictionParameterSummary,
  validatePredictionParameters,
  type PredictionParameterValues
} from "../parameterSettings";

export interface AdvancedParametersProps {
  values: PredictionParameterValues;
  onChange: (values: PredictionParameterValues) => void;
  disabled?: boolean;
  showMicroBatchSize?: boolean;
  title?: string;
  /** Optional; callers can derive validity directly with parameterValidationMessage. */
  onValidationChange?: (error: string | null) => void;
  /** Clear an earlier request error when restoring defaults. */
  onReset?: () => void;
}

/** Shared controlled editor: drafts survive tab changes when kept by the parent. */
export function AdvancedParameters({ values, onChange, disabled = false, showMicroBatchSize = false,
  title = "Parameters", onValidationChange, onReset }: AdvancedParametersProps) {
  const id = useId();
  const errors = validatePredictionParameters(values, showMicroBatchSize);
  const validationMessage = parameterValidationMessage(values, showMicroBatchSize);
  const selectedDevice = values.device.trim().toLowerCase();
  const deviceChoice = ["", "auto", "cpu", "mps", "cuda"].includes(selectedDevice) ? selectedDevice : "custom";
  useEffect(() => { onValidationChange?.(validationMessage); }, [onValidationChange, validationMessage]);
  function update<K extends keyof PredictionParameterValues>(key: K, value: PredictionParameterValues[K]) {
    onChange({ ...values, [key]: value });
  }
  return (
    <details className="disclosure settings-disclosure">
      <summary><Icon name="settings" /> {title}<span>{predictionParameterSummary(values, showMicroBatchSize)}</span></summary>
      <div className="disclosure-content parameter-editor">
        <div className="parameter-toolbar">
          {disabled ? <small>Locked during run</small> : null}
          <button className="text-button" disabled={disabled} onClick={() => {
            onChange({ ...DEFAULT_PREDICTION_PARAMETERS });
            onReset?.();
          }}>Restore defaults</button>
        </div>
        <div className="parameter-sections">
          <section className="parameter-section parameter-selection" aria-labelledby={`${id}-selection-heading`}>
            <h4 className="parameter-section-title" id={`${id}-selection-heading`}>Residue selection</h4>
            <div className="inline-fields parameter-grid">
              <div className="field">
                <label htmlFor={`${id}-threshold`}>Model-score cutoff</label>
                <input id={`${id}-threshold`} type="number" inputMode="decimal" min={0} max={1} step="any"
                  value={values.threshold} disabled={disabled} aria-invalid={Boolean(errors.threshold)}
                  aria-describedby={`${id}-threshold-help${errors.threshold ? ` ${id}-threshold-error` : ""}`}
                  onChange={(event) => update("threshold", event.target.value)} />
                <small className="field-help" id={`${id}-threshold-help`}>0–1 · selects score &gt; cutoff</small>
                {errors.threshold ? <span className="inline-error" role="alert" id={`${id}-threshold-error`}>{errors.threshold}</span> : null}
              </div>
              <div className="field">
                <label htmlFor={`${id}-distance`}>Cluster distance (Å)</label>
                <input id={`${id}-distance`} type="number" inputMode="decimal" min={0} step="any"
                  value={values.clusterCutoff} disabled={disabled} aria-invalid={Boolean(errors.clusterCutoff)}
                  aria-describedby={`${id}-distance-help${errors.clusterCutoff ? ` ${id}-distance-error` : ""}`}
                  onChange={(event) => update("clusterCutoff", event.target.value)} />
                <small className="field-help" id={`${id}-distance-help`}>Greater than 0 · Cα separation</small>
                {errors.clusterCutoff ? <span className="inline-error" role="alert" id={`${id}-distance-error`}>{errors.clusterCutoff}</span> : null}
              </div>
            </div>
            <label className="checkbox-line truncation-control">
              <input type="checkbox" checked={values.allowTruncation} disabled={disabled}
                aria-labelledby={`${id}-truncation-label`} aria-describedby={`${id}-truncation-help`}
                onChange={(event) => update("allowTruncation", event.target.checked)} />
              <span><strong id={`${id}-truncation-label`}>Allow long-chain truncation</strong><small id={`${id}-truncation-help`}>Keep the first 1,022 residues per chain; remainder unscored.</small></span>
            </label>
          </section>
          <section className="parameter-section parameter-execution" aria-labelledby={`${id}-execution-heading`}>
            <h4 className="parameter-section-title" id={`${id}-execution-heading`}>Execution</h4>
            <div className="inline-fields parameter-grid">
              <div className="field">
                <label htmlFor={`${id}-device`}>Inference device</label>
                <select id={`${id}-device`} value={deviceChoice} disabled={disabled} aria-describedby={`${id}-device-help`}
                  onChange={(event) => update("device", event.target.value === "custom" ? "cuda:0" : event.target.value)}>
                  <option value="">Active runtime</option>
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">NVIDIA CUDA</option>
                  <option value="mps">Apple MPS</option>
                  <option value="custom">CUDA device…</option>
                </select>
                <small className="field-help" id={`${id}-device-help`}>Run override; requires available hardware.</small>
              </div>
              {showMicroBatchSize ? (
                <div className="field">
                  <label htmlFor={`${id}-batch-size`}>Structures per microbatch</label>
                  <input id={`${id}-batch-size`} type="number" inputMode="numeric" min={1} max={4} step={1}
                    value={values.batchSize} disabled={disabled} aria-invalid={Boolean(errors.batchSize)}
                    aria-describedby={`${id}-batch-help${errors.batchSize ? ` ${id}-batch-error` : ""}`}
                    onChange={(event) => update("batchSize", event.target.value)} />
                  <small className="field-help" id={`${id}-batch-help`}>1–4 · lower to reduce memory use</small>
                  {errors.batchSize ? <span className="inline-error" role="alert" id={`${id}-batch-error`}>{errors.batchSize}</span> : null}
                </div>
              ) : null}
              {deviceChoice === "custom" ? (
                <div className="field">
                  <label htmlFor={`${id}-cuda`}>CUDA device index</label>
                  <input id={`${id}-cuda`} type="number" inputMode="numeric" min={0} step={1}
                    value={values.device.toLowerCase().startsWith("cuda:") ? values.device.slice(5) : ""} disabled={disabled} placeholder="0"
                    aria-invalid={Boolean(errors.device)} aria-describedby={`${id}-cuda-help${errors.device ? ` ${id}-device-error` : ""}`}
                    onChange={(event) => update("device", `cuda:${event.target.value}`)} />
                  <small className="field-help" id={`${id}-cuda-help`}>0 = cuda:0 · integer ≥ 0</small>
                  {errors.device ? <span className="inline-error" role="alert" id={`${id}-device-error`}>{errors.device}</span> : null}
                </div>
              ) : null}
            </div>
          </section>
        </div>
        <details className="parameter-reference">
          <summary>Parameter reference</summary>
          <div>
            <p>Model scores rank predicted binding-site residues; they are not independently calibrated probabilities.</p>
            <p>Clusters are connected components of selected Cα pairs within the distance cutoff (≤). Changing either cutoff leaves model scores unchanged.</p>
            <p>Device overrides use the installed runtime; they do not install hardware dependencies.</p>
          </div>
        </details>
      </div>
    </details>
  );
}
