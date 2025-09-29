import type { USE_GCU } from "./types/paddle_types.js";
import type {
  ORT,
  ORTBufferType,
  ORTParametersOptionType,
  ORTRunFeedsType,
  ORTSessionReturnType,
  ORTTensorType,
} from "./types/type.js";

export const onnxRuntimeGPUOption: ORTParametersOptionType = {
  executionProviders: ["cuda", "dml", "webgpu", "webgl", "cpu", "wasm"],
};

export const onnxRuntimeCPUOption: ORTParametersOptionType = {
  executionProviders: ["cpu", "wasm"],
};

export function get_output_name_fn(session: ORTSessionReturnType): string[] {
  return [...session.outputNames];
}

export function get_input_name_fn(session: ORTSessionReturnType): string[] {
  return [...session.inputNames];
}

export function get_input_feed_fn(
  input_names: string[],
  input_data: ORTTensorType
): ORTRunFeedsType {
  const feeds: { [name: string]: ORTTensorType } = {};
  for (const name of input_names) {
    feeds[name] = input_data;
  }
  const ort_feeds: ORTRunFeedsType = feeds;
  return ort_feeds;
}

export async function create_onnx_session_fn(
  ort: ORT,
  modelArrayBuffer: ORTBufferType,
  use_gpu: USE_GCU
): Promise<ORTSessionReturnType> {
  const options = use_gpu ? onnxRuntimeGPUOption : onnxRuntimeCPUOption;
  return await ort.InferenceSession.create(modelArrayBuffer, options);
}
