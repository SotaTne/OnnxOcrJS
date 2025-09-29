import {
  get_input_feed_fn,
  get_input_name_fn,
  get_output_name_fn,
} from "./onnx_runtime.js";
import type { USE_GCU } from "./types/paddle_types.js";
import type {
  ORT,
  ORTBufferType,
  ORTSessionReturnType,
  ORTTensorType,
} from "./types/type.js";

export abstract class PredictBase {
  abstract ort: ORT;

  static get_onnx_session(
    modelArrayBuffer: ORTBufferType,
    use_gpu: USE_GCU,
    ort: ORT
  ): Promise<ORTSessionReturnType> {
    throw new Error("Method not implemented.");
  }

  get_output_name(session: ORTSessionReturnType) {
    return get_output_name_fn(session);
  }

  get_input_name(session: ORTSessionReturnType) {
    return get_input_name_fn(session);
  }

  get_input_feed(input_names: string[], input_data: ORTTensorType) {
    return get_input_feed_fn(input_names, input_data);
  }
}
