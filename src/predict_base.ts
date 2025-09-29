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
  abstract onnx_session?: ORTSessionReturnType;

  abstract create_onnx_session(
    modelArrayBuffer: ORTBufferType,
    use_gpu: USE_GCU
  ): Promise<ORTSessionReturnType>;

  get_onnx_session(): ORTSessionReturnType {
    if (!this.onnx_session) {
      throw new Error("ONNX session is not initialized.");
    }
    return this.onnx_session;
  }

  get_output_name(session: ORTSessionReturnType) {
    return get_output_name_fn(session);
  }

  get_input_name(session: ORTSessionReturnType) {
    return get_input_name_fn(session);
  }

  get_input_feed(session: ORTSessionReturnType, input_data: ORTTensorType) {
    const input_names = this.get_input_name(session);
    return get_input_feed_fn(input_names, input_data);
  }
}
