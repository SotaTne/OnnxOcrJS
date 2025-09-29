import type { Mat } from "@techstark/opencv-js";
import type {
  CV2,
  Data,
  DataValues,
  ORT,
  ORTBufferType,
  ORTSessionReturnType,
  Point,
} from "./types/type.js";
import {
  createOperator,
  transform,
  type PreProcessOperator,
} from "./utils/operators.js";
import { DBPostProcess, type DBPostProcessParams } from "./db_postprocess.js";
import type {
  DET_BOX_TYPE,
  DET_DB_BOX_THRESH,
  DET_DB_SCORE_MODE,
  DET_DB_THRESH,
  DET_DB_UNCLIP_RATIO,
  USE_DILATION,
  USE_GCU,
} from "./types/paddle_types.js";
import { PredictBase } from "./predict_base.js";
import type { InferenceSession } from "onnxruntime-web";
import { create_onnx_session_fn } from "./onnx_runtime.js";

let _det_onnx_session: ORTSessionReturnType | undefined = undefined;

export type TextDetectorParams = {
  drop_score?: number;
  limit_side_len: number;
  det_limit_type: "max" | "min" | "resize_long" | null;
  det_db_thresh: DET_DB_THRESH;
  det_db_box_thresh: DET_DB_BOX_THRESH;
  det_db_unclip_ratio: DET_DB_UNCLIP_RATIO;
  use_dilation: USE_DILATION;
  det_db_score_mode: DET_DB_SCORE_MODE;
  det_box_type: DET_BOX_TYPE;
  cv: CV2;
  ort: ORT;
};

export class TextDetector extends PredictBase {
  cv: CV2;
  postprocess_op: DBPostProcess;
  preprocess_op: PreProcessOperator[];
  ort: ORT;
  onnx_session?: InferenceSession;

  constructor(params: TextDetectorParams) {
    super();
    this.cv = params.cv;
    this.ort = params.ort;
    this.preprocess_op = [
      createOperator({
        type: "DetResizeForTest",
        params: {
          cv: this.cv,
          limit_side_len: params.limit_side_len,
          limit_type: params.det_limit_type,
        },
      }),
      createOperator({
        type: "NormalizeImage",
        params: {
          cv: this.cv,
          std: [0.229, 0.224, 0.225],
          mean: [0.485, 0.456, 0.406],
          scale: 1.0 / 255.0,
          order: "hwc",
        },
      }),
      createOperator({
        type: "ToCHWImage",
        params: {
          cv: this.cv,
        },
      }),
      createOperator({
        type: "KeepKeys",
        params: {
          keep_keys: ["image", "shape"],
        },
      }),
    ];
    const postprocess_params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: params.det_db_thresh,
      box_thresh: params.det_db_box_thresh,
      max_candidates: 1000,
      unclip_ratio: params.det_db_unclip_ratio,
      use_dilation: params.use_dilation,
      score_mode: params.det_db_score_mode,
      box_type: params.det_box_type,
      cv: this.cv,
    };
    this.postprocess_op = new DBPostProcess(postprocess_params);
  }

  order_points_clockwise(pts: Point[]): Point[] {
    const rect: Point[] = [
      [0, 0], // top-left
      [0, 0], // top-right
      [0, 0], // bottom-right
      [0, 0], // bottom-left
    ];

    const sumPts = pts.map((p) => p[0] + p[1]);

    const minIndex = sumPts.indexOf(Math.min(...sumPts)); // top-left
    const maxIndex = sumPts.indexOf(Math.max(...sumPts)); // bottom-right

    rect[0] = pts[minIndex]!;
    rect[2] = pts[maxIndex]!;

    const tmp = pts.filter((_, i) => i !== minIndex && i !== maxIndex);

    const diff = tmp.map((p) => p[0] - p[1]);
    const minDiffIndex = diff.indexOf(Math.min(...diff)); // top-right
    const maxDiffIndex = diff.indexOf(Math.max(...diff)); // bottom-left

    rect[1] = tmp[minDiffIndex]!;
    rect[3] = tmp[maxDiffIndex]!;

    return rect;
  }

  clip_det_res() {}

  filter_tag_det_res() {}

  filter_tag_det_res_only_clip() {}

  execute(_img: Mat) {
    const ori_im = _img.clone();
    const data: Data = { image: ori_im, shape: null };
    const transformed = transform(data, this.preprocess_op) as
      | DataValues[]
      | null;
    if (transformed === null) {
      return [null, 0];
    }
    const img = transformed[0]! as Data["image"];
    const shape = transformed[0]! as Data["shape"];
  }

  async create_onnx_session(
    modelArrayBuffer: ORTBufferType,
    use_gpu: USE_GCU
  ): Promise<ORTSessionReturnType> {
    if (_det_onnx_session) {
      this.onnx_session = _det_onnx_session;
      return _det_onnx_session;
    }
    _det_onnx_session = await create_onnx_session_fn(
      this.ort,
      modelArrayBuffer,
      use_gpu
    );
    this.onnx_session = _det_onnx_session;
    return _det_onnx_session;
  }
}
