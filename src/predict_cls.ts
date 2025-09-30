import type { Mat } from "@techstark/opencv-js";
import { ClsPostProcess } from "./cls_postprocess.js";
import { create_onnx_session_fn } from "./onnx_runtime.js";
import { PredictBase } from "./predict_base.js";
import type {
  CLS_BATCH_NUM,
  CLS_IMAGE_SHAPE_NUMBER,
  CLS_THRESH,
  LABEL_LIST,
  USE_GCU,
} from "./types/paddle_types.js";
import type {
  CV2,
  ORT,
  ORTBufferType,
  ORTRunFetchesType,
  ORTSessionReturnType,
} from "./types/type.js";
import {
  argsort,
  cloneNdArray,
  matToNdArray,
  ndArrayToList,
  tensorToNdArray,
} from "./utils/func.js";
import ndarray, { type NdArray } from "ndarray";
import ops from "ndarray-ops";

let _cls_onnx_session: ORTSessionReturnType | undefined = undefined;

export type TextClassifierParams = {
  cv: CV2;
  cls_image_shape: CLS_IMAGE_SHAPE_NUMBER | null;
  cls_thresh: CLS_THRESH | null;
  label_list: LABEL_LIST | null;
  cls_batch_num: CLS_BATCH_NUM | null;
  ort: ORT;
  det_model_array_buffer: ORTBufferType;
  use_gpu: USE_GCU;
};

export class TextClassifier extends PredictBase {
  cv: CV2;
  cls_image_shape: CLS_IMAGE_SHAPE_NUMBER;
  cls_thresh: CLS_THRESH;
  postprocess_op: ClsPostProcess;
  cls_batch_num: CLS_BATCH_NUM;

  cls_onnx_session: ORTSessionReturnType;

  ort: ORT;
  cls_model_array_buffer: ORTBufferType;

  cls_input_name: string[];
  cls_output_name: string[];

  use_gpu: USE_GCU;

  constructor(
    params: TextClassifierParams & {
      cls_onnx_session: ORTSessionReturnType;
    }
  ) {
    super();
    this.cv = params.cv;
    this.cls_image_shape = params.cls_image_shape ?? [3, 48, 192];
    this.cls_thresh = params.cls_thresh ?? 0.9;
    this.cls_batch_num = params.cls_batch_num ?? 6;
    this.ort = params.ort;
    this.cls_model_array_buffer = params.det_model_array_buffer;
    this.use_gpu = params.use_gpu;
    this.postprocess_op = new ClsPostProcess({
      label_list: params.label_list,
    });
    this.cls_onnx_session = params.cls_onnx_session;
    this.cls_input_name = this.get_input_name(this.cls_onnx_session);
    this.cls_output_name = this.get_output_name(this.cls_onnx_session);
  }

  static async create(params: TextClassifierParams) {
    const cls_onnx_session = await TextClassifier.get_onnx_session(
      params.det_model_array_buffer,
      params.use_gpu,
      params.ort
    );
    return new TextClassifier({ ...params, cls_onnx_session });
  }

  resize_norm_img(img: Mat): NdArray {
    const imgC = this.cls_image_shape[0]!;
    const imgH = this.cls_image_shape[1]!;
    const imgW = this.cls_image_shape[2]!;
    const h = img.rows;
    const w = img.cols;
    const ratio = w / h;
    let resized_w = Math.ceil(imgW * ratio);
    if (resized_w > imgW) {
      resized_w = imgW;
    }
    const resized_image = new this.cv.Mat();
    this.cv.resize(img, resized_image, new this.cv.Size(resized_w, imgH));
    const ndarray_img = matToNdArray(resized_image, this.cv);
    let resized_ndarray_image: NdArray;
    if (this.cls_image_shape[0] === 1) {
      const img_3c_list = (
        ndArrayToList(cloneNdArray(ndarray_img)) as number[][][]
      ).map((row) => row.map((col) => [col[0]!, col[0]!, col[0]!]));
      const listData = img_3c_list.flat(2).map((v) => (v / 255 - 0.5) / 0.5);
      const ndarray_img_3c = ndarray(Float32Array.from(listData), [
        ndarray_img.shape[0]!,
        ndarray_img.shape[1]!,
        3,
      ]); // (H,W,3)
      resized_ndarray_image = ndarray_img_3c.transpose(2, 0, 1); // (3,H,W)
    } else {
      const listData = (
        ndArrayToList(cloneNdArray(ndarray_img)) as number[][][]
      )
        .flat(2)
        .map((v) => (v / 255 - 0.5) / 0.5);
      const ndarray_img_3c = ndarray(Float32Array.from(listData), [
        ndarray_img.shape[0]!,
        ndarray_img.shape[1]!,
        3,
      ]);
      resized_ndarray_image = ndarray_img_3c.transpose(2, 0, 1); // (3,H,W)
    }
    const zeroImg = ndarray(
      Float32Array.from(Array(imgC * imgH * resized_w).fill(0)),
      [imgC, imgH, imgW]
    );

    const zeroView = zeroImg.hi(imgC, imgH, resized_w);

    ops.assign(zeroView, resized_ndarray_image.hi(imgC, imgH, resized_w));

    return zeroView;
  }

  async execute(img_list: Mat[]): Promise<[Mat[], [string, number][]]> {
    const imgC = this.cls_image_shape[0]!;
    const imgH = this.cls_image_shape[1]!;
    const imgW = this.cls_image_shape[2]!;
    const cloned_img_list: Mat[] = img_list.map((img) => img.clone());
    const img_num = cloned_img_list.length;
    const width_list: number[] = [];
    for (const img of cloned_img_list) {
      width_list.push(img.cols / img.rows);
    }
    const indices = argsort(width_list);

    const cls_res: [string, number][] = new Array(img_num).fill(["", 0]);

    const batch_num = this.cls_batch_num;

    for (let beg_img_no = 0; beg_img_no < img_num; beg_img_no += batch_num) {
      const end_img_no = Math.min(beg_img_no + batch_num, img_num);
      const norm_img_batch: number[][][][] = [];
      let max_wh_ratio = 0;
      for (let ino = beg_img_no; ino < end_img_no; ino++) {
        const h = img_list[indices[ino]!]?.rows!;
        const w = img_list[indices[ino]!]?.cols!;
        let wh_ratio = w / h;
        max_wh_ratio = Math.max(max_wh_ratio, wh_ratio);
      }
      for (let ino = beg_img_no; ino < end_img_no; ino++) {
        const norm_img = this.resize_norm_img(img_list[indices[ino]!]!);
        norm_img_batch.push(ndArrayToList(norm_img) as number[][][]);
      }
      const images_shape = [norm_img_batch.length, imgC, imgH, imgW];
      const img_buffer = Float32Array.from([...norm_img_batch].flat(Infinity));
      const tensor_imgs = new this.ort.Tensor(
        "float32",
        img_buffer,
        images_shape
      );
      const input_feed = this.get_input_feed(this.cls_input_name, tensor_imgs);
      const ort_run_fetches: ORTRunFetchesType = this.cls_output_name;

      const outputs = await this.cls_onnx_session.run(
        input_feed,
        ort_run_fetches
      );
      const result_prod = outputs[0];
      if (!result_prod) {
        throw new Error("No output from the ONNX model.");
      }

      const prodNdArray = tensorToNdArray(result_prod);
      const prodList = ndArrayToList(prodNdArray) as number[][];

      const cls_result = this.postprocess_op.execute(
        prodList,
        prodNdArray.shape,
        null
      );
      for (let rno = 0; rno < cls_result.length; rno++) {
        const [label, score] = cls_result[rno]!;
        cls_res[indices[beg_img_no + rno]!] = [label, score];
        if (label.includes("180") && score > this.cls_thresh) {
          const dstMat = new this.cv.Mat();
          this.cv.rotate(
            img_list[indices[beg_img_no + rno]!]!,
            dstMat,
            this.cv.ROTATE_180
          );
          img_list[indices[beg_img_no + rno]!]?.delete();
          img_list[indices[beg_img_no + rno]!] = dstMat;
        }
      }
    }

    return [img_list, cls_res];
  }

  static async get_onnx_session(
    modelArrayBuffer: ORTBufferType,
    use_gpu: USE_GCU,
    ort: ORT
  ): Promise<ORTSessionReturnType> {
    if (_cls_onnx_session) {
      return _cls_onnx_session;
    }
    _cls_onnx_session = await create_onnx_session_fn(
      ort,
      modelArrayBuffer,
      use_gpu
    );
    return _cls_onnx_session;
  }
}
