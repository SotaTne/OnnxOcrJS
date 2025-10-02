import type { Mat } from "@techstark/opencv-js";
import { TextSystem, type TextSystemParams } from "./predict_system.js";
import type { Box, Point } from "./types/type.js";

type PartialOmit<T, K extends keyof any> = Partial<Omit<T, K>>;

type PartialOmitTextSystemParams = PartialOmit<
  TextSystemParams,
  | "det_model_array_buffer"
  | "cv"
  | "ort"
  | "cls_model_array_buffer"
  | "rec_model_array_buffer"
  | "rec_char_dict"
>;

type OmitTextSystemParams = Omit<
  TextSystemParams,
  | "det_model_array_buffer"
  | "cv"
  | "ort"
  | "cls_model_array_buffer"
  | "rec_model_array_buffer"
  | "rec_char_dict"
>;

type PickedTextSystemParams = Pick<
  TextSystemParams,
  | "det_model_array_buffer"
  | "cv"
  | "ort"
  | "cls_model_array_buffer"
  | "rec_model_array_buffer"
  | "rec_char_dict"
>;

const initArgs: OmitTextSystemParams = {
  limit_side_len: 960,
  det_limit_type: "max",
  det_db_thresh: 0.3, // 閾値下げる
  det_db_box_thresh: 0.6, // ボックス閾値も下げる
  det_db_unclip_ratio: 1.5,
  use_dilation: false,
  det_db_score_mode: "fast",
  det_box_type: "quad",
  use_gpu: false,
  cls_image_shape: [3, 48, 192], // "3,48,192" を配列に展開
  cls_thresh: 0.9,
  label_list: ["0", "180"],
  cls_batch_num: 6,
  drop_score: 0.5,
  rec_batch_num: 6,
  rec_algorithm: "SVTR_LCNet",
  rec_image_shape: [3, 48, 320],
  use_space_char: false,
  use_angle_cls: true,
};

export class ONNXPaddleOCR {
  params: OmitTextSystemParams = initArgs;
  constructor(params: PartialOmitTextSystemParams) {
    this.params = { ...this.params, ...params };
  }
  async init(params: PickedTextSystemParams) {
    return await TextSystem.create({ ...this.params, ...params });
  }
  async ocr(
    textSystem: TextSystem,
    img: Mat,
    det: true,
    rec: true,
    cls: true | false,
  ): Promise<[Box, [string, number]][]>;

  async ocr(
    textSystem: TextSystem,
    img: Mat,
    det: true,
    rec: false,
    cls: true | false,
  ): Promise<Point[][][]>;

  async ocr(
    textSystem: TextSystem,
    img: Mat,
    det: false,
    rec: true,
    cls: true | false,
  ): Promise<[string, number][][]>;

  async ocr(
    textSystem: TextSystem,
    img: Mat,
    det = true,
    rec = true,
    cls = true,
  ) {
    if (cls == true && this.params.use_angle_cls == false) {
      console.warn(
        "Since the angle classifier is not initialized, the angle classifier will not be used during the forward process",
      );
    }
    if (det && rec) {
      const ocr_res = [];
      const [dt_boxes, rec_res] = await textSystem.execute(img, cls);
      if (dt_boxes == null || rec_res == null) {
        return [] as [Box, [string, number]][];
      }
      const tmp_res: [Box, [string, number]][] = dt_boxes.map((box, i) => [
        box,
        rec_res[i]! as [string, number],
      ]);
      ocr_res.push(tmp_res);
      return ocr_res;
    } else if (det && !rec) {
      const ocr_res = [];
      const dt_boxes = await textSystem.text_detector.execute(img);
      if (dt_boxes == null) {
        return [] as Point[][][];
      }
      ocr_res.push(dt_boxes);
      return ocr_res;
    } else {
      const ocr_res: [string, number][][] = [];
      const cls_res: [string, number][][] = [];
      let cls_result_img = [img];
      if (this.params.use_angle_cls && cls && textSystem.text_classifier) {
        const [result_img, cls_res_tmp] =
          await textSystem.text_classifier.execute([img]);
        cls_result_img = result_img;
        if (!rec) {
          cls_res.push(cls_res_tmp);
        }
      }
      const rec_res = await textSystem.text_recognizer.execute(cls_result_img);
      ocr_res.push(rec_res);

      if (!rec) {
        return cls_res;
      }
      return ocr_res;
    }
  }
}
