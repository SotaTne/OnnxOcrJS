import type { Mat } from "@techstark/opencv-js";
import { TextClassifier, type TextClassifierParams } from "./predict_cls.js";
import { TextDetector, type TextDetectorParams } from "./predict_det.js";
import { TextRecognizer, type TextRecognizerParams } from "./predict_rec.js";
import type {
  DET_BOX_TYPE,
  DROP_SCORE,
  USE_ANGLE_CLS,
} from "./types/paddle_types.js";
import type { Box, CV2 } from "./types/type.js";
import { get_minarea_rect_crop, get_rotate_crop_image } from "./utils/func.js";

type TextSystemParams = {
  text_detector: TextDetector;
  text_recognizer: TextRecognizer;
  text_classifier: TextClassifier | null;
  drop_score?: DROP_SCORE;
  use_angle_cls: USE_ANGLE_CLS;
  det_box_type: DET_BOX_TYPE;
  cv: CV2;
} & TextDetectorParams &
  TextRecognizerParams &
  TextClassifierParams;

export class TextSystem {
  text_detector: TextDetector;
  text_recognizer: TextRecognizer;
  text_classifier: TextClassifier | null = null;
  drop_score: DROP_SCORE;
  use_angle_cls: USE_ANGLE_CLS;
  det_box_type: DET_BOX_TYPE = "quad";
  cv: CV2;
  // save_crop_res: boolean = false;

  constructor(
    params: TextSystemParams & {
      text_detector: TextDetector;
      text_recognizer: TextRecognizer;
    }
  ) {
    this.text_detector = params.text_detector;
    this.text_recognizer = params.text_recognizer;
    this.drop_score = params.drop_score ?? 0.7;
    this.use_angle_cls = params.use_angle_cls;
    if (this.use_angle_cls) {
      this.text_classifier = new TextClassifier(params);
    }
    this.cv = params.cv;
  }

  static async create(params: TextSystemParams) {
    const text_detector = await TextDetector.create(params);
    const text_recognizer = await TextRecognizer.create(params);
    return new TextSystem({ ...params, text_detector, text_recognizer });
  }

  async execute(img: Mat, cls = true): Promise<[Box[] | null, any[] | null]> {
    const ori_img = img.clone();

    // 1. Detection
    let dt_boxes: Box[] | null = (await this.text_detector.execute(ori_img)) as
      | Box[]
      | null;
    if (!Array.isArray(dt_boxes) || dt_boxes.length === 0) {
      return [null, null];
    }

    let img_crop_list: Mat[] = [];
    dt_boxes = sortedBoxes(dt_boxes);

    for (const box of dt_boxes) {
      const tmp_box: Box = [...box];
      const img_crop: Mat =
        this.det_box_type === "quad"
          ? get_rotate_crop_image(ori_img, tmp_box, this.cv)
          : get_minarea_rect_crop(ori_img, tmp_box, this.cv);
      img_crop_list.push(img_crop);
    }
    if (this.use_angle_cls && cls && this.text_classifier) {
      [img_crop_list] = this.text_classifier.execute(img_crop_list);
    }
    const rec_res = await this.text_recognizer.execute(img_crop_list);
    // if (this.save_crop_res) {
    //   // Save cropped images
    // }
    const filtered_boxes: Box[] = [];
    const filtered_rec_res = [];
    if (dt_boxes.length !== rec_res.length) {
      throw new Error("dt_boxes and rec_res length mismatch");
    }
    for (let i = 0; i < dt_boxes.length; i++) {
      const box = dt_boxes[i]!;
      const rec_result = rec_res[i]!;
      const [_text, score] = rec_result;
      if (score > this.drop_score) {
        filtered_boxes.push(box);
        filtered_rec_res.push(rec_result);
      }
    }
    return [filtered_boxes, filtered_rec_res];
  }
}

export function sortedBoxes(dt_boxes: Box[]): Box[] {
  const numBoxes = dt_boxes.length;

  // まず (y, x) で大まかにソート
  const sorted = dt_boxes.slice().sort((a, b) => {
    if (a[0][1] === b[0][1]) {
      return a[0][0] - b[0][0]; // y が同じなら x でソート
    }
    return a[0][1] - b[0][1]; // y でソート
  });

  const boxes = [...sorted];

  for (let i = 0; i < numBoxes - 1; i++) {
    for (let j = i; j >= 0; j--) {
      const yDiff = Math.abs(boxes[j + 1]![0][1] - boxes[j]![0][1]);
      const xRight = boxes[j + 1]![0][0];
      const xLeft = boxes[j]![0][0];

      if (yDiff < 10 && xRight < xLeft) {
        // swap
        const tmp = boxes[j]!;
        boxes[j] = boxes[j + 1]!;
        boxes[j + 1] = tmp;
      } else {
        break;
      }
    }
  }

  return boxes;
}
