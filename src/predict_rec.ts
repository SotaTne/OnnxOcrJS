import type { DROP_SCORE } from "./types/paddle_types.js";
import type { Mat } from "@techstark/opencv-js";
import type { CV2 } from "./types/type.js";

export type TextRecognizerParams = {
  drop_score?: DROP_SCORE;
  cv: CV2;
}

export class TextRecognizer {
  cv: CV2;
  constructor(params:TextRecognizerParams) {
    this.cv = params.cv;
  }

  execute(img:Mat) {
    const ori_img = img.clone();
  }
}