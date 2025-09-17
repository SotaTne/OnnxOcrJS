import type { Mat } from '@techstark/opencv-js';
import type { CV2 } from './types/type.js';

export type TextDetectorParams = {
  drop_score?: number;
  cv: CV2;
}

export class TextDetector{
    cv: CV2;
  
  constructor(params: TextDetectorParams) {
    this.cv = params.cv;
  }


  execute(img: Mat) {
    const ori_im = img.clone();

  }
}