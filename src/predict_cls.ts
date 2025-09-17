import type { CV2 } from './types/type.js';
export type TextClassifierParams = {
  cv: CV2;
}

export class TextClassifier{
  cv: CV2;
      
  constructor(params: TextClassifierParams) {
    this.cv = params.cv;
  }
}