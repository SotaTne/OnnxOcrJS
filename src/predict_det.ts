import type { Mat } from '@techstark/opencv-js';
import type { CV2 } from './types/type.js';
import { createOperator } from './utils/operators.js';
import type {DBPostProcessParams} from './db_postprocess.js';
import type { DET_BOX_TYPE, DET_DB_BOX_THRESH, DET_DB_SCORE_MODE, DET_DB_THRESH,DET_DB_UNCLIP_RATIO, USE_DILATION } from './types/paddle_types.js';

export type TextDetectorParams = {
  drop_score?: number;
  limit_side_len: number;
  det_limit_type:"max"|"min"|"resize_long"|null;
  det_db_thresh:DET_DB_THRESH
  det_db_box_thresh:DET_DB_BOX_THRESH
  det_db_unclip_ratio:DET_DB_UNCLIP_RATIO
  use_dilation:USE_DILATION
  det_db_score_mode:DET_DB_SCORE_MODE
  det_box_type:DET_BOX_TYPE
  cv: CV2;
}

export class TextDetector{
    cv: CV2;
  
  constructor(params: TextDetectorParams) {
    this.cv = params.cv;
    const pre_process_list = [
      createOperator({
        type: "DetResizeForTest",
        params: {
          cv: this.cv,
          limit_side_len: params.limit_side_len,
          limit_type: params.det_limit_type,
        }
      }),
      createOperator({
        type: "NormalizeImage",
        params: {
          cv:this.cv,
          std:[0.229, 0.224, 0.225],
          mean:[0.485, 0.456, 0.406],
          scale:1.0/255.0,
          order:"hwc"
        }
      }),
      createOperator({
        type: "ToCHWImage",
        params: {
          cv:this.cv
        }
      }),
      createOperator({
        type: "KeepKeys",
        params: {
          keep_keys: ["image", "shape"]
        }
      })
    ]
    const postprocess_params:DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: params.det_db_thresh,
      box_thresh: params.det_db_box_thresh,
      max_candidates: 1000,
      unclip_ratio: params.det_db_unclip_ratio,
      use_dilation: params.use_dilation,
      score_mode: params.det_db_score_mode,
      box_type: params.det_box_type,
      cv: this.cv
    }
    
  }


  execute(img: Mat) {
    const ori_im = img.clone();

  }
}