import type { NdArray } from "ndarray";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import type { CV2,Data } from "../../types/type.js";
import { matToList, broadcastTo, matToLine } from "../func.js";

export type DecodeImageParams = {
  img_mode:"RGB"|"GRAY"|null
  channel_first:boolean|null
  ignore_orientation:boolean|null
  cv:CV2
}

export class DecodeImage {
  img_mode:"RGB"|"GRAY"="RGB"
  cv:CV2
  channel_first:boolean
  ignore_orientation:boolean


  constructor(params: DecodeImageParams) {
    this.cv = params.cv;
    this.img_mode = params.img_mode || "RGB";
    this.channel_first = params.channel_first || false;
    this.ignore_orientation = params.ignore_orientation || false;
  }

  execute(data:Data):Data{
    const img = data.image;
    const row = img.rows;
    const col = img.cols;
    const channel = img.channels();
    const {data: imgLine} = matToLine(img, this.cv);

    if (imgLine === null || imgLine === undefined) {
      throw new Error("NormalizeImage: unsupported Mat type");
    }
    const shape = [row, col, channel];
    const ndArrayImg = ndarray(Uint8Array.from(imgLine), shape);
    if( this.ignore_orientation){
      const cvImg = cv.imread
      ndArrayImg = 
    }
  }
}