import type { NdArray } from "ndarray";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import type { CV2,Data } from "../../types/type.js";
import { matToList, broadcastTo } from "../func.js";

export type NormalizeImageParams = {
  scale: number|null;
  mean: number[]|null;
  std: number[]|null;
  order: 'hwc' | 'chw' | null;
  cv:CV2
}

export class NormalizeImage {
  scale: number;
  mean: NdArray<Float32Array>;
  std: NdArray<Float32Array>;
  shape: [number, number, number]
  shapedMean:NdArray<Float32Array>
  shapedStd:NdArray<Float32Array>
  cv:CV2

  constructor(params: NormalizeImageParams) {
    this.cv = params.cv;
    this.scale = params.scale || 1.0 /255.0;
    this.mean = Array.isArray(params.mean) ? ndarray(Float32Array.from(params.mean),[1,3]) : ndarray(Float32Array.from([0.485, 0.456, 0.406]),[1,3]);
    this.std = Array.isArray(params.std) ? ndarray(Float32Array.from(params.std),[1,3]) : ndarray(Float32Array.from([0.229, 0.224, 0.225]),[1,3]);
    this.shape = params.order === 'hwc' ? [1,1,3] : [3,1,1];

    this.shapedMean = ndarray(this.mean.data,this.shape)
    this.shapedStd = ndarray(this.std.data,this.shape)
  }

  execute(data:Data):Data{
    const img = data.image;
    const row = img.rows;
    const col = img.cols;
    const channel = img.channels();
    const imgList = matToList(img, this.cv,false) as number[][][];
    if (imgList === null || imgList === undefined) {
      throw new Error("NormalizeImage: unsupported Mat type");
    }
    const shape = [row, col, channel];
    const ndArrayImg = ndarray(Float32Array.from(imgList.flat(2)), shape);
    // img * scale
    const scaledImg = ndarray(new Float32Array(row * col * channel), shape);
    ops.assign(scaledImg, ndArrayImg);
    const result = ops.mulseq(scaledImg, this.scale);
    if (result === false){
      throw new Error("NormalizeImage: failed to scale image");
    }

    // scaledImg - mean
    const subImg = ndarray(new Float32Array(row * col * channel), shape);
    ops.sub(subImg, scaledImg, broadcastTo(this.shapedMean, shape));

    // (scaledImg - mean) / std
    const divImg = ndarray(new Float32Array(row * col * channel), shape);
    ops.div(divImg, subImg, broadcastTo(this.shapedStd, shape));

    const normalizedImg = this.cv.matFromArray(
      row, col, channel === 1 ? this.cv.CV_32F : this.cv.CV_32FC3, divImg
    )
    img.delete();
    return {
      ...data,
      image: normalizedImg
    };
  }
}