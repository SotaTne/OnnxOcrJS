import type { NdArray } from "ndarray";
import type { CV2, Data } from "../types/type.js";
import { broadcastTo, fillZero, matToLine, matToList } from "./func.js";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import type { DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, SR_IMAGE_SHAPE } from "../types/paddle_types.js";
import type { Mat } from "@techstark/opencv-js";

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

export type DetResizeForTestParams = ({
  image_shape: [number,number];
  keep_ratio:boolean;
}|{
  limit_side_len:number;
  limit_type:DET_LIMIT_TYPE | null;
}|{
  resize_long:number | null;
} | null) &{
  cv:CV2
};

export class DetResizeForTest {
  resize_type=0;
  keep_ratio=false;
  limit_type:DET_LIMIT_TYPE='max';
  limit_side_len:DET_LIMIT_SIDE_LEN=960;
  image_shape:[number,number]|null=null;
  resize_long:number|null=null;
  cv:CV2;
  constructor(params:DetResizeForTestParams){
    this.cv = params.cv;
    if ('image_shape' in params){
      this.image_shape = params.image_shape;
      this.resize_type = 1
      if (params.keep_ratio) {
        this.keep_ratio = params.keep_ratio;
      }
    } else if ('limit_side_len' in params){
      this.limit_side_len = params.limit_side_len;
      this.limit_type = params.limit_type || "min";
    } else if ('resize_long' in params){
      this.resize_type = 2;
      this.resize_long = params.resize_long || 960;
    } else{ 
      this.limit_side_len = 736;
      this.limit_type = 'min'; 
    }
  }

  execute(data:Data):Data{
    let img = data.image;
    const imgH = img.rows;
    const imgW = img.cols;
    let ratio_h:number;
    let ratio_w:number;
    if (imgH + imgW < 64) {
      img = this.image_padding(img);
    }

    if (this.resize_type === 0){
      const result = this.resize_image_type0(img);
      img = result.img;
      ratio_h = result.ratio_h;
      ratio_w = result.ratio_w;
    }
  }

  image_padding(img:Mat,value=0):Mat{
    const h = img.rows;
    const w = img.cols;
    const c = img.channels();
    const defaultIm = ndarray(Uint8Array.from(matToLine(img,this.cv).data),[h,w,c]);
    const im_pad = fillZero(defaultIm, [Math.max(32,h),Math.max(32,w),c]);

    const isSuccess = ops.addseq(im_pad,value);
    if (!isSuccess){
      throw new Error("image_padding: failed to pad image");
    }
    im_pad.data.set()

  }

  resize_image_type0(img:Mat):{img:Mat,ratio_h:number,ratio_w:number} {
    
  }
}