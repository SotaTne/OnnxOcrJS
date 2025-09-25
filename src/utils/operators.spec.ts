import { expect ,describe, it, beforeAll} from 'vitest'
import type cvReadyPromiseType from "@techstark/opencv-js";
import { NormalizeImage,DetResizeForTest,ToCHWImage,KeepKeys } from "./operators.js";
import type { Data } from '../types/type.js';
import ndarray from 'ndarray';

let cv: Awaited<typeof cvReadyPromiseType>;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  cv = await cvReadyPromise;
});

describe('NormalizeImage',()=>{
  it('normalizes an image correctly hwc',async ()=>{
    const normalizeImage = new NormalizeImage({
      scale: 1.0 / 255.0,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      order: 'hwc',
      cv
    });
    const imgData = cv.matFromArray(1000, 1000, cv.CV_8UC3, Array(1000 * 1000 * 3).fill(2));
    const result = normalizeImage.execute({ image: imgData, shape: null });
    expect(result).toBeDefined();
  })
  it('normalizes an image correctly chw',async ()=>{
    const normalizeImage = new NormalizeImage({
      scale: 1.0 / 255.0,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      order: 'chw',
      cv
    });
    const imgData = cv.matFromArray(3, 1000, cv.CV_8UC4, Array(3 * 1000 * 4).fill(2));
    const result = normalizeImage.execute({ image: imgData, shape: null });
    expect(result).toBeDefined();
  })
})

describe("DetResizeForTest",()=>{

  it("resize_type=0, limit_type=max", ()=>{
    const op = new DetResizeForTest({ limit_side_len: 64, limit_type: "max", cv });
    const imgData = cv.matFromArray(200, 100, cv.CV_8UC3, Array(200*100*3).fill(1));
    const result = op.execute({ image: imgData, shape: null });
    expect(result.image.rows).toBeLessThanOrEqual(64);
    expect(result.image.cols).toBeLessThanOrEqual(64);
  });

  it("resize_type=0, limit_type=min", ()=>{
    const op = new DetResizeForTest({ limit_side_len: 128, limit_type: "min", cv });
    const imgData = cv.matFromArray(60, 200, cv.CV_8UC3, Array(60*200*3).fill(1));
    const result = op.execute({ image: imgData, shape: null });
    expect(result.image.rows).toBeGreaterThanOrEqual(128);
    expect(result.image.cols).toBeGreaterThanOrEqual(128);
  });

  it("resize_type=0, limit_type=resize_long", ()=>{
    const op = new DetResizeForTest({ limit_side_len: 100, limit_type: "resize_long", cv });
    const imgData = cv.matFromArray(400, 200, cv.CV_8UC3, Array(400*200*3).fill(1));
    const result = op.execute({ image: imgData, shape: null });
    expect(result.image.rows).toBeLessThanOrEqual(100);
    expect(result.image.cols).toBeLessThanOrEqual(100);
  });

  it("resize_type=0, invalid limit_type throws", ()=>{
    const op = new DetResizeForTest({ limit_side_len: 64, limit_type:"min", cv });
    // @ts-expect-error: force invalid value
    op.limit_type = "unknown";
    const imgData = cv.matFromArray(100, 100, cv.CV_8UC3, Array(100*100*3).fill(1));
    expect(()=>op.execute({ image: imgData, shape: null })).toThrow();
  });

  it("resize_type=1, keep_ratio=false", ()=>{
    const op = new DetResizeForTest({ image_shape: [128,256], keep_ratio:false, cv });
    const imgData = cv.matFromArray(200, 100, cv.CV_8UC3, Array(200*100*3).fill(1));
    const result = op.execute({ image: imgData, shape: null });
    expect(result.image.rows).toBe(128);
    expect(result.image.cols).toBe(256);
  });

  it("resize_type=1, keep_ratio=true", ()=>{
    const op = new DetResizeForTest({ image_shape: [128,128], keep_ratio:true, cv });
    const imgData = cv.matFromArray(100, 50, cv.CV_8UC3, Array(100*50*3).fill(1));
    const result = op.execute({ image: imgData, shape: null });
    expect(result.image.rows).toBe(128);
    expect(result.image.cols % 32).toBe(0); // keep_ratioで32倍数
  });

  it("resize_type=1 throws when image_shape is null", ()=>{
    const op = new DetResizeForTest({ cv });
    // 強制的にresize_type=1にしてテスト
    op.resize_type = 1;
    op.image_shape = null;
    const imgData = cv.matFromArray(100, 100, cv.CV_8UC3, Array(100*100*3).fill(1));
    expect(()=>op.execute({ image: imgData, shape: null })).toThrow();
  });

  it("resize_type=2, normal case", ()=>{
    const op = new DetResizeForTest({ resize_long: 128, cv });
    const imgData = cv.matFromArray(400, 200, cv.CV_8UC3, Array(400*200*3).fill(1));
    const result = op.execute({ image: imgData, shape: null });
    expect(result.image.rows % 128).toBe(0);
    expect(result.image.cols % 128).toBe(0);
  });

  it("resize_type=2 throws when resize_long=null", ()=>{
    const op = new DetResizeForTest({ resize_long: 128, cv });
    op.resize_long = null;
    const imgData = cv.matFromArray(100, 100, cv.CV_8UC3, Array(100*100*3).fill(1));
    expect(()=>op.execute({ image: imgData, shape: null })).toThrow();
  });

  it("pads very small image", ()=>{
    const op = new DetResizeForTest({ limit_side_len: 32, limit_type:"min", cv });
    const imgData = cv.matFromArray(10, 10, cv.CV_8UC3, Array(10*10*3).fill(1));
    const result = op.execute({ image: imgData, shape: null });
    expect(result.image.rows).toBeGreaterThanOrEqual(32);
    expect(result.image.cols).toBeGreaterThanOrEqual(32);
  });

  it("throws if input is not cv.Mat", ()=>{
    const op = new DetResizeForTest({ limit_side_len: 32, limit_type:"min", cv });
    expect(()=>op.execute({ image: "not a Mat", shape:null } as any)).toThrow();
  });

});

describe("ToCHWImage", () => {
  it("converts Mat image to CHW ndarray", () => {
    const toCHW = new ToCHWImage({ cv });
    // 3チャネルの画像を作る (高さ=2, 幅=2)
    const mat = cv.matFromArray(2, 2, cv.CV_8UC3, [
      1, 2, 3,   4, 5, 6,
      7, 8, 9,   10,11,12,
    ]);
    const result = toCHW.execute({ image: mat, shape: null });
    expect(result.image.shape).toEqual([3, 2, 2]); // (C,H,W)
  });

  it("throws error if input is not Mat", () => {
    const toCHW = new ToCHWImage({ cv });
    expect(() => toCHW.execute({ image: "not a Mat", shape: null } as any)).toThrow();
  });
});

describe("KeepKeys", () => {
  it("keeps only specified keys", () => {
    const array = ndarray(new Float32Array([1, 2, 3, 4]), [1, 3, 1, 1]);
    const shape = [1, 3, 1, 1] as [number, number, number, number];
    const data:Data = {
      image: array,
      shape,
    }
    const keepKeys = new KeepKeys({ keep_keys: ["shape", "image"] });
    const result = keepKeys.execute(data);
    expect(result).toEqual([
      shape,
      array
    ]);
  });

  it("ignores keys not in data", () => {
    const mat = cv.matFromArray(2, 2, cv.CV_8UC1, [1, 2, 3, 4]);
    const shape = [1, 1, 2, 2] as [number, number, number, number];
    const data: Data = {
      image: mat,
      shape,
    };

    const keepKeys = new KeepKeys({ keep_keys: ["shape", "nonexistent"] as any });
    const result = keepKeys.execute(data);

    expect(result).toEqual([shape]);
    mat.delete();
  });
});