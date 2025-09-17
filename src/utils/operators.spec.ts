import { expect ,describe, it, beforeAll} from 'vitest'
import type cvReadyPromiseType from "@techstark/opencv-js";
import { NormalizeImage } from "./operators.js";

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
    const result = normalizeImage.execute({ image: imgData });
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
    const result = normalizeImage.execute({ image: imgData });
    expect(result).toBeDefined();
  })
})