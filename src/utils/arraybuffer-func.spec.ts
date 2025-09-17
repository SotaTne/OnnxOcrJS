import {matToArrayBuffer} from "./func.js";
import { expect ,describe, it, beforeAll} from 'vitest'
import type cvReadyPromiseType from "@techstark/opencv-js";

let cv: Awaited<typeof cvReadyPromiseType>;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  cv = await cvReadyPromise;
});

describe('matToArrayBuffer', () => {

  // 8-bit unsigned
  describe('8UC', () => {
    it('returns data for CV_8UC1', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8UC1, [0,1,2,3]); // 2*2*1=4
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint8Array);
      expect(Array.from(data as Uint8Array)).toEqual([0,1,2,3]);
      mat.delete();
    });
    it('returns data for CV_8UC2', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8UC2, [0,1,2,3,4,5,6,7]); // 2*2*2=8
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint8Array);
      expect(Array.from(data as Uint8Array)).toEqual([0,1,2,3,4,5,6,7]);
      mat.delete();
    });
    it('returns data for CV_8UC3', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8UC3, [0,1,2,3,4,5,6,7,8,9,10,11]); // 2*2*3=12
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint8Array);
      expect(Array.from(data as Uint8Array)).toEqual([0,1,2,3,4,5,6,7,8,9,10,11]);
      mat.delete();
    });
    it('returns data for CV_8UC4', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8UC4, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]); // 2*2*4=16
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint8Array);
      expect(Array.from(data as Uint8Array)).toEqual([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
      mat.delete();
    });
  });

  // 8-bit signed
  describe('8SC', () => {
    it('returns data for CV_8SC1', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8SC1, [0,1,2,3]); // 2*2*1=4
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int8Array);
      expect(Array.from(data as Int8Array)).toEqual([0,1,2,3]);
      mat.delete();
    });
    it('returns data for CV_8SC2', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8SC2, [0,1,2,3,4,5,6,7]); // 2*2*2=8
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int8Array);
      expect(Array.from(data as Int8Array)).toEqual([0,1,2,3,4,5,6,7]);
      mat.delete();
    });
    it('returns data for CV_8SC3', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8SC3, [0,1,2,3,4,5,6,7,8,9,10,11]); // 2*2*3=12
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int8Array);
      expect(Array.from(data as Int8Array)).toEqual([0,1,2,3,4,5,6,7,8,9,10,11]);
      mat.delete();
    });
    it('returns data for CV_8SC4', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_8SC4, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]); // 2*2*4=16
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int8Array);
      expect(Array.from(data as Int8Array)).toEqual([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
      mat.delete();
    });
  });

  // 16-bit unsigned
  describe('16UC', () => {
    it('returns data for CV_16UC1', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16UC1, [1000,2000,3000,4000]); // 2*2*1=4
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint16Array);
      expect(Array.from(data as Uint16Array)).toEqual([1000,2000,3000,4000]);
      mat.delete();
    });
    it('returns data for CV_16UC2', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16UC2, [1000,2000,3000,4000,5000,6000,7000,8000]); // 2*2*2=8
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint16Array);
      expect(Array.from(data as Uint16Array)).toEqual([1000,2000,3000,4000,5000,6000,7000,8000]);
      mat.delete();
    });
    it('returns data for CV_16UC3', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16UC3, [1,2,3,4,5,6,7,8,9,10,11,12]); // 2*2*3=12
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint16Array);
      expect(Array.from(data as Uint16Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12]);
      mat.delete();
    });
    it('returns data for CV_16UC4', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16UC4, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]); // 2*2*4=16
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Uint16Array);
      expect(Array.from(data as Uint16Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
      mat.delete();
    });
  });

  // 16-bit signed
  describe('16SC', () => {
    it('returns data for CV_16SC1', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16SC1, [-1,-2,-3,-4]); // 2*2*1=4
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int16Array);
      expect(Array.from(data as Int16Array)).toEqual([-1,-2,-3,-4]);
      mat.delete();
    });
    it('returns data for CV_16SC2', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16SC2, [-1,-2,-3,-4,-5,-6,-7,-8]); // 2*2*2=8
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int16Array);
      expect(Array.from(data as Int16Array)).toEqual([-1,-2,-3,-4,-5,-6,-7,-8]);
      mat.delete();
    });
    it('returns data for CV_16SC3', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16SC3, [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12]); // 2*2*3=12
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int16Array);
      expect(Array.from(data as Int16Array)).toEqual([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12]);
      mat.delete();
    });
    it('returns data for CV_16SC4', () => {
      const mat = cv.matFromArray(2, 2, cv.CV_16SC4, [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16]); // 2*2*4=16
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int16Array);
      expect(Array.from(data as Int16Array)).toEqual([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16]);
      mat.delete();
    });
  });

  // 32-bit signed
  describe('32SC', () => {
    it('returns data for CV_32SC1', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32SC1,[1,2,3,4]); // 2*2*1=4
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int32Array);
      expect(Array.from(data as Int32Array)).toEqual([1,2,3,4]);
      mat.delete();
    });
    it('returns data for CV_32SC2', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32SC2,[1,2,3,4,5,6,7,8]); // 2*2*2=8
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int32Array);
      expect(Array.from(data as Int32Array)).toEqual([1,2,3,4,5,6,7,8]);
      mat.delete();
    });
    it('returns data for CV_32SC3', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32SC3,[1,2,3,4,5,6,7,8,9,10,11,12]); // 2*2*3=12
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int32Array);
      expect(Array.from(data as Int32Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12]);
      mat.delete();
    });
    it('returns data for CV_32SC4', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32SC4,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]); // 2*2*4=16
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Int32Array);
      expect(Array.from(data as Int32Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
      mat.delete();
    });
  });

  // 32-bit float
  describe('32FC', () => {
    it('returns data for CV_32FC1', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32FC1,[1,2,3,4]); // 2*2*1=4
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float32Array);
      expect(Array.from(data as Float32Array)).toEqual([1,2,3,4]);
      mat.delete();
    });
    it('returns data for CV_32FC2', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32FC2,[1,2,3,4,5,6,7,8]); // 2*2*2=8
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float32Array);
      expect(Array.from(data as Float32Array)).toEqual([1,2,3,4,5,6,7,8]);
      mat.delete();
    });
    it('returns data for CV_32FC3', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32FC3,[1,2,3,4,5,6,7,8,9,10,11,12]); // 2*2*3=12
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float32Array);
      expect(Array.from(data as Float32Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12]);
      mat.delete();
    });
    it('returns data for CV_32FC4', () => {
      const mat = cv.matFromArray(2,2,cv.CV_32FC4,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]); // 2*2*4=16
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float32Array);
      expect(Array.from(data as Float32Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
      mat.delete();
    });
  });

  // 64-bit float
  describe('64FC', () => {
    it('returns data for CV_64FC1', () => {
      const mat = cv.matFromArray(2,2,cv.CV_64FC1,[1,2,3,4]); // 2*2*1=4
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float64Array);
      expect(Array.from(data as Float64Array)).toEqual([1,2,3,4]);
      mat.delete();
    });
    it('returns data for CV_64FC2', () => {
      const mat = cv.matFromArray(2,2,cv.CV_64FC2,[1,2,3,4,5,6,7,8]); // 2*2*2=8
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float64Array);
      expect(Array.from(data as Float64Array)).toEqual([1,2,3,4,5,6,7,8]);
      mat.delete();
    });
    it('returns data for CV_64FC3', () => {
      const mat = cv.matFromArray(2,2,cv.CV_64FC3,[1,2,3,4,5,6,7,8,9,10,11,12]); // 2*2*3=12
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float64Array);
      expect(Array.from(data as Float64Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12]);
      mat.delete();
    });
    it('returns data for CV_64FC4', () => {
      const mat = cv.matFromArray(2,2,cv.CV_64FC4,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]); // 2*2*4=16
      const data = matToArrayBuffer(mat, cv);
      expect(data).toBeInstanceOf(Float64Array);
      expect(Array.from(data as Float64Array)).toEqual([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
      mat.delete();
    });
  });
});