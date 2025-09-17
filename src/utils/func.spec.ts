import type { Box, Point } from "../types/type.js";
import {boxToLine, boxToMat, broadcastTo, euclideanDistance, fillValue, matToLine, matToList, matToPoints} from "./func.js";
import { expect ,describe, it, beforeAll} from 'vitest'
import type cvReadyPromiseType from "@techstark/opencv-js";
import ndarray from "ndarray";

let cv: Awaited<typeof cvReadyPromiseType>;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  cv = await cvReadyPromise;
});


describe('euclideanDistance', () => {
  it('calculates the correct distance between two points', () => {
    const point1: Point = [0, 0];
    const point2: Point = [3, 4];
    const distance = euclideanDistance(point1, point2);
    expect(distance).toBe(5);
  });
  it('returns 0 for identical points', () => {
    const point: Point = [1, 1];
    const distance = euclideanDistance(point, point);
    expect(distance).toBe(0);
  });
  it('handles negative coordinates', () => {
    const point1: Point = [-1, -1];
    const point2: Point = [2, 3];
    const distance = euclideanDistance(point1, point2);
    expect(distance).toBe(5);
  });
});


describe('boxToLine', ()=>{
  it('converts a box to a line array', ()=>{
    const box:Box = [[0,1],[2,3],[4,5],[6,7]] as const;
    const line = [0,1,2,3,4,5,6,7];
    expect(boxToLine(box)).toEqual(line);
  });
})

describe('boxToMat',()=>{
  it('converts a box to a Mat object',async ()=>{
    const box:Box = [[0,1],[2,3],[4,5],[6,7]];
    const mat = boxToMat(box, cv);
    expect(mat.rows).toBe(4);
    expect(mat.cols).toBe(1);
    expect(mat.type()).toBe(cv.CV_32FC2);
    expect(Array.from(mat.data32F)).toEqual([0,1,2,3,4,5,6,7]);
    mat.delete();
  })
})

describe('boxToMat / matToPoints', () => {
  it('converts a box to a Mat and back', () => {
    const baseBox: Box = [[0,1],[2,3],[4,5],[6,7]];
    const mat = boxToMat(baseBox, cv);
    const points = matToPoints(mat, cv);
    expect(points).toEqual(baseBox);
    mat.delete();
  });
});

describe('matToLine',()=>{
  it('converts a Mat object to a line array', ()=>{
    const Baseline = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
    const mat = cv.matFromArray(4, 2, cv.CV_32SC2, Baseline);
    const {data, row, col, channel} = matToLine(mat,cv);
    expect(data).toEqual(Baseline);
    expect(row).toBe(4);
    expect(col).toBe(2);
    expect(channel).toBe(2);
    mat.delete();
  })
})


describe('matToList', () => {

  it('converts CV_8UC1 2*2*1 to nested list with minCol=false', () => {
    const baseList = [
      [[0],[1]],
      [[2],[3]]
    ]; // 2*2*1=4
    const flatted = baseList.flat(2); // [0,1,2,3]
    const mat = cv.matFromArray(2,2,cv.CV_8UC1, flatted); // 2*2*1=4
    const result = matToList(mat, cv, false);
    expect(result).toEqual(baseList);
    mat.delete();
  });

  it('converts CV_8UC3 3*1*3 to nested list with minCol=false', () => {
    const baseList = [
      [[0,1,2]],
      [[3,4,5]],
      [[6,7,8]]
    ]; // 3*1*3=9
    const flatted = baseList.flat(2); // [0,1,2,3,4,5,6,7,8]
    const mat = cv.matFromArray(3,1,cv.CV_8UC3, flatted); // 3*1*3=9
    const result = matToList(mat, cv, false);
    expect(result).toEqual(baseList);
    mat.delete();
  });

  it('converts CV_16UC2 2*2*2 to nested list with minCol=false', () => {
    const baseList = [
      [[1000,2000],[3000,4000]],
      [[5000,6000],[7000,8000]]
    ]; // 2*2*2=8
    const flatted = baseList.flat(2); // [1000,2000,3000,4000,5000,6000,7000,8000]
    const mat = cv.matFromArray(2,2,cv.CV_16UC2, flatted); // 2*2*2=8
    const result = matToList(mat, cv, false);
    expect(result).toEqual(baseList);
    mat.delete();
  });

  it('converts CV_32FC1 3*2*1 to nested list with minCol=false', () => {
    const baseList = [
      [[1],[2]],
      [[3],[4]],
      [[5],[6]]
    ]; // 3*2*1=6
    const flatted = baseList.flat(2); // [1,2,3,4,5,6]
    const mat = cv.matFromArray(3,2,cv.CV_32FC1, flatted); // 3*2*1=6
    const result = matToList(mat, cv, false);
    expect(result).toEqual(baseList);
    mat.delete();
  });

  it('converts CV_32FC3 2*1*3 to nested list with minCol=true', () => {
    const baseList = [
      [1,2,3],
      [4,5,6]
    ]; // 2*1*3=6
    const flatted = baseList.flat(2); // [1,2,3,4,5,6]
    const mat = cv.matFromArray(2,1,cv.CV_32FC3, flatted); // 2*1*3=6
    const result = matToList(mat, cv, true);
    expect(result).toEqual(baseList);
    mat.delete();
  });

  it('converts CV_64FC4 2*2*4 to nested list with minCol=false', () => {
    const baseList = [
      [[1,2,3,4],[5,6,7,8]],
      [[9,10,11,12],[13,14,15,16]]
    ]; // 2*2*4=16
    const flatted = baseList.flat(2); // [1..16]
    const mat = cv.matFromArray(2,2,cv.CV_64FC4, flatted); // 2*2*4=16
    const result = matToList(mat, cv, false);
    expect(result).toEqual(baseList);
    mat.delete();
  });

});


describe("broadcastTo", () => {
  // 成功するケース（右揃えで NumPy 互換）
  it("broadcast (1,1,3) -> (2,3,3)", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [1, 1, 3]);
    const out = broadcastTo(arr, [2, 3, 3]);
    expect(out.shape).toEqual([2, 3, 3]);
    expect(out.get(0, 0, 0)).toBe(1);
    expect(out.get(1, 2, 2)).toBe(3);
  });

  it("broadcast (3,) -> (3,3)", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [3]);
    const out = broadcastTo(arr, [3, 3]);
    expect(out.shape).toEqual([3, 3]);
    expect(out.get(0, 0)).toBe(1);
    expect(out.get(2, 2)).toBe(3);
  });

  it("broadcast (1,3) -> (4,3)", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [1, 3]);
    const out = broadcastTo(arr, [4, 3]);
    expect(out.shape).toEqual([4, 3]);
    expect(out.get(0, 0)).toBe(1);
    expect(out.get(3, 2)).toBe(3);
  });

  it("broadcast (3,1) -> (3,4)", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [3, 1]);
    const out = broadcastTo(arr, [3, 4]);
    expect(out.shape).toEqual([3, 4]);
    expect(out.get(0, 0)).toBe(1);
    expect(out.get(2, 3)).toBe(3);
  });

  it("broadcast (3,) -> (4,3) succeeds", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [3]);
    const out = broadcastTo(arr, [4, 3]);
    expect(out.shape).toEqual([4, 3]);
    expect(out.get(0, 0)).toBe(1);
    expect(out.get(3, 2)).toBe(3);
  });

  it("broadcast (2,3) -> (3,2) fails", () => {
    const arr = ndarray(new Float32Array([1,2,3,4,5,6]), [2,3]);
    expect(() => broadcastTo(arr, [3,2])).toThrow();
  });

  // エッジケース（NumPy的には成功）
  it("broadcast (1,1,3) -> (1,1,3) identity", () => {
    const arr = ndarray(new Float32Array([1,2,3]), [1,1,3]);
    const out = broadcastTo(arr, [1,1,3]);
    expect(out.shape).toEqual([1,1,3]);
    expect(out.get(0,0,2)).toBe(3);
  });

  it("broadcast (1,1,3) -> (2,1,3) edge", () => {
    const arr = ndarray(new Float32Array([1,2,3]), [1,1,3]);
    const out = broadcastTo(arr, [2,1,3]);
    expect(out.shape).toEqual([2,1,3]);
    expect(out.get(1,0,1)).toBe(2);
  });

  // 追加の成功ケース
  it("broadcast scalar (1,) -> (2,3,4)", () => {
    const arr = ndarray(new Float32Array([5]), [1]);
    const out = broadcastTo(arr, [2, 3, 4]);
    expect(out.shape).toEqual([2, 3, 4]);
    expect(out.get(0, 0, 0)).toBe(5);
    expect(out.get(1, 2, 3)).toBe(5);
  });

  it("broadcast (2,1,1) -> (2,3,4)", () => {
    const arr = ndarray(new Float32Array([10, 20]), [2, 1, 1]);
    const out = broadcastTo(arr, [2, 3, 4]);
    expect(out.shape).toEqual([2, 3, 4]);
    expect(out.get(0, 0, 0)).toBe(10);
    expect(out.get(0, 2, 3)).toBe(10);
    expect(out.get(1, 0, 0)).toBe(20);
    expect(out.get(1, 2, 3)).toBe(20);
  });

  it("broadcast (4,) -> (2,3,4) multi-dimensional extension", () => {
    const arr = ndarray(new Float32Array([1, 2, 3, 4]), [4]);
    const out = broadcastTo(arr, [2, 3, 4]);
    expect(out.shape).toEqual([2, 3, 4]);
    expect(out.get(0, 0, 0)).toBe(1);
    expect(out.get(0, 0, 3)).toBe(4);
    expect(out.get(1, 2, 0)).toBe(1);
    expect(out.get(1, 2, 3)).toBe(4);
  });

  // 追加の失敗ケース
  it("broadcast (3,2) -> (2,3) fails - incompatible dimensions", () => {
    const arr = ndarray(new Float32Array([1,2,3,4,5,6]), [3, 2]);
    expect(() => broadcastTo(arr, [2, 3])).toThrow();
  });

  it("broadcast (2,3,4) -> (2,3) fails - cannot reduce dimensions", () => {
    const arr = ndarray(new Float32Array(24).fill(0).map((_, i) => i + 1), [2, 3, 4]);
    expect(() => broadcastTo(arr, [2, 3])).toThrow();
  });

  it("broadcast (5,) -> (3,4) fails - incompatible trailing dimension", () => {
    const arr = ndarray(new Float32Array([1, 2, 3, 4, 5]), [5]);
    expect(() => broadcastTo(arr, [3, 4])).toThrow();
  });
});

describe("fillValue", () => {
  // --- 成功するケース ---
  it("fillValue (2,2) -> (4,4)", () => {
    const arr = ndarray(new Float32Array([1,2,3,4]), [2,2]);
    const out = fillValue(arr, [4,4]);
    expect(out.shape).toEqual([4,4]);
    expect(out.get(0,0)).toBe(1);
    expect(out.get(1,1)).toBe(4);
    expect(out.get(3,3)).toBe(0); // padding
  });

  it("fillValue (1,3) -> (2,3)", () => {
    const arr = ndarray(new Uint8Array([10,20,30]), [1,3]);
    const out = fillValue(arr, [2,3]);
    expect(out.shape).toEqual([2,3]);
    expect(out.get(0,2)).toBe(30);
    expect(out.get(1,0)).toBe(0); // padded row
  });

  // --- エッジケース（成功）---
  it("fillValue identity (3,3) -> (3,3)", () => {
    const arr = ndarray(new Float32Array([1,2,3,4,5,6,7,8,9]), [3,3]);
    const out = fillValue(arr, [3,3]);
    expect(out.shape).toEqual([3,3]);
    expect(out.get(2,2)).toBe(9);
  });

  it("fillValue minimal expansion (2,1) -> (2,4)  // 2*1=2", () => {
    // 元配列は 2 行 1 列 = 2 要素
    const arr = ndarray(new Uint8Array([7, 8]), [2, 1]); // 2*1=2
    const out = fillValue(arr, [2, 4]);
    expect(out.shape).toEqual([2, 4]);

    // 元の値が左上に保存されているか（行, 列）
    expect(out.get(0, 0)).toBe(7);
    expect(out.get(1, 0)).toBe(8);

    // パディング領域が 0 で埋まっているか確認（列の末尾など）
    expect(out.get(0, 1)).toBe(0);
    expect(out.get(0, 3)).toBe(0);
    expect(out.get(1, 2)).toBe(0);
    expect(out.get(1, 3)).toBe(0);

    // 必要なら内部バッファ型の確認も（Uint8Array を使っていること）
    expect(out.data).toBeInstanceOf(Uint8Array);
  });

  it("fillValue RGB image expansion (2,2,3) -> (4,4,3)", () => {
    // 元は 2x2 の RGB = 2*2*3 = 12 要素
    const arr = ndarray(
      new Uint8Array([
        // row 0, col 0: RGB=(1,2,3)
        1, 2, 3,
        // row 0, col 1: RGB=(4,5,6)
        4, 5, 6,
        // row 1, col 0: RGB=(7,8,9)
        7, 8, 9,
        // row 1, col 1: RGB=(10,11,12)
        10, 11, 12,
      ]),
      [2, 2, 3]
    );

    const out = fillValue(arr, [4, 4, 3]);

    expect(out.shape).toEqual([4, 4, 3]);

    // 元のデータが左上にコピーされているか確認
    expect(out.get(0, 0, 0)).toBe(1);   // row0,col0,R
    expect(out.get(0, 0, 2)).toBe(3);   // row0,col0,B
    expect(out.get(1, 1, 1)).toBe(11);  // row1,col1,G

    // パディング領域がゼロか確認
    expect(out.get(0, 2, 0)).toBe(0); // row0,col2,R
    expect(out.get(3, 3, 2)).toBe(0); // row3,col3,B

    // 型が一致していること
    expect(out.data).toBeInstanceOf(Uint8Array);
  });

  // --- 失敗するケース ---
  it("fillValue (4,4) -> (2,4) fails", () => {
    const arr = ndarray(new Uint8Array(16).fill(1), [4,4]);
    expect(() => fillValue(arr, [2,4])).toThrow();
  });

  it("fillValue (2,3) -> (2,2) fails", () => {
    const arr = ndarray(new Uint8Array([1,2,3,4,5,6]), [2,3]);
    expect(() => fillValue(arr, [2,2])).toThrow();
  });
});

describe("fillValue with custom value", () => {
  it("fillValue (2,2) -> (4,4) with value=7", () => {
    const arr = ndarray(new Float32Array([1,2,3,4]), [2,2]);
    const out = fillValue(arr, [4,4], 7); // padding=7
    expect(out.shape).toEqual([4,4]);

    // 元の値はコピーされている
    expect(out.get(0,0)).toBe(1);
    expect(out.get(1,1)).toBe(4);

    // パディング領域が 7 で埋まっている
    expect(out.get(2,2)).toBe(7);
    expect(out.get(3,3)).toBe(7);
  });

  it("fillValue (1,3) -> (2,5) with value=255", () => {
    const arr = ndarray(new Uint8Array([10,20,30]), [1,3]);
    const out = fillValue(arr, [2,5], 255);

    expect(out.shape).toEqual([2,5]);

    // 元データがコピーされている
    expect(out.get(0,0)).toBe(10);
    expect(out.get(0,2)).toBe(30);

    // パディング部分が 255 で埋まっている
    expect(out.get(0,3)).toBe(255);
    expect(out.get(1,4)).toBe(255);
  });

  it("fillValue RGB image (2,2,3) -> (4,4,3) with value=128", () => {
    const arr = ndarray(
      new Uint8Array([
        1,2,3, 4,5,6,
        7,8,9, 10,11,12
      ]),
      [2,2,3]
    );

    const out = fillValue(arr, [4,4,3], 128);

    // 元データがコピーされている
    expect(out.get(0,0,0)).toBe(1);
    expect(out.get(1,1,2)).toBe(12);

    // パディング部分が 128
    expect(out.get(3,3,0)).toBe(128);
    expect(out.get(2,2,2)).toBe(128);
  });
});