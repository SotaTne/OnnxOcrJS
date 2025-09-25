import type { Box, Point } from "../types/type.js";
import {boxToLine, boxToMat, broadcastTo, clip, cloneNdArray, euclideanDistance, fillValue, matToLine, matToList, matToNdArray, matToPoints, ndArrayToList, pickAndSet, polygonArea, polygonPerimeter, unclip, type NdArrayListData} from "./func.js";
import { expect ,describe, it, beforeAll} from 'vitest'
import type cvReadyPromiseType from "@techstark/opencv-js";
import ndarray, { type NdArray } from "ndarray";
import ops from "ndarray-ops";

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

describe('matToNdArray', () => {
  // 成功ケース: 基本的な変換
  it('converts CV_8UC3 Mat to 3D NdArray', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // 2*2*3=12
    const mat = cv.matFromArray(2, 2, cv.CV_8UC3, data);
    
    const result = matToNdArray(mat, cv);
    
    expect(result.shape).toEqual([2, 2, 3]);
    expect(result.dtype).toBe('uint8');
    expect(result.get(0, 0, 0)).toBe(1);
    expect(result.get(1, 1, 2)).toBe(12);
    mat.delete();
  });

  // 成功ケース: skip_channel=trueでグレースケール画像
  it('converts CV_8UC1 Mat to 2D NdArray with skip_channel=true', () => {
    const data = [10, 20, 30, 40]; // 2*2*1=4
    const mat = cv.matFromArray(2, 2, cv.CV_8UC1, data);
    
    const result = matToNdArray(mat, cv, true);
    
    expect(result.shape).toEqual([2, 2]); // チャンネル次元がスキップされる
    expect(result.get(0, 0)).toBe(10);
    expect(result.get(1, 1)).toBe(40);
    mat.delete();
  });

  // 成功ケース: Float32データ型
  it('converts CV_32FC1 Mat to NdArray', () => {
    const data = [1.5, 2.7, 3.14]; // 3*1*1=3
    const mat = cv.matFromArray(3, 1, cv.CV_32FC1, data);
    
    const result = matToNdArray(mat, cv);
    
    expect(result.shape).toEqual([3, 1, 1]);
    expect(result.dtype).toBe('float32');
    expect(result.get(0, 0, 0)).toBeCloseTo(1.5);
    expect(result.get(2, 0, 0)).toBeCloseTo(3.14);
    mat.delete();
  });

  // 成功ケース: 16ビット整数データ
  it('converts CV_16UC2 Mat to NdArray', () => {
    const data = [1000, 2000, 3000, 4000]; // 2*1*2=4
    const mat = cv.matFromArray(2, 1, cv.CV_16UC2, data);
    
    const result = matToNdArray(mat, cv);
    
    expect(result.shape).toEqual([2, 1, 2]);
    expect(result.dtype).toBe('uint16');
    expect(result.get(0, 0, 1)).toBe(2000);
    expect(result.get(1, 0, 0)).toBe(3000);
    mat.delete();
  });

  // エッジケース: skip_channel=trueだが3チャンネル画像
  it('converts CV_8UC3 Mat to 3D NdArray even with skip_channel=true', () => {
    const data = [1, 2, 3, 4, 5, 6]; // 2*1*3=6
    const mat = cv.matFromArray(2, 1, cv.CV_8UC3, data);
    
    const result = matToNdArray(mat, cv, true);
    
    // チャンネル数が1でないため、skip_channelは無視される
    expect(result.shape).toEqual([2, 1, 3]);
    expect(result.get(0, 0, 0)).toBe(1);
    expect(result.get(1, 0, 2)).toBe(6);
    mat.delete();
  });

  // エッジケース: 1x1の最小サイズ画像
  it('converts 1x1 CV_32FC1 Mat to NdArray', () => {
    const data = [42.5]; // 1*1*1=1
    const mat = cv.matFromArray(1, 1, cv.CV_32FC1, data);
    
    const result = matToNdArray(mat, cv, true);
    
    expect(result.shape).toEqual([1, 1]); // skip_channel適用
    expect(result.get(0, 0)).toBeCloseTo(42.5);
    mat.delete();
  });

  // エッジケース: 空のMat入力
  it('handles empty Mat gracefully', () => {
    // 0x0のMat作成を試行
    const emptyMat = new cv.Mat();
    
    const result = matToNdArray(emptyMat, cv);
    
    expect(result.shape).toEqual([0, 0, 1]); // 空のMatは(0,0,1)として扱う
    expect(result.size).toBe(0);
    expect(result.data.length).toBe(0);
    expect(result.get(0, 0, 0)).toBeUndefined(); // データは存在しない
    
    emptyMat.delete();
  });

  it('returns empty NdArray without throwing for zero-sized Mat', () => {
    const emptyMat = new cv.Mat(0, 0, cv.CV_8UC1);
    let result: NdArray | null = null;
    const convert = () => {
      result = matToNdArray(emptyMat, cv);
    };
    try {
      expect(convert).not.toThrow();
      if (result === null) {
        throw new Error('matToNdArray did not return a value');
      }
      expect((result as NdArray).shape).toEqual([0, 0, 1]);
      expect((result as NdArray).size).toBe(0);
    } finally {
      emptyMat.delete();
    }
  });
});

describe("pickAndSet", () => {
  // --- 基本ケース ---
  it("1次元配列の特定要素をclipで変更できる", () => {
    const arr = ndarray(new Float32Array([1, 5, 10, 20]), [4]);

    pickAndSet(
      arr,
      (view) => {
        for (let i = 0; i < view.shape[0]!; i++) {
          view.set(i, Math.min(view.get(i), 8));
        }
        return view;
      },
      -1 // 全要素
    );

    expect(arr.data).toEqual(new Float32Array([1, 5, 8, 8]));
  });

  it("2次元配列の列をclipで変更できる (box[:,0] の代替)", () => {
    const arr = ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);

    pickAndSet(
      arr,
      (view) => {
        for (let i = 0; i < view.shape[0]!; i++) {
          view.set(i, Math.min(view.get(i), 2));
        }
        return view;
      },
      -1, 0 // column 0
    );

    expect(arr.data).toEqual(new Float32Array([1, 2, 2, 4]));
  });

  // --- 多次元 ---
  it("多次元配列 (RGB画像) のチャンネルだけ加工できる", () => {
    const arr = ndarray(
      new Uint8Array([
        10, 20, 30, // row0 col0 RGB
        40, 50, 60, // row0 col1 RGB
      ]),
      [1, 2, 3]
    );

    pickAndSet(
      arr,
      (view) => {
        // view は pick(-1,-1,0) で shape=[1,2] になっている（2次元）
        for (let i = 0; i < view.shape[0]!; i++) {
          for (let j = 0; j < view.shape[1]!; j++) {
            view.set(i, j, 0); // R チャンネルをゼロ化
          }
        }
        return view;
      },
      -1, -1, 0
    );

    expect(arr.get(0, 0, 0)).toBe(0);  // row0 col0 R
    expect(arr.get(0, 0, 1)).toBe(20); // G unchanged
    expect(arr.get(0, 1, 0)).toBe(0);  // row0 col1 R
    expect(arr.get(0, 1, 2)).toBe(60); // B unchanged
  });

  // --- エッジケース ---
  it("空配列に対してpickすると例外を投げる", () => {
    const arr = ndarray(new Float32Array([]), [0]);
    expect(() =>
      pickAndSet(arr, (v) => v, -1)
    ).toThrow();
  });

  it("範囲外インデックスを指定すると例外を投げる", () => {
    const arr = ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);
    expect(() =>
      pickAndSet(arr, (v) => v, 5) // 存在しないindex
    ).toThrow();
  });

  it("pickで全null指定 (コピー) の場合も反映される", () => {
    const arr = ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);

    pickAndSet(
      arr,
      (view) => {
        for (let i = 0; i < view.data.length; i++) {
          view.data[i]! *= 2;
        }
        return view;
      },
      -1, -1
    );

    expect(arr.data).toEqual(new Float32Array([2, 4, 6, 8]));
  });

  it("set関数が何もしなくても元配列は変更されない", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [3]);

    pickAndSet(arr, (view) => view, -1);

    expect(arr.data).toEqual(new Float32Array([1, 2, 3]));
  });

  it("set関数が新しい配列を返した場合、値がコピーされる", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [3]);

    pickAndSet(
      arr,
      (view) => {
        // 新しい配列を返す
        return ndarray(new Float32Array([99, 88, 77]), [3]);
      },
      -1
    );

    expect(arr.get(0)).toBe(99);
    expect(arr.get(1)).toBe(88);
    expect(arr.get(2)).toBe(77);
  });

  it("set関数が元のビューを変更して返した場合", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [3]);

    pickAndSet(
      arr,
      (view) => {
        view.set(0, 99);
        return view; // 同じビューを返す
      },
      -1
    );

    expect(arr.get(0)).toBe(99);
    expect(arr.get(1)).toBe(2);
    expect(arr.get(2)).toBe(3);
  });
});

describe("pickAndSet - Box座標スケーリング", () => {
  it("(N,2)のboxでx座標とy座標に異なるスケーリングを適用", () => {
    // 元画像サイズ: 800x600
    // 変換後サイズ: 400x300
    const originalWidth = 800;
    const originalHeight = 600;
    const destWidth = 400;
    const destHeight = 300;

    // バウンディングボックス座標 (x, y) のペア
    const box = ndarray(new Float32Array([
      100, 150,  // box 0: (100, 150)
      200, 300,  // box 1: (200, 300) 
      400, 450,  // box 2: (400, 450)
      600, 500   // box 3: (600, 500)
    ]), [4, 2]);

    // x座標 (列0) をスケーリング: x' = clip(round(x / 800 * 400), 0, 400)
    pickAndSet(box, (view) => {
      for (let i = 0; i < view.shape[0]!; i++) {
        const x = view.get(i);
        const scaledX = Math.round(x / originalWidth * destWidth);
        const clippedX = Math.max(0, Math.min(scaledX, destWidth));
        view.set(i, clippedX);
      }
      return view;
    }, -1, 0); // すべての行の0列目 (x座標)

    // y座標 (列1) をスケーリング: y' = clip(round(y / 600 * 300), 0, 300)
    pickAndSet(box, (view) => {
      for (let i = 0; i < view.shape[0]!; i++) {
        const y = view.get(i);
        const scaledY = Math.round(y / originalHeight * destHeight);
        const clippedY = Math.max(0, Math.min(scaledY, destHeight));
        view.set(i, clippedY);
      }
      return view;
    }, -1, 1); // すべての行の1列目 (y座標)

    // 期待値:
    // box[0]: (100, 150) -> (50, 75)   : 100*400/800=50, 150*300/600=75
    // box[1]: (200, 300) -> (100, 150) : 200*400/800=100, 300*300/600=150
    // box[2]: (400, 450) -> (200, 225) : 400*400/800=200, 450*300/600=225
    // box[3]: (600, 500) -> (300, 250) : 600*400/800=300, 500*300/600=250
    
    expect(box.get(0, 0)).toBe(50);   // x0
    expect(box.get(0, 1)).toBe(75);   // y0
    expect(box.get(1, 0)).toBe(100);  // x1
    expect(box.get(1, 1)).toBe(150);  // y1
    expect(box.get(2, 0)).toBe(200);  // x2
    expect(box.get(2, 1)).toBe(225);  // y2
    expect(box.get(3, 0)).toBe(300);  // x3
    expect(box.get(3, 1)).toBe(250);  // y3
  });

  it("範囲外座標がクリッピングされる", () => {
    const destWidth = 100;
    const destHeight = 100;

    // 範囲外の座標を含むボックス
    const box = ndarray(new Float32Array([
      -50, -30,   // 負の値
      150, 200,   // 範囲外の正の値
      50, 80      // 正常な値
    ]), [3, 2]);

    // x座標をクリッピング (0 <= x <= 100)
    pickAndSet(box, (view) => {
      for (let i = 0; i < view.shape[0]!; i++) {
        const x = view.get(i);
        view.set(i, Math.max(0, Math.min(x, destWidth)));
      }
      return view;
    }, -1, 0);

    // y座標をクリッピング (0 <= y <= 100)  
    pickAndSet(box, (view) => {
      for (let i = 0; i < view.shape[0]!; i++) {
        const y = view.get(i);
        view.set(i, Math.max(0, Math.min(y, destHeight)));
      }
      return view;
    }, -1, 1);

    expect(box.get(0, 0)).toBe(0);    // -50 -> 0
    expect(box.get(0, 1)).toBe(0);    // -30 -> 0
    expect(box.get(1, 0)).toBe(100);  // 150 -> 100
    expect(box.get(1, 1)).toBe(100);  // 200 -> 100
    expect(box.get(2, 0)).toBe(50);   // 50 -> 50 (変化なし)
    expect(box.get(2, 1)).toBe(80);   // 80 -> 80 (変化なし)
  });

  it("x座標のみ変更、y座標は保持", () => {
    const box = ndarray(new Float32Array([
      10, 20,
      30, 40,
      50, 60
    ]), [3, 2]);

    // x座標のみを2倍にする
    pickAndSet(box, (view) => {
      for (let i = 0; i < view.shape[0]!; i++) {
        view.set(i, view.get(i) * 2);
      }
      return view;
    }, -1, 0); // x座標のみ

    expect(box.get(0, 0)).toBe(20);  // 10 * 2
    expect(box.get(0, 1)).toBe(20);  // y座標は変化なし
    expect(box.get(1, 0)).toBe(60);  // 30 * 2  
    expect(box.get(1, 1)).toBe(40);  // y座標は変化なし
    expect(box.get(2, 0)).toBe(100); // 50 * 2
    expect(box.get(2, 1)).toBe(60);  // y座標は変化なし
  });

  it("実用的なOCR座標変換シミュレーション", () => {
    // 実際のOCRでよくある変換パターン
    const originalSize = { width: 1920, height: 1080 };
    const targetSize = { width: 640, height: 480 };
    
    // 検出されたテキストボックス座標
    const textBoxes = ndarray(new Float32Array([
      100, 200,   // テキスト1
      500, 300,   // テキスト2  
      1000, 600,  // テキスト3
      1500, 900   // テキスト4
    ]), [4, 2]);

    // x座標の変換とクリッピング
    pickAndSet(textBoxes, (view) => {
      for (let i = 0; i < view.shape[0]!; i++) {
        const x = view.get(i);
        const scaledX = Math.round(x * targetSize.width / originalSize.width);
        view.set(i, Math.max(0, Math.min(scaledX, targetSize.width)));
      }
      return view;
    }, -1, 0);

    // y座標の変換とクリッピング  
    pickAndSet(textBoxes, (view) => {
      for (let i = 0; i < view.shape[0]!; i++) {
        const y = view.get(i);
        const scaledY = Math.round(y * targetSize.height / originalSize.height);
        view.set(i, Math.max(0, Math.min(scaledY, targetSize.height)));
      }
      return view;
    }, -1, 1);

    // 変換結果の検証
    // 100 * 640/1920 ≈ 33, 200 * 480/1080 ≈ 89
    expect(textBoxes.get(0, 0)).toBe(33);
    expect(textBoxes.get(0, 1)).toBe(89);
    
    // 500 * 640/1920 ≈ 167, 300 * 480/1080 ≈ 133  
    expect(textBoxes.get(1, 0)).toBe(167);
    expect(textBoxes.get(1, 1)).toBe(133);
    
    // 1000 * 640/1920 ≈ 333, 600 * 480/1080 ≈ 267
    expect(textBoxes.get(2, 0)).toBe(333);
    expect(textBoxes.get(2, 1)).toBe(267);
    
    // 1500 * 640/1920 = 500, 900 * 480/1080 = 400
    expect(textBoxes.get(3, 0)).toBe(500);
    expect(textBoxes.get(3, 1)).toBe(400);
  });
});

describe('clip', () => {
  // 1次元配列のテスト
  it('clips 1D array [1,2,3,4,5,6,7,8,9,10] with min=3, max=7', () => {
    const src = ndarray(new Float32Array([1,2,3,4,5,6,7,8,9,10]), [10]);
    const dest = ndarray(new Float32Array(10), [10]);
    
    clip(dest, src, 3, 7);
    
    // 期待値: [3,3,3,4,5,6,7,7,7,7]
    expect(dest.data).toEqual(new Float32Array([3,3,3,4,5,6,7,7,7,7]));
  });

  it('clips 1D array with min=0, max=5', () => {
    const src = ndarray(new Float32Array([1,2,3,4,5,6,7,8,9,10]), [10]);
    const dest = ndarray(new Float32Array(10), [10]);
    
    clip(dest, src, 0, 5);
    
    // 期待値: [1,2,3,4,5,5,5,5,5,5]
    expect(dest.data).toEqual(new Float32Array([1,2,3,4,5,5,5,5,5,5]));
  });

  it('clips 1D array with negative range min=-2, max=8', () => {
    const src = ndarray(new Float32Array([-5,-1,0,1,2,3,4,5,6,15]), [10]);
    const dest = ndarray(new Float32Array(10), [10]);
    
    clip(dest, src, -2, 8);
    
    // 期待値: [-2,-1,0,1,2,3,4,5,6,8]
    expect(dest.data).toEqual(new Float32Array([-2,-1,0,1,2,3,4,5,6,8]));
  });

  // 2次元配列のテスト
  it('clips 2D array (2x3)', () => {
    const src = ndarray(new Float32Array([1,8,3,12,5,-2]), [2,3]);
    const dest = ndarray(new Float32Array(6), [2,3]);
    
    clip(dest, src, 2, 6);
    
    // 期待値: [2,6,3,6,5,2] (1->2, 8->6, 3->3, 12->6, 5->5, -2->2)
    expect(dest.data).toEqual(new Float32Array([2,6,3,6,5,2]));
  });

  // エッジケース
  it('handles case where all values are within range', () => {
    const src = ndarray(new Float32Array([3,4,5,6,7]), [5]);
    const dest = ndarray(new Float32Array(5), [5]);
    
    clip(dest, src, 2, 8);
    
    // すべて範囲内なので変化なし
    expect(dest.data).toEqual(new Float32Array([3,4,5,6,7]));
  });

  it('handles case where all values are below min', () => {
    const src = ndarray(new Float32Array([1,2,3,4,5]), [5]);
    const dest = ndarray(new Float32Array(5), [5]);
    
    clip(dest, src, 10, 20);
    
    // すべてminでクリップ
    expect(dest.data).toEqual(new Float32Array([10,10,10,10,10]));
  });

  it('handles case where all values are above max', () => {
    const src = ndarray(new Float32Array([15,20,25,30,35]), [5]);
    const dest = ndarray(new Float32Array(5), [5]);
    
    clip(dest, src, 5, 10);
    
    // すべてmaxでクリップ
    expect(dest.data).toEqual(new Float32Array([10,10,10,10,10]));
  });

  it('handles single element array', () => {
    const src = ndarray(new Float32Array([5]), [1]);
    const dest = ndarray(new Float32Array(1), [1]);
    
    clip(dest, src, 2, 8);
    
    expect(dest.data).toEqual(new Float32Array([5]));
  });

  it('clips with min=max (all values become the same)', () => {
    const src = ndarray(new Float32Array([1,5,10,15]), [4]);
    const dest = ndarray(new Float32Array(4), [4]);
    
    clip(dest, src, 7, 7);
    
    // すべて7になる
    expect(dest.data).toEqual(new Float32Array([7,7,7,7]));
  });

  // 異なるデータ型のテスト
  it('works with Int32Array', () => {
    const src = ndarray(new Int32Array([1,2,3,4,5,6,7,8,9,10]), [10]);
    const dest = ndarray(new Int32Array(10), [10]);
    
    clip(dest, src, 3, 7);
    
    expect(dest.data).toEqual(new Int32Array([3,3,3,4,5,6,7,7,7,7]));
  });

  it('works with Uint8Array', () => {
    const src = ndarray(new Uint8Array([0,50,100,150,200,255]), [6]);
    const dest = ndarray(new Uint8Array(6), [6]);
    
    clip(dest, src, 64, 192);
    
    // 期待値: [64,64,100,150,192,192]
    expect(dest.data).toEqual(new Uint8Array([64,64,100,150,192,192]));
  });

  // 失敗ケース
  it('throws error when min > max', () => {
    const src = ndarray(new Float32Array([1,2,3]), [3]);
    const dest = ndarray(new Float32Array(3), [3]);
    
    expect(() => {
      clip(dest, src, 10, 5); // min > max
    }).toThrow('min (10) must be <= max (5)');
  });

  it('handles floating point precision', () => {
    const src = ndarray(new Float32Array([1.1, 2.5, 3.7, 4.9, 5.2]), [5]);
    const dest = ndarray(new Float32Array(5), [5]);
    
    clip(dest, src, 2.0, 4.0);
    
    // 期待値: [2.0, 2.5, 3.7, 4.0, 4.0]
    expect(dest.get(0)).toBeCloseTo(2.0);
    expect(dest.get(1)).toBeCloseTo(2.5);
    expect(dest.get(2)).toBeCloseTo(3.7);
    expect(dest.get(3)).toBeCloseTo(4.0);
    expect(dest.get(4)).toBeCloseTo(4.0);
  });

  // 実用例：画像のピクセル値クリッピング
  it('clips image pixel values to valid range [0, 255]', () => {
    // 画像処理でよくある：計算結果が範囲外になったピクセル値をクリップ
    const pixelValues = ndarray(new Float32Array([
      -10, 50, 100, 300, 255, 500, 0, 128, -5, 260
    ]), [10]);
    const clippedPixels = ndarray(new Float32Array(10), [10]);
    
    clip(clippedPixels, pixelValues, 0, 255);
    
    // 期待値: [0, 50, 100, 255, 255, 255, 0, 128, 0, 255]
    expect(clippedPixels.data).toEqual(new Float32Array([
      0, 50, 100, 255, 255, 255, 0, 128, 0, 255
    ]));
  });

  // 実用例：座標のクリッピング
  it('clips bounding box coordinates to image bounds', () => {
    // バウンディングボックスの座標を画像範囲内にクリップ
    const coordinates = ndarray(new Float32Array([
      -20, 50, 800, 1200  // x1, y1, x2, y2
    ]), [4]);
    const clippedCoords = ndarray(new Float32Array(4), [4]);
    
    const imageWidth = 640;
    const imageHeight = 480;
    
    clip(clippedCoords, coordinates, 0, Math.max(imageWidth, imageHeight));
    
    // 期待値: [0, 50, 640, 640] (最大値で全てクリップ)
    expect(clippedCoords.data).toEqual(new Float32Array([0, 50, 640, 640]));
  });
});

describe('cloneNdArray', () => {
  // 基本的なクローンテスト
  it('creates a deep copy of Float32Array ndarray', () => {
    const original = ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const cloned = cloneNdArray(original);
    
    // 形状が同じ
    expect(cloned.shape).toEqual(original.shape);
    expect(cloned.stride).toEqual(original.stride);
    expect(cloned.offset).toBe(original.offset);
    
    // データが同じ
    expect(cloned.data).toEqual(original.data);
    
    // しかし、異なるメモリ領域
    expect(cloned.data).not.toBe(original.data);
    
    // データ型も同じ
    expect(cloned.dtype).toBe(original.dtype);
  });

  it('cloned array is independent from original', () => {
    const original = ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const cloned = cloneNdArray(original);
    
    // クローンを変更
    cloned.set(0, 0, 99);
    
    // 元の配列は変更されない
    expect(original.get(0, 0)).toBe(1);
    expect(cloned.get(0, 0)).toBe(99);
  });

  // 様々なデータ型のテスト
  it('clones Int32Array correctly', () => {
    const original = ndarray(new Int32Array([10, 20, 30]), [3]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.dtype).toBe('int32');
    expect(cloned.data).toBeInstanceOf(Int32Array);
    expect(cloned.data).toEqual(original.data);
    expect(cloned.data).not.toBe(original.data);
  });

  it('clones Uint8Array correctly', () => {
    const original = ndarray(new Uint8Array([100, 150, 200, 255]), [2, 2]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.dtype).toBe('uint8');
    expect(cloned.data).toBeInstanceOf(Uint8Array);
    expect(cloned.data).toEqual(original.data);
    expect(cloned.data).not.toBe(original.data);
  });

  it('clones Int16Array correctly', () => {
    const original = ndarray(new Int16Array([1000, 2000, 3000, 4000]), [2, 2]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.dtype).toBe('int16');
    expect(cloned.data).toBeInstanceOf(Int16Array);
    expect(cloned.data).toEqual(original.data);
    expect(cloned.data).not.toBe(original.data);
  });

  it('clones Float64Array correctly', () => {
    const original = ndarray(new Float64Array([1.5, 2.7, 3.14]), [3]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.dtype).toBe('float64');
    expect(cloned.data).toBeInstanceOf(Float64Array);
    expect(cloned.data).toEqual(original.data);
    expect(cloned.data).not.toBe(original.data);
  });

  // 複雑な形状のテスト
  it('clones 3D array correctly', () => {
    const original = ndarray(
      new Float32Array([1,2,3,4,5,6,7,8,9,10,11,12]), 
      [2, 2, 3]
    );
    const cloned = cloneNdArray(original);
    
    expect(cloned.shape).toEqual([2, 2, 3]);
    expect(cloned.get(0, 1, 2)).toBe(original.get(0, 1, 2));
    expect(cloned.get(1, 1, 2)).toBe(original.get(1, 1, 2));
    
    // 独立性確認
    cloned.set(1, 1, 2, 999);
    expect(original.get(1, 1, 2)).not.toBe(999);
  });

  it('clones 1D array correctly', () => {
    const original = ndarray(new Float32Array([10, 20, 30, 40, 50]), [5]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.shape).toEqual([5]);
    expect(cloned.get(2)).toBe(30);
    
    cloned.set(2, 999);
    expect(original.get(2)).toBe(30);
  });

  // ストライド付き配列のテスト
  it('preserves custom stride', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const original = ndarray(data, [2, 2], [3, 1], 1); // カスタムストライドとオフセット
    const cloned = cloneNdArray(original);
    
    expect(cloned.shape).toEqual(original.shape);
    expect(cloned.stride).toEqual(original.stride);
    expect(cloned.offset).toBe(original.offset);
    
    // 同じ要素にアクセス可能
    expect(cloned.get(0, 0)).toBe(original.get(0, 0));
    expect(cloned.get(1, 1)).toBe(original.get(1, 1));
  });

  // トランスポーズされた配列のテスト  
  it('clones transposed array correctly', () => {
    const original = ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const transposed = original.transpose(1, 0);
    const cloned = cloneNdArray(transposed);
    
    expect(cloned.shape).toEqual(transposed.shape);
    expect(cloned.stride).toEqual(transposed.stride);
    
    // 転置後の値が正しい
    expect(cloned.get(0, 0)).toBe(transposed.get(0, 0));
    expect(cloned.get(0, 1)).toBe(transposed.get(0, 1));
    expect(cloned.get(1, 0)).toBe(transposed.get(1, 0));
    expect(cloned.get(1, 1)).toBe(transposed.get(1, 1));
  });

  // スライスされた配列のテスト
  it('clones sliced array correctly', () => {
    const original = ndarray(new Float32Array([1,2,3,4,5,6,7,8,9]), [3, 3]);
    const sliced = original.hi(2, 2).lo(1, 1); // 中央1x1の領域
    const cloned = cloneNdArray(sliced);
    
    expect(cloned.shape).toEqual(sliced.shape);
    expect(cloned.stride).toEqual(sliced.stride);
    expect(cloned.offset).toBe(sliced.offset);
    
    // スライス後の値が正しい
    expect(cloned.get(0, 0)).toBe(sliced.get(0, 0));
  });

  // エッジケース
  it('clones empty array', () => {
    const original = ndarray(new Float32Array([]), [0]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.shape).toEqual([0]);
    expect(cloned.size).toBe(0);
    expect(cloned.data.length).toBe(0);
  });

  it('clones single element array', () => {
    const original = ndarray(new Float32Array([42]), [1]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.shape).toEqual([1]);
    expect(cloned.get(0)).toBe(42);
    
    cloned.set(0, 99);
    expect(original.get(0)).toBe(42);
  });

  // 型保持のテスト
  it('preserves TypeScript type information', () => {
    const original: NdArray<Float32Array> = ndarray(new Float32Array([1, 2, 3]), [3]);
    const cloned: NdArray<Float32Array> = cloneNdArray(original);
    
    // TypeScriptの型チェックが通ることを確認
    expect(cloned.data).toBeInstanceOf(Float32Array);
  });

  // number[]型のテスト
  it('clones regular number array correctly', () => {
    const original = ndarray([1, 2, 3, 4, 5, 6], [2, 3]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.dtype).toBe('array');
    expect(cloned.data).toBeInstanceOf(Array);
    expect(cloned.shape).toEqual([2, 3]);
    expect(cloned.data).toEqual([1, 2, 3, 4, 5, 6]);
    expect(cloned.data).not.toBe(original.data);
    
    // 独立性確認
    cloned.set(0, 0, 99);
    expect(original.get(0, 0)).toBe(1);
    expect(cloned.get(0, 0)).toBe(99);
  });

  // メモリ共有の完全な独立性テスト
  it('ensures complete memory independence after multiple operations', () => {
    const original = ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const cloned = cloneNdArray(original);
    
    // 複数の操作を実行
    cloned.set(0, 0, 100);
    cloned.set(1, 2, 200);
    
    // 元の配列のデータが変更されていないことを確認
    expect(original.get(0, 0)).toBe(1);
    expect(original.get(1, 2)).toBe(6);
    expect(cloned.get(0, 0)).toBe(100);
    expect(cloned.get(1, 2)).toBe(200);
    
    // データ配列自体も異なることを確認
    expect(original.data === cloned.data).toBe(false);
    expect(original.data[0]).toBe(1);
    expect(cloned.data[0]).toBe(100);
  });

  it('memory independence with ndarray-ops operations', () => {
    const original = ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const cloned = cloneNdArray(original);
    
    // ndarray-opsを使った操作
    ops.mulseq(cloned, 10); // clonedを10倍
    ops.addseq(cloned, 5);  // clonedに5を加算
    
    // 元の配列は変更されていない
    expect(original.get(0, 0)).toBe(1);
    expect(original.get(1, 1)).toBe(4);
    
    // クローンは変更されている
    expect(cloned.get(0, 0)).toBe(15); // (1 * 10) + 5
    expect(cloned.get(1, 1)).toBe(45); // (4 * 10) + 5
    
    // 内部データも独立している
    expect(original.data[0]).toBe(1);
    expect(cloned.data[0]).toBe(15);
  });

  it('memory independence with complex data manipulation', () => {
    const original = ndarray([10, 20, 30, 40, 50, 60], [2, 3]);
    const cloned = cloneNdArray(original);
    
    // 複雑なデータ操作
    for (let i = 0; i < cloned.shape[0]!; i++) {
      for (let j = 0; j < cloned.shape[1]!; j++) {
        const val = cloned.get(i, j);
        cloned.set(i, j, val * 2 + 1);
      }
    }
    
    // 元のデータは変更されていない
    expect(original.data).toEqual([10, 20, 30, 40, 50, 60]);
    
    // クローンは変更されている
    expect(cloned.data).toEqual([21, 41, 61, 81, 101, 121]);
    
    // メモリアドレスが異なることを確認
    expect(original.data).not.toBe(cloned.data);
  });

  // 実用例：画像データのクローン
  it('clones RGB image data correctly', () => {
    // 2x2のRGB画像 (2*2*3 = 12要素)
    const imageData = new Uint8Array([
      255, 0, 0,    // 赤
      0, 255, 0,    // 緑
      0, 0, 255,    // 青  
      255, 255, 0   // 黄
    ]);
    const original = ndarray(imageData, [2, 2, 3]);
    const cloned = cloneNdArray(original);
    
    expect(cloned.shape).toEqual([2, 2, 3]);
    expect(cloned.get(0, 0, 0)).toBe(255); // 赤チャンネル
    expect(cloned.get(0, 1, 1)).toBe(255); // 緑チャンネル
    
    // 独立性確認
    cloned.set(0, 0, 0, 128);
    expect(original.get(0, 0, 0)).toBe(255);
  });
});

// ndarrayからネスト配列へ変換するケースを網羅
describe('ndArrayToList - Dimensional Tests', () => {
  // 0次元配列（スカラー）
  it('converts 0D array (scalar) to list', () => {
    const scalar = ndarray(new Float32Array([42]), []);
    const result = ndArrayToList(scalar);
    
    expect(result).toEqual([42]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(1);
  });

  it('converts 0D array with different data types', () => {
    const intScalar = ndarray(new Int32Array([100]), []);
    const floatScalar = ndarray(new Float64Array([3.14]), []);
    
    expect(ndArrayToList(intScalar)).toEqual([100]);
    expect(ndArrayToList(floatScalar)).toEqual([3.14]);
  });

  // 1次元配列
  it('converts 1D array to list', () => {
    const arr1d = ndarray(new Float32Array([1, 2, 3, 4, 5]), [5]);
    const result = ndArrayToList(arr1d);
    
    expect(result).toEqual([1, 2, 3, 4, 5]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(5);
  });

  it('converts 1D empty array', () => {
    const empty1d = ndarray(new Float32Array([]), [0]);
    const result = ndArrayToList(empty1d);
    
    expect(result).toEqual([]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(0);
  });

  it('converts 1D single element array', () => {
    const single1d = ndarray(new Float32Array([99]), [1]);
    const result = ndArrayToList(single1d);
    
    expect(result).toEqual([99]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(1);
  });

  // 2次元配列
  it('converts 2D array to nested list', () => {
    const arr2d = ndarray(new Float32Array([
      1, 2, 3,
      4, 5, 6
    ]), [2, 3]);
    const result = ndArrayToList(arr2d);
    
    expect(result).toEqual([
      [1, 2, 3],
      [4, 5, 6]
    ]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(2);
    expect(Array.isArray(result[0])).toBe(true);
    expect((result[0] as number[]).length).toBe(3);
  });

  it('converts 2D square matrix', () => {
    const matrix = ndarray(new Float32Array([
      1, 2,
      3, 4
    ]), [2, 2]);
    const result = ndArrayToList(matrix);
    
    expect(result).toEqual([
      [1, 2],
      [3, 4]
    ]);
  });

  it('converts 2D single row matrix', () => {
    const rowMatrix = ndarray(new Float32Array([10, 20, 30]), [1, 3]);
    const result = ndArrayToList(rowMatrix);
    
    expect(result).toEqual([[10, 20, 30]]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(1);
  });

  it('converts 2D single column matrix', () => {
    const colMatrix = ndarray(new Float32Array([10, 20, 30]), [3, 1]);
    const result = ndArrayToList(colMatrix);
    
    expect(result).toEqual([[10], [20], [30]]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(3);
  });

  // 3次元配列
  it('converts 3D array to triple-nested list', () => {
    const arr3d = ndarray(new Float32Array([
      1, 2, 3, 4,  // [0, :, :]
      5, 6, 7, 8   // [1, :, :]
    ]), [2, 2, 2]);
    const result = ndArrayToList(arr3d);
    
    expect(result).toEqual([
      [[1, 2], [3, 4]],  // [0, :, :]
      [[5, 6], [7, 8]]   // [1, :, :]
    ]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(2);
  });

  it('converts 3D RGB image-like array', () => {
    // 2x2 RGB画像 (height=2, width=2, channels=3)
    const rgbImage = ndarray(new Uint8Array([
      255, 0, 0,    // pixel[0,0] RGB
      0, 255, 0,    // pixel[0,1] RGB
      0, 0, 255,    // pixel[1,0] RGB
      255, 255, 0   // pixel[1,1] RGB
    ]), [2, 2, 3]);
    const result = ndArrayToList(rgbImage);
    
    expect(result).toEqual([
      [[255, 0, 0], [0, 255, 0]],      // row 0
      [[0, 0, 255], [255, 255, 0]]     // row 1
    ]);
  });

  it('converts 3D single element per dimension', () => {
    const arr3d = ndarray(new Float32Array([42]), [1, 1, 1]);
    const result = ndArrayToList(arr3d);
    
    expect(result).toEqual([[[42]]]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(1);
  });

  // ビュー（部分領域）をリストへ正しく変換できるか
  it('converts 2D ndarray views without losing data', () => {
    const data = new Float32Array(9);
    for (let i = 0; i < data.length; i++) {
      data[i] = i;
    }
    const base = ndarray(data, [3, 3]);
    const view = base.lo(1, 1).hi(2, 2);

    const result = ndArrayToList(view);

    expect(result).toEqual([
      [4, 5],
      [7, 8],
    ]);
  });

  // 3チャンネル画像ビューをリストへ変換できるか
  it('converts 3-channel ndarray views correctly', () => {
    const data = new Uint8Array([
      // row 0
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12,
      // row 1
      13, 14, 15,
      16, 17, 18,
      19, 20, 21,
      22, 23, 24,
    ]);
    const base = ndarray(data, [2, 4, 3]);
    const view = base.lo(1, 1, 0).hi(1, 2, 3);

    const result = ndArrayToList(view);

    expect(result).toEqual([
      [
        [16, 17, 18],
        [19, 20, 21],
      ],
    ]);
  });

  // 転置ビューでも期待どおり変換できるか
  it('converts transposed ndarray views', () => {
    const data = new Float32Array([
      1, 2, 3,
      4, 5, 6,
    ]);
    const base = ndarray(data, [2, 3]);
    const view = base.transpose(1, 0);

    const result = ndArrayToList(view);

    expect(result).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  // 4次元配列
  it('converts 4D array to quadruple-nested list', () => {
    // バッチサイズ2, 高さ2, 幅2, チャンネル1のテンソル
    const arr4d = ndarray(new Float32Array([
      1, 2, 3, 4,  // batch 0
      5, 6, 7, 8   // batch 1
    ]), [2, 2, 2, 1]);
    const result = ndArrayToList(arr4d);
    
    expect(result).toEqual([
      [[[1], [2]], [[3], [4]]],  // batch 0
      [[[5], [6]], [[7], [8]]]   // batch 1
    ]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(2);
  });

  it('converts 4D tensor with multiple channels', () => {
    // 1x1x2x2 テンソル（バッチ1, 高さ1, 幅2, チャンネル2）
    const tensor = ndarray(new Float32Array([
      1, 2, 3, 4  // [batch0, height0, [width0_ch0, width0_ch1], [width1_ch0, width1_ch1]]
    ]), [1, 1, 2, 2]);
    const result = ndArrayToList(tensor);
    
    expect(result).toEqual([
      [[[1, 2], [3, 4]]]
    ]);
  });

  // 5次元配列
  it('converts 5D array to quintuple-nested list', () => {
    // 非常に小さな5次元配列
    const arr5d = ndarray(new Float32Array([
      1, 2, 3, 4, 5, 6, 7, 8
    ]), [2, 2, 2, 1, 1]);
    const result = ndArrayToList(arr5d);
    
    expect(result).toEqual([
      [[[[1]], [[2]]], [[[3]], [[4]]]],  // first major group
      [[[[5]], [[6]]], [[[7]], [[8]]]]   // second major group
    ]);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(2);
  });

  it('converts 5D minimal array', () => {
    const minimal5d = ndarray(new Float32Array([100]), [1, 1, 1, 1, 1]);
    const result = ndArrayToList(minimal5d);
    
    expect(result).toEqual([[[[[100]]]]]);
    expect(Array.isArray(result)).toBe(true);
  });

  // エッジケースと型検証
  it('maintains number precision with different data types', () => {
    const floatArr = ndarray(new Float64Array([1.234567, 2.345678]), [2]);
    const intArr = ndarray(new Int32Array([-100, 0, 100]), [3]);
    
    expect(ndArrayToList(floatArr)).toEqual([1.234567, 2.345678]);
    expect(ndArrayToList(intArr)).toEqual([-100, 0, 100]);
  });

  it('works with regular number arrays', () => {
    const regularArray = ndarray([10, 20, 30, 40], [2, 2]);
    const result = ndArrayToList(regularArray);
    
    expect(result).toEqual([[10, 20], [30, 40]]);
  });

  it('handles negative numbers correctly', () => {
    const negativeArr = ndarray(new Float32Array([
      -1, -2,
      -3, -4
    ]), [2, 2]);
    const result = ndArrayToList(negativeArr);
    
    expect(result).toEqual([[-1, -2], [-3, -4]]);
  });

  // パフォーマンステスト（小規模）
  it('handles moderately sized arrays efficiently', () => {
    // 10x10の2次元配列
    const largeData = new Float32Array(100);
    for (let i = 0; i < 100; i++) {
      largeData[i] = i;
    }
    const largeArr = ndarray(largeData, [10, 10]);
    
    const result = ndArrayToList(largeArr);
    
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(10);
    expect((result[0] as number[]).length).toBe(10);
    expect((result[0] as number[])[0]).toBe(0);
    expect((result[9] as number[])[9]).toBe(99);
  });

  // 型安全性テスト
  it('returns consistent nested array structure', () => {
    const arr3d = ndarray(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
    const result = ndArrayToList(arr3d);
    
    // TypeScriptの型チェックが通ることを確認
    expect(typeof result).toBe('object');
    expect(Array.isArray(result)).toBe(true);
    
    // ネストレベルの確認
    const level1 = result;
    expect(Array.isArray(level1[0])).toBe(true);

    const level2 = level1[0] as NdArrayListData;
    expect(Array.isArray(level2[0])).toBe(true);
    
    const level3 = level2[0] as number[];
    expect(typeof level3[0]).toBe('number');
  });
});

describe("polygonArea", () => {
  it("computes area of a square", () => {
    const square = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ];
    expect(polygonArea(square)).toBe(100);
  });

  it("computes area of a triangle", () => {
    const tri = [
      { x: 0, y: 0 },
      { x: 4, y: 0 },
      { x: 0, y: 3 },
    ];
    expect(polygonArea(tri)).toBe(6);
  });

  it("returns 0 for a line-like polygon", () => {
    const line = [
      { x: 0, y: 0 },
      { x: 5, y: 0 },
      { x: 10, y: 0 },
    ];
    expect(polygonArea(line)).toBe(0);
  });

  it("returns 0 for a single point", () => {
    expect(polygonArea([{ x: 1, y: 1 }])).toBe(0);
  });

  it("returns 0 for two-point polygon", () => {
    const line2 = [
      { x: 0, y: 0 },
      { x: 3, y: 4 },
    ];
    expect(polygonArea(line2)).toBe(0);
  });
});

describe("polygonPerimeter", () => {
  it("computes perimeter of a square", () => {
    const square = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ];
    expect(polygonPerimeter(square)).toBeCloseTo(40);
  });

  it("computes perimeter of a triangle", () => {
    const tri = [
      { x: 0, y: 0 },
      { x: 3, y: 0 },
      { x: 0, y: 4 },
    ];
    expect(polygonPerimeter(tri)).toBeCloseTo(12);
  });

  it("returns 0 for a single point", () => {
    expect(polygonPerimeter([{ x: 1, y: 1 }])).toBe(0);
  });

  it("returns double the distance for two points (closed loop)", () => {
    const line2 = [
      { x: 0, y: 0 },
      { x: 3, y: 4 },
    ];
    // distance = 5, closed loopなので往復=10
    expect(polygonPerimeter(line2)).toBeCloseTo(10);
  });
});

describe("unclip", () => {
  it("expands a square polygon outward", async () => {
    const square = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ];

    const expanded = await unclip(square, 2.0);
    expect(Array.isArray(expanded)).toBe(true);
    expect(expanded!.length).toBeGreaterThan(0);

    const expandedArea = polygonArea(expanded![0]!);
    expect(expandedArea).toBeGreaterThan(polygonArea(square));
  });

  it("shrinks polygon if unclipRatio is negative", async () => {
    const rect = [
      { x: 0, y: 0 },
      { x: 20, y: 0 },
      { x: 20, y: 10 },
      { x: 0, y: 10 },
    ];
    const shrunk = await unclip(rect, -0.5);
    expect(Array.isArray(shrunk)).toBe(true);

    const shrunkArea = polygonArea(shrunk![0]!);
    expect(shrunkArea).toBeLessThan(polygonArea(rect));
  });

  it("returns undefined or empty array if delta is too small", async () => {
    const tiny = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
    const result = await unclip(tiny, 0.0);
    expect(result.length === 0).toBe(true);
  });

  it("handles negative unclipRatio with small polygons", async () => {
    const tri = [
      { x: 0, y: 0 },
      { x: 3, y: 0 },
      { x: 0, y: 4 },
    ];
    const shrunk = await unclip(tri, -1.0);
    expect(Array.isArray(shrunk)).toBe(true);
    if (shrunk && shrunk.length > 0) {
      expect(polygonArea(shrunk[0]!)).toBeLessThan(polygonArea(tri));
    }
  });

  it("works with larger complex polygon", async () => {
    const hexagon = [
      { x: 0, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 2 },
      { x: 2, y: 4 },
      { x: 0, y: 4 },
      { x: -1, y: 2 },
    ];
    const expanded = await unclip(hexagon, 1.5);
    expect(Array.isArray(expanded)).toBe(true);
    expect(expanded!.length).toBeGreaterThan(0);
  });

  // ★ 追加: 2点だけの線
  it("returns undefined or empty for two-point polygon", async () => {
    const line2 = [
      { x: 0, y: 0 },
      { x: 5, y: 0 },
    ];
    const result = await unclip(line2, 1.0);
    expect(result === undefined || result.length === 0).toBe(true);
  });
});
