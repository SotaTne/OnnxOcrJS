import ndarray from "ndarray";
import { beforeAll, describe, expect, it } from "vitest";
import { ndArrayToMat } from "./ndarray-to-mat.js";
import type { Mat } from "@techstark/opencv-js";
import type cvReadyPromiseType from "@techstark/opencv-js";

let cv: Awaited<typeof cvReadyPromiseType>;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  cv = await cvReadyPromise;
});

describe("ndArrayToMat", () => {
  // --- 正常系 ---
  it("converts 1D float array to single-column Mat", () => {
    const arr = ndarray(new Float32Array([1.1, 2.2, 3.3]), [3]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(3);
    expect(mat.cols).toBe(1);
    expect(mat.channels()).toBe(1);
    expect(mat.data32F[0]).toBeCloseTo(1.1);
    expect(mat.data32F[2]).toBeCloseTo(3.3);

    mat.delete();
  });

  it("converts 2D int array to Mat", () => {
    const arr = ndarray(new Int32Array([1, 2, 3, 4]), [2, 2]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(2);
    expect(mat.cols).toBe(2);
    expect(mat.channels()).toBe(1);
    expect(mat.data32S[0]).toBe(1);
    expect(mat.data32S[3]).toBe(4);

    mat.delete();
  });

  it("converts 3D uint8 array (RGB image-like) to Mat", () => {
    const arr = ndarray(
      new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]),
      [2, 2, 3],
    );

    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(2);
    expect(mat.cols).toBe(2);
    expect(mat.channels()).toBe(3);
    expect(mat.type()).toBe(cv.CV_8UC3);

    expect(mat.ucharPtr(0, 0)).toEqual(new Uint8Array([255, 0, 0]));
    expect(mat.ucharPtr(1, 1)).toEqual(new Uint8Array([255, 255, 0]));

    mat.delete();
  });

  it("accepts plain number arrays", () => {
    const arr = ndarray(
      [
        [1, 2],
        [3, 4],
      ],
      [2, 2],
    );
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(2);
    expect(mat.cols).toBe(2);
    expect(mat.channels()).toBe(1);
    expect(mat.data32S[0]).toBe(1);
    expect(mat.data32S[3]).toBe(4);

    mat.delete();
  });

  it("detects float precision from number arrays", () => {
    const arr = ndarray([[1.1, 2.2]], [1, 2]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.type()).toBe(cv.CV_64FC1);
    expect(mat.data64F[0]).toBeCloseTo(1.1);

    mat.delete();
  });

  // Arrayベースのビューでも部分領域だけをMat化できること
  it("converts array-backed ndarray views without reading full buffer", () => {
    const base = ndarray(
      [
        [1, 2],
        [3, 4],
      ],
      [2, 2],
    );
    const view = base.lo(1, 0); // [[3,4]]

    const mat = ndArrayToMat(view, cv);

    expect(mat.rows).toBe(1);
    expect(mat.cols).toBe(2);
    expect(mat.channels()).toBe(1);
    expect(mat.data32S[0]).toBe(3);
    expect(mat.data32S[1]).toBe(4);

    mat.delete();
  });

  // --- エッジケース ---
  it("supports 2 channels", () => {
    const arr = ndarray(new Uint16Array([100, 200, 300, 400]), [2, 1, 2]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(2);
    expect(mat.cols).toBe(1);
    expect(mat.channels()).toBe(2);
    expect(mat.type()).toBe(cv.CV_16UC2);

    expect(mat.ushortPtr(0, 0)).toEqual(new Uint16Array([100, 200]));
    expect(mat.ushortPtr(1, 0)).toEqual(new Uint16Array([300, 400]));

    mat.delete();
  });

  it("supports 4 channels", () => {
    const arr = ndarray(new Uint8Array([10, 20, 30, 40]), [1, 1, 4]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(1);
    expect(mat.cols).toBe(1);
    expect(mat.channels()).toBe(4);
    expect(mat.type()).toBe(cv.CV_8UC4);

    expect(mat.ucharPtr(0, 0)).toEqual(new Uint8Array([10, 20, 30, 40]));

    mat.delete();
  });

  // --- エラーケース ---
  it("throws for 0D array", () => {
    const arr = ndarray(new Float32Array([42]), []);
    expect(() => ndArrayToMat(arr, cv)).toThrow(/0D arrays are not supported/);
  });

  it("throws for >3D array", () => {
    const arr = ndarray(new Float32Array([1, 2]), [1, 1, 1, 2]);
    expect(() => ndArrayToMat(arr, cv)).toThrow(
      /only 1D, 2D, 3D arrays are supported/,
    );
  });

  it("throws for unsupported channels", () => {
    const arr = ndarray(new Float32Array([1, 2, 3, 4, 5]), [1, 1, 5]);
    expect(() => ndArrayToMat(arr, cv)).toThrow(
      /only 1,2,3,4 channels are supported/,
    );
  });

  it("throws for length mismatch in typed array", () => {
    const arr = ndarray(new Float32Array([1, 2, 3]), [2, 2]);
    expect(() => ndArrayToMat(arr, cv)).toThrow(/data length mismatch/);
  });

  it("throws for length mismatch in number array", () => {
    const arr = ndarray([[1, 2, 3]], [1, 2]); // shapeと長さ不一致
    expect(() => ndArrayToMat(arr, cv)).toThrow(/data length mismatch/);
  });

  it("throws for unsupported data container", () => {
    const arr = { shape: [2, 2], data: new Set([1, 2, 3, 4]) } as any;
    expect(() => ndArrayToMat(arr, cv)).toThrow(/unsupported data container/);
  });
});

// ndarrayからMatへのエッジケース処理を検証
describe("ndArrayToMat - Edge Cases", () => {
  // --- 空配列 ---
  it("returns empty Mat for empty 1D array (len=0)", () => {
    const arr = ndarray(new Float32Array([]), [0]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(0);
    expect(mat.cols).toBe(1); // shape[0]=0, cols=1扱い
    expect(mat.empty()).toBe(true);

    mat.delete();
  });

  it("returns empty Mat for empty 2D array (0x0)", () => {
    const arr = ndarray(new Float32Array([]), [0, 0]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(0);
    expect(mat.cols).toBe(0);
    expect(mat.empty()).toBe(true);

    mat.delete();
  });

  // --- 最小サイズ ---
  it("converts 1x1 scalar matrix (int)", () => {
    const arr = ndarray(new Int32Array([123]), [1, 1]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(1);
    expect(mat.cols).toBe(1);
    expect(mat.channels()).toBe(1);
    expect(mat.data32S[0]).toBe(123);

    mat.delete();
  });

  it("converts 1x1x3 RGB pixel", () => {
    const arr = ndarray(new Uint8Array([10, 20, 30]), [1, 1, 3]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.rows).toBe(1);
    expect(mat.cols).toBe(1);
    expect(mat.channels()).toBe(3);
    expect(mat.ucharPtr(0, 0)).toEqual(new Uint8Array([10, 20, 30]));

    mat.delete();
  });

  // --- 境界値 ---
  it("handles negative numbers in int arrays", () => {
    const arr = ndarray(new Int16Array([-32768, 0, 32767]), [3, 1]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.type()).toBe(cv.CV_16SC1);
    expect(mat.shortAt(0, 0)).toBe(-32768);
    expect(mat.shortAt(2, 0)).toBe(32767);

    mat.delete();
  });

  it("handles large float values", () => {
    const arr = ndarray(new Float64Array([1e10, -1e10]), [2, 1]);
    const mat = ndArrayToMat(arr, cv);

    expect(mat.type()).toBe(cv.CV_64FC1);
    expect(mat.doubleAt(0, 0)).toBeCloseTo(1e10);
    expect(mat.doubleAt(1, 0)).toBeCloseTo(-1e10);

    mat.delete();
  });

  // --- 型判定 ---
  it("distinguishes between int and float in number arrays", () => {
    const intArr = ndarray([[1, 2]], [1, 2]);
    const floatArr = ndarray([[1.1, 2.2]], [1, 2]);

    const intMat = ndArrayToMat(intArr, cv);
    const floatMat = ndArrayToMat(floatArr, cv);

    expect(intMat.type()).toBe(cv.CV_32SC1);
    expect(floatMat.type()).toBe(cv.CV_64FC1);

    intMat.delete();
    floatMat.delete();
  });

  // --- サポート外ケース ---
  it("throws for channels > 4", () => {
    const arr = ndarray(new Uint8Array([1, 2, 3, 4, 5]), [1, 1, 5]);
    expect(() => ndArrayToMat(arr, cv)).toThrow(
      /only 1,2,3,4 channels are supported/,
    );
  });

  it("throws for unsupported container (Set)", () => {
    const arr = { shape: [2, 2], data: new Set([1, 2, 3, 4]) } as any;
    expect(() => ndArrayToMat(arr, cv)).toThrow(/unsupported data container/);
  });
});

// ndarrayからMatへの変換パスを網羅的に検証
// Matからndarrayへの相互変換ロジックを検証
describe("ndArrayToMat/view", () => {
  // ndarrayの部分ビューを連続メモリに変換できるか
  it("converts ndarray views without using the original buffer head", () => {
    const data = new Float32Array(9);
    for (let i = 0; i < data.length; i++) {
      data[i] = i;
    }
    const base = ndarray(data, [3, 3]);
    const view = base.lo(1, 1).hi(2, 2);

    let createdMat: Mat | null = null;
    const createMat = () => {
      createdMat = ndArrayToMat(view, cv);
    };

    try {
      expect(createMat).not.toThrow();
      if (createdMat === null) {
        throw new Error("ndArrayToMat did not produce a Mat instance");
      }
      expect(Array.from((createdMat as Mat).data32F)).toEqual([4, 5, 7, 8]);
    } finally {
      (createdMat as null | Mat)?.delete();
    }
  });

  // 3チャンネルビューを正しくMat化できるか
  it("converts 3-channel ndarray views without falling back to base buffer", () => {
    const data = new Uint8Array([
      // row 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      // row 1
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    ]);
    const base = ndarray(data, [2, 4, 3]); // (rows, cols, channels)
    const view = base.lo(1, 1, 0).hi(1, 2, 3); // 1x2 region, all channels

    let createdMat: Mat | null = null;
    const createMat = () => {
      createdMat = ndArrayToMat(view, cv);
    };

    try {
      expect(createMat).not.toThrow();
      if (createdMat === null) {
        throw new Error("ndArrayToMat did not produce a Mat instance");
      }
      expect(Array.from((createdMat as Mat).data)).toEqual([
        16, 17, 18, 19, 20, 21,
      ]);
    } finally {
      (createdMat as null | Mat)?.delete();
    }
  });

  // 転置ビューをMatへ変換できるか
  it("converts transposed ndarray views", () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const base = ndarray(data, [2, 3]);
    const view = base.transpose(1, 0); // shape [3,2], non-contiguous stride

    let createdMat: Mat | null = null;
    const createMat = () => {
      createdMat = ndArrayToMat(view, cv);
    };

    try {
      expect(createMat).not.toThrow();
      if (createdMat === null) {
        throw new Error("ndArrayToMat did not produce a Mat instance");
      }
      expect(Array.from((createdMat as Mat).data32F)).toEqual([
        1, 4, 2, 5, 3, 6,
      ]);
    } finally {
      (createdMat as null | Mat)?.delete();
    }
  });
});
