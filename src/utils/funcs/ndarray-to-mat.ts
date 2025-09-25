
import type { NdArray } from "ndarray";
import type { CV2 } from "../../types/type.js";
import type ndarray from "ndarray";
import type { Mat } from "@techstark/opencv-js";

type CvDepth = "8U" | "8S" | "16U" | "16S" | "32S" | "32F" | "64F";

function inferDepthFromTypedArray(buf: ArrayBufferView): CvDepth {
  const C = buf.constructor;
  if (C === Uint8Array || C === Uint8ClampedArray) return "8U";
  if (C === Int8Array) return "8S";
  if (C === Uint16Array) return "16U";
  if (C === Int16Array) return "16S";
  if (C === Int32Array || C === Uint32Array) return "32S";
  if (C === Float32Array) return "32F";
  if (C === Float64Array) return "64F";
  throw new Error("ndArrayToMat: unsupported typed array");
}

function getCvType(cv: CV2, depth: CvDepth, ch: number): number {
  const key = `CV_${depth}C${ch}` as keyof CV2;
  const t = cv[key];
  if (typeof t !== "number") {
    throw new Error(`ndArrayToMat: unsupported cv type ${String(key)}`);
  }
  return t as unknown as number;
}

function calcShapeMeta(arr: NdArray<any>): {
  rank: number;
  rows: number;
  cols: number;
  channels: number;
  expectedLen: number;
} {
  const rank = arr.shape.length;
  if (rank === 0) throw new Error("ndArrayToMat: 0D arrays are not supported");
  if (rank > 3) throw new Error("ndArrayToMat: only 1D, 2D, 3D arrays are supported");

  const rows = arr.shape[0]!;
  const cols = rank === 1 ? 1 : arr.shape[1]!;
  const channels = rank === 3 ? arr.shape[2]! : 1;

  if (channels < 1 || channels > 4) {
    throw new Error("ndArrayToMat: only 1,2,3,4 channels are supported");
  }

  const expectedLen = rows * cols * channels;
  return { rank, rows, cols, channels, expectedLen };
}

function harvestTypedContiguous<T extends ndarray.Data>(
  arr: NdArray<T>,
  meta: { rank: number; rows: number; cols: number; channels: number; expectedLen: number },
  typed: ArrayBufferView
): ArrayBufferView {
  const { rank, rows, cols, channels, expectedLen } = meta;
  const contiguous = new ((typed.constructor as { new(length: number): ArrayBufferView }))(expectedLen);
  const stride0 = arr.stride[0] ?? (rank > 1 ? cols : 1);
  const stride1 = arr.stride[1] ?? (rank === 3 ? channels : 1);
  const stride2 = arr.stride[2] ?? 1;
  let out = 0;

  const src = typed as any;
  const ensure = (idx: number) => {
    if (idx < 0 || idx >= src.length) {
      throw new Error("ndArrayToMat: data length mismatch (access out of bounds)");
    }
  };

  if (rank === 1) {
    for (let i = 0; i < rows; i++) {
      const idx = arr.offset + i * stride0;
      ensure(idx);
      (contiguous as any)[out++] = src[idx];
    }
    return contiguous;
  }
  if (rank === 2) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = arr.offset + r * stride0 + c * stride1;
        ensure(idx);
        (contiguous as any)[out++] = src[idx];
      }
    }
    return contiguous;
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      for (let ch = 0; ch < channels; ch++) {
        const idx = arr.offset + r * stride0 + c * stride1 + ch * stride2;
        ensure(idx);
        (contiguous as any)[out++] = src[idx];
      }
    }
  }
  return contiguous;
}

function flattenNumberContainer<T extends ndarray.Data>(
  arr: NdArray<T>,
  meta: { rank: number; rows: number; cols: number; channels: number; expectedLen: number },
  raw: unknown[]
): number[] {
  // raw は多次元 (Array) を含む可能性があるので flat(Infinity)
  const flat: number[] = (raw as unknown[]).flat(Infinity) as number[];
  // offset が 0 の場合は単純長さチェック
  if (arr.offset === 0 && flat.length !== meta.expectedLen) {
    throw new Error(
      `ndArrayToMat: data length mismatch (got ${flat.length}, expected ${meta.expectedLen})`
    );
  }
  // stride/offset が絡む場合は再収集
  if (arr.offset !== 0 || flat.length !== meta.expectedLen) {
    return harvestNumberContiguous(arr, meta, flat);
  }
  return flat;
}

function harvestNumberContiguous<T extends ndarray.Data>(
  arr: NdArray<T>,
  meta: { rank: number; rows: number; cols: number; channels: number; expectedLen: number },
  data: number[]
): number[] {
  const { rank, rows, cols, channels, expectedLen } = meta;
  const harvested = new Array<number>(expectedLen);
  const stride0 = arr.stride[0] ?? (rank > 1 ? cols : 1);
  const stride1 = arr.stride[1] ?? (rank === 3 ? channels : 1);
  const stride2 = arr.stride[2] ?? 1;
  let out = 0;

  const ensure = (idx: number) => {
    if (idx < 0 || idx >= data.length) {
      throw new Error("ndArrayToMat: data length mismatch (access out of bounds)");
    }
  };

  if (rank === 1) {
    for (let i = 0; i < rows; i++) {
      const idx = arr.offset + i * stride0;
      ensure(idx);
      harvested[out++] = data[idx]!;
    }
    return harvested;
  }
  if (rank === 2) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = arr.offset + r * stride0 + c * stride1;
        ensure(idx);
        harvested[out++] = data[idx]!;
      }
    }
    return harvested;
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      for (let ch = 0; ch < channels; ch++) {
        const idx = arr.offset + r * stride0 + c * stride1 + ch * stride2;
        ensure(idx);
        harvested[out++] = data[idx]!;
      }
    }
  }
  return harvested;
}


export function chooseDepthForNumbers(nums: number[]): CvDepth {
  const hasFloat = nums.some((v) => !Number.isInteger(v));

  if (hasFloat) {
    // NumPy はデフォルトで float64 を選ぶ
    return "64F";
  }

  // JS の number は 53bit 精度を持つので int64 相当まで扱える
  // ただし OpenCV に int64 は無いので int32 に寄せる
  const maxInt32 = 2_147_483_647;
  const minInt32 = -2_147_483_648;

  const needs64 = nums.some((v) => v > maxInt32 || v < minInt32);

  if (needs64) {
    // NumPy なら int64 だが OpenCV に無いので float64 にフォールバック
    return "64F";
  }

  return "32S";
}

export function ndArrayToMat<T extends ndarray.Data>(arr: NdArray<T>, cv: CV2): Mat {
  const meta = calcShapeMeta(arr);
  const data = (arr as any).data;

  // 数値配列 (ネスト配列) の場合
  if (Array.isArray(data)) {
    const flat = flattenNumberContainer(arr, meta, data);
    if (flat.length !== meta.expectedLen) {
      throw new Error(
        `ndArrayToMat: data length mismatch (got ${flat.length}, expected ${meta.expectedLen})`
      );
    }
    const depth = chooseDepthForNumbers(flat);
    return cv.matFromArray(meta.rows, meta.cols, getCvType(cv, depth, meta.channels), flat);
  }

  // TypedArray 系
  if (
    data instanceof Uint8Array || data instanceof Uint8ClampedArray ||
    data instanceof Int8Array || data instanceof Uint16Array ||
    data instanceof Int16Array || data instanceof Uint32Array ||
    data instanceof Int32Array || data instanceof Float32Array ||
    data instanceof Float64Array
  ) {
    if (data.length < meta.expectedLen) {
      // 足りない場合は stride/offset 参照で収集
      const contiguous = harvestTypedContiguous(arr, meta, data);
      const depth = inferDepthFromTypedArray(contiguous);
      if ((contiguous as any).length !== meta.expectedLen) {
        throw new Error("ndArrayToMat: data length mismatch after harvest");
      }
      return cv.matFromArray(
        meta.rows,
        meta.cols,
        getCvType(cv, depth, meta.channels),
        contiguous as any
      );
    } else if (arr.offset !== 0 || !isStandardContiguous(arr, meta)) {
      const contiguous = harvestTypedContiguous(arr, meta, data);
      const depth = inferDepthFromTypedArray(contiguous);
      return cv.matFromArray(
        meta.rows,
        meta.cols,
        getCvType(cv, depth, meta.channels),
        contiguous as any
      );
    } else {
      // 標準連続メモリ
      const depth = inferDepthFromTypedArray(data);
      return cv.matFromArray(meta.rows, meta.cols, getCvType(cv, depth, meta.channels), data as any);
    }
  }

  throw new Error("ndArrayToMat: unsupported data container");
}

function isStandardContiguous<T extends ndarray.Data>(
  arr: NdArray<T>,
  meta: { rank: number; rows: number; cols: number; channels: number }
): boolean {
  const { rank, rows, cols, channels } = meta;
  const stride = arr.stride;
  if (rank === 1) {
    return stride[0] === 1;
  }
  if (rank === 2) {
    return stride[0] === cols && stride[1] === 1;
  }
  // rank === 3
  return stride[0] === cols * channels && stride[1] === channels && stride[2] === 1;
}