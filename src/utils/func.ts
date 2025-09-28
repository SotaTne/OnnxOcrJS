import type { Mat } from "@techstark/opencv-js";
import type { Box, CV2, Point } from "../types/type.js";
import type { NdArray } from "ndarray";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import * as clipperLib from "js-angusj-clipper";

export function euclideanDistance(point1: Point, point2: Point): number {
  return Math.sqrt(
    Math.pow(point1[0] - point2[0], 2) + Math.pow(point1[1] - point2[1], 2)
  );
}

export function boxToLine(
  box: Box
): [number, number, number, number, number, number, number, number] {
  const [p1, p2, p3, p4] = box;
  return [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]];
}

export function boxToMat(box: Box, cv: CV2): Mat {
  return cv.matFromArray(4, 1, cv.CV_32FC2, boxToLine(box));
}

// Mat(4x1x2, CV_32F or CV_32S) → [[x,y], [x,y], [x,y], [x,y]]
export function matToPoints(mat: Mat, cv: CV2): Point[] {
  if (mat.type() !== cv.CV_32FC2 && mat.type() !== cv.CV_32SC2) {
    throw new Error("matToList: expected CV_32FC2 or CV_32SC2 Mat");
  }

  if (mat.cols !== 1 || mat.channels() !== 2 || mat.rows !== 4) {
    throw new Error("matToPoints: expected 4x1x2 Mat");
  }

  const points: Point[] = matToList(mat, cv) as Point[];

  return points;
}

export function matToArrayBuffer(mat: Mat, cv: CV2) {
  switch (mat.type()) {
    // 8bit unsigned integer
    case cv.CV_8UC1:
    case cv.CV_8UC2:
    case cv.CV_8UC3:
    case cv.CV_8UC4:
      return mat.data;

    // 8bit signed integer
    case cv.CV_8SC1:
    case cv.CV_8SC2:
    case cv.CV_8SC3:
    case cv.CV_8SC4:
      return mat.data8S;

    // 16bit unsigned integer
    case cv.CV_16UC1:
    case cv.CV_16UC2:
    case cv.CV_16UC3:
    case cv.CV_16UC4:
      return mat.data16U;
    // 16bit signed integer
    case cv.CV_16SC1:
    case cv.CV_16SC2:
    case cv.CV_16SC3:
    case cv.CV_16SC4:
      return mat.data16S;
    // 32bit signed integer
    case cv.CV_32SC1:
    case cv.CV_32SC2:
    case cv.CV_32SC3:
    case cv.CV_32SC4:
      return mat.data32S;
    // 32bit float
    case cv.CV_32FC1:
    case cv.CV_32FC2:
    case cv.CV_32FC3:
    case cv.CV_32FC4:
      return mat.data32F;
    // 64bit float
    case cv.CV_64FC1:
    case cv.CV_64FC2:
    case cv.CV_64FC3:
    case cv.CV_64FC4:
      return mat.data64F;
    default:
      return null;
  }
}

export function matToLine(
  mat: Mat,
  cv: CV2
): { data: number[]; col: number; row: number; channel: number } {
  const data = matToArrayBuffer(mat, cv);
  if (
    !(
      data instanceof Float32Array ||
      data instanceof Int32Array ||
      data instanceof Uint8Array ||
      data instanceof Int8Array ||
      data instanceof Uint16Array ||
      data instanceof Int16Array ||
      data instanceof Float64Array
    )
  ) {
    console.log(data);
    throw new Error("matToLine: unsupported Mat type");
  }
  const array = Array.from(data);
  const col = mat.cols;
  const row = mat.rows;
  const channel = mat.channels();
  if (array.length !== col * row * channel) {
    throw new Error(
      `matToLine: data length mismatch\n  expected: ${col} * ${row} * ${channel} = ${
        col * row * channel
      }\n  actual: ${array.length}`
    );
  }
  return { data: array, col: col, row: row, channel: channel };
}

export function matToList(
  mat: Mat,
  cv: CV2,
  minCol = true
): number[][] | number[][][] {
  const { data, col, row, channel } = matToLine(mat, cv);

  return shapeAndDataToList({ col, row, channel, data }, minCol);
}

export function shapeAndDataToList(
  {
    col,
    row,
    channel,
    data,
  }: {
    col: number;
    row: number;
    channel: number;
    data: number[];
  },
  minCol = true
): number[][] | number[][][] {
  if (minCol && col === 1) {
    const list: number[][] = [];
    for (let r = 0; r < row; r++) {
      const rowArray: number[] = [];
      for (let ch = 0; ch < channel; ch++) {
        const index = r * channel + ch;
        rowArray.push(data[index]!);
      }
      list.push(rowArray);
    }
    return list;
  } else {
    const list: number[][][] = [];
    for (let r = 0; r < row; r++) {
      const rowArray: number[][] = [];
      for (let c = 0; c < col; c++) {
        const pixelArray: number[] = [];
        for (let ch = 0; ch < channel; ch++) {
          const index = (r * col + c) * channel + ch;
          pixelArray.push(data[index]!);
        }
        rowArray.push(pixelArray);
      }
      list.push(rowArray);
    }

    return list;
  }
}

export function get_rotate_crop_image(img: Mat, points: Box, cv: CV2): Mat {
  if (points.length !== 4) {
    throw new Error("shape of points must be 4*2");
  }
  const img_crop_width = Math.trunc(
    Math.max(
      euclideanDistance(points[0], points[1]),
      euclideanDistance(points[2], points[3])
    )
  );
  const img_crop_height = Math.trunc(
    Math.max(
      euclideanDistance(points[0], points[3]),
      euclideanDistance(points[1], points[2])
    )
  );

  const pts_std: Box = [
    [0, 0],
    [img_crop_width, 0],
    [img_crop_width, img_crop_height],
    [0, img_crop_height],
  ];

  const srcTri = boxToMat(points, cv);
  const dstTri = boxToMat(pts_std, cv);

  const M = cv.getPerspectiveTransform(srcTri, dstTri);
  const dst_img = new cv.Mat();
  cv.warpPerspective(
    img,
    dst_img,
    M,
    new cv.Size(img_crop_width, img_crop_height),
    cv.BORDER_REPLICATE,
    cv.INTER_CUBIC
  );
  const imgSize = dst_img.size();
  if ((imgSize.height * 1.0) / imgSize.width >= 1.5) {
    cv.rotate(dst_img, dst_img, cv.ROTATE_90_CLOCKWISE);
  }
  srcTri.delete();
  dstTri.delete();
  M.delete();
  return dst_img;
}

export function get_minarea_rect_crop(img: Mat, arg_points: Box, cv: CV2): Mat {
  const srcTri = boxToMat(arg_points, cv);
  const bounding_box = cv.minAreaRect(srcTri);
  const matPoints: Mat = new cv.Mat();
  cv.boxPoints(bounding_box, matPoints);
  const beforePoints: Point[] = matToPoints(matPoints, cv);
  if (beforePoints.length !== 4) {
    throw new Error("shape of points must be 4*2");
  }
  const points = beforePoints.sort((a, b) => a[0] - b[0]) as Box;

  let index_a = 0,
    index_b = 1,
    index_c = 2,
    index_d = 3;

  if (points[1][1] > points[0][1]) {
    index_a = 0;
    index_d = 1;
  } else {
    index_a = 1;
    index_d = 0;
  }

  if (points[3][1] > points[2][1]) {
    index_b = 2;
    index_c = 3;
  } else {
    index_b = 3;
    index_c = 2;
  }

  const sortedBox: Box = [
    points[index_a]!,
    points[index_b]!,
    points[index_c]!,
    points[index_d]!,
  ];

  const crop_img = get_rotate_crop_image(img, sortedBox, cv);
  srcTri.delete();
  matPoints.delete();
  return crop_img;
}

export function matToNdArray(mat: Mat, cv: CV2, skip_channel = false): NdArray {
  const buffer = matToArrayBuffer(mat, cv);
  if (
    !(
      buffer instanceof Float32Array ||
      buffer instanceof Int32Array ||
      buffer instanceof Uint8Array ||
      buffer instanceof Int8Array ||
      buffer instanceof Uint16Array ||
      buffer instanceof Int16Array ||
      buffer instanceof Float64Array
    )
  ) {
    throw new Error("matToNdArray: unsupported Mat type");
  }
  if (skip_channel && mat.channels() === 1) {
    return ndarray(buffer, [mat.rows, mat.cols]);
  }
  return ndarray(buffer, [mat.rows, mat.cols, mat.channels()]);
}

export function broadcastTo<T extends ndarray.Data>(
  arr: NdArray<T>,
  targetShape: number[]
): NdArray<T> {
  const inShape = arr.shape;
  const inStrides = arr.stride.slice();

  const ndim = targetShape.length;
  const newStrides: number[] = new Array(ndim).fill(0);

  // 右揃えで各次元を比較
  for (let i = 0; i < ndim; i++) {
    const inDim = inShape[inShape.length - 1 - i] ?? 1;
    const outDim = targetShape[targetShape.length - 1 - i];

    if (inDim === outDim) {
      // 次元が一致 → stride をそのまま使う
      newStrides[ndim - 1 - i] = inStrides[inShape.length - 1 - i] ?? 0;
    } else if (inDim === 1) {
      // broadcast → stride を 0 にする
      newStrides[ndim - 1 - i] = 0;
    } else {
      // NumPy ルールに反する場合はエラー
      throw new Error(`Cannot broadcast dimension ${inDim} to ${outDim}`);
    }
  }

  return ndarray(arr.data, targetShape, newStrides, arr.offset);
}

export function fillValue<T extends ndarray.Data>(
  arr: NdArray<T>,
  targetShape: number[],
  value = 0
) {
  if (arr.shape.length !== targetShape.length) {
    throw new Error(
      `fillZero: shape length mismatch\n  expected: ${targetShape.length}\n  actual: ${arr.shape.length}`
    );
  }
  for (let i = 0; i < targetShape.length; i++) {
    if (arr.shape[i]! > targetShape[i]!) {
      throw new Error(
        `fillZero: shape mismatch at dimension ${i}\n  expected: <= ${targetShape[i]}\n  actual: ${arr.shape[i]}`
      );
    }
  }

  // 新しい配列を確保
  const size = targetShape.reduce((a, b) => a * b, 1);
  const NewArrayCtor = arr.data.constructor as { new (size: number): T };
  const buffer = new NewArrayCtor(size);

  // 全体を value で埋める（効率的に）
  if ((buffer as any).fill) {
    (buffer as any).fill(value);
  } else {
    for (let i = 0; i < buffer.length; i++) {
      (buffer as any)[i] = value;
    }
  }

  const newArr = ndarray(buffer, targetShape);

  // 元データをコピー
  const slices = arr.shape.map((dim) => [0, dim] as [number, number]);
  ops.assign(
    newArr.hi(...slices.map((s) => s[1])).lo(...slices.map((s) => s[0])),
    arr
  );

  return newArr;
}

export function pickAndSet<T extends ndarray.Data>(
  arr: NdArray<T>,
  set: (view: NdArray<T>) => NdArray<T>,
  ...args: number[]
): NdArray<T> {
  // 配列が空の場合は例外
  if (arr.shape.length === 0 || arr.shape.some((s) => s === 0)) {
    throw new Error("Cannot pick from empty array");
  }

  // インデックスが配列次元数を超える場合は例外
  if (args.length > arr.shape.length) {
    throw new Error("Too many indices");
  }

  // 範囲チェック (-1は全範囲扱い)
  args.forEach((a, i) => {
    if (a !== -1 && (a < 0 || a >= arr.shape[i]!)) {
      throw new Error(`Index ${a} out of range for dimension ${i}`);
    }
  });

  const picked = arr.pick(...args);

  // set を呼び出す - 副作用で picked が変更される
  const updated = set(cloneNdArray(picked));

  // updated が picked と異なる配列の場合のみ、値をコピーバック
  if (updated !== picked && updated.data !== picked.data) {
    ops.assign(picked, updated);
  }

  // 元の配列を返す（pickedの変更は既に反映されている）
  return arr;
}

export function clip(
  dest: NdArray<ndarray.TypedArray | ndarray.GenericArray<number> | number[]>,
  src: NdArray<ndarray.TypedArray | ndarray.GenericArray<number> | number[]>,
  min: number,
  max: number
): void {
  if (min > max) {
    throw new Error(`min (${min}) must be <= max (${max})`);
  }

  ops.assign(dest, src);
  ops.minseq(dest, max);
  ops.maxseq(dest, min);
}

export function cloneNdArray<T extends ndarray.Data>(
  src: NdArray<T>
): NdArray<T> {
  let newData: T;
  if (Array.isArray(src.data)) {
    newData = src.data.map((item) =>
      Array.isArray(item) ? [...item] : item
    ) as T;
  } else {
    newData = new (src.data.constructor as any)(src.data);
  }
  return ndarray(newData, src.shape, src.stride, src.offset);
}

export type NdArrayListData = number[] | NdArrayListData[];

export function ndArrayToList<T extends ndarray.Data>(
  arr: NdArray<T>
): NdArrayListData {
  // 0次元配列の場合は1要素の配列として扱う
  if (arr.shape.length === 0) {
    return [arr.get()];
  }

  // 1次元配列の場合
  if (arr.shape.length === 1) {
    const list: number[] = [];
    for (let i = 0; i < arr.shape[0]!; i++) {
      list.push(arr.get(i));
    }
    return list;
  }
  // この後テストの追加

  // 多次元配列の場合
  const list: NdArrayListData[] = [];
  for (let i = 0; i < arr.shape[0]!; i++) {
    const subArr = arr.pick(i);
    list.push(ndArrayToList(subArr));
  }
  return list;
}

let _clipper = null as clipperLib.ClipperLibWrapper | null;

export async function unclip(
  box: { x: number; y: number }[],
  unclipRatio: number
): Promise<clipperLib.Paths> {
  if (_clipper === null) {
    _clipper = await clipperLib.loadNativeClipperLibInstanceAsync(
      clipperLib.NativeClipperLibRequestedFormat.WasmWithAsmJsFallback
    );
  }
  const clipper = _clipper;

  const area = polygonArea(box);
  const perimeter = polygonPerimeter(box);
  const distance = (area * unclipRatio) / perimeter;

  if (Math.abs(distance) < 1e-6) {
    return [] as clipperLib.Paths; // 明示的に空を返す
  }

  return (
    clipper.offsetToPaths({
      delta: distance,
      offsetInputs: [
        {
          data: box,
          joinType: clipperLib.JoinType.Round,
          endType: clipperLib.EndType.ClosedPolygon,
        },
      ],
    }) ?? []
  );
}

// 補助関数（Pythonのpoly.area, poly.length相当）
export function polygonArea(points: { x: number; y: number }[]): number {
  let area = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const { x: x0, y: y0 } = points[i]!;
    const { x: x1, y: y1 } = points[(i + 1) % n]!;
    area += x0 * y1 - x1 * y0;
  }
  return Math.abs(area) / 2;
}

export function polygonPerimeter(points: { x: number; y: number }[]): number {
  let perimeter = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const { x: x0, y: y0 } = points[i]!;
    const { x: x1, y: y1 } = points[(i + 1) % n]!;
    perimeter += Math.hypot(x1 - x0, y1 - y0);
  }
  return perimeter;
}

export type NdArrayLike = number | NdArrayLike[];

/** shape を推定 */
function shape(arr: any): number[] {
  const shp: number[] = [];
  let cur = arr;
  while (Array.isArray(cur)) {
    shp.push(cur.length);
    cur = cur[0];
  }
  return shp;
}

function normalizeAxis(axis: number, ndim: number): number {
  const ax = axis < 0 ? axis + ndim : axis;
  if (ax < 0 || ax >= ndim) {
    throw new Error(`axis out of range: ${axis} for ndim=${ndim}`);
  }
  return ax;
}

/** max reduce (任意次元) */
function reduceMax(arr: NdArrayLike, axis: number): NdArrayLike {
  const shp = shape(arr);
  const ndim = shp.length;
  if (ndim === 0) throw new Error("not an array");
  const ax = normalizeAxis(axis, ndim);

  // 1D
  if (ndim === 1) {
    const xs = arr as number[];
    if (xs.length === 0) throw new Error("cannot reduce empty array");
    let best = xs[0]; // 初期値は最初の要素
    for (let i = 1; i < xs.length; i++) {
      const v = xs[i];
      if (typeof v !== "number") throw new Error("non-number encountered");
      if (v > best!) best = v;
    }
    return best!;
  }

  // axis == 0
  if (ax === 0) {
    const len = (arr as NdArrayLike[]).length;
    if (len === 0) throw new Error("cannot reduce empty array (axis=0)");
    const restShape = shape((arr as NdArrayLike[])[0]);

    const build = (indices: number[]): any => {
      let best: number | undefined = undefined;
      for (let i = 0; i < len; i++) {
        let cur: any = (arr as NdArrayLike[])[i]!;
        for (const idx of indices) cur = cur[idx];
        if (typeof cur !== "number") throw new Error("non-number encountered");
        if (best === undefined || cur > best) {
          best = cur;
        }
      }
      return best!;
    };

    const recur = (depth: number, indices: number[]): any => {
      if (depth === restShape.length) return build(indices);
      const out: any[] = [];
      for (let j = 0; j < restShape[depth]!; j++) {
        out.push(recur(depth + 1, [...indices, j]));
      }
      return out;
    };

    return recur(0, []);
  }

  // axis > 0
  return (arr as NdArrayLike[]).map((sub) => reduceMax(sub, ax - 1));
}

/** argmax reduce (任意次元) */
function reduceArgmax(arr: NdArrayLike, axis: number): NdArrayLike {
  const shp = shape(arr);
  const ndim = shp.length;
  if (ndim === 0) throw new Error("not an array");
  const ax = normalizeAxis(axis, ndim);

  if (ndim === 1) {
    const xs = arr as number[];
    if (xs.length === 0) throw new Error("cannot reduce empty array");
    let bestV = xs[0];
    let bestI = 0;
    for (let i = 1; i < xs.length; i++) {
      const v = xs[i];
      if (typeof v !== "number") throw new Error("non-number encountered");
      if (v > bestV!) {
        bestV = v;
        bestI = i;
      }
    }
    return bestI;
  }

  if (ax === 0) {
    const len = (arr as NdArrayLike[]).length;
    const restShape = shape((arr as NdArrayLike[])[0]);

    const build = (indices: number[]): any => {
      let bestV: number | undefined = undefined;
      let bestI = 0;
      for (let i = 0; i < len; i++) {
        let cur: any = (arr as NdArrayLike[])[i];
        for (const idx of indices) cur = cur[idx];
        if (typeof cur !== "number") throw new Error("non-number encountered");
        if (bestV === undefined || cur > bestV) {
          bestV = cur;
          bestI = i;
        }
      }
      return bestI;
    };

    const recur = (depth: number, indices: number[]): any => {
      if (depth === restShape.length) return build(indices);
      const out: any[] = [];
      for (let j = 0; j < restShape[depth]!; j++) {
        out.push(recur(depth + 1, [...indices, j]));
      }
      return out;
    };

    return recur(0, []);
  }

  return (arr as NdArrayLike[]).map((sub) => reduceArgmax(sub, ax - 1));
}

/** 公開API */
export function max(arr: NdArrayLike, axis: number) {
  return reduceMax(arr, axis);
}

export function argmax(arr: NdArrayLike, axis: number) {
  return reduceArgmax(arr, axis);
}
