import type { Mat } from "@techstark/opencv-js";
import type { Box, CV2, Point } from "../types/type.js";
import type { NdArray } from "ndarray";
import ndarray from "ndarray";
import ops from "ndarray-ops";

export function euclideanDistance(point1: Point, point2: Point): number {
  return Math.sqrt(Math.pow(point1[0] - point2[0], 2) + Math.pow(point1[1] - point2[1], 2));
}

export function boxToLine(box: Box): [number,number, number,number, number, number,number,number] {
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

  const points: Point[] = matToList(mat,cv) as Point[];

  return points;
}

export function matToArrayBuffer(mat: Mat,cv:CV2) {
  switch (mat.type()) {
    // 8bit unsigned integer
    case cv.CV_8UC1:
    case cv.CV_8UC2:
    case cv.CV_8UC3:
    case cv.CV_8UC4:
      return mat.data
        
    // 8bit signed integer
    case cv.CV_8SC1:
    case cv.CV_8SC2:
    case cv.CV_8SC3:
    case cv.CV_8SC4:
      return mat.data8S
        
    // 16bit unsigned integer
    case cv.CV_16UC1:
    case cv.CV_16UC2:
    case cv.CV_16UC3:
    case cv.CV_16UC4:
      return mat.data16U
    // 16bit signed integer
    case cv.CV_16SC1:
    case cv.CV_16SC2:
    case cv.CV_16SC3:
    case cv.CV_16SC4:
      return mat.data16S
    // 32bit signed integer
    case cv.CV_32SC1:
    case cv.CV_32SC2:
    case cv.CV_32SC3:
    case cv.CV_32SC4:
      return mat.data32S
    // 32bit float
    case cv.CV_32FC1:
    case cv.CV_32FC2:
    case cv.CV_32FC3:
    case cv.CV_32FC4:
      return mat.data32F
    // 64bit float
    case cv.CV_64FC1:
    case cv.CV_64FC2:
    case cv.CV_64FC3:
    case cv.CV_64FC4:
      return mat.data64F
    default:
      return null;
  }
}

export function matToLine(mat: Mat,cv:CV2):{data:number[], col:number, row:number, channel:number} {
  const data = matToArrayBuffer(mat,cv);
  if(!(data instanceof Float32Array || data instanceof Int32Array || data instanceof Uint8Array || data instanceof Int8Array || data instanceof Uint16Array || data instanceof Int16Array || data instanceof Float64Array)){
    console.log(data);
    throw new Error("matToLine: unsupported Mat type");
  }
  const array = Array.from(data)
  const col = mat.cols;
  const row = mat.rows;
  const channel = mat.channels();
  if (array.length !== col * row * channel) {
    throw new Error(`matToLine: data length mismatch\n  expected: ${col} * ${row} * ${channel} = ${col * row * channel}\n  actual: ${array.length}`);
  }
  return {data:array, col:col, row:row, channel:channel};
}

export function matToList(mat: Mat, cv: CV2, minCol = true): number[][] | number[][][] {
  const { data, col, row, channel } = matToLine(mat, cv);

  return shapeAndDataToList({ col, row, channel, data }, minCol);
}

export function shapeAndDataToList({
  col,
  row,
  channel,
  data,
}: {
  col: number;
  row: number;
  channel: number;
  data: number[];
}, minCol = true): number[][] | number[][][] {
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

export function get_rotate_crop_image(img:Mat, points:Box, cv:CV2):Mat{
  if (points.length !== 4) {
    throw new Error("shape of points must be 4*2");
  }
  const img_crop_width = Math.trunc(Math.max(
    euclideanDistance(points[0], points[1]),
    euclideanDistance(points[2], points[3])
  ));
  const img_crop_height = Math.trunc(Math.max(
    euclideanDistance(points[0], points[3]),
    euclideanDistance(points[1], points[2])
  ));

  const pts_std:Box = [
    [0,0],
    [img_crop_width,0],
    [img_crop_width,img_crop_height],
    [0,img_crop_height]
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
    cv.INTER_CUBIC,
  );
  const imgSize = dst_img.size();
  if (imgSize.height * 1.0 / imgSize.width >= 1.5) {
    cv.rotate(dst_img, dst_img, cv.ROTATE_90_CLOCKWISE);
  }
  srcTri.delete(); 
  dstTri.delete(); 
  M.delete();
  return dst_img;
}

export function get_minarea_rect_crop(img:Mat, arg_points:Box, cv:CV2):Mat{
  const srcTri = boxToMat(arg_points, cv);
  const bounding_box = cv.minAreaRect(srcTri);
  const matPoints: Mat = new cv.Mat();
  cv.boxPoints(bounding_box, matPoints);
  const beforePoints: Point[] = matToPoints(matPoints, cv);
  if (beforePoints.length !== 4) {
    throw new Error("shape of points must be 4*2");
  }
  const points = beforePoints.sort((a, b) => a[0] - b[0]) as Box;

   let index_a = 0, index_b = 1, index_c = 2, index_d = 3;

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
      newStrides[ndim - 1 - i] =
        inStrides[inShape.length - 1 - i] ?? 0;
    } else if (inDim === 1) {
      // broadcast → stride を 0 にする
      newStrides[ndim - 1 - i] = 0;
    } else {
      // NumPy ルールに反する場合はエラー
      throw new Error(
        `Cannot broadcast dimension ${inDim} to ${outDim}`
      );
    }
  }

  return ndarray(arr.data, targetShape, newStrides, arr.offset);
}

export function fillValue<T extends ndarray.Data>(arr:NdArray<T>,targetShape: number[],value=0){
  if (arr.shape.length !== targetShape.length){
    throw new Error(`fillZero: shape length mismatch\n  expected: ${targetShape.length}\n  actual: ${arr.shape.length}`);
  }
  for (let i = 0; i < targetShape.length; i++){
    if (arr.shape[i]! > targetShape[i]!){
      throw new Error(`fillZero: shape mismatch at dimension ${i}\n  expected: <= ${targetShape[i]}\n  actual: ${arr.shape[i]}`);
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