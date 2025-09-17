import type { Mat } from "@techstark/opencv-js";
import type cvReadyPromise from "@techstark/opencv-js";
import type { NdArray } from "ndarray";

export type Point = [number, number];  // [x, y]
export type Box = [Point, Point, Point, Point];  // 四角形を構成する4点
export type CV2 = Awaited<typeof cvReadyPromise>
export type DataKeys = keyof Data;
export type DataValues = Data[keyof Data];
export type Data = MatData | NdArrayData ;
export type MatData = {
  image:Mat;
  shape: [number, number, number, number] | null;
}
export type NdArrayData = {
  image:NdArray;
  shape: [number, number, number, number] | null;
}