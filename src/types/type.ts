import type { Mat } from "@techstark/opencv-js";
import type cvReadyPromise from "@techstark/opencv-js";

export type Point = [number, number];  // [x, y]
export type Box = [Point, Point, Point, Point];  // 四角形を構成する4点
export type CV2 = Awaited<typeof cvReadyPromise>
export type Data = {
  image:Mat,
}