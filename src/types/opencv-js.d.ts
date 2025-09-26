import type { RotatedRect, Point2f } from "@techstark/opencv-js";

declare module "@techstark/opencv-js" {
  // 既存の型定義を保持しつつ boxPoints を上書き
  function boxPoints(box: RotatedRect): Point2f[];
}
