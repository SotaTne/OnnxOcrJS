import type { Mat } from "@techstark/opencv-js";
import type { Box, CV2, Point } from "../../types/type.js";
import type { DET_BOX_TYPE } from "../../types/paddle_types.js";
import { boxToMat, euclideanDistance, matToPoints } from "../func.js";

export function sortedBoxes(dt_boxes: Point[][]): Point[][] {
  const numBoxes = dt_boxes.length;

  // まず (y, x) で大まかにソート
  const sorted = dt_boxes.slice().sort((a, b) => {
    if (a[0]![1] === b[0]![1]) {
      return a[0]![0] - b[0]![0]; // y が同じなら x でソート
    }
    return a[0]![1] - b[0]![1]; // y でソート
  });

  const boxes = [...sorted];

  for (let i = 0; i < numBoxes - 1; i++) {
    for (let j = i; j >= 0; j--) {
      const yDiff = Math.abs(boxes[j + 1]![0]![1] - boxes[j]![0]![1]);
      const xRight = boxes[j + 1]![0]![0];
      const xLeft = boxes[j]![0]![0];

      if (yDiff < 10 && xRight < xLeft) {
        // swap
        const tmp = boxes[j]!;
        boxes[j] = boxes[j + 1]!;
        boxes[j + 1] = tmp;
      } else {
        break;
      }
    }
  }

  return boxes;
}

export function getImgCropList(
  ori_img: Mat,
  dt_boxes: Point[][],
  det_box_type: DET_BOX_TYPE,
  cv: CV2,
): Mat[] {
  const img_crop_list: Mat[] = [];
  dt_boxes = sortedBoxes(dt_boxes);

  for (const box of dt_boxes) {
    const tmp_box: Box = [...box.map((pt) => [...pt] as Point)] as Box;
    const img_crop: Mat =
      det_box_type === "quad"
        ? get_rotate_crop_image(ori_img, tmp_box, cv)
        : get_minarea_rect_crop(ori_img, tmp_box, cv);
    img_crop_list.push(img_crop);
  }
  return img_crop_list;
}

export function get_rotate_crop_image(img: Mat, points: Box, cv: CV2): Mat {
  if (points.length !== 4) {
    throw new Error("shape of points must be 4*2");
  }
  const img_crop_width = Math.floor(
    Math.max(
      euclideanDistance(points[0], points[1]),
      euclideanDistance(points[2], points[3]),
    ),
  );
  const img_crop_height = Math.floor(
    Math.max(
      euclideanDistance(points[0], points[3]),
      euclideanDistance(points[1], points[2]),
    ),
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
    cv.INTER_CUBIC, // flags
    cv.BORDER_REPLICATE, // borderMode
  );

  const imgSize = dst_img.size();
  if ((imgSize.height * 1.0) / imgSize.width >= 1.5) {
    cv.rotate(dst_img, dst_img, cv.ROTATE_90_COUNTERCLOCKWISE);
  }

  srcTri.delete();
  dstTri.delete();
  M.delete();
  return dst_img;
}

export function get_minarea_rect_crop(
  img: Mat,
  arg_points: Point[],
  cv: CV2,
): Mat {
  // OpenCV.js では minAreaRect の引数は MatOfPoint2f
  const pointsMat = cv.matFromArray(
    arg_points.length,
    1,
    cv.CV_32FC2,
    arg_points.flat(2),
  );

  const bounding_box = cv.minAreaRect(pointsMat);
  // const srcTri = boxToMat(arg_points, cv);
  // const bounding_box = cv.minAreaRect(srcTri);

  // OpenCV.js の boxPoints は Point[] を返す
  const boxPoints = cv.boxPoints(bounding_box);
  const beforePoints: Point[] = boxPoints.map((p) => [p.x, p.y]);

  // Python: sorted(..., key=lambda x: x[0])
  const points = [...beforePoints].sort((a, b) => a[0] - b[0]);

  let index_a = 0,
    index_b = 1,
    index_c = 2,
    index_d = 3;

  if (points[1]![1] > points[0]![1]) {
    index_a = 0;
    index_d = 1;
  } else {
    index_a = 1;
    index_d = 0;
  }

  if (points[3]![1] > points[2]![1]) {
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

  //srcTri.delete();
  pointsMat.delete();
  return crop_img;
}
