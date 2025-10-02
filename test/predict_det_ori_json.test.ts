import { join } from "path";
import { beforeAll, describe, expect, it } from "vitest";
import { TextDetector, TextDetectorParams } from "../src/predict_det.js";
import { model_dir, test_image_dir } from "./config.js";
import fs from "fs/promises";
import type cvReadyPromiseType from "@techstark/opencv-js";
import { Box, ORT } from "../src/types/type.js";
import { Jimp } from "jimp";
import ori_im from "./ori_im.json" with { type: "json" };

let cv: Awaited<typeof cvReadyPromiseType>;
let ort: ORT;

const trueBoxes: Box[] = [
  [
    [89, 417],
    [379, 428],
    [376, 523],
    [86, 512],
  ],

  [
    [21, 229],
    [559, 229],
    [559, 341],
    [21, 341],
  ],

  [
    [276, 121],
    [332, 121],
    [332, 180],
    [276, 180],
  ],
];

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  const nodeORT = require("onnxruntime-node");
  ort = nodeORT;
  cv = await cvReadyPromise;
});

describe("TextDetector", () => {
  it("detect: 簡単なケース", async () => {
    const ori_im_data = ori_im as number[][][];
    const shape = [579, 584, 3];
    const imageMat = cv.matFromArray(
      shape[0],
      shape[1],
      cv.CV_8UC3,
      Uint8Array.from(ori_im_data.flat(2)),
    );

    // pngでは透過分がalphaチャンネルに入るので、3チャンネルに変換しておく
    const det_model_src = join(model_dir, "ppocrv5", "det", "det.onnx");
    const det_model_buffer = await fs.readFile(det_model_src);

    const textDetectorParams: TextDetectorParams = {
      limit_side_len: 960,
      det_limit_type: "max",
      det_db_thresh: 0.3, // 閾値下げる
      det_db_box_thresh: 0.6, // ボックス閾値も下げる
      det_db_unclip_ratio: 1.5,
      use_dilation: false,
      det_db_score_mode: "fast",
      det_box_type: "quad",
      cv,
      ort,
      det_model_array_buffer: det_model_buffer,
      use_gpu: false,
      drop_score: null,
    };
    const detector = await TextDetector.create(textDetectorParams);

    const boxes = await detector.execute(imageMat);
    expect(boxes).toBeDefined();
    expect(boxes).toEqual(trueBoxes);
  });
});
