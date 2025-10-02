import { join } from "path";
import { beforeAll, describe, expect, it } from "vitest";
import { TextDetector, TextDetectorParams } from "../src/predict_det.js";
import { model_dir, test_image_dir } from "./config.js";
import fs from "fs/promises";
import type cvReadyPromiseType from "@techstark/opencv-js";
import { Box, ORT } from "../src/types/type.js";
import { Jimp } from "jimp";

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
    const model_src = join(model_dir, "ppocrv5", "det", "det.onnx");
    const model_buffer = await fs.readFile(model_src);

    const image_src = join(test_image_dir, "onnxocr_logo.png");
    const jimpImage = await Jimp.read(image_src);

    const imageMat = cv.matFromImageData(jimpImage.bitmap);

    // pngでは透過分がalphaチャンネルに入るので、3チャンネルに変換しておく
    const imageMat3Ch = new cv.Mat();
    cv.cvtColor(imageMat, imageMat3Ch, cv.COLOR_RGBA2BGR);

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
      det_model_array_buffer: model_buffer,
      use_gpu: false,
      drop_score: null,
    };
    const detector = await TextDetector.create(textDetectorParams);

    const boxes = await detector.execute(imageMat3Ch);
    expect(boxes).toBeDefined();
    console.log("boxes:", boxes);
    expect(boxes!.length).toBeGreaterThanOrEqual(3);
    expect(boxes!.flat(Infinity)).toEqual(trueBoxes.flat(Infinity));
  });
});
