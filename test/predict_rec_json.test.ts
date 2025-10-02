import { TextClassifier, TextClassifierParams } from "../src/predict_cls.js";

import { join } from "path";

import { beforeAll, describe, expect, it } from "vitest";
import { TextDetector, TextDetectorParams } from "../src/predict_det.js";
import { model_dir, test_image_dir } from "./config.js";
import fs from "fs/promises";
import type cvReadyPromiseType from "@techstark/opencv-js";
import { Box, ORT } from "../src/types/type.js";
import { Jimp } from "jimp";
import type { Mat } from "@techstark/opencv-js";

import { TextRecognizer, TextRecognizerParams } from "../src/predict_rec.js";

import ori_im_list from "./ori_im_list.json" with { type: "json" };

let cv: Awaited<typeof cvReadyPromiseType>;
let ort: ORT;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  const nodeORT = require("onnxruntime-node");
  ort = nodeORT;
  cv = await cvReadyPromise;
});

describe("TextRecognition / TextDetector", () => {
  it("recognition: 簡単なケース", async () => {
    const charset_path = join(model_dir, "ppocrv5", "ppocrv5_dict.txt");
    const charset = (await fs.readFile(charset_path, "utf-8")).toString();

    // pngでは透過分がalphaチャンネルに入るので、3チャンネルに変換しておく

    const rec_model_src = join(model_dir, "ppocrv5", "rec", "rec.onnx");
    const rec_model_buffer = await fs.readFile(rec_model_src);

    const det_box_type = "quad";

    const imageMatList = (ori_im_list as number[][][][]).map((ori_im_data) => {
      const h = ori_im_data.length;
      const w = ori_im_data[0].length;
      return cv.matFromArray(
        h,
        w,
        cv.CV_8UC3,
        Uint8Array.from(ori_im_data.flat(2)),
      );
    });

    const textRecognizerParams: TextRecognizerParams = {
      drop_score: 0.5,
      rec_batch_num: 6,
      rec_algorithm: "SVTR_LCNet",
      rec_image_shape: [3, 48, 320],
      cv: cv,
      ort: ort,
      rec_model_array_buffer: rec_model_buffer,
      rec_char_dict: charset,
      use_space_char: false,
      use_gpu: false,
    };

    const textRecognizer = await TextRecognizer.create(textRecognizerParams);
    const rec_res = await textRecognizer.execute(imageMatList);
    console.log("rec_res:", rec_res);
  });
});
