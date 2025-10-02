import { TextClassifierParams } from "../src/predict_cls.js";
import { TextSystem, TextSystemParams } from "../src/predict_system.js";

import { join } from "path";

import { beforeAll, describe, expect, it } from "vitest";
import { TextDetectorParams } from "../src/predict_det.js";
import { model_dir, test_image_dir } from "./config.js";
import fs from "fs/promises";
import type cvReadyPromiseType from "@techstark/opencv-js";
import { ORT } from "../src/types/type.js";
import { Jimp } from "jimp";
import { TextRecognizerParams } from "../src/predict_rec.js";

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
    // det

    const det_model_src = join(model_dir, "ppocrv5", "det", "det.onnx");
    const det_model_buffer = await fs.readFile(det_model_src);

    const image_src = join(test_image_dir, "onnxocr_logo.png");
    const jimpImage = await Jimp.read(image_src);

    const charset_path = join(model_dir, "ppocrv5", "ppocrv5_dict.txt");
    const charset = (await fs.readFile(charset_path, "utf-8")).toString();

    const imageMat = cv.matFromImageData(jimpImage.bitmap);

    // pngでは透過分がalphaチャンネルに入るので、3チャンネルに変換しておく
    const imageMat3Ch = new cv.Mat();
    cv.cvtColor(imageMat, imageMat3Ch, cv.COLOR_RGBA2BGR);

    const cls_model_src = join(model_dir, "ppocrv5", "cls", "cls.onnx");
    const cls_model_buffer = await fs.readFile(cls_model_src);

    const rec_model_src = join(model_dir, "ppocrv5", "rec", "rec.onnx");
    const rec_model_buffer = await fs.readFile(rec_model_src);

    const det_box_type = "quad";

    const textDetectorParams: TextDetectorParams = {
      limit_side_len: 960,
      det_limit_type: "max",
      det_db_thresh: 0.3, // 閾値下げる
      det_db_box_thresh: 0.6, // ボックス閾値も下げる
      det_db_unclip_ratio: 1.5,
      use_dilation: false,
      det_db_score_mode: "fast",
      det_box_type: det_box_type,
      cv,
      ort,
      det_model_array_buffer: det_model_buffer,
      use_gpu: false,
      drop_score: null,
    };

    const textClassifierParams: TextClassifierParams = {
      cv,
      cls_image_shape: [3, 48, 192], // "3,48,192" を配列に展開
      cls_thresh: 0.9,
      label_list: ["0", "180"],
      cls_batch_num: 6,
      ort,
      cls_model_array_buffer: cls_model_buffer,
      use_gpu: false,
    };

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

    const textSystemParams: TextSystemParams = {
      ...textDetectorParams,
      ...textRecognizerParams,
      ...textClassifierParams,
      use_angle_cls: true,
    };

    const textSystem = await TextSystem.create(textSystemParams);
    const results = await textSystem.execute(imageMat3Ch);
    expect(results[0]).toBeDefined();
    expect(results[1]).toBeDefined();
    expect(results[0]!.length).toEqual(results[1]!.length);
    console.log("results:", results);
  });
  it("recognition: 簡単なケース(万有館)", async () => {
    // det

    const det_model_src = join(model_dir, "ppocrv5", "det", "det.onnx");
    const det_model_buffer = await fs.readFile(det_model_src);

    const image_src = join(test_image_dir, "banyuukann.png");
    const jimpImage = await Jimp.read(image_src);

    const charset_path = join(model_dir, "ppocrv5", "ppocrv5_dict.txt");
    const charset = (await fs.readFile(charset_path, "utf-8")).toString();

    const imageMat = cv.matFromImageData(jimpImage.bitmap);

    // pngでは透過分がalphaチャンネルに入るので、3チャンネルに変換しておく
    const imageMat3Ch = new cv.Mat();
    cv.cvtColor(imageMat, imageMat3Ch, cv.COLOR_RGBA2BGR);

    const cls_model_src = join(model_dir, "ppocrv5", "cls", "cls.onnx");
    const cls_model_buffer = await fs.readFile(cls_model_src);

    const rec_model_src = join(model_dir, "ppocrv5", "rec", "rec.onnx");
    const rec_model_buffer = await fs.readFile(rec_model_src);

    const det_box_type = "quad";

    const textDetectorParams: TextDetectorParams = {
      limit_side_len: 960,
      det_limit_type: "max",
      det_db_thresh: 0.3, // 閾値下げる
      det_db_box_thresh: 0.6, // ボックス閾値も下げる
      det_db_unclip_ratio: 1.5,
      use_dilation: false,
      det_db_score_mode: "fast",
      det_box_type: det_box_type,
      cv,
      ort,
      det_model_array_buffer: det_model_buffer,
      use_gpu: false,
      drop_score: null,
    };

    const textClassifierParams: TextClassifierParams = {
      cv,
      cls_image_shape: [3, 48, 192], // "3,48,192" を配列に展開
      cls_thresh: 0.9,
      label_list: ["0", "180"],
      cls_batch_num: 6,
      ort,
      cls_model_array_buffer: cls_model_buffer,
      use_gpu: false,
    };

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

    const textSystemParams: TextSystemParams = {
      ...textDetectorParams,
      ...textRecognizerParams,
      ...textClassifierParams,
      use_angle_cls: true,
    };

    const textSystem = await TextSystem.create(textSystemParams);
    const results = await textSystem.execute(imageMat3Ch);
    expect(results[0]).toBeDefined();
    expect(results[1]).toBeDefined();
    expect(results[0]!.length).toEqual(results[1]!.length);
    console.log("results:", results);
  });
});
