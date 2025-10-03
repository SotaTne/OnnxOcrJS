import { beforeAll, describe, it } from "vitest";
import { ONNXPaddleOCR } from "../src/onnx_paddleocr.js";
import { Jimp } from "jimp";
import { join } from "path";
import { model_dir, test_image_dir } from "./config.js";
import * as fs from "fs/promises";
import { CV2, ORT } from "../src/types/type.js";

let cv: CV2;
let ort: ORT;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  const nodeORT = require("onnxruntime-node");
  ort = nodeORT;
  cv = await cvReadyPromise;
});

describe("ONNXPaddleOCR", () => {
  it("source code image", async () => {
    const det_model_src = join(model_dir, "ppocrv5", "det", "det.onnx");
    const det_model_buffer = await fs.readFile(det_model_src);

    const image_src = join(test_image_dir, "enhancedscrollbarsearch.png");
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
    const ocr = new ONNXPaddleOCR({});
    const text_system = await ocr.init({
      det_model_array_buffer: det_model_buffer,
      cv: cv,
      ort: ort,
      cls_model_array_buffer: cls_model_buffer,
      rec_model_array_buffer: rec_model_buffer,
      rec_char_dict: charset,
    });
    const res = await ocr.ocr(text_system, imageMat3Ch, true, true, true);
    res.forEach((r) => {
      console.log(r);
    });
    // expect(res.length).toBeGreaterThan(0);
  });
});
