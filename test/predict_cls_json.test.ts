import { TextClassifier, TextClassifierParams } from "../src/predict_cls.js";
import { join } from "path";

import { beforeAll, describe, expect, it } from "vitest";
import { model_dir } from "./config.js";
import fs from "fs/promises";
import type cvReadyPromiseType from "@techstark/opencv-js";
import { ORT } from "../src/types/type.js";

import { matToList } from "../src/utils/func.js";

import ori_im_list from "./ori_im_list.json" with { type: "json" };
import ori_im_list_before_cls from "./ori_im_list_before_cls.json" with { type: "json" };
import { blake2bHex } from "blakejs";

let cv: Awaited<typeof cvReadyPromiseType>;
let ort: ORT;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  const nodeORT = require("onnxruntime-node");
  ort = nodeORT;
  cv = await cvReadyPromise;
});

describe("TextClassifier / TextDetector", () => {
  it("classifier: 簡単なケース", async () => {
    const cls_model_src = join(model_dir, "ppocrv5", "cls", "cls.onnx");
    const cls_model_buffer = await fs.readFile(cls_model_src);

    // cls

    const imageMatList = (ori_im_list_before_cls as number[][][][]).map(
      (ori_im_data) => {
        const h = ori_im_data.length;
        const w = ori_im_data[0].length;
        return cv.matFromArray(
          h,
          w,
          cv.CV_8UC3,
          Uint8Array.from(ori_im_data.flat(2)),
        );
      },
    );

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
    const classifier = await TextClassifier.create(textClassifierParams);
    const [cls_img_list, cls_res] = await classifier.execute(imageMatList);

    expect(cls_res).toBeDefined();
    expect(cls_res.length).toBe(imageMatList.length);
    expect(cls_img_list.length).toBe(cls_res.length);

    const cls_img_list_array = cls_img_list.map(
      (mat) => matToList(mat, cv) as number[][][],
    );

    expect(cls_img_list_array[0].slice(0, 10)).toEqual(
      (ori_im_list as number[][][][])[0].slice(0, 10),
    );
    expect(cls_img_list_array[1].slice(0, 10)).toEqual(
      (ori_im_list as number[][][][])[1].slice(0, 10),
    );
    // expect(cls_img_list_array[2].slice(0,10)).toEqual(((ori_im_list as number[][][][])[2]).slice(0,10));

    const cls_img_list_array_hash = blake2bHex(
      Uint8Array.from(cls_img_list_array.flat(3)),
      undefined,
    );
    const ori_img_list_true_hash = blake2bHex(
      Uint8Array.from((ori_im_list as number[][][][]).flat(3)),
      undefined,
    );

    expect(cls_img_list_array_hash).toEqual(ori_img_list_true_hash);
  });
});
