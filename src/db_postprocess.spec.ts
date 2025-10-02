import { expect, describe, it, beforeAll } from "vitest";
import type cvReadyPromiseType from "@techstark/opencv-js";
import ndarray, { type NdArray } from "ndarray";
import { DBPostProcess, type DBPostProcessParams } from "./db_postprocess.js";
import { ndArrayToList } from "./utils/func.js";
import type { Point } from "./types/type.js";
import outs_dict from "./outs_dict.json" with { type: "json" };

let cv: Awaited<typeof cvReadyPromiseType>;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  cv = await cvReadyPromise;
});

// ---- 期待される結果 ----
const expectedResults = {
  binary_map_correct: [
    [
      [7, 17],
      [112, 17],
      [112, 102],
      [7, 102],
    ],
  ],
  binary_map_with_dilation: [
    [
      [3, 3],
      [77, 3],
      [77, 67],
      [3, 67],
    ],
  ],
  poly_box_detect: [
    [
      [104, 60],
      [107, 63],
      [109, 67],
      [110, 70],
      [110, 99],
      [109, 104],
      [106, 107],
      [102, 109],
      [99, 110],
      [70, 110],
      [65, 109],
      [62, 106],
      [60, 102],
      [59, 99],
      [59, 70],
      [60, 65],
      [63, 62],
      [67, 60],
      [70, 59],
      [99, 59],
    ],
    [
      [54, 10],
      [57, 13],
      [59, 17],
      [60, 20],
      [60, 49],
      [59, 54],
      [56, 57],
      [52, 59],
      [49, 60],
      [20, 60],
      [15, 59],
      [12, 56],
      [10, 52],
      [9, 49],
      [9, 20],
      [10, 15],
      [13, 12],
      [17, 10],
      [20, 9],
      [49, 9],
    ],
  ],
};

describe("DBPostProcess (large maps)", () => {
  it("binary_map_correct matches expected points", async () => {
    const params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: 0.3,
      box_thresh: 0.5,
      max_candidates: 1000,
      unclip_ratio: 2.0,
      use_dilation: false,
      score_mode: "fast",
      box_type: "quad",
      cv,
    };
    const H = 128,
      W = 128;
    const ndarrayData = new Float32Array(H * W).fill(0);
    for (let y = 40; y < 80; y++) {
      for (let x = 30; x < 90; x++) {
        ndarrayData[y * W + x] = 1.0;
      }
    }
    for (let x = 30; x < 90; x++) {
      ndarrayData[60 * W + x] = (x % 10) / 10.0;
    }

    const results = (await new DBPostProcess(params).execute(
      { maps: ndarray(ndarrayData, [1, 1, H, W]) },
      [[H, W, 1.0, 1.0]],
    )) as { points: NdArray }[];

    expect(ndArrayToList(results[0]!.points)).toEqual(
      expectedResults.binary_map_correct,
    );
  });

  it("binary_map_with_dilation matches expected points", async () => {
    const params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: 0.3,
      box_thresh: 0.7,
      max_candidates: 1000,
      unclip_ratio: 2.0,
      use_dilation: true,
      score_mode: "fast",
      box_type: "quad",
      cv,
    };
    const H = 128,
      W = 128;
    const ndarrayData = new Float32Array(H * W).fill(0);
    for (let y = 20; y < 50; y++) {
      for (let x = 20; x < 60; x++) {
        ndarrayData[y * W + x] = 1.0;
      }
      ndarrayData[y * W + 40] = (y % 15) / 15.0;
    }

    const results = (await new DBPostProcess(params).execute(
      { maps: ndarray(ndarrayData, [1, 1, H, W]) },
      [[H, W, 1.0, 1.0]],
    )) as { points: NdArray }[];

    expect(ndArrayToList(results[0]!.points)).toEqual(
      expectedResults.binary_map_with_dilation,
    );
  });

  it("poly_box_detect matches expected polygon", async () => {
    const params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: 0.2,
      box_thresh: 0.3,
      max_candidates: 1000,
      unclip_ratio: 1.5,
      use_dilation: false,
      score_mode: "fast",
      box_type: "poly",
      cv,
    };
    const H = 128,
      W = 128;
    const ndarrayData = new Float32Array(H * W).fill(0);

    // 左上の四角
    for (let y = 20; y < 50; y++) {
      for (let x = 20; x < 50; x++) {
        ndarrayData[y * W + x] = 1.0;
      }
    }

    // 右下の四角
    for (let y = 70; y < 100; y++) {
      for (let x = 70; x < 100; x++) {
        ndarrayData[y * W + x] = 1.0;
      }
    }

    const results = (await new DBPostProcess(params).execute(
      { maps: ndarray(ndarrayData, [1, 1, H, W]) },
      [[H, W, 1.0, 1.0]],
    )) as { points: Point[][] }[];
    expect(results[0]!.points).toEqual(expectedResults.poly_box_detect);
  });

  it("throws when shape_list length differs", async () => {
    const params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: 0.3,
      box_thresh: 0.7,
      max_candidates: 10,
      unclip_ratio: 2.0,
      use_dilation: false,
      score_mode: "fast",
      box_type: "quad",
      cv,
    };
    const H = 128,
      W = 128;
    const maps = ndarray(new Float32Array(2 * H * W), [2, 1, H, W]);

    await expect(() =>
      new DBPostProcess(params).execute({ maps }, [[H, W, 1.0, 1.0]]),
    ).rejects.toThrow(/shape_list length mismatch/);
  });

  it("実際の値での動作確認", async () => {
    const params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: 0.3,
      box_thresh: 0.6,
      max_candidates: 1000,
      unclip_ratio: 1.5,
      use_dilation: false,
      score_mode: "fast",
      box_type: "quad",
      cv,
    };
    const shape = [[579, 584, 0.99481865, 0.98630137]];
    const trueResult = [
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
    const mapList = outs_dict as number[][][][];
    const maps = ndarray(new Float32Array(mapList.flat(3)), [1, 1, 576, 576]);

    const results = (await new DBPostProcess(params).execute(
      { maps },
      shape as [number, number, number, number][],
    )) as { points: NdArray }[];

    const result = ndArrayToList(results[0]!.points) as number[][][];

    console.log(results);

    console.log(trueResult);

    expect(result.flat(Infinity)).toEqual(trueResult.flat(Infinity));
  });
});
