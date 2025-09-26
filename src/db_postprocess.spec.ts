import { expect, describe, it, beforeAll } from "vitest";
import type cvReadyPromiseType from "@techstark/opencv-js";
import ndarray from "ndarray";
import {
  DBPostProcess,
  type DBPostProcessParams,
  type OutDict,
} from "./db_postprocess.js";

let cv: Awaited<typeof cvReadyPromiseType>;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  cv = await cvReadyPromise;
});

describe("DBPostProcess", () => {
  it("processes a binary map correctly", async () => {
    const params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: 0.3,
      box_thresh: 0.7,
      max_candidates: 1000,
      unclip_ratio: 2.0,
      use_dilation: false,
      score_mode: "fast",
      box_type: "quad",
      cv,
    };

    const dBPostProcess = new DBPostProcess(params);

    const imageNumber = 1; // ← 1枚だけでテスト
    const channels = 1;
    const height = 3;
    const width = 8;

    // (N, C, H, W) の配列を作る
    const ndarrayData = new Float32Array(
      imageNumber * channels * height * width
    );

    // 中央に "1" の領域を作成（白い矩形っぽくする）
    // shape: (1,1,3,8)
    // row=1 に [0,1,1,1,0,0,0,0]
    ndarrayData.set([
      // batch=0, channel=0, row=0
      0, 0, 0, 0, 0, 0, 0, 0,
      // row=1
      0, 1, 1, 1, 0, 0, 0, 0,
      // row=2
      0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    const outsDict: OutDict = {
      maps: ndarray(ndarrayData, [imageNumber, channels, height, width]),
    };

    // shape_list は (src_h, src_w, ratio_h, ratio_w)
    const shape_list: [number, number, number, number][] = [
      [height, width, 1.0, 1.0],
    ];

    const results = await dBPostProcess.execute(outsDict, shape_list);

    // 検出結果の形式確認
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(1);
    expect(results[0]).toHaveProperty("points");

    // points は ndarray またはリスト形式のはず
    const points = results[0]!.points;
    expect(points).toBeTruthy();

    // 矩形が1つ以上検出されていることを期待
    if ("shape" in points) {
      expect(points.shape[1]).toBe(4); // (N,4,2) のはず
      expect(points.shape[2]).toBe(2);
    }
  });
});
