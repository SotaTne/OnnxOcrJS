// ClsPostProcess.spec.ts
import { describe, it, expect } from "vitest";
import { ClsPostProcess } from "./cls_postprocess.js";

describe("ClsPostProcess", () => {
  it("works with label_list (string[])", () => {
    const preds = [
      [0.1, 0.7, 0.2],
      [0.3, 0.3, 0.4],
    ]; // shape [2,3]
    const shape = [2, 3];
    const labelList = ["class0", "class1", "class2"];
    const proc = new ClsPostProcess({ label_list: labelList });
    const result = proc.execute(preds, shape, null);
    expect(result).toEqual([
      ["class1", 0.7],
      ["class2", 0.4],
    ]);
  });

  it("works without label_list (default to index string)", () => {
    const preds = [
      [0.1, 0.7, 0.2],
      [0.3, 0.3, 0.4],
    ];
    const shape = [2, 3];
    const proc = new ClsPostProcess({});
    const result = proc.execute(preds, shape, null);
    expect(result).toEqual([
      ["1", 0.7],
      ["2", 0.4],
    ]);
  });

  it("returns both decode_out and label when label is provided", () => {
    const preds = [
      [0.1, 0.7, 0.2],
      [0.3, 0.3, 0.4],
    ];
    const shape = [2, 3];
    const labelList = ["class0", "class1", "class2"];
    const proc = new ClsPostProcess({ label_list: labelList });

    const labels = [2, 0];
    const [result, labelResult] = proc.execute(preds, shape, labels);

    expect(result).toEqual([
      ["class1", 0.7],
      ["class2", 0.4],
    ]);
    expect(labelResult).toEqual([
      ["class2", 1],
      ["class0", 1],
    ]);
  });
});
