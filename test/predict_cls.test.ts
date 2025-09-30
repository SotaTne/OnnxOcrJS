import { describe, expect, it } from "vitest";
import { CTCLabelDecode } from "../src/rec_postprocess.js";

describe("CTCLabelDecode / BaseRecLabelDecode", () => {
  it("decode: 重複除去 + 無視トークン(0) を適用し、平均信頼度を返す", () => {
    // 使う文字集合を小さくして可読性を上げる
    const dec = new CTCLabelDecode({
      character_str: "abcd",
      use_space_char: false,
    });
  });
});
