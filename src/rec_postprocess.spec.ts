// rec_postprocess.spec.ts
import { describe, expect, it } from "vitest";
import { CTCLabelDecode } from "./rec_postprocess.js";

/**
 * ヘルパー: argmax/max が参照する 3 次元テンソル [B, T, C] を生成
 * - idxs[t] が取りたいクラス index
 * - probs[t] がそのタイムステップの最大値（=confidence）
 */
function makePreds(
  idxs: number[],
  probs: number[] | undefined,
  numClasses: number
): number[][][] {
  const T = idxs.length;
  const batchOne: number[][] = [];
  for (let t = 0; t < T; t++) {
    const row = new Array(numClasses).fill(0);
    row[idxs[t]!] = probs?.[t] ?? 1;
    batchOne.push(row);
  }
  return [batchOne]; // [B=1, T, C]
}

describe("CTCLabelDecode / BaseRecLabelDecode", () => {
  it("decode: 重複除去 + 無視トークン(0) を適用し、平均信頼度を返す", () => {
    // 使う文字集合を小さくして可読性を上げる
    const dec = new CTCLabelDecode({
      character_str: "abcd",
      use_space_char: false,
    });

    // text_index: [[1, 1, 0, 2, 2, 3]]
    // - 連続重複の 2 個目以降は落ちる
    // - 0 は無視トークンとして落ちる
    const text_index = [[1, 1, 0, 2, 2, 3]];
    const text_prob = [[0.9, 0.8, 0.1, 0.7, 0.5, 0.6]];

    const out = dec.decode(text_index, text_prob, true);
    // 残る index は [1, 2, 3] => 文字は ["b","c","d"]
    // conf は [0.9, 0.7, 0.6] の平均 = 2.2 / 3
    expect(out).toHaveLength(1);
    expect(out[0]![0]).toBe("bcd");
    expect(out[0]![1]).toBeCloseTo(2.2 / 3, 6);
  });

  it("decode: すべて無視トークン(0) の場合は空文字 & 信頼度0", () => {
    const dec = new CTCLabelDecode({
      character_str: "abcd",
      use_space_char: false,
    });
    const out = dec.decode([[0, 0, 0]], [[0.1, 0.2, 0.3]], true);
    expect(out).toEqual([["", 0]]);
  });

  it("decode: use_space_char=true で末尾に空白が追加される（index=2）", () => {
    const dec = new CTCLabelDecode({
      character_str: "ab",
      use_space_char: true,
    });
    // dec.character は ["a","b"," "] の順になる
    // 無視トークンは index=0（= "a"）なので、[1,2] -> "b" + " " が残る
    const out = dec.decode([[1, 2]], null, false);
    expect(out).toEqual([["b ", 1]]); // text_prob=null なので conf は 1 の平均
  });

  it("pred_reverse: 記号で分割し、英数塊をまとめて逆順に並べ替える", () => {
    const dec = new CTCLabelDecode({
      character_str: "abcd",
      use_space_char: false,
    });
    // pred_reverse は BaseRecLabelDecode のメソッド（private ではない）
    // 許容文字 [a-zA-Z0-9 :*\.\/%+-] の連続を一塊、その他は1文字で分割 → 逆順結合
    const reversed = (dec as any).pred_reverse("abc☆def 12/34");
    expect(reversed).toBe("def 12/34☆abc");
  });

  it("execute: label なし → [ [text, conf] ] を返す", () => {
    const dec = new CTCLabelDecode({
      character_str: "abcd",
      use_space_char: false,
    });

    // 4 クラス想定（"a","b","c","d"）。argmax が [1,1,2,3] になるように行を作る
    const idxs = [1, 1, 2, 3];
    const probs = [0.9, 0.8, 0.7, 0.6]; // max() が拾う値
    const preds = makePreds(idxs, probs, 4);

    // execute は is_remove_duplicate=true で decode する
    // -> 重複の 2 個目 (idx=1 の2つ目) は落ちる。残り [1,2,3] => "bcd"
    // conf は [0.9, 0.7, 0.6] の平均 = 2.2 / 3
    const out = dec.execute(preds, null) as [string, number][];
    expect(out).toHaveLength(1);
    expect(out[0]![0]).toBe("bcd");
    expect(out[0]![1]).toBeCloseTo(2.2 / 3, 6);
  });

  it("execute: label あり → [ [ [text, conf] ], [ [labelText, 1] ] ] を返す", () => {
    const dec = new CTCLabelDecode({
      character_str: "abcd",
      use_space_char: false,
    });

    // 予測
    const idxs = [2, 2, 3]; // -> "cd"（重複除去で 2 の2個目は落ちる）
    const probs = [0.4, 0.5, 0.9]; // conf -> [0.4, 0.9] の平均 = 0.65
    const preds = makePreds(idxs, probs, 4);

    // ラベル（decode(label, null) は is_remove_duplicate=false で conf は全て1）
    const label = [[1, 1, 3]]; // "bbd"
    const out = dec.execute(preds, label) as [
      [string, number][],
      [string, number][]
    ];

    // 予測側
    expect(out[0]).toHaveLength(1);
    expect(out[0][0]![0]).toBe("cd");
    expect(out[0][0]![1]).toBeCloseTo((0.4 + 0.9) / 2, 6);

    // ラベル側（conf は 1 の平均 = 1）
    expect(out[1]).toHaveLength(1);
    expect(out[1][0]![0]).toBe("bbd");
    expect(out[1][0]![1]).toBe(1);
  });

  it("decode: 複数バッチも処理できる", () => {
    const dec = new CTCLabelDecode({
      character_str: "wxyz",
      use_space_char: false,
    });
    const text_index = [
      [1, 1, 2, 2, 3], // -> 重複除去 + 0 無視なし => "xyz"
      [0, 1, 1, 3], // -> 0 は無視、重複除去で 2つ目の1は落ちて "yz"
    ];
    const text_prob = [
      [0.5, 0.4, 0.6, 0.6, 0.7], // 採用 [0.5, 0.6, 0.7] -> 平均 0.6
      [0.2, 0.9, 0.8, 0.4], // 採用 [0.9, 0.4]     -> 平均 0.65
    ];

    const out = dec.decode(text_index, text_prob, true);
    expect(out).toHaveLength(2);

    expect(out[0]![0]).toBe("xyz");
    expect(out[0]![1]).toBeCloseTo(0.6, 6);

    expect(out[1]![0]).toBe("xz");
    expect(out[1]![1]).toBeCloseTo((0.9 + 0.4) / 2, 6);
  });
});
