import { argmax, max } from "./utils/func.js";

export type BaseRecLabelDecodeParams = {
  character_str: string | null;
  use_space_char: boolean;
};

export class BaseRecLabelDecode {
  character: string[];
  dict: Record<string, number>;
  use_space_char: boolean;

  constructor(params: BaseRecLabelDecodeParams) {
    this.use_space_char = params.use_space_char ?? false;

    let dict_character: string[];

    if (!params.character_str) {
      dict_character = "0123456789abcdefghijklmnopqrstuvwxyz".split("");
    } else {
      if (params.character_str.includes("\n")) {
        // ファイル形式（1行ごと）
        dict_character = params.character_str
          .split(/\r?\n/)
          .map((c) => (c === " " ? "" : c))
          .filter((c) => c.length > 0);
      } else {
        // 単なる文字列 → 1文字ずつ
        dict_character = params.character_str.split("");
      }

      if (this.use_space_char) {
        dict_character.push(" ");
      }
    }

    dict_character = this.add_special_char(dict_character);

    this.character = dict_character;
    this.dict = {};
    for (const [i, ch] of dict_character.entries()) {
      this.dict[ch] = i;
    }
  }

  add_special_char(dict_character: string[]): string[] {
    return dict_character;
  }

  decode(
    text_index: number[][],
    text_prob: number[][] | null,
    is_remove_duplicate = false
  ): [string, number][] {
    const result_list: [string, number][] = [];
    const ignored_tokens = this.get_ignored_tokens();

    for (let b = 0; b < text_index.length; b++) {
      const seq = text_index[b]!;
      const selection = new Array(seq.length).fill(true);

      // Python版と同じ: 連続重複の除去
      if (is_remove_duplicate) {
        for (let i = 1; i < seq.length; i++) {
          if (seq[i] === seq[i - 1]) selection[i] = false;
        }
      }

      // 無視トークン（blank=0）の除去
      for (let i = 0; i < seq.length; i++) {
        if (ignored_tokens.includes(seq[i]!)) {
          selection[i] = false;
        }
      }

      const char_list: string[] = [];
      const conf_list: number[] = [];

      for (let i = 0; i < seq.length; i++) {
        if (!selection[i]) continue;
        const idx = seq[i]!;
        char_list.push(this.character[idx] ?? "");
        conf_list.push(text_prob ? text_prob[b]![i]! : 1);
      }

      if (conf_list.length === 0) conf_list.push(0);

      const text = char_list.join("");
      const avg_conf = conf_list.reduce((a, b) => a + b, 0) / conf_list.length;
      result_list.push([text, avg_conf]);
    }

    return result_list;
  }

  get_ignored_tokens(): number[] {
    return [0]; // for CTC blank
  }

  // Python版にある pred_reverse を追加
  pred_reverse(pred: string): string {
    const pred_re: string[] = [];
    let c_current = "";
    for (const c of pred) {
      if (!/[a-zA-Z0-9 :*./%+-]/.test(c)) {
        if (c_current !== "") pred_re.push(c_current);
        pred_re.push(c);
        c_current = "";
      } else {
        c_current += c;
      }
    }
    if (c_current !== "") pred_re.push(c_current);
    return pred_re.reverse().join("");
  }
}

export class CTCLabelDecode extends BaseRecLabelDecode {
  constructor(params: BaseRecLabelDecodeParams) {
    super(params);
  }

  execute(preds: number[][][], label: null): [string, number][];

  execute(
    preds: number[][][],
    label: number[][]
  ): [[string, number][], [string, number][]];

  execute(preds: number[][][], label: number[][] | null = null) {
    // preds: [batch, seq, num_classes]
    const preds_idx: number[][] = argmax(preds, 2) as number[][];
    const preds_prob: number[][] = max(preds, 2) as number[][];

    const text = this.decode(preds_idx, preds_prob, true);

    if (label == null) {
      return text;
    }
    const lab = this.decode(label, null);
    return [text, lab];
  }

  add_special_char(dict_character: string[]): string[] {
    // Python版と完全一致: 先頭に blank を追加
    return ["blank", ...dict_character];
  }
}
