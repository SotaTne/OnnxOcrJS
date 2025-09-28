import { argmax, max } from "./utils/func.js";

export type BaseRecLabelDecodeParams = {
  character_str: string | null;
  use_space_char: boolean | null;
};

export class BaseRecLabelDecode {
  character_str: string;
  character: string[];
  use_space_char: boolean;

  constructor(params: BaseRecLabelDecodeParams) {
    this.use_space_char = params.use_space_char || false;
    let dict_character: string[];
    const dict: Record<string, number> = {};
    if (!params.character_str) {
      this.character_str = "0123456789abcdefghijklmnopqrstuvwxyz";
      dict_character = this.character_str.split("");
    } else {
      this.character_str = params.character_str
        .split("")
        .filter((c) => c !== " " && c !== "\n" && c !== "\r\n")
        .join("");
      if (this.use_space_char) this.character_str += " ";
      // Todo: reverse for arabic
      dict_character = this.character_str.split("");
    }
    for (const [i, char] of dict_character.entries()) {
      dict[char] = i;
    }
    this.character = dict_character;
  }

  pred_reverse(pred: string): string {
    const pred_re: string[] = [];
    let c_current = "";
    for (const c of pred) {
      if (!/[a-zA-Z0-9 :*\.\/%+-]/.test(c)) {
        if (c_current !== "") {
          pred_re.push(c_current);
        }
        pred_re.push(c);
        c_current = "";
      } else {
        c_current += c;
      }
    }
    if (c_current !== "") {
      pred_re.push(c_current);
    }
    return pred_re.reverse().join("");
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
    const batch_size = text_index.length;

    for (let batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      const seq = text_index[batch_idx]!;

      // np.ones(len(text_index[batch_idx]), dtype=bool)
      const selection: boolean[] = new Array(seq.length).fill(true);

      if (is_remove_duplicate) {
        for (let i = 1; i < seq.length; i++) {
          if (seq[i] === seq[i - 1]) selection[i] = false;
        }
      }

      for (const ignored of ignored_tokens) {
        for (let i = 0; i < seq.length; i++) {
          if (seq[i] === ignored) selection[i] = false;
        }
      }

      const char_list: string[] = [];
      const conf_list: number[] = [];

      for (let i = 0; i < seq.length; i++) {
        if (!selection[i]) continue;
        char_list.push(this.character[seq[i]!]!);
        conf_list.push(text_prob ? text_prob[batch_idx]![i]! : 1);
      }

      if (conf_list.length === 0) conf_list.push(0);

      let text = char_list.join("");

      // Todo: reverse for arabic
      // if (this.reverse) {
      //   text = this.pred_reverse(text);
      // }

      // np.mean(conf_list)
      const avg_conf = conf_list.reduce((a, b) => a + b, 0) / conf_list.length;

      result_list.push([text, avg_conf]);
    }

    return result_list;
  }

  get_ignored_tokens() {
    return [0];
  }
}

export class CTCLabelDecode extends BaseRecLabelDecode {
  constructor(params: BaseRecLabelDecodeParams) {
    super(params);
  }

  execute(preds: number[][][], label: number[][] | null = null) {
    // preds: shape [batch, seq, num_classes]
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
    return ["blank", ...dict_character];
  }
}
