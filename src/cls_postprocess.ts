import { argmax } from "./utils/func.js";

export type ClsPostProcessParams = {
  label_list?: string[] | null;
  key?: string | null;
};

export type DecodeOutput = Array<[string, number]>;

export class ClsPostProcess {
  label_list: string[] | null;
  key: string | null;

  constructor(params: ClsPostProcessParams = {}) {
    this.label_list = params.label_list ?? null;
    this.key = params.key ?? null;
  }

  execute(preds: number[][], predsShape: number[], label: null): DecodeOutput;
  execute(
    preds: number[][],
    predsShape: number[],
    label: number[]
  ): [DecodeOutput, DecodeOutput];

  execute(preds: number[][], predsShape: number[], label: number[] | null) {
    // preds: [batch, num_classes]
    let labelList: string[] = this.label_list === null ? [] : this.label_list;

    // デフォルトラベル（"0","1","2",...）を作る
    if (this.label_list === null) {
      const numClasses = predsShape[predsShape.length - 1]!; // 最後の次元が num_classes
      labelList = [];
      for (let idx = 0; idx < numClasses; idx++) {
        labelList[idx] = idx.toString();
      }
    }

    const predsIdx: number[] = argmax(preds, 1) as number[];

    let decode_out: [string, number][] = [];
    for (const [i, idx] of predsIdx.entries()) {
      decode_out.push([labelList[idx]!, preds[i]![idx]!]);
    }

    if (label === null) {
      return decode_out;
    }

    const new_label = label.map(
      (idx) => [labelList[idx]!, 1] as [string, number]
    );
    return [decode_out, new_label];
  }
}
