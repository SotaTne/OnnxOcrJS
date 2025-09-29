import type { Mat } from "@techstark/opencv-js";
import type cvReadyPromise from "@techstark/opencv-js";
import type { NdArray } from "ndarray";
import type * as webort from "onnxruntime-web";
import type * as nodeort from "onnxruntime-node";
import type * as reactnativeort from "onnxruntime-react-native";

export type Point = [number, number]; // [x, y]
export type Box = [Point, Point, Point, Point]; // 四角形を構成する4点
export type CV2 = Awaited<typeof cvReadyPromise>;
export type DataKeys = keyof Data;
export type DataValues = Data[keyof Data];
export type Data = MatData | NdArrayData;
export type MatData = {
  image: Mat;
  shape: [number, number, number, number] | null;
};
export type NdArrayData = {
  image: NdArray;
  shape: [number, number, number, number] | null;
};

export type BufferType = {
  size: 8 | 16 | 32 | 64;
  type: "int" | "uint" | "float";
};

export type ORT = typeof webort | typeof nodeort | typeof reactnativeort;

export type ORTSessionType = ORT["InferenceSession"];

export type ORTSessionArgsType = Parameters<ORTSessionType["create"]>;

export type ORTSessionReturnType = Awaited<
  ReturnType<ORTSessionType["create"]>
>;

export type ORTParametersOptionType = NonNullable<ORTSessionArgsType[1]>;

export type ORTBufferType = NonNullable<ORTSessionArgsType[0]>;

export type ORTTensorType = InstanceType<ORT["Tensor"]>;

export type ORTRunType = ORTSessionReturnType["run"];

export type ORTRunArgsType = Parameters<ORTRunType>;

export type ORTRunFeedsType = ORTRunArgsType[0];
export type ORTRunFetchesType = ORTRunArgsType[1];
export type ORTRunOptionsType = NonNullable<ORTRunArgsType[2]>;
