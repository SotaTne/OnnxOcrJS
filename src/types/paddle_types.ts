// PaddleOCR Args TypeScript型定義

// 基本設定
export type USE_GPU = boolean;
export type USE_XPU = boolean;
export type USE_NPU = boolean;
export type USE_MLU = boolean;
export type USE_GCU = boolean;
export type IR_OPTIM = boolean;
export type USE_TENSORRT = boolean;
export type MIN_SUBGRAPH_SIZE = number;
export type PRECISION = "fp32" | "fp16" | "int8";
export type GPU_MEM = number;
export type GPU_ID = number;

// テキスト検出器パラメータ
export type IMAGE_DIR = string;
export type PAGE_NUM = number;
export type DET_ALGORITHM =
  | "DB"
  | "DB++"
  | "EAST"
  | "SAST"
  | "PSE"
  | "FCE"
  | "CT";
export type DET_MODEL_DIR = `${string}.onnx`;
export type DET_LIMIT_SIDE_LEN = number;
export type DET_LIMIT_TYPE = "max" | "min";
export type DET_BOX_TYPE = "quad" | "poly";

// DB/DB++パラメータ
export type DET_DB_THRESH = number;
export type DET_DB_BOX_THRESH = number;
export type DET_DB_UNCLIP_RATIO = number;
export type MAX_BATCH_SIZE = number;
export type USE_DILATION = boolean;
export type DET_DB_SCORE_MODE = "slow" | "fast";

// EASTパラメータ
export type DET_EAST_SCORE_THRESH = number;
export type DET_EAST_COVER_THRESH = number;
export type DET_EAST_NMS_THRESH = number;

// SASTパラメータ
export type DET_SAST_SCORE_THRESH = number;
export type DET_SAST_NMS_THRESH = number;

// PSEパラメータ
export type DET_PSE_THRESH = number;
export type DET_PSE_BOX_THRESH = number;
export type DET_PSE_MIN_AREA = number;
export type DET_PSE_SCALE = number;

// FCEパラメータ
export type SCALES = number[];
export type ALPHA = number;
export type BETA = number;
export type FOURIER_DEGREE = number;

// テキスト認識器パラメータ
export type REC_ALGORITHM =
  | "SVTR_LCNet"
  // | "CRNN"
  // | "SVTR_HGNet"
  // | "SRN"
  | "NRTR"
  | "RFL"
  | "RARE"
  | "ViTSTR";
export type REC_MODEL_DIR = `${string}.onnx`;
export type REC_IMAGE_INVERSE = boolean;
export type REC_IMAGE_SHAPE = string; // "3, 48, 320"形式
export type REC_IMAGE_SHAPE_NUMBER = number[]; // [3, 48, 320]
export type REC_BATCH_NUM = number;
export type MAX_TEXT_LENGTH = number;
export type REC_CHAR_DICT_PATH = string;
export type USE_SPACE_CHAR = boolean;
export type VIS_FONT_PATH = string;
export type DROP_SCORE = number;

// End-to-Endパラメータ
export type E2E_ALGORITHM = "PGNet" | string;
export type E2E_MODEL_DIR = string;
export type E2E_LIMIT_SIDE_LEN = number;
export type E2E_LIMIT_TYPE = "max" | "min";

// PGNetパラメータ
export type E2E_PGNET_SCORE_THRESH = number;
export type E2E_CHAR_DICT_PATH = string;
export type E2E_PGNET_VALID_SET = "totaltext" | string;
export type E2E_PGNET_MODE = "fast" | "slow";

// テキスト分類器パラメータ
export type USE_ANGLE_CLS = boolean;
export type CLS_MODEL_DIR = `${string}.onnx`;
export type CLS_IMAGE_SHAPE = string; // "3, 48, 192"形式
export type LABEL_LIST = string[];
export type CLS_BATCH_NUM = number;
export type CLS_THRESH = number;

// システム最適化パラメータ
export type ENABLE_MKLDNN = boolean | null;
export type CPU_THREADS = number;
export type USE_PDSERVING = boolean;
export type WARMUP = boolean;

// Super Resolutionパラメータ
export type SR_MODEL_DIR = string;
export type SR_IMAGE_SHAPE = string; // "3, 32, 128"形式
export type SR_BATCH_NUM = number;

// 出力・保存パラメータ
export type DRAW_IMG_SAVE_DIR = string;
export type SAVE_CROP_RES = boolean;
export type CROP_RES_SAVE_DIR = string;

// マルチプロセスパラメータ
export type USE_MP = boolean;
export type TOTAL_PROCESS_NUM = number;
export type PROCESS_ID = number;

// ログ・ベンチマークパラメータ
export type BENCHMARK = boolean;
export type SAVE_LOG_PATH = string;
export type SHOW_LOG = boolean;

// ONNXパラメータ
export type USE_ONNX = boolean;
export type ONNX_PROVIDERS = string[] | boolean;
export type ONNX_SESS_OPTIONS = any[] | boolean;

// 拡張機能パラメータ
export type RETURN_WORD_BOX = boolean;
