import { build } from "esbuild";

const shared = {
  entryPoints: ["src/index.ts"], // ライブラリのエントリポイント
  bundle: true,                  // 依存をまとめる
  sourcemap: true,               // ソースマップ生成
  minify: true,                 // ライブラリなら通常 minify は不要
  target: "esnext",              // 出力ターゲット (必要なら es2018 などに変更)
  external: [                    // peerDependencies は bundle に含めない
    "@techstark/opencv-js",
    "onnxruntime-node",
  ],
};

await build({
  ...shared,
  format: "esm",
  outfile: "dist/index.mjs",
  platform: "browser",
  alias: {
    fs: "./stubs/fs.js",
    path: "./stubs/path.js"
  }
});

await build({
  ...shared,
  format: "cjs",
  outfile: "dist/index.cjs",
  platform: "node",
  external: ["fs", "path"],
});