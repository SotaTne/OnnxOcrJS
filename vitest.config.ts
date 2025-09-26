// vitest.config.ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["**/*.{test,spec}.ts?(x)", "test/**/*.ts?(x)"],
    deps: {
      interopDefault: true, // デフォルトエクスポートを正しく解釈
    },
    testTimeout: 20000,
  },
});
