import http from "http";
import handler from "serve-handler";
import path from "path";
import { test, expect } from "@playwright/test";
import { fileURLToPath } from "url";

let server: http.Server;
let port: number;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.beforeAll(async () => {
  server = http.createServer((req, res) => {
    return handler(req, res, {
      public: path.resolve(__dirname, ".."), // ← プロジェクトルート想定
      cleanUrls: false,
      rewrites: [{ source: "/models/**", destination: "/models/$1" }],
    });
  });

  await new Promise<void>((resolve) => {
    server.listen(0, () => {
      port = (server.address() as any).port;
      console.log("Test server running at:", `http://localhost:${port}`);
      resolve();
    });
  });
});

test.afterAll(() => server.close());

test.setTimeout(2000000); // OCRが重いのでタイムアウトを長めに設定

test("run OCR demo", async ({ page }) => {
  page.on("console", (msg) => {
    console.log("BROWSER LOG:", msg.type(), msg.text());
  });
  page.on("pageerror", (err) => {
    console.error("BROWSER ERROR:", err);
  });

  await page.goto(`http://localhost:${port}/e2e/big_ocr_test.html`, {
    waitUntil: "domcontentloaded",
  });

  const resultLocator = page.locator("#result");
  await expect(resultLocator).toContainText("[", { timeout: 2000000 });

  const resultText = await resultLocator.textContent();
  expect(resultText).toBeTruthy();
  console.log("OCR Results:", resultText);
});
