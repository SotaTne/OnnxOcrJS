# Repository Guidelines
This guide keeps OnnxOcrJS contributors aligned on structure, tooling, and expectations when extending the ONNX-based PaddleOCR pipeline.

## Project Structure & Module Organization
Core TypeScript sources live under `src/`. `src/predict_system.ts` orchestrates detection (`predict_det.ts`), recognition (`predict_rec.ts`), and optional angle classification (`predict_cls.ts`). Shared geometry helpers sit in `src/utils/`, and reusable types are in `src/types/`. Pretrained weights and configs are in `models/` (e.g., `ppocrv5/`); treat them as read-only artifacts and document provenance when swapping files. The compiler writes build artifacts to `dist/`; never edit generated output directly.

## Build, Test, and Development Commands
- `pnpm install` — install dependencies pinned by `pnpm-lock.yaml`.
- `pnpm build` — run the TypeScript compiler (`tsc`) using `tsconfig.json`, emitting `.js` and declaration files.
- `pnpm test` — execute the Vitest test suite; append `--watch` when iterating.
Verify large model folders with `du -sh models/*` before committing to catch accidental bloat.

## Coding Style & Naming Conventions
Use 2-space indentation and keep files as ES modules. Apply `camelCase` for variables/functions, `PascalCase` for classes (`TextSystem`, `TextDetector`), and UPPER_SNAKE_CASE for constants. Prefer named exports, grouping external imports before relative ones. Honor strict typing—narrow `unknown` results, avoid `any`, and co-locate helper functions with their domain (`utils` for math/image transforms, `types` for structures).

## Testing Guidelines
Vitest is the primary framework. Co-locate new specs as `*.test.ts`/`*.spec.ts` next to the code under test (e.g., `src/utils/boxes.test.ts`, `src/utils/boxes.spec.ts`). Use descriptive `describe` and `it` names that reflect OCR scenarios, and cover threshold logic, box ordering, and angle classification branches. When OpenCV operations are heavy, stub them with lightweight fixtures housed in `src/utils/__fixtures__/` to keep tests fast.

## Commit & Pull Request Guidelines
Existing history follows Conventional Commits (`feat:`, `fix:`, `chore:`). Keep each commit focused, include relevant script or doc updates, and run `pnpm build`/`pnpm test` beforehand. PRs should describe behaviour changes, list the validation commands, link tracking issues, and highlight any model asset updates (include hashes or download instructions).

## Model Assets & Configuration Tips
Large ONNX files can overwhelm diffs; consider linking to external storage and documenting checksum/expected dimensions in the PR. Track preprocessing parameters (mean/scale, padding, angle settings) with the code change so integrators can reproduce results.
