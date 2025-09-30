import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const model_dir = join(__dirname, "../models");
export const test_image_dir = join(__dirname, "./images");
