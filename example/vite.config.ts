import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// No cross-origin isolation here: the GPU renderer keeps its state on the device, so it needs no SharedArrayBuffer.
export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  // Relative paths so the build works under any GitHub Pages sub-path.
  base: "./",
  resolve: {
    // The example imports the package's build, exactly as an app installed from npm would.
    alias: { "@tinypoly/gpu-renderer": fileURLToPath(new URL("../dist/index.js", import.meta.url)) },
  },
  // naga finds its WebAssembly next to its own module; pre-bundling would move the module away from it.
  optimizeDeps: { exclude: ["naga-wasm"] },
});
