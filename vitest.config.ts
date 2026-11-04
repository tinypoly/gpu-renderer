import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    // Packing a scene and walking a BVH over every triangle is slower than the default budget on a cold machine.
    testTimeout: 30000,
  },
});
