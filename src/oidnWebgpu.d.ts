// The package's lib/ imports its modules without extensions, which only a bundler resolves; its dist/ is one ES module
// that a browser and Node both load, with the same exports as lib/main.
declare module "oidn-webgpu/dist/oidn.js" {
  export * from "oidn-webgpu";
}
