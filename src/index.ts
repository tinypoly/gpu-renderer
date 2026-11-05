// The public API: path trace a Three.js scene on the GPU, in WGSL compute shaders. `attachCanvas` puts the image on a
// canvas while it converges; `render()` resolves with the finished pixels.

export { isWebGpuAvailable, requestRenderDevice, requestWebGpuSupport } from "./device.js";
export type { RenderDevice, RenderDeviceOptions, WebGpuSupport } from "./device.js";

export { GpuRenderer, isAbortError } from "./renderer.js";

export type {
  GpuRendererEvents,
  GpuRendererOptions,
  RenderCamera,
  RenderImage,
  RenderPhase,
  RenderPixels,
  RenderScene,
} from "./renderer.js";

export { attachCanvas } from "./canvas.js";

export { DEFAULT_DENOISER } from "./denoise.js";
export { DEFAULT_OIDN_WEIGHTS } from "./oidnPass.js";
export type { DenoiserAlgorithm, DenoiserSettings } from "./denoise.js";

// The settings only this renderer reads, beside the CPU renderer's, and the defaults it renders with.
export {
  DEFAULT_GPU_RENDER_SETTINGS,
  effectiveExposure,
  mergeGpuSettings,
  physicalExposure,
  whiteBalanceMatrix,
} from "./settings.js";

export type { GpuRenderSettings, GpuRenderSettingsInput } from "./settings.js";
export { subsurfaceMedium } from "./subsurface.js";
export type { SubsurfaceInput, SubsurfaceMaterial, SubsurfaceMedium } from "./subsurface.js";

export type { Bucket, Listener } from "./upstream.js";

// Scene packing, published because the sizes it produces are what a host checks against a device's limits.
export { packGeometry } from "./geometry.js";
export type { PackedGeometry } from "./geometry.js";
export { buildBvh, MAX_BVH_DEPTH } from "./bvh.js";
export type { PackedBvh } from "./bvh.js";
// The BVH worker: what `createBvhWorker` has to return, and the worker this package starts by default.
export { BVH_WORKER_MIN_TRIANGLES, createDefaultBvhWorker, respondToBvhRequest } from "./bvhBuilder.js";
export type { BvhRequest, BvhResponse, BvhWorker, BvhWorkerFactory } from "./bvhBuilder.js";
export { sceneByteLength } from "./sceneUpload.js";
export type { SceneByteLength } from "./sceneUpload.js";
export { LIGHT_KIND, packLights } from "./lights.js";
export type { PackedLights } from "./lights.js";
export { MATERIAL_LIT, MATERIAL_PBR, materialProperties } from "./materialLayout.js";
export type { MaterialMap, MaterialProperties } from "./materialLayout.js";
export { NO_TEXTURE, packTextures } from "./texturePack.js";
export type { PackedTextures, TextureExtras } from "./texturePack.js";
export { loadEnvironment } from "./environment.js";
export type { PackedEnvironment } from "./environment.js";

// Settings, the environment and the scene snapshot come from the CPU renderer, unchanged.
export {
  DEFAULT_RENDER_ENVIRONMENT,
  DEFAULT_RENDER_SETTINGS,
  mergeSettings,
  sceneSignature,
  serializeCamera,
  serializeScene,
} from "./upstream.js";

export type {
  CameraState,
  RenderBackgroundMode,
  RenderEnvironment,
  RenderSettings,
  RenderSettingsInput,
  RenderToneMapping,
  SerializedScene,
  SerializedSceneResult,
} from "./upstream.js";
