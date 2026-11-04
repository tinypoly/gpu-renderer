// Everything this renderer takes from @tinypoly/cpu-renderer passes through here, so a version bump has one place to
// audit. The sibling package owns the scene snapshot, the settings and the tone mapping; this one owns the GPU.

export {
  DEFAULT_RENDER_ENVIRONMENT,
  DEFAULT_RENDER_SETTINGS,
  mergeSettings,
  sceneSignature,
  serializeCamera,
  serializeScene,
} from "@tinypoly/cpu-renderer";

// The renderer's own public shapes, so a host can swap one renderer for the other without touching its types.
export type {
  Bucket,
  CameraState,
  Listener,
  RenderBackgroundMode,
  RenderCamera,
  RenderEnvironment,
  RenderImage,
  RenderPhase,
  RenderPixels,
  RenderScene,
  RenderSettings,
  RenderSettingsInput,
  RenderToneMapping,
  SerializedScene,
  SerializedSceneResult,
} from "@tinypoly/cpu-renderer";

// The engine entry point is the lower level one, and the sibling's README warns that it moves more often.
export { EventEmitter } from "@tinypoly/cpu-renderer/engine";

export { encodeColor, srgbToLinear } from "@tinypoly/cpu-renderer/engine";

// The GLSL interpreter runs vertex shaders here, where a vertex's position has to be known before the hierarchy is
// built; the same interpreter bakes a procedural sky and a material's compute simulation, exactly as the CPU renderer
// does. Only fragment shaders are translated to WGSL, because only they run per sample.
export {
  CpuComputePipeline,
  CpuEnvironment,
  CpuShader,
  matrixValue,
  sampleTexture,
} from "@tinypoly/cpu-renderer/engine";

export type {
  SerializedFog,
  SerializedLight,
  SerializedMaterial,
  SerializedMesh,
  SerializedShader,
  SerializedTexture,
  UniformValue,
} from "@tinypoly/cpu-renderer/engine";
