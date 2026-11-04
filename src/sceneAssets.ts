import { environmentFromImage, type PackedEnvironment } from "./environment.js";
import { compileShaderPrograms, type ShaderPrograms } from "./shaderProgram.js";
import { CpuComputePipeline, CpuEnvironment } from "./upstream.js";
import type { SerializedMaterial, SerializedScene, SerializedTexture, UniformValue } from "./upstream.js";

/**
 * What a scene needs before it can be packed, beyond the snapshot itself: the outputs of the compute simulations its
 * materials declare, its procedural sky baked into an image, and its shaders translated. All of it depends on the scene
 * alone, so it is prepared once per scene and kept while only the camera or the settings change.
 */
export interface SceneAssets {
  /** The snapshot with every simulation's outputs bound to the uniforms that read them. */
  scene: SerializedScene;
  programs: ShaderPrograms;
  sky: ProceduralSkyEnvironment | null;
}

export interface ProceduralSkyEnvironment {
  environment: PackedEnvironment;
  /** The strength the sky lights the scene with; its background ignores it. */
  intensity: number;
  /** Whether the sky paints the background at all. */
  background: boolean;
  uniforms: Record<string, UniformValue>;
}

/** How wide the sky is baked. Its foreground, the sharp part, is evaluated per pixel instead. */
const SKY_WIDTH = 256;

/** Runs the CPU renderer's compute simulations, as its scene pipeline does, and binds their outputs. */
function bakeComputations(source: SerializedScene): SerializedScene {
  const copies = new Map<SerializedMaterial, SerializedMaterial>();
  const textures: SerializedTexture[] = [...source.textures];
  const known = new Set(textures.map(texture => texture.id));
  const computed = new Map<string, Record<string, SerializedTexture>>();

  const copy = (material: SerializedMaterial) => {
    let result = copies.get(material);

    if (result) return result;

    result = material.shader
      ? { ...material, shader: { ...material.shader, uniforms: { ...material.shader.uniforms } } }
      : material;

    copies.set(material, result);

    const { compute } = material;

    if (compute && result.shader) {
      let outputs = computed.get(compute.id);

      if (!outputs) {
        const pipeline = new CpuComputePipeline(compute.pipeline, compute.id, compute.data ?? {});

        if (compute.snapshot) pipeline.restore(compute.snapshot);
        else for (const _step of pipeline.tick(material.time ?? 0, 1 / 60)) { /* runs to completion */ }

        outputs = pipeline.outputs();
        computed.set(compute.id, outputs);
      }

      for (const [uniform, texture] of Object.entries(outputs)) {
        result.shader.uniforms[uniform] = { texture: texture.id };

        if (!known.has(texture.id)) {
          textures.push(texture);
          known.add(texture.id);
        }
      }
    }

    return result;
  };

  return {
    ...source,
    textures,
    meshes: source.meshes.map(mesh => ({
      ...mesh,
      materials: mesh.materials.map(material => (material ? copy(material) : material)),
    })),
  };
}

/** Bakes a procedural sky with the CPU renderer's own interpreter, at the size its environment uses. */
function bakeSky(scene: SerializedScene): ProceduralSkyEnvironment | null {
  const sky = scene.sky;

  if (!sky) return null;

  const generator = CpuEnvironment.fromProcedural(sky, SKY_WIDTH);
  let step = generator.next();

  while (!step.done) step = generator.next();

  // The snapshot is the environment's public face; its texture is the baked panorama.
  const texture = step.value.snapshot().texture;

  if (!texture || !(texture.data instanceof Float32Array)) return null;

  const environment = environmentFromImage(texture.data, texture.width, texture.height, "procedural-sky");

  if (!environment) return null;

  return { environment, intensity: sky.intensity, background: sky.background, uniforms: sky.uniforms };
}

export async function prepareSceneAssets(source: SerializedScene): Promise<SceneAssets> {
  const scene = source.meshes.some(mesh => mesh.materials.some(material => material?.compute))
    ? bakeComputations(source)
    : source;

  const sky = bakeSky(scene);
  const programs = await compileShaderPrograms(scene, scene.sky?.foregroundShader);

  return { scene, programs, sky };
}
