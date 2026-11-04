import { environmentFromImage, type PackedEnvironment } from "./environment.js";
import { compileShaderPrograms, type ShaderPrograms } from "./shaderProgram.js";
import { bakeSkyOnDevice, compileSkyProgram } from "./skyBake.js";
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
  /** The baked sky, kept by the renderer so a scene with the same sky does not bake it again. */
  skyBake: { key: string; baked: Promise<ProceduralSkyEnvironment | null> } | null;
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

type ProceduralSky = NonNullable<SerializedScene["sky"]>;

/** Bakes a procedural sky with the CPU renderer's own interpreter: the fallback, when the device cannot run it. */
function bakeSkyOnCpu(sky: ProceduralSky): Float32Array | null {
  const generator = CpuEnvironment.fromProcedural(sky, SKY_WIDTH);
  let step = generator.next();

  while (!step.done) step = generator.next();

  // The snapshot is the environment's public face; its texture is the baked panorama.
  const texture = step.value.snapshot().texture;

  return texture && texture.data instanceof Float32Array ? texture.data : null;
}

/**
 * Bakes a procedural sky at the size its environment uses. On a device the sky's shader is translated and run there,
 * over every texel at once; the CPU renderer's interpreter, which runs it texel by texel, is what is left when there is
 * no device or the shader does not translate.
 */
async function bakeSky(sky: ProceduralSky, device?: GPUDevice): Promise<ProceduralSkyEnvironment | null> {
  let radiance: Float32Array | null = null;

  if (device) {
    try {
      radiance = await bakeSkyOnDevice(device, await compileSkyProgram(sky.fragmentShader), sky.uniforms, SKY_WIDTH);
    } catch(error) {
      console.warn("GPU renderer: the procedural sky is baked on the CPU, its shader did not run on the device.", error);
    }
  }

  radiance ??= bakeSkyOnCpu(sky);

  if (!radiance) return null;

  const environment = environmentFromImage(radiance, SKY_WIDTH, SKY_WIDTH / 2, "procedural-sky");

  if (!environment) return null;

  return { environment, intensity: sky.intensity, background: sky.background, uniforms: sky.uniforms };
}

/** What decides a baked sky: the same shader with the same uniforms bakes the same panorama. */
export function skyKey(sky: ProceduralSky | undefined): string {
  return sky ? JSON.stringify([sky.fragmentShader, sky.uniforms]) : "";
}

export interface SceneAssetOptions {
  /** Where the sky is baked; without one, on the CPU. */
  device?: GPUDevice;
  /** A sky baked for an earlier scene, reused when its key matches this scene's. */
  sky?: { key: string; baked: Promise<ProceduralSkyEnvironment | null> } | null;
}

export async function prepareSceneAssets(
  source: SerializedScene, options: SceneAssetOptions = {}): Promise<SceneAssets> {
  const scene = source.meshes.some(mesh => mesh.materials.some(material => material?.compute))
    ? bakeComputations(source)
    : source;

  const key = skyKey(scene.sky);
  const reused = options.sky && options.sky.key === key ? options.sky.baked : null;
  const baked = reused ?? (scene.sky ? bakeSky(scene.sky, options.device) : Promise.resolve(null));
  const [sky, programs] = await Promise.all([baked, compileShaderPrograms(scene, scene.sky?.foregroundShader)]);
  // The sky the scene uses, with its baked panorama, for the next scene to reuse.
  const skyBake = scene.sky ? { key, baked } : null;

  // The panorama comes from the shader and its uniforms; how strongly it lights and whether it paints the background
  // are this scene's own.
  const current = sky && scene.sky
    ? { ...sky, intensity: scene.sky.intensity, background: scene.sky.background, uniforms: scene.sky.uniforms }
    : null;

  return { scene, programs, sky: current, skyBake };
}
