import { FRAME_LAYOUT, FRAME_WORDS } from "./frameLayout.js";
import { createShaderModule } from "./shaderModule.js";
import { programsWgsl, type ShaderPrograms } from "./shaderProgram.js";
import type { SceneBuffers } from "./sceneUpload.js";
import { writeStruct } from "./structLayout.js";
import { BSDF_WGSL } from "./wgsl/bsdf.wgsl.js";
import { BVH_WGSL } from "./wgsl/bvh.wgsl.js";
import { COMMON_WGSL } from "./wgsl/common.wgsl.js";
import { ENVIRONMENT_WGSL } from "./wgsl/environment.wgsl.js";
import { LIGHTS_WGSL } from "./wgsl/lights.wgsl.js";
import { PATH_TRACE_WGSL } from "./wgsl/pathTrace.wgsl.js";
import { RANDOM_WGSL } from "./wgsl/random.wgsl.js";
import { SAMPLING_WGSL } from "./wgsl/sampling.wgsl.js";
import { SHADER_SUPPORT_WGSL } from "./wgsl/shaderSupport.wgsl.js";
import { SURFACE_WGSL } from "./wgsl/surface.wgsl.js";
import { TEXTURE_WGSL } from "./wgsl/texture.wgsl.js";

/** One dispatch worth of state: which band, which samples, and everything the kernel reads about the frame. */
export type FrameUniforms = Record<keyof typeof FRAME_LAYOUT.offsets & string, number | boolean | ArrayLike<number>>;

/**
 * The whole kernel, in the order WGSL needs: every stage may only call what comes before it, as there are no forward
 * declarations. The scene's translated shaders sit between the texture code they call and the surface code that calls
 * them.
 */
export function pathTraceSource(programs: ShaderPrograms | null = null): string {
  return COMMON_WGSL + RANDOM_WGSL + BVH_WGSL + SAMPLING_WGSL + TEXTURE_WGSL + SHADER_SUPPORT_WGSL
    + programsWgsl(programs) + ENVIRONMENT_WGSL + SURFACE_WGSL + BSDF_WGSL + LIGHTS_WGSL + PATH_TRACE_WGSL;
}

const FRAME_BYTES = FRAME_WORDS * 4;

const WORKGROUP = 8;

export class PathTracePass {
  private pipeline: GPUComputePipeline;
  private readonly uniforms: GPUBuffer;
  private readonly values = new ArrayBuffer(FRAME_BYTES);
  private bindGroup: GPUBindGroup | null = null;
  private programKey = "";

  constructor(private readonly device: GPUDevice) {
    this.pipeline = this.build(null);
    this.uniforms = device.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Recompiles the kernel when the scene's shaders changed; the same set of programs keeps the pipeline it has. */
  setPrograms(programs: ShaderPrograms | null) {
    const key = programs?.key ?? "";

    if (key === this.programKey) return;

    this.programKey = key;
    this.pipeline = this.build(programs);
    this.bindGroup = null;
  }

  private build(programs: ShaderPrograms | null): GPUComputePipeline {
    const module = createShaderModule(this.device, "path trace", pathTraceSource(programs));

    return this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  }

  /** Rebinds after the scene or the accumulation buffer was replaced. */
  setBindings(scene: SceneBuffers, accumulation: GPUBuffer) {
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: scene.positions } },
        { binding: 2, resource: { buffer: scene.attributes } },
        { binding: 3, resource: { buffer: scene.indices } },
        { binding: 4, resource: { buffer: scene.materials } },
        { binding: 5, resource: { buffer: scene.nodes } },
        { binding: 6, resource: { buffer: scene.lights } },
        { binding: 7, resource: { buffer: accumulation } },
        { binding: 8, resource: { buffer: scene.textures } },
      ],
    });
  }

  encode(encoder: GPUCommandEncoder, frame: FrameUniforms) {
    if (!this.bindGroup) throw new Error("GPU renderer: the path tracing pass has no scene bound.");

    new Uint32Array(this.values).fill(0);
    writeStruct(FRAME_LAYOUT, this.values, 0, frame);
    this.device.queue.writeBuffer(this.uniforms, 0, this.values);

    const pass = encoder.beginComputePass({ label: "path trace" });
    const size = frame.bandSize as ArrayLike<number>;

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(size[0] / WORKGROUP), Math.ceil(size[1] / WORKGROUP));
    pass.end();
  }
}
