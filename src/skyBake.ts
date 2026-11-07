import { compileProgram, type ShaderProgram } from "./shaderProgram.js";
import { createShaderModule } from "./shaderModule.js";
import type { UniformValue } from "./upstream.js";
import { traversalConstants } from "./pathTracePass.js";
import { BVH_WGSL } from "./wgsl/bvh.wgsl.js";
import { COMMON_WGSL } from "./wgsl/common.wgsl.js";
import { RANDOM_WGSL } from "./wgsl/random.wgsl.js";
import { SAMPLING_WGSL } from "./wgsl/sampling.wgsl.js";
import { SHADER_SUPPORT_WGSL } from "./wgsl/shaderSupport.wgsl.js";
import { TEXTURE_WGSL } from "./wgsl/texture.wgsl.js";

const WORKGROUP = 8;

/**
 * The bake's own kernel: the kernel's support code up to the translated shaders, the sky's program, and an entry that
 * runs it once per texel of the panorama, along the direction the CPU renderer's bake gives that texel. The output
 * goes to the binding the path tracer accumulates into, which is the one read-write buffer the support code declares.
 */
export function skyBakeSource(program: ShaderProgram, width: number, height: number): string {
  return `${traversalConstants(false)}${COMMON_WGSL}${RANDOM_WGSL}${BVH_WGSL}${SAMPLING_WGSL}${TEXTURE_WGSL}${SHADER_SUPPORT_WGSL}
${program.wgsl}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${width}u || id.y >= ${height}u) {
    return;
  }

  let u = (f32(id.x) + 0.5) / ${width}.0;
  let v = (f32(id.y) + 0.5) / ${height}.0;
  let phi = (u - 0.5) * 2.0 * PI;
  let theta = v * PI;
  var input: ShaderInput;
  input.data = 0u;
  input.direction = vec3<f32>(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
  input.fragCoord = vec4<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5, 0.0, 1.0);
  input.diffuseColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  accumulation[id.y * ${width}u + id.x] = p${program.id}_run(input).color;
}
`;
}

/** Translates a procedural sky's shader for the bake; the sky reads its direction from `vDirection`, as the CPU's does. */
export function compileSkyProgram(fragmentShader: string): Promise<ShaderProgram> {
  return compileProgram(0, "foreground", { fragment: fragmentShader, defines: {} }, true);
}

/**
 * Runs a procedural sky's shader on the device over an equirectangular panorama and reads the radiance back, as RGBA
 * floats, row by row from the top: the same image the CPU renderer's interpreter bakes, in a fraction of the time. A
 * texel that comes out not finite reads black, where the CPU bake would have refused the whole sky.
 */
export async function bakeSkyOnDevice(
  device: GPUDevice, program: ShaderProgram, uniforms: Record<string, UniformValue>, width: number,
): Promise<Float32Array> {
  const height = width / 2;
  const texels = width * height;
  const words = program.packUniforms(uniforms, new Map());
  const module = createShaderModule(device, "sky bake", skyBakeSource(program, width, height));

  const layout = device.createBindGroupLayout({
    label: "sky bake",
    entries: [
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipeline = await device.createComputePipelineAsync({
    label: "sky bake",
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: "main" },
  });

  const output = device.createBuffer({ label: "sky bake", size: texels * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const data = device.createBuffer({
    label: "sky bake",
    size: Math.max(16, words.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const staging = device.createBuffer({ label: "sky bake", size: texels * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  try {
    device.queue.writeBuffer(data, 0, words);

    const encoder = device.createCommandEncoder({ label: "sky bake" });
    const pass = encoder.beginComputePass({ label: "sky bake" });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout,
      entries: [{ binding: 7, resource: { buffer: output } }, { binding: 8, resource: { buffer: data } }],
    }));
    pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP), Math.ceil(height / WORKGROUP));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, texels * 16);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);

    const radiance = new Float32Array(staging.getMappedRange().slice(0));

    staging.unmap();

    for (let i = 0; i < radiance.length; i++) if (!Number.isFinite(radiance[i])) radiance[i] = 0;

    return radiance;
  } finally {
    output.destroy();
    data.destroy();
    staging.destroy();
  }
}
