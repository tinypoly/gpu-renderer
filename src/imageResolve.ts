import { createShaderModule } from "./shaderModule.js";
import type { RenderToneMapping } from "./upstream.js";
import { PRESENT_WGSL } from "./wgsl/present.wgsl.js";

/** The order the presentation shader's `toneMapping` switch expects. */
const TONE_MAPPING = ["aces", "agx", "neutral", "linear"] as const;

const UNIFORM_BYTES = 48;

const WORKGROUP = 8;

/**
 * The finished image, resolved on the device: the presentation pass's display transform, run as a compute pass that
 * writes straight-alpha sRGB bytes, four per pixel. Reading those back is a sixth of the float accumulation and needs
 * no loop over the pixels on the page, which is what held it for seconds at the end of a large frame.
 *
 * It follows the image `render()` has always resolved: no grain, and the vignette measured from each pixel's corner.
 */
export const IMAGE_RESOLVE_WGSL = /* wgsl */ `${PRESENT_WGSL}
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;

fn encodeByte(value: f32) -> u32 {
  return u32(round(clamp(value, 0.0, 1.0) * 255.0));
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn resolveMain(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = present.resolution.x;
  let height = present.resolution.y;

  if (id.x >= width || id.y >= height) {
    return;
  }

  let index = id.y * width + id.x;
  let stored = accumulation[index];
  let samples = stored.w;
  var alpha = 1.0;

  if (present.transparent != 0u) {
    alpha = select(0.0, clamp(accumulation[width * height + index].x / samples, 0.0, 1.0), samples > 0.0);
  }

  var color = vec3<f32>(0.0);

  if (samples > 0.0 && alpha > 0.0) {
    let mean = select(stored.rgb / samples, denoised[index].rgb, present.denoised != 0u);
    let centred = vec2<f32>(f32(id.x) / f32(width), f32(id.y) / f32(height)) - 0.5;
    let falloff = max(0.0, 1.0 - present.vignette * dot(centred, centred) * 2.0);

    color = mean / alpha * falloff;
  }

  let encoded = linearToSrgb(toneMap(color));

  pixels[index] = encodeByte(encoded.r) | (encodeByte(encoded.g) << 8u) | (encodeByte(encoded.b) << 16u)
    | (encodeByte(alpha) << 24u);
}
`;

export interface ResolveSettings {
  exposure: number;
  toneMapping: RenderToneMapping;
  vignette: number;
  contrast: number;
  transparent: boolean;
}

export class ImageResolvePass {
  private readonly pipeline: GPUComputePipeline;
  private readonly uniforms: GPUBuffer;
  private readonly values = new ArrayBuffer(UNIFORM_BYTES);

  constructor(private readonly device: GPUDevice) {
    const module = createShaderModule(device, "image resolve", IMAGE_RESOLVE_WGSL);

    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "resolveMain" } });
    this.uniforms = device.createBuffer({
      size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** The image as RGBA bytes, straight alpha, row by row from the top. */
  async read(accumulation: GPUBuffer, denoised: GPUBuffer | null, width: number, height: number,
    settings: ResolveSettings): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const floats = new Float32Array(this.values), integers = new Uint32Array(this.values);

    integers.fill(0);
    integers[0] = width;
    integers[1] = height;
    floats[2] = settings.exposure;
    integers[3] = Math.max(0, TONE_MAPPING.indexOf(settings.toneMapping));
    floats[4] = settings.vignette;
    integers[7] = settings.transparent ? 1 : 0;
    integers[8] = denoised ? 1 : 0;
    floats[9] = settings.contrast;
    this.device.queue.writeBuffer(this.uniforms, 0, this.values);

    const bytes = width * height * 4;
    const output = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    try {
      const encoder = this.device.createCommandEncoder({ label: "image resolve" });
      const pass = encoder.beginComputePass({ label: "image resolve" });

      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniforms } },
          { binding: 1, resource: { buffer: accumulation } },
          { binding: 2, resource: { buffer: denoised ?? accumulation } },
          { binding: 3, resource: { buffer: output } },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP), Math.ceil(height / WORKGROUP));
      pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, bytes);
      this.device.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);

      const data = new Uint8ClampedArray(staging.getMappedRange().slice(0));

      staging.unmap();

      return data;
    } finally {
      output.destroy();
      staging.destroy();
    }
  }

  dispose() {
    this.uniforms.destroy();
  }
}
