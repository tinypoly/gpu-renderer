import type { UNet } from "oidn-webgpu";
import { initUNetFromURL } from "oidn-webgpu/dist/oidn.js";
import { createShaderModule } from "./shaderModule.js";

/**
 * The weights the OIDN denoiser loads by default: the HDR ray tracing network that takes the albedo and the normal as
 * well as the colour (RenderKit's oidn-weights, Apache-2.0), shipped with the package. A bundler that moves the package
 * elsewhere, or an app that serves its own copy, passes `denoiser.oidnWeights` instead.
 */
export const DEFAULT_OIDN_WEIGHTS = new URL("../weights/rt_hdr_alb_nrm.tza", import.meta.url).href;

const WORKGROUP = 8;
const METER_GRID = [64, 36] as const;

/**
 * Reads the accumulation into the three images OIDN takes, and back.
 *
 * The network was trained on images whose average sits near middle grey, which its own CPU path reaches by an
 * automatic exposure; on the device that scale is left to the caller. `meter` takes the mean of the logarithms of the
 * image's luminance over a grid, as the lens pass's meter does, `prepare` writes the colour times the scale that brings
 * it to middle grey, the albedo (diffuse and specular, so a mirror is not taken for black) and the normal, and `finish`
 * divides the network's output by the same scale and mixes it with the traced image by the strength.
 */
export const OIDN_WGSL = /* wgsl */ `
struct Params {
  size: vec2<u32>,
  count: u32,
  strength: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> accumulation: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> color: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> albedo: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> normal: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> scale: array<f32>;

const MIDDLE_GREY: f32 = 0.18;
const GRID: vec2<u32> = vec2<u32>(${METER_GRID[0]}u, ${METER_GRID[1]}u);

var<workgroup> sums: array<vec2<f32>, 256>;

fn luminance(value: vec3<f32>) -> f32 {
  return dot(value, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn mean(index: u32) -> vec3<f32> {
  let stored = accumulation[index];

  return select(vec3<f32>(0.0), stored.rgb / stored.w, stored.w > 0.0);
}

@compute @workgroup_size(256, 1, 1)
fn meter(@builtin(local_invocation_index) thread: u32) {
  var sum = vec2<f32>(0.0);

  for (var k = thread; k < GRID.x * GRID.y; k = k + 256u) {
    let uv = (vec2<f32>(f32(k % GRID.x), f32(k / GRID.x)) + 0.5) / vec2<f32>(GRID);
    let pixel = min(vec2<u32>(uv * vec2<f32>(params.size)), params.size - 1u);
    let value = luminance(mean(pixel.y * params.size.x + pixel.x));

    if (value > 0.0) {
      sum = sum + vec2<f32>(log(value), 1.0);
    }
  }

  sums[thread] = sum;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (thread < stride) {
      sums[thread] = sums[thread] + sums[thread + stride];
    }

    workgroupBarrier();
  }

  if (thread == 0u) {
    let total = sums[0];

    scale[0] = select(1.0, clamp(MIDDLE_GREY / exp(total.x / total.y), 1e-6, 1e6), total.y > 0.0);
  }
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn prepare(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.size.x || id.y >= params.size.y) {
    return;
  }

  let index = id.y * params.size.x + id.x;
  let samples = max(accumulation[index].w, 1.0);
  let traced = mean(index) * scale[0];
  let normalSum = accumulation[3u * params.count + index].xyz;
  let size = length(normalSum);
  let surface = size > 1e-6;
  let reflectance = (accumulation[2u * params.count + index].rgb + accumulation[5u * params.count + index].rgb)
    / samples;

  color[index] = vec4<f32>(max(traced, vec3<f32>(0.0)), 1.0);
  // Where the camera sees no surface, the sky, the albedo is the colour itself, as OIDN asks of the background.
  albedo[index] = vec4<f32>(select(clamp(traced, vec3<f32>(0.0), vec3<f32>(1.0)),
    clamp(reflectance, vec3<f32>(0.0), vec3<f32>(1.0)), surface), 1.0);
  normal[index] = vec4<f32>(select(vec3<f32>(0.0), normalSum / size, surface), 0.0);
}

// The network's output arrives in the colour binding; the albedo binding takes the result.
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.size.x || id.y >= params.size.y) {
    return;
  }

  let index = id.y * params.size.x + id.x;
  let traced = mean(index);
  let denoised = max(color[index].rgb, vec3<f32>(0.0)) / scale[0];

  albedo[index] = vec4<f32>(mix(traced, denoised, clamp(params.strength, 0.0, 1.0)), 1.0);
}
`;

const networks = new WeakMap<GPUDevice, Map<string, Promise<UNet>>>();

/** One network per device and weights file, kept for as long as the device lives: loading it compiles its kernels. */
function network(device: GPUDevice, weights: string): Promise<UNet> {
  let byUrl = networks.get(device);

  if (!byUrl) networks.set(device, byUrl = new Map());

  let unet = byUrl.get(weights);

  if (!unet) {
    unet = initUNetFromURL(weights, { device }, { aux: true, hdr: true });
    // A file that failed to load may load on a later attempt.
    unet.catch(() => byUrl.delete(weights));
    byUrl.set(weights, unet);
  }

  return unet;
}

/**
 * Open Image Denoise (Intel's neural denoiser, through oidn-webgpu): the image, its albedo and its normal through a
 * U-Net trained on path traced images, entirely on the device. It keeps texture and fine detail that a hand-made
 * filter blurs, at the cost of a fraction of a second for a large image.
 */
export class OidnPass {
  private readonly layout: GPUBindGroupLayout;
  private readonly meter: GPUComputePipeline;
  private readonly prepare: GPUComputePipeline;
  private readonly finish: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private readonly scale: GPUBuffer;
  private buffers: GPUBuffer[] = [];
  private width = 0;
  private height = 0;
  /** The denoised radiance, one vec4 per pixel, once `run` has resolved. */
  output: GPUBuffer | null = null;

  constructor(private readonly device: GPUDevice) {
    const module = createShaderModule(device, "oidn", OIDN_WGSL);
    const visibility = GPUShaderStage.COMPUTE;
    const storage = (type: GPUBufferBindingType) => ({ binding: 0, visibility, buffer: { type } });

    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility, buffer: { type: "uniform" } },
        { ...storage("read-only-storage"), binding: 1 },
        { ...storage("storage"), binding: 2 },
        { ...storage("storage"), binding: 3 },
        { ...storage("storage"), binding: 4 },
        { ...storage("storage"), binding: 5 },
      ],
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ layout, compute: { module, entryPoint } });

    this.meter = pipeline("meter");
    this.prepare = pipeline("prepare");
    this.finish = pipeline("finish");
    this.params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.scale = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  }

  private resize(width: number, height: number) {
    if (width === this.width && height === this.height && this.buffers.length) return;

    for (const buffer of this.buffers) buffer.destroy();

    const size = Math.max(16, width * height * 16);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    // Colour, albedo and normal for the network, then the result.
    this.buffers = Array.from({ length: 4 }, () => this.device.createBuffer({ size, usage }));
    this.output = this.buffers[3];
    this.width = width;
    this.height = height;
  }

  private group(accumulation: GPUBuffer, color: GPUBuffer, albedo: GPUBuffer, normal: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: accumulation } },
        { binding: 2, resource: { buffer: color } },
        { binding: 3, resource: { buffer: albedo } },
        { binding: 4, resource: { buffer: normal } },
        { binding: 5, resource: { buffer: this.scale } },
      ],
    });
  }

  private dispatch(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup, x: number,
    y: number) {
    const pass = encoder.beginComputePass({ label: "oidn" });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }

  /**
   * Denoises what `accumulation` holds into `output`. The images are read from it as it stands on the queue when this
   * is called; the promise resolves once the result has been written, and rejects when the network cannot load.
   */
  async run(accumulation: GPUBuffer, width: number, height: number, strength: number, weights: string,
    signal?: { cancelled: boolean }): Promise<boolean> {
    const unet = await network(this.device, weights);

    if (signal?.cancelled) return false;

    this.resize(width, height);

    const [color, albedo, normal, output] = this.buffers;
    const values = new ArrayBuffer(16);

    new Uint32Array(values).set([width, height, width * height]);
    new Float32Array(values)[3] = strength;
    this.device.queue.writeBuffer(this.params, 0, values);

    const groups = [Math.ceil(width / WORKGROUP), Math.ceil(height / WORKGROUP)] as const;
    const inputs = this.group(accumulation, color, albedo, normal);
    const encoder = this.device.createCommandEncoder({ label: "oidn prepare" });

    this.dispatch(encoder, this.meter, inputs, 1, 1);
    this.dispatch(encoder, this.prepare, inputs, ...groups);
    this.device.queue.submit([encoder.finish()]);

    const denoised = await new Promise<GPUBuffer | null>((resolve, reject) => {
      const abort = unet.tileExecute({
        color: { data: color, width, height },
        albedo: { data: albedo, width, height },
        normal: { data: normal, width, height },
        done: result => resolve(result.data),
        error: reject,
        progress: () => {
          if (signal?.cancelled) {
            abort();
            resolve(null);
          }
        },
      });
    });

    if (!denoised || signal?.cancelled) return false;

    const finishing = this.device.createCommandEncoder({ label: "oidn finish" });

    this.dispatch(finishing, this.finish, this.group(accumulation, denoised, output, normal), ...groups);
    this.device.queue.submit([finishing.finish()]);

    return true;
  }

  destroy() {
    for (const buffer of [...this.buffers, this.params, this.scale]) buffer.destroy();

    this.buffers = [];
    this.output = null;
  }
}
