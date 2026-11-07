import { createShaderModule } from "./shaderModule.js";

/**
 * Which denoiser filters the image: `"oidn"`, Intel's Open Image Denoise, a neural network that keeps texture and fine
 * detail, or `"svgf"`, the edge-aware wavelet filter, faster and with the controls below.
 */
export type DenoiserAlgorithm = "oidn" | "svgf";

/** How the denoiser filters. Every field is optional where it is taken, and falls back to `DEFAULT_DENOISER`. */
export interface DenoiserSettings {
  /** The denoiser that runs; OIDN falls back to SVGF where its network cannot load. */
  algorithm: DenoiserAlgorithm;
  /** Where OIDN's weights are fetched from; empty takes the ones shipped with the package. */
  oidnWeights: string;
  /** How much of the filtered image replaces the traced one, from 0 (none) to 1. */
  strength: number;
  /** SVGF only from here: wavelet passes, 1 to 5, how far the filter reaches, from 2 to 62 pixels. */
  passes: number;
  /**
   * How large a luminance difference between two pixels still counts as noise, in standard deviations of the pixel's
   * own noise. Lower keeps more detail and removes less noise.
   */
  edge: number;
  /** Leave the pixels that have converged as they were traced, and filter only where noise is still visible. */
  onlyNoisy: boolean;
  /** Clamp a pixel brighter than every one of its neighbours to the brightest of them, before filtering. */
  antiFirefly: boolean;
  /**
   * How far the filter lets go of the normal and depth tests over a material that scatters light inside, from 0 to 1.
   * Light leaves such a surface after wandering under it, so its shading hardly follows the normal at each point.
   */
  subsurface: number;
  /**
   * The smallest footprint a smooth reflection is filtered over, in pixels; 0 leaves a mirror alone. Within it, taps
   * also have to agree on how far away the reflected thing is, so the edge of a reflected object stays sharp.
   */
  mirrorRadius: number;
}

export const DEFAULT_DENOISER: Readonly<DenoiserSettings> = Object.freeze({
  algorithm: "oidn",
  oidnWeights: "",
  strength: 1,
  passes: 3,
  edge: 2,
  onlyNoisy: true,
  antiFirefly: true,
  subsurface: 1,
  mirrorRadius: 3,
});

/** The settings with `input` applied, each field kept in range. */
export function mergeDenoiser(base: DenoiserSettings, input: Partial<DenoiserSettings> = {}): DenoiserSettings {
  const merged = { ...base, ...Object.fromEntries(Object.entries(input).filter(([, value]) => value !== undefined)) };

  return {
    algorithm: merged.algorithm === "svgf" ? "svgf" : "oidn",
    oidnWeights: typeof merged.oidnWeights === "string" ? merged.oidnWeights : "",
    strength: Math.min(1, Math.max(0, Number(merged.strength) || 0)),
    passes: Math.min(5, Math.max(1, Math.round(Number(merged.passes) || 1))),
    edge: Math.max(0.01, Number(merged.edge) || DEFAULT_DENOISER.edge),
    onlyNoisy: Boolean(merged.onlyNoisy),
    antiFirefly: Boolean(merged.antiFirefly),
    subsurface: Math.min(1, Math.max(0, Number(merged.subsurface) || 0)),
    mirrorRadius: Math.min(16, Math.max(0, Number(merged.mirrorRadius) || 0)),
  };
}

/** The error a pixel counts as converged under, the one adaptive sampling stops at. */
export const CONVERGED_ERROR = 0.004;

/** What the camera says about how large things look, which the specular filter's radius depends on. */
export interface DenoiseView {
  /** World units one pixel spans at unit distance (perspective) or anywhere (orthographic). */
  pixelScale: number;
  orthographic: boolean;
}

/**
 * The denoiser: the spatial half of SVGF (Schied and others, 2017), with the diffuse and the specular light filtered
 * apart, as NVIDIA's RELAX and REBLUR do.
 *
 * The path tracer splits every path at its first scatter, by the lobe it took, so each pixel holds a diffuse and a
 * specular signal, each with its own variance. Each is divided by its own albedo (the diffuse colour, and the GGX
 * lobe's directional albedo), so the filter sees light and never blurs a texture or a colour; both are multiplied back
 * at the end. A pixel brighter than all of its neighbours is first clamped to the brightest of them.
 *
 * Then a few passes of an edge-avoiding a-trous wavelet, its taps twice as far apart each pass. The diffuse light is
 * weighed by the agreement of normals and depths and by its luminance against the pixel's own variance. The specular
 * light also by roughness, by normals more sharply the smoother the surface, and within the footprint its lobe leaves on
 * the screen: the lobe's angle times the distance to what it reflects, seen from the camera. A sharp reflection of
 * something near is left almost alone; a rough reflection of something far is smoothed across a wide area.
 *
 * The weights are normalized, which keeps every flat area at exactly its mean, and each signal is mixed with what was
 * traced by how noisy it still is, so what has converged keeps exactly its traced value.
 */
export const DENOISE_WGSL = /* wgsl */ `
struct Params {
  size: vec2<u32>,
  step: u32,
  count: u32,
  edge: f32,
  strength: f32,
  onlyNoisy: u32,
  converged: f32,
  pixelScale: f32,
  orthographic: u32,
  subsurface: f32,
  mirrorRadius: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> accumulation: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> diffuseIn: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> specularIn: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> diffuseOut: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> specularOut: array<vec4<f32>>;
// The guides each tap compares, worked out once per pixel by prepare: the normal and the depth, then the roughness, or
// -1 where no sample has landed. The same numbers the accumulation gives, two reads a tap instead of four.
@group(0) @binding(6) var<storage, read_write> guides: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> roughnesses: array<f32>;

fn luminance(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn samples(index: u32) -> f32 {
  return max(accumulation[index].w, 1.0);
}

/** Each signal's mean radiance: the diffuse one stored apart, the specular one the rest. */
fn diffuseMean(index: u32) -> vec3<f32> {
  return accumulation[4u * params.count + index].rgb / samples(index);
}

fn specularMean(index: u32) -> vec3<f32> {
  return accumulation[index].rgb / samples(index) - diffuseMean(index);
}

/** The albedos the light is divided by, kept off zero so the division never explodes. */
fn diffuseAlbedo(index: u32) -> vec3<f32> {
  return max(accumulation[2u * params.count + index].rgb / samples(index), vec3<f32>(0.01));
}

fn specularAlbedo(index: u32) -> vec3<f32> {
  return max(accumulation[5u * params.count + index].rgb / samples(index), vec3<f32>(0.01));
}

fn guideNormal(index: u32) -> vec3<f32> {
  let sum = accumulation[3u * params.count + index].xyz;
  let size = length(sum);

  return select(vec3<f32>(0.0), sum / size, size > 1e-6);
}

fn guideDepth(index: u32) -> f32 {
  return accumulation[3u * params.count + index].w / samples(index);
}

fn guideRoughness(index: u32) -> f32 {
  return accumulation[2u * params.count + index].w / samples(index);
}

/** How much of the first surface scatters light inside, averaged over the pixel's samples. */
fn guideSubsurface(index: u32) -> f32 {
  return accumulation[5u * params.count + index].w / samples(index);
}

fn specularDistance(index: u32) -> f32 {
  return accumulation[4u * params.count + index].w / samples(index);
}

/** The variance of a signal's mean, in the space of its demodulated light. */
fn variance(mean: vec3<f32>, square: f32, count: f32, albedo: vec3<f32>) -> f32 {
  let m = luminance(mean);
  let a = max(luminance(albedo), 0.01);

  return max(square / count - m * m, 0.0) / count / (a * a);
}

/** Both signals, divided by their albedos, with the variance of their means. */
@compute @workgroup_size(8, 8, 1)
fn prepare(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.size.x || id.y >= params.size.y) {
    return;
  }

  let index = id.y * params.size.x + id.x;
  let count = accumulation[index].w;

  if (count <= 0.0) {
    diffuseOut[index] = vec4<f32>(0.0);
    specularOut[index] = vec4<f32>(0.0);
    guides[index] = vec4<f32>(0.0);
    roughnesses[index] = -1.0;

    return;
  }

  guides[index] = vec4<f32>(guideNormal(index), guideDepth(index));
  roughnesses[index] = guideRoughness(index);

  let squares = accumulation[params.count + index];
  let diffuse = diffuseMean(index);
  let specular = specularMean(index);

  diffuseOut[index] = vec4<f32>(diffuse / diffuseAlbedo(index), variance(diffuse, squares.z, count, diffuseAlbedo(index)));
  specularOut[index] = vec4<f32>(
    specular / specularAlbedo(index), variance(specular, squares.w, count, specularAlbedo(index)));
}

fn inside(q: vec2<i32>) -> bool {
  return q.x >= 0 && q.y >= 0 && q.x < i32(params.size.x) && q.y < i32(params.size.y);
}

/** A value brighter than every neighbour, scaled down to the brightest of them: a firefly, not a feature. */
fn clampFirefly(value: vec4<f32>, brightest: f32) -> vec4<f32> {
  let own = luminance(value.rgb);

  if (own <= brightest || own <= 0.0) {
    return value;
  }

  return vec4<f32>(value.rgb * (brightest / own), value.w);
}

@compute @workgroup_size(8, 8, 1)
fn antiFirefly(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.size.x || id.y >= params.size.y) {
    return;
  }

  let pixel = vec2<i32>(id.xy);
  let index = id.y * params.size.x + id.x;
  var diffuse = 0.0;
  var specular = 0.0;

  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let q = pixel + vec2<i32>(dx, dy);

      if ((dx == 0 && dy == 0) || !inside(q)) {
        continue;
      }

      let other = u32(q.y) * params.size.x + u32(q.x);
      diffuse = max(diffuse, luminance(diffuseIn[other].rgb));
      specular = max(specular, luminance(specularIn[other].rgb));
    }
  }

  diffuseOut[index] = clampFirefly(diffuseIn[index], diffuse);
  specularOut[index] = clampFirefly(specularIn[index], specular);
}

const KERNEL = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

/** The variance around a pixel, blurred over 3 x 3, which steadies the luminance test on a few samples. */
fn localVariance(pixel: vec2<i32>, specular: bool) -> f32 {
  var sum = 0.0;
  var weight = 0.0;

  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let q = pixel + vec2<i32>(dx, dy);

      if (!inside(q)) {
        continue;
      }

      let other = u32(q.y) * params.size.x + u32(q.x);
      let w = select(0.5, 1.0, dx == 0) * select(0.5, 1.0, dy == 0);
      sum = sum + select(diffuseIn[other].w, specularIn[other].w, specular) * w;
      weight = weight + w;
    }
  }

  return sum / max(weight, 1e-6);
}

/**
 * How many pixels the specular lobe's footprint covers: the lobe's angle times the distance to what it reflects, seen
 * from the camera across the distance to the surface plus that.
 */
fn specularRadius(index: u32, depth: f32) -> f32 {
  let alpha = guideRoughness(index) * guideRoughness(index);
  let reflected = specularDistance(index);

  if (params.orthographic != 0u) {
    return min(alpha * reflected / params.pixelScale, 64.0);
  }

  return min(alpha * reflected / max(depth + reflected, 1e-6) / params.pixelScale, 64.0);
}

/** One a-trous pass over both signals, its taps params.step pixels apart. */
@compute @workgroup_size(8, 8, 1)
fn atrous(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.size.x || id.y >= params.size.y) {
    return;
  }

  let pixel = vec2<i32>(id.xy);
  let index = id.y * params.size.x + id.x;
  let diffuseCenter = diffuseIn[index];
  let specularCenter = specularIn[index];

  if (accumulation[index].w <= 0.0) {
    diffuseOut[index] = diffuseCenter;
    specularOut[index] = specularCenter;

    return;
  }

  let guide = guides[index];
  let normal = guide.xyz;
  let depth = guide.w;
  let roughness = roughnesses[index];
  let diffuseSigma = params.edge * sqrt(max(localVariance(pixel, false), 0.0))
    * (1.0 + guideSubsurface(index) * params.subsurface) + 1e-6;
  let specularSigma = params.edge * sqrt(max(localVariance(pixel, true), 0.0)) + 1e-6;
  // A smooth reflection is filtered over at least the mirror radius, by taps that see the reflected thing at the same
  // distance: that test keeps the edges of what it reflects, which lets the normal test relax over a curved mirror,
  // where neighbouring normals part fast and would otherwise leave the reflection unfiltered.
  let mirror = params.mirrorRadius > 0.0 && roughness < 0.3;
  let radius = select(specularRadius(index, depth), max(specularRadius(index, depth), params.mirrorRadius), mirror);
  let reflected = specularDistance(index);
  // A smoother surface's reflection turns faster with its normal: the normal test sharpens with it, short of shutting
  // out the neighbours of a curved mirror altogether once the reflected distance guards it.
  let specularPower = min(128.0 / max(roughness * roughness, 0.03), select(4096.0, 64.0, mirror));
  // Under a surface that scatters light inside, the normal and the depth say little about the light that leaves it.
  let scattering = guideSubsurface(index) * params.subsurface;
  let diffusePower = mix(128.0, 1.0, scattering);
  let depthSlack = mix(1.0, 4.0, scattering);
  let step = i32(params.step);
  var diffuse = vec4<f32>(0.0);
  var specular = vec4<f32>(0.0);
  var diffuseTotal = 0.0;
  var specularTotal = 0.0;

  for (var dy = -2; dy <= 2; dy = dy + 1) {
    for (var dx = -2; dx <= 2; dx = dx + 1) {
      let q = pixel + vec2<i32>(dx, dy) * step;

      if (!inside(q)) {
        continue;
      }

      let other = u32(q.y) * params.size.x + u32(q.x);
      let d = diffuseIn[other];
      let s = specularIn[other];
      let base = KERNEL[dx + 2] * KERNEL[dy + 2];
      var wd = base;
      var ws = base;

      if (dx != 0 || dy != 0) {
        let otherRoughness = roughnesses[other];

        // No sample has landed there.
        if (otherRoughness < 0.0) {
          continue;
        }

        let otherGuide = guides[other];
        let offset = length(vec2<f32>(f32(dx * step), f32(dy * step)));
        let alignment = max(dot(normal, otherGuide.xyz), 0.0);
        let depthWeight = exp(-abs(depth - otherGuide.w) / (0.02 * depth * offset + 1e-3));
        let diffuseDepth = exp(-abs(depth - otherGuide.w) / (0.02 * depthSlack * depth * offset + 1e-3));

        wd = wd * pow(alignment, diffusePower) * diffuseDepth
          * exp(-abs(luminance(d.rgb) - luminance(diffuseCenter.rgb)) / diffuseSigma);

        ws = ws * pow(alignment, specularPower) * depthWeight
          * exp(-abs(otherRoughness - roughness) * 10.0)
          * exp(-abs(luminance(s.rgb) - luminance(specularCenter.rgb)) / specularSigma)
          * select(0.0, exp(-offset * offset / (radius * radius)), radius >= 0.5);

        if (mirror) {
          ws = ws * exp(-abs(specularDistance(other) - reflected) / (0.05 * reflected + 1e-3));
        }
      }

      diffuse = diffuse + vec4<f32>(d.rgb * wd, d.w * wd * wd);
      specular = specular + vec4<f32>(s.rgb * ws, s.w * ws * ws);
      diffuseTotal = diffuseTotal + wd;
      specularTotal = specularTotal + ws;
    }
  }

  diffuseOut[index] = vec4<f32>(diffuse.rgb / diffuseTotal, diffuse.w / (diffuseTotal * diffuseTotal));
  specularOut[index] = vec4<f32>(specular.rgb / specularTotal, specular.w / (specularTotal * specularTotal));
}

/** How much of the filtered signal replaces the traced one: by how noisy the signal still is, when asked. */
fn amount(mean: vec3<f32>, square: f32, count: f32) -> f32 {
  if (params.onlyNoisy == 0u || count <= 1.0) {
    return params.strength;
  }

  let m = luminance(mean);
  let error = sqrt(max(square / count - m * m, 0.0) / count) / sqrt(max(m, 0.0) + 1e-3);

  return params.strength * smoothstep(params.converged, 4.0 * params.converged, error);
}

/** Both filtered signals, times the albedos they were divided by, each mixed with what was traced. */
@compute @workgroup_size(8, 8, 1)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.size.x || id.y >= params.size.y) {
    return;
  }

  let index = id.y * params.size.x + id.x;
  let count = accumulation[index].w;

  if (count <= 0.0) {
    diffuseOut[index] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    return;
  }

  let squares = accumulation[params.count + index];
  let diffuse = diffuseMean(index);
  let specular = specularMean(index);
  let filteredDiffuse = diffuseIn[index].rgb * diffuseAlbedo(index);
  let filteredSpecular = specularIn[index].rgb * specularAlbedo(index);

  diffuseOut[index] = vec4<f32>(
    mix(diffuse, filteredDiffuse, amount(diffuse, squares.z, count))
    + mix(specular, filteredSpecular, amount(specular, squares.w, count)), 1.0);
}
`;

/** The a-trous passes, their taps 1, 2, 4, 8 and 16 pixels apart. */
const STEPS = [1, 2, 4, 8, 16];

/** Uniforms: prepare, anti-firefly, one per a-trous pass, finish. */
const STAGES = STEPS.length + 3;

export class DenoisePass {
  private readonly layout: GPUBindGroupLayout;
  private readonly prepare: GPUComputePipeline;
  private readonly antiFirefly: GPUComputePipeline;
  private readonly atrous: GPUComputePipeline;
  private readonly finish: GPUComputePipeline;
  /** One uniform per stage, so a whole denoise is a single submit. */
  private readonly params: GPUBuffer[];
  /** Two ping-pong pairs, diffuse and specular, the output, then the guides and the roughness. */
  private buffers: GPUBuffer[] = [];
  /** Bind groups by the buffers they bind, and a number for each buffer seen, which is what the key is made of. */
  private readonly groups = new Map<string, GPUBindGroup>();
  private readonly ids = new WeakMap<GPUBuffer, number>();
  private nextId = 0;

  private idOf(buffer: GPUBuffer): number {
    let id = this.ids.get(buffer);

    if (id === undefined) this.ids.set(buffer, id = this.nextId++);

    return id;
  }

  private width = 0;
  private height = 0;

  /** The denoised radiance, one vec4 per pixel, once `encode` has run. */
  output: GPUBuffer | null = null;

  constructor(private readonly device: GPUDevice) {
    const module = createShaderModule(device, "denoise", DENOISE_WGSL);
    const storage = (type: GPUBufferBindingType) => ({ type });
    const visibility = GPUShaderStage.COMPUTE;

    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility, buffer: { type: "uniform" } },
        { binding: 1, visibility, buffer: storage("read-only-storage") },
        { binding: 2, visibility, buffer: storage("read-only-storage") },
        { binding: 3, visibility, buffer: storage("read-only-storage") },
        { binding: 4, visibility, buffer: storage("storage") },
        { binding: 5, visibility, buffer: storage("storage") },
        { binding: 6, visibility, buffer: storage("storage") },
        { binding: 7, visibility, buffer: storage("storage") },
      ],
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ layout, compute: { module, entryPoint } });

    this.prepare = pipeline("prepare");
    this.antiFirefly = pipeline("antiFirefly");
    this.atrous = pipeline("atrous");
    this.finish = pipeline("finish");
    this.params = Array.from({ length: STAGES }, () =>
      device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }

  private resize(width: number, height: number) {
    if (width === this.width && height === this.height && this.buffers.length) return;

    for (const buffer of this.buffers) buffer.destroy();

    const size = Math.max(16, width * height * 16);

    this.buffers = Array.from({ length: 6 }, () =>
      this.device.createBuffer({ label: "svgf", size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    this.buffers.push(
      this.device.createBuffer({ label: "svgf", size: Math.max(16, width * height * 4), usage: GPUBufferUsage.STORAGE }));
    this.output = this.buffers[4];
    this.groups.clear();
    this.width = width;
    this.height = height;
  }

  private writeParams(width: number, height: number, settings: DenoiserSettings, view: DenoiseView) {
    const steps = [1, 1, ...STEPS, 1];

    steps.forEach((step, k) => {
      const values = new ArrayBuffer(48);
      const words = new Uint32Array(values);
      const floats = new Float32Array(values);

      words.set([width, height, step, width * height]);
      floats[4] = settings.edge;
      floats[5] = settings.strength;
      words[6] = settings.onlyNoisy ? 1 : 0;
      floats[7] = CONVERGED_ERROR;
      floats[8] = view.pixelScale;
      words[9] = view.orthographic ? 1 : 0;
      floats[10] = settings.subsurface;
      floats[11] = settings.mirrorRadius;
      this.device.queue.writeBuffer(this.params[k], 0, values);
    });
  }

  /** Denoises what `accumulation` holds into `output`. */
  encode(
    encoder: GPUCommandEncoder,
    accumulation: GPUBuffer,
    width: number,
    height: number,
    settings: DenoiserSettings = DEFAULT_DENOISER,
    view: DenoiseView = { pixelScale: 1 / height, orthographic: false }) {
    this.resize(width, height);
    this.writeParams(width, height, settings, view);

    const [diffuseA, diffuseB, specularA, specularB, output, guides, roughnesses] = this.buffers;
    const groups = [Math.ceil(width / 8), Math.ceil(height / 8)] as const;

    const run = (pipeline: GPUComputePipeline, params: GPUBuffer,
      sources: [GPUBuffer, GPUBuffer], targets: [GPUBuffer, GPUBuffer]) => {
      // The same few groups serve every denoise of an image size: they are made once and kept.
      const key = [params, accumulation, ...sources, ...targets].map(buffer => this.idOf(buffer)).join();
      let group = this.groups.get(key);

      if (!group) {
        group = this.device.createBindGroup({
          layout: this.layout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: accumulation } },
            { binding: 2, resource: { buffer: sources[0] } },
            { binding: 3, resource: { buffer: sources[1] } },
            { binding: 4, resource: { buffer: targets[0] } },
            { binding: 5, resource: { buffer: targets[1] } },
            { binding: 6, resource: { buffer: guides } },
            { binding: 7, resource: { buffer: roughnesses } },
          ],
        });
        // A new accumulation buffer comes with a new size, which clears these; the cap is only a guard.
        if (this.groups.size > 64) this.groups.clear();

        this.groups.set(key, group);
      }

      const pass = encoder.beginComputePass({ label: "denoise" });

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups[0], groups[1]);
      pass.end();
    };

    // prepare reads nothing from its sources; the B buffers stand in so the bindings are valid.
    run(this.prepare, this.params[0], [diffuseB, specularB], [diffuseA, specularA]);

    let from: [GPUBuffer, GPUBuffer] = [diffuseA, specularA];
    let to: [GPUBuffer, GPUBuffer] = [diffuseB, specularB];

    if (settings.antiFirefly) {
      run(this.antiFirefly, this.params[1], from, to);
      [from, to] = [to, from];
    }

    STEPS.slice(0, settings.passes).forEach((_, k) => {
      run(this.atrous, this.params[k + 2], from, to);
      [from, to] = [to, from];
    });

    // finish writes the result to the output; the free specular buffer fills the second, unused, target.
    run(this.finish, this.params[STAGES - 1], from, [output, to[1]]);
  }

  destroy() {
    for (const buffer of [...this.buffers, ...this.params]) buffer.destroy();

    this.buffers = [];
    this.groups.clear();
    this.output = null;
  }
}
