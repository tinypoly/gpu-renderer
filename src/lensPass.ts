import { createShaderModule } from "./shaderModule.js";
import { NEUTRAL_WHITE_BALANCE, whiteBalanceMatrix, type GpuRenderSettings } from "./settings.js";
import { LENS_WGSL } from "./wgsl/lens.wgsl.js";

const UNIFORM_BYTES = 144;

const WORKGROUP = 8;

/** Halving stops once a level would be smaller than this on its short side, or after this many levels. */
const SMALLEST_LEVEL = 4;
const MAX_LEVELS = 8;

/** Where chromatic aberration `1` moves red and blue apart, as a share of the distance from the centre. */
const ABERRATION_SCALE = 0.005;

/** What the lens pass is told about the lens: the settings that reach it, resolved. */
export interface LensEffects {
  /** Share of the light the halo takes; zero leaves the pyramid unbuilt. */
  bloom: number;
  radius: number;
  aberration: number;
  /** Share of the light the aperture's rays take; zero gathers none. */
  glare: number;
  /** How many rays, how far they reach as a share of the half diagonal, and the first one's angle in radians. */
  rays: number;
  glareLength: number;
  glareAngle: number;
  /** A linear sRGB matrix, row by row, or `null` for a neutral white balance. */
  whiteBalance: number[] | null;
}

/** The lens effects these settings ask for, or `null` when they ask for none and the pass can be skipped. */
export function lensEffects(settings: GpuRenderSettings): LensEffects | null {
  const { bloom, glare, lens, camera } = settings;
  const intensity = bloom.enabled ? Math.min(1, Math.max(0, bloom.intensity)) : 0;
  const streaks = glare.enabled ? Math.min(1, Math.max(0, glare.intensity)) : 0;
  const aberration = Math.max(0, lens.chromaticAberration) * ABERRATION_SCALE;
  const neutral = !Number.isFinite(camera.whiteBalance) || camera.whiteBalance === NEUTRAL_WHITE_BALANCE;
  const blades = Math.min(16, Math.max(3, Math.round(Number(glare.blades) || 6)));

  if (intensity <= 0 && streaks <= 0 && aberration <= 0 && neutral) return null;

  return {
    bloom: intensity,
    radius: Math.min(0.95, Math.max(0, bloom.radius)),
    aberration,
    glare: streaks,
    // Each blade's edge diffracts into a ray each way: an even count's opposite edges share their rays.
    rays: blades % 2 === 0 ? blades : blades * 2,
    glareLength: Math.min(1, Math.max(0, Number(glare.length) || 0)),
    glareAngle: (Number(glare.rotation) || 0) * Math.PI / 180,
    whiteBalance: neutral ? null : whiteBalanceMatrix(camera.whiteBalance),
  };
}

interface Level {
  width: number;
  height: number;
  /** In texels, in both pyramids. */
  offset: number;
}

/** Halved levels of the image, the largest first, until one would be too small to be worth filtering. */
export function pyramidLevels(width: number, height: number): Level[] {
  const levels: Level[] = [];
  let w = width, h = height, offset = 0;

  while (levels.length < MAX_LEVELS) {
    w = Math.max(1, Math.ceil(w / 2));
    h = Math.max(1, Math.ceil(h / 2));

    if (levels.length > 0 && Math.min(w, h) < SMALLEST_LEVEL) break;

    levels.push({ width: w, height: h, offset });
    offset += w * h;
  }

  return levels;
}

/** The level the aperture's rays are gathered at: a quarter of the image, where there is one. */
function streakLevel(levels: Level[]): Level {
  return levels[Math.min(1, levels.length - 1)];
}

/**
 * The lens and the sensor, between the radiance and its display transform: bloom, chromatic aberration and white
 * balance, into a buffer of linear colour the presentation and the image resolve read in place of the radiance.
 */
export class LensPass {
  private readonly layout: GPUBindGroupLayout;
  private readonly downsample: GPUComputePipeline;
  private readonly upsample: GPUComputePipeline;
  private readonly composite: GPUComputePipeline;
  private readonly streaks: GPUComputePipeline;
  /** One uniform buffer per dispatch, so every one of them keeps its own values until the queue runs it. */
  private readonly uniforms: GPUBuffer[] = [];
  private readonly values = new ArrayBuffer(UNIFORM_BYTES);
  private down: GPUBuffer | null = null;
  private up: GPUBuffer | null = null;
  private graded: GPUBuffer | null = null;
  private streaked: GPUBuffer | null = null;
  private size = "";

  constructor(private readonly device: GPUDevice) {
    const module = createShaderModule(device, "lens", LENS_WGSL);
    const storage = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry["buffer"] => ({ type });

    this.layout = device.createBindGroupLayout({
      label: "lens",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: storage("read-only-storage") },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: storage("read-only-storage") },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: storage("storage") },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: storage("storage") },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: storage("storage") },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: storage("storage") },
      ],
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ layout, compute: { module, entryPoint } });

    this.downsample = pipeline("downsample");
    this.upsample = pipeline("upsample");
    this.composite = pipeline("composite");
    this.streaks = pipeline("streaks");
  }

  /** The graded image of the last `encode`, one vec4 of linear colour per pixel. */
  get output(): GPUBuffer | null {
    return this.graded;
  }

  private ensure(width: number, height: number, levels: Level[]) {
    const key = `${width}x${height}`;

    if (key === this.size && this.graded) return;

    this.destroyBuffers();
    this.size = key;

    const last = levels[levels.length - 1];
    const texels = Math.max(1, last.offset + last.width * last.height);
    // A pyramid of one level is copied into the halo as it is, which the two need copy usage for.
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    this.down = this.device.createBuffer({ label: "lens pyramid", size: texels * 16, usage });
    this.up = this.device.createBuffer({ label: "lens halo", size: texels * 16, usage });
    this.graded = this.device.createBuffer({ label: "lens output", size: width * height * 16,
      usage: GPUBufferUsage.STORAGE });

    const streak = streakLevel(levels);

    this.streaked = this.device.createBuffer({ label: "lens rays", size: streak.width * streak.height * 16,
      usage: GPUBufferUsage.STORAGE });
  }

  private uniform(index: number): GPUBuffer {
    while (this.uniforms.length <= index) {
      this.uniforms.push(this.device.createBuffer({
        size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
    }

    return this.uniforms[index];
  }

  /**
   * Encodes the whole pass over the accumulation (or the denoiser's output, when given) and returns the buffer it
   * writes, which holds the result once the encoder has run.
   */
  encode(encoder: GPUCommandEncoder, accumulation: GPUBuffer, denoised: GPUBuffer | null, width: number,
    height: number, effects: LensEffects): GPUBuffer {
    const levels = pyramidLevels(width, height);

    this.ensure(width, height, levels);

    const floats = new Float32Array(this.values), integers = new Uint32Array(this.values);
    const white = effects.whiteBalance ?? [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const halo = levels[0];
    const streak = streakLevel(levels);
    let used = 0;

    const dispatch = (pipeline: GPUComputePipeline, target: { width: number; height: number },
      fill: () => void) => {
      integers.fill(0);
      integers[0] = width;
      integers[1] = height;
      integers[2] = denoised ? 1 : 0;
      integers[11] = halo.offset;
      integers[12] = halo.width;
      integers[13] = halo.height;
      floats[14] = effects.bloom;
      floats[15] = effects.radius;
      floats[16] = effects.aberration;
      floats[17] = effects.glare;
      integers[18] = effects.rays;
      floats[19] = effects.glareLength * Math.hypot(streak.width, streak.height) / 2;
      integers[32] = streak.width;
      integers[33] = streak.height;
      floats[34] = effects.glareAngle;
      for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++)
        floats[20 + row * 4 + column] = white[row * 3 + column];
      fill();

      const buffer = this.uniform(used++);

      this.device.queue.writeBuffer(buffer, 0, this.values);

      const pass = encoder.beginComputePass({ label: "lens" });

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: accumulation } },
          { binding: 2, resource: { buffer: denoised ?? accumulation } },
          { binding: 3, resource: { buffer: this.down as GPUBuffer } },
          { binding: 4, resource: { buffer: this.up as GPUBuffer } },
          { binding: 5, resource: { buffer: this.graded as GPUBuffer } },
          { binding: 6, resource: { buffer: this.streaked as GPUBuffer } },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(target.width / WORKGROUP), Math.ceil(target.height / WORKGROUP));
      pass.end();
    };

    const read = (level: { width: number; height: number; offset: number }) => {
      integers[4] = level.width;
      integers[5] = level.height;
      integers[6] = level.offset;
    };

    const write = (level: Level) => {
      integers[8] = level.width;
      integers[9] = level.height;
      integers[10] = level.offset;
    };

    // The rays are gathered from a level of the same pyramid, so either effect builds it.
    if (effects.bloom > 0 || effects.glare > 0) {
      levels.forEach((level, k) => dispatch(this.downsample, level, () => {
        if (k === 0) {
          read({ width, height, offset: 0 });
          integers[7] = 1;
        } else {
          read(levels[k - 1]);
        }

        write(level);
      }));
    }

    if (effects.glare > 0) {
      dispatch(this.streaks, streak, () => {
        read(streak);
        write({ ...streak, offset: 0 });
      });
    }

    if (effects.bloom > 0) {
      // The smallest level folds into nothing wider; each one above takes the spread of everything below it.
      for (let k = levels.length - 2; k >= 0; k--) {
        dispatch(this.upsample, levels[k], () => {
          read(levels[k + 1]);
          integers[3] = k === levels.length - 2 ? 1 : 0;
          write(levels[k]);
        });
      }

      // A single level has nothing to fold: the halo is that level as it was filtered.
      if (levels.length === 1) {
        encoder.copyBufferToBuffer(this.down as GPUBuffer, 0, this.up as GPUBuffer, 0,
          levels[0].width * levels[0].height * 16);
      }
    }

    dispatch(this.composite, { width, height }, () => {});

    return this.graded as GPUBuffer;
  }

  private destroyBuffers() {
    this.down?.destroy();
    this.up?.destroy();
    this.graded?.destroy();
    this.streaked?.destroy();
    this.down = this.up = this.graded = this.streaked = null;
  }

  destroy() {
    this.destroyBuffers();
    for (const buffer of this.uniforms) buffer.destroy();
    this.uniforms.length = 0;
  }
}
