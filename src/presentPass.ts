import { createShaderModule } from "./shaderModule.js";
import type { RenderToneMapping } from "./upstream.js";
import { PRESENT_WGSL } from "./wgsl/present.wgsl.js";

export interface PresentUniforms {
  width: number;
  height: number;
  exposure: number;
  toneMapping: RenderToneMapping;
  vignette: number;
  grain: number;
  /** AgX's look, from its base curve to Punchy. */
  contrast: number;
  frameIndex: number;
  transparent: boolean;
  /** The denoiser's output, which the colour is read from instead of the accumulation when given. */
  denoised?: GPUBuffer | null;
}

/** The order the shader's `toneMapping` switch expects. */
const TONE_MAPPING = ["aces", "agx", "neutral", "linear"] as const;

const PRESENT_BYTES = 48;

/** Resolves the accumulation buffer onto the canvas: divide by the sample count, expose, tone map, encode. */
export class PresentPass {
  private readonly pipeline: GPURenderPipeline;
  private readonly uniforms: GPUBuffer;
  private readonly values = new ArrayBuffer(PRESENT_BYTES);
  private readonly floats = new Float32Array(this.values);
  private readonly integers = new Uint32Array(this.values);
  private bindGroup: GPUBindGroup | null = null;
  private accumulation: GPUBuffer | null = null;
  private source: GPUBuffer | null = null;

  constructor(private readonly device: GPUDevice, readonly format: GPUTextureFormat) {
    const module = createShaderModule(device, "present", PRESENT_WGSL);

    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vertexMain" },
      fragment: { module, entryPoint: "fragmentMain", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.uniforms = device.createBuffer({
      size: PRESENT_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  setAccumulation(accumulation: GPUBuffer) {
    this.accumulation = accumulation;
    this.bind(null);
  }

  private bind(denoised: GPUBuffer | null) {
    if (!this.accumulation) return;

    // Without a denoised image the accumulation stands in for it; the uniform says which one to read.
    const source = denoised ?? this.accumulation;

    if (this.bindGroup && source === this.source) return;

    this.source = source;
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: this.accumulation } },
        { binding: 2, resource: { buffer: source } },
      ],
    });
  }

  encode(encoder: GPUCommandEncoder, view: GPUTextureView, present: PresentUniforms) {
    this.bind(present.denoised ?? null);

    if (!this.bindGroup) throw new Error("GPU renderer: the presentation pass has no accumulation buffer bound.");

    this.integers[0] = present.width;
    this.integers[1] = present.height;
    this.floats[2] = present.exposure;
    this.integers[3] = Math.max(0, TONE_MAPPING.indexOf(present.toneMapping));
    this.floats[4] = present.vignette;
    this.floats[5] = present.grain;
    this.integers[6] = present.frameIndex;
    this.integers[7] = present.transparent ? 1 : 0;
    this.integers[8] = present.denoised ? 1 : 0;
    this.floats[9] = present.contrast;
    this.device.queue.writeBuffer(this.uniforms, 0, this.values);

    const pass = encoder.beginRenderPass({
      label: "present",
      colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }
}
