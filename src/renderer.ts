import { PerspectiveCamera, type LightProbe, type Material, type Object3D, type Scene } from "three";
import { Accumulator } from "./accumulator.js";
import { type PackedBvh } from "./bvh.js";
import { BvhBuilder, createDefaultBvhWorker, type BvhWorkerFactory } from "./bvhBuilder.js";
import { cameraMatrices, type CameraMatrices } from "./camera.js";
import { requestRenderDevice, type RenderDevice } from "./device.js";
import { loadEnvironment, type PackedEnvironment } from "./environment.js";
import { DispatchScheduler } from "./dispatchScheduler.js";
import { packGeometry, type PackedGeometry } from "./geometry.js";
import { packLights } from "./lights.js";
import { NO_PROGRAM } from "./materialLayout.js";
import { NO_TEXTURE, planTextures, writeTextures, type PackedTextures } from "./texturePack.js";
import { PathTracePass, type FrameUniforms } from "./pathTracePass.js";
import { PresentPass } from "./presentPass.js";
import {
  CONVERGED_ERROR,
  DEFAULT_DENOISER,
  DenoisePass,
  mergeDenoiser,
  type DenoiserSettings,
  type DenoiseView,
} from "./denoise.js";
import { CaptureResolvePass, planarCaptures, type PlanarCaptures } from "./planarCapture.js";
import { prepareSceneAssets, type SceneAssets } from "./sceneAssets.js";
import { packVolumetricLights, VOLUMETRIC_WORDS } from "./volumetric.js";
import { environmentPower, uploadScene, type SceneBuffers } from "./sceneUpload.js";
import {
  DEFAULT_RENDER_ENVIRONMENT,
  DEFAULT_RENDER_SETTINGS,
  encodeColor,
  EventEmitter,
  mergeSettings,
  serializeCamera,
  serializeScene,
  srgbToLinear,
  type Bucket,
  type CameraState,
  type RenderCamera,
  type RenderEnvironment,
  type RenderBackgroundMode,
  type RenderImage,
  type RenderPhase,
  type RenderPixels,
  type RenderScene,
  type RenderSettings,
  type RenderSettingsInput,
  type SerializedScene,
} from "./upstream.js";

export type { RenderCamera, RenderImage, RenderPhase, RenderPixels, RenderScene } from "./upstream.js";

/**
 * The CPU renderer's events, with the same payloads, so a host listens to either renderer the same way. What differs
 * is when they fire: nothing is finished until the last sample lands, so the pixels arrive once.
 */
export interface GpuRendererEvents {
  /** The scene is on the device and tracing starts; `image` has the render size and is empty. */
  start: { width: number; height: number };
  /** The finished image, once, as a single region at (0, 0). `image` already holds it. */
  pixels: RenderPixels;
  /** Rows traced out of `total`, over every sample. `"prepare"` while the scene uploads, `"shade"` while it traces. */
  progress: { completed: number; total: number; phase: RenderPhase };
  /** The band of rows being traced, as a bucket of worker 0, or `null` when the device is idle. */
  bucket: { worker: number; bucket: Bucket | null; width: number; height: number };
  /** One more sample per pixel finished over the whole image. Only this renderer has it. */
  sample: { samples: number; target: number };
  /** `image` is finished. */
  complete: void;
  /** The frame failed; `render()` rejects with the same error. */
  error: Error;
}

/** The CPU renderer's options without its worker pool, which has no counterpart on a device. */
export interface GpuRendererOptions {
  scene?: RenderScene | SerializedScene;
  /** Defaults to a perspective camera at (5, 5, 5) looking at the origin. */
  camera?: RenderCamera;
  /** Merged over `DEFAULT_RENDER_SETTINGS`, group by group. */
  settings?: RenderSettingsInput;
  /** Defaults to `DEFAULT_RENDER_ENVIRONMENT`. */
  environment?: RenderEnvironment;
  /** Image size in CSS pixels; the render size also applies `pixelRatio` and `settings.renderScale`. */
  width: number;
  height: number;
  /** Defaults to 1. */
  pixelRatio?: number;
  /** How the denoiser filters, when `settings.globalIllumination.denoise` turns it on. Merged over `DEFAULT_DENOISER`. */
  denoiser?: Partial<DenoiserSettings>;
  /**
   * Builds the BVH of a large scene in a Web Worker, so the page keeps responding while it is split. The tree is the
   * same either way. Defaults to true where Web Workers exist, or when `createBvhWorker` is given; false builds on the
   * main thread. If the worker cannot start, the renderer warns once and builds on the main thread.
   */
  bvhWorker?: boolean;
  /**
   * Starts the BVH worker. Replace it when the bundler cannot follow `new URL("./bvhWorker.js", import.meta.url)`, or
   * outside the browser, with anything shaped like a Web Worker that runs `dist/bvhWorker.js`.
   */
  createBvhWorker?: BvhWorkerFactory;
}

interface PendingFrame {
  resolve: (image: RenderImage) => void;
  reject: (reason: Error) => void;
}

function abortError(message: string): Error {
  return Object.assign(new Error(message), { name: "AbortError" });
}

/** `render()` rejects with this when a newer frame, `cancel()` or `dispose()` replaced the frame it started. */
export const isAbortError = (error: unknown): boolean => error instanceof Error && error.name === "AbortError";

/** `#rrggbb` to linear rgb, the space the renderer works in. */
function linearColor(color: string | undefined, fallback: readonly number[]): number[] {
  const match = /^#?([0-9a-f]{6})$/i.exec(color ?? "");

  if (!match) return [...fallback];

  const value = parseInt(match[1], 16);

  return [(value >> 16) & 255, (value >> 8) & 255, value & 255].map(channel => srgbToLinear(channel / 255));
}

/** Hands the thread back to the page for a moment, so input and painting go on between submits. */
const yieldToPage = () => new Promise<void>(resolve => setTimeout(resolve, 0));

/**
 * How often the canvas is drawn and the page gets the thread back while a frame traces: about once a display frame.
 * Waiting for an animation frame after every band would leave the device idle for most of each one.
 */
const SHOW_EVERY_MS = 16;

const EMPTY_SCENE: SerializedScene = { meshes: [], lights: [], textures: [] };

/** Paths every pixel takes before adaptive sampling may stop it: below this its variance says too little. */
const ADAPTIVE_MIN = 32;
/** How many times the target a pixel that has not converged may take, when converged pixels leave the work to it. */
const ADAPTIVE_REACH = 4;
/**
 * The error under which a pixel stops: its luminance's standard error over the square root of its luminance, about
 * half a step of an 8 bit sRGB image in the mid tones.
 */
const ADAPTIVE_THRESHOLD = CONVERGED_ERROR;

/** Contiguous runs of the given sorted rows, as [first, end) pairs. */
function rowRanges(rows: readonly number[]): [number, number][] {
  const ranges: [number, number][] = [];

  for (const row of rows) {
    const last = ranges[ranges.length - 1];

    if (last && last[1] === row) last[1] = row + 1;
    else ranges.push([row, row + 1]);
  }

  return ranges;
}

/** How many paths per pixel a planar capture gets. It is a texture a shader distorts, not the image itself. */
const CAPTURE_SAMPLES = 32;

/** A scene on the device, with where its extra words went. */
interface PackedTarget {
  packed: { geometry: PackedGeometry; bvh: PackedBvh; textures: PackedTextures };
  buffers: SceneBuffers;
  skyData: number;
  volumetricData: number;
  volumetricCount: number;
}

/**
 * Path traces a Three.js scene on the GPU, behind the same API as `CpuRenderer`. The setters record the inputs;
 * `render()` opens a frame with whatever is current and accumulates samples until it reaches `settings.samples`.
 */
export class GpuRenderer extends EventEmitter<GpuRendererEvents> {
  private current: RenderSettings;
  private environment: RenderEnvironment;
  // Serializing walks the whole scene, so it is started by the setter and awaited by the frame that needs it.
  private scene: Promise<SerializedScene> = Promise.resolve(EMPTY_SCENE);
  // A LightProbe's harmonics, which the snapshot does not carry: read from the live scene when there is one.
  private probe: number[] = new Array(27).fill(0);
  // A Three camera is kept by reference and read at every `render()`, as the CPU renderer does; a snapshot stays fixed.
  private camera: RenderCamera | null = null;
  private width: number;
  private height: number;
  private pixelRatio: number;
  private output: RenderImage | null = null;
  private frame: PendingFrame | null = null;
  private isPaused = false;
  private resumed: (() => void) | null = null;
  private unpaused: Promise<void> = Promise.resolve();

  private ready: Promise<RenderDevice> | null = null;
  private gpu: RenderDevice | null = null;
  private accumulator: Accumulator | null = null;
  private pathTrace: PathTracePass | null = null;
  private present: PresentPass | null = null;
  private buffers: SceneBuffers | null = null;
  private packed: { geometry: PackedGeometry; bvh: PackedBvh; textures: PackedTextures } | null = null;
  private sceneDirty = true;
  // Compute bakes, the procedural sky and translated shaders: prepared once per scene.
  private assets: Promise<SceneAssets> | null = null;
  // What the packed triangles were built for, when points, lines, sprites or vertex shaders tie them to the camera.
  private packedView = "";
  // Where the packed scene's own extra words went: the sky's uniforms and the volumetric lights.
  private target: PackedTarget | null = null;
  // Whether the scene has planar captures, which are taken from the camera and so follow it.
  private planar = false;
  private resolve: CaptureResolvePass | null = null;
  private denoisePass: DenoisePass | null = null;
  private denoiserSettings: DenoiserSettings = { ...DEFAULT_DENOISER };
  // The size of the last frame that finished, which setDenoiser filters again; null while a frame traces.
  private finished: { width: number; height: number } | null = null;
  // Whether the denoiser's output holds this frame yet: until it does, the canvas shows the accumulation itself.
  private denoised = false;
  // How large a pixel is for the camera of the frame, which the specular filter sizes its footprint by.
  private denoiseView: DenoiseView = { pixelScale: 1, orthographic: false };
  // Reading an environment image is a fetch, so it is started once and awaited by the frame that needs it.
  private environmentSource = "";
  private environmentImage: Promise<PackedEnvironment | null> = Promise.resolve(null);
  private canvas: HTMLCanvasElement | null = null;
  private context: GPUCanvasContext | null = null;
  private scheduler: DispatchScheduler;
  private revision = 0;
  private frameIndex = 0;
  private disposed = false;
  private bvh: BvhBuilder;

  constructor(options: GpuRendererOptions) {
    super();
    this.bvh = new BvhBuilder(
      options.bvhWorker ?? (options.createBvhWorker !== undefined || typeof Worker !== "undefined"),
      options.createBvhWorker ?? createDefaultBvhWorker);
    this.width = Math.max(1, Math.floor(options.width));
    this.height = Math.max(1, Math.floor(options.height));
    this.pixelRatio = checkedPixelRatio(options.pixelRatio ?? 1);
    this.current = mergeSettings(DEFAULT_RENDER_SETTINGS, options.settings);
    this.denoiserSettings = mergeDenoiser(DEFAULT_DENOISER, options.denoiser);
    this.environment = DEFAULT_RENDER_ENVIRONMENT;
    this.scheduler = new DispatchScheduler({ maxRows: this.renderHeight });
    this.setEnvironment(options.environment ?? DEFAULT_RENDER_ENVIRONMENT);

    if (options.scene) this.setScene(options.scene);
    if (options.camera) this.setCamera(options.camera);
  }

  /** The settings the next frame renders with. */
  get settings(): Readonly<RenderSettings> {
    return this.current;
  }

  /** How the denoiser filters. */
  get denoiser(): Readonly<DenoiserSettings> {
    return this.denoiserSettings;
  }

  /**
   * Merges the given fields into the denoiser's settings. Filtering costs nothing like tracing, so a finished image is
   * filtered again at once, without tracing anything: `image` and the canvas take the result and `pixels` fires. A frame
   * in flight uses them from its next pass on.
   */
  setDenoiser(input: Partial<DenoiserSettings>) {
    this.denoiserSettings = mergeDenoiser(this.denoiserSettings, input);
    this.refilter();
  }

  /** Filters the finished image again with the current denoiser settings, or shows it unfiltered when it is off. */
  private refilter() {
    const finished = this.finished, gpu = this.gpu;

    if (!finished || !gpu || !this.accumulator || this.rendering) return;

    const revision = this.revision;
    const denoise = this.current.globalIllumination.denoise;

    void (async() => {
      const encoder = gpu.device.createCommandEncoder();

      if (denoise) this.encodeDenoise(gpu, encoder, finished.width, finished.height);
      else this.denoised = false;

      this.encodePresent(encoder, finished.width, finished.height);
      gpu.device.queue.submit([encoder.finish()]);

      const image = await this.readImage(gpu, finished.width, finished.height,
        denoise ? this.denoisePass?.output ?? null : null);

      // A frame started meanwhile owns the image now.
      if (revision !== this.revision || this.rendering) return;

      this.output = image;
      this.emit("pixels", { x: 0, y: 0, ...image });
    })().catch((error: unknown) => this.emit("error", error instanceof Error ? error : new Error(String(error))));
  }

  /** A frame is in flight: `render()` was called and has not settled. */
  get rendering(): boolean {
    return this.frame !== null;
  }

  get paused(): boolean {
    return this.isPaused;
  }

  /**
   * The image of the last finished frame, or an empty one of the render size while a frame traces; `null` before the
   * first frame starts. Each frame allocates a new buffer, so a reference taken at `complete` stays as it is.
   */
  get image(): RenderImage | null {
    return this.output;
  }

  get renderWidth(): number {
    return Math.max(1, Math.floor(this.width * this.pixelRatio * this.current.renderScale));
  }

  get renderHeight(): number {
    return Math.max(1, Math.floor(this.height * this.pixelRatio * this.current.renderScale));
  }

  /** Image size in CSS pixels for the next frame; `pixelRatio` stays as it is when omitted. */
  setSize(width: number, height: number, pixelRatio = this.pixelRatio) {
    this.width = Math.max(1, Math.floor(width));
    this.height = Math.max(1, Math.floor(height));
    this.pixelRatio = checkedPixelRatio(pixelRatio);
  }

  setPixelRatio(pixelRatio: number) {
    this.pixelRatio = checkedPixelRatio(pixelRatio);
  }

  /**
   * Takes a Three scene, serialized in the background, or the result of `serializeScene`. The live scene is not read
   * again until this is called once more.
   */
  setScene(scene: RenderScene | SerializedScene) {
    this.probe = "isScene" in scene && scene.isScene ? lightProbes(scene as Scene) : new Array(27).fill(0);
    this.scene = "isScene" in scene && scene.isScene
      ? serializeLive(scene as Scene)
      : Promise.resolve("meshes" in scene ? scene : (scene as { scene: SerializedScene }).scene);
    this.sceneDirty = true;
    this.assets = null;
  }

  /**
   * A Three camera is read at every `render()`, so moving it or changing its aspect only needs a new frame.
   * A `CameraState` snapshot is used as it is.
   */
  setCamera(camera: RenderCamera) {
    this.camera = camera;
  }

  /**
   * Merges the given fields into the current settings, group by group. Turning the denoiser on or off also filters the
   * finished image again at once, as `setDenoiser` does, since that needs no tracing.
   */
  setSettings(settings: RenderSettingsInput) {
    const denoise = this.current.globalIllumination.denoise;

    this.current = mergeSettings(this.current, settings);

    if (this.current.globalIllumination.denoise !== denoise) this.refilter();
  }

  setEnvironment(environment: RenderEnvironment) {
    this.environment = environment;

    // Only the image itself has to be read again; its strength and rotation are uniforms.
    const source = environment.kind === "hdri" ? environment.url : "";

    if (source === this.environmentSource) return;

    this.environmentSource = source;
    this.environmentImage = loadEnvironment(environment).catch(() => null);
    // The image shares a buffer with the scene's own textures, so the upload has to happen again.
    this.sceneDirty = true;
  }

  /** Used by `attachCanvas`; pass `null` to detach. */
  setOutputCanvas(canvas: HTMLCanvasElement | null) {
    this.canvas = canvas;
    this.context = null;
  }

  /**
   * Starts a frame with the current scene, camera, settings, environment and size, discarding any frame in
   * progress, and accumulates `settings.samples` samples per pixel. Resolves with the finished image. Rejects with an
   * `AbortError` when a later `render()`, `cancel()` or `dispose()` replaces the frame, and with the error when the
   * frame fails.
   */
  render(): Promise<RenderImage> {
    if (this.disposed) return Promise.reject(new Error("GPU renderer: this renderer was disposed."));

    // The camera as it is at this call, even if it moves while the scene is still serializing.
    const camera = !this.camera ? defaultCamera()
      : "isCamera" in this.camera ? serializeCamera(this.camera) : this.camera;

    this.abort("A newer render() replaced this frame.");

    const revision = ++this.revision;

    const settled = new Promise<RenderImage>((resolve, reject) => {
      this.frame = { resolve, reject };
    });

    this.accumulate(revision, camera).then(
      image => {
        if (revision !== this.revision) return;

        this.emit("complete");
        this.settle()?.resolve(image);
      },
      (error: unknown) => {
        // A replaced frame was already rejected by whatever replaced it.
        if (revision !== this.revision || isAbortError(error)) return;

        this.fail(error instanceof Error ? error : new Error(String(error)));
      });

    return settled;
  }

  /** Stops between two dispatches; `resume()` continues the same frame. */
  pause() {
    if (this.isPaused) return;

    this.isPaused = true;
    this.unpaused = new Promise(resolve => { this.resumed = resolve; });
  }

  resume() {
    if (!this.isPaused) return;

    this.isPaused = false;
    this.resumed?.();
    this.resumed = null;
  }

  /**
   * Stops the frame in flight and keeps the device for the next one. `render()` rejects with an `AbortError` and
   * `image` keeps what it held. A paused frame is cancelled too, and the next one starts unpaused. Does nothing
   * without a frame in flight.
   */
  cancel() {
    if (this.disposed || !this.frame) return;

    this.revision++;
    this.resume();
    this.emitIdle();
    this.abort("The frame was cancelled.");
  }

  /** Releases the device. A frame in flight rejects with an `AbortError`. */
  dispose() {
    if (this.disposed) return;

    this.disposed = true;
    this.revision++;
    this.resume();
    this.abort("The renderer was disposed.");
    this.bvh.dispose();
    this.buffers?.destroy();
    this.accumulator?.destroy();
    this.denoisePass?.destroy();
    this.gpu?.device.destroy();
    this.buffers = null;
    this.accumulator = null;
    this.gpu = null;
    this.ready = null;
  }

  /** Takes the frame in flight, if any, so it can be settled after the events go out. */
  private settle(): PendingFrame | null {
    const frame = this.frame;

    this.frame = null;

    return frame;
  }

  /** Settles the frame in flight, if any, with an `AbortError`. */
  private abort(reason: string) {
    this.settle()?.reject(abortError(reason));
  }

  private fail(error: Error) {
    const frame = this.settle();

    this.emitIdle();
    this.emit("error", error);
    frame?.reject(error);
  }

  private emitIdle() {
    this.emit("bucket", { worker: 0, bucket: null, width: this.output?.width ?? 0, height: this.output?.height ?? 0 });
  }

  private async device(): Promise<RenderDevice> {
    this.ready ??= requestRenderDevice({
      onLost: reason => {
        if (!this.disposed) this.fail(new Error(`GPU renderer: the device was lost (${reason}).`));
      },
    });
    this.gpu = await this.ready;

    return this.gpu;
  }

  private async accumulate(revision: number, camera: CameraState): Promise<RenderImage> {
    this.emit("progress", { completed: 0, total: 1, phase: "prepare" });

    const gpu = await this.device();
    const scene = await this.scene;
    const environment = await this.environmentImage;
    const width = this.renderWidth, height = this.renderHeight;

    this.check(revision);
    this.assets ??= prepareSceneAssets(scene);

    const assets = await this.assets;

    this.check(revision);
    await this.prepare(gpu, revision, assets, camera, width, height, environment);
    this.output = { width, height, data: new Uint8ClampedArray(width * height * 4) };
    this.emit("start", { width, height });

    const target = Math.max(1, Math.round(this.current.samples));
    // Adaptive sampling: the frame has the work of `target` passes over the whole image to spend. Pixels that have
    // converged stop taking samples, and a row with none left stops being dispatched, so what they would have used goes
    // to the rows that still need it, up to a few times `target` paths in the noisiest pixels.
    const adaptive = target > ADAPTIVE_MIN;
    const maxPasses = adaptive ? target * ADAPTIVE_REACH : target;
    const budget = target * height;
    const accumulator = this.accumulator as Accumulator;
    let active: number[] = Array.from({ length: height }, (_, row) => row);
    let spent = 0, band = 0, pass = 0;
    let shown = performance.now();
    const denoise = this.current.globalIllumination.denoise;

    this.denoised = false;
    this.finished = null;

    const matrices = cameraMatrices(camera, width, height);

    // Two over the projection's vertical scale and the height: one pixel's extent, per unit of depth in perspective.
    this.denoiseView = { pixelScale: 2 / (matrices.projection[5] * height), orthographic: matrices.orthographic };

    while (pass < maxPasses && spent < budget && active.length) {
      // Only a pass whose pixels may stop needs its rows flagged; before that every row is busy anyway.
      const flagging = adaptive && pass >= ADAPTIVE_MIN;

      if (flagging) {
        const clear = gpu.device.createCommandEncoder();

        clear.clearBuffer(accumulator.buffer, accumulator.rowFlagsOffset, accumulator.rowFlagsLength);
        gpu.device.queue.submit([clear.finish()]);
      }

      for (const [first, end] of rowRanges(active)) {
        let row = first;

        while (row < end) {
          if (this.isPaused) await this.unpaused;

          this.check(revision);

          const rows = this.scheduler.next(end - row);
          const started = performance.now();
          const encoder = gpu.device.createCommandEncoder();

          this.emit("bucket", { worker: 0, bucket: { x: 0, y: row, width, height: rows, index: band++ }, width, height });
          this.pathTrace?.encode(encoder, {
            ...this.frameUniforms(camera, width, height, assets),
            bandOrigin: [0, row],
            bandSize: [width, rows],
            sampleIndex: pass,
            sampleCount: 1,
            adaptiveMin: adaptive ? ADAPTIVE_MIN : 0,
            adaptiveThreshold: ADAPTIVE_THRESHOLD,
          });
          // The canvas is drawn about once a display frame; in between the device only traces.
          const show = this.canvas !== null && started - shown >= SHOW_EVERY_MS;

          if (show) this.encodePresent(encoder, width, height);

          gpu.device.queue.submit([encoder.finish()]);
          await gpu.device.queue.onSubmittedWorkDone();

          this.scheduler.record(performance.now() - started, rows);
          row += rows;
          spent += rows;
          this.emit("progress", { completed: Math.min(spent, budget), total: budget, phase: "shade" });

          if (show) {
            shown = performance.now();
            await yieldToPage();
          }
        }
      }

      pass++;
      this.frameIndex++;

      // The canvas shows the denoised image from the end of each pass on, and the bands of the next pass refine
      // underneath it until it is denoised again.
      if (denoise && this.canvas && this.context) {
        const encoder = gpu.device.createCommandEncoder();

        this.encodeDenoise(gpu, encoder, width, height);
        this.encodePresent(encoder, width, height);
        gpu.device.queue.submit([encoder.finish()]);
      }

      this.emit("sample", { samples: Math.min(target, Math.round(spent / height)), target });

      if (flagging) active = await this.activeRows(gpu, accumulator);
    }

    // Converged early, or spent: either way the image is as finished as this frame makes it.
    this.emit("progress", { completed: budget, total: budget, phase: "shade" });
    this.emit("sample", { samples: target, target });

    this.check(revision);

    if (denoise) {
      const encoder = gpu.device.createCommandEncoder();

      this.encodeDenoise(gpu, encoder, width, height);
      this.encodePresent(encoder, width, height);
      gpu.device.queue.submit([encoder.finish()]);
    }

    const image = await this.readImage(gpu, width, height, denoise ? this.denoisePass?.output ?? null : null);

    this.check(revision);
    this.output = image;
    this.finished = { width, height };
    this.emitIdle();
    this.emit("pixels", { x: 0, y: 0, ...image });

    return image;
  }

  /** The rows a pass still traced something in, read back from the accumulator's row flags. */
  private async activeRows(gpu: RenderDevice, accumulator: Accumulator): Promise<number[]> {
    const staging = gpu.device.createBuffer({
      size: accumulator.rowFlagsLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = gpu.device.createCommandEncoder();

    encoder.copyBufferToBuffer(accumulator.buffer, accumulator.rowFlagsOffset, staging, 0, accumulator.rowFlagsLength);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);

    const flags = new Float32Array(staging.getMappedRange().slice(0));

    staging.unmap();
    staging.destroy();

    const rows: number[] = [];

    for (let row = 0; row < flags.length / 4; row++) if (flags[row * 4] > 0) rows.push(row);

    return rows;
  }

  private check(revision: number) {
    if (revision !== this.revision || this.disposed) throw abortError("A newer render() replaced this frame.");
  }

  /** Uploads what changed and makes sure the passes point at the current buffers. */
  private async prepare(
    gpu: RenderDevice,
    revision: number,
    assets: SceneAssets,
    camera: CameraState,
    width: number,
    height: number,
    image: PackedEnvironment | null) {
    this.pathTrace ??= new PathTracePass(gpu.device);
    this.pathTrace.setPrograms(assets.programs);

    // A procedural sky stands in for the renderer's own environment, as it does in the CPU renderer.
    const environment = assets.sky?.environment ?? image;
    const matrices = cameraMatrices(camera, width, height);
    const view = `${JSON.stringify(camera)}|${width}|${height}`;
    const moved = (!!this.packed?.geometry.cameraDependent || this.planar) && view !== this.packedView;

    if (this.sceneDirty || !this.buffers || moved) {
      // Planar surfaces need their captures, from this camera, before the scene that samples them is packed.
      const planar = planarCaptures(assets.scene, camera, width, height);

      const packed = await this.packScene(
        gpu, revision, planar?.scene ?? assets.scene, assets, matrices, width, height, environment);

      this.packed = packed.packed;
      this.buffers?.destroy();
      this.buffers = packed.buffers;
      this.target = packed;
      this.sceneDirty = false;
      this.planar = !!planar;

      if (planar) await this.renderCaptures(gpu, revision, planar, assets, environment);

      // Only once the captures are in: a frame replaced while they were taken has to take them again.
      this.packedView = view;
    }

    const resized = !this.accumulator || this.accumulator.resize(width, height);

    this.accumulator ??= new Accumulator(gpu.device, width, height);

    if (resized) this.scheduler = new DispatchScheduler({ maxRows: height });

    this.pathTrace.setBindings(this.buffers, this.accumulator.buffer);
    this.configureCanvas(gpu, width, height);

    // Every frame starts from nothing: the camera, the scene or the settings may have moved since the last one. The
    // canvas is drawn from the cleared accumulation at once, so the last image does not linger until the first band of
    // the new one lands: every pixel goes see-through, and the new samples arrive over whatever the page shows below.
    const encoder = gpu.device.createCommandEncoder();

    this.denoised = false;
    this.accumulator.clear(encoder);
    this.encodePresent(encoder, width, height);
    gpu.device.queue.submit([encoder.finish()]);
  }

  /** Packs and uploads one scene: its textures, geometry, hierarchy, lights and every extra word they point at. */
  private async packScene(
    gpu: RenderDevice,
    revision: number,
    scene: SerializedScene,
    assets: SceneAssets,
    matrices: CameraMatrices,
    width: number,
    height: number,
    environment: PackedEnvironment | null): Promise<PackedTarget> {
    // The textures are planned first: a material refers to one by the index it lands on in that table, and to its
    // extra words by where they land after the texels. The environment rides along in the same buffer.
    const plan = planTextures(scene, environment ? [environment.texture] : []);
    const extra: number[] = [];

    const append = (words: ArrayLike<number>) => {
      const at = plan.extraOffset + extra.length;

      for (let i = 0; i < words.length; i++) extra.push(words[i]);

      return at;
    };

    const cdf = environment
      ? append(new Uint32Array(environment.distribution.buffer, environment.distribution.byteOffset,
        environment.distribution.length))
      : plan.extraOffset;

    const textureData = new Map(scene.textures.map(texture => [texture.id, texture]));

    const geometry = packGeometry(scene, {
      textures: plan.indexOf,
      textureData,
      camera: matrices,
      width,
      height,
      programs: assets.programs,
      extraBase: plan.extraOffset + extra.length,
      fog: scene.fog,
    });

    append(geometry.extraWords);

    const lights = packLights(geometry, scene.lights, { power: environmentPower(environment) },
      { textures: plan.indexOf, extraBase: plan.extraOffset + extra.length });

    append(lights.extraWords);

    const foreground = assets.programs.foreground;

    const skyData = foreground && assets.sky
      ? append(foreground.packUniforms({ ...assets.sky.uniforms, uCpuSky: { texture: "procedural-sky" } }, plan.indexOf))
      : 0;

    const volumetric = packVolumetricLights(scene.lights);
    const volumetricData = append(new Uint32Array(volumetric.buffer));
    const textures = writeTextures(plan, new Uint32Array(extra));
    const bvh = await this.bvh.build(geometry.positions, geometry.triangleCount);

    // Nothing is on the device yet, so a frame replaced while the hierarchy was built has nothing to give back.
    this.check(revision);

    const packed = { geometry, bvh, textures };
    const buffers = uploadScene(gpu.device, geometry, packed.bvh, textures, lights, environment, cdf);

    if (buffers.bytes.largest > gpu.maxStorageBufferBindingSize) {
      buffers.destroy();
      throw new Error("GPU renderer: this scene needs a larger storage buffer than the device allows.");
    }

    return {
      packed,
      buffers,
      skyData,
      volumetricData,
      volumetricCount: Math.min(volumetric.length / VOLUMETRIC_WORDS, 8),
    };
  }

  /**
   * Renders a scene's planar captures with this renderer, from the mirrored and the real camera, and writes each into
   * the texels of its placeholder in the main scene's texture buffer. Everything is queued ahead of the frame itself,
   * which the queue then runs in order.
   */
  private async renderCaptures(gpu: RenderDevice, revision: number, planar: PlanarCaptures, assets: SceneAssets,
    environment: PackedEnvironment | null) {
    const main = this.target as PackedTarget;
    const pathTrace = this.pathTrace as PathTracePass;
    const { width, height } = planar;
    const accumulator = new Accumulator(gpu.device, width, height);
    const empty: SerializedScene = { ...planar.captureScene, meshes: [] };
    const scenes = new Map<string, PackedTarget>();

    this.resolve ??= new CaptureResolvePass(gpu.device);

    try {
      for (const capture of planar.captures) {
        let target = scenes.get(capture.scene);

        if (!target) {
          target = await this.packScene(gpu, revision, capture.scene === "scene" ? planar.captureScene : empty, assets,
            cameraMatrices(capture.camera, width, height), width, height, environment);
          scenes.set(capture.scene, target);
        }

        const clear = gpu.device.createCommandEncoder();

        accumulator.clear(clear);
        gpu.device.queue.submit([clear.finish()]);
        pathTrace.setBindings(target.buffers, accumulator.buffer);

        const uniforms = {
          ...this.frameUniforms(capture.camera, width, height, assets, target, capture.background),
          lens: [0, 1, 0, 0],
          volumetricCount: 0,
          clipPlane: capture.clip ?? [0, 0, 0, 0],
          bandOrigin: [0, 0],
          bandSize: [width, height],
          sampleCount: 1,
        };

        for (let sample = 0; sample < CAPTURE_SAMPLES; sample++) {
          const encoder = gpu.device.createCommandEncoder();

          pathTrace.encode(encoder, { ...uniforms, sampleIndex: sample });
          gpu.device.queue.submit([encoder.finish()]);
        }

        const offset = (id: string | undefined) => {
          const index = id === undefined ? undefined : main.packed.textures.indexOf.get(id);

          return index === undefined ? NO_TEXTURE : main.packed.textures.data[index * 16 + 2];
        };

        const encoder = gpu.device.createCommandEncoder();

        (this.resolve as CaptureResolvePass).encode(encoder, accumulator.buffer, main.buffers.textures, width, height,
          offset(capture.texture), offset(capture.depth));
        gpu.device.queue.submit([encoder.finish()]);
      }
    } finally {
      // The queue keeps what it still needs alive until the work that uses it is done.
      for (const target of scenes.values()) target.buffers.destroy();

      accumulator.destroy();
    }
  }

  private configureCanvas(gpu: RenderDevice, width: number, height: number) {
    if (!this.canvas) return;

    const format = navigator.gpu.getPreferredCanvasFormat();

    this.canvas.width = width;
    this.canvas.height = height;

    // Premultiplied always: a transparent background needs it, and a pixel no sample has reached yet stays see-through
    // rather than black, so a page can show something under the canvas until the image covers it.
    if (!this.context) {
      this.context = this.canvas.getContext("webgpu");
      this.context?.configure({ device: gpu.device, format, alphaMode: "premultiplied" });
    }

    if (!this.present || this.present.format !== format) this.present = new PresentPass(gpu.device, format);

    this.present.setAccumulation((this.accumulator as Accumulator).buffer);
  }

  private encodePresent(encoder: GPUCommandEncoder, width: number, height: number) {
    if (!this.context || !this.present) return;

    this.present.encode(encoder, this.context.getCurrentTexture().createView(), {
      width,
      height,
      exposure: this.current.exposure,
      toneMapping: this.current.toneMapping,
      vignette: this.current.vignette,
      grain: this.current.grain,
      frameIndex: this.frameIndex,
      transparent: this.current.background.mode === "transparent",
      denoised: this.denoised ? this.denoisePass?.output : null,
    });
  }

  /** Denoises the accumulation as it stands, for the canvas and for the finished image. */
  private encodeDenoise(gpu: RenderDevice, encoder: GPUCommandEncoder, width: number, height: number) {
    this.denoisePass ??= new DenoisePass(gpu.device);
    this.denoisePass.encode(encoder, (this.accumulator as Accumulator).buffer, width, height, this.denoiserSettings,
      this.denoiseView);
    this.denoised = true;
  }

  private frameUniforms(
    camera: CameraState,
    width: number,
    height: number,
    assets: SceneAssets,
    target: PackedTarget | null = this.target,
    backgroundOverride?: RenderBackgroundMode)
    : Omit<FrameUniforms, "bandOrigin" | "bandSize" | "sampleIndex" | "sampleCount"> {
    const buffers = target?.buffers ?? null;
    const matrices: CameraMatrices = cameraMatrices(camera, width, height);
    const gradient = this.environment.kind === "gradient" ? this.environment : null;
    const ambient = buffers?.ambient ?? { sky: [0, 0, 0], ground: [0, 0, 0], axis: [0, 1, 0] };
    const surrounding = buffers?.environment;
    const { shadows, fog, depthOfField } = this.current;
    const background = { ...this.current.background, mode: backgroundOverride ?? this.current.background.mode };
    const sky = assets.sky;
    const sceneFog = assets.scene.fog;
    // The CPU renderer's lens: a 35 mm frame, its focal length from the field of view, and the f-number's aperture.
    const focalLength = 0.035 / (2 * Math.tan(matrices.fov * Math.PI / 360));

    const lensRadius = depthOfField.enabled && !matrices.orthographic
      ? focalLength / (2 * Math.max(depthOfField.aperture, 1e-3))
      : 0;

    const backgroundMode = background.mode === "color" ? 1 : background.mode === "transparent" ? 2 : 0;

    return {
      cameraToWorld: matrices.cameraWorld,
      worldToCamera: matrices.view,
      projection: matrices.projection,
      inverseProjection: matrices.inverseProjection,
      environmentTop: [...linearColor(gradient?.topColor, [1, 1, 1]), gradient?.exponent ?? 2],
      environmentBottom: [
        ...linearColor(gradient?.bottomColor, [0, 0, 0]),
        sky ? sky.intensity : this.environment.intensity ?? 1,
      ],
      ambientSky: [...ambient.sky, 0],
      ambientGround: [...ambient.ground, 0],
      ambientAxis: [...ambient.axis, 0],
      ...Object.fromEntries(Array.from({ length: 9 }, (_, k) =>
        [`probe${k}`, [...this.probe.slice(k * 3, k * 3 + 3), 0]])),
      // The settings' fog covers every surface; otherwise the scene's own fog covers the materials that accept it.
      fogColor: fog.enabled
        ? [...linearColor(fog.color, [0.5, 0.5, 0.5]), 1]
        : sceneFog ? [...sceneFog.color, sceneFog.density !== undefined ? 2 : 1] : [0, 0, 0, 0],
      fogParams: fog.enabled
        ? [fog.near, fog.far, 0, 1]
        : [sceneFog?.near ?? 0, sceneFog?.far ?? 0, sceneFog?.density ?? 0, 0],
      background: [...linearColor(background.color, [0, 0, 0]), backgroundMode],
      lens: [lensRadius, Math.max(depthOfField.focusDistance, 1e-3), depthOfField.blades, 0],
      resolution: [width, height],
      cameraKind: matrices.orthographic ? 1 : 0,
      bounces: Math.max(1, Math.round(this.current.globalIllumination.bounces)),
      triangleCount: buffers?.triangleCount ?? 0,
      lightCount: buffers?.lightCount ?? 0,
      areaLightCount: buffers?.areaLightCount ?? 0,
      clampIndirect: Math.max(0, this.current.globalIllumination.clamp),
      // The setting is the angular diameter a light covers, in degrees, as it is for the CPU renderer's shadow blur.
      shadowSpread: shadows.enabled ? Math.tan(Math.max(0, shadows.softness) * Math.PI / 360) : 0,
      shadowsEnabled: shadows.enabled,
      uvOffset: buffers?.uvOffset ?? 0,
      orderOffset: buffers?.orderOffset ?? 0,
      colorOffset: buffers?.colorOffset ?? 0,
      hasColors: buffers?.hasColors ?? false,
      varyingOffset: buffers?.varyingOffset ?? 0,
      environmentTexture: surrounding?.texture ?? NO_TEXTURE,
      environmentCdf: surrounding?.cdf ?? 0,
      environmentCdfSize: [surrounding?.width ?? 0, surrounding?.height ?? 0],
      environmentTotal: surrounding?.total ?? 0,
      // A procedural sky ignores the rotation, so its sun stays where its light is.
      environmentRotation: sky ? 0 : (this.environment.rotation ?? 0) * Math.PI / 180,
      environmentPick: surrounding?.pick ?? 0,
      skyProgram: sky && assets.programs.foreground ? assets.programs.foreground.id : NO_PROGRAM,
      skyData: target?.skyData ?? 0,
      skyBackground: sky ? (sky.background ? 1 : 0) : 1,
      skyIntensity: sky ? Math.max(sky.intensity, 1e-6) : 0,
      pixelCount: width * height,
      volumetricCount: target?.volumetricCount ?? 0,
      volumetricData: target?.volumetricData ?? 0,
    };
  }

  /**
   * Copies the accumulation buffer back and resolves it on the CPU, with the same exposure, tone mapping and sRGB
   * encoding the presentation pass applies. Grain is a presentation effect and is not part of the returned image.
   */
  private async readImage(
    gpu: RenderDevice, width: number, height: number, denoised: GPUBuffer | null): Promise<RenderImage> {
    const accumulator = this.accumulator as Accumulator;
    // The radiance and the coverage; the guides and the row flags after them are not part of the image.
    const length = width * height * 32;

    const read = async(buffer: GPUBuffer, bytes: number) => {
      const usage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
      const staging = gpu.device.createBuffer({ size: bytes, usage });
      const encoder = gpu.device.createCommandEncoder();

      encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
      gpu.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);

      const floats = new Float32Array(staging.getMappedRange().slice(0));

      staging.unmap();
      staging.destroy();

      return floats;
    };

    const floats = await read(accumulator.buffer, length);
    // The denoiser's output is the mean radiance itself, one vec4 per pixel.
    const filtered = denoised ? await read(denoised, width * height * 16) : null;

    const data = new Uint8ClampedArray(width * height * 4);
    const { exposure, toneMapping, vignette } = this.current;
    const transparent = this.current.background.mode === "transparent";
    const coverage = width * height * 4;

    for (let pixel = 0; pixel < width * height; pixel++) {
      const samples = floats[pixel * 4 + 3];
      const x = (pixel % width) / width - 0.5, y = Math.floor(pixel / width) / height - 0.5;
      const falloff = Math.max(0, 1 - vignette * (x * x + y * y) * 2);

      // The radiance is premultiplied by the coverage, which a transparent background makes the alpha.
      const alpha = !transparent ? 1
        : samples > 0 ? Math.min(1, Math.max(0, floats[coverage + pixel * 4] / samples)) : 0;

      const mean = (channel: number) =>
        (filtered ? filtered[pixel * 4 + channel] : floats[pixel * 4 + channel] / samples);

      const color = samples > 0 && alpha > 0
        ? [0, 1, 2].map(channel => mean(channel) / alpha * falloff)
        : [0, 0, 0];

      const [r, g, b] = encodeColor(color, exposure, toneMapping);

      data.set([r, g, b, Math.round(alpha * 255)], pixel * 4);
    }

    return { width, height, data };
  }
}

/** The material types the CPU renderer's snapshot accepts by name. */
const SERIALIZABLE = new Set([
  "PointsMaterial", "LineBasicMaterial", "LineDashedMaterial", "SpriteMaterial", "MeshBasicMaterial",
  "MeshLambertMaterial", "MeshPhongMaterial", "MeshToonMaterial", "MeshStandardMaterial", "MeshPhysicalMaterial",
  "MeshNormalMaterial", "MeshDepthMaterial", "MeshDistanceMaterial", "MeshMatcapMaterial", "ShadowMaterial",
  "ShaderMaterial", "RawShaderMaterial"]);

/**
 * Serializes a live scene. The snapshot accepts materials by their type name, so a subclass of ShaderMaterial with a
 * name of its own (LineMaterial, for fat lines) would be refused; it passes as the class it extends, which is all the
 * snapshot reads of it, and gets its name back once the materials have been read.
 */
async function serializeLive(scene: Scene): Promise<SerializedScene> {
  const renamed = new Map<Material, string>();

  scene.traverse(object => {
    const value = (object as Object3D & { material?: Material | Material[] }).material;

    for (const material of Array.isArray(value) ? value : value ? [value] : []) {
      const shader = material as Material & { isShaderMaterial?: boolean; isRawShaderMaterial?: boolean };

      if (shader.isShaderMaterial && !SERIALIZABLE.has(material.type) && !renamed.has(material)) {
        renamed.set(material, material.type);
        material.type = shader.isRawShaderMaterial ? "RawShaderMaterial" : "ShaderMaterial";
      }
    }
  });

  try {
    // The path tracer samples only the base level of a texture: the mip chain would be built and thrown away.
    return (await serializeScene(scene, { mipmaps: false })).scene;
  } finally {
    for (const [material, type] of renamed) material.type = type;
  }
}

/**
 * The summed coefficients of every LightProbe the renderer would draw, times its intensity, as Three's WebGLLights sums
 * them. The snapshot keeps a probe's colour and intensity but not its harmonics, so they are read from the live scene.
 */
function lightProbes(scene: Scene): number[] {
  const sum = new Array(27).fill(0);

  const visit = (object: Object3D) => {
    if (!object.visible || (object.userData?.cpuRenderer as { exclude?: boolean } | undefined)?.exclude) return;

    const probe = object as LightProbe;

    if (probe.isLightProbe)
      probe.sh.coefficients.forEach((coefficient, k) => {
        sum[k * 3] += coefficient.x * probe.intensity;
        sum[k * 3 + 1] += coefficient.y * probe.intensity;
        sum[k * 3 + 2] += coefficient.z * probe.intensity;
      });

    for (const child of object.children) visit(child);
  };

  visit(scene);

  return sum;
}

function checkedPixelRatio(pixelRatio: number): number {
  if (!Number.isFinite(pixelRatio) || pixelRatio <= 0)
    throw new Error("GPU renderer: pixelRatio must be a positive number.");

  return pixelRatio;
}

/** A perspective camera at (5, 5, 5) looking at the origin, for a frame rendered before any camera is set. */
function defaultCamera(): CameraState {
  const camera = new PerspectiveCamera(50, 1, 0.1, 1000);

  camera.position.set(5, 5, 5);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  return serializeCamera(camera);
}
