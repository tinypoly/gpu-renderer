/** Acquiring a device, and the questions a host asks before choosing this renderer at all. */

/** What the renderer needs from the platform, and what is missing when it cannot run. */
export interface WebGpuSupport {
  /** True when an adapter answered. */
  supported: boolean;
  /** Why the renderer cannot run here, in one sentence. Empty when `supported`. */
  reason: string;
}

export interface RenderDevice {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** Largest storage buffer this device accepts, which is the ceiling on how big a scene can be. */
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

export interface RenderDeviceOptions {
  /** Called when the device is lost. The renderer stops; choosing what to do next is the host's decision. */
  onLost?: (reason: string) => void;
}

/**
 * Whether this context exposes WebGPU at all. It is a cheap, synchronous check for a fallback decision;
 * it does not prove that an adapter exists, which only `requestWebGpuSupport` can tell.
 */
export function isWebGpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && navigator.gpu != null;
}

/**
 * Asks the platform for an adapter, which is the only honest answer to "can this machine render?".
 * A browser can expose `navigator.gpu` and still hand back no adapter (a blocklisted driver, a headless session
 * without a software fallback), so hosts deciding between renderers should await this one.
 */
export async function requestWebGpuSupport(): Promise<WebGpuSupport> {
  if (!isWebGpuAvailable())
    return { supported: false, reason: "This browser does not expose WebGPU." };

  try {
    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) return { supported: false, reason: "WebGPU is exposed, but no adapter was available." };

    return { supported: true, reason: "" };
  } catch(error) {
    return { supported: false, reason: `Requesting a WebGPU adapter failed: ${(error as Error).message}` };
  }
}

/**
 * The device the renderer runs on, asking for everything the adapter offers. Scene geometry, the hierarchy and the
 * accumulation buffer all live in storage buffers, so the default 128 MiB binding size is the first ceiling a real
 * scene meets, and adapters routinely allow far more.
 */
export async function requestRenderDevice(options: RenderDeviceOptions = {}): Promise<RenderDevice> {
  if (!isWebGpuAvailable())
    throw new Error("GPU renderer: this browser does not expose WebGPU.");

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });

  if (!adapter)
    throw new Error("GPU renderer: no WebGPU adapter was available.");

  // Half floats and the adapter's widest workgroups are what the OIDN denoiser runs fastest with; the path tracer
  // needs neither, and an adapter without them still gets a device.
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("shader-f16") ? ["shader-f16"] : [];

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    },
  });

  void device.lost.then(info => options.onLost?.(info.message || String(info.reason)));

  return {
    adapter,
    device,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxBufferSize: device.limits.maxBufferSize,
  };
}
