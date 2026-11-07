import { SLOT_CHANNELS } from "./texturePack.js";

/**
 * Memory diagnostics: what the renderer holds on the device, by label, and the page's JavaScript heap, logged at each
 * step of a frame. Every buffer and texture the device creates is counted until it is destroyed or collected, so a
 * step that leaves the device fuller than it found it shows up as a growing total between frames.
 */

interface Allocation {
  label: string;
  bytes: number;
}

const live = new Map<number, Allocation>();
let nextId = 0;
let enabled = false;

// A resource dropped without destroy() still holds its memory until it is collected; then it leaves the count.
const collected = typeof FinalizationRegistry === "undefined"
  ? null
  : new FinalizationRegistry<number>(id => live.delete(id));

const MB = 1024 * 1024;

/** Turns the memory log on or off. It is off by default. */
export function setMemoryLogging(on: boolean) {
  enabled = on;
}

/** Whether the memory log is on, so a caller can skip measuring what nobody will read. */
export function memoryLogging(): boolean {
  return enabled;
}

function track<T extends GPUBuffer | GPUTexture>(resource: T, label: string, bytes: number): T {
  const id = nextId++;
  const destroy = resource.destroy.bind(resource);

  live.set(id, { label, bytes });
  collected?.register(resource, id);

  resource.destroy = () => {
    live.delete(id);
    destroy();
  };

  return resource;
}

/** Bytes per texel of the formats the renderer and a canvas use; anything else counts as four. */
function texelBytes(format: GPUTextureFormat): number {
  if (format.startsWith("rgba32")) return 16;
  if (format.startsWith("rgba16") || format.startsWith("rg32")) return 8;
  if (format.startsWith("r8") && !format.startsWith("r8g8")) return 1;

  return 4;
}

/** Counts every buffer and texture the device creates from now on. */
export function trackDevice(device: GPUDevice) {
  const createBuffer = device.createBuffer.bind(device);
  const createTexture = device.createTexture.bind(device);

  device.createBuffer = descriptor =>
    track(createBuffer(descriptor), descriptor.label || "unlabeled", descriptor.size);

  device.createTexture = descriptor => {
    const size = descriptor.size as number[] | GPUExtent3DDict;

    const [width, height = 1, layers = 1] = Array.isArray(size)
      ? size
      : [size.width, size.height ?? 1, size.depthOrArrayLayers ?? 1];

    return track(createTexture(descriptor), descriptor.label || "texture",
      width * height * layers * texelBytes(descriptor.format) * (descriptor.mipLevelCount ?? 1));
  };
}

const mb = (bytes: number) => Math.round(bytes / MB * 10) / 10;

/** Bytes of every typed array and ArrayBuffer reachable from a value, each counted once, without crossing into DOM. */
export function retainedBytes(value: unknown): number {
  const seen = new Set<unknown>();
  const buffers = new Set<ArrayBufferLike>();
  const stack = [value];
  let plain = 0;

  while (stack.length) {
    const item = stack.pop();

    if (item === null || typeof item !== "object" || seen.has(item)) {
      if (typeof item === "string") plain += item.length * 2;

      continue;
    }

    seen.add(item);

    if (ArrayBuffer.isView(item)) {
      buffers.add(item.buffer);
      continue;
    }

    if (item instanceof ArrayBuffer) {
      buffers.add(item);
      continue;
    }

    if (Array.isArray(item)) {
      // A plain array of numbers costs eight bytes an element, which is how a large one goes unnoticed.
      plain += item.length * 8;
      for (const element of item) if (typeof element === "object" || typeof element === "string") stack.push(element);
      continue;
    }

    if (typeof (item as { nodeType?: unknown }).nodeType === "number") continue;

    for (const key of Object.keys(item)) stack.push((item as Record<string, unknown>)[key]);
  }

  let typed = 0;

  for (const buffer of buffers) typed += buffer.byteLength;

  return typed + plain;
}

/** What the device holds, in MB, with the largest labels first. */
export function deviceMemory(): { totalMB: number; byLabel: Record<string, number> } {
  const byLabel = new Map<string, number>();
  let total = 0;

  for (const { label, bytes } of live.values()) {
    total += bytes;
    byLabel.set(label, (byLabel.get(label) ?? 0) + bytes);
  }

  const sorted = [...byLabel].sort((a, b) => b[1] - a[1]).slice(0, 12);

  return { totalMB: mb(total), byLabel: Object.fromEntries(sorted.map(([label, bytes]) => [label, mb(bytes)])) };
}

/** The page's JavaScript heap, where the browser tells (Chrome and Edge only). */
function heap(): { usedMB: number; totalMB: number; limitMB: number } | null {
  const memory = (performance as { memory?: { usedJSHeapSize: number; totalJSHeapSize: number;
    jsHeapSizeLimit: number } }).memory;

  return memory
    ? { usedMB: mb(memory.usedJSHeapSize), totalMB: mb(memory.totalJSHeapSize), limitMB: mb(memory.jsHeapSizeLimit) }
    : null;
}

/** Logs one step: the device's total and largest labels, the JavaScript heap, and whatever sizes the step adds. */
export function logMemory(step: string, details: Record<string, unknown> = {}) {
  if (!enabled) return;

  const device = deviceMemory();
  const js = heap();

  // As text, not as an object: a console that folds objects loses them when it is copied.
  const text = Object.entries({ ...details, device: device.byLabel })
    .map(([key, value]) => `\n  ${key}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`)
    .join("");

  // eslint-disable-next-line no-console -- the log is the point of this module
  console.info(`[gpu-renderer] ${step}: device ${device.totalMB} MB`
    + (js ? `, JS heap ${js.usedMB}/${js.limitMB} MB` : "") + text);
}

interface BreakdownMesh {
  primitive?: string;
  attributes: Record<string, { data: ArrayLike<number> } | undefined>;
  index: ArrayLike<number> | null;
  materials: ({ name?: string; type?: string; values?: Record<string, unknown> } | null | undefined)[];
}

/**
 * Where a snapshot's triangles come from: plain meshes, wireframe materials, or lines, points and sprites, which the
 * packer turns into triangles facing the camera (a line segment becomes two). Lines and wireframes that are not
 * meant to be rendered are the usual reason a scene packs far more than it shows. Also the largest meshes, by the
 * triangles each one packs.
 */
export function sceneBreakdown(meshes: readonly BreakdownMesh[]) {
  const kinds: Record<string, { meshes: number; packedTriangles: number }> = {};
  const largest: { kind: string; material: string; packedTriangles: number; vertices: number }[] = [];

  for (const mesh of meshes) {
    const vertices = (mesh.attributes.position?.data.length ?? 0) / 3;
    const count = mesh.index?.length ?? vertices;
    const material = mesh.materials[0];
    const wireframe = !mesh.primitive && mesh.materials.some(entry => entry?.values?.wireframe === true);
    const kind = mesh.primitive ?? (wireframe ? "wireframe" : "mesh");

    // Two triangles a segment or a point; a strip has one segment fewer than vertices.
    const packed = kind === "lineSegments" ? count : kind === "line" || kind === "lineLoop" ? 2 * count
      : kind === "points" ? 2 * count : count / 3;

    kinds[kind] ??= { meshes: 0, packedTriangles: 0 };
    kinds[kind].meshes++;
    kinds[kind].packedTriangles += Math.round(packed);
    largest.push({ kind, material: material?.name || material?.type || "?", packedTriangles: Math.round(packed),
      vertices });
  }

  largest.sort((a, b) => b.packedTriangles - a.packedTriangles);

  return { kinds, largest: largest.slice(0, 10) };
}

interface UsageTexture {
  id: string;
  width: number;
  height: number;
  data: ArrayLike<number>;
}

interface UsageMaterial {
  maps?: Record<string, string>;
}

/**
 * What each texture costs and how much of it is read: the slots that use it, the channels those slots read, and the
 * bytes a texel would take if only those channels were kept (one, two or four bytes; float textures keep their four
 * floats). The total says what keeping only the channels read would save.
 */
export function textureUsage(
  textures: readonly UsageTexture[], materials: readonly (UsageMaterial | null | undefined)[]) {
  const slots = new Map<string, Set<string>>();

  for (const material of new Set(materials)) {
    for (const [slot, id] of Object.entries(material?.maps ?? {})) {
      if (typeof id !== "string") continue;

      let known = slots.get(id);

      if (!known) slots.set(id, known = new Set());

      known.add(slot);
    }
  }

  let nowBytes = 0, packedBytes = 0;

  const rows = textures.map(texture => {
    const used = [...slots.get(texture.id) ?? []];
    // A normal map's third component could be rebuilt from the other two, so it counts as two here.
    const channels = new Set(used.flatMap(slot => [...(slot.endsWith("ormalMap") ? "rg" : SLOT_CHANNELS[slot] ?? "rgba")]));
    const float = texture.data instanceof Float32Array;
    const texels = texture.width * texture.height;
    const now = texels * (float ? 16 : 4);
    // Three channels do not pack into fewer than four bytes on a word boundary; one or two do.
    const kept = float ? 16 : channels.size === 0 ? 0 : channels.size <= 2 ? channels.size : 4;
    const packed = texels * kept;

    nowBytes += now;
    packedBytes += packed;

    return {
      id: texture.id.length > 12 ? `${texture.id.slice(0, 12)}…` : texture.id,
      fullId: texture.id,
      size: `${texture.width}x${texture.height}`,
      MB: mb(now),
      slots: used.join("+") || "unused",
      channels: [...channels].sort().join("") || "-",
      packedMB: mb(packed),
    };
  });

  rows.sort((a, b) => b.MB - a.MB);

  return { totalMB: mb(nowBytes), packedMB: mb(packedBytes), textures: rows };
}

/** Bytes as MB, for the details of a step. */
export const toMB = mb;
