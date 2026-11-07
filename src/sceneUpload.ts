import type { PackedGeometry } from "./geometry.js";
import { packLights, type PackedLights } from "./lights.js";
import { MATERIAL_WORDS, materialProperties, writeMaterial } from "./materialLayout.js";
import { NO_TEXTURE, packTextures, type PackedTextures, type TextureUpload } from "./texturePack.js";
import type { PackedEnvironment } from "./environment.js";

/** Bytes each buffer of a scene takes on the device. Computed without a device, so a host can check it up front. */
export interface SceneByteLength {
  positions: number;
  materials: number;
  /** The UV sets, vertex colours and shader varyings, in one buffer. */
  attributes: number;
  /**
   * One material index per triangle, the hierarchy's triangle order and its inverse, the instance of every instanced
   * triangle, the three vertices of every stored one and the vertices' normals, in one buffer.
   */
  indices: number;
  nodes: number;
  lights: number;
  textures: number;
  total: number;
  /** The largest single binding, which is what a device limit applies to. */
  largest: number;
}

export interface SceneBuffers {
  /** What the upload takes on the device, so the host can compare it with the device's limits. */
  bytes: SceneByteLength;
  positions: GPUBuffer;
  materials: GPUBuffer;
  attributes: GPUBuffer;
  indices: GPUBuffer;
  nodes: GPUBuffer;
  lights: GPUBuffer;
  textures: GPUBuffer;
  /** Where the UV sets, the vertex colours and the varyings start inside `attributes`, and the order inside `indices`. */
  uvOffset: number;
  colorOffset: number;
  varyingOffset: number;
  hasColors: boolean;
  orderOffset: number;
  /** Where each triangle's position in the leaf order starts inside `indices`. */
  leafOffset: number;
  /** Where the instance of every instanced triangle, the corners and the normals start inside `indices`. */
  instanceOffset: number;
  cornerOffset: number;
  normalOffset: number;
  /** Every triangle a hit can name: the scene's own, then each instance's. */
  triangleCount: number;
  /** The scene's own triangles, which lead the numbering. */
  sceneTriangles: number;
  /** Everything the integrator can aim a shadow ray at. */
  lightCount: number;
  /** How many of those are emissive triangles, which lead the buffer. */
  areaLightCount: number;
  /** Where the lights' cumulative probabilities and the emissive ones' triangles sit in `textures`. */
  lightCdf: number;
  lightTriangles: number;
  /** The light tree over the lights that sit in the scene: see lightTree.ts. */
  localLightCount: number;
  localLightPick: number;
  lightTree: number;
  lightLeaves: number;
  lightList: number;
  /** The windows the sky is sampled through: see lights.ts. */
  portals: number;
  portalCount: number;
  portalList: number;
  /** Radiance an ambient or hemisphere light adds, which no shadow ray can reach. */
  ambient: PackedLights["ambient"];
  /** Where the surrounding image and its distributions ended up, for the frame uniform. */
  environment: {
    texture: number;
    cdf: number;
    width: number;
    height: number;
    total: number;
    pick: number;
  };
  destroy(): void;
}

/**
 * WebGPU rejects an empty binding, and one smaller than a single element of its array, which an empty scene would
 * otherwise produce for the hierarchy's 32 byte nodes. This covers the largest element any binding holds.
 */
const MINIMUM = Math.max(64, MATERIAL_WORDS * 4);

const sized = (bytes: number) => Math.max(MINIMUM, bytes);

/** What a scene's hierarchy hands the upload: its nodes, and the stored triangle at each position of its leaf order. */
export interface UploadHierarchy {
  nodes: ArrayBuffer;
  order: Uint32Array;
}

const attributeBytes = (geometry: PackedGeometry) =>
  geometry.uvs.byteLength + (geometry.colors?.byteLength ?? 0) + (geometry.varyings?.byteLength ?? 0);

/** Triangles a hit can name, and the ones stored: the scene's own and every prototype's. */
function triangleCounts(geometry: PackedGeometry) {
  const instanced = geometry.instancing?.materials.length ?? 0;
  const stored = geometry.triangleCount + (geometry.instancing?.triangleCount ?? 0);

  return { instanced, stored, named: geometry.triangleCount + instanced };
}

export function sceneByteLength(
  geometry: PackedGeometry,
  bvh: UploadHierarchy,
  lights: PackedLights = packLights(geometry),
  textures: Pick<PackedTextures, "data"> | Pick<TextureUpload, "byteLength"> = { data: new Uint32Array(4) }): SceneByteLength {
  const { instanced, stored, named } = triangleCounts(geometry);

  const lengths = {
    positions: sized(stored * 36),
    attributes: sized(attributeBytes(geometry)),
    // Materials of every named triangle, the order both ways, each instanced one's instance, corners and normals.
    indices: sized((named + 2 * stored + instanced + 3 * stored + geometry.vertexCount) * 4),
    materials: sized(Math.max(1, geometry.materialCount) * MATERIAL_WORDS * 4),
    nodes: sized(bvh.nodes.byteLength),
    lights: sized(lights.entries.byteLength),
    textures: sized("data" in textures ? textures.data.byteLength : textures.byteLength),
  };

  const values = Object.values(lengths);

  return {
    ...lengths,
    total: values.reduce((sum, value) => sum + value, 0),
    largest: Math.max(...values),
  };
}

/** A unit normal in two signed 16 bit halves of one word, folded onto an octahedron; zero is kept apart as NO_NORMAL. */
export const NO_NORMAL = 0x80008000;

export function octahedralNormal(x: number, y: number, z: number): number {
  const length = Math.abs(x) + Math.abs(y) + Math.abs(z);

  if (!(length > 0) || !Number.isFinite(length)) return NO_NORMAL;

  let u = x / length, v = y / length;

  if (z < 0) {
    const fu = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1);

    v = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1);
    u = fu;
  }

  const qu = Math.round(Math.min(1, Math.max(-1, u)) * 32767);
  const qv = Math.round(Math.min(1, Math.max(-1, v)) * 32767);

  return ((qu & 0xffff) | ((qv & 0xffff) << 16)) >>> 0;
}

/** Every vertex normal, folded into one word each. */
function encodedNormals(geometry: PackedGeometry): Uint32Array {
  const words = new Uint32Array(geometry.vertexCount);
  const n = geometry.normals;

  for (let vertex = 0; vertex < geometry.vertexCount; vertex++)
    words[vertex] = octahedralNormal(n[vertex * 3], n[vertex * 3 + 1], n[vertex * 3 + 2]);

  return words;
}

/** The instance every instanced triangle belongs to, in the order they are numbered. */
function instanceOfTriangles(geometry: PackedGeometry): Uint32Array {
  const instancing = geometry.instancing;
  const words = new Uint32Array(instancing?.materials.length ?? 0);

  instancing?.instances.forEach((instance, k) => {
    const from = instance.firstTriangle - geometry.triangleCount;

    words.fill(k, from, from + instancing.prototypes[instance.prototype].count);
  });

  return words;
}

/** Materials as the shader reads them: one `Material` struct each, laid out by `materialLayout.ts`. */
export function packMaterials(geometry: PackedGeometry): ArrayBuffer {
  const count = Math.max(1, geometry.materialCount);
  const packed = new ArrayBuffer(count * MATERIAL_WORDS * 4);

  // A scene whose meshes carry no material still needs one entry for the shader to index.
  const all = geometry.materialCount ? geometry.materialProperties : [materialProperties(undefined)];

  all.forEach((properties, index) => writeMaterial(packed, index, properties, geometry.materialPlacements?.[index]));

  return packed;
}

/** The power a surrounding image radiates over the whole sphere, which is what weighs it against the scene's lights. */
export function environmentPower(environment: PackedEnvironment | null): number {
  return environment ? environment.total * 2 * Math.PI ** 2 / (environment.width * environment.height) : 0;
}

/**
 * The positions in the hierarchy's leaf order, which is the order its traversal reads them in, so a leaf is one run of
 * the buffer; and where each stored triangle went, for everything that reads a triangle by its own number. A stored
 * triangle past the scene's own is a prototype's, in object space.
 */
export function leafOrder(
  geometry: Pick<PackedGeometry, "positions" | "triangleCount" | "instancing">,
  order: Uint32Array): { positions: Float32Array; inverse: Uint32Array } {
  const ordered = new Float32Array(order.length * 9);
  const inverse = new Uint32Array(order.length);
  const scene = geometry.triangleCount;
  const prototypes = geometry.instancing?.positions ?? new Float32Array(0);

  for (let at = 0; at < order.length; at++) {
    const triangle = order[at];
    const source = triangle < scene ? geometry.positions : prototypes;
    const from = (triangle < scene ? triangle : triangle - scene) * 9;

    ordered.set(source.subarray(from, from + 9), at * 9);
    inverse[triangle] = at;
  }

  return { positions: ordered, inverse };
}

export function uploadScene(
  device: GPUDevice,
  geometry: PackedGeometry,
  bvh: UploadHierarchy,
  textures: PackedTextures | TextureUpload = packTextures({ meshes: [], lights: [], textures: [] }),
  lights: PackedLights = packLights(geometry),
  environment: PackedEnvironment | null = null,
  environmentCdf = textures.extraOffset): SceneBuffers {
  const lengths = sceneByteLength(geometry, bvh, lights, textures);

  const create = (size: number, label: string): GPUBuffer =>
    device.createBuffer({ label: `scene ${label}`, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const write = (data: ArrayBufferView, size: number, label: string): GPUBuffer => {
    const buffer = create(size, label);

    device.queue.writeBuffer(buffer, 0, data);

    return buffer;
  };

  /** Arrays of the same kind, one after the other, so they cost one binding between them. */
  const join = (parts: ArrayBufferView[], size: number, label: string): GPUBuffer => {
    const buffer = create(size, label);
    let offset = 0;

    for (const part of parts) {
      if (part.byteLength) device.queue.writeBuffer(buffer, offset, part);

      offset += part.byteLength;
    }

    return buffer;
  };

  const colors = geometry.colors ?? new Float32Array(0);
  const varyings = geometry.varyings ?? new Float32Array(0);
  const leaves = leafOrder(geometry, bvh.order);
  const { instanced, stored, named } = triangleCounts(geometry);
  const instanceOf = instanceOfTriangles(geometry);
  const orderOffset = named;
  const leafOffset = orderOffset + stored;
  const instanceOffset = leafOffset + stored;
  const cornerOffset = instanceOffset + instanced;
  const normalOffset = cornerOffset + 3 * stored;

  const texturesBuffer = create(lengths.textures, "textures");

  if ("write" in textures) textures.write(device, texturesBuffer);
  else device.queue.writeBuffer(texturesBuffer, 0, textures.data);

  const buffers = {
    positions: write(leaves.positions, lengths.positions, "positions"),
    attributes: join([geometry.uvs, colors, varyings], lengths.attributes, "attributes"),
    indices: join([
      geometry.materials,
      geometry.instancing?.materials ?? new Uint32Array(0),
      bvh.order,
      leaves.inverse,
      instanceOf,
      geometry.corners,
      encodedNormals(geometry),
    ], lengths.indices, "indices"),
    materials: write(new Uint8Array(packMaterials(geometry)), lengths.materials, "materials"),
    nodes: write(new Uint8Array(bvh.nodes), lengths.nodes, "bvh nodes"),
    lights: write(new Uint8Array(lights.entries), lengths.lights, "lights"),
    textures: texturesBuffer,
  };

  return {
    bytes: lengths,
    ...buffers,
    uvOffset: 0,
    colorOffset: geometry.uvs.length,
    varyingOffset: geometry.uvs.length + colors.length,
    hasColors: colors.length > 0,
    orderOffset,
    leafOffset,
    instanceOffset,
    cornerOffset,
    normalOffset,
    triangleCount: named,
    sceneTriangles: geometry.triangleCount,
    lightCount: lights.count,
    areaLightCount: lights.areaCount,
    lightCdf: lights.cdfAt,
    lightTriangles: lights.trianglesAt,
    localLightCount: lights.localCount,
    localLightPick: lights.localPick,
    lightTree: lights.treeAt,
    lightLeaves: lights.leavesAt,
    lightList: lights.listAt,
    portals: lights.portalTreeAt,
    portalCount: lights.portalCount,
    portalList: lights.portalListAt,
    ambient: lights.ambient,
    environment: {
      texture: environment ? textures.indexOf.get(environment.texture.id) ?? NO_TEXTURE : NO_TEXTURE,
      cdf: environmentCdf,
      width: environment?.width ?? 0,
      height: environment?.height ?? 0,
      total: environment?.total ?? 0,
      pick: lights.environmentPick,
    },
    destroy() {
      for (const buffer of Object.values(buffers)) buffer.destroy();
    },
  };
}
