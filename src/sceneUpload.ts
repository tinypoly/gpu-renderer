import type { PackedBvh } from "./bvh.js";
import type { PackedGeometry } from "./geometry.js";
import { packLights, type PackedLights } from "./lights.js";
import { MATERIAL_WORDS, materialProperties, writeMaterial } from "./materialLayout.js";
import { NO_TEXTURE, packTextures, type PackedTextures } from "./texturePack.js";
import type { PackedEnvironment } from "./environment.js";

/** Bytes each buffer of a scene takes on the device. Computed without a device, so a host can check it up front. */
export interface SceneByteLength {
  positions: number;
  materials: number;
  /** Vertex normals, the UV sets, vertex colours and shader varyings, in one buffer. */
  attributes: number;
  /** One material index per triangle and then the hierarchy's triangle order, in one buffer. */
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
  triangleCount: number;
  /** Everything the integrator can aim a shadow ray at. */
  lightCount: number;
  /** How many of those are emissive triangles, which lead the buffer. */
  areaLightCount: number;
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

const attributeBytes = (geometry: PackedGeometry) =>
  geometry.normals.byteLength + geometry.uvs.byteLength + (geometry.colors?.byteLength ?? 0)
  + (geometry.varyings?.byteLength ?? 0);

export function sceneByteLength(
  geometry: PackedGeometry,
  bvh: PackedBvh,
  lights: PackedLights = packLights(geometry),
  textures: Pick<PackedTextures, "data"> = { data: new Uint32Array(4) }): SceneByteLength {
  const lengths = {
    positions: sized(geometry.positions.byteLength),
    attributes: sized(attributeBytes(geometry)),
    indices: sized(geometry.materials.byteLength + bvh.order.byteLength),
    materials: sized(Math.max(1, geometry.materialCount) * MATERIAL_WORDS * 4),
    nodes: sized(bvh.nodes.byteLength),
    lights: sized(lights.entries.byteLength),
    textures: sized(textures.data.byteLength),
  };

  const values = Object.values(lengths);

  return {
    ...lengths,
    total: values.reduce((sum, value) => sum + value, 0),
    largest: Math.max(...values),
  };
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

export function uploadScene(
  device: GPUDevice,
  geometry: PackedGeometry,
  bvh: PackedBvh,
  textures: PackedTextures = packTextures({ meshes: [], lights: [], textures: [] }),
  lights: PackedLights = packLights(geometry),
  environment: PackedEnvironment | null = null,
  environmentCdf = textures.extraOffset): SceneBuffers {
  const lengths = sceneByteLength(geometry, bvh, lights, textures);

  const create = (size: number): GPUBuffer =>
    device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const write = (data: ArrayBufferView, size: number): GPUBuffer => {
    const buffer = create(size);

    device.queue.writeBuffer(buffer, 0, data);

    return buffer;
  };

  /** Arrays of the same kind, one after the other, so they cost one binding between them. */
  const join = (parts: ArrayBufferView[], size: number): GPUBuffer => {
    const buffer = create(size);
    let offset = 0;

    for (const part of parts) {
      if (part.byteLength) device.queue.writeBuffer(buffer, offset, part);

      offset += part.byteLength;
    }

    return buffer;
  };

  const colors = geometry.colors ?? new Float32Array(0);
  const varyings = geometry.varyings ?? new Float32Array(0);

  const buffers = {
    positions: write(geometry.positions, lengths.positions),
    attributes: join([geometry.normals, geometry.uvs, colors, varyings], lengths.attributes),
    indices: join([geometry.materials, bvh.order], lengths.indices),
    materials: write(new Uint8Array(packMaterials(geometry)), lengths.materials),
    nodes: write(new Uint8Array(bvh.nodes), lengths.nodes),
    lights: write(new Uint8Array(lights.entries), lengths.lights),
    textures: write(textures.data, lengths.textures),
  };

  return {
    bytes: lengths,
    ...buffers,
    uvOffset: geometry.normals.length,
    colorOffset: geometry.normals.length + geometry.uvs.length,
    varyingOffset: geometry.normals.length + geometry.uvs.length + colors.length,
    hasColors: colors.length > 0,
    orderOffset: geometry.materials.length,
    triangleCount: geometry.triangleCount,
    lightCount: lights.count,
    areaLightCount: lights.areaCount,
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
