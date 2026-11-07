import { describe, expect, it } from "vitest";
import { buildBvh, NODE_WORDS } from "./bvh.js";
import { cornerNormal, packGeometry, packGeometrySliced, type PackedGeometry } from "./geometry.js";
import { INSTANCE_WORDS, instanceWords, inverseAffine, mergeHierarchies, transformBox, trianglesBox } from "./instancing.js";
import { NO_NORMAL, octahedralNormal, uploadScene, type SceneBuffers } from "./sceneUpload.js";
import { packTextures } from "./texturePack.js";
import type { SerializedMaterial, SerializedMesh, SerializedScene } from "./upstream.js";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function material(values: SerializedMaterial["values"] = {}): SerializedMaterial {
  return { type: "MeshStandardMaterial", name: "m", values, maps: {} };
}

/** A bumpy indexed grid of `size` by `size` quads, with normals and UVs: two triangles a quad. */
function grid(size: number, seed: number) {
  const positions: number[] = [], normals: number[] = [], uvs: number[] = [], index: number[] = [];

  for (let y = 0; y <= size; y++)
    for (let x = 0; x <= size; x++) {
      const height = Math.sin(x * 0.7 + seed) * Math.cos(y * 0.5 + seed) * 0.3;
      const normal = [Math.cos(x + seed) * 0.2, 1, Math.sin(y + seed) * 0.2];
      const length = Math.hypot(normal[0], normal[1], normal[2]);

      positions.push(x / size - 0.5, height, y / size - 0.5);
      normals.push(normal[0] / length, normal[1] / length, normal[2] / length);
      uvs.push(x / size, y / size);
    }

  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const a = y * (size + 1) + x, b = a + size + 1;

      index.push(a, b, a + 1, b, b + 1, a + 1);
    }

  return {
    attributes: {
      position: { data: new Float32Array(positions), itemSize: 3 },
      normal: { data: new Float32Array(normals), itemSize: 3 },
      uv: { data: new Float32Array(uvs), itemSize: 2 },
    },
    index: new Uint32Array(index),
  };
}

type Geometry = ReturnType<typeof grid>;

function mesh(geometry: Geometry, matrixWorld: number[], materials = [material()]): SerializedMesh {
  return {
    attributes: geometry.attributes,
    index: geometry.index,
    groups: [],
    materials,
    matrixWorld,
    castShadow: true,
    receiveShadow: true,
    renderOrder: 0,
  };
}

/** A rotation about y by `angle`, scaled by `scale` on every axis and moved by `t`. */
function transform(angle: number, scale: number, t: number[]) {
  const c = Math.cos(angle) * scale, s = Math.sin(angle) * scale;

  return [c, 0, -s, 0, 0, scale, 0, 0, s, 0, c, 0, t[0], t[1], t[2], 1];
}

const scene = (meshes: SerializedMesh[]): SerializedScene => ({ meshes, lights: [], textures: [] });

/** A copy of the same arrays, so two meshes are equal by content and not by reference. */
const copied = (geometry: Geometry): Geometry => ({
  attributes: {
    position: { data: geometry.attributes.position.data.slice(), itemSize: 3 },
    normal: { data: geometry.attributes.normal.data.slice(), itemSize: 3 },
    uv: { data: geometry.attributes.uv.data.slice(), itemSize: 2 },
  },
  index: geometry.index.slice(),
});

describe("packGeometry with repeated geometry", () => {
  const shared = grid(12, 1);

  it("keeps a repeated mesh once, as a prototype with an instance per copy", () => {
    const packed = packGeometry(scene([
      mesh(shared, transform(0, 1, [0, 0, 0])),
      mesh(copied(shared), transform(1, 2, [3, 0, 0])),
      mesh(copied(shared), transform(2, 0.5, [0, 0, 3])),
    ]));

    expect(packed.triangleCount).toBe(0);
    expect(packed.instancing?.prototypes).toEqual([{ first: 0, count: 288 }]);
    expect(packed.instancing?.instances.map(instance => instance.firstTriangle)).toEqual([0, 288, 576]);
    expect(packed.instancing?.materials.length).toBe(3 * 288);
    // The prototype's vertices are the mesh's own, in object space, shared between its triangles.
    expect(packed.vertexCount).toBe(13 * 13);
    expect([...packed.instancing?.positions.slice(0, 3) ?? []])
      .toEqual([...shared.attributes.position.data.slice(0, 3)]);
  });

  it("gives each instance its own materials", () => {
    const red = material({ color: [1, 0, 0] }), blue = material({ color: [0, 0, 1] });

    const packed = packGeometry(scene([
      mesh(shared, IDENTITY, [red]),
      mesh(copied(shared), transform(1, 1, [2, 0, 0]), [blue]),
    ]));

    const materials = packed.instancing?.materials ?? new Uint32Array(0);

    expect(materials[0]).not.toBe(materials[288]);
    expect(packed.materialProperties[materials[288]].color).toEqual([0, 0, 1]);
  });

  it("leaves alone what is read triangle by triangle, mirrored, small or unique", () => {
    const glowing = material({ emissive: [1, 1, 1], emissiveIntensity: 1 });
    const mirror = transform(0, 1, [0, 0, 0]);

    mirror[0] = -1;

    for (const meshes of [
      [mesh(shared, IDENTITY, [glowing]), mesh(copied(shared), transform(1, 1, [2, 0, 0]), [glowing])],
      [mesh(shared, IDENTITY), mesh(copied(shared), mirror)],
      [mesh(grid(4, 1), IDENTITY), mesh(grid(4, 1), transform(1, 1, [2, 0, 0]))],
      [mesh(shared, IDENTITY), mesh(grid(12, 2), transform(1, 1, [2, 0, 0]))],
    ]) {
      const packed = packGeometry(scene(meshes));

      expect(packed.instancing === null || packed.instancing.instances.length < 2).toBe(true);
    }

    const copies = scene([mesh(shared, IDENTITY), mesh(copied(shared), IDENTITY)]);

    expect(packGeometry(copies, { instancing: false }).instancing)
      .toBeNull();
  });

  it("shares a vertex between the triangles of a smooth mesh", () => {
    const packed = packGeometry(scene([mesh(grid(2, 0), IDENTITY)]));

    expect(packed.triangleCount).toBe(8);
    expect(packed.vertexCount).toBe(9);
    expect(packed.corners.length).toBe(24);
  });

  it("packs the same in slices as at once, handing the thread back on the way", async() => {
    const meshes = [
      mesh(shared, IDENTITY),
      mesh(copied(shared), transform(1, 2, [3, 0, 0])),
      mesh(grid(30, 3), IDENTITY),
    ];

    let pauses = 0;
    const sliced = await packGeometrySliced(scene(meshes), {}, async() => { pauses++; }, 0);

    expect(sliced).toEqual(packGeometry(scene(meshes)));
    expect(pauses).toBeGreaterThan(0);
  });
});

describe("octahedralNormal", () => {
  /** The decoding the kernel does, step for step. */
  function decode(code: number): number[] {
    if (code === NO_NORMAL) return [0, 0, 0];

    const half = (bits: number) => Math.max(((bits << 16) >> 16) / 32767, -1);
    const n = [half(code & 0xffff), half(code >>> 16), 0];

    n[2] = 1 - Math.abs(n[0]) - Math.abs(n[1]);

    const t = Math.max(-n[2], 0);

    n[0] += n[0] >= 0 ? -t : t;
    n[1] += n[1] >= 0 ? -t : t;

    const length = Math.hypot(n[0], n[1], n[2]);

    return n.map(value => value / length);
  }

  it("brings a unit normal back within a hundredth of a degree", () => {
    for (let i = 0; i < 2000; i++) {
      const z = 1 - 2 * ((i + 0.5) / 2000), r = Math.sqrt(1 - z * z), phi = i * 2.399963;
      const normal = [r * Math.cos(phi), r * Math.sin(phi), z];
      const back = decode(octahedralNormal(normal[0], normal[1], normal[2]));
      const cosine = normal[0] * back[0] + normal[1] * back[1] + normal[2] * back[2];

      expect(Math.acos(Math.min(1, cosine)) * 180 / Math.PI).toBeLessThan(0.01);
    }
  });

  it("keeps a zero normal apart, so shading falls back to the face", () => {
    expect(octahedralNormal(0, 0, 0)).toBe(NO_NORMAL);
    expect(decode(NO_NORMAL)).toEqual([0, 0, 0]);
  });
});

/** A device that keeps every buffer's bytes, so the test reads what the kernel would. */
function recordingDevice() {
  (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage ??= { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 };

  const buffers: { label: string; bytes: Uint8Array }[] = [];

  const device = {
    createBuffer({ label, size }: GPUBufferDescriptor) {
      const buffer = { label: label ?? "", bytes: new Uint8Array(size), destroy() {} };

      buffers.push(buffer);

      return buffer;
    },
    queue: {
      writeBuffer(buffer: { bytes: Uint8Array }, offset: number, data: ArrayBufferView) {
        buffer.bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      },
    },
  };

  return { device: device as unknown as GPUDevice, buffers };
}

/** The scene as the kernel sees it: every buffer the upload wrote, and the traversal of bvh.wgsl.ts over them. */
function deviceScene(geometry: PackedGeometry) {
  const bvh = buildBvh(geometry.positions, geometry.triangleCount);
  const repeated = geometry.instancing;
  let hierarchy: { nodes: ArrayBuffer; order: Uint32Array } = bvh;
  let words: Uint32Array = new Uint32Array(0);

  if (repeated) {
    const boxes = repeated.prototypes.map(p => trianglesBox(repeated.positions, p.first, p.count));

    const merged = mergeHierarchies(bvh, geometry.triangleCount,
      repeated.prototypes.map(p =>
        buildBvh(repeated.positions.subarray(p.first * 9, (p.first + p.count) * 9), p.count)),
      repeated.prototypes.map(p => p.count),
      repeated.instances.map(i => transformBox(i.matrix, boxes[i.prototype].low, boxes[i.prototype].high)));

    words = instanceWords(repeated.instances, merged.roots, merged.storageFirst);
    hierarchy = merged;
  }

  const textures = packTextures({ meshes: [], lights: [], textures: [] }, { words });
  const { device, buffers } = recordingDevice();
  const uploaded: SceneBuffers = uploadScene(device, geometry, hierarchy, textures);
  const bytes = (label: string) => (buffers.find(buffer => buffer.label === `scene ${label}`) as { bytes: Uint8Array }).bytes;
  const positions = new Float32Array(bytes("positions").buffer);
  const indices = new Uint32Array(bytes("indices").buffer);
  const nodeFloats = new Float32Array(bytes("bvh nodes").buffer), nodeInts = new Int32Array(bytes("bvh nodes").buffer);
  const textureWords = new Uint32Array(bytes("textures").buffer), textureFloats = new Float32Array(textureWords.buffer);
  const instanceData = textures.extraOffset;
  const NO = 0xffffffff;

  const row = (k: number, r: number) =>
    [0, 1, 2, 3].map(i => textureFloats[instanceData + k * INSTANCE_WORDS + r * 4 + i]);

  const word = (k: number, w: number) => textureWords[instanceData + k * INSTANCE_WORDS + w];

  const apply = (k: number, first: number, v: number[], w: number) =>
    [0, 1, 2].map(r => {
      const m = row(k, first + r);

      return m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * w;
    });

  const instanceOf = (t: number) =>
    t < uploaded.sceneTriangles ? NO : indices[uploaded.instanceOffset + t - uploaded.sceneTriangles];

  const stored = (t: number, k: number) => k === NO ? t : word(k, 25) + t - word(k, 26);
  const leafVertex = (at: number, c: number) => [0, 1, 2].map(i => positions[at * 9 + c * 3 + i]);

  const triangleVertex = (t: number, c: number) => {
    const k = instanceOf(t);
    const p = leafVertex(indices[uploaded.leafOffset + stored(t, k)], c);

    return k === NO ? p : apply(k, 0, p, 1);
  };

  const cornerNormalOnDevice = (t: number, c: number) => {
    const k = instanceOf(t);
    const vertex = indices[uploaded.cornerOffset + stored(t, k) * 3 + c];
    const code = indices[uploaded.normalOffset + vertex];
    const half = (bits: number) => Math.max(((bits << 16) >> 16) / 32767, -1);
    const n = [half(code & 0xffff), half(code >>> 16), 0];

    n[2] = 1 - Math.abs(n[0]) - Math.abs(n[1]);

    const s = Math.max(-n[2], 0);

    n[0] += n[0] >= 0 ? -s : s;
    n[1] += n[1] >= 0 ? -s : s;

    // The inverse transpose: the rows of the inverse, weighed by the components.
    const world = k === NO ? n : [0, 1, 2].map(i => n[0] * row(k, 3)[i] + n[1] * row(k, 4)[i] + n[2] * row(k, 5)[i]);
    const length = Math.hypot(world[0], world[1], world[2]);

    return world.map(value => value / length);
  };

  const intersect = (at: number, origin: number[], direction: number[]) => {
    const a = leafVertex(at, 0), b = leafVertex(at, 1), c = leafVertex(at, 2);
    const e1 = b.map((v, i) => v - a[i]), e2 = c.map((v, i) => v - a[i]);

    const cross = (p: number[], q: number[]) =>
      [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];

    const dot = (p: number[], q: number[]) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    const p = cross(direction, e2), determinant = dot(e1, p);

    if (Math.abs(determinant) < 1e-12) return -1;

    const inverse = 1 / determinant, offset = origin.map((v, i) => v - a[i]);
    const u = dot(offset, p) * inverse;

    if (u < 0 || u > 1) return -1;

    const q = cross(offset, e1), v = dot(direction, q) * inverse;

    if (v < 0 || u + v > 1) return -1;

    return dot(e2, q) * inverse;
  };

  const child = (node: number, side: number) => {
    const at = node * NODE_WORDS + side * 8;

    return {
      low: [nodeFloats[at], nodeFloats[at + 1], nodeFloats[at + 2]],
      high: [nodeFloats[at + 4], nodeFloats[at + 5], nodeFloats[at + 6]],
      link: nodeInts[at + 3],
      count: nodeInts[at + 7],
    };
  };

  const enterBox = (box: { low: number[]; high: number[] }, origin: number[], inverse: number[], limit: number) => {
    let enter = 0, exit = limit;

    for (let axis = 0; axis < 3; axis++) {
      const first = (box.low[axis] - origin[axis]) * inverse[axis];
      const second = (box.high[axis] - origin[axis]) * inverse[axis];

      enter = Math.max(enter, Math.min(first, second));
      exit = Math.min(exit, Math.max(first, second));
    }

    return enter <= exit ? enter : 1e30;
  };

  const rayInverse = (direction: number[]) => direction.map(v => (v >= 0 ? 1 : -1) / Math.max(Math.abs(v), 1e-20));

  /** traceScene, step for step. */
  const trace = (worldOrigin: number[], worldDirection: number[]) => {
    let distance = 1e30, triangle = NO, hitInstance = NO, instance = NO;
    let origin = worldOrigin, direction = worldDirection, inverse = rayInverse(direction);
    const stack: { entry: number | "return" | { parent: number; side: number }; enter: number }[] = [];
    let current: number | null = 0;
    let leaf: { link: number; count: number } | null = null;

    for (;;) {
      if (leaf && leaf.count > 0) {
        for (let i = 0; i < leaf.count; i++) {
          const at = -1 - leaf.link + i, t = intersect(at, origin, direction);

          if (t > 1e-4 && t < distance) {
            distance = t;
            triangle = at;
            hitInstance = instance;
          }
        }

        leaf = null;
      }

      if (leaf && leaf.count < 0) {
        instance = -1 - leaf.link;
        stack.push({ entry: "return", enter: -1 });
        origin = apply(instance, 3, worldOrigin, 1);
        direction = apply(instance, 3, worldDirection, 0);
        inverse = rayInverse(direction);
        current = word(instance, 24);
        leaf = null;
      }

      if (current === null) {
        const top = stack.pop();

        if (!top) break;
        if (top.enter > distance) continue;

        if (top.entry === "return") {
          instance = NO;
          origin = worldOrigin;
          direction = worldDirection;
          inverse = rayInverse(direction);
          continue;
        }

        if (typeof top.entry === "object") {
          leaf = child(top.entry.parent, top.entry.side);
          continue;
        }

        current = top.entry;
      }

      const left = child(current, 0), right = child(current, 1);
      const leftEnter = enterBox(left, origin, inverse, distance);
      const rightEnter = enterBox(right, origin, inverse, distance);
      const leftFirst = leftEnter <= rightEnter;
      const near = leftFirst ? left : right, far = leftFirst ? right : left;
      const farEnter = leftFirst ? rightEnter : leftEnter;
      const parent = current;

      current = null;

      if (Math.min(leftEnter, rightEnter) < 1e30) {
        if (near.link < 0) leaf = near;
        else current = near.link;
      }

      if (farEnter < 1e30)
        stack.push({ entry: far.link < 0 ? { parent, side: leftFirst ? 1 : 0 } : far.link, enter: farEnter });
    }

    if (triangle === NO) return null;

    const storedTriangle = indices[uploaded.orderOffset + triangle];

    const named = hitInstance === NO ? storedTriangle : word(hitInstance, 26) + storedTriangle - word(hitInstance, 25);

    return { distance, triangle: named };
  };

  return { trace, triangleVertex, cornerNormal: cornerNormalOnDevice, materialOf: (t: number) => indices[t] };
}

/** The closest of every triangle, flat and in world space. */
function bruteForce(geometry: PackedGeometry, origin: number[], direction: number[]) {
  let best: { distance: number; triangle: number } | null = null;
  const p = geometry.positions;

  for (let t = 0; t < geometry.triangleCount; t++) {
    const a = [p[t * 9], p[t * 9 + 1], p[t * 9 + 2]];
    const e1 = [p[t * 9 + 3] - a[0], p[t * 9 + 4] - a[1], p[t * 9 + 5] - a[2]];
    const e2 = [p[t * 9 + 6] - a[0], p[t * 9 + 7] - a[1], p[t * 9 + 8] - a[2]];

    const cross = (u: number[], v: number[]) =>
      [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];

    const dot = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const q = cross(direction, e2), determinant = dot(e1, q);

    if (Math.abs(determinant) < 1e-12) continue;

    const offset = origin.map((v, i) => v - a[i]), u = dot(offset, q) / determinant;

    if (u < 0 || u > 1) continue;

    const r = cross(offset, e1), v = dot(direction, r) / determinant;

    if (v < 0 || u + v > 1) continue;

    const distance = dot(e2, r) / determinant;

    if (distance > 1e-4 && (!best || distance < best.distance)) best = { distance, triangle: t };
  }

  return best;
}

describe("the kernel's view of instances", () => {
  // A unique mesh first and copies after: the flat packing numbers its triangles exactly as the instanced one names them.
  const unique = grid(10, 5);
  const shared = grid(14, 2);

  const meshes = [
    mesh(unique, transform(0.3, 3, [0, -1, 0])),
    mesh(shared, transform(0, 1, [0.4, 0, 0])),
    mesh(copied(shared), transform(1.1, 1.7, [-0.8, 0.3, 0.5])),
    mesh(copied(shared), transform(2.5, 0.6, [0.2, 0.6, -0.7])),
    mesh(copied(shared), transform(4, 1.2, [0, 0.9, 0])),
  ];

  const instanced = packGeometry(scene(meshes));
  const flat = packGeometry(scene(meshes), { instancing: false });
  const kernel = deviceScene(instanced);

  it("instances the copies and keeps the unique mesh among the scene's own triangles", () => {
    expect(instanced.triangleCount).toBe(200);
    expect(instanced.instancing?.instances.length).toBe(4);
  });

  it("finds the same closest hit as every triangle tested in world space", () => {
    let hits = 0;

    for (let i = 0; i < 400; i++) {
      const theta = i * 2.399963, height = 1.5 * Math.sin(i * 0.37);
      const origin = [3 * Math.cos(theta), height + 2, 3 * Math.sin(theta)];
      const target = [Math.sin(i * 1.3) * 0.6, Math.cos(i * 0.7) * 0.4, Math.sin(i * 2.1) * 0.6];
      const direction = target.map((v, k) => v - origin[k]);
      const expected = bruteForce(flat, origin, direction);
      const found = kernel.trace(origin, direction);

      expect(found === null).toBe(expected === null);

      if (!found || !expected) continue;

      hits++;
      expect(found.distance).toBeCloseTo(expected.distance, 4);
      expect(found.triangle).toBe(expected.triangle);
    }

    expect(hits).toBeGreaterThan(200);
  });

  it("reads every triangle's corners, normals and material as the flat packing has them", () => {
    for (let t = 0; t < flat.triangleCount; t += 7) {
      for (let c = 0; c < 3; c++) {
        const expected = Array.from(flat.positions.subarray(t * 9 + c * 3, t * 9 + c * 3 + 3));

        kernel.triangleVertex(t, c).forEach((value, axis) => expect(value).toBeCloseTo(expected[axis], 4));

        const normal = cornerNormal(flat, t, c);

        kernel.cornerNormal(t, c).forEach((value, axis) => expect(value).toBeCloseTo(normal[axis], 3));
      }

      expect(kernel.materialOf(t)).toBe(flat.materials[t]);
    }
  });
});

it("inverts an affine transform", () => {
  const m = transform(0.7, 2.5, [1, 2, 3]);
  const inverse = inverseAffine(m);
  const p = [0.3, -1.2, 4];
  const world = [0, 1, 2].map(r => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);

  const back = [0, 1, 2].map(r =>
    inverse[r] * world[0] + inverse[4 + r] * world[1] + inverse[8 + r] * world[2] + inverse[12 + r]);

  back.forEach((value, axis) => expect(value).toBeCloseTo(p[axis], 9));
});
