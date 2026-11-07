/**
 * Geometry a scene repeats, kept once. A mesh whose triangles appear again elsewhere, the same positions, normals, UVs
 * and colours under another transform, becomes an instance of a prototype: the prototype's triangles sit in object
 * space with a hierarchy of their own, and each instance keeps only its transform and the material of each triangle.
 * A tree over the instances' boxes leads into those hierarchies, so the kernel traverses one tree, entering an instance
 * by taking the ray into its space.
 *
 * Only meshes nothing else reads triangle by triangle become instances: lights, windows, volumes and grass are
 * found on the CPU among the scene's own triangles, and points, lines, sprites and vertex shaders depend on the camera.
 */

import { NODE_WORDS, type PackedBvh } from "./bvh.js";

/** A mesh with fewer triangles than this is cheaper to copy than to enter. */
export const INSTANCE_MIN_TRIANGLES = 256;

/**
 * Words per instance in the texture buffer: the object to world transform and its inverse, three rows of four floats
 * each, then the root of the prototype's hierarchy, where its triangles start among the stored ones, and the number
 * the first of this instance's triangles goes by.
 */
export const INSTANCE_WORDS = 28;

/** A prototype: a run of the stored triangles, in object space. */
export interface Prototype {
  /** Its first triangle among the prototype triangles. */
  first: number;
  count: number;
}

export interface Instance {
  prototype: number;
  /** Object to world, column major, as Three's `matrixWorld`. */
  matrix: readonly number[];
  /** The number this instance's first triangle goes by, after every triangle of the scene's own. */
  firstTriangle: number;
}

/** Repeated geometry, as `packGeometry` leaves it. */
export interface PackedInstancing {
  /** Nine coordinates per prototype triangle, in object space. */
  positions: Float32Array;
  triangleCount: number;
  prototypes: Prototype[];
  instances: Instance[];
  /** One material index per instanced triangle, every instance's in turn. */
  materials: Uint32Array;
  /** The box around every instance, in world space. */
  bounds: { low: number[]; high: number[] };
}

/** A seed pair, mixed by every word: two independent 32 bit hashes, so a match is all but certain. */
interface Hash {
  a: number;
  b: number;
}

const scratch = new Float32Array(1);
const scratchWords = new Uint32Array(scratch.buffer);

function mix(hash: Hash, word: number) {
  hash.a = Math.imul(hash.a ^ word, 0x01000193);
  hash.b = Math.imul(hash.b + word, 0x5bd1e995);
  hash.b ^= hash.b >>> 15;
}

function mixArray(hash: Hash, data: ArrayLike<number> | null | undefined) {
  if (!data) {
    mix(hash, 0x9e3779b9);

    return;
  }

  mix(hash, data.length);

  if (data instanceof Float32Array || data instanceof Uint32Array || data instanceof Int32Array) {
    const words = new Uint32Array(data.buffer, data.byteOffset, data.length);

    for (let i = 0; i < words.length; i++) mix(hash, words[i]);

    return;
  }

  for (let i = 0; i < data.length; i++) {
    scratch[0] = data[i];
    mix(hash, scratchWords[0]);
  }
}

/** What makes two meshes' geometry the same, as far as the triangles they pack: one value per array, then the rest. */
interface GeometrySource {
  attributes: Record<string, { data: ArrayLike<number>; itemSize: number } | undefined>;
  index: ArrayLike<number> | null;
  groups: readonly { start: number; count: number; materialIndex: number }[];
}

// A mesh handed over twice shares its arrays: hashing each array once is enough.
const arrayHashes = new WeakMap<object, string>();

function arrayKey(data: ArrayLike<number> | null | undefined): string {
  if (!data) return "-";

  const known = arrayHashes.get(data as object);

  if (known) return known;

  const hash = { a: 0x811c9dc5, b: 0x2545f491 };

  mixArray(hash, data);

  const key = `${(hash.a >>> 0).toString(36)}.${(hash.b >>> 0).toString(36)}`;

  arrayHashes.set(data as object, key);

  return key;
}

/** A key equal for two meshes exactly when their triangles would pack the same in object space. */
export function geometryKey(mesh: GeometrySource, visibleGroups: string): string {
  const { position, normal, uv, uv1, color } = mesh.attributes;
  const groups = mesh.groups.map(group => `${group.start}:${group.count}:${group.materialIndex}`).join(",");

  return [
    arrayKey(position?.data), position?.itemSize,
    arrayKey(normal?.data), normal?.itemSize,
    arrayKey(uv?.data), uv?.itemSize,
    arrayKey(uv1?.data), uv1?.itemSize,
    arrayKey(color?.data), color?.itemSize,
    arrayKey(mesh.index),
    groups, visibleGroups,
  ].join("|");
}

/** Whether a transform keeps a prototype's triangles as they are: affine, invertible and not mirrored. */
export function instanceable(matrix: readonly number[]): boolean {
  if (matrix[3] !== 0 || matrix[7] !== 0 || matrix[11] !== 0 || matrix[15] !== 1) return false;

  const determinant = matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
    - matrix[4] * (matrix[1] * matrix[10] - matrix[2] * matrix[9])
    + matrix[8] * (matrix[1] * matrix[6] - matrix[2] * matrix[5]);

  return determinant > 1e-12 && Number.isFinite(determinant);
}

/** The inverse of an affine transform, column major. */
export function inverseAffine(m: readonly number[]): number[] {
  const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10];
  const c0 = e * i - f * h, c1 = f * g - d * i, c2 = d * h - e * g;
  const s = 1 / (a * c0 + b * c1 + c * c2);

  // Columns of the inverse 3x3: the adjugate over the determinant.
  const r = [
    c0 * s, (c * h - b * i) * s, (b * f - c * e) * s,
    c1 * s, (a * i - c * g) * s, (c * d - a * f) * s,
    c2 * s, (b * g - a * h) * s, (a * e - b * d) * s,
  ];

  const tx = m[12], ty = m[13], tz = m[14];

  return [
    r[0], r[1], r[2], 0,
    r[3], r[4], r[5], 0,
    r[6], r[7], r[8], 0,
    -(r[0] * tx + r[3] * ty + r[6] * tz), -(r[1] * tx + r[4] * ty + r[7] * tz), -(r[2] * tx + r[5] * ty + r[8] * tz), 1,
  ];
}

/** A box transformed by a matrix: the box around its eight corners. */
export function transformBox(m: readonly number[], low: readonly number[], high: readonly number[]) {
  const outLow = [Infinity, Infinity, Infinity], outHigh = [-Infinity, -Infinity, -Infinity];

  for (let corner = 0; corner < 8; corner++) {
    const x = corner & 1 ? high[0] : low[0], y = corner & 2 ? high[1] : low[1], z = corner & 4 ? high[2] : low[2];

    for (let axis = 0; axis < 3; axis++) {
      const value = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];

      outLow[axis] = Math.min(outLow[axis], value);
      outHigh[axis] = Math.max(outHigh[axis], value);
    }
  }

  return { low: outLow, high: outHigh };
}

/** The box around a run of triangles, nine coordinates each. */
export function trianglesBox(positions: Float32Array, first: number, count: number) {
  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];

  for (let at = first * 9; at < (first + count) * 9; at += 3) {
    for (let axis = 0; axis < 3; axis++) {
      low[axis] = Math.min(low[axis], positions[at + axis]);
      high[axis] = Math.max(high[axis], positions[at + axis]);
    }
  }

  return { low, high };
}

/** The words of every instance, as the kernel reads them: see `INSTANCE_WORDS`. */
export function instanceWords(
  instances: readonly Instance[], roots: readonly number[], storageFirst: readonly number[]): Uint32Array {
  const words = new Uint32Array(instances.length * INSTANCE_WORDS);
  const floats = new Float32Array(words.buffer);

  instances.forEach((instance, k) => {
    const at = k * INSTANCE_WORDS;
    const m = instance.matrix;
    const inverse = inverseAffine(m);

    // Rows of the 3x4 transforms: a row times (x, y, z, 1) is one coordinate.
    for (let row = 0; row < 3; row++) {
      floats.set([m[row], m[4 + row], m[8 + row], m[12 + row]], at + row * 4);
      floats.set([inverse[row], inverse[4 + row], inverse[8 + row], inverse[12 + row]], at + 12 + row * 4);
    }

    words[at + 24] = roots[instance.prototype];
    words[at + 25] = storageFirst[instance.prototype];
    words[at + 26] = instance.firstTriangle;
  });

  return words;
}

/** One node of the tree over instances, before it is laid out: a box, and two children or one instance. */
interface TopNode {
  low: number[];
  high: number[];
  left: number;
  right: number;
  instance: number;
}

/**
 * The tree over the instances' boxes, split at the median along the widest spread of their centres, one instance a
 * leaf. Instances are few, so this is cheap, and the median split never fails the way a heuristic one can on boxes
 * that share a centre: the depth is the logarithm of the count, which keeps the kernel's stack bounded.
 */
function buildTopTree(boxes: readonly { low: number[]; high: number[] }[]): TopNode[] {
  const nodes: TopNode[] = [];
  const centre = boxes.map(box => box.low.map((value, axis) => (value + box.high[axis]) / 2));

  const build = (items: number[]): number => {
    const index = nodes.length;
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];

    for (const item of items)
      for (let axis = 0; axis < 3; axis++) {
        low[axis] = Math.min(low[axis], boxes[item].low[axis]);
        high[axis] = Math.max(high[axis], boxes[item].high[axis]);
      }

    nodes.push({ low, high, left: -1, right: -1, instance: items.length === 1 ? items[0] : -1 });

    if (items.length === 1) return index;

    const spread = [0, 1, 2].map(axis => {
      const values = items.map(item => centre[item][axis]);

      return Math.max(...values) - Math.min(...values);
    });

    const axis = spread.indexOf(Math.max(...spread));
    const sorted = [...items].sort((p, q) => centre[p][axis] - centre[q][axis] || p - q);
    const half = sorted.length >> 1;

    nodes[index].left = build(sorted.slice(0, half));
    nodes[index].right = build(sorted.slice(half));

    return index;
  };

  if (boxes.length) build(boxes.map((_, k) => k));

  return nodes;
}

/** The scene's triangles, then every prototype's, in one hierarchy: what the kernel traverses. */
export interface MergedHierarchy {
  nodes: ArrayBuffer;
  nodeCount: number;
  /** The stored triangle at each position of the leaf order: the scene's own first, then each prototype's. */
  order: Uint32Array;
  /** Where each prototype's hierarchy starts among the nodes. */
  roots: number[];
  /** Where each prototype's triangles start among the stored ones. */
  storageFirst: number[];
}

const FAR = 3.4028234663852886e38;

/**
 * Joins the scene's own hierarchy, one per prototype and the tree over the instances into one node array. Node zero
 * holds the scene's tree on one side and the instances' on the other. A leaf of the instances' tree has a count of -1
 * and links to its instance; a prototype's leaves point into its run of the leaf order, after the scene's own.
 */
export function mergeHierarchies(scene: PackedBvh, sceneTriangles: number, prototypes: readonly PackedBvh[],
  prototypeCounts: readonly number[], instanceBoxes: readonly { low: number[]; high: number[] }[]): MergedHierarchy {
  const top = buildTopTree(instanceBoxes);
  const topNodes = top.filter(node => node.instance < 0);
  // A single instance is a tree of one leaf, which still takes a node to hold it.
  const topCount = Math.max(1, topNodes.length);
  const sceneBase = 1;
  const topBase = sceneBase + scene.nodeCount;
  const roots: number[] = [];
  const storageFirst: number[] = [];
  let nodeCount = topBase + topCount;
  let stored = sceneTriangles;

  prototypes.forEach((bvh, p) => {
    roots.push(nodeCount);
    storageFirst.push(stored);
    nodeCount += bvh.nodeCount;
    stored += prototypeCounts[p];
  });

  const nodes = new ArrayBuffer(nodeCount * NODE_WORDS * 4);
  const floats = new Float32Array(nodes);
  const ints = new Int32Array(nodes);
  const order = new Uint32Array(stored);

  order.set(scene.order.subarray(0, sceneTriangles), 0);

  // A tree copied in place: its nodes move by `base`, its leaves by `leafBase` along the leaf order.
  const copyTree = (bvh: PackedBvh, base: number, leafBase: number) => {
    const source = new Int32Array(bvh.nodes);

    floats.set(new Float32Array(bvh.nodes), base * NODE_WORDS);

    for (let node = 0; node < bvh.nodeCount; node++)
      for (const child of [0, 8]) {
        const word = (base + node) * NODE_WORDS + child;
        const link = source[node * NODE_WORDS + child + 3];
        const count = source[node * NODE_WORDS + child + 7];

        if (link >= 0) ints[word + 3] = link + base;
        else if (count > 0) ints[word + 3] = -1 - (-1 - link + leafBase);
      }
  };

  const writeChild = (word: number, low: readonly number[], high: readonly number[], link: number, count: number) => {
    for (let axis = 0; axis < 3; axis++) {
      floats[word + axis] = low[axis];
      floats[word + 4 + axis] = high[axis];
    }

    ints[word + 3] = link;
    ints[word + 7] = count;
  };

  const empty = (word: number) => writeChild(word, [FAR, FAR, FAR], [FAR, FAR, FAR], -1, 0);

  // The box a whole tree takes, from the two children of its first node; an empty child adds nothing.
  const treeBox = (bvh: PackedBvh) => {
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
    const source = new Float32Array(bvh.nodes), links = new Int32Array(bvh.nodes);

    for (const child of [0, 8]) {
      if (links[child + 3] < 0 && links[child + 7] === 0) continue;

      for (let axis = 0; axis < 3; axis++) {
        low[axis] = Math.min(low[axis], source[child + axis]);
        high[axis] = Math.max(high[axis], source[child + 4 + axis]);
      }
    }

    return { low, high };
  };

  if (sceneTriangles > 0 && scene.nodeCount > 0) {
    const box = treeBox(scene);

    copyTree(scene, sceneBase, 0);
    writeChild(0, box.low, box.high, sceneBase, 0);
  } else {
    empty(0);
  }

  // The instances' tree, numbered as it was built without its leaves, which live in their parents.
  const index = new Map<number, number>();

  top.forEach((node, at) => { if (node.instance < 0) index.set(at, topBase + index.size); });

  const topChild = (word: number, at: number) => {
    const node = top[at];

    if (node.instance >= 0) writeChild(word, node.low, node.high, -1 - node.instance, -1);
    else writeChild(word, node.low, node.high, index.get(at) as number, 0);
  };

  if (top.length === 1) {
    topChild(topBase * NODE_WORDS, 0);
    empty(topBase * NODE_WORDS + 8);
  } else {
    top.forEach((node, at) => {
      if (node.instance >= 0) return;

      const word = (index.get(at) as number) * NODE_WORDS;

      topChild(word, node.left);
      topChild(word + 8, node.right);
    });
  }

  if (top.length) writeChild(8, top[0].low, top[0].high, topBase, 0);
  else empty(8);

  prototypes.forEach((bvh, p) => {
    copyTree(bvh, roots[p], storageFirst[p]);

    for (let i = 0; i < prototypeCounts[p]; i++) order[storageFirst[p] + i] = storageFirst[p] + bvh.order[i];
  });

  return { nodes, nodeCount, order, roots, storageFirst };
}
