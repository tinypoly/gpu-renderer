import { expect, it } from "vitest";
import { buildBvh, MAX_BVH_DEPTH, NODE_WORDS, type PackedBvh } from "./bvh.js";

/** A small deterministic generator, so a failure can be reproduced from the seed alone. */
function random(seed: number) {
  let state = seed >>> 0;

  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;

    return state / 4294967296;
  };
}

function randomTriangles(count: number, seed = 7): Float32Array {
  const next = random(seed);
  const positions = new Float32Array(count * 9);

  for (let triangle = 0; triangle < count; triangle++) {
    const x = next() * 20 - 10, y = next() * 20 - 10, z = next() * 20 - 10;

    for (let corner = 0; corner < 3; corner++) {
      positions[triangle * 9 + corner * 3] = x + next() * 2 - 1;
      positions[triangle * 9 + corner * 3 + 1] = y + next() * 2 - 1;
      positions[triangle * 9 + corner * 3 + 2] = z + next() * 2 - 1;
    }
  }

  return positions;
}

/** Closest hit by trying every triangle: the answer the hierarchy has to reproduce. */
function bruteForce(positions: Float32Array, count: number, origin: number[], direction: number[]): number {
  let closest = Infinity;

  for (let triangle = 0; triangle < count; triangle++) {
    const distance = intersect(positions, triangle, origin, direction);

    if (distance < closest) closest = distance;
  }

  return closest;
}

function intersect(positions: Float32Array, triangle: number, origin: number[], direction: number[]): number {
  const o = triangle * 9;

  const e1 = [positions[o + 3] - positions[o], positions[o + 4] - positions[o + 1],
    positions[o + 5] - positions[o + 2]];

  const e2 = [positions[o + 6] - positions[o], positions[o + 7] - positions[o + 1],
    positions[o + 8] - positions[o + 2]];

  const p = cross(direction, e2);
  const determinant = dot(e1, p);

  if (Math.abs(determinant) < 1e-12) return Infinity;

  const inverse = 1 / determinant;
  const t = [origin[0] - positions[o], origin[1] - positions[o + 1], origin[2] - positions[o + 2]];
  const u = dot(t, p) * inverse;

  if (u < 0 || u > 1) return Infinity;

  const q = cross(t, e1);
  const v = dot(direction, q) * inverse;

  if (v < 0 || u + v > 1) return Infinity;

  const distance = dot(e2, q) * inverse;

  return distance > 1e-6 ? distance : Infinity;
}

const cross = (a: number[], b: number[]) =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * The traversal the shader performs, in TypeScript: the same stack, the same slab test and the same leaf encoding.
 * It is the reference the WGSL is written against, and it is what makes a packing mistake fail here instead of on a
 * GPU where nothing can be printed.
 */
function traverse(bvh: PackedBvh, positions: Float32Array, origin: number[], direction: number[]): number {
  const floats = new Float32Array(bvh.nodes);
  const ints = new Int32Array(bvh.nodes);
  const inverse = direction.map(value => 1 / value);
  const stack = [0];
  let closest = Infinity;

  while (stack.length) {
    const node = stack.pop() as number;
    const word = node * NODE_WORDS;
    let near = 0, far = closest;

    for (let axis = 0; axis < 3; axis++) {
      const a = (floats[word + axis] - origin[axis]) * inverse[axis];
      const b = (floats[word + 4 + axis] - origin[axis]) * inverse[axis];

      near = Math.max(near, Math.min(a, b));
      far = Math.min(far, Math.max(a, b));
    }

    if (near > far) continue;

    const link = ints[word + 3];

    if (link < 0) {
      const start = -1 - link;

      for (let i = start; i < start + ints[word + 7]; i++) {
        const distance = intersect(positions, bvh.order[i], origin, direction);

        if (distance < closest) closest = distance;
      }

      continue;
    }

    stack.push(link, ints[word + 7]);
  }

  return closest;
}

it("hands back an empty tree for an empty scene", () => {
  const bvh = buildBvh(new Float32Array(0), 0);

  expect(bvh.nodeCount).toBe(0);
  expect(bvh.order.length).toBe(0);
});

it("keeps every triangle exactly once", () => {
  const bvh = buildBvh(randomTriangles(500), 500);

  expect([...bvh.order].sort((a, b) => a - b)).toEqual([...Array(500).keys()]);
});

it("stays inside the stack depth the shader allocates", () => {
  for (const count of [1, 5, 64, 2000]) {
    const bvh = buildBvh(randomTriangles(count), count);

    expect(bvh.maxDepth).toBeLessThan(MAX_BVH_DEPTH);
  }
});

it("holds every triangle of a leaf inside that leaf's bounds", () => {
  const count = 800;
  const positions = randomTriangles(count);
  const bvh = buildBvh(positions, count);
  const floats = new Float32Array(bvh.nodes);
  const ints = new Int32Array(bvh.nodes);

  for (let node = 0; node < bvh.nodeCount; node++) {
    const word = node * NODE_WORDS;
    const link = ints[word + 3];

    if (link >= 0) continue;

    const start = -1 - link;

    for (let i = start; i < start + ints[word + 7]; i++)
      for (let corner = 0; corner < 3; corner++)
        for (let axis = 0; axis < 3; axis++) {
          const value = positions[bvh.order[i] * 9 + corner * 3 + axis];

          expect(value).toBeGreaterThanOrEqual(floats[word + axis]);
          expect(value).toBeLessThanOrEqual(floats[word + 4 + axis]);
        }
  }
});

it("contains each child inside its parent", () => {
  const bvh = buildBvh(randomTriangles(600), 600);
  const floats = new Float32Array(bvh.nodes);
  const ints = new Int32Array(bvh.nodes);

  for (let node = 0; node < bvh.nodeCount; node++) {
    const word = node * NODE_WORDS;

    if (ints[word + 3] < 0) continue;

    for (const child of [ints[word + 3], ints[word + 7]])
      for (let axis = 0; axis < 3; axis++) {
        expect(floats[child * NODE_WORDS + axis]).toBeGreaterThanOrEqual(floats[word + axis]);
        expect(floats[child * NODE_WORDS + 4 + axis]).toBeLessThanOrEqual(floats[word + 4 + axis]);
      }
  }
});

it("finds the same closest hit as testing every triangle", () => {
  const count = 400;
  const positions = randomTriangles(count, 11);
  const bvh = buildBvh(positions, count);
  const next = random(3);

  for (let ray = 0; ray < 500; ray++) {
    const origin = [next() * 40 - 20, next() * 40 - 20, next() * 40 - 20];
    const target = [next() * 20 - 10, next() * 20 - 10, next() * 20 - 10];
    const direction = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
    const length = Math.hypot(...direction);
    const unit = direction.map(value => value / length);

    expect(traverse(bvh, positions, origin, unit)).toBeCloseTo(bruteForce(positions, count, origin, unit), 4);
  }
});

it("keeps triangles that share a centroid in one leaf instead of looping forever", () => {
  const count = 32;
  const positions = new Float32Array(count * 9);

  for (let triangle = 0; triangle < count; triangle++)
    positions.set([0, 0, 0, 1, 0, 0, 0, 1, 0], triangle * 9);

  const bvh = buildBvh(positions, count);

  expect(bvh.nodeCount).toBe(1);
  expect(bvh.order.length).toBe(count);
});
