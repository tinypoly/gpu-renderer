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

/** One child of a node: its box, and its link and count, as `struct BvhNode` holds them. */
function child(bvh: PackedBvh, node: number, side: 0 | 1) {
  const floats = new Float32Array(bvh.nodes);
  const ints = new Int32Array(bvh.nodes);
  const word = node * NODE_WORDS + side * 8;

  return {
    low: [floats[word], floats[word + 1], floats[word + 2]],
    high: [floats[word + 4], floats[word + 5], floats[word + 6]],
    link: ints[word + 3],
    count: ints[word + 7],
  };
}

function enter(box: { low: number[]; high: number[] }, origin: number[], inverse: number[], limit: number): number {
  let near = 0, far = limit;

  for (let axis = 0; axis < 3; axis++) {
    const a = (box.low[axis] - origin[axis]) * inverse[axis];
    const b = (box.high[axis] - origin[axis]) * inverse[axis];

    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
  }

  return near <= far ? near : Infinity;
}

/**
 * The traversal the shader performs, in TypeScript: the same stack, the same order, the same slab test and the same
 * leaf encoding, a leaf that has to wait included. It is the reference the WGSL is written against, and it is what
 * makes a packing mistake fail here instead of on a GPU where nothing can be printed. Also returns how many triangle
 * tests it took, which says the search was cut short.
 */
function traverse(bvh: PackedBvh, positions: Float32Array, origin: number[], direction: number[]) {
  const inverse = direction.map(value => 1 / value);
  const stack: { entry: { node: number } | { parent: number; side: 0 | 1 }; enter: number }[] = [];
  let closest = Infinity, tests = 0;
  let current: number | null = 0;

  const leaf = (link: number, count: number) => {
    for (let i = -1 - link; i < -1 - link + count; i++) {
      tests++;
      closest = Math.min(closest, intersect(positions, bvh.order[i], origin, direction));
    }
  };

  for (;;) {
    if (current === null) {
      const top = stack.pop();

      if (!top) break;
      if (top.enter > closest) continue;

      if ("parent" in top.entry) {
        const waiting = child(bvh, top.entry.parent, top.entry.side);

        leaf(waiting.link, waiting.count);
        continue;
      }

      current = top.entry.node;
    }

    const left = child(bvh, current, 0), right = child(bvh, current, 1);
    const leftEnter = enter(left, origin, inverse, closest), rightEnter = enter(right, origin, inverse, closest);
    const leftFirst = leftEnter <= rightEnter;

    const [near, nearEnter, far, farEnter] = leftFirst
      ? [left, leftEnter, right, rightEnter] : [right, rightEnter, left, leftEnter];

    const parent: number = current;

    current = null;

    if (nearEnter < Infinity) {
      if (near.link < 0) leaf(near.link, near.count);
      else current = near.link;
    }

    if (farEnter < Infinity && farEnter <= closest) {
      if (current === null) {
        if (far.link < 0) leaf(far.link, far.count);
        else current = far.link;
      } else {
        stack.push({ entry: far.link < 0 ? { parent, side: leftFirst ? 1 : 0 } : { node: far.link }, enter: farEnter });
      }
    }
  }

  return { closest, tests };
}

/** Every leaf, as the run of `order` it covers and the box its parent keeps for it. */
function leaves(bvh: PackedBvh) {
  const found: { start: number; count: number; low: number[]; high: number[] }[] = [];

  for (let node = 0; node < bvh.nodeCount; node++) {
    for (const side of [0, 1] as const) {
      const slot = child(bvh, node, side);

      if (slot.link < 0 && slot.count > 0)
        found.push({ start: -1 - slot.link, count: slot.count, low: slot.low, high: slot.high });
    }
  }

  return found;
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

  for (const { start, count: size, low, high } of leaves(bvh))
    for (let i = start; i < start + size; i++)
      for (let corner = 0; corner < 3; corner++)
        for (let axis = 0; axis < 3; axis++) {
          const value = positions[bvh.order[i] * 9 + corner * 3 + axis];

          expect(value).toBeGreaterThanOrEqual(low[axis]);
          expect(value).toBeLessThanOrEqual(high[axis]);
        }
});

it("covers every triangle with exactly one leaf", () => {
  const bvh = buildBvh(randomTriangles(700), 700);
  const covered = new Uint8Array(700);

  for (const { start, count } of leaves(bvh)) for (let i = start; i < start + count; i++) covered[i]++;

  expect([...covered].every(times => times === 1)).toBe(true);
});

it("contains each child's children inside the box its parent keeps for it", () => {
  const bvh = buildBvh(randomTriangles(600), 600);

  for (let node = 0; node < bvh.nodeCount; node++) {
    for (const side of [0, 1] as const) {
      const slot = child(bvh, node, side);

      if (slot.link < 0) continue;

      for (const inner of [child(bvh, slot.link, 0), child(bvh, slot.link, 1)])
        for (let axis = 0; axis < 3; axis++) {
          expect(inner.low[axis]).toBeGreaterThanOrEqual(slot.low[axis]);
          expect(inner.high[axis]).toBeLessThanOrEqual(slot.high[axis]);
        }
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

    const { closest, tests } = traverse(bvh, positions, origin, unit);

    expect(closest).toBeCloseTo(bruteForce(positions, count, origin, unit), 4);
    expect(tests).toBeLessThan(count);
  }
});

it("keeps triangles that share a centroid in one leaf instead of looping forever", () => {
  const count = 32;
  const positions = new Float32Array(count * 9);

  for (let triangle = 0; triangle < count; triangle++)
    positions.set([0, 0, 0, 1, 0, 0, 0, 1, 0], triangle * 9);

  const bvh = buildBvh(positions, count);

  // One leaf, held by a node beside an empty child.
  expect(bvh.nodeCount).toBe(1);
  expect(bvh.order.length).toBe(count);
  expect(leaves(bvh)).toHaveLength(1);
  expect(child(bvh, 0, 1).count).toBe(0);
});

it("finds a single leaf's closest hit, held beside the empty child", () => {
  const positions = randomTriangles(3, 5);
  const bvh = buildBvh(positions, 3);
  const unit = [0.3, -0.2, 0.93].map(value => value / Math.hypot(0.3, -0.2, 0.93));

  expect(traverse(bvh, positions, [0, 0, -20], unit).closest)
    .toBeCloseTo(bruteForce(positions, 3, [0, 0, -20], unit), 4);
});
