import { describe, expect, it } from "vitest";
import type { PackedGeometry } from "./geometry.js";
import {
  GRASS_BLADE_WORDS,
  GRASS_ENTRY_WORDS,
  GRASS_RECORD_WORDS,
  GRASS_TRIANGLE_WORDS,
  grassField,
  grassPatch,
  packGrass,
  type GrassField,
} from "./grass.js";
import { NODE_WORDS } from "./bvh.js";

const field = (input: object = {}) => grassField({ density: 400, ...input }) as GrassField;

const asFloats = (words: Uint32Array) => new Float32Array(words.buffer, words.byteOffset, words.length);

/** A flat square of ground, two triangles facing up, and a wall beside it facing sideways. */
function ground(grass: GrassField | null): PackedGeometry {
  const positions = new Float32Array([
    0, 0, 0, 0, 0, 2, 2, 0, 2,
    0, 0, 0, 2, 0, 2, 2, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 1, 1,
  ]);

  const normals = new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]);

  return {
    positions,
    // One vertex per corner, as a mesh without shared vertices packs.
    corners: Uint32Array.from({ length: 9 }, (_, k) => k),
    normals,
    vertexCount: 9,
    triangleCount: 3,
    materials: new Uint32Array([0, 0, 0]),
    materialProperties: [{ grass } as PackedGeometry["materialProperties"][number]],
  } as PackedGeometry;
}

describe("grassField", () => {
  it("takes every default from true and none from nothing", () => {
    expect(grassField(true)?.height).toBe(0.09);
    expect(grassField(undefined)).toBeNull();
    expect(grassField(false)).toBeNull();
    expect(grassField({ height: 0 })).toBeNull();
  });

  it("clamps what it reads", () => {
    const resolved = field({ dryness: 4, translucency: -1, maxSlope: 120 });

    expect(resolved.dryness).toBe(1);
    expect(resolved.translucency).toBe(0);
    expect(resolved.maxSlope).toBe(89);
  });
});

describe("grassPatch", () => {
  it("lays every variant's triangles in its own hierarchy, with blades the triangles point at", () => {
    const patch = grassPatch(field());
    const triangles = patch.triangles.length / GRASS_TRIANGLE_WORDS;
    const blades = patch.blades.length / GRASS_BLADE_WORDS;

    expect(triangles).toBeGreaterThan(0);
    expect(patch.nodes).toHaveLength(4);

    for (let t = 0; t < triangles; t++) expect(patch.triangles[t * GRASS_TRIANGLE_WORDS + 15]).toBeLessThan(blades);

    // Every leaf of every variant points inside the triangle list.
    for (const nodes of patch.nodes) {
      const ints = new Int32Array(nodes.buffer);

      for (let at = 0; at < ints.length; at += NODE_WORDS) {
        for (const link of [3, 11]) {
          if (ints[at + link] < 0 && ints[at + link + 4] > 0) {
            expect(-1 - ints[at + link] + ints[at + link + 4]).toBeLessThanOrEqual(triangles);
          }
        }
      }
    }
  });

  it("stands every blade on the ground, below the patch's top", () => {
    const patch = grassPatch(field());
    const floats = asFloats(patch.triangles);

    for (let at = 0; at < floats.length; at += GRASS_TRIANGLE_WORDS) {
      for (let corner = 0; corner < 3; corner++) {
        expect(floats[at + corner * 3 + 1]).toBeGreaterThanOrEqual(0);
        expect(floats[at + corner * 3 + 1]).toBeLessThanOrEqual(patch.top);
      }
    }
  });

  it("repeats the same blades along the border, so any variant meets any other without a seam", () => {
    const patch = grassPatch(field());
    const blades = asFloats(patch.blades);
    const counts = new Map<string, number>();

    for (let at = 0; at < blades.length; at += GRASS_BLADE_WORDS) {
      const x = blades[at], z = blades[at + 1];

      // Only the blade itself, not the copies shifted a tile over, and only next to the border.
      if (x < 0 || x >= patch.tile || z < 0 || z >= patch.tile) continue;
      if (Math.min(x, patch.tile - x, z, patch.tile - z) > 0.01) continue;

      const key = `${x},${z}`;

      counts.set(key, (counts.get(key) ?? 0) + 1);
    }

    expect(counts.size).toBeGreaterThan(0);

    for (const count of counts.values()) expect(count).toBe(patch.nodes.length);
  });

  it("reuses a patch for the same lawn", () => {
    expect(grassPatch(field())).toBe(grassPatch(field({ dryness: 0.5 })));
  });
});

describe("packGrass", () => {
  it("is nothing without a lawn", () => {
    expect(packGrass(ground(null), 100)).toBeNull();
  });

  it("grows on the ground and not on the wall, with offsets into where the words will sit", () => {
    const base = 1000;
    const packed = packGrass(ground(field()), base);

    expect(packed).not.toBeNull();
    expect(packed!.count).toBe(2);

    const words = packed!.words;
    const floats = asFloats(words);
    const entry = packed!.entries - base;

    for (let k = 0; k < packed!.count; k++) {
      const at = entry + k * GRASS_ENTRY_WORDS;

      // Every entry points at the lawn's record, which leads the words.
      expect(words[at]).toBe(base);

      // Anticlockwise on the ground.
      const [ax, az, bx, bz, cx, cz] = Array.from(floats.subarray(at + 1, at + 7));

      expect((bx - ax) * (cz - az) - (bz - az) * (cx - ax)).toBeGreaterThan(0);
      // A flat plane at the height of zero.
      for (const value of floats.subarray(at + 7, at + 10)) expect(Math.abs(value)).toBe(0);
    }

    // The record's patch pointers fall inside the words, before the entries.
    expect(words[3]).toBe(base + GRASS_RECORD_WORDS);
    expect(words[5]).toBeLessThan(packed!.entries);
    expect(packed!.shells).toBeGreaterThan(packed!.entries);
  });

  it("leaves out slopes steeper than the lawn allows", () => {
    expect(packGrass(ground(field({ maxSlope: 0 })), 0)?.count).toBe(2);

    const steep = ground(field({ maxSlope: 10 }));

    // Tilt the ground by about 27 degrees.
    steep.positions.set([0, 0, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 2, 1, 0], 0);

    expect(packGrass(steep, 0)).toBeNull();
  });
});

/**
 * The kernel's search, step for step: the ray clipped to a shell, the tiles walked front to back, each tile's patch
 * searched through its hierarchy. Against it, every triangle of the same variant tried by brute force.
 */
describe("the search through a lawn", () => {
  const lawn = field({ density: 900, patchiness: 0 });
  const base = 0;
  const packed = packGrass(ground(lawn), base)!;
  const words = packed.words;
  const floats = asFloats(words);
  const record = words[packed.entries];
  const tile = floats[record];
  const top = floats[record + 1];
  const trianglesAt = words[record + 3];

  type Vec = [number, number, number];

  const sub = (a: Vec, b: Vec): Vec => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

  const cross = (a: Vec, b: Vec): Vec =>
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

  const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const corner = (at: number): Vec => [floats[at], floats[at + 1], floats[at + 2]];

  function triangle(at: number, origin: Vec, direction: Vec): number {
    const a = corner(at);
    const e1 = sub(corner(at + 3), a), e2 = sub(corner(at + 6), a);
    const p = cross(direction, e2);
    const det = dot(e1, p);

    if (Math.abs(det) < 1e-14) return Infinity;

    const offset = sub(origin, a);
    const u = dot(offset, p) / det;
    const q = cross(offset, e1);
    const v = dot(direction, q) / det;
    const t = dot(e2, q) / det;

    return u < 0 || v < 0 || u + v > 1 ? Infinity : t;
  }

  /** The patch's hierarchy, as tracePatch walks it. */
  function patchByTree(root: number, origin: Vec, direction: Vec, near: number, far: number): number {
    let closest = far;
    const pending = [0];
    const inverse = direction.map(d => (d >= 0 ? 1 : -1) / Math.max(Math.abs(d), 1e-20));

    while (pending.length) {
      const at = root + (pending.pop() as number) * 16;

      for (const side of [0, 8]) {
        const low = [floats[at + side], floats[at + side + 1], floats[at + side + 2]];
        const high = [floats[at + side + 4], floats[at + side + 5], floats[at + side + 6]];
        let enter = 0, exit = closest;

        for (let k = 0; k < 3; k++) {
          const a = (low[k] - origin[k]) * inverse[k], b = (high[k] - origin[k]) * inverse[k];

          enter = Math.max(enter, Math.min(a, b));
          exit = Math.min(exit, Math.max(a, b));
        }

        if (enter > exit) continue;

        const link = new Int32Array(words.buffer)[at + side + 3];
        const count = new Int32Array(words.buffer)[at + side + 7];

        if (link >= 0) {
          pending.push(link);
          continue;
        }

        for (let i = 0; i < count; i++) {
          const t = triangle(trianglesAt + (-1 - link + i) * GRASS_TRIANGLE_WORDS, origin, direction);

          if (t > near && t < closest) closest = t;
        }
      }
    }

    return closest;
  }

  /** The triangles a variant's leaves cover, from its first to its last. */
  function variantRange(root: number): [number, number] {
    let first = Infinity, last = 0;
    const ints = new Int32Array(words.buffer);
    const pending = [0];

    while (pending.length) {
      const at = root + (pending.pop() as number) * 16;

      for (const side of [0, 8]) {
        const link = ints[at + side + 3], count = ints[at + side + 7];

        if (link >= 0) pending.push(link);
        else if (count > 0) {
          first = Math.min(first, -1 - link);
          last = Math.max(last, -1 - link + count);
        }
      }
    }

    return [first, last];
  }

  function patchByBruteForce(root: number, origin: Vec, direction: Vec, near: number, far: number): number {
    const [first, last] = variantRange(root);
    let closest = far;

    for (let i = first; i < last; i++) {
      const t = triangle(trianglesAt + i * GRASS_TRIANGLE_WORDS, origin, direction);

      if (t > near && t < closest) closest = t;
    }

    return closest;
  }

  /** grassInShell over both triangles of the ground, with the search in each tile given. */
  function search(origin: Vec, direction: Vec, inTile: typeof patchByTree): number {
    let best = Infinity;

    for (let k = 0; k < packed.count; k++) {
      const entry = packed.entries + k * GRASS_ENTRY_WORDS;
      const [ax, az, bx, bz, cx, cz, p0, sx, sz] = Array.from(floats.subarray(entry + 1, entry + 10));
      const h0 = origin[1] - p0 - sx * origin[0] - sz * origin[2];
      const dh = direction[1] - sx * direction[0] - sz * direction[2];
      let range: [number, number] = [1e-4, best];

      const clip = (offset: number, rate: number) => {
        if (Math.abs(rate) < 1e-20) {
          if (offset < 0) range = [1, 0];
        } else if (rate > 0) range[0] = Math.max(range[0], -offset / rate);
        else range[1] = Math.min(range[1], -offset / rate);
      };

      const edge = (px: number, pz: number, qx: number, qz: number) => {
        const ex = qx - px, ez = qz - pz;

        clip(ex * (origin[2] - pz) - ez * (origin[0] - px), ex * direction[2] - ez * direction[0]);
      };

      clip(h0, dh);
      clip(top - h0, -dh);
      edge(ax, az, bx, bz);
      edge(bx, bz, cx, cz);
      edge(cx, cz, ax, az);

      if (range[0] >= range[1]) continue;

      let cellX = Math.floor((origin[0] + direction[0] * range[0]) / tile);
      let cellZ = Math.floor((origin[2] + direction[2] * range[0]) / tile);

      const next = (cell: number, o: number, d: number) =>
        Math.abs(d) < 1e-20 ? Infinity : ((cell + (d > 0 ? 1 : 0)) * tile - o) / d;

      let crossX = next(cellX, origin[0], direction[0]);
      let crossZ = next(cellZ, origin[2], direction[2]);
      let start = range[0];

      for (let i = 0; i < 48; i++) {
        const end = Math.min(crossX, crossZ, range[1]);
        // The test takes the first variant everywhere: which one a tile picks is the kernel's hash.
        const root = words[record + 5];

        const t = end > start
          ? inTile(root, [origin[0] - cellX * tile, h0, origin[2] - cellZ * tile], [direction[0], dh, direction[2]],
            start, end)
          : end;

        if (t < end) {
          best = t;
          break;
        }

        if (end >= range[1]) break;

        if (crossX < crossZ) {
          cellX += Math.sign(direction[0]);
          start = crossX;
          crossX += Math.abs(tile / direction[0]);
        } else {
          cellZ += Math.sign(direction[2]);
          start = crossZ;
          crossZ += Math.abs(tile / direction[2]);
        }
      }
    }

    return best;
  }

  const normalize = (v: Vec): Vec => {
    const length = Math.hypot(...v);

    return [v[0] / length, v[1] / length, v[2] / length];
  };

  it("finds the same blade through the hierarchy as by brute force, from above and at a grazing angle", () => {
    let hits = 0;
    let rays = 0;

    for (let i = 0; i < 400; i++) {
      const x = 0.1 + 1.8 * ((i * 0.618034) % 1);
      const z = 0.1 + 1.8 * ((i * 0.414214) % 1);
      const grazing = i % 2 === 1;
      const direction = grazing ? normalize([0.9, -0.08, 0.3]) : normalize([0.05, -1, 0.02]);
      const origin: Vec = grazing ? [x - 0.9, 0.08 + 0.05, z - 0.3] : [x, 1, z];
      const tree = search(origin, direction, patchByTree);
      const brute = search(origin, direction, patchByBruteForce);

      expect(tree).toBeCloseTo(brute, 9);
      rays++;

      if (tree < Infinity) {
        hits++;

        const point = origin.map((value, k) => value + direction[k] * tree);

        // On the lawn, above the ground and under its top.
        expect(point[1]).toBeGreaterThanOrEqual(-1e-6);
        expect(point[1]).toBeLessThanOrEqual(top + 1e-6);
      }
    }

    // A lawn this dense hides most of the ground from a grazing view and a good part of it from above.
    expect(hits / rays).toBeGreaterThan(0.3);
  });
});
