import { expect, it } from "vitest";
import {
  buildLightTree,
  TREE_ENTRY_WORDS,
  TREE_LEAF,
  TREE_NODE_WORDS,
  TREE_ROOT_PARENT,
  unionCone,
  type TreeLight,
} from "./lightTree.js";

const point = (x: number, power = 1): TreeLight => ({
  low: [x, 0, 0], high: [x, 0, 0], power, axis: [0, 0, 1], thetaO: Math.PI, thetaE: Math.PI / 2, density: 1,
});

it("bounds two cones by one that holds both", () => {
  const up = { axis: [0, 1, 0], thetaO: 0, thetaE: Math.PI / 2 };
  const side = { axis: [1, 0, 0], thetaO: 0, thetaE: Math.PI / 2 };
  const both = unionCone(up, side);

  expect(both.thetaO).toBeCloseTo(Math.PI / 4, 10);
  expect(both.axis[0]).toBeCloseTo(Math.SQRT1_2, 6);
  expect(both.axis[1]).toBeCloseTo(Math.SQRT1_2, 6);
  // Opposite axes leave nothing but the whole sphere.
  expect(unionCone(up, { ...up, axis: [0, -1, 0] }).thetaO).toBe(Math.PI);
});

it("puts every light in a leaf of its own, whose parents lead back to the root", () => {
  const lights = [point(0, 1), point(1, 2), point(5, 3), point(9, 4), point(10, 5)];
  const { words, leafOf, listAt } = buildLightTree(lights);
  const floats = new Float32Array(words.buffer);

  expect(listAt / TREE_NODE_WORDS).toBe(2 * lights.length - 1);
  // The root holds all the power.
  expect(floats[3]).toBe(15);
  expect(words[14]).toBe(TREE_ROOT_PARENT);

  lights.forEach((_, light) => {
    const leaf = leafOf[light * 2];
    const slot = listAt + leafOf[light * 2 + 1] * TREE_ENTRY_WORDS;

    // Spread this far apart, no two lights share a leaf.
    expect(words[leaf * TREE_NODE_WORDS + 13]).toBe(1);
    expect(words[slot]).toBe(light);
    expect(floats[slot + 1]).toBe(1);

    let node = leaf, depth = 0;

    while (words[node * TREE_NODE_WORDS + 14] !== TREE_ROOT_PARENT && depth++ < 10) {
      const parent = words[node * TREE_NODE_WORDS + 14];

      expect([words[parent * TREE_NODE_WORDS + 12], words[parent * TREE_NODE_WORDS + 13]]).toContain(node);
      node = parent;
    }

    expect(node).toBe(0);
  });
});

it("lets the triangles of one small lamp share a leaf, picked within it by power", () => {
  // Eight tiny lights a hair apart at the origin, and one ten units off.
  const lamp = Array.from({ length: 8 }, (_, k) => point(k * 0.001, k + 1));
  const { words, leafOf, listAt } = buildLightTree([...lamp, point(10)]);
  const floats = new Float32Array(words.buffer);
  const leaf = leafOf[0];

  expect(lamp.every((_, k) => leafOf[k * 2] === leaf)).toBe(true);
  expect(leafOf[8 * 2]).not.toBe(leaf);
  expect((words[leaf * TREE_NODE_WORDS + 12] & TREE_LEAF) >>> 0).toBe(TREE_LEAF);
  expect(words[leaf * TREE_NODE_WORDS + 13]).toBe(8);

  const start = listAt + (words[leaf * TREE_NODE_WORDS + 12] & ~TREE_LEAF) * TREE_ENTRY_WORDS;
  const chances = lamp.map((_, k) => floats[start + k * TREE_ENTRY_WORDS + 1]);

  // Shares of the power, 1 to 8 over 36, the last closing the range exactly.
  chances.forEach((chance, k) => expect(chance).toBeCloseTo((k + 1) / 36, 6));
  expect(floats[start + 7 * TREE_ENTRY_WORDS + 3]).toBe(1);
});
