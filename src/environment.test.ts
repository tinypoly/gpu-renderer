import { expect, it } from "vitest";
import { environmentDistribution, loadEnvironment } from "./environment.js";

/** A grid of linear rgb, three floats per texel, from a function of its position. */
function grid(width: number, height: number, value: (x: number, y: number) => number) {
  const texels = new Float32Array(width * height * 3);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * 3;
      const level = value(x, y);

      texels[at] = level;
      texels[at + 1] = level;
      texels[at + 2] = level;
    }
  }

  return texels;
}

/** The three parts the shader reads: the row distribution, each row's columns, and the weight of every texel. */
function parts(built: NonNullable<ReturnType<typeof environmentDistribution>>, width: number, height: number) {
  return {
    rows: built.distribution.subarray(0, height),
    columns: built.distribution.subarray(height, height + width * height),
    weights: built.distribution.subarray(height + width * height),
    total: built.total,
  };
}

it("closes both distributions at one, so a unit sample always lands somewhere", () => {
  const width = 8, height = 4;
  const built = environmentDistribution(grid(width, height, () => 1), width, height);

  expect(built).not.toBeNull();

  const { rows, columns } = parts(built!, width, height);

  expect(rows[height - 1]).toBeCloseTo(1);
  for (let y = 0; y < height; y++) expect(columns[y * width + width - 1]).toBeCloseTo(1);
});

it("weights a row by the sine of its colatitude, so the poles are picked less often", () => {
  const width = 4, height = 4;
  const built = environmentDistribution(grid(width, height, () => 1), width, height);
  const { rows } = parts(built!, width, height);
  const first = rows[0];
  const second = rows[1] - rows[0];

  // The second row of four is nearer the equator than the first, so it covers more of the sphere.
  expect(second).toBeGreaterThan(first);
});

it("spreads a lone bright texel over its neighbours, which the bilinear read can reach", () => {
  const width = 8, height = 4;
  const built = environmentDistribution(grid(width, height, (x, y) => (x === 4 && y === 2 ? 100 : 0)), width, height);
  const { weights } = parts(built!, width, height);

  expect(weights[2 * width + 4]).toBeGreaterThan(50);
  // Its neighbour carries the same weight, because the filter can put the same light there.
  expect(weights[2 * width + 5]).toBeCloseTo(weights[2 * width + 4]);
  // Two texels away it drops to the floor that keeps every direction reachable.
  expect(weights[2 * width + 6]).toBeLessThan(weights[2 * width + 4] / 10);
  expect(weights[2 * width + 6]).toBeGreaterThan(0);
});

it("wraps the spread around the seam, longitude having no edge", () => {
  const width = 8, height = 4;
  const built = environmentDistribution(grid(width, height, (x, y) => (x === 0 && y === 2 ? 100 : 0)), width, height);
  const { weights } = parts(built!, width, height);

  expect(weights[2 * width + width - 1]).toBeCloseTo(weights[2 * width]);
});

it("gives every texel some chance, so no direction arrives through a bounce alone", () => {
  const width = 4, height = 2;
  const built = environmentDistribution(grid(width, height, (x, y) => (x === 0 && y === 0 ? 10 : 0)), width, height);
  const { weights } = parts(built!, width, height);

  for (const weight of weights) expect(weight).toBeGreaterThan(0);
});

it("refuses a sky with nothing in it, which nothing could be aimed at", () => {
  expect(environmentDistribution(grid(4, 2, () => 0), 4, 2)).toBeNull();
});

it("leaves a gradient environment alone: there is no image to read", async() => {
  expect(await loadEnvironment({ kind: "gradient", topColor: "#ffffff" })).toBeNull();
});

it("returns nothing for an image it cannot read, rather than failing the frame", async() => {
  expect(await loadEnvironment({ kind: "hdri", url: "" })).toBeNull();
});
