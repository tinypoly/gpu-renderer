import { expect, it } from "vitest";
import type { PackedGeometry } from "./geometry.js";
import { LIGHT_CASTS_SHADOW, LIGHT_KIND, LIGHT_WORDS, packLights } from "./lights.js";
import { NO_PLACEMENT, materialProperties } from "./materialLayout.js";
import type { SerializedLight, SerializedMaterial } from "./upstream.js";

/** Right triangles in the xy plane, one per entry, with the given legs; material i emits `emissive[i]`. */
function geometry(
  triangles: { legs: number; material: number; x?: number; mirror?: boolean }[],
  emissive: number[][]): PackedGeometry {
  const positions = new Float32Array(triangles.length * 9);
  const materials = new Uint32Array(triangles.length);

  triangles.forEach(({ legs, material, x = 0, mirror = false }, index) => {
    const side = mirror ? -legs : legs;

    positions.set([x, 0, 0, x + side, 0, 0, x, legs, 0], index * 9);
    materials[index] = material;
  });

  const properties = emissive.map(values => materialProperties({
    type: "MeshStandardMaterial",
    name: "",
    values: { emissive: values.slice(0, 3), emissiveIntensity: values[3] },
    maps: {},
  } as SerializedMaterial));

  return {
    positions,
    normals: new Float32Array(triangles.length * 9),
    uvs: new Float32Array(triangles.length * 12),
    materials,
    triangleCount: triangles.length,
    materialColors: new Float32Array(properties.flatMap(m => [...m.color, m.opacity])),
    materialEmissive: new Float32Array(properties.flatMap(m => [...m.emissive, m.emissiveIntensity])),
    materialProperties: properties,
    materialPlacements: properties.map(() => NO_PLACEMENT),
    materialCount: properties.length,
    colors: new Float32Array(0),
    varyings: new Float32Array(0),
    extraWords: new Uint32Array(0),
    cameraDependent: false,
  };
}

/** A serialized light with everything Three would carry, so a test only states what it cares about. */
function light(overrides: Partial<SerializedLight>): SerializedLight {
  return {
    type: "PointLight",
    color: [1, 1, 1],
    intensity: 1,
    position: [0, 0, 0],
    target: [0, 0, 0],
    distance: 0,
    decay: 2,
    angle: 0,
    penumbra: 0,
    groundColor: [0, 0, 0],
    width: 0,
    height: 0,
    matrixWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    castShadow: false,
    ...overrides,
  };
}

const read = (lights: ReturnType<typeof packLights>) => ({
  words: new Uint32Array(lights.entries),
  floats: new Float32Array(lights.entries),
});

const at = (index: number, word: number) => index * LIGHT_WORDS + word;

it("keeps only the triangles that emit, in triangle order", () => {
  const lights = packLights(geometry(
    [{ legs: 1, material: 0 }, { legs: 1, material: 1 }, { legs: 1, material: 0 }, { legs: 1, material: 2 }],
    [[0, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 2]]));

  const { words } = read(lights);

  expect(lights.count).toBe(2);
  expect(lights.areaCount).toBe(2);
  expect([words[at(0, 3)], words[at(1, 3)]]).toEqual([1, 3]);
});

it("weights the choice by the power a triangle radiates and closes the cdf at one", () => {
  // Mirror images either side of the scene's centre, so their distance to it cancels and only the power is left.
  // White at 1 and white at 3: the second radiates three times the first.
  const lights = packLights(geometry(
    [{ legs: 1, material: 0, x: -4 }, { legs: 1, material: 1, x: 4, mirror: true }],
    [[1, 1, 1, 1], [1, 1, 1, 3]]));

  const { floats } = read(lights);

  expect(floats[at(0, 1)]).toBeCloseTo(1 / 4);
  expect(floats[at(1, 1)]).toBe(1);
  // The density is per unit area: the probability of the triangle over its area.
  expect(floats[at(0, 2)]).toBeCloseTo(1 / 4 / 0.5);
  expect(floats[at(1, 2)]).toBeCloseTo(3 / 4 / 0.5);
});

it("prefers the light that actually reaches the scene, not the one with the larger number on it", () => {
  // A distant point light of great intensity delivers less than a near one of modest intensity, and Three states the
  // two in different units, so the raw intensity cannot be what decides.
  const lights = packLights(
    geometry([{ legs: 1, material: 0 }], [[0, 0, 0, 1]]),
    [
      light({ type: "PointLight", intensity: 100, position: [40, 0, 0], decay: 2 }),
      light({ type: "PointLight", intensity: 4, position: [1, 0, 0], decay: 2 }),
    ]);

  const { floats } = read(lights);
  const near = 1 - floats[at(0, 1)];

  expect(near).toBeGreaterThan(0.9);
});

it("skips degenerate triangles and materials with zero intensity", () => {
  const lights = packLights(geometry(
    [{ legs: 0, material: 0 }, { legs: 1, material: 1 }, { legs: 1, material: 0 }],
    [[1, 1, 1, 1], [1, 1, 1, 0]]));

  expect(lights.count).toBe(1);
  expect(new Uint32Array(lights.entries)[at(0, 3)]).toBe(2);
  expect(new Float32Array(lights.entries)[at(0, 1)]).toBe(1);
});

it("still produces a bindable buffer for a scene without lights", () => {
  const lights = packLights(geometry([{ legs: 1, material: 0 }], [[0, 0, 0, 1]]));

  expect(lights.count).toBe(0);
  expect(lights.entries.byteLength).toBe(LIGHT_WORDS * 4);
});

it("puts the emissive triangles before the analytic lights", () => {
  const lights = packLights(
    geometry([{ legs: 1, material: 0 }], [[1, 1, 1, 1]]),
    [light({ type: "DirectionalLight", position: [0, 1, 0] })]);

  const { words } = read(lights);

  expect([lights.count, lights.areaCount]).toEqual([2, 1]);
  expect(words[at(0, 0)]).toBe(LIGHT_KIND.area);
  expect(words[at(1, 0)]).toBe(LIGHT_KIND.directional);
});

it("points a directional light from its target towards its position, and folds in the intensity", () => {
  const lights = packLights(
    geometry([], []),
    [light({ type: "DirectionalLight", color: [1, 0.5, 0.25], intensity: 4, position: [0, 3, 0], target: [0, 0, 0] })]);

  const { floats } = read(lights);

  expect([...floats.subarray(at(0, 4), at(0, 7))]).toEqual([4, 2, 1]);
  expect([...floats.subarray(at(0, 12), at(0, 15))]).toEqual([0, 1, 0]);
});

it("carries a spot light's cone as the cosines the shader compares against", () => {
  const angle = Math.PI / 6;

  const lights = packLights(
    geometry([], []),
    [light({ type: "SpotLight", angle, penumbra: 0.5, position: [0, 2, 0], target: [0, 0, 0] })]);

  const { floats } = read(lights);

  expect(floats[at(0, 15)]).toBeCloseTo(Math.cos(angle));
  expect(floats[at(0, 19)]).toBeCloseTo(Math.cos(angle / 2));
});

it("gives a rectangular light half its own extent along the axes of its world matrix", () => {
  const lights = packLights(
    geometry([], []),
    [light({ type: "RectAreaLight", width: 4, height: 2, position: [0, 0, 0] })]);

  const { floats, words } = read(lights);

  expect(words[at(0, 0)]).toBe(LIGHT_KIND.rect);
  expect([...floats.subarray(at(0, 16), at(0, 19))]).toEqual([2, 0, 0]);
  expect([...floats.subarray(at(0, 20), at(0, 23))]).toEqual([0, 1, 0]);
  // The density is per unit area, over the whole rectangle.
  expect(floats[at(0, 2)]).toBeCloseTo(1 / 8);
  // Three has no shadow switch for one, and an area light that cast none would look broken once traced.
  expect(words[at(0, 23)]).toBe(LIGHT_CASTS_SHADOW);
});

it("only marks a light as casting a shadow when Three says it does", () => {
  const lights = packLights(geometry([], []), [
    light({ type: "PointLight", position: [1, 0, 0], castShadow: true }),
    light({ type: "PointLight", position: [2, 0, 0], castShadow: false }),
  ]);

  const { words } = read(lights);

  expect(words[at(0, 23)]).toBe(LIGHT_CASTS_SHADOW);
  expect(words[at(1, 23)]).toBe(0);
});

it("leaves lights with no intensity or no colour out of the buffer", () => {
  const lights = packLights(geometry([], []), [
    light({ type: "PointLight", intensity: 0 }),
    light({ type: "DirectionalLight", color: [0, 0, 0] }),
    light({ type: "PointLight", intensity: 2, position: [1, 0, 0] }),
  ]);

  expect(lights.count).toBe(1);
});

it("turns an ambient light into the radiance of an equally bright surrounding", () => {
  const { ambient } = packLights(geometry([], []), [
    light({ type: "AmbientLight", color: [1, 1, 1], intensity: Math.PI }),
  ]);

  // A surrounding of radiance L delivers pi times L, so Three's irradiance divides by pi.
  expect(ambient.sky[0]).toBeCloseTo(1);
  expect(ambient.ground[0]).toBeCloseTo(1);
});

it("splits a hemisphere light between its sky and its ground, along its own axis", () => {
  const { ambient } = packLights(geometry([], []), [
    light({
      type: "HemisphereLight",
      color: [1, 1, 1],
      groundColor: [0.5, 0.5, 0.5],
      intensity: Math.PI,
      position: [0, 4, 0],
    }),
  ]);

  expect(ambient.sky[0]).toBeCloseTo(1);
  expect(ambient.ground[0]).toBeCloseTo(0.5);
  expect(ambient.axis).toEqual([0, 1, 0]);
});

it("keeps an ambient light out of the sampled buffer, having nothing to aim a shadow ray at", () => {
  const lights = packLights(geometry([], []), [light({ type: "AmbientLight", intensity: 1 })]);

  expect(lights.count).toBe(0);
});

it("puts the lights that sit in the scene before the ones outside it, and closes their share exactly", () => {
  const panel = geometry([{ legs: 1, material: 0 }], [[1, 1, 1, 1]]);
  const sun = light({ type: "DirectionalLight", intensity: 2, position: [0, 1, 0] });
  const bulb = light({ type: "PointLight", intensity: 3, position: [0, 1, 0] });
  const packed = packLights(panel, [sun, bulb]);
  const floats = new Float32Array(packed.entries);
  const words = new Uint32Array(packed.entries);

  expect(packed.localCount).toBe(2);
  expect(words[1 * LIGHT_WORDS]).toBe(LIGHT_KIND.point);
  expect(words[2 * LIGHT_WORDS]).toBe(LIGHT_KIND.directional);
  expect(packed.localPick).toBe(floats[1 * LIGHT_WORDS + 1]);
  expect(packed.treeAt).toBeGreaterThan(0);
});
