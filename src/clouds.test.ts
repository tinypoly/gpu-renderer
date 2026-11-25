import { expect, it } from "vitest";
import { cloudLayerOf, VOLUMETRIC_CLOUDS_UNIFORM } from "./clouds.js";

const sky = { [VOLUMETRIC_CLOUDS_UNIFORM]: 0, uCloudCoverage: 0.3, uCloudDensity: 1, uCloudScale: 1, uCloudSeed: 4,
  uCloudAltitude: 1.4, uCloudThickness: 1.5, uCloudDetail: 0.6, uHaze: 0.5 };

it("reads the layer a sky declares, in the units the kernel walks it in", () => {
  const layer = cloudLayerOf(sky);

  expect(layer).toMatchObject({ coverage: 0.3, density: 1, scale: 1, seed: 4, altitude: 1.4, thickness: 1.5,
    detail: 0.6 });
  // Haze grows with the aerosols of the sky's air.
  expect(layer?.haze).toBeCloseTo(0.021, 6);
});

it("leaves a sky that does not ask for it painting its own clouds", () => {
  const { [VOLUMETRIC_CLOUDS_UNIFORM]: _declared, ...plain } = sky;

  expect(cloudLayerOf(plain)).toBeNull();
  expect(cloudLayerOf(undefined)).toBeNull();
});

it("has no layer for a clear sky or one it could not walk", () => {
  expect(cloudLayerOf({ ...sky, uCloudCoverage: 0 })).toBeNull();
  expect(cloudLayerOf({ ...sky, uCloudThickness: 0 })).toBeNull();
  // A size of zero would divide the pattern by zero.
  expect(cloudLayerOf({ ...sky, uCloudScale: 0 })).toBeNull();
});

it("carries the wind offset of the pattern, and none when the sky gives it", () => {
  expect(cloudLayerOf({ ...sky, uCloudDrift: [12, -3] })?.drift).toEqual([12, -3]);
  expect(cloudLayerOf(sky)?.drift).toEqual([0, 0]);
});

it("carries the age of the clouds, and none when the sky gives it", () => {
  expect(cloudLayerOf({ ...sky, uCloudTime: -2.5 })?.time).toBe(-2.5);
  expect(cloudLayerOf(sky)?.time).toBe(0);
});
