import { expect, it } from "vitest";
import { lensEffects, pyramidLevels } from "./lensPass.js";
import { materialProperties } from "./materialLayout.js";
import {
  DEFAULT_GPU_RENDER_SETTINGS,
  effectiveExposure,
  mergeGpuSettings,
  physicalExposure,
  whiteBalanceMatrix,
} from "./settings.js";
import { subsurfaceMedium, volumeMedium, type SubsurfaceMaterial } from "./subsurface.js";
import { atmosphereExtinction, gradientAverage } from "./atmosphere.js";
import { containingVolume, NO_MATERIAL } from "./volumeContainment.js";
import { DEFAULT_RENDER_SETTINGS } from "./upstream.js";

const apply = (m: number[], v: number[]) =>
  [0, 1, 2].map(row => m[row * 3] * v[0] + m[row * 3 + 1] * v[1] + m[row * 3 + 2] * v[2]);

it("defaults to what a photograph has: long paths, a lens and subsurface scattering", () => {
  const settings = DEFAULT_GPU_RENDER_SETTINGS;

  expect(settings.globalIllumination.bounces).toBe(12);
  expect(settings.globalIllumination.russianRoulette).toBe(true);
  expect(settings.bloom.enabled).toBe(true);
  expect(settings.lens.chromaticAberration).toBeGreaterThan(0);
  expect(settings.camera.whiteBalance).toBe(6500);
  expect(settings.subsurface.enabled).toBe(true);
  // Everything the CPU renderer shares stays its own.
  expect(settings.samples).toBe(DEFAULT_RENDER_SETTINGS.samples);
  expect(settings.toneMapping).toBe("agx");
  expect(settings.glare.enabled).toBe(true);
  expect(settings.atmosphere.enabled).toBe(true);
  expect(settings.volumes.enabled).toBe(true);
});

it("merges the new groups field by field and leaves the base alone", () => {
  const merged = mergeGpuSettings(DEFAULT_GPU_RENDER_SETTINGS,
    { bloom: { intensity: 0.2 }, lens: { distortion: 0.1 } });

  expect(merged.bloom).toEqual({ ...DEFAULT_GPU_RENDER_SETTINGS.bloom, intensity: 0.2 });
  expect(merged.lens.distortion).toBe(0.1);
  expect(merged.lens.chromaticAberration).toBe(DEFAULT_GPU_RENDER_SETTINGS.lens.chromaticAberration);
  expect(DEFAULT_GPU_RENDER_SETTINGS.bloom.intensity).toBe(0.04);
});

it("exposes as a camera does: one over 1.2 at EV 0, and the sunny sixteen rule", () => {
  expect(physicalExposure(1, 1, 100)).toBeCloseTo(1 / 1.2, 10);
  // f/16, 1/100 s at ISO 100 is EV100 = log2(256 * 100) = 14.64: bright sunlight.
  expect(Math.log2(1 / (1.2 * physicalExposure(16, 1 / 100, 100)))).toBeCloseTo(Math.log2(25600), 6);
  // Doubling the ISO doubles the exposure.
  expect(physicalExposure(2.8, 1 / 125, 200) / physicalExposure(2.8, 1 / 125, 100)).toBeCloseTo(2, 10);
});

it("applies physical exposure only when asked, with the exposure as compensation", () => {
  const manual = mergeGpuSettings(DEFAULT_GPU_RENDER_SETTINGS, { exposure: 2 });

  const physical = mergeGpuSettings(manual, {
    camera: { physicalExposure: true, iso: 100, shutterSpeed: 1 },
    depthOfField: { aperture: 1 },
  });

  expect(effectiveExposure(manual)).toBe(2);
  expect(effectiveExposure(physical)).toBeCloseTo(2 / 1.2, 10);
});

it("leaves the image alone at 6500 K", () => {
  const identity = whiteBalanceMatrix(6500);

  [1, 0, 0, 0, 1, 0, 0, 0, 1].forEach((value, k) => expect(identity[k]).toBeCloseTo(value, 5));
});

it("renders light of the balanced temperature as the neutral white", () => {
  // The colour of a 3200 K black body in linear sRGB, from its chromaticity, and the neutral white's.
  const white = (x: number, y: number) => apply([
    3.2404542, -1.5371385, -0.4985314, -0.969266, 1.8760108, 0.041556, 0.0556434, -0.2040259, 1.0572252,
  ], [x / y, 1, (1 - x - y) / y]);

  const tungsten = white(0.4234, 0.399);
  const neutral = white(0.3135, 0.3236);
  const balanced = apply(whiteBalanceMatrix(3200), tungsten);

  balanced.forEach((channel, k) => expect(channel / neutral[k]).toBeCloseTo(1, 1));
});

it("cools the image when balanced for warm light, and warms it when balanced for cold", () => {
  const warm = apply(whiteBalanceMatrix(3200), [1, 1, 1]);
  const cold = apply(whiteBalanceMatrix(10000), [1, 1, 1]);

  expect(warm[2]).toBeGreaterThan(warm[0]);
  expect(cold[0]).toBeGreaterThan(cold[2]);
});

it("skips the lens pass when nothing asks for it", () => {
  const none = mergeGpuSettings(DEFAULT_GPU_RENDER_SETTINGS, {
    bloom: { enabled: false },
    glare: { enabled: false },
    lens: { chromaticAberration: 0 },
    camera: { whiteBalance: 6500 },
  });

  expect(lensEffects(none)).toBeNull();
  expect(lensEffects(DEFAULT_GPU_RENDER_SETTINGS)?.bloom).toBe(0.04);
  expect(lensEffects(mergeGpuSettings(none, { camera: { whiteBalance: 4000 } }))?.whiteBalance).toHaveLength(9);
});

it("halves the image into a pyramid that ends before its levels get too small", () => {
  const levels = pyramidLevels(1920, 1080);

  expect(levels[0]).toEqual({ width: 960, height: 540, offset: 0 });
  expect(levels[1].offset).toBe(960 * 540);
  expect(levels.length).toBe(8);
  expect(Math.min(levels[levels.length - 1].width, levels[levels.length - 1].height)).toBeGreaterThanOrEqual(4);
  // A tiny image still gets one level to filter.
  expect(pyramidLevels(4, 4)).toHaveLength(1);
});

it("reads a subsurface medium with Blender's defaults, in world units", () => {
  const defaults = subsurfaceMedium(true);

  expect(defaults?.weight).toBe(1);
  expect(defaults?.anisotropy).toBe(0);
  [0.05, 0.01, 0.005].forEach((path, k) => expect(defaults?.radius[k]).toBeCloseTo(path, 12));
  expect(subsurfaceMedium({ radius: 2, scale: 0.1, weight: 0.5, anisotropy: 3 }))
    .toEqual({ weight: 0.5, radius: [0.2, 0.2, 0.2], anisotropy: 0.95 });
  expect(subsurfaceMedium({ weight: 0 })).toBeNull();
  expect(subsurfaceMedium(false)).toBeNull();
  expect(subsurfaceMedium(undefined)).toBeNull();
});

it("packs the medium of a physical material, and none for a material Three does not shade physically", () => {
  const skin: SubsurfaceMaterial = {
    type: "MeshPhysicalMaterial", name: "skin", values: {}, maps: {},
    subsurface: { radius: [1, 0.5, 0.25], scale: 0.1 },
  };

  const properties = materialProperties(skin);

  expect(properties.subsurface).toBe(1);
  expect(properties.subsurfaceRadius).toEqual([0.1, 0.05, 0.025]);
  expect(materialProperties({ ...skin, type: "MeshLambertMaterial" }).subsurface).toBe(0);
  const plain: SubsurfaceMaterial = { ...skin, subsurface: undefined };

  expect(materialProperties(plain).subsurface).toBe(0);
});

it("counts the aperture's rays as a lens does: as many as an even count of blades, twice an odd one", () => {
  expect(lensEffects(mergeGpuSettings(DEFAULT_GPU_RENDER_SETTINGS, { glare: { blades: 6 } }))?.rays).toBe(6);
  expect(lensEffects(mergeGpuSettings(DEFAULT_GPU_RENDER_SETTINGS, { glare: { blades: 7 } }))?.rays).toBe(14);
});

it("fades green over the visibility to 2 percent, blue faster in clean air and alike in haze", () => {
  const clean = atmosphereExtinction({ enabled: true, visibility: 1000, haze: 0, color: "" });
  const hazy = atmosphereExtinction({ enabled: true, visibility: 1000, haze: 1, color: "" });

  expect(Math.exp(-clean[1] * 1000)).toBeCloseTo(0.02, 10);
  expect(clean[2] / clean[0]).toBeGreaterThan(5);
  expect(hazy[2] / hazy[0]).toBeCloseTo(((680 / 440) ** 1.3), 6);
  expect(atmosphereExtinction({ enabled: false, visibility: 1000, haze: 0, color: "" })).toEqual([0, 0, 0]);
});

it("averages a gradient over the sphere", () => {
  // An exponent of one is a linear blend in height, which a sphere covers evenly: the plain mean of the two colours.
  expect(gradientAverage([1, 1, 1], [0, 0, 0], 1)).toEqual([0.5, 0.5, 0.5]);
});

it("reads a volume medium, and none without density", () => {
  expect(volumeMedium({ density: 0.5, color: "#ffffff", anisotropy: 0.7 }))
    .toEqual({ density: 0.5, albedo: [1, 1, 1], anisotropy: 0.7 });
  expect(volumeMedium(true)?.density).toBe(1);
  expect(volumeMedium({ density: 0 })).toBeNull();
});

it("finds the volume a point stands inside by the parity of its crossings", () => {
  // A unit cube from -1 to 1, its twelve triangles all belonging to material 3.
  const corner = (x: number, y: number, z: number) => [x, y, z];

  const faces = [
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],
    [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]], [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],
  ];

  const positions = new Float32Array(faces.flatMap(([a, b, c, d]) =>
    [...corner(a[0], a[1], a[2]), ...b, ...c, ...a, ...c, ...d]));

  const volumes = { positions, materials: new Uint32Array(12).fill(3) };

  expect(containingVolume(volumes, [0.1, 0.2, -0.3])).toBe(3);
  expect(containingVolume(volumes, [0, 0, 5])).toBe(NO_MATERIAL);
  expect(containingVolume(null, [0, 0, 0])).toBe(NO_MATERIAL);
});
