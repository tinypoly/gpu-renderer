import type { SerializedMaterial } from "./upstream.js";

/**
 * How a material scatters light inside itself, from `userData.cpuRenderer.subsurface`. `true` takes every default.
 *
 * The material's own colour (with its map) is what the medium reflects in all once light has wandered through it, as
 * in Blender's Principled BSDF, so a textured skin keeps its texture; `radius` says how far each colour gets before it
 * scatters, which is what makes red bleed through an ear and blue stay at the surface.
 */
export type SubsurfaceInput = boolean | {
  /** Share of the diffuse layer that scatters inside rather than at the surface, `0` to `1`. Defaults to `1`. */
  weight?: number;
  /**
   * Mean free path of red, green and blue, as multiples of `scale`; one number for all three. Defaults to
   * `[1, 0.2, 0.1]`, Blender's, which suits skin.
   */
  radius?: number | readonly number[];
  /** World units `radius` is measured in. Defaults to `0.05`: five centimetres in a scene modelled in metres. */
  scale?: number;
  /** Henyey-Greenstein anisotropy of each scattering, `-1` (back) to `1` (forward). Defaults to `0`. */
  anisotropy?: number;
};

/** A medium resolved to what the kernel reads. */
export interface SubsurfaceMedium {
  weight: number;
  /** Mean free path per channel, in world units. */
  radius: [number, number, number];
  anisotropy: number;
}

/** A snapshot material, with the medium `serializeLive` attaches; a snapshot built elsewhere can carry one too. */
export type SubsurfaceMaterial = SerializedMaterial & { subsurface?: SubsurfaceInput };

const DEFAULT_RADIUS: [number, number, number] = [1, 0.2, 0.1];
const DEFAULT_SCALE = 0.05;

const finite = (value: unknown, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

/** The medium an input describes, with its defaults, or `null` when it describes none. */
export function subsurfaceMedium(input: unknown): SubsurfaceMedium | null {
  if (input === true) input = {};
  if (!input || typeof input !== "object") return null;

  const value = input as Exclude<SubsurfaceInput, boolean>;
  const weight = Math.min(1, Math.max(0, finite(value.weight, 1)));

  if (weight <= 0) return null;

  const scale = Math.max(0, finite(value.scale, DEFAULT_SCALE));

  const radius = typeof value.radius === "number"
    ? [value.radius, value.radius, value.radius]
    : Array.isArray(value.radius) && value.radius.length >= 3
      ? value.radius.slice(0, 3)
      : DEFAULT_RADIUS;

  // A path that never travels would divide by zero; a hair of distance keeps it a very dense medium instead.
  const paths = radius.map((channel, k) => Math.max(1e-5, finite(channel, DEFAULT_RADIUS[k]) * scale));

  return {
    weight,
    radius: [paths[0], paths[1], paths[2]],
    anisotropy: Math.min(0.95, Math.max(-0.95, finite(value.anisotropy, 0))),
  };
}

/** The medium a snapshot material carries, if any. */
export function subsurfaceOf(material: SerializedMaterial | undefined): SubsurfaceMedium | null {
  return subsurfaceMedium((material as SubsurfaceMaterial | undefined)?.subsurface);
}

/**
 * A mesh filled with a medium, from `userData.cpuRenderer.volume`: its surface is only the boundary, invisible, and
 * light scatters and fades in the space it encloses. The mesh has to be closed, with its normals pointing out.
 */
export type VolumeInput = {
  /** How much of the light the medium takes per scene unit: one over the mean distance a ray gets. Defaults to `1`. */
  density?: number;
  /** What it keeps at each scattering, `#rrggbb` or linear rgb: white for fog and clouds, dark for soot. */
  color?: string | readonly number[];
  /** Henyey-Greenstein anisotropy, `-1` (back) to `1` (forward). Defaults to `0`; haze and dust are forward, ~0.7. */
  anisotropy?: number;
};

export interface VolumeMedium {
  density: number;
  /** Single scattering albedo, linear rgb. */
  albedo: [number, number, number];
  anisotropy: number;
}

/** A snapshot material that may carry a volume, as `serializeLive` attaches it. */
export type VolumeMaterial = SerializedMaterial & { volume?: VolumeInput };

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

function albedoOf(color: unknown): [number, number, number] {
  if (Array.isArray(color) && color.length >= 3) {
    return [0, 1, 2].map(k => Math.min(1, Math.max(0, finite(color[k], 1)))) as [number, number, number];
  }

  const match = typeof color === "string" ? /^#?([0-9a-f]{6})$/i.exec(color) : null;

  if (!match) return [1, 1, 1];

  const value = parseInt(match[1], 16);

  return [(value >> 16) & 255, (value >> 8) & 255, value & 255].map(c => srgbToLinear(c / 255)) as
    [number, number, number];
}

/** The medium an input describes, with its defaults, or `null` when it describes none. */
export function volumeMedium(input: unknown): VolumeMedium | null {
  if (input === true) input = {};
  if (!input || typeof input !== "object") return null;

  const value = input as VolumeInput;
  const density = Math.max(0, finite(value.density, 1));

  if (density <= 0) return null;

  return {
    density,
    albedo: albedoOf(value.color),
    anisotropy: Math.min(0.95, Math.max(-0.95, finite(value.anisotropy, 0))),
  };
}

/**
 * Rounded edges at shading time, from `userData.cpuRenderer.bevel`: the radius, in scene units, over which the
 * shading normal turns from one face to the next, as Cycles' bevel node rounds them. A number, or `{ radius }`.
 */
export type BevelInput = number | { radius?: number };

/** A snapshot material that may carry a bevel, as `serializeLive` attaches it. */
export type BevelMaterial = SerializedMaterial & { bevel?: BevelInput };

/** The bevel radius an input asks for, or zero. */
export function bevelRadius(input: unknown): number {
  const radius = typeof input === "number" ? input : (input as { radius?: unknown } | null)?.radius;

  return typeof radius === "number" && Number.isFinite(radius) && radius > 0 ? radius : 0;
}

export function bevelOf(material: SerializedMaterial | undefined): number {
  return bevelRadius((material as BevelMaterial | undefined)?.bevel);
}

export function volumeOf(material: SerializedMaterial | undefined): VolumeMedium | null {
  return volumeMedium((material as VolumeMaterial | undefined)?.volume);
}
