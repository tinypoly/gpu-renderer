import type { UniformValue } from "./upstream.js";

/**
 * The uniform a procedural sky declares to say its cumulus can be path traced. The renderer sets it to 1 when it does,
 * and the sky's own shader then leaves those clouds out of what it paints, so they are not counted twice.
 */
export const VOLUMETRIC_CLOUDS_UNIFORM = "uVolumetricClouds";

/** A cloud layer as the kernel reads it. */
export interface CloudLayer {
  /** How much of the sky is covered, from 0 to 1. */
  coverage: number;
  /** How much the clouds take of what is behind them, as a multiple of the usual. */
  density: number;
  /** Size of the clouds, as a multiple of the usual. */
  scale: number;
  /** Which pattern the clouds are drawn in. */
  seed: number;
  /** Altitude of the base of the layer, in kilometres. */
  altitude: number;
  /** How far the layer rises above its base, in kilometres. */
  thickness: number;
  /** How ragged the rim of a cloud is, from 0 to 1. */
  detail: number;
  /** How much of a cloud the air hides per kilometre of distance. */
  haze: number;
  /** How far the sun is up, from 0 at night to 1: the same curve the sky's shader lights its clouds with. */
  sunUp: number;
  /** How much the layer darkens the sun on what is under it, from 0 to 1. */
  shadow: number;
  /** How far the wind has carried the clouds, in kilometres along world X and Z. */
  drift: [number, number];
  /** Hours the clouds have lived since noon: they boil, are born and dissolve with it. */
  time: number;
}

const number = (value: UniformValue | undefined, fallback: number): number =>
  (typeof value === "number" && Number.isFinite(value) ? value : fallback);

/** The wind offset of the pattern, or none when the sky does not move its clouds. */
function driftOf(value: UniformValue | undefined): [number, number] {
  if (!Array.isArray(value) || value.length < 2) return [0, 0];
  const [x, z] = value as number[];
  return [Number.isFinite(x) ? x : 0, Number.isFinite(z) ? z : 0];
}

/** The smooth step around the horizon the sky's shader lights its clouds by. A sky with no sun direction is day. */
function sunElevationWeight(direction: UniformValue | undefined): number {
  if (!Array.isArray(direction) || direction.length < 3) return 1;

  const [x, y, z] = direction as number[], size = Math.hypot(x, y, z) || 1;
  const t = Math.min(1, Math.max(0, (y / size + 0.07) / 0.1));

  return t * t * (3 - 2 * t);
}

/**
 * The cloud layer a procedural sky asks to have path traced, read from the uniforms its own shader draws the same
 * clouds with, or null when it has none: a sky that does not declare {@link VOLUMETRIC_CLOUDS_UNIFORM} keeps painting
 * its clouds itself.
 */
export function cloudLayerOf(uniforms: Record<string, UniformValue> | undefined): CloudLayer | null {
  if (!uniforms || !(VOLUMETRIC_CLOUDS_UNIFORM in uniforms)) return null;

  const coverage = Math.min(1, number(uniforms.uCloudCoverage, 0));
  const altitude = number(uniforms.uCloudAltitude, 0);
  const thickness = number(uniforms.uCloudThickness, 0);
  const scale = number(uniforms.uCloudScale, 1);
  const density = number(uniforms.uCloudDensity, 1);

  if (!(coverage > 0) || !(altitude > 0) || !(thickness > 0) || !(scale > 0) || !(density > 0)) return null;

  return {
    coverage,
    density,
    scale,
    seed: number(uniforms.uCloudSeed, 0),
    altitude,
    thickness,
    detail: Math.min(1, Math.max(0, number(uniforms.uCloudDetail, 0.6))),
    haze: 0.011 + 0.02 * Math.max(0, number(uniforms.uHaze, 0)),
    sunUp: sunElevationWeight(uniforms.uSunDirection),
    shadow: Math.min(1, Math.max(0, number(uniforms.uCloudShadow, 1))),
    drift: driftOf(uniforms.uCloudDrift),
    time: number(uniforms.uCloudTime, 0),
  };
}
