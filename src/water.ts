import type { SerializedMaterial } from "./upstream.js";

/**
 * Water, from `userData.cpuRenderer.water`, on a transmissive material whose mesh bounds the water's volume: the path
 * tracer ripples the surface that faces up or down with waves of its own, and lights what lies under it with the
 * caustics the sun makes through those waves. `true` takes every default.
 *
 * The waves are a sum of directional sines over the horizontal plane, read only as a normal: the geometry stays flat,
 * so a pool needs no finer mesh. The caustics come from manifold next event estimation: for a point under the water
 * and the sun, a few Newton steps find the point of the surface whose refraction sends the sun's light to it, and the
 * spread of the refracted beam there sets how much light it brings.
 */
export type WaterInput = boolean | {
  /** Height of the longest wave, in scene units. Defaults to `0.006`, a pool stirred by a breeze. */
  waveHeight?: number;
  /** Length of the longest wave, in scene units. The others are shorter, down to a sixth of it. Defaults to `0.9`. */
  waveLength?: number;
  /** Direction the waves travel to, in radians on the horizontal plane from +x towards +z. Defaults to `0.6`. */
  wind?: number;
  /** Changes the arrangement of the waves. */
  seed?: number;
};

/** Water resolved to its defaults. */
export interface WaterField {
  waveHeight: number;
  waveLength: number;
  wind: number;
  seed: number;
}

/** A snapshot material that may carry water, as `serializeLive` attaches it. */
export type WaterMaterial = SerializedMaterial & { water?: WaterInput };

const finite = (value: unknown, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

/** The water an input describes, with its defaults, or `null` when it describes none. */
export function waterField(input: unknown): WaterField | null {
  if (input === true) input = {};

  if (!input || typeof input !== "object") return null;

  const value = input as Exclude<WaterInput, boolean>;

  return {
    waveHeight: Math.min(1, Math.max(0, finite(value.waveHeight, 0.006))),
    waveLength: Math.min(100, Math.max(0.01, finite(value.waveLength, 0.9))),
    wind: finite(value.wind, 0.6),
    seed: finite(value.seed, 0),
  };
}

/** Water only reads on a physical material that lets light through: elsewhere it has no surface to ripple. */
export function waterOf(material: SerializedMaterial | undefined): WaterField | null {
  if (material?.type !== "MeshPhysicalMaterial" || !(Number(material.values.transmission) > 0)) return null;

  return waterField((material as WaterMaterial).water);
}

/** Waves in the sum, from the longest down to a sixth of it. The kernel holds the same tables. */
export const WATER_WAVES = 8;
export const WATER_SPREAD = [0, 0.55, -0.4, 0.9, -0.8, 0.25, -0.65, 1.1];
export const WATER_PHASE = [0, 1.7, 4.1, 2.6, 5.3, 0.9, 3.4, 5.9];

/** The slope of the waves at a point of the horizontal plane, dh/dx and dh/dz, as the kernel's waterSlope. */
export function waterSlope(field: WaterField, x: number, z: number): [number, number] {
  let sx = 0;
  let sz = 0;

  for (let i = 0; i < WATER_WAVES; i++) {
    const length = field.waveLength * (1 / 6) ** (i / (WATER_WAVES - 1));
    const k = (2 * Math.PI) / length;
    const amplitude = (field.waveHeight * length) / field.waveLength;
    const angle = field.wind + WATER_SPREAD[i];
    const dx = Math.cos(angle);
    const dz = Math.sin(angle);
    const phase = k * (dx * x + dz * z) + WATER_PHASE[i] + field.seed * (i + 1) * 0.37;
    const slope = amplitude * k * Math.cos(phase);

    sx += dx * slope;
    sz += dz * slope;
  }

  return [sx, sz];
}

type Vec3 = [number, number, number];

const normalize = (v: Vec3): Vec3 => {
  const length = Math.hypot(...v) || 1;

  return [v[0] / length, v[1] / length, v[2] / length];
};

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** The normal of the rippled surface facing up, at a point of the horizontal plane. */
export function waterNormal(field: WaterField, x: number, z: number): Vec3 {
  const [sx, sz] = waterSlope(field, x, z);

  return normalize([-sx, 1, -sz]);
}

/** GLSL's refract: the incident direction bent through a surface whose normal faces against it, or zero past the cut. */
export function refract(incident: Vec3, normal: Vec3, eta: number): Vec3 {
  const cosine = dot(normal, incident);
  const k = 1 - eta * eta * (1 - cosine * cosine);

  if (k < 0) return [0, 0, 0];

  const scale = eta * cosine + Math.sqrt(k);

  return [0, 1, 2].map(axis => eta * incident[axis] - scale * normal[axis]) as Vec3;
}

/**
 * Where on the water's plane (height `surfaceY`) the light coming from `toLight` must enter to reach `point` below it,
 * as the kernel's solver finds it: Newton's method on the half vector of the refraction, which has to stand along the
 * rippled normal. `null` when the steps do not settle.
 */
export function solveCausticPoint(
  field: WaterField, point: Vec3, toLight: Vec3, surfaceY: number, ior: number, start: [number, number],
): [number, number] | null {
  const constraint = (x: number, z: number): [number, number] | null => {
    const out = normalize([point[0] - x, point[1] - surfaceY, point[2] - z]);
    const half: Vec3 = [-(toLight[0] + ior * out[0]), -(toLight[1] + ior * out[1]), -(toLight[2] + ior * out[2])];

    if (half[1] <= 1e-6) return null;

    const [sx, sz] = waterSlope(field, x, z);

    return [half[0] / half[1] + sx, half[2] / half[1] + sz];
  };

  let [x, z] = start;
  const h = 1e-4;

  for (let step = 0; step < 16; step++) {
    const value = constraint(x, z);

    if (!value) return null;
    if (Math.hypot(...value) < 1e-5) return [x, z];

    const ax = constraint(x + h, z);
    const az = constraint(x, z + h);

    if (!ax || !az) return null;

    const j00 = (ax[0] - value[0]) / h;
    const j10 = (ax[1] - value[1]) / h;
    const j01 = (az[0] - value[0]) / h;
    const j11 = (az[1] - value[1]) / h;
    const det = j00 * j11 - j01 * j10;

    if (Math.abs(det) < 1e-12) return null;

    let dx = (j11 * value[0] - j01 * value[1]) / det;
    let dz = (-j10 * value[0] + j00 * value[1]) / det;
    const length = Math.hypot(dx, dz);

    // Long steps leave the basin of the solution: the waves are short, and so is a sensible step.
    if (length > 0.25) {
      dx *= 0.25 / length;
      dz *= 0.25 / length;
    }

    x -= dx;
    z -= dz;
  }

  return null;
}
