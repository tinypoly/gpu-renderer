import type { PackedEnvironment } from "./environment.js";
import type { GpuRenderSettings } from "./settings.js";
import type { SerializedLight } from "./upstream.js";

/**
 * How much more each colour scatters than green, at 680, 550 and 440 nm. Clean air scatters as the inverse fourth
 * power of the wavelength (Rayleigh), haze nearly alike (Mie, an Angstrom exponent of about 1.3).
 */
const RAYLEIGH = [(550 / 680) ** 4, 1, (550 / 440) ** 4];
const MIE = [(550 / 680) ** 1.3, 1, (550 / 440) ** 1.3];

/** Koschmieder: a dark object fades to 2 percent contrast over the visibility, so green's extinction is ln 50 over it. */
const KOSCHMIEDER = Math.log(50);

/** Extinction per scene unit of red, green and blue for the atmosphere setting, or nothing when it is off. */
export function atmosphereExtinction(atmosphere: GpuRenderSettings["atmosphere"]): [number, number, number] {
  const visibility = Number(atmosphere.visibility);

  if (!atmosphere.enabled || !(visibility > 0)) return [0, 0, 0];

  const haze = Math.min(1, Math.max(0, Number(atmosphere.haze) || 0));
  const scale = KOSCHMIEDER / visibility;

  return [0, 1, 2].map(k => scale * (RAYLEIGH[k] + (MIE[k] - RAYLEIGH[k]) * haze)) as [number, number, number];
}

/** The brightest directional light of a scene, which lights the air: its direction towards the light and irradiance. */
export function brightestSun(lights: readonly SerializedLight[]): { direction: number[]; irradiance: number[] } | null {
  let best: { direction: number[]; irradiance: number[] } | null = null;
  let strongest = 0;

  for (const light of lights) {
    if (light.type !== "DirectionalLight" || !(light.intensity > 0)) continue;

    const irradiance = light.color.map(channel => channel * light.intensity);
    const strength = 0.2126 * irradiance[0] + 0.7152 * irradiance[1] + 0.0722 * irradiance[2];
    const axis = [0, 1, 2].map(k => light.position[k] - light.target[k]);
    const size = Math.hypot(...axis);

    if (strength > strongest && size > 0) {
      strongest = strength;
      best = { direction: axis.map(value => value / size), irradiance };
    }
  }

  return best;
}

const averages = new WeakMap<PackedEnvironment, [number, number, number]>();

/**
 * The radiance of an image averaged over the whole sphere: what air scattering every direction alike sends towards the
 * camera. Each row weighs by the sine of its colatitude, the share of the sphere it covers.
 */
export function averageRadiance(environment: PackedEnvironment): [number, number, number] {
  const known = averages.get(environment);

  if (known) return known;

  const { width, height, data } = environment.texture;
  const sum = [0, 0, 0];
  let weight = 0;
  // A byte image is stored as it was encoded; only a float one holds linear light, as an HDR image always does.
  const scale = data instanceof Float32Array ? 1 : 1 / 255;

  for (let y = 0; y < height; y++) {
    const sine = Math.sin((y + 0.5) / height * Math.PI);

    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * 4;

      sum[0] += data[at] * scale * sine;
      sum[1] += data[at + 1] * scale * sine;
      sum[2] += data[at + 2] * scale * sine;
    }

    weight += sine * width;
  }

  const average = sum.map(channel => channel / Math.max(weight, 1e-12)) as [number, number, number];

  averages.set(environment, average);

  return average;
}

/**
 * A vertical gradient averaged over the sphere. Directions are spread evenly in height over a sphere, so the share of
 * the top colour is the mean of h^exponent for h even in [0, 1]: one over exponent plus one.
 */
export function gradientAverage(top: number[], bottom: number[], exponent: number): [number, number, number] {
  const share = 1 / (Math.max(exponent, 0) + 1);

  return [0, 1, 2].map(k => bottom[k] + (top[k] - bottom[k]) * share) as [number, number, number];
}
