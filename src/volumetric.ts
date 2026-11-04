import type { SerializedLight } from "./upstream.js";

/**
 * The CPU renderer's volumetric lights (`userData.cpuRenderer.volumetric` on a point or spot light): light scattered by
 * the air along every view ray, marched as `volumetricLight.js` marches it and added over the image the same way. The
 * kernel reads one record of `VOLUMETRIC_WORDS` floats per light from the texture buffer's tail.
 */
export const VOLUMETRIC_WORDS = 20;

const clamp = (value: number, low = 0, high = 1) => Math.min(high, Math.max(low, value));

function normalize(v: readonly number[]): number[] {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;

  return [v[0] / length, v[1] / length, v[2] / length];
}

/** The records, or an empty array when no light scatters. */
export function packVolumetricLights(lights: readonly SerializedLight[]): Float32Array {
  const volumetric = lights.filter(light => light.volumetric && (light.type === "PointLight" || light.type === "SpotLight"));
  const records = new Float32Array(volumetric.length * VOLUMETRIC_WORDS);

  volumetric.forEach((light, index) => {
    const density = Math.max(0, light.medium?.density ?? 0.01);
    const spread = clamp(light.medium?.spread ?? 0);
    // Scattered light reaches past the cone: up to 1.5 rad (86 degrees) off the axis with a full spread.
    const outer = light.angle + spread * Math.max(0, 1.5 - light.angle);
    const axis = normalize([0, 1, 2].map(k => light.target[k] - light.position[k]));
    const radius = light.distance > 0 ? light.distance : -3 * Math.log(0.0001);
    const spot = light.type === "SpotLight";

    records.set([
      ...light.position, spot ? 1 : 0,
      ...axis, radius,
      ...light.color.map(channel => channel * light.intensity), density,
      light.angle, light.penumbra, light.distance, outer,
      spread, clamp(light.medium?.anisotropy ?? 0, -0.95, 0.95), light.castShadow ? 1 : 0, 0,
    ], index * VOLUMETRIC_WORDS);
  });

  return records;
}
