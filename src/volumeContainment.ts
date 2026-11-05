import type { PackedGeometry } from "./geometry.js";
import { MATERIAL_VOLUME } from "./materialLayout.js";

/** What the kernel reads as no material at all. */
export const NO_MATERIAL = 0xffffffff;

/** The triangles that bound a volume, each with its material's index. */
export interface VolumeTriangles {
  /** Nine floats a triangle. */
  positions: Float32Array;
  materials: Uint32Array;
}

/** The boundaries of every volume in a packed scene, or `null` when it has none. */
export function volumeTriangles(geometry: PackedGeometry): VolumeTriangles | null {
  const volume = geometry.materialProperties.map(properties => (properties.flags & MATERIAL_VOLUME) !== 0);

  if (!volume.some(Boolean)) return null;

  const picked: number[] = [];

  for (let triangle = 0; triangle < geometry.triangleCount; triangle++)
    if (volume[geometry.materials[triangle]]) picked.push(triangle);

  const positions = new Float32Array(picked.length * 9);
  const materials = new Uint32Array(picked.length);

  picked.forEach((triangle, k) => {
    positions.set(geometry.positions.subarray(triangle * 9, triangle * 9 + 9), k * 9);
    materials[k] = geometry.materials[triangle];
  });

  return { positions, materials };
}

/** An irrational direction, so a ray from the camera does not run along an edge of an axis aligned box. */
const PROBE = [0.4123, 0.7071, 0.5745].map((value, _, all) => value / Math.hypot(...all));

/**
 * The volume a point stands inside, by the parity of the boundary crossings of a ray from it: an odd count of one
 * volume's triangles means inside it. The camera usually stands in none, so the kernel only asks this of the camera.
 */
export function containingVolume(volumes: VolumeTriangles | null, point: readonly number[]): number {
  if (!volumes) return NO_MATERIAL;

  const crossings = new Map<number, number>();
  const p = volumes.positions;
  const [dx, dy, dz] = PROBE;

  for (let k = 0; k < volumes.materials.length; k++) {
    const at = k * 9;
    const e1x = p[at + 3] - p[at], e1y = p[at + 4] - p[at + 1], e1z = p[at + 5] - p[at + 2];
    const e2x = p[at + 6] - p[at], e2y = p[at + 7] - p[at + 1], e2z = p[at + 8] - p[at + 2];
    // Moller and Trumbore.
    const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
    const det = e1x * hx + e1y * hy + e1z * hz;

    if (Math.abs(det) < 1e-12) continue;

    const inverse = 1 / det;
    const sx = point[0] - p[at], sy = point[1] - p[at + 1], sz = point[2] - p[at + 2];
    const u = (sx * hx + sy * hy + sz * hz) * inverse;

    if (u < 0 || u > 1) continue;

    const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * inverse;

    if (v < 0 || u + v > 1) continue;

    if ((e2x * qx + e2y * qy + e2z * qz) * inverse > 0) {
      const material = volumes.materials[k];

      crossings.set(material, (crossings.get(material) ?? 0) + 1);
    }
  }

  for (const [material, count] of crossings) if (count % 2 === 1) return material;

  return NO_MATERIAL;
}
