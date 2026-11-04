// Small matrix helpers. Three is a peer dependency, but the packing path stays free of it so it can be tested in Node
// without a scene. Matrices arrive as Three's column-major sixteen number arrays.

/** Applies a column-major 4x4 to a point, writing three floats at `offset`. */
export function transformPoint(m: readonly number[], x: number, y: number, z: number,
  out: Float32Array, offset: number) {
  const w = m[3] * x + m[7] * y + m[11] * z + m[15] || 1;

  out[offset] = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
  out[offset + 1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
  out[offset + 2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
}

/**
 * The inverse transpose of the upper-left 3x3, as a column-major 3x3. Non-uniform scale makes the world matrix itself
 * wrong for normals, which is what this corrects; a singular matrix falls back to the identity.
 */
export function normalMatrix(m: readonly number[]): Float64Array {
  const a = m[0], b = m[1], c = m[2];
  const d = m[4], e = m[5], f = m[6];
  const g = m[8], h = m[9], i = m[10];
  const cofactor0 = e * i - f * h, cofactor1 = f * g - d * i, cofactor2 = d * h - e * g;
  const determinant = a * cofactor0 + b * cofactor1 + c * cofactor2;

  if (determinant === 0) return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

  const s = 1 / determinant;

  // The inverse is the adjugate over the determinant; transposing it is the same as reading the adjugate row-wise.
  return new Float64Array([
    cofactor0 * s, cofactor1 * s, cofactor2 * s,
    (c * h - b * i) * s, (a * i - c * g) * s, (b * g - a * h) * s,
    (b * f - c * e) * s, (c * d - a * f) * s, (a * e - b * d) * s,
  ]);
}

/** Applies a column-major 3x3 to a direction and normalizes it, writing three floats at `offset`. */
export function transformNormal(n: Float64Array, x: number, y: number, z: number,
  out: Float32Array, offset: number) {
  const nx = n[0] * x + n[3] * y + n[6] * z;
  const ny = n[1] * x + n[4] * y + n[7] * z;
  const nz = n[2] * x + n[5] * y + n[8] * z;
  const length = Math.hypot(nx, ny, nz) || 1;

  out[offset] = nx / length;
  out[offset + 1] = ny / length;
  out[offset + 2] = nz / length;
}
