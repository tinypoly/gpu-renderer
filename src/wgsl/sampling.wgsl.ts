/** The geometric primitives the lobes and the light sampling are built from. */
export const SAMPLING_WGSL = /* wgsl */ `
/** Branchless orthonormal basis around a unit normal (Duff and others, 2017). */
fn orthonormalBasis(n: vec3<f32>) -> mat3x3<f32> {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;

  return mat3x3<f32>(
    vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
    vec3<f32>(b, s + n.y * n.y * a, -n.y),
    n);
}

/** Cosine weighted direction in the hemisphere around the normal; the weight cancels the Lambert cosine. */
fn cosineDirection(normal: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
  let radius = sqrt(u.x);
  let angle = 2.0 * PI * u.y;
  let local = vec3<f32>(radius * cos(angle), radius * sin(angle), sqrt(max(0.0, 1.0 - u.x)));

  return normalize(orthonormalBasis(normal) * local);
}

/** A direction scattered from forward by the Henyey-Greenstein phase function, which its own sampling cancels. */
fn sampleHenyeyGreenstein(forward: vec3<f32>, g: f32, u: vec2<f32>) -> vec3<f32> {
  var cosine = 1.0 - 2.0 * u.x;

  if (abs(g) > 1e-3) {
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
    cosine = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
  }

  let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
  let angle = 2.0 * PI * u.y;

  return normalize(orthonormalBasis(forward) * vec3<f32>(sine * cos(angle), sine * sin(angle), cosine));
}

/**
 * The Henyey-Greenstein phase function, per steradian, for a turn whose cosine is given from the direction of travel:
 * the density sampleHenyeyGreenstein draws with.
 */
fn henyeyGreenstein(cosine: f32, g: f32) -> f32 {
  let denominator = 1.0 + g * g - 2.0 * g * cosine;

  return (1.0 - g * g) / (4.0 * PI * denominator * sqrt(max(denominator, 1e-12)));
}

/** The unit normal of the triangle's plane; which side it faces does not matter, every emitter is two sided. */
fn triangleNormal(triangle: u32) -> vec3<f32> {
  let a = triangleVertex(triangle, 0u);

  return normalize(cross(triangleVertex(triangle, 1u) - a, triangleVertex(triangle, 2u) - a));
}
`;
