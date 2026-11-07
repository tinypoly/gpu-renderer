/**
 * Water: the waves that ripple its surface and the caustics the sun makes through them on what lies below. See
 * water.ts, which holds the same tables and mirrors the solver for the tests.
 */

/** The waves, which the surface reads before lighting: compiled in every kernel, they cost nothing until a material asks. */
export const WATER_SURFACE_WGSL = /* wgsl */ `
const WATER_WAVES: u32 = 8u;

/** The slope of the waves at a point of the horizontal plane, dh/dx and dh/dz: a sum of directional sines. */
fn waterSlope(m: u32, p: vec2<f32>) -> vec2<f32> {
  var spread = array<f32, 8>(0.0, 0.55, -0.4, 0.9, -0.8, 0.25, -0.65, 1.1);
  var phases = array<f32, 8>(0.0, 1.7, 4.1, 2.6, 5.3, 0.9, 3.4, 5.9);
  let water = materials[m].water;
  var slope = vec2<f32>(0.0);

  for (var i = 0u; i < WATER_WAVES; i = i + 1u) {
    // From the longest wave down to a sixth of it, all with the same steepness.
    let wavelength = water.y * pow(1.0 / 6.0, f32(i) / f32(WATER_WAVES - 1u));
    let k = 2.0 * PI / wavelength;
    let amplitude = water.x * wavelength / water.y;
    let angle = water.z + spread[i];
    let direction = vec2<f32>(cos(angle), sin(angle));
    let phase = k * dot(direction, p) + phases[i] + water.w * f32(i + 1u) * 0.37;

    slope = slope + direction * (amplitude * k * cos(phase));
  }

  return slope;
}

/** The rippled normal of a water surface that faces up or down, on the side of the facing normal; walls stay flat. */
fn waterNormal(m: u32, point: vec3<f32>, facing: vec3<f32>) -> vec3<f32> {
  if (abs(facing.y) < 0.5) {
    return facing;
  }

  let slope = waterSlope(m, point.xz);
  let up = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));

  return select(up, -up, facing.y < 0.0);
}
`;

/**
 * Manifold next event estimation through one rippled interface: for a point under the water and a sample of the sun,
 * the point of the surface whose refraction sends that light to it, and how much of the light arrives.
 */
export const WATER_CAUSTICS_WGSL = /* wgsl */ `
/** The light a point under water gets from the sun through the surface: the way to it, and the share that arrives. */
struct Caustic {
  found: bool,
  // From the shaded point towards the point of the surface the light comes in through.
  direction: vec3<f32>,
  // Irradiance on the receiving surface over the light's own, already over the cosine the lobes will apply.
  share: vec3<f32>,
};

fn noCaustic() -> Caustic {
  var caustic: Caustic;
  caustic.found = false;
  caustic.direction = vec3<f32>(0.0, 1.0, 0.0);
  caustic.share = vec3<f32>(0.0);
  return caustic;
}

/** How far the half vector of the refraction at (x, z) leans from the rippled normal there: zero at a solution. */
fn causticConstraint(m: u32, point: vec3<f32>, toLight: vec3<f32>, surfaceY: f32, ior: f32, spot: vec2<f32>) -> vec3<f32> {
  let toPoint = normalize(point - vec3<f32>(spot.x, surfaceY, spot.y));
  let bisector = -(toLight + ior * toPoint);

  if (bisector.y <= 1e-6) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }

  let slope = waterSlope(m, spot);

  return vec3<f32>(bisector.x / bisector.y + slope.x, bisector.z / bisector.y + slope.y, 0.0);
}

/** Where the sun's light entering the water at (x, z) meets the receiving plane, in that plane's own coordinates. */
fn causticLanding(m: u32, toLight: vec3<f32>, surfaceY: f32, ior: f32, spot: vec2<f32>, origin: vec3<f32>,
  basis: mat3x3<f32>) -> vec3<f32> {
  let entry = vec3<f32>(spot.x, surfaceY, spot.y);
  let normal = waterNormal(m, entry, vec3<f32>(0.0, 1.0, 0.0));
  let inside = refract(-toLight, normal, 1.0 / ior);
  let across = dot(inside, basis[2]);

  if (abs(across) < 1e-4 || all(inside == vec3<f32>(0.0))) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }

  let landing = entry + inside * (dot(origin - entry, basis[2]) / across);

  return vec3<f32>(dot(landing - origin, basis[0]), dot(landing - origin, basis[1]), 0.0);
}

/** Largest gain a focused caustic may bring, which keeps a fold of the light from becoming a lone bright pixel. */
const CAUSTIC_GAIN: f32 = 40.0;

/**
 * The sun's light at a point under the water m, through its rippled surface: Newton's method finds where it enters,
 * then the way to it is checked both under and above the water, and the spread of the refracted beam over the
 * receiving plane (normal) sets the irradiance, with the share the interface lets in and what the water absorbs.
 */
fn waterCaustic(m: u32, position: vec3<f32>, normal: vec3<f32>, chosen: LightSample, rng: ptr<function, PathSampler>) -> Caustic {
  let toLight = chosen.direction;
  let ior = max(materials[m].ior, 1.0);

  if (toLight.y <= 0.02) {
    return noCaustic();
  }

  // The surface above: along the way the light would come by if the water were still.
  let still = refract(-toLight, vec3<f32>(0.0, 1.0, 0.0), 1.0 / ior);
  let up = traceScene(position, -still, 1e30);

  if (up.triangle == NO_HIT || materialIndexOf(up.triangle) != m) {
    return noCaustic();
  }

  let surfaceY = position.y - still.y * up.distance;
  var spot = position.xz - still.xz * up.distance;
  var settled = false;
  let probe = 1e-4;

  for (var i = 0u; i < 16u; i = i + 1u) {
    let value = causticConstraint(m, position, toLight, surfaceY, ior, spot);

    if (value.z != 0.0) {
      return noCaustic();
    }

    if (length(value.xy) < 1e-5) {
      settled = true;
      break;
    }

    let ax = causticConstraint(m, position, toLight, surfaceY, ior, spot + vec2<f32>(probe, 0.0));
    let az = causticConstraint(m, position, toLight, surfaceY, ior, spot + vec2<f32>(0.0, probe));
    let j = mat2x2<f32>((ax.xy - value.xy) / probe, (az.xy - value.xy) / probe);
    let det = determinant(j);

    if (ax.z != 0.0 || az.z != 0.0 || abs(det) < 1e-12) {
      return noCaustic();
    }

    var delta = vec2<f32>(j[1].y * value.x - j[1].x * value.y, -j[0].y * value.x + j[0].x * value.y) / det;
    let size = length(delta);

    // The waves are short, and so is a step that stays in reach of the same solution.
    if (size > 0.25) {
      delta = delta * (0.25 / size);
    }

    spot = spot - delta;
  }

  if (!settled) {
    return noCaustic();
  }

  let entry = vec3<f32>(spot.x, surfaceY, spot.y);
  let toEntry = entry - position;
  let travel = length(toEntry);
  let direction = toEntry / max(travel, 1e-6);
  let cosine = dot(direction, normal);

  if (direction.y <= 0.0 || cosine <= 0.02) {
    return noCaustic();
  }

  // Under the water: nothing between the point and the surface but the water itself.
  let under = traceScene(position, direction, travel + 1e-3);

  if (under.triangle == NO_HIT || materialIndexOf(under.triangle) != m || abs(under.distance - travel) > 1e-2) {
    return noCaustic();
  }

  // Above it: the sun's own shadow ray, past the water it starts on.
  let through = lightTransmittance(entry + vec3<f32>(0.0, RAY_EPSILON, 0.0), chosen, rng, m, 0.0);

  if (all(through <= vec3<f32>(0.0))) {
    return noCaustic();
  }

  // The spread of the refracted beam: area on the receiving plane per area of the water's plane.
  let basis = orthonormalBasis(normal);
  let landing = causticLanding(m, toLight, surfaceY, ior, spot, position, basis);
  let alongX = causticLanding(m, toLight, surfaceY, ior, spot + vec2<f32>(1e-3, 0.0), position, basis);
  let alongZ = causticLanding(m, toLight, surfaceY, ior, spot + vec2<f32>(0.0, 1e-3), position, basis);

  if (landing.z != 0.0 || alongX.z != 0.0 || alongZ.z != 0.0) {
    return noCaustic();
  }

  let spread = abs(determinant(mat2x2<f32>((alongX.xy - landing.xy) / 1e-3, (alongZ.xy - landing.xy) / 1e-3)));
  let gain = min(1.0 / max(spread, 1e-6), CAUSTIC_GAIN);
  let rippled = waterNormal(m, entry, vec3<f32>(0.0, 1.0, 0.0));
  let enters = 1.0 - fresnelDielectric(dot(toLight, rippled), 1.0 / ior);

  var caustic: Caustic;
  caustic.found = true;
  caustic.direction = direction;
  caustic.share = through * exp(-absorptionOf(m) * travel) * (toLight.y * enters * gain / cosine);
  return caustic;
}
`;

/** The same entry points for a scene with no water, which compile to nothing. */
export const WATER_CAUSTICS_STUB_WGSL = /* wgsl */ `
struct Caustic {
  found: bool,
  direction: vec3<f32>,
  share: vec3<f32>,
};

fn waterCaustic(m: u32, position: vec3<f32>, normal: vec3<f32>, chosen: LightSample, rng: ptr<function, PathSampler>) -> Caustic {
  var caustic: Caustic;
  caustic.found = false;
  caustic.direction = vec3<f32>(0.0, 1.0, 0.0);
  caustic.share = vec3<f32>(0.0);
  return caustic;
}
`;
