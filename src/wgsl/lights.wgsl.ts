/**
 * Picking a light and aiming a shadow ray at it. Emissive triangles and Three's analytic lights answer the same
 * question, so the integrator asks it once: given two random numbers, where does light come from, how much of it
 * arrives, and how likely was this direction to be found this way.
 *
 * The falloff, the spot cone and the rectangle's projected area follow `material.js` in the CPU renderer, so direct
 * light from an analytic light agrees with it.
 */
export const LIGHTS_WGSL = /* wgsl */ `
struct LightSample {
  // Unit vector from the shaded point towards the light.
  direction: vec3<f32>,
  // How far an occlusion ray may travel before it is past the light.
  distance: f32,
  // What the light delivers, already divided by the density it was sampled with.
  value: vec3<f32>,
  // Solid angle density of this direction, for the power heuristic. Zero when no bounce ray could find this light,
  // which is every light that occupies no geometry, and then the light strategy keeps the whole contribution.
  misPdf: f32,
  valid: bool,
  occlude: bool,
  kind: u32,
};

fn noLight() -> LightSample {
  var sample: LightSample;
  sample.valid = false;

  return sample;
}

/** The light whose cumulative probability range contains u. */
fn lightByProbability(u: f32) -> Light {
  var low = 0u;
  var high = frame.lightCount - 1u;

  loop {
    if (low >= high) {
      break;
    }

    let middle = (low + high) / 2u;

    if (lights[middle].cdf > u) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }

  return lights[low];
}

/**
 * The area density with which direct sampling would have picked a point on this triangle, or zero if it never would.
 * Only the leading entries are emissive triangles, and they are in triangle order, so one binary search answers it.
 */
fn lightPdfOfTriangle(triangle: u32) -> f32 {
  var low = 0u;
  var high = frame.areaLightCount;

  loop {
    if (low >= high) {
      break;
    }

    let middle = (low + high) / 2u;
    let candidate = lights[middle].triangle;

    if (candidate == triangle) {
      return lights[middle].pdf;
    }

    if (candidate < triangle) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }

  return 0.0;
}

/** The power heuristic: how much of a contribution the strategy that produced it keeps, against the other one. */
fn misWeight(pdf: f32, other: f32) -> f32 {
  let a = pdf * pdf;
  let b = other * other;

  return select(a / (a + b), 0.0, a + b <= 0.0);
}

/**
 * Spreads a direction over a cone, which is how a light of some apparent size softens its own shadow. The settings give
 * the angular diameter; a rasterizer would blur the shadow map by the same amount.
 */
fn spreadDirection(direction: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
  if (frame.shadowSpread <= 0.0) {
    return direction;
  }

  let basis = orthonormalBasis(direction);
  let radius = sqrt(u.x) * frame.shadowSpread;
  let angle = 2.0 * PI * u.y;

  return normalize(direction + basis[0] * (radius * cos(angle)) + basis[1] * (radius * sin(angle)));
}

/** Three's getDistanceAttenuation: the inverse power law, cut off smoothly at the light's range. */
fn distanceFalloff(light: Light, distance: f32) -> f32 {
  var falloff = 1.0 / max(pow(distance, light.decay), 0.01);

  if (light.cutoff > 0.0) {
    let edge = clamp(1.0 - pow(distance / light.cutoff, 4.0), 0.0, 1.0);

    falloff = falloff * edge * edge;
  }

  return falloff;
}

/** A point uniformly distributed over a rectangular light, and the axis it emits along. */
fn rectanglePoint(light: Light, u: vec2<f32>) -> vec3<f32> {
  return light.position + light.tangent * (u.x * 2.0 - 1.0) + light.bitangent * (u.y * 2.0 - 1.0);
}

/**
 * Where the light that pick selects arrives from, and how much of it. u places the point on a surface light or spreads a
 * point light's shadow, so the same two dimensions serve every kind.
 */
fn sampleLight(position: vec3<f32>, pick: f32, u: vec2<f32>) -> LightSample {
  if (frame.lightCount == 0u) {
    return noLight();
  }

  return sampleLightEntry(lightByProbability(pick), position, u);
}

/** Three's spotLightMap: inside the projection's frustum the texture tints the light; outside it leaves it alone. */
fn spotLightTint(light: Light, position: vec3<f32>) -> vec3<f32> {
  if ((light.flags & LIGHT_HAS_MAP) == 0u) {
    return vec3<f32>(1.0);
  }

  let at = light.triangle;
  let map = textures[at];
  let matrix = mat4x4<f32>(extraVec4(at + 1u), extraVec4(at + 5u), extraVec4(at + 9u), extraVec4(at + 13u));
  let coordinate = matrix * vec4<f32>(position, 1.0);

  if (coordinate.w <= 0.0) {
    return vec3<f32>(1.0);
  }

  let projected = coordinate.xyz / coordinate.w;

  if (any(projected <= vec3<f32>(0.0)) || any(projected >= vec3<f32>(1.0))) {
    return vec3<f32>(1.0);
  }

  return sampleTextureRaw(map, projected.xy).rgb;
}

fn sampleLightEntry(light: Light, position: vec3<f32>, u: vec2<f32>) -> LightSample {
  var sample: LightSample;
  sample.valid = true;
  sample.kind = light.kind;
  // Emissive geometry is always tested: a panel that lit the room through its own wall would not be a path tracer.
  sample.occlude = light.kind == LIGHT_AREA
    || ((light.flags & LIGHT_CASTS_SHADOW) != 0u && frame.shadowsEnabled != 0u);
  sample.misPdf = 0.0;

  if (light.kind == LIGHT_ENVIRONMENT) {
    let drawn = sampleEnvironmentDirection(u);

    if (drawn.pdf <= 0.0) {
      return noLight();
    }

    let density = light.pdf * drawn.pdf;

    sample.direction = drawn.direction;
    sample.distance = 1e30;
    sample.value = environmentRadiance(drawn.direction) / density;
    // A bounce ray that escapes finds the same light, so the two strategies share it.
    sample.misPdf = density;

    return sample;
  }

  if (light.kind == LIGHT_DIRECTIONAL) {
    sample.direction = spreadDirection(light.direction, u);
    sample.distance = 1e30;
    // The colour already carries the intensity, which for a directional light is the irradiance it delivers.
    sample.value = light.color / light.pdf;

    return sample;
  }

  if (light.kind == LIGHT_AREA || light.kind == LIGHT_RECT) {
    let area = light.kind == LIGHT_AREA;
    let bary = triangleBarycentric(u);
    let point = select(rectanglePoint(light, u), trianglePointAt(light.triangle, bary), area);
    let axis = select(normalize(cross(light.tangent, light.bitangent)), triangleNormal(light.triangle), area);
    let offset = point - position;
    let distanceSquared = dot(offset, offset);
    let distance = sqrt(distanceSquared);

    if (distance <= RAY_EPSILON) {
      return noLight();
    }

    sample.direction = offset / distance;
    // An emissive triangle radiates from both sides in this milestone; a rectangular light only from the side it faces.
    let cosine = dot(axis, sample.direction);
    let cosLight = select(max(cosine, 0.0), abs(cosine), area);

    if (cosLight <= 1e-6) {
      return noLight();
    }

    // Stop just short of the light so its own surface is not reported as the blocker.
    sample.distance = distance * (1.0 - 1e-3);
    let density = light.pdf * distanceSquared / cosLight;
    // The emissive map is read at the very point that was sampled, so this agrees with what a bounce ray would find.
    let emission = select(light.color, emissionAt(light.triangle, bary), area);

    sample.value = emission / density;
    // Only a triangle of the scene's geometry can also be found by a bounce ray, so only it needs the heuristic.
    sample.misPdf = select(0.0, density, area);

    return sample;
  }

  // A point or a spot light: both fall off with distance and are sampled at their centre.
  let offset = light.position - position;
  let distance = length(offset);

  if (distance <= RAY_EPSILON) {
    return noLight();
  }

  let towards = offset / distance;
  var power = distanceFalloff(light, distance);

  if (light.kind == LIGHT_SPOT) {
    let cosine = dot(light.direction, towards);

    if (light.innerCos <= light.outerCos) {
      power = power * select(0.0, 1.0, cosine >= light.outerCos);
    } else {
      let t = clamp((cosine - light.outerCos) / (light.innerCos - light.outerCos), 0.0, 1.0);

      power = power * t * t * (3.0 - 2.0 * t);
    }

    if (power <= 0.0) {
      return noLight();
    }
  }

  sample.direction = spreadDirection(towards, u);
  sample.distance = distance;
  sample.value = light.color * (power / light.pdf);

  if (light.kind == LIGHT_SPOT) {
    sample.value = sample.value * spotLightTint(light, position);
  }

  return sample;
}

/**
 * Whether a material stops every shadow ray that meets it, wherever: opaque, no cutout, no transmission, no clipping
 * or wireframe, no shader that might discard, and normal blending. Only then can the first triangle found decide.
 */
fn blocksAll(m: u32) -> bool {
  let flags = materials[m].flags;
  let blending = materials[m].blending;

  return (flags & (MATERIAL_TRANSPARENT | MATERIAL_ALPHA_HASH | MATERIAL_WIREFRAME)) == 0u
    && materials[m].alphaTest <= 0.0
    && materials[m].transmission <= 0.0
    && materials[m].clipCount == 0u
    && materials[m].program == NO_PROGRAM
    && materials[m].kind != KIND_SHADOW
    && (blending == BLEND_NORMAL || blending == BLEND_NONE);
}

/** How many surfaces a shadow ray may pass through before it gives up and calls the light blocked. */
const MAX_SHADOW_SKIPS: u32 = 8u;

/** No material to skip: every blocker counts. */
const NO_MATERIAL: u32 = 0xffffffffu;

/**
 * What a shadow ray keeps of a thin transmissive wall it meets: the share Three lets through, tinted by the base colour,
 * less what the interface reflects. It is the expectation of the straight crossing in sampleDielectric, so a light
 * seen through a window is found the same by both estimators. A volume bends the ray it lets through, so it blocks.
 */
fn thinCrossing(triangle: u32, bary: vec2<f32>, direction: vec3<f32>) -> vec3<f32> {
  let m = materialIndexOf(triangle);

  if (materials[m].transmission <= 0.0) {
    return vec3<f32>(0.0);
  }

  let metalness = materials[m].metalness * sampleMap(materials[m].metalnessMap, triangle, bary).b;
  let transmission = transmissionAt(m, triangle, bary, metalness);
  let thickness = materials[m].thickness * sampleMap(materials[m].thicknessMap, triangle, bary).g;

  if (transmission <= 0.0 || thickness > 0.0) {
    return vec3<f32>(0.0);
  }

  let albedo = materials[m].color.rgb * sampleMap(materials[m].map, triangle, bary).rgb;
  let reflectance = fresnelDielectric(abs(dot(authoredNormal(triangle, bary), direction)), 1.0 / max(materials[m].ior, 1.0));

  return albedo * (transmission * (1.0 - reflectance));
}

/** The colour a blended surface filters what is behind it by: its base colour, with the map and vertex colours. */
fn blendColor(triangle: u32, bary: vec2<f32>) -> vec3<f32> {
  let m = materialIndexOf(triangle);
  var color = materials[m].color.rgb * sampleMap(materials[m].map, triangle, bary).rgb;

  if ((materials[m].flags & MATERIAL_VERTEX_COLORS) != 0u) {
    color = color * triangleColor(triangle, bary).rgb;
  }

  return color;
}

/**
 * How much of the light this sample came from reaches the shaded point. A surface that only partly covers its triangle
 * blocks the ray only that often, which is what makes a cutout leaf cast a cutout shadow; a thin transmissive wall
 * lets its own share through, tinted.
 */
fn lightTransmittance(
  position: vec3<f32>, chosen: LightSample, rng: ptr<function, PathSampler>, skipMaterial: u32) -> vec3<f32> {
  if (!chosen.occlude) {
    return vec3<f32>(1.0);
  }

  // Most blockers are plainly opaque, and for those any triangle on the way settles it: the search stops at the first
  // one found instead of looking for the closest. Anything that lets light through goes the long way below.
  let first = traceAny(position, chosen.direction, chosen.distance);

  if (first == NO_HIT) {
    return vec3<f32>(1.0);
  }

  let firstMaterial = materialIndexOf(first);

  if (firstMaterial != skipMaterial && blocksAll(firstMaterial)) {
    return vec3<f32>(0.0);
  }

  var origin = position;
  var remaining = chosen.distance;
  var through = vec3<f32>(1.0);

  for (var skip = 0u; skip < MAX_SHADOW_SKIPS; skip = skip + 1u) {
    let blocker = traceScene(origin, chosen.direction, remaining);

    if (blocker.triangle == NO_HIT) {
      return through;
    }

    let bary = vec2<f32>(blocker.u, blocker.v);
    let m = materialIndexOf(blocker.triangle);
    let blending = materials[m].blending;

    // A shadow catcher catches shadows rather than casting them, and a surface that adds light blocks none.
    if (m != skipMaterial && materials[m].kind != KIND_SHADOW && blending != BLEND_ADDITIVE) {
      let coverage = coverageAt(blocker.triangle, bary, chosen.direction);

      if (blending == BLEND_MULTIPLY) {
        through = through * mix(vec3<f32>(1.0), blendColor(blocker.triangle, bary), coverage);
      } else if (blending == BLEND_SUBTRACTIVE) {
        through = through * (vec3<f32>(1.0) - blendColor(blocker.triangle, bary) * coverage);
      } else if (coverage >= 1.0 || sample2D(rng).x < coverage) {
        through = through * thinCrossing(blocker.triangle, bary, chosen.direction);
      }

      if (all(through <= vec3<f32>(0.0))) {
        return vec3<f32>(0.0);
      }
    }

    // The ray carries on from just past the surface it saw through.
    origin = origin + chosen.direction * (blocker.distance + RAY_EPSILON);
    remaining = remaining - blocker.distance - RAY_EPSILON;

    if (remaining <= RAY_EPSILON) {
      return through;
    }
  }

  return vec3<f32>(0.0);
}

/**
 * How shadowed a point is, as Three's ShadowMaterial reads it: for every light that casts a shadow, whether something
 * stands between it and the point, the darkest of them winning as in the CPU renderer. A rectangular light casts no
 * shadow in Three, so it takes no part.
 */
fn shadowMaskAt(position: vec3<f32>, rng: ptr<function, PathSampler>) -> f32 {
  if (frame.shadowsEnabled == 0u) {
    return 0.0;
  }

  var shadow = 0.0;

  for (var i = frame.areaLightCount; i < frame.lightCount; i = i + 1u) {
    let light = lights[i];

    if (light.kind == LIGHT_ENVIRONMENT || light.kind == LIGHT_RECT || (light.flags & LIGHT_CASTS_SHADOW) == 0u) {
      continue;
    }

    let chosen = sampleLightEntry(light, position, sample2D(rng));

    if (!chosen.valid) {
      continue;
    }

    let visible = luminanceOf(lightTransmittance(position, chosen, rng, NO_MATERIAL));

    shadow = max(shadow, 1.0 - clamp(visible, 0.0, 1.0));
  }

  return shadow;
}
`;
