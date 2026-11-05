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

    // The probabilities packed tight in the texture buffer: the same floats as lights[].cdf, a word apart.
    if (extraFloat(frame.lightCdf + middle) > u) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }

  return lights[low];
}

const TREE_NODE_WORDS: u32 = 16u;
const TREE_LEAF: u32 = 0x80000000u;
const TREE_ROOT_PARENT: u32 = 0xffffffffu;
const NO_LIGHT: u32 = 0xffffffffu;

/**
 * How much the lights under a node of the light tree can deliver to a point: their power over the square of the
 * distance, times how well their cone of directions and the receiving surface face each other, each widened by the
 * angle the node's box takes from the point (Conty Estevez and Kulla's importance). A point inside the box rules
 * nothing out. A zero normal is a receiver that takes light from everywhere, such as a point in a volume.
 */
fn lightNodeImportance(node: u32, position: vec3<f32>, normal: vec3<f32>) -> f32 {
  let at = frame.lightTree + node * TREE_NODE_WORDS;
  let low = extraVec4(at);
  let high = extraVec4(at + 4u);
  let cone = extraVec4(at + 8u);

  if (low.w <= 0.0) {
    return 0.0;
  }

  let centre = (low.xyz + high.xyz) * 0.5;
  let radius = length(high.xyz - low.xyz) * 0.5;
  let offset = position - centre;
  let distance2 = dot(offset, offset);

  if (distance2 <= radius * radius) {
    return low.w / max(radius * radius, 1e-12);
  }

  let distance = sqrt(distance2);
  let spread = asin(clamp(radius / distance, 0.0, 1.0));
  var orientation = 1.0;

  // A cone of minus one is the whole sphere: such lights shine towards every point alike.
  if (high.w > -1.0) {
    let theta = acos(clamp(dot(cone.xyz, offset / distance), -1.0, 1.0));
    let past = max(0.0, theta - acos(high.w) - spread);

    orientation = select(0.0, cos(past), past <= acos(clamp(cone.w, -1.0, 1.0)));
  }

  var facing = 1.0;

  if (any(normal != vec3<f32>(0.0))) {
    let past = max(0.0, acos(clamp(dot(normal, -offset / distance), -1.0, 1.0)) - spread);

    facing = select(0.0, cos(past), past < 0.5 * PI);
  }

  return low.w * orientation * facing / distance2;
}

/** The chance of stepping from a node of the tree to each of its children, from this point. */
fn lightChildOdds(left: u32, right: u32, position: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
  var importance = vec2<f32>(0.0);

  // One loop over the two children, so the importance is written into the kernel once per walk.
  for (var k = 0u; k < 2u; k = k + 1u) {
    importance[k] = lightNodeImportance(select(right, left, k == 0u), position, normal);
  }

  let total = importance.x + importance.y;

  return select(importance / total, vec2<f32>(0.0), total <= 0.0);
}

/** A light picked down the tree by importance from this point, with the chance of picking it; NO_LIGHT for none. */
fn sampleLightTree(position: vec3<f32>, normal: vec3<f32>, u: f32, pdf: ptr<function, f32>,
  density: ptr<function, f32>) -> u32 {
  var node = 0u;
  var chance = 1.0;
  var x = u;

  for (var depth = 0u; depth < 64u; depth = depth + 1u) {
    let at = frame.lightTree + node * TREE_NODE_WORDS;
    let first = textures[at + 12u];

    if ((first & TREE_LEAF) != 0u) {
      *pdf = chance;
      *density = extraFloat(at + 13u);

      return first & ~TREE_LEAF;
    }

    let second = textures[at + 13u];
    let odds = lightChildOdds(first, second, position, normal);

    if (odds.x + odds.y <= 0.0) {
      return NO_LIGHT;
    }

    if (x < odds.x) {
      node = first;
      chance = chance * odds.x;
      x = x / odds.x;
    } else {
      node = second;
      chance = chance * odds.y;
      x = (x - odds.x) / max(odds.y, 1e-20);
    }

    x = min(x, 0.99999994);
  }

  return NO_LIGHT;
}

/** The chance the tree would have picked the local light at this index from this point, walked up from its leaf. */
fn lightTreePdf(index: u32, position: vec3<f32>, normal: vec3<f32>) -> f32 {
  var node = textures[frame.lightLeaves + index];
  var chance = extraFloat(frame.lightTree + node * TREE_NODE_WORDS + 13u);

  for (var depth = 0u; depth < 64u; depth = depth + 1u) {
    let parent = textures[frame.lightTree + node * TREE_NODE_WORDS + 14u];

    if (parent == TREE_ROOT_PARENT) {
      return chance;
    }

    let at = frame.lightTree + parent * TREE_NODE_WORDS;
    let first = textures[at + 12u];
    let odds = lightChildOdds(first, textures[at + 13u], position, normal);

    chance = chance * select(odds.y, odds.x, node == first);
    node = parent;
  }

  return 0.0;
}

/**
 * The area density with which direct sampling from this point would have picked a point on this triangle, or zero if
 * it never would. Only the leading entries are emissive triangles, and they are in triangle order, so one binary search
 * finds the entry; the tree, when there is one, says how likely it was from here.
 */
fn lightPdfOfTriangle(triangle: u32, position: vec3<f32>, normal: vec3<f32>) -> f32 {
  var low = 0u;
  var high = frame.areaLightCount;

  loop {
    if (low >= high) {
      break;
    }

    let middle = (low + high) / 2u;
    let candidate = textures[frame.lightTriangles + middle];

    if (candidate == triangle) {
      if (frame.lightTree != 0u) {
        return frame.localLightPick * lightTreePdf(middle, position, normal);
      }

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
fn sampleLight(position: vec3<f32>, normal: vec3<f32>, pick: f32, u: vec2<f32>) -> LightSample {
  if (frame.lightCount == 0u) {
    return noLight();
  }

  // The lights in the scene go through the tree, by what they can deliver here; what lies outside it by its weight.
  if (frame.lightTree != 0u && pick < frame.localLightPick) {
    var chance = 0.0;
    var density = 1.0;
    let index = sampleLightTree(position, normal, pick / frame.localLightPick, &chance, &density);

    if (index == NO_LIGHT || chance <= 0.0) {
      return noLight();
    }

    return sampleLightEntry(lights[index], frame.localLightPick * chance * density, position, u);
  }

  let light = lightByProbability(pick);

  return sampleLightEntry(light, light.pdf, position, u);
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

/** pdf is the chance of picking this light, times the area density of a point on it where it has an area. */
fn sampleLightEntry(light: Light, pdf: f32, position: vec3<f32>, u: vec2<f32>) -> LightSample {
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

    let density = pdf * drawn.pdf;

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
    sample.value = light.color / pdf;

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
    let density = pdf * distanceSquared / cosLight;
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
  sample.value = light.color * (power / pdf);

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

  return (flags & (MATERIAL_TRANSPARENT | MATERIAL_ALPHA_HASH | MATERIAL_WIREFRAME | MATERIAL_VOLUME)) == 0u
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
 * lets its own share through, tinted. A volume's boundary lets the ray in or out, and the medium inside takes its
 * share over the distance between; medium is the density at the shaded point itself, zero outside any volume.
 */
fn lightTransmittance(position: vec3<f32>, chosen: LightSample, rng: ptr<function, PathSampler>, skipMaterial: u32,
  medium: f32) -> vec3<f32> {
  // A light that casts no shadow is blocked by nothing, but a medium on the way still takes its share: then the walk
  // below only looks for volumes and passes every other surface.
  if (!chosen.occlude && !VOLUMES) {
    return vec3<f32>(1.0);
  }

  if (chosen.occlude) {
    // Most blockers are plainly opaque, and for those any triangle on the way settles it: the search stops at the
    // first one found instead of looking for the closest. Anything that lets light through goes the long way below.
    let first = traceAny(position, chosen.direction, chosen.distance);

    if (first == NO_HIT) {
      return vec3<f32>(exp(-medium * chosen.distance));
    }

    let firstMaterial = materialIndexOf(first);

    if (firstMaterial != skipMaterial && blocksAll(firstMaterial)) {
      return vec3<f32>(0.0);
    }
  }

  var origin = position;
  var remaining = chosen.distance;
  var through = vec3<f32>(1.0);
  // The medium the ray is in, and how far along the ray it entered it.
  var density = medium;
  var travelled = 0.0;
  var enteredAt = 0.0;

  for (var skip = 0u; skip < MAX_SHADOW_SKIPS; skip = skip + 1u) {
    let blocker = traceScene(origin, chosen.direction, remaining);

    if (blocker.triangle == NO_HIT) {
      return through * exp(-density * (travelled + remaining - enteredAt));
    }

    let bary = vec2<f32>(blocker.u, blocker.v);
    let m = materialIndexOf(blocker.triangle);
    let blending = materials[m].blending;

    if (VOLUMES && (materials[m].flags & MATERIAL_VOLUME) != 0u) {
      let at = travelled + blocker.distance;

      if (dot(authoredNormal(blocker.triangle, bary), chosen.direction) < 0.0) {
        density = select(0.0, materials[m].volume.w, frame.volumesEnabled != 0u);
        enteredAt = at;
      } else {
        through = through * exp(-density * (at - enteredAt));
        density = 0.0;
      }
    } else if (chosen.occlude && m != skipMaterial && materials[m].kind != KIND_SHADOW && blending != BLEND_ADDITIVE) {
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
    travelled = travelled + blocker.distance + RAY_EPSILON;

    if (remaining <= RAY_EPSILON) {
      return through * exp(-density * (travelled - enteredAt));
    }
  }

  // Too many surfaces to see through: a shadow ray gives up dark, a ray that only counts the media keeps what it has.
  return select(vec3<f32>(0.0), through, !chosen.occlude);
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

    let chosen = sampleLightEntry(light, light.pdf, position, sample2D(rng));

    if (!chosen.valid) {
      continue;
    }

    let visible = luminanceOf(lightTransmittance(position, chosen, rng, NO_MATERIAL, 0.0));

    shadow = max(shadow, 1.0 - clamp(visible, 0.0, 1.0));
  }

  return shadow;
}
`;
