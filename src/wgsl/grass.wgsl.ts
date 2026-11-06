/**
 * Grass grown on a material's surface: see grass.ts for the layout of what the texture buffer holds.
 *
 * A triangle that grows grass is a shell, its footprint on the ground times the height of the lawn above its plane, and
 * the shells have a hierarchy of their own. Inside a shell the ray walks the world's tiles front to back, and in each
 * one it is taken into the space of that tile's patch, where the plane is the ground and blades stand on it: a shear
 * that keeps the ray's distances as they are. The search in a tile is held to the stretch of the ray over that tile and
 * that triangle, so a blade that leans over a border is seen once, from the patch it is drawn in, and the first tile
 * with a hit holds the closest one in the shell.
 */
export const GRASS_WGSL = /* wgsl */ `
const GRASS_NONE: u32 = 0xffffffffu;
const GRASS_MAX_TILES: u32 = 48u;
// What a hit on a blade carries as its triangle, which no scene triangle can be.
const GRASS_TRIANGLE: u32 = 0xfffffffeu;

struct GrassHit {
  found: bool,
  distance: f32,
  // The lawn's record and the patch triangle hit, both as offsets into the texture buffer.
  record: u32,
  triangle: u32,
  u: f32,
  v: f32,
  // The ground plane's slope and where the tile starts, which take the patch back to the world.
  slope: vec2<f32>,
  tile: vec2<f32>,
};

fn noGrass(maxDistance: f32) -> GrassHit {
  var hit: GrassHit;
  hit.found = false;
  hit.distance = maxDistance;

  return hit;
}

fn grassNode(at: u32) -> BvhNode {
  var node: BvhNode;
  node.leftLow = vec3<f32>(extraFloat(at), extraFloat(at + 1u), extraFloat(at + 2u));
  node.leftLink = bitcast<i32>(textures[at + 3u]);
  node.leftHigh = vec3<f32>(extraFloat(at + 4u), extraFloat(at + 5u), extraFloat(at + 6u));
  node.leftCount = bitcast<i32>(textures[at + 7u]);
  node.rightLow = vec3<f32>(extraFloat(at + 8u), extraFloat(at + 9u), extraFloat(at + 10u));
  node.rightLink = bitcast<i32>(textures[at + 11u]);
  node.rightHigh = vec3<f32>(extraFloat(at + 12u), extraFloat(at + 13u), extraFloat(at + 14u));
  node.rightCount = bitcast<i32>(textures[at + 15u]);

  return node;
}

fn grassVec3(at: u32) -> vec3<f32> {
  return vec3<f32>(extraFloat(at), extraFloat(at + 1u), extraFloat(at + 2u));
}

fn grassHash(cell: vec2<i32>) -> u32 {
  return pcgHash((bitcast<u32>(cell.x) * 1597334677u) ^ pcgHash(bitcast<u32>(cell.y) * 3812015801u));
}

/** Smooth value noise in zero to one, for what changes across the lawn rather than from blade to blade. */
fn grassNoise(p: vec2<f32>) -> f32 {
  let cell = floor(p);
  let f = p - cell;
  let s = f * f * (3.0 - 2.0 * f);
  let c = vec2<i32>(cell);
  let corner = vec4<f32>(
    f32(grassHash(c) >> 8u), f32(grassHash(c + vec2<i32>(1, 0)) >> 8u),
    f32(grassHash(c + vec2<i32>(0, 1)) >> 8u), f32(grassHash(c + vec2<i32>(1, 1)) >> 8u)) / 16777216.0;

  return mix(mix(corner.x, corner.y, s.x), mix(corner.z, corner.w, s.x), s.y);
}

/** Whether a blade grows where its root stands: the lawn thins out in patches, and each blade has its own number. */
fn grassKeeps(record: u32, root: vec2<f32>, rand: f32) -> bool {
  let patchiness = extraFloat(record + 22u);

  if (patchiness <= 0.0) {
    return true;
  }

  let thin = smoothstep(0.3, 0.75, grassNoise(root * extraFloat(record + 23u) + vec2<f32>(31.7, 11.3)));

  return rand >= patchiness * thin;
}

/**
 * The closest blade of one patch along the ray, between near and the hit so far: origin and direction already in the
 * patch's space. Records the triangle in best and returns whether it found one.
 */
fn tracePatch(record: u32, root: u32, tile: vec2<f32>, origin: vec3<f32>, direction: vec3<f32>, near: f32,
  far: f32, best: ptr<function, GrassHit>) -> bool {
  let triangles = textures[record + 3u];
  let blades = textures[record + 4u];
  let signs = select(vec3<f32>(-1.0), vec3<f32>(1.0), direction >= vec3<f32>(0.0));
  let inverse = signs / max(abs(direction), vec3<f32>(1e-20));
  var closest = far;
  var found = false;
  var stack: array<u32, 32>;
  var depth = 0u;
  var current = 0u;

  loop {
    let node = grassNode(root + current * 16u);
    let leftEnter = enterBox(node.leftLow, node.leftHigh, origin, inverse, closest);
    let rightEnter = enterBox(node.rightLow, node.rightHigh, origin, inverse, closest);
    var next = NO_NODE;

    for (var side = 0u; side < 2u; side = side + 1u) {
      let enter = select(leftEnter, rightEnter, side == 1u);
      let link = select(node.leftLink, node.rightLink, side == 1u);
      let count = select(node.leftCount, node.rightCount, side == 1u);

      if (enter >= 1e30) {
        continue;
      }

      if (link >= 0) {
        if (next == NO_NODE) {
          next = u32(link);
        } else if (depth < 32u) {
          stack[depth] = u32(link);
          depth = depth + 1u;
        }

        continue;
      }

      let start = u32(-1 - link);

      for (var i = 0u; i < u32(count); i = i + 1u) {
        let at = triangles + (start + i) * 16u;
        let a = grassVec3(at);
        let edge1 = grassVec3(at + 3u) - a;
        let edge2 = grassVec3(at + 6u) - a;
        let p = cross(direction, edge2);
        let determinant = dot(edge1, p);

        if (abs(determinant) < 1e-14) {
          continue;
        }

        let invDet = 1.0 / determinant;
        let offset = origin - a;
        let u = dot(offset, p) * invDet;
        let q = cross(offset, edge1);
        let v = dot(direction, q) * invDet;
        let t = dot(edge2, q) * invDet;

        if (u < 0.0 || v < 0.0 || u + v > 1.0 || t <= near || t >= closest) {
          continue;
        }

        let blade = blades + textures[at + 15u] * 4u;
        let rootAt = tile + vec2<f32>(extraFloat(blade), extraFloat(blade + 1u));

        if (!grassKeeps(record, rootAt, extraFloat(blade + 2u))) {
          continue;
        }

        closest = t;
        found = true;
        (*best).triangle = at;
        (*best).u = u;
        (*best).v = v;
      }
    }

    if (next == NO_NODE) {
      if (depth == 0u) {
        break;
      }

      depth = depth - 1u;
      next = stack[depth];
    }

    current = next;
  }

  if (found) {
    (*best).distance = closest;
  }

  return found;
}

/** Keeps the part of [range.x, range.y] where offset + rate t stays at or above zero. */
fn grassClip(range: vec2<f32>, offset: f32, rate: f32) -> vec2<f32> {
  if (abs(rate) < 1e-20) {
    return select(range, vec2<f32>(1.0, 0.0), offset < 0.0);
  }

  let at = -offset / rate;

  return select(vec2<f32>(range.x, min(range.y, at)), vec2<f32>(max(range.x, at), range.y), rate > 0.0);
}

/** The edge from p to q of an anticlockwise footprint, as the ray crosses it on the ground: inside is on the left. */
fn grassEdge(range: vec2<f32>, p: vec2<f32>, q: vec2<f32>, origin: vec2<f32>, direction: vec2<f32>) -> vec2<f32> {
  let e = q - p;

  return grassClip(range, e.x * (origin.y - p.y) - e.y * (origin.x - p.x), e.x * direction.y - e.y * direction.x);
}

/** The grass over one triangle, against the closest hit so far. */
fn grassInShell(entry: u32, origin: vec3<f32>, direction: vec3<f32>, best: ptr<function, GrassHit>) {
  let record = textures[entry];
  let a = vec2<f32>(extraFloat(entry + 1u), extraFloat(entry + 2u));
  let b = vec2<f32>(extraFloat(entry + 3u), extraFloat(entry + 4u));
  let c = vec2<f32>(extraFloat(entry + 5u), extraFloat(entry + 6u));
  let p0 = extraFloat(entry + 7u);
  let slope = vec2<f32>(extraFloat(entry + 8u), extraFloat(entry + 9u));
  let tile = extraFloat(record);
  let top = extraFloat(record + 1u);
  // The ray's height over the plane, at its start and per unit of distance.
  let h0 = origin.y - p0 - slope.x * origin.x - slope.y * origin.z;
  let dh = direction.y - slope.x * direction.x - slope.y * direction.z;
  let ground = origin.xz;
  let along = direction.xz;
  var range = vec2<f32>(RAY_EPSILON, (*best).distance);

  range = grassClip(range, h0, dh);
  range = grassClip(range, top - h0, -dh);
  range = grassEdge(range, a, b, ground, along);
  range = grassEdge(range, b, c, ground, along);
  range = grassEdge(range, c, a, ground, along);

  if (range.x >= range.y) {
    return;
  }

  // Amanatides and Woo over the tiles of the ground.
  var cell = floor((ground + along * range.x) / tile);
  let stepping = select(vec2<f32>(-1.0), vec2<f32>(1.0), along >= vec2<f32>(0.0));
  let span = abs(vec2<f32>(tile) / select(along, vec2<f32>(1e-20), abs(along) < vec2<f32>(1e-20)));
  var crossing = vec2<f32>(1e30);

  if (abs(along.x) >= 1e-20) {
    crossing.x = ((cell.x + select(0.0, 1.0, along.x > 0.0)) * tile - ground.x) / along.x;
  }

  if (abs(along.y) >= 1e-20) {
    crossing.y = ((cell.y + select(0.0, 1.0, along.y > 0.0)) * tile - ground.y) / along.y;
  }

  var start = range.x;
  let variants = textures[record + 2u];

  for (var i = 0u; i < GRASS_MAX_TILES; i = i + 1u) {
    let end = min(min(crossing.x, crossing.y), range.y);
    let corner = cell * tile;
    let variant = grassHash(vec2<i32>(cell)) % max(variants, 1u);
    let patchOrigin = vec3<f32>(origin.x - corner.x, h0, origin.z - corner.y);
    let patchDirection = vec3<f32>(direction.x, dh, direction.z);

    if (end > start
      && tracePatch(record, textures[record + 5u + variant], corner, patchOrigin, patchDirection, start, end, best)) {
      (*best).found = true;
      (*best).record = record;
      (*best).slope = slope;
      (*best).tile = corner;

      return;
    }

    if (end >= range.y) {
      return;
    }

    if (crossing.x < crossing.y) {
      cell.x = cell.x + stepping.x;
      start = crossing.x;
      crossing.x = crossing.x + span.x;
    } else {
      cell.y = cell.y + stepping.y;
      start = crossing.y;
      crossing.y = crossing.y + span.y;
    }
  }
}

/** The closest blade along a ray before maxDistance, in any lawn of the scene. */
fn traceGrass(origin: vec3<f32>, direction: vec3<f32>, maxDistance: f32) -> GrassHit {
  var best = noGrass(maxDistance);

  if (frame.grassShells == GRASS_NONE) {
    return best;
  }

  let signs = select(vec3<f32>(-1.0), vec3<f32>(1.0), direction >= vec3<f32>(0.0));
  let inverse = signs / max(abs(direction), vec3<f32>(1e-20));
  var stack: array<u32, 32>;
  var enters: array<f32, 32>;
  var depth = 0u;
  var current = 0u;

  loop {
    if (current == NO_NODE) {
      if (depth == 0u) {
        break;
      }

      depth = depth - 1u;

      if (enters[depth] > best.distance) {
        continue;
      }

      current = stack[depth];
    }

    let node = grassNode(frame.grassShells + current * 16u);
    let leftEnter = enterBox(node.leftLow, node.leftHigh, origin, inverse, best.distance);
    let rightEnter = enterBox(node.rightLow, node.rightHigh, origin, inverse, best.distance);
    let leftFirst = leftEnter <= rightEnter;

    current = NO_NODE;

    for (var k = 0u; k < 2u; k = k + 1u) {
      // The nearer child first; a leaf is searched at once, a node is visited next or waits.
      let left = (k == 0u) == leftFirst;
      let enter = select(rightEnter, leftEnter, left);
      let link = select(node.rightLink, node.leftLink, left);
      let count = select(node.rightCount, node.leftCount, left);

      if (enter >= best.distance) {
        continue;
      }

      if (link < 0) {
        let first = u32(-1 - link);

        for (var i = 0u; i < u32(count); i = i + 1u) {
          grassInShell(frame.grassEntries + (first + i) * 12u, origin, direction, &best);
        }
      } else if (current == NO_NODE) {
        current = u32(link);
      } else if (depth < 32u) {
        stack[depth] = u32(link);
        enters[depth] = enter;
        depth = depth + 1u;
      }
    }
  }

  return best;
}

/**
 * A blade as a surface: thin, waxy and translucent, a quarter of its light let through to the far side. The colour runs
 * from the root to the tip, changes from blade to blade and across the lawn, and some blades are dry.
 */
fn grassSurface(hit: GrassHit, direction: vec3<f32>) -> Surface {
  let at = hit.triangle;
  let record = hit.record;
  let w = 1.0 - hit.u - hit.v;
  let height = w * extraFloat(at + 9u) + hit.u * extraFloat(at + 10u) + hit.v * extraFloat(at + 11u);
  let side = w * extraFloat(at + 12u) + hit.u * extraFloat(at + 13u) + hit.v * extraFloat(at + 14u);
  let blade = textures[record + 4u] + textures[at + 15u] * 4u;
  let root = hit.tile + vec2<f32>(extraFloat(blade), extraFloat(blade + 1u));
  let rand = extraFloat(blade + 2u);
  let yaw = extraFloat(blade + 3u);
  let a = grassVec3(at);
  let local = cross(grassVec3(at + 3u) - a, grassVec3(at + 6u) - a);
  // Normals go back through the inverse transpose of the shear.
  var normal = normalize(vec3<f32>(local.x - hit.slope.x * local.y, local.y, local.z - hit.slope.y * local.y));

  if (dot(normal, direction) > 0.0) {
    normal = -normal;
  }

  let across = normalize(vec3<f32>(cos(yaw), hit.slope.x * cos(yaw) + hit.slope.y * sin(yaw), sin(yaw)));
  // A blade is rounded across: its normal turns towards each edge.
  var shading = normalize(normal + across * (side * 0.55));

  if (dot(shading, direction) >= 0.0) {
    shading = normal;
  }

  let variation = extraFloat(record + 21u);
  let scale = extraFloat(record + 23u);
  let wide = grassNoise(root * scale);
  let fine = grassNoise(root * scale * 4.3 + vec2<f32>(5.2, 1.7));
  var albedo = mix(grassVec3(record + 9u), grassVec3(record + 12u), smoothstep(0.0, 1.0, pow(height, 0.75)));

  // Across the lawn, greener and yellower stretches; from blade to blade, lighter and darker.
  albedo = albedo * mix(vec3<f32>(1.0), vec3<f32>(1.12, 1.0, 0.7), variation * (wide - 0.4));
  albedo = albedo * (1.0 + variation * 0.6 * (fract(rand * 13.37) - 0.5));

  let dryness = extraFloat(record + 18u);
  let dry = grassVec3(record + 15u) * (0.85 + 0.3 * fract(rand * 7.77));

  if (fract(rand * 31.1) < dryness * (0.4 + 1.2 * fine)) {
    albedo = mix(dry * 0.8, dry, height);
  } else {
    // Tips dry first.
    albedo = mix(albedo, dry, smoothstep(0.82, 1.0, height) * dryness * 1.5);
  }

  // The sheath at the root is paler and duller.
  albedo = mix(albedo * 0.7 + vec3<f32>(0.03, 0.03, 0.01), albedo, smoothstep(0.0, 0.2, height));
  albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(0.9));

  let translucency = extraFloat(record + 19u);
  var surface: Surface;

  surface.material = 0u;
  surface.flags = MATERIAL_LIT | MATERIAL_PBR;
  surface.kind = KIND_STANDARD;
  surface.blending = BLEND_NORMAL;
  surface.facing = normal;
  surface.front = true;
  surface.normal = shading;
  surface.albedo = albedo * (1.0 - 0.5 * translucency);
  // Light through a leaf is more saturated and yellower than off it.
  surface.translucent = min(albedo * translucency * vec3<f32>(1.05, 1.2, 0.6), vec3<f32>(0.9));
  surface.opacity = 1.0;
  surface.alpha = 1.0;
  surface.roughness = clamp(extraFloat(record + 20u), 0.0525, 1.0);
  surface.metalness = 0.0;
  surface.ior = 1.5;
  surface.specularColor = vec3<f32>(0.04);
  surface.specularF90 = 1.0;
  surface.specularStrength = 1.0;
  surface.envMap = NO_TEXTURE;
  surface.envMapIntensity = 1.0;
  surface.gradientMap = NO_TEXTURE;

  return surface;
}
`;

/** What the kernel calls when no material grows grass: the same names, doing nothing. */
export const GRASS_STUB_WGSL = /* wgsl */ `
const GRASS_NONE: u32 = 0xffffffffu;
const GRASS_TRIANGLE: u32 = 0xfffffffeu;

struct GrassHit {
  found: bool,
  distance: f32,
};

fn traceGrass(origin: vec3<f32>, direction: vec3<f32>, maxDistance: f32) -> GrassHit {
  var hit: GrassHit;
  hit.found = false;
  hit.distance = maxDistance;

  return hit;
}

fn grassSurface(hit: GrassHit, direction: vec3<f32>) -> Surface {
  var surface: Surface;

  return surface;
}
`;
