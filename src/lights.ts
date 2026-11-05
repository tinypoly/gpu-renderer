import type { PackedGeometry } from "./geometry.js";
import { buildLightTree, type TreeLight } from "./lightTree.js";
import type { SerializedLight } from "./upstream.js";

/**
 * What the integrator can aim a shadow ray at. Emissive triangles and Three's analytic lights share one buffer and one
 * cumulative distribution, so picking a light is a single binary search however the scene is lit.
 *
 * The emissive triangles come first, in triangle order, which lets the same buffer answer the other question the
 * integrator asks: given a triangle a bounce ray happened to hit, how likely was direct sampling to have found it. That
 * is a second binary search over the same prefix.
 */
export const LIGHT_KIND = {
  /** An emissive triangle of the scene's geometry. */
  area: 0,
  directional: 1,
  point: 2,
  spot: 3,
  /** A `RectAreaLight`: a rectangle that is not part of the geometry, so no bounce ray can hit it. */
  rect: 4,
  /** The surrounding image, sampled where it is brightest. */
  environment: 5,
} as const;

/** Twenty four words per light, as `Light` in `wgsl/common.wgsl.ts`. */
export const LIGHT_WORDS = 24;

/** Set when the light's own `castShadow` is on, which is what gates the occlusion ray. */
export const LIGHT_CASTS_SHADOW = 1;
/** A spot light with a projected map; its triangle word says where the map's record is in the texture buffer. */
export const LIGHT_HAS_MAP = 2;

/** Where the lights' own records go, and how their textures are found. */
export interface LightPackOptions {
  /** The index in the texture table of each texture, by id. */
  textures?: ReadonlyMap<string, number>;
  /** Where the words these lights append will start in the texture buffer. */
  extraBase?: number;
}

export interface PackedLights {
  /** `LIGHT_WORDS` words per light: see `Light` in `wgsl/common.wgsl.ts`. */
  entries: ArrayBuffer;
  count: number;
  /** How many leading entries are emissive triangles. The rest are analytic and cannot be hit by a bounce ray. */
  areaCount: number;
  /**
   * Radiance an `AmbientLight` or a `HemisphereLight` adds from every direction, which a path tracer can only honestly
   * express as light arriving from outside the scene. `sky` applies along `axis`, `ground` against it.
   */
  ambient: { sky: number[]; ground: number[]; axis: number[] };
  /** How often direct sampling picks the environment rather than a light. Zero when there is nothing to aim at. */
  environmentPick: number;
  /**
   * Words to append to the texture buffer: a spot light's map index and the matrix that projects it, then the
   * cumulative probability of every entry and the triangle of every emissive one, packed tight so the two searches the
   * shader makes over them read four bytes a step instead of a whole entry.
   */
  extraWords: Uint32Array;
  /** Where the cumulative probabilities start in the texture buffer, one float per entry. */
  cdfAt: number;
  /** Where the emissive entries' triangles start in the texture buffer, one word per emissive entry. */
  trianglesAt: number;
  /**
   * How many leading entries sit in the scene (emissive triangles, point, spot and rectangular lights), which the light
   * tree picks among; the directional lights and the environment follow them.
   */
  localCount: number;
  /** The chance of picking one of those: exactly the cumulative probability of the last of them. */
  localPick: number;
  /** Where the light tree's nodes start in the texture buffer, and the leaf of every local entry; zero for no tree. */
  treeAt: number;
  leavesAt: number;
}

const luminance = (color: ArrayLike<number>) => 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];

/** Twice the area of the triangle at `offset`. */
function doubleArea(positions: Float32Array, offset: number): number {
  const ax = positions[offset + 3] - positions[offset], ay = positions[offset + 4] - positions[offset + 1];
  const az = positions[offset + 5] - positions[offset + 2];
  const bx = positions[offset + 6] - positions[offset], by = positions[offset + 7] - positions[offset + 1];
  const bz = positions[offset + 8] - positions[offset + 2];

  return Math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
}

function normalize(vector: readonly number[]): number[] {
  const length = Math.hypot(vector[0], vector[1], vector[2]);

  return length > 0 ? [vector[0] / length, vector[1] / length, vector[2] / length] : [0, 1, 0];
}

const scale = (color: readonly number[], factor: number) => [color[0] * factor, color[1] * factor, color[2] * factor];

/**
 * Where the scene is and how big it is, which is what makes the lights comparable. Three states a directional light in
 * lux and a point light in candela, so their numbers cannot be weighed against each other directly; what can is the
 * irradiance each of them actually delivers to the scene.
 */
interface Bounds {
  centre: number[];
  radius: number;
}

function sceneBounds(positions: Float32Array): Bounds {
  if (!positions.length) return { centre: [0, 0, 0], radius: 1 };

  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];

  for (let at = 0; at < positions.length; at += 3) {
    for (let axis = 0; axis < 3; axis++) {
      low[axis] = Math.min(low[axis], positions[at + axis]);
      high[axis] = Math.max(high[axis], positions[at + axis]);
    }
  }

  const centre = [0, 1, 2].map(axis => (low[axis] + high[axis]) / 2);
  const radius = Math.hypot(...[0, 1, 2].map(axis => (high[axis] - low[axis]) / 2));

  return { centre, radius: radius > 0 ? radius : 1 };
}

/**
 * How far a light is from the scene, never closer than a quarter of its radius. A light sitting exactly at the centre
 * would otherwise be credited with an unbounded irradiance and swallow every sample.
 */
function reach(position: readonly number[], bounds: Bounds): number {
  const distance = Math.hypot(...[0, 1, 2].map(axis => position[axis] - bounds.centre[axis]));

  return Math.max(distance, bounds.radius * 0.25);
}

/** A candidate before the cumulative distribution is laid over it: the words it fills, and the power it radiates. */
interface Candidate {
  kind: number;
  weight: number;
  write(words: Uint32Array, floats: Float32Array, at: number): void;
  /** Area measure density of a point on it, before the pick probability is folded in. */
  density?: number;
  /** What the light tree knows of a light that sits in the scene; none for a directional light or the environment. */
  tree?: TreeLight;
}

const corners = (points: number[][]) => ({
  low: [0, 1, 2].map(k => Math.min(...points.map(p => p[k]))),
  high: [0, 1, 2].map(k => Math.max(...points.map(p => p[k]))),
});

/**
 * Emissive triangles weighted by the power they radiate (area times emitted luminance), so a large or bright panel is
 * chosen more often than a dim sliver.
 */
function emissiveTriangles(geometry: PackedGeometry, bounds: Bounds): Candidate[] {
  const candidates: Candidate[] = [];

  for (let triangle = 0; triangle < geometry.triangleCount; triangle++) {
    const material = geometry.materials[triangle] * 4;
    const emissive = geometry.materialEmissive;

    if (material + 3 >= emissive.length) continue;

    const glow = luminance(emissive.subarray(material, material + 3)) * emissive[material + 3];
    const area = doubleArea(geometry.positions, triangle * 9) / 2;

    if (!(glow > 0) || !(area > 0)) continue;

    const at = triangle * 9;

    const centroid = [0, 1, 2].map(axis =>
      (geometry.positions[at + axis] + geometry.positions[at + 3 + axis] + geometry.positions[at + 6 + axis]) / 3);

    const p = geometry.positions;
    const e1 = [p[at + 3] - p[at], p[at + 4] - p[at + 1], p[at + 5] - p[at + 2]];
    const e2 = [p[at + 6] - p[at], p[at + 7] - p[at + 1], p[at + 8] - p[at + 2]];

    candidates.push({
      kind: LIGHT_KIND.area,
      // The irradiance it delivers, so a bright panel and a bright lamp are weighed in the same unit.
      weight: glow * area / reach(centroid, bounds) ** 2,
      density: 1 / area,
      // It radiates from both sides, so its cone is the whole sphere.
      tree: {
        ...corners([0, 3, 6].map(v => [p[at + v], p[at + v + 1], p[at + v + 2]])),
        power: glow * area,
        axis: normalize([e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]),
        thetaO: Math.PI,
        thetaE: Math.PI / 2,
        density: 1 / area,
      },
      write(words, _floats, at) {
        words[at] = LIGHT_KIND.area;
        words[at + 3] = triangle;
        // The emission itself is read from the material, so the colour words stay zero.
        words[at + 23] = LIGHT_CASTS_SHADOW;
      },
    });
  }

  return candidates;
}

/**
 * Three's analytic lights. The falloff, the cone and the rectangle's projected area follow `material.js` in the CPU
 * renderer, so direct light from these agrees with it.
 */
function analyticLight(
  light: SerializedLight, bounds: Bounds, options: LightPackOptions, extra: number[]): Candidate | null {
  const intensity = light.intensity;

  if (!(intensity > 0)) return null;

  const color = scale(light.color, intensity);
  const emitted = luminance(color);

  if (!(emitted > 0)) return null;

  // A directional light's intensity already is an irradiance; the others have to be carried to the scene first.
  const distance = reach(light.position, bounds);
  const falloff = 1 / Math.max(distance ** (light.decay || 0), 0.01);

  // Three points a directional or spot light from its target towards its position, and so does the CPU renderer.
  const axis = normalize([
    light.position[0] - light.target[0],
    light.position[1] - light.target[1],
    light.position[2] - light.target[2]]);

  // Three ignores `castShadow` on a RectAreaLight, but an area light that casts none looks broken once it is traced.
  const flags = light.castShadow || light.type === "RectAreaLight" ? LIGHT_CASTS_SHADOW : 0;

  const common = (words: Uint32Array, floats: Float32Array, at: number, kind: number) => {
    words[at] = kind;
    floats.set(color, at + 4);
    floats[at + 7] = light.distance;
    floats.set(light.position, at + 8);
    floats[at + 11] = light.decay;
    floats.set(axis, at + 12);
    words[at + 23] = flags;
  };

  if (light.type === "DirectionalLight") {
    return {
      kind: LIGHT_KIND.directional,
      weight: emitted,
      write: (words, floats, at) => common(words, floats, at, LIGHT_KIND.directional),
    };
  }

  if (light.type === "PointLight") {
    return {
      kind: LIGHT_KIND.point,
      weight: emitted * falloff,
      tree: { low: [...light.position], high: [...light.position], power: emitted, axis: [0, 0, 1], thetaO: Math.PI,
        thetaE: Math.PI / 2, density: 1 },
      write: (words, floats, at) => common(words, floats, at, LIGHT_KIND.point),
    };
  }

  if (light.type === "SpotLight") {
    const outer = Math.cos(light.angle);
    const map = light.map !== undefined ? options.textures?.get(light.map) : undefined;
    let record = -1;

    // Three's spotLightMap: the texture and the matrix that takes a world position to its UV.
    if (map !== undefined && light.mapMatrix?.length === 16) {
      record = (options.extraBase ?? 0) + extra.length;
      extra.push(map, ...new Uint32Array(new Float32Array(light.mapMatrix).buffer));
    }

    return {
      kind: LIGHT_KIND.spot,
      // Only the cone is lit, so only that share of the scene can expect anything from it.
      weight: emitted * falloff * (1 - Math.cos(Math.min(light.angle, Math.PI))) / 2,
      // It shines from its position towards its target, within its outer cone.
      tree: { low: [...light.position], high: [...light.position], power: emitted, axis: scale(axis, -1),
        thetaO: Math.min(light.angle, Math.PI), thetaE: 0, density: 1 },
      write(words, floats, at) {
        common(words, floats, at, LIGHT_KIND.spot);
        floats[at + 15] = outer;
        floats[at + 19] = Math.cos(light.angle * (1 - light.penumbra));

        if (record >= 0) {
          words[at + 3] = record;
          words[at + 23] |= LIGHT_HAS_MAP;
        }
      },
    };
  }

  if (light.type === "RectAreaLight") {
    const matrix = light.matrixWorld;
    // The light's own axes, from its world matrix: Three lights the rectangle along its local -z.
    const right = normalize([matrix[0], matrix[1], matrix[2]]);
    const up = normalize([matrix[4], matrix[5], matrix[6]]);
    const area = Math.max(light.width * light.height, 1e-12);

    const half = [scale(right, light.width / 2), scale(up, light.height / 2)];

    const facing = normalize([
      right[1] * up[2] - right[2] * up[1], right[2] * up[0] - right[0] * up[2], right[0] * up[1] - right[1] * up[0]]);

    return {
      kind: LIGHT_KIND.rect,
      weight: emitted * area / distance ** 2,
      density: 1 / area,
      // One sided: it lights the side its axes' cross product turns away from, over a hemisphere.
      tree: {
        ...corners([[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) =>
          [0, 1, 2].map(k => light.position[k] + half[0][k] * a + half[1][k] * b))),
        power: emitted * area,
        axis: scale(facing, -1),
        thetaO: 0,
        thetaE: Math.PI / 2,
        density: 1 / area,
      },
      write(words, floats, at) {
        common(words, floats, at, LIGHT_KIND.rect);
        floats.set(scale(right, light.width / 2), at + 16);
        floats.set(scale(up, light.height / 2), at + 20);
      },
    };
  }

  return null;
}

/**
 * The irradiance an `AmbientLight` or a `HemisphereLight` adds, turned into the radiance of an equally bright
 * surrounding. Three adds these straight to a surface's diffuse term, which no path can carry; light that arrives from
 * everywhere is light from outside the scene, so that is where this renderer puts it. In a closed room it therefore
 * contributes nothing, which is the difference a host has to know about.
 */
function ambientLight(lights: readonly SerializedLight[]): PackedLights["ambient"] {
  const sky = [0, 0, 0], ground = [0, 0, 0];
  let axis: number[] | null = null;

  for (const light of lights) {
    if (!(light.intensity > 0)) continue;

    // A uniform surrounding of radiance L delivers an irradiance of pi times L.
    const strength = light.intensity / Math.PI;

    if (light.type === "AmbientLight") {
      for (let channel = 0; channel < 3; channel++) {
        sky[channel] += light.color[channel] * strength;
        ground[channel] += light.color[channel] * strength;
      }
    }

    if (light.type === "HemisphereLight") {
      // Three takes the axis from the light's position; several of them cannot share one gradient, so the first wins.
      axis ??= normalize(light.position);

      for (let channel = 0; channel < 3; channel++) {
        sky[channel] += light.color[channel] * strength;
        ground[channel] += light.groundColor[channel] * strength;
      }
    }
  }

  return { sky, ground, axis: axis ?? [0, 1, 0] };
}

/**
 * Everything the integrator may aim a shadow ray at, with the cumulative distribution that picks one. The weights mix
 * two unit systems, the radiant power of a surface and the intensity of an analytic light, so the balance between the
 * two kinds is a heuristic; it costs variance when a scene leans on both at once and never costs correctness, because
 * every contribution is divided by the probability it was actually chosen with.
 */
export function packLights(
  geometry: PackedGeometry,
  lights: readonly SerializedLight[] = [],
  environment?: { power: number } | null,
  options: LightPackOptions = {}): PackedLights {
  const bounds = sceneBounds(geometry.positions);
  const areas = emissiveTriangles(geometry, bounds);
  const extra: number[] = [];

  const analytic = lights
    .map(light => analyticLight(light, bounds, options, extra))
    .filter((light): light is Candidate => light !== null);

  const surrounding: Candidate[] = environment && environment.power > 0
    ? [{
      kind: LIGHT_KIND.environment,
      weight: environment.power,
      write(words, _floats, at) {
        words[at] = LIGHT_KIND.environment;
        // Nothing occludes the sky except the scene itself, which is exactly what the shadow ray tests.
        words[at + 23] = LIGHT_CASTS_SHADOW;
      },
    }]
    : [];

  // What sits in the scene first, so the tree's lights are the leading entries; then what lies outside it.
  const local = analytic.filter(candidate => candidate.tree);
  const distant = analytic.filter(candidate => !candidate.tree);
  const candidates = [...areas, ...local, ...distant, ...surrounding];
  const localCount = areas.length + local.length;
  const total = candidates.reduce((sum, candidate) => sum + candidate.weight, 0);
  const entries = new ArrayBuffer(Math.max(1, candidates.length) * LIGHT_WORDS * 4);
  const words = new Uint32Array(entries);
  const floats = new Float32Array(entries);
  let cumulative = 0;

  candidates.forEach((candidate, index) => {
    const at = index * LIGHT_WORDS;
    const probability = candidate.weight / total;

    cumulative += probability;
    candidate.write(words, floats, at);
    // The last entry closes the range exactly, so a unit sample never falls past it.
    floats[at + 1] = index === candidates.length - 1 ? 1 : cumulative;
    floats[at + 2] = probability * (candidate.density ?? 1);
  });

  const environmentPick = surrounding.length ? surrounding[0].weight / total : 0;
  const base = options.extraBase ?? 0;
  const cdfAt = base + extra.length;

  for (let index = 0; index < candidates.length; index++) extra.push(words[index * LIGHT_WORDS + 1]);

  const trianglesAt = base + extra.length;

  for (let index = 0; index < areas.length; index++) extra.push(words[index * LIGHT_WORDS + 3]);

  let treeAt = 0, leavesAt = 0;

  if (localCount > 0) {
    const tree = buildLightTree(candidates.slice(0, localCount).map(candidate => candidate.tree as TreeLight));

    treeAt = base + extra.length;
    for (const word of tree.words) extra.push(word);
    leavesAt = base + extra.length;
    for (const leaf of tree.leafOf) extra.push(leaf);
  }

  return {
    localCount,
    localPick: localCount > 0 ? floats[(localCount - 1) * LIGHT_WORDS + 1] : 0,
    treeAt,
    leavesAt,
    cdfAt,
    trianglesAt,
    entries,
    count: candidates.length,
    areaCount: areas.length,
    ambient: ambientLight(lights),
    environmentPick,
    extraWords: new Uint32Array(extra),
  };
}
