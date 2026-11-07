import { buildBvh, NODE_WORDS } from "./bvh.js";
import type { PackedGeometry } from "./geometry.js";
import { linearAlbedo } from "./subsurface.js";
import type { SerializedMaterial } from "./upstream.js";

/**
 * A lawn grown on a material's surface, from `userData.cpuRenderer.grass`: only the path tracer draws it, and nothing
 * of it is geometry the host holds. `true` takes every default.
 *
 * The blades are real triangles, in a periodic patch of lawn a few metres across that tiles the world on the horizontal
 * plane and follows each triangle's slope by shearing, so the lawn costs the same whatever the size of the surface.
 */
export type GrassInput = boolean | {
  /** The tallest blade, in scene units. Defaults to `0.09`. */
  height?: number;
  /** Blades per square scene unit. Defaults to `6000`, a full lawn. */
  density?: number;
  /** The widest a blade gets, in scene units. Defaults to `0.005`. */
  width?: number;
  /** How far blades lean and curl over, `0` (upright) to `1`. Defaults to `0.35`. */
  bend?: number;
  /** Colour at the root, `#rrggbb` or linear rgb. */
  color?: string | readonly number[];
  /** Colour at the tip. */
  tipColor?: string | readonly number[];
  /** Colour of the dry blades. */
  dryColor?: string | readonly number[];
  /** Share of dry blades, `0` to `1`. Defaults to `0.12`. */
  dryness?: number;
  /** How much the colour changes from blade to blade and across the lawn, `0` to `1`. Defaults to `0.35`. */
  variation?: number;
  /** How much the lawn thins out in patches, `0` to `1`. Defaults to `0.2`. */
  patchiness?: number;
  /** Share of the light a blade lets through to its far side, `0` to `1`. Defaults to `0.45`. */
  translucency?: number;
  /** Roughness of the blade's waxy surface. Defaults to `0.5`. */
  roughness?: number;
  /** The steepest slope, in degrees, that still grows grass. Defaults to `50`. */
  maxSlope?: number;
  /** Changes the arrangement of the blades. */
  seed?: number;
};

/** A lawn resolved to its defaults. */
export interface GrassField {
  height: number;
  density: number;
  width: number;
  bend: number;
  color: [number, number, number];
  tipColor: [number, number, number];
  dryColor: [number, number, number];
  dryness: number;
  variation: number;
  patchiness: number;
  translucency: number;
  roughness: number;
  maxSlope: number;
  seed: number;
}

/** A snapshot material that may carry a lawn, as `serializeLive` attaches it. */
export type GrassMaterial = SerializedMaterial & { grass?: GrassInput };

const DEFAULT_COLOR = "#365a1c";
const DEFAULT_TIP = "#739a42";
const DEFAULT_DRY = "#a8975a";

/** Blades one patch may hold, so a dense lawn cannot take the whole device. */
const MAX_PATCH_BLADES = 24000;

/** Different patches the tiles pick from, which keeps the repetition out of sight. */
const VARIANTS = 4;

const finite = (value: unknown, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

const unit = (value: unknown, fallback: number) => Math.min(1, Math.max(0, finite(value, fallback)));

/** The lawn an input describes, with its defaults, or `null` when it describes none. */
export function grassField(input: unknown): GrassField | null {
  if (input === true) input = {};
  if (!input || typeof input !== "object") return null;

  const value = input as Exclude<GrassInput, boolean>;
  const height = finite(value.height, 0.09);
  const density = finite(value.density, 6000);

  if (!(height > 0) || !(density > 0)) return null;

  return {
    height,
    density,
    width: Math.max(1e-4, finite(value.width, 0.005)),
    bend: unit(value.bend, 0.35),
    color: linearAlbedo(value.color ?? DEFAULT_COLOR),
    tipColor: linearAlbedo(value.tipColor ?? DEFAULT_TIP),
    dryColor: linearAlbedo(value.dryColor ?? DEFAULT_DRY),
    dryness: unit(value.dryness, 0.12),
    variation: unit(value.variation, 0.35),
    patchiness: unit(value.patchiness, 0.2),
    translucency: unit(value.translucency, 0.45),
    roughness: Math.min(1, Math.max(0.05, finite(value.roughness, 0.5))),
    maxSlope: Math.min(89, Math.max(0, finite(value.maxSlope, 50))),
    seed: Math.round(finite(value.seed, 0)),
  };
}

export function grassOf(material: SerializedMaterial | undefined): GrassField | null {
  return grassField((material as GrassMaterial | undefined)?.grass);
}

/** What the frame uniform says when no triangle grows grass. */
export const GRASS_NONE = 0xffffffff;

/** Words the kernel reads per patch triangle: its corners, the height along the blade and the side of each, the blade. */
export const GRASS_TRIANGLE_WORDS = 16;
/** Words per blade: its root, a random number of its own and the angle its width runs at. */
export const GRASS_BLADE_WORDS = 4;
/** Words per lawn: see writeRecord. */
export const GRASS_RECORD_WORDS = 32;
/** Words per triangle that grows grass: see writeEntry. */
export const GRASS_ENTRY_WORDS = 12;

/** One patch of lawn, every variant of it: triangles in patch space, with x and z across the tile and y up. */
export interface GrassPatch {
  tile: number;
  /** The tallest point of any blade. */
  top: number;
  /** Sixteen words a triangle, every variant one after the other in its own hierarchy's leaf order. */
  triangles: Uint32Array;
  blades: Uint32Array;
  /** Each variant's hierarchy, its leaves pointing into `triangles`. */
  nodes: Uint32Array[];
}

/** Mulberry32: small, fast and the same everywhere, so a lawn grows the same way on every machine. */
function random(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A normally distributed number, by Box and Muller. */
const gaussian = (next: () => number) => Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());

interface Blade {
  x: number;
  z: number;
  rand: number;
  yaw: number;
  /** Corners in patch space relative to the root, and per corner the height along the blade and the side. */
  corners: number[];
  heights: number[];
  sides: number[];
  /** Three corner indices a triangle. */
  faces: number[];
  reach: number;
}

/** How far a blade's tip can get from its root, which the border shared by every variant has to cover. */
function bladeReach(field: GrassField): number {
  return field.height * (0.15 + 1.1 * field.bend) + field.width * 2;
}

/**
 * One blade: a spine curving away from its root as a quadratic, tapering to a point, with a slight twist. It leans away
 * from the middle of its clump, as grass grows.
 */
function makeBlade(field: GrassField, next: () => number, x: number, z: number, lean: [number, number]): Blade {
  const height = field.height * (0.45 + 0.55 * Math.pow(next(), 0.6));
  const width = field.width * (0.7 + 0.6 * next());
  const amount = field.bend * (0.35 + 0.75 * next());
  const twist = (next() - 0.5) * 0.9;
  const facing = Math.atan2(lean[1], lean[0]) + Math.PI / 2 + (next() - 0.5) * 0.7;
  const segments = height > 0.18 ? 6 : 4;
  const corners: number[] = [];
  const heights: number[] = [];
  const sides: number[] = [];
  const faces: number[] = [];
  let reach = 0;

  const spine = (v: number) => {
    const out = amount * height * v * v;
    // A blade that bends over loses height, roughly as an arc of the same length would.
    const up = height * v * (1 - 0.4 * amount * v);

    return [lean[0] * out, up, lean[1] * out];
  };

  for (let k = 0; k < segments; k++) {
    const v = k / segments;
    const point = spine(v);
    // Widest a little above the root, closing to the tip.
    const half = width * 0.5 * Math.pow(Math.max(0, 1 - Math.pow(v, 2.2)), 0.6) * (0.8 + 0.2 * Math.min(1, v / 0.15));
    const angle = facing + twist * v;
    const across = [Math.cos(angle) * half, Math.sin(angle) * half];

    corners.push(point[0] - across[0], point[1], point[2] - across[1],
      point[0] + across[0], point[1], point[2] + across[1]);
    heights.push(v, v);
    sides.push(-1, 1);
    reach = Math.max(reach, Math.hypot(point[0], point[2]) + half);
  }

  const tip = spine(1);

  corners.push(tip[0], tip[1], tip[2]);
  heights.push(1);
  sides.push(0);
  reach = Math.max(reach, Math.hypot(tip[0], tip[2]));

  for (let k = 0; k < segments - 1; k++) {
    const l0 = 2 * k, r0 = 2 * k + 1, l1 = 2 * k + 2, r1 = 2 * k + 3;

    faces.push(l0, r0, r1, l0, r1, l1);
  }

  faces.push(2 * (segments - 1), 2 * (segments - 1) + 1, 2 * segments);

  return { x, z, rand: next(), yaw: facing + twist * 0.5, corners, heights, sides, faces, reach };
}

/**
 * Every blade of one arrangement over the tile: clumps on a jittered grid, each a handful of blades scattered around its
 * middle, wrapped around the tile so the arrangement repeats without a seam.
 */
function arrangement(field: GrassField, tile: number, seed: number, count: number): Blade[] {
  const next = random(seed);
  const spacing = Math.max(0.025, Math.min(0.12, field.height * 0.7));
  const cells = Math.max(1, Math.round(tile / spacing));
  const step = tile / cells;
  const perClump = count / (cells * cells);
  const blades: Blade[] = [];

  for (let i = 0; i < cells; i++) {
    for (let j = 0; j < cells; j++) {
      const cx = (i + next()) * step;
      const cz = (j + next()) * step;
      // A whole number of blades on average, with the fraction drawn.
      const whole = Math.floor(perClump);
      const n = whole + (next() < perClump - whole ? 1 : 0);

      for (let b = 0; b < n; b++) {
        const dx = gaussian(next) * step * 0.4;
        const dz = gaussian(next) * step * 0.4;
        const away = Math.hypot(dx, dz);
        const angle = next() * Math.PI * 2;
        // Away from the middle of the clump, with some of every direction mixed in.
        const mixX = (away > 1e-6 ? dx / away : 0) * 0.7 + Math.cos(angle) * 0.5;
        const mixZ = (away > 1e-6 ? dz / away : 0) * 0.7 + Math.sin(angle) * 0.5;
        const length = Math.hypot(mixX, mixZ) || 1;
        const x = ((cx + dx) % tile + tile) % tile;
        const z = ((cz + dz) % tile + tile) % tile;

        blades.push(makeBlade(field, next, x, z, [mixX / length, mixZ / length]));
      }
    }
  }

  return blades;
}

/** The patches already built, by the parameters that shape them: a scene packed again reuses them. */
const patches = new Map<string, GrassPatch>();

/**
 * The patch of a lawn. The band along the tile's border, as wide as a blade reaches, is the same in every variant and
 * wraps around, so any variant meets any other without a seam; inside it each variant grows its own blades.
 */
export function grassPatch(field: GrassField): GrassPatch {
  const key = JSON.stringify([field.height, field.density, field.width, field.bend, field.seed]);
  const cached = patches.get(key);

  if (cached) return cached;

  const reach = bladeReach(field);
  const tile = Math.max(1, reach * 4);
  const count = Math.min(MAX_PATCH_BLADES, Math.round(field.density * tile * tile));
  const inBorder = (blade: Blade) => Math.min(blade.x, tile - blade.x, blade.z, tile - blade.z) < reach;
  const border = arrangement(field, tile, field.seed * 7919 + 1, count).filter(inBorder);
  const triangles: number[] = [];
  const blades: number[] = [];
  const floats = new Float32Array(1);
  const words = new Uint32Array(floats.buffer);
  const word = (value: number) => (floats[0] = value, words[0]);
  const nodes: Uint32Array[] = [];
  let top = 0;

  for (let variant = 0; variant < VARIANTS; variant++) {
    const inner = arrangement(field, tile, field.seed * 7919 + 101 + variant * 31, count)
      .filter(blade => !inBorder(blade));

    // A blade over the border also shows, shifted by a whole tile, in the tiles on the other side of it.
    const placed: { blade: Blade; ox: number; oz: number }[] = [];

    for (const blade of [...border, ...inner]) {
      for (const ox of [-tile, 0, tile]) {
        for (const oz of [-tile, 0, tile]) {
          const x = blade.x + ox, z = blade.z + oz;

          if (x + blade.reach < 0 || x - blade.reach > tile || z + blade.reach < 0 || z - blade.reach > tile) continue;

          placed.push({ blade, ox, oz });
        }
      }
    }

    const positions: number[] = [];
    const records: number[][] = [];

    for (const { blade, ox, oz } of placed) {
      const index = blades.length / GRASS_BLADE_WORDS;
      const x = blade.x + ox, z = blade.z + oz;

      blades.push(word(x), word(z), word(blade.rand), word(blade.yaw));

      for (let f = 0; f < blade.faces.length; f += 3) {
        const record: number[] = [];

        for (let c = 0; c < 3; c++) {
          const corner = blade.faces[f + c];
          const px = blade.corners[corner * 3] + x;
          const py = blade.corners[corner * 3 + 1];
          const pz = blade.corners[corner * 3 + 2] + z;

          positions.push(px, py, pz);
          record.push(px, py, pz);
          top = Math.max(top, py);
        }

        for (let c = 0; c < 3; c++) record.push(blade.heights[blade.faces[f + c]]);
        for (let c = 0; c < 3; c++) record.push(blade.sides[blade.faces[f + c]]);

        records.push([...record, index]);
      }
    }

    const bvh = buildBvh(new Float32Array(positions), records.length);
    const start = triangles.length / GRASS_TRIANGLE_WORDS;

    for (let at = 0; at < bvh.order.length; at++) {
      const record = records[bvh.order[at]];

      for (let k = 0; k < 15; k++) triangles.push(word(record[k]));
      triangles.push(record[15]);
    }

    nodes.push(offsetLeaves(bvh.nodes, start));
  }

  const patch: GrassPatch = {
    tile,
    top,
    triangles: new Uint32Array(triangles),
    blades: new Uint32Array(blades),
    nodes,
  };

  patches.set(key, patch);

  return patch;
}

/** A hierarchy's nodes with every leaf moved along by `start` entries, so several can share one list. */
function offsetLeaves(source: ArrayBuffer, start: number): Uint32Array {
  const nodes = new Int32Array(source.slice(0));

  for (let at = 0; at < nodes.length; at += NODE_WORDS) {
    for (const link of [3, 11]) {
      if (nodes[at + link] < 0 && nodes[at + link + 4] > 0) nodes[at + link] = -1 - (-1 - nodes[at + link] + start);
    }
  }

  return new Uint32Array(nodes.buffer);
}

/** What packGrass adds to the texture buffer's extra words, and where its root is. */
export interface PackedGrass {
  words: Uint32Array;
  /** Where the hierarchy of the triangles that grow grass starts, in the texture buffer. */
  shells: number;
  /** Where their entries start. */
  entries: number;
  /** How many triangles grow grass. */
  count: number;
}

/**
 * The lawns of a packed scene, laid out to sit at `base` in the texture buffer: each lawn's record, patch triangles,
 * blades and hierarchies, then one entry per triangle that grows grass and the hierarchy over those triangles. `null`
 * when nothing grows any.
 */
export function packGrass(geometry: PackedGeometry, base: number): PackedGrass | null {
  const fields = geometry.materialProperties.map(properties => properties.grass ?? null);

  if (!fields.some(Boolean)) return null;

  const words: number[] = [];
  const floats = new Float32Array(1);
  const view = new Uint32Array(floats.buffer);
  const word = (value: number) => (floats[0] = value, view[0]);
  const records = new Map<number, { at: number; field: GrassField; patch: GrassPatch }>();

  fields.forEach((field, material) => {
    if (!field) return;

    const patch = grassPatch(field);
    const at = base + words.length;
    const trianglesAt = at + GRASS_RECORD_WORDS;
    const bladesAt = trianglesAt + patch.triangles.length;
    let nodesAt = bladesAt + patch.blades.length;
    const roots: number[] = [];

    for (const nodes of patch.nodes) {
      roots.push(nodesAt);
      nodesAt += nodes.length;
    }

    const record = new Array<number>(GRASS_RECORD_WORDS).fill(0);

    record[0] = word(patch.tile);
    record[1] = word(patch.top);
    record[2] = patch.nodes.length;
    record[3] = trianglesAt;
    record[4] = bladesAt;
    roots.forEach((root, k) => (record[5 + k] = root));
    field.color.forEach((channel, k) => (record[9 + k] = word(channel)));
    field.tipColor.forEach((channel, k) => (record[12 + k] = word(channel)));
    field.dryColor.forEach((channel, k) => (record[15 + k] = word(channel)));
    record[18] = word(field.dryness);
    record[19] = word(field.translucency);
    record[20] = word(field.roughness);
    record[21] = word(field.variation);
    record[22] = word(field.patchiness);
    // The size of the patches the lawn thins out in and changes colour over, a few metres, larger for taller grass.
    record[23] = word(1 / Math.max(1.5, field.height * 30));

    words.push(...record);
    pushAll(words, patch.triangles);
    pushAll(words, patch.blades);
    for (const nodes of patch.nodes) pushAll(words, nodes);

    records.set(material, { at, field, patch });
  });

  const entries: number[][] = [];
  const boxes: number[] = [];
  const p = geometry.positions;
  const n = geometry.normals;

  for (let triangle = 0; triangle < geometry.triangleCount; triangle++) {
    const lawn = records.get(geometry.materials[triangle]);

    if (!lawn) continue;

    const at = triangle * 9;
    const e1 = [p[at + 3] - p[at], p[at + 4] - p[at + 1], p[at + 5] - p[at + 2]];
    const e2 = [p[at + 6] - p[at], p[at + 7] - p[at + 1], p[at + 8] - p[at + 2]];
    const geometric = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const length = Math.hypot(geometric[0], geometric[1], geometric[2]);

    if (length < 1e-12) continue;

    // The side the surface faces: its vertex normals where it has them, which survive a winding turned around.
    const [v0, v1, v2] = [0, 1, 2].map(corner => geometry.corners[triangle * 3 + corner] * 3);
    const authored = [0, 1, 2].map(axis => n[v0 + axis] + n[v1 + axis] + n[v2 + axis]);
    const facing = Math.hypot(authored[0], authored[1], authored[2]) > 1e-6 ? authored[1] : geometric[1];
    const up = Math.abs(geometric[1]) / length;

    if (facing <= 0 || up < Math.cos(lawn.field.maxSlope * Math.PI / 180)) continue;

    // The plane as a height over the ground: y = p0 + sx x + sz z.
    const sx = -geometric[0] / geometric[1];
    const sz = -geometric[2] / geometric[1];
    const p0 = p[at + 1] - sx * p[at] - sz * p[at + 2];
    const a = [p[at], p[at + 2]];
    let b = [p[at + 3], p[at + 5]], c = [p[at + 6], p[at + 8]];

    // Anticlockwise on the ground, which the kernel's edge test assumes.
    if ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0) [b, c] = [c, b];

    entries.push([lawn.at, a[0], a[1], b[0], b[1], c[0], c[1], p0, sx, sz]);

    const low = [Math.min(a[0], b[0], c[0]), Math.min(p[at + 1], p[at + 4], p[at + 7]), Math.min(a[1], b[1], c[1])];

    const high = [Math.max(a[0], b[0], c[0]), Math.max(p[at + 1], p[at + 4], p[at + 7]) + lawn.patch.top,
      Math.max(a[1], b[1], c[1])];

    const middle = low.map((value, axis) => (value + high[axis]) / 2);

    // A stand-in triangle with the shell's box as its bounds and its middle as its centroid.
    boxes.push(...low, ...high, ...middle);
  }

  if (!entries.length) return null;

  const bvh = buildBvh(new Float32Array(boxes), entries.length);
  const entriesAt = base + words.length;

  for (let at = 0; at < bvh.order.length; at++) {
    const entry = entries[bvh.order[at]];

    words.push(entry[0], ...entry.slice(1).map(word), 0, 0);
  }

  const shells = base + words.length;

  pushAll(words, new Uint32Array(bvh.nodes));

  return { words: new Uint32Array(words), shells, entries: entriesAt, count: entries.length };
}

/** Appends a typed array without spreading it, which a large patch would overflow the stack with. */
function pushAll(target: number[], source: Uint32Array) {
  for (let i = 0; i < source.length; i++) target.push(source[i]);
}
