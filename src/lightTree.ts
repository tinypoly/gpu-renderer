/**
 * A bounding hierarchy over the lights that sit in the scene (emissive triangles, point, spot and rectangular lights),
 * which lets the integrator pick one by how much it can deliver to the point being shaded rather than by its power
 * alone: Conty Estevez and Kulla, "Importance Sampling of Many Lights with Adaptive Tree Splitting" (2018), without the
 * splitting, as Cycles uses it. Every node keeps its box, the power under it and a cone bounding the directions its
 * lights face; the kernel walks down choosing a child by each one's importance from the shaded point, and walks the same
 * path up to tell how likely a light a bounce ray hit was to be chosen that way.
 */

/** A light as the tree sees it: where it is, how strong, which way it faces, and the area density of its points. */
export interface TreeLight {
  low: number[];
  high: number[];
  /** Radiant intensity scale: what a unit distance away receives, so lights of every kind compare. */
  power: number;
  /** The axis its emission is centred on, the half angle of the normals around it, and how far past them it emits. */
  axis: number[];
  thetaO: number;
  thetaE: number;
  /** Area density of a point on it, one for a point light. */
  density: number;
}

/** Sixteen words a node: low and power, high and cos theta_o, axis and cos theta_e, then the links. */
export const TREE_NODE_WORDS = 16;

/**
 * The high bit of a node's first link marks a leaf, whose low bits say where its lights start in the list after the
 * nodes; its second link says how many there are.
 */
export const TREE_LEAF = 0x80000000;

/** Four words a light in a leaf's list: the light, its chance within the leaf, its area density, the running chance. */
export const TREE_ENTRY_WORDS = 4;

/**
 * A leaf holds up to this many lights, when they are so close together, against how far the lights spread, that from
 * any point the tree could not tell them apart: the triangles of one small lamp. Lights further apart each keep a leaf,
 * which keeps every choice between them by distance and facing.
 */
const LEAF_LIGHTS = 8;
const LEAF_EXTENT = 0.02;

export const TREE_ROOT_PARENT = 0xffffffff;

interface Cone {
  axis: number[];
  thetaO: number;
  thetaE: number;
}

const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function normalized(v: number[]): number[] {
  const length = Math.hypot(v[0], v[1], v[2]);

  return length > 0 ? v.map(c => c / length) : [0, 0, 1];
}

/** Rotates a towards b by an angle, in the plane they span. */
function rotateTowards(a: number[], b: number[], angle: number): number[] {
  const along = dot(a, b);
  const ortho = normalized(b.map((c, k) => c - a[k] * along));

  return normalized(a.map((c, k) => c * Math.cos(angle) + ortho[k] * Math.sin(angle)));
}

/** The smallest cone bounding two, as Conty and Kulla merge them. */
export function unionCone(first: Cone, second: Cone): Cone {
  const [a, b] = first.thetaO >= second.thetaO ? [first, second] : [second, first];
  const thetaE = Math.max(a.thetaE, b.thetaE);
  const thetaD = Math.acos(Math.min(1, Math.max(-1, dot(a.axis, b.axis))));

  if (Math.min(thetaD + b.thetaO, Math.PI) <= a.thetaO) return { axis: a.axis, thetaO: a.thetaO, thetaE };

  const thetaO = (a.thetaO + thetaD + b.thetaO) / 2;

  if (thetaO >= Math.PI) return { axis: a.axis, thetaO: Math.PI, thetaE };

  // Two opposite axes have no plane between them: the cone then takes the whole sphere.
  if (thetaD >= Math.PI - 1e-6) return { axis: a.axis, thetaO: Math.PI, thetaE };

  return { axis: rotateTowards(a.axis, b.axis, thetaO - a.thetaO), thetaO, thetaE };
}

interface Node {
  low: number[];
  high: number[];
  power: number;
  cone: Cone;
  left: number;
  right: number;
  parent: number;
  /** The lights of a leaf, or none. */
  lights: number[];
}

/**
 * The tree over these lights, as words for the texture buffer, the leaf and list slot each light ended in, and where
 * the list of the leaves' lights starts. Split at the middle of the longest axis of the centres, which needs no cost
 * model and keeps a tight group, such as one lamp's triangles, on one side to share a leaf.
 */
export function buildLightTree(lights: readonly TreeLight[]): {
  words: Uint32Array; leafOf: Uint32Array; listAt: number;
} {
  const nodes: Node[] = [];
  // Two words a light: its leaf, and where it sits in the list of lights after the nodes.
  const leafOf = new Uint32Array(lights.length * 2);
  let span = 0;

  const build = (indices: number[], parent: number): number => {
    const index = nodes.length;
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
    let power = 0;
    let cone: Cone | null = null;

    for (const i of indices) {
      const light = lights[i];

      for (let k = 0; k < 3; k++) {
        low[k] = Math.min(low[k], light.low[k]);
        high[k] = Math.max(high[k], light.high[k]);
      }

      power += light.power;

      const own = { axis: normalized(light.axis), thetaO: light.thetaO, thetaE: light.thetaE };

      cone = cone ? unionCone(cone, own) : own;
    }

    const node: Node = { low, high, power, cone: cone as Cone, left: -1, right: -1, parent, lights: [] };

    nodes.push(node);

    const extent = Math.hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]);

    if (index === 0) span = extent;

    if (indices.length === 1 || (indices.length <= LEAF_LIGHTS && extent <= span * LEAF_EXTENT)) {
      node.lights = indices;

      return index;
    }

    const centre = (i: number, k: number) => (lights[i].low[k] + lights[i].high[k]) / 2;

    const spread = [0, 1, 2].map(k => {
      const values = indices.map(i => centre(i, k));

      return Math.max(...values) - Math.min(...values);
    });

    const axis = spread.indexOf(Math.max(...spread));
    const sorted = [...indices].sort((a, b) => centre(a, axis) - centre(b, axis));
    // Split at the middle of the extent, which keeps a tight group of lights together on one side; the median only
    // when that leaves a side empty, as lights all at one place do.
    const middle = (centre(sorted[0], axis) + centre(sorted[sorted.length - 1], axis)) / 2;
    const below = sorted.findIndex(i => centre(i, axis) > middle);
    const half = below > 0 ? below : sorted.length >> 1;

    node.left = build(sorted.slice(0, half), index);
    node.right = build(sorted.slice(half), index);

    return index;
  };

  if (lights.length) build(lights.map((_, i) => i), TREE_ROOT_PARENT);

  const listAt = nodes.length * TREE_NODE_WORDS;
  const entries = nodes.reduce((sum, node) => sum + node.lights.length, 0);
  const words = new Uint32Array(listAt + entries * TREE_ENTRY_WORDS);
  const floats = new Float32Array(words.buffer);
  let entry = 0;

  nodes.forEach((node, k) => {
    const at = k * TREE_NODE_WORDS;

    floats.set(node.low, at);
    floats[at + 3] = node.power;
    floats.set(node.high, at + 4);
    // A cone that takes the whole sphere is stored as minus one, which the kernel reads as no orientation at all.
    floats[at + 7] = node.cone.thetaO >= Math.PI ? -1 : Math.cos(node.cone.thetaO);
    floats.set(node.cone.axis, at + 8);
    floats[at + 11] = Math.cos(Math.min(node.cone.thetaE, Math.PI));

    if (node.lights.length) {
      words[at + 12] = (TREE_LEAF | entry) >>> 0;
      words[at + 13] = node.lights.length;

      // Within a leaf, by power: the lights are too close together for anything else to tell them apart.
      const total = node.lights.reduce((sum, i) => sum + lights[i].power, 0);
      let running = 0;

      node.lights.forEach((light, j) => {
        const chance = total > 0 ? lights[light].power / total : 1 / node.lights.length;
        const slot = listAt + entry * TREE_ENTRY_WORDS;

        running += chance;
        words[slot] = light;
        floats[slot + 1] = chance;
        floats[slot + 2] = lights[light].density;
        // The last closes the range exactly, so a unit sample never falls past it.
        floats[slot + 3] = j === node.lights.length - 1 ? 1 : running;
        leafOf[light * 2] = k;
        leafOf[light * 2 + 1] = entry;
        entry++;
      });
    } else {
      words[at + 12] = node.left;
      words[at + 13] = node.right;
    }

    words[at + 14] = node.parent >>> 0;
  });

  return { words, leafOf, listAt };
}
