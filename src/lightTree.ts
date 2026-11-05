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

/** The high bit of a node's first link marks a leaf, whose low bits are its light's index. */
export const TREE_LEAF = 0x80000000;

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
  /** The light of a leaf, or -1. */
  light: number;
  density: number;
}

/**
 * The tree over these lights, as words for the texture buffer, and the leaf each light ended in. Split at the median
 * of the longest axis of the centres, which needs no cost model and keeps the depth near log2 of the count.
 */
export function buildLightTree(lights: readonly TreeLight[]): { words: Uint32Array; leafOf: Uint32Array } {
  const nodes: Node[] = [];
  const leafOf = new Uint32Array(lights.length);

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

    const node: Node = { low, high, power, cone: cone as Cone, left: -1, right: -1, parent, light: -1, density: 1 };

    nodes.push(node);

    if (indices.length === 1) {
      node.light = indices[0];
      node.density = lights[indices[0]].density;
      leafOf[indices[0]] = index;

      return index;
    }

    const centre = (i: number, k: number) => (lights[i].low[k] + lights[i].high[k]) / 2;

    const extent = [0, 1, 2].map(k => {
      const values = indices.map(i => centre(i, k));

      return Math.max(...values) - Math.min(...values);
    });

    const axis = extent.indexOf(Math.max(...extent));
    const sorted = [...indices].sort((a, b) => centre(a, axis) - centre(b, axis));
    const half = sorted.length >> 1;

    node.left = build(sorted.slice(0, half), index);
    node.right = build(sorted.slice(half), index);

    return index;
  };

  if (lights.length) build(lights.map((_, i) => i), TREE_ROOT_PARENT);

  const words = new Uint32Array(nodes.length * TREE_NODE_WORDS);
  const floats = new Float32Array(words.buffer);

  nodes.forEach((node, k) => {
    const at = k * TREE_NODE_WORDS;

    floats.set(node.low, at);
    floats[at + 3] = node.power;
    floats.set(node.high, at + 4);
    // A cone that takes the whole sphere is stored as minus one, which the kernel reads as no orientation at all.
    floats[at + 7] = node.cone.thetaO >= Math.PI ? -1 : Math.cos(node.cone.thetaO);
    floats.set(node.cone.axis, at + 8);
    floats[at + 11] = Math.cos(Math.min(node.cone.thetaE, Math.PI));

    if (node.light >= 0) {
      words[at + 12] = (TREE_LEAF | node.light) >>> 0;
      floats[at + 13] = node.density;
    } else {
      words[at + 12] = node.left;
      words[at + 13] = node.right;
    }

    words[at + 14] = node.parent >>> 0;
  });

  return { words, leafOf };
}
