/**
 * Binned SAH bounding volume hierarchy, built straight into the layout the shader reads: eight words per node, the
 * first four a minimum corner and a left child or leaf start, the last four a maximum corner and a right child or
 * triangle count, matching `struct BvhNode` in `wgsl/bvh.wgsl.ts`.
 *
 * The bounds are computed from the same `Float32Array` the shader traverses, so every corner is already a float and
 * needs no widening. Building from double precision positions, as the CPU renderer does, would need each minimum
 * rounded down and each maximum rounded up before they reach the GPU.
 */

/** Words per node in the packed buffer. */
export const NODE_WORDS = 8;

/** Triangles a leaf may hold. */
const MAX_LEAF_SIZE = 4;

/** Candidate split positions evaluated per axis. */
const BIN_COUNT = 12;

/** What descending one more level costs, relative to intersecting one triangle. */
const TRAVERSAL_COST = 0.125;

/**
 * Deepest node the builder produces. The shader traverses with a fixed stack, and one entry per level is enough
 * because a node only pushes its far child. Depth is capped rather than trusted.
 */
export const MAX_BVH_DEPTH = 32;

export interface PackedBvh {
  /** `nodeCount * NODE_WORDS` words, read as floats for the corners and as signed integers for the links. */
  nodes: ArrayBuffer;
  nodeCount: number;
  /** Triangle indices in leaf order: a leaf covers `order[start] .. order[start + count - 1]`. */
  order: Uint32Array;
  maxDepth: number;
}

interface Range {
  start: number;
  count: number;
  node: number;
  depth: number;
}

/** Axis aligned bounds under construction, in doubles so the sweep does not accumulate rounding. */
class Bounds {
  min = [Infinity, Infinity, Infinity];
  max = [-Infinity, -Infinity, -Infinity];

  reset() {
    this.min = [Infinity, Infinity, Infinity];
    this.max = [-Infinity, -Infinity, -Infinity];
  }

  addPoint(x: number, y: number, z: number) {
    const point = [x, y, z];

    for (let axis = 0; axis < 3; axis++) {
      if (point[axis] < this.min[axis]) this.min[axis] = point[axis];
      if (point[axis] > this.max[axis]) this.max[axis] = point[axis];
    }
  }

  addBounds(other: Bounds) {
    for (let axis = 0; axis < 3; axis++) {
      if (other.min[axis] < this.min[axis]) this.min[axis] = other.min[axis];
      if (other.max[axis] > this.max[axis]) this.max[axis] = other.max[axis];
    }
  }

  /** Surface area, the cost the split heuristic weighs each side by. Empty bounds cost nothing. */
  area(): number {
    const x = this.max[0] - this.min[0], y = this.max[1] - this.min[1], z = this.max[2] - this.min[2];

    return x < 0 || y < 0 || z < 0 ? 0 : 2 * (x * y + y * z + z * x);
  }
}

/** Per-triangle bounds and centroids, the only inputs the split needs. */
function triangleBounds(positions: Float32Array, triangleCount: number) {
  const min = new Float32Array(triangleCount * 3);
  const max = new Float32Array(triangleCount * 3);
  const centroid = new Float64Array(triangleCount * 3);

  for (let triangle = 0; triangle < triangleCount; triangle++) {
    for (let axis = 0; axis < 3; axis++) {
      const a = positions[triangle * 9 + axis];
      const b = positions[triangle * 9 + 3 + axis];
      const c = positions[triangle * 9 + 6 + axis];

      min[triangle * 3 + axis] = Math.min(a, b, c);
      max[triangle * 3 + axis] = Math.max(a, b, c);
      centroid[triangle * 3 + axis] = (a + b + c) / 3;
    }
  }

  return { min, max, centroid };
}

export function buildBvh(positions: Float32Array, triangleCount: number): PackedBvh {
  const order = new Uint32Array(triangleCount);

  for (let i = 0; i < triangleCount; i++) order[i] = i;

  if (triangleCount === 0)
    return { nodes: new ArrayBuffer(0), nodeCount: 0, order, maxDepth: 0 };

  const { min, max, centroid } = triangleBounds(positions, triangleCount);
  // The heuristic may split down to one triangle per leaf, so the pool starts at the common case and grows.
  let nodes = new ArrayBuffer(Math.max(4, Math.ceil(triangleCount / 2)) * NODE_WORDS * 4);
  let floats = new Float32Array(nodes);
  let ints = new Int32Array(nodes);

  const reserve = (needed: number) => {
    if (needed * NODE_WORDS * 4 <= nodes.byteLength) return;

    const grown = new ArrayBuffer(Math.max(needed, Math.ceil(nodes.byteLength / (NODE_WORDS * 4) * 1.6))
      * NODE_WORDS * 4);

    new Float32Array(grown).set(floats);
    nodes = grown;
    floats = new Float32Array(nodes);
    ints = new Int32Array(nodes);
  };

  const stack: Range[] = [{ start: 0, count: triangleCount, node: 0, depth: 0 }];
  const bounds = new Bounds();
  let nodeCount = 1;
  let maxDepth = 0;

  while (stack.length) {
    const range = stack.pop() as Range;

    maxDepth = Math.max(maxDepth, range.depth);
    bounds.reset();

    for (let i = range.start; i < range.start + range.count; i++) {
      const triangle = order[i] * 3;

      bounds.addPoint(min[triangle], min[triangle + 1], min[triangle + 2]);
      bounds.addPoint(max[triangle], max[triangle + 1], max[triangle + 2]);
    }

    const word = range.node * NODE_WORDS;

    floats[word] = bounds.min[0];
    floats[word + 1] = bounds.min[1];
    floats[word + 2] = bounds.min[2];
    floats[word + 4] = bounds.max[0];
    floats[word + 5] = bounds.max[1];
    floats[word + 6] = bounds.max[2];

    const split = range.count > MAX_LEAF_SIZE && range.depth + 1 < MAX_BVH_DEPTH
      ? partition(order, centroid, min, max, range, bounds.area())
      : -1;

    if (split < 0) {
      // A leaf points into `order` instead of into the node array, which the sign of the first link tells apart.
      ints[word + 3] = -1 - range.start;
      ints[word + 7] = range.count;
      continue;
    }

    reserve(nodeCount + 2);

    const left = nodeCount++;
    const right = nodeCount++;

    ints[word + 3] = left;
    ints[word + 7] = right;
    stack.push({ start: range.start, count: split - range.start, node: left, depth: range.depth + 1 });
    stack.push({ start: split, count: range.start + range.count - split, node: right, depth: range.depth + 1 });
  }

  return { nodes: nodes.slice(0, nodeCount * NODE_WORDS * 4), nodeCount, order, maxDepth };
}

/**
 * Reorders `order` in place so the range splits in two, and returns where the right side starts, or -1 when no split
 * beats keeping the range as a leaf. Bins the centroids on all three axes and sweeps the surface area heuristic.
 */
function partition(order: Uint32Array, centroid: Float64Array, min: Float32Array, max: Float32Array,
  range: Range, parentArea: number): number {
  if (parentArea <= 0) return -1;

  const end = range.start + range.count;
  const centroidBounds = new Bounds();

  for (let i = range.start; i < end; i++) {
    const triangle = order[i] * 3;

    centroidBounds.addPoint(centroid[triangle], centroid[triangle + 1], centroid[triangle + 2]);
  }

  // Splitting costs a traversal step, so a split only wins when it beats shading the whole range as one leaf.
  let bestCost = range.count, bestAxis = -1, bestBin = -1, bestScale = 0;

  for (let axis = 0; axis < 3; axis++) {
    const extent = centroidBounds.max[axis] - centroidBounds.min[axis];

    // Every centroid sits at the same place on this axis: no split here separates anything.
    if (extent <= 0) continue;

    const scale = BIN_COUNT / extent;
    const binBounds = Array.from({ length: BIN_COUNT }, () => new Bounds());
    const binCounts = new Int32Array(BIN_COUNT);

    for (let i = range.start; i < end; i++) {
      const triangle = order[i] * 3;
      const bin = binOf(centroid[triangle + axis], centroidBounds.min[axis], scale);

      binCounts[bin]++;
      binBounds[bin].addPoint(min[triangle], min[triangle + 1], min[triangle + 2]);
      binBounds[bin].addPoint(max[triangle], max[triangle + 1], max[triangle + 2]);
    }

    // Sweeping once from each side turns the heuristic into two passes instead of one per candidate split.
    const rightArea = new Float64Array(BIN_COUNT);
    const rightCount = new Int32Array(BIN_COUNT);
    const accumulated = new Bounds();
    let running = 0;

    for (let bin = BIN_COUNT - 1; bin > 0; bin--) {
      accumulated.addBounds(binBounds[bin]);
      running += binCounts[bin];
      rightArea[bin] = accumulated.area();
      rightCount[bin] = running;
    }

    accumulated.reset();
    running = 0;

    for (let bin = 0; bin < BIN_COUNT - 1; bin++) {
      accumulated.addBounds(binBounds[bin]);
      running += binCounts[bin];

      if (!running || !rightCount[bin + 1]) continue;

      const cost = TRAVERSAL_COST
        + (accumulated.area() * running + rightArea[bin + 1] * rightCount[bin + 1]) / parentArea;

      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestBin = bin;
        bestScale = scale;
      }
    }
  }

  if (bestAxis < 0) return -1;

  const pivot = centroidBounds.min[bestAxis];
  let left = range.start;

  for (let i = range.start; i < end; i++) {
    if (binOf(centroid[order[i] * 3 + bestAxis], pivot, bestScale) <= bestBin) {
      const swap = order[i];

      order[i] = order[left];
      order[left] = swap;
      left++;
    }
  }

  // The sweep only proposes splits both sides occupy, so this is a guard against rounding, not an expected outcome.
  return left === range.start || left === end ? -1 : left;
}

function binOf(value: number, origin: number, scale: number): number {
  return Math.min(BIN_COUNT - 1, Math.max(0, Math.floor((value - origin) * scale)));
}
