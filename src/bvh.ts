/**
 * Binned SAH bounding volume hierarchy. It is built as a binary tree of boxes, eight words a node, and then laid out
 * the way the shader reads it: each node holds the boxes of its two children, so one read tells the traversal both
 * what it may enter and where to go, and a leaf is tested without being read at all. That is sixteen words a node,
 * matching `struct BvhNode` in `wgsl/common.wgsl.ts`: per child a minimum corner and a link, then a maximum corner and
 * a triangle count. A link of zero or more is the node to go to; a negative one is a leaf, `-1 - start` into `order`.
 *
 * The bounds are computed from the same `Float32Array` the shader traverses, so every corner is already a float and
 * needs no widening. Building from double precision positions, as the CPU renderer does, would need each minimum
 * rounded down and each maximum rounded up before they reach the GPU.
 */

/** Words per node in the packed buffer: both children's corners, links and counts. */
export const NODE_WORDS = 16;

/** Words per node of the binary tree the builder works on: its own corners, and its children or its leaf. */
const BINARY_WORDS = 8;

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
  /**
   * `nodeCount * NODE_WORDS` words, read as floats for the corners and as signed integers for the links. Node zero
   * holds the root's two children, or the root itself beside an empty child when the whole scene is one leaf.
   */
  nodes: ArrayBuffer;
  nodeCount: number;
  /** Triangle indices in leaf order: a leaf covers `order[start] .. order[start + count - 1]`. */
  order: Uint32Array;
  maxDepth: number;
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
  let nodes = new ArrayBuffer(Math.max(4, Math.ceil(triangleCount / 2)) * BINARY_WORDS * 4);
  let floats = new Float32Array(nodes);
  let ints = new Int32Array(nodes);

  const reserve = (needed: number) => {
    if (needed * BINARY_WORDS * 4 <= nodes.byteLength) return;

    const grown = new ArrayBuffer(Math.max(needed, Math.ceil(nodes.byteLength / (BINARY_WORDS * 4) * 1.6))
      * BINARY_WORDS * 4);

    new Float32Array(grown).set(floats);
    nodes = grown;
    floats = new Float32Array(nodes);
    ints = new Int32Array(nodes);
  };

  // The pending ranges, four integers each: start, count, node and depth. Typed, so the loop allocates nothing.
  let stack = new Int32Array(4 * 2 * MAX_BVH_DEPTH + 8);
  let top = 1;
  const bins = new Bins();
  let nodeCount = 1;
  let maxDepth = 0;

  stack[0] = 0;
  stack[1] = triangleCount;
  stack[2] = 0;
  stack[3] = 0;

  while (top) {
    top--;

    const start = stack[top * 4], count = stack[top * 4 + 1], node = stack[top * 4 + 2], depth = stack[top * 4 + 3];
    const end = start + count;

    if (depth > maxDepth) maxDepth = depth;

    // The node's bounds and its centroids' bounds, in one pass. Doubles, so the corners are exactly the floats read.
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let cMinX = Infinity, cMinY = Infinity, cMinZ = Infinity, cMaxX = -Infinity, cMaxY = -Infinity, cMaxZ = -Infinity;

    for (let i = start; i < end; i++) {
      const triangle = order[i] * 3;
      let value = min[triangle];

      if (value < minX) minX = value;
      value = min[triangle + 1];
      if (value < minY) minY = value;
      value = min[triangle + 2];
      if (value < minZ) minZ = value;
      value = max[triangle];
      if (value > maxX) maxX = value;
      value = max[triangle + 1];
      if (value > maxY) maxY = value;
      value = max[triangle + 2];
      if (value > maxZ) maxZ = value;
      value = centroid[triangle];
      if (value < cMinX) cMinX = value;
      if (value > cMaxX) cMaxX = value;
      value = centroid[triangle + 1];
      if (value < cMinY) cMinY = value;
      if (value > cMaxY) cMaxY = value;
      value = centroid[triangle + 2];
      if (value < cMinZ) cMinZ = value;
      if (value > cMaxZ) cMaxZ = value;
    }

    const word = node * BINARY_WORDS;

    floats[word] = minX;
    floats[word + 1] = minY;
    floats[word + 2] = minZ;
    floats[word + 4] = maxX;
    floats[word + 5] = maxY;
    floats[word + 6] = maxZ;

    let split = -1;

    if (count > MAX_LEAF_SIZE && depth + 1 < MAX_BVH_DEPTH) {
      bins.centroidMin[0] = cMinX;
      bins.centroidMin[1] = cMinY;
      bins.centroidMin[2] = cMinZ;
      bins.centroidMax[0] = cMaxX;
      bins.centroidMax[1] = cMaxY;
      bins.centroidMax[2] = cMaxZ;
      split = partition(order, centroid, min, max, start, end, area(minX, minY, minZ, maxX, maxY, maxZ), bins);
    }

    if (split < 0) {
      // A leaf points into `order` instead of into the node array, which the sign of the first link tells apart.
      ints[word + 3] = -1 - start;
      ints[word + 7] = count;
      continue;
    }

    reserve(nodeCount + 2);

    const left = nodeCount++;
    const right = nodeCount++;

    ints[word + 3] = left;
    ints[word + 7] = right;

    if ((top + 2) * 4 > stack.length) {
      const grown = new Int32Array(stack.length * 2);

      grown.set(stack);
      stack = grown;
    }

    // Left first, then right, so the right side is built first, as the order the tree was always built in.
    stack.set([start, split - start, left, depth + 1, split, end - split, right, depth + 1], top * 4);
    top += 2;
  }

  return { ...widen(floats, ints, nodeCount), order, maxDepth };
}

/**
 * The binary tree laid out for the shader: one node per interior node of it, holding both children's boxes. Interior
 * nodes are numbered depth first, left before right, so a node and the child it leads to tend to share a cache line.
 */
function widen(floats: Float32Array, ints: Int32Array, binaryCount: number): { nodes: ArrayBuffer; nodeCount: number } {
  const leaf = (node: number) => ints[node * BINARY_WORDS + 3] < 0;
  const rootLeaf = leaf(0);
  // A full binary tree has one fewer interior node than leaves; a single leaf still needs a node to stand in.
  const nodeCount = rootLeaf ? 1 : (binaryCount - 1) / 2;
  const nodes = new ArrayBuffer(nodeCount * NODE_WORDS * 4);
  const outFloats = new Float32Array(nodes);
  const outInts = new Int32Array(nodes);
  const index = new Int32Array(binaryCount);

  const writeChild = (word: number, child: number) => {
    const from = child * BINARY_WORDS;

    for (let axis = 0; axis < 3; axis++) {
      outFloats[word + axis] = floats[from + axis];
      outFloats[word + 4 + axis] = floats[from + 4 + axis];
    }

    outInts[word + 3] = leaf(child) ? ints[from + 3] : index[child];
    outInts[word + 7] = leaf(child) ? ints[from + 7] : 0;
  };

  if (rootLeaf) {
    writeChild(0, 0);

    // A box at the far corner of the float range, which no ray reaches, around a leaf of nothing.
    for (let axis = 0; axis < 3; axis++) outFloats[8 + axis] = outFloats[12 + axis] = 3.4028234663852886e38;

    outInts[11] = -1;
    outInts[15] = 0;

    return { nodes, nodeCount };
  }

  const interior: number[] = [];
  const pending = [0];

  while (pending.length) {
    const node = pending.pop() as number;

    index[node] = interior.length;
    interior.push(node);

    const left = ints[node * BINARY_WORDS + 3], right = ints[node * BINARY_WORDS + 7];

    if (!leaf(right)) pending.push(right);
    if (!leaf(left)) pending.push(left);
  }

  interior.forEach((node, at) => {
    writeChild(at * NODE_WORDS, ints[node * BINARY_WORDS + 3]);
    writeChild(at * NODE_WORDS + 8, ints[node * BINARY_WORDS + 7]);
  });

  return { nodes, nodeCount };
}

/** Surface area, the cost the split heuristic weighs each side by. Empty bounds cost nothing. */
function area(minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): number {
  const x = maxX - minX, y = maxY - minY, z = maxZ - minZ;

  return x < 0 || y < 0 || z < 0 ? 0 : 2 * (x * y + y * z + z * x);
}

/** Scratch space for one split, kept across every node so binning allocates nothing. */
class Bins {
  readonly centroidMin = new Float64Array(3);
  readonly centroidMax = new Float64Array(3);
  /** Six doubles per bin and axis: the minimum corner, then the maximum. */
  readonly bounds = new Float64Array(3 * BIN_COUNT * 6);
  readonly counts = new Int32Array(3 * BIN_COUNT);
  readonly rightArea = new Float64Array(BIN_COUNT);
  readonly rightCount = new Int32Array(BIN_COUNT);
  readonly scale = new Float64Array(3);

  reset() {
    const bounds = this.bounds;

    for (let i = 0; i < bounds.length; i += 6) {
      bounds[i] = bounds[i + 1] = bounds[i + 2] = Infinity;
      bounds[i + 3] = bounds[i + 4] = bounds[i + 5] = -Infinity;
    }

    this.counts.fill(0);
  }
}

/**
 * Reorders `order` in place so the range splits in two, and returns where the right side starts, or -1 when no split
 * beats keeping the range as a leaf. Bins the centroids on all three axes in one pass and sweeps the surface area
 * heuristic on each.
 */
function partition(order: Uint32Array, centroid: Float64Array, min: Float32Array, max: Float32Array,
  start: number, end: number, parentArea: number, bins: Bins): number {
  if (parentArea <= 0) return -1;

  const { centroidMin, centroidMax, bounds, counts, rightArea, rightCount, scale } = bins;
  let binned = false;

  for (let axis = 0; axis < 3; axis++) {
    const extent = centroidMax[axis] - centroidMin[axis];

    // Every centroid sits at the same place on this axis: no split here separates anything.
    scale[axis] = extent > 0 ? BIN_COUNT / extent : 0;
    binned ||= extent > 0;
  }

  if (!binned) return -1;

  bins.reset();

  for (let i = start; i < end; i++) {
    const triangle = order[i] * 3;
    const x0 = min[triangle], y0 = min[triangle + 1], z0 = min[triangle + 2];
    const x1 = max[triangle], y1 = max[triangle + 1], z1 = max[triangle + 2];

    for (let axis = 0; axis < 3; axis++) {
      if (scale[axis] === 0) continue;

      const bin = axis * BIN_COUNT + binOf(centroid[triangle + axis], centroidMin[axis], scale[axis]);
      const at = bin * 6;

      counts[bin]++;

      if (x0 < bounds[at]) bounds[at] = x0;
      if (y0 < bounds[at + 1]) bounds[at + 1] = y0;
      if (z0 < bounds[at + 2]) bounds[at + 2] = z0;
      if (x1 > bounds[at + 3]) bounds[at + 3] = x1;
      if (y1 > bounds[at + 4]) bounds[at + 4] = y1;
      if (z1 > bounds[at + 5]) bounds[at + 5] = z1;
    }
  }

  // Splitting costs a traversal step, so a split only wins when it beats shading the whole range as one leaf.
  let bestCost = end - start, bestAxis = -1, bestBin = -1;

  for (let axis = 0; axis < 3; axis++) {
    if (scale[axis] === 0) continue;

    // Sweeping once from each side turns the heuristic into two passes instead of one per candidate split.
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let running = 0;

    for (let bin = BIN_COUNT - 1; bin > 0; bin--) {
      const at = (axis * BIN_COUNT + bin) * 6;

      if (bounds[at] < minX) minX = bounds[at];
      if (bounds[at + 1] < minY) minY = bounds[at + 1];
      if (bounds[at + 2] < minZ) minZ = bounds[at + 2];
      if (bounds[at + 3] > maxX) maxX = bounds[at + 3];
      if (bounds[at + 4] > maxY) maxY = bounds[at + 4];
      if (bounds[at + 5] > maxZ) maxZ = bounds[at + 5];
      running += counts[axis * BIN_COUNT + bin];
      rightArea[bin] = area(minX, minY, minZ, maxX, maxY, maxZ);
      rightCount[bin] = running;
    }

    minX = minY = minZ = Infinity;
    maxX = maxY = maxZ = -Infinity;
    running = 0;

    for (let bin = 0; bin < BIN_COUNT - 1; bin++) {
      const at = (axis * BIN_COUNT + bin) * 6;

      if (bounds[at] < minX) minX = bounds[at];
      if (bounds[at + 1] < minY) minY = bounds[at + 1];
      if (bounds[at + 2] < minZ) minZ = bounds[at + 2];
      if (bounds[at + 3] > maxX) maxX = bounds[at + 3];
      if (bounds[at + 4] > maxY) maxY = bounds[at + 4];
      if (bounds[at + 5] > maxZ) maxZ = bounds[at + 5];
      running += counts[axis * BIN_COUNT + bin];

      if (!running || !rightCount[bin + 1]) continue;

      const cost = TRAVERSAL_COST
        + (area(minX, minY, minZ, maxX, maxY, maxZ) * running + rightArea[bin + 1] * rightCount[bin + 1]) / parentArea;

      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestBin = bin;
      }
    }
  }

  if (bestAxis < 0) return -1;

  const pivot = centroidMin[bestAxis];
  const bestScale = scale[bestAxis];
  let left = start;

  for (let i = start; i < end; i++) {
    if (binOf(centroid[order[i] * 3 + bestAxis], pivot, bestScale) <= bestBin) {
      const swap = order[i];

      order[i] = order[left];
      order[left] = swap;
      left++;
    }
  }

  // The sweep only proposes splits both sides occupy, so this is a guard against rounding, not an expected outcome.
  return left === start || left === end ? -1 : left;
}

function binOf(value: number, origin: number, scale: number): number {
  return Math.min(BIN_COUNT - 1, Math.max(0, Math.floor((value - origin) * scale)));
}
