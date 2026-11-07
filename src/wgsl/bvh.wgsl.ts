/**
 * Triangle intersection and the iterative traversal, with the fixed stack the builder caps its depth to.
 *
 * The positions are uploaded in the hierarchy's leaf order, so a leaf is a run of triangles read straight off the
 * buffer; the scene's own triangle number is looked up once, for the hit a trace returns. Every other reader goes the
 * other way, from a triangle number to where its corners sit.
 *
 * With INSTANCES compiled in, a leaf with a count of -1 is an instance: the ray is taken into its object space and
 * the traversal goes on in its prototype's hierarchy, with a marker on the stack that brings it back to world space.
 * The distance along the ray is the same in both spaces, since the direction is transformed and not normalized.
 */
export const BVH_WGSL = /* wgsl */ `
/** A corner of the triangle at this position of the hierarchy's leaf order. */
fn leafVertex(at: u32, corner: u32) -> vec3<f32> {
  let base = at * 9u + corner * 3u;

  return vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
}

/** Where a triangle of the scene sits in the leaf order the positions are stored in. */
fn leafOf(triangle: u32) -> u32 {
  return indices[frame.leafOffset + triangle];
}

/** A corner of a triangle a hit named, in world space. */
fn triangleVertex(triangle: u32, corner: u32) -> vec3<f32> {
  let instance = instanceOf(triangle);
  let p = leafVertex(leafOf(storedTriangle(triangle, instance)), corner);

  if (instance != NO_INSTANCE) {
    return instancePoint(instance, p);
  }

  return p;
}

/** The stack entry that leaves an instance: the traversal goes back to world space. */
const RETURN_ENTRY: u32 = 0xffffffffu;

/** A huge finite reciprocal instead of an infinite one: an axis aligned slab would otherwise produce 0 * inf. */
fn rayInverse(direction: vec3<f32>) -> vec3<f32> {
  let signs = select(vec3<f32>(-1.0), vec3<f32>(1.0), direction >= vec3<f32>(0.0));

  return signs / max(abs(direction), vec3<f32>(1e-20));
}

/** The number a hit goes by, from where it sits in the leaf order and the instance it was found in. */
fn namedTriangle(at: u32, instance: u32) -> u32 {
  let stored = orderedTriangle(at);

  if (INSTANCES && instance != NO_INSTANCE) {
    return instanceWord(instance, 26u) + stored - instanceWord(instance, 25u);
  }

  return stored;
}

/** Moller-Trumbore. Returns the distance in x and the barycentric coordinates in yz, or a negative distance. */
fn intersectTriangle(at: u32, origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
  let miss = vec3<f32>(-1.0, 0.0, 0.0);
  let a = leafVertex(at, 0u);
  let edge1 = leafVertex(at, 1u) - a;
  let edge2 = leafVertex(at, 2u) - a;
  let p = cross(direction, edge2);
  let determinant = dot(edge1, p);

  if (abs(determinant) < 1e-12) {
    return miss;
  }

  let inverse = 1.0 / determinant;
  let offset = origin - a;
  let u = dot(offset, p) * inverse;

  if (u < 0.0 || u > 1.0) {
    return miss;
  }

  let q = cross(offset, edge1);
  let v = dot(direction, q) * inverse;

  if (v < 0.0 || u + v > 1.0) {
    return miss;
  }

  return vec3<f32>(dot(edge2, q) * inverse, u, v);
}

/** Where a ray enters a box, or 1e30 when it misses it or only meets it past maxDistance. */
fn enterBox(low: vec3<f32>, high: vec3<f32>, origin: vec3<f32>, inverse: vec3<f32>, maxDistance: f32) -> f32 {
  let first = (low - origin) * inverse;
  let second = (high - origin) * inverse;
  let near = min(first, second);
  let far = max(first, second);
  let enter = max(max(near.x, near.y), max(near.z, 0.0));
  let exit = min(min(far.x, far.y), min(far.z, maxDistance));

  return select(1e30, enter, enter <= exit);
}

/** The triangles of a leaf, against the closest hit so far, which records where in the leaf order it found one. */
fn intersectLeaf(link: i32, count: i32, origin: vec3<f32>, direction: vec3<f32>, hit: ptr<function, Hit>) {
  let start = u32(-1 - link);

  for (var i = 0u; i < u32(count); i = i + 1u) {
    let result = intersectTriangle(start + i, origin, direction);

    if (result.x > RAY_EPSILON && result.x < (*hit).distance) {
      (*hit).distance = result.x;
      (*hit).triangle = start + i;
      (*hit).u = result.y;
      (*hit).v = result.z;
    }
  }
}

/** Whether any triangle of a leaf lies along the ray before maxDistance: its position in the leaf order, or NO_HIT. */
fn anyInLeaf(link: i32, count: i32, origin: vec3<f32>, direction: vec3<f32>, maxDistance: f32) -> u32 {
  let start = u32(-1 - link);

  for (var i = 0u; i < u32(count); i = i + 1u) {
    let result = intersectTriangle(start + i, origin, direction);

    if (result.x > RAY_EPSILON && result.x < maxDistance) {
      return start + i;
    }
  }

  return NO_HIT;
}

/** No node to go to: the traversal takes the next entry off the stack. */
const NO_NODE: u32 = 0xffffffffu;

/**
 * A stack entry that is a leaf rather than a node: the node holding it and which of its two children it is. A leaf
 * has no node of its own, so this is how one waits its turn.
 */
const LEAF_ENTRY: u32 = 0x80000000u;

fn leafEntry(parent: u32, right: bool) -> u32 {
  return LEAF_ENTRY | (parent << 1u) | select(0u, 1u, right);
}

/** A waiting leaf's link and count, read back from the node that holds it. */
fn waitingLeaf(entry: u32) -> vec2<i32> {
  let parent = (entry & ~LEAF_ENTRY) >> 1u;

  if ((entry & 1u) != 0u) {
    return vec2<i32>(nodes[parent].rightLink, nodes[parent].rightCount);
  }

  return vec2<i32>(nodes[parent].leftLink, nodes[parent].leftCount);
}

/**
 * The closest hit along a ray. A node holds both of its children's boxes, so one read tests them both. The nearer is
 * visited first, so the closest hit shrinks the search early and whole subtrees behind it are skipped; the farther one
 * waits on the stack with where the ray entered it, and is dropped unopened if something closer turned up meanwhile.
 * A leaf is tested at the top of the next step, the one place the loop tests triangles, which keeps the kernel small.
 * This visits what the binary tree's traversal visited, in the same order, so it finds the same hit.
 */
fn traceScene(worldOrigin: vec3<f32>, worldDirection: vec3<f32>, maxDistance: f32) -> Hit {
  var hit: Hit;
  hit.distance = maxDistance;
  hit.triangle = NO_HIT;
  hit.u = 0.0;
  hit.v = 0.0;

  if (frame.triangleCount == 0u) {
    return hit;
  }

  var origin = worldOrigin;
  var direction = worldDirection;
  var inverse = rayInverse(direction);
  // The instance the traversal is inside, and the one the closest hit so far was found in.
  var instance = NO_INSTANCE;
  var hitInstance = NO_INSTANCE;

  var stack: array<u32, BVH_STACK>;
  var enters: array<f32, BVH_STACK>;
  var depth = 0u;
  var current = 0u;
  // A leaf to test before anything else: its link, and its triangle count, zero when there is none.
  var leaf = vec2<i32>(0, 0);

  loop {
    if (leaf.y > 0) {
      let before = hit.distance;

      intersectLeaf(leaf.x, leaf.y, origin, direction, &hit);

      if (INSTANCES && hit.distance < before) {
        hitInstance = instance;
      }

      leaf.y = 0;
    }

    if (INSTANCES && leaf.y < 0) {
      leaf.y = 0;

      if (depth < BVH_STACK) {
        instance = u32(-1 - leaf.x);
        stack[depth] = RETURN_ENTRY;
        enters[depth] = -1.0;
        depth = depth + 1u;
        origin = instanceToObject(instance, worldOrigin, 1.0);
        direction = instanceToObject(instance, worldDirection, 0.0);
        inverse = rayInverse(direction);
        current = instanceWord(instance, 24u);
      }
    }

    if (current == NO_NODE) {
      if (depth == 0u) {
        break;
      }

      depth = depth - 1u;

      // Something closer was found since this was pushed.
      if (enters[depth] > hit.distance) {
        continue;
      }

      let entry = stack[depth];

      if (INSTANCES && entry == RETURN_ENTRY) {
        instance = NO_INSTANCE;
        origin = worldOrigin;
        direction = worldDirection;
        inverse = rayInverse(direction);
        continue;
      }

      if ((entry & LEAF_ENTRY) != 0u) {
        leaf = waitingLeaf(entry);
        continue;
      }

      current = entry;
    }

    let node = nodes[current];
    let leftEnter = enterBox(node.leftLow, node.leftHigh, origin, inverse, hit.distance);
    let rightEnter = enterBox(node.rightLow, node.rightHigh, origin, inverse, hit.distance);
    let leftFirst = leftEnter <= rightEnter;
    let nearLink = select(node.rightLink, node.leftLink, leftFirst);
    let farEnter = select(leftEnter, rightEnter, leftFirst);
    let farLink = select(node.leftLink, node.rightLink, leftFirst);
    let parent = current;

    current = NO_NODE;

    // The nearer child: a miss means both missed.
    if (min(leftEnter, rightEnter) < 1e30) {
      if (nearLink < 0) {
        leaf = vec2<i32>(nearLink, select(node.rightCount, node.leftCount, leftFirst));
      } else {
        current = u32(nearLink);
      }
    }

    // The builder caps its depth so the far children always fit; the guard keeps a malformed tree inside the stack.
    if (farEnter < 1e30 && depth < BVH_STACK) {
      stack[depth] = select(u32(farLink), leafEntry(parent, leftFirst), farLink < 0);
      enters[depth] = farEnter;
      depth = depth + 1u;
    }
  }

  if (hit.triangle != NO_HIT) {
    hit.triangle = namedTriangle(hit.triangle, hitInstance);
  }

  return hit;
}

/**
 * Whether anything at all lies along a ray before maxDistance, stopping at the first triangle it finds rather than the
 * closest: what a shadow ray asks when every surface it could meet blocks it. Returns that triangle, or NO_HIT. The
 * right child goes first and the left one waits, the order the binary tree's traversal took.
 */
fn traceAny(worldOrigin: vec3<f32>, worldDirection: vec3<f32>, maxDistance: f32) -> u32 {
  if (frame.triangleCount == 0u) {
    return NO_HIT;
  }

  var origin = worldOrigin;
  var direction = worldDirection;
  var inverse = rayInverse(direction);
  var instance = NO_INSTANCE;

  var stack: array<u32, BVH_STACK>;
  var depth = 0u;
  var current = 0u;
  var leaf = vec2<i32>(0, 0);

  loop {
    if (leaf.y > 0) {
      let found = anyInLeaf(leaf.x, leaf.y, origin, direction, maxDistance);

      if (found != NO_HIT) {
        return namedTriangle(found, instance);
      }

      leaf.y = 0;
    }

    if (INSTANCES && leaf.y < 0) {
      leaf.y = 0;

      if (depth < BVH_STACK) {
        instance = u32(-1 - leaf.x);
        stack[depth] = RETURN_ENTRY;
        depth = depth + 1u;
        origin = instanceToObject(instance, worldOrigin, 1.0);
        direction = instanceToObject(instance, worldDirection, 0.0);
        inverse = rayInverse(direction);
        current = instanceWord(instance, 24u);
      }
    }

    if (current == NO_NODE) {
      if (depth == 0u) {
        break;
      }

      depth = depth - 1u;

      let entry = stack[depth];

      if (INSTANCES && entry == RETURN_ENTRY) {
        instance = NO_INSTANCE;
        origin = worldOrigin;
        direction = worldDirection;
        inverse = rayInverse(direction);
        continue;
      }

      if ((entry & LEAF_ENTRY) != 0u) {
        leaf = waitingLeaf(entry);
        continue;
      }

      current = entry;
    }

    let node = nodes[current];
    let rightHit = enterBox(node.rightLow, node.rightHigh, origin, inverse, maxDistance) < 1e30;
    let leftHit = enterBox(node.leftLow, node.leftHigh, origin, inverse, maxDistance) < 1e30;
    let parent = current;

    current = NO_NODE;

    // The right child is taken now and the left one waits; with only one of them hit, that one is taken now.
    let takeRight = rightHit;
    let takenLink = select(node.leftLink, node.rightLink, takeRight);

    if (rightHit || leftHit) {
      if (takenLink < 0) {
        leaf = vec2<i32>(takenLink, select(node.leftCount, node.rightCount, takeRight));
      } else {
        current = u32(takenLink);
      }
    }

    if (rightHit && leftHit && depth < BVH_STACK) {
      stack[depth] = select(u32(node.leftLink), leafEntry(parent, false), node.leftLink < 0);
      depth = depth + 1u;
    }
  }

  return NO_HIT;
}
`;
