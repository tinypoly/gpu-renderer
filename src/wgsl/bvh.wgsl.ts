/** Triangle intersection and the iterative traversal, with the fixed stack the builder caps its depth to. */
export const BVH_WGSL = /* wgsl */ `
fn triangleVertex(triangle: u32, corner: u32) -> vec3<f32> {
  let base = triangle * 9u + corner * 3u;

  return vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
}

/** Moller-Trumbore. Returns the distance in x and the barycentric coordinates in yz, or a negative distance. */
fn intersectTriangle(triangle: u32, origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
  let miss = vec3<f32>(-1.0, 0.0, 0.0);
  let a = triangleVertex(triangle, 0u);
  let edge1 = triangleVertex(triangle, 1u) - a;
  let edge2 = triangleVertex(triangle, 2u) - a;
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

/** Where a ray enters a node's box, or 1e30 when it misses it or only meets it past maxDistance. */
fn enterBox(node: BvhNode, origin: vec3<f32>, inverse: vec3<f32>, maxDistance: f32) -> f32 {
  let first = (node.low - origin) * inverse;
  let second = (node.high - origin) * inverse;
  let near = min(first, second);
  let far = max(first, second);
  let enter = max(max(near.x, near.y), max(near.z, 0.0));
  let exit = min(min(far.x, far.y), min(far.z, maxDistance));

  return select(1e30, enter, enter <= exit);
}

/** The triangles of a leaf, against the closest hit so far. */
fn intersectLeaf(node: BvhNode, origin: vec3<f32>, direction: vec3<f32>, hit: ptr<function, Hit>) {
  let start = u32(-1 - node.link);

  for (var i = 0u; i < u32(node.count); i = i + 1u) {
    let triangle = orderedTriangle(start + i);
    let result = intersectTriangle(triangle, origin, direction);

    if (result.x > RAY_EPSILON && result.x < (*hit).distance) {
      (*hit).distance = result.x;
      (*hit).triangle = triangle;
      (*hit).u = result.y;
      (*hit).v = result.z;
    }
  }
}

/**
 * The closest hit along a ray. Both children of a node are tested together and the nearer one is visited first, so
 * the closest hit shrinks the search early and whole subtrees behind it are skipped; each node remembers where the ray
 * entered it, and one whose entry is already past the closest hit is dropped without being opened.
 */
fn traceScene(origin: vec3<f32>, direction: vec3<f32>, maxDistance: f32) -> Hit {
  var hit: Hit;
  hit.distance = maxDistance;
  hit.triangle = NO_HIT;
  hit.u = 0.0;
  hit.v = 0.0;

  if (frame.triangleCount == 0u) {
    return hit;
  }

  // A huge finite reciprocal instead of an infinite one: an axis aligned slab would otherwise produce 0 * inf.
  let signs = select(vec3<f32>(-1.0), vec3<f32>(1.0), direction >= vec3<f32>(0.0));
  let inverse = signs / max(abs(direction), vec3<f32>(1e-20));
  let rootEnter = enterBox(nodes[0], origin, inverse, hit.distance);

  if (rootEnter >= 1e30) {
    return hit;
  }

  var stack: array<u32, 32>;
  var enters: array<f32, 32>;
  var depth = 1u;
  stack[0] = 0u;
  enters[0] = rootEnter;

  loop {
    if (depth == 0u) {
      break;
    }

    depth = depth - 1u;

    // Something closer was found since this node was pushed.
    if (enters[depth] > hit.distance) {
      continue;
    }

    let node = nodes[stack[depth]];

    if (node.link < 0) {
      intersectLeaf(node, origin, direction, &hit);
      continue;
    }

    let leftIndex = u32(node.link);
    let rightIndex = u32(node.count);
    let leftEnter = enterBox(nodes[leftIndex], origin, inverse, hit.distance);
    let rightEnter = enterBox(nodes[rightIndex], origin, inverse, hit.distance);
    let leftHit = leftEnter < 1e30;
    let rightHit = rightEnter < 1e30;

    // The builder caps its depth so both children always fit; the guard keeps a malformed tree inside the stack.
    if (depth + 2u > 32u) {
      continue;
    }

    if (leftHit && rightHit) {
      // The farther first, so the nearer comes off the stack next.
      let leftNearer = leftEnter <= rightEnter;
      stack[depth] = select(leftIndex, rightIndex, leftNearer);
      enters[depth] = select(leftEnter, rightEnter, leftNearer);
      stack[depth + 1u] = select(rightIndex, leftIndex, leftNearer);
      enters[depth + 1u] = select(rightEnter, leftEnter, leftNearer);
      depth = depth + 2u;
    } else if (leftHit) {
      stack[depth] = leftIndex;
      enters[depth] = leftEnter;
      depth = depth + 1u;
    } else if (rightHit) {
      stack[depth] = rightIndex;
      enters[depth] = rightEnter;
      depth = depth + 1u;
    }
  }

  return hit;
}

/**
 * Whether anything at all lies along a ray before maxDistance, stopping at the first triangle it finds rather than the
 * closest: what a shadow ray asks when every surface it could meet blocks it. Returns that triangle, or NO_HIT.
 */
fn traceAny(origin: vec3<f32>, direction: vec3<f32>, maxDistance: f32) -> u32 {
  if (frame.triangleCount == 0u) {
    return NO_HIT;
  }

  let signs = select(vec3<f32>(-1.0), vec3<f32>(1.0), direction >= vec3<f32>(0.0));
  let inverse = signs / max(abs(direction), vec3<f32>(1e-20));

  if (enterBox(nodes[0], origin, inverse, maxDistance) >= 1e30) {
    return NO_HIT;
  }

  var stack: array<u32, 32>;
  var depth = 1u;
  stack[0] = 0u;

  loop {
    if (depth == 0u) {
      break;
    }

    depth = depth - 1u;

    let node = nodes[stack[depth]];

    if (node.link < 0) {
      let start = u32(-1 - node.link);

      for (var i = 0u; i < u32(node.count); i = i + 1u) {
        let triangle = orderedTriangle(start + i);
        let result = intersectTriangle(triangle, origin, direction);

        if (result.x > RAY_EPSILON && result.x < maxDistance) {
          return triangle;
        }
      }

      continue;
    }

    let leftIndex = u32(node.link);
    let rightIndex = u32(node.count);

    if (depth + 2u > 32u) {
      continue;
    }

    if (enterBox(nodes[leftIndex], origin, inverse, maxDistance) < 1e30) {
      stack[depth] = leftIndex;
      depth = depth + 1u;
    }

    if (enterBox(nodes[rightIndex], origin, inverse, maxDistance) < 1e30) {
      stack[depth] = rightIndex;
      depth = depth + 1u;
    }
  }

  return NO_HIT;
}
`;
