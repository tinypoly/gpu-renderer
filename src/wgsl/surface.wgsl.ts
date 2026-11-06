/**
 * What a hit actually is once the maps are read: a material resolved at one point on one triangle. The integrator
 * resolves a hit once and then asks the lobes about it, so no texture is read twice for the same point.
 *
 * The channel each map reads from and the way each value combines mirror Three's own shader chunks (r185), which the
 * CPU renderer follows too: roughness from green, metalness from blue, the alpha map from green, clearcoat from red,
 * its roughness from green, sheen roughness from alpha, and so on.
 */
export const SURFACE_WGSL = /* wgsl */ `
struct Surface {
  // The shading normal, maps applied, facing the ray.
  normal: vec3<f32>,
  albedo: vec3<f32>,
  // What the surface radiates by itself: its emission, or the whole colour of a material Three does not light.
  emissive: vec3<f32>,
  // The dielectric's reflectance at normal incidence and at grazing angles, before metalness turns it to the albedo.
  specularColor: vec3<f32>,
  specularF90: f32,
  roughness: f32,
  metalness: f32,
  // How much of the surface actually covers this point: a cutout, a transparency or a discarded fragment.
  opacity: f32,
  // Three's diffuseColor.a, which blending other than normal weighs the surface by.
  alpha: f32,
  flags: u32,
  kind: u32,
  // Share of the diffuse layer that lets light through, with metal already taken out of it.
  transmission: f32,
  ior: f32,
  thickness: f32,
  dispersion: f32,
  // What a unit of distance inside the volume absorbs, per channel: zero for a volume that never darkens.
  absorption: vec3<f32>,
  // The interpolated normal facing the ray, before any map: thin walls, shadow rays and the clearcoat read this one.
  facing: vec3<f32>,
  // Whether the ray arrived from the side the normals point to, which is the outside of a closed volume.
  front: bool,
  clearcoat: f32,
  clearcoatRoughness: f32,
  clearcoatNormal: vec3<f32>,
  sheenColor: vec3<f32>,
  sheenRoughness: f32,
  iridescence: f32,
  iridescenceFresnel: vec3<f32>,
  // Anisotropic GGX: its strength, the roughness along the tangent, and the tangent frame it stretches along.
  anisotropy: f32,
  alphaT: f32,
  anisotropyT: vec3<f32>,
  anisotropyB: vec3<f32>,
  shininess: f32,
  // Phong's specular map, which scales its highlight.
  specularStrength: f32,
  gradientMap: u32,
  // The material's own environment map, which the paths leaving it see instead of the scene's.
  envMap: u32,
  envMapIntensity: f32,
  // Basic, Lambert and Phong blend their environment map over the lit colour: what the map shows here, and how much.
  legacyEnv: vec3<f32>,
  legacyAmount: f32,
  combine: u32,
  // Light a light map bakes into this point, which only the camera sees: the path traces the rest itself.
  lightMap: vec3<f32>,
  blending: u32,
  material: u32,
  // Share of the diffuse layer that scatters inside the material instead of at its surface.
  subsurface: f32,
  // What a translucent surface lets through to its far side, diffusely: grass blades. Zero for everything else.
  translucent: vec3<f32>,
};

/** Barycentric weights of the two trailing vertices, for a point drawn uniformly over a triangle's area. */
fn triangleBarycentric(u: vec2<f32>) -> vec2<f32> {
  let root = sqrt(u.x);

  return vec2<f32>(1.0 - root, u.y * root);
}

fn trianglePointAt(triangle: u32, bary: vec2<f32>) -> vec3<f32> {
  return triangleVertex(triangle, 0u) * (1.0 - bary.x - bary.y)
    + triangleVertex(triangle, 1u) * bary.x
    + triangleVertex(triangle, 2u) * bary.y;
}

fn triangleUv(triangle: u32, bary: vec2<f32>, channel: u32) -> vec2<f32> {
  return attributeUv(triangle, 0u, channel) * (1.0 - bary.x - bary.y)
    + attributeUv(triangle, 1u, channel) * bary.x
    + attributeUv(triangle, 2u, channel) * bary.y;
}

fn triangleColor(triangle: u32, bary: vec2<f32>) -> vec4<f32> {
  return attributeColor(triangle, 0u) * (1.0 - bary.x - bary.y)
    + attributeColor(triangle, 1u) * bary.x
    + attributeColor(triangle, 2u) * bary.y;
}

/** A map slot, sampled through the UV set the texture asks for. Missing maps read as one, so a multiply is always safe. */
fn sampleMap(index: u32, triangle: u32, bary: vec2<f32>) -> vec4<f32> {
  if (index == NO_TEXTURE) {
    return vec4<f32>(1.0);
  }

  return sampleTextureAt(index, triangleUv(triangle, bary, textureInfo(index).channel));
}

/** Whether the ray sees the side Three calls the front: the winding, which packing keeps counter-clockwise. */
fn isFrontFace(triangle: u32, direction: vec3<f32>) -> bool {
  return dot(triangleNormal(triangle), direction) < 0.0;
}

/** Three culls the side a material does not draw. */
fn isCulled(material: u32, triangle: u32, direction: vec3<f32>) -> bool {
  let side = materials[material].side;
  let front = isFrontFace(triangle, direction);

  return (side == SIDE_FRONT && !front) || (side == SIDE_BACK && front);
}

/** What the triangle radiates at this point, its emissive map included so direct and indirect light agree on it. */
fn emissionAt(triangle: u32, bary: vec2<f32>) -> vec3<f32> {
  let m = materialIndexOf(triangle);

  // An unlit material emits its colour as the camera would see it, so a bounce ray finds what a light sample counted.
  if ((materials[m].flags & MATERIAL_UNLIT_LIGHT) != 0u) {
    var color = materials[m].color.rgb * sampleMap(materials[m].map, triangle, bary).rgb;

    if ((materials[m].flags & MATERIAL_VERTEX_COLORS) != 0u) {
      color = color * triangleColor(triangle, bary).rgb;
    }

    return color;
  }

  let map = sampleMap(materials[m].emissiveMap, triangle, bary);

  return materials[m].emissive.rgb * materials[m].emissive.w * map.rgb;
}

/** Three's material clipping planes: a point on the far side of one (or of all, with clipIntersection) is cut away. */
fn isClipped(m: u32, point: vec3<f32>) -> bool {
  let count = materials[m].clipCount;

  if (count == 0u) {
    return false;
  }

  let intersection = (materials[m].flags & MATERIAL_CLIP_INTERSECTION) != 0u;
  var every = true;

  for (var i = 0u; i < count; i = i + 1u) {
    let plane = extraVec4(materials[m].clipStart + i * 4u);
    let outside = dot(plane.xyz, point) + plane.w < 0.0;

    if (outside && !intersection) {
      return true;
    }

    every = every && outside;
  }

  return intersection && every;
}

/** How large one pixel is at this point, in world units: what a one pixel wide line covers. */
fn pixelFootprint(point: vec3<f32>) -> f32 {
  let scale = 2.0 / (frame.projection[1][1] * f32(frame.resolution.y));

  if (frame.cameraKind == 1u) {
    return scale;
  }

  return scale * max(-(frame.worldToCamera * vec4<f32>(point, 1.0)).z, 1e-6);
}

fn edgeDistance(point: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
  let edge = b - a;

  return length(cross(point - a, edge)) / max(length(edge), 1e-20);
}

/** A wireframe draws only the edges of each triangle, one pixel wide as WebGL draws every line. */
fn onWireframe(triangle: u32, point: vec3<f32>) -> bool {
  let a = triangleVertex(triangle, 0u);
  let b = triangleVertex(triangle, 1u);
  let c = triangleVertex(triangle, 2u);
  let nearest = min(edgeDistance(point, a, b), min(edgeDistance(point, b, c), edgeDistance(point, c, a)));

  return nearest <= 0.5 * pixelFootprint(point);
}

/** Three's diffuseColor.a: opacity, the colour map's alpha, the alpha map's green and the vertex alpha. */
fn materialAlpha(m: u32, triangle: u32, bary: vec2<f32>) -> f32 {
  return materialAlphaFrom(m, triangle, bary, sampleMap(materials[m].map, triangle, bary));
}

/** The same, from the colour map's texel at this point. Multiplied in the order resolveSurface multiplies it. */
fn materialAlphaFrom(m: u32, triangle: u32, bary: vec2<f32>, base: vec4<f32>) -> f32 {
  var alpha = materials[m].color.w * base.a * sampleMap(materials[m].alphaMap, triangle, bary).g;

  if ((materials[m].flags & MATERIAL_VERTEX_COLORS) != 0u) {
    alpha = alpha * triangleColor(triangle, bary).a;
  }

  return alpha;
}

/**
 * How much of the surface is there, from its alpha the way Three's chunks use it: alphaTest cuts, alphaHash keeps the
 * share, and only a transparent material with normal blending lets the rest through; an opaque one sets alpha to one.
 */
fn coverageOf(m: u32, alpha: f32) -> f32 {
  if (materials[m].alphaTest > 0.0 && alpha < materials[m].alphaTest) {
    return 0.0;
  }

  var coverage = 1.0;

  if ((materials[m].flags & MATERIAL_ALPHA_HASH) != 0u) {
    coverage = clamp(alpha, 0.0, 1.0);
  }

  if ((materials[m].flags & MATERIAL_TRANSPARENT) != 0u && materials[m].blending == BLEND_NORMAL) {
    coverage = coverage * clamp(alpha, 0.0, 1.0);
  }

  return coverage;
}

/** Barycentrics on the triangle's plane of a ray, extrapolated past its edges: where a neighbouring pixel lands. */
fn planeBarycentric(triangle: u32, origin: vec3<f32>, direction: vec3<f32>, fallback: vec2<f32>) -> vec2<f32> {
  let a = triangleVertex(triangle, 0u);
  let e1 = triangleVertex(triangle, 1u) - a;
  let e2 = triangleVertex(triangle, 2u) - a;
  let n = cross(e1, e2);
  let denominator = dot(n, direction);

  if (abs(denominator) < 1e-20) {
    return fallback;
  }

  let p = origin + direction * (dot(n, a - origin) / denominator) - a;
  let d00 = dot(e1, e1);
  let d01 = dot(e1, e2);
  let d11 = dot(e2, e2);
  let d20 = dot(p, e1);
  let d21 = dot(p, e2);
  let det = d00 * d11 - d01 * d01;

  if (abs(det) < 1e-30) {
    return fallback;
  }

  return vec2<f32>((d11 * d20 - d01 * d21) / det, (d00 * d21 - d01 * d20) / det);
}

/** The camera ray through a point given in normalized device coordinates: origin on the near plane, and its length. */
fn ndcRay(ndc: vec2<f32>) -> array<vec4<f32>, 2> {
  let nearView = frame.inverseProjection * vec4<f32>(ndc, -1.0, 1.0);
  let farView = frame.inverseProjection * vec4<f32>(ndc, 1.0, 1.0);
  let near = (frame.cameraToWorld * vec4<f32>(nearView.xyz / nearView.w, 1.0)).xyz;
  let far = (frame.cameraToWorld * vec4<f32>(farView.xyz / farView.w, 1.0)).xyz;
  let span = far - near;
  let extent = max(length(span), 1e-20);

  return array<vec4<f32>, 2>(vec4<f32>(near, 0.0), vec4<f32>(span / extent, extent));
}

/** What a fragment shader knows about the point it runs at, in GL's own terms. */
fn shaderInputAt(m: u32, triangle: u32, bary: vec2<f32>, direction: vec3<f32>) -> ShaderInput {
  var input: ShaderInput;
  let point = trianglePointAt(triangle, bary);
  let view = frame.worldToCamera * vec4<f32>(point, 1.0);
  let clip = frame.projection * view;
  let w = select(clip.w, 1e-20, abs(clip.w) < 1e-20);
  let ndc = clip.xyz / w;
  let size = vec2<f32>(frame.resolution);
  let pixel = (ndc.xy * 0.5 + 0.5) * size;
  let right = ndcRay(vec2<f32>(ndc.x + 2.0 / size.x, ndc.y));
  let above = ndcRay(vec2<f32>(ndc.x, ndc.y + 2.0 / size.y));

  input.data = materials[m].shaderData;
  input.material = m;
  input.triangle = triangle;
  input.frontFacing = isFrontFace(triangle, direction);
  input.bary = bary;
  input.baryDx = planeBarycentric(triangle, right[0].xyz, right[1].xyz, bary);
  input.baryDy = planeBarycentric(triangle, above[0].xyz, above[1].xyz, bary);
  input.fragCoord = vec4<f32>(pixel, ndc.z * 0.5 + 0.5, 1.0 / w);
  let uv = triangleUv(triangle, bary, 0u);
  input.uv = uv;
  input.fragColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  // A point's quad carries gl_PointCoord in its first UV set, with v running up as a texture's does.
  input.pointCoord = vec2<f32>(uv.x, 1.0 - uv.y);
  input.direction = direction;
  input.diffuseColor = vec4<f32>(1.0);
  input.normal = vec3<f32>(0.0, 0.0, 1.0);
  input.geometryNormal = vec3<f32>(0.0, 0.0, 1.0);
  input.roughness = 1.0;
  input.metalness = 0.0;
  input.emissive = vec3<f32>(0.0);
  input.viewPosition = -view.xyz;

  return input;
}

/** World to view space for a direction, which is the space Three's shaders keep their normals in. */
fn toView(direction: vec3<f32>) -> vec3<f32> {
  return normalize((frame.worldToCamera * vec4<f32>(direction, 0.0)).xyz);
}

fn toWorld(direction: vec3<f32>) -> vec3<f32> {
  return normalize((frame.cameraToWorld * vec4<f32>(direction, 0.0)).xyz);
}

/**
 * How much of the surface covers this point for a shadow ray: the same alpha, cut, clipping and wireframe the camera
 * sees, and a shader's own discard. Blending other than normal is handled by the caller.
 */
fn coverageAt(triangle: u32, bary: vec2<f32>, direction: vec3<f32>) -> f32 {
  let m = materialIndexOf(triangle);
  let flags = materials[m].flags;

  if (materials[m].clipCount > 0u || (flags & MATERIAL_WIREFRAME) != 0u) {
    let point = trianglePointAt(triangle, bary);

    if (isClipped(m, point)) {
      return 0.0;
    }

    if ((flags & MATERIAL_WIREFRAME) != 0u && !onWireframe(triangle, point)) {
      return 0.0;
    }
  }

  var alpha = materialAlpha(m, triangle, bary);

  if (materials[m].program != NO_PROGRAM) {
    var input = shaderInputAt(m, triangle, bary, direction);
    input.diffuseColor = vec4<f32>(materials[m].color.rgb, alpha);
    let output = runProgram(materials[m].program, input);

    if (output.discarded) {
      return 0.0;
    }

    alpha = select(output.color.a, output.diffuseColor.a, (flags & MATERIAL_PATCHED) != 0u);
  }

  return coverageOf(m, alpha);
}

/** A tangent frame from the triangle's own UVs, which is what Three's tangents amount to for a flat triangle. */
struct TangentFrame {
  tangent: vec3<f32>,
  bitangent: vec3<f32>,
  valid: bool,
};

fn tangentFrame(triangle: u32, channel: u32, normal: vec3<f32>) -> TangentFrame {
  var tangents: TangentFrame;
  tangents.valid = false;

  let p0 = triangleVertex(triangle, 0u);
  let edge1 = triangleVertex(triangle, 1u) - p0;
  let edge2 = triangleVertex(triangle, 2u) - p0;
  let uv0 = attributeUv(triangle, 0u, channel);
  let delta1 = attributeUv(triangle, 1u, channel) - uv0;
  let delta2 = attributeUv(triangle, 2u, channel) - uv0;
  let determinant = delta1.x * delta2.y - delta2.x * delta1.y;

  // A triangle with no UV area has no tangent frame to offer.
  if (abs(determinant) < 1e-16) {
    let basis = orthonormalBasis(normal);
    tangents.tangent = basis[0];
    tangents.bitangent = basis[1];

    return tangents;
  }

  let raw = (edge1 * delta2.y - edge2 * delta1.y) / determinant;
  // Made perpendicular to the shading normal, so a map tilts the interpolated normal rather than the flat one.
  let tangent = normalize(raw - normal * dot(normal, raw));

  if (!all(tangent == tangent)) {
    let basis = orthonormalBasis(normal);
    tangents.tangent = basis[0];
    tangents.bitangent = basis[1];

    return tangents;
  }

  tangents.tangent = tangent;
  tangents.bitangent = cross(normal, tangent) * select(-1.0, 1.0, determinant >= 0.0);
  tangents.valid = true;

  return tangents;
}

/** A tangent space normal map over a normal, scaled the way Three's normalScale scales it. */
fn tangentNormal(index: u32, triangle: u32, bary: vec2<f32>, normal: vec3<f32>, scale: vec2<f32>) -> vec3<f32> {
  if (index == NO_TEXTURE) {
    return normal;
  }

  let channel = textureInfo(index).channel;
  let tangents = tangentFrame(triangle, channel, normal);

  if (!tangents.valid) {
    return normal;
  }

  let texel = sampleTextureAt(index, triangleUv(triangle, bary, channel)).xyz * 2.0 - 1.0;
  let mapped = normalize(
    tangents.tangent * (texel.x * scale.x)
    + tangents.bitangent * (texel.y * scale.y)
    + normal * texel.z);

  return select(mapped, normal, !all(mapped == mapped));
}

/** The normal the material asks for: an object space or tangent space normal map, or else a bump map. */
fn mappedNormal(m: u32, triangle: u32, bary: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
  let normalMap = materials[m].normalMap;

  if (normalMap != NO_TEXTURE && (materials[m].flags & MATERIAL_OBJECT_NORMAL_MAP) != 0u) {
    let texel = sampleMap(normalMap, triangle, bary).xyz * 2.0 - 1.0;
    let world = normalize(vec3<f32>(
      dot(materials[m].normalRow0.xyz, texel),
      dot(materials[m].normalRow1.xyz, texel),
      dot(materials[m].normalRow2.xyz, texel)));

    // Seen from behind, a double sided surface turns its map around with it.
    return select(world, -world, dot(world, normal) < 0.0);
  }

  if (normalMap != NO_TEXTURE) {
    return tangentNormal(normalMap, triangle, bary, normal, vec2<f32>(materials[m].normalScaleX, materials[m].normalScaleY));
  }

  let bumpMap = materials[m].bumpMap;

  if (bumpMap != NO_TEXTURE) {
    // The CPU renderer's bump: the height's change over one texel, along the tangent frame of the map's UVs.
    let info = textureInfo(bumpMap);
    let channel = info.channel;
    let tangents = tangentFrame(triangle, channel, normal);
    let uv = triangleUv(triangle, bary, channel);
    let du = 1.0 / f32(max(info.size.x, 1u));
    let dv = 1.0 / f32(max(info.size.y, 1u));
    let scale = materials[m].bumpScale;
    let height = sampleTextureAt(bumpMap, uv).x;
    let dx = (sampleTextureAt(bumpMap, uv + vec2<f32>(du, 0.0)).x - height) * scale;
    let dy = (sampleTextureAt(bumpMap, uv + vec2<f32>(0.0, dv)).x - height) * scale;
    let bumped = normalize(normal - (tangents.tangent * dx + tangents.bitangent * dy));

    return select(bumped, normal, !all(bumped == bumped));
  }

  return normal;
}

/** The interpolated vertex normal as the mesh authored it, pointing out of a closed volume. */
fn authoredNormal(triangle: u32, bary: vec2<f32>) -> vec3<f32> {
  let base = triangle * 9u;
  let n0 = vec3<f32>(attributes[base], attributes[base + 1u], attributes[base + 2u]);
  let n1 = vec3<f32>(attributes[base + 3u], attributes[base + 4u], attributes[base + 5u]);
  let n2 = vec3<f32>(attributes[base + 6u], attributes[base + 7u], attributes[base + 8u]);

  return normalize(n0 * (1.0 - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);
}

/** The interpolated vertex normal, or the face's for flat shading, turned to face the ray. */
fn interpolatedNormal(m: u32, triangle: u32, bary: vec2<f32>, direction: vec3<f32>) -> vec3<f32> {
  var normal = authoredNormal(triangle, bary);

  if ((materials[m].flags & MATERIAL_FLAT) != 0u || !all(normal == normal)) {
    normal = triangleNormal(triangle);
  }

  return select(normal, -normal, dot(normal, direction) > 0.0);
}

/** Three's transmission at this point: the factor, its map's red channel, and none of it on metal. */
fn transmissionAt(m: u32, triangle: u32, bary: vec2<f32>, metalness: f32) -> f32 {
  if (materials[m].transmission <= 0.0) {
    return 0.0;
  }

  let map = sampleMap(materials[m].transmissionMap, triangle, bary).r;

  return clamp(materials[m].transmission * map, 0.0, 1.0) * (1.0 - metalness);
}

/** Beer-Lambert: the colour reached at the attenuation distance fixes how fast each channel fades. */
fn absorptionOf(m: u32) -> vec3<f32> {
  let attenuation = materials[m].attenuation;

  if (attenuation.w <= 0.0) {
    return vec3<f32>(0.0);
  }

  return -log(max(attenuation.rgb, vec3<f32>(1e-6))) / attenuation.w;
}

/** Three's evalSensitivity: the XYZ response of a thin film, Fourier-fitted, in linear sRGB. */
fn filmSensitivity(opd: f32, shift: vec3<f32>) -> vec3<f32> {
  let phase = 2.0 * PI * opd * 1.0e-9;
  let value = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
  let position = vec3<f32>(1.6810e+06, 1.7953e+06, 2.2084e+06);
  let variance = vec3<f32>(4.3278e+09, 9.3046e+09, 6.6121e+09);
  var xyz = value * sqrt(2.0 * PI * variance) * cos(position * phase + shift) * exp(-phase * phase * variance);

  xyz.x = xyz.x + 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x)
    * exp(-4.5282e+09 * phase * phase);
  xyz = xyz / 1.0685e-7;

  return mat3x3<f32>(
    vec3<f32>(3.2404542, -0.9692660, 0.0556434),
    vec3<f32>(-1.5371385, 1.8760108, -0.2040259),
    vec3<f32>(-0.4985314, 0.0415560, 1.0572252)) * xyz;
}

fn schlick3(f0: vec3<f32>, f90: f32, cosine: f32) -> vec3<f32> {
  let t = 1.0 - cosine;
  let k = t * t * t * t * t;

  return f0 + (vec3<f32>(f90) - f0) * k;
}

/** Three's evalIridescence (Belcour and Barla 2017): the Fresnel of a thin film over a base of reflectance baseF0. */
fn evalIridescence(filmIor: f32, cosTheta1: f32, thickness: f32, baseF0: vec3<f32>) -> vec3<f32> {
  let ior = mix(1.0, filmIor, smoothstep(0.0, 0.03, thickness));
  let sinTheta2Sq = (1.0 / ior) * (1.0 / ior) * (1.0 - cosTheta1 * cosTheta1);
  let cosTheta2Sq = 1.0 - sinTheta2Sq;

  if (cosTheta2Sq < 0.0) {
    return vec3<f32>(1.0);
  }

  let cosTheta2 = sqrt(cosTheta2Sq);
  let r0 = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
  let r12 = r0 + (1.0 - r0) * pow(1.0 - cosTheta1, 5.0);
  let t121 = 1.0 - r12;
  let phi12 = select(0.0, PI, ior < 1.0);
  let phi21 = PI - phi12;
  let root = sqrt(clamp(baseF0, vec3<f32>(0.0), vec3<f32>(0.9999)));
  let baseIor = (vec3<f32>(1.0) + root) / (vec3<f32>(1.0) - root);
  let r1 = ((baseIor - vec3<f32>(ior)) / (baseIor + vec3<f32>(ior))) * ((baseIor - vec3<f32>(ior)) / (baseIor + vec3<f32>(ior)));
  let r23 = schlick3(r1, 1.0, cosTheta2);
  let phi23 = select(vec3<f32>(0.0), vec3<f32>(PI), baseIor < vec3<f32>(ior));
  let opd = 2.0 * ior * thickness * cosTheta2;
  let phi = vec3<f32>(phi21) + phi23;
  let r123 = clamp(r12 * r23, vec3<f32>(1e-5), vec3<f32>(0.9999));
  let r123Root = sqrt(r123);
  let rs = t121 * t121 * r23 / (vec3<f32>(1.0) - r123);
  var intensity = vec3<f32>(r12) + rs;
  var cm = rs - vec3<f32>(t121);

  for (var m = 1; m <= 2; m = m + 1) {
    cm = cm * r123Root;
    intensity = intensity + cm * 2.0 * filmSensitivity(f32(m) * opd, f32(m) * phi);
  }

  return max(intensity, vec3<f32>(0.0));
}

/** Three's toon ramp: the gradient map across dot(n, l) from -1 to 1, or a single step at 0.7 without one. */
fn toonRamp(gradientMap: u32, cosine: f32) -> f32 {
  let coordinate = cosine * 0.5 + 0.5;

  if (gradientMap == NO_TEXTURE) {
    return select(0.7, 1.0, coordinate >= 0.7);
  }

  return sampleTextureRaw(gradientMap, vec2<f32>(coordinate, 0.0)).r;
}

/**
 * Whether a material's coverage can be told from its maps alone, before the rest of the surface is read: no shader that
 * might rewrite the alpha or discard, normal blending, and something that makes it partial at all.
 */
fn coverageFirst(m: u32) -> bool {
  let flags = materials[m].flags;
  let blending = materials[m].blending;

  return materials[m].program == NO_PROGRAM
    && (blending == BLEND_NORMAL || blending == BLEND_NONE)
    && (materials[m].alphaTest > 0.0 || (flags & (MATERIAL_ALPHA_HASH | MATERIAL_TRANSPARENT)) != 0u);
}

/** Everything the lobes need about one point of one triangle. */
fn resolveSurface(triangle: u32, bary: vec2<f32>, direction: vec3<f32>) -> Surface {
  let m = materialIndexOf(triangle);

  return resolveSurfaceFrom(triangle, bary, direction, sampleMap(materials[m].map, triangle, bary));
}

/** The same, from the colour map's texel at this point, for a caller that already read it. */
fn resolveSurfaceFrom(triangle: u32, bary: vec2<f32>, direction: vec3<f32>, base: vec4<f32>) -> Surface {
  let m = materialIndexOf(triangle);
  let flags = materials[m].flags;
  let kind = materials[m].kind;
  let facing = interpolatedNormal(m, triangle, bary, direction);
  let view = -direction;
  var surface: Surface;

  surface.material = m;
  surface.flags = flags;
  surface.kind = kind;
  surface.blending = materials[m].blending;
  surface.facing = facing;
  surface.front = dot(authoredNormal(triangle, bary), direction) < 0.0;
  surface.normal = mappedNormal(m, triangle, bary, facing);

  // A normal map may tilt past the horizon, where the surface would be lit from behind itself.
  if (dot(surface.normal, view) <= 0.0) {
    surface.normal = facing;
  }

  var albedo = materials[m].color.rgb * base.rgb;
  var alpha = materials[m].color.w * base.a * sampleMap(materials[m].alphaMap, triangle, bary).g;

  if ((flags & MATERIAL_VERTEX_COLORS) != 0u) {
    let color = triangleColor(triangle, bary);
    albedo = albedo * color.rgb;
    alpha = alpha * color.a;
  }

  var roughness = materials[m].specular.w * sampleMap(materials[m].roughnessMap, triangle, bary).g;
  var metalness = materials[m].metalness * sampleMap(materials[m].metalnessMap, triangle, bary).b;
  var emissive = vec3<f32>(0.0);

  // An unlit light's colour is added as what it shows, below; emissionAt gives the same colour to light samples.
  if ((flags & MATERIAL_UNLIT_LIGHT) == 0u) {
    emissive = emissionAt(triangle, bary);
  }

  surface.opacity = 1.0;

  // A built-in material's hooks run on its values before lighting, in Three's view space; a shader material's output
  // is the colour itself.
  if (materials[m].program != NO_PROGRAM) {
    var input = shaderInputAt(m, triangle, bary, direction);
    input.diffuseColor = vec4<f32>(albedo, alpha);
    input.normal = toView(surface.normal);
    input.geometryNormal = toView(facing);
    input.roughness = roughness;
    input.metalness = metalness;
    input.emissive = emissive;
    let output = runProgram(materials[m].program, input);

    if (output.discarded) {
      surface.opacity = 0.0;
    }

    if ((flags & MATERIAL_PATCHED) != 0u) {
      albedo = output.diffuseColor.rgb;
      alpha = output.diffuseColor.a;
      roughness = output.roughness;
      metalness = output.metalness;
      emissive = output.emissive;
      let normal = toWorld(output.normal);

      if (all(normal == normal)) {
        surface.normal = select(normal, -normal, dot(normal, view) < 0.0);
      }
    } else {
      albedo = output.color.rgb;
      alpha = output.color.a;
    }
  }

  surface.opacity = surface.opacity * coverageOf(m, alpha);
  surface.alpha = alpha;
  surface.albedo = albedo;
  surface.roughness = clamp(roughness, 0.0525, 1.0);
  surface.metalness = clamp(metalness, 0.0, 1.0);
  surface.emissive = emissive;
  surface.transmission = transmissionAt(m, triangle, bary, surface.metalness);
  surface.ior = max(materials[m].ior, 1.0);
  surface.thickness = materials[m].thickness * sampleMap(materials[m].thicknessMap, triangle, bary).g;
  surface.dispersion = materials[m].dispersion;
  surface.subsurface = select(0.0, materials[m].subsurface.w, frame.subsurfaceEnabled != 0u);
  surface.absorption = absorptionOf(m);
  surface.shininess = materials[m].shininess;
  surface.specularStrength = sampleMap(materials[m].specularMap, triangle, bary).r;
  surface.gradientMap = materials[m].gradientMap;
  surface.envMap = NO_TEXTURE;
  surface.envMapIntensity = materials[m].envMapIntensity;
  surface.legacyAmount = 0.0;
  surface.legacyEnv = vec3<f32>(0.0);
  surface.combine = materials[m].combine;
  surface.lightMap = vec3<f32>(0.0);
  surface.clearcoat = 0.0;
  surface.iridescence = 0.0;
  surface.anisotropy = 0.0;
  surface.sheenColor = vec3<f32>(0.0);
  surface.specularColor = materials[m].specular.rgb;
  surface.specularF90 = 1.0;

  if ((flags & MATERIAL_PBR) != 0u) {
    if ((flags & MATERIAL_SPECULAR) != 0u) {
      // A physical material: the index of refraction, the specular colour and intensity and their maps.
      let intensity = materials[m].specularIntensity * sampleMap(materials[m].specularIntensityMap, triangle, bary).a;
      let color = materials[m].specular.rgb * sampleMap(materials[m].specularColorMap, triangle, bary).rgb;

      surface.specularColor = min(materials[m].dielectric * color, vec3<f32>(1.0)) * intensity;
      surface.specularF90 = mix(intensity, 1.0, surface.metalness);
    } else {
      // Three's standard material has no index of refraction and reflects 0.04.
      surface.specularColor = vec3<f32>(0.04);
    }

    surface.envMap = materials[m].envMap;

    if (materials[m].clearcoat > 0.0) {
      surface.clearcoat = clamp(materials[m].clearcoat * sampleMap(materials[m].clearcoatMap, triangle, bary).r, 0.0, 1.0);
      surface.clearcoatRoughness = clamp(
        materials[m].clearcoatRoughness * sampleMap(materials[m].clearcoatRoughnessMap, triangle, bary).g, 0.0525, 1.0);
      // The coat ignores the base's normal map: it has its own, over the interpolated normal.
      surface.clearcoatNormal = tangentNormal(materials[m].clearcoatNormalMap, triangle, bary, facing,
        vec2<f32>(materials[m].clearcoatNormalScaleX, materials[m].clearcoatNormalScaleY));

      if (dot(surface.clearcoatNormal, view) <= 0.0) {
        surface.clearcoatNormal = facing;
      }
    }

    if (any(materials[m].sheen.rgb > vec3<f32>(0.0))) {
      let map = sampleMap(materials[m].sheenRoughnessMap, triangle, bary).a;
      surface.sheenColor = materials[m].sheen.rgb * sampleMap(materials[m].sheenColorMap, triangle, bary).rgb;
      surface.sheenRoughness = clamp(materials[m].sheen.w, 0.0001, 1.0) * map;
    }

    if (materials[m].iridescence > 0.0) {
      var thickness = materials[m].iridescenceThicknessMax;

      if (materials[m].iridescenceThicknessMap != NO_TEXTURE) {
        thickness = mix(materials[m].iridescenceThicknessMin, materials[m].iridescenceThicknessMax,
          sampleMap(materials[m].iridescenceThicknessMap, triangle, bary).g);
      }

      if (thickness > 0.0) {
        surface.iridescence = clamp(
          materials[m].iridescence * sampleMap(materials[m].iridescenceMap, triangle, bary).r, 0.0, 1.0);

        let nv = clamp(dot(surface.normal, view), 0.0, 1.0);
        surface.iridescenceFresnel = mix(
          evalIridescence(materials[m].iridescenceIor, nv, thickness, surface.specularColor),
          evalIridescence(materials[m].iridescenceIor, nv, thickness, albedo),
          surface.metalness);
      }
    }

    if (materials[m].anisotropy != 0.0) {
      let rotation = materials[m].anisotropyRotation;
      var vector = vec2<f32>(cos(rotation), sin(rotation)) * materials[m].anisotropy;
      let channel = select(0u, textureInfo(materials[m].anisotropyMap).channel, materials[m].anisotropyMap != NO_TEXTURE);
      let tangents = tangentFrame(triangle, channel, surface.normal);

      if (materials[m].anisotropyMap != NO_TEXTURE) {
        let polar = sampleMap(materials[m].anisotropyMap, triangle, bary).rgb;
        let turn = mat2x2<f32>(vector.x, vector.y, -vector.y, vector.x);
        vector = turn * normalize(2.0 * polar.rg - vec2<f32>(1.0)) * polar.b;
      }

      let strength = length(vector);
      let along = select(vector / max(strength, 1e-20), vec2<f32>(1.0, 0.0), strength == 0.0);

      surface.anisotropy = clamp(strength, 0.0, 1.0);
      surface.alphaT = mix(surface.roughness * surface.roughness, 1.0, surface.anisotropy * surface.anisotropy);
      surface.anisotropyT = tangents.tangent * along.x + tangents.bitangent * along.y;
      surface.anisotropyB = tangents.bitangent * along.x - tangents.tangent * along.y;
    }
  } else if (materials[m].envMap != NO_TEXTURE
    && (kind == KIND_BASIC || kind == KIND_LAMBERT || kind == KIND_PHONG)) {
    // Basic, Lambert and Phong blend their own map over the lit colour, mirrored or bent the way the map says.
    let info = textureInfo(materials[m].envMap);
    let bent = refract(direction, surface.normal, materials[m].refractionRatio);
    let lookup = select(reflect(direction, surface.normal), bent, (info.flags & 16u) != 0u && any(bent != vec3<f32>(0.0)));

    surface.legacyEnv = sampleEquirect(materials[m].envMap, lookup).rgb * materials[m].envMapIntensity;
    surface.legacyAmount = clamp(materials[m].reflectivity * surface.specularStrength, 0.0, 1.0);
  }

  if (materials[m].lightMap != NO_TEXTURE) {
    let baked = sampleMap(materials[m].lightMap, triangle, bary).rgb * materials[m].lightMapIntensity;
    surface.lightMap = baked * albedo * (1.0 - surface.metalness) / PI;
  }

  // A material Three does not light reads as its own colour, which in a path tracer is the same as emitting it.
  if ((flags & MATERIAL_LIT) == 0u) {
    var color = albedo;

    if (kind == KIND_NORMAL) {
      color = toView(surface.normal) * 0.5 + 0.5;
    } else if (kind == KIND_DEPTH) {
      let clip = frame.projection * (frame.worldToCamera * vec4<f32>(trianglePointAt(triangle, bary), 1.0));
      color = vec3<f32>(1.0 - (clip.z / clip.w * 0.5 + 0.5));
    } else if (kind == KIND_MATCAP) {
      let point = trianglePointAt(triangle, bary);
      let viewDirection = normalize(-(frame.worldToCamera * vec4<f32>(point, 1.0)).xyz);
      let x = normalize(vec3<f32>(viewDirection.z, 0.0, -viewDirection.x));
      let y = cross(viewDirection, x);
      let normal = toView(surface.normal);
      let uv = vec2<f32>(dot(x, normal), dot(y, normal)) * 0.495 + 0.5;
      var matcap = vec3<f32>(mix(0.2, 0.8, uv.y));

      if (materials[m].matcap != NO_TEXTURE) {
        matcap = sampleTextureRaw(materials[m].matcap, uv).rgb;
      }

      color = albedo * matcap;
    } else if (kind == KIND_BASIC && materials[m].lightMap != NO_TEXTURE) {
      // A basic material with a light map shows the map's light instead of full brightness.
      color = albedo * sampleMap(materials[m].lightMap, triangle, bary).rgb * materials[m].lightMapIntensity / PI;
    }

    if (surface.legacyAmount > 0.0) {
      let amount = surface.legacyAmount;

      if (surface.combine == 1u) {
        color = mix(color, surface.legacyEnv, amount);
      } else if (surface.combine == 2u) {
        color = color + surface.legacyEnv * amount;
      } else {
        color = mix(color, color * surface.legacyEnv, amount);
      }

      surface.legacyAmount = 0.0;
    }

    surface.emissive = surface.emissive + color;
    surface.lightMap = vec3<f32>(0.0);
  }

  return surface;
}
`;
