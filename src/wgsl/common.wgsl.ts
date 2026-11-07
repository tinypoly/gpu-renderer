import { FRAME_LAYOUT } from "../frameLayout.js";
import { MATERIAL_KIND, MATERIAL_LAYOUT } from "../materialLayout.js";
import { SOBOL_TABLE, SRGB_TABLE, TEXTURE_TABLE } from "../texturePack.js";

/** Bindings and structs shared by every stage of the path tracing module. */
export const COMMON_WGSL = /* wgsl */ `
${FRAME_LAYOUT.wgsl}
// Sixty four bytes: the boxes of a node's two children, each a corner and a link, then a corner and a count. A link of
// zero or more is the child's own node; a leaf stores -1 - start in it, into the hierarchy's leaf order. See bvh.ts.
struct BvhNode {
  leftLow: vec3<f32>,
  leftLink: i32,
  leftHigh: vec3<f32>,
  leftCount: i32,
  rightLow: vec3<f32>,
  rightLink: i32,
  rightHigh: vec3<f32>,
  rightCount: i32,
};

/** As materialLayout.ts packs it. A map index of NO_TEXTURE means the slot is empty. */
${MATERIAL_LAYOUT.wgsl}
/** Sixty four bytes, as texturePack.ts writes it. */
struct TextureInfo {
  size: vec2<u32>,
  // Where the texels start, in words for a byte texture and in floats for a float one.
  offset: u32,
  flags: u32,
  wrapS: u32,
  wrapT: u32,
  // Which of the geometry's UV sets this texture reads.
  channel: u32,
  padding: u32,
  // The rows of Three's own UV transform: xyz against (u, v, 1).
  rowU: vec4<f32>,
  rowV: vec4<f32>,
};

/**
 * Something the integrator can aim a shadow ray at. The emissive triangles come first and carry only a triangle index,
 * their emission being read from its material; the analytic lights that follow carry Three's own parameters.
 *
 * pdf is the density with which this entry is chosen, in the measure that suits its kind: per unit area for a surface,
 * and simply the probability of the pick for a light that occupies no area.
 */
struct Light {
  kind: u32,
  cdf: f32,
  pdf: f32,
  // The triangle of an emissive entry; for a spot light with a projected map, where that map's record starts.
  triangle: u32,
  color: vec3<f32>,
  // Three's light.distance: beyond it a point or spot light delivers nothing. Zero means no cutoff.
  cutoff: f32,
  position: vec3<f32>,
  decay: f32,
  // Towards the light from its target, as Three orients a directional or spot light.
  direction: vec3<f32>,
  outerCos: f32,
  // Half the width of a rectangular light, along its own axes.
  tangent: vec3<f32>,
  innerCos: f32,
  bitangent: vec3<f32>,
  flags: u32,
};

struct Hit {
  distance: f32,
  triangle: u32,
  u: f32,
  v: f32,
};

const NO_HIT: u32 = 0xffffffffu;
const RAY_EPSILON: f32 = 1e-4;
const PI: f32 = 3.141592653589793;

const LIGHT_AREA: u32 = 0u;
const LIGHT_DIRECTIONAL: u32 = 1u;
const LIGHT_POINT: u32 = 2u;
const LIGHT_SPOT: u32 = 3u;
const LIGHT_RECT: u32 = 4u;
const LIGHT_ENVIRONMENT: u32 = 5u;
const LIGHT_CASTS_SHADOW: u32 = 1u;
// A spot light that projects a texture; its triangle field says where the map and its matrix are.
const LIGHT_HAS_MAP: u32 = 2u;

// A surface Three shades with light at all: a basic, points, line or sprite material is not one.
const MATERIAL_LIT: u32 = 1u;
// A surface with Three's metalness workflow, which is the only one that gets a GGX specular lobe.
const MATERIAL_PBR: u32 = 2u;
const MATERIAL_TRANSPARENT: u32 = 4u;
const MATERIAL_VERTEX_COLORS: u32 = 8u;
const MATERIAL_FLAT: u32 = 16u;
const MATERIAL_WIREFRAME: u32 = 32u;
const MATERIAL_FOG: u32 = 64u;
const MATERIAL_CLIP_INTERSECTION: u32 = 128u;
const MATERIAL_OBJECT_NORMAL_MAP: u32 = 256u;
const MATERIAL_PATCHED: u32 = 512u;
const MATERIAL_ALPHA_HASH: u32 = 2048u;
const MATERIAL_SPECULAR: u32 = 4096u;
// The mesh only bounds a medium: its surface is invisible and the space inside scatters light.
const MATERIAL_VOLUME: u32 = 8192u;
// An unlit material of a fixed colour, which direct sampling aims at: it emits what it shows.
const MATERIAL_UNLIT_LIGHT: u32 = 16384u;
const MATERIAL_WATER: u32 = 32768u;

const KIND_STANDARD: u32 = ${MATERIAL_KIND.standard}u;
const KIND_PHYSICAL: u32 = ${MATERIAL_KIND.physical}u;
const KIND_LAMBERT: u32 = ${MATERIAL_KIND.lambert}u;
const KIND_PHONG: u32 = ${MATERIAL_KIND.phong}u;
const KIND_TOON: u32 = ${MATERIAL_KIND.toon}u;
const KIND_BASIC: u32 = ${MATERIAL_KIND.basic}u;
const KIND_NORMAL: u32 = ${MATERIAL_KIND.normal}u;
const KIND_DEPTH: u32 = ${MATERIAL_KIND.depth}u;
const KIND_MATCAP: u32 = ${MATERIAL_KIND.matcap}u;
const KIND_SHADOW: u32 = ${MATERIAL_KIND.shadow}u;
const KIND_SHADER: u32 = ${MATERIAL_KIND.shader}u;

// Three's side and blending constants.
const SIDE_FRONT: u32 = 0u;
const SIDE_BACK: u32 = 1u;
const BLEND_NONE: u32 = 0u;
const BLEND_NORMAL: u32 = 1u;
const BLEND_ADDITIVE: u32 = 2u;
const BLEND_SUBTRACTIVE: u32 = 3u;
const BLEND_MULTIPLY: u32 = 4u;

/** What a material's map index reads as when it has no such map. */
const NO_TEXTURE: u32 = 0xffffffffu;
const NO_PROGRAM: u32 = 0xffffffffu;

// The lookup tables that lead the texture buffer, and where its table of textures starts after them: see texturePack.ts.
const SOBOL_TABLE: u32 = ${SOBOL_TABLE}u;
const SRGB_TABLE: u32 = ${SRGB_TABLE}u;
const TEXTURE_TABLE: u32 = ${TEXTURE_TABLE}u;

// A WebGPU device only has to offer eight storage buffers per stage, so arrays of the same kind share one binding and
// the uniform says where the second one starts. See sceneUpload.ts, which joins them.
@group(0) @binding(0) var<uniform> frame: Frame;
// Nine coordinates per stored triangle, in the hierarchy's leaf order: see leafOf in bvh.wgsl.ts. A prototype's are
// in object space.
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// Per vertex: the UV sets from frame.uvOffset and vertex colours from frame.colorOffset; then shader varyings, per
// corner, from frame.varyingOffset.
@group(0) @binding(2) var<storage, read> attributes: array<f32>;
// One material index per triangle a hit can name, then the hierarchy's triangle order from frame.orderOffset on (the
// stored triangle at each position of the leaf order), then from frame.leafOffset on the other way (the position of
// each stored triangle), the instance of every instanced triangle from frame.instanceOffset, the three vertices of each
// stored triangle from frame.cornerOffset and every vertex's folded normal from frame.normalOffset.
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;
@group(0) @binding(5) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(6) var<storage, read> lights: array<Light>;
// Radiance and sample count per pixel, then the coverage per pixel from frame.pixelCount on.
@group(0) @binding(7) var<storage, read_write> accumulation: array<vec4<f32>>;
// The texture table, then the texels it points at, then everything else that is read by offset: the environment's
// distributions, clipping planes, projected light maps and shader uniforms. See texturePack.ts.
@group(0) @binding(8) var<storage, read> textures: array<u32>;

fn materialIndexOf(triangle: u32) -> u32 {
  return indices[triangle];
}

/** The triangle the hierarchy holds at this position of its order. */
fn orderedTriangle(at: u32) -> u32 {
  return indices[frame.orderOffset + at];
}

/** What a triangle of the scene's own belongs to: no instance. */
const NO_INSTANCE: u32 = 0xffffffffu;

/** The instance a triangle a hit named belongs to, or NO_INSTANCE for one of the scene's own. */
fn instanceOf(triangle: u32) -> u32 {
  if (!INSTANCES || triangle < frame.sceneTriangles) {
    return NO_INSTANCE;
  }

  return indices[frame.instanceOffset + triangle - frame.sceneTriangles];
}

/** One word of an instance's record: see INSTANCE_WORDS in instancing.ts. */
fn instanceWord(instance: u32, word: u32) -> u32 {
  return textures[frame.instanceData + instance * 28u + word];
}

/** A row of an instance's transforms: 0 to 2 take object space to world, 3 to 5 back. */
fn instanceRow(instance: u32, row: u32) -> vec4<f32> {
  let at = frame.instanceData + instance * 28u + row * 4u;

  return vec4<f32>(bitcast<f32>(textures[at]), bitcast<f32>(textures[at + 1u]), bitcast<f32>(textures[at + 2u]),
    bitcast<f32>(textures[at + 3u]));
}

/** Where a named triangle is stored: itself, or for an instance's, the prototype's triangle it repeats. */
fn storedTriangle(triangle: u32, instance: u32) -> u32 {
  if (instance == NO_INSTANCE) {
    return triangle;
  }

  return instanceWord(instance, 25u) + triangle - instanceWord(instance, 26u);
}

/** A point of an instance's prototype, in world space. */
fn instancePoint(instance: u32, p: vec3<f32>) -> vec3<f32> {
  let q = vec4<f32>(p, 1.0);

  return vec3<f32>(dot(instanceRow(instance, 0u), q), dot(instanceRow(instance, 1u), q), dot(instanceRow(instance, 2u), q));
}

/** A world space point (w 1) or direction (w 0), in an instance's object space. */
fn instanceToObject(instance: u32, v: vec3<f32>, w: f32) -> vec3<f32> {
  let q = vec4<f32>(v, w);

  return vec3<f32>(dot(instanceRow(instance, 3u), q), dot(instanceRow(instance, 4u), q), dot(instanceRow(instance, 5u), q));
}

/** An object space normal in world space, by the inverse transpose; not normalized. */
fn instanceNormal(instance: u32, n: vec3<f32>) -> vec3<f32> {
  return n.x * instanceRow(instance, 3u).xyz + n.y * instanceRow(instance, 4u).xyz + n.z * instanceRow(instance, 5u).xyz;
}

/** The three vertices of a triangle a hit named. */
fn triangleCorners(triangle: u32) -> vec3<u32> {
  let at = frame.cornerOffset + storedTriangle(triangle, instanceOf(triangle)) * 3u;

  return vec3<u32>(indices[at], indices[at + 1u], indices[at + 2u]);
}

/** One UV set of a vertex: four components, two per set. */
fn vertexUv(vertex: u32, channel: u32) -> vec2<f32> {
  let at = frame.uvOffset + vertex * 4u + channel * 2u;

  return vec2<f32>(attributes[at], attributes[at + 1u]);
}

/** A vertex's colour, or white when the scene has none. */
fn vertexColor(vertex: u32) -> vec4<f32> {
  if (frame.hasColors == 0u) {
    return vec4<f32>(1.0);
  }

  let at = frame.colorOffset + vertex * 4u;

  return vec4<f32>(attributes[at], attributes[at + 1u], attributes[at + 2u], attributes[at + 3u]);
}

/** A normal folded onto an octahedron in two signed 16 bit halves, as sceneUpload.ts folds it; zero for NO_NORMAL. */
fn vertexNormal(vertex: u32) -> vec3<f32> {
  let code = indices[frame.normalOffset + vertex];

  if (code == 0x80008000u) {
    return vec3<f32>(0.0);
  }

  let e = unpack2x16snorm(code);
  var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);

  n.x = n.x + select(t, -t, n.x >= 0.0);
  n.y = n.y + select(t, -t, n.y >= 0.0);

  return normalize(n);
}

fn extraFloat(at: u32) -> f32 {
  return bitcast<f32>(textures[at]);
}

fn extraVec4(at: u32) -> vec4<f32> {
  return vec4<f32>(extraFloat(at), extraFloat(at + 1u), extraFloat(at + 2u), extraFloat(at + 3u));
}

/** One entry of the texture table, unpacked from the words it shares with the texels. */
fn textureInfo(index: u32) -> TextureInfo {
  let at = TEXTURE_TABLE + index * 16u;
  var info: TextureInfo;

  info.size = vec2<u32>(textures[at], textures[at + 1u]);
  info.offset = textures[at + 2u];
  info.flags = textures[at + 3u];
  info.wrapS = textures[at + 4u];
  info.wrapT = textures[at + 5u];
  info.channel = textures[at + 6u];
  info.rowU = vec4<f32>(
    bitcast<f32>(textures[at + 8u]), bitcast<f32>(textures[at + 9u]), bitcast<f32>(textures[at + 10u]), 0.0);
  info.rowV = vec4<f32>(
    bitcast<f32>(textures[at + 12u]), bitcast<f32>(textures[at + 13u]), bitcast<f32>(textures[at + 14u]), 0.0);

  return info;
}
`;
