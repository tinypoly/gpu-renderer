import { FRAME_LAYOUT } from "../frameLayout.js";
import { MATERIAL_KIND, MATERIAL_LAYOUT } from "../materialLayout.js";

/** Bindings and structs shared by every stage of the path tracing module. */
export const COMMON_WGSL = /* wgsl */ `
${FRAME_LAYOUT.wgsl}
// Thirty two bytes: a corner and a link, twice. A leaf stores -1 - start in place of its left child.
struct BvhNode {
  low: vec3<f32>,
  link: i32,
  high: vec3<f32>,
  count: i32,
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

// A WebGPU device only has to offer eight storage buffers per stage, so arrays of the same kind share one binding and
// the uniform says where the second one starts. See sceneUpload.ts, which joins them.
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// Normals, then the UV sets from frame.uvOffset, vertex colours from frame.colorOffset and shader varyings from
// frame.varyingOffset.
@group(0) @binding(2) var<storage, read> attributes: array<f32>;
// One material index per triangle, then the hierarchy's triangle order from frame.orderOffset on.
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

/** One word of a UV set: three vertices of four components, two per set. */
fn attributeUv(triangle: u32, corner: u32, channel: u32) -> vec2<f32> {
  let at = frame.uvOffset + triangle * 12u + corner * 4u + channel * 2u;

  return vec2<f32>(attributes[at], attributes[at + 1u]);
}

/** A corner's vertex colour, or white when the scene has none. */
fn attributeColor(triangle: u32, corner: u32) -> vec4<f32> {
  if (frame.hasColors == 0u) {
    return vec4<f32>(1.0);
  }

  let at = frame.colorOffset + triangle * 12u + corner * 4u;

  return vec4<f32>(attributes[at], attributes[at + 1u], attributes[at + 2u], attributes[at + 3u]);
}

fn extraFloat(at: u32) -> f32 {
  return bitcast<f32>(textures[at]);
}

fn extraVec4(at: u32) -> vec4<f32> {
  return vec4<f32>(extraFloat(at), extraFloat(at + 1u), extraFloat(at + 2u), extraFloat(at + 3u));
}

/** One entry of the texture table, unpacked from the words it shares with the texels. */
fn textureInfo(index: u32) -> TextureInfo {
  let at = index * 16u;
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
