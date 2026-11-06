import { layoutStruct, writeStruct, type StructField } from "./structLayout.js";
import { grassOf, type GrassField } from "./grass.js";
import { bevelOf, subsurfaceOf, volumeOf } from "./subsurface.js";
import { NO_TEXTURE } from "./texturePack.js";
import type { SerializedMaterial, UniformValue } from "./upstream.js";

/**
 * What a surface is, reduced to what the integrator reads. Three's own defaults and its shading models are resolved
 * here, following `material.js` in the CPU renderer and Three's own shader chunks, so the two renderers start from the
 * same numbers even though they use them differently.
 */
export interface MaterialProperties {
  kind: number;
  /** Linear rgb. */
  color: number[];
  opacity: number;
  /** Linear rgb. */
  emissive: number[];
  emissiveIntensity: number;
  /**
   * The specular colour: for a physical material Three's `specularColor`, before the index of refraction turns it into
   * a reflectance; for Phong its `specular`. A standard material reflects 0.04 regardless.
   */
  specular: number[];
  specularIntensity: number;
  /** Reflectance of the dielectric at normal incidence from the index of refraction, as Three derives it. */
  dielectric: number;
  roughness: number;
  metalness: number;
  alphaTest: number;
  flags: number;
  /** How far the normal map tilts the surface, along the two texture axes. */
  normalScale: number[];
  /** Share of the diffuse layer that lets light through instead, as Three's `transmission`. Physical materials only. */
  transmission: number;
  ior: number;
  /** Zero is a thin wall, which light crosses without bending; above it the surface bounds a volume that refracts. */
  thickness: number;
  /** Linear rgb that light inside the volume fades to over `attenuationDistance`. */
  attenuationColor: number[];
  /** Zero when the volume does not absorb at all, which is Three's infinite default. */
  attenuationDistance: number;
  clearcoat: number;
  clearcoatRoughness: number;
  clearcoatNormalScale: number[];
  /** Three's `sheenColor` already multiplied by `sheen`, as its own uniform is. */
  sheenColor: number[];
  sheenRoughness: number;
  iridescence: number;
  iridescenceIor: number;
  iridescenceThickness: number[];
  anisotropy: number;
  anisotropyRotation: number;
  dispersion: number;
  shininess: number;
  bumpScale: number;
  lightMapIntensity: number;
  envMapIntensity: number;
  reflectivity: number;
  refractionRatio: number;
  combine: number;
  side: number;
  blending: number;
  /**
   * Share of the diffuse layer that scatters light inside the material, from `userData.cpuRenderer.subsurface`; zero
   * for none. Physical and standard materials only.
   */
  subsurface: number;
  /** Mean free path of red, green and blue inside the material, in world units. */
  subsurfaceRadius: number[];
  subsurfaceAnisotropy: number;
  /** The medium a volume's boundary encloses: its single scattering albedo, density per scene unit and anisotropy. */
  volumeAlbedo: number[];
  volumeDensity: number;
  volumeAnisotropy: number;
  /** The radius over which edges are rounded at shading time, in scene units; zero leaves them sharp. */
  bevel: number;
  /**
   * The lawn grown on the surface, from `userData.cpuRenderer.grass`. It lives in the texture buffer's extra words
   * rather than in the struct: see grass.ts.
   */
  grass?: GrassField | null;
  /** An index into the texture table per slot, or `NO_TEXTURE`. */
  maps: Record<MaterialMap, number>;
}

/** The parts of a material entry that depend on the mesh and the packing, not on the material alone. */
export interface MaterialPlacement {
  /** Where the material's clipping planes start in the texture buffer, and how many there are. */
  clipStart: number;
  clipCount: number;
  /** The translated fragment shader, or `NO_PROGRAM`. */
  program: number;
  /** Where its uniform values start in the texture buffer. */
  shaderData: number;
  /** Where the shader's per-corner varyings start in the attribute buffer, how many floats a corner has, and which
   *  triangle they begin at. */
  varyingBase: number;
  varyingStride: number;
  firstTriangle: number;
  /** Rows of the mesh's world normal matrix, which an object space normal map is turned by. */
  normalMatrix: number[];
  /** The program a hook after `<opaque_fragment>` became, and where its uniforms start. */
  afterProgram: number;
  afterData: number;
}

/**
 * The map slots the integrator reads, by Three's own property names, so a material's `maps` record is looked up directly.
 * A baked ambient occlusion map is deliberately absent: this renderer traces the occlusion it stands for, and applying
 * both would darken the same shadow twice. `displacementMap` moves vertices, so it is applied while packing instead.
 */
export const MATERIAL_MAPS = [
  "map", "emissiveMap", "roughnessMap", "metalnessMap", "normalMap", "alphaMap", "transmissionMap", "thicknessMap",
  "clearcoatMap", "clearcoatRoughnessMap", "clearcoatNormalMap", "sheenColorMap", "sheenRoughnessMap",
  "iridescenceMap", "iridescenceThicknessMap", "anisotropyMap", "specularMap", "specularIntensityMap",
  "specularColorMap", "bumpMap", "lightMap", "gradientMap", "matcap", "envMap"] as const;

export type MaterialMap = typeof MATERIAL_MAPS[number];

/** How the integrator shades a surface. Each kind is one of Three's shading models. */
export const MATERIAL_KIND = {
  standard: 0,
  physical: 1,
  lambert: 2,
  phong: 3,
  toon: 4,
  /** Unlit: basic, points, lines and sprites. */
  basic: 5,
  normal: 6,
  /** Depth and distance materials. */
  depth: 7,
  matcap: 8,
  shadow: 9,
  /** A `ShaderMaterial` or `RawShaderMaterial`: unlit, its colour from the translated fragment shader. */
  shader: 10,
} as const;

/** A surface Three shades with light at all. The rest return their own colour and take no part in it. */
export const MATERIAL_LIT = 1;
/** Three's metalness workflow, which is the only one with a GGX specular lobe. */
export const MATERIAL_PBR = 2;
/** Three's `transparent`: only then does the opacity let anything through. */
export const MATERIAL_TRANSPARENT = 4;
export const MATERIAL_VERTEX_COLORS = 8;
export const MATERIAL_FLAT = 16;
export const MATERIAL_WIREFRAME = 32;
/** The material accepts `scene.fog`. */
export const MATERIAL_FOG = 64;
export const MATERIAL_CLIP_INTERSECTION = 128;
export const MATERIAL_OBJECT_NORMAL_MAP = 256;
/** A built-in material whose `onBeforeCompile` hooks run before lighting, as a program. */
export const MATERIAL_PATCHED = 512;
/** The material's own environment map bends the view ray instead of mirroring it. */
export const MATERIAL_ENV_REFRACTION = 1024;
/** Three's `alphaHash`, which keeps the coverage stochastic without `transparent`. */
export const MATERIAL_ALPHA_HASH = 2048;
/** A physical material, whose specular colour and intensity reach the lobe and whose grazing reflectance follows them. */
export const MATERIAL_SPECULAR = 4096;

/**
 * The mesh only bounds a medium, from `userData.cpuRenderer.volume`: its surface is invisible and the space inside
 * scatters light.
 */
export const MATERIAL_VOLUME = 8192;

/**
 * An unlit material whose colour is known without running anything: opaque, normal blending, no light map, environment
 * map, alpha map or shader. What it shows is what it emits, so direct sampling can aim at it like an emissive one.
 */
export const MATERIAL_UNLIT_LIGHT = 16384;

/** What a material's `program` reads as when it has no shader. */
export const NO_PROGRAM = 0xffffffff;

const FIELDS: StructField[] = [
  { name: "color", type: "vec4f", doc: "rgb is the base colour in linear light, w the opacity." },
  { name: "emissive", type: "vec4f", doc: "rgb is the emissive colour, w its intensity." },
  { name: "specular", type: "vec4f", doc: "rgb is the specular colour, w the roughness." },
  { name: "attenuation", type: "vec4f", doc: "rgb is the colour a volume fades to over w; a w of zero never absorbs." },
  { name: "sheen", type: "vec4f", doc: "rgb is the sheen colour times its strength, w its roughness." },
  { name: "metalness", type: "f32" },
  { name: "flags", type: "u32" },
  { name: "alphaTest", type: "f32" },
  { name: "transmission", type: "f32" },
  { name: "normalScaleX", type: "f32" },
  { name: "normalScaleY", type: "f32" },
  { name: "ior", type: "f32" },
  { name: "thickness", type: "f32", doc: "Zero is a thin wall; above it the surface bounds a volume that refracts." },
  { name: "clearcoat", type: "f32" },
  { name: "clearcoatRoughness", type: "f32" },
  { name: "clearcoatNormalScaleX", type: "f32" },
  { name: "clearcoatNormalScaleY", type: "f32" },
  { name: "iridescence", type: "f32" },
  { name: "iridescenceIor", type: "f32" },
  { name: "iridescenceThicknessMin", type: "f32" },
  { name: "iridescenceThicknessMax", type: "f32" },
  { name: "anisotropy", type: "f32" },
  { name: "anisotropyRotation", type: "f32" },
  { name: "dispersion", type: "f32" },
  { name: "shininess", type: "f32" },
  { name: "bumpScale", type: "f32" },
  { name: "lightMapIntensity", type: "f32" },
  { name: "envMapIntensity", type: "f32" },
  { name: "reflectivity", type: "f32" },
  { name: "refractionRatio", type: "f32" },
  { name: "combine", type: "u32" },
  { name: "side", type: "u32" },
  { name: "blending", type: "u32" },
  { name: "clipStart", type: "u32" },
  { name: "clipCount", type: "u32" },
  { name: "program", type: "u32" },
  { name: "shaderData", type: "u32" },
  { name: "varyingBase", type: "u32" },
  { name: "varyingStride", type: "u32" },
  { name: "firstTriangle", type: "u32" },
  { name: "kind", type: "u32" },
  { name: "specularIntensity", type: "f32" },
  { name: "dielectric", type: "f32" },
  { name: "afterProgram", type: "u32", doc: "What a hook does after <opaque_fragment>, or NO_PROGRAM." },
  { name: "afterData", type: "u32" },
  { name: "subsurface", type: "vec4f", doc: "rgb is the mean free path of each colour inside, w the share of the diffuse layer." },
  { name: "subsurfaceAnisotropy", type: "f32" },
  { name: "volume", type: "vec4f", doc: "rgb is the medium's single scattering albedo, w its density per scene unit." },
  { name: "volumeAnisotropy", type: "f32" },
  { name: "bevel", type: "f32", doc: "The radius edges are rounded over at shading time; zero leaves them sharp." },
  { name: "normalRow0", type: "vec4f", doc: "The mesh's world normal matrix, for an object space normal map." },
  { name: "normalRow1", type: "vec4f" },
  { name: "normalRow2", type: "vec4f" },
  ...MATERIAL_MAPS.map(slot => ({ name: slot, type: "u32" as const })),
];

/** The `Material` struct, as both sides of the device see it. */
export const MATERIAL_LAYOUT = layoutStruct("Material", FIELDS);

export const MATERIAL_WORDS = MATERIAL_LAYOUT.words;

const KIND_OF: Record<string, number> = {
  MeshStandardMaterial: MATERIAL_KIND.standard,
  MeshPhysicalMaterial: MATERIAL_KIND.physical,
  MeshLambertMaterial: MATERIAL_KIND.lambert,
  MeshPhongMaterial: MATERIAL_KIND.phong,
  MeshToonMaterial: MATERIAL_KIND.toon,
  MeshNormalMaterial: MATERIAL_KIND.normal,
  MeshDepthMaterial: MATERIAL_KIND.depth,
  MeshDistanceMaterial: MATERIAL_KIND.depth,
  MeshMatcapMaterial: MATERIAL_KIND.matcap,
  ShadowMaterial: MATERIAL_KIND.shadow,
  ShaderMaterial: MATERIAL_KIND.shader,
  RawShaderMaterial: MATERIAL_KIND.shader,
};

/** The material types Three lights, exactly as `LIT_MATERIALS` in the CPU renderer. */
const LIT = new Set([
  "MeshLambertMaterial", "MeshPhongMaterial", "MeshToonMaterial", "MeshStandardMaterial", "MeshPhysicalMaterial"]);

const PBR = new Set(["MeshStandardMaterial", "MeshPhysicalMaterial"]);

/** Three's `ShaderMaterial` keeps `fog` off by default; every built-in material has it on. */
const FOG_OFF_BY_DEFAULT = new Set(["ShaderMaterial", "RawShaderMaterial"]);

/** Three's `0x111111`, the Phong default specular colour, in linear light. */
const PHONG_SPECULAR = 0.005605391906346412;

/** Reads a serialized Three colour or vector, falling back when the material does not carry it. */
function vector(value: UniformValue | undefined, fallback: readonly number[], size = 3): number[] {
  return Array.isArray(value) && value.length >= size && value.slice(0, size).every(entry => typeof entry === "number")
    ? (value as number[]).slice(0, size)
    : [...fallback];
}

function number(value: UniformValue | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function flag(value: UniformValue | undefined, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

/** Whether a material is drawn at all: Three skips one whose `visible` is off. */
export function isMaterialVisible(material: SerializedMaterial | undefined): boolean {
  return flag(material?.values?.visible, true) && flag(material?.values?.colorWrite, true);
}

/**
 * Three's resolution of a material, defaults and all. A material outside the metalness workflow keeps its own specular
 * colour; one Three does not light reads as its own colour.
 */
export function materialProperties(
  material: SerializedMaterial | undefined,
  textures: ReadonlyMap<string, number> = new Map()): MaterialProperties {
  const values = material?.values ?? {};
  const type = material?.type ?? "MeshStandardMaterial";
  const pbr = PBR.has(type);
  const physical = type === "MeshPhysicalMaterial";
  const ior = number(values.ior, 1.5);
  const attenuationDistance = number(values.attenuationDistance, 0);
  const transparent = flag(values.transparent, false);
  const sheen = physical ? Math.max(0, number(values.sheen, 0)) : 0;
  const anisotropy = physical ? number(values.anisotropy, 0) : 0;
  const thickness = vector(values.iridescenceThicknessRange, [100, 400], 2);
  const medium = pbr ? subsurfaceOf(material) : null;
  const volume = volumeOf(material);

  const maps = Object.fromEntries(MATERIAL_MAPS.map(slot => {
    const id = material?.maps?.[slot];

    return [slot, id !== undefined ? textures.get(id) ?? NO_TEXTURE : NO_TEXTURE];
  })) as Record<MaterialMap, number>;

  // A shader material's own colour comes from its program; a patched built-in keeps its type's shading.
  const shader = material?.shader;
  const kind = KIND_OF[type] ?? MATERIAL_KIND.basic;

  const flags = (LIT.has(type) ? MATERIAL_LIT : 0)
    | (pbr ? MATERIAL_PBR : 0)
    | (transparent ? MATERIAL_TRANSPARENT : 0)
    | (flag(values.vertexColors, false) ? MATERIAL_VERTEX_COLORS : 0)
    | (flag(values.flatShading, false) ? MATERIAL_FLAT : 0)
    | (flag(values.wireframe, false) ? MATERIAL_WIREFRAME : 0)
    | (flag(values.fog, !FOG_OFF_BY_DEFAULT.has(type)) ? MATERIAL_FOG : 0)
    | (flag(values.clipIntersection, false) ? MATERIAL_CLIP_INTERSECTION : 0)
    | (number(values.normalMapType, 0) === 1 ? MATERIAL_OBJECT_NORMAL_MAP : 0)
    | (shader?.patched ? MATERIAL_PATCHED : 0)
    | (flag(values.alphaHash, false) ? MATERIAL_ALPHA_HASH : 0)
    | (physical ? MATERIAL_SPECULAR : 0)
    | (volume ? MATERIAL_VOLUME : 0)
    | (type === "MeshBasicMaterial" && !shader && !transparent && !flag(values.alphaHash, false)
      && !flag(values.wireframe, false) && number(values.blending, 1) === 1 && number(values.opacity, 1) >= 1
      && number(values.alphaTest, 0) <= 0 && maps.alphaMap === NO_TEXTURE && maps.envMap === NO_TEXTURE
      && maps.lightMap === NO_TEXTURE && !volume ? MATERIAL_UNLIT_LIGHT : 0);

  return {
    kind,
    color: vector(values.color, [1, 1, 1]),
    opacity: number(values.opacity, 1),
    emissive: vector(values.emissive, [0, 0, 0]),
    emissiveIntensity: number(values.emissiveIntensity, 1),
    specular: physical
      ? vector(values.specularColor, [1, 1, 1])
      : vector(values.specular, [PHONG_SPECULAR, PHONG_SPECULAR, PHONG_SPECULAR]),
    specularIntensity: physical ? number(values.specularIntensity, 1) : 1,
    // A standard material has no index of refraction in Three's shader and reflects 0.04, which is what 1.5 gives.
    dielectric: ((ior - 1) / (ior + 1)) ** 2,
    roughness: number(values.roughness, 1),
    metalness: number(values.metalness, 0),
    alphaTest: number(values.alphaTest, 0),
    flags,
    normalScale: vector(values.normalScale, [1, 1], 2),
    // Three only lets light through a physical material. Like its own shader, metal takes the share back.
    transmission: physical ? clamp01(number(values.transmission, 0)) : 0,
    ior,
    thickness: Math.max(0, number(values.thickness, 0)),
    attenuationColor: vector(values.attenuationColor, [1, 1, 1]),
    attenuationDistance: attenuationDistance > 0 ? attenuationDistance : 0,
    clearcoat: physical ? clamp01(number(values.clearcoat, 0)) : 0,
    clearcoatRoughness: number(values.clearcoatRoughness, 0),
    clearcoatNormalScale: vector(values.clearcoatNormalScale, [1, 1], 2),
    sheenColor: vector(values.sheenColor, [0, 0, 0]).map(channel => channel * sheen),
    sheenRoughness: Math.min(1, Math.max(1e-4, number(values.sheenRoughness, 1))),
    iridescence: physical ? Math.max(0, number(values.iridescence, 0)) : 0,
    iridescenceIor: number(values.iridescenceIOR, 1.3),
    iridescenceThickness: thickness,
    anisotropy,
    anisotropyRotation: number(values.anisotropyRotation, 0),
    dispersion: physical ? Math.max(0, number(values.dispersion, 0)) : 0,
    shininess: Math.max(0, number(values.shininess, 30)),
    bumpScale: number(values.bumpScale, 1),
    lightMapIntensity: number(values.lightMapIntensity, 1),
    envMapIntensity: number(values.envMapIntensity, 1),
    reflectivity: number(values.reflectivity, 1),
    refractionRatio: number(values.refractionRatio, 0.98),
    combine: number(values.combine, 0),
    side: number(values.side, 0),
    blending: number(values.blending, 1),
    subsurface: medium?.weight ?? 0,
    subsurfaceRadius: medium?.radius ?? [1, 1, 1],
    subsurfaceAnisotropy: medium?.anisotropy ?? 0,
    volumeAlbedo: volume?.albedo ?? [1, 1, 1],
    volumeDensity: volume?.density ?? 0,
    volumeAnisotropy: volume?.anisotropy ?? 0,
    bevel: LIT.has(type) ? bevelOf(material) : 0,
    grass: LIT.has(type) ? grassOf(material) : null,
    maps,
  };
}

/** A material entry with no mesh specific parts: no clipping, no program, no varyings, an identity normal matrix. */
export const NO_PLACEMENT: MaterialPlacement = {
  clipStart: 0,
  clipCount: 0,
  program: NO_PROGRAM,
  shaderData: 0,
  varyingBase: 0,
  varyingStride: 0,
  firstTriangle: 0,
  normalMatrix: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  afterProgram: NO_PROGRAM,
  afterData: 0,
};

/** The bytes of one material, as the shader reads them. */
export function writeMaterial(
  target: ArrayBuffer, index: number, properties: MaterialProperties, placement: MaterialPlacement = NO_PLACEMENT) {
  const p = properties;
  const n = placement.normalMatrix;

  writeStruct(MATERIAL_LAYOUT, target, index * MATERIAL_WORDS, {
    color: [...p.color, p.opacity],
    emissive: [...p.emissive, p.emissiveIntensity],
    specular: [...p.specular, p.roughness],
    attenuation: [...p.attenuationColor, p.attenuationDistance],
    sheen: [...p.sheenColor, p.sheenRoughness],
    metalness: p.metalness,
    flags: p.flags,
    alphaTest: p.alphaTest,
    transmission: p.transmission,
    normalScaleX: p.normalScale[0],
    normalScaleY: p.normalScale[1],
    ior: p.ior,
    thickness: p.thickness,
    clearcoat: p.clearcoat,
    clearcoatRoughness: p.clearcoatRoughness,
    clearcoatNormalScaleX: p.clearcoatNormalScale[0],
    clearcoatNormalScaleY: p.clearcoatNormalScale[1],
    iridescence: p.iridescence,
    iridescenceIor: p.iridescenceIor,
    iridescenceThicknessMin: p.iridescenceThickness[0],
    iridescenceThicknessMax: p.iridescenceThickness[1],
    anisotropy: p.anisotropy,
    anisotropyRotation: p.anisotropyRotation,
    dispersion: p.dispersion,
    shininess: p.shininess,
    bumpScale: p.bumpScale,
    lightMapIntensity: p.lightMapIntensity,
    envMapIntensity: p.envMapIntensity,
    reflectivity: p.reflectivity,
    refractionRatio: p.refractionRatio,
    combine: p.combine,
    side: p.side,
    blending: p.blending,
    clipStart: placement.clipStart,
    clipCount: placement.clipCount,
    program: placement.program,
    shaderData: placement.shaderData,
    varyingBase: placement.varyingBase,
    varyingStride: placement.varyingStride,
    firstTriangle: placement.firstTriangle,
    kind: p.kind,
    specularIntensity: p.specularIntensity,
    dielectric: p.dielectric,
    afterProgram: placement.afterProgram,
    afterData: placement.afterData,
    subsurface: [...p.subsurfaceRadius, p.subsurface],
    subsurfaceAnisotropy: p.subsurfaceAnisotropy,
    volume: [...p.volumeAlbedo, p.volumeDensity],
    volumeAnisotropy: p.volumeAnisotropy,
    bevel: p.bevel,
    normalRow0: [n[0], n[3], n[6], 0],
    normalRow1: [n[1], n[4], n[7], 0],
    normalRow2: [n[2], n[5], n[8], 0],
    ...Object.fromEntries(MATERIAL_MAPS.map(slot => [slot, p.maps[slot]])),
  });
}
