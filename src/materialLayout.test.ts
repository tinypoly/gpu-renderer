import { expect, it } from "vitest";
import {
  MATERIAL_ALPHA_HASH,
  MATERIAL_FOG,
  MATERIAL_KIND,
  MATERIAL_LAYOUT,
  MATERIAL_LIT,
  MATERIAL_MAPS,
  MATERIAL_PBR,
  MATERIAL_SPECULAR,
  MATERIAL_TRANSPARENT,
  MATERIAL_VERTEX_COLORS,
  MATERIAL_WORDS,
  materialProperties,
  NO_PROGRAM,
  writeMaterial,
} from "./materialLayout.js";
import { NO_TEXTURE } from "./texturePack.js";
import type { SerializedMaterial, UniformValue } from "./upstream.js";

const material = (type: string, values: Record<string, UniformValue> = {}, maps: Record<string, string> = {}) =>
  ({ type, name: "", values, maps } as SerializedMaterial);

const written = (properties: ReturnType<typeof materialProperties>) => {
  const target = new ArrayBuffer(MATERIAL_WORDS * 4);

  writeMaterial(target, 0, properties);

  const floats = new Float32Array(target);
  const words = new Uint32Array(target);
  const at = MATERIAL_LAYOUT.offsets;

  return {
    float: (name: string, k = 0) => floats[at[name] + k],
    word: (name: string) => words[at[name]],
  };
};

it("derives the dielectric reflectance from the index of refraction, as Three does", () => {
  // The default 1.5 gives the 0.04 every renderer quotes.
  expect(materialProperties(material("MeshStandardMaterial")).dielectric).toBeCloseTo(0.04);
  expect(materialProperties(material("MeshPhysicalMaterial", { ior: 2 })).dielectric).toBeCloseTo(1 / 9);
});

it("keeps a physical material's specular colour and intensity for the shader to combine with their maps", () => {
  const properties = materialProperties(material("MeshPhysicalMaterial", {
    specularColor: [1, 0.5, 0], specularIntensity: 0.5,
  }));

  expect(properties.specular).toEqual([1, 0.5, 0]);
  expect(properties.specularIntensity).toBe(0.5);
  expect(properties.flags & MATERIAL_SPECULAR).toBeTruthy();
  // A standard material has neither in Three's shader.
  expect(materialProperties(material("MeshStandardMaterial", { specularIntensity: 0.5 })).specularIntensity).toBe(1);
});

it("keeps a Phong material's own specular colour, with Three's 0x111111 default in linear light", () => {
  expect(materialProperties(material("MeshPhongMaterial")).specular[0]).toBeCloseTo(0.0056, 4);
  expect(materialProperties(material("MeshPhongMaterial", { specular: [1, 0, 0] })).specular).toEqual([1, 0, 0]);
  expect(materialProperties(material("MeshPhongMaterial")).shininess).toBe(30);
});

it("marks which materials Three lights and which get a specular lobe", () => {
  const flags = (type: string) => materialProperties(material(type)).flags & (MATERIAL_LIT | MATERIAL_PBR);

  expect(flags("MeshStandardMaterial")).toBe(MATERIAL_LIT | MATERIAL_PBR);
  expect(flags("MeshPhysicalMaterial")).toBe(MATERIAL_LIT | MATERIAL_PBR);
  expect(flags("MeshLambertMaterial")).toBe(MATERIAL_LIT);
  expect(flags("MeshPhongMaterial")).toBe(MATERIAL_LIT);
  expect(flags("MeshToonMaterial")).toBe(MATERIAL_LIT);
  expect(flags("MeshBasicMaterial")).toBe(0);
  expect(flags("SpriteMaterial")).toBe(0);
});

it("gives each of Three's shading models its own kind", () => {
  const kind = (type: string) => materialProperties(material(type)).kind;

  expect(kind("MeshPhongMaterial")).toBe(MATERIAL_KIND.phong);
  expect(kind("MeshToonMaterial")).toBe(MATERIAL_KIND.toon);
  expect(kind("MeshMatcapMaterial")).toBe(MATERIAL_KIND.matcap);
  expect(kind("MeshNormalMaterial")).toBe(MATERIAL_KIND.normal);
  expect(kind("MeshDistanceMaterial")).toBe(MATERIAL_KIND.depth);
  expect(kind("ShadowMaterial")).toBe(MATERIAL_KIND.shadow);
  expect(kind("RawShaderMaterial")).toBe(MATERIAL_KIND.shader);
  expect(kind("PointsMaterial")).toBe(MATERIAL_KIND.basic);
});

it("reads the flags Three's own material properties set", () => {
  const flags = (values: Record<string, UniformValue>, type = "MeshStandardMaterial") =>
    materialProperties(material(type, values)).flags;

  expect(flags({ transparent: true }) & MATERIAL_TRANSPARENT).toBeTruthy();
  expect(flags({}) & MATERIAL_TRANSPARENT).toBe(0);
  expect(flags({ vertexColors: true }) & MATERIAL_VERTEX_COLORS).toBeTruthy();
  expect(flags({ alphaHash: true }) & MATERIAL_ALPHA_HASH).toBeTruthy();
  // Fog is on for every built-in material and off for a shader unless it asks.
  expect(flags({}) & MATERIAL_FOG).toBeTruthy();
  expect(flags({}, "ShaderMaterial") & MATERIAL_FOG).toBe(0);
  expect(flags({ fog: false }) & MATERIAL_FOG).toBe(0);
});

it("falls back to Three's defaults for anything the material does not carry", () => {
  const properties = materialProperties(undefined);

  expect(properties.color).toEqual([1, 1, 1]);
  expect(properties.roughness).toBe(1);
  expect(properties.metalness).toBe(0);
  expect(properties.opacity).toBe(1);
  expect(properties.normalScale).toEqual([1, 1]);
  expect(properties.side).toBe(0);
  expect(properties.blending).toBe(1);
  expect(properties.iridescenceThickness).toEqual([100, 400]);
});

it("multiplies the sheen colour by the sheen strength, as Three's own uniform does", () => {
  const properties = materialProperties(material("MeshPhysicalMaterial", { sheen: 0.5, sheenColor: [1, 0.5, 0] }));

  expect(properties.sheenColor).toEqual([0.5, 0.25, 0]);
  // Only a physical material has the layers.
  expect(materialProperties(material("MeshStandardMaterial", { clearcoat: 1, sheen: 1 })).clearcoat).toBe(0);
});

it("resolves each map slot to its place in the texture table, or to nothing", () => {
  const properties = materialProperties(
    material("MeshStandardMaterial", {}, { map: "a", normalMap: "b", envMap: "c", aoMap: "d" }),
    new Map([["a", 3], ["b", 7]]));

  expect(properties.maps.map).toBe(3);
  expect(properties.maps.normalMap).toBe(7);
  // A texture the table never received.
  expect(properties.maps.envMap).toBe(NO_TEXTURE);
  expect(properties.maps.roughnessMap).toBe(NO_TEXTURE);
  // The baked occlusion map is not a slot at all: the renderer traces what it stands for.
  expect(MATERIAL_MAPS).not.toContain("aoMap");
});

it("writes a material into the words the shader reads it from", () => {
  const properties = materialProperties(material("MeshStandardMaterial", {
    color: [0.25, 0.5, 0.75], opacity: 0.5, emissive: [1, 0, 0], emissiveIntensity: 3,
    roughness: 0.25, metalness: 0.75, alphaTest: 0.1, normalScale: [2, -2],
  }, { roughnessMap: "a" }), new Map([["a", 5]]));

  const read = written(properties);

  expect([0, 1, 2, 3].map(k => read.float("color", k))).toEqual([0.25, 0.5, 0.75, 0.5]);
  expect([0, 1, 2, 3].map(k => read.float("emissive", k))).toEqual([1, 0, 0, 3]);
  expect(read.float("specular", 3)).toBe(0.25);
  expect(read.float("metalness")).toBe(0.75);
  expect(read.word("flags") & (MATERIAL_LIT | MATERIAL_PBR)).toBe(MATERIAL_LIT | MATERIAL_PBR);
  expect(read.float("alphaTest")).toBeCloseTo(0.1);
  expect(read.word("map")).toBe(NO_TEXTURE);
  expect(read.word("roughnessMap")).toBe(5);
  expect([read.float("normalScaleX"), read.float("normalScaleY")]).toEqual([2, -2]);
  expect(read.word("program")).toBe(NO_PROGRAM);
});

it("reads transmission from physical materials only, with Three's thin wall and non absorbing defaults", () => {
  const glass = materialProperties(material("MeshPhysicalMaterial", { transmission: 1, ior: 1.33 }));

  expect(glass.transmission).toBe(1);
  expect(glass.ior).toBe(1.33);
  expect(glass.thickness).toBe(0);
  // Three's default distance is infinite, which never absorbs and is written as zero.
  expect(materialProperties(material("MeshPhysicalMaterial", { attenuationDistance: Infinity })).attenuationDistance)
    .toBe(0);
  expect(materialProperties(material("MeshStandardMaterial", { transmission: 1 })).transmission).toBe(0);
});

it("lays the struct out with WGSL's alignment, every vec4 on a sixteen byte boundary", () => {
  for (const [name, type] of Object.entries(MATERIAL_LAYOUT.types))
    if (type === "vec4f") expect(MATERIAL_LAYOUT.offsets[name] % 4).toBe(0);

  expect(MATERIAL_WORDS % 4).toBe(0);
  expect(MATERIAL_LAYOUT.wgsl).toContain("envMap: u32,");
});

it("writes the transmission words and their maps where the struct says", () => {
  const properties = materialProperties(material("MeshPhysicalMaterial", {
    transmission: 0.75, ior: 1.5, thickness: 2, attenuationColor: [0.5, 0.25, 1], attenuationDistance: 3,
  }, { transmissionMap: "t", thicknessMap: "k" }), new Map([["t", 4], ["k", 9]]));

  const read = written(properties);

  expect(read.float("transmission")).toBe(0.75);
  expect(read.word("transmissionMap")).toBe(4);
  expect(read.word("thicknessMap")).toBe(9);
  expect([read.float("ior"), read.float("thickness")]).toEqual([1.5, 2]);
  expect([0, 1, 2, 3].map(k => read.float("attenuation", k))).toEqual([0.5, 0.25, 1, 3]);
});
