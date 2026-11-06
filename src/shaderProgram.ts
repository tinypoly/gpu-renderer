import type * as Naga from "naga-wasm";
import { ShaderChunk } from "three";
import { NO_TEXTURE } from "./texturePack.js";
import { CpuShader } from "./upstream.js";
import type { SerializedMaterial, SerializedScene, SerializedShader, UniformValue } from "./upstream.js";

/**
 * Three's GLSL, run inside the path tracing kernel.
 *
 * A fragment shader is translated to WGSL by naga, after a rewrite that turns it from a program into functions: every
 * uniform, varying and built-in becomes a module scope global the kernel fills in, every texture call and derivative
 * becomes a function the kernel provides, and `discard` becomes a flag. What naga writes back is renamed into the
 * kernel's namespace and given an entry the kernel calls at a hit. See the decision in ARCHITECTURE.md.
 *
 * Vertex shaders are not translated: they run once per vertex while the scene is packed, on the CPU renderer's own
 * interpreter, because their output decides where the triangles are and the hierarchy is built from that.
 *
 * Derivatives are real. A shader that takes one runs three times per hit, for the pixel to the right, the pixel above
 * and the pixel itself, in that order; each call site keeps what the neighbours computed there, and the third run reads
 * the difference. That is a quad's worth of work, in sequence, and it holds wherever the three runs take the same path
 * through the shader, which is the same assumption a GPU's own quads make.
 */

/** Which kind of entry a translated shader gets. */
export type ProgramKind =
  /** A `ShaderMaterial` or `RawShaderMaterial`: its output colour is what the surface shows. */
  | "surface"
  /** A built-in material's `onBeforeCompile` hooks: they edit the material's values before lighting. */
  | "patched"
  /** What a built-in material's hooks do after `<opaque_fragment>`: they edit the colour the surface ended up with. */
  | "after"
  /** A procedural sky's foreground, painted over the background per pixel. */
  | "foreground";

export interface ProgramVarying {
  name: string;
  /** Floats per corner. */
  size: number;
  /** Where it starts within a corner. */
  offset: number;
  /** Taken from one vertex rather than interpolated, as GLSL's `flat` (and every integer varying) asks. */
  flat: boolean;
}

export interface ShaderProgram {
  id: number;
  kind: ProgramKind;
  /** The program's functions, renamed into its own namespace, with the entry the kernel calls. */
  wgsl: string;
  varyings: ProgramVarying[];
  /** Floats each triangle corner stores for this program. */
  varyingStride: number;
  usesDerivatives: boolean;
  /** Words of uniform data one material entry needs. */
  uniformWords: number;
  /** The uniform values, laid out as the generated loader reads them. Textures become indices into the table. */
  packUniforms(values: Record<string, UniformValue>, textures: ReadonlyMap<string, number>): Uint32Array;
}

/** Everything translated for one scene. */
export interface ShaderPrograms {
  programs: ShaderProgram[];
  /** The fragment program a material runs, if it has one and it translated. */
  fragment(material: SerializedMaterial): ShaderProgram | null;
  /** The vertex shader a material runs on the CPU, if it has one. */
  vertex(material: SerializedMaterial): CpuShader | null;
  /** What a built-in material's hooks do after `<opaque_fragment>`, if they do anything there. */
  after(material: SerializedMaterial): ShaderProgram | null;
  /** A procedural sky's foreground, if there is one. */
  foreground: ShaderProgram | null;
  /** Shaders that failed to translate, by material name, with the reason; those materials render as their colour. */
  failures: string[];
  /** Identifies the set of programs, which is what decides whether the kernel has to be compiled again. */
  key: string;
}

/** The chunks a shader may include, with the output transforms the renderer applies to the image itself. */
export const SHADER_CHUNKS: Record<string, string> = {
  ...ShaderChunk as unknown as Record<string, string>,
  tonemapping_fragment: "",
  colorspace_fragment: "",
  encodings_fragment: "",
  // Three's fog chunks are honoured by the integrator for every material that accepts fog, shaders included.
  fog_fragment: "",
};

/** What a sampler becomes in the rewritten GLSL: a texture table index, typed so the overloads below can tell them apart. */
const SAMPLER_TYPES: Record<string, string> = {
  sampler2D: "uint",
  isampler2D: "uint",
  usampler2D: "uint",
  sampler2DShadow: "uint",
  samplerExternalOES: "uint",
  samplerCube: "int",
  samplerCubeShadow: "int",
  sampler3D: "uvec2",
  sampler2DArray: "uvec2",
  isampler3D: "uvec2",
  usampler3D: "uvec2",
  sampler2DArrayShadow: "uvec2",
};

/** The texture calls the kernel provides, by the name the rewrite gives them. */
const TEXTURE_CALLS: [RegExp, string][] = [
  [/\b(texture2DLodEXT|texture2DLod|textureCubeLodEXT|textureCubeLod|textureLod)\s*\(/g, "tp_texLod("],
  [/\b(texture2DGradEXT|textureCubeGradEXT|textureGrad)\s*\(/g, "tp_texGrad("],
  [/\b(texture2DProj|textureProj)\s*\(/g, "tp_texProj("],
  [/\btexelFetch\s*\(/g, "tp_texelFetch("],
  [/\btextureSize\s*\(/g, "tp_texSize("],
  [/\b(textureCube|texture2D|texture)\s*\(/g, "tp_tex("],
];

/** Built-in variables, which become globals the kernel sets. */
const BUILTINS: [RegExp, string][] = [
  [/\bgl_FragData\s*\[\s*0\s*\]/g, "tp_FragColor"],
  [/\bgl_FragColor\b/g, "tp_FragColor"],
  [/\bgl_FragCoord\b/g, "tp_FragCoord"],
  [/\bgl_FrontFacing\b/g, "tp_FrontFacing"],
  [/\bgl_PointCoord\b/g, "tp_PointCoord"],
  [/\bgl_FragDepth(EXT)?\b/g, "tp_FragDepth"],
];

/**
 * Stand-ins for everything the kernel replaces. naga keeps their signatures, which is how the WGSL that comes back is
 * matched to the kernel's own implementation; their bodies are thrown away.
 */
const INTERNAL_PRELUDE = `
vec4 tp_FragColor;
vec4 tp_FragCoord;
bool tp_FrontFacing;
vec2 tp_PointCoord;
float tp_FragDepth;
bool tp_discarded;
int tp_lane;
void tp_discard() { tp_discarded = true; }
vec4 tp_tex(uint s, vec2 uv) { return vec4(0.0); }
vec4 tp_tex(uint s, vec2 uv, float bias) { return vec4(0.0); }
vec4 tp_tex(int s, vec3 direction) { return vec4(0.0); }
vec4 tp_tex(int s, vec3 direction, float bias) { return vec4(0.0); }
vec4 tp_tex(uvec2 s, vec3 uvw) { return vec4(0.0); }
vec4 tp_tex(uvec2 s, vec3 uvw, float bias) { return vec4(0.0); }
vec4 tp_texLod(uint s, vec2 uv, float lod) { return vec4(0.0); }
vec4 tp_texLod(int s, vec3 direction, float lod) { return vec4(0.0); }
vec4 tp_texLod(uvec2 s, vec3 uvw, float lod) { return vec4(0.0); }
vec4 tp_texGrad(uint s, vec2 uv, vec2 dx, vec2 dy) { return vec4(0.0); }
vec4 tp_texGrad(int s, vec3 direction, vec3 dx, vec3 dy) { return vec4(0.0); }
vec4 tp_texProj(uint s, vec3 uv) { return vec4(0.0); }
vec4 tp_texProj(uint s, vec4 uv) { return vec4(0.0); }
vec4 tp_texelFetch(uint s, ivec2 texel, int lod) { return vec4(0.0); }
ivec2 tp_texSize(uint s, int lod) { return ivec2(0); }
ivec2 tp_texSize(int s, int lod) { return ivec2(0); }
`;

/** What Three declares in front of every `ShaderMaterial` fragment shader, minus its output transforms. */
const SHADER_MATERIAL_PRELUDE = `
mat4 viewMatrix;
vec3 cameraPosition;
bool isOrthographic;
vec4 linearToOutputTexel(vec4 value) { return value; }
vec3 toneMapping(vec3 color) { return color; }
`;

/** The surface state a built-in material's hooks read and edit, as Three's own fragment shader names it. */
const PATCHED_CONTEXT: [string, string][] = [
  ["vec4", "diffuseColor"],
  ["vec3", "normal"],
  ["float", "roughnessFactor"],
  ["float", "metalnessFactor"],
  ["vec3", "totalEmissiveRadiance"],
  ["vec3", "vViewPosition"],
  ["vec2", "vUv"],
  ["vec3", "outgoingLight"],
  ["vec3", "geometryNormal"],
  ["vec3", "nonPerturbedNormal"],
  ["vec3", "geometryPosition"],
  ["vec3", "geometryViewDir"],
  ["float", "faceDirection"],
  ["vec3", "vNormal"],
];

/**
 * Three's physical light function, which a hook may wrap to change its direct light (a relief shadowing its own
 * joints). The integrator lights the surface, so the function is only there for the hook to compile and does nothing.
 */
const PHYSICAL_LIGHT_STUB = `
struct PhysicalMaterial { vec3 diffuseColor; float roughness; vec3 specularColor; float specularF90; float dispersion; };
void RE_Direct_Physical(IncidentLight directLight, vec3 geometryPosition, vec3 geometryNormal, vec3 geometryViewDir,
  vec3 geometryClearcoatNormal, PhysicalMaterial material, inout ReflectedLight reflectedLight) {}
`;

/** Light helpers Three's hooks expect from its lighting chunks, which a patched shader no longer includes. */
const PATCHED_HELPERS = `
float getDistanceAttenuation(float lightDistance, float cutoffDistance, float decayExponent) {
  float falloff = 1.0 / max(pow(lightDistance, decayExponent), 0.01);
  if (cutoffDistance > 0.0) falloff *= pow(clamp(1.0 - pow(lightDistance / cutoffDistance, 4.0), 0.0, 1.0), 2.0);
  return falloff;
}
float getSpotAttenuation(float coneCosine, float penumbraCosine, float angleCosine) {
  return smoothstep(coneCosine, penumbraCosine, angleCosine);
}
`;

/** The GLSL handed to naga, and what the rewrite learned about the shader on the way. */
export interface PreparedGlsl {
  source: string;
  uniforms: string[];
  /** Inputs from the vertex stage, in declaration order. */
  varyings: { name: string; flat: boolean }[];
  /** Outputs a GLSL 3 shader declares itself, instead of writing gl_FragColor. */
  outputs: string[];
  /** How many derivative call sites the shader has. */
  derivativeSites: number;
}

/** Expands `#include <chunk>` the way Three does, recursively. */
export function expandIncludes(source: string, chunks: Record<string, string> = SHADER_CHUNKS, depth = 0): string {
  if (depth > 16) throw new Error("GPU renderer: shader includes nest too deeply.");

  return source.replace(/^[ \t]*#include +<([\w\d./]+)>/gm, (_match, name: string) => {
    const chunk = chunks[name];

    if (chunk === undefined) throw new Error(`GPU renderer: unknown shader include <${name}>.`);

    return expandIncludes(chunk, chunks, depth + 1);
  });
}

/** Three's unrollLoops, for the loops whose bounds are plain numbers; others stay loops. */
function unrollLoops(source: string): string {
  const pattern = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;

  return source
    .replace(pattern, (_match, start: string, end: string, body: string) => {
      let unrolled = "";

      for (let i = Number(start); i < Number(end); i++)
        unrolled += body.replace(/\[\s*i\s*\]/g, `[ ${i} ]`).replace(/UNROLLED_LOOP_INDEX/g, String(i));

      return unrolled;
    })
    .replace(/#pragma unroll_loop_(start|end)/g, "");
}

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, match => match.replace(/[^\n]/g, " ")).replace(/\/\/[^\n]*/g, "");
}

const QUALIFIERS = new Set([
  "uniform", "varying", "attribute", "in", "out", "flat", "smooth", "noperspective", "centroid", "invariant",
  "highp", "mediump", "lowp", "const"]);

/**
 * Rewrites the declarations at global scope: uniforms and varyings lose their storage qualifier and become plain
 * globals, which naga keeps as module scope variables the kernel can fill in; samplers become texture indices.
 */
function rewriteGlobals(source: string, stage: "fragment") {
  const uniforms: string[] = [];
  const varyings: { name: string; flat: boolean }[] = [];
  const outputs: string[] = [];
  let output = "";
  let depth = 0;
  let statement = "";
  let lineStart = true;

  const declarators = (rest: string) => rest
    .split(",")
    .map(part => /^\s*(\w+)/.exec(part)?.[1])
    .filter((name): name is string => !!name);

  const flush = (terminator: string) => {
    const text = statement;

    statement = "";

    const layout = /^(\s*)layout\s*\([^)]*\)\s*/.exec(text);
    const body = layout ? text.slice(layout[0].length) : text.trimStart();
    const leading = layout ? layout[1] : text.slice(0, text.length - text.trimStart().length);
    const words = body.split(/\s+/);
    let index = 0;
    const qualifiers: string[] = [];

    while (index < words.length && QUALIFIERS.has(words[index])) qualifiers.push(words[index++]);

    const storage = qualifiers.find(q => q === "uniform" || q === "varying" || q === "attribute" || q === "in" || q === "out");

    if (!storage) {
      output += text + terminator;

      return;
    }

    const remainder = words.slice(index).join(" ");
    const type = /^(\w+)/.exec(remainder)?.[1] ?? "";
    const names = declarators(remainder.slice(type.length));

    // An interface block: `uniform Block { ... } name;` becomes a struct with that declarator.
    if (terminator === "{") {
      output += `${leading}struct ${type} {`;
      depth++;

      return;
    }

    if (storage === "uniform") uniforms.push(...names);
    else if (stage === "fragment" && (storage === "varying" || storage === "in"))
      varyings.push(...names.map(name => ({ name, flat: qualifiers.includes("flat") })));
    else if (storage === "out") outputs.push(...names);

    output += `${leading}${remainder}${terminator}`;
  };

  for (let i = 0; i < source.length; i++) {
    const char = source[i];

    // Preprocessor lines pass through whole, wherever they appear.
    if (lineStart && char === "#") {
      const end = source.indexOf("\n", i);
      const line = end < 0 ? source.slice(i) : source.slice(i, end + 1);

      if (depth === 0) {
        output += statement + line;
        statement = "";
      } else {
        output += line;
      }

      i += line.length - 1;
      lineStart = true;
      continue;
    }

    if (char === "\n") lineStart = true;
    else if (char !== " " && char !== "\t" && char !== "\r") lineStart = false;

    if (depth > 0) {
      output += char;

      if (char === "{") depth++;
      if (char === "}") depth--;

      continue;
    }

    if (char === ";") {
      flush(";");
      continue;
    }

    if (char === "{") {
      flush("{");

      if (!output.endsWith("{")) output += "{";

      depth = Math.max(depth, 1);
      continue;
    }

    if (char === "}") {
      output += statement + char;
      statement = "";
      continue;
    }

    statement += char;
  }

  output += statement;

  return { source: output, uniforms, varyings, outputs };
}

/**
 * Turns one of Three's fragment shaders into the GLSL 450 naga reads, with every hook the kernel needs. Exported for
 * the tests, which check the rewrite without a device.
 */
export function prepareFragmentShader(
  shader: Pick<SerializedShader, "fragment" | "defines">,
  kind: ProgramKind,
  options: { raw?: boolean } = {}): PreparedGlsl {
  const defines = Object.entries(shader.defines ?? {})
    .filter(([, value]) => value !== false)
    .map(([name, value]) => `#define ${name} ${value === true ? "" : value}`)
    .join("\n");

  let body = shader.fragment;

  if (kind === "patched" || kind === "after") body = `${SHADER_CHUNKS.common}\n${PHYSICAL_LIGHT_STUB}\n${body}`;

  body = unrollLoops(expandIncludes(body));
  body = stripComments(body)
    .replace(/^[ \t]*#version[^\n]*$/gm, "")
    .replace(/^[ \t]*#extension[^\n]*$/gm, "")
    .replace(/^[ \t]*precision\s+\w+\s+\w+\s*;/gm, "")
    .replace(/\b(highp|mediump|lowp)\b/g, "");

  const rewritten = rewriteGlobals(body, "fragment");

  body = rewritten.source;

  for (const [type, replacement] of Object.entries(SAMPLER_TYPES))
    body = body.replace(new RegExp(`\\b${type}\\b`, "g"), replacement);

  for (const [pattern, replacement] of [...BUILTINS, ...TEXTURE_CALLS]) body = body.replace(pattern, replacement);

  body = body
    .replace(/\bdiscard\s*;/g, "tp_discard();")
    .replace(/\bconst\s+(?=(in|inout|out)\b)/g, "")
    .replace(/([(,]\s*)const\s+(?=\w)/g, "$1");

  // Every derivative call site gets its own slot, which the neighbouring runs fill in and the centre one reads.
  let sites = 0;
  const siteFunctions: string[] = [];

  body = body.replace(/\b(dFdx|dFdy|fwidth)\s*\(/g, (_match, operation: string) => {
    const site = sites++;

    siteFunctions.push(derivativeSite(site, operation));

    return `tp_${operation}_${site}(`;
  });

  // Only declarations at global scope count: a parameter or a local of the same name declares nothing for main().
  const global = globalScope(body);
  const declared = (name: string) => new RegExp(`\\b(vec[234]|float|int|bool|mat[234])\\s+${name}\\s*[;=\\[,]`).test(global);

  const patched = kind === "patched" || kind === "after"
    ? PATCHED_CONTEXT.filter(([, name]) => !declared(name)).map(([type, name]) => `${type} ${name};`).join("\n")
      + PATCHED_HELPERS
    : "";

  const shaderPrelude = !options.raw ? SHADER_MATERIAL_PRELUDE
    .split("\n")
    .filter(line => {
      const name = /^\w+\s+(\w+)\s*;/.exec(line)?.[1];

      return !name || !declared(name);
    })
    .join("\n") : "";

  const source = [
    "#version 450",
    defines,
    INTERNAL_PRELUDE,
    siteFunctions.join("\n"),
    shaderPrelude,
    patched,
    body,
  ].join("\n");

  return {
    source,
    uniforms: [...new Set(rewritten.uniforms)],
    varyings: rewritten.varyings,
    outputs: rewritten.outputs,
    derivativeSites: sites,
  };
}

/** The source with every block and parenthesis emptied, which leaves the declarations at global scope. */
function globalScope(source: string): string {
  let depth = 0;
  let out = "";

  for (const char of source) {
    if (char === "{" || char === "(") depth++;

    if (depth === 0) out += char;

    if ((char === "}" || char === ")") && depth > 0) {
      depth--;

      if (depth === 0) out += ";";
    }
  }

  return out;
}

/** One derivative call site, overloaded for every float type, with the slot its neighbours write. */
function derivativeSite(site: number, operation: string): string {
  const types: [string, string, string][] = [["float", ".x", "0.0"], ["vec2", ".xy", "vec2(0.0)"],
    ["vec3", ".xyz", "vec3(0.0)"], ["vec4", "", "vec4(0.0)"]];

  const slots = `vec4 tp_sx_${site};\nvec4 tp_sy_${site};\n`;

  return slots + types.map(([type, swizzle, zero]) => {
    const write = (slot: string) => swizzle ? `${slot}${swizzle} = v;` : `${slot} = v;`;
    const read = (slot: string) => `${slot}${swizzle}`;

    if (operation === "dFdx")
      return `${type} tp_dFdx_${site}(${type} v) { if (tp_lane == 1) { ${write(`tp_sx_${site}`)} }`
        + ` return tp_lane == 0 ? ${read(`tp_sx_${site}`)} - v : ${zero}; }`;

    if (operation === "dFdy")
      return `${type} tp_dFdy_${site}(${type} v) { if (tp_lane == 2) { ${write(`tp_sy_${site}`)} }`
        + ` return tp_lane == 0 ? ${read(`tp_sy_${site}`)} - v : ${zero}; }`;

    return `${type} tp_fwidth_${site}(${type} v) { if (tp_lane == 1) { ${write(`tp_sx_${site}`)} }`
      + ` if (tp_lane == 2) { ${write(`tp_sy_${site}`)} }`
      + ` return tp_lane == 0 ? abs(${read(`tp_sx_${site}`)} - v) + abs(${read(`tp_sy_${site}`)} - v) : ${zero}; }`;
  }).join("\n");
}

// ---------------------------------------------------------------------------------------------------------------------
// The WGSL naga writes back.

interface WgslItem {
  text: string;
  kind: "struct" | "var" | "const" | "fn" | "alias" | "entry" | "other";
  name: string;
  /** For a `var<private>`, its type; for a function, its parameter types. */
  type?: string;
  params?: string[];
}

/** Splits a module into its top level declarations. naga's output is regular enough for a brace counting scan. */
export function splitWgsl(source: string): WgslItem[] {
  const items: WgslItem[] = [];
  let depth = 0;
  let start = 0;

  const push = (end: number) => {
    const text = source.slice(start, end).trim();

    start = end;

    if (!text) return;

    const entry = /^@(fragment|vertex|compute)/.test(text);
    const fn = /^(?:@[\w]+(?:\([^)]*\))?\s*)*fn\s+(\w+)\s*\(([\s\S]*?)\)\s*(?:->\s*[^{]+)?\{/.exec(text);

    if (fn) {
      const params = fn[2].split(",").map(p => p.trim()).filter(Boolean).map(p => p.slice(p.indexOf(":") + 1).trim());

      items.push({ text, kind: entry ? "entry" : "fn", name: fn[1], params });

      return;
    }

    const struct = /^struct\s+(\w+)/.exec(text);

    if (struct) {
      items.push({ text, kind: "struct", name: struct[1] });

      return;
    }

    const variable = /^var<private>\s+(\w+)\s*:\s*([^=;]+)/.exec(text);

    if (variable) {
      items.push({ text, kind: "var", name: variable[1], type: variable[2].trim() });

      return;
    }

    const constant = /^(?:const|override)\s+(\w+)/.exec(text);

    if (constant) {
      items.push({ text, kind: "const", name: constant[1] });

      return;
    }

    const alias = /^alias\s+(\w+)/.exec(text);

    items.push({ text, kind: alias ? "alias" : "other", name: alias?.[1] ?? "" });
  };

  for (let i = 0; i < source.length; i++) {
    const char = source[i];

    if (char === "{") depth++;

    if (char === "}") {
      depth--;

      if (depth === 0) push(i + 1);

      continue;
    }

    if (char === ";" && depth === 0) push(i + 1);
  }

  push(source.length);

  return items;
}

/** Prefixes every module scope name, leaving member accesses and struct member declarations alone. */
function renameItems(items: WgslItem[], rename: Map<string, string>): WgslItem[] {
  const replace = (text: string, insideStruct: boolean) => {
    let result = "";
    const identifier = /[A-Za-z_][A-Za-z0-9_]*/g;
    let last = 0;
    let match: RegExpExecArray | null;

    while ((match = identifier.exec(text))) {
      const name = match[0];
      const before = text.slice(0, match.index).trimEnd();
      const after = text.slice(match.index + name.length).trimStart();
      const member = before.endsWith(".");
      // Inside a struct, a name followed by a colon is a member being declared.
      const declaration = insideStruct && after.startsWith(":") && !/[<(,]$/.test(before);
      const next = rename.get(name);

      result += text.slice(last, match.index) + (next && !member && !declaration ? next : name);
      last = match.index + name.length;
    }

    return result + text.slice(last);
  };

  return items.map(item => ({
    ...item,
    text: replace(item.text, item.kind === "struct"),
    name: rename.get(item.name) ?? item.name,
  }));
}

/** A WGSL type, parsed enough to load a value of it from words. */
type WgslType =
  | { kind: "scalar"; scalar: "f32" | "i32" | "u32" | "bool" }
  | { kind: "vector"; size: number; scalar: "f32" | "i32" | "u32" | "bool" }
  | { kind: "matrix"; columns: number; rows: number }
  | { kind: "array"; element: WgslType; count: number }
  | { kind: "struct"; name: string; members: { name: string; type: WgslType }[] };

function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0, start = 0;

  for (let i = 0; i < text.length; i++) {
    if (text[i] === "<") depth++;
    if (text[i] === ">") depth--;

    if (text[i] === "," && depth === 0) {
      parts.push(text.slice(start, i).trim());
      start = i + 1;
    }
  }

  parts.push(text.slice(start).trim());

  return parts.filter(Boolean);
}

function parseType(text: string, structs: Map<string, { name: string; type: string }[]>): WgslType | null {
  const type = text.trim();

  if (type === "f32" || type === "i32" || type === "u32" || type === "bool") return { kind: "scalar", scalar: type };

  const vector = /^vec([234])<(f32|i32|u32|bool)>$/.exec(type);

  if (vector) return { kind: "vector", size: Number(vector[1]), scalar: vector[2] as "f32" };

  const shorthand = /^vec([234])([fiu])$/.exec(type);

  if (shorthand)
    return { kind: "vector", size: Number(shorthand[1]), scalar: ({ f: "f32", i: "i32", u: "u32" } as const)[shorthand[2] as "f"] };

  const matrix = /^mat([234])x([234])(?:<f32>|f)$/.exec(type);

  if (matrix) return { kind: "matrix", columns: Number(matrix[1]), rows: Number(matrix[2]) };

  const array = /^array<([\s\S]+)>$/.exec(type);

  if (array) {
    const [element, count] = splitTopLevel(array[1]);
    const parsed = parseType(element, structs);
    const size = Number(count);

    return parsed && Number.isInteger(size) ? { kind: "array", element: parsed, count: size } : null;
  }

  const members = structs.get(type);

  if (members) {
    const parsed = members.map(member => ({ name: member.name, type: parseType(member.type, structs) }));

    if (parsed.some(member => !member.type)) return null;

    return { kind: "struct", name: type, members: parsed as { name: string; type: WgslType }[] };
  }

  return null;
}

function wordsOf(type: WgslType): number {
  switch (type.kind) {
    case "scalar": return 1;
    case "vector": return type.size;
    case "matrix": return type.columns * type.rows;
    case "array": return type.count * wordsOf(type.element);
    case "struct": return type.members.reduce((sum, member) => sum + wordsOf(member.type), 0);
  }
}

const scalarRead = (scalar: string, word: string) => {
  switch (scalar) {
    case "f32": return `bitcast<f32>(${word})`;
    case "i32": return `bitcast<i32>(${word})`;
    case "bool": return `(${word} != 0u)`;
    default: return word;
  }
};

/** WGSL that assigns a value of `type` to `target` from the words at `base + offset`. */
function loadCode(target: string, type: WgslType, offset: string, indent: string, loop = 0): string {
  const word = (k: number | string) => `textures[data + ${offset} + ${k}u]`;

  switch (type.kind) {
    case "scalar":
      return `${indent}${target} = ${scalarRead(type.scalar, word(0))};\n`;

    case "vector": {
      const parts = Array.from({ length: type.size }, (_, k) => scalarRead(type.scalar, word(k)));

      return `${indent}${target} = vec${type.size}<${type.scalar}>(${parts.join(", ")});\n`;
    }

    case "matrix": {
      const parts = Array.from({ length: type.columns * type.rows }, (_, k) => scalarRead("f32", word(k)));

      return `${indent}${target} = mat${type.columns}x${type.rows}<f32>(${parts.join(", ")});\n`;
    }

    case "array": {
      const index = `i${loop}`;
      const stride = wordsOf(type.element);

      return `${indent}for (var ${index} = 0u; ${index} < ${type.count}u; ${index} = ${index} + 1u) {\n`
        + loadCode(`${target}[${index}]`, type.element, `${offset} + ${index} * ${stride}u`, `${indent}  `, loop + 1)
        + `${indent}}\n`;
    }

    case "struct": {
      let code = "";
      let at = 0;

      for (const member of type.members) {
        code += loadCode(`${target}.${member.name}`, member.type, `${offset} + ${at}u`, indent, loop);
        at += wordsOf(member.type);
      }

      return code;
    }
  }
}

/** Numbers out of a serialized uniform value: a colour, a vector, a matrix, an array or a plain number. */
function numbersOf(value: UniformValue | undefined): number[] {
  if (typeof value === "number") return [value];
  if (typeof value === "boolean") return [value ? 1 : 0];
  if (Array.isArray(value)) return value.flatMap(entry => numbersOf(entry));

  if (value && typeof value === "object") {
    if ("matrix" in value) return numbersOf(value.matrix);
    if ("elements" in value) return numbersOf(value.elements);
  }

  return [];
}

/** Writes a value of `type` into words, as `loadCode` reads it back. */
function packValue(type: WgslType, value: UniformValue | undefined, out: Uint32Array, floats: Float32Array,
  ints: Int32Array, at: number, textures: ReadonlyMap<string, number>) {
  const scalar = (kind: string, raw: UniformValue | undefined, index: number) => {
    // A texture becomes its place in the table, whatever integer type the sampler was rewritten to.
    if (raw && typeof raw === "object" && !Array.isArray(raw) && typeof raw.texture === "string") {
      out[index] = textures.get(raw.texture) ?? NO_TEXTURE;

      return;
    }

    const number = typeof raw === "boolean" ? Number(raw) : typeof raw === "number" ? raw : 0;

    if (kind === "f32") floats[index] = number;
    else if (kind === "i32") ints[index] = Math.trunc(number);
    else if (kind === "bool") out[index] = number ? 1 : 0;
    else out[index] = Math.max(0, Math.trunc(number)) >>> 0;
  };

  switch (type.kind) {
    case "scalar":
      scalar(type.scalar, Array.isArray(value) ? value[0] : value, at);
      break;

    case "vector": {
      const numbers = typeof value === "number" ? Array(type.size).fill(value) : numbersOf(value);

      for (let k = 0; k < type.size; k++) scalar(type.scalar, numbers[k] ?? 0, at + k);
      break;
    }

    case "matrix": {
      const numbers = numbersOf(value);
      const sourceSize = Math.round(Math.sqrt(numbers.length));

      // Three hands mat3 uniforms over as nine numbers and mat4 as sixteen, both column-major.
      for (let column = 0; column < type.columns; column++)
        for (let row = 0; row < type.rows; row++) {
          const square = sourceSize * sourceSize === numbers.length;
          const index = square ? column * sourceSize + row : column * type.rows + row;
          floats[at + column * type.rows + row] = numbers[index] ?? (column === row ? 1 : 0);
        }

      break;
    }

    case "array": {
      const stride = wordsOf(type.element);
      const flat = !Array.isArray(value) ? [] : value;

      // A float array may arrive flat (Float32Array becomes numbers) for vector elements.
      const elements = type.element.kind !== "scalar" && flat.every(entry => typeof entry === "number")
        ? Array.from({ length: type.count }, (_, k) => flat.slice(k * stride, (k + 1) * stride) as UniformValue)
        : flat;

      for (let k = 0; k < type.count; k++)
        packValue(type.element, elements[k], out, floats, ints, at + k * stride, textures);
      break;
    }

    case "struct": {
      const record = value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, UniformValue> : {};
      let offset = 0;

      for (const member of type.members) {
        packValue(member.type, record[member.name], out, floats, ints, at + offset, textures);
        offset += wordsOf(member.type);
      }

      break;
    }
  }
}

/** The kernel's implementation of each texture stand-in, chosen by its name and the types of its parameters. */
function textureImplementation(base: string, params: string[]): string {
  const sampler = params[0] ?? "";
  const unsupported = "return vec4<f32>(0.0);";

  if (base === "tp_texSize") return `return shaderTextureSize(u32(${"s"}));`;
  if (base === "tp_texelFetch") return "return shaderTexelFetch(s, a1);";

  if (sampler.startsWith("vec2")) return unsupported;

  if (base === "tp_texProj")
    return params[1]?.startsWith("vec4") ? "return shaderSample2D(s, a1.xy / a1.w);" : "return shaderSample2D(s, a1.xy / a1.z);";

  if (sampler === "i32") return "return shaderSampleCube(u32(s), a1);";

  return "return shaderSample2D(s, a1);";
}

let naga: Promise<typeof Naga> | null = null;

/** naga is only loaded for a scene that has shaders; its WebAssembly has to be instantiated once in a browser. */
async function loadNaga() {
  naga ??= import("naga-wasm").then(async module => {
    // In Node the module instantiates itself on import; in a browser, init() fetches the binary next to it.
    if (typeof window !== "undefined") await module.init();

    return module;
  });

  return naga;
}

/** naga's GLSL to WGSL, with its own error text on failure. */
async function translate(source: string): Promise<string> {
  const module = await loadNaga();
  const parsed = module.parseGlsl(source, { stage: "fragment" });

  try {
    const info = module.validate(parsed);

    try {
      return module.writeWgsl(parsed, info);
    } finally {
      info.free();
    }
  } finally {
    parsed.free();
  }
}

/** Builds a program from naga's WGSL: renamed, its stand-ins replaced, with a loader, its varyings and an entry. */
export function buildProgram(
  id: number, kind: ProgramKind, wgsl: string, prepared: PreparedGlsl): ShaderProgram {
  const prefix = `p${id}_`;
  const items = splitWgsl(wgsl).filter(item => item.kind !== "entry");
  const names = new Map<string, string>();

  for (const item of items) if (item.name) names.set(item.name, prefix + item.name);

  const structSource = new Map<string, { name: string; type: string }[]>();

  for (const item of items.filter(entry => entry.kind === "struct")) {
    const body = item.text.slice(item.text.indexOf("{") + 1, item.text.lastIndexOf("}"));

    const members = splitTopLevel(body.replace(/@\w+(\([^)]*\))?/g, "")).map(member => {
      const colon = member.indexOf(":");

      return { name: member.slice(0, colon).trim(), type: member.slice(colon + 1).trim() };
    });

    structSource.set(item.name, members);
  }

  const globals = new Map(items.filter(item => item.kind === "var").map(item => [item.name, item.type as string]));

  /** naga keeps a global's name unless it collides with a WGSL keyword, in which case it suffixes it. */
  const globalNamed = (name: string) => {
    if (globals.has(name)) return name;

    // naga's namer suffixes a global whose name a parameter or a keyword already took: name_, name_1, name_2...
    const pattern = new RegExp(`^${name}_\\d*$`);

    for (const candidate of globals.keys()) if (pattern.test(candidate)) return candidate;

    return null;
  };

  const renamed = renameItems(items, names);

  // The stand-ins: their WGSL bodies are replaced by calls into the kernel.
  const stand = /^tp_(tex|texLod|texGrad|texProj|texelFetch|texSize)(_\d+)?$/;
  const output: string[] = [];

  renamed.forEach((item, index) => {
    const original = items[index];

    if (original.kind === "fn" && stand.test(original.name)) {
      const base = `tp_${stand.exec(original.name)?.[1]}`;
      const params = original.params ?? [];
      const signature = params.map((type, k) => `${k === 0 ? "s" : `a${k}`}: ${type}`).join(", ");
      const returns = base === "tp_texSize" ? "vec2<i32>" : "vec4<f32>";

      output.push(`fn ${item.name}(${signature}) -> ${returns} {\n  ${textureImplementation(base, params)}\n}`);

      return;
    }

    output.push(item.text);
  });

  // The uniform loader.
  const uniformTypes: { global: string; name: string; type: WgslType }[] = [];

  for (const name of prepared.uniforms) {
    const global = globalNamed(name);
    const type = global ? parseType(globals.get(global) as string, structSource) : null;

    if (global && type) uniformTypes.push({ global, name, type });
  }

  // The shader prelude's built-ins are uniforms too, filled the same way.
  for (const name of ["viewMatrix", "cameraPosition", "isOrthographic"]) {
    const global = globalNamed(name);

    if (global && !uniformTypes.some(entry => entry.global === global)) {
      const type = parseType(globals.get(global) as string, structSource);

      if (type) uniformTypes.push({ global, name, type });
    }
  }

  let loader = `fn ${prefix}load(data: u32) {\n`;
  let words = 0;

  for (const uniform of uniformTypes) {
    loader += loadCode(prefix + uniform.global, uniform.type, `${words}u`, "  ");
    words += wordsOf(uniform.type);
  }

  loader += "}\n";

  // Varyings, laid out per corner in declaration order.
  const varyings: ProgramVarying[] = [];
  let stride = 0;
  let varyingCode = "";

  const sourceVaryings = kind === "foreground" ? [] : prepared.varyings;

  for (const varying of sourceVaryings) {
    const global = globalNamed(varying.name);
    const type = global ? parseType(globals.get(global) as string, structSource) : null;

    if (!global || !type || (type.kind !== "scalar" && type.kind !== "vector")) continue;

    const size = type.kind === "scalar" ? 1 : type.size;
    const scalar = type.scalar;
    const flat = varying.flat || scalar !== "f32";

    varyings.push({ name: varying.name, size, offset: stride, flat });

    const component = (k: number) => {
      const read = `shaderVarying(at, ${stride + k}u, bary, ${flat ? "true" : "false"})`;

      return scalar === "f32" ? read : scalar === "bool" ? `(${read} != 0.0)` : `${scalar}(round(${read}))`;
    };

    const value = size === 1 ? component(0)
      : `vec${size}<${scalar}>(${Array.from({ length: size }, (_, k) => component(k)).join(", ")})`;

    varyingCode += `  ${prefix}${global} = ${value};\n`;
    stride += size;
  }

  const set = (name: string, value: string) => {
    const global = globalNamed(name);

    return global ? `  ${prefix}${global} = ${value};\n` : "";
  };

  const usesDerivatives = prepared.derivativeSites > 0;
  const outputName = prepared.outputs.map(globalNamed).find((name): name is string => !!name) ?? globalNamed("tp_FragColor");

  let setup = `fn ${prefix}setup(input: ShaderInput, lane: i32) {\n`;

  setup += set("tp_lane", "lane");
  setup += "  let bary = select(select(input.bary, input.baryDy, lane == 2), input.baryDx, lane == 1);\n";
  setup += "  var coord = input.fragCoord;\n  if (lane == 1) { coord.x = coord.x + 1.0; }\n"
    + "  if (lane == 2) { coord.y = coord.y + 1.0; }\n";
  setup += set("tp_FragCoord", "coord");
  setup += set("tp_FrontFacing", "input.frontFacing");
  setup += set("tp_PointCoord", "input.pointCoord");
  setup += set("tp_discarded", "false");

  // After <opaque_fragment>, gl_FragColor already holds the surface's colour.
  if (outputName) setup += `  ${prefix}${outputName} = ${kind === "after" ? "input.fragColor" : "vec4<f32>(0.0, 0.0, 0.0, 1.0)"};\n`;

  if (kind === "foreground") {
    setup += set("vDirection", "input.direction");
  } else if (varyingCode) {
    setup += "  let at = shaderVaryingAddress(input.material, input.triangle);\n" + varyingCode;
  }

  if (kind === "patched" || kind === "after") {
    setup += set("diffuseColor", "input.diffuseColor");
    setup += set("normal", "input.normal");
    setup += set("roughnessFactor", "input.roughness");
    setup += set("metalnessFactor", "input.metalness");
    setup += set("totalEmissiveRadiance", "input.emissive");
    setup += set("vViewPosition", "input.viewPosition");
    setup += set("vUv", "input.uv");
    setup += set("outgoingLight", "input.diffuseColor.rgb");
    setup += set("geometryNormal", "input.geometryNormal");
    setup += set("nonPerturbedNormal", "input.geometryNormal");
    setup += set("geometryPosition", "-input.viewPosition");
    setup += set("geometryViewDir", "normalize(input.viewPosition)");
    setup += set("faceDirection", "select(-1.0, 1.0, input.frontFacing)");
    setup += set("vNormal", "input.geometryNormal");
  }

  setup += "}\n";

  // naga renames GLSL's main to main_1 and gives the name to its entry point, which was dropped above.
  const functions = new Set(items.filter(item => item.kind === "fn").map(item => item.name));
  const main = names.get(functions.has("main_1") ? "main_1" : "main");

  if (!main || !functions.has(functions.has("main_1") ? "main_1" : "main"))
    throw new Error("GPU renderer: the translated shader has no main function.");
  let run = `fn ${prefix}run(input: ShaderInput) -> ShaderOutput {\n  ${prefix}load(input.data);\n`;

  if (usesDerivatives)
    run += `  ${prefix}setup(input, 1);\n  ${main}();\n  ${prefix}setup(input, 2);\n  ${main}();\n`;

  run += `  ${prefix}setup(input, 0);\n  ${main}();\n  var result = noShaderOutput(input);\n`;

  if (outputName) run += `  result.color = ${prefix}${outputName};\n`;

  run += globalNamed("tp_discarded") ? `  result.discarded = ${prefix}${globalNamed("tp_discarded")};\n` : "";

  if (kind === "patched") {
    const read = (field: string, name: string) => (globalNamed(name) ? `  result.${field} = ${prefix}${globalNamed(name)};\n` : "");

    run += read("diffuseColor", "diffuseColor") + read("normal", "normal") + read("roughness", "roughnessFactor")
      + read("metalness", "metalnessFactor") + read("emissive", "totalEmissiveRadiance");
  }

  run += "  return result;\n}\n";

  const totalWords = words;

  return {
    id,
    kind,
    wgsl: `// program ${id} (${kind})\n${output.join("\n\n")}\n\n${loader}\n${setup}\n${run}`,
    varyings,
    varyingStride: stride,
    usesDerivatives,
    uniformWords: totalWords,
    packUniforms(values, textures) {
      const out = new Uint32Array(Math.max(1, totalWords));
      const floats = new Float32Array(out.buffer);
      const ints = new Int32Array(out.buffer);
      let at = 0;

      for (const uniform of uniformTypes) {
        packValue(uniform.type, values[uniform.name], out, floats, ints, at, textures);
        at += wordsOf(uniform.type);
      }

      return out;
    },
  };
}

/** Translates one fragment shader into a program, or explains why it could not be. */
export async function compileProgram(
  id: number, kind: ProgramKind, shader: Pick<SerializedShader, "fragment" | "defines">, raw = false): Promise<ShaderProgram> {
  const prepared = prepareFragmentShader(shader, kind, { raw });
  let wgsl: string;

  try {
    wgsl = await translate(prepared.source);
  } catch(error) {
    // WGSL has no fallthrough. Only then are the shader's switches rewritten, so every other shader keeps naga's own.
    if (!/fall-through/.test((error as { formatted?: string }).formatted ?? (error as Error).message)) throw error;

    prepared.source = rewriteSwitches(prepared.source);
    wgsl = await translate(prepared.source);
  }

  return buildProgram(id, kind, wgsl, prepared);
}

/** Where the bracket opened at `open` closes, counting the same kind of bracket. */
function closing(source: string, open: number): number {
  const pair: Record<string, string> = { "(": ")", "{": "}" };
  const opener = source[open], closer = pair[opener];
  let depth = 0;

  for (let i = open; i < source.length; i++) {
    if (source[i] === opener) depth++;

    if (source[i] === closer && --depth === 0) return i;
  }

  return -1;
}

let switchCount = 0;

/**
 * GLSL switches turned into what WGSL can say, for a shader whose cases fall through: one pass of a loop, so `break`
 * still leaves it, and each case an `if` that is entered when its label matches or the case before it fell through.
 * `default` is entered when no label matches. A `continue` inside such a case leaves the switch instead of continuing
 * the loop around it, which is the one thing this does not keep.
 */
export function rewriteSwitches(source: string): string {
  const pattern = /\bswitch\s*\(/g;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(source))) {
    const open = source.indexOf("(", match.index);
    const close = closing(source, open);
    const brace = source.indexOf("{", close);
    const end = closing(source, brace);

    if (close < 0 || brace < 0 || end < 0) break;

    const selector = source.slice(open + 1, close);
    const body = rewriteSwitches(source.slice(brace + 1, end));
    const labels: { at: number; length: number; value: string | null }[] = [];
    const label = /\b(case\s+([^:]+?)|default)\s*:/g;
    let found: RegExpExecArray | null;

    while ((found = label.exec(body))) {
      // Only labels of this switch: at its own nesting depth, not inside a block or a call.
      const before = body.slice(0, found.index);
      const depth = (before.match(/[{(]/g)?.length ?? 0) - (before.match(/[})]/g)?.length ?? 0);

      if (depth === 0) labels.push({ at: found.index, length: found[0].length, value: found[2] ?? null });
    }

    const n = switchCount++;
    const values = labels.filter(entry => entry.value !== null).map(entry => `tp_sv${n} == int(${entry.value})`);

    let rewritten = `for (int tp_sw${n} = 0; tp_sw${n} < 1; tp_sw${n}++) {\n`
      + `int tp_sv${n} = int(${selector});\nbool tp_sf${n} = false;\n`;

    labels.forEach((entry, k) => {
      const statements = body.slice(entry.at + entry.length, labels[k + 1]?.at ?? body.length);

      const condition = entry.value !== null
        ? `tp_sv${n} == int(${entry.value})`
        : `!(${values.join(" || ") || "false"})`;

      rewritten += `if (tp_sf${n} || ${condition}) { tp_sf${n} = true;\n${statements}\n}\n`;
    });

    rewritten += "}";
    source = source.slice(0, match.index) + rewritten + source.slice(end + 1);
    pattern.lastIndex = match.index + rewritten.length;
  }

  return source;
}

/** The CPU renderer's foreground wrapper, with the sky texture it samples as a plain 2D lookup. */
export function foregroundSource(source: string): string {
  const replaced = source.replace(/\btextureCube\s*\(\s*uSkyMap\s*,/g, "cpuSkyMap(");

  return `
varying vec3 vDirection;
uniform sampler2D uCpuSky;
vec4 cpuSkyMap(vec3 direction) {
  vec3 d = normalize(direction);
  return texture2D(uCpuSky, vec2(atan(d.z, d.x) / 6.283185307179586 + 0.5, acos(clamp(d.y, -1.0, 1.0)) / 3.141592653589793));
}
${replaced}
void main() { gl_FragColor = vec4(skyForeground(normalize(vDirection)), 1.0); }
`;
}

const NO_PROGRAMS: ShaderPrograms = {
  programs: [],
  fragment: () => null,
  vertex: () => null,
  after: () => null,
  foreground: null,
  failures: [],
  key: "",
};

/**
 * Translates every shader a scene uses. A shader that fails to translate is reported and its material falls back to
 * its colour, as before this existed, rather than failing the frame.
 */
export async function compileShaderPrograms(
  scene: SerializedScene, foreground?: string): Promise<ShaderPrograms> {
  const materials = new Set<SerializedMaterial>();

  for (const mesh of scene.meshes) for (const material of mesh.materials) if (material?.shader) materials.add(material);

  if (!materials.size && !foreground) return NO_PROGRAMS;

  const programs: ShaderProgram[] = [];
  const byKey = new Map<string, Promise<ShaderProgram | null>>();
  // Keyed by what decides the program, not by the material object: a material copied to bind other uniforms (a
  // planar capture's, for one) runs the same shaders.
  const fragments = new Map<string, ShaderProgram>();
  const vertices = new Map<string, CpuShader>();
  const afters = new Map<string, ShaderProgram>();
  const vertexByKey = new Map<string, CpuShader>();
  const failures: string[] = [];
  const counts = lightCounts(scene);

  const programFor = (key: string, compile: (id: number) => Promise<ShaderProgram>, label: string) => {
    let pending = byKey.get(key);

    if (!pending) {
      const id = byKey.size;

      pending = compile(id).then(
        program => {
          programs[id] = program;

          return program;
        },
        (error: unknown) => {
          failures.push(`${label}: ${(error as { formatted?: string }).formatted ?? (error as Error).message ?? error}`);

          return null;
        });
      byKey.set(key, pending);
    }

    return pending;
  };

  for (const material of materials) {
    const shader = material.shader as SerializedShader;
    const kind: ProgramKind = shader.patched ? "patched" : "surface";
    const raw = material.type === "RawShaderMaterial";
    const defines = { ...shader.defines, ...counts };
    const key = `${kind}|${raw}|${JSON.stringify(defines)}|${shader.fragment}`;

    const program = await programFor(key, id => compileProgram(id, kind, { fragment: shader.fragment, defines }, raw),
      material.name);

    if (program) fragments.set(shaderKey(material), program);

    // The CPU renderer's snapshot keeps what a hook does after <opaque_fragment> as a second shader, often empty.
    const after = shader.patched ? shader.fragmentAfter : undefined;

    if (after && hasStatements(after)) {
      const afterProgram = await programFor(`after|${JSON.stringify(defines)}|${after}`,
        id => compileProgram(id, "after", { fragment: after, defines }), `${material.name} (after opaque)`);

      if (afterProgram) afters.set(shaderKey(material), afterProgram);
    }

    const vertexKey = `${JSON.stringify(defines)}|${shader.vertex}`;

    try {
      let vertex = vertexByKey.get(vertexKey);

      if (!vertex) {
        vertex = new CpuShader(shader.vertex, defines, SHADER_CHUNKS);
        vertexByKey.set(vertexKey, vertex);
      }

      vertices.set(shaderKey(material), vertex);
    } catch(error) {
      failures.push(`${material.name} (vertex): ${(error as Error).message}`);
    }
  }

  const sky = foreground
    ? await programFor(`foreground|${foreground}`,
      id => compileProgram(id, "foreground", { fragment: foregroundSource(foreground), defines: {} }, true), "sky foreground")
    : null;

  const compiled = programs.filter(Boolean);

  for (const failure of failures) console.warn(`GPU renderer: a shader could not be translated and renders as its colour. ${failure}`);

  return {
    programs: compiled,
    fragment: material => (material.shader ? fragments.get(shaderKey(material)) ?? null : null),
    vertex: material => (material.shader ? vertices.get(shaderKey(material)) ?? null : null),
    after: material => (material.shader ? afters.get(shaderKey(material)) ?? null : null),
    foreground: sky,
    failures,
    key: compiled.map(program => program.wgsl).join("\n"),
  };
}

/** What decides a material's programs: its type and the source and defines of its shaders. */
function shaderKey(material: SerializedMaterial): string {
  const shader = material.shader as SerializedShader;

  return JSON.stringify([material.type, shader.patched, shader.vertex, shader.fragment, shader.fragmentAfter ?? "",
    shader.defines]);
}

/** Whether a shader's main does anything: a hook template with nothing added after a chunk is only empty lines. */
function hasStatements(source: string): boolean {
  const start = source.indexOf("void main()");
  const open = start < 0 ? -1 : source.indexOf("{", start);

  if (open < 0) return false;

  const end = closing(source, open);

  return stripComments(source.slice(open + 1, end < 0 ? undefined : end)).replace(/#[^\n]*/g, "").trim().length > 0;
}

/** The light counts Three defines for a patched shader, as the CPU renderer passes them. */
function lightCounts(scene: SerializedScene): Record<string, number> {
  return {
    NUM_DIR_LIGHTS: scene.lights.filter(light => light.type === "DirectionalLight").length,
    NUM_POINT_LIGHTS: scene.lights.filter(light => light.type === "PointLight").length,
    NUM_SPOT_LIGHTS: scene.lights.filter(light => light.type === "SpotLight").length,
  };
}

/** The WGSL every program relies on, and the dispatcher the surface code calls. */
export function programsWgsl(programs: ShaderPrograms | null): string {
  const list = programs?.programs ?? [];
  const cases = list.map(program => `    case ${program.id}u: { return p${program.id}_run(input); }`).join("\n");

  return `${list.map(program => program.wgsl).join("\n\n")}

/** Runs the translated shader with this id; an unknown one leaves the surface as it was. */
fn runProgram(id: u32, input: ShaderInput) -> ShaderOutput {
  switch id {
${cases}
    default: { }
  }

  return noShaderOutput(input);
}
`;
}
