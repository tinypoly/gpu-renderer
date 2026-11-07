import type { ShaderPrograms } from "./shaderProgram.js";
import type { SerializedMaterial, SerializedScene, SerializedTexture } from "./upstream.js";

/**
 * Every texture of a scene in one storage buffer, led by a table saying where each one lives. Core WebGPU has no binding
 * arrays and `texture_2d_array` wants one size and one format for every layer, which a Three.js scene does not provide,
 * so the texels are read by hand. See the decision in ARCHITECTURE.md.
 *
 * Byte textures keep their eight bits per channel, four to a word, and are decoded from sRGB as they are read. A byte
 * texture of which only one channel is ever read (a roughness or a bump map) keeps that channel alone, four texels to a
 * word. Float textures keep all four, because their range is the reason they are float at all.
 */
export interface PackedTextures {
  /**
   * One buffer for all of it: the lookup tables first (see `TEXTURE_TABLE`), then `TEXTURE_WORDS` words per texture, as
   * `TextureInfo` in `wgsl/common.wgsl.ts`, then the texels each entry points at. A byte texture takes one word per
   * texel, RGBA in the low to high bytes; a float texture takes four, each the bits of one linear channel. Keeping them
   * in one binding is what holds the shader inside the eight storage buffers a WebGPU device has to offer.
   */
  data: Uint32Array;
  count: number;
  /** The index in the table of each texture, by the id the materials refer to it with. */
  indexOf: Map<string, number>;
  /** Where the words a caller appended begin, which is how the environment's distribution rides along. */
  extraOffset: number;
}

/** Textures and raw words that are not part of the scene snapshot, such as the environment and its distribution. */
export interface TextureExtras {
  textures?: readonly SerializedTexture[];
  words?: Uint32Array;
}

/** Sixteen words per entry: see `TextureInfo` in `wgsl/common.wgsl.ts`. */
export const TEXTURE_WORDS = 16;

/**
 * Lookup tables that lead the buffer, at offsets the shader knows as constants: the second Sobol dimension one byte of
 * the index at a time (four tables of 256), then the linear value of each of the 256 sRGB codes. Both replace work the
 * kernel did on every read with the very numbers it computed, so nothing they feed changes.
 */
export const SOBOL_TABLE = 0;
export const SRGB_TABLE = SOBOL_TABLE + 4 * 256;
/** Where the texture table starts, after the lookup tables. */
export const TEXTURE_TABLE = SRGB_TABLE + 256;

/** The four byte tables of the Sobol sequence's second dimension, whose generator matrix is Pascal's triangle mod 2. */
function sobolTables(): Uint32Array {
  const directions: number[] = [];
  let direction = 0x80000000;

  for (let bit = 0; bit < 32; bit++) {
    directions.push(direction >>> 0);
    direction = (direction ^ (direction >>> 1)) >>> 0;
  }

  const tables = new Uint32Array(4 * 256);

  for (let byte = 0; byte < 4; byte++) {
    for (let value = 0; value < 256; value++) {
      let xor = 0;

      for (let bit = 0; bit < 8; bit++) if (value & (1 << bit)) xor ^= directions[byte * 8 + bit];

      tables[byte * 256 + value] = xor >>> 0;
    }
  }

  return tables;
}

/** The linear value of every sRGB code, worked out in double precision and rounded once. */
function srgbTable(): Float32Array {
  return Float32Array.from({ length: 256 }, (_, code) => {
    const value = Math.fround(code / 255);

    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
}

const LOOKUP = (() => {
  const words = new Uint32Array(TEXTURE_TABLE);

  words.set(sobolTables(), SOBOL_TABLE);
  words.set(new Uint32Array(srgbTable().buffer), SRGB_TABLE);

  return words;
})();

/** Where a texture's entry starts in the buffer. */
export const textureEntry = (index: number) => TEXTURE_TABLE + index * TEXTURE_WORDS;

/** What a material's map index means when the material has no such map. */
export const NO_TEXTURE = 0xffffffff;

const TEXTURE_FLOAT = 1;
const TEXTURE_SRGB = 2;
const TEXTURE_NEAREST = 4;
const TEXTURE_FLIP_Y = 8;
/** An environment map whose mapping bends the view ray instead of mirroring it. */
const TEXTURE_REFRACTION = 16;
/** One byte a texel: the only channel the scene reads of it, which the shader hands back in every channel. */
const TEXTURE_SINGLE = 32;

/** Three's own wrap constants, which the shader compares against. */
const REPEAT = 1000;
const MIRRORED_REPEAT = 1002;

/** How many UV sets the geometry carries, and so the highest channel a texture may ask for. */
export const UV_CHANNELS = 2;

/** Whether a word's first byte is its lowest, as it is on the device: then RGBA bytes are already RGBA words. */
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

const texelCount = (texture: SerializedTexture) => Math.max(0, texture.width) * Math.max(0, texture.height);

const isFloat = (texture: SerializedTexture) => texture.data instanceof Float32Array;

/**
 * Whether the shader has to decode this texture's colour from sRGB. Only byte textures can still be encoded: a float
 * texture was converted to linear when the scene was serialized, and had its colour space rewritten to say so.
 */
const isEncoded = (texture: SerializedTexture) => !isFloat(texture) && texture.colorSpace === "srgb";

/**
 * The channels the kernel reads from each material map slot, as the WGSL reads them: roughness from green, metalness
 * from blue, and so on. A slot missing here counts as reading all four.
 */
export const SLOT_CHANNELS: Readonly<Record<string, string>> = {
  map: "rgba", emissiveMap: "rgb", roughnessMap: "g", metalnessMap: "b", normalMap: "rgb", clearcoatNormalMap: "rgb",
  alphaMap: "g", transmissionMap: "r", thicknessMap: "g", clearcoatMap: "r", clearcoatRoughnessMap: "g",
  sheenColorMap: "rgb", sheenRoughnessMap: "a", iridescenceMap: "r", iridescenceThicknessMap: "g", anisotropyMap: "rgb",
  specularMap: "r", specularIntensityMap: "a", specularColorMap: "rgb", bumpMap: "r", displacementMap: "r",
  lightMap: "rgb", gradientMap: "r", matcap: "rgb", envMap: "rgb",
};

/** Every string inside a value, with the first path it was found at, without walking into typed arrays. */
function collectStrings(value: unknown, path: string, into: Map<string, string>, seen = new Set<unknown>()) {
  if (typeof value === "string") {
    if (!into.has(value)) into.set(value, path);

    return;
  }

  if (!value || typeof value !== "object" || seen.has(value) || ArrayBuffer.isView(value)) return;

  seen.add(value);

  for (const [key, entry] of Object.entries(value)) collectStrings(entry, `${path}.${key}`, into, seen);
}

/** How each texture of a scene is stored: one channel, or all four and why. */
export interface TextureChannels {
  /** The channel kept (0 for red to 3 for alpha) of each texture stored one byte a texel. */
  single: Map<string, number>;
  /** Why every other texture keeps all four. */
  whole: Map<string, string>;
}

/** Texture indices no table reaches, one per texture, which a program's packed uniforms give back where it reads one. */
const PROBE = 0x7f000000;

/**
 * The textures a material's translated programs read: their uniforms are packed with a probe index for every texture,
 * and the probes that come back are the textures the programs declare. A texture a hook's uniforms name but its
 * program never declares (Three hands a patched material's every map to its shader) is not read by it.
 */
function programTextures(material: SerializedMaterial, programs: Pick<ShaderPrograms, "fragment" | "after">,
  textures: readonly SerializedTexture[]): Set<string> {
  const probes = new Map(textures.map((texture, index) => [texture.id, PROBE + index]));
  const uniforms = { ...material.shader?.uniforms, ...material.time !== undefined ? { time: material.time } : {} };
  const read = new Set<string>();

  for (const program of [programs.fragment(material), programs.after(material)]) {
    if (!program) continue;

    for (const word of program.packUniforms(uniforms, probes))
      if (word >= PROBE && word < PROBE + textures.length) read.add(textures[word - PROBE].id);
  }

  return read;
}

/**
 * Which byte textures of a scene are read through one channel only: every reference to the texture is a material map
 * slot, and every such slot reads the same channel. A texture named anywhere else (a light's projected map, the sky, a
 * planar capture) is read in ways this cannot see, and keeps all four. With the scene's translated programs, a
 * shader's uniforms count only for the textures its program reads; without them, any texture they name keeps all four.
 */
export function textureChannels(scene: SerializedScene,
  programs?: Pick<ShaderPrograms, "fragment" | "after">): TextureChannels {
  const read = new Map<string, Set<string>>();
  const elsewhere = new Map<string, string>();
  const materials = new Set(scene.meshes.flatMap(mesh => mesh.materials));

  for (const material of materials) {
    if (!material) continue;

    const { maps, ...everything } = material;
    const name = `material "${material.name || material.type}"`;
    // The shader's uniforms are read only through its translated programs, which say what they sample.
    const { shader, ...rest } = everything;

    if (programs && shader) {
      for (const id of programTextures(material, programs, scene.textures ?? []))
        if (!elsewhere.has(id)) elsewhere.set(id, `the shader of ${name}`);
    } else if (shader) {
      collectStrings(shader, `${name}.shader`, elsewhere);
    }

    for (const [slot, id] of Object.entries(maps ?? {})) {
      if (typeof id !== "string") continue;

      let channels = read.get(id);

      if (!channels) read.set(id, channels = new Set());

      for (const channel of SLOT_CHANNELS[slot] ?? "rgba") channels.add(channel);
    }

    collectStrings(rest, name, elsewhere);
  }

  // The rest of the scene: its lights, its sky and whatever else names a texture.
  for (const [key, value] of Object.entries(scene))
    if (key !== "meshes" && key !== "textures") collectStrings(value, key, elsewhere);

  const single = new Map<string, number>();
  const whole = new Map<string, string>();

  for (const texture of scene.textures ?? []) {
    const channels = read.get(texture.id);
    const named = elsewhere.get(texture.id);

    if (isFloat(texture)) whole.set(texture.id, "float");
    else if (!channels) whole.set(texture.id, named ? `named by ${named}` : "not read by any map");
    else if (channels.size !== 1) whole.set(texture.id, `reads ${[...channels].join("")}`);
    else if (named) whole.set(texture.id, `named by ${named}`);
    else single.set(texture.id, "rgba".indexOf([...channels][0]));
  }

  return { single, whole };
}

/** The byte textures of a scene read through one channel only, with that channel: see `textureChannels`. */
export function singleChannelTextures(scene: SerializedScene,
  programs?: Pick<ShaderPrograms, "fragment" | "after">): Map<string, number> {
  return textureChannels(scene, programs).single;
}

/** Words a texture's texels take in the buffer. */
function texelWords(texture: SerializedTexture, single: boolean): number {
  const texels = texelCount(texture);

  return isFloat(texture) ? texels * 4 : single ? Math.ceil(texels / 4) : texels;
}

/**
 * Where everything will go, before anything is written: which index each texture gets and where the words after the
 * texels begin. Materials and lights point into that tail by absolute offset, so they need it before the buffer exists.
 */
export interface TexturePlan {
  textures: SerializedTexture[];
  indexOf: Map<string, number>;
  /** The channel kept of each texture stored one byte a texel, by id. */
  channelOf: ReadonlyMap<string, number>;
  /** The lookup tables and the texture table together: where the texels begin. */
  tableWords: number;
  /** Where the words appended after the texels begin. */
  extraOffset: number;
}

export function planTextures(scene: SerializedScene, extraTextures: readonly SerializedTexture[] = [],
  channelOf: ReadonlyMap<string, number> = singleChannelTextures(scene)): TexturePlan {
  const seen = new Set<string>();

  // A texture id appears once, whoever refers to it; the first copy wins.
  const textures = [...scene.textures ?? [], ...extraTextures]
    .filter(texture => texelCount(texture) > 0 && texture.data.length >= texelCount(texture) * 4)
    .filter(texture => !seen.has(texture.id) && !!seen.add(texture.id));

  const indexOf = new Map(textures.map((texture, index) => [texture.id, index]));
  const tableWords = TEXTURE_TABLE + Math.max(1, textures.length) * TEXTURE_WORDS;
  const kept = new Map([...channelOf].filter(([id]) => indexOf.has(id)));
  const words = textures.reduce((sum, texture) => sum + texelWords(texture, kept.has(texture.id)), 0);

  return { textures, indexOf, channelOf: kept, tableWords, extraOffset: tableWords + words };
}

export function packTextures(scene: SerializedScene, extras: TextureExtras = {}): PackedTextures {
  return writeTextures(planTextures(scene, extras.textures), extras.words ?? new Uint32Array(0));
}

/**
 * The lookup tables and the texture table a plan describes, written at the head of `words`. Returns where each
 * texture's texels start, in words.
 */
function writeTable(words: Uint32Array, plan: TexturePlan): number[] {
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const offsets: number[] = [];
  let texelAt = plan.tableWords;

  words.set(LOOKUP, 0);

  plan.textures.forEach((texture, index) => {
    const at = textureEntry(index);
    const float = isFloat(texture);

    offsets.push(texelAt);
    words[at] = texture.width;
    words[at + 1] = texture.height;
    words[at + 2] = texelAt;

    words[at + 3] = (float ? TEXTURE_FLOAT : 0)
      | (isEncoded(texture) ? TEXTURE_SRGB : 0)
      | (texture.nearest ? TEXTURE_NEAREST : 0)
      | (texture.flipY ? TEXTURE_FLIP_Y : 0)
      | (texture.refraction ? TEXTURE_REFRACTION : 0)
      | (plan.channelOf.has(texture.id) ? TEXTURE_SINGLE : 0);

    words[at + 4] = texture.wrapS;
    words[at + 5] = texture.wrapT;
    words[at + 6] = Math.min(Math.max(0, texture.channel | 0), UV_CHANNELS - 1);
    words[at + 7] = 0;

    // Three's Matrix3 arrives column major, so a row of the affine transform reads across the columns.
    const matrix = texture.matrix?.length === 9 ? texture.matrix : [1, 0, 0, 0, 1, 0, 0, 0, 1];

    floats.set([matrix[0], matrix[3], matrix[6], 0], at + 8);
    floats.set([matrix[1], matrix[4], matrix[7], 0], at + 12);
    texelAt += texelWords(texture, plan.channelOf.has(texture.id));
  });

  return offsets;
}

/**
 * A texture's texels as the buffer holds them: viewed straight from the snapshot where the bytes already agree, the one
 * channel kept copied out four texels to a word, or null for a byte texture a big endian page has to turn around.
 */
function texelView(texture: SerializedTexture, channel: number | undefined): ArrayBufferView | null {
  const texels = texelCount(texture);

  if (isFloat(texture)) return (texture.data as Float32Array).subarray(0, texels * 4);

  if (channel !== undefined) {
    const source = texture.data as Uint8Array;
    // Bytes in the order the device reads a word's, lowest first, so this needs no turning around on any page.
    const kept = new Uint8Array(Math.ceil(texels / 4) * 4);

    for (let texel = 0; texel < texels; texel++) kept[texel] = source[texel * 4 + channel];

    return kept;
  }

  return LITTLE_ENDIAN ? (texture.data as Uint8Array).subarray(0, texels * 4) : null;
}

/** The buffer a plan describes, with `extraWords` appended at its `extraOffset`. */
export function writeTextures(plan: TexturePlan, extraWords: Uint32Array): PackedTextures {
  const { textures, indexOf } = plan;
  // WebGPU rejects an empty binding, and a scene with no textures at all is ordinary.
  const data = new Uint32Array(Math.max(4, plan.extraOffset + extraWords.length));
  const offsets = writeTable(data, plan);

  textures.forEach((texture, index) => {
    const texelAt = offsets[index];
    const view = texelView(texture, plan.channelOf.get(texture.id));

    if (view) {
      // The channels keep their bits, and bytes already sit in the order the words want them: copied whole.
      new Uint8Array(data.buffer, texelAt * 4, view.byteLength)
        .set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));

      return;
    }

    const source = texture.data as Uint8Array;

    for (let texel = 0; texel < texelCount(texture); texel++) {
      const offset = texel * 4;

      data[texelAt + texel] = source[offset]
        | (source[offset + 1] << 8)
        | (source[offset + 2] << 16)
        | (source[offset + 3] << 24);
    }
  });

  data.set(extraWords, plan.extraOffset);

  return { data, count: textures.length, indexOf, extraOffset: plan.extraOffset };
}

/**
 * The same buffer `writeTextures` builds, written to the device a piece at a time: the tables, then each texture's
 * texels straight from the snapshot, then the extra words. The page never holds the whole buffer, which for a scene of
 * many large textures is hundreds of megabytes it would only hold to hand over.
 */
export interface TextureUpload {
  byteLength: number;
  count: number;
  indexOf: Map<string, number>;
  extraOffset: number;
  /** Where each texture's texels start, in words, by its index in the table. */
  offsets: number[];
  write(device: GPUDevice, buffer: GPUBuffer): void;
}

export function textureUpload(plan: TexturePlan, extraWords: Uint32Array): TextureUpload {
  const byteLength = Math.max(4, plan.extraOffset + extraWords.length) * 4;

  // A big endian page has to turn every byte texel around, which needs the whole buffer anyway.
  if (!LITTLE_ENDIAN) {
    const packed = writeTextures(plan, extraWords);

    return {
      byteLength,
      count: packed.count,
      indexOf: packed.indexOf,
      extraOffset: packed.extraOffset,
      offsets: plan.textures.map((_, index) => packed.data[textureEntry(index) + 2]),
      write: (device, buffer) => device.queue.writeBuffer(buffer, 0, packed.data),
    };
  }

  const header = new Uint32Array(plan.tableWords);
  const offsets = writeTable(header, plan);

  return {
    byteLength,
    count: plan.textures.length,
    indexOf: plan.indexOf,
    extraOffset: plan.extraOffset,
    offsets,
    write(device, buffer) {
      device.queue.writeBuffer(buffer, 0, header);

      plan.textures.forEach((texture, index) => {
        const view = texelView(texture, plan.channelOf.get(texture.id)) as ArrayBufferView;

        if (view.byteLength) device.queue.writeBuffer(buffer, offsets[index] * 4, view);
      });

      if (extraWords.length) device.queue.writeBuffer(buffer, plan.extraOffset * 4, extraWords);
    },
  };
}

export { MIRRORED_REPEAT, REPEAT };
