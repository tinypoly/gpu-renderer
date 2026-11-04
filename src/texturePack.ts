import type { SerializedScene, SerializedTexture } from "./upstream.js";

/**
 * Every texture of a scene in one storage buffer, led by a table saying where each one lives. Core WebGPU has no binding
 * arrays and `texture_2d_array` wants one size and one format for every layer, which a Three.js scene does not provide,
 * so the texels are read by hand. See the decision in ARCHITECTURE.md.
 *
 * Byte textures keep their eight bits per channel, four to a word, and are decoded from sRGB as they are read. Float
 * textures keep all four, because their range is the reason they are float at all.
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
 * Where everything will go, before anything is written: which index each texture gets and where the words after the
 * texels begin. Materials and lights point into that tail by absolute offset, so they need it before the buffer exists.
 */
export interface TexturePlan {
  textures: SerializedTexture[];
  indexOf: Map<string, number>;
  /** The lookup tables and the texture table together: where the texels begin. */
  tableWords: number;
  /** Where the words appended after the texels begin. */
  extraOffset: number;
}

export function planTextures(scene: SerializedScene, extraTextures: readonly SerializedTexture[] = []): TexturePlan {
  const seen = new Set<string>();

  // A texture id appears once, whoever refers to it; the first copy wins.
  const textures = [...scene.textures ?? [], ...extraTextures]
    .filter(texture => texelCount(texture) > 0 && texture.data.length >= texelCount(texture) * 4)
    .filter(texture => !seen.has(texture.id) && !!seen.add(texture.id));

  const indexOf = new Map(textures.map((texture, index) => [texture.id, index]));
  const tableWords = TEXTURE_TABLE + Math.max(1, textures.length) * TEXTURE_WORDS;
  const texelWords = textures.reduce((sum, texture) => sum + texelCount(texture) * (isFloat(texture) ? 4 : 1), 0);

  return { textures, indexOf, tableWords, extraOffset: tableWords + texelWords };
}

export function packTextures(scene: SerializedScene, extras: TextureExtras = {}): PackedTextures {
  return writeTextures(planTextures(scene, extras.textures), extras.words ?? new Uint32Array(0));
}

/** The buffer a plan describes, with `extraWords` appended at its `extraOffset`. */
export function writeTextures(plan: TexturePlan, extraWords: Uint32Array): PackedTextures {
  const { textures, indexOf, tableWords } = plan;

  // WebGPU rejects an empty binding, and a scene with no textures at all is ordinary.
  const data = new Uint32Array(Math.max(4, plan.extraOffset + extraWords.length));
  const words = data;
  const floats = new Float32Array(data.buffer);
  let texelAt = tableWords;

  data.set(LOOKUP, 0);

  textures.forEach((texture, index) => {
    const at = textureEntry(index);
    const texels = texelCount(texture);
    const float = isFloat(texture);

    words[at] = texture.width;
    words[at + 1] = texture.height;
    words[at + 2] = texelAt;
    words[at + 3] = (float ? TEXTURE_FLOAT : 0)
      | (isEncoded(texture) ? TEXTURE_SRGB : 0)
      | (texture.nearest ? TEXTURE_NEAREST : 0)
      | (texture.flipY ? TEXTURE_FLIP_Y : 0)
      | (texture.refraction ? TEXTURE_REFRACTION : 0);
    words[at + 4] = texture.wrapS;
    words[at + 5] = texture.wrapT;
    words[at + 6] = Math.min(Math.max(0, texture.channel | 0), UV_CHANNELS - 1);
    words[at + 7] = 0;

    // Three's Matrix3 arrives column major, so a row of the affine transform reads across the columns.
    const matrix = texture.matrix?.length === 9 ? texture.matrix : [1, 0, 0, 0, 1, 0, 0, 0, 1];

    floats.set([matrix[0], matrix[3], matrix[6], 0], at + 8);
    floats.set([matrix[1], matrix[4], matrix[7], 0], at + 12);

    if (float) {
      // The channels keep their bits and the shader reads them back as floats.
      floats.set((texture.data as Float32Array).subarray(0, texels * 4), texelAt);
      texelAt += texels * 4;
    } else {
      const source = texture.data as Uint8Array;

      if (LITTLE_ENDIAN) {
        // The bytes already sit in the order the words want them: copied whole, as the GPU reads them.
        new Uint8Array(data.buffer, texelAt * 4, texels * 4).set(source.subarray(0, texels * 4));
      } else {
        for (let texel = 0; texel < texels; texel++) {
          const offset = texel * 4;

          words[texelAt + texel] = source[offset]
            | (source[offset + 1] << 8)
            | (source[offset + 2] << 16)
            | (source[offset + 3] << 24);
        }
      }

      texelAt += texels;
    }
  });

  data.set(extraWords, texelAt);

  return { data, count: textures.length, indexOf, extraOffset: texelAt };
}

export { MIRRORED_REPEAT, REPEAT };
