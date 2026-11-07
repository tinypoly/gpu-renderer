import { describe, expect, it } from "vitest";
import {
  NO_TEXTURE,
  packTextures,
  planTextures,
  SOBOL_TABLE,
  SRGB_TABLE,
  TEXTURE_TABLE,
  TEXTURE_WORDS,
  textureChannels,
  textureUpload,
  writeTextures,
} from "./texturePack.js";
import type { SerializedMaterial, SerializedMesh, SerializedScene, SerializedTexture } from "./upstream.js";

const texture = (overrides: Partial<SerializedTexture>): SerializedTexture => ({
  id: "t",
  width: 1,
  height: 1,
  data: new Uint8Array([0, 0, 0, 255]),
  colorSpace: "srgb",
  wrapS: 1000,
  wrapT: 1001,
  matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  flipY: false,
  channel: 0,
  nearest: false,
  ...overrides,
});

const scene = (textures: SerializedTexture[]): SerializedScene => ({ meshes: [], lights: [], textures });

const FLOAT = 1, SRGB = 2, NEAREST = 4, FLIP_Y = 8, SINGLE = 32;

/** A mesh whose one material uses these maps, and whatever else the material says. */
function meshWith(maps: Record<string, string>, rest: Partial<SerializedMaterial> = {}): SerializedMesh {
  return {
    attributes: { position: { data: new Float32Array(9), itemSize: 3 } },
    index: null,
    groups: [],
    materials: [{ type: "MeshStandardMaterial", name: "m", values: {}, maps, ...rest }],
    matrixWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    castShadow: true,
    receiveShadow: true,
    renderOrder: 0,
  };
}

/** readTexel in texture.wgsl.ts, for a byte texture: the four channels of one texel. */
function readTexel(words: Uint32Array, index: number, texel: number): number[] {
  const at = TEXTURE_TABLE + index * TEXTURE_WORDS;
  const offset = words[at + 2], flags = words[at + 3];
  const linear = (code: number) => new Float32Array(words.buffer)[SRGB_TABLE + code];

  if (flags & SINGLE) {
    const code = (words[offset + (texel >> 2)] >>> ((texel & 3) * 8)) & 0xff;
    const colour = flags & SRGB ? linear(code) : code / 255;

    return [colour, colour, colour, code / 255];
  }

  const word = words[offset + texel];
  const code = (shift: number) => (word >>> shift) & 0xff;

  return flags & SRGB
    ? [linear(code(0)), linear(code(8)), linear(code(16)), code(24) / 255]
    : [code(0) / 255, code(8) / 255, code(16) / 255, code(24) / 255];
}

const T = TEXTURE_TABLE;

it("writes the size, the flags and the wrap modes of each texture", () => {
  const packed = packTextures(scene([
    texture({ id: "a", width: 2, height: 3, data: new Uint8Array(24), wrapS: 1002, wrapT: 1000, nearest: true }),
  ]));

  expect(packed.count).toBe(1);
  expect(packed.indexOf.get("a")).toBe(0);
  expect([packed.data[T + 0], packed.data[T + 1]]).toEqual([2, 3]);
  expect(packed.data[T + 3]).toBe(SRGB | NEAREST);
  expect([packed.data[T + 4], packed.data[T + 5]]).toEqual([1002, 1000]);
});

it("marks a byte texture as encoded only when its colour space says so", () => {
  const encoded = packTextures(scene([texture({ colorSpace: "srgb" })]));
  const linear = packTextures(scene([texture({ colorSpace: "srgb-linear" })]));

  expect(encoded.data[T + 3] & SRGB).toBe(SRGB);
  expect(linear.data[T + 3] & SRGB).toBe(0);
});

it("never marks a float texture as encoded, the snapshot having linearised it already", () => {
  const packed = packTextures(scene([
    texture({ data: new Float32Array([0.25, 0.5, 0.75, 1]), colorSpace: "srgb" }),
  ]));

  expect(packed.data[T + 3] & FLOAT).toBe(FLOAT);
  expect(packed.data[T + 3] & SRGB).toBe(0);
});

it("carries a texture's flip and its UV channel", () => {
  const packed = packTextures(scene([texture({ flipY: true, channel: 1 })]));

  expect(packed.data[T + 3] & FLIP_Y).toBe(FLIP_Y);
  expect(packed.data[T + 6]).toBe(1);
});

it("clamps a UV channel the geometry does not carry", () => {
  expect(packTextures(scene([texture({ channel: 7 })])).data[T + 6]).toBe(1);
});

it("reads Three's column major matrix as the rows of an affine transform", () => {
  // Scale two across, three down, offset by a quarter and a half.
  const packed = packTextures(scene([texture({ matrix: [2, 0, 0, 0, 3, 0, 0.25, 0.5, 1] })]));
  const floats = new Float32Array(packed.data.buffer);

  expect([...floats.subarray(T + 8, T + 11)]).toEqual([2, 0, 0.25]);
  expect([...floats.subarray(T + 12, T + 15)]).toEqual([0, 3, 0.5]);
});

it("packs a byte texel into one word, red in the low byte", () => {
  const packed = packTextures(scene([texture({ data: new Uint8Array([1, 2, 3, 4]) })]));

  expect(packed.data[packed.data[T + 2]]).toBe(1 | (2 << 8) | (3 << 16) | (4 << 24));
});

it("keeps a float texel's bits, four words to a texel", () => {
  const values = new Float32Array([0.25, 0.5, 0.75, 1]);
  const packed = packTextures(scene([texture({ data: values })]));
  const at = packed.data[T + 2];
  const floats = new Float32Array(packed.data.buffer);

  expect([...floats.subarray(at, at + 4)]).toEqual([0.25, 0.5, 0.75, 1]);
});

it("puts the texels after the table, one entry per texture", () => {
  const packed = packTextures(scene([
    texture({ id: "a", width: 2, height: 2, data: new Uint8Array(16) }),
    texture({ id: "b", data: new Uint8Array(4) }),
  ]));

  expect(packed.data[T + 2]).toBe(T + 2 * TEXTURE_WORDS);
  expect(packed.data[T + TEXTURE_WORDS + 2]).toBe(T + 2 * TEXTURE_WORDS + 4);
});

it("skips a texture whose data is shorter than its own size claims", () => {
  const packed = packTextures(scene([texture({ width: 4, height: 4, data: new Uint8Array(8) })]));

  expect(packed.count).toBe(0);
  expect(packed.indexOf.size).toBe(0);
});

it("appends a caller's own words after the texels and says where they start", () => {
  const packed = packTextures(
    scene([texture({ data: new Uint8Array([9, 0, 0, 0]) })]),
    { words: new Uint32Array([111, 222]) });

  expect(packed.extraOffset).toBe(T + TEXTURE_WORDS + 1);
  expect([packed.data[packed.extraOffset], packed.data[packed.extraOffset + 1]]).toEqual([111, 222]);
});

it("lets a caller add textures of its own, such as the environment", () => {
  const packed = packTextures(
    scene([texture({ id: "a" })]),
    { textures: [texture({ id: "environment", data: new Float32Array([1, 1, 1, 1]) })] });

  expect(packed.count).toBe(2);
  expect(packed.indexOf.get("environment")).toBe(1);
});

it("still produces a bindable buffer for a scene with no textures", () => {
  const packed = packTextures(scene([]));

  expect(packed.count).toBe(0);
  expect(packed.data.length).toBeGreaterThan(0);
  // Nothing refers to entry zero, and a material with no map says so with its own index.
  expect(NO_TEXTURE).toBe(0xffffffff);
});

it("leads with the second Sobol dimension, byte by byte, as the bit by bit fold computes it", () => {
  const { data } = packTextures(scene([]));

  const folded = (index: number) => {
    let second = 0, direction = 0x80000000, bits = index >>> 0;

    for (; bits; bits >>>= 1, direction = (direction ^ (direction >>> 1)) >>> 0) if (bits & 1) second ^= direction;

    return second >>> 0;
  };

  const looked = (index: number) => (data[SOBOL_TABLE + (index & 255)] ^ data[SOBOL_TABLE + 256 + ((index >>> 8) & 255)]
    ^ data[SOBOL_TABLE + 512 + ((index >>> 16) & 255)] ^ data[SOBOL_TABLE + 768 + (index >>> 24)]) >>> 0;

  for (let i = 0; i < 4096; i++) expect(looked(i)).toBe(folded(i));
  for (const index of [0xffffffff, 0x80000001, 0x12345678, 0xdeadbeef]) expect(looked(index)).toBe(folded(index));
});

it("leads with the linear value of every sRGB code", () => {
  const floats = new Float32Array(packTextures(scene([])).data.buffer);

  expect(floats[SRGB_TABLE]).toBe(0);
  expect(floats[SRGB_TABLE + 255]).toBe(1);
  expect(floats[SRGB_TABLE + 128]).toBeCloseTo(0.21586, 5);
});

it("writes the same bytes to the device as the buffer it would have built, without building it", () => {
  const bytes = (width: number, height: number, seed: number) => texture({
    id: `b${seed}`,
    width,
    height,
    data: Uint8Array.from({ length: width * height * 4 }, (_, k) => (k * 31 + seed) & 255),
  });

  const floats = texture({
    id: "f",
    width: 3,
    height: 2,
    data: Float32Array.from({ length: 3 * 2 * 4 }, (_, k) => k / 7),
  });

  // The last one is read through its red channel alone, so it is kept one byte a texel.
  const scene: SerializedScene = {
    meshes: [meshWith({ map: "b1", bumpMap: "b9" })],
    lights: [],
    textures: [bytes(5, 3, 1), floats, bytes(5, 3, 9)],
  };

  const extra = Uint32Array.from([7, 8, 9, 10]);
  const plan = planTextures(scene);
  const whole = writeTextures(plan, extra);
  const upload = textureUpload(plan, extra);
  const written = new Uint8Array(upload.byteLength);

  const device = {
    queue: {
      writeBuffer(_buffer: unknown, offset: number, data: ArrayBufferView) {
        written.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      },
    },
  } as unknown as GPUDevice;

  upload.write(device, {} as GPUBuffer);

  expect(upload.byteLength).toBe(whole.data.byteLength);
  expect([...new Uint32Array(written.buffer)]).toEqual([...whole.data]);
  expect(upload.offsets)
    .toEqual(plan.textures.map((_, index) => whole.data[TEXTURE_TABLE + index * TEXTURE_WORDS + 2]));
});

describe("textures read through one channel", () => {
  const bytes = (id: string, colorSpace = "") => texture({
    id,
    width: 7,
    height: 3,
    colorSpace,
    data: Uint8Array.from({ length: 7 * 3 * 4 }, (_, k) => (k * 37 + id.length * 11) & 255),
  });

  it("keeps the only channel read, one byte a texel, and reads back what the full texel said in it", () => {
    for (const [slot, channel, colorSpace] of [["roughnessMap", 1, ""], ["bumpMap", 0, ""], ["alphaMap", 1, "srgb"]] as const) {
      const source = bytes("t", colorSpace);
      const single = packTextures({ meshes: [meshWith({ [slot]: "t" })], lights: [], textures: [source] });

      const whole = writeTextures(planTextures({ meshes: [], lights: [], textures: [source] }, [], new Map()),
        new Uint32Array(0));

      expect(single.data[TEXTURE_TABLE + 3] & SINGLE).toBe(SINGLE);
      expect(single.data.length).toBe(whole.data.length - 21 + Math.ceil(21 / 4));

      for (let texel = 0; texel < 21; texel++)
        expect(readTexel(single.data, 0, texel)[channel]).toBeCloseTo(readTexel(whole.data, 0, texel)[channel], 6);
    }
  });

  it("keeps all four where a slot reads another channel, or something besides a map names the texture", () => {
    const textures = [bytes("orm"), bytes("lit"), bytes("uniform"), bytes("rough")];

    const plan = textureChannels({
      meshes: [
        meshWith({ roughnessMap: "orm", metalnessMap: "orm", bumpMap: "lit" }),
        meshWith({ roughnessMap: "uniform" }, { shader: { uniforms: { tMap: { texture: "uniform" } } } as never }),
        meshWith({ roughnessMap: "rough" }),
      ],
      lights: [{ map: "lit" } as never],
      textures,
    });

    expect([...plan.single]).toEqual([["rough", 1]]);
    expect(plan.whole.get("orm")).toBe("reads gb");
    expect(plan.whole.get("lit")).toBe("named by lights.0.map");
    expect(plan.whole.get("uniform")).toBe('named by material "m".shader.uniforms.tMap.texture');
  });

  it("counts a shader's uniforms only for the textures its translated program reads", () => {
    const hooked = meshWith({ bumpMap: "bump", roughnessMap: "rough" }, {
      shader: { uniforms: { bumpMap: { texture: "bump" }, roughnessMap: { texture: "rough" } } } as never,
    });

    // A program that declares only the roughness map among the uniforms it is handed.
    const program = {
      packUniforms: (values: Record<string, { texture?: string }>, textures: ReadonlyMap<string, number>) =>
        Uint32Array.from([textures.get(values.roughnessMap?.texture ?? "") ?? NO_TEXTURE, 7]),
    };

    const programs = {
      fragment: (material: SerializedMaterial) => material === hooked.materials[0] ? program as never : null,
      after: () => null,
    };

    const plan = textureChannels({ meshes: [hooked], lights: [], textures: [bytes("bump"), bytes("rough")] }, programs);

    expect([...plan.single]).toEqual([["bump", 0]]);
    expect(plan.whole.get("rough")).toBe('named by the shader of material "m"');
  });
});
