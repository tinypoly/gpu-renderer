import { expect, it } from "vitest";
import { NO_TEXTURE, packTextures, TEXTURE_WORDS } from "./texturePack.js";
import type { SerializedScene, SerializedTexture } from "./upstream.js";

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

const FLOAT = 1, SRGB = 2, NEAREST = 4, FLIP_Y = 8;

it("writes the size, the flags and the wrap modes of each texture", () => {
  const packed = packTextures(scene([
    texture({ id: "a", width: 2, height: 3, data: new Uint8Array(24), wrapS: 1002, wrapT: 1000, nearest: true }),
  ]));

  expect(packed.count).toBe(1);
  expect(packed.indexOf.get("a")).toBe(0);
  expect([packed.data[0], packed.data[1]]).toEqual([2, 3]);
  expect(packed.data[3]).toBe(SRGB | NEAREST);
  expect([packed.data[4], packed.data[5]]).toEqual([1002, 1000]);
});

it("marks a byte texture as encoded only when its colour space says so", () => {
  const encoded = packTextures(scene([texture({ colorSpace: "srgb" })]));
  const linear = packTextures(scene([texture({ colorSpace: "srgb-linear" })]));

  expect(encoded.data[3] & SRGB).toBe(SRGB);
  expect(linear.data[3] & SRGB).toBe(0);
});

it("never marks a float texture as encoded, the snapshot having linearised it already", () => {
  const packed = packTextures(scene([
    texture({ data: new Float32Array([0.25, 0.5, 0.75, 1]), colorSpace: "srgb" }),
  ]));

  expect(packed.data[3] & FLOAT).toBe(FLOAT);
  expect(packed.data[3] & SRGB).toBe(0);
});

it("carries a texture's flip and its UV channel", () => {
  const packed = packTextures(scene([texture({ flipY: true, channel: 1 })]));

  expect(packed.data[3] & FLIP_Y).toBe(FLIP_Y);
  expect(packed.data[6]).toBe(1);
});

it("clamps a UV channel the geometry does not carry", () => {
  expect(packTextures(scene([texture({ channel: 7 })])).data[6]).toBe(1);
});

it("reads Three's column major matrix as the rows of an affine transform", () => {
  // Scale two across, three down, offset by a quarter and a half.
  const packed = packTextures(scene([texture({ matrix: [2, 0, 0, 0, 3, 0, 0.25, 0.5, 1] })]));
  const floats = new Float32Array(packed.data.buffer);

  expect([...floats.subarray(8, 11)]).toEqual([2, 0, 0.25]);
  expect([...floats.subarray(12, 15)]).toEqual([0, 3, 0.5]);
});

it("packs a byte texel into one word, red in the low byte", () => {
  const packed = packTextures(scene([texture({ data: new Uint8Array([1, 2, 3, 4]) })]));

  expect(packed.data[packed.data[2]]).toBe(1 | (2 << 8) | (3 << 16) | (4 << 24));
});

it("keeps a float texel's bits, four words to a texel", () => {
  const values = new Float32Array([0.25, 0.5, 0.75, 1]);
  const packed = packTextures(scene([texture({ data: values })]));
  const at = packed.data[2];
  const floats = new Float32Array(packed.data.buffer);

  expect([...floats.subarray(at, at + 4)]).toEqual([0.25, 0.5, 0.75, 1]);
});

it("puts the texels after the table, one entry per texture", () => {
  const packed = packTextures(scene([
    texture({ id: "a", width: 2, height: 2, data: new Uint8Array(16) }),
    texture({ id: "b", data: new Uint8Array(4) }),
  ]));

  expect(packed.data[2]).toBe(2 * TEXTURE_WORDS);
  expect(packed.data[TEXTURE_WORDS + 2]).toBe(2 * TEXTURE_WORDS + 4);
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

  expect(packed.extraOffset).toBe(TEXTURE_WORDS + 1);
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
