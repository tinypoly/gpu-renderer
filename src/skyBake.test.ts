import { parseWgsl, validate, type NagaError } from "naga-wasm";
import { expect, it } from "vitest";
import { prepareSceneAssets, skyKey } from "./sceneAssets.js";
import { compileSkyProgram, skyBakeSource } from "./skyBake.js";
import type { SerializedScene } from "./upstream.js";

function check(source: string) {
  try {
    validate(parseWgsl(source));
  } catch(error) {
    throw new Error((error as NagaError).formatted ?? String(error), { cause: error });
  }
}

// A sky as the CPU renderer's snapshot carries it: the direction comes in as a varying, uniforms beside it.
const SKY = `varying vec3 vDirection;
uniform vec3 uZenith;
uniform float uSharpness;
void main() {
  float up = clamp(normalize(vDirection).y, 0.0, 1.0);
  vec3 color = vec3(0.0);
  for (int i = 0; i < 4; i++) color += uZenith * pow(up, uSharpness + float(i)) * 0.25;
  gl_FragColor = vec4(color, 1.0);
}`;

function skyScene(zenith = [0.2, 0.4, 0.8]): SerializedScene {
  return {
    meshes: [],
    lights: [],
    textures: [],
    sky: { fragmentShader: SKY, uniforms: { uZenith: zenith, uSharpness: 1 }, intensity: 1, background: true },
  };
}

it("translates a procedural sky into a bake kernel naga accepts", async() => {
  const program = await compileSkyProgram(SKY);

  expect(program.uniformWords).toBe(4);
  expect(() => check(skyBakeSource(program, 256, 128))).not.toThrow();
});

it("bakes the sky on the CPU when there is no device, as before", async() => {
  const assets = await prepareSceneAssets(skyScene());

  expect(assets.sky?.environment).toBeTruthy();
  expect(assets.skyBake?.key).toBe(skyKey(skyScene().sky));
});

it("reuses a baked sky when the next scene has the same one, and bakes a different one again", async() => {
  const first = await prepareSceneAssets(skyScene());
  const same = { ...skyScene(), sky: { ...skyScene().sky!, intensity: 3 } };
  const reused = await prepareSceneAssets(same, { sky: first.skyBake });

  expect(reused.skyBake?.baked).toBe(first.skyBake?.baked);
  // The panorama is shared; how strongly it lights stays the scene's own.
  expect(reused.sky?.intensity).toBe(3);

  const other = await prepareSceneAssets(skyScene([1, 0, 0]), { sky: first.skyBake });

  expect(other.skyBake?.baked).not.toBe(first.skyBake?.baked);
});
