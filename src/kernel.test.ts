import { parseWgsl, validate, type NagaError } from "naga-wasm";
import { expect, it } from "vitest";
import { pathTraceSource } from "./pathTracePass.js";
import { compileShaderPrograms } from "./shaderProgram.js";
import { DENOISE_WGSL } from "./denoise.js";
import { PRESENT_WGSL } from "./wgsl/present.wgsl.js";
import { IMAGE_RESOLVE_WGSL } from "./imageResolve.js";
import { LENS_WGSL } from "./wgsl/lens.wgsl.js";
import type { SerializedMaterial, SerializedScene } from "./upstream.js";

/**
 * The kernel is validated here by naga, which is close enough to what a browser's compiler accepts that most mistakes
 * in the WGSL fail in CI instead of on a device, where a shader that does not compile only prints a line number.
 */
function check(source: string) {
  try {
    validate(parseWgsl(source));
  } catch(error) {
    throw new Error((error as NagaError).formatted ?? String(error), { cause: error });
  }
}

it("validates the path tracing kernel without any shader", () => {
  expect(() => check(pathTraceSource(null))).not.toThrow();
});

it("validates the kernel without the subsurface walk or volumes, as a plain scene compiles it", () => {
  expect(() => check(pathTraceSource(null, false, false, false, false, false))).not.toThrow();
  expect(() => check(pathTraceSource(null, false, false, true, false, false))).not.toThrow();
  expect(() => check(pathTraceSource(null, true, false, false, false, false))).not.toThrow();
  expect(() => check(pathTraceSource(null, false, true, false, false, false))).not.toThrow();
});

it("validates the kernel with grass and nothing else", () => {
  expect(() => check(pathTraceSource(null, false, false, false, false, true))).not.toThrow();
});

it("validates the presentation pass", () => {
  expect(() => check(PRESENT_WGSL)).not.toThrow();
});

it("validates the pass that resolves the finished image to bytes", () => {
  expect(() => check(IMAGE_RESOLVE_WGSL)).not.toThrow();
});

it("validates the denoiser", () => {
  expect(() => check(DENOISE_WGSL)).not.toThrow();
});

it("validates the lens pass", () => {
  expect(() => check(LENS_WGSL)).not.toThrow();
});

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function shaderScene(materials: SerializedMaterial[]): SerializedScene {
  return {
    lights: [],
    textures: [],
    meshes: materials.map(material => ({
      attributes: { position: { data: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), itemSize: 3 } },
      index: null,
      groups: [],
      materials: [material],
      matrixWorld: IDENTITY,
      castShadow: true,
      receiveShadow: true,
      renderOrder: 0,
    })),
  };
}

const shader = (fragment: string, extra: Partial<SerializedMaterial> = {}): SerializedMaterial => ({
  type: "ShaderMaterial",
  name: "test",
  values: {},
  maps: {},
  shader: {
    vertex: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragment,
    uniforms: {},
    defines: {},
    patched: false,
  },
  ...extra,
});

it("validates a kernel with translated shaders of every kind", async() => {
  const programs = await compileShaderPrograms(shaderScene([
    shader(`
      uniform sampler2D map;
      uniform samplerCube envMap;
      uniform vec3 colors[2];
      uniform float time;
      uniform bool flip;
      struct Wave { float height; vec2 direction; };
      uniform Wave waves[2];
      varying vec2 vUv;
      #include <common>
      float grid(vec2 p) { vec2 g = abs(fract(p - 0.5) - 0.5) / fwidth(p); return 1.0 - min(min(g.x, g.y), 1.0); }
      void main() {
        if (vUv.x < 0.0) discard;
        vec4 texel = texture2D(map, vUv) + textureCube(envMap, vec3(vUv, 1.0));
        float h = waves[1].height * dot(waves[0].direction, vUv);
        gl_FragColor = vec4(colors[1] * grid(vUv * 10.0) + texel.rgb * sin(time) + h, flip ? 0.5 : 1.0);
        gl_FragColor.rgb += vec3(gl_FragCoord.x, float(gl_FrontFacing), dFdx(vUv.x));
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`),
    {
      ...shader(`#version 300 es
        precision highp float;
        in vec2 vUv;
        flat in int vId;
        out vec4 fragColor;
        uniform sampler2D tDiffuse;
        void main() { fragColor = texture(tDiffuse, vUv) * float(vId) + textureLod(tDiffuse, vUv, 0.0); }`),
      type: "RawShaderMaterial",
    },
    {
      type: "MeshStandardMaterial",
      name: "patched",
      values: {},
      maps: {},
      shader: {
        vertex: "void main() {}",
        fragment: "uniform float uAmount;\nvoid main() {\ndiffuseColor.rgb *= uAmount;\nroughnessFactor = 0.5;\n"
          + "normal = normalize(normal + vec3(0.1));\n}",
        uniforms: { uAmount: 0.5 },
        defines: {},
        patched: true,
      },
    },
  ]), "vec3 skyForeground(vec3 direction) { return textureCube(uSkyMap, direction).rgb + vec3(direction.y); }");

  expect(programs.failures).toEqual([]);
  expect(programs.programs.length).toBe(4);
  expect(programs.foreground).not.toBeNull();
  expect(() => check(pathTraceSource(programs))).not.toThrow();
});

it("rewrites a switch whose cases fall through, which WGSL cannot say", async() => {
  const programs = await compileShaderPrograms(shaderScene([shader(`
    uniform int mode;
    void main() {
      float value = 0.0;
      switch (mode) {
        case 0: value += 0.125;
        case 1: { value += 0.125; break; }
        default: value = 1.0;
      }
      gl_FragColor = vec4(vec3(value), 1.0);
    }`)]));

  expect(programs.failures).toEqual([]);
  expect(() => check(pathTraceSource(programs))).not.toThrow();
});

it("translates what a hook does after <opaque_fragment> into a second program", async() => {
  const patched: SerializedMaterial = {
    type: "MeshLambertMaterial",
    name: "after",
    values: {},
    maps: {},
    shader: {
      vertex: "void main() {}",
      fragment: "void main() {\n}",
      fragmentAfter: "uniform float uDim;\nvoid main() {\ngl_FragColor.rgb *= uDim;\n}",
      uniforms: { uDim: 0.5 },
      defines: {},
      patched: true,
    },
  };

  const programs = await compileShaderPrograms(shaderScene([patched]));

  expect(programs.failures).toEqual([]);
  expect(programs.after(patched)?.kind).toBe("after");
  expect(() => check(pathTraceSource(programs))).not.toThrow();
});

it("compiles a hook that wraps Three's physical direct light and samples the material's maps", async() => {
  const patched: SerializedMaterial = {
    type: "MeshStandardMaterial",
    name: "relief",
    values: {},
    maps: {},
    shader: {
      vertex: "void main() {}",
      fragment: `uniform sampler2D map;
varying vec2 vMapUv;
void reliefDirect(const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal,
  const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material,
  inout ReflectedLight reflectedLight) {
  RE_Direct_Physical(directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material,
    reflectedLight);
}
#undef RE_Direct
#define RE_Direct reliefDirect
void main() {
#ifdef USE_MAP
diffuseColor.rgb = texture2D(map, vMapUv * 2.0).rgb;
#endif
}`,
      uniforms: { map: { texture: "map" } },
      defines: { USE_MAP: "" },
      patched: true,
    },
  };

  const programs = await compileShaderPrograms(shaderScene([patched]));

  expect(programs.failures).toEqual([]);
  expect(programs.fragment(patched)?.varyings.map(varying => varying.name)).toEqual(["vMapUv"]);
  expect(() => check(pathTraceSource(programs))).not.toThrow();
});
