# @tinypoly/gpu-renderer

[![npm version](https://img.shields.io/npm/v/@tinypoly/gpu-renderer)](https://www.npmjs.com/package/@tinypoly/gpu-renderer)
[![npm downloads](https://img.shields.io/npm/dm/@tinypoly/gpu-renderer)](https://www.npmjs.com/package/@tinypoly/gpu-renderer)
[![CI](https://github.com/tinypoly/gpu-renderer/actions/workflows/ci.yml/badge.svg)](https://github.com/tinypoly/gpu-renderer/actions/workflows/ci.yml)
[![types](https://img.shields.io/npm/types/@tinypoly/gpu-renderer)](https://www.npmjs.com/package/@tinypoly/gpu-renderer)
[![license](https://img.shields.io/npm/l/@tinypoly/gpu-renderer)](LICENSE)

A WebGPU path tracer for [Three.js](https://threejs.org) scenes. It takes an ordinary `THREE.Scene` and traces it in WGSL compute shaders, accumulating paths until the image converges, and refines a canvas in place while it does.

It is the GPU sibling of [@tinypoly/cpu-renderer](https://github.com/tinypoly/cpu-renderer), which powers the render mode of the [Tinypoly](https://tinypoly.com) editor: the same API, the same scene snapshot and the same settings, so a host picks one at startup and writes the rest once.

**[Live example](https://tinypoly.github.io/gpu-renderer/)**

- Unidirectional path tracing with multiple importance sampling: every light is aimed at with a shadow ray and weighed against the bounce that might find it, so a small bright lamp or a sun in an HDR converges in tens of samples.
- Physically based shading as Three r185 lights it: Standard and Physical materials (transmission, dispersion, clearcoat, sheen, iridescence, anisotropy), Lambert, Phong, Toon, Basic, Matcap, Normal, Depth and `ShadowMaterial`, with all of their maps.
- Real glass: thin walls that light crosses, and closed volumes that refract and absorb.
- Perspective and orthographic cameras, any projection Three builds; meshes, lines (`Line`, `LineSegments`, `LineLoop`), dashed lines, fat lines (`Line2`), sprites and `PointsMaterial`.
- Your custom `ShaderMaterial` GLSL, `RawShaderMaterial` and `onBeforeCompile` hooks: fragment shaders are translated to WGSL by [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) and run in the kernel, with real derivatives; vertex shaders run on the CPU renderer's interpreter.
- HDR, gradient or procedural sky environments, light probes, depth of field with polygonal bokeh, fog, volumetric lights, vignette and grain, and ACES, AgX or Neutral tone mapping.
- Adaptive sampling: pixels whose noise is gone stop taking paths, and the work they leave goes to the dark, glossy and out of focus areas that still need it.
- A variance-guided denoiser (the spatial filter of SVGF, with diffuse and specular light filtered apart as NVIDIA's RELAX does): each signal is separated from its albedo, filtered by an edge-avoiding wavelet that stops at normals, depths and roughness and trusts each pixel as far as its own variance says, and multiplied back, so textures stay sharp, a sharp reflection stays sharp and flat areas keep their exact mean.
- Progressive output: the canvas refines sample by sample, and the render can be paused, resumed and cancelled.
- No worker pool and no `SharedArrayBuffer`: the scene lives on the device, so the page needs no cross-origin isolation. A single worker builds the BVH of a large scene, so the page keeps responding while it is prepared.

## Install

```sh
npm install @tinypoly/gpu-renderer three
```

`three` is a peer dependency (0.185 or newer). `@tinypoly/cpu-renderer` comes along: the scene snapshot, the settings and the tone mapping are shared with it rather than written twice. The browser needs WebGPU.

## Usage

```ts
import * as THREE from "three";
import { GpuRenderer, attachCanvas, requestWebGpuSupport } from "@tinypoly/gpu-renderer";

const { supported, reason } = await requestWebGpuSupport();
if (!supported) throw new Error(reason); // the host decides what to fall back to: the CPU renderer, for one

const scene = new THREE.Scene();
const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.MeshPhysicalMaterial({ color: "tomato" }));
scene.add(sphere);

const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
camera.position.set(0, 1, 5);
camera.lookAt(0, 0, 0);

const canvas = document.querySelector("canvas")!;

const renderer = new GpuRenderer({
  width: canvas.clientWidth,
  height: canvas.clientHeight,
  pixelRatio: window.devicePixelRatio,
  scene,
  camera,
  settings: { samples: 256 }, // paths per pixel
  environment: { kind: "gradient", topColor: "#9cb8d8", bottomColor: "#2b2a28" },
});

attachCanvas(renderer, canvas); // sizes the canvas and refines it in place while the image converges
renderer.on("sample", ({ samples, target }) => console.log(`${samples} / ${target}`));
const { width, height, data } = await renderer.render(); // RGBA bytes of the finished image
```

`width` and `height` are pixels; the render size multiplies them by `pixelRatio` and `settings.renderScale`. Every mesh takes part in shadows: a traced shadow comes from the geometry that casts it, so `castShadow` and `receiveShadow` on a mesh are not read.

Without a canvas, take the image when it is done:

```ts
renderer.on("start", ({ width, height }) => { /* the render size */ });
renderer.on("pixels", ({ x, y, width, height, data }) => { /* the whole image, once, when the last sample lands */ });
```

Setters record the inputs; `render()` starts a frame with the current ones:

```ts
camera.position.x += 1;                  // the camera is read at every render()
renderer.setSettings({ samples: 1024, globalIllumination: { bounces: 8 } });
const image = await renderer.render();
```

## API

### `new GpuRenderer(options)`

Every option has a setter of the same name, except `bvhWorker` and `createBvhWorker`, which are fixed for the renderer's life.

| Option | Type | |
| --- | --- | --- |
| `scene` | `THREE.Scene` or the result of `serializeScene` | A Three scene is serialized in the background. Defaults to an empty scene. |
| `camera` | `THREE.Camera` or `CameraState` | A Three camera is read at every `render()`. Any projection works: perspective, orthographic, zoom and view offsets. Defaults to a perspective camera at (5, 5, 5) looking at the origin. |
| `settings` | `RenderSettingsInput` | Merged over `DEFAULT_RENDER_SETTINGS`, group by group. See [Settings](#settings). |
| `environment` | `RenderEnvironment` | See [Environment](#environment). Defaults to `DEFAULT_RENDER_ENVIRONMENT`, a gradient. |
| `width`, `height` | `number` | Required. Image size in CSS pixels. |
| `pixelRatio` | `number` | Defaults to 1. Pass `window.devicePixelRatio` for a sharp image on a high-density screen. |
| `denoiser` | `Partial<DenoiserSettings>` | How the denoiser filters. Merged over `DEFAULT_DENOISER`. See [Denoiser](#denoiser). |
| `bvhWorker` | `boolean` | Builds the BVH of a scene with at least `BVH_WORKER_MIN_TRIANGLES` (50 000) triangles in a Web Worker, so the page keeps responding. The tree, and so the image, is the same either way. Defaults to `true` where Web Workers exist or when `createBvhWorker` is given; `false` builds on the main thread. See [BVH worker](#bvh-worker). |
| `createBvhWorker` | `() => BvhWorker` | Starts the BVH worker. Defaults to `createDefaultBvhWorker`. Passing one turns the worker on. |

### Methods

| Method | |
| --- | --- |
| `render()` | Starts a frame with the current inputs and discards the one in progress. Resolves with the finished image, `{ width, height, data }`: `data` is RGBA bytes, straight sRGB, `width * height * 4` long. Rejects with an `AbortError` (see `isAbortError`) when a later `render()`, `cancel()` or `dispose()` replaces the frame, and with the error when the frame fails, a lost device included. |
| `setScene(scene)` | A Three scene is serialized in the background; `render()` waits for it. |
| `setCamera(camera)` | A Three camera is read as it is at every `render()`, so moving it or changing its aspect only needs a new frame. A `CameraState` from `serializeCamera` stays fixed. |
| `setSettings(settings)` | Merges the given fields into the current settings, group by group: `{ fog: { enabled: true } }` keeps the other fog fields. |
| `setEnvironment(environment)` | Replaces the environment. A change of intensity or rotation alone does not load the image again. |
| `setSize(width, height, pixelRatio?)` | Image size in CSS pixels for the next frame; the pixel ratio stays unless given. |
| `setPixelRatio(pixelRatio)` | |
| `setDenoiser(denoiser)` | Merges the given fields into the denoiser's settings. A finished image is filtered again at once, without tracing anything: `image` and the canvas take the result and `pixels` fires. |
| `pause()`, `resume()` | Stops or continues the frame between two dispatches. |
| `cancel()` | Stops the frame in flight and keeps the device for the next one. `render()` rejects with an `AbortError` and `image` keeps what it held. A paused frame is cancelled too, and the next one starts unpaused. |
| `dispose()` | Releases the device. |
| `settings` | The settings the next frame renders with. |
| `denoiser` | The denoiser's settings. |
| `rendering`, `paused` | Whether a frame is in flight and whether it is paused. |
| `image` | The last finished image, in the shape `render()` resolves with, or an empty one of the render size while a frame traces. `null` before the first frame starts. Each frame allocates a new buffer, so the image one `render()` resolved with stays as it is. |
| `renderWidth`, `renderHeight` | The size the next frame renders at: CSS pixels times `pixelRatio` and `settings.renderScale`. |

### Events

`renderer.on(event, listener)` returns the function that removes the listener; `once` and `off` exist too. The payloads are the CPU renderer's; what differs is when they fire, because no pixel is finished before the last sample.

| Event | Payload | |
| --- | --- | --- |
| `start` | `{ width, height }` | The scene is on the device and tracing starts. `image` has the render size and is empty. |
| `sample` | `{ samples, target }` | One more path per pixel over the whole image. Only this renderer has it. |
| `pixels` | `{ x, y, width, height, data }` | The finished image, **once**, as a single region at (0, 0). `image` already holds it. |
| `progress` | `{ completed, total, phase }` | Rows traced out of `total`, over every sample. `phase` is `"prepare"` while the scene uploads and `"shade"` while it traces. |
| `bucket` | `{ worker, bucket, width, height }` | The band of rows on the device, always as worker 0, or `null` when it is idle. The renderer sizes its dispatches to keep the page responsive. |
| `complete` | | The image is finished. `render()` resolves with it right after. |
| `error` | `Error` | The frame failed. `render()` rejects with the same error. |

### `attachCanvas(renderer, canvas)`

Draws the output on an `HTMLCanvasElement` through a WebGPU context: the canvas takes the render size and is refined in place after every dispatch. A pixel no sample has reached yet stays see-through, so a page can show a preview under the canvas until the image covers it. The canvas must not already hold another context. Returns the function that detaches it.

A WebGPU canvas does not keep what it presented: to save the image, use the pixels `render()` resolves with, or `image`.

### WebGPU support

| Function | |
| --- | --- |
| `isWebGpuAvailable()` | The cheap synchronous check: whether `navigator.gpu` exists. It does not prove an adapter does. |
| `requestWebGpuSupport()` | Asks for an adapter, which is the only honest answer, because a browser can expose `navigator.gpu` and still hand back nothing. Resolves with `{ supported, reason }`. |

The renderer never falls back on its own. Choosing another renderer when WebGPU is missing is the host's decision, and the CPU renderer takes the same calls.

### Settings

`RenderSettings`, grouped by feature, the CPU renderer's own. Defaults are `DEFAULT_RENDER_SETTINGS`; `mergeSettings(base, input)` applies a partial `RenderSettingsInput` the way `setSettings` does. A path tracer reads some of them differently:

| Field | Default | |
| --- | --- | --- |
| `renderScale` | `1` | Multiplier on the image size. |
| `samples` | `4` | **Paths per pixel**, on average: one number does the antialiasing and the global illumination in a path tracer, and a few hundred is a clean image. Above 32, a pixel that has converged stops early and the noisiest ones take up to four times as many. |
| `toneMapping` | `"aces"` | `"aces"`, `"agx"`, `"neutral"` or `"linear"`. |
| `exposure` | `1` | |
| `background.mode` | `"environment"` | `"environment"`, `"color"` or `"transparent"`, which makes the image's alpha the coverage of every pixel. |
| `background.color` | `"#1a1a1a"` | Used by `mode: "color"`. |
| `shadows.enabled` | `true` | Whether Three's lights are occlusion tested. Emissive geometry always is. |
| `shadows.softness` | `1` | Angular diameter of the lights in degrees. `0` gives hard shadows. |
| `globalIllumination.bounces` | `3` | How far a path goes. Glass wants four or more. |
| `globalIllumination.clamp` | `0` | Maximum luminance one indirect contribution may add, which trades a little bias for fewer fireflies. `0` disables clamping. |
| `depthOfField.enabled` | `false` | |
| `depthOfField.focusDistance` | `10` | World units. |
| `depthOfField.aperture` | `2.8` | f-number. |
| `depthOfField.blades` | `0` | Polygon sides of the bokeh. `0` is circular. |
| `fog.enabled` | `false` | Linear fog over every surface. Without it, `scene.fog` applies. |
| `fog.color` | `"#b8c4d0"` | |
| `fog.near`, `fog.far` | `10`, `60` | View depth. |
| `vignette` | `0` | `0` to `1`. |
| `grain` | `0` | `0` to `1`. |

`globalIllumination.denoise` (default `true`) turns the denoiser on: on the canvas at the end of every pass, and on the image `render()` resolves with. Switching it alone filters the finished image again without tracing.

### Denoiser

A variance-guided wavelet filter, the spatial half of SVGF, with the diffuse and the specular light filtered apart, as NVIDIA's RELAX and REBLUR do. Every path is split at its first scatter by the lobe it took; each signal is divided by its own albedo (the diffuse colour, the GGX lobe's directional albedo), filtered by passes that stop at normals and depths, and multiplied back, so textures stay sharp. The specular filter also stops at roughness, and reaches only as far as the footprint the lobe leaves on the screen, from the roughness and the distance to what it reflects: a mirror is left alone, a rough reflection of something far is smoothed widely. Each pixel is trusted as far as its own variance says. `DenoiserSettings`, defaults in `DEFAULT_DENOISER`:

| Field | Default | |
| --- | --- | --- |
| `strength` | `1` | How much of the filtered image replaces the traced one, from `0` to `1`. |
| `passes` | `3` | Wavelet passes, `1` to `5`: how far the filter reaches, 2, 6, 14, 30 or 62 pixels. |
| `edge` | `2` | How large a luminance difference still counts as noise, in standard deviations of the pixel's own noise. Lower keeps more detail and removes less noise. |
| `onlyNoisy` | `true` | Leaves the pixels that have converged as they were traced, and filters only where noise is still visible. |
| `antiFirefly` | `true` | Before filtering, clamps a pixel brighter than every one of its eight neighbours to the brightest of them, in each signal. |

A denoiser trades noise for detail. What it smooths most is what its guides cannot see: the relief of a lacquer, an out of focus area. With `onlyNoisy` it leaves alone whatever has already converged, so more samples always mean less filtering.

```ts
renderer.setDenoiser({ strength: 0.6, edge: 1 }); // gentler: filters the finished image again, no tracing
```

Global illumination is always on, because tracing paths is what this renderer does: `globalIllumination.enabled`, `samples`, `intensity`, `cache` and `cacheSpacing`, `ambientOcclusion` and `tileSize` are the CPU renderer's and are not read.

### Environment

`RenderEnvironment`. Lights the scene and paints the background when `background.mode` is `"environment"`. An HDR image gets a luminance distribution, so a shadow ray can aim at its sun.

```ts
{ kind: "hdri", url: "studio.hdr" }                                            // equirectangular .hdr
{ kind: "gradient", topColor: "#9cb8d8", bottomColor: "#2b2a28", exponent: 2 } // exponent defaults to 2
{ kind: "hdri", url: "studio.hdr", intensity: 0.8, rotation: 90 }              // intensity 1 and rotation 0 by default
```

### Object options

The renderer reads the CPU renderer's optional settings from `userData.cpuRenderer` on Three objects; the key keeps that name, so a scene renders the same in both. A plain scene needs none of them.

| Where | Field | |
| --- | --- | --- |
| Any object | `exclude` | Leaves the object and its children out of the render. |
| Light | `volumetric` | Scatters the light in the air: `true`, or a medium `{ density, anisotropy, spread }`. |
| Scene | `sky` | A `ProceduralSky`: a shader that replaces the environment. |
| Material | `time` | Value of the shader's `time` uniform. |
| Material | `planarCapture` | Renders planar reflection and refraction textures for a flat surface, path traced from the mirrored camera. The surface's local +Y is its normal. |
| Material | `compute` | A `MaterialCompute`: fragment passes baked on the CPU into the material's uniforms. |
| Material | `uniforms` | The live uniforms of a material patched in `onBeforeCompile`. |
| Material | `prepare` | Called with the scene before it is serialized. |

```ts
light.userData.cpuRenderer = { volumetric: { density: 0.05 } };
```

### Helpers

| Function | |
| --- | --- |
| `serializeScene(scene, options?)` | Async. The snapshot `setScene` builds from a Three scene, to reuse across renderers, the CPU one included. A snapshot only for this renderer can pass `{ mipmaps: false }`: it samples the base level of each texture, so the mip chains would be built for nothing. |
| `serializeCamera(camera)` | A fixed `CameraState` snapshot of a Three camera, including zoom and view offsets. |
| `sceneSignature(scene)` | A string that changes whenever something the renderer reads changes. Compare it before re-serializing an interactive scene. |
| `isAbortError(error)` | Whether a `render()` rejection means the frame was replaced or cancelled rather than failed. |
| `createDefaultBvhWorker()` | The BVH worker this package ships, started from `new URL("./bvhWorker.js", import.meta.url)`. |
| `respondToBvhRequest(request)` | The worker's side of the exchange, for a worker of your own: returns the response and the buffers to transfer. |
| `packGeometry`, `buildBvh`, `packLights`, `packTextures`, `sceneByteLength` | The scene packing, published because the sizes it produces are what a host checks against a device's limits before uploading. |

## Bundlers

The renderer loads naga's WebAssembly the first time a scene has a shader, from `new URL("naga_bg.wasm", import.meta.url)` next to the `naga-wasm` module, which Vite, webpack 5, Rspack and Parcel resolve. With **Vite**, keep it out of dependency pre-bundling, or the dev server moves the module away from its binary:

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ["naga-wasm"] },
});
```

The BVH worker is started the same way, from `new URL("./bvhWorker.js", import.meta.url)` next to the package's own module, which the same bundlers emit as a separate chunk. Where yours does not, pass `createBvhWorker`, or `bvhWorker: false`; see [BVH worker](#bvh-worker).

### BVH worker

Splitting a large scene into its BVH takes about a second per million triangles, which would hold the page if it ran there. The renderer sends a copy of the positions to a Web Worker instead, starts it on the first scene large enough to need it, and keeps it until `dispose()`. A smaller scene is built on the main thread, where it takes less time than the trip to the worker and back.

If the worker cannot start or fails to load, the renderer warns once in the console and builds every BVH on the main thread from then on: the image does not change, only the page holds while the scene is prepared.

```ts
// Off: every BVH on the main thread.
new GpuRenderer({ width, height, scene, bvhWorker: false });

// A worker of your own, when the bundler cannot follow the default one's URL.
new GpuRenderer({
  width, height, scene,
  createBvhWorker: () => new Worker(new URL("@tinypoly/gpu-renderer/bvhWorker.js", import.meta.url),
    { type: "module" }),
});
```

Anything shaped like a Web Worker fits `BvhWorker`: `postMessage(request, transfer)`, `terminate()`, `onmessage` and `onerror`. A worker of your own answers each `BvhRequest` with `respondToBvhRequest(request)`, posting its `response` with its `transfer` list.

## Limits

This is a path tracer, not Three.js or WebGL conformance, and not a match for the CPU renderer's image: shadows, indirect light, reflections and transmission come out differently, usually better, never identical. Where it deliberately disagrees with Three (ambient lights as surroundings, stochastic transparency, unlit surfaces that emit, refracting glass) and what it reads, [ARCHITECTURE.md](ARCHITECTURE.md) lists in full. A scene has to fit the device's largest storage binding, and it is not available in Node, which has no WebGPU.

## Development

`pnpm dev` opens the example gallery, the same scenes as the CPU renderer's, traced here. `pnpm test` runs the CI gate, which needs no GPU; `pnpm verify` renders scenes with answers known on paper on a real GPU in headless Chrome; `pnpm capture:scenes` renders the gallery pictures with the example running. `pnpm verify --page divergence.html` measures what the megakernel loses to divergence: the same grid of spheres with one material at a time and with all of them mixed.

## License

[MIT](LICENSE)
