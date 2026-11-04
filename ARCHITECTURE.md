# Architecture

How `@tinypoly/gpu-renderer` works inside, and the decisions behind it. For installation and usage, see the
[README](README.md).

The renderer takes a serialized Three.js scene, uploads geometry, a BVH, materials, textures and lights to the GPU, and
runs a path tracing compute shader that accumulates radiance into a float buffer. A small render pipeline presents that
buffer, dividing by the sample count and applying exposure, tone mapping, vignette and grain. There is no worker pool and
no `SharedArrayBuffer`: the state lives on the device.

## Relationship with @tinypoly/cpu-renderer

The two renderers share a contract instead of a code base. Everything that reads Three.js or defines the shape of the
data comes from the CPU renderer, through a single module (`src/upstream.ts`), so a version bump has one place to audit:

- `serializeScene`, `serializeCamera`, `sceneSignature` and the `Serialized*` types: the scene snapshot.
- `USER_DATA_KEY` and the `userData.cpuRenderer` extensions. The key stays `cpuRenderer` in both renderers: it is
  historical, and renaming it would break scenes that already carry it.
- `RenderSettings`, `FrameSettings` and the tone mapping transforms.
- The GPGPU pipeline descriptors a module publishes.

What is rewritten in WGSL: material evaluation, the integrator, texture sampling and the environment. The CPU sources
stay the algorithmic reference, line by line.

## Modules

- `device.ts`: adapter and device acquisition, requested limits, loss handling.
- `camera.ts`: the camera's matrices. Every ray goes through the inverse of the camera's own projection.
- `geometry.ts`, `bvh.ts`: flat triangle and attribute buffers, and the BVH built straight into 32 byte nodes. Packing
  also builds what Three draws in screen space (points, lines, sprites, dashes) as triangles facing the camera, applies
  displacement maps, and runs vertex shaders.
- `structLayout.ts`, `frameLayout.ts`, `materialLayout.ts`: one description of each struct, from which both the WGSL
  declaration and the TypeScript writer are generated, so their word offsets cannot drift apart.
- `texturePack.ts`, `lights.ts`: the remaining storage buffer layouts. `lights.ts` holds the emissive triangles and
  Three's analytic lights in one cumulative distribution, the triangles first and in triangle order, so the same buffer
  answers both questions the integrator asks: pick a light by probability, and, given a triangle a bounce ray hit, how
  likely direct sampling was to have found it.
- `environment.ts`: the surrounding image and the distribution that lets a shadow ray aim at its bright parts.
- `shaderProgram.ts`, `sceneAssets.ts`: the GLSL translation, the procedural sky and the compute bakes (see below).
- `planarCapture.ts`, `volumetric.ts`: the CPU renderer's planar captures and volumetric lights.
- `sceneUpload.ts`: the uploads, and the joining of buffers described below.
- `accumulator.ts`, `pathTracePass.ts`, `presentPass.ts`, `dispatchScheduler.ts`: the frame loop.
- `wgsl/*.wgsl.ts`: the shader source.

## Decisions worth not relitigating

**A megakernel, not a wavefront.** One compute invocation per pixel, the bounce loop inside the kernel, `8 x 8`
workgroups so neighbouring pixels stay coherent. A wavefront pays off with many distinct materials and high register
pressure; here there is a single evaluator behind flags, and the queue traffic (a ray state per pixel per bounce) costs
most on exactly the weak hardware this has to serve. The WGSL is split into pure functions so that moving to a wavefront
later rewrites the scheduler, not the BRDFs. The signal to revisit: a kernel past roughly 700 lines, or throughput
falling off a cliff when a lobe is added.

**Arrays of the same kind share a binding.** A WebGPU device only has to offer eight storage buffers per shader stage,
and the kernel wants eleven arrays. Normals and the UV sets go in one buffer, the per-triangle material indices and the
hierarchy's triangle order in another, and the texture table sits in front of the texels it points at, with the uniform
saying where each second half begins. Positions, materials, nodes, lights and the accumulation buffer keep their own.
Merging costs a scene some headroom against `maxStorageBufferBindingSize`, which is why positions stay separate: they are
the largest array and the hottest one.

**Textures in a storage buffer, sampled by hand.** Core WebGPU has no binding arrays, and `texture_2d_array` demands one
size and one format for every layer, which the scene snapshot does not provide (arbitrary sizes, per texture wrap and UV
matrix, authored mipmaps). Compute shaders have no implicit derivatives either, so the level of detail is computed by
hand regardless and half of a hardware sampler's advantage disappears. A fast path can add `texture_2d_array` later, for
scenes where every texture happens to be uniform.

**WGSL lives in `src/wgsl/*.wgsl.ts`, as template literals.** The build is plain `tsc`, and `tsc` does not copy `.wgsl`
files into `dist`. The Tinypoly rule of keeping shader source in its own file exists because of a build time obfuscation
plugin that does not apply to a published library.

**The hierarchy is built here, not imported.** The CPU renderer has a binned SAH builder, but it stores bounds as
doubles for a scalar traversal, and re-packing that tree would still leave every minimum needing to be rounded down and
every maximum up before a float32 shader may use it. Building from the same `Float32Array` the shader reads makes every
corner exactly representable and the widening unnecessary. Revisit if the two builders start to drift.

**Shaders: vertex stages interpreted, fragment stages translated by naga.** A Three shader is two different problems.

- **A vertex shader decides where the triangles are**, so it has to run before the hierarchy is built, which happens on
  the CPU. It runs there, once per vertex, on the CPU renderer's own GLSL interpreter (`CpuShader`), with the inputs the
  CPU renderer gives it. Its `gl_Position` is taken back to world space through the camera, which is how displacement,
  billboards and `gl_PointSize` come out exactly as drawn; its varyings are stored per triangle corner.
- **A fragment shader runs per sample**, inside the kernel. `naga-wasm` (naga 30) translates it to WGSL after a rewrite
  that turns a program into functions: every uniform, varying and built-in becomes a module scope global the kernel
  fills in, every texture call becomes a function the kernel provides (reading the texture table by hand), and `discard`
  sets a flag. Uniforms are loaded from the texture buffer by code generated from the types naga wrote, so arrays,
  structs and matrices need no table of their own. What comes back is renamed into the kernel's namespace and called
  through a `switch` on the material's program. naga is loaded only for a scene that has shaders.
- **Derivatives are real.** A shader that takes `dFdx`, `dFdy` or `fwidth` runs three times per hit, for the pixel to the
  right, the pixel above and the pixel itself; each call site stores what the neighbours computed there, and the third run
  reads the difference. The neighbours' varyings come from the barycentrics of their camera rays on the hit triangle's
  plane. It is a quad's work in sequence, correct wherever the three runs take the same path through the shader, which is
  the assumption a GPU's own quads make.
- **An `onBeforeCompile` hook** runs on the material's values before lighting (`diffuseColor`, `normal` in view space,
  `roughnessFactor`, `metalnessFactor`, `totalEmissiveRadiance`), which is the part of Three's shader the CPU renderer's
  snapshot captures. The path tracer then lights the result itself.

The CPU renderer's tree-walking front end, considered in an earlier version of this decision, is not used: naga parses,
type checks and restructures, which was the bulk of the work. A shader that naga refuses (a `switch` that falls through
was the one known case, and is now rewritten) is reported with `console.warn` and its material renders as its colour,
rather than failing the frame.

**Adaptive sampling spends a fixed budget where the noise is.** A frame has the work of `settings.samples` passes over
the whole image to spend. Every pixel keeps the sum of its squared luminance beside its radiance, which gives its
variance; after 32 paths, a pixel whose standard error (over the square root of its luminance, roughly how visible the
error is once encoded) is under half an 8 bit step stops, provided its 3 x 3 neighbourhood has too. The neighbourhood is
what keeps a rare bright path honest: a pixel judged alone would stop where it happened not to have seen one yet, and
that area would converge too dark. Each traced row sets a flag, read back after the pass, and the next pass dispatches
only the rows that still have work, until the budget is spent or a pixel reaches four times its share. Stopping on the
samples themselves is a slight bias, the usual price of adaptive sampling.

**The denoiser is the spatial filter of SVGF, with diffuse and specular light apart.** The temporal half of SVGF, and of
NVIDIA's NRD, has nothing to do here: the image does not move while it converges, and the accumulation buffer already
is a perfect history. What carries over from NRD's RELAX and REBLUR is splitting the signal.

- **Two signals.** Every path is split at its first scatter: what the diffuse lobe carried on (and what the camera sees
  directly) is diffuse, what a specular lobe, the clearcoat or transmission carried on is specular. Direct light at that
  vertex is split by evaluating the diffuse term alone. Each pixel keeps the diffuse radiance beside the total, and the
  squared luminance of each signal, which gives each its own variance.
- **Guides**, taken at the first surface the camera ray stops at: the diffuse albedo, the specular one (the GGX lobe's
  directional albedo from Karis' fit, with the clearcoat, white through glass), the normal, the distance, the roughness,
  and how far the specular continuation went before it hit something.
- **Filtering.** Each signal is divided by its albedo, so the filter sees light and never blurs a texture. A pixel
  brighter than all eight neighbours is clamped to the brightest of them (anti-firefly). Then up to five passes of a
  5 x 5 a-trous wavelet, taps 1 to 16 pixels apart. Diffuse taps are weighed by the agreement of normals and depths and
  by their luminance against the pixel's own variance, prefiltered over 3 x 3. Specular taps also by roughness, by
  normals more sharply the smoother the surface, and within the footprint the lobe leaves on the screen: the lobe's angle
  (alpha) times the distance to what it reflects, over that distance plus the surface's, in pixels. A mirror is left
  alone, a rough reflection of something far is smoothed widely. The variance is filtered along, so each pass trusts the
  last.
- **Mixing back.** Weights are normalized, so a flat area keeps its mean exactly. Each signal is then mixed with what
  was traced by how noisy it still is, by the error adaptive sampling measures, from where it counts as converged to four
  times that: a converged pixel keeps exactly what was traced. `DenoiserSettings` scales the mix, the reach, the
  luminance tolerance and the anti-firefly; changing it filters the finished accumulation again, without tracing.

It runs in its own pipelines, outside the kernel's eight storage buffers, on the canvas at the end of every pass and on
the image `render()` resolves with. The accumulation holds six vec4 per pixel for all of this, 96 bytes, and the filter
five more buffers of 16: a 2000 x 1350 image takes some 460 MB on the device.

## Compatibility

This is a path tracer, **not Three.js/WebGL conformance, and not a match for the CPU renderer's image**. What follows is
what the renderer reads today. Anything absent is ignored in silence unless it is marked as an error, so a host reading
this table is the only way to know.

### Geometry and cameras

| Works | Notes |
| --- | --- |
| Triangle meshes, transformed to world space | Every triangle carries its own three vertices |
| `InstancedMesh`, with `instanceColor` | The scene snapshot expands it to one mesh per instance |
| `SkinnedMesh`, morph targets | The snapshot bakes the deformed positions |
| Per-group materials, indexed or not, `drawRange` | |
| Vertex normals, or a face normal when absent; `flatShading` | |
| Vertex colours (`vertexColors`), three or four components | |
| Two UV sets, `uv` and `uv1` | A texture asking for a higher channel falls back to `uv1` |
| A mirroring transform | The winding is turned around, as Three flips its front face |
| `Points` | A square of `size` pixels facing the camera, with `sizeAttenuation`, as `gl_PointCoord` maps it |
| `Line`, `LineSegments`, `LineLoop`, `LineDashedMaterial` | One pixel wide, as WebGL draws every line; dashes are cut into pieces |
| `Sprite` | Laid out around its `center`, with `rotation` and `sizeAttenuation` |
| Any camera with a projection matrix | Perspective, orthographic, zoom, `setViewOffset`, film offset |

Camera rays start on the near plane and stop at the far plane, which clip a surface as Three clips a fragment.
Points, lines, sprites and vertex shaders face or depend on the camera, so a scene with them is packed again whenever the
camera or the image size changes.

`material.side` culls the faces Three culls, for the camera's own rays only: a mesh still casts shadows and holds light
in from both sides, which is what a closed room drawn `BackSide` for a cutaway view needs. Per-mesh `castShadow` and
`receiveShadow` are ignored: a traced shadow comes from the geometry that casts it.

### Materials

| Type | What happens |
| --- | --- |
| `MeshStandardMaterial` | Three r185's direct lighting: Lambert diffuse, GGX with its multiple scattering term |
| `MeshPhysicalMaterial` | Adds transmission, dispersion, clearcoat, sheen, iridescence, anisotropy, specular colour and intensity |
| `MeshLambertMaterial` | Lambert diffuse |
| `MeshPhongMaterial` | Lambert diffuse and Three's Blinn-Phong highlight, sampled by its own lobe |
| `MeshToonMaterial` | Three's ramp, from `gradientMap` or its two steps, in place of the cosine |
| `MeshBasicMaterial`, `PointsMaterial`, `LineBasicMaterial`, `SpriteMaterial` | Unlit, which in a path tracer means the surface emits its own colour |
| `MeshNormalMaterial`, `MeshDepthMaterial`, `MeshDistanceMaterial`, `MeshMatcapMaterial` | Unlit, computed as Three's shaders compute them |
| `ShadowMaterial` | A shadow catcher: its colour where a shadow casting light is blocked, nothing elsewhere |
| `ShaderMaterial`, `RawShaderMaterial` | The fragment shader runs, translated; its output is the colour the surface shows, unlit |
| A built-in material patched with `onBeforeCompile` | The hooks run on the material's values before lighting |

Maps read: `map`, `alphaMap`, `emissiveMap`, `roughnessMap`, `metalnessMap`, `normalMap` (tangent or object space),
`bumpMap`, `displacementMap`, `lightMap`, `envMap`, `specularMap`, `specularColorMap`, `specularIntensityMap`,
`transmissionMap`, `thicknessMap`, `clearcoatMap`, `clearcoatRoughnessMap`, `clearcoatNormalMap`, `sheenColorMap`,
`sheenRoughnessMap`, `iridescenceMap`, `iridescenceThicknessMap`, `anisotropyMap`, `gradientMap`, `matcap`. Each one reads
the channel Three's chunks read and honours the texture's own UV transform, wrap modes, filter and colour space.

Coverage follows Three's chunks: `alphaTest` cuts, `alphaHash` keeps the share, and only a `transparent` material with
normal blending lets the rest through; an opaque one is opaque whatever its opacity. Additive blending adds the surface
to what is behind it, multiply and subtractive filter it. Material clipping planes (with `clipIntersection`),
`wireframe` (one pixel wide edges) and `visible`/`colorWrite` are honoured.

A **material's own `envMap`** is what the paths leaving it see instead of the scene's surroundings; on Basic, Lambert
and Phong it blends over the colour through `combine` and `reflectivity`, mirrored or bent by `refractionRatio`. A
**light map** adds its light to what the camera sees of the surface and is not traced further: the path traces that light
itself, and counting the bake again would double it.

A shader's output is taken as linear light, as in the CPU renderer: the tone mapping and colour space chunks are emptied,
because the renderer applies both to the image. Its fog chunk is emptied too, because the integrator lays Three's fog over
every material that accepts it. A fragment shader runs only where a path hits the surface, so a shader whose output
depends on `gl_FragCoord` sees the camera's projection of the hit, which off screen can be anywhere.

**Transmission** follows Three's split: `transmission`, times its map and less metalness, is the share of the diffuse
layer that lets light through, and the specular lobe stays on top. A path picks that share by chance and then meets a
dielectric of index `ior`, reflecting by the exact Fresnel term and refracting the rest, rough or smooth. The base colour
tints light once, on the way in. `thickness` decides what the surface is:

- **Zero (Three's default) is a thin wall**, a window: light crosses it without bending, and so do shadow rays, tinted
  by the same amount. A room lit by the sun through its glazing works as it should.
- **Above zero the mesh bounds a volume** that refracts, and `attenuationColor` over `attenuationDistance` absorbs along
  the path inside it (Beer-Lambert). The mesh has to be closed with outward normals. A shadow ray cannot bend, so a volume
  blocks it: **a glass ball under a directional, point or spot light casts a solid shadow**, with no caustic, because
  those lights occupy no area a refracted path could find. Under an area light, emissive geometry or the environment
  the caustic appears, slowly. **Dispersion** splits a path into one colour the first time it enters such a volume, which
  then refracts with that colour's index, spread as Three spreads it.

A refraction does not spend a bounce; a reflection inside the glass does.

`aoMap` is ignored deliberately, not by omission. It bakes the occlusion this renderer traces, and applying both would
darken the same shadow twice.

Only the top mip level of a texture is read. A rasterizer needs the lower levels so minified detail does not alias; this
renderer already draws many jittered samples through every pixel, which averages that detail properly. Levels would buy
memory locality, not correctness. A bump map's slope is taken over one texel, as in the CPU renderer, where Three takes
it over one pixel.

### Lights

| Works | Notes |
| --- | --- |
| `DirectionalLight` | Aimed at directly, with `shadows.softness` as its angular diameter |
| `PointLight` | Three's own distance falloff and cutoff |
| `SpotLight` | Cone and penumbra as Three computes them, and its projected `map` |
| `RectAreaLight` | A real area light, sampled over its rectangle, so its penumbra is traced rather than approximated |
| `AmbientLight`, `HemisphereLight` | See the divergence below |
| Emissive geometry | Any triangle whose material emits, weighted by area and power |

`light.castShadow` gates the occlusion ray, as does `settings.shadows.enabled`. A `RectAreaLight` always occludes,
because Three has no shadow switch for one and an area light that cast none would look broken once traced. A toon
surface lit from behind is not shadowed by its own mesh, as Three's shadow map leaves it.

### The environment and the background

A vertical gradient, an equirectangular `.hdr` image, or the scene's procedural sky (`userData.cpuRenderer.sky`), with a
luminance distribution built over the image so a shadow ray can aim at a sun. `intensity` and `rotation` apply to the
gradient and the image. A procedural sky is baked by the CPU renderer's interpreter, lights with its own `intensity`,
ignores the rotation so its sun stays with its light, and paints the background without its intensity, with its
`foregroundShader` evaluated per pixel over it.

`settings.background` decides what the camera sees where nothing is in the way: the environment, a colour, or nothing,
in which case the image's alpha is the coverage of every pixel and the canvas is configured premultiplied. The
background shows through transmission, as Three's transmission shows it, and not in a reflection, which shows the
environment.

`scene.fog` (`Fog` and `FogExp2`) is laid over what the camera sees through the first surface, at that surface's view
depth, for every material that accepts fog, and the background stays clear, as in Three. `settings.fog`, when enabled,
covers every surface instead.

### Settings

Read: `samples`, `renderScale`, `exposure`, `toneMapping`, `vignette`, `grain`, `background`, `fog`, `depthOfField`
(the CPU renderer's 35 mm lens, `focusDistance`, `aperture` as an f-number, `blades` for polygonal bokeh),
`globalIllumination.bounces`, `globalIllumination.clamp`, `shadows.enabled`, `shadows.softness`.

Ignored: `tileSize`, `ambientOcclusion`, and everything else under `globalIllumination` (`enabled`, `samples`,
`intensity`, `cache`, `cacheSpacing`, `denoise`).

### The CPU renderer's extensions, and the rest

- **Light probes.** A `LightProbe`'s harmonics are radiance in Three's SH3 basis; they join the ambient surrounding,
  which delivers exactly the irradiance Three's shader convolves from them. The snapshot does not carry them, so they are
  read from the live scene `setScene` receives: a scene handed over already serialized has none.
- **Volumetric lights** (`userData.cpuRenderer.volumetric` on a point or spot light) are marched along every camera ray
  as the CPU renderer's `volumetricLight.js` marches them, up to the first surface, and added over the image the same way.
- **Planar captures** (`planarCapture`) are rendered by this renderer, path traced from the CPU renderer's mirrored and
  real cameras with its clip planes, into the textures the material samples. They follow the camera, so a scene with one
  is captured again whenever it moves.
- **Compute simulations** (`compute`) are baked on the CPU, by the CPU renderer's own pipeline.
- **Instanced geometry drawn by a shader**, such as `Line2`, `LineSegments2` and `Wireframe`: the vertex shader runs over
  the template once per instance, with the per-instance attributes at that instance.
- **A `switch` whose cases fall through**, which WGSL cannot express, is rewritten for that shader only: one pass of a
  loop, so `break` still leaves it, with an `if` per case. A `continue` inside such a case leaves the switch instead of
  continuing the loop around it.
- **Hooks after `<opaque_fragment>`** run on the radiance the path found on the camera's line of sight from that
  surface on, before the fog, as Three runs them on the lit colour. For a hook that is not linear in `gl_FragColor` this
  is applied per path rather than to the converged colour.

### Where this deliberately diverges from Three

- **`settings.samples` means paths per pixel here**, where the CPU renderer reads it as antialiasing samples and takes
  its path count from `globalIllumination.samples`. One number does both jobs in a path tracer.
- **An `AmbientLight` or a `HemisphereLight` becomes an equally bright surrounding.** Three adds them straight to a
  surface's diffuse term, which no path can carry; light that arrives from every direction is light from outside the
  scene. The irradiance matches Three exactly wherever the sky is open, and **contributes nothing inside a closed room**,
  which is the difference a host has to know about. It is not painted in the background, as in Three.
- **Transparency is stochastic.** A surface with an opacity below one lets that share of rays straight through, which is
  what makes a cutout cast a cutout shadow, and what makes the result converge rather than composite.
- **Unlit surfaces emit.** A basic material, a shader material, a sprite: whatever colour Three would draw, the surface
  radiates, so it also lights its surroundings (found by bounce rays, not aimed at).
- Global illumination is always on. There is no flag for it: tracing paths is what this renderer does.
- **Glass refracts** where Three's transmission samples a blurred copy of the opaque scene behind it. A volume bends what
  is behind it and reflects its surroundings properly; a thin wall is exact. Neither is what Three draws.

## Testing

The CI gate needs no GPU: scene packing (points, lines, sprites, dashes, displacement and winding included), BVH
packing, buffer layouts and alignment, the scheduler's feedback loop, and the upstream symbols imported from the CPU
renderer. `kernel.test.ts` validates the whole kernel with naga, with translated shaders of every kind in it, so a WGSL
mistake fails in CI rather than on a device. The hierarchy is checked against
brute force over every triangle, with a traversal in the test file that mirrors the WGSL one step for step, so a packing
mistake fails here instead of on a device where nothing can be printed.

GPU suites are named `*.gpu.test.ts` and skip themselves when no adapter is available. They exist to catch WGSL that does
not compile, which is the most common and most tedious error to find in a browser.

Headless browsers do not judge images here. The software path for WebGPU differs from real drivers in floating point
results, optional features and limits, which are the three dimensions a path tracer diverges in.

What does judge the integrator is `pnpm verify`, which needs a real GPU and so stays out of CI. It drives headless Chrome
over `scripts/verify.html` and prints a reading against an expectation for each check. Every scene in it has an answer
known on paper rather than a stored reference image: a unit albedo Lambert surface facing a light of irradiance `pi/4`
leaves a radiance of exactly `0.25`, whichever kind of light delivered it, so the same number checks the directional
falloff, the inverse square law, the spot cone, the rectangle's projected area, the ambient conversion, the sRGB decode
of a texture and the Fresnel split of the specular lobe. A wrong density or a misread texel compiles perfectly and simply
produces the wrong picture; these numbers say which part is wrong.

`pnpm verify --page other.html` runs another page under `scripts/` instead, for chasing one check without running all
of them.

It also reports how far neighbouring pixels sit apart at a low sample count, which is what a broken estimator and a badly
weighted light both show up as, and it is the measurement to trust over any single image.
