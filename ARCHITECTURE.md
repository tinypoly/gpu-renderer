# Architecture

How `@tinypoly/gpu-renderer` works inside, and the decisions behind it. For installation and usage, see the
[README](README.md).

The renderer takes a serialized Three.js scene, uploads geometry, a BVH, materials, textures and lights to the GPU, and
runs a path tracing compute shader that accumulates radiance into a float buffer. A small render pipeline presents that
buffer, dividing by the sample count and applying exposure, tone mapping, vignette and grain. There is no worker pool and
no `SharedArrayBuffer`: the state lives on the device. The one worker there is builds the BVH of a large scene off the
page's thread, and changes nothing about the tree it builds.

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
- `geometry.ts`, `bvh.ts`: flat triangle and attribute buffers, and the BVH: built as a binary tree, then laid out in
  64 byte nodes that each hold their two children's boxes, so one read tests both and a leaf is never read. Packing
  also builds what Three draws in screen space (points, lines, sprites, dashes) as triangles facing the camera, applies
  displacement maps, and runs vertex shaders. A plain mesh, with no vertex shader and no displacement, takes a faster
  path that transforms each vertex once into typed arrays and writes exactly what the general one would.
- `bvhBuilder.ts`, `bvhWorker.ts`: the BVH built in a Web Worker from a copy of the positions, for scenes of at least
  `BVH_WORKER_MIN_TRIANGLES`; on the main thread below that, when `bvhWorker` is false, or for good once the worker
  fails to start or load.
- `structLayout.ts`, `frameLayout.ts`, `materialLayout.ts`: one description of each struct, from which both the WGSL
  declaration and the TypeScript writer are generated, so their word offsets cannot drift apart.
- `texturePack.ts`, `lights.ts`: the remaining storage buffer layouts. `lights.ts` holds the emissive triangles and
  Three's analytic lights in one cumulative distribution, the triangles first and in triangle order, so the same buffer
  answers both questions the integrator asks: pick a light by probability, and, given a triangle a bounce ray hit, how
  likely direct sampling was to have found it.
- `environment.ts`: the surrounding image and the distribution that lets a shadow ray aim at its bright parts.
- `shaderProgram.ts`, `sceneAssets.ts`, `skyBake.ts`: the GLSL translation, the procedural sky and the compute bakes (see below).
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
hierarchy's triangle order (both ways) in another, and the lookup tables and the texture table sit in front of the
texels they point at, with the uniform saying where each second half begins. The positions are uploaded in the
hierarchy's leaf order, so a leaf is one run of the buffer; anything that reads a triangle by its own number goes through
the inverse order. Positions, materials, nodes, lights and the accumulation buffer keep their own.
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
only the rows that still have work, until the budget is spent or a pixel reaches four times its share. Before 32 paths
no pixel can stop, so those passes take up to four paths a dispatch and read and write the accumulation once for them. Stopping on the
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

It runs in its own pipelines, outside the kernel's eight storage buffers, on the canvas a few times a second while a
frame traces (at the end of a pass, at most every 200 ms) and on the image `render()` resolves with. The accumulation
holds six vec4 per pixel for all of this, 96 bytes, and the filter six more buffers of 16 and one of 4, the last two
the guides each tap compares, worked out once per pixel: a 2000 x 1350 image takes some 530 MB on the device.

**Light inside materials and volumes is a walk, compiled in only where a scene has it.** Both are media the path
crosses instead of surfaces it bounces off, and both are written into the bounce loop rather than as loops of their
own, so the one traversal call in the loop serves them.

- **Subsurface scattering** is a random walk, as in Cycles. A vertex on such a material takes the walk instead of its
  surface lobes by chance, in proportion to the light each accounts for, so neither is weighted against the other; the
  lobes it keeps lose that share of the diffuse layer. Inside, each colour has its own mean free path (`radius`) and the
  single scattering albedo that the material's colour asks for (van de Hulst's relation, inverted); distances are drawn
  for a colour picked by what the path still carries of it and weighed over all three (spectral MIS). Where the walk
  reaches a surface it leaves through a white Lambert surface facing out, which takes direct light there.
- **Volumes** are homogeneous media bounded by an invisible mesh. A medium of one grey density lets the distance be
  drawn after the segment is traced: the next hit is found, then a collision before it is drawn, and either way the
  weight is one. A collision is a vertex whose lobe is the phase function (Henyey-Greenstein), with light sampled from
  there. Shadow rays fade through every boundary they cross, and through the volume they start in; a light that casts no
  shadow still fades, since the medium is not geometry. Whether the camera stands inside one is found on the CPU, by the
  parity of a ray's crossings, once per frame.
- `SUBSURFACE_WALK` and `VOLUMES` are WGSL constants the pipeline is compiled with, true only for a scene with such a
  material. Carrying the walk's code in every scene measured 2 to 5 percent slower on scenes that never run it (see
  the next decision); folded away, a plain scene renders the same image as before, byte for byte.
- The snapshot has no field for either medium, so `serializeLive` marks the material's name while the CPU renderer
  serializes it, as it already renames a shader material's type, and attaches the medium after. A snapshot built
  elsewhere can carry `subsurface` or `volume` on its materials itself.

**The lens and the sensor are a pass of their own, after the radiance.** Bloom (Jimenez's 13 tap pyramid, folded
back with a tent), the aperture's rays (gathered along each ray from a quarter sized level), lateral chromatic
aberration and the white balance (a von Kries scaling in Bradford's cone space, exactly the identity at 6500 K) work on
linear light between the denoiser and the display transform, into a buffer the presentation and the image resolve read
in its place. Bloom and rays only move light, with weights that sum to one, so a flat image keeps its mean. Aerial
perspective is not part of it: it needs each sample's distance, so the kernel lays it over the camera's first surface,
as it lays Three's fog, with the extinction Koschmieder's law gives each colour and the glow of the surroundings and of
the brightest directional light scattered by the air's phase function. Lens distortion bends the camera rays, so it
needs no resampling.

**Lights are picked through a tree, by what they can deliver to the shaded point.** Picking by power alone sends a
small lamp beside a surface as few samples as its power earns against a large panel across the room, and its light
then arrives through bounce rays as fireflies. The lights that sit in the scene (emissive triangles, point, spot and
rectangular lights) lead the light buffer and are bound by a tree built on the CPU (`lightTree.ts`): each node keeps
its box, its power and a cone of the directions its lights face, and the kernel walks down choosing a child by its
importance from the point, power over squared distance times how well the cone and the receiving normal face each other,
widened by the angle the box takes (Conty Estevez and Kulla, 2018, without the splitting). The same walk, taken up from
a leaf, gives the chance a bounce ray's light was picked from the vertex it left, so the heuristic stays exact. The
directional lights and the environment follow, picked as before by the irradiance they deliver; the share between the
two groups is what the old distribution gave them. The nodes live in the texture buffer's extra words, sixteen a node,
because the kernel's eight storage bindings are taken.

**Bevel rounds edges at shading time, by one probe a sample.** As Cycles' bevel node, but stochastic: a point on a disk
of the radius around the shaded point, across an axis picked along the normal half the time and each tangent a
quarter, and a short ray along it that finds the same material there. Its normal turns the shading normal by as much
as it differs from the normal here, which keeps a normal map, and the pixel's samples average the turn into a rounded
edge. It costs a traversal a vertex on such materials, so `BEVEL` is compiled in only for a scene that has one.

**The kernel is bound by its size, not by its arithmetic.** Measured on an Apple GPU (`scripts/bench.html`, every
build timed in turns against the other), the nodes that hold their children's boxes made whole renders 16 to 21 percent
faster, but only once the traversal loop tested leaves in one place: with the leaf test written out wherever a leaf
turned up, three times a traversal, the same layout was slower than the binary tree it replaced. Replacing the sampler's
bit loop and the sRGB powers with tables gained only 3 to 5 percent. Every function call is inlined, so a traversal, a
light sample or a shader is copied wherever it is called, and what each copy costs in registers is paid across the whole
kernel. Fewer copies are not automatically better either: folding the closest and any hit searches into one call, or
calling the direct light from one place, measured slower. Measure a kernel change; do not judge it by the work it seems
to remove.

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
| Emissive geometry | Any triangle whose material emits, picked through the light tree by what it delivers to the shaded point |

`light.castShadow` gates the occlusion ray, as does `settings.shadows.enabled`. A `RectAreaLight` always occludes,
because Three has no shadow switch for one and an area light that cast none would look broken once traced. A toon
surface lit from behind is not shadowed by its own mesh, as Three's shadow map leaves it.

### The environment and the background

A vertical gradient, an equirectangular `.hdr` image, or the scene's procedural sky (`userData.cpuRenderer.sky`), with a
luminance distribution built over the image so a shadow ray can aim at a sun. `intensity` and `rotation` apply to the
gradient and the image. A procedural sky's shader is translated like any other and run on the device over the
panorama, one invocation per texel, which is read back for the luminance distribution; the CPU renderer's interpreter
bakes it only when the shader does not translate. The bake is kept while the shader and its uniforms stay the same,
so a scene that changes anything else does not bake the sky again. The sky lights with its own `intensity`,
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
`globalIllumination.bounces`, `globalIllumination.clamp`, `shadows.enabled`, `shadows.softness`, and the groups only this
renderer has: `globalIllumination.russianRoulette` and `rouletteStart`, `bloom`, `glare`, `lens`, `camera`,
`atmosphere`, `subsurface` and `volumes`. Its defaults differ from the CPU renderer's in `toneMapping` (AgX),
`globalIllumination.bounces` (12) and `globalIllumination.clamp` (10).

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
  radiates, so it also lights its surroundings. An opaque `MeshBasicMaterial` of a fixed colour (normal blending, no
  alpha, light or environment map) is aimed at as a light, its colour times its map and vertex colours read at the
  sampled point; anything else unlit is only found by bounce rays, since what it shows is not known until it runs.
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
