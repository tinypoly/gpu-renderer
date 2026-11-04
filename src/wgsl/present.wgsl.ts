/**
 * The presentation pass: divides the accumulation buffer by its sample count and applies exposure, tone mapping,
 * vignette and grain. The operators are ports of `color.ts` in the CPU renderer, which are in turn Three's ACES, AgX
 * and Khronos neutral curves, so the two renderers agree on the display transform even when they disagree on the light.
 */
export const PRESENT_WGSL = /* wgsl */ `
struct Present {
  resolution: vec2<u32>,
  exposure: f32,
  toneMapping: u32,
  vignette: f32,
  grain: f32,
  frameIndex: u32,
  // A transparent background: the canvas gets the coverage as alpha, premultiplied as it is configured.
  transparent: u32,
  // Whether the colour comes from the denoiser's output rather than straight from the accumulation.
  denoised: u32,
};

@group(0) @binding(0) var<uniform> present: Present;
@group(0) @binding(1) var<storage, read> accumulation: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> denoised: array<vec4<f32>>;

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  // One oversized triangle covers the viewport with no vertex buffer.
  let x = f32(index / 2u) * 4.0 - 1.0;
  let y = f32(index & 1u) * 4.0 - 1.0;

  return vec4<f32>(x, y, 0.0, 1.0);
}

fn toneMapAces(color: vec3<f32>) -> vec3<f32> {
  let toAces = mat3x3<f32>(
    vec3<f32>(0.59719, 0.076, 0.0284),
    vec3<f32>(0.35458, 0.90834, 0.13383),
    vec3<f32>(0.04823, 0.01566, 0.83777));
  let fromAces = mat3x3<f32>(
    vec3<f32>(1.60475, -0.10208, -0.00327),
    vec3<f32>(-0.53108, 1.10813, -0.07276),
    vec3<f32>(-0.07367, -0.00605, 1.07602));
  let v = toAces * (color / 0.6);

  return fromAces * ((v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.432951) + 0.238081));
}

fn toneMapAgx(color: vec3<f32>) -> vec3<f32> {
  let toAgx = mat3x3<f32>(
    vec3<f32>(0.6274, 0.0691, 0.0164),
    vec3<f32>(0.3293, 0.9195, 0.088),
    vec3<f32>(0.0433, 0.0113, 0.8956));
  let inset = mat3x3<f32>(
    vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
    vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
    vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859));
  let outset = mat3x3<f32>(
    vec3<f32>(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
    vec3<f32>(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
    vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
  let fromAgx = mat3x3<f32>(
    vec3<f32>(1.6605, -0.1246, -0.0182),
    vec3<f32>(-0.5876, 1.1329, -0.1006),
    vec3<f32>(-0.0728, -0.0083, 1.1187));
  let logged = inset * (toAgx * color);
  let x = clamp((log2(max(logged, vec3<f32>(1e-10))) + 12.47393) / 16.499999, vec3<f32>(0.0), vec3<f32>(1.0));
  let curved = 15.5 * pow(x, vec3<f32>(6.0)) - 40.14 * pow(x, vec3<f32>(5.0)) + 31.96 * pow(x, vec3<f32>(4.0))
    - 6.868 * pow(x, vec3<f32>(3.0)) + 0.4298 * x * x + 0.1191 * x - 0.00232;

  return fromAgx * pow(max(outset * curved, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn toneMapNeutral(color: vec3<f32>) -> vec3<f32> {
  let low = min(color.r, min(color.g, color.b));
  let offset = select(0.04, low - 6.25 * low * low, low < 0.08);
  let shifted = color - offset;
  let peak = max(shifted.r, max(shifted.g, shifted.b));

  if (peak < 0.76) {
    return shifted;
  }

  let newPeak = 1.0 - 0.24 * 0.24 / (peak + 0.24 - 0.76);
  let g = 1.0 - 1.0 / (0.15 * (peak - newPeak) + 1.0);

  return shifted * (newPeak / peak) * (1.0 - g) + newPeak * g;
}

fn toneMap(color: vec3<f32>) -> vec3<f32> {
  let exposed = max(color * present.exposure, vec3<f32>(0.0));
  var mapped = exposed;

  if (present.toneMapping == 0u) {
    mapped = toneMapAces(exposed);
  } else if (present.toneMapping == 1u) {
    mapped = toneMapAgx(exposed);
  } else if (present.toneMapping == 2u) {
    mapped = toneMapNeutral(exposed);
  }

  return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSrgb(color: vec3<f32>) -> vec3<f32> {
  let low = color * 12.92;
  let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;

  return select(high, low, color <= vec3<f32>(0.0031308));
}

fn hash(value: u32) -> f32 {
  var state = value * 747796405u + 2891336453u;
  state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

  return f32((state >> 22u) ^ state) * 2.3283064365386963e-10;
}

@fragment
fn fragmentMain(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let pixel = vec2<u32>(position.xy);
  let index = pixel.y * present.resolution.x + pixel.x;
  let stored = accumulation[index];

  // Nothing has been traced into this pixel yet: it stays see-through, so whatever is under the canvas shows.
  if (stored.w <= 0.0) {
    return vec4<f32>(0.0);
  }

  var color = select(stored.rgb / stored.w, vec3<f32>(0.0), stored.w <= 0.0);

  if (present.denoised != 0u) {
    color = denoised[index].rgb;
  }
  var alpha = 1.0;

  if (present.transparent != 0u) {
    let coverage = accumulation[present.resolution.x * present.resolution.y + index].x;
    alpha = select(clamp(coverage / stored.w, 0.0, 1.0), 0.0, stored.w <= 0.0);
    // The radiance is premultiplied by the coverage; the tone curve applies to the colour itself.
    color = select(color / alpha, vec3<f32>(0.0), alpha <= 0.0);
  }

  let centred = position.xy / vec2<f32>(present.resolution) - 0.5;
  let falloff = 1.0 - present.vignette * dot(centred, centred) * 2.0;
  var output = linearToSrgb(toneMap(color * max(falloff, 0.0)));

  if (present.grain > 0.0) {
    let noise = hash(index * 9781u + present.frameIndex * 6151u) - 0.5;
    output = clamp(output + noise * present.grain * 0.1, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  return vec4<f32>(output * alpha, alpha);
}
`;
