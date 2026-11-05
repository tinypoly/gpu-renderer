/**
 * What the lens and the sensor do to the light before it is exposed: a halo around bright light (bloom), colour
 * fringes towards the corners (lateral chromatic aberration) and the camera's white balance. Everything here is linear
 * light, between the traced (or denoised) radiance and the presentation pass, which exposes and tone maps the result.
 *
 * The halo is built over a pyramid of half sized copies of the image, as Jimenez's "Next generation post processing in
 * Call of Duty" (2014) builds it: each level is a 13 tap filter of the one above, and the levels are folded back from
 * the smallest up with a tent filter, each mixed with the one below by the radius. The result is a wide, smooth spread
 * whose weights sum to one, so a flat image is left as it was and the halo takes light from where it is rather than
 * adding any: the image keeps its exact mean.
 *
 * The rays a bright light throws through the aperture are gathered the same way, from a quarter sized level: each
 * texel takes the light of the texels behind it along every ray's direction, fading with the distance, with weights
 * that again sum to one.
 */
export const LENS_WGSL = /* wgsl */ `
struct Lens {
  resolution: vec2<u32>,
  // Whether the colour comes from the denoiser's output rather than straight from the accumulation.
  denoised: u32,
  // Upsampling only: the level read from sits in the downsampled pyramid (the smallest one) instead of the folded one.
  fromDown: u32,
  sourceSize: vec2<u32>,
  // Where the level read from starts in its pyramid; a size of the full resolution reads the image itself.
  sourceOffset: u32,
  // The first downsample reads the image itself, and averages its taps by luminance against fireflies.
  fromImage: u32,
  targetSize: vec2<u32>,
  targetOffset: u32,
  // Where the largest level of the folded pyramid starts, which the composite reads the halo from.
  haloOffset: u32,
  haloSize: vec2<u32>,
  // How much of the light the halo takes, and how much each level takes from the one below it.
  intensity: f32,
  radius: f32,
  // How far red and blue move apart at the corners, as a share of the distance from the centre.
  aberration: f32,
  // How much of the light the aperture's rays take, how many there are, and how far they reach in streak texels.
  glare: f32,
  rays: u32,
  glareLength: f32,
  // The white balance, a linear sRGB matrix, row by row.
  white0: vec4<f32>,
  white1: vec4<f32>,
  white2: vec4<f32>,
  // The size of the level the rays are gathered at, and the angle of the first ray, in radians.
  streakSize: vec2<u32>,
  glareAngle: f32,
  pad0: f32,
};

const PI: f32 = 3.14159265358979;

/** Taps along each of the aperture's rays. */
const STREAK_TAPS: i32 = 24;

@group(0) @binding(0) var<uniform> lens: Lens;
@group(0) @binding(1) var<storage, read> accumulation: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> denoised: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> down: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> up: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> graded: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> streaked: array<vec4<f32>>;

/** The image's own texel: its mean so far, or nothing where no path has landed yet. */
fn imageTexel(x: i32, y: i32) -> vec3<f32> {
  let size = vec2<i32>(lens.resolution);
  let index = u32(clamp(y, 0, size.y - 1) * size.x + clamp(x, 0, size.x - 1));

  if (lens.denoised != 0u) {
    return denoised[index].rgb;
  }

  let stored = accumulation[index];

  return select(stored.rgb / stored.w, vec3<f32>(0.0), stored.w <= 0.0);
}

fn downTexel(offset: u32, size: vec2<u32>, x: i32, y: i32) -> vec3<f32> {
  let s = vec2<i32>(size);

  return down[offset + u32(clamp(y, 0, s.y - 1) * s.x + clamp(x, 0, s.x - 1))].rgb;
}

fn upTexel(offset: u32, size: vec2<u32>, x: i32, y: i32) -> vec3<f32> {
  let s = vec2<i32>(size);

  return up[offset + u32(clamp(y, 0, s.y - 1) * s.x + clamp(x, 0, s.x - 1))].rgb;
}

/** Bilinear weights and the texel below and left of a point given in texels, edges clamped. */
fn bilinearAt(position: vec2<f32>) -> vec4<f32> {
  let p = position - 0.5;
  let base = floor(p);

  return vec4<f32>(base, p - base);
}

fn imageBilinear(position: vec2<f32>) -> vec3<f32> {
  let b = bilinearAt(position);
  let x = i32(b.x);
  let y = i32(b.y);

  return mix(
    mix(imageTexel(x, y), imageTexel(x + 1, y), b.z),
    mix(imageTexel(x, y + 1), imageTexel(x + 1, y + 1), b.z), b.w);
}

fn downBilinear(offset: u32, size: vec2<u32>, position: vec2<f32>) -> vec3<f32> {
  let b = bilinearAt(position);
  let x = i32(b.x);
  let y = i32(b.y);

  return mix(
    mix(downTexel(offset, size, x, y), downTexel(offset, size, x + 1, y), b.z),
    mix(downTexel(offset, size, x, y + 1), downTexel(offset, size, x + 1, y + 1), b.z), b.w);
}

fn upBilinear(offset: u32, size: vec2<u32>, position: vec2<f32>) -> vec3<f32> {
  let b = bilinearAt(position);
  let x = i32(b.x);
  let y = i32(b.y);

  return mix(
    mix(upTexel(offset, size, x, y), upTexel(offset, size, x + 1, y), b.z),
    mix(upTexel(offset, size, x, y + 1), upTexel(offset, size, x + 1, y + 1), b.z), b.w);
}

/** The level being filtered, read bilinearly at a point given in its own texels. */
fn sourceAt(position: vec2<f32>) -> vec3<f32> {
  if (lens.fromImage != 0u) {
    return imageBilinear(position);
  }

  return downBilinear(lens.sourceOffset, lens.sourceSize, position);
}

fn luminance(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/** Karis' average: a group of taps weighed by one over one plus its luminance, so a lone firefly cannot flare. */
fn karis(sum: vec3<f32>) -> vec4<f32> {
  let weight = 1.0 / (1.0 + luminance(sum));

  return vec4<f32>(sum * weight, weight);
}

/** One level of the pyramid from the one above it: Jimenez's 13 taps, four overlapping boxes and a centre one. */
@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= lens.targetSize.x || id.y >= lens.targetSize.y) {
    return;
  }

  let scale = vec2<f32>(lens.sourceSize) / vec2<f32>(lens.targetSize);
  let centre = (vec2<f32>(id.xy) + 0.5) * scale;
  let a = sourceAt(centre + vec2<f32>(-2.0, -2.0));
  let b = sourceAt(centre + vec2<f32>(0.0, -2.0));
  let c = sourceAt(centre + vec2<f32>(2.0, -2.0));
  let d = sourceAt(centre + vec2<f32>(-2.0, 0.0));
  let e = sourceAt(centre);
  let f = sourceAt(centre + vec2<f32>(2.0, 0.0));
  let g = sourceAt(centre + vec2<f32>(-2.0, 2.0));
  let h = sourceAt(centre + vec2<f32>(0.0, 2.0));
  let i = sourceAt(centre + vec2<f32>(2.0, 2.0));
  let j = sourceAt(centre + vec2<f32>(-1.0, -1.0));
  let k = sourceAt(centre + vec2<f32>(1.0, -1.0));
  let l = sourceAt(centre + vec2<f32>(-1.0, 1.0));
  let m = sourceAt(centre + vec2<f32>(1.0, 1.0));
  var result: vec3<f32>;

  if (lens.fromImage != 0u) {
    let groups = karis((j + k + l + m) * 0.25) * 0.5
      + karis((a + b + d + e) * 0.25) * 0.125
      + karis((b + c + e + f) * 0.25) * 0.125
      + karis((d + e + g + h) * 0.25) * 0.125
      + karis((e + f + h + i) * 0.25) * 0.125;

    result = groups.rgb / max(groups.w, 1e-6);
  } else {
    result = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }

  down[lens.targetOffset + id.y * lens.targetSize.x + id.x] = vec4<f32>(max(result, vec3<f32>(0.0)), 1.0);
}

/** A 3 x 3 tent over the smaller level, read at a point given in its texels. */
fn tent(position: vec2<f32>) -> vec3<f32> {
  var sum = vec3<f32>(0.0);

  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let weight = f32((2 - abs(dx)) * (2 - abs(dy))) / 16.0;
      let at = position + vec2<f32>(f32(dx), f32(dy));

      if (lens.fromDown != 0u) {
        sum = sum + downBilinear(lens.sourceOffset, lens.sourceSize, at) * weight;
      } else {
        sum = sum + upBilinear(lens.sourceOffset, lens.sourceSize, at) * weight;
      }
    }
  }

  return sum;
}

/** One level folded back: its own light, mixed by the radius with the wider spread of every level below it. */
@compute @workgroup_size(8, 8, 1)
fn upsample(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= lens.targetSize.x || id.y >= lens.targetSize.y) {
    return;
  }

  let scale = vec2<f32>(lens.sourceSize) / vec2<f32>(lens.targetSize);
  let wider = tent((vec2<f32>(id.xy) + 0.5) * scale);
  let index = lens.targetOffset + id.y * lens.targetSize.x + id.x;

  up[index] = vec4<f32>(mix(down[index].rgb, wider, lens.radius), 1.0);
}

/** The aperture's rays at one texel of the streak level: the light behind it along each ray, fading with distance. */
@compute @workgroup_size(8, 8, 1)
fn streaks(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= lens.targetSize.x || id.y >= lens.targetSize.y) {
    return;
  }

  let centre = vec2<f32>(id.xy) + 0.5;
  var sum = vec3<f32>(0.0);
  var total = 0.0;

  for (var r = 0u; r < lens.rays; r = r + 1u) {
    let angle = lens.glareAngle + f32(r) * 2.0 * PI / f32(lens.rays);
    let along = vec2<f32>(cos(angle), sin(angle));

    for (var j = 1; j <= STREAK_TAPS; j = j + 1) {
      let t = f32(j) / f32(STREAK_TAPS);
      let weight = exp(-4.0 * t);

      sum = sum + downBilinear(lens.sourceOffset, lens.sourceSize, centre - along * (t * lens.glareLength)) * weight;
      total = total + weight;
    }
  }

  streaked[id.y * lens.targetSize.x + id.x] = vec4<f32>(sum / max(total, 1e-6), 1.0);
}

fn streakTexel(x: i32, y: i32) -> vec3<f32> {
  let s = vec2<i32>(lens.streakSize);

  return streaked[u32(clamp(y, 0, s.y - 1) * s.x + clamp(x, 0, s.x - 1))].rgb;
}

fn streakBilinear(position: vec2<f32>) -> vec3<f32> {
  let b = bilinearAt(position);
  let x = i32(b.x);
  let y = i32(b.y);

  return mix(mix(streakTexel(x, y), streakTexel(x + 1, y), b.z), mix(streakTexel(x, y + 1), streakTexel(x + 1, y + 1), b.z), b.w);
}

/** The folded halo at full resolution: a tent over its largest level, which is half the image's size. */
fn tentHalo(position: vec2<f32>) -> vec3<f32> {
  var sum = vec3<f32>(0.0);

  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let weight = f32((2 - abs(dx)) * (2 - abs(dy))) / 16.0;

      sum = sum + upBilinear(lens.haloOffset, lens.haloSize, position + vec2<f32>(f32(dx), f32(dy)) * 0.5) * weight;
    }
  }

  return sum;
}

/** The image as the lens delivers it: fringed towards the corners, with its halo, balanced for the camera's white. */
@compute @workgroup_size(8, 8, 1)
fn composite(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= lens.resolution.x || id.y >= lens.resolution.y) {
    return;
  }

  let size = vec2<f32>(lens.resolution);
  let position = vec2<f32>(id.xy) + 0.5;
  var color = imageTexel(i32(id.x), i32(id.y));

  // Lateral chromatic aberration: each colour is magnified a little differently, red outwards and blue inwards,
  // in proportion to the distance from the centre, which is how a lens's dispersion moves them.
  if (lens.aberration > 0.0) {
    let centre = size * 0.5;
    let away = position - centre;
    let red = imageBilinear(centre + away * (1.0 - lens.aberration));
    let blue = imageBilinear(centre + away * (1.0 + lens.aberration));

    color = vec3<f32>(red.r, color.g, blue.b);
  }

  if (lens.intensity > 0.0) {
    let halo = tentHalo(position * vec2<f32>(lens.haloSize) / size);

    color = mix(color, halo, lens.intensity);
  }

  if (lens.glare > 0.0) {
    color = mix(color, streakBilinear(position * vec2<f32>(lens.streakSize) / size), lens.glare);
  }

  let balanced = vec3<f32>(dot(lens.white0.xyz, color), dot(lens.white1.xyz, color), dot(lens.white2.xyz, color));

  graded[id.y * lens.resolution.x + id.x] = vec4<f32>(max(balanced, vec3<f32>(0.0)), 1.0);
}
`;
