/**
 * Reading a texel by hand, because core WebGPU offers no other way here: there are no binding arrays, and one
 * texture_2d_array would force a single size and format on a scene that has neither. The filtering, the wrap modes and
 * the sRGB decode follow texture.js in the CPU renderer so the two agree on what a texture says.
 *
 * Only the top level is read. A rasterizer needs mip levels to stop minified texture detail from aliasing, but this
 * renderer already draws many jittered samples through every pixel, which averages that detail properly instead of
 * approximating it. Levels would buy memory locality, not correctness, so they can wait.
 */
export const TEXTURE_WGSL = /* wgsl */ `
const TEXTURE_FLOAT: u32 = 1u;
const TEXTURE_SRGB: u32 = 2u;
const TEXTURE_NEAREST: u32 = 4u;
const TEXTURE_FLIP_Y: u32 = 8u;

// Three's own wrap constants. Anything else clamps, as it does in the CPU renderer.
const WRAP_REPEAT: u32 = 1000u;
const WRAP_MIRROR: u32 = 1002u;

/** Brings a texel index inside the texture the way the wrap mode says to. */
fn wrapIndex(index: i32, size: i32, mode: u32) -> i32 {
  if (mode == WRAP_REPEAT) {
    return ((index % size) + size) % size;
  }

  if (mode == WRAP_MIRROR) {
    let period = size * 2;
    let folded = ((index % period) + period) % period;

    return select(period - 1 - folded, folded, folded < size);
  }

  return clamp(index, 0, size - 1);
}

fn srgbChannelToLinear(value: f32) -> f32 {
  return select(pow((value + 0.055) / 1.055, 2.4), value / 12.92, value <= 0.04045);
}

/** One texel, in linear light. Alpha is never encoded, as in Three. */
fn readTexel(info: TextureInfo, x: i32, y: i32) -> vec4<f32> {
  let index = u32(y) * info.size.x + u32(x);

  if ((info.flags & TEXTURE_FLOAT) != 0u) {
    let at = info.offset + index * 4u;

    return vec4<f32>(
      bitcast<f32>(textures[at]),
      bitcast<f32>(textures[at + 1u]),
      bitcast<f32>(textures[at + 2u]),
      bitcast<f32>(textures[at + 3u]));
  }

  let word = textures[info.offset + index];
  let raw = vec4<f32>(
    f32(word & 0xffu),
    f32((word >> 8u) & 0xffu),
    f32((word >> 16u) & 0xffu),
    f32((word >> 24u) & 0xffu)) / 255.0;

  if ((info.flags & TEXTURE_SRGB) != 0u) {
    return vec4<f32>(
      srgbChannelToLinear(raw.r), srgbChannelToLinear(raw.g), srgbChannelToLinear(raw.b), raw.a);
  }

  return raw;
}

/** One texture of the table, sampled with its own transform, wrap modes and filter. */
fn sampleTextureAt(index: u32, uv: vec2<f32>) -> vec4<f32> {
  let info = textureInfo(index);
  // The affine transform Three keeps on the texture, one row at a time.
  let transformed = vec2<f32>(
    dot(info.rowU.xyz, vec3<f32>(uv, 1.0)),
    dot(info.rowV.xyz, vec3<f32>(uv, 1.0)));

  return sampleTextureInfo(info, transformed);
}

/**
 * A texture read the way GLSL's texture2D reads it: at the coordinates given, with no transform. Three applies a
 * texture's matrix in its own materials' shaders, so a shader of the scene's, an environment lookup and a light's
 * projected map do not get it.
 */
fn sampleTextureRaw(index: u32, uv: vec2<f32>) -> vec4<f32> {
  return sampleTextureInfo(textureInfo(index), uv);
}

/** Three's equirectUv: longitude across, and latitude up from the bottom of the image, as GL reads it. */
fn equirectUv(direction: vec3<f32>) -> vec2<f32> {
  let d = normalize(direction);

  return vec2<f32>(atan2(d.z, d.x) / (2.0 * PI) + 0.5, asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5);
}

/** An environment map of the scene's: an equirectangular image, or a cube map the snapshot already unwrapped into one. */
fn sampleEquirect(index: u32, direction: vec3<f32>) -> vec4<f32> {
  return sampleTextureRaw(index, equirectUv(direction));
}

fn sampleTextureInfo(info: TextureInfo, transformed: vec2<f32>) -> vec4<f32> {
  if (!all(transformed == transformed) || info.size.x == 0u || info.size.y == 0u) {
    return vec4<f32>(0.0);
  }

  let flipped = select(transformed.y, 1.0 - transformed.y, (info.flags & TEXTURE_FLIP_Y) != 0u);
  let size = vec2<i32>(info.size);
  // The half texel offset that puts a sample in the middle of a texel, as on a graphics pipeline.
  let centred = vec2<f32>(transformed.x, flipped) * vec2<f32>(info.size) - 0.5;

  if ((info.flags & TEXTURE_NEAREST) != 0u) {
    let at = vec2<i32>(floor(centred + 0.5));

    return readTexel(info, wrapIndex(at.x, size.x, info.wrapS), wrapIndex(at.y, size.y, info.wrapT));
  }

  let low = floor(centred);
  let fraction = centred - low;
  let base = vec2<i32>(low);
  let x0 = wrapIndex(base.x, size.x, info.wrapS);
  let x1 = wrapIndex(base.x + 1, size.x, info.wrapS);
  let y0 = wrapIndex(base.y, size.y, info.wrapT);
  let y1 = wrapIndex(base.y + 1, size.y, info.wrapT);

  // The four corners are blended in linear light, which is why the decode happens per texel above.
  return mix(
    mix(readTexel(info, x0, y0), readTexel(info, x1, y0), fraction.x),
    mix(readTexel(info, x0, y1), readTexel(info, x1, y1), fraction.x),
    fraction.y);
}
`;
