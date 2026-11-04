/**
 * What a translated shader reads from the kernel: its inputs at a hit, its texture calls and its varyings. The
 * programs themselves are generated per scene by shaderProgram.ts and come right after this.
 */
export const SHADER_SUPPORT_WGSL = /* wgsl */ `
/** Everything a fragment shader can ask about the point it runs at. */
struct ShaderInput {
  // Where the material's uniform values start in the texture buffer.
  data: u32,
  material: u32,
  triangle: u32,
  frontFacing: bool,
  bary: vec2<f32>,
  // The barycentrics the pixel to the right and the pixel above would have found on the same plane.
  baryDx: vec2<f32>,
  baryDy: vec2<f32>,
  // GL window coordinates: x right, y up from the bottom, z the depth, w the reciprocal of clip w.
  fragCoord: vec4<f32>,
  pointCoord: vec2<f32>,
  // A sky's view direction.
  direction: vec3<f32>,
  // A built-in material's state before lighting, for its hooks, in Three's view space.
  diffuseColor: vec4<f32>,
  normal: vec3<f32>,
  roughness: f32,
  metalness: f32,
  emissive: vec3<f32>,
  viewPosition: vec3<f32>,
  geometryNormal: vec3<f32>,
  uv: vec2<f32>,
  // What gl_FragColor holds when a hook after <opaque_fragment> starts.
  fragColor: vec4<f32>,
};

struct ShaderOutput {
  color: vec4<f32>,
  discarded: bool,
  diffuseColor: vec4<f32>,
  normal: vec3<f32>,
  roughness: f32,
  metalness: f32,
  emissive: vec3<f32>,
};

/** The output of a shader that changed nothing, which is what a hook that does not touch a value leaves. */
fn noShaderOutput(input: ShaderInput) -> ShaderOutput {
  var output: ShaderOutput;
  output.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  output.discarded = false;
  output.diffuseColor = input.diffuseColor;
  output.normal = input.normal;
  output.roughness = input.roughness;
  output.metalness = input.metalness;
  output.emissive = input.emissive;

  return output;
}

/** texture2D: an unbound sampler reads black, as WebGL's does. */
fn shaderSample2D(index: u32, uv: vec2<f32>) -> vec4<f32> {
  if (index == NO_TEXTURE) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  return sampleTextureRaw(index, uv);
}

/** textureCube: the snapshot hands a cube map over unwrapped, mirrored the way Three samples a cube texture. */
fn shaderSampleCube(index: u32, direction: vec3<f32>) -> vec4<f32> {
  if (index == NO_TEXTURE) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  return sampleEquirect(index, direction);
}

fn shaderTexelFetch(index: u32, texel: vec2<i32>) -> vec4<f32> {
  if (index == NO_TEXTURE) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  let info = textureInfo(index);
  let size = vec2<i32>(info.size);
  let at = clamp(texel, vec2<i32>(0), max(size - vec2<i32>(1), vec2<i32>(0)));
  // texelFetch reads rows as GL stores them; a texture flipped on upload is flipped back here.
  let row = select(at.y, size.y - 1 - at.y, (info.flags & 8u) != 0u);

  return readTexel(info, at.x, row);
}

fn shaderTextureSize(index: u32) -> vec2<i32> {
  if (index == NO_TEXTURE) {
    return vec2<i32>(0);
  }

  return vec2<i32>(textureInfo(index).size);
}

/** Where a triangle's varyings start (x) and how many floats one corner has (y). */
fn shaderVaryingAddress(material: u32, triangle: u32) -> vec2<u32> {
  let stride = materials[material].varyingStride;
  let local = triangle - materials[material].firstTriangle;

  return vec2<u32>(frame.varyingOffset + materials[material].varyingBase + local * 3u * stride, stride);
}

/** One component of a varying at a point: interpolated, or the last vertex's for a flat one, as GL takes it. */
fn shaderVarying(at: vec2<u32>, component: u32, bary: vec2<f32>, flat: bool) -> f32 {
  let v0 = attributes[at.x + component];
  let v1 = attributes[at.x + at.y + component];
  let v2 = attributes[at.x + 2u * at.y + component];

  if (flat) {
    return v2;
  }

  return v0 * (1.0 - bary.x - bary.y) + v1 * bary.x + v2 * bary.y;
}
`;
