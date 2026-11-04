/**
 * The light that surrounds the scene, and how a shadow ray aims at the bright parts of it.
 *
 * A sun in an equirectangular image is the same problem a small bright lamp is: a cosine weighted bounce almost never
 * finds it, so the image takes thousands of samples to settle. The cumulative distributions built in environment.ts
 * turn two random numbers into a direction drawn by luminance, which the power heuristic then balances against the
 * bounce that might have found it anyway.
 */
export const ENVIRONMENT_WGSL = /* wgsl */ `
struct EnvironmentSample {
  direction: vec3<f32>,
  // Density per solid angle, before the probability of picking the environment at all.
  pdf: f32,
};

/** From world space into the image's own frame, which is what the rotation setting turns. */
fn environmentRotate(direction: vec3<f32>) -> vec3<f32> {
  let c = cos(frame.environmentRotation);
  let s = sin(frame.environmentRotation);

  return vec3<f32>(direction.x * c - direction.z * s, direction.y, direction.x * s + direction.z * c);
}

fn environmentUnrotate(direction: vec3<f32>) -> vec3<f32> {
  let c = cos(frame.environmentRotation);
  let s = sin(frame.environmentRotation);

  return vec3<f32>(direction.x * c + direction.z * s, direction.y, -direction.x * s + direction.z * c);
}

/** Where a direction lands on an equirectangular image: longitude across, colatitude down from the zenith. */
fn environmentUv(direction: vec3<f32>) -> vec2<f32> {
  return vec2<f32>(
    atan2(direction.z, direction.x) / (2.0 * PI) + 0.5,
    acos(clamp(direction.y, -1.0, 1.0)) / PI);
}

/** And back again. */
fn environmentDirection(uv: vec2<f32>) -> vec3<f32> {
  let phi = (uv.x - 0.5) * 2.0 * PI;
  let theta = uv.y * PI;

  return vec3<f32>(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

/** What arrives from a direction: the image if there is one, and the gradient otherwise. */
fn environmentRadiance(direction: vec3<f32>) -> vec3<f32> {
  if (frame.environmentTexture == NO_TEXTURE) {
    let height = pow(clamp(direction.y * 0.5 + 0.5, 0.0, 1.0), frame.environmentTop.w);

    return mix(frame.environmentBottom.rgb, frame.environmentTop.rgb, height) * frame.environmentBottom.w;
  }

  let uv = environmentUv(environmentRotate(direction));

  return sampleTextureAt(frame.environmentTexture, uv).rgb * frame.environmentBottom.w;
}

fn environmentRow(y: u32) -> f32 {
  return bitcast<f32>(textures[frame.environmentCdf + y]);
}

fn environmentColumn(index: u32) -> f32 {
  return bitcast<f32>(textures[frame.environmentCdf + frame.environmentCdfSize.y + index]);
}

fn environmentTexelLuminance(index: u32) -> f32 {
  let base = frame.environmentCdf + frame.environmentCdfSize.y
    + frame.environmentCdfSize.x * frame.environmentCdfSize.y;

  return bitcast<f32>(textures[base + index]);
}

/** The first entry of a cumulative range whose value passes u. */
fn searchCdf(start: u32, count: u32, u: f32, rows: bool) -> u32 {
  var low = 0u;
  var high = count - 1u;

  loop {
    if (low >= high) {
      break;
    }

    let middle = (low + high) / 2u;
    let value = select(environmentColumn(start + middle), environmentRow(middle), rows);

    if (u < value) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }

  return low;
}

/**
 * A direction drawn by the image's own luminance. The remainder of each random number places the sample inside the
 * texel it chose, so two dimensions do the work of four.
 */
fn sampleEnvironmentDirection(u: vec2<f32>) -> EnvironmentSample {
  var result: EnvironmentSample;
  result.pdf = 0.0;
  result.direction = vec3<f32>(0.0, 1.0, 0.0);

  if (frame.environmentTotal <= 0.0) {
    return result;
  }

  let width = frame.environmentCdfSize.x;
  let height = frame.environmentCdfSize.y;
  let y = searchCdf(0u, height, u.x, true);
  let rowStart = select(environmentRow(y - 1u), 0.0, y == 0u);
  let rowSpan = environmentRow(y) - rowStart;
  let x = searchCdf(y * width, width, u.y, false);
  let columnStart = select(environmentColumn(y * width + x - 1u), 0.0, x == 0u);
  let columnSpan = environmentColumn(y * width + x) - columnStart;
  let jitterY = select(0.5, clamp((u.x - rowStart) / rowSpan, 0.0, 0.999999), rowSpan > 0.0);
  let jitterX = select(0.5, clamp((u.y - columnStart) / columnSpan, 0.0, 0.999999), columnSpan > 0.0);
  let uv = vec2<f32>((f32(x) + jitterX) / f32(width), (f32(y) + jitterY) / f32(height));

  result.direction = normalize(environmentUnrotate(environmentDirection(uv)));
  result.pdf = environmentTexelLuminance(y * width + x) * f32(width) * f32(height)
    / (2.0 * PI * PI * frame.environmentTotal);

  return result;
}

/** The density with which that sampling would have produced this direction, for the power heuristic. */
fn environmentPdf(direction: vec3<f32>) -> f32 {
  if (frame.environmentTotal <= 0.0) {
    return 0.0;
  }

  let width = frame.environmentCdfSize.x;
  let height = frame.environmentCdfSize.y;
  let uv = environmentUv(environmentRotate(direction));
  let x = min(u32(max(floor(uv.x * f32(width)), 0.0)), width - 1u);
  let y = min(u32(max(floor(uv.y * f32(height)), 0.0)), height - 1u);

  return environmentTexelLuminance(y * width + x) * f32(width) * f32(height)
    / (2.0 * PI * PI * frame.environmentTotal);
}
`;
