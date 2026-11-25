/**
 * A layer of cumulus as a participating medium: a camera ray that leaves the scene crosses a shell of the planet
 * between two altitudes and is marched through it. Each step inside a cloud measures the sun through the cloud that
 * stands in its way, and the many scatterings that bring light to the inside are summed as octaves, each one reaching
 * deeper, weaker and less forward. A random walk converges on the same light far more slowly, and what it leaves as
 * noise the denoiser turns into blur.
 *
 * The density is the one a procedural sky's own shader paints clouds with, written by hand here: a translated shader
 * reloads its uniforms on every call, and a ray asks for the density hundreds of times. The two have to stay the same
 * function, or the clouds of a render stand somewhere else than those of the editor that framed them.
 *
 * Lengths are kilometres here and scene units are metres. Only the camera's own ray enters the layer: bounce rays
 * keep reading the baked sky, as they did, so nothing changes in how the scene is lit.
 */
export const CLOUDS_WGSL = /* wgsl */`
const CLOUD_PLANET_RADIUS: f32 = 6371.0;
// What the densest cloud takes per kilometre.
const CLOUD_EXTINCTION: f32 = 40.0;
// The longest stretch a ray is followed inside the layer, near the horizon.
const CLOUD_MAX_SPAN: f32 = 14.0;
// Farther than this the haze has hidden everything.
const CLOUD_MAX_DISTANCE: f32 = 90.0;
const CLOUD_FREQUENCY: f32 = 0.5;
// The same for the large billows, which cut the outline of a cloud.
const CLOUD_BROAD_FREQUENCY: f32 = 4.0;
// The largest puff of the surface, in kilometres, and how many generations grow on it. Only large ones: a small sphere
// reads as a bead.
const CLOUD_PUFF_SIZE: f32 = 0.2;
const CLOUD_PUFF_OCTAVES: u32 = 2u;
// Frequency and strength of the erosion, which makes all the fine detail.
const CLOUD_ERODE_FREQUENCY: f32 = 8.0;
const CLOUD_ERODE_DEPTH: f32 = 0.048;
// How much a fold covered by its neighbour darkens.
const CLOUD_RELIEF_SHADE: f32 = 0.6;
// How far the puffs rise inside a cloud per hour, like the currents of a cumulus: the surface boils even when the cloud
// stands still.
const CLOUD_BOIL_RISE: f32 = 6.0;
// Cells of the cloud grid per drawing of the map: each holds one cumulus at most.
const CLOUD_GRID: f32 = 1.6;
// How much longer a cumulus is along the wind than across it.
const CLOUD_STRETCH: f32 = 1.6;
const CLOUD_BILLOW_OCTAVES: u32 = 7u;
// The size of a pixel one kilometre away: relief smaller than that is left out.
const CLOUD_PIXEL: f32 = 0.0006;
// Steps the camera ray takes across the layer.
const CLOUD_VIEW_STEPS: u32 = 72u;
// Steps inside a cloud that get lit. With the view steps, this bounds what a pixel costs: a band of clouds that
// outlasts the watchdog of the system takes the whole device down with it.
const CLOUD_LIT_STEPS: u32 = 24u;
// Stretches the ray towards the sun is split into, each 2.6 times as long as the one before.
const CLOUD_SUN_STEPS: u32 = 6u;
const CLOUD_SUN_STEP: f32 = 0.012;
// Orders of scattering summed, and what each one keeps of the energy and of the depth of the one before.
const CLOUD_SCATTER_ORDERS: u32 = 5u;
const CLOUD_ORDER_ENERGY: f32 = 0.75;
const CLOUD_ORDER_DEPTH: f32 = 0.28;
const CLOUD_SUN_GAIN: f32 = 1.79;
// How much of the sky's light the base of a cloud still gets.
const CLOUD_BASE_AMBIENT: f32 = 0.82;
// How much of the brightest light a cloud takes at night.
const CLOUD_MOONLIT: f32 = 0.042;
// Steps of a sun ray through the layer, what a thick cloud still lets through, and how much scattering lightens it.
const CLOUD_SHADOW_STEPS: u32 = 6u;
const CLOUD_SHADOW_FLOOR: f32 = 0.06;
const CLOUD_SHADOW_SCATTER: f32 = 0.3;

/** What the layer adds to a camera ray, and how much of what is behind it still shows. */
struct CloudLayer {
  radiance: vec3<f32>,
  through: f32,
  // How far along the ray the clouds are, in scene units.
  distance: f32,
  // Which way the visible face of the cloud looks: the denoiser keeps the relief it can see in this.
  faceNormal: vec3<f32>,
};

fn cloudHash2(p: vec2<f32>) -> f32 {
  var q = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
  q = q + dot(q, q.yzx + 33.33);

  return fract((q.x + q.y) * q.z);
}

fn cloudHash3(p: vec3<f32>) -> f32 {
  var q = fract(p * 0.1031);
  q = q + dot(q, q.zyx + 31.32);

  return fract((q.x + q.y) * q.z);
}

fn cloudNoise(p: vec2<f32>) -> f32 {
  let cell = floor(p);
  let f = p - cell;
  let s = f * f * (3.0 - 2.0 * f);
  let a = cloudHash2(cell);
  let b = cloudHash2(cell + vec2<f32>(1.0, 0.0));
  let c = cloudHash2(cell + vec2<f32>(0.0, 1.0));
  let d = cloudHash2(cell + vec2<f32>(1.0, 1.0));

  return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

/** Relief of a cloud, from 0 in the creases to 1 on top of the billows: folded waves, turned and shrunk per octave. */
fn cloudBillows(start: vec3<f32>, levels: f32) -> f32 {
  var p = start;
  var sum = 0.0;
  var total = 0.0;
  var amplitude = 0.5;

  for (var i = 0u; i < CLOUD_BILLOW_OCTAVES; i++) {
    let weight = amplitude * clamp(levels - f32(i), 0.0, 1.0);

    if (weight <= 0.0) {
      break;
    }

    let wave = dot(cos(p), sin(p.yzx));

    sum = sum + weight * (sqrt(wave * wave + 0.03) - 0.17);
    total = total + weight;
    p = vec3<f32>(0.8 * p.y + 0.6 * p.z, -0.8 * p.x + 0.36 * p.y - 0.48 * p.z, -0.6 * p.x - 0.48 * p.y + 0.64 * p.z) * 2.07
      + vec3<f32>(1.7, 4.1, 2.9);
    amplitude = amplitude * 0.52;
  }

  return clamp(sum / max(total, 1e-4) * 0.85, 0.0, 1.0);
}

fn cloudRemap(value: f32, low: f32, high: f32) -> f32 {
  return clamp((value - low) / max(high - low, 1e-4), 0.0, 1.0);
}

fn cloudOctave(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(1.6 * p.x + 1.2 * p.y, -1.2 * p.x + 1.6 * p.y) + vec2<f32>(3.1, 1.7);
}

fn cloudSmoothMax(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;

  return max(a, b) + h * h * k * 0.25;
}

fn cloudSmoothMin(a: f32, b: f32, k: f32) -> f32 {
  return -cloudSmoothMax(-a, -b, k);
}

/**
 * Distance, in kilometres, to the body of the nearest cloud. The clouds are placed one by one, as the convection cells
 * that make them: each cell of the map draws whether it has one, where and how large, and each cloud is a handful of
 * ellipsoids resting on the base. A broad noise leaves full regions and clearings, another bends the outline.
 */
/**
 * Whether a cell of the map holds a cloud, as the number it drew, or 2 when it holds none. Broad regions fill up and
 * others clear.
 */
/** How full a region of the map is: broad areas fill up and others stay open, which is what groups clouds. */
fn cloudRegion(at: vec2<f32>) -> f32 {
  return 0.62 * cloudNoise(at * 0.05 + vec2<f32>(3.7, 8.1)) + 0.38 * cloudNoise(at * 0.16 + vec2<f32>(11.0, 5.0));
}

// Where the observer is in the cloud pattern: the seed picks the pattern and the wind drags it.
fn cloudOrigin() -> vec2<f32> {
  return vec2<f32>(frame.cloudShape.w * 17.0, frame.cloudShape.w * 9.0) - frame.cloudSun.zw / frame.cloudShape.z;
}

// The draw does not look at the distance to the observer: with wind it changes, and a cloud would vanish between frames.
fn cloudDraw(at: vec2<f32>) -> f32 {
  let chance = clamp(frame.cloudShape.x * 1.6 + (cloudRegion(at) - 0.5) * 3.2, 0.0, 1.0);
  let draw = cloudHash2(at + 71.7);

  return select(2.0, draw, draw < chance);
}

// A cumulus lives from about 15 minutes to almost an hour, longer the larger it is: it is born small and flat, grows, and
// dissolves. Returns how far it has grown and which life it is in, and each life has another shape. The moment of birth
// is drawn, so they are not born together.
fn cloudLife(at: vec2<f32>) -> vec2<f32> {
  let size = cloudHash2(at + 41.3);
  let period = mix(0.25, 0.9, size * size) * (0.85 + 0.3 * cloudHash2(at + 23.9));
  let age = frame.cloudClock.x / period + cloudHash2(at + 61.3);
  let cycle = floor(age);
  let life = age - cycle;

  return vec2<f32>(smoothstep(0.0, 0.3, life) * (1.0 - smoothstep(0.75, 1.0, life)), cycle);
}

// Fewer clouds far from the observer, or the horizon closes into a carpet. Each one shrinks gradually with distance
// instead of vanishing, which keeps the wind continuous.
fn cloudFade(at: vec2<f32>) -> f32 {
  let cellSpot = (at + 0.5) / CLOUD_GRID;
  let spot = vec2<f32>(0.8 * cellSpot.x - 0.6 * cellSpot.y, 0.6 * cellSpot.x + 0.8 * cellSpot.y);
  let remote = length(spot - cloudOrigin() * CLOUD_FREQUENCY) / CLOUD_FREQUENCY * frame.cloudShape.z;
  let keep = 1.0 - 0.88 * smoothstep(7.0, 32.0, remote);
  let pick = cloudHash2(at + 57.1);

  return smoothstep(pick - 0.12, pick, keep);
}

fn cloudBody(p: vec2<f32>, lift: f32) -> f32 {
  let coverage = frame.cloudShape.x;
  let bent = p + 0.7 * (vec2<f32>(cloudNoise(p * 1.7 + 3.0), cloudNoise(p * 1.7 + vec2<f32>(9.2, 4.1))) - 0.5)
    + 0.35 * (vec2<f32>(cloudNoise(p * 5.3 + 1.7), cloudNoise(p * 5.3 + vec2<f32>(6.1, 2.9))) - 0.5);
  // The grid is turned so its rows line up with nothing.
  let grid = vec2<f32>(0.8 * bent.x + 0.6 * bent.y, -0.6 * bent.x + 0.8 * bent.y) * CLOUD_GRID;
  let cell = floor(grid);
  let f = grid - cell;
  let up = lift * CLOUD_FREQUENCY * CLOUD_GRID;
  // Coverage brings more clouds, not larger ones: only a very loaded sky fuses them.
  let coverage2 = coverage * coverage;
  let reach = 0.95 + 0.9 * coverage2 * coverage2;
  // The tallest lobe fits the layer, or its ceiling would cut the cloud straight.
  let ceiling = 0.75 * frame.cloudLayer.y / frame.cloudShape.z * CLOUD_FREQUENCY * CLOUD_GRID;
  var best = 1.0;

  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let offset = vec2<f32>(f32(x), f32(y));
      let at = cell + offset;

      // Drawn per cloud, not per point: the line of the draw would otherwise cut a cloud in two.
      let draw = cloudDraw(at);
      let life = cloudLife(at);
      let grown = cloudFade(at) * life.x;

      // A cloud gives way to a neighbour that drew lower: touching clouds line up into rows and false necks.
      if (draw < 2.0 && draw <= cloudDraw(at + vec2<f32>(1.0, 0.0)) && draw <= cloudDraw(at - vec2<f32>(1.0, 0.0))
        && draw <= cloudDraw(at + vec2<f32>(0.0, 1.0)) && draw <= cloudDraw(at - vec2<f32>(0.0, 1.0))
        && grown > 0.0) {
        // Each life of the cloud has its own place and lobes; the swap happens while it is gone.
        let born = at + life.y * vec2<f32>(17.31, 9.73);
        let drawn = cloudHash2(at + 41.3);
        let size = min(drawn * drawn * (0.45 + 1.3 * cloudRegion(at)), 1.0);
        let center = offset + 0.1 + 0.8 * vec2<f32>(cloudHash2(born), cloudHash2(born + 19.7)) - f;
        // A little larger at its peak: over its life it covers the same sky as before.
        let radius = min((0.14 + 0.28 * size) * reach * 1.15, 0.4) * grown;

        // A cloud far from this point changes nothing, and that is most of the reads.
        if (length(center) > radius * 2.4 + 0.4) {
          continue;
        }

        for (var lobe = 0; lobe < 6; lobe++) {
          let key = born + f32(lobe) * vec2<f32>(7.7, 9.1);
          let away = center + (vec2<f32>(cloudHash2(key + 3.1), cloudHash2(key + 5.3)) - 0.5) * radius * 1.15
            * vec2<f32>(CLOUD_STRETCH, 1.0);
          let lobeRadius = radius * (0.6 + 0.35 * cloudHash2(key + 8.9));
          // A small cloud is shallow, a large one rises; while it is being born it has not risen yet.
          let tall = min((0.55 + 0.4 * size) * (0.7 + 0.6 * cloudHash2(key + 2.3)) * (0.35 + 0.65 * life.x),
            ceiling / (lobeRadius * 1.65));
          // A dome on a column that narrows a little to the base: the lower half of a sphere would leave a lip under
          // the cloud for every lobe.
          let above = up / tall;
          let shaft = lobeRadius * (0.85 + 0.15 * smoothstep(0.0, 1.0, above / (0.65 * lobeRadius)));
          let column = vec3<f32>(away.x / CLOUD_STRETCH, max(above - 0.65 * lobeRadius, 0.0), away.y);
          let gap = (length(column) - shaft) * min(tall, 1.0);

          best = cloudSmoothMin(best, gap, 0.1);
        }
      }
    }
  }

  var d = best / (CLOUD_FREQUENCY * CLOUD_GRID);
  // At the end of the range the sky closes for good, cloud drawn or not.
  let closed = smoothstep(0.75, 1.0, coverage);

  if (closed > 0.0) {
    let deck = frame.cloudLayer.y / frame.cloudShape.z * closed * (0.6 + 0.4 * cloudNoise(p * 2.0));

    d = min(d, (lift - deck) * 0.7);
  }

  return d;
}

/** Height of a point above the ground, with the planet's curvature. */
fn cloudAltitude(p: vec3<f32>) -> f32 {
  return p.y + dot(p.xz, p.xz) * (0.5 / CLOUD_PLANET_RADIUS);
}

/** How far along a ray rising with sine \`rise\` the altitude is reached. */
fn cloudShellDistance(rise: f32, altitude: f32) -> f32 {
  let b = CLOUD_PLANET_RADIUS * rise;
  let c = (2.0 * CLOUD_PLANET_RADIUS + altitude) * altitude;

  return c / (b + sqrt(b * b + c));
}

/** Distance to the nearest sphere of a grid, each of its own size. */
fn cloudPuffs(p: vec3<f32>) -> f32 {
  let cell = floor(p);
  let f = p - cell;
  var best = 1000.0;

  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let corner = vec3<f32>(f32(x), f32(y), f32(z));

        best = min(best, length(f - corner) - (0.25 + 0.4 * cloudHash3(cell + corner + 0.37)));
      }
    }
  }

  return best;
}

/**
 * Density at a point of the layer, from 0 to 1. The map gives the body of the cloud as a distance to its surface, and
 * each generation of spheres grows on the one before: the cauliflower, running down the sides to a flat base. \`levels\`
 * is how much relief goes in and \`range\` the distance the point is seen from: far away the rim softens, as what is
 * smaller than a pixel would otherwise turn to grain.
 */
fn cloudBaseLumps(q: vec3<f32>, levels: f32) -> f32 {
  return cloudBillows(q * CLOUD_ERODE_FREQUENCY * 0.7 + vec3<f32>(2.3, 0.7, 5.1), min(levels, 4.0));
}

fn cloudDensity(p: vec3<f32>, levels: f32, range: f32) -> f32 {
  return cloudDensityOver(p, levels, range, 0.0);
}

/** Density at a point of the layer. blur is the stretch of ray a reading stands for: it counts how much of it is cloud. */
fn cloudDensityOver(p: vec3<f32>, levels: f32, range: f32, blur: f32) -> f32 {
  let altitude = cloudAltitude(p);
  let rise = altitude - frame.cloudLayer.x;

  if (rise <= 0.0 || rise >= frame.cloudLayer.y) {
    return 0.0;
  }

  let q = vec3<f32>(p.x, altitude, p.z) / frame.cloudShape.z + vec3<f32>(cloudOrigin().x, 0.0, cloudOrigin().y);
  let lift = rise / frame.cloudShape.z;
  var d = cloudBody(q.xz * CLOUD_FREQUENCY, lift);
  var size = CLOUD_PUFF_SIZE;

  if (d > size * 2.2) {
    return 0.0;
  }

  // The puffs rise with time, the small ones faster: it is the boiling of a cumulus.
  let rising = vec3<f32>(0.0, frame.cloudClock.x * CLOUD_BOIL_RISE, 0.0);
  // Large billows push the surface in and out.
  let broadSpot = q - rising * 0.5;
  let broadPoint = broadSpot * CLOUD_BROAD_FREQUENCY + 0.6 * sin(broadSpot.zxy * 1.3 + vec3<f32>(1.2, 3.4, 0.7));

  d = d + (0.45 - cloudBillows(broadPoint, 2.0)) * 0.1 * smoothstep(0.0, 0.25, lift);

  if (d > size * 1.5) {
    return 0.0;
  }

  let octaves = min(levels - 1.0, f32(CLOUD_PUFF_OCTAVES));
  var x = (q - rising) / size + vec3<f32>(1.7, 0.4, 2.9);

  for (var i = 0u; i < CLOUD_PUFF_OCTAVES; i++) {
    let fade = clamp(octaves - f32(i), 0.0, 1.0);

    // Away from the surface the smaller puffs change nothing.
    if (fade <= 0.0 || abs(d) > size * 1.5) {
      break;
    }

    // The point is bent and the union very smooth: a clean sphere would read as a bead.
    let bent = x + 0.6 * sin(x.yzx * vec3<f32>(1.9, 1.7, 2.3) + vec3<f32>(1.3, 2.1, 0.4));
    let puff = cloudSmoothMax(size * cloudPuffs(bent), d - 0.25 * size, 0.45 * size);

    d = mix(d, cloudSmoothMin(puff, d, 0.4 * size), fade);
    x = vec3<f32>(0.8 * x.y + 0.6 * x.z, -0.8 * x.x + 0.36 * x.y - 0.48 * x.z, -0.6 * x.x - 0.48 * x.y + 0.64 * x.z) * 2.03
      + vec3<f32>(1.7, 4.1, 2.9);
    size = size * 0.5;
  }

  // Continuous erosion cuts the surface into irregular folds and wisps.
  if (abs(d) < CLOUD_ERODE_DEPTH * 2.5) {
    var carved = (q - rising * 1.5) * CLOUD_ERODE_FREQUENCY;

    carved = carved + 1.1 * sin(carved.zxy * vec3<f32>(0.37, 0.41, 0.33) + vec3<f32>(1.3, 2.2, 0.6));
    d = d + CLOUD_ERODE_DEPTH * 2.0 * (0.48 - cloudBillows(carved, min(levels, 3.0 + 4.0 * frame.cloudLayer.z))) * (1.0 - smoothstep(12.0, 48.0, range));
  }

  // The base is a plane: condensation starts at the same height across the whole cloud.
  var floorHeight = 0.02 + 0.1 * cloudNoise(q.xz * 1.3 + vec2<f32>(5.1, 2.7)) + 0.06 * cloudNoise(q.xz * 4.1 + vec2<f32>(1.9, 7.3))
    + 0.03 * cloudNoise(q.xz * 11.0 + vec2<f32>(8.2, 3.6));

  // The pockets of the base, seen from below.
  if (abs(lift - floorHeight) < 0.1) {
    floorHeight = floorHeight + 0.075 * (cloudBaseLumps(q, levels) - 0.4) * (1.0 - smoothstep(12.0, 48.0, range));
  }

  d = cloudSmoothMax(d, floorHeight - lift, 0.05);
  d = cloudSmoothMax(d, lift + 0.04 - frame.cloudLayer.y / frame.cloudShape.z, 0.1);

  let density = clamp((0.5 * blur - d) / max(0.008 + 0.0018 * range, blur), 0.0, 1.0);

  // Condensation sets in gradually: the base is diffuse, with no marked line.
  return density * density * (3.0 - 2.0 * density) * smoothstep(0.0, 0.07, lift - floorHeight);
}

/** Henyey-Greenstein: how much of the light arriving from one direction leaves in another. */
fn cloudPhase(cosine: f32, g: f32) -> f32 {
  return (1.0 - g * g) / (4.0 * PI * pow(max(1.0 + g * g - 2.0 * g * cosine, 1e-4), 1.5));
}

/** How much cloud stands between a point and the sun, as optical depth. */
fn cloudSunDepth(point: vec3<f32>, towards: vec3<f32>, levels: f32, range: f32, extinction: f32, jitter: f32) -> f32 {
  var depth = 0.0;
  var reach = CLOUD_SUN_STEP;
  var along = 0.0;

  for (var j = 0u; j < CLOUD_SUN_STEPS; j++) {
    // Close to the point the relief is what shades one billow on the next; farther on, only the body counts.
    let relief = select(3.0, min(levels, 6.5), j < 3u);

    depth = depth + cloudDensity(point + towards * (along + reach * jitter), relief, range) * reach;
    along = along + reach;
    reach = reach * 2.6;
  }

  return depth * extinction;
}

/** How much light reaches this point of the face: a fold covered from above and from the sun is darker than a bulge. */
fn cloudRelief(point: vec3<f32>, towards: vec3<f32>, levels: f32, range: f32) -> f32 {
  let reach = max(0.012, range * 0.004);
  let covered = 0.5 * (cloudDensity(point + vec3<f32>(0.0, reach, 0.0), levels, range)
    + cloudDensity(point + towards * reach * 1.6, levels, range));

  let altitude = cloudAltitude(point);
  let q = vec3<f32>(point.x, altitude, point.z) / frame.cloudShape.z + vec3<f32>(cloudOrigin().x, 0.0, cloudOrigin().y);
  let base = 1.0 - smoothstep(0.03, 0.18, (altitude - frame.cloudLayer.x) / frame.cloudLayer.y);

  return mix(1.0, CLOUD_RELIEF_SHADE, covered) * mix(1.0, 0.72 + 0.6 * cloudBaseLumps(q, levels), base);
}

/** The sun scattered towards the camera at that optical depth, every order of scattering summed. */
fn cloudSunScatter(depth: f32, cosine: f32) -> f32 {
  var sum = 0.0;
  var energy = 1.0;
  var reach = 1.0;
  var forward = 1.0;

  for (var order = 0u; order < CLOUD_SCATTER_ORDERS; order++) {
    sum = sum + energy * exp(-depth * reach)
      * min(0.65 * cloudPhase(cosine, 0.8 * forward) + 0.35 * cloudPhase(cosine, -0.15 * forward), 0.3);
    energy = energy * CLOUD_ORDER_ENERGY;
    reach = reach * CLOUD_ORDER_DEPTH;
    forward = forward * 0.6;
  }

  return sum * CLOUD_SUN_GAIN;
}

/** The outward normal of the cloud at a point, from where its density falls. Facing the camera if it has no slope. */
fn cloudSurfaceNormal(point: vec3<f32>, direction: vec3<f32>, range: f32, levels: f32) -> vec3<f32> {
  let e = max(0.003, range * CLOUD_PIXEL * 2.0);
  let here = cloudDensity(point, levels, range);
  let slope = vec3<f32>(
    here - cloudDensity(point + vec3<f32>(e, 0.0, 0.0), levels, range),
    here - cloudDensity(point + vec3<f32>(0.0, e, 0.0), levels, range),
    here - cloudDensity(point + vec3<f32>(0.0, 0.0, e), levels, range));
  let size = length(slope);

  if (size < 1e-5) {
    return -direction;
  }

  let outward = slope / size;

  // A face the camera could not see is the noise of the slope: it is laid back onto the visible side.
  return normalize(outward - direction * max(0.0, dot(outward, direction)) * 1.01 - direction * 0.05);
}

/** How much of the sun gets through the layer to a point in metres. */
fn cloudSunThrough(position: vec3<f32>, sun: vec3<f32>, jitter: f32) -> f32 {
  let point = position * 0.001;
  let bottom = frame.cloudLayer.x - point.y;

  if (sun.y <= 0.02 || bottom <= 0.0) {
    return 1.0;
  }

  let enter = cloudShellDistance(sun.y, bottom);

  if (enter >= CLOUD_MAX_DISTANCE) {
    return 1.0;
  }

  let span = min(cloudShellDistance(sun.y, bottom + frame.cloudLayer.y) - enter, CLOUD_MAX_SPAN);
  let stepLength = span / f32(CLOUD_SHADOW_STEPS);
  let blur = min(stepLength / frame.cloudShape.z, CLOUD_PUFF_SIZE * 2.0);
  var depth = 0.0;

  for (var i = 0u; i < CLOUD_SHADOW_STEPS; i++) {
    depth = depth + cloudDensityOver(point + sun * (enter + stepLength * (f32(i) + jitter)), 3.0, enter, blur);
  }

  depth = depth * stepLength * CLOUD_EXTINCTION * frame.cloudShape.y;

  return mix(1.0, mix(CLOUD_SHADOW_FLOOR, 1.0, exp(-depth * CLOUD_SHADOW_SCATTER)), frame.cloudSun.y);
}

/** The layer along a camera ray that left the scene. */
fn cloudScatter(rayOrigin: vec3<f32>, direction: vec3<f32>, rng: ptr<function, PathSampler>) -> CloudLayer {
  var result: CloudLayer;

  result.radiance = vec3<f32>(0.0);
  result.through = 1.0;
  result.distance = 1e6;
  result.faceNormal = -direction;

  let origin = rayOrigin * 0.001;
  let bottom = frame.cloudLayer.x - origin.y;

  // From inside or above the layer nothing is drawn: the camera of a building stands under it.
  if (direction.y <= 0.0 || bottom <= 0.0) {
    return result;
  }

  let enter = cloudShellDistance(direction.y, bottom);

  if (enter >= CLOUD_MAX_DISTANCE) {
    return result;
  }

  let span = min(cloudShellDistance(direction.y, bottom + frame.cloudLayer.y) - enter, CLOUD_MAX_SPAN);
  let extinction = CLOUD_EXTINCTION * frame.cloudShape.y;
  let stepLength = span / f32(CLOUD_VIEW_STEPS);
  let pick = hashSample2D(rng);
  let sunDirection = normalize(frame.atmosphereSun.xyz);
  let sunColor = frame.atmosphereSunColor.rgb * step(0.0, sunDirection.y);
  // Many scatterings wash the tint of the sun out of a cloud, which reads white where the light itself is warm.
  let sunLight = mix(sunColor, vec3<f32>(luminanceOf(sunColor)), mix(0.45, 0.7, smoothstep(0.05, 0.4, sunDirection.y))) * mix(0.72, 1.0, smoothstep(0.03, 0.3, sunDirection.y));
  let cosine = dot(direction, sunDirection);
  let skyAbove = clearSkyRadiance(vec3<f32>(0.0, 1.0, 0.0));
  let skyBelow = clearSkyRadiance(vec3<f32>(0.0, -1.0, 0.0));
  let skyGray = vec3<f32>((skyAbove.r + skyAbove.g + skyAbove.b) / 3.0);
  // With the sun down the brightest light is the moon, which a cloud only just shows.
  let moonlit = mix(CLOUD_MOONLIT, 1.0, frame.cloudSun.x);
  var travelled = enter + stepLength * pick.x;
  var transmittance = 1.0;
  var gathered = vec3<f32>(0.0);
  var relief = 1.0;
  var lit = 0u;
  var first = -1.0;
  // Where the ray has lost half of its light: the face of the cloud as the camera sees it.
  var front = -1.0;
  var frontLevels = 2.0;

  var stepNow = stepLength;
  var inside = false;

  for (var i = 0u; i < CLOUD_VIEW_STEPS && lit < CLOUD_LIT_STEPS; i++) {
    let levels = clamp(log2(0.74 / (CLOUD_PIXEL * 3.0 * travelled)), 2.0, 8.0);
    var density = cloudDensity(origin + direction * travelled, levels, travelled);

    // A long step lands anywhere inside the rim, which blurs the face over the samples: the face is searched for, and
    // the ray goes on in short steps from it, each a little longer than the last.
    if (density > 0.0 && !inside) {
      var low = travelled - stepNow;
      var high = travelled;

      for (var b = 0u; b < 5u; b++) {
        let middle = 0.5 * (low + high);

        if (cloudDensity(origin + direction * middle, levels, middle) > 0.0) {
          high = middle;
        } else {
          low = middle;
        }
      }

      inside = true;
      stepNow = max(0.004, high * 0.001);
      travelled = high + 1e-4 + stepNow * pick.x;
      density = cloudDensity(origin + direction * travelled, levels, travelled);
    }

    if (density > 0.0) {
      lit = lit + 1u;

      if (first < 0.0) {
        first = travelled;
      }

      let point = origin + direction * travelled;

      if (lit == 1u) {
        relief = cloudRelief(point, sunDirection, levels, travelled);
      }

      let depth = cloudSunDepth(point, sunDirection, levels, travelled, extinction, pick.y);
      // A wisp at the rim has little around it to scatter light back out.
      let powder = 0.5 + 0.5 * (1.0 - exp(-density * 8.0));
      let height = clamp((cloudAltitude(point) - frame.cloudLayer.x) / frame.cloudLayer.y, 0.0, 1.0);
      // The base sees the ground, but also the light of the sky scattered through the air around it.
      let ambient = mix(skyBelow, 0.5 * (skyAbove + skyGray), 0.55 + 0.45 * height)
        * (CLOUD_BASE_AMBIENT + (1.0 - CLOUD_BASE_AMBIENT) * height);
      let kept = exp(-density * extinction * stepNow);

      gathered = gathered + transmittance * (1.0 - kept) * relief * (sunLight * moonlit * cloudSunScatter(depth, cosine) * powder + ambient);
      transmittance = transmittance * kept;

      if (front < 0.0 && transmittance < 0.5) {
        front = travelled;
        frontLevels = levels;
      }

      if (transmittance < 0.02) {
        transmittance = 0.0;
        break;
      }

      stepNow = min(stepNow * 1.22, stepLength);
    } else if (inside) {
      // The step grows gradually after a gap, or it would jump over the cloud right behind it.
      stepNow = min(stepNow * 1.6, stepLength);
      inside = stepNow < stepLength;
    }

    travelled = travelled + stepNow;

    if (travelled > enter + span) {
      break;
    }
  }

  if (first < 0.0) {
    return result;
  }

  // Far clouds fade into the haze: what is behind them shows instead.
  let haze = max(1.0 - exp(-first * frame.cloudLayer.w * 4.0), smoothstep(0.45 * CLOUD_MAX_DISTANCE, 0.9 * CLOUD_MAX_DISTANCE, enter));
  let edge = smoothstep(0.0, 0.012, direction.y);
  let sky = clearSkyRadiance(direction);

  result.through = 1.0 - (1.0 - transmittance) * edge;
  result.radiance = (gathered * (1.0 - haze) + sky * (1.0 - transmittance) * haze) * edge;
  result.distance = first * 1000.0;

  // A thin veil has no face to keep, and the normal stays towards the camera.
  if (front > 0.0) {
    result.distance = front * 1000.0;
    result.faceNormal = cloudSurfaceNormal(origin + direction * front, direction, front, min(frontLevels, 7.0));
  }

  return result;
}
`;

/** A scene with no cloud layer keeps the name the kernel calls, and none of the code. */
export const CLOUDS_STUB_WGSL = /* wgsl */`
fn cloudSunThrough(position: vec3<f32>, sun: vec3<f32>, jitter: f32) -> f32 {
  return 1.0;
}

struct CloudLayer {
  radiance: vec3<f32>,
  through: f32,
  distance: f32,
  faceNormal: vec3<f32>,
};

fn cloudScatter(rayOrigin: vec3<f32>, direction: vec3<f32>, rng: ptr<function, PathSampler>) -> CloudLayer {
  var result: CloudLayer;

  result.radiance = vec3<f32>(0.0);
  result.through = 1.0;
  result.distance = 1e6;
  result.faceNormal = -direction;

  return result;
}
`;
