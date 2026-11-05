/**
 * The integrator: camera rays, the bounce loop and the accumulation buffer.
 *
 * Light reaches a path two ways, and the two are combined with multiple importance sampling: a shadow ray towards a
 * light at every vertex, and whatever the sampled bounce direction happens to run into. Each contribution is weighted by
 * the power heuristic, which keeps the pair unbiased while each strategy covers the other's worst case. A small bright
 * emitter is found by the shadow ray; a sharp reflection of a large one is found by the bounce.
 *
 * What only the camera sees is handled on the camera's own line of sight, the part of a path before its first
 * reflection: culled faces, the background, Three's fog, a light map's baked light and the coverage a transparent
 * background turns into alpha. Transmission keeps a path on that line, as Three's transmission shows what is behind.
 */
export const PATH_TRACE_WGSL = /* wgsl */ `
/** Light from an ambient or hemisphere light, which this renderer turns into an equally bright surrounding. */
fn ambientRadiance(direction: vec3<f32>) -> vec3<f32> {
  let latitude = clamp(dot(direction, frame.ambientAxis.xyz) * 0.5 + 0.5, 0.0, 1.0);

  return mix(frame.ambientGround.rgb, frame.ambientSky.rgb, latitude) + probeRadiance(direction);
}

/**
 * A LightProbe's light. Three's coefficients are radiance projected on the SH3 basis, which its shader convolves with
 * the cosine to get irradiance; evaluated directly they are the surrounding that delivers exactly that irradiance.
 */
fn probeRadiance(d: vec3<f32>) -> vec3<f32> {
  let radiance = frame.probe0.rgb * 0.282095
    + frame.probe1.rgb * (0.488603 * d.y)
    + frame.probe2.rgb * (0.488603 * d.z)
    + frame.probe3.rgb * (0.488603 * d.x)
    + frame.probe4.rgb * (1.092548 * d.x * d.y)
    + frame.probe5.rgb * (1.092548 * d.y * d.z)
    + frame.probe6.rgb * (0.315392 * (3.0 * d.z * d.z - 1.0))
    + frame.probe7.rgb * (1.092548 * d.x * d.z)
    + frame.probe8.rgb * (0.546274 * (d.x * d.x - d.y * d.y));

  return max(radiance, vec3<f32>(0.0));
}

/** What the camera sees where nothing is in the way, as the background setting says. */
fn backgroundRadiance(direction: vec3<f32>) -> vec3<f32> {
  if (frame.background.w == 1.0) {
    return frame.background.rgb;
  }

  // A procedural sky paints the background with the strength it lights with, and its foreground over it. The CPU
  // renderer leaves that strength out of what the camera sees, which a renderer exposing for the light cannot: the sky
  // that lights a wall five times brighter than itself would show as a dark band behind it.
  if (frame.skyProgram != NO_PROGRAM || frame.skyIntensity > 0.0) {
    if (frame.skyBackground == 0u) {
      return vec3<f32>(0.0);
    }

    if (frame.skyProgram != NO_PROGRAM) {
      var input: ShaderInput;
      input.data = frame.skyData;
      input.direction = direction;
      input.diffuseColor = vec4<f32>(1.0);

      return runProgram(frame.skyProgram, input).color.rgb * frame.skyIntensity;
    }

    return environmentRadiance(direction);
  }

  return environmentRadiance(direction);
}

/**
 * How much of the sky a bounce ray that escaped keeps. Direct sampling aims at the bright parts of an image too, so the
 * two share it; when direct sampling took all of it, as it does from a toon surface, the bounce keeps none.
 */
fn escapeWeight(direction: vec3<f32>, bsdfPdf: f32, lightsOwnDirect: bool, scatteredAt: vec3<f32>) -> f32 {
  if (frame.environmentPick <= 0.0) {
    return 1.0;
  }

  if (lightsOwnDirect) {
    return 0.0;
  }

  if (bsdfPdf <= 0.0) {
    return 1.0;
  }

  // The sky was aimed at from the vertex the bounce left, through its windows as well when that vertex is inside.
  return misWeight(bsdfPdf, frame.environmentPick * skyPdf(scatteredAt, direction, portalChance(scatteredAt)));
}

/** Scales a contribution down to the luminance ceiling, which trades a little bias for the fireflies it removes. */
fn clampIndirect(contribution: vec3<f32>) -> vec3<f32> {
  let luminance = luminanceOf(contribution);

  if (frame.clampIndirect <= 0.0 || luminance <= frame.clampIndirect) {
    return contribution;
  }

  return contribution * (frame.clampIndirect / luminance);
}

/**
 * How much of the emission a bounce ray found the renderer keeps. Direct sampling had its own chance at this triangle,
 * and the heuristic splits the two so neither counts it twice.
 */
fn emissionWeight(hit: Hit, direction: vec3<f32>, bsdfPdf: f32, lightsOwnDirect: bool, scatteredAt: vec3<f32>,
  normal: vec3<f32>) -> f32 {
  if (frame.areaLightCount == 0u) {
    return 1.0;
  }

  // From the vertex the bounce left, which is where direct sampling would have picked it from.
  let areaPdf = lightPdfOfTriangle(hit.triangle, scatteredAt, normal);

  if (areaPdf <= 0.0) {
    return 1.0;
  }

  if (lightsOwnDirect) {
    return 0.0;
  }

  if (bsdfPdf <= 0.0) {
    return 1.0;
  }

  let cosLight = max(abs(dot(triangleNormal(hit.triangle), direction)), 1e-6);

  return misWeight(bsdfPdf, areaPdf * hit.distance * hit.distance / cosLight);
}

/**
 * The normal the light tree weighs lights by, or none where light from behind the surface counts too: a point in a
 * volume, a toon ramp read all round, a clearcoat that may face what its base turns from, and transmission.
 */
fn treeNormal(surface: Surface) -> vec3<f32> {
  if (surface.kind == KIND_VOLUME || surface.kind == KIND_TOON || surface.clearcoat > 0.0 || surface.transmission > 0.0) {
    return vec3<f32>(0.0);
  }

  return surface.normal;
}

/** One shadow ray from the surface towards a light, weighted against the chance the bounce would have found it too. */
fn directLight(
  surface: Surface,
  position: vec3<f32>,
  view: vec3<f32>,
  pick: f32,
  u: vec2<f32>,
  rng: ptr<function, PathSampler>,
  diffusePart: ptr<function, vec3<f32>>,
  medium: f32) -> vec3<f32> {
  *diffusePart = vec3<f32>(0.0);

  let chosen = sampleLight(position, treeNormal(surface), pick, u);

  if (!chosen.valid) {
    return vec3<f32>(0.0);
  }

  // A material with its own environment map sees that map in place of the scene's surroundings.
  if (chosen.kind == LIGHT_ENVIRONMENT && surface.envMap != NO_TEXTURE) {
    return vec3<f32>(0.0);
  }

  let reflected = evaluateBsdf(surface, view, chosen.direction);

  if (all(reflected.value <= vec3<f32>(0.0))) {
    return vec3<f32>(0.0);
  }

  let toon = surface.kind == KIND_TOON;
  // Three's shadow map leaves a toon mesh's own far side unshadowed; tracing through the mesh would not.
  let skip = select(NO_MATERIAL, surface.material, toon && dot(surface.normal, chosen.direction) < 0.1);
  let through = lightTransmittance(position, chosen, rng, skip, medium);

  if (all(through <= vec3<f32>(0.0))) {
    return vec3<f32>(0.0);
  }

  // A light that no bounce ray could have found keeps its whole contribution; an emissive triangle shares it. A toon
  // surface takes all of its direct light this way, because its bounce is not the lobe that lit it.
  let weight = select(1.0, misWeight(chosen.misPdf, reflected.pdf), chosen.misPdf > 0.0 && reflected.pdf > 0.0 && !toon);

  let light = chosen.value * through * weight;

  // The share of it the diffuse lobe reflects, which the denoiser filters apart from the specular one.
  *diffusePart = min(diffuseValue(surface, view, chosen.direction), reflected.value) * light;

  return reflected.value * light;
}

/** Three's fog factor at a view depth: linear between near and far, or exponential squared. */
fn fogFactor(depth: f32, kind: f32) -> f32 {
  if (kind == 2.0) {
    let density = frame.fogParams.z;

    return clamp(1.0 - exp(-density * density * depth * depth), 0.0, 1.0);
  }

  return smoothstep(frame.fogParams.x, frame.fogParams.y, depth);
}

/** The colour the diffuse signal is divided by at this surface; white where the surface shows what it emits. */
fn diffuseGuide(surface: Surface) -> vec3<f32> {
  if ((surface.flags & MATERIAL_LIT) == 0u) {
    return vec3<f32>(1.0);
  }

  return surface.albedo * (1.0 - surface.metalness) * (1.0 - clamp(surface.transmission, 0.0, 1.0));
}

/**
 * How often a vertex on a material that scatters inside takes the walk into it instead of its surface lobes: by the
 * share of the light each accounts for, as lobeWeights chooses between lobes. Never all of the time, so the specular
 * layer on top stays reachable.
 */
fn subsurfaceChance(surface: Surface, view: vec3<f32>) -> f32 {
  if (surface.subsurface <= 0.0 || (surface.flags & MATERIAL_PBR) == 0u) {
    return 0.0;
  }

  let diffuse = luminanceOf(surface.albedo * (1.0 - surface.metalness)) * coatTransmittance(surface, view);
  let inside = diffuse * surface.subsurface;
  let outside = diffuse * (1.0 - surface.subsurface) + luminanceOf(specularAlbedo(surface, view)) + 0.02;

  return clamp(inside / max(inside + outside, 1e-6), 0.0, 0.95);
}

/**
 * The single scattering albedo that gives this multiple scattering albedo in a semi-infinite medium: van de Hulst's
 * relation, inverted as Cycles inverts it for its random walk. What the material's colour says it reflects in all
 * is what light keeps at each scattering inside it.
 */
fn subsurfaceAlbedo(albedo: vec3<f32>) -> vec3<f32> {
  let a = clamp(albedo, vec3<f32>(0.0), vec3<f32>(0.999));
  let s = 4.09712 + 4.20863 * a - sqrt(9.59217 + 41.6808 * a + 17.7126 * a * a);

  return vec3<f32>(1.0) - s * s;
}

/**
 * How far the walk goes before it next scatters, drawn for one colour picked in proportion to what the path still
 * carries of it. choice keeps those odds, which the next step weighs its distance by over all three colours, so each
 * colour keeps its own mean free path and the estimate stays exact (spectral MIS, as Cycles' random walk does it).
 */
fn walkDistance(throughput: vec3<f32>, sigma: vec3<f32>, u: vec2<f32>, choice: ptr<function, vec3<f32>>) -> f32 {
  let weights = max(throughput, vec3<f32>(0.0));
  let total = weights.x + weights.y + weights.z;
  let odds = select(vec3<f32>(1.0 / 3.0), weights / total, total > 0.0);
  let chosen = select(select(sigma.z, sigma.y, u.x < odds.x + odds.y), sigma.x, u.x < odds.x);

  *choice = odds;

  return -log(max(1.0 - u.y, 1e-12)) / chosen;
}

/**
 * Where a walk leaves the material: a white Lambert surface facing out, as Cycles leaves its random walk. The colour
 * was taken inside, scattering by scattering, so nothing tints the light again on the way out.
 */
fn subsurfaceExit(m: u32, triangle: u32, bary: vec2<f32>, direction: vec3<f32>) -> Surface {
  var surface: Surface;
  // Turned to face against the reversed ray: outwards, the side the walk is leaving through.
  let out = interpolatedNormal(m, triangle, bary, -direction);

  surface.normal = out;
  surface.facing = out;
  surface.albedo = vec3<f32>(1.0);
  surface.flags = MATERIAL_LIT;
  surface.kind = KIND_LAMBERT;
  surface.opacity = 1.0;
  surface.alpha = 1.0;
  surface.blending = BLEND_NORMAL;
  surface.material = m;
  surface.roughness = 1.0;
  surface.ior = 1.5;
  surface.specularF90 = 1.0;
  surface.envMap = NO_TEXTURE;
  surface.envMapIntensity = 1.0;
  surface.gradientMap = NO_TEXTURE;

  return surface;
}

/** A point inside a volume where the light scatters: the medium's phase function in place of a surface's lobes. */
fn volumePoint(direction: vec3<f32>, albedo: vec3<f32>, anisotropy: f32) -> Surface {
  var surface: Surface;

  surface.normal = -direction;
  surface.facing = -direction;
  surface.albedo = albedo;
  surface.flags = MATERIAL_LIT;
  surface.kind = KIND_VOLUME;
  surface.anisotropy = anisotropy;
  surface.opacity = 1.0;
  surface.alpha = 1.0;
  surface.blending = BLEND_NORMAL;
  surface.material = NO_MATERIAL;
  surface.roughness = 1.0;
  surface.ior = 1.0;
  surface.specularF90 = 1.0;
  surface.envMap = NO_TEXTURE;
  surface.envMapIntensity = 1.0;
  surface.gradientMap = NO_TEXTURE;

  return surface;
}

/**
 * The shading normal with the edges near it rounded, as Cycles' bevel node rounds them, by one probe a sample. A
 * point is drawn on a disk of the bevel radius around the shaded point, across an axis picked along the normal half
 * the time and along each tangent a quarter; a short ray along that axis through it finds the surface of the same
 * material there, and that surface's normal turns this one by as much as it differs from the normal here. Over the
 * samples a pixel takes, the normal blends from one face to the next within the radius of every edge, so a box reads
 * as a machined one. A probe that finds nothing, or a normal turned away from the view, leaves the normal as it was.
 */
fn bevelNormal(m: u32, point: vec3<f32>, surface: Surface, view: vec3<f32>, radius: f32, u: vec2<f32>,
  v: vec2<f32>) -> vec3<f32> {
  let basis = orthonormalBasis(surface.facing);
  var axis = basis[2];
  var first = basis[0];
  var second = basis[1];

  if (v.x >= 0.5) {
    axis = select(basis[1], basis[0], v.x < 0.75);
    first = select(basis[2], basis[1], v.x < 0.75);
    second = select(basis[0], basis[2], v.x < 0.75);
  }

  let reach = radius * sqrt(u.x);
  let angle = 2.0 * PI * u.y;
  let across = point + first * (reach * cos(angle)) + second * (reach * sin(angle));
  // Along the axis one way or the other at random, from the far edge of the radius across it.
  let way = select(-1.0, 1.0, v.y < 0.5);
  let hit = traceScene(across - axis * (radius * way), axis * way, 2.0 * radius);

  if (hit.triangle == NO_HIT || materialIndexOf(hit.triangle) != m) {
    return surface.normal;
  }

  // The authored normal there, turned the way this side's own is: the facing normal is the authored one, or its reverse.
  let there = authoredNormal(hit.triangle, vec2<f32>(hit.u, hit.v)) * select(-1.0, 1.0, surface.front);
  let turned = normalize(surface.normal + there - surface.facing);

  return select(surface.normal, turned, dot(turned, view) > 0.0 && all(turned == turned));
}

struct PathResult {
  radiance: vec3<f32>,
  alpha: f32,
  // How far along the camera ray the first surface it stopped at is, or 1e30 when it stopped at none.
  depth: f32,
  // The denoiser's two signals: what of the radiance came through the first surface's diffuse lobe (and what the camera
  // sees directly), the rest being specular, and what guides each at that surface.
  diffuse: vec3<f32>,
  diffuseAlbedo: vec3<f32>,
  specularAlbedo: vec3<f32>,
  normal: vec3<f32>,
  // Along the camera ray to the first surface, its roughness, and how far the specular continuation went from it.
  guideDistance: f32,
  roughness: f32,
  specularDistance: f32,
  // How much of the first surface scatters light inside, which the denoiser relaxes its tests over.
  subsurface: f32,
};

/** How many surfaces a path may pass through without scattering (cutouts, culled faces, glass) before it gives up. */
const MAX_STEPS: u32 = 64u;

/** Scatterings one walk inside a material may take; the roulette ends most walks well before. */
const MAX_WALK: u32 = 128u;

fn traceRadiance(
  rayOrigin: vec3<f32>, rayDirection: vec3<f32>, rayLength: f32, rng: ptr<function, PathSampler>) -> PathResult {
  var radiance = vec3<f32>(0.0);
  var throughput = vec3<f32>(1.0);
  var origin = rayOrigin;
  var direction = rayDirection;
  var limit = rayLength;
  // Solid angle density of the current direction; zero for the camera ray, which no light sampling competes with.
  var bsdfPdf = 0.0;
  // What the volume the ray is travelling through absorbs per unit of distance; zero outside any.
  var absorption = vec3<f32>(0.0);
  // Still on the camera's own line of sight, which transmission keeps a path on.
  var primary = true;
  // Still the camera ray itself, in a straight line: only this one is culled and clipped by the far plane.
  var straight = true;
  var alpha = 0.0;
  var bounce = 0u;
  // The vertex the path last scattered at took all of its direct light by sampling lights: a toon surface.
  var lightsOwnDirect = false;
  // Where the path last scattered and the normal the light tree weighed lights by there, for the heuristic's other half.
  var lastPosition = rayOrigin;
  var lastNormal = vec3<f32>(0.0);
  // That vertex's own environment map, which a ray escaping from it sees instead of the scene's.
  var escapeMap = NO_TEXTURE;
  var escapeIntensity = 1.0;
  // Which colour a dispersive path carries, once a dispersive volume has split it; all three before that.
  var channel = vec3<f32>(1.0);
  // Three's fog, laid over everything the camera sees through the first surface.
  var fogged = false;
  var fogAmount = 0.0;
  var fogStart = vec3<f32>(0.0);
  var fogThroughput = vec3<f32>(0.0);
  // The air between the camera and the first surface it sees: aerial perspective.
  var hazed = false;
  var hazeDistance = 0.0;
  var hazeStart = vec3<f32>(0.0);
  var hazeThroughput = vec3<f32>(0.0);
  var hazeDirection = vec3<f32>(0.0);
  // A built-in material's hooks after <opaque_fragment> edit the colour the camera sees of it, which is everything the
  // path gathers from that surface on.
  var afterMaterial = NO_MATERIAL;
  var afterTriangle = 0u;
  var afterBary = vec2<f32>(0.0);
  var afterDirection = vec3<f32>(0.0);
  var afterAlbedo = vec4<f32>(0.0);
  var afterStart = vec3<f32>(0.0);
  var afterThroughput = vec3<f32>(0.0);
  var depth = 1e30;
  // The denoiser's guides, taken at the first surface the camera ray stops at.
  var guided = false;
  var diffuseAlbedo = vec3<f32>(1.0);
  var specularAlbedoGuide = vec3<f32>(1.0);
  var guideNormal = vec3<f32>(0.0);
  var guideDistance = 1e6;
  var guideRoughness = 1.0;
  var guideSubsurface = 0.0;
  // Where the path first scattered, which decides which of the two signals the rest of it feeds.
  var split = false;
  var splitBase = vec3<f32>(0.0);
  var splitDirect = vec3<f32>(0.0);
  var splitDirectDiffuse = vec3<f32>(0.0);
  var splitDiffuse = true;
  var waitingForSpecular = false;
  var specularDistance = 1e6;
  // Inside a material that scatters light, between two scatterings: its extinction and single scattering albedo per
  // colour, its anisotropy, the odds each colour had when the current step's distance was drawn, and the steps so far.
  var inMedium = false;
  var sigma = vec3<f32>(0.0);
  var walkAlbedo = vec3<f32>(0.0);
  var walkG = 0.0;
  var walkOdds = vec3<f32>(0.0);
  var walkSteps = 0u;
  // Inside a volume: its density per unit of distance, what it keeps at each scattering and its anisotropy. A camera
  // inside one starts there.
  var inVolume = false;
  var volumeDensity = 0.0;
  var volumeAlbedo = vec3<f32>(0.0);
  var volumeG = 0.0;

  if (VOLUMES && frame.cameraVolume != NO_MATERIAL && frame.volumesEnabled != 0u) {
    inVolume = true;
    volumeDensity = materials[frame.cameraVolume].volume.w;
    volumeAlbedo = materials[frame.cameraVolume].volume.rgb;
    volumeG = materials[frame.cameraVolume].volumeAnisotropy;
  }

  for (var step = 0u; step < MAX_STEPS + MAX_WALK; step = step + 1u) {
    let hit = traceScene(origin, direction, limit);

    // A step of the walk inside a material that ends in the medium: the light scatters there, keeping each colour's
    // share of it. It needs three numbers, which a plain hash gives for a fraction of what the sequence costs.
    if (SUBSURFACE_WALK && inMedium && hit.triangle == NO_HIT) {
      let travelled = exp(-sigma * limit);
      let density = dot(walkOdds, sigma * travelled);
      // A new direction, the next distance (a colour and how far), and the roulette's chance.
      let turn = hashSample2D(rng);
      let next = hashSample2D(rng);
      let chance = hashSample2D(rng).x;

      throughput = throughput * sigma * walkAlbedo * travelled / max(density, 1e-20);
      walkSteps = walkSteps + 1u;

      let survival = max3(throughput);

      if (walkSteps >= MAX_WALK || survival <= 0.0) {
        break;
      }

      if (survival < 0.2) {
        if (chance > survival) {
          break;
        }

        throughput = throughput / survival;
      }

      origin = origin + direction * limit;
      direction = sampleHenyeyGreenstein(direction, walkG, turn);
      limit = walkDistance(throughput, sigma, next, &walkOdds);
      continue;
    }

    // Every vertex draws the same dimensions in the same order, whatever the path then does with them: from the
    // sequence for the first vertices, from a hash past SEQUENCE_DEPTH bounces.
    let deep = bounce >= SEQUENCE_DEPTH;
    let coverage = pathSample2D(rng, deep);
    let pick = pathSample2D(rng, deep);
    let lightSample = pathSample2D(rng, deep);
    let lobe = pathSample2D(rng, deep);
    let bounceSample = pathSample2D(rng, deep);
    var mediumSample = vec2<f32>(0.0);

    if (VOLUMES) {
      mediumSample = pathSample2D(rng, deep);
    }

    // The walk left the material here, and the surface below is its way out rather than what the material shows.
    var exiting = false;

    if (SUBSURFACE_WALK && inMedium) {
      // The walk reached the surface before scattering again: the chance of travelling that far, per colour.
      let travelled = exp(-sigma * hit.distance);

      throughput = throughput * travelled / max(dot(walkOdds, travelled), 1e-20);
      inMedium = false;
      exiting = true;
    }

    // Inside a volume, the medium may stop the ray before whatever it hit. The distance is drawn now that the segment's
    // length is known, which a homogeneous medium allows; one grey density makes the weight of either outcome one.
    var scatterAt = -1.0;

    if (VOLUMES && inVolume && !exiting) {
      let free = -log(max(1.0 - mediumSample.x, 1e-12)) / max(volumeDensity, 1e-12);

      if (free < select(hit.distance, limit, hit.triangle == NO_HIT)) {
        scatterAt = free;
      }
    }

    let scattering = scatterAt >= 0.0;

    // The specular continuation's first hit: how far the reflection the denoiser filters actually is.
    if (waitingForSpecular) {
      specularDistance = select(min(hit.distance, 1e6), 1e6, hit.triangle == NO_HIT);
      waitingForSpecular = false;
    }

    if (hit.triangle == NO_HIT && !scattering) {
      if (primary) {
        // A transparent background leaves the pixel empty, which its alpha says.
        if (frame.background.w != 2.0) {
          alpha = 1.0;
          radiance = radiance + throughput * backgroundRadiance(direction);
        }
      } else {
        var sky = ambientRadiance(direction);

        if (escapeMap != NO_TEXTURE) {
          sky = sky + sampleEquirect(escapeMap, direction).rgb * escapeIntensity;
        } else {
          sky = sky + environmentRadiance(direction) * escapeWeight(direction, bsdfPdf, lightsOwnDirect, lastPosition);
        }

        let contribution = throughput * sky;

        radiance = radiance + select(contribution, clampIndirect(contribution), bounce > 1u);
      }

      break;
    }

    let travel = select(hit.distance, scatterAt, scattering);

    throughput = throughput * exp(-absorption * travel);

    let bary = vec2<f32>(hit.u, hit.v);
    let point = origin + direction * travel;
    var m = 0u;

    if (!scattering) {
      m = materialIndexOf(hit.triangle);
    }

    var surface: Surface;

    if (SUBSURFACE_WALK && exiting) {
      surface = subsurfaceExit(m, hit.triangle, bary, direction);
      // Seen from outside, along its normal: the side the light leaves through.
      direction = -surface.normal;
    } else if (VOLUMES && scattering) {
      surface = volumePoint(direction, volumeAlbedo, volumeG);
    } else {

    // A volume's boundary is not a surface: the ray goes in or out, and the medium is what it meets.
    if (VOLUMES && (materials[m].flags & MATERIAL_VOLUME) != 0u) {
      if (dot(authoredNormal(hit.triangle, bary), direction) < 0.0) {
        if (frame.volumesEnabled != 0u) {
          inVolume = true;
          volumeDensity = materials[m].volume.w;
          volumeAlbedo = materials[m].volume.rgb;
          volumeG = materials[m].volumeAnisotropy;
        }
      } else {
        inVolume = false;
      }

      origin = point + direction * RAY_EPSILON;
      limit = select(1e30, limit - hit.distance, straight);
      continue;
    }

    // Three culls the side a material does not draw. Only the camera's rays are culled: a mesh still casts shadows and
    // holds light in from both sides.
    // A planar capture drops what lies behind its plane, for the camera's own ray.
    let clipped = any(frame.clipPlane != vec4<f32>(0.0)) && dot(frame.clipPlane.xyz, point) + frame.clipPlane.w < 0.0;

    if (straight && (clipped || isCulled(m, hit.triangle, direction))) {
      origin = point + direction * RAY_EPSILON;
      limit = limit - hit.distance;
      continue;
    }

    if (isClipped(m, point)
      || ((materials[m].flags & MATERIAL_WIREFRAME) != 0u && !onWireframe(hit.triangle, point))) {
      origin = point + direction * RAY_EPSILON;
      limit = select(1e30, limit - hit.distance, straight);
      continue;
    }

    let base = sampleMap(materials[m].map, hit.triangle, bary);

    // A cutout lets the ray through before anything else about the surface is read: the normal map, the other maps and
    // the lobes would only be thrown away. The draw and the comparison are the ones made below, so nothing changes.
    if (coverageFirst(m)) {
      let opacity = coverageOf(m, materialAlphaFrom(m, hit.triangle, bary, base));

      if (opacity < 1.0 && coverage.x >= opacity) {
        origin = point + direction * RAY_EPSILON;
        limit = select(1e30, limit - hit.distance, straight);
        continue;
      }
    }

    surface = resolveSurfaceFrom(hit.triangle, bary, direction, base);

    // Deep in the path, light reflects off a material that scatters inside as off a diffuse one of the same colour.
    if (SUBSURFACE_WALK && bounce >= frame.subsurfaceDepth) {
      surface.subsurface = 0.0;
    }

    if (BEVEL && frame.bevelEnabled != 0u && materials[m].bevel > 0.0) {
      let disk = hashSample2D(rng);
      let axis = hashSample2D(rng);

      surface.normal = bevelNormal(m, point, surface, -direction, materials[m].bevel, disk, axis);
    }
    }

    // Blending other than normal: the surface adds its light to, or filters, whatever is behind it, and the path goes on.
    if (surface.blending == BLEND_ADDITIVE || surface.blending == BLEND_MULTIPLY || surface.blending == BLEND_SUBTRACTIVE) {
      // Three blends these with the fragment's alpha whether or not the material is transparent.
      let strength = surface.opacity * clamp(surface.alpha, 0.0, 1.0);
      let color = select(surface.albedo, surface.emissive, (surface.flags & MATERIAL_LIT) == 0u);

      if (surface.blending == BLEND_ADDITIVE) {
        var added = surface.emissive;

        if ((surface.flags & MATERIAL_LIT) != 0u) {
          var unused: vec3<f32>;
          added = added + directLight(surface, point + surface.normal * RAY_EPSILON, -direction, pick.x, lightSample, rng,
            &unused, select(0.0, volumeDensity, inVolume));
        }

        radiance = radiance + throughput * added * strength;
      } else if (surface.blending == BLEND_MULTIPLY) {
        throughput = throughput * mix(vec3<f32>(1.0), color, strength);
      } else {
        throughput = throughput * (vec3<f32>(1.0) - color * strength);
      }

      origin = point + direction * RAY_EPSILON;
      limit = select(1e30, limit - hit.distance, straight);
      continue;
    }

    // A surface that only partly covers its triangle lets the ray straight through the rest of the time. The direction
    // does not change, so the density that found it still stands and the next hit is weighted against the same one.
    if (surface.opacity < 1.0 && coverage.x >= surface.opacity) {
      origin = point + direction * RAY_EPSILON;
      limit = select(1e30, limit - hit.distance, straight);
      continue;
    }

    // Three's ShadowMaterial shows its colour only where a light is blocked, and nothing elsewhere.
    if (surface.kind == KIND_SHADOW) {
      let shadow = shadowMaskAt(point + surface.facing * RAY_EPSILON, rng);

      if (coverage.y >= shadow * clamp(materials[m].color.w, 0.0, 1.0)) {
        origin = point + direction * RAY_EPSILON;
        limit = select(1e30, limit - hit.distance, straight);
        continue;
      }

      if (primary) {
        alpha = 1.0;
      }

      radiance = radiance + throughput * surface.albedo;
      break;
    }

    if (straight && depth >= 1e30) {
      depth = length(point - rayOrigin);
    }

    // A window the camera ray goes straight through is not what the camera sees: what lies behind it is, so the
    // denoiser takes its guides there, and the air its distance. Glass has next to no diffuse colour and one flat
    // normal, which would leave everything seen through it unfiltered. The window's own guides stand until then, for
    // the paths that reflect off it instead.
    let window = surface.transmission > 0.0 && surface.thickness <= 0.0;

    if (straight && !guided) {
      guided = !window;
      diffuseAlbedo = diffuseGuide(surface);
      specularAlbedoGuide = select(specularAlbedo(surface, -direction), vec3<f32>(1.0), (surface.flags & MATERIAL_LIT) == 0u);
      guideNormal = surface.normal;
      guideDistance = length(point - rayOrigin);
      guideRoughness = surface.roughness;
      guideSubsurface = select(0.0, surface.subsurface, SUBSURFACE_WALK);

      if (surface.kind == KIND_PHONG) {
        // Phong's exponent as the roughness of the GGX lobe that is about as wide.
        guideRoughness = pow(2.0 / (surface.shininess + 2.0), 0.25);
      }
    }

    if (primary) {
      alpha = 1.0;

      if (!scattering && !fogged && afterMaterial == NO_MATERIAL && materials[m].afterProgram != NO_PROGRAM) {
        afterMaterial = m;
        afterTriangle = hit.triangle;
        afterBary = bary;
        afterDirection = direction;
        afterAlbedo = vec4<f32>(surface.albedo, surface.alpha);
        afterStart = radiance;
        afterThroughput = throughput;
      }

      if (!fogged) {
        let everySurface = frame.fogParams.w > 0.0;
        let kind = frame.fogColor.w;

        if (everySurface || (kind > 0.0 && (surface.flags & MATERIAL_FOG) != 0u)) {
          let depth = -(frame.worldToCamera * vec4<f32>(point, 1.0)).z;

          fogged = true;
          fogAmount = fogFactor(depth, select(kind, 1.0, everySurface));
          fogStart = radiance;
          fogThroughput = throughput;
        }
      }

      if (!hazed && !window && frame.atmosphereColor.w > 0.0) {
        hazed = true;
        // From the eye, where the camera ray only starts on the near plane; an orthographic view has no eye.
        hazeDistance = length(point - select(frame.cameraToWorld[3].xyz, rayOrigin, frame.cameraKind == 1u));
        hazeStart = radiance;
        hazeThroughput = throughput;
        hazeDirection = direction;
      }

      radiance = radiance + throughput * surface.lightMap;
    }

    if (any(surface.emissive > vec3<f32>(0.0))) {
      let contribution = throughput * surface.emissive * emissionWeight(hit, direction, bsdfPdf, lightsOwnDirect, lastPosition, lastNormal);

      // Seen by the camera or one bounce away this is direct light and stays exact; deeper it may be clamped.
      radiance = radiance + select(contribution, clampIndirect(contribution), bounce > 1u);
    }

    // Three does not light a basic, points, line or sprite material, so nothing carries on from one.
    // Past the last bounce a path still crosses a window straight on, which scatters nothing, to find the sky behind
    // it: cut there, the daylight a room gets through its glazing would go missing from every last bounce.
    let spent = bounce >= frame.bounces;

    if ((surface.flags & MATERIAL_LIT) == 0u || (spent && !window)) {
      break;
    }

    // Lambert and Phong blend their own environment map over the lit colour, which here is a share of the path.
    if (surface.legacyAmount > 0.0) {
      let amount = surface.legacyAmount;

      if (surface.combine == 1u) {
        radiance = radiance + throughput * surface.legacyEnv * amount;
        throughput = throughput * (1.0 - amount);
      } else if (surface.combine == 2u) {
        radiance = radiance + throughput * surface.legacyEnv * amount;
      } else {
        throughput = throughput * mix(vec3<f32>(1.0), surface.legacyEnv, amount);
      }
    }

    let view = -direction;
    var next = direction;
    var nextOrigin = point + direction * RAY_EPSILON;

    // Transmission replaces part of the diffuse layer, so that share of the time the surface is a dielectric instead.
    // Choosing it in proportion to its weight leaves both halves unweighted. The opaque half aims at a light; on a
    // window that lets only part of the light through, it does so on every path, weighed by its share, rather than on
    // the paths that chose it: a sunlit pane seen against a dim room would otherwise be lit on one path in five and
    // dark on the rest, which is the grain glass shows.
    let dielectric = surface.transmission > 0.0 && lobe.y < surface.transmission;
    let windowLight = window && !spent && surface.transmission < 1.0 && surface.subsurface <= 0.0;

    if (!dielectric || windowLight) {
      var chance = 0.0;

      if (!dielectric) {
        // Light that enters the material: a random walk inside it from here, in place of the share of the diffuse
        // layer that scatters inside. Chosen by chance, so neither branch is weighted against the other.
        // The window's opaque share is a surface like any other, which a spent path stops at.
        if (spent) {
          break;
        }

        if (SUBSURFACE_WALK) {
          chance = subsurfaceChance(surface, view);
        }

        if (SUBSURFACE_WALK && lobe.x < chance) {
          let medium = materials[surface.material].subsurface;

          throughput = throughput
            * ((1.0 - surface.metalness) * surface.subsurface * coatTransmittance(surface, view) / chance);
          sigma = vec3<f32>(1.0) / max(medium.rgb, vec3<f32>(1e-6));
          walkAlbedo = subsurfaceAlbedo(surface.albedo);
          walkG = materials[surface.material].subsurfaceAnisotropy;
          walkSteps = 0u;

          // Light that scattered inside is diffuse light, which is the signal it feeds.
          if (!split) {
            split = true;
            splitBase = radiance;
            splitDiffuse = true;
          }

          inMedium = true;
          primary = false;
          straight = false;
          bsdfPdf = 0.0;
          lightsOwnDirect = false;
          escapeMap = NO_TEXTURE;
          bounce = bounce + 1u;
          direction = cosineDirection(-surface.facing, bounceSample);
          origin = point - surface.facing * RAY_EPSILON;
          limit = walkDistance(throughput, sigma, pick, &walkOdds);
          continue;
        }

        throughput = throughput / (1.0 - chance);
      }

      let position = point + surface.normal * RAY_EPSILON;
      var directDiffuse: vec3<f32>;
      let lit = select(throughput, throughput * (1.0 - surface.transmission), windowLight);
      let direct = lit * directLight(surface, position, view, pick.x, lightSample, rng, &directDiffuse,
        select(0.0, volumeDensity, inVolume));
      let added = select(direct, clampIndirect(direct), bounce > 0u);
      let before = radiance;

      radiance = radiance + added;

      if (!dielectric) {
        let scattered = sampleBsdf(surface, view, (lobe.x - chance) / (1.0 - chance), bounceSample);

        // The first scatter splits the path: its direct light by lobe, and everything after it by the lobe it chose.
        if (!split) {
          split = true;
          splitBase = before;
          splitDirect = added;
          splitDirectDiffuse = lit * directDiffuse * select(1.0, luminanceOf(added) / max(luminanceOf(direct), 1e-20),
            bounce > 0u);
          splitDiffuse = !scattered.valid || scattered.diffuse;
          waitingForSpecular = scattered.valid && !scattered.diffuse;
        }

        if (!scattered.valid) {
          break;
        }

        throughput = throughput * scattered.weight;
        bsdfPdf = scattered.pdf;
        lastPosition = position;
        lastNormal = treeNormal(surface);
        lightsOwnDirect = surface.kind == KIND_TOON;
        escapeMap = surface.envMap;
        escapeIntensity = surface.envMapIntensity;
        next = scattered.direction;
        nextOrigin = position;
        primary = false;
        straight = false;
        bounce = bounce + 1u;
      }
    }

    if (dielectric) {
      var ior = surface.ior;

      // Three's dispersion: each colour refracts with its own index, spread around the material's. The path picks one
      // colour the first time it enters such a volume and carries only that one from then on.
      if (surface.dispersion > 0.0 && surface.thickness > 0.0) {
        if (all(channel == vec3<f32>(1.0))) {
          let which = min(u32(coverage.y * 3.0), 2u);

          channel = vec3<f32>(select(0.0, 1.0, which == 0u), select(0.0, 1.0, which == 1u), select(0.0, 1.0, which == 2u));
          throughput = throughput * channel * 3.0;
        }

        let spread = (ior - 1.0) * 0.025 * surface.dispersion;

        ior = dot(channel, vec3<f32>(ior - spread, ior, ior + spread));
      }

      let crossing = sampleDielectric(surface, view, ior, lobe.y / surface.transmission, bounceSample);

      // A reflection off the glass is a scattering, which a spent path no longer has.
      if (!crossing.valid || (spent && !crossing.straight)) {
        break;
      }

      // Bending or reflecting off glass is the specular signal.
      if (!split && !crossing.straight) {
        split = true;
        splitBase = radiance;
        splitDiffuse = false;
        waitingForSpecular = true;
      }

      throughput = throughput * crossing.weight;

      // Straight through a thin wall the direction stands, and with it the density that will weigh the next hit.
      if (!crossing.straight) {
        straight = false;
        next = crossing.direction;
        nextOrigin = point + surface.normal * select(RAY_EPSILON, -RAY_EPSILON, crossing.crossed);
        // A direction no light sampling could have produced: whatever the path finds next keeps its whole value.
        bsdfPdf = 0.0;
        lightsOwnDirect = false;
        escapeMap = surface.envMap;
        escapeIntensity = surface.envMapIntensity;

        if (crossing.crossed) {
          absorption = select(vec3<f32>(0.0), surface.absorption, surface.front);
        } else {
          primary = false;
        }
      }

      if (!primary) {
        bounce = bounce + 1u;
      }
    }

    // A path that carries almost nothing is ended by chance rather than by a cut, which keeps the estimate unbiased.
    let survival = max(throughput.r, max(throughput.g, throughput.b));

    if (survival <= 0.0) {
      break;
    }

    if (survival < frame.rouletteThreshold && bounce >= frame.rouletteStart) {
      if (pick.y > survival) {
        break;
      }

      throughput = throughput / survival;
    }

    origin = nextOrigin;
    limit = select(1e30, limit - hit.distance, straight);
    direction = next;
  }

  // The diffuse signal: everything before the first scatter, its diffuse direct light, and the rest if the diffuse lobe
  // carried the path on. A path that never scattered is all diffuse, which is what the camera sees directly.
  var diffuse = radiance;

  if (split) {
    let after = radiance - splitBase - splitDirect;
    diffuse = splitBase + splitDirectDiffuse + select(vec3<f32>(0.0), after, splitDiffuse);
  }

  if (afterMaterial != NO_MATERIAL && all(afterThroughput > vec3<f32>(0.0))) {
    var input = shaderInputAt(afterMaterial, afterTriangle, afterBary, afterDirection);
    input.data = materials[afterMaterial].afterData;
    input.diffuseColor = afterAlbedo;
    input.fragColor = vec4<f32>((radiance - afterStart) / afterThroughput, 1.0);

    let edited = runProgram(materials[afterMaterial].afterProgram, input);

    radiance = afterStart + afterThroughput * edited.color.rgb;
    // A hook rewrites the colour as a whole: there is no split left to keep.
    diffuse = radiance;
  }

  // Three lays its fog after the hooks, over what they made of the surface.
  if (fogged) {
    let fog = frame.fogColor.rgb * fogThroughput * fogAmount;

    radiance = fogStart + (radiance - fogStart) * (1.0 - fogAmount) + fog;
    // What came before the fogged surface is diffuse, so the fog's own colour joins that signal.
    diffuse = fogStart + (diffuse - fogStart) * (1.0 - fogAmount) + fog;
  }

  // The air takes its share of what lies behind it and adds the light it scatters towards the camera; the background,
  // which already is the sky, is left as it is.
  if (hazed) {
    let through = exp(-frame.atmosphereExtinction.rgb * hazeDistance);
    // The sun lights the air by the phase function of what fills it: Rayleigh's for clean air, a forward lobe for haze
    // (Henyey-Greenstein at 0.76), which is what makes the air glow towards the sun.
    let turn = dot(hazeDirection, frame.atmosphereSun.xyz);
    let phase = mix(3.0 / (16.0 * PI) * (1.0 + turn * turn), henyeyGreenstein(turn, 0.76), frame.atmosphereExtinction.w);
    let air = frame.atmosphereColor.rgb + frame.atmosphereSunColor.rgb * phase;
    let glow = air * hazeThroughput * (vec3<f32>(1.0) - through);

    radiance = hazeStart + (radiance - hazeStart) * through + glow;
    diffuse = hazeStart + (diffuse - hazeStart) * through + glow;
  }

  var result: PathResult;
  result.radiance = radiance;
  result.alpha = alpha;
  result.depth = depth;
  result.diffuse = diffuse;
  result.diffuseAlbedo = diffuseAlbedo;
  result.specularAlbedo = specularAlbedoGuide;
  result.normal = guideNormal;
  result.guideDistance = guideDistance;
  result.roughness = guideRoughness;
  result.specularDistance = specularDistance;
  result.subsurface = guideSubsurface;

  return result;
}

/** A point on the lens: a disk, or Three-free polygon bokeh shaped as the CPU renderer shapes it. */
fn lensPoint(u: vec2<f32>) -> vec2<f32> {
  var radius = sqrt(u.x) * frame.lens.x;
  let angle = u.y * 2.0 * PI;
  let blades = frame.lens.z;

  if (blades >= 3.0) {
    let wedge = 2.0 * PI / blades;
    radius = radius * cos(PI / blades) / cos((angle - floor(angle / wedge) * wedge) - PI / blades);
  }

  return vec2<f32>(cos(angle), sin(angle)) * radius;
}

/**
 * Where a sample's camera ray starts, which way it points and how far it may go, for any camera: the ray is cast through
 * the inverse of the camera's own projection, from the near plane to the far one, which clip it as Three clips a
 * fragment. A lens moves the origin across its aperture and aims at the point in focus.
 */
fn cameraRay(pixel: vec2<u32>, jitter: vec2<f32>, lens: vec2<f32>) -> array<vec4<f32>, 2> {
  let uv = (vec2<f32>(pixel) + jitter) / vec2<f32>(frame.resolution);
  var ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

  // A lens's radial distortion: the pixel sees the scene at a radius scaled by the square of its own, which is one at
  // the corners. Positive reaches further out at the edges (barrel), negative less far (pincushion).
  if (frame.lensDistortion != 0.0) {
    let aspect = f32(frame.resolution.x) / f32(frame.resolution.y);
    let radius2 = (ndc.x * ndc.x * aspect * aspect + ndc.y * ndc.y) / (aspect * aspect + 1.0);

    ndc = ndc * (1.0 + frame.lensDistortion * radius2);
  }

  let ray = ndcRay(ndc);

  if (frame.lens.x <= 0.0 || frame.cameraKind == 1u) {
    return ray;
  }

  let position = frame.cameraToWorld[3].xyz;
  let forward = -normalize(frame.cameraToWorld[2].xyz);
  let focus = position + ray[1].xyz * (frame.lens.y / max(dot(ray[1].xyz, forward), 1e-6));
  let offset = lensPoint(lens);
  let origin = position + normalize(frame.cameraToWorld[0].xyz) * offset.x + normalize(frame.cameraToWorld[1].xyz) * offset.y;

  return array<vec4<f32>, 2>(vec4<f32>(origin, 0.0), vec4<f32>(normalize(focus - origin), 1e30));
}

/** The window depth GL would store for the first surface the camera ray stopped at: one where it stopped at none. */
fn windowDepth(point: vec3<f32>, distance: f32) -> f32 {
  if (distance >= 1e30) {
    return 1.0;
  }

  let clip = frame.projection * (frame.worldToCamera * vec4<f32>(point, 1.0));

  return clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
}

/** smoothstep as the CPU renderer writes it, which also holds when the two edges meet. */
fn edge(low: f32, high: f32, value: f32) -> f32 {
  let t = select(select(0.0, 1.0, value >= high), clamp((value - low) / (high - low), 0.0, 1.0), high > low);

  return t * t * (3.0 - 2.0 * t);
}

/** How much of what lies at this point the fog lets through to the camera. */
fn fogVisibility(point: vec3<f32>) -> f32 {
  let everySurface = frame.fogParams.w > 0.0;
  let kind = frame.fogColor.w;

  if (!everySurface && kind <= 0.0) {
    return 1.0;
  }

  let depth = -(frame.worldToCamera * vec4<f32>(point, 1.0)).z;

  return 1.0 - fogFactor(depth, select(kind, 1.0, everySurface));
}

/** The most volumetric lights one view ray marches through. */
const MAX_VOLUMETRIC: u32 = 8u;

/**
 * Steps of one march. The CPU renderer marches 64 once per pixel; here every path marches, a few hundred times a
 * pixel, each at its own random offset, so a few steps a path add up to a finer integral than 64 once, at a fraction of
 * the shadow rays.
 */
const MARCH_STEPS: u32 = 8u;

/**
 * The CPU renderer's volumetric lights (volumetricLight.js): one march across the union of every light's
 * reach along the view ray, Henyey-Greenstein scattering towards the camera, a shadow ray for a spot light that casts
 * one, and the same compression of the sum before it is added over the image.
 */
fn volumetricGlow(origin: vec3<f32>, direction: vec3<f32>, distance: f32, jitter: f32) -> vec3<f32> {
  let count = min(frame.volumetricCount, MAX_VOLUMETRIC);
  let far = min(distance, 40.0);
  var tStart = far;
  var tEnd = 0.0;

  for (var k = 0u; k < count; k = k + 1u) {
    let at = frame.volumetricData + k * 20u;
    let position = extraVec4(at).xyz;
    let spot = extraFloat(at + 3u) > 0.5;
    let axis = extraVec4(at + 4u).xyz;
    let radius = extraFloat(at + 7u);
    let outer = extraFloat(at + 15u);
    let delta = origin - position;
    let b = dot(delta, direction);
    let disc = b * b - dot(delta, delta) + radius * radius;

    if (disc <= 0.0) {
      continue;
    }

    let start = max(0.0, -b - sqrt(disc));
    let end = min(far, -b + sqrt(disc));

    if (end <= start) {
      continue;
    }

    var cuts = array<f32, 4>(start, end, end, end);
    var cutCount = 2u;
    let cone = cos(outer);

    if (spot) {
      let ar = dot(axis, direction);
      let ac = dot(axis, delta);
      let cos2 = cone * cone;
      let qa = ar * ar - cos2;
      let qb = ar * ac - dot(direction, delta) * cos2;
      let qc = ac * ac - dot(delta, delta) * cos2;
      let discriminant = qb * qb - qa * qc;
      var roots = array<f32, 2>(-1.0, -1.0);

      if (abs(qa) < 1e-8) {
        if (abs(qb) > 1e-8) {
          roots[0] = -qc / (2.0 * qb);
        }
      } else if (discriminant >= 0.0) {
        roots[0] = (-qb - sqrt(discriminant)) / qa;
        roots[1] = (-qb + sqrt(discriminant)) / qa;
      }

      for (var r = 0u; r < 2u; r = r + 1u) {
        if (roots[r] > start && roots[r] < end) {
          cuts[cutCount] = roots[r];
          cutCount = cutCount + 1u;
        }
      }

      // Sort the few cuts in place.
      for (var i = 1u; i < cutCount; i = i + 1u) {
        var j = i;

        loop {
          if (j == 0u || cuts[j - 1u] <= cuts[j]) {
            break;
          }

          let swap = cuts[j];
          cuts[j] = cuts[j - 1u];
          cuts[j - 1u] = swap;
          j = j - 1u;
        }
      }
    }

    for (var segment = 1u; segment < cutCount; segment = segment + 1u) {
      let lo = cuts[segment - 1u];
      let hi = cuts[segment];

      if (spot && dot(normalize(delta + direction * ((lo + hi) * 0.5)), axis) < cone) {
        continue;
      }

      tStart = min(tStart, lo);
      tEnd = max(tEnd, hi);
    }
  }

  if (tEnd <= tStart) {
    return vec3<f32>(0.0);
  }

  let step = (tEnd - tStart) / f32(MARCH_STEPS);
  var transmittance: array<f32, 8>;

  for (var k = 0u; k < count; k = k + 1u) {
    transmittance[k] = exp(-extraFloat(frame.volumetricData + k * 20u + 11u) * tStart);
  }

  var result = vec3<f32>(0.0);

  for (var i = 0u; i < MARCH_STEPS; i = i + 1u) {
    let t = tStart + (f32(i) + jitter) * step;
    let p = origin + direction * t;
    let fogged = fogVisibility(p);

    if (fogged <= 0.0001) {
      break;
    }

    var lit = false;

    for (var k = 0u; k < count; k = k + 1u) {
      if (transmittance[k] <= 0.003) {
        continue;
      }

      lit = true;

      let at = frame.volumetricData + k * 20u;
      let position = extraVec4(at).xyz;
      let spot = extraFloat(at + 3u) > 0.5;
      let axis = extraVec4(at + 4u).xyz;
      let color = extraVec4(at + 8u).rgb;
      let density = extraFloat(at + 11u);
      let angle = extraFloat(at + 12u);
      let penumbra = extraFloat(at + 13u);
      let range = extraFloat(at + 14u);
      let outer = extraFloat(at + 15u);
      let spread = extraFloat(at + 16u);
      let anisotropy = extraFloat(at + 17u);
      let casts = extraFloat(at + 18u) > 0.5;
      let toLight = position - p;
      let dist = length(toLight);
      let toward = toLight / max(dist, 1e-20);
      var attenuation = exp(-dist / (select(10.0, range, range > 0.0) * 0.3));

      if (range > 0.0) {
        attenuation = attenuation * (1.0 - edge(range * 0.8, range, dist));
      }

      if (spot) {
        let along = -dot(toward, axis);
        let inner = cos(angle * (1.0 - penumbra));
        let core = edge(cos(angle), inner, along);
        let scattered = spread * pow(edge(cos(outer), inner, along), 2.0);

        attenuation = attenuation * select(core, max(core, scattered), spread > 0.0);
      }

      if (attenuation <= 0.0001) {
        continue;
      }

      // Shadowing follows Three's volumetric lights, which only have shadow maps for spots.
      if (spot && casts && frame.shadowsEnabled != 0u && traceAny(p, toward, dist * (1.0 - 1e-3)) != NO_HIT) {
        continue;
      }

      let cosine = dot(toward, direction);
      let g = anisotropy;
      let phase = select((1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosine, 1.5), 1.0, g == 0.0);

      // The air between the camera and this point, exactly: a long step evaluated at its start would overstate it.
      result = result + color * (attenuation * density * step * exp(-density * t) * fogged * phase * 1.5);
    }

    if (!lit) {
      break;
    }

    for (var k = 0u; k < count; k = k + 1u) {
      transmittance[k] = transmittance[k] * exp(-extraFloat(frame.volumetricData + k * 20u + 11u) * step);
    }
  }

  return result / (vec3<f32>(1.0) + result);
}

/**
 * How far a pixel's estimate may still be from its mean: the standard error of its luminance, over the square root of
 * the luminance, which is roughly how visible that error is once the image is encoded for display.
 */
fn pixelError(pixel: vec2<u32>) -> f32 {
  let index = pixel.y * frame.resolution.x + pixel.x;
  let sum = accumulation[index];

  if (sum.w < 2.0) {
    return 1e30;
  }

  let squares = accumulation[frame.pixelCount + index];
  let diffuse = accumulation[4u * frame.pixelCount + index].rgb;

  // Each of the denoiser's signals has its own variance; the pixel has converged when both have.
  return max(
    channelError(luminanceOf(diffuse) / sum.w, squares.z / sum.w, sum.w),
    channelError(luminanceOf(sum.rgb - diffuse) / sum.w, squares.w / sum.w, sum.w));
}

fn channelError(mean: f32, square: f32, samples: f32) -> f32 {
  let variance = max(square - mean * mean, 0.0);

  return sqrt(variance / samples) / sqrt(max(mean, 0.0) + 1e-3);
}

/**
 * Whether a pixel has converged, judged over its neighbourhood: a rare bright path lands in one pixel, and a pixel
 * judged alone would stop where its neighbours had not yet seen one and leave that area too dark.
 */
fn converged(pixel: vec2<u32>) -> bool {
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let x = clamp(i32(pixel.x) + dx, 0, i32(frame.resolution.x) - 1);
      let y = clamp(i32(pixel.y) + dy, 0, i32(frame.resolution.y) - 1);

      if (pixelError(vec2<u32>(u32(x), u32(y))) > frame.adaptiveThreshold) {
        return false;
      }
    }
  }

  return true;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= frame.bandSize.x || id.y >= frame.bandSize.y) {
    return;
  }

  let pixel = frame.bandOrigin + id.xy;

  if (pixel.x >= frame.resolution.x || pixel.y >= frame.resolution.y) {
    return;
  }

  let index = pixel.y * frame.resolution.x + pixel.x;

  // Adaptive sampling: a pixel whose neighbourhood has converged takes no more samples, which leaves the pass to
  // the pixels that still need them.
  if (frame.adaptiveMin > 0u && accumulation[index].w >= f32(frame.adaptiveMin) && converged(pixel)) {
    return;
  }

  var radiance = vec3<f32>(0.0);
  var alpha = 0.0;
  var depthSum = 0.0;
  var diffuseSquares = 0.0;
  var specularSquares = 0.0;
  var diffuseSum = vec3<f32>(0.0);
  var diffuseAlbedoSum = vec3<f32>(0.0);
  var specularAlbedoSum = vec3<f32>(0.0);
  var normalSum = vec3<f32>(0.0);
  var distanceSum = 0.0;
  var roughnessSum = 0.0;
  var specularDistanceSum = 0.0;
  var subsurfaceSum = 0.0;

  for (var s = 0u; s < frame.sampleCount; s = s + 1u) {
    var rng = makeSampler(pixel, frame.sampleIndex + s);
    let jitter = sample2D(&rng);
    let lens = sample2D(&rng);
    let ray = cameraRay(pixel, jitter, lens);
    let result = traceRadiance(ray[0].xyz, ray[1].xyz, ray[1].w, &rng);
    var sampleAlpha = result.alpha;

    var sampled = result.radiance;
    var sampledDiffuse = result.diffuse;

    if (frame.volumetricCount > 0u) {
      let glow = volumetricGlow(ray[0].xyz, ray[1].xyz, result.depth, sample2D(&rng).x);

      // As the CPU renderer composites it: added over the image, and raising the coverage just enough to hold it.
      sampled = sampled + glow;
      sampledDiffuse = sampledDiffuse + glow;
      sampleAlpha = max(sampleAlpha, max3(glow));
    }

    let sampledSpecular = sampled - sampledDiffuse;

    radiance = radiance + sampled;
    diffuseSum = diffuseSum + sampledDiffuse;
    diffuseSquares = diffuseSquares + luminanceOf(sampledDiffuse) * luminanceOf(sampledDiffuse);
    specularSquares = specularSquares + luminanceOf(sampledSpecular) * luminanceOf(sampledSpecular);
    diffuseAlbedoSum = diffuseAlbedoSum + result.diffuseAlbedo;
    specularAlbedoSum = specularAlbedoSum + result.specularAlbedo;
    normalSum = normalSum + result.normal;
    distanceSum = distanceSum + result.guideDistance;
    roughnessSum = roughnessSum + result.roughness;
    specularDistanceSum = specularDistanceSum + result.specularDistance;
    subsurfaceSum = subsurfaceSum + result.subsurface;

    alpha = alpha + sampleAlpha;
    depthSum = depthSum + windowDepth(ray[0].xyz + ray[1].xyz * min(result.depth, 1e20), result.depth);
  }

  let coverage = accumulation[frame.pixelCount + index];
  accumulation[index] = accumulation[index] + vec4<f32>(radiance, f32(frame.sampleCount));
  accumulation[frame.pixelCount + index] = vec4<f32>(
    coverage.x + alpha, coverage.y + depthSum, coverage.z + diffuseSquares, coverage.w + specularSquares);
  // The denoiser's signals and guides, summed like the radiance so they average over the pixel's samples.
  let pc = frame.pixelCount;
  accumulation[2u * pc + index] = accumulation[2u * pc + index] + vec4<f32>(diffuseAlbedoSum, roughnessSum);
  accumulation[3u * pc + index] = accumulation[3u * pc + index] + vec4<f32>(normalSum, distanceSum);
  accumulation[4u * pc + index] = accumulation[4u * pc + index] + vec4<f32>(diffuseSum, specularDistanceSum);
  accumulation[5u * pc + index] = accumulation[5u * pc + index] + vec4<f32>(specularAlbedoSum, subsurfaceSum);
  // This row still has work: every pixel that traced writes the same value, so the race is harmless.
  accumulation[6u * pc + pixel.y] = vec4<f32>(1.0);
}
`;
