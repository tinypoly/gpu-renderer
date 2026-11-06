/**
 * Material evaluation and the sampling the CPU renderer has no need for. Its `material.js` only ever asks what a surface
 * reflects towards a light it already knows about; a path tracer also has to ask the opposite question, which direction
 * to continue in, and with what density it chose it.
 *
 * The lobes evaluate as Three r185's direct lighting does (`lights_physical_pars_fragment`, `BRDF_BlinnPhong`, the toon
 * ramp): GGX with its multiple scattering term, anisotropy, sheen and its energy compensation, a clearcoat on its own
 * normal, and an iridescent film blended into the Fresnel. They are sampled with the standard methods for each: cosine
 * for the diffuse lobe, the visible normal distribution (Heitz 2018) for GGX, anisotropic or not, and the Blinn-Phong
 * lobe for Phong, mixed by how much each lobe reflects.
 */
export const BSDF_WGSL = /* wgsl */ `
/** How a surface answers a direction: what it reflects, and how likely sampling was to have chosen it. */
struct BsdfEval {
  // The bidirectional distribution times the cosine at the surface, which is what a contribution is multiplied by.
  value: vec3<f32>,
  // Solid angle density of this direction under this renderer's own sampling, for the power heuristic.
  pdf: f32,
};

struct BsdfSample {
  direction: vec3<f32>,
  // The distribution over its own density: what the path's throughput is multiplied by.
  weight: vec3<f32>,
  pdf: f32,
  valid: bool,
  // The diffuse lobe chose the direction, rather than a specular one: which of the denoiser's two signals it feeds.
  diffuse: bool,
};

fn luminanceOf(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn max3(color: vec3<f32>) -> f32 {
  return max(color.r, max(color.g, color.b));
}

/** Schlick's approximation, as in Three: the reflectance always reaches one at a grazing angle. */
fn fresnelSchlick(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
  return schlick3(f0, 1.0, cosine);
}

/** Reflectance at normal incidence: the dielectric's own, turned towards the base colour as the surface turns metal. */
fn surfaceF0(surface: Surface) -> vec3<f32> {
  return mix(surface.specularColor, surface.albedo, surface.metalness);
}

/** GGX, with alpha the square of the roughness, as Three parameterises it. */
fn ggxDistribution(alpha2: f32, nh: f32) -> f32 {
  let denominator = nh * nh * (alpha2 - 1.0) + 1.0;

  return alpha2 / (PI * denominator * denominator);
}

/** Smith's height correlated visibility, which already carries the 1/(4 nl nv) the lobe would otherwise need. */
fn ggxVisibility(alpha2: f32, nl: f32, nv: f32) -> f32 {
  return 0.5 / (nl * sqrt(nv * nv * (1.0 - alpha2) + alpha2) + nv * sqrt(nl * nl * (1.0 - alpha2) + alpha2) + 1e-7);
}

/** The separable masking term, which is the one the visible normal sampling is built on. */
fn ggxMasking(alpha2: f32, nv: f32) -> f32 {
  return 2.0 * nv / (nv + sqrt(alpha2 + (1.0 - alpha2) * nv * nv));
}

/** Karis' analytic fit of the GGX directional albedo (Three's DFGApprox): the scale and bias applied to F0 and F90. */
fn dfgApprox(roughness: f32, nv: f32) -> vec2<f32> {
  let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022) + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;

  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

/** Three's IBLSheenBRDF: the Charlie sheen lobe integrated over the hemisphere. */
fn sheenAlbedo(cosine: f32, roughness: f32) -> f32 {
  let r2 = roughness * roughness;
  let inverse = 1.0 / (roughness + 0.1);
  let a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * inverse;
  let b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * inverse;

  return clamp(exp(a * clamp(cosine, 0.0, 1.0) + b), 0.0, 1.0);
}

/** Three's D_Charlie times V_Neubelt. */
fn sheenLobe(roughness: f32, nh: f32, nv: f32, nl: f32) -> f32 {
  let inverse = 1.0 / (roughness * roughness);
  let sin2h = max(1.0 - nh * nh, 0.0078125);
  let distribution = (2.0 + inverse) * pow(sin2h, inverse * 0.5) / (2.0 * PI);

  return distribution * clamp(1.0 / (4.0 * (nl + nv - nl * nv)), 0.0, 1.0);
}

/** The frame the specular lobe is defined in: Three's anisotropy direction when there is one, any frame otherwise. */
fn specularBasis(surface: Surface) -> mat3x3<f32> {
  let normal = surface.normal;

  if (surface.anisotropy > 0.0) {
    let t = surface.anisotropyT - normal * dot(normal, surface.anisotropyT);
    let tangent = normalize(t);

    if (all(tangent == tangent) && dot(t, t) > 1e-12) {
      return mat3x3<f32>(tangent, cross(normal, tangent), normal);
    }
  }

  return orthonormalBasis(normal);
}

/** The GGX roughness along the frame's two axes: Three's alphaT along the anisotropy, alpha across it. */
fn specularAlphas(surface: Surface) -> vec2<f32> {
  let alpha = surface.roughness * surface.roughness;

  if (surface.anisotropy > 0.0) {
    return vec2<f32>(max(surface.alphaT, 1e-4), max(alpha, 1e-4));
  }

  return vec2<f32>(max(alpha, 1e-4));
}

/** Anisotropic GGX in the lobe's own frame; with equal alphas it is exactly the isotropic distribution. */
fn ggxAnisotropic(h: vec3<f32>, alphas: vec2<f32>) -> f32 {
  let x = h.x / alphas.x;
  let y = h.y / alphas.y;
  let d = x * x + y * y + h.z * h.z;

  return 1.0 / (PI * alphas.x * alphas.y * d * d);
}

/** Smith's masking of one direction, in the lobe's frame. */
fn ggxMaskingAnisotropic(v: vec3<f32>, alphas: vec2<f32>) -> f32 {
  let projected = alphas.x * alphas.x * v.x * v.x + alphas.y * alphas.y * v.y * v.y;

  return 2.0 * v.z / (v.z + sqrt(v.z * v.z + projected));
}

/** Three's V_GGX_SmithCorrelated_Anisotropic, in the lobe's frame. */
fn ggxVisibilityAnisotropic(l: vec3<f32>, v: vec3<f32>, alphas: vec2<f32>) -> f32 {
  let gv = l.z * length(vec3<f32>(alphas.x * v.x, alphas.y * v.y, v.z));
  let gl = v.z * length(vec3<f32>(alphas.x * l.x, alphas.y * l.y, l.z));

  return 0.5 / max(gv + gl, 1e-7);
}

/**
 * A half vector drawn from the distribution of normals the view direction can actually see (Heitz 2018), which is what
 * keeps the specular lobe from wasting samples on microfacets facing away. The view is in the lobe's frame.
 */
fn sampleVisibleNormal(view: vec3<f32>, alphas: vec2<f32>, u: vec2<f32>) -> vec3<f32> {
  let stretched = normalize(vec3<f32>(alphas.x * view.x, alphas.y * view.y, view.z));
  let lengthSquared = stretched.x * stretched.x + stretched.y * stretched.y;
  let t1 = select(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-stretched.y, stretched.x, 0.0) * inverseSqrt(max(lengthSquared, 1e-20)),
    lengthSquared > 1e-20);
  let t2 = cross(stretched, t1);
  let radius = sqrt(u.x);
  let angle = 2.0 * PI * u.y;
  let px = radius * cos(angle);
  let raw = radius * sin(angle);
  let lift = 0.5 * (1.0 + stretched.z);
  let py = (1.0 - lift) * sqrt(max(0.0, 1.0 - px * px)) + lift * raw;
  let normal = t1 * px + t2 * py + stretched * sqrt(max(0.0, 1.0 - px * px - py * py));

  return normalize(vec3<f32>(alphas.x * normal.x, alphas.y * normal.y, max(normal.z, 1e-6)));
}

/** What the clearcoat takes back from everything under it, seen from this view: Three's (1 - clearcoat * Fcc). */
fn coatTransmittance(surface: Surface, view: vec3<f32>) -> f32 {
  if (surface.clearcoat <= 0.0) {
    return 1.0;
  }

  let nv = clamp(dot(surface.clearcoatNormal, view), 0.0, 1.0);

  return 1.0 - surface.clearcoat * schlick3(vec3<f32>(0.04), 1.0, nv).x;
}

/**
 * A point inside a volume, where light scatters by the medium's phase function instead of off a surface. Only the
 * integrator makes one: its normal faces back along the ray, its albedo is the medium's, its anisotropy the phase's.
 */
const KIND_VOLUME: u32 = 100u;

/**
 * Share of the diffuse lobe's directions drawn on the far side of a surface that lets light through it diffusely, as a
 * grass blade does: by how much of the diffuse light each side accounts for.
 */
fn translucentShare(surface: Surface) -> f32 {
  if (!GRASS) {
    return 0.0;
  }

  let through = luminanceOf(surface.translucent);

  if (through <= 0.0) {
    return 0.0;
  }

  let off = luminanceOf(surface.albedo * ((1.0 - surface.metalness) * (1.0 - surface.subsurface)));

  return through / max(through + off, 1e-6);
}

/** How often each lobe is chosen, as (diffuse, specular, clearcoat): by how much of the reflection each accounts for. */
fn lobeWeights(surface: Surface, view: vec3<f32>) -> vec3<f32> {
  let nv = max(dot(surface.normal, view), 1e-4);

  if (surface.kind == KIND_PHONG) {
    let diffuse = luminanceOf(surface.albedo) + 0.02;
    let specular = luminanceOf(surface.specularColor) * surface.specularStrength * (surface.shininess * 0.125 + 1.0) + 0.02;

    return vec3<f32>(diffuse, specular, 0.0) / (diffuse + specular);
  }

  if ((surface.flags & MATERIAL_PBR) == 0u) {
    return vec3<f32>(1.0, 0.0, 0.0);
  }

  let base = coatTransmittance(surface, view);
  // The floors keep every lobe reachable: the diffuse one also carries the sheen and the multiple scattering term.
  let diffuse = (luminanceOf(surface.albedo * ((1.0 - surface.metalness) * (1.0 - surface.subsurface)))
    + max3(surface.sheenColor) + select(0.0, luminanceOf(surface.translucent), GRASS)) * base + 0.02;
  let specular = luminanceOf(schlick3(surface.specularColor, surface.specularF90, nv)
    * (1.0 - surface.metalness) + surface.albedo * surface.metalness) * base + 0.05;
  let coat = select(0.0, (1.0 - base) + 0.02, surface.clearcoat > 0.0);
  let total = diffuse + specular + coat;

  return vec3<f32>(diffuse, specular, coat) / total;
}

/** Three's physical direct lighting for one light direction, times the cosine. */
fn physicalValue(surface: Surface, view: vec3<f32>, light: vec3<f32>) -> vec3<f32> {
  let normal = surface.normal;
  let nl = dot(normal, light);
  let nv = max(dot(normal, view), 1e-4);
  var value = vec3<f32>(0.0);

  if (nl > 0.0) {
    let basis = specularBasis(surface);
    let alphas = specularAlphas(surface);
    let half = normalize(light + view);
    let vh = clamp(dot(view, half), 0.0, 1.0);
    let f0 = surfaceF0(surface);
    var fresnel = schlick3(f0, surface.specularF90, vh);

    if (surface.iridescence > 0.0) {
      fresnel = mix(fresnel, surface.iridescenceFresnel, surface.iridescence);
    }

    let localLight = vec3<f32>(dot(light, basis[0]), dot(light, basis[1]), nl);
    let localView = vec3<f32>(dot(view, basis[0]), dot(view, basis[1]), nv);
    let localHalf = vec3<f32>(dot(half, basis[0]), dot(half, basis[1]), max(dot(half, normal), 0.0));
    let single = fresnel * (ggxVisibilityAnisotropic(localLight, localView, alphas) * ggxAnisotropic(localHalf, alphas));

    // Three's BRDF_GGX_Multiscatter: the energy single scattering loses at high roughness, returned.
    let dfgV = dfgApprox(surface.roughness, nv);
    let dfgL = dfgApprox(surface.roughness, nl);
    let essV = f0 * dfgV.x + surface.specularF90 * dfgV.y;
    let essL = f0 * dfgL.x + surface.specularF90 * dfgL.y;
    let emsV = 1.0 - (dfgV.x + dfgV.y);
    let emsL = 1.0 - (dfgL.x + dfgL.y);
    let average = f0 + (vec3<f32>(1.0) - f0) * 0.047619;
    let multiple = essV * essL * average / (vec3<f32>(1.0) - emsV * emsL * average + 1e-7) * (emsV * emsL);

    // What scatters inside the material leaves it elsewhere, by the random walk, not from this point.
    var diffuse = surface.albedo * ((1.0 - surface.metalness) * (1.0 - surface.subsurface)) / PI;
    var specular = single + multiple;

    // Three's sheen: its lobe sits on top and takes back the energy it reflects from the layers below.
    if (any(surface.sheenColor > vec3<f32>(0.0))) {
      let energy = 1.0 - max3(surface.sheenColor)
        * max(sheenAlbedo(nv, surface.sheenRoughness), sheenAlbedo(nl, surface.sheenRoughness));

      diffuse = diffuse * energy;
      specular = specular * energy
        + surface.sheenColor * sheenLobe(surface.sheenRoughness, clamp(dot(normal, half), 0.0, 1.0), nv, nl);
    }

    value = (diffuse + specular) * nl;
  }

  if (surface.clearcoat > 0.0) {
    let coatNormal = surface.clearcoatNormal;
    value = value * coatTransmittance(surface, view);

    let nlCoat = dot(coatNormal, light);
    let nvCoat = max(dot(coatNormal, view), 1e-4);

    if (nlCoat > 0.0) {
      let half = normalize(light + view);
      let alpha = surface.clearcoatRoughness * surface.clearcoatRoughness;
      let alpha2 = alpha * alpha;
      let fresnel = schlick3(vec3<f32>(0.04), 1.0, clamp(dot(view, half), 0.0, 1.0)).x;
      let lobe = fresnel * ggxVisibility(alpha2, nlCoat, nvCoat) * ggxDistribution(alpha2, max(dot(coatNormal, half), 0.0));

      value = value + vec3<f32>(surface.clearcoat * lobe * nlCoat);
    }
  }

  // Light through a translucent surface, from its far side.
  if (GRASS && nl < 0.0) {
    value = value + surface.translucent / PI * (-nl);
  }

  return value;
}

/** Three's D_BlinnPhong times its implicit visibility of a quarter: the Phong highlight's distribution. */
fn blinnPhong(shininess: f32, nh: f32) -> f32 {
  return 0.25 * (shininess * 0.5 + 1.0) / PI * pow(max(nh, 0.0), shininess);
}

/** Density of the Blinn-Phong lobe's sampling, per solid angle of the light direction. */
fn blinnPhongPdf(shininess: f32, nh: f32, vh: f32) -> f32 {
  return (shininess + 1.0) / (2.0 * PI) * pow(max(nh, 0.0), shininess) / max(4.0 * vh, 1e-6);
}

/** Density with which the GGX lobe (VNDF) would have produced this light direction, in the lobe's frame. */
fn ggxPdf(localLight: vec3<f32>, localView: vec3<f32>, alphas: vec2<f32>) -> f32 {
  let half = normalize(localLight + localView);

  if (half.z <= 0.0 || localView.z <= 0.0) {
    return 0.0;
  }

  return ggxMaskingAnisotropic(localView, alphas) * ggxAnisotropic(half, alphas) / (4.0 * localView.z);
}

/** The mixture density of every lobe this surface samples with. */
fn bsdfPdf(surface: Surface, view: vec3<f32>, light: vec3<f32>) -> f32 {
  let normal = surface.normal;
  let nl = dot(normal, light);
  let weights = lobeWeights(surface, view);
  var pdf = weights.x * max(nl, 0.0) / PI;

  if (GRASS) {
    pdf = weights.x * mix(max(nl, 0.0), max(-nl, 0.0), translucentShare(surface)) / PI;
  }

  if (surface.kind == KIND_PHONG) {
    let half = normalize(light + view);

    return pdf + weights.y * blinnPhongPdf(surface.shininess, dot(normal, half), max(dot(view, half), 0.0));
  }

  if ((surface.flags & MATERIAL_PBR) == 0u) {
    return pdf;
  }

  let basis = specularBasis(surface);
  let alphas = specularAlphas(surface);

  pdf = pdf + weights.y * ggxPdf(
    vec3<f32>(dot(light, basis[0]), dot(light, basis[1]), nl),
    vec3<f32>(dot(view, basis[0]), dot(view, basis[1]), dot(view, normal)),
    alphas);

  if (weights.z > 0.0) {
    let coatBasis = orthonormalBasis(surface.clearcoatNormal);
    let alpha = max(surface.clearcoatRoughness * surface.clearcoatRoughness, 1e-4);

    pdf = pdf + weights.z * ggxPdf(
      vec3<f32>(dot(light, coatBasis[0]), dot(light, coatBasis[1]), dot(light, surface.clearcoatNormal)),
      vec3<f32>(dot(view, coatBasis[0]), dot(view, coatBasis[1]), dot(view, surface.clearcoatNormal)),
      vec2<f32>(alpha));
  }

  return pdf;
}

/**
 * What the surface sends towards the view direction, and the density of that direction. A Lambert material keeps
 * Three's Lambert term alone, which is all its own shading has; a toon material reads its ramp in place of the cosine.
 */
fn evaluateBsdf(surface: Surface, view: vec3<f32>, light: vec3<f32>) -> BsdfEval {
  let normal = surface.normal;
  var result: BsdfEval;
  result.value = vec3<f32>(0.0);
  result.pdf = 0.0;

  let nl = dot(normal, light);
  let nv = dot(normal, view);

  // In a medium every direction is open: the phase function stands for the lobes, and samples itself exactly.
  if (surface.kind == KIND_VOLUME) {
    let phase = henyeyGreenstein(dot(-view, light), surface.anisotropy);

    result.value = surface.albedo * phase;
    result.pdf = phase;

    return result;
  }

  if (nv <= 0.0) {
    return result;
  }

  if (surface.kind == KIND_TOON) {
    // Three's toon lighting reads its ramp across the whole sphere of directions, the side facing away included.
    result.value = surface.albedo / PI * toonRamp(surface.gradientMap, nl);
    result.pdf = max(nl, 0.0) / PI;

    return result;
  }

  // A clearcoat may face a light its base turns away from, so only a surface without one stops here.
  if (nl <= 0.0 && surface.clearcoat <= 0.0 && !(GRASS && any(surface.translucent > vec3<f32>(0.0)))) {
    return result;
  }

  if ((surface.flags & MATERIAL_PBR) != 0u) {
    result.value = physicalValue(surface, view, light);
  } else if (surface.kind == KIND_PHONG) {
    let half = normalize(light + view);
    let fresnel = schlick3(surface.specularColor, 1.0, clamp(dot(view, half), 0.0, 1.0));
    let specular = fresnel * blinnPhong(surface.shininess, dot(normal, half)) * surface.specularStrength;

    result.value = (surface.albedo / PI + specular) * max(nl, 0.0);
  } else {
    result.value = surface.albedo / PI * max(nl, 0.0);
  }

  result.pdf = bsdfPdf(surface, view, light);

  return result;
}

/**
 * Reflectance of a smooth dielectric interface, from both polarisations rather than Schlick's fit: refraction needs the
 * exact cut-off past which nothing gets through. eta is the index the ray leaves over the index it enters.
 */
fn fresnelDielectric(cosine: f32, eta: f32) -> f32 {
  let c = clamp(cosine, 0.0, 1.0);
  let sin2 = eta * eta * (1.0 - c * c);

  if (sin2 >= 1.0) {
    return 1.0;
  }

  let t = sqrt(1.0 - sin2);
  let s = (eta * c - t) / (eta * c + t);
  let p = (c - eta * t) / (c + eta * t);

  return 0.5 * (s * s + p * p);
}

/** Where a transmissive surface sends the path: off it, or through it. */
struct DielectricSample {
  direction: vec3<f32>,
  weight: vec3<f32>,
  // The path went through the surface instead of reflecting off it.
  crossed: bool,
  // It went through without bending, so the density that found the next hit is still the one before this surface.
  straight: bool,
  valid: bool,
};

/**
 * The transmissive share of a physical material. Reflection and refraction are chosen in proportion to the Fresnel
 * reflectance on a microfacet drawn from the visible normals, which leaves only the masking of the outgoing direction
 * in the weight. Light is tinted by the base colour once, on the way in, which is where Three applies it.
 *
 * A thin wall (zero thickness, Three's default) bends nothing: what is not reflected carries straight on, exactly as a
 * shadow ray crossing it does in thinCrossing, so the two estimators agree on a window. ior is the index this path
 * sees, which dispersion makes depend on the colour the path carries.
 */
fn sampleDielectric(surface: Surface, view: vec3<f32>, ior: f32, lobe: f32, u: vec2<f32>) -> DielectricSample {
  let normal = surface.normal;
  var result: DielectricSample;
  result.valid = false;
  result.crossed = false;
  result.straight = false;
  result.weight = vec3<f32>(0.0);

  let nv = dot(normal, view);

  if (nv <= 0.0) {
    return result;
  }

  let alpha = max(surface.roughness * surface.roughness, 1e-4);
  let alpha2 = alpha * alpha;
  let basis = orthonormalBasis(normal);
  let localView = vec3<f32>(dot(view, basis[0]), dot(view, basis[1]), nv);
  let localHalf = sampleVisibleNormal(localView, vec2<f32>(alpha), u);
  let half = basis[0] * localHalf.x + basis[1] * localHalf.y + normal * localHalf.z;

  if (surface.thickness <= 0.0) {
    let reflectance = fresnelDielectric(abs(dot(surface.facing, view)), 1.0 / ior);

    if (lobe >= reflectance) {
      result.direction = -view;
      result.weight = surface.albedo;
      result.crossed = true;
      result.straight = true;
      result.valid = true;

      return result;
    }

    result.direction = reflect(-view, half);
  } else {
    let eta = select(ior, 1.0 / ior, surface.front);
    let refracted = refract(-view, half, eta);
    // Past the critical angle refract() returns nothing, and everything reflects.
    let reflectance = select(fresnelDielectric(dot(view, half), eta), 1.0, all(refracted == vec3<f32>(0.0)));

    if (lobe >= reflectance) {
      result.direction = normalize(refracted);

      let no = -dot(normal, result.direction);

      if (no <= 0.0) {
        return result;
      }

      result.weight = vec3<f32>(ggxMasking(alpha2, no)) * select(vec3<f32>(1.0), surface.albedo, surface.front);
      result.crossed = true;
      result.valid = true;

      return result;
    }

    result.direction = reflect(-view, half);
  }

  let nl = dot(normal, result.direction);

  if (nl <= 0.0) {
    return result;
  }

  result.weight = vec3<f32>(ggxMasking(alpha2, nl));
  result.valid = true;

  return result;
}

/** A half vector around the normal distributed as cos^s, which is the Blinn-Phong lobe's own shape. */
fn sampleBlinnPhongHalf(normal: vec3<f32>, shininess: f32, u: vec2<f32>) -> vec3<f32> {
  let cosine = pow(u.x, 1.0 / (shininess + 1.0));
  let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
  let angle = 2.0 * PI * u.y;

  return normalize(orthonormalBasis(normal) * vec3<f32>(sine * cos(angle), sine * sin(angle), cosine));
}

/** Which way the path carries on, and what its throughput is multiplied by. */
fn sampleBsdf(surface: Surface, view: vec3<f32>, lobe: f32, u: vec2<f32>) -> BsdfSample {
  let normal = surface.normal;
  var result: BsdfSample;
  result.valid = false;
  result.weight = vec3<f32>(0.0);
  result.pdf = 0.0;
  result.diffuse = true;

  if (surface.kind == KIND_VOLUME) {
    result.direction = sampleHenyeyGreenstein(-view, surface.anisotropy, u);
    result.weight = surface.albedo;
    result.pdf = henyeyGreenstein(dot(-view, result.direction), surface.anisotropy);
    result.valid = true;

    return result;
  }

  if (dot(normal, view) <= 0.0) {
    return result;
  }

  // Lambert, and the bounce of a toon surface: Three lights a toon surface's indirect light with a plain Lambert term.
  if ((surface.flags & MATERIAL_PBR) == 0u && surface.kind != KIND_PHONG) {
    result.direction = cosineDirection(normal, u);
    // The cosine weight cancels the Lambert cosine exactly, which leaves the albedo alone.
    result.weight = surface.albedo;
    result.pdf = max(dot(normal, result.direction), 0.0) / PI;
    result.valid = result.pdf > 0.0;

    return result;
  }

  let weights = lobeWeights(surface, view);

  result.diffuse = lobe < weights.x;

  if (lobe < weights.x) {
    // A translucent surface sends that share of its diffuse light through, to the far side.
    let through = GRASS && lobe < weights.x * translucentShare(surface);

    result.direction = cosineDirection(select(normal, -normal, through), u);
  } else if (surface.kind == KIND_PHONG) {
    result.direction = reflect(-view, sampleBlinnPhongHalf(normal, surface.shininess, u));
  } else if (lobe < weights.x + weights.y) {
    let basis = specularBasis(surface);
    // The lobe is drawn in its own frame, so the view direction goes into that frame and the result comes back out.
    let localView = vec3<f32>(dot(view, basis[0]), dot(view, basis[1]), dot(view, normal));
    let localHalf = sampleVisibleNormal(localView, specularAlphas(surface), u);
    let half = basis * localHalf;

    result.direction = reflect(-view, half);
  } else {
    let coatNormal = surface.clearcoatNormal;
    let basis = orthonormalBasis(coatNormal);
    let alpha = max(surface.clearcoatRoughness * surface.clearcoatRoughness, 1e-4);
    let localView = vec3<f32>(dot(view, basis[0]), dot(view, basis[1]), dot(view, coatNormal));
    let half = basis * sampleVisibleNormal(localView, vec2<f32>(alpha), u);

    result.direction = reflect(-view, half);
  }

  let evaluated = evaluateBsdf(surface, view, result.direction);

  if (evaluated.pdf <= 0.0 || all(evaluated.value <= vec3<f32>(0.0))) {
    return result;
  }

  result.pdf = evaluated.pdf;
  result.weight = evaluated.value / evaluated.pdf;
  result.valid = true;

  return result;
}

/**
 * Only the diffuse part of evaluateBsdf, times the cosine: what of the light from this direction the denoiser counts
 * as diffuse. The rest of the value is specular.
 */
fn diffuseValue(surface: Surface, view: vec3<f32>, light: vec3<f32>) -> vec3<f32> {
  let normal = surface.normal;
  let nl = dot(normal, light);
  let nv = max(dot(normal, view), 1e-4);

  if (surface.kind == KIND_TOON) {
    return surface.albedo / PI * toonRamp(surface.gradientMap, nl);
  }

  // Light scattered in a medium is all diffuse, as the denoiser sees it.
  if (surface.kind == KIND_VOLUME) {
    return surface.albedo * henyeyGreenstein(dot(-view, light), surface.anisotropy);
  }

  if (GRASS && nl < 0.0) {
    return surface.translucent / PI * (-nl);
  }

  if (nl <= 0.0) {
    return vec3<f32>(0.0);
  }

  if ((surface.flags & MATERIAL_PBR) == 0u) {
    return surface.albedo / PI * nl;
  }

  var diffuse = surface.albedo * ((1.0 - surface.metalness) * (1.0 - surface.subsurface)) / PI;

  if (any(surface.sheenColor > vec3<f32>(0.0))) {
    diffuse = diffuse * (1.0 - max3(surface.sheenColor)
      * max(sheenAlbedo(nv, surface.sheenRoughness), sheenAlbedo(nl, surface.sheenRoughness)));
  }

  return diffuse * nl * coatTransmittance(surface, view);
}

/** How much of the light a surface's specular lobes reflect in all, seen from this view: what that signal is divided by. */
fn specularAlbedo(surface: Surface, view: vec3<f32>) -> vec3<f32> {
  let nv = clamp(dot(surface.normal, view), 0.0, 1.0);

  if (surface.kind == KIND_PHONG) {
    return surface.specularColor * surface.specularStrength;
  }

  if ((surface.flags & MATERIAL_PBR) == 0u) {
    return vec3<f32>(0.0);
  }

  let dfg = dfgApprox(surface.roughness, nv);
  var albedo = surfaceF0(surface) * dfg.x + vec3<f32>(surface.specularF90 * dfg.y);

  if (surface.clearcoat > 0.0) {
    let coatDfg = dfgApprox(surface.clearcoatRoughness, clamp(dot(surface.clearcoatNormal, view), 0.0, 1.0));
    albedo = albedo * coatTransmittance(surface, view) + vec3<f32>(surface.clearcoat * (0.04 * coatDfg.x + coatDfg.y));
  }

  // What crosses a transmissive surface is the specular signal too, and it is not filtered by any colour of the surface.
  return mix(albedo, vec3<f32>(1.0), clamp(surface.transmission, 0.0, 1.0));
}
`;
