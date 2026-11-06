import { DEFAULT_RENDER_SETTINGS, mergeSettings, type RenderSettings } from "./upstream.js";

/**
 * The settings this renderer reads beyond the CPU renderer's own: what makes a path traced image look photographed.
 * They ride in the same object as the shared settings, so a host hands one object to either renderer; the CPU renderer
 * leaves the groups it does not know alone.
 */
export interface GpuRenderSettings extends Omit<RenderSettings, "globalIllumination"> {
  globalIllumination: RenderSettings["globalIllumination"] & {
    /**
     * Ends a path that carries little light by chance instead of by its bounce count, weighting the survivors up so
     * the mean stays exact. It is what makes a high bounce count affordable.
     */
    russianRoulette: boolean;
    /** Bounces every path takes before the roulette may end it: the first ones carry most of the light. */
    rouletteStart: number;
    /**
     * Picks the light to aim at by what it can deliver to the point being shaded (its power, distance and the way it
     * faces), through a tree over the scene's lights, instead of by its power alone. A small lamp near a surface then
     * gets the samples a large panel far away would have taken.
     */
    lightTree: boolean;
    /**
     * Inside a room, aims at the sky through its windows (every thin transmissive material) as well as by its
     * brightness, as Cycles' portals do, so daylight that enters through glass is found instead of stumbled upon.
     * A point counts as inside when something is above it; elsewhere the sky is sampled as before.
     */
    portals: boolean;
  };
  /**
   * The light a real lens and sensor scatter around anything bright: a soft halo that fades with distance. Energy
   * conserving, with no threshold, so it only shows where something is much brighter than its surroundings.
   */
  bloom: {
    enabled: boolean;
    /** Share of the light the halo takes, `0` to `1`. A real lens scatters a few percent. */
    intensity: number;
    /** How far the halo reaches, `0` to `1`: from a tight glow to one across the frame. */
    radius: number;
  };
  /**
   * The rays that stream from a bright light through a lens: diffraction at the edges of the aperture's blades. An
   * even number of blades makes as many rays, an odd number twice as many. Energy conserving like the bloom, so only
   * light far brighter than its surroundings shows them.
   */
  glare: {
    enabled: boolean;
    /** Share of the light the rays take, `0` to `1`. */
    intensity: number;
    /** Blades of the aperture, `3` to `16`. */
    blades: number;
    /** How far the rays reach, as a share of the frame's half diagonal. */
    length: number;
    /** Degrees the aperture is turned by. */
    rotation: number;
  };
  /** What an actual lens does to the image besides focusing it. */
  lens: {
    /**
     * Lateral chromatic aberration, `0` to `1`: red and blue land slightly apart towards the corners, as through any
     * real glass. `1` shifts them by half a percent of the frame's half diagonal at the corners.
     */
    chromaticAberration: number;
    /** Radial distortion: positive bulges straight lines outwards (barrel), negative pinches them (pincushion). */
    distortion: number;
  };
  /** A camera's exposure triangle and white balance. */
  camera: {
    /**
     * The contrast and saturation of AgX's look, `0` to `1`: `0` is its base curve, flat on purpose, and `1` a look
     * like Blender's Punchy, pivoted on middle grey so the image keeps its brightness. The other tone mappings ignore it.
     */
    contrast: number;
    /**
     * Exposes from `iso`, `shutterSpeed` and `depthOfField.aperture` as a camera would, with `exposure` as a
     * compensation on top. The scene's lights then have to be in physical units (lux, candela, nits).
     */
    physicalExposure: boolean;
    /**
     * Meters the image as a camera in automatic does and exposes it to middle grey, from the mean of the logarithms of
     * its luminance, weighed towards the centre: a bright window or sky cannot darken the rest. It takes the place of
     * `exposure` and of the physical exposure, with `exposureCompensation` on top.
     */
    autoExposure: boolean;
    /** Stops added to the metered exposure: positive brightens, as on a camera's dial. */
    exposureCompensation: number;
    /**
     * How much of a dark scene's darkness the automatic exposure keeps, `0` to `1`. The meter brings any image to
     * middle grey, so a night lit by a few lamps comes out as dusk; below a dim average this keeps that share of the
     * missing stops, as the eye does at night. `0` exposes every scene alike; `1` brightens nothing below that average.
     */
    darkAdaptation: number;
    iso: number;
    /** Seconds. */
    shutterSpeed: number;
    /**
     * The colour temperature the camera takes as white, in kelvin. `6500` is neutral; a lower value cools the image,
     * as a camera set for tungsten light does in daylight.
     */
    whiteBalance: number;
  };
  /** Light that enters a material and leaves it somewhere else: skin, wax, marble, leaves, milk. */
  subsurface: {
    /** Materials with `userData.cpuRenderer.subsurface` scatter light inside themselves. Off renders them opaque. */
    enabled: boolean;
    /**
     * Bounces at which light still walks inside such a material: past them it reflects off it as a diffuse surface of
     * the same colour. Light that has already bounced that often hardly shows the difference, and every walk costs as
     * much as many bounces.
     */
    depth: number;
  };
  /**
   * Aerial perspective: the air between the camera and what it sees, which fades distant things towards the colour of
   * the sky, bluer for clean air and greyer for haze. It needs a scene in metres to look right.
   */
  atmosphere: {
    enabled: boolean;
    /**
     * Meteorological visibility, in scene units: the distance at which a dark object fades to 2 percent contrast.
     * Twenty kilometres is a clear day, two a hazy one.
     */
    visibility: number;
    /** `0` is clean air, scattering blue far more than red (Rayleigh); `1` is haze, scattering every colour alike. */
    haze: number;
    /** The colour the air glows with, `#rrggbb`; empty takes the average of the environment's light. */
    color: string;
  };
  /** Edges rounded at shading time, where a material asks for it with `userData.cpuRenderer.bevel`. */
  bevel: {
    enabled: boolean;
  };
  /** Meshes filled with a medium: smoke, fog banks, murky water, a beam of light through dust. */
  volumes: {
    /** Meshes whose material has `userData.cpuRenderer.volume` scatter light inside. Off leaves them out. */
    enabled: boolean;
  };
}

/** Any subset of the settings, group by group: `{ bloom: { intensity: 0.1 } }` keeps the other fields. */
export type GpuRenderSettingsInput = {
  [K in keyof GpuRenderSettings]?: GpuRenderSettings[K] extends object
    ? Partial<GpuRenderSettings[K]>
    : GpuRenderSettings[K];
};

/**
 * The CPU renderer's defaults, with what a photograph has turned on: paths long enough for light to fill a room, a
 * film's tone curve, a lens's halo, rays and colour fringes, the air's haze, and materials and volumes that scatter
 * light inside where the scene asks for it.
 */
export const DEFAULT_GPU_RENDER_SETTINGS: Readonly<GpuRenderSettings> = Object.freeze(mergeGpuSettings(
  {
    ...DEFAULT_RENDER_SETTINGS,
    // AgX rolls highlights off as film does, where ACES pushes saturated colours to the gamut's edge.
    toneMapping: "agx",
    globalIllumination: { ...DEFAULT_RENDER_SETTINGS.globalIllumination, russianRoulette: true, rouletteStart: 1,
      lightTree: true, portals: true },
    bloom: { enabled: true, intensity: 0.04, radius: 0.65 },
    glare: { enabled: true, intensity: 0.06, blades: 6, length: 0.25, rotation: 15 },
    lens: { chromaticAberration: 0.05, distortion: 0 },
    camera: {
      contrast: 1, physicalExposure: false, autoExposure: false, exposureCompensation: 0, darkAdaptation: 0, iso: 100,
      shutterSpeed: 1 / 125, whiteBalance: 6500,
    },
    subsurface: { enabled: true, depth: 2 },
    atmosphere: { enabled: true, visibility: 20000, haze: 0.6, color: "" },
    volumes: { enabled: true },
    bevel: { enabled: true },
  },
  // Cycles' indirect light clamp: a rare bright path through a mirror or off a tiny lamp is cut down to ten, which
  // trades a percent of indirect light for most of the fireflies it would leave.
  { globalIllumination: { bounces: 12, clamp: 10 } }));

/** Settings with `input` applied over `base`, group by group, as the CPU renderer's `mergeSettings` does. */
export function mergeGpuSettings(base: GpuRenderSettings, input: GpuRenderSettingsInput = {}): GpuRenderSettings {
  return mergeSettings(base as unknown as RenderSettings, input as never) as unknown as GpuRenderSettings;
}

/**
 * What a camera's exposure triangle multiplies the scene's luminance by: one over the luminance that saturates the
 * sensor, from the exposure value at ISO 100 (Lagarde and de Rousiers, "Moving Frostbite to PBR", 2014).
 */
export function physicalExposure(aperture: number, shutterSpeed: number, iso: number): number {
  const n = Math.max(aperture, 0.5), t = Math.max(shutterSpeed, 1e-6), s = Math.max(iso, 1);
  const ev100 = Math.log2(n * n / t * 100 / s);

  return 1 / (1.2 * 2 ** ev100);
}

/**
 * The exposure the image is multiplied by: the setting itself, or a camera's on top of it as compensation, or with
 * the automatic exposure only its compensation, the meter's own exposure being applied on the device.
 */
export function effectiveExposure(settings: GpuRenderSettings): number {
  const { camera, depthOfField, exposure } = settings;

  // The meter's exposure is applied on the device, by the lens pass; what is left here is the compensation.
  if (camera.autoExposure) return 2 ** (Number(camera.exposureCompensation) || 0);

  return camera.physicalExposure
    ? exposure * physicalExposure(depthOfField.aperture, camera.shutterSpeed, camera.iso)
    : exposure;
}

/** Chromaticity of a black body at this temperature (Kim et al.'s fit of the Planckian locus, 1667 K to 25000 K). */
function planckian(kelvin: number): [number, number] {
  const t = Math.min(25000, Math.max(1667, kelvin));

  const x = t <= 4000
    ? -0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2 + 0.8776956e3 / t + 0.17991
    : -3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2 + 0.2226347e3 / t + 0.24039;

  const y = t <= 2222
    ? -1.1063814 * x ** 3 - 1.3481102 * x ** 2 + 2.18555832 * x - 0.20219683
    : t <= 4000
      ? -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
      : 3.081758 * x ** 3 - 5.8733867 * x ** 2 + 3.75112997 * x - 0.37001483;

  return [x, y];
}

type Matrix = number[];

const multiply = (a: Matrix, b: Matrix): Matrix => Array.from({ length: 9 }, (_, k) => {
  const row = Math.floor(k / 3), column = k % 3;

  return a[row * 3] * b[column] + a[row * 3 + 1] * b[3 + column] + a[row * 3 + 2] * b[6 + column];
});

const apply = (m: Matrix, v: number[]) =>
  [0, 1, 2].map(row => m[row * 3] * v[0] + m[row * 3 + 1] * v[1] + m[row * 3 + 2] * v[2]);

const SRGB_TO_XYZ = [
  0.4124564, 0.3575761, 0.1804375,
  0.2126729, 0.7151522, 0.072175,
  0.0193339, 0.119192, 0.9503041];

const XYZ_TO_SRGB = [
  3.2404542, -1.5371385, -0.4985314,
  -0.969266, 1.8760108, 0.041556,
  0.0556434, -0.2040259, 1.0572252];

const BRADFORD = [
  0.8951, 0.2664, -0.1614,
  -0.7502, 1.7135, 0.0367,
  0.0389, -0.0685, 1.0296];

const BRADFORD_INVERSE = [
  0.9869929, -0.1470543, 0.1599627,
  0.4323053, 0.5183603, 0.0492912,
  -0.0085287, 0.0400428, 0.9684867];

/** The cone response to white light of this temperature. */
function coneWhite(kelvin: number): number[] {
  const [x, y] = planckian(kelvin);

  return apply(BRADFORD, [x / y, 1, (1 - x - y) / y]);
}

/** The temperature the image is neutral at: white balance at this value changes nothing. */
export const NEUTRAL_WHITE_BALANCE = 6500;

/**
 * The linear sRGB matrix, row by row, that a camera balanced for this temperature applies: a von Kries scaling of the
 * cone responses (Bradford), which renders light of that temperature as the white of 6500 K. Exactly the identity at
 * 6500 K, since both whites come from the same curve.
 */
export function whiteBalanceMatrix(kelvin: number): Matrix {
  const source = coneWhite(kelvin), target = coneWhite(NEUTRAL_WHITE_BALANCE);
  const scale = [target[0] / source[0], 0, 0, 0, target[1] / source[1], 0, 0, 0, target[2] / source[2]];

  return multiply(XYZ_TO_SRGB, multiply(BRADFORD_INVERSE, multiply(scale, multiply(BRADFORD, SRGB_TO_XYZ))));
}
