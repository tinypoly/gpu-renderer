import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import type { RenderEnvironment, SerializedTexture } from "./upstream.js";

/**
 * The light that surrounds the scene. A gradient is three numbers the shader evaluates directly; an equirectangular
 * image is a texture plus the tables that let a shadow ray aim at its bright parts.
 *
 * Without those tables a sun in an image is exactly the problem direct light sampling exists to solve: a tiny, very
 * bright region that a cosine weighted bounce almost never finds. The construction is the one in `environment.js` of
 * the CPU renderer: a cumulative distribution over rows weighted by luminance and the sine of the colatitude, then one
 * over the columns of each row.
 */
export interface PackedEnvironment {
  /** The image itself, ready to join the scene's own textures. */
  texture: SerializedTexture;
  /** Rows, then each row's columns, then the luminance of every texel: `2 * width * height + height` floats. */
  distribution: Float32Array;
  width: number;
  height: number;
  /** The sum of luminance times the sine of the colatitude, which turns a texel's share into a density. */
  total: number;
}

/** The grid the distribution is built on, as in the CPU renderer. Finer than this buys nothing but memory. */
const GRID_WIDTH = 256;
const GRID_HEIGHT = 128;

/** How large an image is kept for the sharp lookup. A full resolution sky costs more memory than it adds detail here. */
const MAX_WIDTH = 1024;

const luminance = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** An area averaged reduction to `width` by `height`, which is what the distribution is built from. */
function reduce(source: Float32Array, sourceWidth: number, sourceHeight: number, width: number, height: number) {
  const target = new Float32Array(width * height * 3);
  const scaleX = sourceWidth / width, scaleY = sourceHeight / height;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const x0 = Math.floor(x * scaleX), x1 = Math.max(x0 + 1, Math.ceil((x + 1) * scaleX));
      const y0 = Math.floor(y * scaleY), y1 = Math.max(y0 + 1, Math.ceil((y + 1) * scaleY));
      let r = 0, g = 0, b = 0, n = 0;

      for (let sy = y0; sy < Math.min(sourceHeight, y1); sy++) {
        for (let sx = x0; sx < Math.min(sourceWidth, x1); sx++) {
          const at = (sy * sourceWidth + sx) * 4;

          r += source[at];
          g += source[at + 1];
          b += source[at + 2];
          n++;
        }
      }

      const at = (y * width + x) * 3;
      const weight = n || 1;

      target[at] = r / weight;
      target[at + 1] = g / weight;
      target[at + 2] = b / weight;
    }
  }

  return target;
}

/** Never picked less often than this share of the average, so no direction is left without a density. */
const FLOOR = 0.02;

/**
 * The luminance a direction is picked by. It is the neighbourhood's brightest rather than the texel's own, because the
 * image is read back with a bilinear filter: that filter spreads a sun into the texels around it, and a direction the
 * distribution gave no chance of being picked would arrive through the bounce alone, as a firefly. Taking the maximum
 * over the neighbours keeps the density above what the filter can produce there.
 */
function weights(grid: Float32Array, width: number, height: number): Float32Array {
  const raw = new Float32Array(width * height);
  const spread = new Float32Array(width * height);
  let sum = 0;

  for (let index = 0; index < width * height; index++) {
    raw[index] = luminance(grid[index * 3], grid[index * 3 + 1], grid[index * 3 + 2]);
    sum += raw[index];
  }

  const floor = sum / (width * height) * FLOOR;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let peak = 0;

      for (let dy = -1; dy <= 1; dy++) {
        const row = Math.min(Math.max(y + dy, 0), height - 1);

        for (let dx = -1; dx <= 1; dx++) {
          // Longitude wraps around, so the texel left of the first column is the last one.
          peak = Math.max(peak, raw[row * width + (x + dx + width) % width]);
        }
      }

      spread[y * width + x] = peak + floor;
    }
  }

  return spread;
}

/**
 * Cumulative distributions over the reduced image. A row is weighted by the sine of its colatitude, because a row near
 * a pole covers far less of the sphere than one at the equator.
 */
export function environmentDistribution(grid: Float32Array, width: number, height: number) {
  const rows = new Float32Array(height);
  const columns = new Float32Array(width * height);
  const values = weights(grid, width, height);
  let total = 0;

  for (let y = 0; y < height; y++) {
    const sine = Math.sin((y + 0.5) / height * Math.PI);
    let row = 0;

    for (let x = 0; x < width; x++) {
      const index = y * width + x;

      row += values[index] * sine;
      columns[index] = row;
    }

    for (let x = 0; x < width; x++) {
      const index = y * width + x;

      // A row with nothing in it is still chosen uniformly, so a search over it always terminates somewhere.
      columns[index] = row > 0 ? columns[index] / row : (x + 1) / width;
    }

    total += row;
    rows[y] = total;
  }

  if (!(total > 0)) return null;

  for (let y = 0; y < height; y++) rows[y] /= total;

  const distribution = new Float32Array(height + width * height * 2);

  distribution.set(rows, 0);
  distribution.set(columns, height);
  distribution.set(values, height + width * height);

  return { distribution, total };
}

/**
 * The distribution of an image with no light in it. Its total is zero, which the shader reads as nothing to sample;
 * the rows and columns are uniform only so that the buffer keeps its layout.
 */
function darkDistribution(width: number, height: number) {
  const distribution = new Float32Array(height + width * height * 2);

  for (let y = 0; y < height; y++) {
    distribution[y] = (y + 1) / height;

    for (let x = 0; x < width; x++) distribution[height + y * width + x] = (x + 1) / width;
  }

  return { distribution, total: 0 };
}

/**
 * Loads an equirectangular image and builds everything the shader needs from it. Returns null for an environment that
 * is not an image, or one whose file could not be read, which leaves the renderer on its gradient.
 */
export async function loadEnvironment(environment: RenderEnvironment): Promise<PackedEnvironment | null> {
  if (environment.kind !== "hdri" || !environment.url) return null;

  const loaded = await new Promise<{ data: Float32Array; width: number; height: number } | null>(resolve => {
    new HDRLoader()
      .setDataType(1015 as never)
      .load(
        environment.url,
        texture => {
          const image = texture.image as { data: Float32Array; width: number; height: number } | undefined;

          resolve(image?.data instanceof Float32Array ? image : null);
          texture.dispose();
        },
        undefined,
        () => resolve(null));
  });

  if (!loaded || loaded.width < 1 || loaded.height < 1) return null;

  return environmentFromImage(loaded.data, loaded.width, loaded.height);
}

/**
 * An equirectangular image already in memory, linear RGBA floats with the zenith in the first row, made into an
 * environment: a procedural sky baked by the CPU renderer's interpreter arrives this way.
 */
export function environmentFromImage(
  data: Float32Array, sourceWidth: number, sourceHeight: number, id = "environment"): PackedEnvironment | null {
  const loaded = { data, width: sourceWidth, height: sourceHeight };

  if (loaded.width < 1 || loaded.height < 1 || data.length < loaded.width * loaded.height * 4) return null;

  // The sharp lookup keeps at most MAX_WIDTH across; the distribution is built on a much coarser grid either way.
  const width = Math.min(loaded.width, MAX_WIDTH);
  const height = Math.max(1, Math.round(loaded.height * (width / loaded.width)));

  const scaled = width === loaded.width
    ? loaded.data
    : (() => {
      const reduced = reduce(loaded.data, loaded.width, loaded.height, width, height);
      const rgba = new Float32Array(width * height * 4);

      for (let texel = 0; texel < width * height; texel++) {
        rgba[texel * 4] = reduced[texel * 3];
        rgba[texel * 4 + 1] = reduced[texel * 3 + 1];
        rgba[texel * 4 + 2] = reduced[texel * 3 + 2];
        rgba[texel * 4 + 3] = 1;
      }

      return rgba;
    })();

  const gridWidth = Math.min(GRID_WIDTH, width);
  const gridHeight = Math.min(GRID_HEIGHT, height);

  // A black image is still an environment: a night sky lights nothing, and it must not give way to the default one.
  const built = environmentDistribution(reduce(scaled, width, height, gridWidth, gridHeight), gridWidth, gridHeight)
    ?? darkDistribution(gridWidth, gridHeight);

  const texture: SerializedTexture = {
    id,
    width,
    height,
    data: scaled,
    // An HDR file is linear already, and the loader hands back floats.
    colorSpace: "srgb-linear",
    // Longitude wraps around; latitude stops at the poles.
    wrapS: 1000,
    wrapT: 1001,
    matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    flipY: false,
    channel: 0,
    nearest: false,
  };

  return { texture, distribution: built.distribution, width: gridWidth, height: gridHeight, total: built.total };
}
