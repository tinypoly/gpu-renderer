/**
 * The sample generator: a Sobol sequence with hash based Owen scrambling (Burley, "Practical Hash-based Owen
 * Scrambling", 2020). Every pixel draws its own scrambled sequence, decorrelated from its neighbours, and inside a pixel
 * the samples are stratified in every pair of dimensions, so the estimate converges faster than with independent
 * random numbers. Only the sample index, the pixel and a dimension counter feed the generator, so a pixel draws the same
 * numbers whichever dispatch happens to cover it and the image does not depend on the tiling.
 */
export const RANDOM_WGSL = /* wgsl */ `
fn pcgHash(value: u32) -> u32 {
  let state = value * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

  return (word >> 22u) ^ word;
}

struct PathSampler {
  seed: u32,
  index: u32,
  dimension: u32,
};

fn makeSampler(pixel: vec2<u32>, index: u32) -> PathSampler {
  var rng: PathSampler;
  rng.seed = pcgHash(pixel.x * 1973u + pixel.y * 9277u + 1u);
  rng.index = index;
  rng.dimension = 0u;

  return rng;
}

/**
 * The Laine-Karras hash, which permutes the high bits in a way that nests: what Owen scrambling asks for. Every
 * multiplier has to be even, or a bit stops depending on itself and the map is no longer a permutation.
 */
fn laineKarras(value: u32, seed: u32) -> u32 {
  var x = value + seed;
  x = x ^ (x * 0x6c50b47cu);
  x = x ^ (x * 0xb82f1e52u);
  x = x ^ (x * 0xc7afe784u);
  x = x ^ (x * 0x8d22f6e6u);

  return x;
}

fn nestedScramble(value: u32, seed: u32) -> u32 {
  return reverseBits(laineKarras(reverseBits(value), seed));
}

/**
 * The first two Sobol dimensions, as fixed point fractions. The second is the index times Pascal's triangle mod 2,
 * which is linear in the bits of the index, so it is read one byte at a time from the tables texturePack.ts writes
 * instead of folded in one bit at a time: four reads where the scrambled index made it thirty two steps.
 */
fn sobol2(index: u32) -> vec2<u32> {
  let second = textures[SOBOL_TABLE + (index & 0xffu)]
    ^ textures[SOBOL_TABLE + 256u + ((index >> 8u) & 0xffu)]
    ^ textures[SOBOL_TABLE + 512u + ((index >> 16u) & 0xffu)]
    ^ textures[SOBOL_TABLE + 768u + (index >> 24u)];

  return vec2<u32>(reverseBits(index), second);
}

fn toUnit(value: u32) -> f32 {
  return min(f32(value) * 2.3283064365386963e-10, 0.99999994);
}

/**
 * The next pair of dimensions, in [0, 1). The sample index is shuffled per pair as well, which pads the two dimensional
 * sequence out to as many dimensions as a path needs without correlating them.
 */
fn sample2D(rng: ptr<function, PathSampler>) -> vec2<f32> {
  let pair = (*rng).seed + (*rng).dimension * 2654435761u;
  (*rng).dimension = (*rng).dimension + 1u;

  let shuffled = nestedScramble((*rng).index, pcgHash(pair));
  let point = sobol2(shuffled);

  return vec2<f32>(
    toUnit(nestedScramble(point.x, pcgHash(pair + 0x9e3779b9u))),
    toUnit(nestedScramble(point.y, pcgHash(pair + 0x3c6ef372u))));
}
`;
