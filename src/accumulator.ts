/**
 * Where the samples pile up: one `vec4<f32>` per pixel, rgb summed radiance and w the number of samples that reached
 * it, then a second `vec4<f32>` per pixel whose x sums the coverage, which is what a transparent background reads as
 * alpha, y the depth of the first surface, z and w the squared luminance of the diffuse and the specular signal, which
 * give each its variance. Then the denoiser's guides and signals, each summed: the diffuse albedo with the roughness,
 * the normal with the distance, the diffuse radiance with the specular hit distance, and the specular albedo. Last comes
 * one `vec4<f32>` per row, set when a pass traced anything in that row: adaptive sampling reads it back to know which
 * rows still have pixels that have not converged. A storage buffer rather than a storage texture, because `rgba32float` only allows write access as a texture,
 * and the presentation pass has to read back what is already there.
 */
export class Accumulator {
  buffer: GPUBuffer;
  width: number;
  height: number;

  constructor(private readonly device: GPUDevice, width: number, height: number) {
    this.width = Math.max(1, Math.floor(width));
    this.height = Math.max(1, Math.floor(height));
    this.buffer = this.create();
  }

  get byteLength(): number {
    return (this.width * this.height * 6 + this.height) * 16;
  }

  /** Where the per-row flags start, in bytes. */
  get rowFlagsOffset(): number {
    return this.width * this.height * 96;
  }

  get rowFlagsLength(): number {
    return this.height * 16;
  }

  /** Reallocates only when the size actually changed; the caller clears it either way. */
  resize(width: number, height: number): boolean {
    const next = { width: Math.max(1, Math.floor(width)), height: Math.max(1, Math.floor(height)) };

    if (next.width === this.width && next.height === this.height) return false;

    this.buffer.destroy();
    this.width = next.width;
    this.height = next.height;
    this.buffer = this.create();

    return true;
  }

  clear(encoder: GPUCommandEncoder) {
    encoder.clearBuffer(this.buffer);
  }

  destroy() {
    this.buffer.destroy();
  }

  private create(): GPUBuffer {
    return this.device.createBuffer({
      label: "accumulation",
      size: this.byteLength,
      // clearBuffer is a copy into the buffer: without COPY_DST it fails validation and clears nothing.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }
}
