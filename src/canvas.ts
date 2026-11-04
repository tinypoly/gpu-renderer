import type { GpuRenderer } from "./renderer.js";

/**
 * Sends the renderer's output to a canvas. It configures the WebGPU context, sizes the canvas to the render
 * resolution and presents after every submit, so the image refines in place while it converges.
 *
 * Returns the function that detaches it.
 */
export function attachCanvas(renderer: GpuRenderer, canvas: HTMLCanvasElement): () => void {
  renderer.setOutputCanvas(canvas);

  return () => renderer.setOutputCanvas(null);
}
