import type { CpuRenderer } from "@tinypoly/cpu-renderer";
import { afterEach, expect, it } from "vitest";
import { GpuRenderer, isAbortError } from "./renderer.js";

/** What a host calls on the CPU renderer. `pnpm typecheck` fails here the moment the two drift apart. */
type SharedApi = Pick<CpuRenderer,
  | "settings" | "rendering" | "paused" | "image"
  | "setScene" | "setCamera" | "setSettings" | "setEnvironment" | "setSize" | "setPixelRatio"
  | "render" | "pause" | "resume" | "cancel" | "dispose"
  | "on" | "once" | "off" | "removeAllListeners">;

export const sharedApi = (renderer: GpuRenderer): SharedApi => renderer;

const original = Object.getOwnPropertyDescriptor(globalThis, "navigator");

afterEach(() => {
  if (original) Object.defineProperty(globalThis, "navigator", original);
  else Reflect.deleteProperty(globalThis, "navigator");
});

function withoutWebGpu() {
  Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true, writable: true });
}

it("records the inputs as the CPU renderer does", () => {
  const renderer = new GpuRenderer({ width: 100, height: 50, settings: { samples: 8 } });

  renderer.setSettings({ shadows: { softness: 3 } });
  renderer.setSize(200, 100, 2);

  expect(renderer.settings.samples).toBe(8);
  expect(renderer.settings.shadows).toEqual({ enabled: true, softness: 3 });
  expect(renderer.renderWidth).toBe(400);
  expect(renderer.image).toBeNull();
  expect(renderer.rendering).toBe(false);
  expect(() => renderer.setPixelRatio(0)).toThrow("pixelRatio");
});

it("renders without a scene or a camera, failing only for want of a device", async() => {
  withoutWebGpu();

  const renderer = new GpuRenderer({ width: 4, height: 4 });
  const errors: Error[] = [];

  renderer.on("error", error => errors.push(error));

  const frame = renderer.render();

  expect(renderer.rendering).toBe(true);
  await expect(frame).rejects.toThrow("does not expose WebGPU");
  expect(errors).toHaveLength(1);
  expect(renderer.rendering).toBe(false);
});

it("rejects a cancelled frame with an AbortError and no error event", async() => {
  withoutWebGpu();

  const renderer = new GpuRenderer({ width: 4, height: 4 });
  const errors: Error[] = [];

  renderer.on("error", error => errors.push(error));

  const frame = renderer.render().catch((error: unknown) => error);

  renderer.cancel();

  expect(isAbortError(await frame)).toBe(true);
  expect(errors).toHaveLength(0);
});

it("aborts the previous frame when a newer one starts, and every frame on dispose", async() => {
  withoutWebGpu();

  const renderer = new GpuRenderer({ width: 4, height: 4 });
  const first = renderer.render().catch((error: unknown) => error);
  const second = renderer.render().catch((error: unknown) => error);

  renderer.dispose();

  expect(isAbortError(await first)).toBe(true);
  expect(isAbortError(await second)).toBe(true);
  await expect(renderer.render()).rejects.toThrow("disposed");
});

it("pauses and resumes like the CPU renderer", () => {
  const renderer = new GpuRenderer({ width: 4, height: 4 });

  renderer.pause();
  expect(renderer.paused).toBe(true);
  renderer.resume();
  expect(renderer.paused).toBe(false);
});
