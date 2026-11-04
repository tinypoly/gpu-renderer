import { afterEach, expect, it } from "vitest";
import { isWebGpuAvailable, requestWebGpuSupport } from "./index.js";

const original = Object.getOwnPropertyDescriptor(globalThis, "navigator");

function setNavigator(value: unknown) {
  Object.defineProperty(globalThis, "navigator", { value, configurable: true, writable: true });
}

afterEach(() => {
  if (original) Object.defineProperty(globalThis, "navigator", original);
  else Reflect.deleteProperty(globalThis, "navigator");
});

it("sees WebGPU when the navigator exposes it", () => {
  setNavigator({ gpu: {} });

  expect(isWebGpuAvailable()).toBe(true);
});

it("sees no WebGPU on a navigator without it", () => {
  setNavigator({});

  expect(isWebGpuAvailable()).toBe(false);
});

it("explains a missing entry point instead of throwing", async() => {
  setNavigator({});

  const support = await requestWebGpuSupport();

  expect(support.supported).toBe(false);
  expect(support.reason).toContain("does not expose WebGPU");
});

it("explains an adapter that was not granted", async() => {
  setNavigator({ gpu: { requestAdapter: async() => null } });

  const support = await requestWebGpuSupport();

  expect(support.supported).toBe(false);
  expect(support.reason).toContain("no adapter");
});

it("reports support once an adapter answers", async() => {
  setNavigator({ gpu: { requestAdapter: async() => ({}) } });

  expect(await requestWebGpuSupport()).toEqual({ supported: true, reason: "" });
});

it("turns a rejected adapter request into a reason", async() => {
  setNavigator({ gpu: { requestAdapter: async() => { throw new Error("driver blocklisted"); } } });

  const support = await requestWebGpuSupport();

  expect(support.supported).toBe(false);
  expect(support.reason).toContain("driver blocklisted");
});
