import { expect, it } from "vitest";
import { deviceMemory, retainedBytes, sceneBreakdown, textureUsage, trackDevice } from "./memoryProbe.js";

const MB = 1024 * 1024;

function fakeDevice(): GPUDevice {
  const resource = () => ({ destroy() {} });

  return { createBuffer: resource, createTexture: resource } as unknown as GPUDevice;
}

it("counts typed arrays once, even when two views share a buffer", () => {
  const buffer = new ArrayBuffer(1024);

  expect(retainedBytes({ a: new Float32Array(buffer), b: new Uint8Array(buffer, 16), c: [buffer] })).toBe(1024 + 8);
});

it("counts a plain array of numbers at eight bytes an element", () => {
  expect(retainedBytes({ words: new Array(100).fill(0) })).toBe(800);
});

it("counts a buffer until it is destroyed", () => {
  const device = fakeDevice();

  trackDevice(device);

  const before = deviceMemory().totalMB;
  const buffer = device.createBuffer({ label: "probe test", size: 4 * MB, usage: 0 });

  expect(deviceMemory().totalMB - before).toBe(4);
  expect(deviceMemory().byLabel["probe test"]).toBe(4);

  buffer.destroy();

  expect(deviceMemory().totalMB).toBe(before);
});

it("sizes a texture by its format", () => {
  const device = fakeDevice();

  trackDevice(device);

  const texture = device.createTexture({ label: "probe texture", size: [1024, 1024], format: "rgba32float", usage: 0 });

  expect(deviceMemory().byLabel["probe texture"]).toBe(16);
  texture.destroy();
});

it("tells lines and wireframes apart from plain meshes", () => {
  const positions = { data: new Float32Array(6 * 3) };

  const { kinds, largest } = sceneBreakdown([
    { attributes: { position: positions }, index: null, materials: [{ type: "MeshStandardMaterial" }] },
    { attributes: { position: positions }, index: null, materials: [{ type: "MeshBasicMaterial", values: { wireframe: true } }] },
    { primitive: "lineSegments", attributes: { position: positions }, index: null, materials: [{ type: "LineBasicMaterial" }] },
  ]);

  expect(kinds.mesh).toEqual({ meshes: 1, packedTriangles: 2 });
  expect(kinds.wireframe).toEqual({ meshes: 1, packedTriangles: 2 });
  // Three segments, two triangles each.
  expect(kinds.lineSegments).toEqual({ meshes: 1, packedTriangles: 6 });
  expect(largest[0].kind).toBe("lineSegments");
});

it("counts only the channels a texture's slots read", () => {
  const texture = (id: string) => ({ id, width: 1024, height: 1024, data: new Uint8Array(0) });

  const usage = textureUsage([texture("color"), texture("orm"), texture("rough"), texture("normal")], [
    { maps: { map: "color", roughnessMap: "orm", metalnessMap: "orm", normalMap: "normal" } },
    { maps: { roughnessMap: "rough" } },
  ]);

  const byId = Object.fromEntries(usage.textures.map(row => [row.id, row]));

  expect(byId.color.channels).toBe("abgr".split("").sort().join(""));
  expect(byId.orm.channels).toBe("bg");
  expect(byId.rough.channels).toBe("g");
  expect(byId.normal.channels).toBe("gr");
  // Four bytes a texel for the colour, two for roughness with metalness and for the normal, one for roughness alone.
  expect(usage.totalMB).toBe(16);
  expect(usage.packedMB).toBe(4 + 2 + 1 + 2);
});
