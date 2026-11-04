import { PerspectiveCamera } from "three";
import { expect, it } from "vitest";
import { cameraMatrices } from "./camera.js";
import { instancedAttributes, packGeometry } from "./geometry.js";
import { serializeCamera } from "./upstream.js";
import type { SerializedMaterial, SerializedMesh, SerializedScene } from "./upstream.js";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function material(values: SerializedMaterial["values"] = {}): SerializedMaterial {
  return { type: "MeshStandardMaterial", name: "m", values, maps: {} };
}

function mesh(positions: number[], overrides: Partial<SerializedMesh> = {}): SerializedMesh {
  return {
    attributes: { position: { data: new Float32Array(positions), itemSize: 3 } },
    index: null,
    groups: [],
    materials: [material()],
    matrixWorld: IDENTITY,
    castShadow: true,
    receiveShadow: true,
    renderOrder: 0,
    ...overrides,
  };
}

function scene(meshes: SerializedMesh[]): SerializedScene {
  return { meshes, lights: [], textures: [] };
}

const TRIANGLE = [0, 0, 0, 1, 0, 0, 0, 1, 0];

it("packs one triangle per three vertices", () => {
  const packed = packGeometry(scene([mesh([...TRIANGLE, ...TRIANGLE])]));

  expect(packed.triangleCount).toBe(2);
  expect(packed.positions.length).toBe(18);
});

it("applies the world matrix to positions", () => {
  const translated = [...IDENTITY];
  translated[12] = 2;
  translated[13] = 3;

  const packed = packGeometry(scene([mesh(TRIANGLE, { matrixWorld: translated })]));

  expect([...packed.positions.slice(0, 3)]).toEqual([2, 3, 0]);
  expect([...packed.positions.slice(3, 6)]).toEqual([3, 3, 0]);
});

it("rotates normals by the inverse transpose, so non-uniform scale does not skew them", () => {
  // Scaling x by four flattens the triangle; a normal along x must stay along x, normalized.
  const scaled = [...IDENTITY];
  scaled[0] = 4;

  const packed = packGeometry(scene([mesh(TRIANGLE, {
    matrixWorld: scaled,
    attributes: {
      position: { data: new Float32Array(TRIANGLE), itemSize: 3 },
      normal: { data: new Float32Array([1, 0, 0, 1, 0, 0, 1, 0, 0]), itemSize: 3 },
    },
  })]));

  expect([...packed.normals.slice(0, 3)]).toEqual([1, 0, 0]);
});

it("falls back to the face normal when the geometry has none", () => {
  const packed = packGeometry(scene([mesh(TRIANGLE)]));

  expect([...packed.normals.slice(0, 3)]).toEqual([0, 0, 1]);
  expect([...packed.normals.slice(6, 9)]).toEqual([0, 0, 1]);
});

it("reads the index buffer", () => {
  const packed = packGeometry(scene([mesh([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], {
    index: new Uint32Array([0, 1, 2, 2, 1, 3]),
  })]));

  expect(packed.triangleCount).toBe(2);
  expect([...packed.positions.slice(9, 12)]).toEqual([0, 1, 0]);
});

it("gives each group its own material and keeps one entry per material", () => {
  const red = material({ color: [1, 0, 0] });
  const blue = material({ color: [0, 0, 1] });

  const packed = packGeometry(scene([
    mesh([...TRIANGLE, ...TRIANGLE], {
      groups: [{ start: 0, count: 3, materialIndex: 0 }, { start: 3, count: 3, materialIndex: 1 }],
      materials: [red, blue],
    }),
    mesh(TRIANGLE, { materials: [red] }),
  ]));

  expect([...packed.materials]).toEqual([0, 1, 0]);
  expect(packed.materialCount).toBe(2);
  expect([...packed.materialColors]).toEqual([1, 0, 0, 1, 0, 0, 1, 1]);
});

it("keeps the emissive colour and its intensity", () => {
  const packed = packGeometry(scene([mesh(TRIANGLE, {
    materials: [material({ emissive: [2, 1, 0.5], emissiveIntensity: 3 })],
  })]));

  expect([...packed.materialEmissive]).toEqual([2, 1, 0.5, 3]);
});

it("builds no points, lines or sprites without a camera to face", () => {
  const packed = packGeometry(scene([
    mesh(TRIANGLE, { primitive: "points" }),
    mesh(TRIANGLE, { primitive: "line" }),
    mesh(TRIANGLE),
  ]));

  expect(packed.triangleCount).toBe(1);
  expect(packed.cameraDependent).toBe(true);
});

/** A camera at (0, 0, 5) looking down -z, with a 90 degree field of view over a 100 by 100 image. */
function view() {
  const camera = new PerspectiveCamera(90, 1, 0.1, 100);

  camera.position.set(0, 0, 5);
  camera.updateMatrixWorld();

  return { camera: cameraMatrices(serializeCamera(camera), 100, 100), width: 100, height: 100 };
}

it("turns a point into a square of its size in pixels, facing the camera", () => {
  const points = mesh([0, 0, 0], {
    primitive: "points",
    materials: [{ type: "PointsMaterial", name: "p", values: { size: 10, sizeAttenuation: false }, maps: {} }],
  });

  const packed = packGeometry(scene([points]), view());

  expect(packed.triangleCount).toBe(2);
  // Ten pixels of a hundred at five units with a 90 degree view: a tenth of the ten units the image spans.
  const xs = [0, 3, 6, 9, 12, 15].map(at => packed.positions[at]);

  expect(Math.max(...xs) - Math.min(...xs)).toBeCloseTo(1, 3);
  // Each corner faces the camera, which is straight ahead along +z.
  expect(packed.normals[2]).toBeGreaterThan(0.98);
});

it("cuts a dashed line into its dashes", () => {
  const line = mesh([-5, 0, 0, 5, 0, 0], {
    primitive: "lineSegments",
    attributes: {
      position: { data: new Float32Array([-5, 0, 0, 5, 0, 0]), itemSize: 3 },
      lineDistance: { data: new Float32Array([0, 10]), itemSize: 1 },
    },
    materials: [{ type: "LineDashedMaterial", name: "d", values: { dashSize: 1, gapSize: 1 }, maps: {} }],
  });

  // Ten units at a period of two: five dashes, two triangles each.
  expect(packGeometry(scene([line]), view()).triangleCount).toBe(10);
});

it("lays a sprite out around its centre, facing the camera", () => {
  const sprite = mesh([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], {
    primitive: "sprite",
    index: new Uint32Array([0, 1, 2, 0, 2, 3]),
    materials: [{ type: "SpriteMaterial", name: "s", values: {}, maps: {} }],
    matrixWorld: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1],
  });

  const packed = packGeometry(scene([sprite]), view());

  expect(packed.triangleCount).toBe(2);
  expect([...packed.positions.slice(0, 3)].map(v => Math.round(v * 1000) / 1000)).toEqual([0, -1, 0]);
});

it("keeps vertex colours, with an opaque alpha for three components", () => {
  const coloured = mesh(TRIANGLE, {
    attributes: {
      position: { data: new Float32Array(TRIANGLE), itemSize: 3 },
      color: { data: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), itemSize: 3 },
    },
    materials: [material({ vertexColors: true })],
  });

  const packed = packGeometry(scene([coloured]));

  expect([...packed.colors]).toEqual([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]);
  // Without a material that reads them, no colours are stored at all.
  expect(packGeometry(scene([mesh(TRIANGLE)])).colors.length).toBe(0);
});

it("turns a mirrored triangle around, so its winding still says which side is the front", () => {
  const mirrored = [...IDENTITY];
  mirrored[0] = -1;

  const packed = packGeometry(scene([mesh(TRIANGLE, { matrixWorld: mirrored })]));
  const corner = (k: number) => [...packed.positions.slice(k * 3, k * 3 + 3)];
  const a = corner(0), b = corner(1), c = corner(2);
  const z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);

  // The original faces +z; mirrored in x it still has to.
  expect(z).toBeGreaterThan(0);
});

it("leaves out a material Three would not draw", () => {
  const packed = packGeometry(scene([mesh(TRIANGLE, { materials: [material({ visible: false })] })]));

  expect(packed.triangleCount).toBe(0);
});

it("writes a material's clipping planes to the extra words and points at them", () => {
  const packed = packGeometry(scene([mesh(TRIANGLE, {
    materials: [material({ clippingPlanes: [[1, 0, 0, -2]] })],
  })]), { extraBase: 100 });

  expect(packed.materialPlacements[0].clipStart).toBe(100);
  expect(packed.materialPlacements[0].clipCount).toBe(1);
  expect([...new Float32Array(packed.extraWords.buffer)]).toEqual([1, 0, 0, -2]);
});

it("moves vertices along their normals by the displacement map", () => {
  const texture = {
    id: "d", width: 1, height: 1, data: new Float32Array([0.5, 0.5, 0.5, 1]), colorSpace: "srgb-linear",
    wrapS: 1001, wrapT: 1001, matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1], flipY: false, channel: 0, nearest: true,
  };

  const displaced = mesh(TRIANGLE, {
    attributes: {
      position: { data: new Float32Array(TRIANGLE), itemSize: 3 },
      normal: { data: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), itemSize: 3 },
    },
    materials: [{ ...material({ displacementScale: 2 }), maps: { displacementMap: "d" } }],
  });

  const packed = packGeometry(scene([displaced]), { textureData: new Map([["d", texture]]) });

  expect(packed.positions[2]).toBeCloseTo(1);
});

it("ignores a group that runs past the end of the buffer", () => {
  const packed = packGeometry(scene([mesh(TRIANGLE, {
    groups: [{ start: 0, count: 300, materialIndex: 0 }],
  })]));

  expect(packed.triangleCount).toBe(1);
});

it("finds the attributes an instanced geometry keeps per instance, and how many instances there are", () => {
  const instanced = mesh([...TRIANGLE, 1, 1, 0], {
    attributes: {
      position: { data: new Float32Array([...TRIANGLE, 1, 1, 0]), itemSize: 3 },
      instanceStart: { data: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]), itemSize: 3 },
      instanceEnd: { data: new Float32Array([1, 0, 0, 2, 0, 0, 3, 0, 0]), itemSize: 3 },
    },
  });

  expect(instancedAttributes(instanced)).toEqual({ names: new Set(["instanceStart", "instanceEnd"]), count: 3 });
  expect(instancedAttributes(mesh(TRIANGLE))).toBeNull();
});
