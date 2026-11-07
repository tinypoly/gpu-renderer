import { describe, expect, it } from "vitest";
import { materialProperties, MATERIAL_WATER } from "./materialLayout.js";
import type { SerializedMaterial } from "./upstream.js";
import { refract, solveCausticPoint, waterField, waterNormal, waterOf, waterSlope, WATER_PHASE, WATER_SPREAD, WATER_WAVES, type WaterField } from "./water.js";

type Vec3 = [number, number, number];

/** The height whose slope waterSlope gives, to check it against. */
function height(field: WaterField, x: number, z: number): number {
  let h = 0;

  for (let i = 0; i < WATER_WAVES; i++) {
    const length = field.waveLength * (1 / 6) ** (i / (WATER_WAVES - 1));
    const k = (2 * Math.PI) / length;
    const angle = field.wind + WATER_SPREAD[i];
    const phase = k * (Math.cos(angle) * x + Math.sin(angle) * z) + WATER_PHASE[i] + field.seed * (i + 1) * 0.37;

    h += ((field.waveHeight * length) / field.waveLength) * Math.sin(phase);
  }

  return h;
}

const normalize = (v: Vec3): Vec3 => {
  const length = Math.hypot(...v);

  return [v[0] / length, v[1] / length, v[2] / length];
};

const physical = (water: unknown, transmission = 1) =>
  ({ type: "MeshPhysicalMaterial", values: { transmission, ior: 1.333, thickness: 1 }, water }) as unknown as SerializedMaterial;

describe("água", () => {
  it("lê os padrões e só vale num material que deixa a luz passar", () => {
    expect(waterField(true)).toEqual({ waveHeight: 0.006, waveLength: 0.9, wind: 0.6, seed: 0 });
    expect(waterField({ waveHeight: 5, waveLength: -1 })).toMatchObject({ waveHeight: 1, waveLength: 0.01 });
    expect(waterField(false)).toBeNull();
    expect(waterOf(physical(true))).not.toBeNull();
    expect(waterOf(physical(true, 0))).toBeNull();
    const properties = materialProperties(physical({ waveHeight: 0.01 }));
    expect(properties.flags & MATERIAL_WATER).toBe(MATERIAL_WATER);
    expect(properties.water?.waveHeight).toBe(0.01);
    expect(materialProperties(physical(undefined)).flags & MATERIAL_WATER).toBe(0);
  });

  it("a inclinação é a derivada da altura das ondas", () => {
    const field = waterField({ waveHeight: 0.01, waveLength: 1.2, wind: 0.3, seed: 2 })!;
    const h = 1e-5;

    for (const [x, z] of [[0.3, -1.7], [4.2, 2.5], [-3, 0.01]]) {
      const [sx, sz] = waterSlope(field, x, z);
      expect(sx).toBeCloseTo((height(field, x + h, z) - height(field, x - h, z)) / (2 * h), 5);
      expect(sz).toBeCloseTo((height(field, x, z + h) - height(field, x, z - h)) / (2 * h), 5);
    }
  });

  it("acha o ponto da superfície que refrata a luz até o fundo, pela lei de Snell", () => {
    const field = waterField({ waveHeight: 0.008, waveLength: 0.9 })!;
    const toLight = normalize([0.4, 1, 0.25]);
    const point: Vec3 = [1.3, -1.4, 0.7];

    // Água parada: o caminho reto da refração.
    const still = solveCausticPoint({ ...field, waveHeight: 0 }, point, toLight, 0, 1.333, [point[0], point[2]])!;
    const inside = refract([-toLight[0], -toLight[1], -toLight[2]], [0, 1, 0], 1 / 1.333);
    // Do fundo, de volta pelo raio refratado até a superfície.
    const t = point[1] / inside[1];
    const flat: [number, number] = [point[0] - inside[0] * t, point[2] - inside[2] * t];
    expect(still[0]).toBeCloseTo(flat[0], 6);
    expect(still[1]).toBeCloseTo(flat[1], 6);

    // Com ondas, a luz refratada no ponto achado vai direto ao fundo.
    const spot = solveCausticPoint(field, point, toLight, 0, 1.333, flat)!;
    expect(spot).not.toBeNull();
    const bent = refract([-toLight[0], -toLight[1], -toLight[2]], waterNormal(field, spot[0], spot[1]), 1 / 1.333);
    const toPoint = normalize([point[0] - spot[0], point[1], point[2] - spot[1]]);
    expect(bent[0] * toPoint[0] + bent[1] * toPoint[1] + bent[2] * toPoint[2]).toBeGreaterThan(0.999999);
  });

  it("as cáusticas só redistribuem a luz: na média, o fundo recebe o mesmo que sob a água parada", () => {
    const field = waterField({ waveHeight: 0.004, waveLength: 1.2 })!;
    const toLight = normalize([0.3, 1, 0.2]);
    const depth = 1.2;

    // O mapa da entrada para o pouso no fundo plano; o ganho é o inverso de quanto ele espalha a área.
    const landing = (x: number, z: number): [number, number] => {
      const d = refract([-toLight[0], -toLight[1], -toLight[2]], waterNormal(field, x, z), 1 / 1.333);
      const t = -depth / d[1];

      return [x + d[0] * t, z + d[2] * t];
    };

    let sum = 0;
    let count = 0;

    // Fundo varrido numa grade larga, para as ondas mais longas entrarem inteiras.
    for (let i = 0; i < 60; i++) {
      for (let j = 0; j < 60; j++) {
        const x = i * 0.11 + 0.013;
        const z = j * 0.11 + 0.029;
        const a = landing(x, z);
        const bx = landing(x + 1e-4, z);
        const bz = landing(x, z + 1e-4);
        const spread = Math.abs(((bx[0] - a[0]) * (bz[1] - a[1]) - (bz[0] - a[0]) * (bx[1] - a[1])) / 1e-8);
        // A área que chega ao fundo, em média, é a que saiu da superfície.
        sum += spread;
        count++;
      }
    }

    expect(sum / count).toBeCloseTo(1, 1);
  });
});
