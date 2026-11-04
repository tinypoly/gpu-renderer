import { expect, it, vi } from "vitest";
import { buildBvh } from "./bvh.js";
import { BvhBuilder, respondToBvhRequest, type BvhRequest, type BvhWorker } from "./bvhBuilder.js";

function triangles(count: number): Float32Array {
  let state = 11;
  const positions = new Float32Array(count * 9);

  for (let i = 0; i < positions.length; i++) {
    state = (state * 1664525 + 1013904223) >>> 0;
    positions[i] = (state / 4294967296) * 10 + (i % 9 < 3 ? 0 : (state % 64) / 64);
  }

  return positions;
}

/** A worker on this thread, answering a turn later, as a real one would; it can also fail to load. */
function fakeWorker(options: { fails?: boolean } = {}) {
  const received: BvhRequest[] = [];

  const worker: BvhWorker & { terminated: boolean } = {
    terminated: false,
    onmessage: null,
    onerror: null,
    postMessage(message) {
      received.push(message);

      setTimeout(() => {
        if (options.fails) return worker.onerror?.({ message: "404" });

        worker.onmessage?.({ data: respondToBvhRequest(message).response });
      }, 0);
    },
    terminate() {
      worker.terminated = true;
    },
  };

  return { worker, received };
}

const same = (a: { nodes: ArrayBuffer; order: Uint32Array }, b: { nodes: ArrayBuffer; order: Uint32Array }) =>
  Buffer.from(a.nodes).equals(Buffer.from(b.nodes)) && Buffer.from(a.order.buffer).equals(Buffer.from(b.order.buffer));

it("builds in the worker the same tree the page would, from its own copy of the positions", async() => {
  const positions = triangles(200);
  const before = positions.slice();
  const { worker, received } = fakeWorker();
  const builder = new BvhBuilder(true, () => worker, 0);

  const bvh = await builder.build(positions, 200);

  expect(received).toHaveLength(1);
  expect(received[0].positions).not.toBe(positions);
  expect(positions).toEqual(before);
  expect(same(bvh, buildBvh(positions, 200))).toBe(true);
});

it("starts the worker once and keeps it for the next builds", async() => {
  const { worker } = fakeWorker();
  const factory = vi.fn(() => worker);
  const builder = new BvhBuilder(true, factory, 0);

  const [a, b] = await Promise.all([builder.build(triangles(50), 50), builder.build(triangles(80), 80)]);

  expect(factory).toHaveBeenCalledTimes(1);
  expect(a.order).toHaveLength(50);
  expect(b.order).toHaveLength(80);
});

it("never starts a worker when it is disabled, or for a scene below the threshold", async() => {
  const factory = vi.fn(() => fakeWorker().worker);

  await new BvhBuilder(false, factory, 0).build(triangles(100), 100);
  await new BvhBuilder(true, factory, 1000).build(triangles(100), 100);

  expect(factory).not.toHaveBeenCalled();
});

it("falls back to the page, once and for good, when the worker fails to load", async() => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const { worker } = fakeWorker({ fails: true });
  const factory = vi.fn(() => worker);
  const builder = new BvhBuilder(true, factory, 0);
  const positions = triangles(120);

  const bvh = await builder.build(positions, 120);

  expect(same(bvh, buildBvh(positions, 120))).toBe(true);
  expect(worker.terminated).toBe(true);
  expect(builder.usesWorker).toBe(false);

  await builder.build(positions, 120);

  expect(factory).toHaveBeenCalledTimes(1);
  expect(warn).toHaveBeenCalledTimes(1);
  warn.mockRestore();
});

it("falls back to the page when the worker cannot even be created", async() => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

  const builder = new BvhBuilder(true, () => {
    throw new Error("no workers here");
  }, 0);

  const bvh = await builder.build(triangles(30), 30);

  expect(bvh.order).toHaveLength(30);
  expect(warn).toHaveBeenCalledOnce();
  warn.mockRestore();
});

it("rejects the builds still in the worker with an AbortError when disposed", async() => {
  const { worker } = fakeWorker();
  const builder = new BvhBuilder(true, () => worker, 0);
  const pending = builder.build(triangles(40), 40);

  builder.dispose();

  await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  await expect(builder.build(triangles(40), 40)).rejects.toMatchObject({ name: "AbortError" });
  expect(worker.terminated).toBe(true);
});

it("reports a build the worker could not finish", async() => {
  const worker: BvhWorker = {
    onmessage: null,
    onerror: null,
    postMessage(message) {
      setTimeout(() => worker.onmessage?.({ data: { id: message.id, error: "out of memory" } }), 0);
    },
    terminate() {},
  };

  await expect(new BvhBuilder(true, () => worker, 0).build(triangles(10), 10)).rejects.toThrow(/out of memory/);
});
