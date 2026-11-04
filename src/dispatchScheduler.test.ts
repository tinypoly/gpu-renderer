import { expect, it } from "vitest";
import { DispatchScheduler } from "./dispatchScheduler.js";

it("starts on the small unmeasured band", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1080, initialRows: 16 });

  expect(scheduler.next(1080)).toBe(16);
});

it("never asks for more rows than are left", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1080, initialRows: 64 });

  expect(scheduler.next(5)).toBe(5);
});

it("always asks for at least one row", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1080, initialRows: 1 });

  scheduler.record(500, 1);

  expect(scheduler.next(1080)).toBe(1);
});

it("grows towards the whole image when submits come back fast", () => {
  const scheduler = new DispatchScheduler({ maxRows: 400, targetMs: 12 });

  for (let submit = 0; submit < 50; submit++) scheduler.record(1, scheduler.next(400));

  expect(scheduler.next(400)).toBe(400);
});

it("shrinks when a submit takes longer than the target", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1080, initialRows: 256, targetMs: 12 });
  const before = scheduler.next(1080);

  scheduler.record(400, before);

  expect(scheduler.next(1080)).toBeLessThan(before);
});

it("settles instead of oscillating between the extremes", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1000, initialRows: 100, targetMs: 10 });

  // A machine that renders ten rows per millisecond: the band should end up near one hundred rows.
  for (let submit = 0; submit < 200; submit++) {
    const rows = scheduler.next(1000);

    scheduler.record(rows / 10, rows);
  }

  expect(scheduler.next(1000)).toBeGreaterThan(60);
  expect(scheduler.next(1000)).toBeLessThan(140);
});

it("ignores a measurement that says nothing", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1080, initialRows: 32 });

  scheduler.record(Number.NaN, 32);
  scheduler.record(0, 32);

  expect(scheduler.next(1080)).toBe(32);
});

it("goes back to the unmeasured band on reset", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1080, initialRows: 32 });

  scheduler.record(1, 32);
  scheduler.reset();

  expect(scheduler.next(1080)).toBe(32);
});

it("widens the band when the clock is mostly the cost of submitting at all", () => {
  // A machine where every submit costs 10ms to hand over and a hundredth of a millisecond to trace a row. A rule that
  // only asked whether a submit came back late would read every band as too slow and shrink to a single row.
  const scheduler = new DispatchScheduler({ maxRows: 1000, initialRows: 16, targetMs: 12 });

  for (let submit = 0; submit < 60; submit++) {
    const rows = scheduler.next(1000);

    scheduler.record(10 + rows * 0.01, rows);
  }

  expect(scheduler.next(1000)).toBeGreaterThan(100);
});

it("gives ground when the scene turns heavier under it", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1000, initialRows: 16, targetMs: 12 });

  // First a machine that races, then the same one bogged down by a hundred times the work.
  for (let submit = 0; submit < 30; submit++) scheduler.record(scheduler.next(1000) / 100, scheduler.next(1000));

  const fast = scheduler.next(1000);

  for (let submit = 0; submit < 60; submit++) scheduler.record(scheduler.next(1000), scheduler.next(1000));

  expect(scheduler.next(1000)).toBeLessThan(fast / 10);
});

it("never widens a band by more than one step at a time", () => {
  const scheduler = new DispatchScheduler({ maxRows: 100000, initialRows: 8, targetMs: 12 });

  scheduler.record(0.001, 8);

  // However fast the answer came back, the next band is only a few times wider, so a slow machine meets the
  // watchdog gradually rather than on its first long submit.
  expect(scheduler.next(100000)).toBeLessThanOrEqual(32);
});

it("covers that many times fewer rows when a submit takes several paths per pixel", () => {
  const scheduler = new DispatchScheduler({ maxRows: 1000, initialRows: 100, targetMs: 10, maxSamples: 4 });

  // Ten rows a millisecond at one path each: a band of four paths per pixel should settle near a quarter as many rows.
  for (let submit = 0; submit < 200; submit++) {
    const rows = scheduler.next(1000, 4);

    scheduler.record(rows * 4 / 10, rows, 4);
  }

  expect(scheduler.next(1000, 4)).toBeGreaterThan(15);
  expect(scheduler.next(1000, 4)).toBeLessThan(35);
  expect(scheduler.next(1000, 1)).toBeGreaterThan(60);
});
