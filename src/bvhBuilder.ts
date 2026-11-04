import { buildBvh, type PackedBvh } from "./bvh.js";

/** What the page sends the worker: its own copy of the positions, transferred. */
export interface BvhRequest {
  id: number;
  positions: Float32Array;
  triangleCount: number;
}

/** The hierarchy back, its buffers transferred, or why it could not be built. */
export type BvhResponse =
  | { id: number; nodes: ArrayBuffer; nodeCount: number; order: Uint32Array; maxDepth: number }
  | { id: number; error: string };

/**
 * What the renderer needs from a worker: the shape of a Web `Worker`. Any transport that carries structured clones and
 * transfer lists fits, such as a `worker_threads` thread behind an adapter.
 */
export interface BvhWorker {
  postMessage(message: BvhRequest, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((event: { data: BvhResponse }) => void) | null;
  onerror: ((event: { message?: string; preventDefault?(): void }) => void) | null;
}

export type BvhWorkerFactory = () => BvhWorker;

/**
 * Below this many triangles the hierarchy is built on the page: it takes a few milliseconds, less than copying the
 * positions to a worker and the hierarchy back.
 */
export const BVH_WORKER_MIN_TRIANGLES = 50_000;

/** The worker this package ships, next to this module. Replace it when the bundler cannot follow the URL. */
export function createDefaultBvhWorker(): BvhWorker {
  if (typeof Worker === "undefined")
    throw new Error("Web Workers are unavailable here: pass `createBvhWorker`, or `bvhWorker: false`.");

  return new Worker(new URL("./bvhWorker.js", import.meta.url), { type: "module" }) as unknown as BvhWorker;
}

/** The worker's side of the exchange, apart from the worker itself so it can be run anywhere. */
export function respondToBvhRequest(request: BvhRequest): { response: BvhResponse; transfer: Transferable[] } {
  try {
    const bvh = buildBvh(request.positions, request.triangleCount);

    return {
      response: { id: request.id, ...bvh },
      transfer: [bvh.nodes, bvh.order.buffer],
    };
  } catch(error) {
    const message = error instanceof Error ? error.message : String(error);

    return { response: { id: request.id, error: message }, transfer: [] };
  }
}

interface Pending {
  positions: Float32Array;
  triangleCount: number;
  resolve: (bvh: PackedBvh) => void;
  reject: (error: Error) => void;
}

/**
 * Builds hierarchies in a worker, so a large scene does not hold the page while it is split. The worker is started on
 * the first build that needs it and kept for the next ones. If it cannot be started or fails to load, every build from
 * then on happens on the page, with the same result: the worker only moves the work, it never changes it.
 */
export class BvhBuilder {
  private worker: BvhWorker | null = null;
  private failed = false;
  private disposed = false;
  private next = 0;
  private readonly pending = new Map<number, Pending>();

  constructor(
    private readonly enabled: boolean,
    private readonly createWorker: BvhWorkerFactory = createDefaultBvhWorker,
    private readonly minTriangles = BVH_WORKER_MIN_TRIANGLES) {}

  /** Whether builds go to the worker: it is enabled and has not failed. */
  get usesWorker(): boolean {
    return this.enabled && !this.failed && !this.disposed;
  }

  build(positions: Float32Array, triangleCount: number): Promise<PackedBvh> {
    if (this.disposed) return Promise.reject(abortError());

    const worker = this.usesWorker && triangleCount >= this.minTriangles ? this.connect() : null;

    if (!worker) return Promise.resolve(buildBvh(positions, triangleCount));

    const id = this.next++;
    // The page still uploads the positions, so the worker gets a copy it can own.
    const copy = positions.slice(0, triangleCount * 9);

    return new Promise<PackedBvh>((resolve, reject) => {
      this.pending.set(id, { positions, triangleCount, resolve, reject });
      worker.postMessage({ id, positions: copy, triangleCount }, [copy.buffer]);
    });
  }

  /** Stops the worker. Builds still waiting for it reject with an `AbortError`. */
  dispose() {
    this.disposed = true;
    this.worker?.terminate();
    this.worker = null;

    for (const pending of this.pending.values()) pending.reject(abortError());

    this.pending.clear();
  }

  private connect(): BvhWorker | null {
    if (this.worker) return this.worker;

    let worker: BvhWorker;

    try {
      worker = this.createWorker();
    } catch(error) {
      this.fail(error instanceof Error ? error.message : String(error));

      return null;
    }

    worker.onmessage = event => this.receive(event.data);

    worker.onerror = event => {
      event.preventDefault?.();
      this.fail(event.message || "the worker failed to load");
    };

    this.worker = worker;

    return worker;
  }

  private receive(response: BvhResponse) {
    const pending = this.pending.get(response.id);

    if (!pending) return;

    this.pending.delete(response.id);

    if ("error" in response) pending.reject(new Error(`GPU renderer: building the BVH failed: ${response.error}`));
    else pending.resolve({
      nodes: response.nodes, nodeCount: response.nodeCount, order: response.order, maxDepth: response.maxDepth,
    });
  }

  /** The worker is gone for good: what it was building, and everything after, is built on the page instead. */
  private fail(reason: string) {
    if (!this.failed)
      console.warn(`GPU renderer: the BVH worker is unavailable (${reason}); building on the main thread instead.`);

    this.failed = true;
    this.worker?.terminate();
    this.worker = null;

    const waiting = [...this.pending.values()];

    this.pending.clear();

    for (const pending of waiting) {
      try {
        pending.resolve(buildBvh(pending.positions, pending.triangleCount));
      } catch(error) {
        pending.reject(error instanceof Error ? error : new Error(String(error)));
      }
    }
  }
}

function abortError(): Error {
  return Object.assign(new Error("The renderer was disposed."), { name: "AbortError" });
}
