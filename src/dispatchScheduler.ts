/**
 * How many rows of pixels one submit covers. The system watchdog kills a command that runs too long (around two
 * seconds on Windows), and a tab that submits the whole image at once stops answering, so the work is cut into bands
 * and the band grows or shrinks towards a target time per submit.
 *
 * The first band of a new scene is deliberately small: nothing is known about the machine yet, and a slow integrated
 * GPU should discover that on a band it survives.
 */
export interface DispatchSchedulerOptions {
  /** Milliseconds one submit should take. */
  targetMs?: number;
  /** Rows the first submit covers, before anything has been measured. */
  initialRows?: number;
  /** Rows in the whole image. */
  maxRows: number;
}

/** The most one measurement may widen the band, so an unmeasured machine meets the watchdog gradually. */
const GROW = 4;
/** How quickly the estimate gives ground when submits come back slower than they used to. */
const DECAY = 0.8;

export class DispatchScheduler {
  private readonly targetMs: number;
  private readonly initialRows: number;
  private readonly maxRows: number;
  private rows: number;
  /** Rows a millisecond, as far as anything has shown. Zero until the first submit says something. */
  private throughput = 0;

  constructor(options: DispatchSchedulerOptions) {
    this.targetMs = options.targetMs ?? 12;
    this.maxRows = Math.max(1, Math.floor(options.maxRows));
    this.initialRows = Math.max(1, Math.min(this.maxRows, Math.floor(options.initialRows ?? 16)));
    this.rows = this.initialRows;
  }

  /** Rows the next submit should cover, never more than what is left. */
  next(remaining: number): number {
    return Math.max(1, Math.min(Math.floor(this.rows), Math.floor(remaining)));
  }

  /**
   * Feeds back how long the last submit took, which moves the band towards the target time.
   *
   * What it estimates is a rate, not a direction. Every submit also costs a fixed round trip to the device, and on a
   * small or simple image that round trip is most of what the clock sees. A rule that only asked whether the submit
   * came back late would then read every band as too slow and shrink until each one covered a single row, which is the
   * worst possible way to spend that round trip and is a hole it could never climb out of. A rate notices instead that
   * a wider band cost no more, and widens.
   *
   * The rate is measured from the rows actually submitted rather than from the current guess, so one band clamped by
   * what was left of the image does not stop the next one from growing.
   */
  record(durationMs: number, rows: number) {
    if (!Number.isFinite(durationMs) || durationMs <= 0 || rows <= 0) return;

    const observed = rows / durationMs;

    // Good news is believed at once, since it can only mean the band was cheaper than the clock suggested. Bad news
    // is taken gradually, so one submit that happened to land behind something else does not collapse the band.
    this.throughput = observed > this.throughput ? observed : this.throughput * DECAY + observed * (1 - DECAY);
    this.rows = Math.max(1, Math.min(this.maxRows, rows * GROW, this.targetMs * this.throughput));
  }

  /** Back to the unmeasured band, for a new scene or a new size. */
  reset() {
    this.rows = this.initialRows;
    this.throughput = 0;
  }
}
