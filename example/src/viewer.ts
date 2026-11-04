import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  attachCanvas,
  DEFAULT_RENDER_SETTINGS,
  GpuRenderer,
  mergeSettings,
  requestWebGpuSupport,
  type Bucket,
  type RenderEnvironment,
  type RenderPhase,
  type RenderSettings,
  type RenderSettingsInput,
  type DenoiserSettings,
  type RenderToneMapping,
} from "@tinypoly/gpu-renderer";
import { SCENE_CATALOG, type SceneId } from "./catalog";
import { ScenePreview } from "./preview";
import { SCENES } from "./scenes";
import { enhanceSelect } from "./select";

// Loaded on the first visit to a scene: the gallery never asks for a GPU.

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const stage = $<HTMLElement>("stage");
const canvas = $<HTMLCanvasElement>("output");
const bucketLayer = $<HTMLDivElement>("buckets");
const deviceInfo = $<HTMLDivElement>("device-info");
const sampleCount = $<HTMLSpanElement>("sample-count");
const phaseBadge = $<HTMLSpanElement>("phase");
const status = $<HTMLParagraphElement>("status");
const elapsed = $<HTMLSpanElement>("elapsed");
const resolution = $<HTMLParagraphElement>("resolution");
const bars = [$<HTMLDivElement>("progress-bar"), $<HTMLDivElement>("meter-bar")];
const sceneSelect = $<HTMLSelectElement>("scene");

// The preview paints the studio the GPU renderer lights with (see studio.ts). `{ kind: "hdri", url }` takes any
// equirectangular .hdr; this one is a blob: URL.
const preview = new ScenePreview($<HTMLCanvasElement>("preview"));
const environment: RenderEnvironment = { kind: "hdri", url: preview.environmentUrl };

// `samples` is paths per pixel here: the path tracer converges by piling them up.
let settings: RenderSettings = mergeSettings(DEFAULT_RENDER_SETTINGS, {
  samples: 256,
  globalIllumination: { bounces: 4 },
  depthOfField: { focusDistance: 6.2, aperture: 1.4 },
});

// ---- Render state, shown in the status card ----

const progress = { completed: 0, total: 0, phase: "prepare" as RenderPhase, done: false, samples: 0, target: 0 };
let error: string | null = null;
let paused = false;
let interacting = false;
// While the gallery is showing, the renderer stops without changing the viewer's own Pause button.
let hidden = false;
let startedAt = performance.now();
let finishedAt: number | null = null;
let currentScene: SceneId | null = null;

// The image size follows the canvas' CSS size: the resize observer below hands it to the renderer.
const renderer = new GpuRenderer({
  width: canvas.clientWidth, height: canvas.clientHeight, pixelRatio: window.devicePixelRatio, settings, environment,
});

// The canvas refines in place while the image converges; pixels nothing has reached yet stay see-through.
attachCanvas(renderer, canvas);

// The canvas keeps the previous image until the renderer starts the new frame.
renderer.on("start", () => stage.classList.remove("stale"));
renderer.on("progress", next => {
  Object.assign(progress, next, { done: false });
  renderStatus();
});
renderer.on("complete", () => {
  progress.done = true;
  renderStatus();
});
renderer.on("sample", ({ samples, target }) => {
  Object.assign(progress, { samples, target });
  sampleCount.textContent = `${samples} / ${target}`;
});
renderer.on("bucket", ({ bucket, height }) => showBand(bucket, height));
renderer.on("error", next => {
  error = next.message;
  renderStatus();
});

/** Starts a frame with the inputs set so far. Errors arrive through the `error` event; a replaced frame is expected. */
function render() {
  // Until the new frame clears it, the canvas shows the last view or scene: the preview stands in meanwhile.
  stage.classList.add("stale");
  startedAt = performance.now();
  finishedAt = null;
  error = null;
  Object.assign(progress, { samples: 0, target: settings.samples });
  renderer.render().catch(() => {});
}

/** The renderer runs unless the viewer paused it, the camera is moving or the gallery covers the viewer. */
function syncPause() {
  if (paused || interacting || hidden)
    renderer.pause();
  else
    renderer.resume();
}

// ---- Device: the adapter, and the band of rows being traced ----

// One frame over the image marks the rows on the device: the renderer sizes its dispatches to stay responsive.
const band = document.createElement("div");
band.className = "bucket";
band.hidden = true;
band.style.setProperty("--worker", "oklch(0.72 0.17 250)");
bucketLayer.append(band);

function chip(name: string, value: string) {
  const element = document.createElement("div");
  element.className = "worker";
  element.dataset.active = "true";
  element.style.setProperty("--worker", "oklch(0.72 0.17 250)");
  element.innerHTML = '<span class="worker-dot"></span><span class="worker-name"></span><span class="worker-count"></span>';
  element.querySelector(".worker-name")!.textContent = name;
  element.querySelector(".worker-count")!.textContent = value;
  deviceInfo.append(element);
}

// The renderer asks for its own device; this only names the adapter, and says plainly when there is none.
void requestWebGpuSupport().then(async({ supported, reason }) => {
  if (!supported) {
    $<HTMLParagraphElement>("device-note").hidden = false;
    error = reason;
    renderStatus();

    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const info = adapter?.info;

  chip("GPU", [info?.vendor, info?.architecture].filter(Boolean).join(" ") || "WebGPU");

  if (info?.description) chip("Adapter", info.description);
});

/** Positions in percent of the render height, so the frame follows the canvas' CSS size at any render scale. */
function showBand(bucket: Bucket | null, height: number) {
  band.hidden = !bucket;

  if (!bucket)
    return;

  Object.assign(band.style, {
    left: "0%",
    top: `${bucket.y / height * 100}%`,
    width: "100%",
    height: `${bucket.height / height * 100}%`,
  });
}

// ---- Camera and the WebGL preview ----

// The renderer never touches WebGL: the camera and its controls are plain Three objects, shared with the preview.
const camera = new THREE.PerspectiveCamera(40, 1, .1, 100);
const controls = new OrbitControls(camera, stage);
controls.enableDamping = false;

preview.setToneMapping(settings.toneMapping, settings.exposure);

let previewFrame = 0;

function requestPreview() {
  cancelAnimationFrame(previewFrame);
  previewFrame = requestAnimationFrame(() => preview.render(camera));
}

/** Hands the camera's current placement to the renderer; the caller starts the frame. */
function updateCamera() {
  renderer.setCamera(camera);
  // Depth of field focuses on the orbit target, wherever the camera ends up.
  const focus = camera.position.distanceTo(controls.target);
  if (settings.depthOfField.enabled && Math.abs(focus - settings.depthOfField.focusDistance) > 1e-3)
    applySettings({ depthOfField: { focusDistance: focus } });
  else
    settings = mergeSettings(settings, { depthOfField: { focusDistance: focus } });
}

/**
 * While the camera moves, the path tracer pauses and only WebGL draws. On release it traces the new view, and the
 * preview shows through the canvas until the first samples cover it.
 */
let settleTimer = 0;

controls.addEventListener("start", () => {
  clearTimeout(settleTimer);
  if (interacting)
    return;
  interacting = true;
  stage.classList.add("interacting");
  syncPause();
  renderStatus();
});

controls.addEventListener("change", requestPreview);

// Scrolling fires start/end on every wheel tick: wait until it stops before rendering.
controls.addEventListener("end", () => {
  clearTimeout(settleTimer);
  settleTimer = window.setTimeout(() => {
    interacting = false;
    stage.classList.remove("interacting");
    updateCamera();
    render();
    syncPause();
    renderStatus();
  }, 150);
});

/** What the viewer had before the current scene's own settings replaced it, to put back when the scene changes. */
let replaced: RenderSettingsInput = {};

/** The current values of the fields `input` sets, in the same shape. */
function pick(input: RenderSettingsInput): RenderSettingsInput {
  return Object.fromEntries(Object.entries(input).map(([key, value]) => {
    const current = settings[key as keyof RenderSettings];

    return [key, value && typeof value === "object"
      ? Object.fromEntries(Object.keys(value).map(field => [field, (current as Record<string, unknown>)[field]]))
      : current];
  }));
}

/** Set once the first scene is handed to the renderer: before that, a resize has nothing to render. */
let loaded = false;

function loadScene(id: SceneId) {
  // The preview still holds the last frame of the previous scene: blank it until the new one draws, and report
  // the new scene as preparing, so it shows up dimmed rather than finished.
  preview.clear();
  Object.assign(progress, { completed: 0, total: 0, phase: "prepare", done: false });
  renderStatus();
  const { scene, cameraPosition, target, enclosed, settings: own = {} } = SCENES[id]();
  camera.position.set(...cameraPosition);
  controls.target.set(...target);
  controls.update();
  // The renderer serializes the scene in the background; a scene picked meanwhile replaces it.
  renderer.setScene(scene);
  loaded = true;
  updateCamera();
  settings = mergeSettings(settings, replaced);
  replaced = pick(own);
  settings = mergeSettings(settings, own);
  renderer.setSettings(settings);
  // A closed room sees none of the studio outside, in either renderer (the preview drops its environment too).
  renderer.setEnvironment({ ...environment, intensity: enclosed ? 0 : 1 });
  preview.setToneMapping(settings.toneMapping, settings.exposure);
  syncControls();
  render();
  preview.setScene(scene, enclosed);
  cancelAnimationFrame(previewFrame);
  preview.render(camera);
}

/** Shows the viewer on `id`. Coming back to the scene already on screen keeps its finished image. */
export function showScene(id: SceneId) {
  hidden = false;
  syncPause();
  if (id === currentScene)
    return;
  currentScene = id;
  // The select follows the route; its change handler only updates the hash, which already matches.
  sceneSelect.value = id;
  sceneSelect.dispatchEvent(new Event("change"));
  loadScene(id);
}

export function hideViewer() {
  hidden = true;
  syncPause();
}

// ---- Sidebar controls ----

for (const { id, title } of SCENE_CATALOG)
  sceneSelect.add(new Option(title, id));

for (const select of document.querySelectorAll<HTMLSelectElement>("select.select"))
  enhanceSelect(select);

// Sections open as the HTML sets them (Scene and Device); what the viewer toggles is remembered per browser.
// Storage may be unavailable (private mode, blocked site data): the sections still toggle.
const SECTIONS_KEY = "gpu-renderer-example:sections";

function readSections(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(SECTIONS_KEY) ?? "{}") as Record<string, boolean>;
  } catch {
    return {};
  }
}

for (const section of document.querySelectorAll<HTMLDetailsElement>("details[data-section]")) {
  const name = section.dataset.section!;
  section.open = readSections()[name] ?? section.open;
  section.addEventListener("toggle", () => {
    try {
      localStorage.setItem(SECTIONS_KEY, JSON.stringify({ ...readSections(), [name]: section.open }));
    } catch {
      // Not remembered.
    }
  });
}

/** Changes settings for the next frame without starting it. */
function applySettings(patch: RenderSettingsInput) {
  settings = mergeSettings(settings, patch);
  renderer.setSettings(patch);
  preview.setToneMapping(settings.toneMapping, settings.exposure);
  requestPreview();
  syncDependents();
}

function updateSettings(patch: RenderSettingsInput) {
  applySettings(patch);
  render();
}

/** Each sidebar control, by element id: the setting it shows and the change it makes. */
const CONTROLS = {
  maxSamples: { read: s => s.samples, write: v => ({ samples: Number(v) }) },
  renderScale: { read: s => s.renderScale, write: v => ({ renderScale: Number(v) }) },
  tonemapping: { read: s => s.toneMapping, write: v => ({ toneMapping: v as RenderToneMapping }) },
  // Global illumination is what a path tracer does: only how far a path goes is a choice.
  bounces: {
    read: s => s.globalIllumination.bounces, write: v => ({ globalIllumination: { bounces: Number(v) } }),
  },
  shadowSoftness: { read: s => s.shadows.softness, write: v => ({ shadows: { softness: Number(v) } }) },
  dofEnabled: { read: s => s.depthOfField.enabled, write: v => ({ depthOfField: { enabled: Boolean(v) } }) },
  denoise: {
    read: s => s.globalIllumination.denoise, write: v => ({ globalIllumination: { denoise: Boolean(v) } }),
  },
  dofAperture: { read: s => s.depthOfField.aperture, write: v => ({ depthOfField: { aperture: Number(v) } }) },
} satisfies Record<string, {
  read: (settings: RenderSettings) => number | boolean | string;
  write: (value: number | boolean | string) => RenderSettingsInput;
}>;

type ControlId = keyof typeof CONTROLS;

// Controls that only matter when a switch is on.
const DEPENDENTS: Partial<Record<ControlId, string[]>> = {
  dofEnabled: ["dofAperture"],
  denoise: ["denoiseStrength", "denoisePasses", "denoiseEdge", "denoiseOnlyNoisy", "denoiseAntiFirefly"],
};

/**
 * A finished image takes a filter change at once, re-filtered without tracing; a frame still in progress starts over,
 * so what the canvas shows always comes from one setting.
 */
function restartIfRendering() {
  if (renderer.rendering)
    render();
}

/**
 * The denoiser's own controls. They change how the finished image is filtered, which the renderer does on the spot:
 * none of them starts a new frame.
 */
const DENOISER_CONTROLS = {
  denoiseStrength: { read: d => d.strength, write: v => ({ strength: Number(v) }) },
  denoisePasses: { read: d => d.passes, write: v => ({ passes: Number(v) }) },
  denoiseEdge: { read: d => d.edge, write: v => ({ edge: Number(v) }) },
  denoiseOnlyNoisy: { read: d => d.onlyNoisy, write: v => ({ onlyNoisy: Boolean(v) }) },
  denoiseAntiFirefly: { read: d => d.antiFirefly, write: v => ({ antiFirefly: Boolean(v) }) },
} satisfies Record<string, {
  read: (denoiser: DenoiserSettings) => number | boolean;
  write: (value: number | boolean | string) => Partial<DenoiserSettings>;
}>;

for (const [id, control] of Object.entries(DENOISER_CONTROLS)) {
  const element = $<HTMLInputElement | HTMLSelectElement>(id), value = control.read(renderer.denoiser);

  if (element instanceof HTMLInputElement && element.type === "checkbox")
    element.checked = Boolean(value);
  else {
    element.value = String(value);
    // The styled select listens for this to show a value set from code.
    if (element instanceof HTMLSelectElement)
      element.dispatchEvent(new Event("sync"));
  }

  element.addEventListener("change", () => {
    renderer.setDenoiser(control.write(
      element instanceof HTMLInputElement && element.type === "checkbox" ? element.checked : element.value));
    restartIfRendering();
  });
}

function syncDependents() {
  for (const [key, ids] of Object.entries(DEPENDENTS) as [ControlId, string[]][])
    for (const id of ids) {
      const control = $<HTMLInputElement>(id), enabled = Boolean(CONTROLS[key].read(settings));
      control.disabled = !enabled;
      control.closest(".field, .switch-row")?.classList.toggle("is-disabled", !enabled);
    }
}

/** Switches, sliders and selects follow the settings when a scene brings its own. */
function syncControls() {
  for (const [id, control] of Object.entries(CONTROLS) as [ControlId, (typeof CONTROLS)[ControlId]][]) {
    const element = $<HTMLInputElement | HTMLSelectElement>(id), value = control.read(settings);

    if (element instanceof HTMLInputElement && element.type === "checkbox")
      element.checked = Boolean(value);
    else {
      element.value = String(value);
      if (element.classList.contains("slider"))
        syncSlider(element as HTMLInputElement);
      if (element instanceof HTMLSelectElement)
        element.dispatchEvent(new Event("sync"));
    }
  }

  syncDependents();
}

/** Sliders show their value and fill the track up to the thumb. */
function syncSlider(slider: HTMLInputElement) {
  const min = Number(slider.min), max = Number(slider.max), value = Number(slider.value);
  slider.style.setProperty("--fill", `${(value - min) / (max - min) * 100}%`);
  const output = document.querySelector<HTMLOutputElement>(`output[for="${slider.id}"]`);
  if (output)
    output.textContent = `${output.dataset.prefix ?? ""}${value.toFixed(1)}${output.dataset.unit ?? ""}`;
}

// Controls that only change how the finished image is filtered: the renderer applies them without tracing again.
const FILTER_ONLY = new Set<ControlId>(["denoise"]);

for (const [id, control] of Object.entries(CONTROLS) as [ControlId, (typeof CONTROLS)[ControlId]][]) {
  const element = $<HTMLInputElement | HTMLSelectElement>(id);
  element.addEventListener("change", () => {
    const patch = control.write(
      element instanceof HTMLInputElement && element.type === "checkbox" ? element.checked : element.value);

    if (FILTER_ONLY.has(id)) {
      applySettings(patch);
      restartIfRendering();
    } else
      updateSettings(patch);
  });
}

// The route owns the scene: picking one here changes the hash, and the page's router loads it.
sceneSelect.addEventListener("change", () => {
  if (location.hash.slice(1) !== sceneSelect.value)
    location.hash = sceneSelect.value;
});

for (const slider of document.querySelectorAll<HTMLInputElement>(".slider")) {
  syncSlider(slider);
  slider.addEventListener("input", () => syncSlider(slider));
}

syncDependents();

const PAUSE_ICON = "<rect x=\"6\" y=\"4\" width=\"4\" height=\"16\" rx=\"1\" /><rect x=\"14\" y=\"4\" width=\"4\" height=\"16\" rx=\"1\" />";
const RESUME_ICON = "<path d=\"M7 4.5v15a1 1 0 0 0 1.5.86l12-7.5a1 1 0 0 0 0-1.72l-12-7.5A1 1 0 0 0 7 4.5Z\" />";
const pause = $<HTMLButtonElement>("pause");

pause.addEventListener("click", () => {
  paused = !paused;
  syncPause();
  pause.querySelector("svg")!.innerHTML = paused ? RESUME_ICON : PAUSE_ICON;
  pause.querySelector("span")!.textContent = paused ? "Resume" : "Pause";
  renderStatus();
});

/**
 * The finished image as a 2D canvas. A WebGPU canvas does not keep what it presented, so the picture comes from the
 * pixels `render()` resolved with, which `image` holds.
 */
export function imageCanvas(): HTMLCanvasElement | null {
  const image = renderer.image;

  if (!image || !progress.done)
    return null;

  const target = document.createElement("canvas");
  target.width = image.width;
  target.height = image.height;
  target.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(image.data), image.width, image.height), 0, 0);

  return target;
}

// Saving is plain canvas-to-PNG, outside the renderer.
$<HTMLButtonElement>("save").addEventListener("click", () => {
  const picture = imageCanvas();
  if (!picture)
    return;
  picture.toBlob(blob => {
    if (!blob)
      return;
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url;
    link.download = `render-${Date.now()}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
});

// ---- Status card ----

/** Badge state, badge label and status line. */
function describe(done: boolean): [string, string, string] {
  const { phase } = progress;
  if (error)
    return ["error", "Error", error];
  if (interacting)
    return ["preview", "Preview", "WebGL preview. Release to trace on the GPU."];
  if (paused && !done)
    return ["paused", "Paused", `Paused at ${progress.samples} of ${progress.target} samples.`];
  if (phase === "prepare" && !done)
    return ["busy", "Preparing", "Preparing the scene…"];
  if (done)
    return ["done", "Done", `Traced ${progress.target} paths per pixel.`];

  return ["busy", "Tracing", `Tracing sample ${Math.min(progress.samples + 1, progress.target)} of ${progress.target}…`];
}

function renderStatus() {
  const { completed, total, done } = progress;
  if (done && finishedAt === null)
    finishedAt = performance.now();

  const [state, label, text] = describe(done);
  phaseBadge.dataset.state = state;
  // The stage dims the WebGL preview while the traced image is unfinished, so the first samples show up over it.
  stage.dataset.state = state;
  phaseBadge.textContent = label;
  status.textContent = text;
  for (const bar of bars)
    bar.style.width = total ? `${completed / total * 100}%` : "0%";
  elapsed.textContent = `${(((finishedAt ?? performance.now()) - startedAt) / 1000).toFixed(1)} s`;
  resolution.textContent = `${canvas.width} × ${canvas.height} px`;
}

// The elapsed time keeps counting between progress events.
setInterval(() => {
  if (finishedAt === null && !paused && !interacting && !hidden)
    renderStatus();
}, 100);

// ---- Layout ----

// The image follows the canvas' CSS size; every resize starts a new frame. Hiding the viewer collapses the canvas
// to nothing and showing it restores the old size: neither is a new frame, so the finished image survives.
let size = "";

new ResizeObserver(() => {
  const { clientWidth: width, clientHeight: height } = canvas;
  if (!width || !height || `${width}x${height}` === size)
    return;
  size = `${width}x${height}`;
  renderer.setSize(width, height, window.devicePixelRatio);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  preview.resize(width, height);
  preview.render(camera);
  if (loaded)
    render();
}).observe(canvas);

// scripts/capture-scenes.js takes the gallery pictures from here.
Object.assign(window, { exampleImage: imageCanvas });
