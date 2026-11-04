import { Matrix3, Matrix4, Plane, Vector3 } from "three";
import { cameraProjection } from "./camera.js";
import { createShaderModule } from "./shaderModule.js";
import type { CameraState, RenderBackgroundMode, SerializedMaterial, SerializedScene, SerializedTexture } from "./upstream.js";

/**
 * The CPU renderer's planar captures (`userData.cpuRenderer.planarCapture`): a flat surface whose shader samples what is
 * reflected in it and seen through it, as `tReflectionMap`, `tEnvironmentReflectionMap`, `tRefractionMap` and
 * `tRefractionDepth`, through `reflectionMatrix` and `refractionMatrix`. The cameras, the clip planes and the matrices are
 * `pipeline.js`'s own; the images are rendered by this renderer, path traced from those cameras.
 */
export interface PlanarCapture {
  /** The texture the image goes to, and the depth texture for the refraction capture. */
  texture: string;
  depth?: string;
  camera: CameraState;
  /** Whether the capture sees the scene, without its planar surfaces, or only the environment. */
  scene: "scene" | "environment";
  background: RenderBackgroundMode;
  /** Drops what lies behind the plane, as the capture's clip plane does: `dot(xyz, p) + w < 0` is cut. */
  clip?: number[];
}

export interface PlanarCaptures {
  /** The scene with the capture textures in its table and each material's capture uniforms bound. */
  scene: SerializedScene;
  /** The scene the captures see: everything but the planar surfaces, which never appear in one. */
  captureScene: SerializedScene;
  captures: PlanarCapture[];
  width: number;
  height: number;
}

/** A capture is rendered at most this wide; it is a texture a shader blurs and distorts, not the image itself. */
const MAX_CAPTURE = 512;

function placeholder(id: string, width: number, height: number): SerializedTexture {
  return {
    id,
    width,
    height,
    data: new Float32Array(width * height * 4),
    colorSpace: "srgb-linear",
    wrapS: 1001,
    wrapT: 1001,
    matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    // The rows are written top first, as the CPU renderer writes them.
    flipY: true,
    channel: 0,
    nearest: false,
  };
}

/** The captures a scene needs from this camera, or null when it has no planar surface. */
export function planarCaptures(
  source: SerializedScene, camera: CameraState, width: number, height: number): PlanarCaptures | null {
  const planar = source.meshes.filter(mesh =>
    !mesh.primitive && mesh.materials.some(material => material?.planarCapture && material.shader));

  if (!planar.length) return null;

  const scale = Math.min(1, MAX_CAPTURE / Math.max(width, height));
  const captureWidth = Math.max(1, Math.round(width * scale));
  const captureHeight = Math.max(1, Math.round(height * scale));
  const textures = [...source.textures];
  const captures: PlanarCapture[] = [];
  const copies = new Map<SerializedMaterial, SerializedMaterial>();
  const cameraWorld = new Matrix4().fromArray(camera.matrixWorld);
  const projection = cameraProjection(camera, width / height);
  const bias = new Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);

  for (const mesh of planar) {
    const model = new Matrix4().fromArray(mesh.matrixWorld);
    const normal = new Vector3(0, 1, 0).applyMatrix3(new Matrix3().getNormalMatrix(model)).normalize();
    const point = new Vector3().setFromMatrixPosition(model);
    const plane = new Plane().setFromNormalAndCoplanarPoint(normal, point);
    const { x, y, z } = normal, d = plane.constant;

    const mirror = new Matrix4().set(
      1 - 2 * x * x, -2 * x * y, -2 * x * z, -2 * d * x,
      -2 * y * x, 1 - 2 * y * y, -2 * y * z, -2 * d * y,
      -2 * z * x, -2 * z * y, 1 - 2 * z * z, -2 * d * z,
      0, 0, 0, 1);

    const reflected = { ...camera, matrixWorld: mirror.clone().multiply(cameraWorld).toArray() };
    const side = plane.distanceToPoint(new Vector3().setFromMatrixPosition(cameraWorld)) >= 0 ? 1 : -1;

    const textureMatrix = (world: number[]) => bias.clone().multiply(projection)
      .multiply(new Matrix4().fromArray(world).invert()).multiply(model).toArray();

    for (const material of mesh.materials) {
      if (!material?.planarCapture || !material.shader) continue;

      const id = material.planarCapture.id;

      const copy: SerializedMaterial = {
        ...material,
        shader: { ...material.shader, uniforms: { ...material.shader.uniforms } },
      };

      const uniforms = (copy.shader as NonNullable<SerializedMaterial["shader"]>).uniforms;

      const names = {
        reflection: `${id}:tReflectionMap`,
        environment: `${id}:tEnvironmentReflectionMap`,
        refraction: `${id}:tRefractionMap`,
        depth: `${id}:tRefractionDepth`,
      };

      Object.assign(uniforms, {
        reflectionMatrix: textureMatrix(reflected.matrixWorld),
        refractionMatrix: textureMatrix(camera.matrixWorld),
        cameraNear: camera.near,
        cameraFar: camera.far,
        tReflectionMap: { texture: names.reflection },
        tEnvironmentReflectionMap: { texture: names.environment },
        tRefractionMap: { texture: names.refraction },
        tRefractionDepth: { texture: names.depth },
      });

      copies.set(material, copy);

      for (const name of Object.values(names)) textures.push(placeholder(name, captureWidth, captureHeight));

      captures.push(
        { texture: names.reflection, camera: reflected, scene: "scene", background: "transparent",
          clip: [x * side, y * side, z * side, d * side + 0.01] },
        { texture: names.environment, camera: reflected, scene: "environment", background: "environment" },
        { texture: names.refraction, depth: names.depth, camera, scene: "scene", background: "environment",
          clip: [-x * side, -y * side, -z * side, -d * side + 0.01] },
      );
    }
  }

  const scene: SerializedScene = {
    ...source,
    textures,
    meshes: source.meshes.map(mesh => ({ ...mesh, materials: mesh.materials.map(m => (m && copies.get(m)) ?? m) })),
  };

  return {
    scene,
    captureScene: { ...source, meshes: source.meshes.filter(mesh => !planar.includes(mesh)) },
    captures,
    width: captureWidth,
    height: captureHeight,
  };
}

/** Writes a finished capture from its accumulation buffer into the texels of its texture, and the depth into another. */
const RESOLVE_WGSL = /* wgsl */ `
struct Resolve {
  size: vec2<u32>,
  color: u32,
  depth: u32,
};

@group(0) @binding(0) var<uniform> resolve: Resolve;
@group(0) @binding(1) var<storage, read> accumulation: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> textures: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let count = resolve.size.x * resolve.size.y;
  let i = id.x;

  if (i >= count) {
    return;
  }

  let sum = accumulation[i];
  let extra = accumulation[count + i];
  let n = max(sum.w, 1.0);
  // Linear light, premultiplied by the coverage, as the CPU renderer's capture holds it.
  let color = vec4<f32>(sum.rgb / n, extra.x / n);

  for (var c = 0u; c < 4u; c = c + 1u) {
    textures[resolve.color + i * 4u + c] = bitcast<u32>(color[c]);
  }

  if (resolve.depth != 0xffffffffu) {
    let depth = extra.y / n;

    for (var c = 0u; c < 3u; c = c + 1u) {
      textures[resolve.depth + i * 4u + c] = bitcast<u32>(depth);
    }

    textures[resolve.depth + i * 4u + 3u] = bitcast<u32>(1.0);
  }
}
`;

export class CaptureResolvePass {
  private readonly pipeline: GPUComputePipeline;
  private readonly uniforms: GPUBuffer;

  constructor(private readonly device: GPUDevice) {
    const module = createShaderModule(device, "capture resolve", RESOLVE_WGSL);

    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    this.uniforms = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** `color` and `depth` are word offsets of the textures' texels; a depth of 0xffffffff writes none. */
  encode(encoder: GPUCommandEncoder, accumulation: GPUBuffer, textures: GPUBuffer,
    width: number, height: number, color: number, depth: number) {
    this.device.queue.writeBuffer(this.uniforms, 0, new Uint32Array([width, height, color, depth]));

    const group = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: accumulation } },
        { binding: 2, resource: { buffer: textures } },
      ],
    });

    const pass = encoder.beginComputePass({ label: "capture resolve" });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(width * height / 64));
    pass.end();
  }
}
