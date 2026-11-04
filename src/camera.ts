import { Matrix4, OrthographicCamera, PerspectiveCamera } from "three";
import type { CameraState } from "./upstream.js";

/**
 * Everything the renderer derives from a camera snapshot. Rays are cast through the inverse of the camera's own
 * projection, so zoom, film offsets, `setViewOffset` and any custom projection come out exactly as Three draws them,
 * and so do the near and far planes, which clip a camera ray the way they clip a rasterized fragment.
 */
export interface CameraMatrices {
  /** Column-major, as Three stores them. */
  cameraWorld: number[];
  view: number[];
  projection: number[];
  inverseProjection: number[];
  orthographic: boolean;
  near: number;
  far: number;
  /** Vertical field of view in degrees, for the lens; 50 when the snapshot has none. */
  fov: number;
  position: number[];
}

/** The camera's projection: the snapshot's own when it has one, otherwise rebuilt from its parameters. */
export function cameraProjection(camera: CameraState, aspect: number): Matrix4 {
  if (camera.projectionMatrix) return new Matrix4().fromArray(camera.projectionMatrix);

  const projection = camera.type === "orthographic"
    ? new OrthographicCamera(camera.left, camera.right, camera.top, camera.bottom, camera.near, camera.far)
    : new PerspectiveCamera(camera.fov ?? 50, camera.aspect ?? aspect, camera.near, camera.far);

  projection.zoom = camera.zoom ?? 1;
  projection.updateProjectionMatrix();

  return projection.projectionMatrix;
}

export function cameraMatrices(camera: CameraState, width: number, height: number): CameraMatrices {
  const world = new Matrix4().fromArray(camera.matrixWorld);
  const projection = cameraProjection(camera, width / height);
  const orthographic = camera.type === "orthographic" || (projection.elements[11] === 0 && projection.elements[15] === 1);

  return {
    cameraWorld: world.toArray(),
    view: world.clone().invert().toArray(),
    projection: projection.toArray(),
    inverseProjection: projection.clone().invert().toArray(),
    orthographic,
    near: camera.near,
    far: camera.far,
    fov: camera.fov ?? 50,
    position: [camera.matrixWorld[12], camera.matrixWorld[13], camera.matrixWorld[14]],
  };
}

/** A column-major 4x4 matrix times a point, with the w it produced. */
export function transform4(matrix: ArrayLike<number>, x: number, y: number, z: number, w = 1): number[] {
  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w,
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w,
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w,
    matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w,
  ];
}

/** A point in clip space back to world space, through the camera that produced it. */
export function clipToWorld(camera: CameraMatrices, clip: readonly number[]): number[] {
  const view = transform4(camera.inverseProjection, clip[0], clip[1], clip[2], clip[3]);
  const w = view[3] || 1e-30;

  return transform4(camera.cameraWorld, view[0] / w, view[1] / w, view[2] / w).slice(0, 3);
}

export function worldToClip(camera: CameraMatrices, point: readonly number[]): number[] {
  const view = transform4(camera.view, point[0], point[1], point[2]);

  return transform4(camera.projection, view[0], view[1], view[2], view[3]);
}
