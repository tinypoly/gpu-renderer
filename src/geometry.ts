import { clipToWorld, transform4, worldToClip, type CameraMatrices } from "./camera.js";
import {
  isMaterialVisible,
  MATERIAL_KIND,
  MATERIAL_UNLIT_LIGHT,
  materialProperties,
  NO_PLACEMENT,
  NO_PROGRAM,
  type MaterialPlacement,
  type MaterialProperties,
} from "./materialLayout.js";
import { normalMatrix, transformNormal, transformPoint } from "./math.js";
import type { ShaderPrograms } from "./shaderProgram.js";
import { type CpuShader, matrixValue, sampleTexture } from "./upstream.js";
import type { SerializedMaterial, SerializedMesh, SerializedScene, SerializedTexture, UniformValue } from "./upstream.js";

/** The flat, GPU ready form of a scene's triangles. Positions and normals are world space, three vertices per entry. */
export interface PackedGeometry {
  /** Nine coordinates per triangle. */
  positions: Float32Array;
  /** Nine normal components per triangle, one per vertex. */
  normals: Float32Array;
  /** Twelve per triangle: two UV sets for each of the three vertices, which is what Three's map channels need. */
  uvs: Float32Array;
  /** Twelve per triangle, RGBA per vertex, when any material reads vertex colours; empty otherwise. */
  colors: Float32Array;
  /** Shader varyings, three corners per triangle of a shader material, laid out as each program asks. */
  varyings: Float32Array;
  /** One material index per triangle. */
  materials: Uint32Array;
  triangleCount: number;
  /** Four floats per material: linear rgb and opacity. */
  materialColors: Float32Array;
  /** Four floats per material: linear emissive rgb and its intensity. */
  materialEmissive: Float32Array;
  /** Every material of the scene, resolved, in the order the triangle indices refer to them. */
  materialProperties: MaterialProperties[];
  /** The mesh specific half of each material entry. */
  materialPlacements: MaterialPlacement[];
  materialCount: number;
  /** Words the materials keep in the texture buffer: clipping planes and shader uniforms. */
  extraWords: Uint32Array;
  /**
   * Whether the triangles depend on the camera: points, lines and sprites face it, and a vertex shader projects through
   * it. Such a scene is packed again whenever the camera or the image size changes.
   */
  cameraDependent: boolean;
}

export interface PackOptions {
  /** The index in the texture table of each texture, by id. */
  textures?: ReadonlyMap<string, number>;
  /** The textures themselves, for the work done here on the CPU: displacement maps and vertex shaders. */
  textureData?: ReadonlyMap<string, SerializedTexture>;
  /** The camera and image size points, lines, sprites and vertex shaders are built for. */
  camera?: CameraMatrices;
  width?: number;
  height?: number;
  /** Translated fragment shaders, for the materials that have one. */
  programs?: ShaderPrograms;
  /** Where the extra words will start in the texture buffer, so the materials can point at them absolutely. */
  extraBase?: number;
  /** Scene fog, which a shader's fog uniforms read. */
  fog?: SerializedScene["fog"];
}

/** Three's wireframe and line width: WebGL draws every line one pixel wide whatever the material asks. */
const LINE_WIDTH = 1;

/** A dashed line split finer than this is drawn solid rather than as millions of dashes. */
const MAX_DASHES = 4096;

/** One corner as it is about to be written: world space, with everything a triangle stores per vertex. */
interface Corner {
  position: number[];
  normal: number[];
  uv0: number[];
  uv1: number[];
  color: number[];
  varyings: Float32Array | null;
}

/** A material entry: what the triangles index, and what it was made from. */
interface Entry {
  index: number;
  properties: MaterialProperties;
  placement: MaterialPlacement;
}

/** Grows a float array as triangles are added, since how many a primitive makes is only known as it is built. */
class FloatStream {
  data: Float32Array;
  length = 0;

  constructor(capacity: number) {
    this.data = new Float32Array(Math.max(16, capacity));
  }

  reserve(count: number) {
    if (this.length + count <= this.data.length) return;

    const next = new Float32Array(Math.max(this.data.length * 2, this.length + count));

    next.set(this.data.subarray(0, this.length));
    this.data = next;
  }

  push(values: ArrayLike<number>, count = values.length) {
    this.reserve(count);

    for (let i = 0; i < count; i++) this.data[this.length + i] = values[i] ?? 0;

    this.length += count;
  }

  pad(count: number) {
    this.reserve(count);
    this.data.fill(0, this.length, this.length + count);
    this.length += count;
  }

  view(): Float32Array {
    return this.data.subarray(0, this.length);
  }
}

function triangleCountOf(mesh: SerializedMesh): number {
  const vertices = mesh.index ? mesh.index.length : mesh.attributes.position.data.length / 3;

  const groups = mesh.groups.length
    ? mesh.groups.reduce((total, group) => total + Math.min(group.count, Math.max(0, vertices - group.start)), 0)
    : vertices;

  return Math.floor(groups / 3);
}

function determinant3(m: readonly number[]): number {
  return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5]);
}

const normalize = (v: readonly number[]) => {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;

  return [v[0] / length, v[1] / length, v[2] / length];
};

function multiply4(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(16).fill(0);

  for (let column = 0; column < 4; column++)
    for (let row = 0; row < 4; row++)
      for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];

  return out;
}

/** Three's normalMatrix for a model view matrix: the inverse transpose of its upper 3x3, column-major. */
function normalMatrix3(m: readonly number[]): number[] {
  return Array.from(normalMatrix(m));
}

/** The uniforms Three sets on every shader for this camera and mesh. */
export function builtinUniforms(camera: CameraMatrices | undefined, matrixWorld: readonly number[],
  fog?: SerializedScene["fog"]): Record<string, UniformValue> {
  const view = camera?.view ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const modelView = multiply4(view, matrixWorld);

  return {
    modelMatrix: [...matrixWorld],
    modelViewMatrix: modelView,
    normalMatrix: normalMatrix3(modelView),
    viewMatrix: [...view],
    projectionMatrix: [...camera?.projection ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, -1, 0, 0, -0.2, 0]],
    cameraPosition: [...camera?.position ?? [0, 0, 0]],
    isOrthographic: camera?.orthographic ?? false,
    cameraNear: camera?.near ?? 0.1,
    cameraFar: camera?.far ?? 1000,
    fogColor: fog?.color ?? [0, 0, 0],
    fogNear: fog?.near ?? 0,
    fogFar: fog?.far ?? 0,
    fogDensity: fog?.density ?? 0,
  };
}

/** The same uniforms in the form the CPU renderer's interpreter reads, where a matrix is a tagged value. */
function interpreterUniforms(uniforms: Record<string, UniformValue>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...uniforms };

  for (const name of ["modelMatrix", "modelViewMatrix", "normalMatrix", "viewMatrix", "projectionMatrix"])
    out[name] = matrixValue(uniforms[name] as number[]);

  return out;
}

/** A flat number array from whatever an interpreter returned, or null when it is not one. */
function flatten(value: unknown): number[] | null {
  if (typeof value === "number") return [value];
  if (typeof value === "boolean") return [value ? 1 : 0];
  if (Array.isArray(value)) return value.flatMap(entry => flatten(entry) ?? [0]);
  if (value && typeof value === "object" && "matrix" in value) return [...(value as { matrix: number[] }).matrix];

  return null;
}

/**
 * Flattens the scene into triangles, materials resolved and transforms applied. Every triangle carries its own three
 * vertices: the path tracer reads a hit through its triangle index, and sharing vertices would cost an indirection on
 * the hottest path of the kernel for memory that low poly scenes do not miss.
 *
 * Points, lines and sprites become triangles that face the camera, sized the way Three draws them, so a path has a
 * surface to find; vertex shaders run here, on the CPU renderer's interpreter, and their varyings ride along per corner.
 */
export function packGeometry(
  scene: SerializedScene, options: PackOptions | ReadonlyMap<string, number> = {}): PackedGeometry {
  const settings: PackOptions = options instanceof Map ? { textures: options } : options as PackOptions;
  const textures = settings.textures ?? new Map<string, number>();
  const textureData = settings.textureData ?? new Map<string, SerializedTexture>();
  const camera = settings.camera;
  const width = Math.max(1, settings.width ?? 1);
  const height = Math.max(1, settings.height ?? 1);
  const programs = settings.programs;
  const extraBase = settings.extraBase ?? 0;
  const meshes = scene.meshes.filter(mesh => mesh.attributes.position);
  const estimate = meshes.reduce((count, mesh) => count + (mesh.primitive ? 0 : triangleCountOf(mesh)), 0);

  const anyVertexColors = meshes.some(mesh =>
    mesh.attributes.color && mesh.materials.some(m => m?.values?.vertexColors === true));

  const positions = new FloatStream(estimate * 9);
  const normals = new FloatStream(estimate * 9);
  const uvs = new FloatStream(estimate * 12);
  const colors = new FloatStream(anyVertexColors ? estimate * 12 : 0);
  const varyings = new FloatStream(0);
  const triangleMaterials: number[] = [];
  const entries: Entry[] = [];
  const shared = new Map<SerializedMaterial | undefined, Entry>();
  const extra: number[] = [];
  let cameraDependent = false;

  /** A material's clipping planes, written once to the extra words. */
  const clipPlanes = new Map<SerializedMaterial, { start: number; count: number }>();

  const planesOf = (material: SerializedMaterial | undefined) => {
    const planes = material?.values?.clippingPlanes;

    if (!material || !Array.isArray(planes) || !planes.length) return { start: 0, count: 0 };

    let known = clipPlanes.get(material);

    if (!known) {
      known = { start: extraBase + extra.length, count: 0 };

      for (const plane of planes) {
        if (!Array.isArray(plane) || plane.length < 4) continue;

        const floats = new Float32Array((plane as number[]).slice(0, 4));

        extra.push(...new Uint32Array(floats.buffer));
        known.count++;
      }

      clipPlanes.set(material, known);
    }

    return known;
  };

  /** One entry per material, or one per mesh and group when the entry carries something of the mesh. */
  const entryFor = (material: SerializedMaterial | undefined, mesh: SerializedMesh): Entry => {
    const program = material ? programs?.fragment(material) ?? null : null;
    const perMesh = !!program || material?.values?.normalMapType === 1;

    if (!perMesh) {
      const known = shared.get(material);

      if (known) return known;
    }

    const properties = materialProperties(material, textures);
    const planes = planesOf(material);

    // Points, lines and sprites always face the camera, so Three never culls them.
    if (mesh.primitive) properties.side = 2;

    const placement: MaterialPlacement = {
      ...NO_PLACEMENT,
      clipStart: planes.start,
      clipCount: planes.count,
      normalMatrix: perMesh ? Array.from(normalMatrix(mesh.matrixWorld)) : NO_PLACEMENT.normalMatrix,
    };

    if (program && material) {
      const uniforms: Record<string, UniformValue> = {
        ...builtinUniforms(camera, mesh.matrixWorld, settings.fog),
        ...material.shader?.uniforms,
        ...material.time !== undefined ? { time: material.time } : {},
      };

      const words = program.packUniforms(uniforms, textures);

      placement.program = program.id;
      placement.shaderData = extraBase + extra.length;
      placement.varyingStride = program.varyingStride;
      placement.varyingBase = varyings.length;
      placement.firstTriangle = triangleMaterials.length;

      for (let i = 0; i < words.length; i++) extra.push(words[i]);

      const after = programs?.after(material) ?? null;

      if (after) {
        const afterWords = after.packUniforms(uniforms, textures);

        placement.afterProgram = after.id;
        placement.afterData = extraBase + extra.length;

        for (let i = 0; i < afterWords.length; i++) extra.push(afterWords[i]);
      }
    }

    const entry = { index: entries.length, properties, placement };

    entries.push(entry);

    if (!perMesh) shared.set(material, entry);

    return entry;
  };

  const writeTriangle = (corners: readonly Corner[], entry: Entry) => {
    // Geometry without normals is shaded flat: the three corners share the face normal.
    const flat = corners.some(corner => corner.normal.length < 3)
      ? faceNormal(corners[0].position, corners[1].position, corners[2].position)
      : null;

    for (const corner of corners) {
      positions.push(corner.position, 3);
      normals.push(flat ?? corner.normal, 3);
      uvs.push([corner.uv0[0], corner.uv0[1], corner.uv1[0], corner.uv1[1]], 4);

      if (anyVertexColors) colors.push(corner.color, 4);
    }

    const stride = entry.placement.varyingStride;

    if (stride > 0)
      for (const corner of corners) {
        if (corner.varyings) varyings.push(corner.varyings, stride);
        else varyings.pad(stride);
      }

    triangleMaterials.push(entry.index);
  };

  /** Every vertex of a plain mesh, transformed once and shared by its groups. */
  let plain: PlainVertices | null = null;

  const writePlainTriangles = (mesh: SerializedMesh, start: number, end: number, mirrored: boolean, entry: Entry) => {
    if (plain?.mesh !== mesh) plain = plainVertices(mesh);

    const { world, normal, hasNormal, uv, color, valid } = plain;
    const index = mesh.index;
    const stride = entry.placement.varyingStride;
    const corners = [0, 0, 0];

    for (let v = start; v + 2 < end; v += 3) {
      const a = index ? index[v] : v, b = index ? index[v + 1] : v + 1, c = index ? index[v + 2] : v + 2;

      if (!valid(a) || !valid(b) || !valid(c)) continue;

      corners[0] = a;
      corners[1] = mirrored ? c : b;
      corners[2] = mirrored ? b : c;

      // Geometry without normals is shaded flat: the three corners share the face normal.
      const flat = !hasNormal[a] || !hasNormal[b] || !hasNormal[c]
        ? faceNormal(
          Array.from(world.subarray(corners[0] * 3, corners[0] * 3 + 3)),
          Array.from(world.subarray(corners[1] * 3, corners[1] * 3 + 3)),
          Array.from(world.subarray(corners[2] * 3, corners[2] * 3 + 3)))
        : null;

      positions.reserve(9);
      normals.reserve(9);
      uvs.reserve(12);

      if (anyVertexColors) colors.reserve(12);

      for (const corner of corners) {
        const p = corner * 3, q = corner * 4;

        positions.data[positions.length++] = world[p];
        positions.data[positions.length++] = world[p + 1];
        positions.data[positions.length++] = world[p + 2];

        if (flat) {
          normals.data[normals.length++] = flat[0];
          normals.data[normals.length++] = flat[1];
          normals.data[normals.length++] = flat[2];
        } else {
          normals.data[normals.length++] = normal[p];
          normals.data[normals.length++] = normal[p + 1];
          normals.data[normals.length++] = normal[p + 2];
        }

        uvs.data[uvs.length++] = uv[q];
        uvs.data[uvs.length++] = uv[q + 1];
        uvs.data[uvs.length++] = uv[q + 2];
        uvs.data[uvs.length++] = uv[q + 3];

        if (anyVertexColors) {
          colors.data[colors.length++] = color[q];
          colors.data[colors.length++] = color[q + 1];
          colors.data[colors.length++] = color[q + 2];
          colors.data[colors.length++] = color[q + 3];
        }
      }

      if (stride > 0) varyings.pad(3 * stride);

      triangleMaterials.push(entry.index);
    }
  };

  for (const mesh of meshes) {
    const groups = mesh.groups.length
      ? mesh.groups
      : [{
        start: 0,
        count: mesh.index ? mesh.index.length : mesh.attributes.position.data.length / 3,
        materialIndex: 0,
      }];

    if (mesh.primitive) cameraDependent = true;

    for (const group of groups) {
      const material = mesh.materials[group.materialIndex] ?? mesh.materials[0];

      if (!isMaterialVisible(material)) continue;

      const entry = entryFor(material, mesh);
      const vertexShader = material ? programs?.vertex(material) ?? null : null;

      if (vertexShader) cameraDependent = true;

      // Primitives that face the camera need one to face; without it there is nothing sensible to build.
      if (mesh.primitive && !camera) continue;

      const context = { camera, width, height, textureData, vertexShader, programs, fog: settings.fog };
      const vertices = new VertexReader(mesh, material, context);
      const count = mesh.index ? mesh.index.length : mesh.attributes.position.data.length / 3;
      const start = Math.max(0, group.start);
      const end = Math.min(group.start + group.count, count);
      const at = (offset: number) => (mesh.index ? mesh.index[offset] : offset);

      if (!mesh.primitive) {
        // A mirroring transform turns the winding around; Three flips its front face to match, and so does this.
        const mirrored = determinant3(mesh.matrixWorld) < 0;
        // Instanced geometry drawn by a shader (Line2, particles built in a vertex shader): the template runs once per
        // instance, reading the per-instance attributes at that instance.
        const instancing = vertexShader ? instancedAttributes(mesh) : null;

        // The common case, a plain mesh, skips the general reader: each vertex is transformed once, straight into
        // typed arrays, and each triangle is copied from them. What it writes is exactly what the reader would.
        if (!vertexShader && !material?.maps?.displacementMap) {
          writePlainTriangles(mesh, start, end, mirrored, entry);
          continue;
        }

        for (let instance = 0; instance < (instancing?.count ?? 1); instance++) {
          const reader = instancing ? new VertexReader(mesh, material, context, instancing.names, instance) : vertices;

          for (let v = start; v + 2 < end; v += 3) {
            const a = reader.corner(at(v)), b = reader.corner(at(v + 1)), c = reader.corner(at(v + 2));

            if (!a || !b || !c) continue;

            writeTriangle(mirrored ? [a, c, b] : [a, b, c], entry);
          }
        }

        continue;
      }

      if (mesh.primitive === "points") {
        for (let v = start; v < end; v++)
          for (const triangle of vertices.point(at(v))) writeTriangle(triangle, entry);

        continue;
      }

      if (mesh.primitive === "sprite") {
        for (let v = start; v + 2 < end; v += 3) {
          const a = vertices.corner(at(v)), b = vertices.corner(at(v + 1)), c = vertices.corner(at(v + 2));

          if (a && b && c) writeTriangle([a, b, c], entry);
        }

        continue;
      }

      // Lines: segments, a strip, or a strip that closes on itself.
      const segments: [number, number][] = [];

      if (mesh.primitive === "lineSegments") {
        for (let v = start; v + 1 < end; v += 2) segments.push([at(v), at(v + 1)]);
      } else {
        for (let v = start; v + 1 < end; v++) segments.push([at(v), at(v + 1)]);

        if (mesh.primitive === "lineLoop" && end - start > 2) segments.push([at(end - 1), at(start)]);
      }

      for (const [a, b] of segments)
        for (const triangle of vertices.segment(a, b)) writeTriangle(triangle, entry);
    }
  }

  const properties = entries.map(entry => entry.properties);

  return {
    positions: positions.view(),
    normals: normals.view(),
    uvs: uvs.view(),
    colors: colors.view(),
    varyings: varyings.view(),
    materials: new Uint32Array(triangleMaterials),
    triangleCount: triangleMaterials.length,
    materialColors: new Float32Array(properties.flatMap(m => [...m.color, m.opacity])),
    // Only a lit or plainly emissive surface is a light: a shader's colour is not known until it runs. An unlit
    // material of a fixed colour emits that colour, so it is one too.
    materialEmissive: new Float32Array(properties.flatMap(m =>
      m.kind === MATERIAL_KIND.shader ? [0, 0, 0, 0]
        : (m.flags & MATERIAL_UNLIT_LIGHT) !== 0 ? [...m.color, 1]
          : [...m.emissive, m.emissiveIntensity])),
    materialProperties: properties,
    materialPlacements: entries.map(entry => entry.placement),
    materialCount: properties.length,
    extraWords: new Uint32Array(extra),
    cameraDependent,
  };
}

/** A plain mesh's vertices in world space, read lazily: only the ones some triangle uses are transformed. */
interface PlainVertices {
  mesh: SerializedMesh;
  world: Float32Array;
  normal: Float32Array;
  hasNormal: Uint8Array;
  /** Four per vertex: the first UV set, then the second. */
  uv: Float32Array;
  /** Four per vertex, RGBA. */
  color: Float32Array;
  /** Whether the vertex exists and has a finite position, transforming it on first use. */
  valid(index: number): boolean;
}

function plainVertices(mesh: SerializedMesh): PlainVertices {
  const position = mesh.attributes.position;
  const count = Math.floor(position.data.length / Math.max(1, position.itemSize));
  const world = new Float32Array(count * 3);
  const normal = new Float32Array(count * 3);
  const hasNormal = new Uint8Array(count);
  const uv = new Float32Array(count * 4);
  const color = new Float32Array(count * 4);
  // 0 not read yet, 1 usable, 2 refused.
  const state = new Uint8Array(count);
  const rotation = normalMatrix(mesh.matrixWorld);
  const matrix = mesh.matrixWorld as number[];
  const { normal: normals, uv: uv0, uv1, color: colors } = mesh.attributes;

  /** Component `c` of a vertex, as the reader's `Array.from(subarray(...))[c]` would see it. */
  const component = (attribute: SerializedMesh["attributes"][string], vertex: number, c: number): number | undefined => {
    const at = vertex * attribute.itemSize + c;

    return c < attribute.itemSize && at < attribute.data.length ? attribute.data[at] : undefined;
  };

  const read = (vertex: number): boolean => {
    const x = component(position, vertex, 0), y = component(position, vertex, 1), z = component(position, vertex, 2);

    if (x === undefined || y === undefined || z === undefined) return false;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;

    transformPoint(matrix, x, y, z, world, vertex * 3);

    if (normals) {
      const nx = component(normals, vertex, 0), ny = component(normals, vertex, 1), nz = component(normals, vertex, 2);

      if (nx !== undefined && ny !== undefined && nz !== undefined) {
        transformNormal(rotation, nx, ny, nz, normal, vertex * 3);
        hasNormal[vertex] = 1;
      }
    }

    // A missing second set falls back to the first, and a missing component to zero, as the pushed stream would.
    const second = uv1 ?? uv0;

    uv[vertex * 4] = uv0 ? component(uv0, vertex, 0) ?? 0 : 0;
    uv[vertex * 4 + 1] = uv0 ? component(uv0, vertex, 1) ?? 0 : 0;
    uv[vertex * 4 + 2] = second ? component(second, vertex, 0) ?? 0 : 0;
    uv[vertex * 4 + 3] = second ? component(second, vertex, 1) ?? 0 : 0;

    if (colors) {
      const available = Math.max(0, Math.min(colors.itemSize, colors.data.length - vertex * colors.itemSize));

      color[vertex * 4] = component(colors, vertex, 0) ?? 1;
      color[vertex * 4 + 1] = component(colors, vertex, 1) ?? 1;
      color[vertex * 4 + 2] = component(colors, vertex, 2) ?? 1;
      color[vertex * 4 + 3] = available >= 4 ? component(colors, vertex, 3) ?? 0 : 1;
    } else {
      color.fill(1, vertex * 4, vertex * 4 + 4);
    }

    return true;
  };

  return {
    mesh,
    world,
    normal,
    hasNormal,
    uv,
    color,
    valid(vertex: number) {
      if (!(vertex >= 0 && vertex < count)) return false;

      if (state[vertex] === 0) state[vertex] = read(vertex) ? 1 : 2;

      return state[vertex] === 1;
    },
  };
}

interface VertexContext {
  camera?: CameraMatrices;
  width: number;
  height: number;
  textureData: ReadonlyMap<string, SerializedTexture>;
  vertexShader: CpuShader | null;
  programs?: ShaderPrograms;
  fog?: SerializedScene["fog"];
}

/** What one vertex of a mesh becomes: computed once, whichever triangles share it. */
interface ProcessedVertex {
  world: number[];
  normal: number[];
  uv0: number[];
  uv1: number[];
  color: number[];
  varyings: Float32Array | null;
  clip: number[] | null;
  pointSize: number;
  lineDistance: number;
}

/** Reads, deforms and projects the vertices of one mesh group, caching each one. */
class VertexReader {
  private readonly cache = new Map<number, ProcessedVertex | null>();
  private readonly rotation: Float64Array;
  private readonly program;
  private readonly shaderInputs: Record<string, unknown> | null;

  constructor(
    private readonly mesh: SerializedMesh,
    private readonly material: SerializedMaterial | undefined,
    private readonly context: VertexContext,
    private readonly perInstance: ReadonlySet<string> = new Set(),
    private readonly instance = 0) {
    this.rotation = normalMatrix(mesh.matrixWorld);
    this.program = material ? context.programs?.fragment(material) ?? null : null;

    this.shaderInputs = context.vertexShader && material?.shader
      ? {
        ...interpreterUniforms(builtinUniforms(context.camera, mesh.matrixWorld, context.fog)),
        ...material.shader.uniforms,
        ...material.time !== undefined ? { time: material.time } : {},
      }
      : null;
  }

  private value(name: string): number {
    const value = this.material?.values?.[name];

    return typeof value === "number" && Number.isFinite(value) ? value : NaN;
  }

  private sample(id: string | undefined, uv: readonly number[]): number[] | null {
    const texture = id ? this.context.textureData.get(id) : undefined;

    return texture ? sampleTexture(texture, [uv[0], uv[1]], true) : null;
  }

  vertex(index: number): ProcessedVertex | null {
    if (this.cache.has(index)) return this.cache.get(index) ?? null;

    const processed = this.process(index);

    this.cache.set(index, processed);

    return processed;
  }

  private attribute(name: string, index: number): number[] | undefined {
    const attribute = this.mesh.attributes[name];

    if (!attribute) return undefined;

    const at = (this.perInstance.has(name) ? this.instance : index) * attribute.itemSize;

    return Array.from(attribute.data.subarray(at, at + attribute.itemSize));
  }

  private process(index: number): ProcessedVertex | null {
    const mesh = this.mesh;
    const camera = this.context.camera;
    let position = this.attribute("position", index);

    if (!position || position.length < 3 || position.some(value => !Number.isFinite(value))) return null;

    let normal = this.attribute("normal", index) ?? null;
    const uv = this.attribute("uv", index) ?? [0, 0];
    const uv1 = this.attribute("uv1", index) ?? uv;
    const color = this.attribute("color", index);
    const displacement = this.material?.maps?.displacementMap;

    // Three moves each vertex along its normal by the displacement map, in the vertex shader, before anything else.
    if (displacement) {
      const channel = this.context.textureData.get(displacement)?.channel ?? 0;
      const texel = this.sample(displacement, channel ? uv1 : uv);
      const n = normal ?? [0, 0, 1];
      const scale = Number.isFinite(this.value("displacementScale")) ? this.value("displacementScale") : 1;
      const bias = Number.isFinite(this.value("displacementBias")) ? this.value("displacementBias") : 0;
      const offset = texel ? texel[0] * scale + bias : 0;

      position = [position[0] + n[0] * offset, position[1] + n[1] * offset, position[2] + n[2] * offset];
    }

    let clip: number[] | null = null;
    let pointSize = Number.isFinite(this.value("size")) ? this.value("size") : 1;
    let varyings: Float32Array | null = null;
    const shader = this.context.vertexShader;

    if (shader && this.shaderInputs) {
      const attributes = Object.fromEntries(Object.keys(mesh.attributes).map(name => {
        const value = this.attribute(name, index) ?? [0];

        return [name, value.length === 1 ? value[0] : value];
      }));

      const result = shader.run({
        ...this.shaderInputs,
        ...attributes,
        uPointScale: this.context.height / (2 * Math.tan((camera?.fov ?? 50) * Math.PI / 360)),
        gl_PointSize: 1,
        position,
        normal: normal ?? [0, 0, 1],
        transformed: position,
        objectNormal: normal ?? [0, 0, 1],
        cpuPosition: position,
        cpuNormal: normal ?? [0, 0, 1],
        gl_Position: [NaN, NaN, NaN, NaN],
        gl_VertexID: index,
      } as never, {
        texture: (sampler, coordinate) => {
          const id = sampler && typeof sampler === "object" && !Array.isArray(sampler)
            ? (sampler as { texture?: unknown }).texture : undefined;

          const texture = typeof id === "string" ? this.context.textureData.get(id) : undefined;

          return texture ? sampleTexture(texture, coordinate, false) : [0, 0, 0, 1];
        },
      });

      if (!result) return null;

      const projected = flatten(result.gl_Position);

      if (this.material?.shader?.patched) {
        position = flatten(result.cpuPosition)?.slice(0, 3) ?? position;
        normal = flatten(result.cpuNormal)?.slice(0, 3) ?? normal;

        if (projected?.length === 4 && projected.every(Number.isFinite)) clip = projected;
      } else if (projected?.length === 4 && projected.every(Number.isFinite)) {
        clip = projected;
      } else {
        return null;
      }

      const size = Number(result.gl_PointSize ?? 1);

      if (Number.isFinite(size)) pointSize = size;

      if (this.program) {
        varyings = new Float32Array(this.program.varyingStride);

        for (const varying of this.program.varyings) {
          const values = flatten(result[varying.name]);

          if (values) for (let i = 0; i < varying.size; i++) varyings[varying.offset + i] = values[i] ?? 0;
        }
      }
    }

    const world = new Float32Array(3);

    transformPoint(mesh.matrixWorld as number[], position[0], position[1], position[2], world, 0);

    let worldPosition = Array.from(world);

    // A vertex shader's clip position is the truth for where the vertex is: displacement and billboards both live there.
    if (clip && camera && clip[3] !== 0) worldPosition = clipToWorld(camera, clip);

    const worldNormal = new Float32Array(3);

    if (normal) transformNormal(this.rotation, normal[0], normal[1], normal[2], worldNormal, 0);

    const lineDistance = (this.attribute("lineDistance", index)?.[0] ?? 0)
      * (Number.isFinite(this.value("scale")) ? this.value("scale") : 1);

    return {
      world: worldPosition,
      normal: normal ? Array.from(worldNormal) : [],
      uv0: uv.slice(0, 2),
      uv1: uv1.slice(0, 2),
      color: color ? [color[0] ?? 1, color[1] ?? 1, color[2] ?? 1, color.length >= 4 ? color[3] : 1] : [1, 1, 1, 1],
      varyings,
      clip,
      pointSize,
      lineDistance,
    };
  }

  /** A mesh corner, or a sprite's, which faces the camera. */
  corner(index: number): Corner | null {
    const vertex = this.vertex(index);

    if (!vertex) return null;

    if (this.mesh.primitive === "sprite" && !this.context.vertexShader) return this.spriteCorner(index, vertex);

    return {
      position: vertex.world,
      normal: vertex.normal,
      uv0: vertex.uv0,
      uv1: vertex.uv1,
      color: vertex.color,
      varyings: vertex.varyings,
    };
  }

  /** Three's sprite vertex shader: the quad is laid out in view space around the sprite's centre, then rotated. */
  private spriteCorner(index: number, vertex: ProcessedVertex): Corner | null {
    const camera = this.context.camera;
    const position = this.attribute("position", index);

    if (!camera || !position) return null;

    const m = this.mesh.matrixWorld;
    const centre = transform4(camera.view, m[12], m[13], m[14]);
    const attenuation = !camera.orthographic && this.material?.values?.sizeAttenuation === false ? -centre[2] : 1;
    const sx = Math.hypot(m[0], m[1], m[2]) * attenuation;
    const sy = Math.hypot(m[4], m[5], m[6]) * attenuation;
    const pivot = this.mesh.center ?? [0.5, 0.5];
    const x = (position[0] - (pivot[0] - 0.5)) * sx;
    const y = (position[1] - (pivot[1] - 0.5)) * sy;
    const rotation = Number.isFinite(this.value("rotation")) ? this.value("rotation") : 0;
    const c = Math.cos(rotation), s = Math.sin(rotation);
    const view = [centre[0] + x * c - y * s, centre[1] + x * s + y * c, centre[2]];
    const world = transform4(camera.cameraWorld, view[0], view[1], view[2]).slice(0, 3);

    return {
      position: world,
      normal: this.towardsCamera(world),
      uv0: vertex.uv0,
      uv1: vertex.uv1,
      color: vertex.color,
      varyings: vertex.varyings,
    };
  }

  private towardsCamera(point: readonly number[]): number[] {
    const camera = this.context.camera;

    if (!camera) return [0, 0, 1];

    if (camera.orthographic) return normalize([camera.cameraWorld[8], camera.cameraWorld[9], camera.cameraWorld[10]]);

    return normalize([camera.position[0] - point[0], camera.position[1] - point[1], camera.position[2] - point[2]]);
  }

  private clipOf(vertex: ProcessedVertex): number[] {
    return vertex.clip ?? worldToClip(this.context.camera as CameraMatrices, vertex.world);
  }

  /** A clip space corner turned back into a world space one, carrying the vertex's attributes. */
  private cornerAt(vertex: ProcessedVertex, clip: number[], uv0?: number[]): Corner {
    const world = clipToWorld(this.context.camera as CameraMatrices, clip);

    return {
      position: world,
      normal: this.towardsCamera(world),
      uv0: uv0 ?? vertex.uv0,
      uv1: vertex.uv1,
      color: vertex.color,
      varyings: vertex.varyings,
    };
  }

  /** A point: a square of `size` pixels facing the camera, its UV running over the square as gl_PointCoord does. */
  point(index: number): Corner[][] {
    const vertex = this.vertex(index);
    const camera = this.context.camera;

    if (!vertex || !camera) return [];

    const clip = this.clipOf(vertex);
    let size = vertex.pointSize;

    // Three's points vertex shader scales the size by the distance when the material asks for attenuation.
    if (!this.context.vertexShader && !camera.orthographic && this.material?.values?.sizeAttenuation !== false) {
      const depth = -transform4(camera.view, vertex.world[0], vertex.world[1], vertex.world[2])[2];

      size *= this.context.height / (2 * depth);
    }

    if (!(size > 0) || !(clip[3] > 0) || !Number.isFinite(size)) return [];

    const corners = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => this.cornerAt(vertex, [
      clip[0] + (u * 2 - 1) * size / this.context.width * clip[3],
      clip[1] + (v * 2 - 1) * size / this.context.height * clip[3],
      clip[2],
      clip[3],
    ], [u, v]));

    return [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]];
  }

  /** A line segment: a quad one pixel wide in screen space, cut into dashes for a dashed material. */
  segment(first: number, second: number): Corner[][] {
    const a = this.vertex(first), b = this.vertex(second);

    if (!a || !b || !this.context.camera) return [];

    const dash = this.material?.type === "LineDashedMaterial"
      ? { size: Number.isFinite(this.value("dashSize")) ? this.value("dashSize") : 3,
        gap: Number.isFinite(this.value("gapSize")) ? this.value("gapSize") : 1 }
      : null;

    const pieces: [number, number][] = [];

    if (dash && dash.size > 0 && dash.size + dash.gap > 0 && dash.gap > 0) {
      const period = dash.size + dash.gap;
      const from = a.lineDistance, to = b.lineDistance;
      const low = Math.min(from, to), high = Math.max(from, to);

      if ((high - low) / period > MAX_DASHES) {
        pieces.push([0, 1]);
      } else if (high > low) {
        for (let k = Math.floor(low / period); k * period < high; k++) {
          const start = Math.max(low, k * period), end = Math.min(high, k * period + dash.size);

          if (end > start) {
            const t0 = (start - from) / (to - from), t1 = (end - from) / (to - from);

            pieces.push([Math.min(t0, t1), Math.max(t0, t1)]);
          }
        }
      } else if (((low % period) + period) % period <= dash.size) {
        pieces.push([0, 1]);
      }
    } else {
      pieces.push([0, 1]);
    }

    return pieces.flatMap(([t0, t1]) => this.quad(a, b, t0, t1));
  }

  private quad(a: ProcessedVertex, b: ProcessedVertex, t0: number, t1: number): Corner[][] {
    const lerp = (x: number[], y: number[], t: number) => x.map((value, i) => value + (y[i] - value) * t);
    const clipA = this.clipOf(a), clipB = this.clipOf(b);
    let start = lerp(clipA, clipB, t0), end = lerp(clipA, clipB, t1);

    // The centre line is clipped against the near and far planes before the divide, as the rasterizer does.
    for (const sign of [1, -1]) {
      const da = start[3] + sign * start[2], db = end[3] + sign * end[2];

      if (da < 0 && db < 0) return [];

      if (da < 0 || db < 0) {
        const t = da / (da - db);
        const cut = lerp(start, end, t);

        if (da < 0) start = cut;
        else end = cut;
      }
    }

    if (start[3] <= 0 || end[3] <= 0) return [];

    const dx = (end[0] / end[3] - start[0] / start[3]) * this.context.width;
    const dy = (end[1] / end[3] - start[1] / start[3]) * this.context.height;
    const length = Math.hypot(dx, dy);

    if (length === 0) return [];

    const offset = (clip: number[], sign: number) => [
      clip[0] - dy / length * LINE_WIDTH / this.context.width * clip[3] * sign,
      clip[1] + dx / length * LINE_WIDTH / this.context.height * clip[3] * sign,
      clip[2],
      clip[3],
    ];

    const vertexAt = (t: number): ProcessedVertex => {
      const w = t0 + (t1 - t0) * t;

      return {
        ...a,
        world: lerp(a.world, b.world, w),
        uv0: lerp(a.uv0, b.uv0, w),
        uv1: lerp(a.uv1, b.uv1, w),
        color: lerp(a.color, b.color, w),
        varyings: a.varyings && b.varyings
          ? Float32Array.from(a.varyings, (value, i) => value + ((b.varyings as Float32Array)[i] - value) * w)
          : a.varyings,
      };
    };

    const head = vertexAt(0), tail = vertexAt(1);

    const corners = [
      this.cornerAt(head, offset(start, -1)),
      this.cornerAt(tail, offset(end, -1)),
      this.cornerAt(tail, offset(end, 1)),
      this.cornerAt(head, offset(start, 1)),
    ];

    return [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]];
  }
}

/**
 * The attributes of an instanced geometry, which the snapshot keeps as plain attributes with one element per instance
 * rather than per vertex, and how many instances there are.
 */
export function instancedAttributes(mesh: SerializedMesh): { names: Set<string>; count: number } | null {
  const vertices = mesh.attributes.position.data.length / 3;
  const names = new Set<string>();
  let count = Infinity;

  for (const [name, attribute] of Object.entries(mesh.attributes)) {
    const elements = Math.floor(attribute.data.length / attribute.itemSize);

    if (elements !== vertices && elements > 0) {
      names.add(name);
      count = Math.min(count, elements);
    }
  }

  return names.size ? { names, count } : null;
}

/** Flat shading for geometry that arrives without normals: the three vertices share the face normal. */
export function faceNormal(a: readonly number[], b: readonly number[], c: readonly number[]): number[] {
  const ax = b[0] - a[0], ay = b[1] - a[1], az = b[2] - a[2];
  const bx = c[0] - a[0], by = c[1] - a[1], bz = c[2] - a[2];

  return normalize([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]);
}

export { NO_PROGRAM };
