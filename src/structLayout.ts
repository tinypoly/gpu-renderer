/**
 * One description of a struct, from which both the WGSL declaration and the TypeScript writer come. The kernel's
 * `Material` and `Frame` structs have dozens of fields each; keeping their word offsets by hand in two languages is the
 * kind of mistake that compiles on both sides and then reads the wrong float on the device, where nothing can be printed.
 *
 * The alignment rules are WGSL's own for the host-shareable types used here.
 */
export type FieldType = "f32" | "u32" | "vec2u" | "vec4f" | "mat4f";

export interface StructField {
  name: string;
  type: FieldType;
  /** One line of documentation, carried into the generated WGSL. */
  doc?: string;
}

export interface StructLayout {
  name: string;
  /** The WGSL declaration. */
  wgsl: string;
  /** Size in 32 bit words, rounded up to the struct's alignment, which is also its array stride. */
  words: number;
  /** Where each field starts, in words. */
  offsets: Record<string, number>;
  types: Record<string, FieldType>;
}

const SPEC: Record<FieldType, { words: number; align: number; wgsl: string }> = {
  f32: { words: 1, align: 1, wgsl: "f32" },
  u32: { words: 1, align: 1, wgsl: "u32" },
  vec2u: { words: 2, align: 2, wgsl: "vec2<u32>" },
  vec4f: { words: 4, align: 4, wgsl: "vec4<f32>" },
  mat4f: { words: 16, align: 4, wgsl: "mat4x4<f32>" },
};

const roundUp = (value: number, align: number) => Math.ceil(value / align) * align;

export function layoutStruct(name: string, fields: readonly StructField[]): StructLayout {
  const offsets: Record<string, number> = {};
  const types: Record<string, FieldType> = {};
  const lines: string[] = [];
  let at = 0;
  let align = 1;

  for (const field of fields) {
    const spec = SPEC[field.type];

    if (field.name in offsets) throw new Error(`Duplicate field ${field.name} in ${name}.`);

    at = roundUp(at, spec.align);
    offsets[field.name] = at;
    types[field.name] = field.type;
    at += spec.words;
    align = Math.max(align, spec.align);

    if (field.doc) lines.push(`  // ${field.doc}`);

    lines.push(`  ${field.name}: ${spec.wgsl},`);
  }

  // A struct holding a vec4 is aligned to 16 bytes; anything smaller to 4, which a uniform still pads to 16.
  const words = roundUp(at, Math.max(align, 4));

  return { name, wgsl: `struct ${name} {\n${lines.join("\n")}\n};\n`, words, offsets, types };
}

/** Writes named values into a struct's words. Missing fields stay zero; unknown ones are an error. */
export function writeStruct(
  layout: StructLayout,
  target: ArrayBuffer,
  wordOffset: number,
  values: Record<string, number | readonly number[] | ArrayLike<number> | boolean>) {
  const floats = new Float32Array(target);
  const words = new Uint32Array(target);

  for (const [name, value] of Object.entries(values)) {
    const offset = layout.offsets[name];

    if (offset === undefined) throw new Error(`${layout.name} has no field ${name}.`);

    const at = wordOffset + offset;
    const type = layout.types[name];

    if (type === "u32" || type === "vec2u") {
      if (typeof value === "number" || typeof value === "boolean") words[at] = Number(value) >>> 0;
      else for (let i = 0; i < Math.min(value.length, SPEC[type].words); i++) words[at + i] = value[i] >>> 0;
    } else if (typeof value === "number" || typeof value === "boolean") {
      floats[at] = Number(value);
    } else {
      for (let i = 0; i < Math.min(value.length, SPEC[type].words); i++) floats[at + i] = value[i];
    }
  }
}
