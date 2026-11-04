/**
 * Creating a shader module never throws: a WGSL mistake only surfaces later, as a pipeline that fails to build or a
 * device error with no line number. Asking for the compilation messages turns it back into something readable, which
 * is the difference between a five minute fix and an afternoon.
 */
export function createShaderModule(device: GPUDevice, label: string, code: string): GPUShaderModule {
  const module = device.createShaderModule({ label, code });

  void module.getCompilationInfo?.().then(info => {
    for (const message of info.messages) {
      const text = `GPU renderer: ${label} shader ${message.type} on line ${message.lineNum}: ${message.message}`;

      if (message.type === "error") console.error(text);
      else if (message.type === "warning") console.warn(text);
    }
  });

  return module;
}
