import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import test from "node:test";
import { reflect_shader } from "../shader_preprocessor.js";
import { MaterialPassType, ShaderResourceType } from "../../engine/src/renderer/renderer_types.js";

test("cooked bindings preserve actual entry-point visibility, including unused declarations", () => {
  const reflection = reflect_shader(`
    @group(0) @binding(0) var<storage, read> positions: array<vec4f>;
    @group(0) @binding(1) var<storage, read> colors: array<vec4f>;
    @group(0) @binding(2) var<storage, read> unused_data: array<vec4f>;
    fn read_color() -> vec4f { return colors[0]; }
    @vertex fn vs() -> @builtin(position) vec4f { return positions[0]; }
    @fragment fn fs() -> @location(0) vec4f { return read_color(); }
  `);
  assert.deepEqual(
    reflection.bind_groups[0].map((binding) => binding.visibility),
    [1, 2, 0]
  );
});

test("material pipeline layouts use each reflected binding's stage mask", () => {
  // Isolate the class from browser-only imports (including Vite's WASM loader).
  // Execute its real pipeline method with CPU-only WebGPU collaborators.
  const source = readFileSync(
    new URL("../../engine/src/renderer/material.js", import.meta.url),
    "utf8"
  );
  const class_source = source.slice(
    source.indexOf("export class MaterialTemplate"),
    source.indexOf("export class Material {")
  );
  const MaterialTemplate = runInNewContext(
    class_source.replace("export class", "class") + "\nMaterialTemplate;",
    {
      MaterialPassType,
      ShaderResourceType,
      GPUShaderStage: { VERTEX: 1, FRAGMENT: 2 },
      Shader: { resource_type_from_reflection_type: (type) => type },
      BindGroup: { create_layout: (name, bindings) => bindings },
      PipelineState: { create_render: (name, descriptor) => descriptor },
    }
  );
  const template = Object.create(MaterialTemplate.prototype);
  const bindings = [1, 2, 0, undefined].map((visibility, binding) => ({
    binding,
    visibility,
    access: "read",
    resourceType: ShaderResourceType.Storage,
  }));
  Object.assign(template, {
    name: "stage_visibility_test",
    shader: {
      module: {},
      reflection: { get_bind_groups: () => [bindings], entry: { fragment: [] } },
    },
    pipeline_state_config: {},
  });
  const pipeline = template.create_pipeline_state([]);
  assert.deepEqual(
    Array.from(pipeline.bind_layouts[0], (binding) => binding.visibility),
    [1, 2, 0, 3]
  );
});
