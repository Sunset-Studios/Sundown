import { DataType, BufferType } from "../meta/fragment_generator_types.js";

const LightFragment = {
  name: "Light",
  fields: {
    position: {
      type: DataType.FLOAT32,
      vector: { x: true, y: true, z: true },
      stride: 1,
    },
    direction: {
      type: DataType.FLOAT32,
      vector: { x: true, y: true, z: true },
      stride: 1,
    },
    color: {
      type: DataType.FLOAT32,
      vector: { r: true, g: true, b: true },
      stride: 1,
    },
    type: {
      type: DataType.UINT8,
      stride: 1,
    },
    intensity: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    radius: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    attenuation: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    outer_angle: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    active: {
      type: DataType.UINT8,
      stride: 1,
    },
    dirty: {
      type: DataType.UINT8,
      stride: 1,
    },
  },
  buffers: {
    light_fragment: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE,
      stride: 16,
      gpu_data: `
      let total_active = 0;
      for (let i = 0; i < this.size; i++) {
        if (this.data.active[i]) {
          total_active++;
        }
      }

      const gpu_data = new Float32Array(Math.max(total_active * 16, 16));
      let offset = 0;
      for (let i = 0; i < this.size; i++) {
        if (!this.data.active[i]) {
          continue;
        }

        gpu_data[offset] = this.data.position.x[i];
        gpu_data[offset + 1] = this.data.position.y[i];
        gpu_data[offset + 2] = this.data.position.z[i];
        gpu_data[offset + 3] = 0; // padding
        gpu_data[offset + 4] = this.data.direction.x[i];
        gpu_data[offset + 5] = this.data.direction.y[i];
        gpu_data[offset + 6] = this.data.direction.z[i];
        gpu_data[offset + 7] = 0; // padding
        gpu_data[offset + 8] = this.data.color.r[i];
        gpu_data[offset + 9] = this.data.color.g[i];
        gpu_data[offset + 10] = this.data.color.b[i];
        gpu_data[offset + 11] = this.data.type[i];
        gpu_data[offset + 12] = this.data.intensity[i];
        gpu_data[offset + 13] = this.data.radius[i];
        gpu_data[offset + 14] = this.data.attenuation[i];
        gpu_data[offset + 15] = this.data.outer_angle[i];

        offset += 16;
      }
      `,
    },
  },
};

const StaticMeshFragment = {
  name: "StaticMesh",
  constants: {
    material_slot_stride: 64,
  },
  fields: {
    mesh: {
      type: DataType.BIGINT64,
      stride: 1,
    },
    material_slots: {
      type: DataType.BIGINT64,
      stride: 64,
      getter: `StaticMeshFragment.data.material_slots.slice(this.current_entity * StaticMeshFragment.material_slot_stride, (this.current_entity + 1) * StaticMeshFragment.material_slot_stride);`,
      setter: `
      if (
        Array.isArray(value) &&
        value.length <= StaticMeshFragment.material_slot_stride
      ) {
        for (let i = 0; i < value.length; i++) {
          StaticMeshFragment.data.material_slots[this.current_entity * StaticMeshFragment.material_slot_stride + i] = BigInt(value[i]);
        }
      }
      if (StaticMeshFragment.data.dirty) {
        StaticMeshFragment.data.dirty[this.current_entity] = 1;
      }
      StaticMeshFragment.data.gpu_data_dirty = true;
      `,
    },
    instance_count: {
      type: DataType.BIGINT64,
      stride: 1,
    },
    dirty: {
      type: DataType.UINT8,
      stride: 1,
    },
  },
  overrides: {
    remove_entity: {
      skip_default: true,
      pre: `
      super.remove_entity(entity);
      const entity_data = this.get_entity_data(entity);
      entity_data.mesh = 0n;
      entity_data.material_slots = Array(this.material_slot_stride).fill(0);
      entity_data.instance_count = 0n;
      `,
    },
  },
};

const TransformFragment = {
  name: "Transform",
  fields: {
    position: {
      type: DataType.FLOAT32,
      stride: 4,
      getter: `[
        TransformFragment.data.position[this.current_entity * 4],
        TransformFragment.data.position[this.current_entity * 4 + 1],
        TransformFragment.data.position[this.current_entity * 4 + 2],
      ];
      `,
      setter: `
      TransformFragment.data.position[this.current_entity * 4] = value[0];
      TransformFragment.data.position[this.current_entity * 4 + 1] = value[1];
      TransformFragment.data.position[this.current_entity * 4 + 2] = value[2];
      TransformFragment.data.position[this.current_entity * 4 + 3] = 1.0;
      TransformFragment.data.position_buffer.write_raw(
        Renderer.get().graphics_context,
        TransformFragment.data.position.subarray(this.current_entity * 4, this.current_entity * 4 + 4),
        this.current_entity * 4 * Float32Array.BYTES_PER_ELEMENT
      );
      if (TransformFragment.data.dirty) {
        TransformFragment.data.dirty[this.current_entity] = 1;
        TransformFragment.data.dirty_flags_buffer.write_raw(
          Renderer.get().graphics_context,
          TransformFragment.data.dirty.subarray(this.current_entity, this.current_entity + 1),
          this.current_entity * Uint32Array.BYTES_PER_ELEMENT
        );
      }
      TransformFragment.data.gpu_data_dirty = true;
      `,
    },
    rotation: {
      type: DataType.FLOAT32,
      stride: 4,
      getter: `[
        TransformFragment.data.rotation[this.current_entity * 4],
        TransformFragment.data.rotation[this.current_entity * 4 + 1],
        TransformFragment.data.rotation[this.current_entity * 4 + 2],
        TransformFragment.data.rotation[this.current_entity * 4 + 3],
      ];
      `,
      setter: `
      TransformFragment.data.rotation[this.current_entity * 4] = value[0];
      TransformFragment.data.rotation[this.current_entity * 4 + 1] = value[1];
      TransformFragment.data.rotation[this.current_entity * 4 + 2] = value[2];
      TransformFragment.data.rotation[this.current_entity * 4 + 3] = value[3];
      TransformFragment.data.rotation_buffer.write_raw(
        Renderer.get().graphics_context,
        TransformFragment.data.rotation.subarray(this.current_entity * 4, this.current_entity * 4 + 4),
        this.current_entity * 4 * Float32Array.BYTES_PER_ELEMENT
      );
      if (TransformFragment.data.dirty) {
        TransformFragment.data.dirty[this.current_entity] = 1;
        TransformFragment.data.dirty_flags_buffer.write_raw(
          Renderer.get().graphics_context,
          TransformFragment.data.dirty.subarray(this.current_entity, this.current_entity + 1),
          this.current_entity * Uint32Array.BYTES_PER_ELEMENT
        );
      }
      TransformFragment.data.gpu_data_dirty = true;
      `,
    },
    scale: {
      type: DataType.FLOAT32,
      stride: 4,
      default: 1,
      getter: `[
        TransformFragment.data.scale[this.current_entity * 4],
        TransformFragment.data.scale[this.current_entity * 4 + 1],
        TransformFragment.data.scale[this.current_entity * 4 + 2],
      ];
      `,
      setter: `
      TransformFragment.data.scale[this.current_entity * 4] = value[0];
      TransformFragment.data.scale[this.current_entity * 4 + 1] = value[1];
      TransformFragment.data.scale[this.current_entity * 4 + 2] = value[2];
      TransformFragment.data.scale[this.current_entity * 4 + 3] = 0.0;
      TransformFragment.data.scale_buffer.write_raw(
        Renderer.get().graphics_context,
        TransformFragment.data.scale.subarray(this.current_entity * 4, this.current_entity * 4 + 4),
        this.current_entity * 4 * Float32Array.BYTES_PER_ELEMENT
      );
      if (TransformFragment.data.dirty) {
        TransformFragment.data.dirty[this.current_entity] = 1;
        TransformFragment.data.dirty_flags_buffer.write_raw(
          Renderer.get().graphics_context,
          TransformFragment.data.dirty.subarray(this.current_entity, this.current_entity + 1),
          this.current_entity * Uint32Array.BYTES_PER_ELEMENT
        );
      }
      TransformFragment.data.gpu_data_dirty = true;
      `,
    },
    dirty: {
      type: DataType.UINT32,
      stride: 1,
    },
  },
  buffers: {
    position: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE_SRC,
      stride: 4,
      cpu_buffer: true,
    },
    rotation: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE_SRC,
      stride: 4,
      cpu_buffer: true,
    },
    scale: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE_SRC,
      stride: 4,
      cpu_buffer: true,
    },
    dirty_flags: {
      type: DataType.UINT32,
      usage: BufferType.STORAGE_SRC,
      stride: 1,
      gpu_data: `
      const gpu_data = this.data.dirty;
      `,
    },
    transforms: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE,
      stride: 32,
    },
    inverse_transforms: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE,
      stride: 32,
    },
    bounds_data: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE,
      stride: 8,
    },
  },
  overrides: {
    remove_entity: {
      skip_default: true,
      pre: `
      super.remove_entity(entity);
      const entity_data = this.get_entity_data(entity);
      entity_data.position.x = 0;
      entity_data.position.y = 0;
      entity_data.position.z = 0;
      entity_data.rotation.x = 0;
      entity_data.rotation.y = 0;
      entity_data.rotation.z = 0;
      entity_data.rotation.w = 1;
      entity_data.scale.x = 1;
      entity_data.scale.y = 1;
      entity_data.scale.z = 1;
      `,
    },
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      return {
        position: {
          x: this.data.position[entity * 4],
          y: this.data.position[entity * 4 + 1],
          z: this.data.position[entity * 4 + 2],
        },
        rotation: {
          x: this.data.rotation[entity * 4],
          y: this.data.rotation[entity * 4 + 1],
          z: this.data.rotation[entity * 4 + 2],
          w: this.data.rotation[entity * 4 + 3],
        },
        scale: {
          x: this.data.scale[entity * 4],
          y: this.data.scale[entity * 4 + 1],
          z: this.data.scale[entity * 4 + 2],
        },
      };
      `,
    },
    to_gpu_data: {
      skip_default: true,
      pre: `
      if (!this.data) {
        this.initialize();
      }

      return {
        transforms_buffer: this.data.transforms_buffer,
        inverse_transforms_buffer: this.data.inverse_transforms_buffer,
        bounds_data_buffer: this.data.bounds_data_buffer,
        position_buffer: this.data.position_buffer,
        rotation_buffer: this.data.rotation_buffer,
        scale_buffer: this.data.scale_buffer,
        dirty_flags_buffer: this.data.dirty_flags_buffer,
      };
      `,
    },
    rebuild_buffers: {
      post: `
      const dirty_flags_buffer_size = this.data.dirty.byteLength;
      if (
        !this.data.dirty_flags_buffer ||
        this.data.dirty_flags_buffer.config.size < dirty_flags_buffer_size
      ) {
        this.data.dirty_flags_buffer = Buffer.create(context, {
          name: "transform_fragment_dirty_flags_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: this.data.dirty,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      }

      if (
        !this.data.transforms_buffer ||
        this.data.transforms_buffer.config.size <
        this.size * 32 * Float32Array.BYTES_PER_ELEMENT
      ) {
        this.data.transforms_buffer = Buffer.create(context, {
          name: "transform_fragment_transforms_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: new Float32Array(this.size * 32),
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      }

      if (
        !this.data.inverse_transforms_buffer ||
        this.data.inverse_transforms_buffer.config.size <
          this.size * 32 * Float32Array.BYTES_PER_ELEMENT
      ) {
        this.data.inverse_transforms_buffer = Buffer.create(context, {
          name: "transform_fragment_inverse_transforms_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: new Float32Array(this.size * 32),
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      }

      if (
        !this.data.bounds_data_buffer ||
        this.data.bounds_data_buffer.config.size <
        this.size * 8 * Float32Array.BYTES_PER_ELEMENT
      ) {
        this.data.bounds_data_buffer = Buffer.create(context, {
          name: "transform_fragment_bounds_data_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: new Float32Array(this.size * 8),
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      }
      `,
    },
  },
  hooks: {
    on_post_render: {
      body: `
    if (!this.data) {
      return;
    }

    await this.sync_buffers(Renderer.get().graphics_context);
    `,
    },
  },
};

const SceneGraphFragment = {
  name: "SceneGraph",
  imports: {
    Tree: "../../../memory/container.js",
  },
  fields: {
    parent: {
      type: DataType.INT32,
      stride: 1,
      setter: `
      SceneGraphFragment.data.parent[this.current_entity] = value ?? -1;
      SceneGraphFragment.data.scene_graph.add(value ?? null, this.current_entity);
      if (SceneGraphFragment.data.dirty) {
        SceneGraphFragment.data.dirty[this.current_entity] = 1;
      }
      SceneGraphFragment.data.gpu_data_dirty = true;
      `,
    },
    children: {
      type: DataType.INT32,
      stride: 1,
      getter: `SceneGraphFragment.data.scene_graph.find_node(this.current_entity)?.children ?? [];`,
      setter: `
      if (Array.isArray(value)) {
        SceneGraphFragment.data.scene_graph.add_multiple(this.current_entity, value, true /* replace_children */);
      }
      if (SceneGraphFragment.data.dirty) {
        SceneGraphFragment.data.dirty[this.current_entity] = 1;
      }
      SceneGraphFragment.data.gpu_data_dirty = true;
      `,
      no_fragment_array: true,
    },
  },
  members: {
    scene_graph: "new Tree()",
    scene_graph_layer_counts: "[]",
    scene_graph_uniforms: "[]",
  },
  buffers: {
    scene_graph: {
      type: DataType.INT32,
      usage: BufferType.STORAGE,
      stride: 2,
      gpu_data: `
      const { result, layer_counts } = this.data.scene_graph.flatten(Int32Array);
      this.data.scene_graph_layer_counts = layer_counts;

      const num_elements = result?.length ?? 0;
      const gpu_data = new Int32Array(Math.max(num_elements * 2, 2));
      for (let i = 0; i < num_elements; ++i) {
        gpu_data[i * 2] = result[i];
        gpu_data[i * 2 + 1] = this.data.parent[result[i]];
      }

      this.data.scene_graph_uniforms = new Array(layer_counts.length);
      let layer_offset = 0;
      for (let i = 0; i < layer_counts.length; ++i) {
        this.data.scene_graph_uniforms[i] = Buffer.create(context, {
          name: "scene_graph_uniforms_" + i,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          raw_data: new Uint32Array([layer_counts[i], layer_offset]),
          force: true,
        });
        layer_offset += layer_counts[i];
      }
      `,
    },
  },
  overrides: {
    remove_entity: {
      skip_default: true,
      post: `
      super.remove_entity(entity);
      this.data.parent[entity] = -1;
      this.data.scene_graph.remove(entity);
      this.data.gpu_data_dirty = true;
      `,
    },
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      const node = this.data.scene_graph.find_node(entity);
      return {
        parent: node?.parent ?? null,
        children: node?.children ?? [],
      };
      `,
    },
  },
};

const VisibilityFragment = {
  name: "Visibility",
  fields: {
    visible: {
      type: DataType.UINT8,
      stride: 4,
    },
  },
  buffers: {
    visible: {
      type: DataType.UINT8,
      usage: BufferType.STORAGE,
      stride: 4,
    },
  },
};

const UserInterfaceFragment = {
  name: "UserInterface",
  fields: {
    allows_cursor_events: {
      type: DataType.UINT8,
      stride: 1,
    },
    auto_size: {
      type: DataType.UINT8,
      stride: 1,
    },
    was_cursor_inside: {
      type: DataType.UINT8,
      stride: 1,
    },
    is_cursor_inside: {
      type: DataType.UINT8,
      stride: 1,
    },
    was_clicked: {
      type: DataType.UINT8,
      stride: 1,
    },
    is_clicked: {
      type: DataType.UINT8,
      stride: 1,
    },
    is_pressed: {
      type: DataType.UINT8,
      stride: 1,
    },
    was_pressed: {
      type: DataType.UINT8,
      stride: 1,
    },
    dirty: {
      type: DataType.UINT8,
      stride: 1,
    },
  },
};

const TextFragment = {
  name: "Text",
  fields: {
    text: {
      type: DataType.UINT32,
      stride: 1,
      is_container: true,
      getter: `String.fromCodePoint(...TextFragment.data.text.get_data_for_entity(this.current_entity));`,
      setter: `
      if (value) {
        const code_points = Array.from(value).map((char) => char.codePointAt(0));
        TextFragment.data.text.update(this.current_entity, code_points);
      }
      if (TextFragment.data.dirty) {
        TextFragment.data.dirty[this.current_entity] = 1;
      }
      TextFragment.data.gpu_data_dirty = true;
      `,
    },
    font: {
      type: DataType.INT32,
      stride: 1,
    },
    dirty: {
      type: DataType.UINT8,
      stride: 1,
    },
  },
  buffers: {
    text: {
      type: DataType.UINT32,
      usage: BufferType.STORAGE,
      stride: 1,
      gpu_data: `
const gpu_data = this.data.text.get_data();
      `,
    },
    dirty: {
      type: DataType.UINT32,
      usage: BufferType.STORAGE_SRC,
      stride: 1,
    },
  },
  overrides: {
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      const data = {};
      data.text = String.fromCodePoint(...this.data.text.get_data_for_entity(entity));
      data.font = this.data.font[entity];
      return data;
      `,
    },
  },
};

export const definitions = [
  LightFragment,
  StaticMeshFragment,
  TransformFragment,
  SceneGraphFragment,
  VisibilityFragment,
  UserInterfaceFragment,
  TextFragment,
];
