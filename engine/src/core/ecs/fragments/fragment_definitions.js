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
  overrides: {
    update_entity_data: {
      post: `
      this.data.active[entity] = 1;
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
      this.update_entity_data(entity, {
        mesh: 0n,
        material_slots: Array(this.material_slot_stride).fill(0),
        instance_count: 0n,
      });
      `,
    },
    update_entity_data: {
      skip_default: true,
      pre: `
      if (!this.data) {
        this.initialize();
      }

      this.data.mesh[entity] = BigInt(data.mesh) ?? 0n;
      this.data.instance_count[entity] = BigInt(data.instance_count) ?? 0n;

      if (
        Array.isArray(data.material_slots) &&
        data.material_slots.length <= this.material_slot_stride
      ) {
        for (let i = 0; i < data.material_slots.length; i++) {
          this.data.material_slots[entity * this.material_slot_stride + i] = BigInt(data.material_slots[i]);
        }
      }

      this.data.dirty[entity] = 1;
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
    },
    rotation: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    scale: {
      type: DataType.FLOAT32,
      stride: 4,
      default: 1,
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
      this.update_entity_data(entity, {
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        scale: { x: 1, y: 1, z: 1 },
      });
      `,
    },
    get_entity_data: {
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
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      const data = this.get_entity_data(entity);
      return {
        position: { x: data.position.x, y: data.position.y, z: data.position.z },
        rotation: {
          x: data.rotation.x,
          y: data.rotation.y,
          z: data.rotation.z,
          w: data.rotation.w,
        },
        scale: { x: data.scale.x, y: data.scale.y, z: data.scale.z },
      };
      `,
    },
    update_entity_data: {
      skip_default: true,
      pre: `
      if (!this.data) {
        this.initialize();
      }

      const context = Renderer.get().graphics_context;

      if (data.position) {
        this.data.position[entity * 4 + 0] = data.position.x;
        this.data.position[entity * 4 + 1] = data.position.y;
        this.data.position[entity * 4 + 2] = data.position.z;
        this.data.position[entity * 4 + 3] = 1.0;
        this.data.position_buffer.write_raw(
          context,
          this.data.position.subarray(entity * 4, entity * 4 + 4),
          entity * 4 * Float32Array.BYTES_PER_ELEMENT
        );
      }
      if (data.rotation) {
        this.data.rotation[entity * 4 + 0] = data.rotation.x;
        this.data.rotation[entity * 4 + 1] = data.rotation.y;
        this.data.rotation[entity * 4 + 2] = data.rotation.z;
        this.data.rotation[entity * 4 + 3] = data.rotation.w;
        this.data.rotation_buffer.write_raw(
          context,
          this.data.rotation.subarray(entity * 4, entity * 4 + 4),
          entity * 4 * Float32Array.BYTES_PER_ELEMENT
        );
      }
      if (data.scale) {
        this.data.scale[entity * 4 + 0] = data.scale.x;
        this.data.scale[entity * 4 + 1] = data.scale.y;
        this.data.scale[entity * 4 + 2] = data.scale.z;
        this.data.scale[entity * 4 + 3] = 0.0;
        this.data.scale_buffer.write_raw(
          context,
          this.data.scale.subarray(entity * 4, entity * 4 + 4),
          entity * 4 * Float32Array.BYTES_PER_ELEMENT
        );
      }

      this.data.dirty[entity] = 1;
      this.data.dirty_flags_buffer.write_raw(
        context,
        this.data.dirty.subarray(entity, entity + 1),
        entity * Uint32Array.BYTES_PER_ELEMENT
      );
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
    `
    } ,
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
    },
  },
  members: {
    scene_graph: 'new Tree()',
    scene_graph_layer_counts: '[]',
    scene_graph_uniforms: '[]',
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
    add_entity: {
      skip_default: true,
      post: `
      super.add_entity(entity, null);

      if (data) {
        this.data.parent[entity] = data.parent || -1;
      }

      this.data.scene_graph.add(data?.parent ?? null, entity);
      if (Array.isArray(data.children)) {
        this.data.scene_graph.add_multiple(entity, data.children);
      }

      this.data.gpu_data_dirty = true;
      `,
    },
    remove_entity: {
      skip_default: true,
      post: `
      super.remove_entity(entity);
      this.data.parent[entity] = -1;
      this.data.scene_graph.remove(entity);
      this.data.gpu_data_dirty = true;
      `,
    },
    get_entity_data: {
      skip_default: true,
      pre: `
      const node = this.data.scene_graph.find_node(entity);
      return {
        parent: node?.parent ?? null,
        children: node?.children ?? [],
      };
      `,
    },
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      return this.get_entity_data(entity);
      `,
    },
    update_entity_data: {
      skip_default: true,
      post: `
      if (!this.data) {
        this.initialize();
      }

      const { children, ...rest } = data;

      super.update_entity_data(entity, rest);

      if (children) {
        this.data.scene_graph.add_multiple(entity, children, true /* replace_children */);
      }

      if (this.data.dirty) {
        this.data.dirty[entity] = 1;
      }
      this.data.gpu_data_dirty = true;
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
  overrides: {
    get_entity_data: {
      skip_default: true,
      pre: `
      return {
        allows_cursor_events: this.data.allows_cursor_events[entity],
        auto_size: this.data.auto_size[entity],
        was_cursor_inside: this.data.was_cursor_inside[entity],
        is_cursor_inside: this.data.is_cursor_inside[entity],
        was_clicked: this.data.was_clicked[entity],
        is_clicked: this.data.is_clicked[entity],
        is_pressed: this.data.is_pressed[entity],
        was_pressed: this.data.was_pressed[entity],
      };
      `,
    },
  },
};

export const definitions = [LightFragment, StaticMeshFragment, TransformFragment, SceneGraphFragment, VisibilityFragment, UserInterfaceFragment];
