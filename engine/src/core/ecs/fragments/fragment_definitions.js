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
      getter: `return StaticMeshFragment.data.material_slots.slice(this.absolute_entity * StaticMeshFragment.material_slot_stride, (this.absolute_entity + 1) * StaticMeshFragment.material_slot_stride);`,
      setter: `
      if (
        Array.isArray(value) &&
        value.length <= StaticMeshFragment.material_slot_stride
      ) {
        for (let i = 0; i < value.length; i++) {
          StaticMeshFragment.data.material_slots[this.absolute_entity * StaticMeshFragment.material_slot_stride + i] = BigInt(value[i]);
        }
      }
      if (StaticMeshFragment.data.dirty) {
        StaticMeshFragment.data.dirty[this.absolute_entity] = 1;
      }
      StaticMeshFragment.data.gpu_data_dirty = true;
      `,
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
      const entity_offset = EntityID.get_absolute_index(entity);
      const entity_instances = EntityID.get_instance_count(entity);
      for (let i = 0; i < entity_instances; i++) {
        this.data.mesh[entity_offset + i] = 0n;
        for (let j = 0; j < this.material_slot_stride; j++) {
          this.data.material_slots[(entity_offset + i) * this.material_slot_stride + j] = 0n;
        }
        this.data.dirty[entity_offset + i] = 1;
      }
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
      getter: `return [
        TransformFragment.data.position[this.absolute_entity * 4],
        TransformFragment.data.position[this.absolute_entity * 4 + 1],
        TransformFragment.data.position[this.absolute_entity * 4 + 2],
      ];
      `,
      setter: `
      TransformFragment.data.position[this.absolute_entity * 4] = value[0];
      TransformFragment.data.position[this.absolute_entity * 4 + 1] = value[1];
      TransformFragment.data.position[this.absolute_entity * 4 + 2] = value[2];
      TransformFragment.data.position[this.absolute_entity * 4 + 3] = 1.0;

      if (TransformFragment.data.dirty) {
        TransformFragment.data.dirty[this.absolute_entity] = 1;
      }
      TransformFragment.data.gpu_data_dirty = true;
      `,
    },
    rotation: {
      type: DataType.FLOAT32,
      stride: 4,
      getter: `return [
        TransformFragment.data.rotation[this.absolute_entity * 4],
        TransformFragment.data.rotation[this.absolute_entity * 4 + 1],
        TransformFragment.data.rotation[this.absolute_entity * 4 + 2],
        TransformFragment.data.rotation[this.absolute_entity * 4 + 3],
      ];
      `,
      setter: `
      TransformFragment.data.rotation[this.absolute_entity * 4] = value[0];
      TransformFragment.data.rotation[this.absolute_entity * 4 + 1] = value[1];
      TransformFragment.data.rotation[this.absolute_entity * 4 + 2] = value[2];
      TransformFragment.data.rotation[this.absolute_entity * 4 + 3] = value[3];

      if (TransformFragment.data.dirty) {
        TransformFragment.data.dirty[this.absolute_entity] = 1;
      }
      TransformFragment.data.gpu_data_dirty = true;
      `,
    },
    scale: {
      type: DataType.FLOAT32,
      stride: 4,
      default: 1,
      getter: `return [
        TransformFragment.data.scale[this.absolute_entity * 4],
        TransformFragment.data.scale[this.absolute_entity * 4 + 1],
        TransformFragment.data.scale[this.absolute_entity * 4 + 2],
      ];
      `,
      setter: `
      TransformFragment.data.scale[this.absolute_entity * 4] = value[0];
      TransformFragment.data.scale[this.absolute_entity * 4 + 1] = value[1];
      TransformFragment.data.scale[this.absolute_entity * 4 + 2] = value[2];
      TransformFragment.data.scale[this.absolute_entity * 4 + 3] = 0.0;

      if (TransformFragment.data.dirty) {
        TransformFragment.data.dirty[this.absolute_entity] = 1;
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
      stride: 64,
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
      const entity_offset = EntityID.get_absolute_index(entity);
      const entity_instances = EntityID.get_instance_count(entity);
      for (let i = 0; i < entity_instances; i++) {
        this.data.position[(entity_offset + i) * 4] = 0;
        this.data.position[(entity_offset + i) * 4 + 1] = 0;
        this.data.position[(entity_offset + i) * 4 + 2] = 0;
        this.data.position[(entity_offset + i) * 4 + 3] = 1;
        this.data.rotation[(entity_offset + i) * 4] = 0;
        this.data.rotation[(entity_offset + i) * 4 + 1] = 0;
        this.data.rotation[(entity_offset + i) * 4 + 2] = 0;
        this.data.rotation[(entity_offset + i) * 4 + 3] = 0;
        this.data.scale[(entity_offset + i) * 4] = 1;
        this.data.scale[(entity_offset + i) * 4 + 1] = 1;
        this.data.scale[(entity_offset + i) * 4 + 2] = 1;
        this.data.scale[(entity_offset + i) * 4 + 3] = 0;
        this.data.dirty[entity_offset + i] = 1;
      }
      `,
    },
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      const entity_offset = EntityID.get_absolute_index(entity);
      return {
        position: [
          this.data.position[entity_offset * 4],
          this.data.position[entity_offset * 4 + 1],
          this.data.position[entity_offset * 4 + 2],
        ],
        rotation: [
          this.data.rotation[entity_offset * 4],
          this.data.rotation[entity_offset * 4 + 1],
          this.data.rotation[entity_offset * 4 + 2],
          this.data.rotation[entity_offset * 4 + 3],
        ],
        scale: [
          this.data.scale[entity_offset * 4],
          this.data.scale[entity_offset * 4 + 1],
          this.data.scale[entity_offset * 4 + 2],
        ],
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
        this.data.dirty_flags_buffer = Buffer.create({
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
        this.size * 64 * Float32Array.BYTES_PER_ELEMENT
      ) {
        this.data.transforms_buffer = Buffer.create({
          name: "transform_fragment_transforms_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: new Float32Array(this.size * 64),
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      }

      if (
        !this.data.bounds_data_buffer ||
        this.data.bounds_data_buffer.config.size <
        this.size * 8 * Float32Array.BYTES_PER_ELEMENT
      ) {
        this.data.bounds_data_buffer = Buffer.create({
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

    await this.sync_buffers();
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
      SceneGraphFragment.data.parent[this.absolute_entity] = value ?? -1;
      SceneGraphFragment.data.scene_graph.remove(this.current_entity);
      SceneGraphFragment.data.scene_graph.add(value ?? null, this.current_entity);
      if (SceneGraphFragment.data.dirty) {
        SceneGraphFragment.data.dirty[this.absolute_entity] = 1;
      }
      SceneGraphFragment.data.gpu_data_dirty = true;
      `,
    },
    children: {
      type: DataType.INT32,
      stride: 1,
      getter: `
      const node = SceneGraphFragment.data.scene_graph.find_node(this.current_entity);
      return [...(SceneGraphFragment.data.scene_graph.get_children(node))].map(child => child.data)
      `,
      setter: `
      if (Array.isArray(value)) {
        SceneGraphFragment.data.scene_graph.add_multiple(this.current_entity, value, true /* replace_children */, true /* unique */);
      }
      if (SceneGraphFragment.data.dirty) {
        SceneGraphFragment.data.dirty[this.absolute_entity] = 1;
      }
      SceneGraphFragment.data.gpu_data_dirty = true;
      `,
      no_fragment_array: true,
    },
  },
  members: {
    scene_graph: "new Tree()",
    scene_graph_flattened: "[]",
    scene_graph_layer_counts: "[]",
    scene_graph_uniforms: "[]",
  },
  buffers: {
    scene_graph: {
      type: DataType.INT32,
      usage: BufferType.STORAGE,
      stride: 2,
      gpu_data: `
      const { result, layer_counts } = this.data.scene_graph.flatten(Int32Array, (result, node, result_size) => {
        let instance_count = EntityID.get_instance_count(node.data);
        for (let i = 0; i < instance_count; i++) {
          result[result_size + i] = EntityID.get_absolute_index(node.data) + i;
        }
        return instance_count;
      }, (node) => {
        return EntityID.get_instance_count(node.data);
      });

      this.data.scene_graph_flattened = result;
      this.data.scene_graph_layer_counts = layer_counts;

      const num_elements = result?.length ?? 0;
      const gpu_data = new Int32Array(Math.max(num_elements * 2, 2));
      for (let i = 0; i < num_elements; ++i) {
        gpu_data[i * 2] = result[i];
        gpu_data[i * 2 + 1] = this.data.parent[result[i]] >= 0 ? EntityID.get_absolute_index(this.data.parent[result[i]]) : -1;
      }

      this.data.scene_graph_uniforms = new Array(layer_counts.length);
      let layer_offset = 0;
      for (let i = 0; i < layer_counts.length; ++i) {
        this.data.scene_graph_uniforms[i] = Buffer.create({
          name: "scene_graph_uniforms_" + i,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          raw_data: new Uint32Array([layer_counts[i], layer_offset, i]),
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
      const entity_offset = EntityID.get_absolute_index(entity);
      const entity_instances = EntityID.get_instance_count(entity);
      for (let i = 0; i < entity_instances; i++) {
        this.data.parent[entity_offset + i] = -1;
      }
      this.data.scene_graph.remove(entity);
      this.data.gpu_data_dirty = true;
      `,
    },
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      const entity_offset = EntityID.get_absolute_index(entity);
      const node = this.data.scene_graph.find_node(entity_offset);
      return {
        parent: this.data.scene_graph.get_parent(node),
        children: this.data.scene_graph.get_children(node).map(child => child.data),
      };
      `,
    },
  },
};

const VisibilityFragment = {
  name: "Visibility",
  fields: {
    visible: {
      type: DataType.UINT32,
      stride: 1,
    },
    dirty: {
      type: DataType.UINT8,
      stride: 1,
    },
  },
  buffers: {
    visible: {
      type: DataType.UINT32,
      usage: BufferType.STORAGE,
      stride: 1,
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
    consume_events: {
      type: DataType.UINT8,
      stride: 1,
    },
    color: {
      type: DataType.FLOAT32,
      vector: { r: true, g: true, b: true, a: true },
      stride: 1,
    },
    emissive: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    rounding: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    dirty: {
      type: DataType.UINT8,
      stride: 1,
    },
  },
  buffers: {
    element_data: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE,
      stride: 8,
      gpu_data: `
      const gpu_data = new Float32Array(Math.max(this.size * 8, 8));
      let offset = 0;
      for (let i = 0; i < this.size; i++) {
        gpu_data[offset + 0] = this.data.color.r[i];
        gpu_data[offset + 1] = this.data.color.g[i];
        gpu_data[offset + 2] = this.data.color.b[i];
        gpu_data[offset + 3] = this.data.color.a[i];
        gpu_data[offset + 4] = this.data.emissive[i];
        gpu_data[offset + 5] = this.data.rounding[i];
        gpu_data[offset + 6] = 0; // padding
        gpu_data[offset + 7] = 0; // padding
        offset += 8;
      }
      `,
    },
  },
};

const TextFragment = {
  name: "Text",
  imports: {
    FontCache: "../../../ui/text/font_cache.js",
  },
  fields: {
    text: {
      type: DataType.UINT32,
      stride: 1,
      is_container: true,
      no_instance_count_resize: true,
      getter: `
      const font = FontCache.get_font_object(TextFragment.data.font[this.current_entity]);
      return TextFragment.data.text.get_data_for_entity(this.current_entity);
      `,
      setter: `
      if (value) {
        const font = FontCache.get_font_object(TextFragment.data.font[this.current_entity]);
        const code_point_indexes = Array.from(value).map((char) => font.code_point_index_map.get(char.codePointAt(0)));
        TextFragment.data.text.update(this.current_entity, code_point_indexes);
        EntityManager.set_entity_instance_count(this.current_entity, code_point_indexes.length);
      }
      if (TextFragment.data.dirty) {
        TextFragment.data.dirty[this.current_entity] = 1;
      }
      TextFragment.data.gpu_data_dirty = true;
      `,
    },
    offsets: {
      type: DataType.FLOAT32,
      stride: 1,
      is_container: true,
      no_instance_count_resize: true,
    },
    font: {
      type: DataType.INT32,
      stride: 1,
      no_instance_count_resize: true,
    },
    font_size: {
      type: DataType.UINT32,
      stride: 1,
      no_instance_count_resize: true,
    },
    color: {
      type: DataType.FLOAT32,
      vector: { r: true, g: true, b: true, a: true },
      stride: 1,
      no_instance_count_resize: true,
    },
    emissive: {
      type: DataType.FLOAT32,
      stride: 1,
      no_instance_count_resize: true,
    },
    dirty: {
      type: DataType.UINT8,
      stride: 1,
      no_instance_count_resize: true,
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
    string_data: {
      type: DataType.FLOAT32,
      usage: BufferType.STORAGE,
      stride: 12,
      gpu_data: `
      const gpu_data = new Float32Array(Math.max(this.size * 12, 12));
      for (let i = 0; i < this.size; i++) {
        const metadata = this.data.text.get_metadata(i);
        const font = FontCache.get_font_object(this.data.font[i]);
        const gpu_data_offset = i * 12;
        gpu_data[gpu_data_offset + 0] = metadata?.start ?? 0;
        gpu_data[gpu_data_offset + 1] = metadata?.count ?? 0;
        gpu_data[gpu_data_offset + 2] = font?.texture_width ?? 0;
        gpu_data[gpu_data_offset + 3] = font?.texture_height ?? 0;
        gpu_data[gpu_data_offset + 4] = this.data.color.r[i];
        gpu_data[gpu_data_offset + 5] = this.data.color.g[i];
        gpu_data[gpu_data_offset + 6] = this.data.color.b[i];
        gpu_data[gpu_data_offset + 7] = this.data.color.a[i];
        gpu_data[gpu_data_offset + 8] = this.data.emissive[i]; 
        gpu_data[gpu_data_offset + 9] = 0; // padding
        gpu_data[gpu_data_offset + 10] = 0; // padding
        gpu_data[gpu_data_offset + 11] = 0; // padding
      }
      `,
    },
  },
  overrides: {
    duplicate_entity_data: {
      skip_default: true,
      pre: `
      const data = {};
      data.text = String.fromCodePoint(...this.data.text.get_data_for_entity(entity));
      data.font = this.data.font[entity];
      data.font_size = this.data.font_size[entity];
      data.emissive = this.data.emissive[entity];
      data.color = {
        r: this.data.color.r[entity],
        g: this.data.color.g[entity],
        b: this.data.color.b[entity],
        a: this.data.color.a[entity],
      };
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
