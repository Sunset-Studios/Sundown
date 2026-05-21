import { DataType, BufferType } from "../meta/fragment_generator_types.js";

const LightFragment = {
  name: "Light",
  members: {
    total_shadow_casting_lights: "0",
  },
  fields: {
    position: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    direction: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    color: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    type: {
      type: DataType.FLOAT32,
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
    shadow_casting: {
      type: DataType.FLOAT32,
      default: 1,
      stride: 1,
    },
    active: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    view_index: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    shadow_index: {
      type: DataType.FLOAT32,
      default: -1,
      stride: 1,
    },
    shadow_clipmaps: {
      type: DataType.FLOAT32,
      stride: 1,
      default: 12,
    },
    is_primary_sun: {
      type: DataType.FLOAT32,
      stride: 1,
      default: 0,
    },
    shadows_dirty: {
      type: DataType.FLOAT32,
      stride: 1,
      default: 0,
    },
  },
  gpu_buffers: {
    light_fragment: {
      fields: [
        "position",
        "direction",
        "color",
        "type",
        "intensity",
        "radius",
        "attenuation",
        "outer_angle",
        "shadow_casting",
        "active",
        "view_index",
        "shadow_index",
        "shadow_clipmaps",
        "is_primary_sun",
        "shadows_dirty",
      ],
      usage: BufferType.STORAGE,
    },
  },
};

const StaticMeshFragment = {
  name: "StaticMesh",
  imports: {
    RenderTaskQueue: "../../../renderer/render_task_queue.js",
  },
  constants: {
    material_slot_stride: 16,
  },
  fields: {
    mesh: {
      type: DataType.BIGINT64,
      stride: 1,
      gpu: false,
      setter: `
      typed_array[element_offset] = BigInt(value);
        RenderTaskQueue.mark_meshes_dirty(true, this.entity);
      `,
    },
    material_slots: {
      type: DataType.BIGINT64,
      stride: 16,
      gpu: false,
      setter: `
      typed_array[element_offset] = BigInt(value);
        RenderTaskQueue.mark_meshes_dirty(true, this.entity);
      `,
    },
    mesh_asset_id: {
      type: DataType.UINT32,
      stride: 1,
      default: 4294967295,
      gpu: true,
      usage: BufferType.STORAGE,
      setter: `
      typed_array[element_offset] = value;
        RenderTaskQueue.mark_meshes_dirty(true, this.entity);
      `,
    },
    material_table_offset: {
      type: DataType.UINT32,
      stride: 1,
      gpu: true,
      usage: BufferType.STORAGE,
      setter: `
      typed_array[element_offset] = value;
        RenderTaskQueue.mark_meshes_dirty(true, this.entity);
      `,
    },
  },
};

const TransformFragment = {
  name: "Transform",
  imports: {
    EntityFlags: "../../minimal.js",
    EntityManager: "../entity.js",
    DEFAULT_CHUNK_CAPACITY: "../solar/types.js",
    BVH: "../../../acceleration/bvh.js",
  },
  members: {
    dirty_entities: "new Set()",
  },
  fields: {
    position: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
      setter: `
      if (
        typed_array
          .subarray(element_offset, element_offset + 4)
          .every((v, i) => v === value[i])
      ) {
        return;
      }

      this.chunk.flags_meta[this.slot + this.instance] |=
        EntityFlags.MOVED | EntityFlags.TRANSFORM_DIRTY;

      typed_array.set(value, element_offset);
      TransformFragment.mark_entity_dirty(this.entity);
      `,
    },
    rotation: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
      setter: `
      if (
        typed_array
          .subarray(element_offset, element_offset + 4)
          .every((v, i) => v === value[i])
      ) {
        return;
      }

      this.chunk.flags_meta[this.slot + this.instance] |=
        EntityFlags.MOVED | EntityFlags.TRANSFORM_DIRTY;

      typed_array.set(value, element_offset);
      TransformFragment.mark_entity_dirty(this.entity);
      `,
    },
    scale: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      default: 1,
      usage: BufferType.STORAGE_SRC,
      setter: `
      if (
        typed_array
          .subarray(element_offset, element_offset + 4)
          .every((v, i) => v === value[i])
      ) {
        return;
      }

      this.chunk.flags_meta[this.slot + this.instance] |=
        EntityFlags.MOVED | EntityFlags.TRANSFORM_DIRTY;

      typed_array.set(value, element_offset);
      TransformFragment.mark_entity_dirty(this.entity);
      `,
    },
    bounds: {
      type: DataType.FLOAT32,
      stride: 8,
      gpu: true,
      buffer_multiplier: 2,
      usage: BufferType.STORAGE,
    },
    transforms: {
      type: DataType.FLOAT32,
      stride: 48,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
    },
    world_position: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
    },
    world_rotation: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
    },
    world_scale: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      default: 1,
      usage: BufferType.STORAGE_SRC,
    },
  },
  custom_methods: {
    mark_entity_dirty: {
      params: `entity`,
      body: `
      if (entity) {
        this.dirty_entities.add(entity);
      }
      `,
    },
    consume_dirty_entities: {
      params: ``,
      body: `
      const dirty_entities = Array.from(this.dirty_entities);
      this.dirty_entities.clear();
      return dirty_entities;
      `,
    },
    get_world_position: {
      params: `entity, instance = 0`,
      body: `
      const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    return transform_fragment ? transform_fragment.world_position.slice(0, 3) : null;
      `,
    },
    get_world_rotation: {
      params: `entity, instance = 0`,
      body: `
      const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    return transform_fragment ? transform_fragment.world_rotation.slice() : null;
      `,
    },
    get_world_scale: {
      params: `entity, instance = 0`,
      body: `
      const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    return transform_fragment ? transform_fragment.world_scale.slice(0, 3) : null;
      `,
    },
    add_world_offset: {
      params: `entity, offset, instance = 0 `,
      body: `
      const local_transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );

    let position = local_transform_fragment.position;
    position[0] += offset[0];
    position[1] += offset[1];
    position[2] += offset[2];
    local_transform_fragment.position = position;
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
      gpu: true,
      usage: BufferType.STORAGE,
    },
    occluder: {
      type: DataType.UINT32,
      stride: 1,
      default: 1,
      gpu: true,
      usage: BufferType.STORAGE,
    },
    got_shadow_feedback: {
      type: DataType.UINT32,
      stride: 1,
      default: 0,
      gpu: true,
      usage: BufferType.STORAGE,
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
    ui_emissive: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    ui_rounding: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    ui_origin: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    ui_x_axis: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    ui_y_axis: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    ui_uv_rect: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    ui_color: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    ui_border_color: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    ui_params: {
      type: DataType.FLOAT32,
      stride: 4,
    },
    ui_text_glyph: {
      type: DataType.UINT32,
      stride: 1,
      gpu: true,
    },
  },
  gpu_buffers: {
    ui_data: {
      fields: [
        "ui_origin",
        "ui_x_axis",
        "ui_y_axis",
        "ui_uv_rect",
        "ui_color",
        "ui_border_color",
        "ui_params",
      ],
      usage: BufferType.STORAGE,
    },
  },
};

const TextFragment = {
  name: "Text",
  imports: {
    FontCache: "../../../ui/text/font_cache.js",
    EntityManager: "../entity.js",
    EntityFlags: "../../minimal.js",
    DEFAULT_CHUNK_CAPACITY: "../solar/types.js",
  },
  fields: {
    text: {
      type: DataType.UINT32,
      stride: 1,
      gpu: true,
      setter: `
       if (value === null || value === undefined) {
          value = ""; // Treat null/undefined as a request to clear, equivalent to empty string.
        }

        const text_value_as_string = String(value);
        const target_instance_count = text_value_as_string.length;

        if (
          !this.entity ||
          !this.entity.segments ||
          this.entity.segments.length === 0
        ) {
          if (this.entity) {
            // Still try to set instance count if handle exists
            EntityManager.set_entity_instance_count(
              this.entity,
              target_instance_count,
            );
          }
          return;
        }

        // Get font ID from the entity's *current* primary segment's chunk and slot
        const current_primary_segment = this.entity.segments[0];
        const chunk_for_font_lookup = current_primary_segment.chunk;
        const slot_for_font_lookup = current_primary_segment.slot; // This is the base slot for the entity in this chunk

        const font_data_array =
          chunk_for_font_lookup.fragment_views[this.fragment_id]?.font;

        if (!font_data_array) {
          EntityManager.set_entity_instance_count(
            this.entity,
            target_instance_count,
          );
          return;
        }

        // The 'font' field is a single Int32, so use the entity's base slot in its current primary chunk
        const font_id = font_data_array[slot_for_font_lookup];
        const font = FontCache.get_font_object(font_id);

        if (!font && target_instance_count > 0) {
          EntityManager.set_entity_instance_count(
            this.entity,
            target_instance_count,
          );
          return;
        }

        let code_point_indexes = [];
        if (target_instance_count > 0 && font) {
          code_point_indexes = Array.from(text_value_as_string).map((char) => {
            const code_point = char.codePointAt(0);
            const index = font.code_point_index_map.get(code_point);
            if (index === undefined) {
              const fallback_index = font.code_point_index_map.get(
                " ".codePointAt(0),
              );
              return fallback_index !== undefined ? fallback_index : 0;
            }
            return index;
          });
        }

        // Determine how many instances we had before and how many we want now:
        EntityManager.set_entity_instance_count(
          this.entity,
          target_instance_count,
        );

        // Write data to the (now potentially new) segments stored in entity_handle.segments.
        let write_offset = 0;
        const segments_to_write = this.entity.segments || [];
        for (let i = 0; i < segments_to_write.length; i++) {
          const { chunk, slot, count } = segments_to_write[i];

          const frag_views_in_current_chunk =
            chunk.fragment_views[this.fragment_id];
          const text_array_in_current_chunk = frag_views_in_current_chunk?.text;

          const slice_to_write = code_point_indexes.slice(
            write_offset,
            write_offset + count,
          );

          text_array_in_current_chunk.set(slice_to_write, slot);

          for (let j = 0; j < count; j++) {
            // Ensure we don't write past flags_meta if count is unexpectedly large
            chunk.flags_meta[slot + j] |= EntityFlags.DIRTY;
          }
          chunk.mark_dirty('text');

          write_offset += count;
        }`,
    },
    font: {
      type: DataType.INT32,
      stride: 1,
      gpu: false,
    },
    font_size: {
      type: DataType.UINT32,
      stride: 1,
      gpu: false,
    },
    text_color: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
    },
    text_emissive: {
      type: DataType.FLOAT32,
      stride: 1,
      gpu: true,
    },
  },
  gpu_buffers: {
    string_data: {
      usage: BufferType.STORAGE,
      stride: 32,
      gpu_data(chunk, fragment) {
        const row_count = DEFAULT_CHUNK_CAPACITY;
        const fragment_views = chunk.fragment_views[fragment.id];
        const packed_data = new Float32Array(Math.max(row_count * 8, 8));
        for (let row = 0; row < row_count; row++) {
          const gpu_data_offset = row * 8;
          const font = FontCache.get_font_object(fragment_views.font[row]);
          packed_data[gpu_data_offset + 0] = fragment_views.text_color[row * 4];
          packed_data[gpu_data_offset + 1] = fragment_views.text_color[row * 4 + 1];
          packed_data[gpu_data_offset + 2] = fragment_views.text_color[row * 4 + 2];
          packed_data[gpu_data_offset + 3] = fragment_views.text_color[row * 4 + 3];
          packed_data[gpu_data_offset + 4] = font?.texture_width ?? 0;
          packed_data[gpu_data_offset + 5] = font?.texture_height ?? 0;
          packed_data[gpu_data_offset + 6] = fragment_views.text_emissive[row];
          packed_data[gpu_data_offset + 7] = 0; // padding
        }
        return { packed_data, row_count };
      },
    },
  },
};

export const definitions = [
  LightFragment,
  StaticMeshFragment,
  TransformFragment,
  VisibilityFragment,
  UserInterfaceFragment,
  TextFragment,
];
