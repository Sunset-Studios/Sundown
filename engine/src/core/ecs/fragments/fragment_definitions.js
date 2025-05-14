import { DataType, BufferType } from "../meta/fragment_generator_types.js";

const LightFragment = {
  name: "Light",
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
    active: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    padding1: {
      type: DataType.FLOAT32,
      stride: 1,
    },
    padding2: {
      type: DataType.FLOAT32,
      stride: 1,
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
        "active",
        "padding1",
        "padding2",
      ],
      usage: BufferType.STORAGE,
    },
  },
};

const StaticMeshFragment = {
  name: "StaticMesh",
  constants: {
    material_slot_stride: 16,
  },
  fields: {
    mesh: {
      type: DataType.BIGINT64,
      stride: 1,
      gpu: false,
    },
    material_slots: {
      type: DataType.BIGINT64,
      stride: 16,
      gpu: false,
    },
  },
};

const TransformFragment = {
  name: "Transform",
  fields: {
    position: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
    },
    rotation: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
    },
    scale: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
    },
    aabb_node_index: {
      type: DataType.UINT32,
      stride: 1,
      gpu: true,
      usage: BufferType.STORAGE,
    },
    transforms: {
      type: DataType.FLOAT32,
      stride: 32,
      gpu: true,
      usage: BufferType.STORAGE_SRC,
      cpu_readback: true,
    },
  },
  custom_methods: {
    get_world_position: {
      params: `entity, instance = 0`,
      body: `
      const transform_fragment = EntityManager.get_fragment(entity, TransformFragment, instance);
      const transform = transform_fragment.transform;
      
      return [
        transform[12],
        transform[13],
        transform[14]
      ];
      `,
    },
    get_world_rotation: {
      params: `entity, instance = 0`,
      body: `
      const transform_fragment = EntityManager.get_fragment(entity, TransformFragment, instance);
      const transform = transform_fragment.transform;

      const m00 = transform[0];
      const m01 = transform[1];
      const m02 = transform[2];
      const m10 = transform[4];
      const m11 = transform[5];
      const m12 = transform[6];
      const m20 = transform[8];
      const m21 = transform[9];
      const m22 = transform[10];
      
      const trace = m00 + m11 + m22;
      let qx, qy, qz, qw;
      
      if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (m21 - m12) * s;
        qy = (m02 - m20) * s;
        qz = (m10 - m01) * s;
      } else if (m00 > m11 && m00 > m22) {
        const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
        qw = (m21 - m12) / s;
        qx = 0.25 * s;
        qy = (m01 + m10) / s;
        qz = (m02 + m20) / s;
      } else if (m11 > m22) {
        const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
        qw = (m02 - m20) / s;
        qx = (m01 + m10) / s;
        qy = 0.25 * s;
        qz = (m12 + m21) / s;
      } else {
        const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
        qw = (m10 - m01) / s;
        qx = (m02 + m20) / s;
        qy = (m12 + m21) / s;
        qz = 0.25 * s;
      }
      
      return [qx, qy, qz, qw];
      `,
    },
    get_world_scale: {
      params: `entity, instance = 0`,
      body: `
      const transform_fragment = EntityManager.get_fragment(entity, TransformFragment, instance);
      const transform = transform_fragment.transform;

      const scale_x = Math.sqrt(
        transform[0] * transform[0] +
        transform[1] * transform[1] +
        transform[2] * transform[2]
      );
      
      const scale_y = Math.sqrt(
        transform[4] * transform[4] +
        transform[5] * transform[5] +
        transform[6] * transform[6]
      );
      
      const scale_z = Math.sqrt(
        transform[8] * transform[8] +
        transform[9] * transform[9] +
        transform[10] * transform[10]
      );
      
      return [scale_x, scale_y, scale_z];
      `,
    },
    add_world_offset: {
      params: `entity, offset, instance = 0 `,
      body: `
      const transform_fragment = EntityManager.get_fragment(entity, TransformFragment, instance);
      const transform = transform_fragment.transform;

      transform[12] += offset[0];
      transform[13] += offset[1];
      transform[14] += offset[2];

      transform_fragment.transform = transform;
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
  },
};

const UserInterfaceFragment = {
  name: "UserInterface",
  fields: {
    allows_cursor_events: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    auto_size: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    was_cursor_inside: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    is_cursor_inside: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    was_clicked: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    is_clicked: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    is_pressed: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    was_pressed: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    consume_events: {
      type: DataType.UINT8,
      stride: 1,
      gpu: false,
    },
    element_color: {
      type: DataType.FLOAT32,
      stride: 4,
      gpu: true,
    },
    element_emissive: {
      type: DataType.FLOAT32,
      stride: 1,
      gpu: true,
    },
    element_rounding: {
      type: DataType.FLOAT32,
      stride: 1,
      gpu: true,
    },
  },
  buffers: {
    element_data: {
      fields: ["element_color", "element_emissive", "element_rounding"],
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
  },
  fields: {
    text: {
      type: DataType.UINT32,
      stride: 1,
      gpu: true,
      setter: `
  if (!value) return;

  const font_typed_array = this.chunk.fragment_views[this.fragment_id]?.font;
  const font = FontCache.get_font_object(font_typed_array[this.slot]);

  // 1) turn string → array of code-point indices
  const code_point_indexes = Array.from(value).map((char) => {
    return font.code_point_index_map.get(char.codePointAt(0));
  });

  // 2) re-allocate the entity to match the new length
  EntityManager.set_entity_instance_count(this.entity, code_point_indexes.length);

  // 3) pull back the brand-new layout (one or more segments)
  let write_offset = 0;
  for (let i = 0; i < this.entity.segments.length; i++) {
    const { chunk, slot, count } = this.entity.segments[i];

    // grab the *fresh* views for this segment
    const frag_views = chunk.fragment_views[this.fragment_id];
    const text_array = frag_views.text;          // Uint32Array view for .text
    const slice = code_point_indexes.slice(write_offset, write_offset + count);

    // write into the correct slot
    text_array.set(slice, slot);

    for (let j = 0; j < count; j++) {
      chunk.flags_meta[slot + j] |= EntityFlags.DIRTY;
    }

    // mark it so it ends up in the next GPU flush
    chunk.mark_dirty();

    write_offset += count;
  }
      `,
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
        const row_count = chunk.capacity;
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
