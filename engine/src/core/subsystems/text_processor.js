import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { TextFragment } from "../ecs/fragments/text_fragment.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { profile_scope } from "../../utility/performance.js";
import { FontCache } from "../../ui/text/font_cache.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";

const text_processor_update_key = "text_processor_update";

export class TextProcessor extends SimulationLayer {
  entity_query = null;

  init() {
    this.entity_query = EntityManager.create_query([TextFragment, StaticMeshFragment, TransformFragment]);
    this._update_internal = this._update_internal.bind(this);
  }

  update(delta_time) {
    profile_scope(text_processor_update_key, this._update_internal);
  }

  #entity_to_segments = new Map();
  _update_internal() {
    // 1) gather up every segment per entity
    this.#entity_to_segments.clear();
    this.entity_query.for_each((chunk, slot, instance_count) => {
      // lookup the original, stable entity ID from the allocator
      const entity_id = EntityManager.get_entity_for(chunk, slot);
      const entity_flags = EntityManager.get_entity_flags(entity_id);
      if ((entity_flags & EntityFlags.DIRTY) === 0) {
        return;
      }

      let segments = this.#entity_to_segments.get(entity_id);
      if (!segments) {
        segments = [];
        this.#entity_to_segments.set(entity_id, segments);
      }
      segments.push({ chunk, slot, instance_count });
    });

    // 2) process each entity exactly once
    const entity_to_segment_entries = this.#entity_to_segments.entries();
    for (let i = 0; i < entity_to_segment_entries.length; i++) {
      const [entity_id, segments] = entity_to_segment_entries[i];

      // check the dirty flag on the first segment only
      const first = segments[0];
      const text_views = first.chunk.get_fragment_view(TextFragment);

      // 3) pull all code points across all segments into one array
      const code_points = [];
      for (let i = 0; i < segments.length; i++) {
        const { chunk, slot, instance_count } = segments[i];
        const seg_text_views = chunk.get_fragment_view(TextFragment);
        for (let j = 0; j < instance_count; j++) {
          code_points.push(seg_text_views.text[slot + j]);
        }
      }

      // 4) compute offsets for the full text
      const font_index = text_views.font[first.slot];
      const font_object = FontCache.get_font_object(font_index);
      const font_scale = text_views.font_size[first.slot] / font_object.texture_height;

      const offsets = new Array(code_points.length).fill(0);
      for (let i = 1; i < code_points.length; i++) {
        const prev_cp = code_points[i - 1];
        const cp = code_points[i];
        const kern = font_object.kerning_matrix.get_adjacent_value(
          font_object.code_point[prev_cp],
          font_object.code_point[cp]
        );
        offsets[i] =
          offsets[i - 1] +
          (font_object.width[prev_cp] +
            font_object.x_advance[cp] +
            font_object.x_offset[cp] +
            kern) *
            font_scale;
      }
      const total_width = offsets[offsets.length - 1];

      // 5) write back into each segment's transform chunk
      let global_index = 0;
      for (let i = 0; i < segments.length; i++) {
        const { chunk, slot, instance_count } = segments[i];

        const transform_views = chunk.get_fragment_view(TransformFragment);
        for (let j = 0; j < instance_count; j++) {
          const idx = global_index + j;
          const base_pos_index = (slot + j) * 3;
          const base_scale_index = (slot + j) * 3;

          // read current position
          const x = transform_views.position[base_pos_index + 0];
          const y = transform_views.position[base_pos_index + 1];
          const z = transform_views.position[base_pos_index + 2];

          // write new position
          transform_views.position[base_pos_index + 0] = x + (offsets[idx] - total_width * 0.5);
          transform_views.position[base_pos_index + 1] =
            y -
            (font_object.y_offset[code_points[idx]] * 2 +
              font_object.height[code_points[idx]] -
              font_object.line_height) *
              font_scale;
          // z stays the same

          // write new scale
          transform_views.scale[base_scale_index + 0] =
            (font_object.width[code_points[idx]] / font_object.texture_width) *
            text_views.font_size[first.slot];
          transform_views.scale[base_scale_index + 1] =
            (font_object.height[code_points[idx]] / font_object.texture_height) *
            text_views.font_size[first.slot];
          // scale[2] = 1.0
          transform_views.scale[base_scale_index + 2] = 1.0;
        }

        chunk.mark_dirty();

        global_index += instance_count;
      }
    }
  }
}
