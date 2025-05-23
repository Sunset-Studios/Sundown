import { EntityFlags } from "../minimal.js";
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
    this.entity_query = EntityManager.create_query([
      TextFragment,
      StaticMeshFragment,
      TransformFragment,
    ]);
    this._update_internal = this._update_internal.bind(this);
    this._text_entities_iter = this._text_entities_iter.bind(this);
  }

  update(delta_time) {
    profile_scope(text_processor_update_key, this._update_internal);
  }

  #text_entities_set = new Set();
  _text_entities_iter(chunk, slot, instance_count, archetype) {
    this.#text_entities_set.add(EntityManager.get_entity_for(chunk, slot));
  }

  _update_internal() {
    // 1) gather up every segment per entity
    this.#text_entities_set.clear();
    this.entity_query.for_each(this._text_entities_iter, true /* dirty_only */);

    // 2) process each entity exactly once
    const text_entities = Array.from(this.#text_entities_set);
    const code_points = [];
    for (let i = 0; i < text_entities.length; i++) {
      const entity = text_entities[i];
      const segments = entity.segments;

      // 3) pull all code points across all segments into one array
      code_points.length = 0;
      for (let k = 0; k < segments.length; k++) {
        const { chunk, slot, count } = segments[k];
        const seg_text_views = chunk.get_fragment_view(TextFragment);
        for (let j = 0; j < count; j++) {
          code_points.push(seg_text_views.text[slot + j]);
        }
      }

      const first = segments[0];
      const text_views = first.chunk.get_fragment_view(TextFragment);

      // 4) compute offsets for the full text
      const font_index = text_views.font[first.slot];
      const font_object = FontCache.get_font_object(font_index);
      const font_size = text_views.font_size[first.slot];

      if (!font_object || !font_object.texture_height) {
        // Ensures texture_height is non-zero and valid
        continue; // Skip processing this entity if font data is problematic
      }
      const font_scale = font_size / font_object.texture_height;

      const offsets = new Array(code_points.length).fill(0);
      for (let k = 1; k < code_points.length; k++) {
        const prev_cp = code_points[k - 1];
        const cp = code_points[k];
        // Ensure prev_cp and cp are valid indices before accessing font_object arrays
        if (
          prev_cp === undefined ||
          cp === undefined ||
          prev_cp >= font_object.code_point.length ||
          cp >= font_object.code_point.length ||
          prev_cp >= font_object.width.length // Assuming width, x_advance, x_offset have similar indexing
        ) {
          // Default or skip kerning/offset if indices are bad
          offsets[k] = offsets[k - 1] + (font_object.default_advance || 10) * font_scale; // Use a default advance
          continue;
        }

        const kern = font_object.kerning_matrix.get_adjacent_value(
          font_object.code_point[prev_cp],
          font_object.code_point[cp]
        );
        offsets[k] =
          offsets[k - 1] +
          (font_object.width[prev_cp] +
            font_object.x_advance[cp] +
            font_object.x_offset[cp] +
            (kern || 0)) * // Use kern || 0 to handle undefined kerning
            font_scale;
      }

      // 5) write back into each segment's transform chunk
      let global_index = 0;
      let anchor_x = 0;
      let anchor_y = 0;
      let anchor_z = 0;

      for (let k = 0; k < segments.length; k++) {
        const { chunk, slot, count } = segments[k];

        const transform_views = chunk.get_fragment_view(TransformFragment);
        for (let j = 0; j < count; j++) {
          const base_pos_index = (slot + j) * 4;
          const base_scale_index = (slot + j) * 4;

          // read current position
          const x = transform_views.position[base_pos_index + 0];
          const y = transform_views.position[base_pos_index + 1];
          const z = transform_views.position[base_pos_index + 2];

          // write new position
          if (global_index === 0) {
            anchor_x = x;
            anchor_y = y;
            anchor_z = z;
          } else {
            // Ensure code_points[global_index] is valid before complex y_offset access
            const current_metric_idx_for_y = code_points[global_index];
            let y_offset_value = 0;
            if (
              current_metric_idx_for_y !== undefined &&
              current_metric_idx_for_y < font_object.y_offset.length &&
              current_metric_idx_for_y < font_object.height.length
            ) {
              y_offset_value =
                (font_object.y_offset[current_metric_idx_for_y] * 2 +
                  font_object.height[current_metric_idx_for_y] -
                  font_object.line_height) *
                font_scale;
            }

            transform_views.position[base_pos_index + 0] = anchor_x + offsets[global_index];
            transform_views.position[base_pos_index + 1] = anchor_y - y_offset_value;
            transform_views.position[base_pos_index + 2] = anchor_z;
          }

          // write new scale – make sure *all four* floats get set each time
          const current_char_metric_index = code_points[global_index];

          if (
            current_char_metric_index === undefined ||
            current_char_metric_index >= font_object.width.length ||
            current_char_metric_index >= font_object.height.length
          ) {
            // Invalid metric index, make glyph invisible or use default small scale
            transform_views.scale[base_scale_index + 0] = 0;
            transform_views.scale[base_scale_index + 1] = 0;
            transform_views.scale[base_scale_index + 2] = 1.0;
            transform_views.scale[base_scale_index + 3] = 1.0; // clear any stale W
          } else {
            transform_views.scale[base_scale_index + 0] =
              font_object.width[current_char_metric_index] * font_scale;
            transform_views.scale[base_scale_index + 1] =
              font_object.height[current_char_metric_index] * font_scale;
            transform_views.scale[base_scale_index + 2] = 1.0;
            transform_views.scale[base_scale_index + 3] = 0.0; // clear any stale W
          }

          ++global_index;
        }

        chunk.mark_dirty();
      }
    }
  }
}
