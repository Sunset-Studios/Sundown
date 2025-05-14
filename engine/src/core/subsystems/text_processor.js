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
  }

  update(delta_time) {
    profile_scope(text_processor_update_key, this._update_internal);
  }

  #text_entities_set = new Set();
  _update_internal() {
    // 1) gather up every segment per entity
    this.#text_entities_set.clear();

    this.entity_query.for_each((chunk, slot, instance_count, archetype) => {
      // lookup the original, stable entity ID from the allocator
      const entity = EntityManager.get_entity_for(chunk, slot);
      const entity_flags = EntityManager.get_entity_flags(entity);
      if ((entity_flags & EntityFlags.DIRTY) === 0) {
        return;
      }
      this.#text_entities_set.add(entity);
    });

    // 2) process each entity exactly once
    const text_entities = Array.from(this.#text_entities_set);
    for (let i = 0; i < text_entities.length; i++) {
      const entity = text_entities[i];
      const segments = entity.segments;

      // 3) pull all code points across all segments into one array
      const code_points = [];
      for (let i = 0; i < segments.length; i++) {
        const { chunk, slot, count } = segments[i];
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
      const font_scale = font_size / font_object.texture_height;

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

      // 5) write back into each segment's transform chunk
      let global_index = 0;
      let anchor_x = 0;
      let anchor_y = 0;

      for (let i = 0; i < segments.length; i++) {
        const { chunk, slot, count } = segments[i];

        const transform_views = chunk.get_fragment_view(TransformFragment);
        for (let j = 0; j < count; j++) {
          const base_pos_index = (slot + j) * 4;
          const base_scale_index = (slot + j) * 4;

          // read current position
          const x = transform_views.position[base_pos_index + 0];
          const y = transform_views.position[base_pos_index + 1];

          // write new position
          if (global_index === 0) {
            anchor_x = x;
            anchor_y = y;
          } else {
            transform_views.position[base_pos_index + 0] = anchor_x + offsets[global_index];
            transform_views.position[base_pos_index + 1] =
              anchor_y -
              (font_object.y_offset[code_points[global_index]] * 2 +
                font_object.height[code_points[global_index]] -
                font_object.line_height) *
                font_scale;
          }

          // write new scale
          transform_views.scale[base_scale_index + 0] =
            (font_object.width[code_points[global_index]] / font_object.texture_width) * font_size;
          transform_views.scale[base_scale_index + 1] =
            (font_object.height[code_points[global_index]] / font_object.texture_height) *
            font_size;
          transform_views.scale[base_scale_index + 2] = 1.0;

          ++global_index;
        }

        chunk.mark_dirty();
      }
    }
  }
}
