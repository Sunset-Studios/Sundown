import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager, EntityID } from "../ecs/entity.js";
import { TextFragment } from "../ecs/fragments/text_fragment.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { profile_scope } from "../../utility/performance.js";
import { FontCache } from "../../ui/text/font_cache.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";

const text_processor_update_key = "text_processor_update";

export class TextProcessor extends SimulationLayer {
  entity_query = null;

  init() {
    this.entity_query = EntityManager.create_query({
      fragment_requirements: [TextFragment, StaticMeshFragment, TransformFragment],
    });
  }

  update(delta_time) {
    profile_scope(text_processor_update_key, () => {
      const texts = EntityManager.get_fragment_array(TextFragment);
      if (!texts) {
        return;
      }

      const transform = EntityManager.get_fragment_array(TransformFragment);
      if (!transform) {
        return;
      }

      const matching_entity_data = this.entity_query.matching_entities.get_data();
      for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
        const entity = matching_entity_data[i];

        if (!texts.dirty[entity]) {
          continue;
        }

        const text_data = TextFragment.get_entity_data(entity);
        const text = text_data.text;

        if (text) {
          EntityManager.change_entity_instance_count(entity, text.length);

          const font = text_data.font;
          const font_object = FontCache.get_font_object(font);
          const font_scale = text_data.font_size / font_object.texture_height;

          // Calculate offsets for each character
          let offsets = Array(text.length).fill(0);
          for (let j = 1; j < text.length; ++j) {
            const prev_code_point_index = text[j - 1]; // Use previous character for offset
            const code_point_index = text[j];
            const kerning = font_object.kerning_matrix.get_adjacent_value(
              font_object.code_point[prev_code_point_index],
              font_object.code_point[code_point_index]
            );
            offsets[j] =
              offsets[j - 1] +
              (font_object.width[prev_code_point_index] +
                font_object.x_advance[code_point_index] +
                font_object.x_offset[code_point_index] +
                kerning) *
                font_scale;
          }

          if (transform && transform.position) {
            // Calculate the total width of the text block
            const total_width = offsets[offsets.length - 1];

            // Update position for each glyph
            for (let j = 0; j < text.length; ++j) {
              const code_point_index = text[j];
              const transform_fragment = EntityManager.get_fragment(entity, TransformFragment, j);
              if (!transform_fragment) continue;

              const position = transform_fragment.position;
              transform_fragment.position = [
                position[0] + (offsets[j] - total_width * 0.5),
                position[1] -
                  (font_object.y_offset[code_point_index] * 2.0 +
                    font_object.height[code_point_index]) *
                    font_scale,
                position[2],
              ];

              // Convert from texture space to world space while maintaining pixel ratio
              const base_scale_x = font_object.width[code_point_index] / font_object.texture_width;
              const base_scale_y =
                font_object.height[code_point_index] / font_object.texture_height;
              transform_fragment.scale = [
                base_scale_x * text_data.font_size,
                base_scale_y * text_data.font_size,
                1.0,
              ];
            }
          }

          text_data.offsets = offsets;
        }

        texts.dirty[entity] = 0;

        // We discard the result as we only want to write the underlying buffers, which are already mapped to the GPU
        TextFragment.to_gpu_data();
      }
    });
  }
}
