import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager, EntityID } from "../ecs/entity.js";
import { TextFragment } from "../ecs/fragments/text_fragment.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { profile_scope } from "../../utility/performance.js";
import { FontCache } from "../../ui/text/font_cache.js";

const text_processor_update_key = "text_processor_update";

export class TextProcessor extends SimulationLayer {
  entity_query = null;

  init() {
    this.entity_query = EntityManager.create_query({
      fragment_requirements: [TextFragment, StaticMeshFragment],
    });
  }

  update(delta_time) {
    profile_scope(text_processor_update_key, () => {
      const texts = EntityManager.get_fragment_array(TextFragment);
      if (!texts) {
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

          let offsets = [0];
          for (let i = 1; i < text.length; ++i)
          {
            const font = text_data.font;
            const font_object = FontCache.get_font_object(font);
            const code_point_index = text[i];
            offsets.push(offsets[i - 1] + font_object.width[code_point_index] + font_object.x_advance[code_point_index]);
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
