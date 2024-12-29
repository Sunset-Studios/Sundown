import { Renderer } from "../../renderer/renderer.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { TextFragment } from "../ecs/fragments/text_fragment.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { profile_scope } from "../../utility/performance.js";
import { FontCache } from "../../ui/text/font_cache.js";

const text_processor_update_key = "text_processor_update";

export class TextProcessor extends SimulationLayer {
  entity_query = null;

  init() {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [TextFragment, StaticMeshFragment],
    });
  }

  update(delta_time) {
    profile_scope(text_processor_update_key, () => {
      const texts =
      EntityManager.get().get_fragment_array(TextFragment);
      if (!texts) {
        return;
      }

      for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
        const entity = this.entity_query.matching_entities[i];

        if (!texts.dirty[entity]) {
          continue;
        }

        const text_data = TextFragment.get_entity_data(entity);
        const text = text_data.text;

        if (text) {
          const text_mesh = StaticMeshFragment.get_entity_data(entity);
          text_mesh.instance_count = text.length;

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
        TextFragment.to_gpu_data(Renderer.get().graphics_context);
      }
    });
  }
}
