import { Renderer } from "../../renderer/renderer.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager, EntityID } from "../ecs/entity.js";
import { UserInterfaceFragment } from "../ecs/fragments/user_interface_fragment.js";
import { Element3D } from "../../ui/3d/element.js";
import { InputProvider } from "../../input/input_provider.js";
import { InputKey } from "../../input/input_types.js";
import { profile_scope } from "../../utility/performance.js";

export class UI3DProcessor extends SimulationLayer {
  entity_query = null;
  scene = null;

  init() {
    this.entity_query = EntityManager.create_query({
      fragment_requirements: [UserInterfaceFragment],
    });
  }

  update(delta_time) {
    profile_scope("UI3DProcessor.update", () => {
      const user_interfaces = EntityManager.get_fragment_array(UserInterfaceFragment);
      if (!user_interfaces || user_interfaces.dirty.length === 0) {
        return;
      }

      let updated = false;

      const matching_entities = this.entity_query.matching_entities.get_data();
      const matching_entity_ids = this.entity_query.matching_entity_ids.get_data();
      for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
        const entity = matching_entities[i];
        const entity_index = matching_entity_ids[i];

        if (!user_interfaces.dirty[entity_index]) {
          continue;
        }

        if (user_interfaces.allows_cursor_events[entity_index]) {
          user_interfaces.was_cursor_inside[entity_index] = user_interfaces.is_cursor_inside[entity_index];
          user_interfaces.is_cursor_inside[entity_index] = entity === this.scene.get_cursor_pixel_entity();

          user_interfaces.was_clicked[entity_index] = user_interfaces.is_clicked[entity_index];
          user_interfaces.is_clicked[entity_index] = user_interfaces.is_cursor_inside[entity_index] && InputProvider.get().get_action(InputKey.B_mouse_left);

          user_interfaces.was_pressed[entity_index] = user_interfaces.is_pressed[entity_index];
          user_interfaces.is_pressed[entity_index] = user_interfaces.is_cursor_inside[entity_index] && InputProvider.get().get_state(InputKey.B_mouse_left);

          if (!user_interfaces.was_cursor_inside[entity_index] && user_interfaces.is_cursor_inside[entity_index]) {
            Element3D.trigger(entity, "hover");
          } else if (user_interfaces.was_cursor_inside[entity_index] && !user_interfaces.is_cursor_inside[entity_index]) {
            Element3D.trigger(entity, "leave");
          }

          if (user_interfaces.is_clicked[entity_index]) {
            Element3D.trigger(entity, "selected");
            if (user_interfaces.consume_events[entity_index]) {
              InputProvider.get().consume_action(InputKey.B_mouse_left);
            }
          }
          if (user_interfaces.is_pressed[entity_index]) {
            Element3D.trigger(entity, "pressed");
            if (user_interfaces.consume_events[entity_index]) {
              InputProvider.get().consume_action(InputKey.B_mouse_left);
            }
          }
        }

        updated = true;
      }

      if (updated) {
        UserInterfaceFragment.to_gpu_data();
      }
    });
  }

  set_scene(scene) {
    this.scene = scene;
  }
}
