import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { UserInterfaceFragment } from "../ecs/fragments/user_interface_fragment.js";
import { Element3D } from "../../ui/3d/element.js";
import { InputProvider } from "../../input/input_provider.js";
import { InputKey } from "../../input/input_types.js";
import { profile_scope } from "../../utility/performance.js";

export class UI3DProcessor extends SimulationLayer {
  entity_query = null;
  scene = null;

  init() {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [UserInterfaceFragment],
    });
  }

  update(delta_time) {
    profile_scope("UI3DProcessor.update", () => {
      const user_interfaces = EntityManager.get().get_fragment_array(UserInterfaceFragment);
      if (!user_interfaces || user_interfaces.dirty.length === 0) {
        return;
      }

      for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
        const entity = this.entity_query.matching_entities[i];

        if (!user_interfaces.dirty[entity]) {
          continue;
        }

        if (user_interfaces.allows_cursor_events[entity]) {
          user_interfaces.was_cursor_inside[entity] = user_interfaces.is_cursor_inside[entity];
          user_interfaces.is_cursor_inside[entity] = entity === this.scene.get_cursor_pixel_entity();

          user_interfaces.was_clicked[entity] = user_interfaces.is_clicked[entity];
          user_interfaces.is_clicked[entity] = user_interfaces.is_cursor_inside[entity] && InputProvider.get().get_action(InputKey.B_mouse_left);

          user_interfaces.was_pressed[entity] = user_interfaces.is_pressed[entity];
          user_interfaces.is_pressed[entity] = user_interfaces.is_cursor_inside[entity] && InputProvider.get().get_state(InputKey.B_mouse_left);

          if (!user_interfaces.was_cursor_inside[entity] && user_interfaces.is_cursor_inside[entity]) {
            Element3D.trigger(entity, "hover");
          } else if (user_interfaces.was_cursor_inside[entity] && !user_interfaces.is_cursor_inside[entity]) {
            Element3D.trigger(entity, "leave");
          }

          if (user_interfaces.is_clicked[entity]) {
            Element3D.trigger(entity, "selected");
            InputProvider.get().consume_action(InputKey.B_mouse_left);
          }
          if (user_interfaces.is_pressed[entity]) {
            Element3D.trigger(entity, "pressed");
            InputProvider.get().consume_action(InputKey.B_mouse_left);
          }
        }
      }
    });
  }

  set_scene(scene) {
    this.scene = scene;
  }
}
