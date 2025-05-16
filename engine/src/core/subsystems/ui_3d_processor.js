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
    this.entity_query = EntityManager.create_query([UserInterfaceFragment]);
    this._update_internal = this._update_internal.bind(this);
  }

  update(delta_time) {
    profile_scope("UI3DProcessor.update", this._update_internal);
  }

  _update_internal() {
    this.entity_query.for_each((chunk, slot, instance_count, archetype) => {
      const entity_flags = chunk.flags_meta[slot];
      if ((entity_flags & EntityFlags.PENDING_DELETE) !== 0) {
        return;
      }
      
      const entity = EntityManager.get_entity_for(chunk, slot);
      
      for (let i = 0; i < instance_count; ++i) {
        const user_interfaces = chunk.get_fragment_view(UserInterfaceFragment);
        const index = slot + i;

        if (user_interfaces.allows_cursor_events[index]) {
          user_interfaces.was_cursor_inside[index] = user_interfaces.is_cursor_inside[index];
          user_interfaces.is_cursor_inside[index] = entity === this.scene.get_cursor_pixel_entity();

          user_interfaces.was_clicked[index] = user_interfaces.is_clicked[index];
          user_interfaces.is_clicked[index] =
            user_interfaces.is_cursor_inside[index] &&
            InputProvider.get_action(InputKey.B_mouse_left);

          user_interfaces.was_pressed[index] = user_interfaces.is_pressed[index];
          user_interfaces.is_pressed[index] =
            user_interfaces.is_cursor_inside[index] &&
            InputProvider.get_state(InputKey.B_mouse_left);

          if (
            !user_interfaces.was_cursor_inside[index] &&
            user_interfaces.is_cursor_inside[index]
          ) {
            Element3D.trigger(entity, "hover");
          } else if (
            user_interfaces.was_cursor_inside[index] &&
            !user_interfaces.is_cursor_inside[index]
          ) {
            Element3D.trigger(entity, "leave");
          }

          if (user_interfaces.is_clicked[index]) {
            Element3D.trigger(entity, "selected");
            if (user_interfaces.consume_events[index]) {
              InputProvider.consume_action(InputKey.B_mouse_left);
            }
          }
          if (user_interfaces.is_pressed[index]) {
            Element3D.trigger(entity, "pressed");
            if (user_interfaces.consume_events[index]) {
              InputProvider.consume_action(InputKey.B_mouse_left);
            }
          }

          chunk.mark_dirty();
        }
      }
    });
  }

  set_scene(scene) {
    this.scene = scene;
  }
}
