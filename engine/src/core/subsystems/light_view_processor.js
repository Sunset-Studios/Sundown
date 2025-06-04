import { LightType } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { LightFragment } from "../ecs/fragments/light_fragment.js";
import { SharedViewBuffer } from "../shared_data.js";
import { EntityFlags } from "../minimal.js";
import { quat, vec4 } from "gl-matrix";

export class LightViewProcessor extends SimulationLayer {
  entity_query = null;

  init() {
    super.init();
    this.entity_query = EntityManager.create_query([LightFragment]);
    this._update_internal_iter_chunk = this._update_internal_iter_chunk.bind(this);
  }

  cleanup() {
    this.entity_query = null;
    super.cleanup();
  }

  update(delta_time) {
    LightFragment.total_shadow_casting_lights = 0;
    this.entity_query.for_each_chunk(this._update_internal_iter_chunk);
  }

  _update_internal_iter_chunk(chunk, flags, counts, archetype) {
    const lights = chunk.get_fragment_view(LightFragment);

    let slot = 0;
    while (slot < DEFAULT_CHUNK_CAPACITY) {
      const flag = flags[slot];
      if ((flag & EntityFlags.ALIVE) === 0 || lights.active[slot] === 0) {
        slot += counts[slot] || 1;
        continue;
      }

      LightFragment.total_shadow_casting_lights += lights.shadow_casting[slot];

      if (!lights.view_index[slot] || lights.view_index[slot] < 0) {
        const view = SharedViewBuffer.add_view_data();
        view.renderable_state = lights.shadow_casting[slot];
        view.occlusion_enabled = 0;

        lights.view_index[slot] = view.get_index();

        const position = [
          lights.position[slot * 4 + 0],
          lights.position[slot * 4 + 1],
          lights.position[slot * 4 + 2],
          1.0,
        ];
        view.view_position = position;
        
        const rotation = quat.fromValues(0, 0, 0, 1);
        if (lights.type[slot] === LightType.DIRECTIONAL) {
          const direction = vec4.negate(vec4.create(), position);
          quat.rotationTo(rotation, [0, 0, 1], direction);
        } else {
          const direction = vec4.fromValues(
            lights.direction[slot * 4 + 0],
            lights.direction[slot * 4 + 1],
            lights.direction[slot * 4 + 2],
            lights.direction[slot * 4 + 3]
          );
          quat.rotationTo(rotation, [0, 0, 1], direction);
        }

        view.view_rotation = rotation;

        chunk.mark_dirty();
      } else {
        // TODO: Frustum-frustum intersection tests to cull lights?
      }

      slot += counts[slot] || 1;
    }
  }
}
