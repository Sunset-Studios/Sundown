import { LightType, EntityFlags, WORLD_FORWARD, WORLD_UP } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { LightFragment } from "../ecs/fragments/light_fragment.js";
import { SharedViewBuffer, SharedFrameInfoBuffer } from "../shared_data.js";
import {
  compute_directional_light_rotation,
  build_directional_light_projection_matrix,
  compute_directional_light_position_for_clip,
  ShadowAllocator,
} from "../../renderer/shadows/shadow_utils.js";
import { quat, vec4, mat4 } from "gl-matrix";

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

      const light_type = lights.type[slot];

      if (!lights.view_index[slot] || lights.view_index[slot] < 0) {
        const view = SharedViewBuffer.add_view_data();
        view.renderable_state = lights.shadow_casting[slot];
        view.occlusion_enabled = 0;
        view.culling_enabled = 0;

        const light_position = [
          lights.position[slot * 4 + 0],
          lights.position[slot * 4 + 1],
          lights.position[slot * 4 + 2],
          1.0,
        ];

        if (light_type === LightType.DIRECTIONAL) {
          const camera_view_index = SharedFrameInfoBuffer.get_view_index();
          const camera_view = SharedViewBuffer.get_view_data(camera_view_index);

          view.custom_projection_enabled = 1;

          // ---------------------------------------------------------------------------
          // Use centralized utilities for stable rotation & projection ----------------
          const rotation = compute_directional_light_rotation(light_position);
          const { position: light_pos } = compute_directional_light_position_for_clip(
            rotation,
            camera_view.view_position,
            camera_view.inverse_view_projection_matrix
          );

          view.projection_matrix = build_directional_light_projection_matrix(
            camera_view.far,
          );

          view.view_position = light_pos;
          view.view_rotation = rotation;
        } else {
          view.fov = 90.0;

          const direction = vec4.fromValues(
            lights.direction[slot * 4 + 0],
            lights.direction[slot * 4 + 1],
            lights.direction[slot * 4 + 2],
            lights.direction[slot * 4 + 3]
          );
          const rotation = quat.rotationTo(quat.create(), [0, 0, 1], direction);

          view.view_position = light_position;
          view.view_rotation = rotation;
        }

        lights.view_index[slot] = view.get_index();

        chunk.mark_dirty();
      } else {
        // Update existing view each frame – particularly important for directional lights so that
        // their orthographic projection follows the camera. We recompute the light-aligned
        // orthographic projection and dependent matrices here rather than relying on
        // SharedViewBuffer.update_transforms (which uses fixed −1..1 extents for orthographic
        // projections).
        if (light_type === LightType.DIRECTIONAL) {
          const view_index = lights.view_index[slot];
          if (view_index < 0) {
            slot += counts[slot] || 1;
            continue;
          }

          // Retrieve the active camera view to build a camera-relative projection.
          const camera_view_index = SharedFrameInfoBuffer.get_view_index();
          const camera_view = SharedViewBuffer.get_view_data(camera_view_index);

          // Guard against invalid indices (e.g. when no camera yet available).
          if (!camera_view) {
            slot += counts[slot] || 1;
            continue;
          }

          const light_position = [
            lights.position[slot * 4 + 0],
            lights.position[slot * 4 + 1],
            lights.position[slot * 4 + 2],
            1.0,
          ];
          const light_view = SharedViewBuffer.get_view_data(view_index);

          const rotation = compute_directional_light_rotation(light_position);
          const { position: light_pos2 } = compute_directional_light_position_for_clip(
            rotation,
            camera_view.view_position,
            camera_view.inverse_view_projection_matrix
          );

          // Orthographic projection centred on the origin (stable virtual address).
          light_view.projection_matrix = build_directional_light_projection_matrix(
            camera_view.far,
          );
          light_view.view_rotation = rotation;
          light_view.view_position = light_pos2;
        }
      }

      // Shadow index management
      if (lights.shadow_index[slot] < 0 && lights.shadow_casting[slot] > 0) {
        lights.shadow_index[slot] = ShadowAllocator.allocate();
        chunk.mark_dirty();
      } else if (lights.shadow_index[slot] >= 0 && lights.shadow_casting[slot] === 0) {
        ShadowAllocator.free(lights.shadow_index[slot]);
        lights.shadow_index[slot] = -1;
        chunk.mark_dirty();
      }

      slot += counts[slot] || 1;
    }
  }
}
