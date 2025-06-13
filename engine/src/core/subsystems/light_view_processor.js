import { LightType, EntityFlags, WORLD_UP } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { LightFragment } from "../ecs/fragments/light_fragment.js";
import { SharedViewBuffer, SharedFrameInfoBuffer } from "../shared_data.js";
import { quat, vec4, vec3, mat4 } from "gl-matrix";

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
        
        const position = [
          lights.position[slot * 4 + 0],
          lights.position[slot * 4 + 1],
          lights.position[slot * 4 + 2],
          1.0,
        ];

        if (light_type === LightType.DIRECTIONAL) {
          const camera_view_index = SharedFrameInfoBuffer.get_view_index();
          const camera_view = SharedViewBuffer.get_view_data(camera_view_index);

          view.fov = 0.0;
          view.zoom = 0.05;
          view.far = camera_view.far;

          const direction = vec4.negate(vec4.create(), position);
          const rotation = quat.rotationTo(quat.create(), [0, 0, 1], direction);

          view.view_position = position;
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

          view.view_position = position;
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

          // --- Compute bounding sphere based on the camera far plane ---
          const camera_position = camera_view.view_position;
          const camera_far = camera_view.far || 100.0;

          const light_view = SharedViewBuffer.get_view_data(view_index);
          const light_forward = light_view.forward;
          const light_zoom = light_view.zoom;

          // Center the light volume around the camera position and move it backwards along
          // the light direction so the camera is roughly centred within the depth range.
          const light_position = vec3.scaleAndAdd(
            vec3.create(),
            camera_position,
            light_forward,
            -camera_far * light_zoom
          );

          const direction = vec4.negate(vec4.create(), light_position);
          const rotation = quat.rotationTo(quat.create(), [0, 0, 1], direction);

          // light_view.view_position = vec4.fromValues(
          //   light_position[0],
          //   light_position[1],
          //   light_position[2],
          //   1.0
          // );
          //light_view.view_rotation = rotation;
        }

        // TODO: Frustum-frustum intersection tests to cull lights?
      }

      slot += counts[slot] || 1;
    }
  }
}
