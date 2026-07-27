import { Renderer } from "../../renderer/renderer.js";
import { LightType, EntityFlags, WORLD_FORWARD } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { LightFragment } from "../ecs/fragments/light_fragment.js";
import { SharedViewBuffer, SharedFrameInfoBuffer, SharedEnvironmentData } from "../shared_data.js";
import {
  compute_directional_light_rotation,
  compute_directional_light_view_projection,
  ShadowAllocator,
} from "../../renderer/shadows/shadow_utils.js";
import { quat, vec4, vec3 } from "gl-matrix";

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
    const shadows_enabled = Renderer.get().is_shadows_enabled();

    let slot = 0;
    while (slot < DEFAULT_CHUNK_CAPACITY) {
      const flag = flags[slot];
      if ((flag & EntityFlags.ALIVE) === 0 || lights.active[slot] === 0) {
        slot += counts[slot] || 1;
        continue;
      }

      LightFragment.total_shadow_casting_lights += lights.shadow_casting[slot];

      const light_type = lights.type[slot];
      const camera_view_index = SharedFrameInfoBuffer.get_view_index();
      const camera_view = SharedViewBuffer.get_view_data(camera_view_index);

      // Reset shadow dirty flag if set
      if (lights.shadows_dirty[slot] > 0) {
        lights.shadows_dirty[slot] = 0;
        chunk.mark_dirty();
      }

      if (!lights.view_index[slot] || lights.view_index[slot] < 0) {
        const view = SharedViewBuffer.add_view_data();
        view.clipmap_count = shadows_enabled ? lights.shadow_clipmaps[slot] : 1;
        view.occlusion_enabled = 0;
        view.far = camera_view.far;
        view.renderable_state = shadows_enabled ? lights.shadow_casting[slot] : 0;

        const light_position = [
          lights.position[slot * 4 + 0],
          lights.position[slot * 4 + 1],
          lights.position[slot * 4 + 2],
          1.0,
        ];

        if (light_type === LightType.DIRECTIONAL) {
          view.custom_projection_enabled = 1;
          view.custom_view_matrix_enabled = 1;

          // ---------------------------------------------------------------------------
          // Use centralized utilities for stable rotation & projection ----------------

          const rotation = compute_directional_light_rotation(light_position);
          const light_dir = vec3.negate(
            vec3.create(),
            vec3.normalize(
              vec3.create(),
              vec3.transformQuat(vec3.create(), WORLD_FORWARD, rotation)
            )
          );

          let { view: light_view, proj: light_proj } = compute_directional_light_view_projection(
            camera_view.inverse_view_projection_matrix,
            light_dir,
            view.far
          );
          view.view_matrix = light_view;
          view.projection_matrix = light_proj;

          view.view_rotation = rotation;
          view.view_position = light_position;
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

        if (lights.is_primary_sun[slot] > 0 && SharedEnvironmentData.get_skydome_data() !== null) {
          SharedEnvironmentData.set_skydome_view(view.get_index());
        }

        lights.view_index[slot] = view.get_index();

        chunk.mark_dirty();
      } else {
        // Update existing view each frame – particularly important for directional lights so that
        // their orthographic projection follows the camera. We recompute the light-aligned
        // orthographic projection and dependent matrices here rather than relying on
        // SharedViewBuffer.update_transforms (which uses fixed −1..1 extents for orthographic
        // projections).
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

        if (light_type === LightType.DIRECTIONAL) {
          const rotation = compute_directional_light_rotation(light_position);
          const light_dir = vec3.negate(
            vec3.create(),
            vec3.normalize(vec3.create(), vec3.transformQuat(vec3.create(), [0, 0, 1], rotation))
          );

          let { view: light_view_mat, proj: light_proj_mat } =
            compute_directional_light_view_projection(
              camera_view.inverse_view_projection_matrix,
              light_dir,
              light_view.far
            );
          light_view.projection_matrix = light_proj_mat;
          light_view.view_matrix = light_view_mat;

          // If the light rotation has changed, mark the light as dirty so we can re-render all tiles
          let prev_light_rotation = light_view.view_rotation;
          if (!quat.equals(prev_light_rotation, rotation)) {
            lights.shadows_dirty[slot] = 1;
            chunk.mark_dirty();
          }

          light_view.view_rotation = rotation;
          light_view.view_position = light_position;
        } else {
          // If the light position or rotation has changed, mark the light as dirty so we can re-render all tiles
          let prev_light_position = light_view.view_position;
          if (!vec3.equals(prev_light_position, light_position)) {
            lights.shadows_dirty[slot] = 1;
            chunk.mark_dirty();
          }

          light_view.view_position = light_position;
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

      // Sky dome light management (if it's the primary sun)
      let light_view_index = lights.view_index[slot];
      if (
        lights.is_primary_sun[slot] > 0 &&
        light_view_index !== SharedEnvironmentData.get_skydome_view() &&
        SharedEnvironmentData.get_skydome_data() !== null
      ) {
        SharedEnvironmentData.set_skydome_view(light_view_index);
        chunk.mark_dirty();
      }

      slot += counts[slot] || 1;
    }
  }
}
