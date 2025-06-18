import { LightType, EntityFlags, WORLD_FORWARD, WORLD_UP, WORLD_RIGHT } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { LightFragment } from "../ecs/fragments/light_fragment.js";
import { SharedViewBuffer, SharedFrameInfoBuffer } from "../shared_data.js";
import { quat, vec4, vec3, mat4, mat3 } from "gl-matrix";

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

        const light_position = [
          lights.position[slot * 4 + 0],
          lights.position[slot * 4 + 1],
          lights.position[slot * 4 + 2],
          1.0,
        ];

        if (light_type === LightType.DIRECTIONAL) {
          const camera_view_index = SharedFrameInfoBuffer.get_view_index();
          const camera_view = SharedViewBuffer.get_view_data(camera_view_index);

          const clip0_extent = 8.0;

          view.fov = 0.0;
          view.far = camera_view.far;
          view.custom_projection_enabled = 1;
          view.custom_view_matrix_enabled = 1;

          // ---------------------------------------------------------------------------
          // build a *stable* world→light rotation -------------------------------------
          const light_forward = vec3.normalize(vec3.create(), vec3.negate(vec3.create(), light_position));

          // X axis = cross(forward , up)  -----------------------------
          let x_axis = vec3.cross(vec3.create(), light_forward, WORLD_UP);
          if (vec3.length(x_axis) < 1e-4)            // poles: pick any stable axis
              x_axis = vec3.clone(WORLD_RIGHT);
          vec3.normalize(x_axis, x_axis);

          // *** make its sign deterministic (here: always point +X) ***
          if (x_axis[0] < 0.0) vec3.negate(x_axis, x_axis);

          // Y axis, basis, quaternion --------------------------------
          const y_axis = vec3.cross(vec3.create(), light_forward, x_axis);
          vec3.normalize(y_axis, y_axis);

          const rot_rows = mat3.fromValues(
              x_axis[0], x_axis[1], x_axis[2],
              y_axis[0], y_axis[1], y_axis[2],
              light_forward[0], light_forward[1], light_forward[2]
          );
          const rotation = quat.fromMat3(quat.create(), rot_rows);
          // ---------------------------------------------------------------------------

          view.projection_matrix = mat4.fromValues(
            2.0 / clip0_extent,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / clip0_extent,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / camera_view.far,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0
          );
          view.view_matrix = mat4.fromRotationTranslation(mat4.create(),
                                                          rotation,
                                                          camera_view.view_position);
          view.view_position = camera_view.view_position;
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

          const clip0_extent = 8.0;

          const light_position = [
            lights.position[slot * 4 + 0],
            lights.position[slot * 4 + 1],
            lights.position[slot * 4 + 2],
            1.0,
          ];

          // ---------------------------------------------------------------------------
          // build a *stable* world→light rotation -------------------------------------
          const light_forward = vec3.normalize(vec3.create(), vec3.negate(vec3.create(), light_position));

          // X axis = cross(forward , up)  -----------------------------
          let x_axis = vec3.cross(vec3.create(), light_forward, WORLD_UP);
          if (vec3.length(x_axis) < 1e-4)            // poles: pick any stable axis
              x_axis = vec3.clone(WORLD_RIGHT);
          vec3.normalize(x_axis, x_axis);

          // *** make its sign deterministic (here: always point +X) ***
          if (x_axis[0] < 0.0) vec3.negate(x_axis, x_axis);

          // Y axis, basis, quaternion --------------------------------
          const y_axis = vec3.cross(vec3.create(), light_forward, x_axis);
          vec3.normalize(y_axis, y_axis);

          const rot_rows = mat3.fromValues(
              x_axis[0], x_axis[1], x_axis[2],
              y_axis[0], y_axis[1], y_axis[2],
              light_forward[0], light_forward[1], light_forward[2]
          );
          const rotation = quat.fromMat3(quat.create(), rot_rows);
          // ---------------------------------------------------------------------------

          const light_view = SharedViewBuffer.get_view_data(view_index);

          // Orthographic projection centred on the origin (stable virtual address).
          light_view.projection_matrix = mat4.fromValues(
            2.0 / clip0_extent,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / clip0_extent,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / camera_view.far,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0
          );
          light_view.view_matrix = mat4.fromRotationTranslation(mat4.create(),
                                                                  rotation,
                                                                  camera_view.view_position);
          light_view.view_position = camera_view.view_position;
          light_view.view_rotation = rotation;
        }
      }

      slot += counts[slot] || 1;
    }

    // After SharedViewBuffer.update_transforms has run
  }
}
