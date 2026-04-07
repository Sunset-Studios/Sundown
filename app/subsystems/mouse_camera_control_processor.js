import { quat, vec3, vec4 } from "gl-matrix";

import { InputProvider } from "../../engine/src/input/input_provider.js";
import { InputKey, InputRange } from "../../engine/src/input/input_types.js";
import { radians } from "../../engine/src/utility/math.js";
import { WORLD_UP } from "../../engine/src/core/minimal.js";
import { SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { SimulationLayer } from "../../engine/src/core/simulation_layer.js";

const UNIT_RIGHT = vec3.fromValues(1, 0, 0);
const UNIT_UP = vec3.fromValues(0, 1, 0);
const UNIT_FORWARD = vec3.fromValues(0, 0, 1);

export class MouseCameraControlProcessor extends SimulationLayer {
  orbit_sensitivity = 7.0;
  pan_sensitivity = 28.0;
  zoom_sensitivity = 0.025;
  pivot_distance = 12.0;
  min_pivot_distance = 1.5;
  max_pivot_distance = 250.0;
  scene = null;
  drag_pivot_point = vec3.fromValues(0, 0, -12);
  drag_mode = null;

  init() {
    super.init();
  }

  cleanup() {
    this.scene = null;
    this.drag_mode = null;
    super.cleanup();
  }

  sync_drag_pivot(position, rotation) {
    const forward = vec3.transformQuat(vec3.create(), UNIT_FORWARD, rotation);
    vec3.scaleAndAdd(this.drag_pivot_point, position, forward, this.pivot_distance);
  }

  update(delta_time) {
    super.update(delta_time);

    const view_data = SharedViewBuffer.get_view_data(this.context.current_view);
    if (!view_data) {
      return;
    }

    const position = vec3.fromValues(
      view_data.view_position[0],
      view_data.view_position[1],
      view_data.view_position[2]
    );
    const rotation = quat.clone(view_data.view_rotation);

    const left_mouse =
      InputProvider.get_state(InputKey.B_mouse_left) ||
      InputProvider.get_action(InputKey.B_mouse_left);
    const right_mouse =
      InputProvider.get_state(InputKey.B_mouse_right) ||
      InputProvider.get_action(InputKey.B_mouse_right);
    const shift_held =
      InputProvider.get_state(InputKey.K_LShift) || InputProvider.get_state(InputKey.K_RShift);

    const mouse_x = InputProvider.get_range(InputRange.M_x);
    const mouse_y = InputProvider.get_range(InputRange.M_y);
    const wheel = InputProvider.get_range(InputRange.M_wheel);

    let moved = false;

    let next_drag_mode = null;
    if (left_mouse && shift_held) {
      next_drag_mode = "orbit";
    } else if (left_mouse) {
      next_drag_mode = "pan_xy";
    } else if (right_mouse) {
      next_drag_mode = "pan_xz";
    }

    if (next_drag_mode !== this.drag_mode) {
      if (next_drag_mode === "orbit") {
        this.sync_drag_pivot(position, rotation);
      }
      this.drag_mode = next_drag_mode;
    }

    if (this.drag_mode && (mouse_x || mouse_y)) {
      if (this.drag_mode === "orbit") {
        const yaw = -mouse_x * this.orbit_sensitivity;
        const pitch = -mouse_y * this.orbit_sensitivity;

        const yaw_rotation = quat.setAxisAngle(quat.create(), WORLD_UP, yaw);
        const rotated_rotation = quat.multiply(quat.create(), yaw_rotation, rotation);
        const rotated_right = vec3.transformQuat(vec3.create(), UNIT_RIGHT, rotated_rotation);
        const pitch_rotation = quat.setAxisAngle(quat.create(), rotated_right, pitch);
        const delta_rotation = quat.multiply(quat.create(), pitch_rotation, yaw_rotation);

        const offset = vec3.sub(vec3.create(), position, this.drag_pivot_point);
        vec3.transformQuat(offset, offset, delta_rotation);
        vec3.add(position, this.drag_pivot_point, offset);
        quat.multiply(rotation, delta_rotation, rotation);
        quat.normalize(rotation, rotation);
        moved = true;
      } else if (this.drag_mode === "pan_xy") {
        const pan_scale = Math.max(this.pivot_distance, this.min_pivot_distance) * this.pan_sensitivity;
        const pan_delta = vec3.create();
        const right = vec3.transformQuat(vec3.create(), UNIT_RIGHT, rotation);
        const up = vec3.transformQuat(vec3.create(), UNIT_UP, rotation);
        vec3.scaleAndAdd(pan_delta, pan_delta, right, -mouse_x * pan_scale);
        vec3.scaleAndAdd(pan_delta, pan_delta, up, -mouse_y * pan_scale);
        vec3.add(position, position, pan_delta);
        moved = true;
      } else if (this.drag_mode === "pan_xz") {
        const pan_scale = Math.max(this.pivot_distance, this.min_pivot_distance) * this.pan_sensitivity;
        const pan_delta = vec3.create();
        const right = vec3.transformQuat(vec3.create(), UNIT_RIGHT, rotation);
        const forward = vec3.transformQuat(vec3.create(), UNIT_FORWARD, rotation);
        vec3.scaleAndAdd(pan_delta, pan_delta, right, -mouse_x * pan_scale);
        vec3.scaleAndAdd(pan_delta, pan_delta, forward, -mouse_y * pan_scale);
        vec3.add(position, position, pan_delta);
        moved = true;
      }
    }

    if (wheel) {
      const forward = vec3.transformQuat(vec3.create(), UNIT_FORWARD, rotation);
      const zoom_amount = -wheel * this.zoom_sensitivity;
      vec3.scaleAndAdd(position, position, forward, zoom_amount);
      this.pivot_distance = Math.min(
        this.max_pivot_distance,
        Math.max(this.min_pivot_distance, this.pivot_distance - zoom_amount)
      );

      if (this.drag_mode === "orbit") {
        this.sync_drag_pivot(position, rotation);
      }
      moved = true;
    }

    if (moved) {
      view_data.view_position = vec4.fromValues(position[0], position[1], position[2], 1.0);
      view_data.view_rotation = rotation;
    }
  }

  set_scene(scene) {
    this.scene = scene;
    this.context.current_view = scene.context.current_view;
    this.drag_mode = null;

    const view_data = SharedViewBuffer.get_view_data(this.scene.context.current_view);
    view_data.fov = radians(75);
  }
}
