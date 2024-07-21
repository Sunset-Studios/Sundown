import { Renderer } from '../../renderer/renderer.js';
import { SimulationLayer } from '../simulation_layer.js';
import { SharedViewBuffer } from '../shared_data.js';
import { InputProvider } from '../../input/input_provider.js';
import { InputKey, InputRange } from '../../input/input_types.js';
import { radians } from '../../utility/math.js';
import { vec4, quat, vec3 } from 'gl-matrix';
import { WORLD_FORWARD, WORLD_UP } from '../minimal.js';

export class FreeformArcballControlProcessor extends SimulationLayer {
    constructor() {
        super();

        this.move_speed = 10.0;
        this.rotation_speed = 1.0; // Adjusted for smoother rotation
        this.orbit_distance = 10; // Fixed distance from pivot point
    }

    init(parent_context) {
        super.init(parent_context);

        const camera_position = vec4.fromValues(0, 0, -2, 1);
        const camera_rotation = quat.fromValues(0, 0, 0, 1);

        SharedViewBuffer.get().set_view_data(Renderer.get().graphics_context, parent_context.current_view, {
            position: camera_position,
            rotation: camera_rotation,
            aspect_ratio: Renderer.get().graphics_context.aspect_ratio,
            fov: radians(75),
        });
    }

    update(delta_time, parent_context) {
        super.update(delta_time, parent_context);

        const view_data = SharedViewBuffer.get().get_view_data(parent_context.current_view);
        let position = vec4.clone(view_data.position);
        let rotation = quat.clone(view_data.rotation);

        let moved = false;
        if (InputProvider.get().get_state(InputKey.K_w)) {
            const forward = vec4.scale(vec4.create(), view_data.view_forward ?? WORLD_FORWARD, this.move_speed * delta_time);
            vec4.add(position, position, forward);
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_s)) {
            const backward = vec4.scale(vec4.create(), view_data.view_forward ?? WORLD_FORWARD, -this.move_speed * delta_time);
            vec4.add(position, position, backward);
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_a)) {
            const left = vec4.scale(vec4.create(), view_data.view_right ?? WORLD_RIGHT, -this.move_speed * delta_time);
            vec4.add(position, position, left);
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_d)) {
            const right = vec4.scale(vec4.create(), view_data.view_right ?? WORLD_RIGHT, this.move_speed * delta_time);
            vec4.add(position, position, right);
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_q)) {
            const up = vec4.scale(vec4.create(), WORLD_UP, this.move_speed * delta_time);
            vec4.add(position, position, up);
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_e)) {
            const down = vec4.scale(vec4.create(), WORLD_UP, -this.move_speed * delta_time);
            vec4.add(position, position, down);
            moved = true;
        }

        const x = InputProvider.get().get_range(InputRange.M_x);
        const y = InputProvider.get().get_range(InputRange.M_y);
        if (x || y) {
            const orbit_distance = 10; // Fixed distance from pivot point
            const pivot_point = vec3.scaleAndAdd(vec3.create(), position, view_data.view_forward ?? WORLD_FORWARD, orbit_distance);

            // Calculate rotation quaternions for pitch and yaw
            const pitch_rotation = quat.setAxisAngle(quat.create(), [1, 0, 0], y * this.rotation_speed);
            const yaw_rotation = quat.setAxisAngle(quat.create(), [0, 1, 0], x * this.rotation_speed);

            // Combine rotations
            const delta_rotation = quat.multiply(quat.create(), yaw_rotation, pitch_rotation);

            // Apply rotation to current rotation
            quat.multiply(rotation, delta_rotation, rotation);

            // Calculate new position
            const offset = vec3.sub(vec3.create(), position, pivot_point);
            vec3.transformQuat(offset, offset, delta_rotation);
            vec3.add(position, pivot_point, offset);

            moved = true;
        }

        const context = Renderer.get().graphics_context;
        if (moved) {
            SharedViewBuffer.get().set_view_data(context, parent_context.current_view, {
                position: position,
                rotation: rotation,
            });
        }
        SharedViewBuffer.get().update_transforms(context, parent_context.current_view);
    }
}