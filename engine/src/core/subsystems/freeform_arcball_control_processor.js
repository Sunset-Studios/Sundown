import { Renderer } from '../../renderer/renderer.js'; 
import { SimulationLayer } from '../simulation_layer.js';
import { SharedViewBuffer } from '../shared_data.js';
import { InputProvider } from '../../input/input_provider.js';
import { InputKey, InputRange } from '../../input/input_types.js';
import { radians } from '../../utility/math.js';
import { vec4, quat, vec3 } from 'gl-matrix';
import { WORLD_FORWARD, WORLD_UP } from '../minimal.js';
import { global_dispatcher } from '../dispatcher.js';

export class FreeformArcballControlProcessor extends SimulationLayer {
    move_speed = 10.0;
    rotation_speed = 3.0; // Adjusted for smoother rotation
    orbit_distance = 20; // Fixed distance from pivot point
    scene = null;

    init() {
        super.init();
        global_dispatcher.on("resolution_change", this.on_resolution_change.bind(this));
    }

    pre_update(delta_time) {
        super.pre_update(delta_time);

        const ctrl_held = InputProvider.get_state(InputKey.K_LControl) || InputProvider.get_state(InputKey.K_RControl);
        if (ctrl_held) {
            return;
        }
        
        const view_data = SharedViewBuffer.get_view_data(this.context.current_view);
        let position = vec4.clone(view_data.position);
        let rotation = quat.clone(view_data.rotation);
        
        let moved = false;
        if (InputProvider.get_state(InputKey.K_w)) {
            const forward = vec4.scale(vec4.create(), view_data.view_forward ?? WORLD_FORWARD, this.move_speed * delta_time);
            vec4.add(position, position, forward);
            moved = true;
        }
        if (InputProvider.get_state(InputKey.K_s)) {
            const backward = vec4.scale(vec4.create(), view_data.view_forward ?? WORLD_FORWARD, -this.move_speed * delta_time);
            vec4.add(position, position, backward);
            moved = true;
        }
        if (InputProvider.get_state(InputKey.K_a)) {
            const left = vec4.scale(vec4.create(), view_data.view_right ?? WORLD_RIGHT, -this.move_speed * delta_time);
            vec4.add(position, position, left);
            moved = true;
        }
        if (InputProvider.get_state(InputKey.K_d)) {
            const right = vec4.scale(vec4.create(), view_data.view_right ?? WORLD_RIGHT, this.move_speed * delta_time);
            vec4.add(position, position, right);
            moved = true;
        }
        if (InputProvider.get_state(InputKey.K_q)) {
            const up = vec4.scale(vec4.create(), WORLD_UP, this.move_speed * delta_time);
            vec4.add(position, position, up);
            moved = true;
        }
        if (InputProvider.get_state(InputKey.K_e)) {
            const down = vec4.scale(vec4.create(), WORLD_UP, -this.move_speed * delta_time);
            vec4.add(position, position, down);
            moved = true;
        }
        
        const shift_held = InputProvider.get_state(InputKey.K_LShift) || InputProvider.get_state(InputKey.K_RShift);
        const x = InputProvider.get_range(InputRange.M_x);
        const y = InputProvider.get_range(InputRange.M_y);

        if ((x || y) && shift_held) {
            const pivot_point = vec3.scaleAndAdd(vec3.create(), position, view_data.view_forward ?? WORLD_FORWARD, this.orbit_distance);

            // Calculate rotation based on mouse movement
            const rotationX = quat.setAxisAngle(quat.create(), WORLD_UP, x * this.rotation_speed);
            const rotationY = quat.setAxisAngle(quat.create(), vec3.cross(vec3.create(), WORLD_UP, view_data.view_forward), -y * this.rotation_speed);
            
            // Combine rotations
            const delta_rotation = quat.multiply(quat.create(), rotationX, rotationY);
            
            // Apply rotation to current rotation
            quat.multiply(rotation, delta_rotation, rotation);

            // Calculate new position
            const offset = vec3.sub(vec3.create(), position, pivot_point);
            vec3.transformQuat(offset, offset, delta_rotation);
            vec3.add(position, pivot_point, offset);

            moved = true;
        }

        if (moved) {
            SharedViewBuffer.set_view_data(this.context.current_view, {
                position: position,
                rotation: rotation
            });
        }
        SharedViewBuffer.update_transforms(this.context.current_view);
    }

    on_resolution_change() {
        SharedViewBuffer.set_view_data(this.context.current_view, {
            aspect_ratio: Renderer.get().aspect_ratio
        });
    }

    set_scene(scene) {
        this.scene = scene;
        this.context.current_view = scene.context.current_view;

        const camera_position = vec4.fromValues(0, 0, 0, 1);
        const camera_rotation = quat.fromEuler(quat.create(), 0, 180.0, 0);

        SharedViewBuffer.set_view_data(this.scene.context.current_view, {
            position: camera_position,
            rotation: camera_rotation,
            aspect_ratio: Renderer.get().aspect_ratio,
            fov: radians(75),
        });
    }
}