import Renderer from '@/renderer/renderer';
import { SimulationLayer } from '@/core/simulation_layer';
import { SharedViewBuffer } from '@/renderer/shared_data';
import { InputProvider } from '@/input/input_provider';
import { InputKey } from '@/input/input_types';
import { radians } from '@/utility/math';
import { vec4, quat } from 'gl-matrix';
import application_state from '@/core/application_state';

export class FreeformViewControlProcessor extends SimulationLayer {
    constructor() {
        super();

        this.move_speed = 10.0;
    }

    init() {
        super.init();

        const camera_position = vec4.fromValues(0, 0, -2, 1);
        const camera_rotation = quat.fromValues(0, 0, 0, 1);

        SharedViewBuffer.get().set_view_data(Renderer.get().graphics_context, application_state.current_view, {
            position: camera_position,
            rotation: camera_rotation,
            aspect_ratio: Renderer.get().graphics_context.aspect_ratio,
            fov: radians(75),
        });
    }

    update(delta_time) {
        super.update(delta_time);

        const view_data = SharedViewBuffer.get().get_view_data(application_state.current_view);
        let position = vec4.clone(view_data.position);

        let moved = false;
        if (InputProvider.get().get_state(InputKey.K_w)) {
            vec4.add(position, position, vec4.fromValues(0, 0, this.move_speed * delta_time, 0));
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_s)) {
            vec4.add(position, position, vec4.fromValues(0, 0, -this.move_speed * delta_time, 0));
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_a)) {
            vec4.add(position, position, vec4.fromValues(this.move_speed * delta_time, 0, 0, 0));
            moved = true;
        }
        if (InputProvider.get().get_state(InputKey.K_d)) {
            vec4.add(position, position, vec4.fromValues(-this.move_speed * delta_time, 0, 0, 0));
            moved = true;
        }

        const context = Renderer.get().graphics_context;
        if (moved) {
            SharedViewBuffer.get().set_view_data(context, application_state.current_view, {
                position: position,
            });
        }
        SharedViewBuffer.get().update_transforms(context, application_state.current_view);
    }
}