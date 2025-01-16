import { Renderer } from "./renderer.js";

export class CommandQueue {
    static create_encoder(name) {
        const renderer = Renderer.get();
        return renderer.device.createCommandEncoder({ label: name });
    }

    static submit(encoder, post_render_cb = null) {
        const renderer = Renderer.get();
        const command_buffer = encoder.finish();
        renderer.device.queue.submit([command_buffer]);
        renderer.device.queue.onSubmittedWorkDone().then(() => {
            renderer.execution_queue.update();
            if (post_render_cb) {
                post_render_cb();
            }
        });
    }
}