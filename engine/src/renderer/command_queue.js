import { Renderer } from "./renderer.js";

export class CommandQueue {
    static create_encoder(name) {
        const renderer = Renderer.get();
        return renderer.device.createCommandEncoder({ label: name });
    }

    static submit(encoder) {
        const renderer = Renderer.get();
        const command_buffer = encoder.finish();
        renderer.device.queue.submit([command_buffer]);
        renderer.device.queue.onSubmittedWorkDone().then(() => {
            renderer.execution_queue.update();
        });
    }
}