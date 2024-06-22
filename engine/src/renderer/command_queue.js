export class CommandQueue {
    queue = null;

    constructor() { }

    static create_encoder(context, name) {
        return context.device.createCommandEncoder({ label: name });
    }

    static submit(context, encoder) {
        const commandBuffer = encoder.finish();
        context.queue.submit([commandBuffer]);
    }
}