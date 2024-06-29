export class CommandQueue {
    queue = null;

    static create_encoder(context, name) {
        return context.device.createCommandEncoder({ label: name });
    }

    static submit(context, encoder) {
        const commandBuffer = encoder.finish();
        context.device.queue.submit([commandBuffer]);
    }
}