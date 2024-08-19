export class CommandQueue {
    static create_encoder(context, name) {
        return context.device.createCommandEncoder({ label: name });
    }

    static submit(context, encoder) {
        const command_buffer = encoder.finish();
        context.device.queue.submit([command_buffer]);
        context.device.queue.onSubmittedWorkDone().then(() => {
            context.execution_queue.update();
        });
    }
}