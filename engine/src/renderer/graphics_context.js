const MAX_BUFFERED_FRAMES = 2;

export default class GraphicsContext {
    canvas = null;
    adapter = null;
    device = null;
    context = null;
    canvas_format = null;
    frame_number = 0;

    constructor() { }

    async setup(canvas) {
        if (!navigator.gpu) {
            throw Error('WebGPU is not supported');
        }

        this.canvas = canvas;
        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            throw Error('Unable to request WebGPU adapter');
        }
        
        this.device = await this.adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        this.canvas_format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.canvas_format,
            alphaMode: 'premultiplied'
        });
    }

    cleanup() {
        this.canvas = null;
        this.adapter = null;
        this.device = null;
        this.context = null;
    }

    advance_frame() {
        this.frame_number++;
    }

    get_frame_number() {
        return this.frame_number;
    }

    get_buffered_frame_number() {
        return this.frame_number % MAX_BUFFERED_FRAMES;
    }

    draw_pass(render_pass, triangles) {
        render_pass.pass.draw(triangles);
    }

    static create(canvas) {
        return new GraphicsContext(canvas);
    }
}
