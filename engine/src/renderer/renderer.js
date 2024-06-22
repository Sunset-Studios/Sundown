import GraphicsContext from '@/renderer/graphics_context.js';
import PipelineState from '@/renderer/pipeline_state.js';
import Shader from '@/renderer/shader.js';

export default class Renderer {
    graphics_context = null;
    simple_shader = null;
    simple_pipeline = null;
    simple_render_pass = null;

    constructor() {
        if (Renderer.instance) {
            return Renderer.instance;
        }
        Renderer.instance = this;
    }

    static get() {
        if (!Renderer.instance) {
            Renderer.instance = new Renderer()
        }
        return Renderer.instance;
    }

    setup(canvas) {
        this.graphics_context = GraphicsContext.create(canvas);
        console.log('here')
        // TODO: Initialize the render graph

        this.simple_shader = Shader.create(
            this.graphics_context,
            '../../shaders/simple.wgsl'
        );

        this.simple_pipeline = PipelineState.create_render(
            this.graphics_context,
            'default_pipeline',
            {
                vertex: {
                    module: this.simple_shader,
                },
                fragment: {
                    module: this.simple_shader,
                    targets: [
                        {
                            format: this.graphics_context.canvas_format,
                        },
                    ],
                },
            }
        );

        this.simple_render_pass = RenderPass.create(this.graphics_context, 'simple_pass', {
            colorAttachments: {
                clearValue: [0.3, 0.3, 0.3, 1.0],
                loadOp: 'clear',
                storeOp: 'store',
                view: this.graphics_context.context.getCurrentTexture().createView(),
            },
        });
    }

    begin_frame() {
        this.graphics_context.advance_frame();

        const encoder = CommandQueue.create_encoder(this.graphics_context);
        this.simple_render_pass.begin(this.graphics_context, encoder, this.simple_pipeline);

        this.draw();

        this.simple_render_pass.end(this.graphics_context);
    }

    draw() {
        this.graphics_context.draw_pass(this.simple_render_pass, 3);
    }

    end_frame() {
        CommandQueue.submit(this.graphics_context, encoder);
    }

    render() {
        this.begin_frame();
        this.draw();
        this.end_frame();
    }

    cleanup() {
        this.graphics_context.cleanup();
        this.graphics_context = null;
    }
}