import { GraphicsContext } from '@/renderer/graphics_context.js';
import { RenderGraph } from '@/renderer/render_graph.js';
import { SimpleShadingStrategy } from '@/renderer/strategies/simple_shading.js';

export default class Renderer {
    graphics_context = null;
    render_strategy = null;
    simple_shader = null;
    simple_vertex_buffer = null;
    render_graph = null;

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

    async setup(canvas) {
        this.graphics_context = await GraphicsContext.create(canvas);
        
        this.render_graph = RenderGraph.create();

        this.render_strategy = new SimpleShadingStrategy();
    }

    render() {
        performance.mark("frame_render");

        this.graphics_context.advance_frame();

        this.render_graph.begin(this.graphics_context);

        this.render_strategy.draw(this.graphics_context, this.render_graph);
    }
}