import { Image } from "@/renderer/image.js";
import { RenderPassFlags } from "@/renderer/render_pass.js";
import { Buffer } from '@/renderer/buffer.js';

export class SimpleShadingStrategy {
    setup(context, render_graph) {
        // Temp theses simple triangle vertices in for now
        const vertices = [
            { position: [0.0, 0.5, 0.0, 1.0], color: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 0.0, 0.0, 0.0] },
            { position: [-0.5, -0.5, 0.0, 1.0], color: [0.0, 1.0, 0.0, 1.0], uv: [0.0, 1.0, 0.0, 0.0] },
            { position: [0.5, -0.5, 0.0, 1.0], color: [0.0, 0.0, 1.0, 1.0], uv: [1.0, 0.0, 0.0, 0.0] },
        ];
        const buffer = Buffer.create(context, {
            name: 'vertex_buffer',
            data: vertices.map(v => v.position.concat(v.color, v.uv)),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        render_graph.queue_global_buffer_writes([{
            buffer: buffer,
            offset: 0,
            size: buffer.config.size,
        }]);
    }

    draw(context, render_graph) {
        const swapchain_image = Image.create_from_image(context.context.getCurrentTexture(), 'swapchain');
        const rg_output_image = render_graph.register_image(swapchain_image.config.name);

        const shader_setup = {
            pipeline_shaders: {
                vertex: {
                    path: '/simple.wgsl' 
                },
                fragment: {
                    path: '/simple.wgsl'
                }
            }
        }

        render_graph.add_pass(
            "simple_pass",
            RenderPassFlags.Present,
            { inputs: [], outputs: [rg_output_image], shader_setup },
            (graph, frame_data, encoder) => {
                const pass = graph.get_physical_pass(frame_data.current_pass);
                pass.pass.draw(3, 1);
            }
        );

        render_graph.submit(context);
    }
}