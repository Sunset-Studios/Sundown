import { Image } from "@/renderer/image.js";
import { RenderPassFlags } from "@/renderer/render_pass.js";
import { SharedVertexBuffer } from '@/renderer/shared_data.js';
import { SharedViewBuffer } from '@/renderer/shared_data.js';
import { MeshTaskQueue } from "@/renderer/mesh_task_queue";

export class SimpleShadingStrategy {
    initialized = false;

    setup(context, render_graph) {
        SharedVertexBuffer.get().build(context);
        SharedViewBuffer.get().build(context);

        render_graph.queue_global_buffer_writes([{
            buffer: SharedVertexBuffer.get().buffer,
            offset: 0,
            size: SharedVertexBuffer.get().size,
        }]);

        render_graph.queue_global_buffer_writes([{
            buffer: SharedViewBuffer.get().buffer,
            offset: 0,
            size: SharedViewBuffer.get().size,
        }]);
    }

    draw(context, render_graph) {
        if (!this.initialized) {
            this.setup(context, render_graph);
            this.initialized = true;
        }

        MeshTaskQueue.get().sort();

        const swapchain_image = Image.create_from_image(context.context.getCurrentTexture(), 'swapchain');
        const rg_output_image = render_graph.register_image(swapchain_image.config.name);

        const shader_setup = {
            pipeline_shaders: {
                vertex: {
                    path: 'simple.wgsl' 
                },
                fragment: {
                    path: 'simple.wgsl'
                }
            }
        }

        render_graph.add_pass(
            "simple_pass",
            RenderPassFlags.Present,
            { inputs: [], outputs: [rg_output_image], shader_setup },
            (graph, frame_data, encoder) => {
                const pass = graph.get_physical_pass(frame_data.current_pass);
                MeshTaskQueue.get().submit_indexed_draws(pass);
            }
        );

        render_graph.submit(context);
    }
}