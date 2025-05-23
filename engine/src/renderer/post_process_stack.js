import { MeshTaskQueue } from "./mesh_task_queue.js";
import { RenderPassFlags } from "./renderer_types.js";
import { Name } from "../utility/names.js";
import { RingBufferAllocator } from "../memory/allocator.js";

const fullscreen_shader_path = "fullscreen.wgsl";

class RGPostProcessPassConfig {
    name = null;
    inputs = [];
    outputs = [];
    shader_setup = {
        pipeline_shaders: {
            vertex: { path: '' },
            fragment: { path: '' }
        }
    };
}

class PostUniformsConfig {
  name = null;
  data = [];
  usage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
}

class PostProcessPass {
    params = null;
    enabled = true;
    param_buffer = null;
    rg_config = new RGPostProcessPassConfig();
    uniforms_config = new PostUniformsConfig();
};

export class PostProcessStack {
    static view_passes = new Map();
    static view_pass_order = new Map();
    static pass_allocator = new RingBufferAllocator(24, PostProcessPass);
    static uniforms_allocator = new RingBufferAllocator(24, PostUniformsConfig);

    /**
     * Register a new post-process pass for a specific view
     * @param {number} view_id - The view to register the pass for
     * @param {string} name - Unique name for the post process pass
     * @param {string} shader_path - Path to the post process shader
     * @param {Object} [params={}] - Uniform parameters for the shader
     * @param {boolean} [enabled=true] - Whether the pass is initially enabled
     */
    static register_pass(view_id, name, shader_path, params = {}, enabled = true) {
        // Initialize view maps if they don't exist
        if (!this.view_passes.has(view_id)) {
            this.view_passes.set(view_id, new Map());
            this.view_pass_order.set(view_id, []);
        }

        const passes = this.view_passes.get(view_id);
        const pass_order = this.view_pass_order.get(view_id);

        const new_pass = this.pass_allocator.allocate();
        new_pass.params = params;
        new_pass.enabled = enabled;
        new_pass.rg_config.name = `post_process_${view_id}_${name}`;
        new_pass.rg_config.shader_setup.pipeline_shaders.vertex.path = fullscreen_shader_path;
        new_pass.rg_config.shader_setup.pipeline_shaders.fragment.path = shader_path;

        new_pass.uniforms_config.name = `post_process_${view_id}_${name}_uniforms`;
        new_pass.uniforms_config.data = Object.values(params).flat();
        
        passes.set(name, new_pass);
        pass_order.push(name);
    }

    /**
     * Enable or disable a post-process pass for a specific view
     */
    static set_pass_enabled(view_id, name, enabled) {
        const passes = this.view_passes.get(view_id);
        if (passes) {
            const pass = passes.get(name);
            if (pass) {
                pass.enabled = enabled;
            }
        }
    }

    /**
     * Update parameters for a post-process pass for a specific view
     */
    static update_pass_params(view_id, name, params) {
        const passes = this.view_passes.get(view_id);
        if (passes) {
            const pass = passes.get(name);
            if (pass) {
                Object.assign(pass.params, params);
                pass.uniforms_config.data = Object.values(pass.params).flat();
            }
        }
    }

    /**
     * Get all passes for a specific view
     */
    static get_view_passes(view_id) {
        return this.view_passes.get(view_id);
    }

    /**
     * Get the pass order for a specific view
     */
    static get_view_pass_order(view_id) {
        return this.view_pass_order.get(view_id);
    }

    /**
     * Clear all passes for a specific view
     */
    static clear_view(view_id) {
        this.view_passes.delete(view_id);
        this.view_pass_order.delete(view_id);
    }

    /**
     * Compile all enabled post-process passes for a specific view into the render graph
     */
    static compile_passes(view_id, render_graph, image_config, input_image, depth_image, normal_image) {
        const passes = this.get_view_passes(view_id);
        const pass_order = this.get_view_pass_order(view_id);
        
        if (!passes || passes.size === 0) return input_image;

        let current_input = input_image;

        // Create ping-pong textures for post processing
        const ping = render_graph.create_image({
            name: `post_process_ping_view${view_id}`,
            width: image_config.width,
            height: image_config.height,
            format: image_config.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        const pong = render_graph.create_image({
            name: `post_process_pong_view${view_id}`,
            width: image_config.width,
            height: image_config.height,
            format: image_config.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        let using_ping = true;

        // Add each enabled post process pass to the render graph
        for (const pass_name of pass_order) {
            const pass = passes.get(pass_name);

            if (!pass || !pass.enabled) continue;

            const output = using_ping ? ping : pong;

            const uniforms_buffer = render_graph.create_buffer(pass.uniforms_config);

            pass.rg_config.inputs.length = 0;
            pass.rg_config.inputs.push(uniforms_buffer);
            pass.rg_config.inputs.push(input_image);
            pass.rg_config.inputs.push(depth_image);
            pass.rg_config.inputs.push(normal_image);
            pass.rg_config.outputs.length = 0;
            pass.rg_config.outputs.push(output);

            render_graph.add_pass(
                pass.rg_config.name,
                RenderPassFlags.Graphics,
                pass.rg_config,
                (graph, frame_data, encoder) => {
                    const physical_pass = graph.get_physical_pass(frame_data.current_pass);
                    MeshTaskQueue.draw_quad(physical_pass);
                }
            );

            current_input = output;
            using_ping = !using_ping;
        }

        return current_input;
    }

    /**
     * Reset all post process stacks
     */
    static reset() {
        this.view_passes.clear();
        this.view_pass_order.clear();
        this.pass_allocator.reset();
    }
} 