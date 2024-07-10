import { Image } from "../image.js";
import { RenderPassFlags } from "../render_pass.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";

export class DeferredShadingStrategy {
  initialized = false;

  setup(context, render_graph) { }

  draw(context, render_graph) {
    if (!this.initialized) {
      this.setup(context, render_graph);
      this.initialized = true;
    }

    MeshTaskQueue.get().sort_and_batch(context);

    const gpu_data = TransformFragment.to_gpu_data(context);
    const entity_transforms = render_graph.register_buffer(
      gpu_data.gpu_buffer.config.name
    );

    const object_instance_buffer =
      MeshTaskQueue.get().get_object_instance_buffer();
    const object_instances = render_graph.register_buffer(
      object_instance_buffer.config.name
    );

    // GBuffer Base Pass
    let main_albedo_image = null;
    let main_depth_image = null;
    let main_smra_image = null;
    let main_cc_image = null;
    let main_normal_image = null;
    let main_position_image = null;

    {
      const shader_setup = {
        pipeline_shaders: {
          vertex: {
            path: "gbuffer.wgsl",
          },
          fragment: {
            path: "gbuffer.wgsl",
          },
        },
      };

      const image_extent = context.get_canvas_resolution();
      main_albedo_image = render_graph.create_image({
        name: "main_albedo",
        format: "bgra8unorm",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      main_smra_image = render_graph.create_image({
        name: "main_smra",
        format: "bgra8unorm",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      main_normal_image = render_graph.create_image({
        name: "main_normal",
        format: "rgba16float",
        width: image_extent.width,
        height: image_extent.height,
        depth: 1,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      main_position_image = render_graph.create_image({
        name: "main_position",
        format: "rgba16float",
        width: image_extent.width,
        height: image_extent.height,
        depth: 1,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      main_depth_image = render_graph.create_image({
        name: "main_depth",
        format: "depth32float",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

      render_graph.add_pass(
        "gbuffer_base_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [entity_transforms, object_instances],
          outputs: [
            main_albedo_image,
            main_smra_image,
            main_normal_image,
            main_position_image,
            main_depth_image,
          ],
          shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.get().submit_indexed_indirect_draws(pass);
        }
      );
    }

    // Fullscreen Pass
    {
      const swapchain_image = Image.create_from_image(
        context.context.getCurrentTexture(),
        "swapchain"
      );
      const rg_output_image = render_graph.register_image(
        swapchain_image.config.name
      );

      const shader_setup = {
        pipeline_shaders: {
          vertex: {
            path: "fullscreen.wgsl",
          },
          fragment: {
            path: "fullscreen.wgsl",
          },
        },
      };

      render_graph.add_pass(
        "fullscreen_present_pass",
        RenderPassFlags.Graphics | RenderPassFlags.Present,
        {
          inputs: [main_albedo_image],
          outputs: [rg_output_image],
          shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.get().draw_quad(frame_data.context, pass);
        }
      );
    }

    render_graph.submit(context);
  }
}
