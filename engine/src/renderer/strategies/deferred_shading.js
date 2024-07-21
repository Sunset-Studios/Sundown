import { Texture } from "../texture.js";
import { RenderPassFlags } from "../render_pass.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { SharedEnvironmentMapData } from "../../core/shared_data.js";
import { ImageFlags } from "../texture.js";
import { Material } from "../material.js";

export class DeferredShadingStrategy {
  initialized = false;

  setup(context, render_graph) {}

  draw(context, render_graph) {
    if (!this.initialized) {
      this.setup(context, render_graph);
      this.initialized = true;
    }

    MeshTaskQueue.get().sort_and_batch(context);

    const transform_gpu_data = TransformFragment.to_gpu_data(context);
    const entity_transforms = render_graph.register_buffer(
      transform_gpu_data.gpu_buffer.config.name
    );

    const light_gpu_data = LightFragment.to_gpu_data(context);
    const lights = render_graph.register_buffer(
      light_gpu_data.gpu_buffer.config.name
    );

    const object_instance_buffer =
      MeshTaskQueue.get().get_object_instance_buffer();
    const object_instances = render_graph.register_buffer(
      object_instance_buffer.config.name
    );

    let skybox_image = null;
    let main_albedo_image = null;
    let main_depth_image = null;
    let main_smra_image = null;
    let main_cc_image = null;
    let main_normal_image = null;
    let main_position_image = null;
    let main_entity_id_image = null;
    let post_lighting_image = null;

    const image_extent = context.get_canvas_resolution();

    // Skybox Pass
    {
      const shader_setup = {
        pipeline_shaders: {
          vertex: {
            path: "skybox.wgsl",
          },
          fragment: {
            path: "skybox.wgsl",
          },
        },
        rasterizer_state: {
          cull_mode: "none",
        },
        depth_write_enabled: false,
      };

      const skybox = SharedEnvironmentMapData.get().get_skybox();
      const skybox_texture = render_graph.register_image(skybox.config.name);

      skybox_image = render_graph.create_image({
        name: "skybox_output",
        format: "bgra8unorm",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
      });

      render_graph.add_pass(
        "skybox_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [skybox_texture],
          outputs: [skybox_image],
          shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.get().draw_cube(frame_data.context, pass);
        }
      );
    }

    // Entity Prepass
    {
      const shader_setup = {
        pipeline_shaders: {
          vertex: {
            path: "entity_prepass.wgsl",
          },
          fragment: {
            path: "entity_prepass.wgsl",
          },
        },
      };

      main_entity_id_image = render_graph.create_image({
        name: "main_entity_id",
        format: "r32uint",
        width: image_extent.width,
        height: image_extent.height,
        depth: 1,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
      });

      main_depth_image = render_graph.create_image({
        name: "main_depth",
        format: "depth32float",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
        load_op: "load",
      });

      render_graph.add_pass(
        "entity_prepass",
        RenderPassFlags.Graphics,
        {
          inputs: [entity_transforms, object_instances],
          outputs: [main_entity_id_image, main_depth_image],
          shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.get().submit_indexed_indirect_draws(
            pass,
            frame_data,
            false /* should_reset */
          );
        }
      );
    }

    // GBuffer Base Pass
    {
      main_albedo_image = render_graph.create_image({
        name: "main_albedo",
        format: "bgra8unorm",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
      });
      main_smra_image = render_graph.create_image({
        name: "main_smra",
        format: "bgra8unorm",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
      });
      main_normal_image = render_graph.create_image({
        name: "main_normal",
        format: "rgba16float",
        width: image_extent.width,
        height: image_extent.height,
        depth: 1,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
      });
      main_position_image = render_graph.create_image({
        name: "main_position",
        format: "rgba16float",
        width: image_extent.width,
        height: image_extent.height,
        depth: 1,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
      });

      const material_buckets = MeshTaskQueue.get().get_material_buckets();
      for (const material_id of material_buckets) {
        const material = Material.get(material_id);

        render_graph.add_pass(
          `g_buffer_${material.template.name}_${material_id}`,
          RenderPassFlags.Graphics,
          {
            inputs: [entity_transforms, object_instances],
            outputs: [
              main_albedo_image,
              main_smra_image,
              main_position_image,
              main_normal_image,
              main_depth_image
            ],
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);

            if (!frame_data.g_buffer_data) {
              frame_data.g_buffer_data = {
                albedo: graph.get_physical_image(main_albedo_image),
                smra: graph.get_physical_image(main_smra_image),
                position: graph.get_physical_image(main_position_image),
                normal: graph.get_physical_image(main_normal_image),
                entity_id: graph.get_physical_image(main_entity_id_image),
                depth: graph.get_physical_image(main_depth_image),
              };

              frame_data.g_buffer_data.albedo.config.load_op = "load";
              frame_data.g_buffer_data.smra.config.load_op = "load";
              frame_data.g_buffer_data.position.config.load_op = "load";
              frame_data.g_buffer_data.normal.config.load_op = "load";
            }

            MeshTaskQueue.get().submit_material_indexed_indirect_draws(
              pass,
              frame_data,
              material_id,
              false /* should_reset */
            );
          }
        );
      }
    }

    // Reset mesh task queue
    {
      render_graph.add_pass(
        "reset_mesh_task_queue",
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          frame_data.g_buffer_data.albedo.config.load_op = "clear";
          frame_data.g_buffer_data.smra.config.load_op = "clear";
          frame_data.g_buffer_data.position.config.load_op = "clear";
          frame_data.g_buffer_data.normal.config.load_op = "clear";
          frame_data.g_buffer_data.depth.config.load_op = "clear";

          MeshTaskQueue.get().reset();
        }
      );
    }

    // Lighting Pass
    {
      const shader_setup = {
        pipeline_shaders: {
          vertex: {
            path: "deferred_lighting.wgsl",
          },
          fragment: {
            path: "deferred_lighting.wgsl",
          },
        },
      };

      post_lighting_image = render_graph.create_image({
        name: "post_lighting",
        format: "bgra8unorm",
        width: image_extent.width,
        height: image_extent.height,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        flags: ImageFlags.Transient,
      });

      render_graph.add_pass(
        "lighting_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [
            skybox_image,
            main_albedo_image,
            main_smra_image,
            main_normal_image,
            main_position_image,
            main_depth_image,
            lights,
          ],
          outputs: [post_lighting_image],
          shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.get().draw_quad(frame_data.context, pass);
        }
      );
    }

    // Fullscreen Pass
    {
      const swapchain_image = Texture.create_from_texture(
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
          inputs: [post_lighting_image],
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
