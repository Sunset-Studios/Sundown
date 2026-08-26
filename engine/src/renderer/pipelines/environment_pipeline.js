import { SharedEnvironmentData } from "../../core/shared_data.js";
import { RenderPassFlags } from "../renderer_types.js";
import {
  rgba16float_format,
} from "../../utility/config_permutations.js";
import { draw_cube } from "../draw_helpers.js";

const skybox_shader_setup = {
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

export class EnvironmentPipeline {
  add_skybox_pass(
    render_graph,
    {
      pass_name,
      image_extent,
      force_recreate = false,
      output_name = "skybox_output",
    }
  ) {
    const skybox_image = render_graph.create_image({
      name: output_name,
      format: rgba16float_format,
      width: image_extent.width,
      height: image_extent.height,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const skydome_data = SharedEnvironmentData.get_skydome_data();
    const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

    const skybox = SharedEnvironmentData.get_skybox();
    const skybox_texture = render_graph.register_image(skybox.config.name);

    render_graph.add_pass(
      pass_name,
      RenderPassFlags.Graphics,
      {
        inputs: [skybox_texture, skydome_data_buffer],
        outputs: [skybox_image],
        shader_setup: skybox_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        draw_cube(pass);
      }
    );

    return skybox_image;
  }
}
