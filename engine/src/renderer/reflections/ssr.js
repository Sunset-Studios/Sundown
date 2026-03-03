import { RenderPassFlags } from "../renderer_types.js";

const ssr_shader_setup = {
  pipeline_shaders: {
    compute: { path: "reflections/ssr.wgsl" },
  },
};

const ssr_output_image_config = {
  name: "ssr_output",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

export class SSR {
  reflection_texture = null;

  add_passes(
    render_graph,
    width,
    height,
    gbuffer_normal,
    gbuffer_position,
    gbuffer_smra,
    skybox_texture,
    force_recreate = false
  ) {
    ssr_output_image_config.width = width;
    ssr_output_image_config.height = height;
    ssr_output_image_config.force = force_recreate;

    this.reflection_texture = render_graph.create_image(ssr_output_image_config);

    render_graph.add_pass(
      "ssr_reflections",
      RenderPassFlags.Compute,
      {
        inputs: [
          gbuffer_normal,
          gbuffer_position,
          gbuffer_smra,
          skybox_texture,
          this.reflection_texture,
        ],
        outputs: [this.reflection_texture],
        shader_setup: ssr_shader_setup,
      },
      (g, fd, encoder) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}
