import { RenderPassFlags } from "../renderer_types.js";

const ssr_trace_shader_setup = {
  pipeline_shaders: {
    compute: { path: "reflections/ssr.wgsl" },
  },
};

const ssr_resolve_shader_setup = {
  pipeline_shaders: {
    compute: { path: "reflections/ssr_resolve.wgsl" },
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

const ssr_trace_image_config = {
  name: "ssr_trace",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const ssr_history_0_image_config = {
  name: "ssr_history_0",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const ssr_history_1_image_config = {
  name: "ssr_history_1",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

export class SSR {
  reflection_texture = null;
  frame_index = 0;

  add_passes(
    render_graph,
    width,
    height,
    gbuffer_normal,
    gbuffer_position,
    prev_gbuffer_normal,
    prev_gbuffer_position,
    gbuffer_motion_emissive,
    gbuffer_smra,
    lighting_history_texture,
    hzb_texture,
    force_recreate = false
  ) {
    ssr_output_image_config.width = width;
    ssr_output_image_config.height = height;
    ssr_output_image_config.force = force_recreate;

    ssr_trace_image_config.width = width;
    ssr_trace_image_config.height = height;
    ssr_trace_image_config.force = force_recreate;

    ssr_history_0_image_config.width = width;
    ssr_history_0_image_config.height = height;
    ssr_history_0_image_config.force = force_recreate;

    ssr_history_1_image_config.width = width;
    ssr_history_1_image_config.height = height;
    ssr_history_1_image_config.force = force_recreate;

    this.reflection_texture = render_graph.create_image(ssr_output_image_config);
    const ssr_trace_texture = render_graph.create_image(ssr_trace_image_config);
    const ssr_history_0 = render_graph.create_image(ssr_history_0_image_config);
    const ssr_history_1 = render_graph.create_image(ssr_history_1_image_config);

    const ping_pong_frame = this.frame_index % 2;
    const ssr_history_prev = ping_pong_frame === 0 ? ssr_history_0 : ssr_history_1;
    const ssr_history_curr = ping_pong_frame === 0 ? ssr_history_1 : ssr_history_0;

    render_graph.add_pass(
      `ssr_trace_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gbuffer_normal,
          gbuffer_position,
          gbuffer_smra,
          lighting_history_texture,
          hzb_texture,
          ssr_trace_texture,
        ],
        outputs: [ssr_trace_texture],
        shader_setup: ssr_trace_shader_setup,
      },
      (g, fd, encoder) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      `ssr_resolve_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          ssr_trace_texture,
          ssr_history_prev,
          gbuffer_normal,
          gbuffer_position,
          prev_gbuffer_normal,
          prev_gbuffer_position,
          gbuffer_motion_emissive,
          gbuffer_smra,
          lighting_history_texture,
          this.reflection_texture,
          ssr_history_curr,
        ],
        outputs: [this.reflection_texture, ssr_history_curr],
        shader_setup: ssr_resolve_shader_setup,
      },
      (g, fd, encoder) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    this.frame_index++;
  }
}
