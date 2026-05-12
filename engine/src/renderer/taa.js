import { Renderer } from "./renderer.js";
import { RenderPassFlags } from "./renderer_types.js";
import { rgba16float_format } from "../utility/config_permutations.js";
import { SharedFrameInfoBuffer, SharedViewBuffer } from "../core/shared_data.js";

const taa_resolve_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "effects/taa_resolve.wgsl",
    },
  },
};

const taa_output_image_config = {
  name: "taa_output",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};

const taa_params_config = {
  name: "taa_params",
  data: [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};

export class TemporalAntiAliasing {
  output_image = null;
  history_valid = false;

  add_passes(
    render_graph,
    width,
    height,
    current_color,
    history_color,
    motion_texture,
    depth_texture,
    prev_depth_texture,
    normal_texture,
    force_recreate = false
  ) {
    if (force_recreate) {
      this.history_valid = false;
    }

    taa_output_image_config.width = width;
    taa_output_image_config.height = height;
    taa_output_image_config.force = force_recreate;
    this.output_image = render_graph.create_image(taa_output_image_config);

    const params = render_graph.create_buffer({
      ...taa_params_config,
      force: force_recreate,
    });
    const history_valid = this.history_valid;

    render_graph.add_pass(
      "taa_resolve",
      RenderPassFlags.Compute,
      {
        inputs: [
          current_color,
          history_color,
          motion_texture,
          depth_texture,
          prev_depth_texture,
          normal_texture,
          this.output_image,
          params,
        ],
        outputs: [this.output_image],
        shader_setup: taa_resolve_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const params_buffer = graph.get_physical_buffer(params);
        const frame_index = SharedFrameInfoBuffer.get_frame_index();
        const resolution = SharedFrameInfoBuffer.frame_info.resolution;
        const current_jitter = SharedViewBuffer.get_temporal_jitter(frame_index, resolution);
        const previous_jitter = SharedViewBuffer.get_temporal_jitter(frame_index - 1, resolution);

        params_buffer.write([
          current_jitter[0],
          current_jitter[1],
          previous_jitter[0],
          previous_jitter[1],
          Renderer.get().get_taa_feedback(),
          history_valid ? 1.0 : 0.0,
        ]);

        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    this.history_valid = true;
  }

  reset_history() {
    this.history_valid = false;
  }
}
