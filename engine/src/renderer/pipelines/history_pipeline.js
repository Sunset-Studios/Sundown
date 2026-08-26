import { RenderPassFlags } from "../renderer_types.js";

const history_lighting_mip_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "reflections/ssr_lighting_mip.wgsl",
    },
  },
};

export class HistoryPipeline {
  add_passes(
    render_graph,
    {
      current_lighting_image,
      current_normal_image,
      current_depth_image,
      prev_lighting_image,
      prev_normal_image,
      prev_depth_image,
      lighting_mip_levels,
    }
  ) {
    this._add_copy_pass(render_graph, {
      current_lighting_image,
      current_normal_image,
      current_depth_image,
      prev_lighting_image,
      prev_normal_image,
      prev_depth_image,
    });

    this._add_lighting_mip_passes(render_graph, prev_lighting_image, lighting_mip_levels);
  }

  _add_copy_pass(
    render_graph,
    {
      current_lighting_image,
      current_normal_image,
      current_depth_image,
      prev_lighting_image,
      prev_normal_image,
      prev_depth_image,
    }
  ) {
    render_graph.add_pass(
      "copy_history",
      RenderPassFlags.GraphLocal,
      {
        inputs: [current_lighting_image, current_normal_image, current_depth_image],
        outputs: [prev_lighting_image, prev_normal_image, prev_depth_image],
      },
      (graph, frame_data, encoder) => {
        const current_lighting = graph.get_physical_image(current_lighting_image);
        const prev_lighting = graph.get_physical_image(prev_lighting_image);
        prev_lighting.copy_texture(encoder, current_lighting);

        const current_normal = graph.get_physical_image(current_normal_image);
        const prev_normal = graph.get_physical_image(prev_normal_image);
        if (prev_normal) {
          prev_normal.copy_texture(encoder, current_normal);
        }

        const current_depth = graph.get_physical_image(current_depth_image);
        const prev_depth = graph.get_physical_image(prev_depth_image);
        if (prev_depth) {
          prev_depth.copy_texture(encoder, current_depth);
        }
      }
    );
  }

  _add_lighting_mip_passes(render_graph, prev_lighting_image, lighting_mip_levels) {
    for (let mip_level = 1; mip_level < lighting_mip_levels; mip_level++) {
      const lighting_mip_params = render_graph.create_buffer({
        name: `prev_lighting_mip_params_${mip_level}`,
        data: [0.0, 0.0, 0.0, 0.0],
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      render_graph.add_pass(
        `prev_lighting_mip_${mip_level}`,
        RenderPassFlags.Compute,
        {
          inputs: [prev_lighting_image, prev_lighting_image, lighting_mip_params],
          outputs: [prev_lighting_image],
          input_views: [mip_level, mip_level + 1],
          shader_setup: history_lighting_mip_shader_setup,
        },
        (graph, frame_data) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const prev_lighting = graph.get_physical_image(prev_lighting_image);
          const params = graph.get_physical_buffer(lighting_mip_params);

          const src_width = Math.max(1, prev_lighting.config.width >> (mip_level - 1));
          const src_height = Math.max(1, prev_lighting.config.height >> (mip_level - 1));
          const dst_width = Math.max(1, prev_lighting.config.width >> mip_level);
          const dst_height = Math.max(1, prev_lighting.config.height >> mip_level);

          params.write([src_width, src_height, dst_width, dst_height]);
          pass.dispatch((dst_width + 7) / 8, (dst_height + 7) / 8, 1);
        }
      );
    }
  }
}
