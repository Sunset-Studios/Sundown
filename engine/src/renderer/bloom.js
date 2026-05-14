import { RenderPassFlags } from "./renderer_types.js";
import { rgba16float_format } from "../utility/config_permutations.js";
import { draw_quad } from "./draw_helpers.js";

const bloom_downsample_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "effects/bloom_downsample.wgsl",
    },
  },
};

const bloom_downsample_first_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "effects/bloom_downsample.wgsl",
      defines: {
        HIGH_QUALITY_DOWNSAMPLE: true,
      },
    },
  },
};

const bloom_blit_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "effects/bloom_blit.wgsl",
    },
  },
};

const bloom_upsample_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "effects/bloom_upsample.wgsl",
    },
  },
};

const bloom_resolve_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "fullscreen.wgsl",
    },
    fragment: {
      path: "effects/bloom_resolve.wgsl",
    },
  },
};

const bloom_pass_params_config = {
  name: "bloom_pass_params",
  data: Array(16).fill(0.0),
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};

const post_bloom_color_image_config = {
  name: "post_bloom_color",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  force: false,
};

const bloom_resolve_pass_name = "bloom_resolve_pass";
const max_bloom_steps = 16;

export class Bloom {
  config = {
    exposure: 1.1,
    intensity: 0.05,
    threshold: 0.1,
    knee: 0.2,
    clamp: 50.0,
    radius: 3.0,
    color: [1.0, 1.0, 1.0],
  };

  output_image = null;
  debug_bloom_image = null;

  constructor(config = {}) {
    Object.assign(this.config, config);
  }

  add_passes(render_graph, width, height, post_lighting_image_desc, force_recreate = false) {
    post_bloom_color_image_config.width = width;
    post_bloom_color_image_config.height = height;
    post_bloom_color_image_config.force = force_recreate;
    this.output_image = render_graph.create_image(post_bloom_color_image_config);

    const full_width = Math.max(1, width);
    const full_height = Math.max(1, height);
    const min_dimension = Math.max(1, Math.min(full_width, full_height));
    const max_iter = this.config.radius - 8.0 + Math.log2(min_dimension);
    const max_iter_int = Math.floor(max_iter);
    const num_iterations = Math.min(Math.max(max_iter_int, 1), max_bloom_steps);
    const bloom_sample_scale = 0.5 + max_iter - max_iter_int;
    const curve_threshold = [
      this.config.threshold - this.config.knee,
      this.config.knee * 2.0,
      0.25 / Math.max(1e-5, this.config.knee),
      this.config.threshold,
    ];
    const bloom_color = this.config.color.map((c) => c * this.config.intensity);

    const bloom_blit = render_graph.create_image({
      name: "bloom_blit",
      format: rgba16float_format,
      width: full_width,
      height: full_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    const bloom_blit_params = render_graph.create_buffer({
      ...bloom_pass_params_config,
      name: "bloom_blit_params",
      force: force_recreate,
    });

    render_graph.add_pass(
      "bloom_blit_pass",
      RenderPassFlags.Compute,
      {
        inputs: [post_lighting_image_desc, bloom_blit, bloom_blit_params],
        outputs: [bloom_blit],
        shader_setup: bloom_blit_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const bloom_params = graph.get_physical_buffer(bloom_blit_params);

        bloom_params.write([
          1.0 / full_width,
          1.0 / full_height,
          bloom_sample_scale,
          this.config.clamp,
          ...curve_threshold,
          bloom_color[0],
          bloom_color[1],
          bloom_color[2],
          this.config.exposure,
          full_width,
          full_height,
          0.0,
          0.0,
        ]);

        pass.dispatch((full_width + 15) / 16, (full_height + 15) / 16, 1);
      }
    );

    let bloom_downsample_chain = [];
    let bloom_downsample_sizes = [];
    let bloom_downsample_params_chain = [];
    let down_width = full_width;
    let down_height = full_height;

    for (let i = 0; i < num_iterations; i++) {
      down_width = Math.max(2, Math.floor(down_width / 2));
      down_height = Math.max(2, Math.floor(down_height / 2));

      bloom_downsample_sizes.push([down_width, down_height]);
      bloom_downsample_chain.push(
        render_graph.create_image({
          name: `bloom_downsample_${i}`,
          format: rgba16float_format,
          width: down_width,
          height: down_height,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          force: force_recreate,
        })
      );
      bloom_downsample_params_chain.push(
        render_graph.create_buffer({
          ...bloom_pass_params_config,
          name: `bloom_downsample_params_${i}`,
          force: force_recreate,
        })
      );
    }

    for (let i = 0; i < num_iterations; i++) {
      const src_width = i === 0 ? full_width : bloom_downsample_sizes[i - 1][0];
      const src_height = i === 0 ? full_height : bloom_downsample_sizes[i - 1][1];
      const dst_width = bloom_downsample_sizes[i][0];
      const dst_height = bloom_downsample_sizes[i][1];
      const shader_setup =
        i === 0 ? bloom_downsample_first_shader_setup : bloom_downsample_shader_setup;

      render_graph.add_pass(
        `bloom_downsample_pass_${i}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            i === 0 ? bloom_blit : bloom_downsample_chain[i - 1],
            bloom_downsample_chain[i],
            bloom_downsample_params_chain[i],
          ],
          outputs: [bloom_downsample_chain[i]],
          shader_setup,
        },
        (graph, frame_data) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const bloom_params = graph.get_physical_buffer(bloom_downsample_params_chain[i]);

          bloom_params.write([
            1.0 / src_width,
            1.0 / src_height,
            bloom_sample_scale,
            0.0,
            ...curve_threshold,
            bloom_color[0],
            bloom_color[1],
            bloom_color[2],
            this.config.exposure,
            dst_width,
            dst_height,
            0.0,
            0.0,
          ]);

          pass.dispatch((dst_width + 15) / 16, (dst_height + 15) / 16, 1);
        }
      );
    }

    let bloom_upsample_chain = [];
    let bloom_upsample_params_chain = [];
    for (let i = 0; i < num_iterations - 1; i++) {
      const [upsample_width, upsample_height] = bloom_downsample_sizes[i];
      bloom_upsample_chain.push(
        render_graph.create_image({
          name: `bloom_upsample_${i}`,
          format: rgba16float_format,
          width: upsample_width,
          height: upsample_height,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          force: force_recreate,
        })
      );
      bloom_upsample_params_chain.push(
        render_graph.create_buffer({
          ...bloom_pass_params_config,
          name: `bloom_upsample_params_${i}`,
          force: force_recreate,
        })
      );
    }

    let bloom_resolve_input = bloom_downsample_chain[num_iterations - 1];
    let bloom_resolve_input_size = bloom_downsample_sizes[num_iterations - 1];

    for (let i = num_iterations - 2; i >= 0; --i) {
      const source_width = bloom_resolve_input_size[0];
      const source_height = bloom_resolve_input_size[1];
      const dst_width = bloom_downsample_sizes[i][0];
      const dst_height = bloom_downsample_sizes[i][1];
      const upsample_output = bloom_upsample_chain[i];
      const upsample_params = bloom_upsample_params_chain[i];

      render_graph.add_pass(
        `bloom_upsample_pass_${i}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            bloom_resolve_input,
            bloom_downsample_chain[i],
            upsample_output,
            upsample_params,
          ],
          outputs: [upsample_output],
          shader_setup: bloom_upsample_shader_setup,
        },
        (graph, frame_data) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const bloom_params = graph.get_physical_buffer(upsample_params);

          bloom_params.write([
            1.0 / source_width,
            1.0 / source_height,
            bloom_sample_scale,
            0.0,
            ...curve_threshold,
            bloom_color[0],
            bloom_color[1],
            bloom_color[2],
            this.config.exposure,
            dst_width,
            dst_height,
            0.0,
            0.0,
          ]);

          pass.dispatch((dst_width + 15) / 16, (dst_height + 15) / 16, 1);
        }
      );

      bloom_resolve_input = upsample_output;
      bloom_resolve_input_size = bloom_downsample_sizes[i];
    }

    this.debug_bloom_image = bloom_resolve_input;

    const bloom_resolve_params_desc = render_graph.create_buffer({
      ...bloom_pass_params_config,
      name: "bloom_resolve_params",
      force: force_recreate,
    });

    render_graph.add_pass(
      bloom_resolve_pass_name,
      RenderPassFlags.Graphics,
      {
        inputs: [post_lighting_image_desc, bloom_resolve_input, bloom_resolve_params_desc],
        outputs: [this.output_image],
        shader_setup: bloom_resolve_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const bloom_params = graph.get_physical_buffer(bloom_resolve_params_desc);

        bloom_params.write([
          1.0 / bloom_resolve_input_size[0],
          1.0 / bloom_resolve_input_size[1],
          bloom_sample_scale,
          0.0,
          ...curve_threshold,
          bloom_color[0],
          bloom_color[1],
          bloom_color[2],
          this.config.exposure,
          0.0,
          0.0,
          0.0,
          0.0,
        ]);

        draw_quad(pass);
      }
    );
  }
}
