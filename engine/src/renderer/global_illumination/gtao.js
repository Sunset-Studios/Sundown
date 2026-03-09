import { RenderPassFlags } from "../renderer_types.js";

const gtao_trace_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gtao.wgsl" },
  },
};

const gtao_temporal_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gtao_temporal.wgsl" },
  },
};

const gtao_denoise_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gtao_bilateral.wgsl" },
  },
};

const ao_raw_image_config = {
  name: "gtao_ao_raw",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};

const ao_temporal_image_config = {
  name: "gtao_ao_temporal",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};

const ao_filter_ping_image_config = {
  name: "gtao_ao_filter_ping",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};

const ao_history_image_config = {
  name: "gtao_ao_history",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};

const bent_raw_image_config = {
  name: "gtao_bent_raw",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const bent_temporal_image_config = {
  name: "gtao_bent_temporal",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const bent_filter_ping_image_config = {
  name: "gtao_bent_filter_ping",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const bent_history_image_config = {
  name: "gtao_bent_history",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const trace_params_buffer_config = {
  name: "gtao_trace_params",
  size: 8,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  force: false,
};

const temporal_params_buffer_config = {
  name: "gtao_temporal_params",
  size: 4,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  force: false,
};

const denoise_x_params_buffer_config = {
  name: "gtao_denoise_x_params",
  size: 8,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  force: false,
};

const denoise_y_params_buffer_config = {
  name: "gtao_denoise_y_params",
  size: 8,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  force: false,
};

export class GTAO {
  constructor(params = {}) {
    this.config = {
      radius: 0.5,
      bias: 0.02,
      sample_count: 6,
      max_radius_px: 16,
      thickness: 0.15,
      temporal_response: 0.14,
      denoise_radius: 4,
      denoise_position_sigma: 0.2,
      denoise_normal_power: 12,
      denoise_ao_sigma: 0.15,
      ...params,
    };

    this.ao_texture = null;
    this.ao_blur_texture = null;
    this.bent_normal_texture = null;

    this.trace_params_data = new Float32Array(8);
    this.temporal_params_data = new Float32Array(4);
    this.denoise_x_params_data = new Float32Array(8);
    this.denoise_y_params_data = new Float32Array(8);
  }

  add_passes(
    render_graph,
    width,
    height,
    gbuffer_position,
    prev_gbuffer_position,
    gbuffer_normal,
    prev_gbuffer_normal,
    gbuffer_albedo,
    gbuffer_smra,
    gbuffer_motion_emissive,
    depth_image,
    hzb_texture,
    tlas_bvh2_bounds,
    tlas_bvh_info,
    blas_bvh2_nodes,
    blas_directory,
    entity_transforms,
    index_buffer,
    dense_lights,
    force_recreate = false
  ) {
    ao_raw_image_config.width = width;
    ao_raw_image_config.height = height;
    ao_raw_image_config.force = force_recreate;

    ao_temporal_image_config.width = width;
    ao_temporal_image_config.height = height;
    ao_temporal_image_config.force = force_recreate;

    ao_filter_ping_image_config.width = width;
    ao_filter_ping_image_config.height = height;
    ao_filter_ping_image_config.force = force_recreate;

    ao_history_image_config.width = width;
    ao_history_image_config.height = height;
    ao_history_image_config.force = force_recreate;

    bent_raw_image_config.width = width;
    bent_raw_image_config.height = height;
    bent_raw_image_config.force = force_recreate;

    bent_temporal_image_config.width = width;
    bent_temporal_image_config.height = height;
    bent_temporal_image_config.force = force_recreate;

    bent_filter_ping_image_config.width = width;
    bent_filter_ping_image_config.height = height;
    bent_filter_ping_image_config.force = force_recreate;

    bent_history_image_config.width = width;
    bent_history_image_config.height = height;
    bent_history_image_config.force = force_recreate;

    trace_params_buffer_config.force = force_recreate;
    temporal_params_buffer_config.force = force_recreate;
    denoise_x_params_buffer_config.force = force_recreate;
    denoise_y_params_buffer_config.force = force_recreate;

    const ao_raw = render_graph.create_image(ao_raw_image_config);
    const ao_temporal = render_graph.create_image(ao_temporal_image_config);
    const ao_filter_ping = render_graph.create_image(ao_filter_ping_image_config);
    const ao_history = render_graph.create_image(ao_history_image_config);

    const bent_raw = render_graph.create_image(bent_raw_image_config);
    const bent_temporal = render_graph.create_image(bent_temporal_image_config);
    const bent_filter_ping = render_graph.create_image(bent_filter_ping_image_config);
    const bent_history = render_graph.create_image(bent_history_image_config);

    const trace_params = render_graph.create_buffer(trace_params_buffer_config);
    const temporal_params = render_graph.create_buffer(temporal_params_buffer_config);
    const denoise_x_params = render_graph.create_buffer(denoise_x_params_buffer_config);
    const denoise_y_params = render_graph.create_buffer(denoise_y_params_buffer_config);

    this.ao_texture = ao_history;
    this.ao_blur_texture = ao_history;
    this.bent_normal_texture = bent_history;

    render_graph.add_pass("gtao_prepare", RenderPassFlags.GraphLocal, {}, (graph) => {
      const trace_params_buffer = graph.get_physical_buffer(trace_params);
      const temporal_params_buffer = graph.get_physical_buffer(temporal_params);
      const denoise_x_params_buffer = graph.get_physical_buffer(denoise_x_params);
      const denoise_y_params_buffer = graph.get_physical_buffer(denoise_y_params);

      this.trace_params_data[0] = this.config.radius;
      this.trace_params_data[1] = this.config.bias;
      this.trace_params_data[2] = this.config.sample_count;
      this.trace_params_data[3] = this.config.max_radius_px;
      this.trace_params_data[4] = this.config.thickness;
      this.trace_params_data[5] = 0.0;
      this.trace_params_data[6] = 0.0;
      this.trace_params_data[7] = 0.0;
      trace_params_buffer.write_raw(this.trace_params_data);

      this.temporal_params_data[0] = this.config.temporal_response;
      this.temporal_params_data[1] = this.config.radius;
      this.temporal_params_data[2] = 0.0;
      this.temporal_params_data[3] = 0.0;
      temporal_params_buffer.write_raw(this.temporal_params_data);

      this.denoise_x_params_data[0] = 1.0;
      this.denoise_x_params_data[1] = 0.0;
      this.denoise_x_params_data[2] = this.config.denoise_radius;
      this.denoise_x_params_data[3] = this.config.denoise_position_sigma;
      this.denoise_x_params_data[4] = this.config.denoise_normal_power;
      this.denoise_x_params_data[5] = this.config.denoise_ao_sigma;
      this.denoise_x_params_data[6] = 0.0;
      this.denoise_x_params_data[7] = 0.0;
      denoise_x_params_buffer.write_raw(this.denoise_x_params_data);

      this.denoise_y_params_data[0] = 0.0;
      this.denoise_y_params_data[1] = 1.0;
      this.denoise_y_params_data[2] = this.config.denoise_radius;
      this.denoise_y_params_data[3] = this.config.denoise_position_sigma;
      this.denoise_y_params_data[4] = this.config.denoise_normal_power;
      this.denoise_y_params_data[5] = this.config.denoise_ao_sigma;
      this.denoise_y_params_data[6] = 0.0;
      this.denoise_y_params_data[7] = 0.0;
      denoise_y_params_buffer.write_raw(this.denoise_y_params_data);
    });

    render_graph.add_pass(
      "gtao_trace",
      RenderPassFlags.Compute,
      {
        inputs: [gbuffer_normal, depth_image, hzb_texture, ao_raw, bent_raw, trace_params],
        outputs: [ao_raw, bent_raw],
        shader_setup: gtao_trace_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      "gtao_temporal",
      RenderPassFlags.Compute,
      {
        inputs: [
          ao_raw,
          bent_raw,
          ao_history,
          bent_history,
          gbuffer_position,
          prev_gbuffer_position,
          gbuffer_normal,
          prev_gbuffer_normal,
          gbuffer_motion_emissive,
          ao_temporal,
          bent_temporal,
          temporal_params,
        ],
        outputs: [ao_temporal, bent_temporal],
        shader_setup: gtao_temporal_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      "gtao_denoise_x",
      RenderPassFlags.Compute,
      {
        inputs: [
          gbuffer_position,
          gbuffer_normal,
          ao_temporal,
          bent_temporal,
          ao_filter_ping,
          bent_filter_ping,
          denoise_x_params,
        ],
        outputs: [ao_filter_ping, bent_filter_ping],
        shader_setup: gtao_denoise_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      "gtao_denoise_y",
      RenderPassFlags.Compute,
      {
        inputs: [
          gbuffer_position,
          gbuffer_normal,
          ao_filter_ping,
          bent_filter_ping,
          ao_history,
          bent_history,
          denoise_y_params,
        ],
        outputs: [ao_history, bent_history],
        shader_setup: gtao_denoise_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}
