import { RenderPassFlags } from "../renderer_types.js";

const vbao_trace_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/vbao.wgsl" },
  },
};

const vbao_resolve_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/vbao_resolve.wgsl" },
  },
};

const ao_raw_image_config = {
  name: "vbao_ao_raw",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};

const ao_temporal_image_config = {
  name: "vbao_ao_temporal",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};

const ao_history_image_config = {
  name: "vbao_ao_history",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};

const vbao_settings_buffer_config = {
  name: "vbao_settings",
  size: 32,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  force: false,
};

export class VBAO {
  config = {
    trace_downsample: 2,
    radius: 16.0,
    strength: 1.1,
    bias: 0.001,
    slice_count: 2,
    sample_count: 8,
    thickness: 0.12,
    temporal_response: 0.06,
  };

  constructor(params = {}) {
    Object.assign(this.config, params);

    this.ao_texture = null;
    this.ao_blur_texture = null;
    this.bent_normal_texture = null;

    this.settings_data = new Float32Array(8);
    this.history_valid = false;
  }

  add_passes(
    render_graph,
    width,
    height,
    gbuffer_normal,
    prev_gbuffer_normal,
    gbuffer_albedo,
    gbuffer_smra,
    gbuffer_motion_emissive,
    depth_image,
    prev_depth_image,
    hzb_texture,
    tlas_bvh2_bounds,
    tlas_bvh_info,
    blas_bvh2_nodes,
    blas_directory,
    entity_transforms,
    _compact_transforms,
    index_buffer,
    dense_lights,
    force_recreate = false
  ) {
    if (force_recreate) {
      this.history_valid = false;
    }

    const trace_width = Math.max(1, Math.ceil(width * (1.0 / this.config.trace_downsample)));
    const trace_height = Math.max(1, Math.ceil(height * (1.0 / this.config.trace_downsample)));

    ao_raw_image_config.width = trace_width;
    ao_raw_image_config.height = trace_height;
    ao_raw_image_config.force = force_recreate;

    ao_temporal_image_config.width = width;
    ao_temporal_image_config.height = height;
    ao_temporal_image_config.force = force_recreate;

    ao_history_image_config.width = width;
    ao_history_image_config.height = height;
    ao_history_image_config.force = force_recreate;

    vbao_settings_buffer_config.force = force_recreate;

    const ao_raw = render_graph.create_image(ao_raw_image_config);
    const ao_temporal = render_graph.create_image(ao_temporal_image_config);
    const ao_history = render_graph.create_image(ao_history_image_config);
    const vbao_settings = render_graph.create_buffer(vbao_settings_buffer_config);

    this.ao_texture = ao_temporal;
    this.ao_blur_texture = ao_temporal;
    this.bent_normal_texture = gbuffer_normal;

    render_graph.add_pass(
      "vbao_prepare",
      RenderPassFlags.GraphLocal,
      {},
      (graph) => {
        const settings_buffer = graph.get_physical_buffer(vbao_settings);

        this.settings_data[0] = this.config.radius;
        this.settings_data[1] = this.config.strength;
        this.settings_data[2] = this.config.bias;
        this.settings_data[3] = this.config.slice_count;
        this.settings_data[4] = this.config.sample_count;
        this.settings_data[5] = this.config.thickness;
        this.settings_data[6] = this.config.temporal_response;
        this.settings_data[7] = this.history_valid ? 1.0 : 0.0;
        settings_buffer.write_raw(this.settings_data);
      }
    );

    render_graph.add_pass(
      "vbao_trace",
      RenderPassFlags.Compute,
      {
        inputs: [gbuffer_normal, depth_image, ao_raw, vbao_settings],
        outputs: [ao_raw],
        shader_setup: vbao_trace_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(trace_width / 8), Math.ceil(trace_height / 8), 1);
      }
    );

    render_graph.add_pass(
      "vbao_resolve_temporal",
      RenderPassFlags.Compute,
      {
        inputs: [
          ao_raw,
          ao_history,
          gbuffer_motion_emissive,
          depth_image,
          prev_depth_image,
          gbuffer_normal,
          prev_gbuffer_normal,
          ao_temporal,
          vbao_settings,
        ],
        outputs: [ao_temporal],
        shader_setup: vbao_resolve_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      "vbao_store_history",
      RenderPassFlags.GraphLocal,
      {
        inputs: [ao_temporal],
        outputs: [ao_history],
      },
      (graph, frame_data, encoder) => {
        const curr_ao = graph.get_physical_image(ao_temporal);
        const history_ao = graph.get_physical_image(ao_history);
        history_ao.copy_texture(encoder, curr_ao);
        this.history_valid = true;
      }
    );
  }
}
