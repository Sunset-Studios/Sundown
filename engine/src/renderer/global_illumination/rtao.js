import { SharedFrameInfoBuffer } from "../../core/shared_data.js";
import { RenderPassFlags } from "../renderer_types.js";
import { Texture } from "../texture.js";

const COMPUTE_WORKGROUP_SIZE = 128;

const gi_reset_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gi_reset.wgsl" },
  },
};

const rtao_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/rtao_trace_init.wgsl" },
  },
};

const rtao_trace_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/rtao_trace_hit.wgsl" },
  },
};

const rtao_resolve_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/rtao_resolve.wgsl" },
  },
};

const rtao_temporal_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/rtao_temporal.wgsl" },
  },
};

export class RTAO {
  ao_texture = null;
  bent_normal_texture = null;

  config = {
    screen_ray_count: 1,
    upscale_factor: 4,
    max_ray_length: 0.5,
  };

  gi_params_data = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

  constructor(params = {}) {
    this.config = { ...this.config, ...params };
  }

  add_passes(
    render_graph,
    width,
    height,
    gbuffer_position,
    gbuffer_normal,
    gbuffer_albedo,
    gbuffer_smra,
    gbuffer_motion_emissive,
    tlas_bvh2_bounds,
    tlas_bvh_info,
    blas_bvh2_nodes,
    blas_directory,
    entity_transforms,
    index_buffer,
    dense_lights,
    force_recreate = false
  ) {
    const safe_upscale_factor = Math.max(1, Math.floor(this.config.upscale_factor));
    const gi_width = Math.max(1, Math.ceil(width / safe_upscale_factor));
    const gi_height = Math.max(1, Math.ceil(height / safe_upscale_factor));
    const total_pixels = gi_width * gi_height;
    const rays_per_frame = total_pixels * this.config.screen_ray_count;

    const ao_raw = render_graph.create_image({
      name: "rtao_ao_raw",
      format: "r32float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
      force: force_recreate,
    });

    const ao_ping = render_graph.create_image({
      name: "rtao_ao_ping",
      format: "r32float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
      force: force_recreate,
    });

    const ao_pong = render_graph.create_image({
      name: "rtao_ao_pong",
      format: "r32float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
      force: force_recreate,
    });

    const ping_pong_index = SharedFrameInfoBuffer.get_frame_index() % 2;
    const ao_history = ping_pong_index === 0 ? ao_pong : ao_ping;
    const ao_output = ping_pong_index === 0 ? ao_ping : ao_pong;
    this.ao_texture = ao_output;

    this.bent_normal_texture = render_graph.create_image({
      name: "rtao_bent_normal",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const gi_params = render_graph.create_buffer({
      name: "rtao_params",
      size: this.gi_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const gi_counters = render_graph.create_buffer({
      name: "rtao_counters",
      size: 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const pixel_path_state = render_graph.create_buffer({
      name: "rtao_pixel_path_state",
      size: rays_per_frame * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const pixel_ray_queue = render_graph.create_buffer({
      name: "rtao_pixel_ray_queue",
      size: rays_per_frame,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const blue_noise = Texture.default_blue_noise();
    const blue_noise_image = render_graph.register_image(blue_noise.config.name);

    render_graph.add_pass(
      "rtao_prepare",
      RenderPassFlags.GraphLocal,
      { },
      (graph) => {
        const gi_params_buf = graph.get_physical_buffer(gi_params);
        const frame_index = SharedFrameInfoBuffer.get_frame_index();

        this.gi_params_data[0] = this.config.screen_ray_count;
        this.gi_params_data[1] = 0;
        this.gi_params_data[2] = 1;
        this.gi_params_data[3] = total_pixels;
        this.gi_params_data[4] = frame_index;
        this.gi_params_data[5] = 0;
        this.gi_params_data[6] = safe_upscale_factor;
        this.gi_params_data[7] = 0;
        this.gi_params_data[8] = width;
        this.gi_params_data[9] = height;
        this.gi_params_data[10] = gi_width;
        this.gi_params_data[11] = gi_height;
        this.gi_params_data[12] = this.config.max_ray_length;
        gi_params_buf.write_raw(this.gi_params_data);
      }
    );

    render_graph.add_pass(
      "rtao_reset",
      RenderPassFlags.Compute,
      {
        inputs: [gi_counters, dense_lights],
        outputs: [gi_counters],
        shader_setup: gi_reset_shader_setup,
      },
      (g, fd) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(1, 1, 1);
      }
    );

    render_graph.add_pass(
      "rtao_trace_init",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          pixel_path_state,
          pixel_ray_queue,
          gbuffer_position,
          gbuffer_normal,
          blue_noise_image,
        ],
        outputs: [pixel_path_state, pixel_ray_queue],
        shader_setup: rtao_trace_init_shader_setup,
      },
      (g, fd) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "rtao_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          pixel_path_state,
          pixel_ray_queue,
          tlas_bvh2_bounds,
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          entity_transforms,
          index_buffer,
        ],
        outputs: [pixel_path_state],
        shader_setup: rtao_trace_hit_shader_setup,
      },
      (g, fd) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      `rtao_resolve_${ping_pong_index}`,
      RenderPassFlags.Compute,
      {
        inputs: [gi_params, pixel_path_state, gbuffer_normal, ao_raw, this.bent_normal_texture],
        outputs: [ao_raw, this.bent_normal_texture],
        shader_setup: rtao_resolve_shader_setup,
      },
      (g, fd) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      `rtao_temporal_${ping_pong_index}`,
      RenderPassFlags.Compute,
      {
        inputs: [ao_raw, ao_history, gbuffer_motion_emissive, ao_output],
        outputs: [ao_output],
        shader_setup: rtao_temporal_shader_setup,
      },
      (g, fd) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}
