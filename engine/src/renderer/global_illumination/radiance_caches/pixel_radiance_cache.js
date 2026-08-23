import { FragmentGpuBuffer } from "../../../core/ecs/solar/memory.js";
import { Texture } from "../../texture.js";
import { RenderPassFlags } from "../../renderer_types.js";
import {
  register_material_buffers,
  register_scene_lighting_data,
  register_texture_pools,
} from "../../render_graph_utils.js";

const COMPUTE_WORKGROUP_SIZE = 128;
export const PIXEL_RADIANCE_CACHE_DIRECT_OUTPUT_NAME = "pixel_rgb_direct_output";
export const PIXEL_RADIANCE_CACHE_DIFFUSE_OUTPUT_NAME = "pixel_rgb_diffuse_output";
export const PIXEL_RADIANCE_CACHE_SPECULAR_OUTPUT_NAME = "pixel_rgb_specular_output";

const compute_shader = (path) => ({ pipeline_shaders: { compute: { path } } });

/**
 * Per-pixel tracing, hit shading, and temporal/spatial RGB accumulation.
 */
export class PerPixelRadianceCache {
  constructor() {
    this.shader_setups = {
      reset: compute_shader("gi/gi_reset.wgsl"),
      compact_emissive: compute_shader("system_compute/compact_emissive_lights.wgsl"),
      trace_init: compute_shader("gi/pixel_trace_init.wgsl"),
      trace_hit: compute_shader("gi/pixel_trace_hit.wgsl"),
      shade: compute_shader("gi/pixel_trace_shade.wgsl"),
      temporal: compute_shader("gi/pixel_temporal_reservoir.wgsl"),
      spatial_wide: compute_shader("gi/pixel_spatial_reservoir_wide.wgsl"),
      spatial_narrow: compute_shader("gi/pixel_spatial_reservoir_narrow.wgsl"),
      accumulate: compute_shader("gi/pixel_accumulate.wgsl"),
      resolve: compute_shader("gi/pixel_upscale_final.wgsl"),
      atrous: compute_shader("gi/ddgi_atrous_diffuse.wgsl"),
    };
    this.final_diffuse_output_name = PIXEL_RADIANCE_CACHE_DIFFUSE_OUTPUT_NAME;
    this.frame_context = null;
    this.params_data = new Float32Array(12);
  }

  add_passes(render_graph, context, surface_cache) {
    if (!surface_cache) {
      throw new Error("Per-pixel radiance cache requires a surface radiance cache");
    }

    this.frame_context = context;
    this._setup_trace_resources(render_graph, context);
    this._setup_shading_resources(render_graph, context);
    this._setup_accumulation_resources(render_graph, context);
    this._record_trace_passes(render_graph, context);
    this._record_shading_passes(render_graph, context, surface_cache);
    this._record_accumulation_passes(render_graph, context);
  }

  _setup_trace_resources(render_graph, context) {
    const { config, rays_per_frame, force_recreate } = context;
    render_graph.create_buffer({
      name: "pixel_trace_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "pixel_trace_counters",
      size: 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "gi_pixel_path_state",
      size: rays_per_frame * 15 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "gi_pixel_ray_queue",
      size: rays_per_frame,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "pixel_trace_emissive_lights",
      size: 4 + Math.max(1, Math.floor(config.max_emissive_lights)) * 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name);
    render_graph.register_image(Texture.default_blue_noise().config.name);
  }

  _setup_shading_resources(render_graph) {
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
    render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name);
  }

  _setup_accumulation_resources(render_graph, context) {
    const { width, height, gi_width, gi_height, force_recreate } = context;
    const reservoir_size = gi_width * gi_height * 28;
    render_graph.create_buffer({
      name: "gi_temporal_reservoir_0",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "gi_temporal_reservoir_1",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "gi_spatial_reservoir_0",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "gi_spatial_reservoir_1",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "gi_spatial_reservoir_stage",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_low_radiance_direct_0",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_low_radiance_direct_1",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_low_radiance_indirect_diffuse_0",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_low_radiance_indirect_diffuse_1",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_low_radiance_indirect_specular_0",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_low_radiance_indirect_specular_1",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: PIXEL_RADIANCE_CACHE_DIRECT_OUTPUT_NAME,
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: PIXEL_RADIANCE_CACHE_DIFFUSE_OUTPUT_NAME,
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: PIXEL_RADIANCE_CACHE_SPECULAR_OUTPUT_NAME,
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_diffuse_atrous_ping",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "gi_diffuse_atrous_pong",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.atrous_params_data = new Float32Array([1, 0.04, 64.0, 1.0]);
    render_graph.create_buffer({
      name: "gi_diffuse_atrous_params",
      size: this.atrous_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
  }

  _record_trace_passes(render_graph, context) {
    const inputs = context.inputs;
    const params = render_graph.get_resource_handle("pixel_trace_params");
    const counters = render_graph.get_resource_handle("pixel_trace_counters");
    const emissive_lights = render_graph.get_resource_handle("pixel_trace_emissive_lights");
    const entity_index_lookup = render_graph.get_resource_handle(
      FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name
    );
    const path_state = render_graph.get_resource_handle("gi_pixel_path_state");
    const ray_queue = render_graph.get_resource_handle("gi_pixel_ray_queue");
    const materials = this.material_buffers;
    const textures = this.texture_pools;

    render_graph.add_pass("pixel_trace_upload_params", RenderPassFlags.GraphLocal, {}, (graph) => {
      this.params_data[0] = context.config.screen_ray_count;
      this.params_data[1] = context.total_pixels;
      this.params_data[2] = context.frame_index;
      this.params_data[3] = context.config.indirect_boost;
      this.params_data[4] = context.safe_upscale_factor;
      this.params_data[5] = context.width;
      this.params_data[6] = context.height;
      this.params_data[7] = context.gi_width;
      this.params_data[8] = context.gi_height;
      this.params_data[9] = context.config.max_ray_length;
      this.params_data[10] = 0;
      this.params_data[11] = 0;
      graph.get_physical_buffer(params).write_raw(this.params_data);
    });

    render_graph.add_pass(
      "pixel_trace_compact_emissive_lights",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.compact_emissive,
        inputs: [
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_directory,
          inputs.index_buffer,
          inputs.entity_transforms,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          entity_index_lookup,
          emissive_lights,
          textures.albedo,
          textures.emission,
        ],
        outputs: [emissive_lights],
      },
      (graph, frame_data) => {
        graph.get_physical_buffer(emissive_lights).write_raw(new Uint32Array([0, 0, 0, 0]), 0);
        const bounds = graph.get_physical_buffer(inputs.tlas_bvh2_bounds);
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(Math.floor(bounds.config.size / 32) / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );
    render_graph.add_pass(
      "pixel_trace_reset_counters",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.reset,
        inputs: [counters, inputs.dense_lights],
        outputs: [counters],
      },
      (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1)
    );

    render_graph.add_pass(
      `pixel_trace_init_${context.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.trace_init,
        inputs: [
          params,
          counters,
          path_state,
          ray_queue,
          inputs.dense_lights,
          emissive_lights,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_albedo,
          inputs.gbuffer_smra,
          inputs.gbuffer_motion_emissive,
          render_graph.get_resource_handle(Texture.default_blue_noise().config.name),
        ],
        outputs: [path_state, ray_queue],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    render_graph.add_pass(
      "pixel_trace_hits",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.trace_hit,
        inputs: [
          params,
          counters,
          path_state,
          ray_queue,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          inputs.compact_transforms,
          inputs.index_buffer,
          entity_index_lookup,
        ],
        outputs: [path_state],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
  }

  _record_shading_passes(render_graph, context, surface) {
    const materials = this.material_buffers;
    const textures = this.texture_pools;
    const lighting = this.scene_lighting_data;
    const path_state = render_graph.get_resource_handle("gi_pixel_path_state");
    render_graph.add_pass(
      "pixel_rgb_shade_hits",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.shade,
        inputs: [
          render_graph.get_resource_handle("pixel_trace_params"),
          lighting.scene_lighting_buffer,
          path_state,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          render_graph.get_resource_handle("surface_cache_elements"),
          render_graph.get_resource_handle(
            FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name
          ),
          textures.albedo,
          textures.normal,
          textures.roughness,
          textures.metallic,
          textures.ao,
          textures.height,
          textures.specular,
          textures.emission,
          lighting.skybox_image,
          render_graph.get_resource_handle("surface_cache_params"),
          render_graph.get_resource_handle("surface_cache_sh"),
          render_graph.get_resource_handle("surface_cache_hashmap_entries"),
        ],
        outputs: [path_state],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.rays_per_frame / 128), 1, 1)
    );
  }

  _record_accumulation_passes(render_graph, context) {
    const inputs = context.inputs;
    const params = render_graph.get_resource_handle("pixel_trace_params");
    const path_state = render_graph.get_resource_handle("gi_pixel_path_state");
    const frame = context.ping_pong_frame;
    const temporal_prev = render_graph.get_resource_handle(`gi_temporal_reservoir_${frame}`);
    const temporal_curr = render_graph.get_resource_handle(`gi_temporal_reservoir_${1 - frame}`);
    const spatial_curr = render_graph.get_resource_handle(`gi_spatial_reservoir_${1 - frame}`);
    const spatial_stage = render_graph.get_resource_handle("gi_spatial_reservoir_stage");
    const previous = {
      direct: render_graph.get_resource_handle(`gi_low_radiance_direct_${frame}`),
      diffuse: render_graph.get_resource_handle(`gi_low_radiance_indirect_diffuse_${frame}`),
      specular: render_graph.get_resource_handle(`gi_low_radiance_indirect_specular_${frame}`),
    };
    const current = {
      direct: render_graph.get_resource_handle(`gi_low_radiance_direct_${1 - frame}`),
      diffuse: render_graph.get_resource_handle(`gi_low_radiance_indirect_diffuse_${1 - frame}`),
      specular: render_graph.get_resource_handle(`gi_low_radiance_indirect_specular_${1 - frame}`),
    };
    render_graph.add_pass(
      `pixel_rgb_temporal_reservoir_${frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.temporal,
        inputs: [
          params,
          path_state,
          temporal_prev,
          temporal_curr,
          inputs.depth_texture,
          inputs.prev_depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_motion_emissive,
          inputs.gbuffer_normal_prev,
        ],
        outputs: [temporal_curr],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 16), Math.ceil(context.gi_height / 16), 1)
    );
    render_graph.add_pass(
      `pixel_rgb_spatial_reservoir_wide_${frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.spatial_wide,
        inputs: [
          params,
          temporal_curr,
          spatial_stage,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_smra,
        ],
        outputs: [spatial_stage],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 16), Math.ceil(context.gi_height / 16), 1)
    );
    render_graph.add_pass(
      `pixel_rgb_spatial_reservoir_narrow_${frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.spatial_narrow,
        inputs: [
          params,
          spatial_stage,
          spatial_curr,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_smra,
        ],
        outputs: [spatial_curr],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 16), Math.ceil(context.gi_height / 16), 1)
    );
    render_graph.add_pass(
      `pixel_rgb_accumulate_${frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.accumulate,
        inputs: [
          params,
          spatial_curr,
          previous.direct,
          previous.diffuse,
          previous.specular,
          inputs.depth_texture,
          inputs.prev_depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_normal_prev,
          inputs.gbuffer_motion_emissive,
          current.direct,
          current.diffuse,
          current.specular,
        ],
        outputs: [current.direct, current.diffuse, current.specular],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 8), Math.ceil(context.gi_height / 8), 1)
    );

    const direct_output = render_graph.get_resource_handle(PIXEL_RADIANCE_CACHE_DIRECT_OUTPUT_NAME);
    const diffuse_output = render_graph.get_resource_handle(
      PIXEL_RADIANCE_CACHE_DIFFUSE_OUTPUT_NAME
    );
    const specular_output = render_graph.get_resource_handle(
      PIXEL_RADIANCE_CACHE_SPECULAR_OUTPUT_NAME
    );
    render_graph.add_pass(
      `pixel_rgb_resolve_${frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.resolve,
        inputs: [
          params,
          current.direct,
          current.diffuse,
          current.specular,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_smra,
          direct_output,
          diffuse_output,
          specular_output,
        ],
        outputs: [direct_output, diffuse_output, specular_output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );

    let final_diffuse = diffuse_output;
    let read = diffuse_output;
    let write = render_graph.get_resource_handle("gi_diffuse_atrous_ping");
    const pass_count =
      context.config.diffuse_atrous_enabled === false
        ? 0
        : Math.max(0, Math.floor(context.config.diffuse_atrous_pass_count || 0));
    for (let pass_index = 0; pass_index < pass_count; pass_index++) {
      render_graph.add_pass(
        `pixel_rgb_atrous_upload_params_${frame}_${pass_index}`,
        RenderPassFlags.GraphLocal,
        {},
        (graph) => {
          this.atrous_params_data[0] = Math.pow(2, pass_index);
          this.atrous_params_data[1] = Math.max(
            0.0001,
            context.config.diffuse_atrous_phi_depth || 0.04
          );
          this.atrous_params_data[2] = Math.max(1, context.config.diffuse_atrous_phi_normal || 64);
          this.atrous_params_data[3] = Math.max(
            0.0001,
            context.config.diffuse_atrous_luma_sigma || 1
          );
          graph
            .get_physical_buffer(render_graph.get_resource_handle("gi_diffuse_atrous_params"))
            .write_raw(this.atrous_params_data);
        }
      );
      render_graph.add_pass(
        `pixel_rgb_atrous_${frame}_${pass_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: this.shader_setups.atrous,
          inputs: [
            render_graph.get_resource_handle("gi_diffuse_atrous_params"),
            read,
            inputs.depth_texture,
            inputs.gbuffer_normal,
            write,
          ],
          outputs: [write],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
      final_diffuse = write;
      read = write;
      write =
        write === render_graph.get_resource_handle("gi_diffuse_atrous_ping")
          ? render_graph.get_resource_handle("gi_diffuse_atrous_pong")
          : render_graph.get_resource_handle("gi_diffuse_atrous_ping");
    }
    this.final_diffuse_output_name = render_graph.get_resource_config(final_diffuse).name;
  }
}
