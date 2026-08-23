import { SharedFrameInfoBuffer } from "../../../core/shared_data.js";
import { FragmentGpuBuffer } from "../../../core/ecs/solar/memory.js";
import { Texture } from "../../texture.js";
import { Buffer } from "../../buffer.js";
import { DebugDrawType, RenderPassFlags } from "../../renderer_types.js";
import { floor_to_multiple, clamp } from "../../../utility/math.js";
import {
  register_material_buffers,
  register_scene_lighting_data,
  register_texture_pools,
} from "../../render_graph_utils.js";

const COMPUTE_WORKGROUP_SIZE = 128;
const SURFACE_CACHE_COUNTERS_NAME = "surface_cache_counters";
const SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT = 9;
const SURFACE_CACHE_TRACE_DISPATCH_OFFSET = 0;
const SURFACE_CACHE_UPDATE_DISPATCH_OFFSET = 3 * Uint32Array.BYTES_PER_ELEMENT;
const SURFACE_CACHE_BOOTSTRAP_UPDATE_DISPATCH_OFFSET = 6 * Uint32Array.BYTES_PER_ELEMENT;
const SURFACE_CACHE_HIT_WORD_COUNT = 12;
const SURFACE_CACHE_RADIANCE_WORD_COUNT = 16;
const EMPTY_EMISSIVE_LIGHT_HEADER = new Uint32Array(4);
export const SURFACE_CACHE_DIRECT_OUTPUT_NAME = "surface_cache_black_output";
export const SURFACE_CACHE_DIFFUSE_OUTPUT_NAME = "surface_cache_diffuse_output";
export const SURFACE_CACHE_SPECULAR_OUTPUT_NAME = "surface_cache_black_output";

const compute_shader = (path) => ({ pipeline_shaders: { compute: { path } } });

/**
 * Hashed surface tracing, hit shading, and SH accumulation.
 */
export class SurfaceRadianceCache {
  constructor() {
    this.shader_setups = {
      compact_emissive: compute_shader("system_compute/compact_emissive_lights.wgsl"),
      prepare_dispatch: compute_shader("gi/surface_cache_prepare_dispatch.wgsl"),
      feedback: compute_shader("gi/surface_cache_feedback.wgsl"),
      trace_hit: compute_shader("gi/surface_cache_trace_hit.wgsl"),
      shade: compute_shader("gi/surface_cache_trace_shade.wgsl"),
      shadow: compute_shader("gi/surface_cache_trace_shadow.wgsl"),
      accumulate: compute_shader("gi/surface_cache_accumulate.wgsl"),
      accumulate_bootstrap: compute_shader("gi/surface_cache_accumulate_bootstrap.wgsl"),
      resolve: compute_shader("gi/surface_cache_resolve.wgsl"),
      temporal: compute_shader("gi/surface_cache_temporal.wgsl"),
      atrous: compute_shader("gi/surface_cache_atrous.wgsl"),
      debug: compute_shader("gi/surface_cache_debug.wgsl"),
    };
    this.final_diffuse_output_name = SURFACE_CACHE_DIFFUSE_OUTPUT_NAME;
    this.frame_context = null;
    this.params_data = new Float32Array(24);
    this.temporal_params_data = new Float32Array(6);
    this.atrous_params_data = new Float32Array(8);
    this.counters_reset_data = new Uint32Array(11);
    this.counters_data = new Uint32Array(11);
    this.counters_buffer = null;
    this.stats_enabled = false;
  }

  add_passes(render_graph, context) {
    this.frame_context = context;
    this._setup_trace_resources(render_graph, context);
    this._setup_shading_resources(render_graph, context);
    this._setup_accumulation_resources(render_graph, context);
    this._record_trace_passes(render_graph, context);
    this._record_shading_passes(render_graph, context);
    this._record_accumulation_passes(render_graph, context);
  }

  add_debug_passes(render_graph, context) {
    return this._record_accumulation_debug_passes(render_graph, context);
  }

  set_stats_enabled(enabled) {
    const next_enabled = !!enabled;
    if (this.stats_enabled === next_enabled) return;

    this.stats_enabled = next_enabled;
    if (this.counters_buffer) {
      this.counters_buffer.config.own_readback = next_enabled;
    }
  }

  _setup_trace_resources(render_graph, context) {
    const { config, width, height, force_recreate } = context;

    context.total_patches = Math.max(16, floor_to_multiple(config.surface_cache_size, 16));
    context.rays_per_patch = Math.max(Math.floor(config.rays_per_patch ?? 1), 1);
    context.total_ray_count = context.total_patches * context.rays_per_patch;
    context.bootstrap_rays_per_patch = Math.max(
      Math.floor(config.bootstrap_rays_per_patch ?? context.rays_per_patch),
      context.rays_per_patch
    );
    context.bootstrap_patch_capacity = clamp(
      Math.floor(config.bootstrap_patch_capacity ?? 0),
      0,
      context.total_patches
    );
    context.bootstrap_enabled =
      context.bootstrap_patch_capacity > 0 &&
      context.bootstrap_rays_per_patch > context.rays_per_patch;
    context.mature_patch_update_period = Math.max(
      Math.floor(config.mature_patch_update_period ?? 1),
      1
    );
    context.maximum_ray_count_per_frame = clamp(
      Math.floor(config.maximum_ray_count_per_frame ?? context.total_ray_count),
      context.rays_per_patch,
      context.total_ray_count
    );
    // Ray work is hard-capped per frame. Size transient hit and shading data
    // for that actual ceiling so increasing persistent cache headroom does not
    // multiply an unrelated per-ray working set.
    context.ray_buffer_capacity = context.maximum_ray_count_per_frame;

    render_graph.create_buffer({
      name: "surface_cache_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_elements",
      size: context.total_patches * 20,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_hashmap_entries",
      size: context.total_patches * 3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_update_indices",
      size: context.total_patches,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_bootstrap_indices",
      size: Math.max(1, context.bootstrap_patch_capacity),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_hit_info",
      size: context.ray_buffer_capacity * SURFACE_CACHE_HIT_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_emissive_lights",
      size: 4 + Math.max(1, Math.floor(config.max_emissive_lights ?? 32768)) * 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_dispatch_args",
      size: SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
      force: force_recreate,
    });
    render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name);
    render_graph.register_buffer(FragmentGpuBuffer.entity_flags_buffer.buffer.config.name);
    if (this.stats_enabled || this.counters_buffer) {
      const needs_stats_buffer = this.stats_enabled && !this.counters_buffer;
      this._ensure_counters(force_recreate || needs_stats_buffer);

      this.counters_buffer.config.own_readback = this.stats_enabled;
      render_graph.register_buffer(SURFACE_CACHE_COUNTERS_NAME);
    } else {
      render_graph.create_buffer({
        name: SURFACE_CACHE_COUNTERS_NAME,
        size: 11,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });
    }
  }

  _setup_shading_resources(render_graph, context) {
    render_graph.create_buffer({
      name: "surface_cache_radiance_info",
      size: context.ray_buffer_capacity * SURFACE_CACHE_RADIANCE_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
  }

  _setup_accumulation_resources(render_graph, context) {
    const { width, height, total_patches, force_recreate } = context;
    const sh_size = total_patches * 6;

    render_graph.create_image({
      name: SURFACE_CACHE_DIRECT_OUTPUT_NAME,
      format: "rgba16float",
      width: 1,
      height: 1,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: SURFACE_CACHE_DIFFUSE_OUTPUT_NAME,
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_image({
      name: "surface_cache_resolve_aux",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_sh",
      size: sh_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    render_graph.create_buffer({
      name: "surface_cache_temporal_params",
      size: this.temporal_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    if (context.config.screen_reconstruction_enabled !== false) {
      for (let frame = 0; frame < 2; frame++) {
        render_graph.create_image({
          name: `surface_cache_diffuse_history_${frame}`,
          format: "rgba16float",
          width,
          height,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          clear_value: { r: 0, g: 0, b: 0, a: 0 },
          force: force_recreate,
        });
      }
      render_graph.create_image({
        name: "surface_cache_atrous_scratch",
        format: "rgba16float",
        width,
        height,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        force: force_recreate,
      });
      render_graph.create_buffer({
        name: "surface_cache_atrous_params",
        size: this.atrous_params_data.length,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });
    }
  }

  _record_trace_passes(render_graph, context) {
    const { config, width, height, rays_per_patch, total_patches, inputs } = context;
    const params = render_graph.get_resource_handle("surface_cache_params");
    const surface_cache = render_graph.get_resource_handle("surface_cache_elements");
    const hashmap_entries = render_graph.get_resource_handle("surface_cache_hashmap_entries");
    const update_indices = render_graph.get_resource_handle("surface_cache_update_indices");
    const bootstrap_indices = render_graph.get_resource_handle("surface_cache_bootstrap_indices");
    const counters = render_graph.get_resource_handle(SURFACE_CACHE_COUNTERS_NAME);
    const dispatch_args = render_graph.get_resource_handle("surface_cache_dispatch_args");
    const hit_info = render_graph.get_resource_handle("surface_cache_hit_info");
    const entity_index_lookup = render_graph.get_resource_handle(
      FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name
    );
    const entity_flags = render_graph.get_resource_handle(
      FragmentGpuBuffer.entity_flags_buffer.buffer.config.name
    );

    render_graph.add_pass(
      "surface_cache_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph) => {
        this.params_data[0] = total_patches;
        this.params_data[1] = total_patches;
        this.params_data[2] = width;
        this.params_data[3] = height;
        this.params_data[4] = SharedFrameInfoBuffer.get_frame_index();
        this.params_data[5] = config.max_ray_length;
        this.params_data[6] = config.history_hysteresis;
        this.params_data[7] = config.max_history_samples;
        this.params_data[8] = config.indirect_boost;
        this.params_data[9] = context.rays_per_patch;
        this.params_data[10] = config.cache_entry_lifetime;
        this.params_data[11] = context.bootstrap_rays_per_patch;
        this.params_data[12] = clamp(config.bootstrap_ray_budget_fraction ?? 0.5, 0.0, 1.0);
        this.params_data[13] = Math.max(config.cache_pixel_footprint ?? 3, 1);
        this.params_data[14] = clamp(Math.floor(config.hash_search_count ?? 10), 1, 64);
        this.params_data[15] = Math.max(config.cache_normal_bias ?? 0, 0);
        this.params_data[16] = Math.max(config.history_footprint_start_samples ?? 4, 0);
        this.params_data[17] = Math.max(
          config.history_footprint_end_samples ?? 32,
          this.params_data[16] + 1
        );
        this.params_data[18] = Math.max(config.history_footprint_max_scale ?? 1, 1);
        this.params_data[19] = context.bootstrap_enabled ? context.bootstrap_patch_capacity : 0;
        this.params_data[20] = context.mature_patch_update_period;
        this.params_data[21] = context.maximum_ray_count_per_frame;
        this.params_data[22] = clamp(config.native_promotion_start_confidence ?? 0.3, 0.0, 0.99);
        this.params_data[23] = clamp(
          config.native_promotion_end_confidence ?? 0.85,
          this.params_data[22] + 0.01,
          1.0
        );
        graph.get_physical_buffer(params).write_raw(this.params_data);
        this.counters_reset_data.fill(0);
        graph.get_physical_buffer(counters).write_raw(this.counters_reset_data);
      }
    );

    render_graph.add_pass(
      "surface_cache_feedback",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.feedback,
        inputs: [
          params,
          surface_cache,
          counters,
          update_indices,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          hashmap_entries,
          bootstrap_indices,
        ],
        outputs: [surface_cache, counters, update_indices, hashmap_entries, bootstrap_indices],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    render_graph.add_pass(
      "surface_cache_prepare_dispatch",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.prepare_dispatch,
        inputs: [params, counters, dispatch_args],
        outputs: [counters, dispatch_args],
      },
      (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1)
    );

    render_graph.add_pass(
      "surface_cache_trace_hits",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.trace_hit,
        inputs: [
          params,
          surface_cache,
          update_indices,
          bootstrap_indices,
          counters,
          hit_info,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          inputs.compact_transforms,
          inputs.index_buffer,
          entity_index_lookup,
          dispatch_args,
        ],
        outputs: [hit_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_TRACE_DISPATCH_OFFSET
          )
    );
  }

  _record_shading_passes(render_graph, context) {
    const params = render_graph.get_resource_handle("surface_cache_params");
    const surface_cache = render_graph.get_resource_handle("surface_cache_elements");
    const hashmap_entries = render_graph.get_resource_handle("surface_cache_hashmap_entries");
    const update_indices = render_graph.get_resource_handle("surface_cache_update_indices");
    const bootstrap_indices = render_graph.get_resource_handle("surface_cache_bootstrap_indices");
    const counters = render_graph.get_resource_handle(SURFACE_CACHE_COUNTERS_NAME);
    const dispatch_args = render_graph.get_resource_handle("surface_cache_dispatch_args");
    const hit_info = render_graph.get_resource_handle("surface_cache_hit_info");
    const emissive_lights = render_graph.get_resource_handle("surface_cache_emissive_lights");
    const entity_index_lookup = render_graph.get_resource_handle(
      FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name
    );
    const radiance_info = render_graph.get_resource_handle("surface_cache_radiance_info");
    const sh = render_graph.get_resource_handle("surface_cache_sh");
    const { inputs } = context;
    const materials = this.material_buffers;
    const textures = this.texture_pools;
    const lighting = this.scene_lighting_data;

    render_graph.add_pass(
      "surface_cache_compact_emissive_lights",
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
        const bounds_buffer = graph.get_physical_buffer(inputs.tlas_bvh2_bounds);
        graph.get_physical_buffer(emissive_lights).write_raw(EMPTY_EMISSIVE_LIGHT_HEADER, 0);
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(Math.floor(bounds_buffer.config.size / 32) / COMPUTE_WORKGROUP_SIZE),
            1,
            1
          );
      }
    );

    render_graph.add_pass(
      "surface_cache_shade_hits",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.shade,
        inputs: [
          params,
          lighting.scene_lighting_buffer,
          surface_cache,
          sh,
          update_indices,
          bootstrap_indices,
          counters,
          hit_info,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          inputs.compact_transforms,
          inputs.dense_lights,
          emissive_lights,
          textures.albedo,
          textures.normal,
          textures.emission,
          lighting.skybox_image,
          radiance_info,
          hashmap_entries,
          dispatch_args,
        ],
        outputs: [radiance_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_TRACE_DISPATCH_OFFSET
          )
    );
    render_graph.add_pass(
      "surface_cache_trace_shadows",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.shadow,
        inputs: [
          params,
          counters,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          inputs.compact_transforms,
          inputs.index_buffer,
          entity_index_lookup,
          radiance_info,
          dispatch_args,
        ],
        outputs: [radiance_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_TRACE_DISPATCH_OFFSET
          )
    );
  }

  _record_accumulation_passes(render_graph, context) {
    const params = render_graph.get_resource_handle("surface_cache_params");
    const surface_cache = render_graph.get_resource_handle("surface_cache_elements");
    const hashmap_entries = render_graph.get_resource_handle("surface_cache_hashmap_entries");
    const update_indices = render_graph.get_resource_handle("surface_cache_update_indices");
    const bootstrap_indices = render_graph.get_resource_handle("surface_cache_bootstrap_indices");
    const counters = render_graph.get_resource_handle(SURFACE_CACHE_COUNTERS_NAME);
    const dispatch_args = render_graph.get_resource_handle("surface_cache_dispatch_args");
    const hit_info = render_graph.get_resource_handle("surface_cache_hit_info");
    const radiance_info = render_graph.get_resource_handle("surface_cache_radiance_info");
    const sh = render_graph.get_resource_handle("surface_cache_sh");
    const direct = render_graph.get_resource_handle(SURFACE_CACHE_DIRECT_OUTPUT_NAME);
    const diffuse = render_graph.get_resource_handle(SURFACE_CACHE_DIFFUSE_OUTPUT_NAME);
    const resolve_aux = render_graph.get_resource_handle("surface_cache_resolve_aux");
    const { width, height, inputs } = context;

    render_graph.add_pass(
      "surface_cache_sh_accumulate",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.accumulate,
        inputs: [
          params,
          surface_cache,
          sh,
          update_indices,
          bootstrap_indices,
          counters,
          hit_info,
          radiance_info,
          dispatch_args,
        ],
        outputs: [surface_cache, sh],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_UPDATE_DISPATCH_OFFSET
          )
    );
    render_graph.add_pass(
      "surface_cache_bootstrap_sh_accumulate",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.accumulate_bootstrap,
        inputs: [
          params,
          surface_cache,
          sh,
          bootstrap_indices,
          counters,
          hit_info,
          radiance_info,
          dispatch_args,
        ],
        outputs: [surface_cache, sh],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_BOOTSTRAP_UPDATE_DISPATCH_OFFSET
          )
    );
    render_graph.add_pass(
      "surface_cache_sh_resolve",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.resolve,
        inputs: [
          params,
          surface_cache,
          sh,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          diffuse,
          direct,
          hashmap_entries,
          resolve_aux,
        ],
        outputs: [diffuse, direct, resolve_aux],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    if (context.config.screen_reconstruction_enabled === false) {
      this.final_diffuse_output_name = SURFACE_CACHE_DIFFUSE_OUTPUT_NAME;
      return;
    }

    const history_frame = SharedFrameInfoBuffer.get_frame_index() & 1;
    const history_prev = render_graph.get_resource_handle(
      `surface_cache_diffuse_history_${history_frame}`
    );
    const history_curr = render_graph.get_resource_handle(
      `surface_cache_diffuse_history_${1 - history_frame}`
    );

    const temporal_params = render_graph.get_resource_handle("surface_cache_temporal_params");
    const temporal_response = Math.fround(clamp(context.config.temporal_response ?? 0.02, 0.0, 1));
    const temporal_max_history_frames = Math.max(
      1,
      Math.floor(context.config.temporal_max_history_frames ?? 64)
    );
    const temporal_depth_threshold = Math.fround(
      Math.max(context.config.temporal_depth_threshold ?? 0.03, 0.0001)
    );
    const temporal_normal_threshold = Math.fround(
      clamp(context.config.temporal_normal_threshold ?? 0.9, 0, 0.9999)
    );
    const spatial_filter_radius = clamp(
      Math.floor((context.config.cache_pixel_footprint ?? 3) * 0.5),
      1,
      8
    );
    const temporal_params_need_upload =
      context.force_recreate ||
      this.temporal_params_data[0] !== temporal_response ||
      this.temporal_params_data[1] !== temporal_max_history_frames ||
      this.temporal_params_data[2] !== temporal_depth_threshold ||
      this.temporal_params_data[3] !== temporal_normal_threshold ||
      this.temporal_params_data[4] !== spatial_filter_radius;
    if (temporal_params_need_upload) {
      this.temporal_params_data[0] = temporal_response;
      this.temporal_params_data[1] = temporal_max_history_frames;
      this.temporal_params_data[2] = temporal_depth_threshold;
      this.temporal_params_data[3] = temporal_normal_threshold;
      this.temporal_params_data[4] = spatial_filter_radius;
      this.temporal_params_data[5] = 0;
      render_graph.add_pass(
        "surface_cache_temporal_upload_params",
        RenderPassFlags.GraphLocal,
        {},
        (graph) => {
          graph.get_physical_buffer(temporal_params).write_raw(this.temporal_params_data);
        }
      );
    }
    render_graph.add_pass(
      `surface_cache_temporal_${history_frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.temporal,
        inputs: [
          temporal_params,
          diffuse,
          history_prev,
          inputs.depth_texture,
          inputs.prev_depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_normal_prev,
          inputs.gbuffer_motion_emissive,
          history_curr,
        ],
        outputs: [history_curr],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    const atrous_enabled = context.config.disocclusion_atrous_enabled !== false;
    const requested_atrous_pass_count = atrous_enabled
      ? clamp(Math.floor(context.config.disocclusion_atrous_pass_count ?? 4), 0, 6)
      : 0;
    // One scratch texture alternates with the current history. An even pass
    // count leaves the filtered result in history_curr for next-frame reuse.
    const atrous_pass_count =
      requested_atrous_pass_count > 0
        ? requested_atrous_pass_count + (requested_atrous_pass_count & 1)
        : 0;
    const atrous_scratch = render_graph.get_resource_handle("surface_cache_atrous_scratch");
    const atrous_params = render_graph.get_resource_handle("surface_cache_atrous_params");
    let atrous_read = history_curr;
    let atrous_write = atrous_scratch;
    for (let pass_index = 0; pass_index < atrous_pass_count; pass_index++) {
      render_graph.add_pass(
        `surface_cache_atrous_upload_${history_frame}_${pass_index}`,
        RenderPassFlags.GraphLocal,
        {},
        (graph) => {
          this.atrous_params_data[0] = 1 << pass_index;
          this.atrous_params_data[1] = Math.max(
            context.config.disocclusion_atrous_phi_depth ?? 0.015,
            0.0001
          );
          this.atrous_params_data[2] = Math.max(
            context.config.disocclusion_atrous_phi_normal ?? 64.0,
            1.0
          );
          this.atrous_params_data[3] = Math.max(
            context.config.disocclusion_atrous_luma_sigma ?? 2.0,
            0.01
          );
          this.atrous_params_data[4] = clamp(
            context.config.disocclusion_atrous_confidence_threshold ?? 0.85,
            0.0,
            1.0
          );
          this.atrous_params_data[5] = Math.max(
            Math.floor(context.config.disocclusion_atrous_history_frames ?? 12),
            1
          );
          this.atrous_params_data[6] = 0;
          this.atrous_params_data[7] = 0;
          graph.get_physical_buffer(atrous_params).write_raw(this.atrous_params_data);
        }
      );
      render_graph.add_pass(
        `surface_cache_atrous_${history_frame}_${pass_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: this.shader_setups.atrous,
          inputs: [
            atrous_params,
            atrous_read,
            resolve_aux,
            inputs.depth_texture,
            inputs.gbuffer_normal,
            atrous_write,
          ],
          outputs: [atrous_write],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
      );
      const previous_read = atrous_read;
      atrous_read = atrous_write;
      atrous_write = previous_read;
    }
    this.final_diffuse_output_name = render_graph.get_resource_config(history_curr).name;
  }

  _record_accumulation_debug_passes(render_graph, context) {
    if (context.debug_view !== DebugDrawType.GI_SurfaceCache) return null;

    const output = render_graph.create_image({
      name: "surface_cache_debug_output",
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: context.force_recreate,
    });
    render_graph.add_pass(
      "surface_cache_debug",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.debug,
        inputs: [
          render_graph.get_resource_handle("surface_cache_params"),
          render_graph.get_resource_handle("surface_cache_elements"),
          render_graph.get_resource_handle("surface_cache_sh"),
          context.inputs.depth_texture,
          context.inputs.gbuffer_normal,
          context.inputs.scene_color,
          output,
          render_graph.get_resource_handle("surface_cache_hashmap_entries"),
        ],
        outputs: [output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );
    return output;
  }

  _ensure_counters(force_recreate) {
    if (!force_recreate && this.counters_buffer) {
      return;
    }

    if (this.counters_buffer) {
      this.counters_buffer.destroy();
    }

    this.counters_buffer = Buffer.create({
      name: SURFACE_CACHE_COUNTERS_NAME,
      raw_data: this.counters_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
      force: force_recreate,
    });
  }

  get_stats(context = this.frame_context) {
    if (!this.counters_data || !context?.total_patches) return null;

    const active_patch_count = Math.min(this.counters_data[0] || 0, context.total_patches);
    const update_patch_count = Math.min(this.counters_data[1] || 0, active_patch_count);
    const bootstrap_patch_count = Math.min(
      this.counters_data[2] || 0,
      context.bootstrap_patch_capacity
    );
    const available_bootstrap_patch_count = Math.min(
      this.counters_data[6] || bootstrap_patch_count,
      context.bootstrap_patch_capacity
    );
    const bootstrap_rays_per_patch =
      bootstrap_patch_count > 0
        ? clamp(
            this.counters_data[3] || context.rays_per_patch,
            1,
            context.bootstrap_rays_per_patch
          )
        : 0;
    const regular_rays_per_patch =
      update_patch_count > 0 ? clamp(this.counters_data[10] || 1, 1, context.rays_per_patch) : 0;
    const deferred_patch_count = Math.max(
      active_patch_count - bootstrap_patch_count - update_patch_count,
      0
    );
    const surface_cache_bytes = context.total_patches * 20 * 4;
    const hashmap_bytes = context.total_patches * 3 * 4;
    const sh_bytes = context.total_patches * 6 * 4;
    const ray_working_set_bytes =
      context.ray_buffer_capacity *
      (SURFACE_CACHE_HIT_WORD_COUNT + SURFACE_CACHE_RADIANCE_WORD_COUNT) *
      4;
    const emissive_light_bytes =
      (4 + Math.max(1, Math.floor(context.config.max_emissive_lights ?? 32768)) * 12) * 4;
    const scheduling_bytes =
      this.params_data.byteLength +
      (context.total_patches + context.bootstrap_patch_capacity) * 4 +
      this.counters_data.byteLength +
      SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT;
    const output_texture_count = context.config.screen_reconstruction_enabled === false ? 2 : 5;
    const output_bytes = context.width * context.height * 8 * output_texture_count + 8;

    return {
      strategy: "scgi",
      width: context.width,
      height: context.height,
      total_patch_count: context.total_patches,
      active_patch_count,
      update_patch_count,
      bootstrap_patch_count,
      available_bootstrap_patch_count,
      pending_bootstrap_patch_count: Math.max(
        available_bootstrap_patch_count - bootstrap_patch_count,
        0
      ),
      deferred_patch_count,
      rays_per_patch: regular_rays_per_patch,
      maximum_regular_rays_per_patch: context.rays_per_patch,
      bootstrap_rays_per_patch,
      maximum_bootstrap_rays_per_patch: context.bootstrap_rays_per_patch,
      bootstrap_ray_budget_fraction: clamp(
        context.config.bootstrap_ray_budget_fraction ?? 0.5,
        0.0,
        1.0
      ),
      bootstrap_patch_capacity: context.bootstrap_patch_capacity,
      total_rays_fired:
        update_patch_count * regular_rays_per_patch +
        bootstrap_patch_count * bootstrap_rays_per_patch,
      active_set_ray_budget: active_patch_count * context.rays_per_patch,
      maximum_ray_count_per_frame: context.maximum_ray_count_per_frame,
      maximum_ray_count: context.total_ray_count,
      max_ray_length: context.config.max_ray_length,
      cache_entry_lifetime: context.config.cache_entry_lifetime,
      cache_pixel_footprint: context.config.cache_pixel_footprint,
      history_footprint_start_samples: context.config.history_footprint_start_samples ?? 0,
      history_footprint_end_samples: context.config.history_footprint_end_samples ?? 0,
      history_footprint_max_scale: context.config.history_footprint_max_scale ?? 1,
      cache_normal_bias: context.config.cache_normal_bias,
      hash_search_count: context.config.hash_search_count,
      history_hysteresis: context.config.history_hysteresis,
      max_history_samples: context.config.max_history_samples,
      mature_patch_update_period: context.mature_patch_update_period,
      feedback_miss_count: this.counters_data[9] || 0,
      surface_cache_bytes,
      hashmap_bytes,
      sh_bytes,
      ray_working_set_bytes,
      emissive_light_bytes,
      scheduling_bytes,
      output_bytes,
      total_memory_bytes:
        surface_cache_bytes +
        hashmap_bytes +
        sh_bytes +
        ray_working_set_bytes +
        emissive_light_bytes +
        scheduling_bytes +
        output_bytes,
    };
  }
}
