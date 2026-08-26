import { SharedFrameInfoBuffer } from "../../../core/shared_data.js";
import { FragmentGpuBuffer } from "../../../core/ecs/solar/memory.js";
import { Buffer } from "../../buffer.js";
import { DebugDrawType, RenderPassFlags } from "../../renderer_types.js";
import { floor_to_multiple, clamp } from "../../../utility/math.js";
import {
  register_material_buffers,
  register_scene_lighting_data,
  register_texture_pools,
} from "../../render_graph_utils.js";

export const SURFACE_CACHE_DIRECT_OUTPUT_NAME = "surface_cache_black_output";
export const SURFACE_CACHE_DIFFUSE_OUTPUT_NAME = "surface_cache_diffuse_output";
export const SURFACE_CACHE_SPECULAR_OUTPUT_NAME = "surface_cache_black_output";

const SURFACE_CACHE_COUNTERS_NAME = "surface_cache_counters";
const SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT = 9;
const SURFACE_CACHE_TRACE_DISPATCH_OFFSET = 0;
const SURFACE_CACHE_UPDATE_DISPATCH_OFFSET = 3 * Uint32Array.BYTES_PER_ELEMENT;
const SURFACE_CACHE_BOOTSTRAP_UPDATE_DISPATCH_OFFSET = 6 * Uint32Array.BYTES_PER_ELEMENT;
const SURFACE_CACHE_HIT_WORD_COUNT = 12;
const SURFACE_CACHE_RADIANCE_WORD_COUNT = 16;
const compute_shader = (path) => ({ pipeline_shaders: { compute: { path } } });

/**
 * Hashed surface tracing, hit shading, and SH accumulation.
 */
export class SurfaceRadianceCache {
  constructor() {
    this.shader_setups = {
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
    this.surface_cache_resources = {};
  }

  add_passes(render_graph, context) {
    this._prepare_context(context);
    this._create_surface_cache_resources(render_graph, context);

    this._record_trace_passes(render_graph, context);
    this._record_shading_passes(render_graph, context);
    this._record_sh_accumulation_passes(render_graph, context);
    this._record_temporal_passes(render_graph, context);
    this._record_atrous_passes(render_graph, context);

    this.final_diffuse_output_name = render_graph.get_resource_config(this.surface_cache_resources.history_curr).name;
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

  _prepare_context(context) {
    context.total_patches = Math.max(16, floor_to_multiple(context.config.surface_cache_size, 16));
    context.rays_per_patch = Math.max(Math.floor(context.config.rays_per_patch ?? 1), 1);
    context.total_ray_count = context.total_patches * context.rays_per_patch;
    context.bootstrap_rays_per_patch = Math.max(
      Math.floor(context.config.bootstrap_rays_per_patch ?? context.rays_per_patch),
      context.rays_per_patch
    );
    context.bootstrap_patch_capacity = clamp(
      Math.floor(context.config.bootstrap_patch_capacity ?? 0),
      0,
      context.total_patches
    );
    context.bootstrap_enabled =
      context.bootstrap_patch_capacity > 0 &&
      context.bootstrap_rays_per_patch > context.rays_per_patch;
    context.mature_patch_update_period = Math.max(
      Math.floor(context.config.mature_patch_update_period ?? 1),
      1
    );
    context.maximum_ray_count_per_frame = clamp(
      Math.floor(context.config.maximum_ray_count_per_frame ?? context.total_ray_count),
      context.rays_per_patch,
      context.total_ray_count
    );
    // Ray work is hard-capped per frame. Size transient hit and shading data
    // for that actual ceiling so increasing persistent cache headroom does not
    // multiply an unrelated per-ray working set.
    context.ray_buffer_capacity = context.maximum_ray_count_per_frame;
    context.history_frame = SharedFrameInfoBuffer.get_frame_index() & 1;

    this.frame_context = context;
  }

  _create_surface_cache_resources(render_graph, context) {
    const resources = this.surface_cache_resources;

    resources.params = render_graph.create_buffer({
      name: "surface_cache_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.surface_cache = render_graph.create_buffer({
      name: "surface_cache_elements",
      size: context.total_patches * 20,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.hashmap_entries = render_graph.create_buffer({
      name: "surface_cache_hashmap_entries",
      size: context.total_patches * 3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.update_indices = render_graph.create_buffer({
      name: "surface_cache_update_indices",
      size: context.total_patches,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.bootstrap_indices = render_graph.create_buffer({
      name: "surface_cache_bootstrap_indices",
      size: Math.max(1, context.bootstrap_patch_capacity),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.hit_info = render_graph.create_buffer({
      name: "surface_cache_hit_info",
      size: context.ray_buffer_capacity * SURFACE_CACHE_HIT_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.dispatch_args = render_graph.create_buffer({
      name: "surface_cache_dispatch_args",
      size: SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
      force: context.force_recreate,
    });

    const needs_stats_buffer =
      context.force_recreate || (this.stats_enabled && !this.counters_buffer);
    this._ensure_counters(needs_stats_buffer);
    resources.counters = render_graph.register_buffer(SURFACE_CACHE_COUNTERS_NAME);
    resources.entity_index_lookup = render_graph.register_buffer(
      FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name
    );
    resources.entity_flags = render_graph.register_buffer(
      FragmentGpuBuffer.entity_flags_buffer.buffer.config.name
    );

    resources.radiance_info = render_graph.create_buffer({
      name: "surface_cache_radiance_info",
      size: context.ray_buffer_capacity * SURFACE_CACHE_RADIANCE_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.emissive_lights = context.inputs.emissive_lights;
    resources.sh = render_graph.create_buffer({
      name: "surface_cache_sh",
      size: context.total_patches * 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.material_buffers = register_material_buffers(render_graph);
    resources.texture_pools = register_texture_pools(render_graph);
    resources.scene_lighting_data = register_scene_lighting_data(render_graph);

    resources.direct = render_graph.create_image({
      name: SURFACE_CACHE_DIRECT_OUTPUT_NAME,
      format: "rgba16float",
      width: 1,
      height: 1,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: context.force_recreate,
    });
    resources.diffuse = render_graph.create_image({
      name: SURFACE_CACHE_DIFFUSE_OUTPUT_NAME,
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: context.force_recreate,
    });
    resources.resolve_aux = render_graph.create_image({
      name: "surface_cache_resolve_aux",
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: context.force_recreate,
    });
    resources.history_prev = render_graph.create_image({
      name: `surface_cache_diffuse_history_${context.history_frame}`,
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      clear_value: { r: 0, g: 0, b: 0, a: 0 },
      force: context.force_recreate,
    });
    resources.history_curr = render_graph.create_image({
      name: `surface_cache_diffuse_history_${1 - context.history_frame}`,
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      clear_value: { r: 0, g: 0, b: 0, a: 0 },
      force: context.force_recreate,
    });
    resources.atrous_scratch = render_graph.create_image({
      name: "surface_cache_atrous_scratch",
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: context.force_recreate,
    });
    resources.temporal_params = render_graph.create_buffer({
      name: "surface_cache_temporal_params",
      size: this.temporal_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.atrous_params = render_graph.create_buffer({
      name: "surface_cache_atrous_params",
      size: this.atrous_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    resources.debug_output =
      context.debug_view === DebugDrawType.GI_SurfaceCache
        ? render_graph.create_image({
          name: "surface_cache_debug_output",
          format: "rgba16float",
          width: context.width,
          height: context.height,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          force: context.force_recreate,
        })
        : null;
    return resources;
  }

  _record_trace_passes(render_graph, context) {
    const {
      params,
      surface_cache,
      hashmap_entries,
      update_indices,
      bootstrap_indices,
      hit_info,
      dispatch_args,
      entity_index_lookup,
      counters,
    } = this.surface_cache_resources;

    render_graph.add_pass(
      "surface_cache_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph) => {
        this.params_data[0] = context.total_patches;
        this.params_data[1] = context.total_patches;
        this.params_data[2] = context.width;
        this.params_data[3] = context.height;
        this.params_data[4] = SharedFrameInfoBuffer.get_frame_index();
        this.params_data[5] = context.config.max_ray_length;
        this.params_data[6] = context.config.history_hysteresis;
        this.params_data[7] = context.config.max_history_samples;
        this.params_data[8] = context.config.indirect_boost;
        this.params_data[9] = context.rays_per_patch;
        this.params_data[10] = context.config.cache_entry_lifetime;
        this.params_data[11] = context.bootstrap_rays_per_patch;
        this.params_data[12] = clamp(context.config.bootstrap_ray_budget_fraction ?? 0.5, 0.0, 1.0);
        this.params_data[13] = Math.max(context.config.cache_pixel_footprint ?? 3, 1);
        this.params_data[14] = clamp(Math.floor(context.config.hash_search_count ?? 10), 1, 64);
        this.params_data[15] = Math.max(context.config.cache_normal_bias ?? 0, 0);
        this.params_data[16] = Math.max(context.config.history_footprint_start_samples ?? 4, 0);
        this.params_data[17] = Math.max(
          context.config.history_footprint_end_samples ?? 32,
          this.params_data[16] + 1
        );
        this.params_data[18] = Math.max(context.config.history_footprint_max_scale ?? 1, 1);
        this.params_data[19] = context.bootstrap_enabled ? context.bootstrap_patch_capacity : 0;
        this.params_data[20] = context.mature_patch_update_period;
        this.params_data[21] = context.maximum_ray_count_per_frame;
        this.params_data[22] = clamp(
          context.config.native_promotion_start_confidence ?? 0.3,
          0.0,
          0.99
        );
        this.params_data[23] = clamp(
          context.config.native_promotion_end_confidence ?? 0.85,
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
          context.inputs.depth_texture,
          context.inputs.gbuffer_normal,
          hashmap_entries,
          bootstrap_indices,
        ],
        outputs: [surface_cache, counters, update_indices, hashmap_entries, bootstrap_indices],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
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
          context.inputs.tlas_bvh2_bounds,
          context.inputs.tlas_bvh_info,
          context.inputs.blas_bvh2_nodes,
          context.inputs.blas_directory,
          context.inputs.compact_transforms,
          context.inputs.index_buffer,
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
    const {
      params,
      surface_cache,
      hashmap_entries,
      update_indices,
      bootstrap_indices,
      counters,
      dispatch_args,
      hit_info,
      entity_index_lookup,
      radiance_info,
      emissive_lights,
      sh,
      material_buffers,
      texture_pools,
      scene_lighting_data,
    } = this.surface_cache_resources;

    render_graph.add_pass(
      "surface_cache_shade_hits",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.shade,
        inputs: [
          params,
          scene_lighting_data.scene_lighting_buffer,
          surface_cache,
          sh,
          update_indices,
          bootstrap_indices,
          counters,
          hit_info,
          material_buffers.params_gpu_buffer,
          material_buffers.material_offsets_buffer,
          material_buffers.material_palette_buffer,
          context.inputs.compact_transforms,
          context.inputs.dense_lights,
          emissive_lights,
          texture_pools.albedo,
          texture_pools.normal,
          texture_pools.emission,
          scene_lighting_data.skybox_image,
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
          context.inputs.tlas_bvh2_bounds,
          context.inputs.tlas_bvh_info,
          context.inputs.blas_bvh2_nodes,
          context.inputs.blas_directory,
          context.inputs.compact_transforms,
          context.inputs.index_buffer,
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

  _record_sh_accumulation_passes(render_graph, context) {
    const {
      params,
      surface_cache,
      hashmap_entries,
      update_indices,
      bootstrap_indices,
      counters,
      dispatch_args,
      hit_info,
      radiance_info,
      sh,
      direct,
      diffuse,
      resolve_aux,
    } = this.surface_cache_resources;

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
          context.inputs.depth_texture,
          context.inputs.gbuffer_normal,
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
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );
  }

  _record_temporal_passes(render_graph, context) {
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
    const { diffuse, history_prev, history_curr, temporal_params } = this.surface_cache_resources;

    const temporal_params_need_upload =
      context.force_recreate ||
      this.temporal_params_data[0] !== temporal_response ||
      this.temporal_params_data[1] !== temporal_max_history_frames ||
      this.temporal_params_data[2] !== temporal_depth_threshold ||
      this.temporal_params_data[3] !== temporal_normal_threshold ||
      this.temporal_params_data[4] !== spatial_filter_radius;

    if (temporal_params_need_upload) {
      render_graph.add_pass(
        "surface_cache_temporal_upload_params",
        RenderPassFlags.GraphLocal,
        {},
        (graph) => {
          this.temporal_params_data[0] = temporal_response;
          this.temporal_params_data[1] = temporal_max_history_frames;
          this.temporal_params_data[2] = temporal_depth_threshold;
          this.temporal_params_data[3] = temporal_normal_threshold;
          this.temporal_params_data[4] = spatial_filter_radius;
          this.temporal_params_data[5] = 0;
          graph.get_physical_buffer(temporal_params).write_raw(this.temporal_params_data);
        }
      );
    }

    render_graph.add_pass(
      `surface_cache_temporal_${context.history_frame}`,
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.temporal,
        inputs: [
          temporal_params,
          diffuse,
          history_prev,
          context.inputs.depth_texture,
          context.inputs.prev_depth_texture,
          context.inputs.gbuffer_normal,
          context.inputs.gbuffer_normal_prev,
          context.inputs.gbuffer_motion_emissive,
          history_curr,
        ],
        outputs: [history_curr],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );
  }

  _record_atrous_passes(render_graph, context, resources) {
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
    const { atrous_params, history_curr, atrous_scratch, resolve_aux } = this.surface_cache_resources;

    let atrous_read = history_curr;
    let atrous_write = atrous_scratch;
    for (let pass_index = 0; pass_index < atrous_pass_count; pass_index++) {
      render_graph.add_pass(
        `surface_cache_atrous_upload_${context.history_frame}_${pass_index}`,
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
        `surface_cache_atrous_${context.history_frame}_${pass_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: this.shader_setups.atrous,
          inputs: [
            atrous_params,
            atrous_read,
            resolve_aux,
            context.inputs.depth_texture,
            context.inputs.gbuffer_normal,
            atrous_write,
          ],
          outputs: [atrous_write],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
      const previous_read = atrous_read;
      atrous_read = atrous_write;
      atrous_write = previous_read;
    }
  }

  _record_accumulation_debug_passes(render_graph, context) {
    if (context.debug_view !== DebugDrawType.GI_SurfaceCache) return null;

    const { params, surface_cache, sh, hashmap_entries, debug_output } =
      this.surface_cache_resources;

    render_graph.add_pass(
      "surface_cache_debug",
      RenderPassFlags.Compute,
      {
        shader_setup: this.shader_setups.debug,
        inputs: [
          params,
          surface_cache,
          sh,
          context.inputs.depth_texture,
          context.inputs.gbuffer_normal,
          context.inputs.scene_color,
          debug_output,
          hashmap_entries,
        ],
        outputs: [debug_output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );

    return debug_output;
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
      cpu_readback: this.stats_enabled,
      own_readback: this.stats_enabled,
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
    const scheduling_bytes =
      this.params_data.byteLength +
      (context.total_patches + context.bootstrap_patch_capacity) * 4 +
      this.counters_data.byteLength +
      SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT;
    const output_bytes = context.width * context.height * 8 * 5 + 8;

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
      scheduling_bytes,
      output_bytes,
      total_memory_bytes:
        surface_cache_bytes +
        hashmap_bytes +
        sh_bytes +
        ray_working_set_bytes +
        scheduling_bytes +
        output_bytes,
    };
  }
}
