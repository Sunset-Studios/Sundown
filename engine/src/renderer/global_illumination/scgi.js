import { DebugDrawType, RenderPassFlags, CacheTypes } from "../renderer_types.js";
import { SharedFrameInfoBuffer } from "../../core/shared_data.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import { ResourceCache } from "../resource_cache.js";
import { Name } from "../../utility/names.js";
import { Texture } from "../texture.js";
import {
  register_material_buffers,
  register_texture_pools,
  register_scene_lighting_data
} from "../render_graph_utils.js";

const COMPUTE_WORKGROUP_SIZE = 128;
const SURFACE_PATCH_STRIDE_WORDS = 24;
const SURFACE_PATCH_SH_STRIDE_WORDS = 6;
const SCGI_HIT_INFO_STRIDE_WORDS = 28;
const SCGI_RADIANCE_INFO_STRIDE_WORDS = 8;
const transforms_name = "transforms";

const scgi_evict_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_evict.wgsl" } } };
const scgi_surface_feedback_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_surface_feedback.wgsl" } } };
const scgi_trace_hit_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_trace_hit.wgsl" } } };
const scgi_trace_shade_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_trace_shade.wgsl" } } };
const scgi_trace_shadow_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_trace_shadow.wgsl" } } };
const scgi_update_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_update.wgsl" } } };
const scgi_filter_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_filter.wgsl" } } };
const scgi_resolve_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_resolve.wgsl" } } };
const scgi_cache_debug_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_cache_debug.wgsl" } } };
const scgi_transform_prepare_shader_setup = { pipeline_shaders: { compute: { path: "gi/scgi_ray_instance_transform_prepare.wgsl" } } };

/**
 * Surface Cache Global Illumination.
 *
 * Visible G-buffer surfaces allocate persistent world-space patches in a
 * bucketed position/normal/LOD hash. Expired patches are evicted first, then
 * depth feedback directly appends visible patches to the active trace stream.
 * Traced radiance is accumulated, filtered in surface space, and resolved.
 */
export class SCGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;

  config = {
    surface_cache_size: 32768,
    surface_cache_cell_size: 0.25,
    surface_cache_lod_count: 4,
    cache_entry_lifetime: 1,
    max_ray_length: 1024.0,
    history_hysteresis: 0.995,
    max_history_samples: 128,
    indirect_boost: 1.0,
  };

  params_data = new Float32Array(16);

  constructor(params = {}) {
    this.config = { ...this.config, ...params };
  }

  add_passes(
    render_graph,
    width,
    height,
    depth_texture,
    _prev_depth_texture,
    gbuffer_normal,
    _gbuffer_normal_prev,
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
    draw_count,
    _hzb_texture,
    force_recreate = false
  ) {
    if (draw_count <= 0) return;

    const frame_index = SharedFrameInfoBuffer.get_frame_index();
    const cache_size = Math.max(16, Math.floor(this.config.surface_cache_size / 16) * 16);
    const lod_count = Math.max(1, Math.floor(this.config.surface_cache_lod_count));
    const total_patches = cache_size * lod_count;

    this.final_gi_texture_direct = render_graph.create_image({
      name: "scgi_final_direct",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.final_gi_texture_indirect_diffuse = render_graph.create_image({
      name: "scgi_final_indirect_diffuse",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.final_gi_texture_indirect_specular = render_graph.create_image({
      name: "scgi_final_indirect_specular",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const scgi_params = render_graph.create_buffer({
      name: "scgi_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const sh_buffer_size = total_patches * SURFACE_PATCH_SH_STRIDE_WORDS;
    const surface_cache = render_graph.create_buffer({
      name: "scgi_surface_cache",
      size: total_patches * SURFACE_PATCH_STRIDE_WORDS,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const surface_cache_sh = render_graph.create_buffer({
      name: "scgi_surface_cache_sh",
      size: sh_buffer_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const surface_cache_sh_filtered = render_graph.create_buffer({
      name: "scgi_surface_cache_sh_filtered",
      size: sh_buffer_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const active_indices = render_graph.create_buffer({
      name: "scgi_active_patch_indices",
      size: total_patches,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const counters = render_graph.create_buffer({
      name: "scgi_counters",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const hit_info = render_graph.create_buffer({
      name: "scgi_hit_info",
      size: total_patches * SCGI_HIT_INFO_STRIDE_WORDS,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const radiance_info = render_graph.create_buffer({
      name: "scgi_radiance_info",
      size: total_patches * SCGI_RADIANCE_INFO_STRIDE_WORDS,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const entity_transform_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      transforms_name
    );
    const entity_transform_count = entity_transform_buffer.max_rows;

    const ray_instance_transforms = render_graph.create_buffer({
      name: "scgi_ray_instance_transforms",
      size: entity_transform_count * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const entity_index_lookup = render_graph.register_buffer(
      FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name
    );

    const material_buffers = register_material_buffers(render_graph);
    const texture_pools = register_texture_pools(render_graph);
    const scene_lighting_data = register_scene_lighting_data(render_graph);

    render_graph.add_pass(
      "scgi_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data) => {
      this.params_data[0] = cache_size;
      this.params_data[1] = this.config.surface_cache_cell_size;
      this.params_data[2] = lod_count;
      this.params_data[3] = total_patches;
      this.params_data[4] = width;
      this.params_data[5] = height;
      this.params_data[6] = frame_index;
      this.params_data[7] = this.config.max_ray_length;
      this.params_data[8] = this.config.history_hysteresis;
      this.params_data[9] = this.config.max_history_samples;
      this.params_data[10] = this.config.indirect_boost;
      this.params_data[11] = 0;
      this.params_data[12] = this.config.cache_entry_lifetime;
      this.params_data[13] = 0;
      this.params_data[14] = 0;
      this.params_data[15] = 0;
      graph.get_physical_buffer(scgi_params).write_raw(this.params_data);
    });

    render_graph.add_pass(
      "scgi_prepare_ray_instance_transforms",
      RenderPassFlags.Compute,
      {
        inputs: [entity_transforms, ray_instance_transforms],
        outputs: [ray_instance_transforms],
        shader_setup: scgi_transform_prepare_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(entity_transform_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );

    render_graph.add_pass(
      "scgi_evict",
      RenderPassFlags.Compute,
      {
        inputs: [scgi_params, surface_cache, surface_cache_sh, surface_cache_sh_filtered, counters],
        outputs: [surface_cache, surface_cache_sh, surface_cache_sh_filtered, counters],
        shader_setup: scgi_evict_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(total_patches / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );

    render_graph.add_pass(
      "scgi_surface_feedback",
      RenderPassFlags.Compute,
      {
        inputs: [
          scgi_params,
          surface_cache,
          surface_cache_sh,
          counters,
          active_indices,
          depth_texture,
          gbuffer_normal,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive
        ],
        outputs: [surface_cache, surface_cache_sh, counters, active_indices],
        shader_setup: scgi_surface_feedback_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    render_graph.add_pass(
      "scgi_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          scgi_params,
          surface_cache,
          active_indices,
          counters,
          hit_info,
          tlas_bvh2_bounds,
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          ray_instance_transforms,
          index_buffer,
          entity_index_lookup
        ],
        outputs: [hit_info],
        shader_setup: scgi_trace_hit_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(total_patches / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );

    render_graph.add_pass(
      "scgi_trace_shade",
      RenderPassFlags.Compute,
      {
        inputs: [
          scgi_params,
          scene_lighting_data.scene_lighting_buffer,
          surface_cache,
          surface_cache_sh,
          active_indices,
          counters,
          hit_info,
          material_buffers.params_gpu_buffer,
          material_buffers.material_offsets_buffer,
          material_buffers.material_palette_buffer,
          entity_index_lookup,
          dense_lights,
          texture_pools.albedo,
          texture_pools.normal,
          texture_pools.roughness,
          texture_pools.metallic,
          texture_pools.ao,
          texture_pools.height,
          texture_pools.specular,
          texture_pools.emission,
          scene_lighting_data.skybox_image,
          radiance_info
        ],
        outputs: [hit_info, radiance_info],
        shader_setup: scgi_trace_shade_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(total_patches / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );

    render_graph.add_pass(
      "scgi_trace_shadow",
      RenderPassFlags.Compute,
      {
        inputs: [
          scgi_params,
          counters,
          hit_info,
          tlas_bvh2_bounds,
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          ray_instance_transforms,
          index_buffer,
          entity_index_lookup,
          radiance_info
        ],
        outputs: [radiance_info],
        shader_setup: scgi_trace_shadow_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(total_patches / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );

    render_graph.add_pass(
      "scgi_update",
      RenderPassFlags.Compute,
      {
        inputs: [
          scgi_params,
          surface_cache,
          surface_cache_sh,
          active_indices,
          counters,
          hit_info,
          radiance_info
        ],
        outputs: [surface_cache, surface_cache_sh],
        shader_setup: scgi_update_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(total_patches / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );

    render_graph.add_pass(
      "scgi_filter",
      RenderPassFlags.Compute,
      {
        inputs: [
          scgi_params,
          surface_cache,
          surface_cache_sh,
          surface_cache_sh_filtered,
          active_indices,
          counters
        ],
        outputs: [surface_cache_sh_filtered],
        shader_setup: scgi_filter_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(total_patches / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );

    render_graph.add_pass(
      "scgi_resolve",
      RenderPassFlags.Compute,
      {
        inputs: [
          scgi_params,
          surface_cache,
          surface_cache_sh_filtered,
          depth_texture,
          gbuffer_normal,
          this.final_gi_texture_direct,
          this.final_gi_texture_indirect_diffuse,
          this.final_gi_texture_indirect_specular
        ],
        outputs: [
          this.final_gi_texture_direct,
          this.final_gi_texture_indirect_diffuse,
          this.final_gi_texture_indirect_specular
        ],
        shader_setup: scgi_resolve_shader_setup
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    this.scgi_params = scgi_params;
    this.surface_cache = surface_cache;
    this.surface_cache_sh = surface_cache_sh;
  }

  add_debug_passes(
    render_graph,
    width,
    height,
    gbuffer_normal,
    depth_texture,
    scene_color,
    debug_view,
    force_recreate = false
  ) {
    if (debug_view !== DebugDrawType.GI_SurfaceCache) {
      return null;
    }

    this.debug_texture = render_graph.create_image({
      name: "scgi_cache_debug",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    render_graph.add_pass(
      "scgi_cache_debug",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.scgi_params,
          this.surface_cache,
          this.surface_cache_sh,
          depth_texture,
          gbuffer_normal,
          scene_color,
          this.debug_texture,
        ],
        outputs: [this.debug_texture],
        shader_setup: scgi_cache_debug_shader_setup,
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    return this.debug_texture;
  }

  reset() {
    this.final_gi_texture_direct = null;
    this.final_gi_texture_indirect_diffuse = null;
    this.final_gi_texture_indirect_specular = null;
  }
}
