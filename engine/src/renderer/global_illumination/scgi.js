import {
  SurfaceRadianceCache,
  SURFACE_CACHE_DIRECT_OUTPUT_NAME,
  SURFACE_CACHE_SPECULAR_OUTPUT_NAME,
} from "./radiance_caches/surface_cache.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { EntityFlags } from "../../core/minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../../core/ecs/solar/types.js";
import { SharedEnvironmentData } from "../../core/shared_data.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";

/** Surface-cache global illumination without a per-pixel tracing branch. */
export class SCGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    surface_cache_size: 262144,
    rays_per_patch: 8,
    bootstrap_patch_capacity: 262144,
    bootstrap_rays_per_patch: 32,
    bootstrap_ray_budget_fraction: 1.0,
    cache_entry_lifetime: 1,
    hash_search_count: 10,
    cache_pixel_footprint: 16.0,
    cache_normal_bias: 0.005,
    history_footprint_start_samples: 0.0,
    history_footprint_end_samples: 512.0,
    history_footprint_max_scale: 2.0,
    native_promotion_start_confidence: 0.2,
    native_promotion_end_confidence: 0.85,
    history_hysteresis: 0.95,
    max_history_samples: 256,
    mature_patch_update_period: 4,
    maximum_ray_count_per_frame: 131072,
    temporal_response: 0.001,
    temporal_max_history_frames: 64,
    temporal_depth_threshold: 0.03,
    temporal_normal_threshold: 0.9,
    disocclusion_atrous_enabled: true,
    disocclusion_atrous_pass_count: 2,
    disocclusion_atrous_phi_depth: 0.03,
    disocclusion_atrous_phi_normal: 64.0,
    disocclusion_atrous_luma_sigma: 2.0,
    disocclusion_atrous_confidence_threshold: 0.85,
    disocclusion_atrous_history_frames: 128,
    indirect_boost: 1.0,
    max_ray_length: 128.0,
    max_emissive_lights: 32768,
  };
  frame_context = {
    config: null,
    width: 0,
    height: 0,
    force_recreate: false,
    inputs: {},
  };
  debug_context = {
    width: 0,
    height: 0,
    debug_view: 0,
    force_recreate: false,
    inputs: {},
  };

  constructor() {
    this.radiance_cache = new SurfaceRadianceCache();
    this.frame_context.config = this.config;
  }

  add_passes(
    render_graph,
    width,
    height,
    depth_texture,
    prev_depth_texture,
    gbuffer_normal,
    gbuffer_normal_prev,
    _gbuffer_albedo,
    _gbuffer_smra,
    gbuffer_motion_emissive,
    tlas_bvh2_bounds,
    tlas_bvh_info,
    blas_bvh2_nodes,
    blas_directory,
    entity_transforms,
    compact_transforms,
    index_buffer,
    dense_lights,
    emissive_lights,
    draw_count,
    _hzb_texture,
    force_recreate = false
  ) {
    if (draw_count <= 0) {
      this.reset();
      return;
    }

    this.frame_context.config = this.config;
    this.frame_context.width = width;
    this.frame_context.height = height;
    this.frame_context.force_recreate = force_recreate;
    this.frame_context.inputs.depth_texture = depth_texture;
    this.frame_context.inputs.prev_depth_texture = prev_depth_texture;
    this.frame_context.inputs.gbuffer_normal = gbuffer_normal;
    this.frame_context.inputs.gbuffer_normal_prev = gbuffer_normal_prev;
    this.frame_context.inputs.gbuffer_motion_emissive = gbuffer_motion_emissive;
    this.frame_context.inputs.tlas_bvh2_bounds = tlas_bvh2_bounds;
    this.frame_context.inputs.tlas_bvh_info = tlas_bvh_info;
    this.frame_context.inputs.blas_bvh2_nodes = blas_bvh2_nodes;
    this.frame_context.inputs.blas_directory = blas_directory;
    this.frame_context.inputs.entity_transforms = entity_transforms;
    this.frame_context.inputs.compact_transforms = compact_transforms;
    this.frame_context.inputs.index_buffer = index_buffer;
    this.frame_context.inputs.dense_lights = dense_lights;
    this.frame_context.inputs.emissive_lights = emissive_lights;

    this.radiance_cache.add_passes(render_graph, this.frame_context);

    this.final_gi_texture_direct = render_graph.get_resource_handle(
      SURFACE_CACHE_DIRECT_OUTPUT_NAME
    );
    this.final_gi_texture_indirect_diffuse = render_graph.get_resource_handle(
      this.radiance_cache.final_diffuse_output_name
    );
    this.final_gi_texture_indirect_specular = render_graph.get_resource_handle(
      SURFACE_CACHE_SPECULAR_OUTPUT_NAME
    );
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
    this.debug_context.width = width;
    this.debug_context.height = height;
    this.debug_context.debug_view = debug_view;
    this.debug_context.force_recreate = force_recreate;
    this.debug_context.inputs.gbuffer_normal = gbuffer_normal;
    this.debug_context.inputs.depth_texture = depth_texture;
    this.debug_context.inputs.scene_color = scene_color;

    this.debug_texture = this.radiance_cache.add_debug_passes(render_graph, this.debug_context);

    return this.debug_texture;
  }

  get_stats() {
    return this.radiance_cache.get_stats();
  }

  set_stats_enabled(enabled) {
    this.radiance_cache.set_stats_enabled(enabled);
  }

  set_config(new_config) {
    this.config = { ...this.config, ...new_config };
  }

  reset() {
    this.final_gi_texture_direct = null;
    this.final_gi_texture_indirect_diffuse = null;
    this.final_gi_texture_indirect_specular = null;
    this.debug_texture = null;
  }
}
