import { GIPipelineComposition } from "./gi_pipeline.js";
import { SurfaceRadianceCache } from "./radiance_caches.js";

/** Surface-cache global illumination without a per-pixel tracing branch. */
export class SCGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    surface_cache_size: 131072,
    rays_per_patch: 4,
    cache_entry_lifetime: 1,
    hash_search_count: 8,
    cache_pixel_footprint: 16.0,
    cache_normal_bias: 0.005,
    history_hysteresis: 0.995,
    max_history_samples: 256,
    screen_reconstruction_enabled: true,
    temporal_response: 0.02,
    temporal_max_history_frames: 64,
    temporal_depth_threshold: 0.03,
    temporal_normal_threshold: 0.9,
    spatial_filter_enabled: true,
    spatial_depth_sigma: 0.025,
    spatial_normal_threshold: 0.8,
    spatial_luminance_sigma: 1.0,
    spatial_filter_strength: 1.0,
    indirect_boost: 1.0,
    max_ray_length: 128.0,
    max_emissive_lights: 32768,
  };

  constructor(params = {}) {
    this.surface_radiance_cache = new SurfaceRadianceCache(params.surface_radiance_cache);
    this.pipeline = new GIPipelineComposition([
      {
        name: "surface",
        module: this.surface_radiance_cache,
      },
    ]);
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
    draw_count,
    _hzb_texture,
    force_recreate = false
  ) {
    if (draw_count <= 0) {
      this.reset();
      return;
    }

    this.pipeline.add_passes(render_graph, {
      config: this.config,
      width,
      height,
      force_recreate,
      inputs: {
        depth_texture,
        prev_depth_texture,
        gbuffer_normal,
        gbuffer_normal_prev,
        gbuffer_motion_emissive,
        tlas_bvh2_bounds,
        tlas_bvh_info,
        blas_bvh2_nodes,
        blas_directory,
        entity_transforms,
        compact_transforms,
        index_buffer,
        dense_lights,
      },
    });

    const output = this.pipeline.get_module("surface");
    this.final_gi_texture_direct = output.get_resource("direct_output");
    this.final_gi_texture_indirect_diffuse = output.get_resource("diffuse_output");
    this.final_gi_texture_indirect_specular = output.get_resource("specular_output");
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
    this.debug_texture = this.pipeline.add_debug_passes(render_graph, {
      width,
      height,
      debug_view,
      force_recreate,
      inputs: {
        gbuffer_normal,
        depth_texture,
        scene_color,
      },
    });
    return this.debug_texture;
  }

  get_stats() {
    return this.pipeline.get_module("surface").get_stats();
  }

  set_stats_enabled(enabled) {
    this.pipeline.get_module("surface").set_stats_enabled(enabled);
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
