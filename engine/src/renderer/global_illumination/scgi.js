import { GIPipelineComposition } from "./gi_pipeline.js";
import { SurfaceRadianceCache } from "./radiance_caches.js";

/** Surface-cache global illumination without a per-pixel tracing branch. */
export class SCGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    surface_cache_size: 262144,
    rays_per_patch: 8,
    cache_entry_lifetime: 1,
    hash_search_count: 10,
    cache_pixel_footprint: 16.0,
    cache_lookup_jitter: 0.75,
    cache_sample_limit: 128,
    history_hysteresis: 0.99,
    max_history_samples: 128,
    importance_sample_count: 8,
    importance_exploration: 0.01,
    indirect_boost: 1.0,
    max_ray_length: 128.0,
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
    _prev_depth_texture,
    gbuffer_normal,
    _gbuffer_normal_prev,
    _gbuffer_albedo,
    _gbuffer_smra,
    _gbuffer_motion_emissive,
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
        gbuffer_normal,
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
