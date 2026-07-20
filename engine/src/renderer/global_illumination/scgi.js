import { GIPipelineComposition } from "./gi_pipeline.js";
import { SurfaceRadianceCache } from "./radiance_caches.js";

/** Composed surface-cache GI pipeline. */
export class SCGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    surface_cache_size: 131072,
    surface_cache_cell_size: 0.25,
    surface_cache_lod_count: 4,
    cache_entry_lifetime: 1,
    max_ray_length: 1024.0,
    history_hysteresis: 0.95,
    max_history_samples: 128,
    indirect_boost: 1.0,
    importance_sample_count: 8,
    importance_exploration: 0.01,
  };

  constructor(params = {}, components = {}) {
    this.radiance_cache = new SurfaceRadianceCache(params.radiance_cache);
    this.pipeline = new GIPipelineComposition([
      {
        name: "surface",
        module: this.radiance_cache,
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
      },
    });

    const radiance_cache = this.pipeline.get_module("surface");
    this.final_gi_texture_direct = radiance_cache.get_resource("direct_output");
    this.final_gi_texture_indirect_diffuse = radiance_cache.get_resource("diffuse_output");
    this.final_gi_texture_indirect_specular = radiance_cache.get_resource("specular_output");
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
      inputs: { gbuffer_normal, depth_texture, scene_color },
    });
    return this.debug_texture;
  }

  reset() {
    this.final_gi_texture_direct = null;
    this.final_gi_texture_indirect_diffuse = null;
    this.final_gi_texture_indirect_specular = null;
  }
}
