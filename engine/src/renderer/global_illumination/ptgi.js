import { SharedFrameInfoBuffer } from "../../core/shared_data.js";
import { GIPipelineComposition } from "./gi_pipeline.js";
import { HashedSurfaceTraceHitCache, PerPixelTraceHitCache } from "./trace_hit_caches.js";
import { PerPixelRGBShadingStrategy, SurfaceCacheSHShadingStrategy } from "./shading_strategies.js";
import { PerPixelRGBAccumulator, SurfaceCacheSHAccumulator } from "./accumulators.js";

/** Surface-cache plus per-pixel path-traced GI composition. */
export class PTGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    screen_ray_count: 1,
    upscale_factor: 4,
    surface_cache_size: 32768,
    surface_cache_cell_size: 0.25,
    surface_cache_lod_count: 4,
    cache_entry_lifetime: 1,
    history_hysteresis: 0.9,
    max_history_samples: 128,
    importance_sample_count: 8,
    importance_exploration: 0.01,
    indirect_boost: 1.0,
    max_ray_length: 128.0,
    max_emissive_lights: 32768,
    diffuse_atrous_enabled: false,
    diffuse_atrous_pass_count: 3,
    diffuse_atrous_phi_depth: 0.04,
    diffuse_atrous_phi_normal: 64.0,
    diffuse_atrous_luma_sigma: 1.0,
  };

  constructor(params = {}, components = {}) {
    this.pipeline = new GIPipelineComposition([
      {
        name: "surface",
        trace_hit_cache: new HashedSurfaceTraceHitCache(),
        shading_strategy: new SurfaceCacheSHShadingStrategy(),
        accumulator: new SurfaceCacheSHAccumulator(),
      },
      {
        name: "pixel",
        dependencies: { radiance_cache: "surface" },
        trace_hit_cache: new PerPixelTraceHitCache(),
        shading_strategy: new PerPixelRGBShadingStrategy(),
        accumulator: new PerPixelRGBAccumulator(),
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

    const safe_upscale_factor = Math.max(1, Math.floor(this.config.upscale_factor));
    const gi_width = Math.max(1, Math.ceil(width / safe_upscale_factor));
    const gi_height = Math.max(1, Math.ceil(height / safe_upscale_factor));
    const total_pixels = gi_width * gi_height;
    const frame_index = SharedFrameInfoBuffer.get_frame_index();

    this.pipeline.add_passes(render_graph, {
      config: this.config,
      width,
      height,
      gi_width,
      gi_height,
      total_pixels,
      rays_per_frame: total_pixels * this.config.screen_ray_count,
      safe_upscale_factor,
      frame_index,
      ping_pong_frame: frame_index % 2,
      force_recreate,
      inputs: {
        depth_texture,
        prev_depth_texture,
        gbuffer_normal,
        gbuffer_normal_prev,
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

    const output = this.pipeline.get_components("pixel").accumulator;
    this.final_gi_texture_direct = output.get_resource("direct_output");
    this.final_gi_texture_indirect_diffuse = output.get_resource("diffuse_output");
    this.final_gi_texture_indirect_specular = output.get_resource("specular_output");
  }

  add_debug_passes(
    render_graph,
    width,
    height,
    main_normal_image,
    depth_texture,
    post_lighting_image_desc,
    debug_view,
    force_recreate = false
  ) {
    this.debug_texture = this.pipeline.add_debug_passes(render_graph, {
      width,
      height,
      debug_view,
      force_recreate,
      inputs: {
        gbuffer_normal: main_normal_image,
        depth_texture,
        scene_color: post_lighting_image_desc,
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
  }
}
