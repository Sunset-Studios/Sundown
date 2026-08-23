import { SharedFrameInfoBuffer } from "../../core/shared_data.js";
import {
  PerPixelRadianceCache,
  PIXEL_RADIANCE_CACHE_DIRECT_OUTPUT_NAME,
  PIXEL_RADIANCE_CACHE_SPECULAR_OUTPUT_NAME,
} from "./radiance_caches/pixel_radiance_cache.js";
import { SurfaceRadianceCache } from "./radiance_caches/surface_cache.js";

/** Surface-cache plus per-pixel path-traced GI composition. */
export class PTGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    screen_ray_count: 1,
    upscale_factor: 4,
    surface_cache_size: 131072,
    rays_per_patch: 1,
    cache_entry_lifetime: 1,
    history_hysteresis: 0.9,
    max_history_samples: 128,
    importance_sample_count: 8,
    importance_exploration: 0.01,
    indirect_boost: 1.0,
    max_ray_length: 128.0,
    max_emissive_lights: 32768,
    screen_reconstruction_enabled: false,
    diffuse_atrous_enabled: false,
    diffuse_atrous_pass_count: 3,
    diffuse_atrous_phi_depth: 0.04,
    diffuse_atrous_phi_normal: 64.0,
    diffuse_atrous_luma_sigma: 1.0,
  };

  constructor() {
    this.surface_cache = new SurfaceRadianceCache();
    this.pixel_cache = new PerPixelRadianceCache();
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

    const safe_upscale_factor = Math.max(1, Math.floor(this.config.upscale_factor));
    const gi_width = Math.max(1, Math.ceil(width / safe_upscale_factor));
    const gi_height = Math.max(1, Math.ceil(height / safe_upscale_factor));
    const total_pixels = gi_width * gi_height;
    const frame_index = SharedFrameInfoBuffer.get_frame_index();

    const frame_context = {
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
        compact_transforms,
        index_buffer,
        dense_lights,
      },
    };

    this.surface_cache.add_passes(render_graph, frame_context);
    this.pixel_cache.add_passes(render_graph, frame_context, this.surface_cache);

    this.final_gi_texture_direct = render_graph.get_resource_handle(
      PIXEL_RADIANCE_CACHE_DIRECT_OUTPUT_NAME
    );
    this.final_gi_texture_indirect_diffuse = render_graph.get_resource_handle(
      this.pixel_cache.final_diffuse_output_name
    );
    this.final_gi_texture_indirect_specular = render_graph.get_resource_handle(
      PIXEL_RADIANCE_CACHE_SPECULAR_OUTPUT_NAME
    );
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
    this.debug_texture = this.surface_cache.add_debug_passes(render_graph, {
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
