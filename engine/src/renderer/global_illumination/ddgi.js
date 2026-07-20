import { GIPipelineComposition } from "./gi_pipeline.js";
import { ProbeVolumeRadianceCache } from "./radiance_caches.js";

/** Scrolling cascaded probe-volume GI composition. */
export class DDGI {
  config = {
    probe_grid_dimensions: [64, 64, 64],
    probe_spacing: 2.0,
    probe_radius: 0.1,
    max_rays_per_probe: 64,
    probes_per_frame: 1024,
    indirect_boost: 1.0,
    cascade_count: 6,
    cascade_spacing_multiplier: 2.0,
    probe_depth_resolutions: [8, 4, 4, 4, 4, 4, 4, 4],
    max_emissive_lights: 32768,
    diffuse_sample_upscale_factor: 1,
    diffuse_atrous_enabled: false,
    diffuse_atrous_pass_count: 3,
    diffuse_atrous_phi_depth: 0.04,
    diffuse_atrous_phi_normal: 64.0,
    diffuse_atrous_luma_sigma: 1.0,
  };

  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  constructor(params = {}, components = {}) {
    this.radiance_cache = new ProbeVolumeRadianceCache(params.radiance_cache);
    this.pipeline = new GIPipelineComposition([
      {
        name: "probes",
        module: this.radiance_cache,
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
    hzb_texture,
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
        hzb_texture,
      },
    });

    const radiance_cache = this.pipeline.get_module("probes");
    this.final_gi_texture_direct = radiance_cache.get_resource("direct_output");
    this.final_gi_texture_indirect_diffuse = radiance_cache.get_resource("diffuse_output");
    this.final_gi_texture_indirect_specular = radiance_cache.get_resource("specular_output");
  }

  add_debug_passes(
    render_graph,
    width,
    height,
    _gbuffer_normal,
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
      inputs: { depth_texture, scene_color },
    });
    return this.debug_texture;
  }

  get_stats() {
    return this.pipeline.get_module("probes").get_stats();
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
