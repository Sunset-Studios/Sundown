import { DebugDrawType, RenderPassFlags } from "../renderer_types.js";
import { GIPipelineComposition } from "./gi_pipeline.js";
import { ScrollingProbeVolumeTraceHitCache } from "./trace_hit_caches.js";
import { ProbeSHShadingStrategy } from "./shading_strategies.js";
import { ProbeSHAccumulator } from "./accumulators.js";

const probe_debug_shader_setup = {
  pipeline_shaders: { compute: { path: "gi/ddgi_sh_probe_debug.wgsl" } },
};

/** Scrolling cascaded probe-volume GI composition. */
export class DDGI {
  config = {
    probe_grid_dimensions: [32, 32, 32],
    probe_spacing: 2.0,
    probe_radius: 0.1,
    max_rays_per_probe: 256,
    probes_per_frame: 1024,
    indirect_boost: 1.0,
    cascade_count: 6,
    cascade_spacing_multiplier: 2.0,
    probe_depth_resolutions: [8, 4, 4, 4, 4, 4, 4, 4],
    max_emissive_lights: 32768,
    diffuse_sample_upscale_factor: 2,
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
  shared_bindings = {
    sh_probes_buffer: null,
    probe_states_buffer: null,
    probe_depth_moments_buffer: null,
    probe_surface_flags_buffer: null,
    probe_msme_stats_buffer: null,
  };

  constructor(params = {}, components = {}) {
    const { pipeline_components = {}, ...config } = params;
    const resolved = { ...pipeline_components, ...components };
    this.config = { ...this.config, ...config };
    this.pipeline = new GIPipelineComposition([
      {
        name: "probes",
        trace_hit_cache: resolved.trace_hit_cache ?? new ScrollingProbeVolumeTraceHitCache(),
        shading_strategy: resolved.shading_strategy ?? new ProbeSHShadingStrategy(),
        accumulator: resolved.accumulator ?? new ProbeSHAccumulator(),
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

    const { trace_hit_cache, accumulator } = this.pipeline.get_components("probes");
    this.ddgi_params = trace_hit_cache.get_resource("params");
    this.final_gi_texture_direct = accumulator.get_resource("direct_output");
    this.final_gi_texture_indirect_diffuse = accumulator.get_resource("diffuse_output");
    this.final_gi_texture_indirect_specular = accumulator.get_resource("specular_output");
    this.shared_bindings.sh_probes_buffer = accumulator.get_resource("sh_probes");
    this.shared_bindings.probe_states_buffer = trace_hit_cache.get_resource("probe_states");
    this.shared_bindings.probe_depth_moments_buffer = accumulator.get_resource("depth_moments");
    this.shared_bindings.probe_surface_flags_buffer = trace_hit_cache.get_resource("surface_flags");
    this.shared_bindings.probe_msme_stats_buffer = accumulator.get_resource("msme_stats");
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
    this.debug_texture = render_graph.create_image({
      name: "ddgi_debug",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    if (debug_view === DebugDrawType.GI_Probes) {
      render_graph.add_pass(
        "probe_sh_debug",
        RenderPassFlags.Compute,
        {
          inputs: [
            this.ddgi_params,
            this.shared_bindings.sh_probes_buffer,
            this.shared_bindings.probe_states_buffer,
            this.shared_bindings.probe_surface_flags_buffer,
            scene_color,
            depth_texture,
            this.debug_texture,
          ],
          outputs: [this.debug_texture],
          shader_setup: probe_debug_shader_setup,
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
      );
    }
  }

  get_stats() {
    return this.pipeline.get_components("probes").trace_hit_cache.get_stats();
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
