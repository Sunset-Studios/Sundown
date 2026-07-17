import { DebugDrawType, RenderPassFlags } from "../renderer_types.js";
import { GIPipelineComposition } from "./gi_pipeline.js";
import { HashedSurfaceTraceHitCache } from "./trace_hit_caches.js";
import { SurfaceCacheSHShadingStrategy } from "./shading_strategies.js";
import { SurfaceCacheSHAccumulator } from "./accumulators.js";

const surface_cache_debug_shader_setup = {
  pipeline_shaders: { compute: { path: "gi/surface_cache_debug.wgsl" } },
};

/** Composed surface-cache GI pipeline. */
export class SCGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    surface_cache_size: 32768,
    surface_cache_cell_size: 0.25,
    surface_cache_lod_count: 6,
    cache_entry_lifetime: 1,
    max_ray_length: 1024.0,
    history_hysteresis: 0.95,
    max_history_samples: 128,
    indirect_boost: 1.0,
    importance_sample_count: 8,
    importance_exploration: 0.01,
  };

  constructor(params = {}, components = {}) {
    const { pipeline_components = {}, ...config } = params;
    const resolved = { ...pipeline_components, ...components };
    this.config = { ...this.config, ...config };
    this.pipeline = new GIPipelineComposition([
      {
        name: "surface",
        trace_hit_cache: resolved.trace_hit_cache ?? new HashedSurfaceTraceHitCache(),
        shading_strategy: resolved.shading_strategy ?? new SurfaceCacheSHShadingStrategy(),
        accumulator: resolved.accumulator ?? new SurfaceCacheSHAccumulator(),
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
    if (draw_count <= 0) return;

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

    const { trace_hit_cache, accumulator } = this.pipeline.get_components("surface");
    this.final_gi_texture_direct = accumulator.get_resource("direct_output");
    this.final_gi_texture_indirect_diffuse = accumulator.get_resource("diffuse_output");
    this.final_gi_texture_indirect_specular = accumulator.get_resource("specular_output");
    this.surface_cache_params = trace_hit_cache.get_resource("params");
    this.surface_cache = trace_hit_cache.get_resource("surface_cache");
    this.surface_cache_sh = accumulator.get_resource("surface_cache_sh");
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
    if (debug_view !== DebugDrawType.GI_SurfaceCache) return null;

    this.debug_texture = render_graph.create_image({
      name: "surface_cache_debug",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    render_graph.add_pass(
      "surface_cache_debug",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.surface_cache_params,
          this.surface_cache,
          this.surface_cache_sh,
          depth_texture,
          gbuffer_normal,
          scene_color,
          this.debug_texture,
        ],
        outputs: [this.debug_texture],
        shader_setup: surface_cache_debug_shader_setup,
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
