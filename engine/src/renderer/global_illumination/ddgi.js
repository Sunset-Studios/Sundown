import { GIPipelineComposition } from "./gi_pipeline.js";
import { ProbeVolumeRadianceCache } from "./radiance_caches.js";
import {
  clone_ddgi_config_value,
  create_ddgi_config,
  ddgi_config_change_requires_rebuild,
  ddgi_config_values_equal,
} from "./ddgi_config.js";

export { DDGI_DEFAULT_CONFIG, create_ddgi_config } from "./ddgi_config.js";

/** Scrolling cascaded probe-volume GI composition. */
export class DDGI {
  config = create_ddgi_config();
  config_resource_rebuild_pending = false;

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
    compact_transforms,
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

    const config_rebuild_pending = this.config_resource_rebuild_pending;
    const rebuild_resources = force_recreate || config_rebuild_pending;
    if (config_rebuild_pending) {
      this.radiance_cache.reset_runtime_state();
    }
    this.pipeline.add_passes(render_graph, {
      config: this.config,
      width,
      height,
      force_recreate: rebuild_resources,
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
        hzb_texture,
      },
    });
    this.config_resource_rebuild_pending = false;

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
    let changed = false;
    for (const [key, value] of Object.entries(new_config || {})) {
      if (ddgi_config_values_equal(this.config[key], value)) {
        continue;
      }
      this.config[key] = clone_ddgi_config_value(value);
      this.config_resource_rebuild_pending ||= ddgi_config_change_requires_rebuild(key);
      changed = true;
    }
    return changed;
  }

  reset_config() {
    return this.set_config(create_ddgi_config());
  }

  reset() {
    this.final_gi_texture_direct = null;
    this.final_gi_texture_indirect_diffuse = null;
    this.final_gi_texture_indirect_specular = null;
  }
}
