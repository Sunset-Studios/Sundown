import {
  ProbeVolumeRadianceCache,
  PROBE_VOLUME_DIRECT_OUTPUT_NAME,
  PROBE_VOLUME_SPECULAR_OUTPUT_NAME,
} from "./radiance_caches/probe_volume_radiance_cache.js";
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

  constructor() {
    this.radiance_cache = new ProbeVolumeRadianceCache();
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
    emissive_lights,
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
    this.radiance_cache.add_passes(render_graph, {
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
        emissive_lights,
        hzb_texture,
      },
    });
    this.config_resource_rebuild_pending = false;

    this.final_gi_texture_direct = render_graph.get_resource_handle(
      PROBE_VOLUME_DIRECT_OUTPUT_NAME
    );
    this.final_gi_texture_indirect_diffuse = render_graph.get_resource_handle(
      this.radiance_cache.final_diffuse_output_name
    );
    this.final_gi_texture_indirect_specular = render_graph.get_resource_handle(
      PROBE_VOLUME_SPECULAR_OUTPUT_NAME
    );
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
    this.debug_texture = this.radiance_cache.add_debug_passes(render_graph, {
      width,
      height,
      debug_view,
      force_recreate,
      inputs: { depth_texture, scene_color },
    });
    return this.debug_texture;
  }

  get_stats() {
    return this.radiance_cache.get_stats();
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
