import { GIPipelineComposition } from "./gi_pipeline.js";
import { SurfaceRadianceCache } from "./radiance_caches.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { EntityFlags } from "../../core/minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../../core/ecs/solar/types.js";
import { SharedEnvironmentData } from "../../core/shared_data.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";

/** Surface-cache global illumination without a per-pixel tracing branch. */
export class SCGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  config = {
    surface_cache_size: 131072,
    rays_per_patch: 8,
    bootstrap_patch_capacity: 131072,
    bootstrap_rays_per_patch: 256,
    bootstrap_ray_budget_fraction: 0.5,
    cache_entry_lifetime: 1,
    hash_search_count: 10,
    cache_pixel_footprint: 20.0,
    cache_normal_bias: 0.005,
    history_footprint_start_samples: 0.0,
    history_footprint_end_samples: 512.0,
    history_footprint_max_scale: 2.0,
    history_hysteresis: 0.95,
    max_history_samples: 256,
    mature_patch_update_period: 4,
    maximum_ray_count_per_frame: 65536,
    screen_reconstruction_enabled: true,
    temporal_response: 0.001,
    temporal_max_history_frames: 256,
    temporal_depth_threshold: 0.03,
    temporal_normal_threshold: 0.9,
    recurrent_blur_enabled: true,
    recurrent_blur_max_radius: 32,
    recurrent_blur_history_frames: 128,
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
    this.frame_context = {
      config: this.config,
      width: 0,
      height: 0,
      force_recreate: false,
      inputs: {},
    };
    this.debug_context = {
      width: 0,
      height: 0,
      debug_view: 0,
      force_recreate: false,
      inputs: {},
    };
    this.light_query = EntityManager.create_query([LightFragment]);
    this.light_snapshot = new Float32Array(32);
    this.light_snapshot_length = 0;
    this.environment_snapshot = new Float32Array(12);
    this.environment_skybox = null;
    this.material_revision = -1;
    this.explicit_invalidation_revision = 1;
    this.applied_invalidation_revision = 0;
    this.light_snapshot_write_index = 0;
    this.radiance_inputs_changed = false;
    this._capture_light_chunk = this._capture_light_chunk.bind(this);
  }

  _ensure_light_snapshot_capacity(required_length) {
    if (required_length <= this.light_snapshot.length) return;

    let capacity = this.light_snapshot.length;
    while (capacity < required_length) capacity *= 2;
    const next_snapshot = new Float32Array(capacity);
    next_snapshot.set(this.light_snapshot);
    this.light_snapshot = next_snapshot;
  }

  _append_light_snapshot_value(value) {
    const snapshot_index = this.light_snapshot_write_index;
    this._ensure_light_snapshot_capacity(snapshot_index + 1);
    this.radiance_inputs_changed ||=
      this.light_snapshot[snapshot_index] !== value;
    this.light_snapshot[snapshot_index] = value;
    this.light_snapshot_write_index = snapshot_index + 1;
  }

  _capture_light_chunk(chunk, flags) {
    // Compare the compact radiance-producing state directly. The persistent
    // snapshot avoids allocating or hashing in this once-per-frame hot path.
    const lights = chunk.get_fragment_view(LightFragment);
    for (let slot = 0; slot < DEFAULT_CHUNK_CAPACITY; slot++) {
      if (
        (flags[slot] & EntityFlags.ALIVE) === 0 ||
        lights.active[slot] === 0
      ) {
        continue;
      }

      const vector_offset = slot * 4;
      this._append_light_snapshot_value(lights.type[slot]);
      this._append_light_snapshot_value(lights.intensity[slot]);
      this._append_light_snapshot_value(lights.radius[slot]);
      this._append_light_snapshot_value(lights.attenuation[slot]);
      this._append_light_snapshot_value(lights.outer_angle[slot]);
      for (let component = 0; component < 4; component++) {
        this._append_light_snapshot_value(lights.position[vector_offset + component]);
      }
      for (let component = 0; component < 4; component++) {
        this._append_light_snapshot_value(lights.direction[vector_offset + component]);
      }
      for (let component = 0; component < 4; component++) {
        this._append_light_snapshot_value(lights.color[vector_offset + component]);
      }
      this._append_light_snapshot_value(lights.is_primary_sun[slot]);
    }
  }

  _radiance_inputs_changed(force_recreate) {
    let changed = force_recreate ||
      this.material_revision !== MaterialAllocationTable.radiance_revision ||
      this.applied_invalidation_revision !== this.explicit_invalidation_revision;
    this.material_revision = MaterialAllocationTable.radiance_revision;
    this.applied_invalidation_revision = this.explicit_invalidation_revision;

    this.light_snapshot_write_index = 0;
    this.radiance_inputs_changed = changed;
    this.light_query.for_each_chunk(this._capture_light_chunk);
    changed = this.radiance_inputs_changed ||
      this.light_snapshot_write_index !== this.light_snapshot_length;
    this.light_snapshot_length = this.light_snapshot_write_index;

    const environment_data = SharedEnvironmentData.skydome_data_buffer;
    for (let index = 0; index < environment_data.length; index++) {
      const value = environment_data[index];
      changed ||= this.environment_snapshot[index] !== value;
      this.environment_snapshot[index] = value;
    }
    const skybox = SharedEnvironmentData.get_skybox();
    changed ||= this.environment_skybox !== skybox;
    this.environment_skybox = skybox;
    return changed;
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

    const frame_context = this.frame_context;
    const inputs = frame_context.inputs;
    frame_context.config = this.config;
    frame_context.width = width;
    frame_context.height = height;
    frame_context.force_recreate = force_recreate;
    frame_context.force_full_update = this._radiance_inputs_changed(force_recreate);
    inputs.depth_texture = depth_texture;
    inputs.prev_depth_texture = prev_depth_texture;
    inputs.gbuffer_normal = gbuffer_normal;
    inputs.gbuffer_normal_prev = gbuffer_normal_prev;
    inputs.gbuffer_motion_emissive = gbuffer_motion_emissive;
    inputs.tlas_bvh2_bounds = tlas_bvh2_bounds;
    inputs.tlas_bvh_info = tlas_bvh_info;
    inputs.blas_bvh2_nodes = blas_bvh2_nodes;
    inputs.blas_directory = blas_directory;
    inputs.entity_transforms = entity_transforms;
    inputs.compact_transforms = compact_transforms;
    inputs.index_buffer = index_buffer;
    inputs.dense_lights = dense_lights;
    this.pipeline.add_passes(render_graph, frame_context);

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
    const debug_context = this.debug_context;
    debug_context.width = width;
    debug_context.height = height;
    debug_context.debug_view = debug_view;
    debug_context.force_recreate = force_recreate;
    debug_context.inputs.gbuffer_normal = gbuffer_normal;
    debug_context.inputs.depth_texture = depth_texture;
    debug_context.inputs.scene_color = scene_color;
    this.debug_texture = this.pipeline.add_debug_passes(render_graph, debug_context);
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
    this.invalidate();
  }

  invalidate() {
    this.explicit_invalidation_revision =
      (this.explicit_invalidation_revision + 1) >>> 0;
  }

  reset() {
    this.final_gi_texture_direct = null;
    this.final_gi_texture_indirect_diffuse = null;
    this.final_gi_texture_indirect_specular = null;
    this.debug_texture = null;
  }
}
