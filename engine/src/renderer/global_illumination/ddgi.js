import { Texture } from "../texture.js";
import {
  SharedEnvironmentData,
  SharedFrameInfoBuffer,
  SharedViewBuffer,
} from "../../core/shared_data.js";
import { DebugDrawType, RenderPassFlags, CacheTypes } from "../renderer_types.js";
import { Buffer } from "../buffer.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";
import { Name } from "../../utility/names.js";
import { ResourceCache } from "../resource_cache.js";
import { ispot, npot } from "../../utility/math.js";

const COMPUTE_WORKGROUP_SIZE = 256;
const DDGI_DEFAULT_PROBE_DEPTH_RESOLUTION = 16;
const DDGI_MAX_CASCADES = 8;

const DDGI_GI_COUNTERS_NAME = "ddgi_gi_counters";
const DDGI_GI_COUNTER_LIGHT_COUNT_INDEX = 0;
const DDGI_GI_COUNTER_NONCULLED_ACTIVE_PROBE_INDEX = 2;
const DDGI_GI_COUNTER_CULLED_ACTIVE_PROBE_INDEX = 3;
const DDGI_GI_COUNTER_PROBE_UPDATE_COUNT_INDEX = 5;

// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ Permutation helpers (CPU-side precomputation for probe cycling)             │
// └─────────────────────────────────────────────────────────────────────────────┘

/**
 * Simple integer hash function matching the shader's hash() function.
 * @param {number} x - Input value
 * @returns {number} - Hashed value as unsigned 32-bit integer
 */
function hash_u32(x) {
  x = x >>> 0; // Ensure unsigned
  x = ((x >>> 16) ^ x) * 0x45d9f3b;
  x = ((x >>> 16) ^ x) * 0x45d9f3b;
  x = (x >>> 16) ^ x;
  return x >>> 0;
}

/**
 * Compute GCD using Euclidean algorithm.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
function gcd_u32(a, b) {
  a = a >>> 0;
  b = b >>> 0;
  while (b !== 0) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/**
 * Find a stride coprime with modulus, starting from a hashed seed.
 * @param {number} sequence_seed
 * @param {number} modulus
 * @returns {number}
 */
function coprime_stride_from_seed(sequence_seed, modulus) {
  modulus = modulus >>> 0;
  if (modulus <= 1) {
    return 1;
  }

  const range = modulus - 1;
  let stride = (hash_u32(sequence_seed) % range) + 1;
  stride = stride | 1; // Bias to odd
  stride = ((stride - 1) % range) + 1;

  // Bounded search for coprime stride
  for (let iter = 0; iter < 32; iter++) {
    if (gcd_u32(stride, modulus) === 1) {
      break;
    }
    stride = stride + 2;
    if (stride >= modulus) {
      stride = stride % modulus;
    }
    if (stride === 0) {
      stride = 1;
    }
  }

  // Fallback
  if (gcd_u32(stride, modulus) !== 1) {
    return 1;
  }
  return stride;
}

/**
 * Precompute the permutation parameters for probe cycling.
 * These values are uniform across all shader invocations.
 * @param {number} probe_count
 * @returns {{stride: number, base_offset: number, frame_stride: number}}
 */
function compute_permutation_params(probe_count) {
  probe_count = Math.max(1, probe_count >>> 0);

  const sequence_seed = hash_u32(probe_count ^ 0xa3c59ac3);
  const stride = coprime_stride_from_seed(sequence_seed, probe_count);
  const base_offset = hash_u32(sequence_seed ^ 0x85ebca6b) % probe_count;
  const frame_stride = coprime_stride_from_seed(sequence_seed ^ 0xc2b2ae35, probe_count);

  return { stride, base_offset, frame_stride };
}

// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ Resource cache / binding names                                               │
// └─────────────────────────────────────────────────────────────────────────────┘
const material_offsets_name = "material_table_offset";
const texture_pool_albedo_name = Name.from("texture_pool_albedo");
const texture_pool_normal_name = Name.from("texture_pool_normal");
const texture_pool_roughness_name = Name.from("texture_pool_roughness");
const texture_pool_metallic_name = Name.from("texture_pool_metallic");
const texture_pool_ao_name = Name.from("texture_pool_ao");
const texture_pool_height_name = Name.from("texture_pool_height");
const texture_pool_specular_name = Name.from("texture_pool_specular");
const texture_pool_emission_name = Name.from("texture_pool_emission");

// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ Shader setups                                                                │
// └─────────────────────────────────────────────────────────────────────────────┘
const ddgi_reset_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_reset.wgsl" },
  },
};

const compact_emissive_lights_shader_setup = {
  pipeline_shaders: {
    compute: { path: "system_compute/compact_emissive_lights.wgsl" },
  },
};

const ddgi_probe_scroll_reset_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_scroll_reset.wgsl" },
  },
};

const ddgi_probe_surface_cull_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_surface_cull.wgsl" },
  },
};

const ddgi_probe_indices_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_indices_init.wgsl" },
  },
};

const ddgi_probe_active_mark_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_active_mark.wgsl" },
  },
};

const ddgi_probe_active_prefix_sum_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_active_prefix_sum.wgsl" },
  },
};

const ddgi_probe_active_block_prefix_scan_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_active_block_prefix_scan.wgsl" },
  },
};

const ddgi_probe_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_trace_init.wgsl" },
  },
};

const ddgi_probe_ray_budget_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_ray_budget.wgsl" },
  },
};

const ddgi_probe_trace_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_trace_hit.wgsl" },
  },
};

const ddgi_probe_trace_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_trace_shade.wgsl" },
  },
};

const ddgi_sh_probe_accumulate_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_sh_probe_accumulate.wgsl" },
  },
};

const ddgi_sh_probe_sample_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_sh_probe_sample.wgsl" },
  },
};

const ddgi_sh_probe_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_sh_probe_debug.wgsl" },
  },
};

const ddgi_probe_state_classify_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_state_classify.wgsl" },
  },
};

const ddgi_probe_cull_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_cull.wgsl" },
  },
};

const ddgi_atrous_diffuse_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_atrous_diffuse.wgsl" },
  },
};

export class DDGI {
  config = {
    probe_grid_dimensions: [64, 64, 64],
    probe_spacing: 2.0,
    probe_radius: 0.1,
    min_rays_per_probe: 32,
    max_rays_per_probe: 256,
    probes_per_frame: 512,
    probe_update_culled_ratio: 0.1,
    indirect_boost: 1.0,
    cascade_count: DDGI_MAX_CASCADES,
    cascade_spacing_multiplier: 2.0,
    probe_depth_resolutions: [8, 4, 4, 4, 4, 4, 4, 4],
    max_emissive_lights: 32768,
    diffuse_atrous_enabled: true,
    diffuse_atrous_pass_count: 3,
    diffuse_atrous_phi_depth: 0.04,
    diffuse_atrous_phi_normal: 64.0,
    diffuse_atrous_luma_sigma: 1.0,
  };

  ddgi_frame_setup = {
    width: 0,
    height: 0,
    frame_index: 0,
    ping_pong_frame: 0,
    dense_lights: null,
    force_recreate: false,
  };

  shared_bindings = {
    sh_probes_buffer: null,
    probe_states_buffer: null,
  };

  // DDGIParams layout:
  // [0-3]   probe_counts        (x=probe_count, y=max_rays_per_probe, z=probes_per_frame, w=probe_spacing)
  // [4-7]   probe_grid_dims     (x=dim_x, y=dim_y, z=dim_z, w=probe_radius)
  // [8-11]  probe_grid_origin   (xyz=grid origin, w=unused)
  // [12-15] probe_grid_log2     (xyz=log2(dim_*), w=unused)
  // [16-19] probe_grid_mask     (xyz=(dim_*-1), w=unused)
  // [20-23] probe_grid_snap_delta (xyz=delta in probe cells, w=active (1/0))
  // [24]    frame_index
  // [25]    indirect_boost
  // [26]    cascade_count
  // [27]    probe_update_culled_ratio
  // [28]    permutation_stride       (precomputed coprime stride for probe cycling)
  // [29]    permutation_base_offset  (precomputed base offset for permutation)
  // [30]    permutation_frame_stride (precomputed frame stride for temporal offset)
  // [31]    min_rays_per_probe
  // [32+]   cascade[0..N]:
  //         origin_spacing(4), scroll_offset(4), snap_delta(4), depth_atlas_info(4)
  // ... (16 floats per cascade)
  ddgi_params_data = new Float32Array(32 + 16 * DDGI_MAX_CASCADES);
  ddgi_params = null;

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ Probe grid snap tracking (CPU-side)                                          │
  // └─────────────────────────────────────────────────────────────────────────────┘
  ddgi_probe_grid_snapped_origin = null;
  ddgi_probe_grid_scroll_offsets = null;
  ddgi_probe_grid_initialized = null;
  ddgi_probe_count = 0;
  ddgi_gi_counters_buffer = null;
  ddgi_gi_counters_data = null;

  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  constructor(params = {}) {
    this.config = { ...this.config, ...params };
  }

  add_passes(
    render_graph,
    width,
    height,
    gbuffer_position,
    gbuffer_position_prev,
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
    // ┌─────────────────────────────────────────────────────────────────────────────┐
    // │ Outputs                                                                     │
    // └─────────────────────────────────────────────────────────────────────────────┘
    this.final_gi_texture_direct = render_graph.create_image({
      name: "ddgi_direct_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    this.final_gi_texture_indirect_specular = render_graph.create_image({
      name: "ddgi_specular_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    this.final_gi_texture_indirect_diffuse = render_graph.create_image({
      name: "ddgi_diffuse_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    if (draw_count > 0) {
      this._add_probe_passes(
        width,
        height,
        render_graph,
        gbuffer_position,
        gbuffer_position_prev,
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
        force_recreate
      );
    }
  }

  add_debug_passes(
    render_graph,
    width,
    height,
    gbuffer_position,
    gbuffer_normal,
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
        "ddgi_sh_probe_debug",
        RenderPassFlags.Compute,
        {
          inputs: [
            this.ddgi_params,
            this.shared_bindings.sh_probes_buffer,
            this.shared_bindings.probe_states_buffer,
            scene_color,
            depth_texture,
            this.debug_texture,
          ],
          outputs: [this.debug_texture],
          shader_setup: ddgi_sh_probe_debug_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
        }
      );
    }
  }

  _ensure_gi_counters_buffer(force_recreate) {
    const needs_recreate = force_recreate || !this.ddgi_gi_counters_buffer;
    if (!needs_recreate) {
      return;
    }

    if (this.ddgi_gi_counters_buffer) {
      this.ddgi_gi_counters_buffer.destroy();
      this.ddgi_gi_counters_buffer = null;
    }

    this.ddgi_gi_counters_data = new Uint32Array(6);
    this.ddgi_gi_counters_buffer = Buffer.create({
      name: DDGI_GI_COUNTERS_NAME,
      raw_data: this.ddgi_gi_counters_data,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
    });
  }

  get_stats() {
    if (!this.ddgi_gi_counters_buffer || !this.ddgi_gi_counters_data) {
      return null;
    }

    const probe_grid_dims = this._sanitize_probe_grid_dimensions(
      this.config.probe_grid_dimensions
    );
    const cascade_count = Math.max(1, Math.floor(this.config.cascade_count));
    const probes_per_cascade = probe_grid_dims[0] * probe_grid_dims[1] * probe_grid_dims[2];
    const total_probe_count = probes_per_cascade * cascade_count;

    const probes_per_frame_cfg = Math.max(0, Math.floor(this.config.probes_per_frame));
    const probes_per_frame_requested =
      probes_per_frame_cfg === 0
        ? total_probe_count
        : Math.min(total_probe_count, probes_per_frame_cfg);
    const probes_per_frame = Math.min(total_probe_count, probes_per_frame_requested);

    const min_rays_per_probe = Math.max(
      1,
      Math.floor(this.config.min_rays_per_probe)
    );
    const max_rays_per_probe = Math.max(
      min_rays_per_probe,
      Math.floor(this.config.max_rays_per_probe)
    );

    const gi_counters_data = this.ddgi_gi_counters_data;
    const active_probe_count_nonculled =
      gi_counters_data[DDGI_GI_COUNTER_NONCULLED_ACTIVE_PROBE_INDEX] || 0;
    const active_probe_count_culled =
      gi_counters_data[DDGI_GI_COUNTER_CULLED_ACTIVE_PROBE_INDEX] || 0;
    const active_probe_count = active_probe_count_nonculled + active_probe_count_culled;
    const probe_update_count = gi_counters_data[DDGI_GI_COUNTER_PROBE_UPDATE_COUNT_INDEX] || 0;
    const min_total_rays_fired = probe_update_count * min_rays_per_probe;
    const max_total_rays_fired = probe_update_count * max_rays_per_probe;

    return {
      probe_grid_dims,
      cascade_count,
      probes_per_cascade,
      total_probe_count,
      probes_per_frame,
      min_rays_per_probe,
      max_rays_per_probe,
      min_total_rays_fired,
      max_total_rays_fired,
      active_probe_count,
      active_probe_count_nonculled,
      active_probe_count_culled,
      probe_update_count,
      light_count: gi_counters_data[DDGI_GI_COUNTER_LIGHT_COUNT_INDEX] || 0,
      probe_spacing: this.config.probe_spacing,
      probe_radius: this.config.probe_radius,
    };
  }

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ Probe passes                                                                │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _add_probe_passes(
    width,
    height,
    render_graph,
    gbuffer_position,
    gbuffer_position_prev,
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
    force_recreate
  ) {
    const grid_dims = this._sanitize_probe_grid_dimensions(this.config.probe_grid_dimensions);
    const probes_per_cascade = grid_dims[0] * grid_dims[1] * grid_dims[2];
    const cascade_count = Math.max(1, Math.floor(this.config.cascade_count));
    const probe_count = probes_per_cascade * cascade_count;

    // ┌─────────────────────────────────────────────────────────────────────────────┐
    // │ Probe update budget (temporal cycling)                                       │
    // └─────────────────────────────────────────────────────────────────────────────┘
    // Update only a subset of probes per frame and rotate through the full set
    // temporally. The selection itself is generated on the GPU by `gi/ddgi_probe_indices_init.wgsl`.
    const probes_per_frame_cfg = Math.max(0, Math.floor(this.config.probes_per_frame));
    const probes_per_frame_requested =
      probes_per_frame_cfg === 0
        ? probe_count
        : Math.min(probe_count, probes_per_frame_cfg);
    const probes_per_frame = Math.min(probe_count, probes_per_frame_requested);
    const min_rays_per_probe = Math.max(
      1,
      Math.floor(this.config.min_rays_per_probe)
    );
    const max_rays_per_probe = Math.max(
      min_rays_per_probe,
      Math.floor(this.config.max_rays_per_probe)
    );
    const probe_primary_ray_count = probes_per_frame * max_rays_per_probe;
    const probe_total_ray_count = probe_primary_ray_count;

    const view_index = SharedFrameInfoBuffer.get_view_index();
    const view = SharedViewBuffer.get_view_data(view_index);
    const camera_position = view.view_position;

    const base_spacing = this.config.probe_spacing;
    const cascade_spacing_multiplier = Math.max(1.0, this.config.cascade_spacing_multiplier);
    const probe_depth_resolutions = this._sanitize_probe_depth_resolutions(cascade_count);

    if (
      !this.ddgi_probe_grid_snapped_origin ||
      this.ddgi_probe_grid_snapped_origin.length !== cascade_count
    ) {
      this.ddgi_probe_grid_snapped_origin = Array.from(
        { length: cascade_count },
        () => new Float32Array(3)
      );
      this.ddgi_probe_grid_scroll_offsets = Array.from(
        { length: cascade_count },
        () => new Int32Array(3)
      );
      this.ddgi_probe_grid_initialized = Array.from({ length: cascade_count }, () => false);
    }

    const cascade_data = new Float32Array(cascade_count * 16);
    let total_depth_texel_count = 0;
    let cascade_has_scroll = false;

    for (let cascade_index = 0; cascade_index < cascade_count; cascade_index += 1) {
      const spacing = base_spacing * Math.pow(cascade_spacing_multiplier, cascade_index);
      const half_extents = [
        (grid_dims[0] - 1) * 0.5 * spacing,
        (grid_dims[1] - 1) * 0.5 * spacing,
        (grid_dims[2] - 1) * 0.5 * spacing,
      ];

      const snapped_origin = this.ddgi_probe_grid_snapped_origin[cascade_index];
      const scroll_offset = this.ddgi_probe_grid_scroll_offsets[cascade_index];

      const next_origin = [
        Math.floor(camera_position[0] / spacing) * spacing - half_extents[0],
        Math.floor(camera_position[1] / spacing) * spacing - half_extents[1],
        Math.floor(camera_position[2] / spacing) * spacing - half_extents[2],
      ];

      let snap_delta_x = 0;
      let snap_delta_y = 0;
      let snap_delta_z = 0;
      const was_initialized = this.ddgi_probe_grid_initialized[cascade_index];

      if (was_initialized) {
        snap_delta_x = Math.round((next_origin[0] - snapped_origin[0]) / spacing);
        snap_delta_y = Math.round((next_origin[1] - snapped_origin[1]) / spacing);
        snap_delta_z = Math.round((next_origin[2] - snapped_origin[2]) / spacing);
      }

      snapped_origin[0] = next_origin[0];
      snapped_origin[1] = next_origin[1];
      snapped_origin[2] = next_origin[2];
      this.ddgi_probe_grid_initialized[cascade_index] = true;

      if (snap_delta_x !== 0 || snap_delta_y !== 0 || snap_delta_z !== 0) {
        scroll_offset[0] =
          ((scroll_offset[0] + snap_delta_x) % grid_dims[0] + grid_dims[0]) % grid_dims[0];
        scroll_offset[1] =
          ((scroll_offset[1] + snap_delta_y) % grid_dims[1] + grid_dims[1]) % grid_dims[1];
        scroll_offset[2] =
          ((scroll_offset[2] + snap_delta_z) % grid_dims[2] + grid_dims[2]) % grid_dims[2];
        cascade_has_scroll = true;
      }

      const depth_resolution = probe_depth_resolutions[cascade_index];
      const depth_texel_count_per_probe = depth_resolution * depth_resolution;
      const cascade_depth_base_offset = total_depth_texel_count;
      total_depth_texel_count += probes_per_cascade * depth_texel_count_per_probe;

      const base_index = cascade_index * 16;
      cascade_data[base_index + 0] = snapped_origin[0];
      cascade_data[base_index + 1] = snapped_origin[1];
      cascade_data[base_index + 2] = snapped_origin[2];
      cascade_data[base_index + 3] = spacing;
      cascade_data[base_index + 4] = scroll_offset[0];
      cascade_data[base_index + 5] = scroll_offset[1];
      cascade_data[base_index + 6] = scroll_offset[2];
      cascade_data[base_index + 7] = 0;
      cascade_data[base_index + 8] = snap_delta_x;
      cascade_data[base_index + 9] = snap_delta_y;
      cascade_data[base_index + 10] = snap_delta_z;
      cascade_data[base_index + 11] =
        snap_delta_x !== 0 || snap_delta_y !== 0 || snap_delta_z !== 0 ? 1 : 0;
      cascade_data[base_index + 12] = depth_resolution;
      cascade_data[base_index + 13] = depth_texel_count_per_probe;
      cascade_data[base_index + 14] = cascade_depth_base_offset;
      cascade_data[base_index + 15] = 0;
    }

    const grid_log2 = [
      Math.round(Math.log2(grid_dims[0])),
      Math.round(Math.log2(grid_dims[1])),
      Math.round(Math.log2(grid_dims[2])),
    ];
    const grid_mask = [grid_dims[0] - 1, grid_dims[1] - 1, grid_dims[2] - 1];

    this.ddgi_frame_setup.width = width;
    this.ddgi_frame_setup.height = height;
    this.ddgi_frame_setup.dense_lights = dense_lights;
    this.ddgi_frame_setup.force_recreate = force_recreate;
    this.ddgi_frame_setup.frame_index = SharedFrameInfoBuffer.get_frame_index();
    this.ddgi_frame_setup.ping_pong_frame = this.ddgi_frame_setup.frame_index % 2;

    const blue_noise = Texture.default_blue_noise();
    const blue_noise_image = render_graph.register_image(blue_noise.config.name);

    const default_texture = Texture.default_array();
    const default_texture_buffer = render_graph.register_image(default_texture.config.name);

    const params_gpu = MaterialAllocationTable.params_buffer;
    const params_gpu_buffer = render_graph.register_buffer(params_gpu.config.name);
    const material_palette = MaterialAllocationTable.palette_buffer;
    const material_palette_buffer = render_graph.register_buffer(material_palette.config.name);
    const material_palette_offsets = EntityManager.get_fragment_gpu_buffer(
      StaticMeshFragment,
      material_offsets_name
    );
    const material_palette_offsets_buffer = render_graph.register_buffer(
      material_palette_offsets.buffer.config.name
    );

    const albedo_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_albedo_name);
    const albedo_pool_buffer = albedo_pool
      ? render_graph.register_image(albedo_pool.config.name)
      : default_texture_buffer;
    const normal_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_normal_name);
    const normal_pool_buffer = normal_pool
      ? render_graph.register_image(normal_pool.config.name)
      : default_texture_buffer;
    const roughness_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_roughness_name);
    const roughness_pool_buffer = roughness_pool
      ? render_graph.register_image(roughness_pool.config.name)
      : default_texture_buffer;
    const metallic_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_metallic_name);
    const metallic_pool_buffer = metallic_pool
      ? render_graph.register_image(metallic_pool.config.name)
      : default_texture_buffer;
    const ao_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_ao_name);
    const ao_pool_buffer = ao_pool
      ? render_graph.register_image(ao_pool.config.name)
      : default_texture_buffer;
    const height_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_height_name);
    const height_pool_buffer = height_pool
      ? render_graph.register_image(height_pool.config.name)
      : default_texture_buffer;
    const specular_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_specular_name);
    const specular_pool_buffer = specular_pool
      ? render_graph.register_image(specular_pool.config.name)
      : default_texture_buffer;
    const emission_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_emission_name);
    const emission_pool_buffer = emission_pool
      ? render_graph.register_image(emission_pool.config.name)
      : default_texture_buffer;

    const skydome_data = SharedEnvironmentData.get_skydome_data();
    const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

    const skybox = SharedEnvironmentData.get_skybox();
    const skybox_texture_buffer = render_graph.register_image(skybox.config.name);

    this.ddgi_params = render_graph.create_buffer({
      name: "ddgi_params",
      size: this.ddgi_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this._ensure_gi_counters_buffer(force_recreate);
    const gi_counters = render_graph.register_buffer(DDGI_GI_COUNTERS_NAME);

    const probe_update_indices = render_graph.create_buffer({
      name: "ddgi_probe_update_indices",
      size: probes_per_frame,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_ray_allocations = render_graph.create_buffer({
      name: "ddgi_probe_ray_allocations",
      size: probes_per_frame * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Active-only probe scheduling with frustum culling priority
    // (cull → mark → prefix sum → scatter)
    // - We build TWO permuted active-flag arrays: one for non-culled active
    //   probes and one for culled active probes.
    // - Non-culled probes are prioritized in the update list.
    // ─────────────────────────────────────────────────────────────────────────
    const probe_active_flags_nonculled = render_graph.create_buffer({
      name: "ddgi_probe_active_flags_nonculled",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_active_flags_culled = render_graph.create_buffer({
      name: "ddgi_probe_active_flags_culled",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_active_prefix_sum_nonculled = render_graph.create_buffer({
      name: "ddgi_probe_active_prefix_sum_nonculled",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_active_prefix_sum_culled = render_graph.create_buffer({
      name: "ddgi_probe_active_prefix_sum_culled",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_active_block_count = Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE);

    const probe_active_block_sums_nonculled = render_graph.create_buffer({
      name: "ddgi_probe_active_block_sums_nonculled",
      size: probe_active_block_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_active_block_sums_culled = render_graph.create_buffer({
      name: "ddgi_probe_active_block_sums_culled",
      size: probe_active_block_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_active_block_prefixes_nonculled = render_graph.create_buffer({
      name: "ddgi_probe_active_block_prefixes_nonculled",
      size: probe_active_block_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_active_block_prefixes_culled = render_graph.create_buffer({
      name: "ddgi_probe_active_block_prefixes_culled",
      size: probe_active_block_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Ray Data Buffer
    // - Small header + per-ray record containing both hit data and shaded radiance
    // - header = 1 atomic<u32> + padding = 4 x u32 words
    // - DDGIProbeRayData = 10 vec4s = 40 x u32 words
    // ─────────────────────────────────────────────────────────────────────────
    const probe_ray_data = render_graph.create_buffer({
      name: "ddgi_probe_ray_data",
      size:
        4 +                         // header = 1 atomic<u32> + padding = 4 x u32 words
        probe_total_ray_count * 40, // 10 vec4s = 40 x u32 words
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Depth Moments Buffers (Directional visibility, octahedral 8x8)
    // Per-cascade depth atlas resolution:
    // - Configurable through config.probe_depth_resolutions[cascade_index]
    // - atlas uses packed u32 moments (f16 mean_t, f16 mean_t2)
    // - total size is the sum over all cascades
    //
    // Each texel stores a single u32 with two f16-packed values:
    // - bits [0..15]  = mean_t   (f16)
    // - bits [16..31] = mean_t2  (f16)
    // ─────────────────────────────────────────────────────────────────────────
    const probe_depth_moments = render_graph.create_buffer({
      name: "ddgi_probe_depth_moments",
      size: total_depth_texel_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────────
    // SH Probe Buffers
    // L1 RGB: 4 coefficients × 3 channels = 12 floats packed to 6 u32 per probe
    // ─────────────────────────────────────────────────────────────────────────
    const sh_probes = render_graph.create_buffer({
      name: "ddgi_sh_probes",
      size: probe_count * 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Probe State Buffers
    // - [0..1] packed_state, sample_count
    // ─────────────────────────────────────────────────────────────────────────
    const probe_states = render_graph.create_buffer({
      name: "ddgi_probe_states",
      size: probe_count * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const emissive_light_header_words = 4;
    const emissive_light_stride_words = 16;
    const max_emissive_lights = Math.max(1, Math.floor(this.config.max_emissive_lights));
    const emissive_lights = render_graph.create_buffer({
      name: "ddgi_emissive_lights",
      size: emissive_light_header_words + max_emissive_lights * emissive_light_stride_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Precompute permutation parameters on CPU (eliminates expensive GCD loops in shader)
    const permutation_params = compute_permutation_params(probe_count);

    render_graph.add_pass(
      "ddgi_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const ddgi_params_buf = graph.get_physical_buffer(this.ddgi_params);

        const primary_origin = this.ddgi_probe_grid_snapped_origin[0];
        this.ddgi_params_data[0] = probe_count;
        this.ddgi_params_data[1] = max_rays_per_probe;
        this.ddgi_params_data[2] = probes_per_frame;
        this.ddgi_params_data[3] = base_spacing;
        this.ddgi_params_data[4] = grid_dims[0];
        this.ddgi_params_data[5] = grid_dims[1];
        this.ddgi_params_data[6] = grid_dims[2];
        this.ddgi_params_data[7] = this.config.probe_radius;
        this.ddgi_params_data[8] = primary_origin[0];
        this.ddgi_params_data[9] = primary_origin[1];
        this.ddgi_params_data[10] = primary_origin[2];
        this.ddgi_params_data[11] = 0;
        this.ddgi_params_data[12] = grid_log2[0];
        this.ddgi_params_data[13] = grid_log2[1];
        this.ddgi_params_data[14] = grid_log2[2];
        this.ddgi_params_data[15] = 0;
        this.ddgi_params_data[16] = grid_mask[0];
        this.ddgi_params_data[17] = grid_mask[1];
        this.ddgi_params_data[18] = grid_mask[2];
        this.ddgi_params_data[19] = 0;
        this.ddgi_params_data[20] = 0;
        this.ddgi_params_data[21] = 0;
        this.ddgi_params_data[22] = 0;
        this.ddgi_params_data[23] = 0;
        this.ddgi_params_data[24] = this.ddgi_frame_setup.frame_index;
        this.ddgi_params_data[25] = this.config.indirect_boost;
        this.ddgi_params_data[26] = cascade_count;
        this.ddgi_params_data[27] = this.config.probe_update_culled_ratio;
        this.ddgi_params_data[28] = permutation_params.stride;
        this.ddgi_params_data[29] = permutation_params.base_offset;
        this.ddgi_params_data[30] = permutation_params.frame_stride;
        this.ddgi_params_data[31] = min_rays_per_probe;

        // Copy cascade data into params buffer (starts at offset 32)
        // Each cascade has 16 floats:
        // origin_spacing(4), scroll_offset(4), snap_delta(4), depth_atlas_info(4)
        for (let i = 0; i < cascade_data.length; i++) {
          this.ddgi_params_data[32 + i] = cascade_data[i];
        }

        ddgi_params_buf.write_raw(this.ddgi_params_data);
      }
    );

    render_graph.add_pass(
      "ddgi_compact_emissive_lights",
      RenderPassFlags.Compute,
      {
        inputs: [
          tlas_bvh2_bounds,
          tlas_bvh_info,
          blas_directory,
          index_buffer,
          entity_transforms,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          albedo_pool_buffer,
          emission_pool_buffer,
          emissive_lights,
        ],
        outputs: [emissive_lights],
        shader_setup: compact_emissive_lights_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const emissive_lights_buf = graph.get_physical_buffer(emissive_lights);
        emissive_lights_buf.write_raw(new Uint32Array([0, 0, 0, 0]), 0);

        const tlas_bvh2_bounds_buf = graph.get_physical_buffer(tlas_bvh2_bounds);
        const tlas_aabb_stride_words = 8;
        const tlas_node_count = Math.floor(tlas_bvh2_bounds_buf.config.size / tlas_aabb_stride_words);
        pass.dispatch(Math.ceil(tlas_node_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_reset",
      RenderPassFlags.Compute,
      {
        inputs: [gi_counters, dense_lights, probe_ray_data],
        outputs: [gi_counters, probe_ray_data],
        shader_setup: ddgi_reset_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1);
      }
    );

    if (cascade_has_scroll) {
      render_graph.add_pass(
        "ddgi_probe_scroll_reset",
        RenderPassFlags.Compute,
        {
          inputs: [
            this.ddgi_params,
            sh_probes,
            probe_depth_moments,
            probe_states,
          ],
          outputs: [sh_probes, probe_depth_moments, probe_states],
          shader_setup: ddgi_probe_scroll_reset_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
        }
      );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Frustum & Occlusion Culling Pass
    // Marks each probe as visible (1) or culled (0) based on:
    // - Frustum culling: Is the probe inside the view frustum?
    // - Occlusion culling: Is the probe visible in the Hierarchical Z-Buffer?
    // Writes cull flags directly into ProbeStateData.cull_flags field
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_probe_cull",
      RenderPassFlags.Compute,
      {
        inputs: [this.ddgi_params, probe_states, hzb_texture],
        outputs: [probe_states],
        shader_setup: ddgi_probe_cull_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe BVH Surface Cull Pass
    // Runs after ddgi_probe_cull so we only BVH-test visible probes
    // Marks probes as SLEEPING/OFF when their cell does not overlap geometry
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_probe_surface_cull",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_states,
          tlas_bvh2_bounds,
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          entity_transforms,
        ],
        outputs: [probe_states],
        shader_setup: ddgi_probe_surface_cull_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Active Mark Pass (with frustum culling priority)
    // Outputs two flag arrays: one for non-culled active, one for culled active
    // Reads cull flags from ProbeStateData.cull_flags (set by ddgi_probe_cull)
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_probe_active_mark",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_states,
          probe_active_flags_nonculled,
          probe_active_flags_culled,
        ],
        outputs: [probe_active_flags_nonculled, probe_active_flags_culled],
        shader_setup: ddgi_probe_active_mark_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Active Prefix Sum Pass (handles both nonculled and culled)
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_probe_active_prefix_sum",
      RenderPassFlags.Compute,
      {
        inputs: [
          probe_active_flags_nonculled,
          probe_active_flags_culled,
          probe_active_prefix_sum_nonculled,
          probe_active_prefix_sum_culled,
          probe_active_block_sums_nonculled,
          probe_active_block_sums_culled,
        ],
        outputs: [
          probe_active_prefix_sum_nonculled,
          probe_active_prefix_sum_culled,
          probe_active_block_sums_nonculled,
          probe_active_block_sums_culled,
        ],
        shader_setup: ddgi_probe_active_prefix_sum_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Active Block Prefix Scan Pass (handles both nonculled and culled)
    // Also writes total counts to gi_counters for scheduling logic
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_probe_active_block_prefix_scan",
      RenderPassFlags.Compute,
      {
        inputs: [
          probe_active_block_sums_nonculled,
          probe_active_block_sums_culled,
          probe_active_block_prefixes_nonculled,
          probe_active_block_prefixes_culled,
          gi_counters,
          this.ddgi_params,
        ],
        outputs: [
          probe_active_block_prefixes_nonculled,
          probe_active_block_prefixes_culled,
          gi_counters,
        ],
        shader_setup: ddgi_probe_active_block_prefix_scan_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Indices Init Pass (with frustum culling priority)
    // Prioritizes non-culled probes, then fills remaining budget with culled
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_probe_indices_init",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_active_flags_nonculled,
          probe_active_prefix_sum_nonculled,
          probe_active_block_prefixes_nonculled,
          probe_active_flags_culled,
          probe_active_prefix_sum_culled,
          probe_active_block_prefixes_culled,
          gi_counters,
        ],
        outputs: [probe_update_indices],
        shader_setup: ddgi_probe_indices_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_probe_ray_budget",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_states,
          gi_counters,
          probe_ray_allocations,
          probe_ray_data,
        ],
        outputs: [probe_ray_allocations, probe_ray_data],
        shader_setup: ddgi_probe_ray_budget_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probes_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_probe_trace_init",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_ray_allocations,
          probe_ray_data,
          gi_counters,
        ],
        outputs: [probe_ray_data],
        shader_setup: ddgi_probe_trace_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(max_rays_per_probe / 16),
          Math.ceil(probes_per_frame / 16),
          1
        );
      }
    );

    render_graph.add_pass(
      "ddgi_probe_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_ray_data,
          tlas_bvh2_bounds,
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          entity_transforms,
          index_buffer,
          dense_lights,
          emissive_lights,
        ],
        outputs: [probe_ray_data],
        shader_setup: ddgi_probe_trace_hit_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_total_ray_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe Depth Moments (visibility / occlusion weighting)
    // Updated inside `ddgi_sh_probe_accumulate.wgsl`:
    // - per-probe scratch clear
    // - per-ray binning (no splatting)
    // - resolve into the persistent octahedral atlas
    // ─────────────────────────────────────────────────────────────────────────

    render_graph.add_pass(
      "ddgi_probe_trace_shade",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          skydome_data_buffer,
          probe_ray_data,
          probe_states,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          sh_probes,
          probe_depth_moments,
          albedo_pool_buffer,
          normal_pool_buffer,
          roughness_pool_buffer,
          metallic_pool_buffer,
          ao_pool_buffer,
          height_pool_buffer,
          specular_pool_buffer,
          emission_pool_buffer,
          skybox_texture_buffer,
        ],
        outputs: [probe_ray_data],
        shader_setup: ddgi_probe_trace_shade_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_total_ray_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // SH Probe Accumulation Pass
    // Projects ray radiance onto L1 spherical harmonics per probe
    // Tracks sample counts for proper weighted temporal averaging
    // Uses probe states to determine hysteresis (fast convergence for newly states)
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_sh_probe_accumulate",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_ray_allocations,
          probe_ray_data,
          sh_probes,
          probe_depth_moments,
          probe_states,
          gi_counters,
        ],
        outputs: [sh_probes, probe_depth_moments, probe_states],
        shader_setup: ddgi_sh_probe_accumulate_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probes_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe State Classification Pass
    // Classifies probes into states based on ray hit data
    // Must run after accumulate so we have complete ray data to analyze
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "ddgi_probe_state_classify",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_ray_allocations,
          probe_ray_data,
          probe_states,
          gi_counters,
        ],
        outputs: [probe_states],
        shader_setup: ddgi_probe_state_classify_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probes_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Probe SH Sampling Pass
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      `ddgi_sh_probe_sample_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          sh_probes,
          probe_states,
          probe_depth_moments,
          gbuffer_position,
          gbuffer_normal,
          this.final_gi_texture_indirect_diffuse,
          skydome_data_buffer,
          skybox_texture_buffer,
        ],
        outputs: [this.final_gi_texture_indirect_diffuse],
        shader_setup: ddgi_sh_probe_sample_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.width / 8),
          Math.ceil(this.ddgi_frame_setup.height / 8),
          1
        );
      }
    );

    this.ddgi_probe_count = probe_count;
    this.shared_bindings.sh_probes_buffer = sh_probes;
    this.shared_bindings.probe_states_buffer = probe_states;
  }

  set_config(new_config) {
    this.config = { ...this.config, ...new_config };
  }

  _sanitize_probe_grid_dimensions(probe_grid_dimensions) {
    // DDGI formulation recommends power-of-two grid resolution per axis.
    // We enforce that here so probe indexing can be implemented with bitwise ops.
    const out = [
      Math.max(1, Math.floor(probe_grid_dimensions[0])),
      Math.max(1, Math.floor(probe_grid_dimensions[1])),
      Math.max(1, Math.floor(probe_grid_dimensions[2])),
    ];

    if (!ispot(out[0])) {
      out[0] = npot(out[0]);
    }
    if (!ispot(out[1])) {
      out[1] = npot(out[1]);
    }
    if (!ispot(out[2])) {
      out[2] = npot(out[2]);
    }

    return out;
  }

  _sanitize_probe_depth_resolutions(cascade_count) {
    const default_resolution = DDGI_DEFAULT_PROBE_DEPTH_RESOLUTION;
    const configured_resolutions = Array.isArray(this.config.probe_depth_resolutions)
      ? this.config.probe_depth_resolutions
      : [];
    const out = new Uint32Array(cascade_count);

    for (let cascade_index = 0; cascade_index < cascade_count; cascade_index += 1) {
      const configured_resolution = configured_resolutions[cascade_index];
      let resolution =
        Number.isFinite(configured_resolution) && configured_resolution > 0
          ? Math.floor(configured_resolution)
          : default_resolution;

      resolution = Math.max(4, Math.min(64, resolution));
      if (!ispot(resolution)) {
        resolution = npot(resolution);
      }

      out[cascade_index] = resolution;
    }

    return out;
  }
}
