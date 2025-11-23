/**
 * GI-1.0 Radiance Caching System
 *
 * Implements a two-level radiance caching scheme following the GI-1.0 paper:
 * - Screen Cache: Fixed grid of probes on primary surfaces with temporal reuse
 * - World Cache: Spatial hash-based cache for secondary+ bounce radiance
 *
 * Screen Probe Algorithm (per tile, per frame):
 * 1. Temporal Upscaling: Only fraction of tiles update each frame (e.g., 1/4)
 * 2. Reprojection: Use motion vectors to find previous probe
 *    - Each pixel in tile tracks backward to find which probe it came from
 *    - Validate: plane_distance < cell_size && normal_dot > 0.95
 *    - Threads compete atomically for best match (closest 3D distance)
 * 3. If reprojection succeeds: Reuse probe data (keep accumulated radiance!)
 * 4. If reprojection fails: Spawn new probe using Halton jitter, seed from world cache
 *    - Query spatial hash to find nearby cached radiance from secondary bounces
 *    - Provides much better initial estimates than starting from zero
 *    - Accelerates convergence and reduces flickering on disocclusions
 * 5. Ray tracing: Trace rays from probe position with temporal jittering
 * 6. Accumulation: Weighted average of ray throughput over time
 *    - Reprojected probes: maintain history for progressive refinement
 *    - New spawns: start with world cache seed for faster convergence
 *    - Sample count clamped to ~32 for responsiveness
 * 7. Gradual convergence as all tiles fill in over multiple frames
 *
 * Features:
 * - ReSTIR-based path sampling for high-quality convergence
 * - Two-level caching with cross-pollination (world cache seeds screen probes)
 * - Separate hit and visibility passes (aligned with path tracer architecture)
 * - Optimized BVH traversal with minimal redundant AABB tests
 * - Temporal stability through geometry-aware probe reuse
 * - Scalable configuration for different performance targets
 *
 * References:
 * - GI-1.0 Paper: https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf
 * - Screen Space Radiance Caching (SSRC)
 * - ReSTIR GI
 */

import { DebugDrawType, RenderPassFlags } from "../renderer_types.js";
import { SharedEnvironmentData, SharedFrameInfoBuffer } from "../../core/shared_data.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";
import { ResourceCache } from "../resource_cache.js";
import { CacheTypes } from "../renderer_types.js";
import { Name } from "../../utility/names.js";
import { Texture } from "../texture.js";

const COMPUTE_WORKGROUP_SIZE = 128;

const material_offsets_name = "material_table_offset";
const texture_pool_albedo_name = Name.from("texture_pool_albedo");
const texture_pool_normal_name = Name.from("texture_pool_normal");
const texture_pool_roughness_name = Name.from("texture_pool_roughness");
const texture_pool_metallic_name = Name.from("texture_pool_metallic");
const texture_pool_ao_name = Name.from("texture_pool_ao");
const texture_pool_height_name = Name.from("texture_pool_height");
const texture_pool_specular_name = Name.from("texture_pool_specular");
const texture_pool_emission_name = Name.from("texture_pool_emission");

const gi_reset_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gi_reset.wgsl" },
  },
};

const world_cache_evict_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_evict.wgsl" },
  },
};

const world_cache_compact_mark_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_active_mark.wgsl" },
  },
};

const world_cache_compact_prefix_sum_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_active_prefix_sum.wgsl" },
  },
};

const world_cache_compact_scatter_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_active_compact.wgsl" },
  },
};

const world_cache_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_trace_init.wgsl" },
  },
};

const world_cache_trace_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_trace_hit.wgsl" },
  },
};

const world_cache_trace_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_trace_shade.wgsl" },
  },
};

const screen_probe_reproject_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_reproject.wgsl" },
  },
};

const screen_probe_patch_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_patch.wgsl" },
  },
};

const screen_probe_spawn_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_spawn.wgsl" },
  },
};

const screen_probe_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_trace_init.wgsl" },
  },
};

const screen_probe_trace_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_trace_hit.wgsl" },
  },
};

const screen_probe_trace_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_trace_shade.wgsl" },
  },
};

const screen_probe_update_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_update.wgsl" },
  },
};

const screen_probe_reconstruct_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_reconstruct.wgsl" },
  },
};

const screen_probe_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_debug.wgsl" },
  },
};

const world_cache_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_debug.wgsl" },
  },
};

export class GI {
  final_gi_texture = null;
  debug_texture = null;

  // Configuration parameters
  config = {
    screen_probe_size: 4, // Side length of probe footprint in pixels
    screen_ray_count: 1, // Rays per screen probe (directional atlas coverage)
    upscale_x: 2, // Temporal upscale factor X (2x2 = 4 frames to fill)
    upscale_y: 2, // Temporal upscale factor Y
    world_cache_size: 32768, // Number of world cache cells (32K)
    world_cache_cell_size: 2.0, // Size of world cache cells in world units (larger = better coverage)
    world_cache_lod_count: 4, // Number of LOD levels for world cache
    indirect_boost: 2.0, // Multiplier for indirect lighting
  };

  constructor(params = {}) {
    // Override defaults with provided parameters
    this.config = { ...this.config, ...params };
    // GI parameters buffer (must match shader GIParams struct)
    this.gi_params_data = new Float32Array([
      0, // screen_probe_size
      0, // screen_ray_count
      0, // world_cache_size
      0, // world_cache_cell_size
      0, // total_screen_probes
      0, // frame_index
      0, // indirect_boost
      0, // upscale_x
      0, // upscale_y
      0, // world_cache_lod_count
      0, // trace_rate
      0, // padding
    ]);
  }

  /**
   * Adds global-illumination passes to the render graph and exposes the final
   * indirect-lighting texture via `final_gi_texture`.
   *
   * @param {RenderGraph} render_graph - The render graph to add passes to
   * @param {number} width - Screen width
   * @param {number} height - Screen height
   * @param {string} gbuffer_position - Position GBuffer texture name
   * @param {string} gbuffer_position_prev - Previous position GBuffer texture name
   * @param {string} gbuffer_normal - Normal GBuffer texture name
   * @param {string} gbuffer_normal_prev - Previous normal GBuffer texture name
   * @param {string} gbuffer_albedo - Albedo GBuffer texture name
   * @param {string} gbuffer_smra - SMRA GBuffer texture name
   * @param {string} gbuffer_motion_emissive - Motion and emissive GBuffer texture name
   * @param {string} tlas_bvh2_bounds - TLAS BVH2 bounds buffer name
   * @param {string} tlas_bvh4_nodes - TLAS BVH4 nodes buffer name
   * @param {string} blas_atlas - BLAS atlas buffer name
   * @param {string} entity_transforms - Entity transforms buffer name
   * @param {string} mesh_asset_ids - Mesh asset IDs buffer name
   * @param {string} index_buffer - Index buffer name
   * @param {string} dense_lights - Dense lights buffer name
   * @param {string} light_count - Light count buffer name
   * @param {boolean} force_recreate - Force recreation of resources
   */
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
    tlas_bvh4_nodes,
    blas_atlas,
    entity_transforms,
    mesh_asset_ids,
    index_buffer,
    dense_lights,
    light_count,
    draw_count,
    force_recreate = false
  ) {
    // =========================================================================
    // Create GI Resources
    // =========================================================================
    let gi_output = render_graph.create_image({
      name: "gi_output",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    if (draw_count > 0) {
        this.add_probe_based_passes(
          render_graph,
          width,
          height,
          tlas_bvh2_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          mesh_asset_ids,
          index_buffer,
          dense_lights,
          light_count,
          gbuffer_position,
          gbuffer_position_prev,
          gbuffer_normal,
          gbuffer_normal_prev,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive,
          gi_output,
          force_recreate
        );
    }

    // Store references for external use
    this.final_gi_texture = gi_output;
  }

  add_probe_based_passes(
    render_graph,
    width,
    height,
    tlas_bvh2_bounds = null,
    tlas_bvh4_nodes = null,
    blas_atlas = null,
    entity_transforms = null,
    mesh_asset_ids = null,
    index_buffer = null,
    dense_lights = null,
    light_count = null,
    gbuffer_position = null,
    gbuffer_position_prev = null,
    gbuffer_normal = null,
    gbuffer_normal_prev = null,
    gbuffer_albedo = null,
    gbuffer_smra = null,
    gbuffer_motion_emissive = null,
    gi_output = null,
    force_recreate = false
  ) {
    let grid_width = Math.ceil(width / this.config.screen_probe_size);
    let grid_height = Math.ceil(height / this.config.screen_probe_size);
    let total_screen_probes = grid_width * grid_height;

    const total_cells = this.config.world_cache_size * this.config.world_cache_lod_count;

    // GI parameters buffer
    const gi_params = render_graph.create_buffer({
      name: "gi_params",
      size: this.gi_params_data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // GI counters buffer: light_count + active_probe_count + padding
    let gi_counters = render_graph.create_buffer({
      name: "gi_counters",
      size: 16, // 4 x u32 (light_count, active_probe_count, active_cache_cell_count, padding)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // World cache storage (persistent across frames)
    // Each cell: radiance+w(16) + data(16) = 32 bytes
    const world_cache = render_graph.create_buffer({
      name: "gi_world_cache",
      size: total_cells * 96,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // World cache compaction buffers
    const world_cache_active_flags = render_graph.create_buffer({
      name: "gi_world_cache_active_flags",
      size: total_cells * 4, // u32 per cell
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_prefix_sum = render_graph.create_buffer({
      name: "gi_world_cache_prefix_sum",
      size: total_cells * 4, // u32 per cell
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_block_sums = render_graph.create_buffer({
      name: "gi_world_cache_block_sums",
      size: Math.ceil(total_cells / COMPUTE_WORKGROUP_SIZE) * 4, // u32 per workgroup
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_compacted_indices = render_graph.create_buffer({
      name: "gi_world_cache_compacted_indices",
      size: total_cells * 4, // u32 per cell (worst case: all active)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_dispatch_params = render_graph.create_buffer({
      name: "gi_world_cache_dispatch_params",
      size: 12, // 3 x u32 (x, y, z)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // World cache path state for tracing from active cells
    // Each active cell traces 1 ray per frame to populate cache with indirect radiance
    const world_cache_path_state = render_graph.create_buffer({
      name: "gi_world_cache_path_state",
      size: total_cells * 13 * 4, // Same structure as WorldCachePathState (13 vec4<f32> per ray)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // =========================================================================
    // Screen Probe Textures (Ping-Pong for Temporal Reprojection)
    // =========================================================================
    // Each probe occupies screen_probe_size x screen_probe_size texels in the atlas
    // Probe grid dimensions
    const probe_atlas_width = grid_width * this.config.screen_probe_size;
    const probe_atlas_height = grid_height * this.config.screen_probe_size;

    // Ping-pong probe radiance textures
    // Each probe is a screen_probe_size x screen_probe_size octahedral atlas
    // RGBA16F: RGB = radiance, A = hit distance for reconstruction
    const screen_probe_radiance_0 = render_graph.create_image({
      name: "gi_screen_probe_radiance_0",
      format: "rgba16float",
      width: probe_atlas_width,
      height: probe_atlas_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const screen_probe_radiance_1 = render_graph.create_image({
      name: "gi_screen_probe_radiance_1",
      format: "rgba16float",
      width: probe_atlas_width,
      height: probe_atlas_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    // Probe metadata buffer: matches ScreenProbe struct
    // Each probe: state(vec4)
    const screen_probe_metadata = render_graph.create_buffer({
      name: "gi_screen_probe_metadata",
      size: total_screen_probes * 16, // 1 x vec4<f32> per probe (16 bytes)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Tile classification buffers for reprojection and patching
    // Queue of empty tiles (need new probes)
    const screen_probe_empty_tiles = render_graph.create_buffer({
      name: "gi_screen_probe_empty_tiles",
      size: total_screen_probes * 4, // u32 per tile (worst case: all empty)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Queue of override tiles (reprojected but may be reassigned)
    const screen_probe_override_tiles = render_graph.create_buffer({
      name: "gi_screen_probe_override_tiles",
      size: total_screen_probes * 4, // u32 per tile (worst case: all override)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Tile classification counters: empty_count, override_count
    const screen_probe_tile_counters = render_graph.create_buffer({
      name: "gi_screen_probe_tile_counters",
      size: 16, // 4 x u32 (empty_count, override_count, padding, padding)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const max_probe_rays = total_screen_probes * this.config.screen_ray_count;
    let probe_path_state = render_graph.create_buffer({
      name: "gi_probe_path_state",
      size: max_probe_rays * 14 * 4, // 14 vec4<f32> per ray
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Determine which probe radiance textures are current vs previous based on ping-pong flip
    // On frame N, we read from the texture we wrote to on frame N-1
    const ping_pong_frame = SharedFrameInfoBuffer.get_frame_index() % 2;
    const probe_radiance_prev = screen_probe_radiance_0;
    const probe_radiance_curr = screen_probe_radiance_1;

    // Get material resources
    const params_gpu = MaterialAllocationTable.params_buffer;
    const material_palette = MaterialAllocationTable.palette_buffer;
    const material_palette_offsets = EntityManager.get_fragment_gpu_buffer(
      StaticMeshFragment,
      material_offsets_name
    );

    const params_gpu_buffer = render_graph.register_buffer(params_gpu.config.name);
    const material_palette_buffer = render_graph.register_buffer(material_palette.config.name);
    const material_palette_offsets_buffer = render_graph.register_buffer(
      material_palette_offsets.buffer.config.name
    );

    // Get texture pools
    const default_texture = Texture.default_array();
    const albedo_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_albedo_name);
    const normal_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_normal_name);
    const roughness_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_roughness_name);
    const metallic_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_metallic_name);
    const ao_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_ao_name);
    const height_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_height_name);
    const specular_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_specular_name);
    const emission_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_emission_name);

    const default_texture_buffer = render_graph.register_image(default_texture.config.name);
    const albedo_pool_buffer = albedo_pool
      ? render_graph.register_image(albedo_pool.config.name)
      : default_texture_buffer;
    const normal_pool_buffer = normal_pool
      ? render_graph.register_image(normal_pool.config.name)
      : default_texture_buffer;
    const roughness_pool_buffer = roughness_pool
      ? render_graph.register_image(roughness_pool.config.name)
      : default_texture_buffer;
    const metallic_pool_buffer = metallic_pool
      ? render_graph.register_image(metallic_pool.config.name)
      : default_texture_buffer;
    const ao_pool_buffer = ao_pool
      ? render_graph.register_image(ao_pool.config.name)
      : default_texture_buffer;
    const height_pool_buffer = height_pool
      ? render_graph.register_image(height_pool.config.name)
      : default_texture_buffer;
    const specular_pool_buffer = specular_pool
      ? render_graph.register_image(specular_pool.config.name)
      : default_texture_buffer;
    const emission_pool_buffer = emission_pool
      ? render_graph.register_image(emission_pool.config.name)
      : default_texture_buffer;

    // Environment data
    const skydome_data = SharedEnvironmentData.get_skydome_data();
    const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

    const skybox = SharedEnvironmentData.get_skybox();
    const skybox_texture_buffer = render_graph.register_image(skybox.config.name);

    render_graph.add_pass(
      "gi_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const gi_params_buf = graph.get_physical_buffer(gi_params);

        // Upload GI parameters
        const frame_index = SharedFrameInfoBuffer.get_frame_index();
        const trace_rate = this.config.upscale_x * this.config.upscale_y;

        this.gi_params_data[0] = this.config.screen_probe_size;
        this.gi_params_data[1] = this.config.screen_ray_count;
        this.gi_params_data[2] = this.config.world_cache_size;
        this.gi_params_data[3] = this.config.world_cache_cell_size;
        this.gi_params_data[4] = total_screen_probes; // Derived from grid dimensions
        this.gi_params_data[5] = frame_index;
        this.gi_params_data[6] = this.config.indirect_boost;
        this.gi_params_data[7] = this.config.upscale_x;
        this.gi_params_data[8] = this.config.upscale_y;
        this.gi_params_data[9] = this.config.world_cache_lod_count;
        this.gi_params_data[10] = trace_rate;
        this.gi_params_data[11] = 0.0;
        gi_params_buf.write_raw(this.gi_params_data);
      }
    );

    render_graph.add_pass(
      "gi_reset",
      RenderPassFlags.Compute,
      {
        inputs: [gi_counters, screen_probe_tile_counters, light_count],
        outputs: [gi_counters, screen_probe_tile_counters],
        shader_setup: gi_reset_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1); // Single thread does the reset
      }
    );

    render_graph.add_pass(
      "gi_world_cache_evict",
      RenderPassFlags.Compute,
      {
        inputs: [world_cache],
        outputs: [world_cache],
        shader_setup: world_cache_evict_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "gi_world_cache_active_mark",
      RenderPassFlags.Compute,
      {
        inputs: [world_cache, world_cache_active_flags],
        outputs: [world_cache_active_flags],
        shader_setup: world_cache_compact_mark_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "gi_world_cache_active_prefix_sum",
      RenderPassFlags.Compute,
      {
        inputs: [world_cache_active_flags, world_cache_prefix_sum, world_cache_block_sums],
        outputs: [world_cache_prefix_sum, world_cache_block_sums],
        shader_setup: world_cache_compact_prefix_sum_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "gi_world_cache_active_compact",
      RenderPassFlags.Compute,
      {
        inputs: [
          world_cache_active_flags,
          world_cache_prefix_sum,
          world_cache_block_sums,
          world_cache_compacted_indices,
          world_cache_dispatch_params,
          gi_counters,
        ],
        outputs: [world_cache_compacted_indices, world_cache_dispatch_params],
        shader_setup: world_cache_compact_scatter_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "gi_world_cache_trace_init_and_sample_lights",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          world_cache,
          world_cache_compacted_indices,
          world_cache_dispatch_params,
          world_cache_path_state,
          light_count,
          dense_lights,
          gi_counters,
        ],
        outputs: [world_cache_path_state],
        shader_setup: world_cache_trace_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const dispatch_buffer = graph.get_physical_buffer(world_cache_dispatch_params);
        // Indirect dispatch based on number of active cells
        pass.dispatch_indirect(dispatch_buffer, 0);
      }
    );

    render_graph.add_pass(
      "gi_world_cache_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          world_cache_path_state,
          tlas_bvh2_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          index_buffer,
          mesh_asset_ids,
          gi_counters,
        ],
        outputs: [world_cache_path_state],
        shader_setup: world_cache_trace_hit_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const dispatch_buffer = graph.get_physical_buffer(world_cache_dispatch_params);
        // Indirect dispatch based on number of active cells
        pass.dispatch_indirect(dispatch_buffer, 0);
      }
    );

    render_graph.add_pass(
      "gi_world_cache_trace_shade",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          skydome_data_buffer,
          world_cache,
          world_cache_compacted_indices,
          world_cache_path_state,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          dense_lights,
          gi_counters,
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
        outputs: [world_cache_path_state, world_cache],
        shader_setup: world_cache_trace_shade_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const dispatch_buffer = graph.get_physical_buffer(world_cache_dispatch_params);
        // Indirect dispatch based on number of active cells
        pass.dispatch_indirect(dispatch_buffer, 0);
      }
    );

    // =========================================================================
    // Screen Probe Reprojection Phase
    // =========================================================================

    // Reprojection pass: Algorithm 1 from GI-1.0 paper
    // For each tile, attempt to reproject a probe from previous frame
    // Outputs: classified tiles into empty_tiles and override_tiles queues
    render_graph.add_pass(
      `gi_screen_probe_reproject_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          probe_radiance_prev,
          screen_probe_metadata,
          gbuffer_position,
          gbuffer_position_prev,
          gbuffer_normal,
          gbuffer_normal_prev,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive,
          screen_probe_empty_tiles,
          screen_probe_override_tiles,
          screen_probe_tile_counters,
          probe_radiance_curr,
        ],
        outputs: [
          probe_radiance_curr,
          screen_probe_metadata,
          screen_probe_empty_tiles,
          screen_probe_override_tiles,
          screen_probe_tile_counters,
        ],
        shader_setup: screen_probe_reproject_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // Dispatch one workgroup per tile
        pass.dispatch(grid_width, grid_height, 1);
      }
    );

    // Patch pass: Algorithm 2 from GI-1.0 paper
    // Reassign some override tiles to fill empty tiles (fixed ray budget)
    // Dispatches one thread per override tile for parallel processing
    render_graph.add_pass(
      "gi_screen_probe_patch",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          screen_probe_empty_tiles,
          screen_probe_override_tiles,
          screen_probe_tile_counters,
        ],
        outputs: [
          screen_probe_empty_tiles,
          screen_probe_override_tiles,
          screen_probe_tile_counters,
        ],
        shader_setup: screen_probe_patch_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // Dispatch based on maximum possible override tiles (worst case: all probes)
        // The shader will bounds-check against actual override_count
        pass.dispatch(Math.ceil(total_screen_probes / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // Spawn pass: Generate new probes for tiles that need them
    // Uses Halton jitter within spawn tiles for temporal upscaling
    render_graph.add_pass(
      "gi_screen_probe_spawn",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          world_cache,
          screen_probe_metadata,
          screen_probe_empty_tiles,
          screen_probe_tile_counters,
          gbuffer_position,
          gbuffer_normal,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive,
        ],
        outputs: [
          gi_counters,
          screen_probe_metadata,
        ],
        shader_setup: screen_probe_spawn_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // Dispatch based on number of tiles that need new probes
        pass.dispatch(Math.ceil(total_screen_probes / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // =========================================================================
    // Screen Probe Ray Tracing Phase
    // =========================================================================
    
    render_graph.add_pass(
      `gi_probe_trace_init_and_sample_lights_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          screen_probe_metadata,
          probe_radiance_prev,
          probe_path_state,
          light_count,
          dense_lights,
          world_cache,
          gbuffer_position,
          gbuffer_normal,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive,
        ],
        outputs: [probe_path_state],
        shader_setup: screen_probe_trace_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const max_rays = total_screen_probes * this.config.screen_ray_count;
        pass.dispatch(Math.ceil(max_rays / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      `gi_probe_trace_hit`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          probe_path_state,
          tlas_bvh2_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          index_buffer,
          mesh_asset_ids,
        ],
        outputs: [probe_path_state],
        shader_setup: screen_probe_trace_hit_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const max_rays = total_screen_probes * this.config.screen_ray_count;
        pass.dispatch(Math.ceil(max_rays / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      `gi_probe_trace_shade`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          skydome_data_buffer,
          gi_counters,
          probe_path_state,
          world_cache,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          dense_lights,
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
        outputs: [probe_path_state],
        shader_setup: screen_probe_trace_shade_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const max_rays = total_screen_probes * this.config.screen_ray_count;
        pass.dispatch(Math.ceil(max_rays / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // Create a third texture for update pass output to avoid read/write conflict
    const probe_radiance_updated = screen_probe_radiance_0;
    
    render_graph.add_pass(
      `gi_screen_probe_update_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          screen_probe_metadata,
          probe_radiance_curr, // Read from curr (reprojection wrote here) - binding 3
          probe_path_state,
          world_cache,
          gbuffer_position,
          gbuffer_normal,
          probe_radiance_updated, // Write to updated (flips back to match next frame's prev) - binding 8
        ],
        outputs: [
          probe_radiance_updated,
          screen_probe_metadata,
          world_cache,
        ],
        shader_setup: screen_probe_update_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(total_screen_probes / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      `gi_reconstruct_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          screen_probe_metadata,
          probe_radiance_updated, // Read from update pass output
          gbuffer_position,
          gbuffer_normal,
          gbuffer_albedo,
          gi_output,
        ],
        outputs: [gi_output],
        shader_setup: screen_probe_reconstruct_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    // Store references for external use and debugging
    this.gi_params = gi_params;
    this.gi_counters = gi_counters;
    this.screen_probe_radiance_0 = screen_probe_radiance_0;
    this.screen_probe_radiance_1 = screen_probe_radiance_1;
    this.screen_probe_metadata = screen_probe_metadata;
    this.world_cache = world_cache;
  }

  add_debug_passes(
    render_graph,
    width,
    height,
    main_position_image,
    main_normal_image,
    post_lighting_image_desc,
    debug_view,
    force_recreate = false
  ) {
    let grid_width = Math.ceil(width / this.config.screen_probe_size);
    let grid_height = Math.ceil(height / this.config.screen_probe_size);
    let total_screen_probes = grid_width * grid_height;

    // Debug visualization texture for screen probes
    this.debug_texture = render_graph.create_image({
      name: "gi_debug_probes",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    if (debug_view === DebugDrawType.GI_ScreenProbes) {
      // Get updated probe radiance texture for debugging (output of update pass)
      const ping_pong_frame = SharedFrameInfoBuffer.get_frame_index() % 2;
      const probe_radiance_final = this.screen_probe_radiance_0;
      
      render_graph.add_pass(
        `gi_debug_probes_composite_${ping_pong_frame}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            this.gi_params,
            this.gi_counters,
            this.screen_probe_metadata,
            probe_radiance_final,
            main_position_image,
            main_normal_image,
            post_lighting_image_desc,
            this.debug_texture,
          ],
          outputs: [this.debug_texture],
          shader_setup: screen_probe_debug_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          // Dispatch one thread per probe (workgroup size = 64)
          pass.dispatch(Math.ceil(total_screen_probes / 64), 1, 1);
        }
      );
    } else if (debug_view === DebugDrawType.GI_WorldCache) {
      // World cache debug visualization
      render_graph.add_pass(
        "gi_debug_world_cache_composite",
        RenderPassFlags.Compute,
        {
          inputs: [
            this.gi_params,
            this.world_cache,
            main_position_image,
            main_normal_image,
            post_lighting_image_desc,
            this.debug_texture,
          ],
          outputs: [this.debug_texture],
          shader_setup: world_cache_debug_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          // Dispatch per pixel (workgroup size = 8x8)
          pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
        }
      );
    }
    return this.debug_texture;
  }

  /**
   * Update configuration parameters at runtime
   */
  set_config(new_config) {
    this.config = { ...this.config, ...new_config };
  }

  /**
   * Force reset of all caches (useful for teleports, scene changes)
   */
  reset() { }
}
