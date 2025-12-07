/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                     PER-PIXEL PATH TRACED GI SYSTEM                       ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  A simplified, high-quality global illumination system that traces rays  ║
 * ║  directly from each screen pixel. Uses a two-level approach:             ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
 * ║  │  LEVEL 1: Per-Pixel First Bounce                                    │  ║
 * ║  │  ─────────────────────────────────                                  │  ║
 * ║  │  • Each pixel traces 1+ rays per frame                              │  ║
 * ║  │  • BRDF-importance sampled ray directions                           │  ║
 * ║  │  • ReSTIR-based path sampling for quality                           │  ║
 * ║  │  • Temporal accumulation with adaptive blending                     │  ║
 * ║  └─────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
 * ║  │  LEVEL 2: World Cache (Multi-Bounce Irradiance)                     │  ║
 * ║  │  ──────────────────────────────────────────────                     │  ║
 * ║  │  • Spatial hash-based radiance cache                                │  ║
 * ║  │  • Provides cached irradiance at ray hit points                     │  ║
 * ║  │  • Temporally recurrent for multi-bounce propagation                │  ║
 * ║  │  • LOD-adaptive quantization for view-dependent quality             │  ║
 * ║  └─────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                           ║
 * ║  Pipeline Flow:                                                           ║
 * ║  ═════════════                                                            ║
 * ║                                                                           ║
 * ║  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 ║
 * ║  │ World Cache  │───▶│ World Cache  │───▶│ World Cache  │                 ║
 * ║  │    Evict     │    │  Trace Init  │    │  Trace Hit   │                 ║
 * ║  └──────────────┘    └──────────────┘    └──────────────┘                 ║
 * ║          │                                      │                         ║
 * ║          ▼                                      ▼                         ║
 * ║  ┌──────────────┐                       ┌──────────────┐                  ║
 * ║  │ World Cache  │◀──────────────────────│ World Cache  │                  ║
 * ║  │   Compact    │                       │  Trace Shade │                  ║
 * ║  └──────────────┘                       └──────────────┘                  ║
 * ║          │                                                                ║
 * ║          ▼                                                                ║
 * ║  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 ║
 * ║  │ Pixel Trace  │───▶│ Pixel Trace  │───▶│ Pixel Trace  │                 ║
 * ║  │     Init     │    │     Hit      │    │    Shade     │                 ║
 * ║  └──────────────┘    └──────────────┘    └──────────────┘                 ║
 * ║                                                 │                         ║
 * ║                                                 ▼                         ║
 * ║                                         ┌──────────────┐                  ║
 * ║                                         │ Pixel Update │                  ║
 * ║                                         │ (Accumulate) │                  ║
 * ║                                         └──────────────┘                  ║
 * ║                                                 │                         ║
 * ║                                                 ▼                         ║
 * ║                                         ┌──────────────┐                  ║
 * ║                                         │  GI Output   │                  ║
 * ║                                         └──────────────┘                  ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
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

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const COMPUTE_WORKGROUP_SIZE = 128;

// Material resource names
const material_offsets_name = "material_table_offset";
const texture_pool_albedo_name = Name.from("texture_pool_albedo");
const texture_pool_normal_name = Name.from("texture_pool_normal");
const texture_pool_roughness_name = Name.from("texture_pool_roughness");
const texture_pool_metallic_name = Name.from("texture_pool_metallic");
const texture_pool_ao_name = Name.from("texture_pool_ao");
const texture_pool_height_name = Name.from("texture_pool_height");
const texture_pool_specular_name = Name.from("texture_pool_specular");
const texture_pool_emission_name = Name.from("texture_pool_emission");

// ═══════════════════════════════════════════════════════════════════════════
// SHADER CONFIGURATIONS
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// System Shaders
// ─────────────────────────────────────────────────────────────────────────────

const gi_reset_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gi_reset.wgsl" },
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// World Cache Shaders
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Per-Pixel Path Tracing Shaders
// ─────────────────────────────────────────────────────────────────────────────

const pixel_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_trace_init.wgsl" },
  },
};

const pixel_trace_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_trace_hit.wgsl" },
  },
};

const pixel_trace_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_trace_shade.wgsl" },
  },
};

const pixel_accumulate_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_accumulate.wgsl" },
  },
};

const pixel_blur_denoise_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_blur_denoise.wgsl" },
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Debug Shaders
// ─────────────────────────────────────────────────────────────────────────────

const world_cache_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_debug.wgsl" },
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// GI SYSTEM CLASS
// ═══════════════════════════════════════════════════════════════════════════

export class GI {
  // Output textures
  final_gi_texture = null;
  debug_texture = null;

  // ─────────────────────────────────────────────────────────────────────────
  // Configuration Parameters
  // ─────────────────────────────────────────────────────────────────────────
  config = {
    screen_ray_count: 1,          // Rays per pixel per frame (1 recommended for real-time)
    upscale_x: 4,                 // Temporal upscale factor X
    upscale_y: 4,                 // Temporal upscale factor Y
    world_cache_size: 32768,      // Number of world cache cells per LOD level
    world_cache_cell_size: 4.0,   // Base cell size in world units
    world_cache_lod_count: 4,     // Number of LOD levels
    indirect_boost: 1.0,         // Multiplier for indirect lighting contribution
  };

  // GI parameters buffer data (matches shader GIParams struct)
  gi_params_data = new Float32Array([
    0,    // screen_ray_count
    0,    // world_cache_size
    0,    // world_cache_cell_size
    0,    // total_pixels
    0,    // frame_index
    0,    // indirect_boost
    0,    // upscale_x
    0,    // upscale_y
    0,    // world_cache_lod_count
    0,    // resolution_x
    0,    // resolution_y
    0,    // padding
  ]);

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────
  constructor(params = {}) {
    // Override defaults with provided parameters
    this.config = { ...this.config, ...params };
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MAIN PASS SETUP
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Adds global-illumination passes to the render graph.
   * Exposes the final indirect-lighting texture via `final_gi_texture`.
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
    // ─────────────────────────────────────────────────────────────────────
    // Create GI Output Texture
    // ─────────────────────────────────────────────────────────────────────
    let gi_output = render_graph.create_image({
      name: "gi_output",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    // Only add GI passes if there are drawable objects
    if (draw_count > 0) {
      this.add_per_pixel_gi_passes(
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

    // Store reference for external use
    this.final_gi_texture = gi_output;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PER-PIXEL GI PASSES
  // ═══════════════════════════════════════════════════════════════════════════

  add_per_pixel_gi_passes(
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
  ) {
    // ─────────────────────────────────────────────────────────────────────
    // Calculate Dimensions
    // ─────────────────────────────────────────────────────────────────────
    const total_pixels = width * height;
    const total_cells = this.config.world_cache_size * this.config.world_cache_lod_count;

    // Tile-based dispatch: only trace one pixel per tile per frame
    // Each tile is upscale_x × upscale_y pixels
    const tile_grid_width = Math.ceil(width / this.config.upscale_x);
    const tile_grid_height = Math.ceil(height / this.config.upscale_y);
    const tiles_per_frame = tile_grid_width * tile_grid_height;
    const rays_per_frame = tiles_per_frame * this.config.screen_ray_count;

    // ═════════════════════════════════════════════════════════════════════
    // CREATE GPU RESOURCES
    // ═════════════════════════════════════════════════════════════════════

    // ─────────────────────────────────────────────────────────────────────
    // Blue Noise Texture
    // ─────────────────────────────────────────────────────────────────────
    const blue_noise = Texture.default_blue_noise();
    const blue_noise_image = render_graph.register_image(blue_noise.config.name);

    // ─────────────────────────────────────────────────────────────────────
    // GI Parameters Buffer
    // ─────────────────────────────────────────────────────────────────────
    const gi_params = render_graph.create_buffer({
      name: "gi_params",
      size: this.gi_params_data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────
    // GI Counters Buffer
    // 6 fields: light_count, active_cache_cell_count, ray_queue_shadow_head,
    //           ray_queue_primary_head, ray_queue_count, padding
    // ─────────────────────────────────────────────────────────────────────
    let gi_counters = render_graph.create_buffer({
      name: "gi_counters",
      size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────
    // World Cache Storage
    // ─────────────────────────────────────────────────────────────────────
    const world_cache = render_graph.create_buffer({
      name: "gi_world_cache",
      size: total_cells * 96,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // World cache compaction buffers
    const world_cache_active_flags = render_graph.create_buffer({
      name: "gi_world_cache_active_flags",
      size: total_cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_prefix_sum = render_graph.create_buffer({
      name: "gi_world_cache_prefix_sum",
      size: total_cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_block_sums = render_graph.create_buffer({
      name: "gi_world_cache_block_sums",
      size: Math.ceil(total_cells / COMPUTE_WORKGROUP_SIZE) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_compacted_indices = render_graph.create_buffer({
      name: "gi_world_cache_compacted_indices",
      size: total_cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_dispatch_params = render_graph.create_buffer({
      name: "gi_world_cache_dispatch_params",
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_path_state = render_graph.create_buffer({
      name: "gi_world_cache_path_state",
      size: total_cells * 11 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────
    // Per-Pixel Path State (sized for rays traced per frame, not all pixels)
    // ─────────────────────────────────────────────────────────────────────
    let pixel_path_state = render_graph.create_buffer({
      name: "gi_pixel_path_state",
      size: rays_per_frame * 13 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Work queue used for persistent threads in the path tracing passes
    let pixel_ray_queue = render_graph.create_buffer({
      name: "gi_pixel_ray_queue",
      size: rays_per_frame * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────
    // Per-Pixel Radiance (Ping-Pong) - BLURRED Output
    // These contain the BLURRED radiance from the recurrent blur pass.
    // pixel_radiance_prev is what pixel_accumulate reads as history, ensuring
    // the temporal accumulation sees the clean blurred background.
    // ─────────────────────────────────────────────────────────────────────
    const pixel_radiance_0 = render_graph.create_image({
      name: "gi_pixel_radiance_0",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const pixel_radiance_1 = render_graph.create_image({
      name: "gi_pixel_radiance_1",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────
    // Raw Accumulation Buffer (Temporary)
    // pixel_accumulate writes raw temporal accumulation here, then recurrent_blur
    // reads this (center) + pixel_radiance_prev (neighbors) and writes the
    // blurred result to pixel_radiance_curr.
    // ─────────────────────────────────────────────────────────────────────
    const raw_accumulation = render_graph.create_image({
      name: "gi_raw_accumulation",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    // Ping-pong selection based on frame index
    const ping_pong_frame = SharedFrameInfoBuffer.get_frame_index() % 2;
    const pixel_radiance_prev = ping_pong_frame === 0 ? pixel_radiance_0 : pixel_radiance_1;
    const pixel_radiance_curr = ping_pong_frame === 0 ? pixel_radiance_1 : pixel_radiance_0;

    // ─────────────────────────────────────────────────────────────────────
    // Get Material Resources
    // ─────────────────────────────────────────────────────────────────────
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

    // ─────────────────────────────────────────────────────────────────────
    // Get Texture Pools
    // ─────────────────────────────────────────────────────────────────────
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

    // ─────────────────────────────────────────────────────────────────────
    // Get Environment Data
    // ─────────────────────────────────────────────────────────────────────
    const skydome_data = SharedEnvironmentData.get_skydome_data();
    const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

    const skybox = SharedEnvironmentData.get_skybox();
    const skybox_texture_buffer = render_graph.register_image(skybox.config.name);

    // ═════════════════════════════════════════════════════════════════════
    // ADD RENDER PASSES
    // ═════════════════════════════════════════════════════════════════════

    // ─────────────────────────────────────────────────────────────────────
    // Pass 0: Upload GI Parameters
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "gi_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const gi_params_buf = graph.get_physical_buffer(gi_params);

        const frame_index = SharedFrameInfoBuffer.get_frame_index();

        // Fill parameter buffer
        this.gi_params_data[0] = this.config.screen_ray_count;
        this.gi_params_data[1] = this.config.world_cache_size;
        this.gi_params_data[2] = this.config.world_cache_cell_size;
        this.gi_params_data[3] = total_pixels;
        this.gi_params_data[4] = frame_index;
        this.gi_params_data[5] = this.config.indirect_boost;
        this.gi_params_data[6] = this.config.upscale_x;
        this.gi_params_data[7] = this.config.upscale_y;
        this.gi_params_data[8] = this.config.world_cache_lod_count;
        this.gi_params_data[9] = width;
        this.gi_params_data[10] = height;
        this.gi_params_data[11] = 0;

        gi_params_buf.write_raw(this.gi_params_data);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Pass 1: Reset Counters
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "gi_reset",
      RenderPassFlags.Compute,
      {
        inputs: [gi_counters, light_count],
        outputs: [gi_counters],
        shader_setup: gi_reset_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1);
      }
    );

    // ═════════════════════════════════════════════════════════════════════
    // WORLD CACHE PASSES
    // ═════════════════════════════════════════════════════════════════════

    // ─────────────────────────────────────────────────────────────────────
    // Pass 2: Evict Stale World Cache Cells
    // ─────────────────────────────────────────────────────────────────────
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

    // ─────────────────────────────────────────────────────────────────────
    // Pass 3: Mark Active World Cache Cells
    // ─────────────────────────────────────────────────────────────────────
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

    // ─────────────────────────────────────────────────────────────────────
    // Pass 4: Prefix Sum for Compaction
    // ─────────────────────────────────────────────────────────────────────
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

    // ─────────────────────────────────────────────────────────────────────
    // Pass 5: Compact Active Cells
    // ─────────────────────────────────────────────────────────────────────
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

    // ─────────────────────────────────────────────────────────────────────
    // Pass 6: World Cache Trace Init
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "gi_world_cache_trace_init",
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
          blue_noise_image,
        ],
        outputs: [world_cache_path_state],
        shader_setup: world_cache_trace_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const dispatch_buffer = graph.get_physical_buffer(world_cache_dispatch_params);
        pass.dispatch_indirect(dispatch_buffer, 0);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Pass 7: World Cache Trace Hit
    // Dispatches 2x rays_per_frame to run shadow and primary rays in parallel:
    //   - First half of threads: shadow ray traces (NEE visibility)
    //   - Second half of threads: primary ray traces (indirect bounce)
    // ─────────────────────────────────────────────────────────────────────
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
          mesh_asset_ids,
          gi_counters,
        ],
        outputs: [world_cache_path_state],
        shader_setup: world_cache_trace_hit_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const dispatch_buffer = graph.get_physical_buffer(world_cache_dispatch_params);
        pass.dispatch_indirect(dispatch_buffer, 0);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Pass 8: World Cache Trace Shade
    // ─────────────────────────────────────────────────────────────────────
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
        pass.dispatch_indirect(dispatch_buffer, 0);
      }
    );

    // ═════════════════════════════════════════════════════════════════════
    // PER-PIXEL PATH TRACING PASSES
    // ═════════════════════════════════════════════════════════════════════

    // ─────────────────────────────────────────────────────────────────────
    // Pass 9: Per-Pixel Trace Init (with NEE light sampling)
    // Dispatches only tiles_per_frame rays (one per tile), not all pixels
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      `gi_pixel_trace_init_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          pixel_path_state,
          pixel_ray_queue,
          light_count,
          dense_lights,
          world_cache,
          gbuffer_position,
          gbuffer_normal,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive,
          blue_noise_image,
        ],
        outputs: [pixel_path_state, pixel_ray_queue],
        shader_setup: pixel_trace_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Pass 10: Per-Pixel Trace Hit (BVH traversal)
    // Dispatches 2x rays_per_frame to run shadow and primary rays in parallel:
    //   - First half of threads: shadow ray traces (NEE visibility)
    //   - Second half of threads: primary ray traces (indirect bounce)
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "gi_pixel_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          pixel_path_state,
          pixel_ray_queue,
          tlas_bvh2_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          mesh_asset_ids,
        ],
        outputs: [pixel_path_state],
        shader_setup: pixel_trace_hit_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // 2x dispatch: first half for shadow rays, second half for primary rays
        pass.dispatch(Math.ceil((2 * rays_per_frame) / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Pass 11: Per-Pixel Trace Shade (material evaluation + world cache query)
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "gi_pixel_trace_shade",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          skydome_data_buffer,
          gi_counters,
          pixel_path_state,
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
        outputs: [pixel_path_state],
        shader_setup: pixel_trace_shade_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Pass 12: Per-Pixel Accumulate (Temporal Accumulation)
    // Reads from pixel_radiance_prev (last frame's BLURRED output) and
    // writes raw accumulated radiance to raw_accumulation buffer.
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      `gi_pixel_accumulate_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          gi_counters,
          pixel_path_state,
          pixel_radiance_prev,
          gbuffer_position,
          gbuffer_position_prev,
          gbuffer_normal,
          gbuffer_normal_prev,
          gbuffer_motion_emissive,
          gbuffer_smra,
          world_cache,
          raw_accumulation,
        ],
        outputs: [raw_accumulation, world_cache],
        shader_setup: pixel_accumulate_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // Dispatch as 8x8 tiles for better cache coherency
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Pass 13: Stabilized Recurrent Blur
    // Adaptive-radius spatial filter using sample count for stabilization
    // Based on NVIDIA's "Fast Denoising with Self-Stabilizing Recurrent Blurs"
    //
    // Key insight: Neighbors are sampled from previous frame's BLURRED output
    // (pixel_radiance_prev), not raw accumulated radiance. This allows
    // temporal redistribution of spatial sampling: 30 FPS × 8 samples =
    // 240 cumulative samples/sec due to the recurrent nature.
    //
    // Inputs:
    //   - raw_accumulation: Current frame's raw temporal accumulation
    //   - pixel_radiance_prev: Previous frame's BLURRED output (clean background)
    //
    // Outputs:
    //   - pixel_radiance_curr: Blurred output (becomes pixel_radiance_prev next frame)
    //   - gi_output: Final GI radiance for deferred lighting passes
    // ─────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      `gi_recurrent_blur_${ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          raw_accumulation,
          pixel_radiance_prev,
          gbuffer_position,
          gbuffer_position_prev,
          gbuffer_normal,
          gbuffer_normal_prev,
          pixel_radiance_curr,
          gi_output,
        ],
        outputs: [pixel_radiance_curr, gi_output],
        shader_setup: pixel_blur_denoise_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // Dispatch as 8x8 tiles for better cache coherency
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    // ─────────────────────────────────────────────────────────────────────
    // Store References for Debug Passes
    // ─────────────────────────────────────────────────────────────────────
    this.gi_params = gi_params;
    this.gi_counters = gi_counters;
    this.world_cache = world_cache;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // DEBUG PASSES
  // ═══════════════════════════════════════════════════════════════════════════

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
    // Create debug visualization texture
    this.debug_texture = render_graph.create_image({
      name: "gi_debug",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    if (debug_view === DebugDrawType.GI_WorldCache) {
      // World cache debug visualization
      render_graph.add_pass(
        "gi_debug_world_cache",
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
          pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
        }
      );
    }

    return this.debug_texture;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CONFIGURATION
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Update configuration parameters at runtime
   */
  set_config(new_config) {
    this.config = { ...this.config, ...new_config };
  }

  /**
   * Force reset of all caches (useful for teleports, scene changes)
   */
  reset() {}
}
