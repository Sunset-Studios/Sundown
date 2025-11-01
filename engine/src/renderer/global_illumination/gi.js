/**
 * GI-1.0 Radiance Caching System
 *
 * Implements a two-level radiance caching scheme following the GI-1.0 paper (Algorithm 1):
 * - Screen Cache: Fixed grid of probes on primary surfaces with temporal reuse
 * - World Cache: Spatial hash-based cache for secondary bounce radiance
 *
 * Screen Probe Algorithm (per tile, per frame):
 * 1. Temporal Upscaling: Only fraction of tiles update each frame (e.g., 1/4)
 * 2. Reprojection: Use motion vectors to find previous probe
 *    - Each pixel in tile tracks backward to find which probe it came from
 *    - Validate: plane_distance < cell_size && normal_dot > 0.95
 *    - Threads compete atomically for best match (closest 3D distance)
 * 3. If reprojection succeeds: Reuse probe data (keep accumulated radiance!)
 * 4. If reprojection fails: Spawn new probe using Halton jitter, reset radiance
 * 5. Ray tracing: Trace rays from probe position with temporal jittering
 * 6. Accumulation: Weighted average of ray throughput over time
 *    - Reprojected probes: maintain history for progressive refinement
 *    - New spawns: start fresh to avoid ghosting
 *    - Sample count clamped to ~32 for responsiveness
 * 7. Gradual convergence as all tiles fill in over multiple frames
 *
 * Features:
 * - ReSTIR-based path sampling for high-quality convergence
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

const material_offsets_name = "material_table_offset";
const texture_pool_albedo_name = Name.from("texture_pool_albedo");
const texture_pool_normal_name = Name.from("texture_pool_normal");
const texture_pool_roughness_name = Name.from("texture_pool_roughness");
const texture_pool_metallic_name = Name.from("texture_pool_metallic");
const texture_pool_ao_name = Name.from("texture_pool_ao");
const texture_pool_height_name = Name.from("texture_pool_height");
const texture_pool_specular_name = Name.from("texture_pool_specular");
const texture_pool_emission_name = Name.from("texture_pool_emission");

// Shader configurations
const gi_reset_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gi_reset.wgsl" },
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

const screen_probe_trace_hit_visibility_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/screen_probe_trace_hit_visibility.wgsl" },
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

export class GI {
  final_gi_texture = null;
  debug_probe_texture = null;
  total_screen_probes = 0; // Calculated from resolution and probe size

  // Configuration parameters
  config = {
    screen_probe_size: 4, // Side length of probe footprint in pixels
    screen_ray_count: 1, // Rays per screen probe
    upscale_x: 2, // Temporal upscale factor X (2x2 = 4 frames to fill)
    upscale_y: 2, // Temporal upscale factor Y
    cell_size_heuristic: 0.5, // Spatial error tolerance for probe reuse (world units)
    world_cache_size: 65536, // Number of world cache cells (64K)
    world_cache_cell_size: 1.0, // Size of world cache cells in world units
    max_bounces: 1, // Maximum path bounces
    indirect_boost: 1.0, // Multiplier for indirect lighting
    reset_caches: false, // Force reset all caches
  };

  constructor(params = {}) {
    // Override defaults with provided parameters
    this.config = { ...this.config, ...params };

    // GI parameters buffer (must match shader GIParams struct)
    this.gi_params_data = new Float32Array([
      0, // screen_probe_size,
      0, // screen_ray_count,
      0, // world_cache_size,
      0, // total_screen_probes,
      0, // frame_index,
      0, // reset_caches,
      0, // indirect_boost,
      0, // upscale_x,
      0, // upscale_y,
      0, // cell_size_heuristic,
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
   * @param {string} gbuffer_normal - Normal GBuffer texture name
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
    gbuffer_normal,
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
    // Calculate total_screen_probes based on grid dimensions
    // =========================================================================
    // Grid dimensions = ceil(resolution / probe_size)
    // Example: 720p with 8x8 probes = (160 x 90) = 14,400 probes
    const grid_width = Math.ceil(width / this.config.screen_probe_size);
    const grid_height = Math.ceil(height / this.config.screen_probe_size);
    this.total_screen_probes = grid_width * grid_height;

    // =========================================================================
    // Create GI Resources
    // =========================================================================
    const gi_params = render_graph.create_buffer({
      name: "gi_params",
      size: this.gi_params_data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Screen probe storage (persistent across frames)
    // Each probe: position+radius(16) + normal+frame(16) + radiance+m(16) + albedo+roughness(16) + material_props(16) + state(16) = 96 bytes
    const screen_probes = render_graph.create_buffer({
      name: "gi_screen_probes",
      size: this.total_screen_probes * 96,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
      persistent: true, // Keep across frames
    });

    // GI counters buffer: light_count + active_probe_count + padding
    const gi_counters = render_graph.create_buffer({
      name: "gi_counters",
      size: 16, // 4 x u32 (light_count, active_probe_count, padding0, padding1)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // World cache storage (persistent across frames)
    // Each cell: position+frame(16) + normal+count(16) + radiance+w(16) + data(16) = 64 bytes
    const world_cache = render_graph.create_buffer({
      name: "gi_world_cache",
      size: this.config.world_cache_size * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
      persistent: true, // Keep across frames
    });

    // Probe path state (rays from screen probes)
    const max_probe_rays = this.total_screen_probes * this.config.screen_ray_count;
    const probe_path_state = render_graph.create_buffer({
      name: "gi_probe_path_state",
      size: max_probe_rays * 12 * 4, // 12 vec4<f32> per ray
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Probe path shade (shading state with ReSTIR)
    const probe_path_shade = render_graph.create_buffer({
      name: "gi_probe_path_shade",
      size: max_probe_rays * 8 * 4, // 2 vec4<f32> per ray
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // Final output texture
    const gi_output = render_graph.create_image({
      name: "gi_output",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    if (draw_count > 0) {
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
      const roughness_pool = ResourceCache.get().fetch(
        CacheTypes.IMAGE,
        texture_pool_roughness_name
      );
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

      // =========================================================================
      // Pass 0: Reset counters and upload parameters
      // =========================================================================
      render_graph.add_pass(
        "gi_upload_params",
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          const gi_params_buf = graph.get_physical_buffer(gi_params);

          // Upload GI parameters
          this.gi_params_data[0] = this.config.screen_probe_size;
          this.gi_params_data[1] = this.config.screen_ray_count;
          this.gi_params_data[2] = this.config.world_cache_size;
          this.gi_params_data[3] = this.total_screen_probes; // Derived from grid dimensions
          this.gi_params_data[4] = SharedFrameInfoBuffer.get_frame_index();
          this.gi_params_data[5] = this.config.reset_caches ? 1 : 0;
          this.gi_params_data[6] = this.config.indirect_boost;
          this.gi_params_data[7] = this.config.upscale_x;
          this.gi_params_data[8] = this.config.upscale_y;
          this.gi_params_data[9] = this.config.cell_size_heuristic;
          gi_params_buf.write_raw(this.gi_params_data);

          // Clear reset flag after use
          this.config.reset_caches = false;
        }
      );

      // =========================================================================
      // Pass 1: Reset counters (GPU compute shader)
      // =========================================================================
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
          pass.dispatch(1, 1, 1); // Single thread does the reset
        }
      );

      // =========================================================================
      // Pass 2: Spawn Screen Probes (Motion vector reprojection + Halton spawn)
      // - Each pixel in tile uses motion vectors to track backward in time
      // - Finds which probe contained that geometry in previous frame
      // - Validates probe: plane_distance < cell_size && normal_dot > 0.95
      // - If valid: reproject probe (keep accumulated radiance!)
      // - If invalid: spawn new probe using Halton jitter (reset radiance)
      // =========================================================================
      render_graph.add_pass(
        "gi_screen_probe_spawn",
        RenderPassFlags.Compute,
        {
          inputs: [
            gi_params,
            gi_counters,
            screen_probes,
            gbuffer_position,
            gbuffer_normal,
            gbuffer_albedo,
            gbuffer_smra,
            gbuffer_motion_emissive,
          ],
          outputs: [gi_counters, screen_probes],
          shader_setup: screen_probe_spawn_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(grid_width, grid_height, 1);
        }
      );

      // =========================================================================
      // Pass 3-N: Screen Probe Path Tracing (Init, Hit, Shade for each bounce)
      // =========================================================================
      render_graph.add_pass(
        "gi_probe_trace_init",
        RenderPassFlags.Compute,
        {
          inputs: [
            gi_params,
            gi_counters,
            screen_probes,
            probe_path_state,
            probe_path_shade,
            light_count,
            dense_lights,
          ],
          outputs: [probe_path_state, probe_path_shade],
          shader_setup: screen_probe_trace_init_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const max_rays = this.total_screen_probes * this.config.screen_ray_count;
          pass.dispatch(Math.ceil(max_rays / 128), 1, 1);
        }
      );

      for (let bounce = 0; bounce < this.config.max_bounces; bounce++) {
        render_graph.add_pass(
          `gi_probe_trace_hit_visibility_${bounce}`,
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
            shader_setup: screen_probe_trace_hit_visibility_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const max_rays = this.total_screen_probes * this.config.screen_ray_count;
            pass.dispatch(Math.ceil(max_rays / 128), 1, 1);
          }
        );

        render_graph.add_pass(
          `gi_probe_trace_hit_${bounce}`,
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
            const max_rays = this.total_screen_probes * this.config.screen_ray_count;
            pass.dispatch(Math.ceil(max_rays / 128), 1, 1);
          }
        );

        const shade_inputs = [
          gi_params,
          skydome_data_buffer,
          gi_counters,
          probe_path_state,
          probe_path_shade,
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
        ];

        render_graph.add_pass(
          `gi_probe_trace_shade_${bounce}`,
          RenderPassFlags.Compute,
          {
            inputs: shade_inputs,
            outputs: [probe_path_state, probe_path_shade],
            shader_setup: screen_probe_trace_shade_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const max_rays = this.total_screen_probes * this.config.screen_ray_count;
            pass.dispatch(Math.ceil(max_rays / 128), 1, 1);
          }
        );
      }

      // =========================================================================
      // Pass N+2: Update Screen Probes (accumulate ray radiance and world cache)
      // - Performs weighted averaging of radiance over time
      // - New spawns start fresh; reprojected probes maintain history
      // - Sample count is clamped to prevent overflow and ensure responsiveness
      // =========================================================================
      render_graph.add_pass(
        "gi_screen_probe_update",
        RenderPassFlags.Compute,
        {
          inputs: [
            gi_params,
            gi_counters,
            screen_probes,
            probe_path_state,
            probe_path_shade,
            world_cache,
          ],
          outputs: [screen_probes, world_cache],
          shader_setup: screen_probe_update_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(this.total_screen_probes / 128), 1, 1);
        }
      );

      // =========================================================================
      // Pass N+3: Reconstruct Final Radiance (interpolate from probes)
      // =========================================================================
      // render_graph.add_pass(
      //   "gi_reconstruct",
      //   RenderPassFlags.Compute,
      //   {
      //     inputs: [
      //       gi_params,
      //       gi_counters,
      //       screen_probes,
      //       gbuffer_position,
      //       gbuffer_normal,
      //       gbuffer_albedo,
      //       gi_output,
      //     ],
      //     outputs: [gi_output],
      //     shader_setup: screen_probe_reconstruct_shader_setup,
      //   },
      //   (graph, frame_data, encoder) => {
      //     const pass = graph.get_physical_pass(frame_data.current_pass);
      //     pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      //   }
      // );
    }

    // Store references for external use
    this.gi_params = gi_params;
    this.gi_counters = gi_counters;
    this.screen_probes = screen_probes;
    this.final_gi_texture = gi_output;
  }

  add_debug_passes(
    render_graph,
    width,
    height,
    main_position_image,
    main_normal_image,
    post_lighting_image_desc,
    force_recreate = false
  ) {
    // Debug visualization texture
    this.debug_probe_texture = render_graph.create_image({
      name: "gi_debug_probes",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    render_graph.add_pass(
      "gi_debug_probes_composite",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.gi_params,
          this.gi_counters,
          this.screen_probes,
          main_position_image,
          main_normal_image,
          post_lighting_image_desc,
          this.debug_probe_texture,
        ],
        outputs: [this.debug_probe_texture],
        shader_setup: screen_probe_debug_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // Dispatch one thread per probe (workgroup size = 64)
        pass.dispatch(Math.ceil(this.total_screen_probes / 64), 1, 1);
      }
    );

    return this.debug_probe_texture;
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
  reset() {
    this.config.reset_caches = true;
  }
}
