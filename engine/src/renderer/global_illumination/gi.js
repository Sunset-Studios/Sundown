/**
 * GI-1.0 Radiance Caching System
 * 
 * Implements a two-level radiance caching scheme inspired by the GI-1.0 paper:
 * - Screen Cache: Probes spawned on primary visible surfaces for high-detail caching
 * - World Cache: Spatial hash-based cache for secondary bounce radiance
 * 
 * Features:
 * - ReSTIR-based path sampling for high-quality convergence
 * - Separate hit and shade passes to accommodate storage buffer limits
 * - Temporal stability through probe reuse and exponential moving averages
 * - Scalable configuration for different performance targets
 * 
 * References:
 * - GI-1.0 Paper: https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf
 * - Screen Space Radiance Caching (SSRC)
 * - ReSTIR GI
 */

import { RenderPassFlags } from "../renderer_types.js";
import { SharedEnvironmentData } from "../../core/shared_data.js";
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

class GI {
  final_gi_texture = null;
  frame_index = 0;
  
  // Configuration parameters
  config = {
    // Screen cache configuration
    screen_probe_spawn_rate: 16,      // 1 in N pixels spawns a probe (16 = ~6.25% of pixels)
    screen_probe_size: 8,              // Side length of probe footprint in pixels
    screen_ray_count: 32,              // Rays per screen probe
    max_screen_probes: 8192,           // Maximum number of screen probes
    // Temporal upscale (GI-1.0 paper section 2.1.1)
    upscale_x: 2,                      // Temporal upscale factor X (2x2 = 4 frames to fill)
    upscale_y: 2,                      // Temporal upscale factor Y
    cell_size_heuristic: 0.5,          // Spatial error tolerance for probe reuse (world units)
    // World cache configuration
    world_cache_size: 65536,           // Number of world cache cells (64K)
    world_cache_cell_size: 1.0,        // Size of world cache cells in world units
    // Path tracing configuration
    max_bounces: 2,                    // Maximum path bounces
    indirect_boost: 1.0,               // Multiplier for indirect lighting
    // Temporal configuration
    temporal_alpha: 0.1,               // Blend factor for temporal filtering
    // Debug/control
    reset_caches: false,               // Force reset all caches
  };
  
  constructor(params = {}) {
    // Override defaults with provided parameters
    this.config = { ...this.config, ...params };

    // GI parameters buffer (must match shader GIParams struct)
    this.gi_params_data = new Float32Array([
      this.config.screen_probe_spawn_rate,
      this.config.screen_probe_size,
      this.config.screen_ray_count,
      this.config.world_cache_size,
      this.config.max_screen_probes,
      this.frame_index,
      this.config.reset_caches ? 1 : 0,
      this.config.indirect_boost,
      this.config.upscale_x,
      this.config.upscale_y,
      this.config.cell_size_heuristic,
      0, // padding for alignment
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
    force_recreate = false
  ) {
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
    // Each probe: position+radius(16) + normal+frame(16) + radiance+m(16) + albedo+roughness(16) + state(16) = 80 bytes
    const screen_probes = render_graph.create_buffer({
      name: "gi_screen_probes",
      size: this.config.max_screen_probes * 80,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
      persistent: true, // Keep across frames
    });
    
    // Screen probe counter (atomic counter for probe allocation)
    const screen_probe_counter = render_graph.create_buffer({
      name: "gi_screen_probe_counter",
      size: 4,
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
    const max_probe_rays = this.config.max_screen_probes * this.config.screen_ray_count;
    const probe_path_state = render_graph.create_buffer({
      name: "gi_probe_path_state",
      size: max_probe_rays * 36 * 4, // 9 vec4<f32> per ray
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    
    // Probe path shade (shading state with ReSTIR)
    const probe_path_shade = render_graph.create_buffer({
      name: "gi_probe_path_shade",
      size: max_probe_rays * 20 * 4, // 5 vec4<f32> per ray
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

    // =========================================================================
    // Pass 0: Reset counters and upload parameters
    // =========================================================================
    render_graph.add_pass(
      "gi_reset",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const gi_params_buf = graph.get_physical_buffer(gi_params);
        const counter_buf = graph.get_physical_buffer(screen_probe_counter);
        
        // Upload GI parameters
        this.gi_params_data[5] = this.frame_index;
        this.gi_params_data[6] = this.config.reset_caches ? 1 : 0;
        this.gi_params_data[7] = this.config.indirect_boost;
        this.gi_params_data[8] = this.config.upscale_x;
        this.gi_params_data[9] = this.config.upscale_y;
        this.gi_params_data[10] = this.config.cell_size_heuristic;
        gi_params_buf.write_raw(this.gi_params_data);
        
        // Reset probe counter
        counter_buf.write_raw(new Uint32Array([0]));
        
        // Update frame tracking
        this.frame_index++;
        this.config.reset_caches = false; // Clear reset flag after use
      }
    );

    // =========================================================================
    // Pass 1: Spawn Screen Probes (with temporal upscale and reprojection)
    // =========================================================================
    render_graph.add_pass(
      "gi_screen_probe_spawn",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          screen_probes,
          screen_probe_counter,
          gbuffer_position,
          gbuffer_normal,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive,
        ],
        outputs: [screen_probes, screen_probe_counter],
        shader_setup: screen_probe_spawn_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        // Dispatch over all pixels (shader will determine which spawn probes)
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    // =========================================================================
    // Pass 2-N: Screen Probe Path Tracing (Init, Hit, Shade for each bounce)
    // =========================================================================
    
    // Initialize probe rays
    render_graph.add_pass(
      "gi_probe_trace_init",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          screen_probes,
          screen_probe_counter,
          probe_path_state,
          probe_path_shade,
        ],
        outputs: [probe_path_state, probe_path_shade],
        shader_setup: screen_probe_trace_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const max_rays = this.config.max_screen_probes * this.config.screen_ray_count;
        pass.dispatch(Math.ceil(max_rays / 64), 1, 1);
      }
    );
    
    // Trace bounces
    for (let bounce = 0; bounce <= this.config.max_bounces; bounce++) {
      // Hit pass
      render_graph.add_pass(
        `gi_probe_trace_hit_${bounce}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            gi_params,
            probe_path_state,
            screen_probe_counter,
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
          const max_rays = this.config.max_screen_probes * this.config.screen_ray_count;
          pass.dispatch(Math.ceil(max_rays / 64), 1, 1);
        }
      );
      
      // Shade pass
      const shade_inputs = [
        gi_params,
        skydome_data_buffer,
        screen_probes,
        screen_probe_counter,
        probe_path_state,
        probe_path_shade,
        world_cache,
        params_gpu_buffer,
        material_palette_offsets_buffer,
        material_palette_buffer,
        dense_lights,
        light_count,
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
          const max_rays = this.config.max_screen_probes * this.config.screen_ray_count;
          pass.dispatch(Math.ceil(max_rays / 64), 1, 1);
        }
      );
    }

    // =========================================================================
    // Pass N+1: Update Screen Probes (accumulate ray radiance and world cache)
    // =========================================================================
    render_graph.add_pass(
      "gi_screen_probe_update",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          screen_probes,
          screen_probe_counter,
          probe_path_state,
          probe_path_shade,
          world_cache,
        ],
        outputs: [screen_probes, world_cache],
        shader_setup: screen_probe_update_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(this.config.max_screen_probes / 64), 1, 1);
      }
    );

    // =========================================================================
    // Pass N+2: Reconstruct Final Radiance (interpolate from probes)
    // =========================================================================
    render_graph.add_pass(
      "gi_reconstruct",
      RenderPassFlags.Compute,
      {
        inputs: [
          gi_params,
          screen_probes,
          screen_probe_counter,
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

    // Expose final GI texture
    this.final_gi_texture = gi_output;
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

export { GI };
