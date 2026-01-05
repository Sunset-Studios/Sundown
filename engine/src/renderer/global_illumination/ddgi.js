import { Renderer } from "../renderer.js";
import { Texture } from "../texture.js";
import {
  SharedEnvironmentData,
  SharedFrameInfoBuffer,
  SharedViewBuffer,
} from "../../core/shared_data.js";
import { DebugDrawType, RenderPassFlags, CacheTypes } from "../renderer_types.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";
import { Name } from "../../utility/names.js";
import { ResourceCache } from "../resource_cache.js";
import { ispot, npot } from "../../utility/math.js";
import { rgba16float_format } from "../../utility/config_permutations.js";

const COMPUTE_WORKGROUP_SIZE = 128;
const DDGI_PROBE_IRRADIANCE_RES = 8;
const DDGI_PROBE_ATLAS_GUTTER = 1;
const DDGI_PROBE_ATLAS_TILE_SIZE = DDGI_PROBE_IRRADIANCE_RES + 2 * DDGI_PROBE_ATLAS_GUTTER;
const DDGI_PROBE_DEPTH_RES = 16;
const DDGI_PROBE_DEPTH_ATLAS_TILE_SIZE = DDGI_PROBE_DEPTH_RES + 2 * DDGI_PROBE_ATLAS_GUTTER;

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

const world_cache_compact_block_prefix_scan_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_active_block_prefix_scan.wgsl" },
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

const specular_mask_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/specular_mask.wgsl" },
  },
};

const pixel_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_trace_init.wgsl", defines: { SPECULAR_MASK_ENABLED: 1 } },
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

const pixel_temporal_reservoir_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_temporal_reservoir.wgsl", defines: { SPECULAR_MASK_ENABLED: 1 } },
  },
};

const pixel_spatial_reservoir_wide_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "gi/pixel_spatial_reservoir_wide.wgsl",
      defines: { SPECULAR_MASK_ENABLED: 1 },
    },
  },
};

const pixel_spatial_reservoir_narrow_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "gi/pixel_spatial_reservoir_narrow.wgsl",
      defines: { SPECULAR_MASK_ENABLED: 1 },
    },
  },
};

const pixel_accumulate_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_accumulate.wgsl", defines: { SPECULAR_MASK_ENABLED: 1 } },
  },
};

const pixel_upscale_final_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/pixel_upscale_final.wgsl", defines: { SPECULAR_MASK_ENABLED: 1 } },
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

const ddgi_probe_accumulate_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_accumulate.wgsl" },
  },
};

const ddgi_probe_indices_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_indices_init.wgsl" },
  },
};

const ddgi_probe_reproject_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_reproject.wgsl" },
  },
};

const ddgi_probe_sample_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_sample.wgsl" },
  },
};

const ddgi_probe_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_debug.wgsl" },
  },
};

const ddgi_probe_atlas_overlay_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_atlas_overlay.wgsl" },
  },
};

const ddgi_probe_depth_atlas_overlay_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_depth_atlas_overlay.wgsl" },
  },
};

const world_cache_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/world_cache_debug.wgsl" },
  },
};

export class DDGI {
  config = {
    probe_grid_dimensions: [16, 8, 16],
    probe_spacing: 8.0,
    probe_radius: 0.5,
    rays_per_probe: 32,
    max_probes_per_frame: 100,
    do_per_pixel_specular: false,
    specular_upscale_factor: 2,
    specular_screen_ray_count: 1,
    world_cache_size: 32768,
    world_cache_cell_size: 2.0,
    world_cache_lod_count: 4,
    indirect_boost: 1.0,
  };

  ddgi_frame_setup = {
    width: 0,
    height: 0,
    gi_width: 0,
    gi_height: 0,
    total_pixels: 0,
    total_cells: 0,
    rays_per_frame: 0,
    frame_index: 0,
    ping_pong_frame: 0,
    safe_upscale_factor: 0,
    light_count: 0,
    force_recreate: false,
  };

  shared_bindings = {
    gi_params: null,
    gi_counters: null,
    use_radiance_cache_as_deferred_lighting: false,
    blue_noise_image: null,
    params_gpu_buffer: null,
    material_palette_offsets_buffer: null,
    material_palette_buffer: null,
    skydome_data_buffer: null,
    albedo_pool_buffer: null,
    normal_pool_buffer: null,
    roughness_pool_buffer: null,
    metallic_pool_buffer: null,
    ao_pool_buffer: null,
    height_pool_buffer: null,
    specular_pool_buffer: null,
    emission_pool_buffer: null,
    skybox_texture_buffer: null,
    probe_radiance_buffer: null,
    probe_depth_buffer: null,
    world_cache: null,
  };

  gi_params_data = new Float32Array([
    0, // specular_screen_ray_count
    0, // world_cache_size
    0, // world_cache_cell_size
    0, // total_pixels
    0, // frame_index
    0, // indirect_boost
    0, // upscale_factor
    0, // world_cache_lod_count
    0, // full_resolution_x
    0, // full_resolution_y
    0, // gi_resolution_x
    0, // gi_resolution_y
  ]);
  ddgi_params = null;
  ddgi_params_data = new Float32Array([
    0,
    0,
    0,
    0, // - probe_counts        (x=probe_count, y=rays_per_probe, z=probes_per_frame, w=probe_spacing)
    0,
    0,
    0,
    0, // - probe_grid_dims     (x=dim_x, y=dim_y, z=dim_z, w=probe_radius)
    0,
    0,
    0,
    0, // - probe_grid_origin   (xyz=grid origin, w=unused)
    0,
    0,
    0,
    0, // - probe_grid_log2     (xyz=log2(dim_*), w=unused)
    0,
    0,
    0,
    0, // - probe_grid_mask     (xyz=(dim_*-1), w=unused)
    0,
    0,
    0,
    0, // - probe_grid_snap_delta (xyz=delta in probe cells, w=active (1/0))
  ]);

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ Probe grid snap tracking (CPU-side)                                          │
  // └─────────────────────────────────────────────────────────────────────────────┘
  ddgi_probe_grid_snapped_origin = new Float32Array(3);

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
    tlas_bvh4_nodes,
    blas_atlas,
    entity_transforms,
    mesh_asset_ids,
    dense_lights,
    light_count,
    draw_count,
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

    if (draw_count === 0) {
      return;
    }

    this._build_shared_bindings(render_graph, force_recreate);

    this._setup_frame_data(
      width,
      height,
      light_count,
      force_recreate,
      this.config.specular_upscale_factor
    );

    this._add_gi_init_passes(render_graph);

    this._add_world_cache_passes(
      render_graph,
      light_count,
      dense_lights,
      tlas_bvh2_bounds,
      tlas_bvh4_nodes,
      blas_atlas,
      entity_transforms,
      mesh_asset_ids,
      this.shared_bindings.skydome_data_buffer,
      this.shared_bindings.params_gpu_buffer,
      this.shared_bindings.material_palette_offsets_buffer,
      this.shared_bindings.material_palette_buffer,
      this.shared_bindings.albedo_pool_buffer,
      this.shared_bindings.normal_pool_buffer,
      this.shared_bindings.roughness_pool_buffer,
      this.shared_bindings.metallic_pool_buffer,
      this.shared_bindings.ao_pool_buffer,
      this.shared_bindings.height_pool_buffer,
      this.shared_bindings.specular_pool_buffer,
      this.shared_bindings.emission_pool_buffer,
      this.shared_bindings.skybox_texture_buffer,
      force_recreate
    );

    if (this.config.do_per_pixel_specular) {
      this._add_specular_gi_passes(
        render_graph,
        tlas_bvh2_bounds,
        tlas_bvh4_nodes,
        blas_atlas,
        entity_transforms,
        mesh_asset_ids,
        dense_lights,
        light_count,
        gbuffer_position,
        gbuffer_position_prev,
        gbuffer_normal,
        gbuffer_normal_prev,
        gbuffer_albedo,
        gbuffer_smra,
        gbuffer_motion_emissive,
        force_recreate
      );
    }

    this._add_probe_passes(
      render_graph,
      gbuffer_position,
      gbuffer_normal,
      tlas_bvh2_bounds,
      tlas_bvh4_nodes,
      blas_atlas,
      entity_transforms,
      mesh_asset_ids,
      force_recreate
    );
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
      name: "ddgi_world_cache_debug",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    if (debug_view === DebugDrawType.GI_WorldCache) {
      render_graph.add_pass(
        "ddgi_debug_world_cache",
        RenderPassFlags.Compute,
        {
          inputs: [
            this.shared_bindings.gi_params,
            this.shared_bindings.world_cache,
            gbuffer_position,
            gbuffer_normal,
            scene_color,
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
    } else if (debug_view === DebugDrawType.GI_Probes) {
      render_graph.add_pass(
        "ddgi_probe_debug",
        RenderPassFlags.Compute,
        {
          inputs: [
            this.ddgi_params,
            this.shared_bindings.probe_radiance_buffer,
            scene_color,
            depth_texture,
            this.debug_texture,
          ],
          outputs: [this.debug_texture],
          shader_setup: ddgi_probe_debug_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
        }
      );
    } else if (debug_view === DebugDrawType.GI_ProbeAtlas) {
      render_graph.add_pass(
        "ddgi_probe_atlas_overlay",
        RenderPassFlags.Compute,
        {
          inputs: [this.shared_bindings.probe_radiance_buffer, scene_color, this.debug_texture],
          outputs: [this.debug_texture],
          shader_setup: ddgi_probe_atlas_overlay_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
        }
      );
    } else if (debug_view === DebugDrawType.GI_ProbeDepthAtlas) {
      render_graph.add_pass(
        "ddgi_probe_depth_atlas_overlay",
        RenderPassFlags.Compute,
        {
          inputs: [this.shared_bindings.probe_depth_buffer, scene_color, this.debug_texture],
          outputs: [this.debug_texture],
          shader_setup: ddgi_probe_depth_atlas_overlay_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
        }
      );
    }
  }

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ Shared setup                                                                │
  // │                                                                             │
  // │ This builds render-graph bindings and derived values that are shared across │
  // │ the DDGI sub-systems (init, per-pixel GI, world cache, probes).             │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _build_shared_bindings(render_graph, force_recreate) {
    // Constants
    this.shared_bindings.use_radiance_cache_as_deferred_lighting =
      Renderer.get().is_use_radiance_cache_as_deferred_lighting();

    // Common textures
    const blue_noise = Texture.default_blue_noise();
    this.shared_bindings.blue_noise_image = render_graph.register_image(blue_noise.config.name);

    const default_texture = Texture.default_array();
    const default_texture_buffer = render_graph.register_image(default_texture.config.name);

    // Material tables
    const params_gpu = MaterialAllocationTable.params_buffer;
    this.shared_bindings.params_gpu_buffer = render_graph.register_buffer(params_gpu.config.name);
    const material_palette = MaterialAllocationTable.palette_buffer;
    this.shared_bindings.material_palette_buffer = render_graph.register_buffer(
      material_palette.config.name
    );
    const material_palette_offsets = EntityManager.get_fragment_gpu_buffer(
      StaticMeshFragment,
      material_offsets_name
    );
    this.shared_bindings.material_palette_offsets_buffer = render_graph.register_buffer(
      material_palette_offsets.buffer.config.name
    );

    // Texture pools
    const albedo_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_albedo_name);
    this.shared_bindings.albedo_pool_buffer = albedo_pool
      ? render_graph.register_image(albedo_pool.config.name)
      : default_texture_buffer;
    const normal_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_normal_name);
    this.shared_bindings.normal_pool_buffer = normal_pool
      ? render_graph.register_image(normal_pool.config.name)
      : default_texture_buffer;
    const roughness_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_roughness_name);
    this.shared_bindings.roughness_pool_buffer = roughness_pool
      ? render_graph.register_image(roughness_pool.config.name)
      : default_texture_buffer;
    const metallic_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_metallic_name);
    this.shared_bindings.metallic_pool_buffer = metallic_pool
      ? render_graph.register_image(metallic_pool.config.name)
      : default_texture_buffer;
    const ao_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_ao_name);
    this.shared_bindings.ao_pool_buffer = ao_pool
      ? render_graph.register_image(ao_pool.config.name)
      : default_texture_buffer;
    const height_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_height_name);
    this.shared_bindings.height_pool_buffer = height_pool
      ? render_graph.register_image(height_pool.config.name)
      : default_texture_buffer;
    const specular_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_specular_name);
    this.shared_bindings.specular_pool_buffer = specular_pool
      ? render_graph.register_image(specular_pool.config.name)
      : default_texture_buffer;
    const emission_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_emission_name);
    this.shared_bindings.emission_pool_buffer = emission_pool
      ? render_graph.register_image(emission_pool.config.name)
      : default_texture_buffer;

    // Environment
    const skydome_data = SharedEnvironmentData.get_skydome_data();
    this.shared_bindings.skydome_data_buffer = render_graph.register_buffer(
      skydome_data.config.name
    );

    const skybox = SharedEnvironmentData.get_skybox();
    this.shared_bindings.skybox_texture_buffer = render_graph.register_image(skybox.config.name);

    // GI uniforms buffers
    this.shared_bindings.gi_params = render_graph.create_buffer({
      name: "ddgi_gi_params",
      size: this.gi_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this.shared_bindings.gi_counters = render_graph.create_buffer({
      name: "ddgi_gi_counters",
      size: 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
  }

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ Frame data setup                                                            │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _setup_frame_data(width, height, light_count, force_recreate, safe_upscale_factor) {
    this.ddgi_frame_setup.width = width;
    this.ddgi_frame_setup.height = height;
    this.ddgi_frame_setup.light_count = light_count;
    this.ddgi_frame_setup.force_recreate = force_recreate;
    this.ddgi_frame_setup.gi_width = Math.max(1, Math.ceil(width / safe_upscale_factor));
    this.ddgi_frame_setup.gi_height = Math.max(1, Math.ceil(height / safe_upscale_factor));
    this.ddgi_frame_setup.total_pixels =
      this.ddgi_frame_setup.gi_width * this.ddgi_frame_setup.gi_height;
    this.ddgi_frame_setup.total_cells =
      this.config.world_cache_size * this.config.world_cache_lod_count;
    this.ddgi_frame_setup.rays_per_frame =
      this.ddgi_frame_setup.total_pixels * this.config.specular_screen_ray_count;
    this.ddgi_frame_setup.frame_index = SharedFrameInfoBuffer.get_frame_index();
    this.ddgi_frame_setup.ping_pong_frame = this.ddgi_frame_setup.frame_index % 2;
    this.ddgi_frame_setup.safe_upscale_factor = Math.max(
      1,
      Math.floor(this.config.specular_upscale_factor)
    );
  }

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ GI init passes                                                               │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _add_gi_init_passes(render_graph) {
    render_graph.add_pass(
      "ddgi_gi_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const gi_params_buf = graph.get_physical_buffer(this.shared_bindings.gi_params);

        // Fill parameter buffer
        this.gi_params_data[0] = this.config.specular_screen_ray_count;
        this.gi_params_data[1] = this.config.world_cache_size;
        this.gi_params_data[2] = this.config.world_cache_cell_size;
        this.gi_params_data[3] = this.ddgi_frame_setup.total_pixels;
        this.gi_params_data[4] = this.ddgi_frame_setup.frame_index;
        this.gi_params_data[5] = this.config.indirect_boost;
        this.gi_params_data[6] = this.ddgi_frame_setup.safe_upscale_factor;
        this.gi_params_data[7] = this.config.world_cache_lod_count;
        this.gi_params_data[8] = this.ddgi_frame_setup.width;
        this.gi_params_data[9] = this.ddgi_frame_setup.height;
        this.gi_params_data[10] = this.ddgi_frame_setup.gi_width;
        this.gi_params_data[11] = this.ddgi_frame_setup.gi_height;

        gi_params_buf.write_raw(this.gi_params_data);
      }
    );

    render_graph.add_pass(
      "ddgi_gi_reset",
      RenderPassFlags.Compute,
      {
        inputs: [this.shared_bindings.gi_counters, this.ddgi_frame_setup.light_count],
        outputs: [this.shared_bindings.gi_counters],
        shader_setup: gi_reset_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1);
      }
    );
  }

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ Specular GI passes                                                           │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _add_specular_gi_passes(
    render_graph,
    tlas_bvh2_bounds,
    tlas_bvh4_nodes,
    blas_atlas,
    entity_transforms,
    mesh_asset_ids,
    dense_lights,
    light_count,
    gbuffer_position,
    gbuffer_position_prev,
    gbuffer_normal,
    gbuffer_normal_prev,
    gbuffer_albedo,
    gbuffer_smra,
    gbuffer_motion_emissive,
    force_recreate
  ) {
    let pixel_path_state = render_graph.create_buffer({
      name: "ddgi_pixel_path_state",
      size: this.ddgi_frame_setup.rays_per_frame * 15 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    let pixel_ray_queue = render_graph.create_buffer({
      name: "ddgi_pixel_ray_queue",
      size: this.ddgi_frame_setup.rays_per_frame,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const reservoir_size = this.ddgi_frame_setup.gi_width * this.ddgi_frame_setup.gi_height * 28;
    const temporal_reservoir_0 = render_graph.create_buffer({
      name: "ddgi_temporal_reservoir_0",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const temporal_reservoir_1 = render_graph.create_buffer({
      name: "ddgi_temporal_reservoir_1",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const spatial_reservoir_0 = render_graph.create_buffer({
      name: "ddgi_spatial_reservoir_0",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const spatial_reservoir_1 = render_graph.create_buffer({
      name: "ddgi_spatial_reservoir_1",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const spatial_reservoir_stage = render_graph.create_buffer({
      name: "ddgi_spatial_reservoir_stage",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const specular_mask_texture = render_graph.create_image({
      name: "ddgi_specular_mask",
      format: "r32uint",
      width: this.ddgi_frame_setup.gi_width,
      height: this.ddgi_frame_setup.gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const gi_low_radiance_direct_0 = render_graph.create_image({
      name: "ddgi_low_radiance_direct_0",
      format: "rgba16float",
      width: this.ddgi_frame_setup.gi_width,
      height: this.ddgi_frame_setup.gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    const gi_low_radiance_direct_1 = render_graph.create_image({
      name: "ddgi_low_radiance_direct_1",
      format: "rgba16float",
      width: this.ddgi_frame_setup.gi_width,
      height: this.ddgi_frame_setup.gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const gi_low_radiance_indirect_diffuse_0 = render_graph.create_image({
      name: "ddgi_low_radiance_indirect_diffuse_0",
      format: "rgba16float",
      width: this.ddgi_frame_setup.gi_width,
      height: this.ddgi_frame_setup.gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    const gi_low_radiance_indirect_diffuse_1 = render_graph.create_image({
      name: "ddgi_low_radiance_indirect_diffuse_1",
      format: "rgba16float",
      width: this.ddgi_frame_setup.gi_width,
      height: this.ddgi_frame_setup.gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const gi_low_radiance_indirect_specular_0 = render_graph.create_image({
      name: "ddgi_low_radiance_indirect_specular_0",
      format: "rgba16float",
      width: this.ddgi_frame_setup.gi_width,
      height: this.ddgi_frame_setup.gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    const gi_low_radiance_indirect_specular_1 = render_graph.create_image({
      name: "ddgi_low_radiance_indirect_specular_1",
      format: "rgba16float",
      width: this.ddgi_frame_setup.gi_width,
      height: this.ddgi_frame_setup.gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const gi_low_radiance_prev_direct =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? gi_low_radiance_direct_0
        : gi_low_radiance_direct_1;
    const gi_low_radiance_prev_indirect_diffuse =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? gi_low_radiance_indirect_diffuse_0
        : gi_low_radiance_indirect_diffuse_1;
    const gi_low_radiance_prev_indirect_specular =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? gi_low_radiance_indirect_specular_0
        : gi_low_radiance_indirect_specular_1;

    const gi_low_radiance_curr_direct =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? gi_low_radiance_direct_1
        : gi_low_radiance_direct_0;
    const gi_low_radiance_curr_indirect_diffuse =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? gi_low_radiance_indirect_diffuse_1
        : gi_low_radiance_indirect_diffuse_0;
    const gi_low_radiance_curr_indirect_specular =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? gi_low_radiance_indirect_specular_1
        : gi_low_radiance_indirect_specular_0;

    const specular_only_indirect_diffuse = render_graph.create_image({
      name: "ddgi_specular_only_indirect_diffuse",
      format: "rgba16float",
      width: this.ddgi_frame_setup.width,
      height: this.ddgi_frame_setup.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const temporal_reservoir_prev =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? temporal_reservoir_0 : temporal_reservoir_1;
    const spatial_reservoir_prev =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? spatial_reservoir_0 : spatial_reservoir_1;
    const temporal_reservoir_curr =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? temporal_reservoir_1 : temporal_reservoir_0;
    const spatial_reservoir_curr =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? spatial_reservoir_1 : spatial_reservoir_0;

    render_graph.add_pass(
      "ddgi_specular_mask",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          gbuffer_normal,
          gbuffer_smra,
          specular_mask_texture,
        ],
        outputs: [specular_mask_texture],
        shader_setup: specular_mask_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.gi_width / 16),
          Math.ceil(this.ddgi_frame_setup.gi_height / 16),
          1
        );
      }
    );

    render_graph.add_pass(
      `ddgi_pixel_trace_init_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          this.shared_bindings.gi_counters,
          pixel_path_state,
          pixel_ray_queue,
          light_count,
          dense_lights,
          this.shared_bindings.world_cache,
          gbuffer_position,
          gbuffer_normal,
          gbuffer_albedo,
          gbuffer_smra,
          gbuffer_motion_emissive,
          this.shared_bindings.blue_noise_image,
          specular_mask_texture,
        ],
        outputs: [pixel_path_state, pixel_ray_queue],
        shader_setup: pixel_trace_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.rays_per_frame / COMPUTE_WORKGROUP_SIZE),
          1,
          1
        );
      }
    );

    render_graph.add_pass(
      "ddgi_pixel_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          this.shared_bindings.gi_counters,
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
        if (this.shared_bindings.use_radiance_cache_as_deferred_lighting) {
          pass.dispatch(
            Math.ceil((2 * this.ddgi_frame_setup.rays_per_frame) / COMPUTE_WORKGROUP_SIZE),
            1,
            1
          );
        } else {
          pass.dispatch(
            Math.ceil(this.ddgi_frame_setup.rays_per_frame / COMPUTE_WORKGROUP_SIZE),
            1,
            1
          );
        }
      }
    );

    render_graph.add_pass(
      "ddgi_pixel_trace_shade",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          this.shared_bindings.skydome_data_buffer,
          this.shared_bindings.gi_counters,
          pixel_path_state,
          this.shared_bindings.world_cache,
          this.shared_bindings.params_gpu_buffer,
          this.shared_bindings.material_palette_offsets_buffer,
          this.shared_bindings.material_palette_buffer,
          dense_lights,
          this.shared_bindings.albedo_pool_buffer,
          this.shared_bindings.normal_pool_buffer,
          this.shared_bindings.roughness_pool_buffer,
          this.shared_bindings.metallic_pool_buffer,
          this.shared_bindings.ao_pool_buffer,
          this.shared_bindings.height_pool_buffer,
          this.shared_bindings.specular_pool_buffer,
          this.shared_bindings.emission_pool_buffer,
          this.shared_bindings.skybox_texture_buffer,
        ],
        outputs: [pixel_path_state],
        shader_setup: pixel_trace_shade_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.rays_per_frame / COMPUTE_WORKGROUP_SIZE),
          1,
          1
        );
      }
    );

    render_graph.add_pass(
      `ddgi_pixel_temporal_reservoir_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          pixel_path_state,
          temporal_reservoir_prev,
          temporal_reservoir_curr,
          gbuffer_position,
          gbuffer_position_prev,
          gbuffer_normal,
          gbuffer_motion_emissive,
          gbuffer_normal_prev,
          specular_mask_texture,
        ],
        outputs: [temporal_reservoir_curr],
        shader_setup: pixel_temporal_reservoir_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.gi_width / 16),
          Math.ceil(this.ddgi_frame_setup.gi_height / 16),
          1
        );
      }
    );

    render_graph.add_pass(
      `ddgi_pixel_spatial_reservoir_wide_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          temporal_reservoir_curr,
          spatial_reservoir_stage,
          gbuffer_position,
          gbuffer_normal,
          specular_mask_texture,
        ],
        outputs: [spatial_reservoir_stage],
        shader_setup: pixel_spatial_reservoir_wide_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.gi_width / 16),
          Math.ceil(this.ddgi_frame_setup.gi_height / 16),
          1
        );
      }
    );

    render_graph.add_pass(
      `ddgi_pixel_spatial_reservoir_narrow_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          spatial_reservoir_stage,
          spatial_reservoir_curr,
          gbuffer_position,
          gbuffer_normal,
          specular_mask_texture,
        ],
        outputs: [spatial_reservoir_curr],
        shader_setup: pixel_spatial_reservoir_narrow_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.gi_width / 16),
          Math.ceil(this.ddgi_frame_setup.gi_height / 16),
          1
        );
      }
    );

    render_graph.add_pass(
      `ddgi_pixel_accumulate_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          spatial_reservoir_curr,
          gi_low_radiance_prev_direct,
          gi_low_radiance_prev_indirect_diffuse,
          gi_low_radiance_prev_indirect_specular,
          gbuffer_position,
          gbuffer_position_prev,
          gbuffer_normal,
          gbuffer_normal_prev,
          gbuffer_motion_emissive,
          gi_low_radiance_curr_direct,
          gi_low_radiance_curr_indirect_diffuse,
          gi_low_radiance_curr_indirect_specular,
          specular_mask_texture,
        ],
        outputs: [
          gi_low_radiance_curr_direct,
          gi_low_radiance_curr_indirect_diffuse,
          gi_low_radiance_curr_indirect_specular,
        ],
        shader_setup: pixel_accumulate_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(this.ddgi_frame_setup.gi_width / 8),
          Math.ceil(this.ddgi_frame_setup.gi_height / 8),
          1
        );
      }
    );

    render_graph.add_pass(
      `ddgi_upscale_final_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          gi_low_radiance_curr_direct,
          gi_low_radiance_curr_indirect_diffuse,
          gi_low_radiance_curr_indirect_specular,
          gbuffer_position,
          gbuffer_normal,
          this.final_gi_texture_direct,
          specular_only_indirect_diffuse,
          this.final_gi_texture_indirect_specular,
          specular_mask_texture,
        ],
        outputs: [
          this.final_gi_texture_direct,
          specular_only_indirect_diffuse,
          this.final_gi_texture_indirect_specular,
        ],
        shader_setup: pixel_upscale_final_shader_setup,
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
  }

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ World cache passes                                                           │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _add_world_cache_passes(
    render_graph,
    light_count,
    dense_lights,
    tlas_bvh2_bounds,
    tlas_bvh4_nodes,
    blas_atlas,
    entity_transforms,
    mesh_asset_ids,
    skydome_data_buffer,
    params_gpu_buffer,
    material_palette_offsets_buffer,
    material_palette_buffer,
    albedo_pool_buffer,
    normal_pool_buffer,
    roughness_pool_buffer,
    metallic_pool_buffer,
    ao_pool_buffer,
    height_pool_buffer,
    specular_pool_buffer,
    emission_pool_buffer,
    skybox_texture_buffer,
    force_recreate
  ) {
    this.shared_bindings.world_cache = render_graph.create_buffer({
      name: "ddgi_world_cache",
      size: this.ddgi_frame_setup.total_cells * 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_active_flags = render_graph.create_buffer({
      name: "ddgi_world_cache_active_flags",
      size: this.ddgi_frame_setup.total_cells,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_prefix_sum = render_graph.create_buffer({
      name: "ddgi_world_cache_prefix_sum",
      size: this.ddgi_frame_setup.total_cells,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_block_sums = render_graph.create_buffer({
      name: "ddgi_world_cache_block_sums",
      size: Math.ceil(this.ddgi_frame_setup.total_cells / COMPUTE_WORKGROUP_SIZE),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_block_prefixes = render_graph.create_buffer({
      name: "ddgi_world_cache_block_prefixes",
      size: Math.ceil(this.ddgi_frame_setup.total_cells / COMPUTE_WORKGROUP_SIZE),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_compacted_indices = render_graph.create_buffer({
      name: "ddgi_world_cache_compacted_indices",
      size: this.ddgi_frame_setup.total_cells,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_dispatch_params = render_graph.create_buffer({
      name: "ddgi_world_cache_dispatch_params",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const world_cache_path_state = render_graph.create_buffer({
      name: "ddgi_world_cache_path_state",
      size: this.ddgi_frame_setup.total_cells * 11 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    render_graph.add_pass(
      "ddgi_world_cache_evict",
      RenderPassFlags.Compute,
      {
        inputs: [this.shared_bindings.world_cache],
        outputs: [this.shared_bindings.world_cache],
        shader_setup: world_cache_evict_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(this.ddgi_frame_setup.total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_world_cache_active_mark",
      RenderPassFlags.Compute,
      {
        inputs: [this.shared_bindings.world_cache, world_cache_active_flags],
        outputs: [world_cache_active_flags],
        shader_setup: world_cache_compact_mark_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(this.ddgi_frame_setup.total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_world_cache_active_prefix_sum",
      RenderPassFlags.Compute,
      {
        inputs: [world_cache_active_flags, world_cache_prefix_sum, world_cache_block_sums],
        outputs: [world_cache_prefix_sum, world_cache_block_sums],
        shader_setup: world_cache_compact_prefix_sum_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(this.ddgi_frame_setup.total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_world_cache_active_block_prefix_scan",
      RenderPassFlags.Compute,
      {
        inputs: [
          world_cache_block_sums,
          world_cache_block_prefixes,
          world_cache_dispatch_params,
          this.shared_bindings.gi_counters,
        ],
        outputs: [
          world_cache_block_prefixes,
          world_cache_dispatch_params,
          this.shared_bindings.gi_counters,
        ],
        shader_setup: world_cache_compact_block_prefix_scan_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_world_cache_active_compact",
      RenderPassFlags.Compute,
      {
        inputs: [
          world_cache_active_flags,
          world_cache_prefix_sum,
          world_cache_block_prefixes,
          world_cache_compacted_indices,
        ],
        outputs: [world_cache_compacted_indices],
        shader_setup: world_cache_compact_scatter_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(this.ddgi_frame_setup.total_cells / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_world_cache_trace_init",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          this.shared_bindings.world_cache,
          world_cache_compacted_indices,
          world_cache_dispatch_params,
          world_cache_path_state,
          light_count,
          dense_lights,
          this.shared_bindings.gi_counters,
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

    render_graph.add_pass(
      "ddgi_world_cache_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          world_cache_path_state,
          tlas_bvh2_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          mesh_asset_ids,
          this.shared_bindings.gi_counters,
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

    render_graph.add_pass(
      "ddgi_world_cache_trace_shade",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          skydome_data_buffer,
          this.shared_bindings.world_cache,
          world_cache_compacted_indices,
          world_cache_path_state,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          dense_lights,
          this.shared_bindings.gi_counters,
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
        outputs: [world_cache_path_state, this.shared_bindings.world_cache],
        shader_setup: world_cache_trace_shade_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const dispatch_buffer = graph.get_physical_buffer(world_cache_dispatch_params);
        pass.dispatch_indirect(dispatch_buffer, 0);
      }
    );
  }

  // ┌─────────────────────────────────────────────────────────────────────────────┐
  // │ Probe passes                                                                │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _add_probe_passes(
    render_graph,
    gbuffer_position,
    gbuffer_normal,
    tlas_bvh2_bounds,
    tlas_bvh4_nodes,
    blas_atlas,
    entity_transforms,
    mesh_asset_ids,
    force_recreate
  ) {
    const grid_dims = this._sanitize_probe_grid_dimensions(this.config.probe_grid_dimensions);
    const probe_count = grid_dims[0] * grid_dims[1] * grid_dims[2];

    const probes_per_frame = probe_count;
    const rays_per_probe = Math.max(1, Math.floor(this.config.rays_per_probe));
    const probe_ray_count = probes_per_frame * rays_per_probe;

    const view_index = SharedFrameInfoBuffer.get_view_index();
    const view = SharedViewBuffer.get_view_data(view_index);
    const camera_position = view.view_position;

    const spacing = this.config.probe_spacing;
    const half_extents = [
      (grid_dims[0] - 1) * 0.5 * spacing,
      (grid_dims[1] - 1) * 0.5 * spacing,
      (grid_dims[2] - 1) * 0.5 * spacing,
    ];

    const snapped_origin = [
      Math.floor(camera_position[0] / spacing) * spacing - half_extents[0],
      Math.floor(camera_position[1] / spacing) * spacing - half_extents[1],
      Math.floor(camera_position[2] / spacing) * spacing - half_extents[2],
    ];

    const snap_delta_x = Math.round(
      (snapped_origin[0] - this.ddgi_probe_grid_snapped_origin[0]) / spacing
    );
    const snap_delta_y = Math.round(
      (snapped_origin[1] - this.ddgi_probe_grid_snapped_origin[1]) / spacing
    );
    const snap_delta_z = Math.round(
      (snapped_origin[2] - this.ddgi_probe_grid_snapped_origin[2]) / spacing
    );

    this.ddgi_probe_grid_snapped_origin[0] = snapped_origin[0];
    this.ddgi_probe_grid_snapped_origin[1] = snapped_origin[1];
    this.ddgi_probe_grid_snapped_origin[2] = snapped_origin[2];

    const grid_log2 = [
      Math.round(Math.log2(grid_dims[0])),
      Math.round(Math.log2(grid_dims[1])),
      Math.round(Math.log2(grid_dims[2])),
    ];
    const grid_mask = [grid_dims[0] - 1, grid_dims[1] - 1, grid_dims[2] - 1];

    const atlas_cols = grid_dims[0] * grid_dims[2];
    const atlas_rows = grid_dims[1];
    const atlas_width = Math.max(1, atlas_cols * DDGI_PROBE_ATLAS_TILE_SIZE);
    const atlas_height = Math.max(1, atlas_rows * DDGI_PROBE_ATLAS_TILE_SIZE);
    const depth_atlas_width = Math.max(1, atlas_cols * DDGI_PROBE_DEPTH_ATLAS_TILE_SIZE);
    const depth_atlas_height = Math.max(1, atlas_rows * DDGI_PROBE_DEPTH_ATLAS_TILE_SIZE);

    this.ddgi_params_data[0] = probe_count;
    this.ddgi_params_data[1] = rays_per_probe;
    this.ddgi_params_data[2] = probes_per_frame;
    this.ddgi_params_data[3] = spacing;
    this.ddgi_params_data[4] = grid_dims[0];
    this.ddgi_params_data[5] = grid_dims[1];
    this.ddgi_params_data[6] = grid_dims[2];
    this.ddgi_params_data[7] = this.config.probe_radius;
    this.ddgi_params_data[8] = snapped_origin[0];
    this.ddgi_params_data[9] = snapped_origin[1];
    this.ddgi_params_data[10] = snapped_origin[2];
    this.ddgi_params_data[11] = 0;
    this.ddgi_params_data[12] = grid_log2[0];
    this.ddgi_params_data[13] = grid_log2[1];
    this.ddgi_params_data[14] = grid_log2[2];
    this.ddgi_params_data[15] = 0;
    this.ddgi_params_data[16] = grid_mask[0];
    this.ddgi_params_data[17] = grid_mask[1];
    this.ddgi_params_data[18] = grid_mask[2];
    this.ddgi_params_data[19] = 0;
    this.ddgi_params_data[20] = snap_delta_x;
    this.ddgi_params_data[21] = snap_delta_y;
    this.ddgi_params_data[22] = snap_delta_z;
    this.ddgi_params_data[23] =
      snap_delta_x !== 0 || snap_delta_y !== 0 || snap_delta_z !== 0 ? 1 : 0;

    this.ddgi_params = render_graph.create_buffer({
      name: "ddgi_params",
      size: this.ddgi_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_update_indices = render_graph.create_buffer({
      name: "ddgi_probe_update_indices",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_ray_radiance = render_graph.create_buffer({
      name: "ddgi_probe_ray_radiance",
      size: probe_ray_count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_ray_hits = render_graph.create_buffer({
      name: "ddgi_probe_ray_hits",
      size: probe_ray_count * 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_irradiance_atlas_0 = render_graph.create_image({
      name: "ddgi_probe_irradiance_atlas_0",
      format: rgba16float_format,
      width: atlas_width,
      height: atlas_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const probe_irradiance_atlas_1 = render_graph.create_image({
      name: "ddgi_probe_irradiance_atlas_1",
      format: rgba16float_format,
      width: atlas_width,
      height: atlas_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const probe_depth_atlas_0 = render_graph.create_image({
      name: "ddgi_probe_depth_atlas_0",
      format: rgba16float_format,
      width: depth_atlas_width,
      height: depth_atlas_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const probe_depth_atlas_1 = render_graph.create_image({
      name: "ddgi_probe_depth_atlas_1",
      format: rgba16float_format,
      width: depth_atlas_width,
      height: depth_atlas_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const probe_irradiance_src =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? probe_irradiance_atlas_0
        : probe_irradiance_atlas_1;
    const probe_irradiance_dst =
      this.ddgi_frame_setup.ping_pong_frame === 0
        ? probe_irradiance_atlas_1
        : probe_irradiance_atlas_0;

    const probe_depth_src =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? probe_depth_atlas_0 : probe_depth_atlas_1;
    const probe_depth_dst =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? probe_depth_atlas_1 : probe_depth_atlas_0;

    render_graph.add_pass(
      "ddgi_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const ddgi_params_buf = graph.get_physical_buffer(this.ddgi_params);
        ddgi_params_buf.write_raw(this.ddgi_params_data);
      }
    );

    render_graph.add_pass(
      "ddgi_probe_indices_init",
      RenderPassFlags.Compute,
      {
        inputs: [this.ddgi_params, probe_update_indices],
        outputs: [probe_update_indices],
        shader_setup: ddgi_probe_indices_init_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const probe_dispatch_count = Math.ceil(probe_count / 128);
        pass.dispatch(probe_dispatch_count, 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_probe_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          this.ddgi_params,
          probe_update_indices,
          probe_ray_hits,
          tlas_bvh2_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          mesh_asset_ids,
        ],
        outputs: [probe_ray_hits],
        shader_setup: ddgi_probe_trace_hit_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_ray_count / 128), 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_probe_trace_shade",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          this.ddgi_params,
          this.shared_bindings.skydome_data_buffer,
          probe_update_indices,
          probe_ray_hits,
          probe_ray_radiance,
          this.shared_bindings.params_gpu_buffer,
          this.shared_bindings.material_palette_offsets_buffer,
          this.shared_bindings.material_palette_buffer,
          this.shared_bindings.world_cache,
          this.shared_bindings.albedo_pool_buffer,
          this.shared_bindings.normal_pool_buffer,
          this.shared_bindings.roughness_pool_buffer,
          this.shared_bindings.metallic_pool_buffer,
          this.shared_bindings.ao_pool_buffer,
          this.shared_bindings.height_pool_buffer,
          this.shared_bindings.specular_pool_buffer,
          this.shared_bindings.emission_pool_buffer,
          this.shared_bindings.skybox_texture_buffer,
        ],
        outputs: [probe_ray_radiance, this.shared_bindings.world_cache],
        shader_setup: ddgi_probe_trace_shade_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_ray_count / 128), 1, 1);
      }
    );

    render_graph.add_pass(
      `ddgi_probe_accumulate_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_ray_hits,
          probe_ray_radiance,
          probe_irradiance_src,
          probe_irradiance_dst,
          probe_depth_src,
          probe_depth_dst,
        ],
        outputs: [probe_irradiance_dst, probe_depth_dst],
        shader_setup: ddgi_probe_accumulate_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probes_per_frame / 64), 1, 1);
      }
    );

    render_graph.add_pass(
      `ddgi_probe_sample_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.shared_bindings.gi_params,
          this.ddgi_params,
          probe_irradiance_dst,
          probe_depth_dst,
          gbuffer_position,
          gbuffer_normal,
          this.final_gi_texture_indirect_diffuse,
        ],
        outputs: [this.final_gi_texture_indirect_diffuse],
        shader_setup: ddgi_probe_sample_shader_setup,
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

    this.shared_bindings.probe_radiance_buffer = probe_irradiance_dst;
    this.shared_bindings.probe_depth_buffer = probe_depth_dst;
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
}
