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
const DDGI_PROBE_DEPTH_ATLAS_GUTTER = 1;
const DDGI_PROBE_DEPTH_RES = 16;
const DDGI_PROBE_DEPTH_ATLAS_TILE_SIZE = DDGI_PROBE_DEPTH_RES + 2 * DDGI_PROBE_DEPTH_ATLAS_GUTTER;

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

const ddgi_probe_indices_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_indices_init.wgsl" },
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

const ddgi_probe_depth_atlas_overlay_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_probe_depth_atlas_overlay.wgsl" },
  },
};

const ddgi_sh_probe_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/ddgi_sh_probe_debug.wgsl" },
  },
};

export class DDGI {
  config = {
    probe_grid_dimensions: [16, 8, 16],
    probe_spacing: 6.0,
    probe_radius: 0.5,
    rays_per_probe: 32,
    max_probes_per_frame: 100,
    indirect_boost: 1.0,
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
    probe_depth_buffer: null,
  };

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
    0, // - frame_index
    0, // - indirect_boost
    0, // - padding
    0, // - padding
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
    tlas_bvh8_nodes,
    blas_atlas,
    entity_transforms,
    mesh_asset_ids,
    dense_lights,
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

    if (draw_count > 0) {
      this._add_probe_passes(
        width,
        height,
        render_graph,
        gbuffer_position,
        gbuffer_normal,
        tlas_bvh2_bounds,
        tlas_bvh8_nodes,
        blas_atlas,
        entity_transforms,
        mesh_asset_ids,
        dense_lights,
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
  // │ Probe passes                                                                │
  // └─────────────────────────────────────────────────────────────────────────────┘
  _add_probe_passes(
    width,
    height,
    render_graph,
    gbuffer_position,
    gbuffer_normal,
    tlas_bvh2_bounds,
    tlas_bvh8_nodes,
    blas_atlas,
    entity_transforms,
    mesh_asset_ids,
    dense_lights,
    force_recreate
  ) {
    const grid_dims = this._sanitize_probe_grid_dimensions(this.config.probe_grid_dimensions);
    const probe_count = grid_dims[0] * grid_dims[1] * grid_dims[2];

    const probes_per_frame = probe_count;
    const rays_per_probe = Math.max(1, Math.floor(this.config.rays_per_probe));
    const probe_shadow_ray_count = probes_per_frame;
    const probe_primary_ray_count = probes_per_frame * rays_per_probe;
    const probe_total_ray_count = probe_shadow_ray_count + probe_primary_ray_count;

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

    const depth_atlas_width = Math.max(1, grid_dims[0] * DDGI_PROBE_DEPTH_ATLAS_TILE_SIZE);
    const depth_atlas_height = Math.max(1, grid_dims[2] * DDGI_PROBE_DEPTH_ATLAS_TILE_SIZE);
    const depth_atlas_layers = Math.max(1, grid_dims[1]);

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

    const gi_counters = render_graph.create_buffer({
      name: "ddgi_gi_counters",
      size: 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

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
      size: probe_total_ray_count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_ray_hits = render_graph.create_buffer({
      name: "ddgi_probe_ray_hits",
      size: probe_total_ray_count * 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const probe_depth_atlas = render_graph.create_image({
      name: "ddgi_probe_depth_atlas",
      format: rgba16float_format,
      width: depth_atlas_width,
      height: depth_atlas_height,
      depth: depth_atlas_layers,
      dimension: "2d-array",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────────
    // SH Probe Buffers
    // L1 RGB: 4 coefficients × 3 channels = 12 floats packed to 6 u32 per probe
    // ─────────────────────────────────────────────────────────────────────────
    const sh_probe_size_u32 = 6;
    const sh_probes_0 = render_graph.create_buffer({
      name: "ddgi_sh_probes_0",
      size: probe_count * sh_probe_size_u32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const sh_probes_1 = render_graph.create_buffer({
      name: "ddgi_sh_probes_1",
      size: probe_count * sh_probe_size_u32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ─────────────────────────────────────────────────────────────────────────
    // SH Probe Sample Count Buffers
    // Tracks accumulated sample count per probe for proper temporal averaging
    // ─────────────────────────────────────────────────────────────────────────
    const sh_sample_counts_0 = render_graph.create_buffer({
      name: "ddgi_sh_sample_counts_0",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const sh_sample_counts_1 = render_graph.create_buffer({
      name: "ddgi_sh_sample_counts_1",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const sh_probes_prev = this.ddgi_frame_setup.ping_pong_frame === 0 ? sh_probes_0 : sh_probes_1;
    const sh_probes_curr = this.ddgi_frame_setup.ping_pong_frame === 0 ? sh_probes_1 : sh_probes_0;

    const sh_sample_counts_prev =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? sh_sample_counts_0 : sh_sample_counts_1;
    const sh_sample_counts_curr =
      this.ddgi_frame_setup.ping_pong_frame === 0 ? sh_sample_counts_1 : sh_sample_counts_0;

    render_graph.add_pass(
      "ddgi_upload_params",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const ddgi_params_buf = graph.get_physical_buffer(this.ddgi_params);

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
        this.ddgi_params_data[24] = this.ddgi_frame_setup.frame_index;
        this.ddgi_params_data[25] = this.config.indirect_boost;

        ddgi_params_buf.write_raw(this.ddgi_params_data);
      }
    );

    render_graph.add_pass(
      "ddgi_gi_reset",
      RenderPassFlags.Compute,
      {
        inputs: [gi_counters, dense_lights],
        outputs: [gi_counters],
        shader_setup: gi_reset_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1);
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
        const probe_dispatch_count = Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE);
        pass.dispatch(probe_dispatch_count, 1, 1);
      }
    );

    render_graph.add_pass(
      "ddgi_probe_trace_hit",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_ray_hits,
          tlas_bvh2_bounds,
          tlas_bvh8_nodes,
          blas_atlas,
          entity_transforms,
          mesh_asset_ids,
          dense_lights,
        ],
        outputs: [probe_ray_hits],
        shader_setup: ddgi_probe_trace_hit_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probe_total_ray_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      `ddgi_probe_trace_shade_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          skydome_data_buffer,
          probe_update_indices,
          probe_ray_hits,
          probe_ray_radiance,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          dense_lights,
          sh_probes_prev,
          probe_depth_atlas,
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
        outputs: [probe_ray_radiance],
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
    // ─────────────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      `ddgi_sh_probe_accumulate_${this.ddgi_frame_setup.ping_pong_frame}`,
      RenderPassFlags.Compute,
      {
        inputs: [
          this.ddgi_params,
          probe_update_indices,
          probe_ray_hits,
          probe_ray_radiance,
          sh_probes_prev,
          sh_probes_curr,
          sh_sample_counts_prev,
          sh_sample_counts_curr,
        ],
        outputs: [sh_probes_curr, sh_sample_counts_curr],
        shader_setup: ddgi_sh_probe_accumulate_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(probes_per_frame / 64), 1, 1);
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
          sh_probes_curr,
          probe_depth_atlas,
          gbuffer_position,
          gbuffer_normal,
          this.final_gi_texture_indirect_diffuse,
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

    this.shared_bindings.probe_depth_buffer = probe_depth_atlas;
    this.shared_bindings.sh_probes_buffer = sh_probes_curr;
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
