// Core imports
import { global_dispatcher } from "../../core/dispatcher.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import {
  SharedEnvironmentData,
  SharedFrameInfoBuffer,
} from "../../core/shared_data.js";

// ECS fragments
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";

// Renderer components
import { Renderer } from "../renderer.js";
import { Texture } from "../texture.js";
import { Material } from "../material.js";
import { MeshData } from "../mesh_data.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";
import { DebugOverlay } from "../debug_overlay.js";
import { PostProcessStack } from "../post_process_stack.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";
import { ComputeRasterTaskQueue } from "../compute_raster_task_queue.js";
import { CullingPipeline } from "../culling_pipeline.js";
import { VisibilityBuffer } from "../visibility_buffer.js";

// Types and utilities
import {
  RenderPassFlags,
  MaterialFamilyType,
  DebugDrawType,
  GIStrategyType,
  AOStrategyType,
  ReflectionStrategyType,
  CacheTypes,
} from "../renderer_types.js";
import { BVH } from "../../acceleration/bvh.js";
import { MeshBLAS } from "../../acceleration/mesh_blas.js";
import { clamp } from "../../utility/math.js";
import { profile_scope } from "../../utility/performance.js";
import {
  rgba8unorm_format,
  rgba16float_format,
  depth32float_format,
  one_one_blend_config,
  src_alpha_one_minus_src_alpha_blend_config,
  load_op_load,
  load_op_clear,
} from "../../utility/config_permutations.js";

// Specialized renderer components
import { PTGI } from "../global_illumination/ptgi.js";
import { DDGI } from "../global_illumination/ddgi.js";
import { VBAO } from "../global_illumination/vbao.js";
import { RTAO } from "../global_illumination/rtao.js";
import { AdaptiveSparseVirtualShadowMaps } from "../shadows/as_vsm.js";
import { SSR } from "../reflections/ssr.js";
import { Bloom } from "../bloom.js";
import { ResourceCache } from "../resource_cache.js";
import { TextureArrayPools } from "../texture_pool.js";
import {
  DEFAULT_LIGHT_CLIP_EXTENT,
  MAX_CLIPMAP_LEVELS,
  VSM_VIRTUAL_DIM,
  ATLAS_SIZE,
  TILE_SIZE,
} from "../shadows/shadow_utils.js";
import { Name } from "../../utility/names.js";

const resolution_change_event_name = "resolution_change";
const deferred_shading_profile_scope_name = "DeferredShadingStrategy.draw";
const transforms_name = "transforms";
const bounds_name = "bounds";
const light_fragment_name = "light_fragment";
const mesh_asset_id_name = "mesh_asset_id";

const main_albedo_image_config = {
  name: "main_albedo",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_smra_image_config = {
  name: "main_smra",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_normal_image_config = {
  name: "main_normal_0",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};
const main_normal_image2_config = {
  name: "main_normal_1",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_DST,
  force: false,
};
const main_motion_emissive_image_config = {
  name: "main_motion_emissive",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_transparency_accum_image_config = {
  name: "main_transparency_accum",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  blend: one_one_blend_config,
  force: false,
};
const main_depth_image_config = {
  name: "main_depth_0",
  format: depth32float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  force: false,
};
const main_depth_image2_config = {
  name: "main_depth_1",
  format: depth32float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  force: false,
};
const prev_lighting_image_config = {
  name: "prev_lighting",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_DST,
  mip_levels: 0,
  b_one_view_per_mip: true,
};

const skybox_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "skybox.wgsl",
    },
    fragment: {
      path: "skybox.wgsl",
    },
  },
  rasterizer_state: {
    cull_mode: "none",
  },
  depth_write_enabled: false,
};
const skybox_output_image_config = {
  name: "skybox_output",
  format: rgba8unorm_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const g_buffer_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "gbuffer_base.wgsl",
    },
    fragment: {
      path: "gbuffer_base.wgsl",
    },
  },
  depth_write_enabled: false,
  depth_stencil_compare_op: "less-equal",
};

const transparency_composite_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "transparency_composite.wgsl",
    },
    fragment: {
      path: "transparency_composite.wgsl",
    },
  },
  attachment_blend: src_alpha_one_minus_src_alpha_blend_config,
};

const prev_lighting_mip_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "reflections/ssr_lighting_mip.wgsl",
    },
  },
};

const dense_lights_buffer_config = {
  name: "dense_lights",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const compact_lights_pass_name = "compact_lights";
const compact_lights_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "system_compute/compact_lights.wgsl",
    },
  },
};

const deferred_lighting_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "deferred_lighting.wgsl",
      defines: {
        GI_ENABLED: false,
        SHADOWS_ENABLED: false,
        AO_ENABLED: false,
      },
    },
    fragment: {
      path: "deferred_lighting.wgsl",
      defines: {
        GI_ENABLED: false,
        SHADOWS_ENABLED: false,
        AO_ENABLED: false,
      },
    },
  },
};
const post_lighting_image_config = {
  name: "post_lighting",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const line_draw_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "line.wgsl",
    },
    fragment: {
      path: "line.wgsl",
    },
  },
  rasterizer_state: {
    cull_mode: "none",
  },
};

const debug_emit_entity_bounds_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "debug/debug_emit_entity_bounds_lines.wgsl",
    },
  },
};
const debug_emit_bvh2_nodes_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "debug/debug_emit_bvh2_nodes_lines.wgsl",
    },
  },
};
const debug_emit_blas_nodes_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "debug/debug_emit_blas_nodes_lines.wgsl",
    },
  },
};
const debug_find_closest_mesh_instances_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "debug/debug_find_closest_mesh_instances.wgsl",
    },
  },
};

const fullscreen_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "fullscreen.wgsl" },
  },
};

const clear_dirty_flags_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "system_compute/clear_dirty_flags.wgsl",
    },
  },
};

const swapchain_name = "swapchain";
const clear_g_buffer_pass_name = "clear_g_buffer";
const skydome_pass_name = "skydome_pass";
const transparency_composite_pass_name = "transparency_composite";
const reset_g_buffer_targets_pass_name = "reset_g_buffer_targets";
const lighting_pass_name = "lighting_pass";
const fullscreen_present_pass_name = "fullscreen_present_pass";

// Debug shader setups for AS-VSM debug views
export class DeferredShadingStrategy {
  initialized = false;
  force_recreate = false;
  force_reinit = false;
  culling_pipeline = null;
  visibility_buffer = null;
  prev_lighting_image = null;
  gi = null;
  vbao = null;
  rtao = null;
  reflections = null;
  bloom = null;
  as_vsm = null;
  debug_overlay = null;

  setup(render_graph) {
    this.debug_overlay = new DebugOverlay();
    this.culling_pipeline = new CullingPipeline();
    this.visibility_buffer = new VisibilityBuffer();

    const gi_strategy_type = Renderer.get().get_gi_strategy_type();
    const ao_strategy_type = Renderer.get().get_ao_strategy_type();
    const reflection_strategy_type = Renderer.get().get_reflection_strategy_type();
    this.gi = gi_strategy_type === GIStrategyType.DDGI ? new DDGI() : new PTGI();
    this.ao = ao_strategy_type === AOStrategyType.RTAO ? new RTAO() : new VBAO();
    this.reflections =
      reflection_strategy_type === ReflectionStrategyType.SSR ? new SSR() : null;
    this.bloom = new Bloom();
    this.as_vsm = new AdaptiveSparseVirtualShadowMaps({
      atlas_size: ATLAS_SIZE,
      tile_size: TILE_SIZE,
      virtual_dim: VSM_VIRTUAL_DIM,
      max_lods: MAX_CLIPMAP_LEVELS,
      clip0_extent: DEFAULT_LIGHT_CLIP_EXTENT,
    });

    global_dispatcher.on(
      resolution_change_event_name,
      this._recreate_persistent_resources.bind(this)
    );
    this._recreate_persistent_resources(render_graph);
  }

  draw(render_graph) {
    profile_scope(
      deferred_shading_profile_scope_name,
      this._draw_internal.bind(this, render_graph)
    );
  }

  refresh(render_graph, reinit = false) {
    this.force_recreate = true;
    this.force_reinit = reinit;
  }

  _draw_internal(render_graph) {
    profile_scope(deferred_shading_profile_scope_name, () => {
      if (!this.initialized || this.force_reinit) {
        this.setup(render_graph);
        this.initialized = true;
        this.force_reinit = false;
      }

      // ═══════════════════════════════════════════════════════════════════════════════
      // ⚙️  SETUP & INITIALIZATION PHASE
      // ═══════════════════════════════════════════════════════════════════════════════
      // Async content (like glTF callbacks) can create entities between simulation
      // flush and render; flush again here so culling sees a current dense row map.
      EntityManager.flush_gpu_buffers();
      MeshTaskQueue.sort_and_batch();
      ComputeTaskQueue.compile_pre_rg_passes(render_graph);

      this.culling_pipeline.reset();

      const renderer = Renderer.get();

      const current_view = SharedFrameInfoBuffer.get_view_index();
      const draw_count = MeshTaskQueue.get_total_draw_count();
      const meshlet_draw_count = MeshTaskQueue.get_total_meshlet_count();
      const debug_view = renderer.get_debug_draw_type();
      const image_extent = renderer.get_canvas_resolution();

      const shadows_enabled = renderer.is_shadows_enabled();
      const gi_enabled = renderer.is_gi_enabled();
      const gi_has_builtin_specular = renderer.get_gi_strategy_type() === GIStrategyType.PTGI;
      const ao_enabled = renderer.is_ao_enabled();
      const reflections_enabled = renderer.is_reflection_enabled() && !gi_has_builtin_specular;
      const depth_prepass_enabled = renderer.is_depth_prepass_enabled();

      if (this.force_recreate) {
        render_graph.mark_pass_cache_bind_groups_dirty(true /* pass_only */);
      }

      const prev_lighting = render_graph.register_image(this.prev_lighting_image.config.name);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📋 Register Core Entity & Transform Buffers                                │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const entity_flags_buffer = FragmentGpuBuffer.entity_flags_buffer;
      const entity_flags = render_graph.register_buffer(entity_flags_buffer.buffer.config.name);

      const entity_index_map_buffer = FragmentGpuBuffer.entity_index_map_buffer;
      const entity_index_lookup = render_graph.register_buffer(
        entity_index_map_buffer.buffer.config.name
      );

      const transforms_buffer = EntityManager.get_fragment_gpu_buffer(
        TransformFragment,
        transforms_name
      );
      const entity_transforms = render_graph.register_buffer(transforms_buffer.buffer.config.name);

      const bounds_buffer = EntityManager.get_fragment_gpu_buffer(TransformFragment, bounds_name);
      const aabb_bounds = render_graph.register_buffer(bounds_buffer.buffer.config.name);

      const aabb_gpu_data = BVH.to_gpu_data();
      const tlas_bvh_info = render_graph.register_buffer(
        aabb_gpu_data.bvh_info_buffer.config.name
      );

      const blas_gpu_data = MeshBLAS.to_gpu_data();
      const blas_directory = render_graph.register_buffer(
        blas_gpu_data.directory_buffer.config.name
      );
      const blas_bvh2_nodes = render_graph.register_buffer(
        blas_gpu_data.bvh2_nodes_buffer.config.name
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 Register Mesh & Instance Buffers                                        │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const object_instances = render_graph.register_buffer(
        MeshTaskQueue.get_object_instance_buffer().config.name
      );
      const meshlet_instances = render_graph.register_buffer(
        MeshTaskQueue.get_meshlet_instance_buffer().config.name
      );

      const mesh_asset_ids = EntityManager.get_fragment_gpu_buffer(
        StaticMeshFragment,
        mesh_asset_id_name
      );
      const mesh_asset_ids_buffer = render_graph.register_buffer(mesh_asset_ids.buffer.config.name);

      const material_offsets_buffer = EntityManager.get_fragment_gpu_buffer(
        StaticMeshFragment,
        "material_table_offset"
      );
      const material_table_offset = render_graph.register_buffer(
        material_offsets_buffer.buffer.config.name
      );

      const mesh_gpu_data = MeshData.to_gpu_data();
      const index_buffer = render_graph.register_buffer(MeshData.index_buffer.config.name);
      const meshlet_buffer = render_graph.register_buffer(mesh_gpu_data.meshlet_buffer.config.name);
      const meshlet_vertex_buffer = render_graph.register_buffer(
        mesh_gpu_data.meshlet_vertex_buffer.config.name
      );
      const meshlet_triangle_buffer = render_graph.register_buffer(
        mesh_gpu_data.meshlet_triangle_buffer.config.name
      );

      const material_params = render_graph.register_buffer(
        MaterialAllocationTable.params_buffer.config.name
      );
      const material_palette = render_graph.register_buffer(
        MaterialAllocationTable.palette_buffer.config.name
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 💡 Setup Lighting System                                                   │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const light_fragment_buffer = EntityManager.get_fragment_gpu_buffer(
        LightFragment,
        light_fragment_name
      );
      const lights = render_graph.register_buffer(light_fragment_buffer.buffer.config.name);

      dense_lights_buffer_config.size = (light_fragment_buffer.buffer.config.size / 4) + 4;
      const dense_lights = render_graph.create_buffer(dense_lights_buffer_config);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖼️  Create G-Buffer & Main Render Targets                                  │
      // └─────────────────────────────────────────────────────────────────────────────┘

      const { main_hzb_image } = this.culling_pipeline.register_targets(render_graph);
      const {
        visibility_entity_image,
        visibility_surface_image,
        visibility_barycentric_image,
      } = this.visibility_buffer.register_targets(render_graph);

      main_normal_image_config.width = image_extent.width;
      main_normal_image_config.height = image_extent.height;
      main_normal_image_config.force = this.force_recreate;

      main_normal_image2_config.width = image_extent.width;
      main_normal_image2_config.height = image_extent.height;
      main_normal_image2_config.force = this.force_recreate;

      main_depth_image_config.width = image_extent.width;
      main_depth_image_config.height = image_extent.height;
      main_depth_image_config.force = this.force_recreate;

      main_depth_image2_config.width = image_extent.width;
      main_depth_image2_config.height = image_extent.height;
      main_depth_image2_config.force = this.force_recreate;

      main_albedo_image_config.width = image_extent.width;
      main_albedo_image_config.height = image_extent.height;
      main_albedo_image_config.force = this.force_recreate;
      main_smra_image_config.width = image_extent.width;
      main_smra_image_config.height = image_extent.height;
      main_smra_image_config.force = this.force_recreate;
      main_motion_emissive_image_config.width = image_extent.width;
      main_motion_emissive_image_config.height = image_extent.height;
      main_motion_emissive_image_config.force = this.force_recreate;
      main_transparency_accum_image_config.width = image_extent.width;
      main_transparency_accum_image_config.height = image_extent.height;
      main_transparency_accum_image_config.force = this.force_recreate;

      let main_albedo_image = render_graph.create_image(main_albedo_image_config);
      let main_smra_image = render_graph.create_image(main_smra_image_config);
      let main_motion_emissive_image = render_graph.create_image(main_motion_emissive_image_config);
      let main_transparency_accum_image = render_graph.create_image(
        main_transparency_accum_image_config
      );
      let main_depth_image = render_graph.create_image(main_depth_image_config);
      let main_normal_image = render_graph.create_image(main_normal_image_config);
      let prev_normal_image = render_graph.create_image(main_normal_image2_config);
      let prev_depth_image = render_graph.create_image(main_depth_image2_config);

      let skybox_image = null;
      let post_lighting_image_desc = null;

      const texture_pool_albedo = this._get_texture_pool(render_graph, "albedo");
      const texture_pool_normal = this._get_texture_pool(render_graph, "normal");
      const texture_pool_roughness = this._get_texture_pool(render_graph, "roughness");
      const texture_pool_metallic = this._get_texture_pool(render_graph, "metallic");
      const texture_pool_ao = this._get_texture_pool(render_graph, "ao");
      const texture_pool_height = this._get_texture_pool(render_graph, "height");
      const texture_pool_specular = this._get_texture_pool(render_graph, "specular");
      const texture_pool_emission = this._get_texture_pool(render_graph, "emission");

      this.culling_pipeline.register_views(render_graph, {
        draw_count,
        main_hzb_image,
        aabb_bounds,
        object_instances,
        entity_index_lookup,
      });

      // ═══════════════════════════════════════════════════════════════════════════════
      // 🎨 RENDERING PIPELINE BEGINS
      // ═══════════════════════════════════════════════════════════════════════════════

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🧹 PASS: Init Views                                                         │
      // │    Initialize all views to a clean slate                                    │
      // └─────────────────────────────────────────────────────────────────────────────┘
      this.culling_pipeline.add_init_view_passes(render_graph, draw_count);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🧹 PASS: Clear G-Buffer Targets                                            │
      // │    Initialize all render targets to a clean slate                          │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        render_graph.add_pass(
          clear_g_buffer_pass_name,
          RenderPassFlags.Graphics,
          {
            outputs: [
              main_albedo_image,
              main_smra_image,
              main_normal_image,
              main_motion_emissive_image,
              visibility_entity_image,
              visibility_surface_image,
              visibility_barycentric_image,
              main_transparency_accum_image,
              main_depth_image,
            ],
            b_skip_pass_pipeline_setup: true,
            b_skip_pass_bind_group_setup: true,
          },
          (graph, frame_data, encoder) => { }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 💡 PASS: Compact Active Lights                                             │
      // │    Pack sparse light data into dense buffers for efficient access         │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        // Compute pass to compact active lights to dense buffer
        render_graph.add_pass(
          compact_lights_pass_name,
          RenderPassFlags.Compute,
          {
            shader_setup: compact_lights_shader_setup,
            inputs: [lights, dense_lights],
            outputs: [dense_lights],
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            // Reset light counters to zero (header u32[4])
            const dense_lights_buf = graph.get_physical_buffer(dense_lights);
            dense_lights_buf.write_raw(new Uint32Array([0, 0, 0, 0]), 0);
            // Dispatch compute to compact lights
            const max_light_count = EntityManager.get_max_rows();
            pass.dispatch((max_light_count + 128 - 1) / 128, 1, 1);
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌌 PASS: Skybox Rendering                                                  │
      // │    Render the environment skybox (or analytic skydome) to provide           |
      // |    distant lighting context                                                 │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        skybox_output_image_config.width = image_extent.width;
        skybox_output_image_config.height = image_extent.height;
        skybox_output_image_config.force = this.force_recreate;
        skybox_image = render_graph.create_image(skybox_output_image_config);

        const skydome_data = SharedEnvironmentData.get_skydome_data();
        const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

        const skybox = SharedEnvironmentData.get_skybox();
        const skybox_texture = render_graph.register_image(skybox.config.name);

        render_graph.add_pass(
          skydome_pass_name,
          RenderPassFlags.Graphics,
          {
            inputs: [skybox_texture, skydome_data_buffer],
            outputs: [skybox_image],
            shader_setup: skybox_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            MeshTaskQueue.draw_cube(pass);
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🔄 PASS: G-Buffer Load State Configuration                                 │
      // │    Configure render targets to preserve existing content for next passes  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        render_graph.add_pass(
          reset_g_buffer_targets_pass_name,
          RenderPassFlags.GraphLocal,
          {},
          (graph, frame_data, encoder) => {
            const albedo = graph.get_physical_image(main_albedo_image);
            const smra = graph.get_physical_image(main_smra_image);
            const normal = graph.get_physical_image(main_normal_image);
            const motion_emissive = graph.get_physical_image(main_motion_emissive_image);
            const entity_id = graph.get_physical_image(visibility_entity_image);
            const transparency_accum = graph.get_physical_image(main_transparency_accum_image);
            const depth = graph.get_physical_image(main_depth_image);

            if (albedo) {
              albedo.config.load_op = load_op_load;
            }
            if (smra) {
              smra.config.load_op = load_op_load;
            }
            if (normal) {
              normal.config.load_op = load_op_load;
            }
            if (motion_emissive) {
              motion_emissive.config.load_op = load_op_load;
            }
            if (entity_id) {
              entity_id.config.load_op = load_op_load;
            }
            if (transparency_accum) {
              transparency_accum.config.load_op = load_op_load;
            }
            if (depth) {
              depth.config.load_op = load_op_load;
            }
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 PASS: Frustum Culling (Phase 1 of 2-Pass Occlusion)                    │
      // │    Eliminate objects outside the camera's view frustum                     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const culling_pass_outputs = this.culling_pipeline.add_frustum_cull_passes(render_graph, {
        current_view,
        draw_count,
        meshlet_draw_count,
        entity_transforms,
        object_instances,
        meshlet_instances,
        entity_index_lookup,
        meshlet_buffer,
        force_recreate: this.force_recreate,
      });
      const {
        frustum_meshlet_list,
        frustum_meshlet_draw_args,
        occlusion_meshlet_list,
        occlusion_meshlet_draw_args,
      } = culling_pass_outputs;

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🏔️  PASS: Depth Pre-Pass                                                   │
      // │    Fill depth buffer early for better GPU efficiency and HZB generation   │
      // └─────────────────────────────────────────────────────────────────────────────┘
      this.visibility_buffer.add_depth_prepass(render_graph, {
        enabled: depth_prepass_enabled,
        meshlet_draw_count,
        depth_image: main_depth_image,
        frustum_meshlet_draw_args,
        inputs: [
          entity_transforms,
          object_instances,
          frustum_meshlet_list,
          meshlet_buffer,
          meshlet_vertex_buffer,
          meshlet_triangle_buffer,
          entity_index_lookup,
          material_params,
          material_table_offset,
          material_palette,
          texture_pool_albedo,
        ],
      });

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌫️  PASS: Occlusion Culling (Phase 2 of 2-Pass Occlusion)                 │
      // │    Use HZB to eliminate objects hidden behind other geometry               │
      // └─────────────────────────────────────────────────────────────────────────────┘
      this.culling_pipeline.add_occlusion_cull_passes(render_graph, {
        current_view,
        draw_count,
        meshlet_draw_count,
        depth_prepass_enabled,
        main_hzb_image,
        main_depth_image,
        prev_depth_image,
        entity_transforms,
        object_instances,
        entity_index_lookup,
        meshlet_buffer,
        culling_pass_outputs,
      });

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖥️  PASS: Compute Rasterization                                            │
      // │    Software rasterization for particles and small geometry                  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      // TODO: Automatically run software rasterization over triangle clusters that fall within some maximum screen size
      {
        // Rasterize particle positions into the G-Buffer (albedo & depth)
        ComputeRasterTaskQueue.compile_rg_passes(render_graph, [
          main_albedo_image,
          main_depth_image,
        ]);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎨 PASS: G-Buffer Base Rendering                                           │
      // │    Fill G-Buffer with geometry data (albedo, normals, material props)     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        this.visibility_buffer.add_visibility_raster_pass(render_graph, {
          meshlet_draw_count,
          depth_prepass_enabled,
          current_view,
          depth_image: main_depth_image,
          occlusion_meshlet_draw_args,
          inputs: [
            entity_transforms,
            object_instances,
            occlusion_meshlet_list,
            meshlet_buffer,
            meshlet_vertex_buffer,
            meshlet_triangle_buffer,
            entity_index_lookup,
            material_params,
            material_table_offset,
            material_palette,
            texture_pool_albedo,
          ],
        });

        this.visibility_buffer.add_gbuffer_resolve_pass(render_graph, {
          meshlet_draw_count,
          current_view,
          depth_image: main_depth_image,
          inputs: [
            entity_transforms,
            meshlet_buffer,
            meshlet_vertex_buffer,
            meshlet_triangle_buffer,
            material_params,
            material_table_offset,
            material_palette,
            texture_pool_albedo,
            texture_pool_normal,
            texture_pool_roughness,
            texture_pool_metallic,
            texture_pool_ao,
            texture_pool_height,
            texture_pool_specular,
            texture_pool_emission,
          ],
          outputs: [
            main_albedo_image,
            main_smra_image,
            main_normal_image,
            main_motion_emissive_image,
          ],
        });

        g_buffer_shader_setup.depth_write_enabled = false;
        g_buffer_shader_setup.depth_stencil_compare_op = "less-equal";

        const material_buckets = MeshTaskQueue.get_material_buckets();
        for (let i = 0; i < material_buckets.length; i++) {
          const material_id = material_buckets[i];
          const material = Material.get(material_id);
          if (material.family !== MaterialFamilyType.Transparent) {
            continue;
          }

          render_graph.add_pass(
            `g_buffer_${material.template.name}_${material_id}`,
            RenderPassFlags.Graphics,
            {
              inputs: [
                entity_transforms,
                entity_flags,
                object_instances,
                this.culling_pipeline.get_occlusion_visibility_buffer(current_view, 0),
                entity_index_lookup,
              ],
              outputs: [
                main_transparency_accum_image,
                main_smra_image,
                main_normal_image,
                main_motion_emissive_image,
                main_depth_image,
              ],
              shader_setup: g_buffer_shader_setup,
              b_skip_pass_pipeline_setup: true,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              MeshTaskQueue.submit_material_indexed_indirect_draws(pass, material_id, current_view);
            }
          );
        }
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌊 PASS: Transparency Composite                                            │
      // │    Blend transparent objects using weighted-blended order-independent     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      render_graph.add_pass(
        transparency_composite_pass_name,
        RenderPassFlags.Graphics,
        {
          inputs: [main_transparency_accum_image],
          outputs: [main_albedo_image],
          shader_setup: transparency_composite_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);

          MeshTaskQueue.draw_quad(pass);
        }
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📏 PASS: Debug Entity Bounds and BVH                                        │
      // │    Render entity bounds and BVH for visualization                           │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (
        debug_view === DebugDrawType.EntityBounds ||
        debug_view === DebugDrawType.BVH ||
        debug_view === DebugDrawType.BLAS_Bounds
      ) {
        let max_nodes_debug = BVH.bvh_size;
        switch (debug_view) {
          case DebugDrawType.BLAS_Bounds:
            max_nodes_debug = MeshBLAS.bounds_size;
            break;
          default:
            break;
        }
        const max_lines = Math.min(max_nodes_debug * 12 * 20, 256000 * 12 * 20);

        const debug_line_data_buf = render_graph.create_buffer({
          name: "debug_line_data",
          size: max_lines,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        if (debug_view === DebugDrawType.EntityBounds) {
          render_graph.add_pass(
            "debug_emit_bounds_lines",
            RenderPassFlags.Compute,
            {
              inputs: [debug_line_data_buf, aabb_bounds],
              outputs: [debug_line_data_buf],
              shader_setup: debug_emit_entity_bounds_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              pass.dispatch(Math.ceil(BVH.bvh_size / 64), 1, 1);
            }
          );
        } else if (debug_view === DebugDrawType.BLAS_Bounds) {
          // Calculate mesh directory size (directory buffer size / bytes per entry / 4 bytes per u32)
          const directory_buffer_size = blas_gpu_data.directory_buffer.config.size;
          const directory_entry_size = 6; // [bvh2_base, bvh2_cap, leaf_count, first_vertex, first_index, padding]
          const mesh_count = Math.floor(directory_buffer_size / (directory_entry_size * 4));

          // Compact per-mesh preprocessing buffers
          const closest_entities_per_mesh_buf = render_graph.create_buffer({
            name: "closest_entities_per_mesh",
            size: mesh_count, // u32 per mesh
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });

          const closest_distances_per_mesh_buf = render_graph.create_buffer({
            name: "closest_distances_per_mesh",
            size: mesh_count, // f32 per mesh
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });

          // ┌─────────────────────────────────────────────────────────────────────────────┐
          // │ 🔍 PASS: Find Closest Mesh Instances                                       │
          // │    Compact preprocessing to find closest entity per mesh asset              │
          // └─────────────────────────────────────────────────────────────────────────────┘
          render_graph.add_pass(
            "debug_init_closest_distances",
            RenderPassFlags.GraphLocal,
            {},
            (graph, frame_data, encoder) => {
              // Initialize distances to infinity
              const distances_buf = graph.get_physical_buffer(closest_distances_per_mesh_buf);
              const infinity_array = new Float32Array(mesh_count);
              infinity_array.fill(Number.MAX_VALUE);
              distances_buf.write(infinity_array);
            }
          );

          render_graph.add_pass(
            "debug_find_closest_instances",
            RenderPassFlags.Compute,
            {
              inputs: [
                closest_entities_per_mesh_buf,
                closest_distances_per_mesh_buf,
                object_instances,
                this.culling_pipeline.get_frustum_visibility_buffer(current_view, 0),
                entity_transforms,
                mesh_asset_ids_buffer,
                entity_index_lookup,
              ],
              outputs: [closest_entities_per_mesh_buf, closest_distances_per_mesh_buf],
              shader_setup: debug_find_closest_mesh_instances_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              pass.dispatch(Math.ceil(draw_count / 64), 1, 1);
            }
          );

          render_graph.add_pass(
            "debug_emit_blas_bounds_lines",
            RenderPassFlags.Compute,
            {
              inputs: [
                debug_line_data_buf,
                blas_directory,
                entity_transforms,
                closest_entities_per_mesh_buf,
                blas_bvh2_nodes,
              ],
              outputs: [debug_line_data_buf],
              shader_setup: debug_emit_blas_nodes_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              const x_dispatch = Math.ceil(max_nodes_debug / 128);
              const y_dispatch = Math.ceil(mesh_count / 2);
              pass.dispatch(x_dispatch, y_dispatch, 1);
            }
          );
        } else {
          // Debug BVH: emit lines from BVH2 nodes
          render_graph.add_pass(
            "debug_emit_bvh2_lines",
            RenderPassFlags.Compute,
            {
              inputs: [debug_line_data_buf, aabb_bounds],
              outputs: [debug_line_data_buf],
              shader_setup: debug_emit_bvh2_nodes_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              pass.dispatch(Math.ceil(max_nodes_debug / 64), 1, 1);
            }
          );
        }

        render_graph.add_pass(
          "debug_line_draw",
          RenderPassFlags.Graphics,
          {
            inputs: [debug_line_data_buf],
            outputs: [
              main_albedo_image,
              main_smra_image,
              main_normal_image,
              main_motion_emissive_image,
              main_depth_image,
            ],
            shader_setup: line_draw_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            MeshTaskQueue.draw_quad(pass, max_lines / 12);
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌚 PASS: Adaptive Sparse Virtual Shadow Maps                               │
      // │    High-quality, efficient shadow mapping with virtual memory management  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (shadows_enabled) {
        this.as_vsm.add_passes(render_graph, {
          depth_texture: main_depth_image,
          entity_flags: entity_flags,
          aabb_bounds: aabb_bounds,
          lights: lights,
          dense_lights_buffer: dense_lights,
          transforms_buffer: entity_transforms,
          object_instances: object_instances,
          entity_index_lookup: entity_index_lookup,
          frustum_culler: this.culling_pipeline.get_frustum_culler(),
          force_recreate: this.force_recreate,
          debug_view: debug_view,
          draw_count: draw_count,
        });
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌟 PASS: Real-Time Global Illumination                                     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (gi_enabled) {
        this.gi.add_passes(
          render_graph,
          image_extent.width,
          image_extent.height,
          main_depth_image,
          prev_depth_image,
          main_normal_image,
          prev_normal_image,
          main_albedo_image,
          main_smra_image,
          main_motion_emissive_image,
          aabb_bounds,
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          entity_transforms,
          index_buffer,
          dense_lights,
          draw_count,
          main_hzb_image,
          this.force_recreate
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌟 PASS: AO                                                                │
      // │    Ambient Occlusion using Real-Time or Ground Truth methods                │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (ao_enabled) {
        this.ao.add_passes(
          render_graph,
          image_extent.width,
          image_extent.height,
          main_normal_image,
          prev_normal_image,
          main_albedo_image,
          main_smra_image,
          main_motion_emissive_image,
          main_depth_image,
          prev_depth_image,
          main_hzb_image,
          aabb_bounds,
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          entity_transforms,
          index_buffer,
          dense_lights,
          this.force_recreate
        );
      }


      if (reflections_enabled) {
        this.reflections.add_passes(
          render_graph,
          image_extent.width,
          image_extent.height,
          main_normal_image,
          main_depth_image,
          prev_normal_image,
          prev_depth_image,
          main_motion_emissive_image,
          main_smra_image,
          prev_lighting,
          main_hzb_image,
          this.force_recreate
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 💡 PASS: Deferred Lighting                                                 │
      // │    Combine G-Buffer data with lights to produce final shaded results      │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        deferred_lighting_shader_setup.force_recreate = this.force_recreate;

        post_lighting_image_config.width = image_extent.width;
        post_lighting_image_config.height = image_extent.height;
        post_lighting_image_config.force = this.force_recreate;
        post_lighting_image_desc = render_graph.create_image(post_lighting_image_config);

        const lighting_inputs = [
          skybox_image,
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
          dense_lights,
        ];

        deferred_lighting_shader_setup.pipeline_shaders.vertex.defines.GI_ENABLED = gi_enabled;
        deferred_lighting_shader_setup.pipeline_shaders.fragment.defines.GI_ENABLED = gi_enabled;

        if (gi_enabled) {
          lighting_inputs.push(
            this.gi.final_gi_texture_direct,
            this.gi.final_gi_texture_indirect_diffuse,
            reflections_enabled
              ? this.reflections.reflection_texture
              : this.gi.final_gi_texture_indirect_specular
          );
        }

        deferred_lighting_shader_setup.pipeline_shaders.vertex.defines.SHADOWS_ENABLED =
          shadows_enabled;
        deferred_lighting_shader_setup.pipeline_shaders.fragment.defines.SHADOWS_ENABLED =
          shadows_enabled;

        if (shadows_enabled) {
          lighting_inputs.push(
            this.as_vsm.shadow_atlas_buf,
            this.as_vsm.page_table,
            this.as_vsm.page_offset,
            this.as_vsm.settings_buf
          );
        }

        deferred_lighting_shader_setup.pipeline_shaders.vertex.defines.AO_ENABLED = ao_enabled;
        deferred_lighting_shader_setup.pipeline_shaders.fragment.defines.AO_ENABLED =
          ao_enabled;

        if (ao_enabled) {
          lighting_inputs.push(this.ao.ao_texture, this.ao.bent_normal_texture);
        }

        render_graph.add_pass(
          lighting_pass_name,
          RenderPassFlags.Graphics,
          {
            inputs: lighting_inputs,
            outputs: [post_lighting_image_desc],
            shader_setup: deferred_lighting_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            MeshTaskQueue.draw_quad(pass);
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🔍 PASS: GI Debug Visualizations                                          │
      // │    - World Cache: Shows spatial hash cached radiance                      │
      // │    (displayed via debug overlay, doesn't affect main rendering pipeline)  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (
        gi_enabled &&
        (debug_view === DebugDrawType.GI_WorldCache ||
          debug_view === DebugDrawType.GI_Probes)
      ) {
        this.gi.add_debug_passes(
          render_graph,
          image_extent.width,
          image_extent.height,
          main_normal_image,
          main_depth_image,
          post_lighting_image_desc,
          debug_view,
          this.force_recreate
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ ✨ PASS: Bloom Post-Processing                                             │
      // │    Multi-pass gaussian blur to create beautiful light bleeding effects    │
      // └─────────────────────────────────────────────────────────────────────────────┘

      this.bloom.add_passes(
        render_graph,
        image_extent.width,
        image_extent.height,
        post_lighting_image_desc,
        this.force_recreate
      );
      const curr_post_bloom = this.bloom.output_image;

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📋 PASS: Copy History                                                    │
      // │    Copy current bloom result into prev_lighting for the next frame        │
      // └─────────────────────────────────────────────────────────────────────────────┘
      render_graph.add_pass(
        "copy_history",
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          const curr_final_lighting = graph.get_physical_image(curr_post_bloom);
          const prev_final_lighting = graph.get_physical_image(prev_lighting);
          prev_final_lighting.copy_texture(encoder, curr_final_lighting);

          const curr_normal = graph.get_physical_image(main_normal_image);
          const prev_normal = graph.get_physical_image(prev_normal_image);
          if (prev_normal) {
            prev_normal.copy_texture(encoder, curr_normal);
          }

          const curr_depth = graph.get_physical_image(main_depth_image);
          const prev_depth = graph.get_physical_image(prev_depth_image);
          if (prev_depth) {
            prev_depth.copy_texture(encoder, curr_depth);
          }
        }
      );

      let prev_lighting_mip_params_chain = [];
      for (let i = 1; i < this.prev_lighting_image.config.mip_levels; i++) {
        prev_lighting_mip_params_chain.push(
          render_graph.create_buffer({
            name: `prev_lighting_mip_params_${i}`,
            data: [0.0, 0.0, 0.0, 0.0],
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          })
        );

        render_graph.add_pass(
          `prev_lighting_mip_${i}`,
          RenderPassFlags.Compute,
          {
            inputs: [prev_lighting, prev_lighting, prev_lighting_mip_params_chain[i - 1]],
            outputs: [prev_lighting],
            input_views: [i, i + 1],
            shader_setup: prev_lighting_mip_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const prevLighting = graph.get_physical_image(prev_lighting);
            const params = graph.get_physical_buffer(prev_lighting_mip_params_chain[i - 1]);

            const srcWidth = Math.max(1, prevLighting.config.width >> (i - 1));
            const srcHeight = Math.max(1, prevLighting.config.height >> (i - 1));
            const dstWidth = Math.max(1, prevLighting.config.width >> i);
            const dstHeight = Math.max(1, prevLighting.config.height >> i);

            params.write([srcWidth, srcHeight, dstWidth, dstHeight]);
            pass.dispatch((dstWidth + 7) / 8, (dstHeight + 7) / 8, 1);
          }
        );
      }

      // Use post-bloom color for antialiased_scene_color_desc
      const antialiased_scene_color_desc = curr_post_bloom;

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎭 PASS: Post-Processing Stack                                              │
      // │    Apply final image enhancements (tone mapping, color grading, etc.)     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const post_processed_image = PostProcessStack.compile_passes(
        0,
        render_graph,
        post_lighting_image_config,
        antialiased_scene_color_desc,
        main_depth_image,
        main_normal_image
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎭 PASS: Debug Overlay                                                     │
      // │    Draw debug overlay on the final output image                            │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (debug_view !== DebugDrawType.None) {
        switch (debug_view) {
          case DebugDrawType.Wireframe:
            break;
          case DebugDrawType.Depth:
            this.debug_overlay.set_properties(
              main_depth_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.Depth
            );
            break;
          case DebugDrawType.Normal:
            this.debug_overlay.set_properties(
              main_normal_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.Normal
            );
            break;
          case DebugDrawType.Emissive:
            this.debug_overlay.set_properties(
              main_motion_emissive_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.Emissive,
              0, // texture_level
              [0.0, 0.0, 0.0, 1.0], // channel_mask (alpha only)
              1 // Use channels
            );
            break;
          case DebugDrawType.Motion:
            this.debug_overlay.set_properties(
              [
                main_motion_emissive_image,
                main_depth_image,
                post_lighting_image_desc,
              ],
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.Motion
            );
            break;
          case DebugDrawType.EntityId:
            this.debug_overlay.set_properties(
              visibility_entity_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.EntityId
            );
            break;
          case DebugDrawType.VisibilityMaterialId:
            this.debug_overlay.set_properties(
              [
                visibility_entity_image,
                visibility_surface_image,
                meshlet_buffer,
                meshlet_vertex_buffer,
                meshlet_triangle_buffer,
                material_table_offset,
                material_palette,
              ],
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.VisibilityMaterialId
            );
            break;
          case DebugDrawType.VisibilityEntityId:
            this.debug_overlay.set_properties(
              visibility_entity_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.VisibilityEntityId
            );
            break;
          case DebugDrawType.VisibilityMeshletId:
            this.debug_overlay.set_properties(
              [
                visibility_entity_image,
                visibility_surface_image,
              ],
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.VisibilityMeshletId
            );
            break;
          case DebugDrawType.VisibilityTriangleId:
            this.debug_overlay.set_properties(
              [
                visibility_entity_image,
                visibility_surface_image,
              ],
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.VisibilityTriangleId
            );
            break;
          case DebugDrawType.HZB:
            const hzb_max_level = Math.max(
              0,
              this.culling_pipeline.get_hzb_mip_level_count() - 1
            );
            const hzb_texture_level = Math.min(
              renderer.get_debug_texture_level(),
              hzb_max_level
            );
            this.debug_overlay.set_properties(
              main_hzb_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.HZB,
              hzb_texture_level + 1
            );
            break;
          case DebugDrawType.ASVSM_ShadowAtlas:
            this.debug_overlay.set_properties(
              this.as_vsm.debug_shadow_atlas_image,
              0,
              0,
              Math.min(image_extent.width, image_extent.height) * 0.35,
              Math.min(image_extent.width, image_extent.height) * 0.35,
              DebugDrawType.ASVSM_ShadowAtlas
            );
            break;
          case DebugDrawType.ASVSM_ShadowPageTable:
            this.debug_overlay.set_properties(
              this.as_vsm.debug_page_table_image,
              0,
              0,
              Math.min(image_extent.width, image_extent.height) * 0.25,
              Math.min(image_extent.width, image_extent.height) * 0.25,
              DebugDrawType.ASVSM_ShadowPageTable
            );
            break;
          case DebugDrawType.ASVSM_TileOverlay:
            this.debug_overlay.set_properties(
              this.as_vsm.debug_tile_overlay_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.ASVSM_TileOverlay
            );
            break;
          case DebugDrawType.ASVSM_TileRenderOutput:
            this.debug_overlay.set_properties(
              this.as_vsm.debug_tile_render_output_image,
              0,
              0,
              Math.min(image_extent.width, image_extent.height) * 0.3,
              Math.min(image_extent.width, image_extent.height) * 0.3,
              DebugDrawType.ASVSM_TileRenderOutput
            );
            break;
          case DebugDrawType.ASVSM_DirtyTiles:
            this.debug_overlay.set_properties(
              this.as_vsm.debug_dirty_tiles_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.ASVSM_DirtyTiles
            );
            break;
          case DebugDrawType.Bloom:
            this.debug_overlay.set_properties(
              this.bloom.debug_bloom_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.Bloom
            );
            break;
          case DebugDrawType.AO:
            this.debug_overlay.set_properties(
              this.ao.ao_blur_texture || this.ao.ao_texture,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.AO
            );
            break;
          case DebugDrawType.BentNormal:
            this.debug_overlay.set_properties(
              this.ao.bent_normal_texture,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.BentNormal
            );
            break;
          case DebugDrawType.GI_Direct:
            this.debug_overlay.set_properties(
              this.gi.final_gi_texture_direct,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.GI_Direct
            );
            break;
          case DebugDrawType.GI_Specular:
            this.debug_overlay.set_properties(
              reflections_enabled
                ? this.reflections.reflection_texture
                : this.gi.final_gi_texture_indirect_specular,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.GI_Specular
            );
            break;
          case DebugDrawType.GI_Diffuse:
            this.debug_overlay.set_properties(
              this.gi.final_gi_texture_indirect_diffuse,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.GI_Diffuse
            );
            break;
          case DebugDrawType.GI_WorldCache:
            this.debug_overlay.set_properties(
              this.gi.debug_texture,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.GI_WorldCache
            );
            break;
          case DebugDrawType.GI_Probes:
            this.debug_overlay.set_properties(
              this.gi.debug_texture,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.GI_Probes
            );
            break;
          case DebugDrawType.GI_Reflections:
            this.debug_overlay.set_properties(
              reflections_enabled
                ? this.reflections.reflection_texture
                : this.gi.final_gi_texture_indirect_specular,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.GI_Reflections
            );
            break;
          case DebugDrawType.PrevLightingPyramid:
            {
              const max_level = Math.max(
                0,
                this.prev_lighting_image.config.mip_levels - 1
              );
              const texture_level = Math.min(
                renderer.get_debug_texture_level(),
                max_level
              );
              this.debug_overlay.set_properties(
                prev_lighting,
                0,
                0,
                image_extent.width,
                image_extent.height,
                DebugDrawType.PrevLightingPyramid,
                texture_level + 1
              );
            }
            break;
          default:
            break;
        }
        this.debug_overlay.add_pass(render_graph, post_processed_image);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖼️  PASS: Final Presentation                                               │
      // │    Present the final rendered image to the screen swapchain               │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        const swapchain_image = Texture.create_from_texture(
          renderer.context.getCurrentTexture(),
          swapchain_name
        );

        const rg_output_image = render_graph.register_image(swapchain_image.config.name);

        render_graph.add_pass(
          fullscreen_present_pass_name,
          RenderPassFlags.Graphics | RenderPassFlags.Present,
          {
            inputs: [post_processed_image],
            outputs: [rg_output_image],
            shader_setup: fullscreen_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            MeshTaskQueue.draw_quad(pass);
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🧽 PASS: G-Buffer Clear State Reset                                        │
      // │    Reset render targets to clear state for next frame                     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        render_graph.add_pass(
          reset_g_buffer_targets_pass_name,
          RenderPassFlags.GraphLocal,
          {},
          (graph, frame_data, encoder) => {
            const albedo = graph.get_physical_image(main_albedo_image);
            const smra = graph.get_physical_image(main_smra_image);
            const normal = graph.get_physical_image(main_normal_image);
            const motion_emissive = graph.get_physical_image(main_motion_emissive_image);
            const entity_id = graph.get_physical_image(visibility_entity_image);
            const transparency_accum = graph.get_physical_image(main_transparency_accum_image);
            const depth = graph.get_physical_image(main_depth_image);

            if (albedo) {
              albedo.config.load_op = load_op_clear;
            }
            if (smra) {
              smra.config.load_op = load_op_clear;
            }
            if (normal) {
              normal.config.load_op = load_op_clear;
            }
            if (motion_emissive) {
              motion_emissive.config.load_op = load_op_clear;
            }
            if (entity_id) {
              entity_id.config.load_op = load_op_clear;
            }
            if (transparency_accum) {
              transparency_accum.config.load_op = load_op_clear;
            }
            if (depth) {
              depth.config.load_op = load_op_clear;
            }
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🚩 PASS: Clear Entity Dirty Flags                                          │
      // │    Reset entity modification flags for next frame's change detection      │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        render_graph.add_pass(
          "clear_dirty_flags",
          RenderPassFlags.Compute,
          {
            inputs: [entity_flags],
            outputs: [entity_flags],
            shader_setup: clear_dirty_flags_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(Math.ceil(EntityManager.get_max_rows() / 128), 1, 1);
          }
        );
      }

      // ═══════════════════════════════════════════════════════════════════════════════
      // 🏁 PIPELINE FINALIZATION
      // ═══════════════════════════════════════════════════════════════════════════════

      this.force_recreate = false;

      ComputeTaskQueue.compile_post_rg_passes(render_graph);

      ComputeTaskQueue.reset();

      render_graph.submit();
    });
  }

  _recreate_persistent_resources(render_graph) {
    this.force_recreate = true;

    const image_extent = Renderer.get().get_canvas_resolution();

    prev_lighting_image_config.mip_levels = Math.max(
      1,
      Math.max(
        Math.floor(Math.log2(image_extent.width)),
        Math.floor(Math.log2(image_extent.height))
      )
    );
    prev_lighting_image_config.width = image_extent.width;
    prev_lighting_image_config.height = image_extent.height;
    prev_lighting_image_config.force = this.force_recreate;

    this.prev_lighting_image = Texture.create(prev_lighting_image_config);

    this.culling_pipeline.recreate_persistent_resources(image_extent, this.force_recreate);
    this.visibility_buffer.recreate_persistent_resources(image_extent, this.force_recreate);
  }

  _get_texture_pool(render_graph, pool_key) {
    const fallback_texture = TextureArrayPools.get_fallback_view();
    const texture =
      ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(`texture_pool_${pool_key}`)) ||
      fallback_texture;
    return render_graph.register_image(texture.config.name);
  };
}
