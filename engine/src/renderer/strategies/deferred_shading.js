// Core imports
import { global_dispatcher } from "../../core/dispatcher.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import { SharedFrameInfoBuffer } from "../../core/shared_data.js";

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
import { PostProcessStack } from "../post_process_stack.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";
import { ComputeRasterTaskQueue } from "../compute_raster_task_queue.js";
import { CullingPipeline } from "../pipelines/culling_pipeline.js";
import { DeferredDebugPipeline } from "../pipelines/deferred_debug_pipeline.js";
import { EnvironmentPipeline } from "../pipelines/environment_pipeline.js";
import { GBufferTargetsPipeline } from "../pipelines/gbuffer_targets_pipeline.js";
import { VisibilityBufferPipeline } from "../pipelines/visibility_buffer_pipeline.js";

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
import { profile_scope } from "../../utility/performance.js";
import {
  rgba16float_format,
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
  visibility_buffer_pipeline = null;
  prev_lighting_image = null;
  gi = null;
  vbao = null;
  rtao = null;
  reflections = null;
  bloom = null;
  as_vsm = null;
  debug_pipeline = null;
  gbuffer_targets_pipeline = null;
  environment_pipeline = null;

  setup(render_graph) {
    this.debug_pipeline = new DeferredDebugPipeline();
    this.culling_pipeline = new CullingPipeline();
    this.environment_pipeline = new EnvironmentPipeline();
    this.gbuffer_targets_pipeline = new GBufferTargetsPipeline();
    this.visibility_buffer_pipeline = new VisibilityBufferPipeline();

    this.gi = Renderer.get().get_gi_strategy_type() === GIStrategyType.DDGI ? new DDGI() : new PTGI();
    this.ao = Renderer.get().get_ao_strategy_type() === AOStrategyType.RTAO ? new RTAO() : new VBAO();
    this.reflections =
      Renderer.get().get_reflection_strategy_type() === ReflectionStrategyType.SSR ? new SSR() : null;

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
      } = this.visibility_buffer_pipeline.register_targets(render_graph);
      let {
        main_albedo_image,
        main_smra_image,
        main_normal_image,
        main_motion_emissive_image,
        main_depth_image,
        prev_depth_image,
        prev_normal_image,
        main_transparency_accum_image,
      } = this.gbuffer_targets_pipeline.create_targets(render_graph, {
        image_extent,
        force_recreate: this.force_recreate,
        include_prev_depth: true,
        include_prev_normal: true,
        include_transparency_accum: true,
      });

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
      this.gbuffer_targets_pipeline.add_clear_pass(render_graph, {
        pass_name: clear_g_buffer_pass_name,
        targets: {
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
          main_transparency_accum_image,
        },
        visibility_targets: [
          visibility_entity_image,
          visibility_surface_image,
          visibility_barycentric_image,
        ],
      });

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
      skybox_image = this.environment_pipeline.add_skybox_pass(render_graph, {
        pass_name: skydome_pass_name,
        image_extent,
        force_recreate: this.force_recreate,
      });

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🔄 PASS: G-Buffer Load State Configuration                                 │
      // │    Configure render targets to preserve existing content for next passes  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      this.gbuffer_targets_pipeline.add_set_load_op_pass(render_graph, {
        pass_name: reset_g_buffer_targets_pass_name,
        targets: {
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
          main_transparency_accum_image,
        },
        visibility_entity_image,
        load_op: load_op_load,
      });

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
      this.visibility_buffer_pipeline.add_depth_prepass(render_graph, {
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
        this.visibility_buffer_pipeline.add_visibility_raster_pass(render_graph, {
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

        this.visibility_buffer_pipeline.add_gbuffer_resolve_pass(render_graph, {
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
      this.debug_pipeline.add_geometry_passes(render_graph, {
        debug_view,
        draw_count,
        current_view,
        aabb_bounds,
        blas_gpu_data,
        blas_directory,
        blas_bvh2_nodes,
        object_instances,
        entity_transforms,
        mesh_asset_ids_buffer,
        entity_index_lookup,
        culling_pipeline: this.culling_pipeline,
        main_albedo_image,
        main_smra_image,
        main_normal_image,
        main_motion_emissive_image,
        main_depth_image,
      });

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
      this.debug_pipeline.add_overlay_pass(render_graph, {
        debug_view,
        image_extent,
        debug_texture_level: renderer.get_debug_texture_level(),
        post_processed_image,
        post_lighting_image: post_lighting_image_desc,
        prev_lighting_image: this.prev_lighting_image,
        prev_lighting,
        main_depth_image,
        main_normal_image,
        main_motion_emissive_image,
        visibility_entity_image,
        visibility_surface_image,
        meshlet_buffer,
        meshlet_vertex_buffer,
        meshlet_triangle_buffer,
        material_table_offset,
        material_palette,
        main_hzb_image,
        culling_pipeline: this.culling_pipeline,
        as_vsm: this.as_vsm,
        bloom: this.bloom,
        ao: this.ao,
        gi: this.gi,
        reflections: this.reflections,
        reflections_enabled,
      });

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
      this.gbuffer_targets_pipeline.add_set_load_op_pass(render_graph, {
        pass_name: reset_g_buffer_targets_pass_name,
        targets: {
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
          main_transparency_accum_image,
        },
        visibility_entity_image,
        load_op: load_op_clear,
      });

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
    this.visibility_buffer_pipeline.recreate_persistent_resources(image_extent, this.force_recreate);
  }

  _get_texture_pool(render_graph, pool_key) {
    const fallback_texture = TextureArrayPools.get_fallback_view();
    const texture =
      ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(`texture_pool_${pool_key}`)) ||
      fallback_texture;
    return render_graph.register_image(texture.config.name);
  };
}
