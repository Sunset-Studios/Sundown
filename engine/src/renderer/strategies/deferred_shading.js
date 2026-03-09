// Core imports
import { global_dispatcher } from "../../core/dispatcher.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import {
  SharedEnvironmentData,
  SharedViewBuffer,
  SharedFrameInfoBuffer,
} from "../../core/shared_data.js";

// ECS fragments
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "../../core/ecs/fragments/visibility_fragment.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";

// Renderer components
import { Renderer } from "../renderer.js";
import { Texture } from "../texture.js";
import { Material } from "../material.js";
import { MeshData } from "../mesh_data.js";
import { DebugOverlay } from "../debug_overlay.js";
import { PostProcessStack } from "../post_process_stack.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";
import { ComputeRasterTaskQueue } from "../compute_raster_task_queue.js";
import { FrustumCuller } from "../cull/frustum_culler.js";
import { OcclusionCuller } from "../cull/occlusion_culler.js";

// Types and utilities
import {
  RenderPassFlags,
  MaterialFamilyType,
  DebugDrawType,
  GIStrategyType,
  AOStrategyType,
  ReflectionStrategyType,
} from "../renderer_types.js";
import { BVH } from "../../acceleration/bvh.js";
import { MeshBLAS } from "../../acceleration/mesh_blas.js";
import { npot, ppot, clamp } from "../../utility/math.js";
import { profile_scope } from "../../utility/performance.js";
import {
  rgba8unorm_format,
  rgba16float_format,
  depth32float_format,
  rgba32float_format,
  r32float_format,
  r32uint_format,
  one_one_blend_config,
  src_alpha_one_minus_src_alpha_blend_config,
  load_op_load,
  load_op_clear,
} from "../../utility/config_permutations.js";

// Specialized renderer components
import { PTGI } from "../global_illumination/ptgi.js";
import { DDGI } from "../global_illumination/ddgi.js";
import { GTAO } from "../global_illumination/gtao.js";
import { RTAO } from "../global_illumination/rtao.js";
import { AdaptiveSparseVirtualShadowMaps } from "../shadows/as_vsm.js";
import { SSR } from "../reflections/ssr.js";
import {
  DEFAULT_LIGHT_CLIP_EXTENT,
  MAX_CLIPMAP_LEVELS,
  VSM_VIRTUAL_DIM,
  ATLAS_SIZE,
  TILE_SIZE,
} from "../shadows/shadow_utils.js";

const resolution_change_event_name = "resolution_change";
const deferred_shading_profile_scope_name = "DeferredShadingStrategy.draw";
const transforms_name = "transforms";
const bounds_name = "bounds";
const occluder_name = "occluder";
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
const main_position_image_config = {
  name: "main_position_0",
  format: rgba32float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};
const main_position_image2_config = {
  name: "main_position_1",
  format: rgba32float_format,
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
  name: "main_depth",
  format: depth32float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
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

const depth_only_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "gbuffer_base.wgsl",
      defines: { DEPTH_ONLY: true },
    },
    fragment: {
      path: "gbuffer_base.wgsl",
      defines: { DEPTH_ONLY: true },
    },
  },
  depth_write_enabled: true,
  depth_stencil_compare_op: "less",
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

const hzb_reduce_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/hzb_reduce.wgsl",
    },
  },
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

const bloom_downsample_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "effects/bloom_downsample.wgsl",
    },
  },
};
const bloom_upsample_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "effects/bloom_upsample.wgsl",
    },
  },
};
const bloom_resolve_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "fullscreen.wgsl",
    },
    fragment: {
      path: "effects/bloom_resolve.wgsl",
    },
  },
};
const bloom_resolve_params_config = {
  name: "bloom_resolve_params",
  data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};
const post_bloom_color_image_config = {
  name: "post_bloom_color",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  force: false,
};
const bloom_params = [
  1.1 /* final exposure */, 0.001 /* bloom intensity */, 0.1 /* bloom threshold */,
  0.2 /* bloom knee */, 0.0 /* near plane (attenuation starts) */,
  50.0 /* far plane (full bloom) */,
];

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

const hzb_image_config = {
  name: "hzb",
  format: r32float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  mip_levels: 0,
  b_one_view_per_mip: true,
  force: false,
};
const entity_id_image_config = {
  name: "entity_id",
  format: r32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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

const swapchain_name = "swapchain";
const clear_g_buffer_pass_name = "clear_g_buffer";
const skydome_pass_name = "skydome_pass";
const depth_prepass_name = "depth_prepass";
const transparency_composite_pass_name = "transparency_composite";
const reset_g_buffer_targets_pass_name = "reset_g_buffer_targets";
const lighting_pass_name = "lighting_pass";
const bloom_resolve_pass_name = "bloom_resolve_pass";
const fullscreen_present_pass_name = "fullscreen_present_pass";

// Debug shader setups for AS-VSM debug views
export class DeferredShadingStrategy {
  initialized = false;
  force_recreate = false;
  force_reinit = false;
  hzb_image = null;
  entity_id_image = null;
  prev_lighting_image = null;
  gi = null;
  gtao = null;
  rtao = null;
  reflections = null;
  as_vsm = null;
  debug_overlay = null;
  frustum_culler = null;
  occlusion_culler = null;

  setup(render_graph) {
    this.debug_overlay = new DebugOverlay();

    const gi_strategy_type = Renderer.get().get_gi_strategy_type();
    const ao_strategy_type = Renderer.get().get_ao_strategy_type();
    const reflection_strategy_type = Renderer.get().get_reflection_strategy_type();
    this.gi = gi_strategy_type === GIStrategyType.DDGI ? new DDGI() : new PTGI();
    this.ao = ao_strategy_type === AOStrategyType.GTAO ? new GTAO() : new RTAO();
    this.reflections =
      reflection_strategy_type === ReflectionStrategyType.SSR ? new SSR() : null;
    this.as_vsm = new AdaptiveSparseVirtualShadowMaps({
      atlas_size: ATLAS_SIZE,
      tile_size: TILE_SIZE,
      virtual_dim: VSM_VIRTUAL_DIM,
      max_lods: MAX_CLIPMAP_LEVELS,
      clip0_extent: DEFAULT_LIGHT_CLIP_EXTENT,
    });

    this.frustum_culler = new FrustumCuller(
      null,
      /* additional_data */ {
        aabb_bounds: 0,
        object_instances: 0,
        main_entity_id_image: 0,
      }
    );

    this.occlusion_culler = new OcclusionCuller(
      this.frustum_culler,
      /* additional_data */ {
        aabb_bounds: 0,
        object_instances: 0,
        main_hzb_image: 0,
        main_entity_id_image: 0,
        entity_occluders: 0,
      }
    );

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

      this.frustum_culler.reset();
      this.occlusion_culler.reset();

      const renderer = Renderer.get();

      const current_view = SharedFrameInfoBuffer.get_view_index();
      const total_views = SharedViewBuffer.get_view_data_count();
      const draw_count = MeshTaskQueue.get_total_draw_count();
      const debug_view = renderer.get_debug_draw_type();
      const image_extent = renderer.get_canvas_resolution();

      const shadows_enabled = renderer.is_shadows_enabled();
      const gi_enabled = renderer.is_gi_enabled();
      const gi_has_builtin_specular = renderer.get_gi_strategy_type() === GIStrategyType.PTGI;
      const ao_enabled = renderer.is_ao_enabled();
      const reflections_enabled = gi_enabled && !gi_has_builtin_specular && !!this.reflections;

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

      const occluder_buffer = EntityManager.get_fragment_gpu_buffer(
        VisibilityFragment,
        occluder_name
      );
      const entity_occluders = render_graph.register_buffer(occluder_buffer.buffer.config.name);

      const aabb_gpu_data = BVH.to_gpu_data();
      const tlas_bvh_info = render_graph.register_buffer(
        aabb_gpu_data.bvh_info_buffer.config.name
      );
      const scene_bounds = render_graph.register_buffer(
        aabb_gpu_data.scene_bounds_buffer.config.name
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

      const mesh_asset_ids = EntityManager.get_fragment_gpu_buffer(
        StaticMeshFragment,
        mesh_asset_id_name
      );
      const mesh_asset_ids_buffer = render_graph.register_buffer(mesh_asset_ids.buffer.config.name);

      const index_buffer = render_graph.register_buffer(MeshData.index_buffer.config.name);

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

      let main_hzb_image = render_graph.register_image(this.hzb_image.config.name);
      let main_entity_id_image = render_graph.register_image(this.entity_id_image.config.name);

      main_position_image_config.width = image_extent.width;
      main_position_image_config.height = image_extent.height;
      main_position_image_config.force = this.force_recreate;

      main_position_image2_config.width = image_extent.width;
      main_position_image2_config.height = image_extent.height;
      main_position_image2_config.force = this.force_recreate;

      main_normal_image_config.width = image_extent.width;
      main_normal_image_config.height = image_extent.height;
      main_normal_image_config.force = this.force_recreate;

      main_normal_image2_config.width = image_extent.width;
      main_normal_image2_config.height = image_extent.height;
      main_normal_image2_config.force = this.force_recreate;

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
      main_depth_image_config.width = image_extent.width;
      main_depth_image_config.height = image_extent.height;
      main_depth_image_config.force = this.force_recreate;

      let main_albedo_image = render_graph.create_image(main_albedo_image_config);
      let main_smra_image = render_graph.create_image(main_smra_image_config);
      let main_motion_emissive_image = render_graph.create_image(main_motion_emissive_image_config);
      let main_transparency_accum_image = render_graph.create_image(
        main_transparency_accum_image_config
      );
      let main_depth_image = render_graph.create_image(main_depth_image_config);
      let main_position_image = render_graph.create_image(main_position_image_config);
      let main_normal_image = render_graph.create_image(main_normal_image_config);
      let prev_position_image = render_graph.create_image(main_position_image2_config);
      let prev_normal_image = render_graph.create_image(main_normal_image2_config);

      let skybox_image = null;
      let post_lighting_image_desc = null;

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📋 Register Per-View Visibility Data                                       │
      // └─────────────────────────────────────────────────────────────────────────────┘
      for (let view_index = 0; view_index < total_views; ++view_index) {
        if (!SharedViewBuffer.is_render_active(view_index)) continue;

        const view_data = SharedViewBuffer.get_view_data(view_index);
        const clipmap_count = view_data.clipmap_count || 1;
        const occlusion_enabled = view_data.occlusion_enabled;

        for (let clipmap_index = 0; clipmap_index < clipmap_count; ++clipmap_index) {
          this.frustum_culler.register_view(render_graph, draw_count, view_index, clipmap_index);
          if (occlusion_enabled) {
            this.occlusion_culler.register_view(
              render_graph,
              draw_count,
              view_index,
              clipmap_index
            );
          }
        }

        this.frustum_culler.additional_data.aabb_bounds = aabb_bounds;
        this.frustum_culler.additional_data.object_instances = object_instances;
        this.frustum_culler.additional_data.entity_index_lookup = entity_index_lookup;

        this.occlusion_culler.additional_data.main_hzb_image = main_hzb_image;
        this.occlusion_culler.additional_data.aabb_bounds = aabb_bounds;
        this.occlusion_culler.additional_data.object_instances = object_instances;
        this.occlusion_culler.additional_data.entity_occluders = entity_occluders;
        this.occlusion_culler.additional_data.main_entity_id_image = main_entity_id_image;
        this.occlusion_culler.additional_data.entity_index_lookup = entity_index_lookup;
      }

      // ═══════════════════════════════════════════════════════════════════════════════
      // 🎨 RENDERING PIPELINE BEGINS
      // ═══════════════════════════════════════════════════════════════════════════════

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🧹 PASS: Init Views                                                         │
      // │    Initialize all views to a clean slate                                    │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.frustum_culler.init_views(render_graph, draw_count);
        this.occlusion_culler.init_views(render_graph, draw_count);
      }

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
              main_position_image,
              main_normal_image,
              main_motion_emissive_image,
              main_entity_id_image,
              main_transparency_accum_image,
              main_depth_image,
            ],
            b_skip_pass_pipeline_setup: true,
            b_skip_pass_bind_group_setup: true,
          },
          (graph, frame_data, encoder) => {}
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
            const position = graph.get_physical_image(main_position_image);
            const normal = graph.get_physical_image(main_normal_image);
            const motion_emissive = graph.get_physical_image(main_motion_emissive_image);
            const entity_id = graph.get_physical_image(main_entity_id_image);
            const transparency_accum = graph.get_physical_image(main_transparency_accum_image);
            const depth = graph.get_physical_image(main_depth_image);

            if (albedo) {
              albedo.config.load_op = load_op_load;
            }
            if (smra) {
              smra.config.load_op = load_op_load;
            }
            if (position) {
              position.config.load_op = load_op_load;
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
      // │ 🔍 PASS: Reset Visibility Buffers                                          │
      // │    Clear per-view visibility data before frustum culling                    │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.frustum_culler.init_visibility(render_graph, draw_count);
        this.occlusion_culler.init_visibility(render_graph, draw_count);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 PASS: Frustum Culling (Phase 1 of 2-Pass Occlusion)                    │
      // │    Eliminate objects outside the camera's view frustum                     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.frustum_culler.submit_cull(render_graph, draw_count);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🏔️  PASS: Depth Pre-Pass                                                   │
      // │    Fill depth buffer early for better GPU efficiency and HZB generation   │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (renderer.is_depth_prepass_enabled()) {
        const visible_buf_no_occlusion = this.frustum_culler.get_visibility_buffer(current_view, 0);

        render_graph.add_pass(
          depth_prepass_name,
          RenderPassFlags.Graphics,
          {
            inputs: [
              entity_transforms,
              entity_flags,
              object_instances,
              visible_buf_no_occlusion,
              entity_index_lookup,
            ],
            outputs: [main_entity_id_image, main_depth_image],
            shader_setup: depth_only_shader_setup,
            b_skip_pass_pipeline_setup: true,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            MeshTaskQueue.submit_indexed_indirect_draws(
              pass,
              current_view,
              0 /* clipmap_index */,
              false /* skip_material_bind */,
              true /* opaque_only */,
              true /* depth_only */
            );
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 PASS: Hierarchical Z-Buffer Generation                                  │
      // │    Create multi-level depth pyramid for efficient occlusion culling      │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        let hzb_params_chain = [];
        for (let i = 0; i < this.hzb_image.config.mip_levels; i++) {
          hzb_params_chain.push(
            render_graph.create_buffer({
              name: `hzb_params_${i}`,
              data: [0.0, 0.0, 0.0, 0.0],
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            })
          );
        }

        for (let i = 0; i < this.hzb_image.config.mip_levels; i++) {
          const src_index = i === 0 ? 0 : i - 1;
          const dst_index = i;

          render_graph.add_pass(
            `reduce_hzb_${i}`,
            RenderPassFlags.Compute,
            {
              inputs: [
                i === 0 ? main_depth_image : main_hzb_image,
                main_hzb_image,
                hzb_params_chain[dst_index],
              ],
              outputs: [main_hzb_image],
              input_views: [i === 0 ? 0 : i, i + 1],
              shader_setup: hzb_reduce_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);

              const depth = graph.get_physical_image(main_depth_image);
              const hzb = graph.get_physical_image(main_hzb_image);

              const hzb_params = graph.get_physical_buffer(hzb_params_chain[dst_index]);

              const src_mip_width = Math.max(
                1,
                i === 0 ? depth.config.width : hzb.config.width >> src_index
              );
              const src_mip_height = Math.max(
                1,
                i === 0 ? depth.config.height : hzb.config.height >> src_index
              );

              const dst_mip_width = Math.max(1, hzb.config.width >> dst_index);
              const dst_mip_height = Math.max(1, hzb.config.height >> dst_index);

              hzb_params.write([src_mip_width, src_mip_height, dst_mip_width, dst_mip_height]);

              pass.dispatch((dst_mip_width + 7) / 8, (dst_mip_height + 7) / 8, 1);
            }
          );
        }
      }

      // TODO: Meshlet cull pass

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌫️  PASS: Occlusion Culling (Phase 2 of 2-Pass Occlusion)                 │
      // │    Use HZB to eliminate objects hidden behind other geometry               │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.occlusion_culler.submit_cull(render_graph, draw_count);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖥️  PASS: Compute Rasterization                                            │
      // │    Software rasterization for particles and small geometry                │
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
        g_buffer_shader_setup.depth_write_enabled = !renderer.is_depth_prepass_enabled();
        g_buffer_shader_setup.depth_stencil_compare_op = !renderer.is_depth_prepass_enabled()
          ? "less"
          : "less-equal";

        const material_buckets = MeshTaskQueue.get_material_buckets();
        for (let i = 0; i < material_buckets.length; i++) {
          const material_id = material_buckets[i];
          const material = Material.get(material_id);

          const outputs = [
            material.family === MaterialFamilyType.Transparent
              ? main_transparency_accum_image
              : main_albedo_image,
            main_smra_image,
            main_position_image,
            main_normal_image,
            main_motion_emissive_image,
            main_depth_image,
          ];

          render_graph.add_pass(
            `g_buffer_${material.template.name}_${material_id}`,
            RenderPassFlags.Graphics,
            {
              inputs: [
                entity_transforms,
                entity_flags,
                object_instances,
                this.occlusion_culler.get_visibility_buffer(current_view, 0),
                entity_index_lookup,
              ],
              outputs: outputs,
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
                this.frustum_culler.get_visibility_buffer(current_view, 0),
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
              main_position_image,
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
          position_texture: main_position_image,
          entity_flags: entity_flags,
          aabb_bounds: aabb_bounds,
          lights: lights,
          dense_lights_buffer: dense_lights,
          transforms_buffer: entity_transforms,
          object_instances: object_instances,
          entity_index_lookup: entity_index_lookup,
          frustum_culler: this.frustum_culler,
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
          main_position_image,
          prev_position_image,
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
            main_position_image,
            prev_position_image,
            main_normal_image,
            prev_normal_image,
            main_albedo_image,
            main_smra_image,
            main_motion_emissive_image,
            main_depth_image,
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
          main_position_image,
          prev_normal_image,
          prev_position_image,
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
          main_position_image,
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
          main_position_image,
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

      post_bloom_color_image_config.width = image_extent.width;
      post_bloom_color_image_config.height = image_extent.height;
      post_bloom_color_image_config.force = this.force_recreate;
      const curr_post_bloom = render_graph.create_image(post_bloom_color_image_config);

      const num_iterations = 4;
      let bloom_blur_chain = [];
      if (num_iterations > 0) {
        const image_extent = renderer.get_canvas_resolution();
        const extent_x = ppot(image_extent.width);
        const extent_y = ppot(image_extent.height);

        let bloom_blur_params_chain = [];
        for (let i = 0; i < num_iterations; i++) {
          bloom_blur_chain.push(
            render_graph.create_image({
              name: `bloom_blur_${i}`,
              format: rgba16float_format,
              width: extent_x >> i,
              height: extent_y >> i,
              usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
              force: this.force_recreate,
            })
          );
          bloom_blur_params_chain.push(
            render_graph.create_buffer({
              name: `bloom_blur_params_${i}`,
              data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
              force: this.force_recreate,
            })
          );
        }

        for (let i = 0; i < num_iterations; i++) {
          const src_index = i === 0 ? 0 : i - 1;
          const dst_index = i;

          render_graph.add_pass(
            `bloom_downsample_pass_${i}`,
            RenderPassFlags.Compute,
            {
              inputs: [
                i === 0 ? post_lighting_image_desc : bloom_blur_chain[src_index],
                bloom_blur_chain[dst_index],
                bloom_blur_params_chain[dst_index],
              ],
              outputs: [bloom_blur_chain[dst_index]],
              shader_setup: bloom_downsample_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);

              const bloom_blur_params = graph.get_physical_buffer(
                bloom_blur_params_chain[dst_index]
              );

              const src_mip_width = clamp(extent_x >> src_index, 1, extent_x);
              const src_mip_height = clamp(extent_y >> src_index, 1, extent_y);

              const dst_mip_width = clamp(extent_x >> dst_index, 1, extent_x);
              const dst_mip_height = clamp(extent_y >> dst_index, 1, extent_y);

              bloom_blur_params.write([
                src_mip_width,
                src_mip_height,
                dst_mip_width,
                dst_mip_height,
                0.0,
                i,
              ]);

              pass.dispatch((dst_mip_width + 15) / 16, (dst_mip_height + 15) / 16, 1);
            }
          );
        }

        for (let i = num_iterations - 1; i > 0; --i) {
          const src_index = i;
          const dst_index = i - 1;

          render_graph.add_pass(
            `bloom_upsample_pass_${i}`,
            RenderPassFlags.Compute,
            {
              inputs: [
                bloom_blur_chain[src_index],
                bloom_blur_chain[dst_index],
                bloom_blur_params_chain[dst_index],
              ],
              outputs: [bloom_blur_chain[dst_index]],
              shader_setup: bloom_upsample_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);

              const bloom_blur_params = graph.get_physical_buffer(
                bloom_blur_params_chain[dst_index]
              );

              const src_mip_width = clamp(extent_x >> src_index, 1, extent_x);
              const src_mip_height = clamp(extent_y >> src_index, 1, extent_y);

              const dst_mip_width = clamp(extent_x >> dst_index, 1, extent_x);
              const dst_mip_height = clamp(extent_y >> dst_index, 1, extent_y);

              bloom_blur_params.write([
                src_mip_width,
                src_mip_height,
                dst_mip_width,
                dst_mip_height,
                6.0,
                i,
              ]);

              pass.dispatch((dst_mip_width + 15) / 16, (dst_mip_height + 15) / 16, 1);
            }
          );
        }

        let bloom_resolve_params_desc = render_graph.create_buffer(bloom_resolve_params_config);

        render_graph.add_pass(
          bloom_resolve_pass_name,
          RenderPassFlags.Graphics,
          {
            inputs: [
              post_lighting_image_desc,
              bloom_blur_chain[0],
              main_depth_image,
              bloom_resolve_params_desc,
            ],
            outputs: [curr_post_bloom],
            shader_setup: bloom_resolve_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);

            const bloom_resolve_params = graph.get_physical_buffer(bloom_resolve_params_desc);

            bloom_resolve_params.write(bloom_params);

            MeshTaskQueue.draw_quad(pass);
          }
        );
      }

      // Copy current bloom result into prev_lighting for the next frame
      render_graph.add_pass(
        "copy_history",
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          const curr_final_lighting = graph.get_physical_image(curr_post_bloom);
          const prev_final_lighting = graph.get_physical_image(prev_lighting);
          prev_final_lighting.copy_texture(encoder, curr_final_lighting);

          const curr_position = graph.get_physical_image(main_position_image);
          const prev_position = graph.get_physical_image(prev_position_image);
          if (prev_position) {
            prev_position.copy_texture(encoder, curr_position);
          }

          const curr_normal = graph.get_physical_image(main_normal_image);
          const prev_normal = graph.get_physical_image(prev_normal_image);
          if (prev_normal) {
            prev_normal.copy_texture(encoder, curr_normal);
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
        post_bloom_color_image_config,
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
                main_position_image,
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
              main_entity_id_image,
              0,
              0,
              image_extent.width,
              image_extent.height,
              DebugDrawType.EntityId
            );
            break;
          case DebugDrawType.HZB:
            const hzb_max_level = Math.max(
              0,
              this.hzb_image.config.mip_levels - 1
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
              bloom_blur_chain[0],
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
            const position = graph.get_physical_image(main_position_image);
            const normal = graph.get_physical_image(main_normal_image);
            const motion_emissive = graph.get_physical_image(main_motion_emissive_image);
            const entity_id = graph.get_physical_image(main_entity_id_image);
            const transparency_accum = graph.get_physical_image(main_transparency_accum_image);
            const depth = graph.get_physical_image(main_depth_image);

            if (albedo) {
              albedo.config.load_op = load_op_clear;
            }
            if (smra) {
              smra.config.load_op = load_op_clear;
            }
            if (position) {
              position.config.load_op = load_op_clear;
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

    const image_width_npot = npot(image_extent.width);
    const image_height_npot = npot(image_extent.height);

    hzb_image_config.mip_levels = Math.max(
      Math.log2(image_width_npot),
      Math.log2(image_height_npot)
    );
    hzb_image_config.width = image_width_npot;
    hzb_image_config.height = image_height_npot;
    hzb_image_config.force = this.force_recreate;

    entity_id_image_config.width = image_extent.width;
    entity_id_image_config.height = image_extent.height;
    entity_id_image_config.force = this.force_recreate;

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

    this.hzb_image = Texture.create(hzb_image_config);
    this.entity_id_image = Texture.create(entity_id_image_config);
    this.prev_lighting_image = Texture.create(prev_lighting_image_config);
  }
}


