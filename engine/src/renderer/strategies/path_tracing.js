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
import { PostProcessStack } from "../post_process_stack.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";
import { FrustumCuller } from "../cull/frustum_culler.js";
import { OcclusionCuller } from "../cull/occlusion_culler.js";

// Types and utilities
import { RenderPassFlags, MaterialFamilyType } from "../renderer_types.js";
import { BVH } from "../../acceleration/bvh.js";
import { MeshBLAS } from "../../acceleration/mesh_blas.js";
import { npot } from "../../utility/math.js";
import { profile_scope } from "../../utility/performance.js";
import {
  rgba16float_format,
  rgba32float_format,
  depth32float_format,
  r32float_format,
  r32uint_format,
  load_op_load,
  load_op_clear,
} from "../../utility/config_permutations.js";

// Specialized renderer components
import { PathTracer } from "../raytracing/path_tracer.js";

const resolution_change_event_name = "resolution_change";
const path_tracing_profile_scope_name = "PathTracingStrategy.draw";
const transforms_name = "transforms";
const bounds_name = "bounds";
const occluder_name = "occluder";
const light_fragment_name = "light_fragment";
const mesh_asset_id_name = "mesh_asset_id";

// G-Buffer configurations
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
    GPUTextureUsage.STORAGE_BINDING,
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
    GPUTextureUsage.STORAGE_BINDING,
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
const main_depth_image_config = {
  name: "main_depth",
  format: depth32float_format,
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

const hzb_reduce_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/hzb_reduce.wgsl",
    },
  },
};

const compact_lights_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "system_compute/compact_lights.wgsl",
    },
  },
};

const dense_lights_buffer_config = {
  name: "dense_lights",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const light_count_buffer_config = {
  name: "light_count",
  size: Uint32Array.BYTES_PER_ELEMENT * 2,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

const path_trace_composite_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "raytracing/path_trace_composite.wgsl",
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

const skybox_output_image_config = {
  name: "skybox_output",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const post_lighting_image_config = {
  name: "post_lighting",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const swapchain_name = "swapchain";
const clear_g_buffer_pass_name = "clear_g_buffer";
const skybox_pass_name = "skybox_pass";
const depth_prepass_name = "depth_prepass";
const reset_g_buffer_targets_pass_name = "reset_g_buffer_targets";
const fullscreen_present_pass_name = "fullscreen_present_pass";
const compact_lights_pass_name = "compact_lights";

/**
 * Hybrid Path Tracing Rendering Strategy
 * 
 * Combines rasterization for the first hit (G-buffer) with path tracing for bounces:
 * - Frustum and occlusion culling for efficient scene traversal
 * - G-buffer generation via rasterization (fast first hit)
 * - Path tracing from G-buffer positions for indirect lighting
 * - Multiple bounces with Resampled Importance Sampling (RIS)
 * - Progressive accumulation over multiple frames
 */
export class PathTracingStrategy {
  initialized = false;
  force_recreate = false;
  hzb_image = null;
  entity_id_image = null;
  path_tracer = null;
  frustum_culler = null;
  occlusion_culler = null;

  // Path tracing parameters (optimized for hybrid mode)
  max_bounces = 6;
  trace_rate = 16; // 1=full res, 2=half, 4=quarter, etc.
  samples_per_pixel = 1; // Number of samples per pixel per frame

  setup(render_graph) {
    this.path_tracer = new PathTracer();

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
      path_tracing_profile_scope_name,
      this._draw_internal.bind(this, render_graph)
    );
  }

  refresh(render_graph) {
    this.force_recreate = true;
  }

  /**
   * Set path tracing parameters
   * @param {Object} params - Path tracing parameters
   * @param {number} params.max_bounces - Maximum number of bounces
   * @param {number} params.trace_rate - Trace rate (1=full res, 2=half, 4=quarter)
   * @param {number} params.samples_per_pixel - Samples per pixel per frame
   */
  set_parameters(params) {
    if (params.max_bounces !== undefined) this.max_bounces = params.max_bounces;
    if (params.trace_rate !== undefined) this.trace_rate = params.trace_rate;
    if (params.samples_per_pixel !== undefined) this.samples_per_pixel = params.samples_per_pixel;
  }

  _draw_internal(render_graph) {
    profile_scope(path_tracing_profile_scope_name, () => {
      if (!this.initialized) {
        this.setup(render_graph);
        this.initialized = true;
      }

      // ═══════════════════════════════════════════════════════════════════════════════
      // ⚙️  SETUP & INITIALIZATION PHASE
      // ═══════════════════════════════════════════════════════════════════════════════
      MeshTaskQueue.sort_and_batch();
      ComputeTaskQueue.compile_pre_rg_passes(render_graph);

      this.frustum_culler.reset();
      this.occlusion_culler.reset();

      const renderer = Renderer.get();

      const current_view = SharedFrameInfoBuffer.get_view_index();
      const total_views = SharedViewBuffer.get_view_data_count();
      const draw_count = MeshTaskQueue.get_total_draw_count();
      const image_extent = renderer.get_canvas_resolution();

      if (this.force_recreate) {
        render_graph.mark_pass_cache_bind_groups_dirty(true /* pass_only */);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📋 Register Core Entity & Transform Buffers                                │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const entity_flags_buffer = FragmentGpuBuffer.entity_flags_buffer;
      const entity_flags = render_graph.register_buffer(entity_flags_buffer.buffer.config.name);

      const entity_index_map_buffer = FragmentGpuBuffer.entity_index_map_buffer;
      const entity_index_lookup = render_graph.register_buffer(entity_index_map_buffer.buffer.config.name);

      const transforms_buffer = EntityManager.get_fragment_gpu_buffer(
        TransformFragment,
        transforms_name
      );
      const bounds_buffer = EntityManager.get_fragment_gpu_buffer(TransformFragment, bounds_name);
      const entity_transforms = render_graph.register_buffer(transforms_buffer.buffer.config.name);
      const aabb_bounds = render_graph.register_buffer(bounds_buffer.buffer.config.name);

      const occluder_buffer = EntityManager.get_fragment_gpu_buffer(
        VisibilityFragment,
        occluder_name
      );
      const entity_occluders = render_graph.register_buffer(occluder_buffer.buffer.config.name);

      const aabb_gpu_data = BVH.to_gpu_data();
      const tlas_bvh4_nodes = render_graph.register_buffer(
        aabb_gpu_data.bvh4_nodes_buffer.config.name
      );

      const blas_gpu_data = MeshBLAS.to_gpu_data();
      const blas_atlas = render_graph.register_buffer(blas_gpu_data.atlas_buffer.config.name);

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

      dense_lights_buffer_config.size = light_fragment_buffer.buffer.config.size;
      const dense_lights = render_graph.create_buffer(dense_lights_buffer_config);

      const light_count = render_graph.create_buffer(light_count_buffer_config);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖼️  Create G-Buffer & Main Render Targets                                  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      let main_hzb_image = render_graph.register_image(this.hzb_image.config.name);
      let main_entity_id_image = render_graph.register_image(this.entity_id_image.config.name);

      main_albedo_image_config.width = image_extent.width;
      main_albedo_image_config.height = image_extent.height;
      main_albedo_image_config.force = this.force_recreate;
      main_smra_image_config.width = image_extent.width;
      main_smra_image_config.height = image_extent.height;
      main_smra_image_config.force = this.force_recreate;
      main_normal_image_config.width = image_extent.width;
      main_normal_image_config.height = image_extent.height;
      main_normal_image_config.force = this.force_recreate;
      main_position_image_config.width = image_extent.width;
      main_position_image_config.height = image_extent.height;
      main_position_image_config.force = this.force_recreate;
      main_motion_emissive_image_config.width = image_extent.width;
      main_motion_emissive_image_config.height = image_extent.height;
      main_motion_emissive_image_config.force = this.force_recreate;
      main_depth_image_config.width = image_extent.width;
      main_depth_image_config.height = image_extent.height;
      main_depth_image_config.force = this.force_recreate;

      let main_albedo_image = render_graph.create_image(main_albedo_image_config);
      let main_smra_image = render_graph.create_image(main_smra_image_config);
      let main_normal_image = render_graph.create_image(main_normal_image_config);
      let main_position_image = render_graph.create_image(main_position_image_config);
      let main_motion_emissive_image = render_graph.create_image(main_motion_emissive_image_config);
      let main_depth_image = render_graph.create_image(main_depth_image_config);

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
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.frustum_culler.init_views(render_graph, draw_count);
        this.occlusion_culler.init_views(render_graph, draw_count);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🧹 PASS: Clear G-Buffer Targets                                            │
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
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        render_graph.add_pass(
          compact_lights_pass_name,
          RenderPassFlags.Compute,
          {
            shader_setup: compact_lights_shader_setup,
            inputs: [lights, light_count, dense_lights],
            outputs: [light_count, dense_lights],
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const count_buf = graph.get_physical_buffer(light_count);
            count_buf.write(new Uint32Array([0, 0]));
            const max_light_count = EntityManager.get_max_rows();
            pass.dispatch((max_light_count + 128 - 1) / 128, 1, 1);
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌌 PASS: Skybox Rendering                                                  │
      // │    Render the environment skybox to provide background                     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      let skybox_image = null;
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
          skybox_pass_name,
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
            const motion = graph.get_physical_image(main_motion_emissive_image);
            const entity_id = graph.get_physical_image(main_entity_id_image);
            const depth = graph.get_physical_image(main_depth_image);

            if (albedo) albedo.config.load_op = load_op_load;
            if (smra) smra.config.load_op = load_op_load;
            if (position) position.config.load_op = load_op_load;
            if (normal) normal.config.load_op = load_op_load;
            if (motion) motion.config.load_op = load_op_load;
            if (entity_id) entity_id.config.load_op = load_op_load;
            if (depth) depth.config.load_op = load_op_load;
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🔍 PASS: Reset Visibility Buffers                                          │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.frustum_culler.init_visibility(render_graph, draw_count);
        this.occlusion_culler.init_visibility(render_graph, draw_count);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 PASS: Frustum Culling                                                   │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.frustum_culler.submit_cull(render_graph, draw_count);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🏔️  PASS: Depth Pre-Pass                                                   │
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
              input_views: [src_index, dst_index],
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

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌫️  PASS: Occlusion Culling                                                │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        this.occlusion_culler.submit_cull(render_graph, draw_count);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎨 PASS: G-Buffer Base Rendering                                           │
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

          // Skip transparent materials in hybrid path tracing
          if (material.family === MaterialFamilyType.Transparent) {
            continue;
          }

          const outputs = [
            main_albedo_image,
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
      // │ 🔆 PASS: Hybrid Path Tracing                                               │
      // │    Use G-buffer for first hit, then path trace from there                 │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        this.path_tracer.add_passes(
          render_graph,
          image_extent.width,
          image_extent.height,
          this.max_bounces,
          this.trace_rate,
          this.samples_per_pixel,
          true, // use_gbuffer - always true for hybrid mode
          aabb_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          mesh_asset_ids_buffer,
          index_buffer,
          dense_lights,
          light_count,
          main_position_image, // G-buffer position
          main_normal_image,   // G-buffer normal
          main_albedo_image,   // G-buffer albedo
          main_smra_image,     // G-buffer SMRA
          main_motion_emissive_image,   // G-buffer motion and emissive
          this.force_recreate
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎨 PASS: Composite Path Trace over Skybox                                  │
      // │    Blend path traced output with skybox background                         │
      // └─────────────────────────────────────────────────────────────────────────────┘
      let composited_image = null;
      {
        const composited_image_config = {
          name: "path_trace_composited",
          format: rgba16float_format,
          width: image_extent.width,
          height: image_extent.height,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          force: this.force_recreate,
        };
        composited_image = render_graph.create_image(composited_image_config);

        render_graph.add_pass(
          "path_trace_composite",
          RenderPassFlags.Compute,
          {
            inputs: [skybox_image, this.path_tracer.output_texture, main_normal_image, composited_image],
            outputs: [composited_image],
            shader_setup: path_trace_composite_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(
              Math.ceil(image_extent.width / 8),
              Math.ceil(image_extent.height / 8),
              1
            );
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎭 PASS: Post-Processing Stack                                             │
      // └─────────────────────────────────────────────────────────────────────────────┘
      post_lighting_image_config.width = image_extent.width;
      post_lighting_image_config.height = image_extent.height;
      post_lighting_image_config.force = this.force_recreate;

      const post_processed_image = PostProcessStack.compile_passes(
        0,
        render_graph,
        post_lighting_image_config,
        composited_image, // Use composited image instead of raw path tracer output
        main_depth_image,
        main_normal_image
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖼️  PASS: Final Presentation                                               │
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
            const motion = graph.get_physical_image(main_motion_emissive_image);
            const entity_id = graph.get_physical_image(main_entity_id_image);
            const depth = graph.get_physical_image(main_depth_image);

            if (albedo) albedo.config.load_op = load_op_clear;
            if (smra) smra.config.load_op = load_op_clear;
            if (position) position.config.load_op = load_op_clear;
            if (normal) normal.config.load_op = load_op_clear;
            if (motion) motion.config.load_op = load_op_clear;
            if (entity_id) entity_id.config.load_op = load_op_clear;
            if (depth) depth.config.load_op = load_op_clear;
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🚩 PASS: Clear Entity Dirty Flags                                          │
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

    this.hzb_image = Texture.create(hzb_image_config);
    this.entity_id_image = Texture.create(entity_id_image_config);
  }
}
