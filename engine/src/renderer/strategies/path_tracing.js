// Core imports
import { global_dispatcher } from "../../core/dispatcher.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import { SharedFrameInfoBuffer } from "../../core/shared_data.js";

// ECS fragments
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";

// Renderer components
import { Renderer } from "../renderer.js";
import { Texture } from "../texture.js";
import { MeshData } from "../mesh_data.js";
import { PostProcessStack } from "../post_process_stack.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";
import { ResourceCache } from "../resource_cache.js";
import { TextureArrayPools } from "../texture_pool.js";
import { EnvironmentPipeline } from "../pipelines/environment_pipeline.js";
import { CullingPipeline } from "../pipelines/culling_pipeline.js";
import { GBufferTargetsPipeline } from "../pipelines/gbuffer_targets_pipeline.js";
import { VisibilityBufferPipeline } from "../pipelines/visibility_buffer_pipeline.js";

// Types and utilities
import { RenderPassFlags, CacheTypes } from "../renderer_types.js";
import { BVH } from "../../acceleration/bvh.js";
import { MeshBLAS } from "../../acceleration/mesh_blas.js";
import { profile_scope } from "../../utility/performance.js";
import { Name } from "../../utility/names.js";
import {
  rgba16float_format,
  depth32float_format,
  load_op_load,
  load_op_clear,
} from "../../utility/config_permutations.js";

// Specialized renderer components
import { PathTracer } from "../raytracing/path_tracer.js";

const resolution_change_event_name = "resolution_change";
const path_tracing_profile_scope_name = "PathTracingStrategy.draw";
const transforms_name = "transforms";
const bounds_name = "bounds";
const light_fragment_name = "light_fragment";
const mesh_asset_id_name = "mesh_asset_id";

const main_depth_image2_config = {
  name: "main_depth_1",
  format: depth32float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_DST,
  force: false,
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
  force_reinit = false;
  culling_pipeline = null;
  visibility_buffer_pipeline = null;
  gbuffer_targets_pipeline = null;
  environment_pipeline = null;
  prev_depth_image = null;
  path_tracer = null;

  // Path tracing parameters (optimized for hybrid mode)
  max_bounces = 6;
  trace_rate = 2; // 1=full res, 2=half, 4=quarter, etc.
  samples_per_pixel = 1; // Number of samples per pixel per frame

  setup(render_graph) {
    this.path_tracer = new PathTracer();
    this.culling_pipeline = new CullingPipeline();
    this.environment_pipeline = new EnvironmentPipeline();
    this.gbuffer_targets_pipeline = new GBufferTargetsPipeline();
    this.visibility_buffer_pipeline = new VisibilityBufferPipeline();

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

  refresh(render_graph, reinit = false) {
    this.force_reinit = reinit;
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
      const visibility_shader_buckets = MeshTaskQueue.get_visibility_shader_buckets();
      const image_extent = renderer.get_canvas_resolution();
      const depth_prepass_enabled = renderer.is_depth_prepass_enabled();

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

      const aabb_gpu_data = BVH.to_gpu_data();
      const tlas_bvh_info = render_graph.register_buffer(
        aabb_gpu_data.bvh_info_buffer.config.name
      );

      const blas_gpu_data = MeshBLAS.to_gpu_data();
      const blas_bvh2_nodes = render_graph.register_buffer(
        blas_gpu_data.bvh2_nodes_buffer.config.name
      );
      const blas_directory = render_graph.register_buffer(
        blas_gpu_data.directory_buffer.config.name
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

      const mesh_gpu_data = MeshData.to_gpu_data();
      const index_buffer = render_graph.register_buffer(MeshData.index_buffer.config.name);
      const meshlet_buffer = render_graph.register_buffer(mesh_gpu_data.meshlet_buffer.config.name);
      const meshlet_vertex_buffer = render_graph.register_buffer(
        mesh_gpu_data.meshlet_vertex_buffer.config.name
      );
      const meshlet_triangle_buffer = render_graph.register_buffer(
        mesh_gpu_data.meshlet_triangle_buffer.config.name
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 💡 Setup Lighting System                                                   │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const light_fragment_buffer = EntityManager.get_fragment_gpu_buffer(
        LightFragment,
        light_fragment_name
      );
      const lights = render_graph.register_buffer(light_fragment_buffer.buffer.config.name);

      // The Light payload follows immediately after the header.
      dense_lights_buffer_config.size = light_fragment_buffer.buffer.config.size + 16;
      const dense_lights = render_graph.create_buffer(dense_lights_buffer_config);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖼️  Create G-Buffer & Main Render Targets                                  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const { main_hzb_image } = this.culling_pipeline.register_targets(render_graph);
      const {
        visibility_entity_image,
        visibility_surface_image,
        visibility_bucket_image,
      } = this.visibility_buffer_pipeline.register_targets(render_graph);
      let {
        main_albedo_image,
        main_smra_image,
        main_normal_image,
        main_motion_emissive_image,
        main_depth_image,
        prev_depth_image,
      } = this.gbuffer_targets_pipeline.create_targets(render_graph, {
        image_extent,
        force_recreate: this.force_recreate,
        include_prev_depth: true,
      });

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
      // └─────────────────────────────────────────────────────────────────────────────┘
      this.culling_pipeline.add_init_view_passes(render_graph, draw_count);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🧹 PASS: Clear G-Buffer Targets                                            │
      // └─────────────────────────────────────────────────────────────────────────────┘
      this.gbuffer_targets_pipeline.add_clear_pass(render_graph, {
        pass_name: clear_g_buffer_pass_name,
        targets: {
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
        },
        visibility_targets: [
          visibility_entity_image,
          visibility_surface_image,
          visibility_bucket_image,
        ],
      });

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 💡 PASS: Compact Active Lights                                             │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
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
      skybox_image = this.environment_pipeline.add_skybox_pass(render_graph, {
        pass_name: skybox_pass_name,
        image_extent,
        force_recreate: this.force_recreate,
      });

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🔄 PASS: G-Buffer Load State Configuration                                 │
      // └─────────────────────────────────────────────────────────────────────────────┘
      this.gbuffer_targets_pipeline.add_set_load_op_pass(render_graph, {
        pass_name: reset_g_buffer_targets_pass_name,
        targets: {
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
        },
        visibility_entity_image,
        visibility_surface_image,
        visibility_bucket_image,
        load_op: load_op_load,
      });

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 PASS: Frustum Culling                                                   │
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
      // └─────────────────────────────────────────────────────────────────────────────┘
      const frustum_bucket_draw_lists =
        this.visibility_buffer_pipeline.build_bucket_meshlet_draw_lists(render_graph, {
          current_view,
          meshlet_draw_count,
          object_instances,
          source_meshlet_list: frustum_meshlet_list,
          source_meshlet_draw_args: frustum_meshlet_draw_args,
          buckets: visibility_shader_buckets,
          stage_name: "frustum",
          force_recreate: this.force_recreate,
        });

      for (const bucket of visibility_shader_buckets) {
        const bucket_draw_resources = frustum_bucket_draw_lists.get(bucket.key);
        this.visibility_buffer_pipeline.add_depth_prepass(render_graph, {
          enabled: depth_prepass_enabled,
          meshlet_draw_count,
          current_view,
          depth_image: main_depth_image,
          frustum_meshlet_draw_args: bucket_draw_resources?.draw_args ?? frustum_meshlet_draw_args,
          bucket,
          inputs: [
            entity_transforms,
            object_instances,
            bucket_draw_resources?.meshlet_list ?? frustum_meshlet_list,
            meshlet_buffer,
            meshlet_vertex_buffer,
            meshlet_triangle_buffer,
            entity_index_lookup,
          ],
        });
      }
      
      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌫️  PASS: Occlusion Culling                                                │
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

      const occlusion_bucket_draw_lists =
        this.visibility_buffer_pipeline.build_bucket_meshlet_draw_lists(render_graph, {
          current_view,
          meshlet_draw_count,
          object_instances,
          source_meshlet_list: occlusion_meshlet_list,
          source_meshlet_draw_args: occlusion_meshlet_draw_args,
          buckets: visibility_shader_buckets,
          stage_name: "occlusion",
          force_recreate: this.force_recreate,
        });

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎨 PASS: G-Buffer Base Rendering                                           │
      // └─────────────────────────────────────────────────────────────────────────────┘
      for (const bucket of visibility_shader_buckets) {
        const bucket_draw_resources = occlusion_bucket_draw_lists.get(bucket.key);
        this.visibility_buffer_pipeline.add_visibility_raster_pass(render_graph, {
          meshlet_draw_count,
          depth_prepass_enabled,
          current_view,
          depth_image: main_depth_image,
          occlusion_meshlet_draw_args: bucket_draw_resources.draw_args,
          bucket,
          inputs: [
            entity_transforms,
            object_instances,
            bucket_draw_resources.meshlet_list,
            meshlet_buffer,
            meshlet_vertex_buffer,
            meshlet_triangle_buffer,
            entity_index_lookup,
          ],
        });

        this.visibility_buffer_pipeline.add_gbuffer_resolve_pass(render_graph, {
          meshlet_draw_count,
          current_view,
          depth_image: main_depth_image,
          bucket,
          inputs: [
            entity_transforms,
            meshlet_buffer,
            meshlet_vertex_buffer,
            meshlet_triangle_buffer,
          ],
          outputs: [
            main_albedo_image,
            main_smra_image,
            main_normal_image,
            main_motion_emissive_image,
          ],
        });
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
          tlas_bvh_info,
          blas_bvh2_nodes,
          blas_directory,
          entity_transforms,
          index_buffer,
          dense_lights,
          visibility_entity_image,
          visibility_surface_image,
          meshlet_buffer,
          meshlet_vertex_buffer,
          meshlet_triangle_buffer,
          main_depth_image,    // G-buffer depth
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

      render_graph.add_pass(
        "copy_depth_history",
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          const curr_depth = graph.get_physical_image(main_depth_image);
          const prev_depth = graph.get_physical_image(prev_depth_image);
          if (prev_depth) {
            prev_depth.copy_texture(encoder, curr_depth);
          }
        }
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
      this.gbuffer_targets_pipeline.add_set_load_op_pass(render_graph, {
        pass_name: reset_g_buffer_targets_pass_name,
        targets: {
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
        },
        visibility_entity_image,
        visibility_surface_image,
        visibility_bucket_image,
        load_op: load_op_clear,
      });

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

    main_depth_image2_config.width = image_extent.width;
    main_depth_image2_config.height = image_extent.height;
    main_depth_image2_config.force = this.force_recreate;

    this.prev_depth_image = Texture.create(main_depth_image2_config);

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
