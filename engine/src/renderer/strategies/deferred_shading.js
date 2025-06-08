// Core imports
import { global_dispatcher } from "../../core/dispatcher.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import {
  SharedEnvironmentMapData,
  SharedViewBuffer,
  SharedFrameInfoBuffer,
} from "../../core/shared_data.js";

// ECS fragments
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";

// Renderer components
import { Renderer } from "../renderer.js";
import { Texture } from "../texture.js";
import { Material } from "../material.js";
import { DebugOverlay } from "../debug_overlay.js";
import { LineRenderer } from "../line_renderer.js";
import { PostProcessStack } from "../post_process_stack.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";
import { ComputeRasterTaskQueue } from "../compute_raster_task_queue.js";

// Types and utilities
import { RenderPassFlags, MaterialFamilyType, DebugView } from "../renderer_types.js";
import { AABB } from "../../acceleration/aabb.js";
import { npot, ppot, clamp } from "../../utility/math.js";
import { profile_scope } from "../../utility/performance.js";
import {
  rgba8unorm_format,
  rgba16float_format,
  r8unorm_format,
  depth32float_format,
  r32float_format,
  rg32uint_format,
  one_one_blend_config,
  src_alpha_one_minus_src_alpha_blend_config,
  load_op_load,
  load_op_clear,
} from "../../utility/config_permutations.js";

// Specialized renderer components
import { GIProbeVolume } from "../global_illumination/ddgi.js";
import { AdaptiveSparseVirtualShadowMaps } from "../shadows/as_vsm.js";

const resolution_change_event_name = "resolution_change";
const deferred_shading_profile_scope_name = "DeferredShadingStrategy.draw";
const transforms_name = "transforms";
const aabb_node_index_name = "aabb_node_index";
const light_fragment_name = "light_fragment";

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
const main_emissive_image_config = {
  name: "main_emissive",
  format: rgba8unorm_format,
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
  format: rgba8unorm_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_normal_image_config = {
  name: "main_normal",
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
  name: "main_position",
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
const main_transparency_reveal_image_config = {
  name: "main_transparency_reveal",
  format: r8unorm_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
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

const clear_visibility_data_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/clear_visibility_data.wgsl",
    },
  },
};

const clear_indirect_instance_counts_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/clear_indirect_instance_counts.wgsl",
    },
  },
};

const compute_cull_frustum_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/cull_frustum.wgsl",
    },
  },
};
const compute_cull_occlusion_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/cull_occlusion.wgsl",
    },
  },
};
const draw_cull_data_config = {
  name: `draw_cull_data`,
  data: [0, 0, 0],
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};

const depth_only_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "gbuffer_base.wgsl",
      defines: ["DEPTH_ONLY"],
    },
    fragment: {
      path: "gbuffer_base.wgsl",
      defines: ["DEPTH_ONLY"],
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

const dense_lights_buffer_config = {
  name: "dense_lights",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const dense_shadow_casting_lights_buffer_config = {
  name: "dense_shadow_casting_lights",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const light_count_buffer_config = {
  name: "light_count",
  size: Uint32Array.BYTES_PER_ELEMENT * 2, // [total_lights, total_shadow_casting_lights]
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
    },
    fragment: {
      path: "deferred_lighting.wgsl",
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

const line_transform_processing_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "system_compute/line_transform_processing.wgsl",
    },
  },
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
  data: [0.0, 0.0, 0.0, 0.0],
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};
const post_bloom_color_image_config = {
  name: "post_bloom_color",
  format: rgba8unorm_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};
const bloom_params = [
  1.5 /* final exposure */, 0.3 /* bloom intensity */, 0.001 /* bloom threshold */,
  0.0 /* bloom knee */,
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
  format: rg32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  force: false,
};
const gi_irradiance_config = {
  name: "gi_irradiance_volume",
  width: 0,
  height: 0,
  depth: 0,
  mip_levels: 1,
  dimension: "3d",
  format: "rgba16float",
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const gi_depth_config = {
  name: "gi_depth_volume",
  width: 0,
  height: 0,
  depth: 0,
  mip_levels: 1,
  dimension: "3d",
  format: "r32float",
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const probe_resolution = 16;
const probe_cubemap_image_config = {
  name: "ddgi_probe_cubemap",
  width: probe_resolution,
  height: probe_resolution,
  dimension: "2d-array",
  array_layer_count: 6,
  format: rgba16float_format,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const swapchain_name = "swapchain";
const clear_g_buffer_pass_name = "clear_g_buffer";
const skybox_pass_name = "skybox_pass";
const depth_prepass_name = "depth_prepass";
const transparency_composite_pass_name = "transparency_composite";
const reset_g_buffer_targets_pass_name = "reset_g_buffer_targets";
const compute_cull_pass_name1 = "compute_cull_frustum";
const compute_cull_pass_name2 = "compute_cull_occlusion";
const lighting_pass_name = "lighting_pass";
const bloom_resolve_pass_name = "bloom_resolve_pass";
const fullscreen_present_pass_name = "fullscreen_present_pass";

// Debug shader setups for AS-VSM debug views
export class DeferredShadingStrategy {
  initialized = false;
  hzb_image = null;
  entity_id_image = null;
  force_recreate = false;
  gi_probe_volume = null;
  as_vsm = null;
  debug_view = DebugView.None;
  debug_overlay = null;

  setup(render_graph) {
    this.gi_probe_volume = new GIProbeVolume();
    this.debug_overlay = new DebugOverlay();

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

  _draw_internal(render_graph) {
    profile_scope(deferred_shading_profile_scope_name, () => {
      if (!this.initialized) {
        this.setup(render_graph);
        this.initialized = true;
      }

      // ═══════════════════════════════════════════════════════════════════════════════
      // ⚙️  SETUP & INITIALIZATION PHASE
      // ═══════════════════════════════════════════════════════════════════════════════
      MeshTaskQueue.sort_and_batch();
      ComputeTaskQueue.compile_rg_passes(render_graph);

      const renderer = Renderer.get();

      const current_view = SharedFrameInfoBuffer.get_view_index();
      const shadows_enabled = renderer.is_shadows_enabled();
      const gi_enabled = renderer.is_gi_enabled();
      const total_views = SharedViewBuffer.get_view_data_count();
      const draw_count = MeshTaskQueue.get_total_draw_count();

      if (this.force_recreate) {
        render_graph.mark_pass_cache_bind_groups_dirty(true /* pass_only */);
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📋 Register Core Entity & Transform Buffers                                │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const entity_flags_buffer = FragmentGpuBuffer.entity_flags_buffer;
      const entity_flags = render_graph.register_buffer(entity_flags_buffer.buffer.config.name);

      const transforms_buffer = EntityManager.get_fragment_gpu_buffer(
        TransformFragment,
        transforms_name
      );
      const entity_transforms = render_graph.register_buffer(transforms_buffer.buffer.config.name);
      const aabb_node_index_buffer = EntityManager.get_fragment_gpu_buffer(
        TransformFragment,
        aabb_node_index_name
      );
      const entity_aabb_node_indices = render_graph.register_buffer(
        aabb_node_index_buffer.buffer.config.name
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 Register Mesh & Instance Buffers                                        │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const object_instances = render_graph.register_buffer(
        MeshTaskQueue.get_object_instance_buffer().config.name
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📋 Register Per-View Visibility Data                                       │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const per_view_visible_no_occlusion_buffers = [];
      const per_view_visible_buffers = [];
      const per_view_indirect_draw_buffers = [];
      const per_view_draw_cull_data = [];
      for (let view_index = 0; view_index < total_views; ++view_index) {
        if (!SharedViewBuffer.is_render_active(view_index)) continue;

        const visible_buf_no_occlusion = render_graph.register_buffer(
          MeshTaskQueue.get_visible_instance_buffer_no_occlusion(view_index).config.name
        );
        const visible_buf = render_graph.register_buffer(
          MeshTaskQueue.get_visible_instance_buffer(view_index).config.name
        );
        const indirect_draw_buf = render_graph.register_buffer(
          MeshTaskQueue.get_indirect_draw_buffer(view_index).config.name
        );

        draw_cull_data_config.name = `draw_cull_data_view_${view_index}`;
        const draw_cull_data = render_graph.create_buffer(draw_cull_data_config);

        per_view_visible_no_occlusion_buffers.push(visible_buf_no_occlusion);
        per_view_visible_buffers.push(visible_buf);
        per_view_indirect_draw_buffers.push(indirect_draw_buf);
        per_view_draw_cull_data.push(draw_cull_data);
      }

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

      dense_shadow_casting_lights_buffer_config.size =
        light_fragment_buffer.max_rows * Uint32Array.BYTES_PER_ELEMENT;
      const dense_shadow_casting_lights = render_graph.create_buffer(
        dense_shadow_casting_lights_buffer_config
      );

      const light_count = render_graph.create_buffer(light_count_buffer_config);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📦 Setup Acceleration Structures & Bounds                                  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const aabb_gpu_data = AABB.to_gpu_data();
      const aabb_bounds = render_graph.register_buffer(
        aabb_gpu_data.node_bounds_buffer.config.name
      );

      let skybox_image = null;
      let post_lighting_image_desc = null;
      let post_bloom_color_desc = null;

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🖼️  Create G-Buffer & Main Render Targets                                  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const image_extent = renderer.get_canvas_resolution();

      let main_hzb_image = render_graph.register_image(this.hzb_image.config.name);
      let main_entity_id_image = render_graph.register_image(this.entity_id_image.config.name);

      main_albedo_image_config.width = image_extent.width;
      main_albedo_image_config.height = image_extent.height;
      main_albedo_image_config.force = this.force_recreate;
      main_emissive_image_config.width = image_extent.width;
      main_emissive_image_config.height = image_extent.height;
      main_emissive_image_config.force = this.force_recreate;
      main_smra_image_config.width = image_extent.width;
      main_smra_image_config.height = image_extent.height;
      main_smra_image_config.force = this.force_recreate;
      main_normal_image_config.width = image_extent.width;
      main_normal_image_config.height = image_extent.height;
      main_normal_image_config.force = this.force_recreate;
      main_position_image_config.width = image_extent.width;
      main_position_image_config.height = image_extent.height;
      main_position_image_config.force = this.force_recreate;
      main_transparency_accum_image_config.width = image_extent.width;
      main_transparency_accum_image_config.height = image_extent.height;
      main_transparency_accum_image_config.force = this.force_recreate;
      main_transparency_reveal_image_config.width = image_extent.width;
      main_transparency_reveal_image_config.height = image_extent.height;
      main_transparency_reveal_image_config.force = this.force_recreate;
      main_depth_image_config.width = image_extent.width;
      main_depth_image_config.height = image_extent.height;
      main_depth_image_config.force = this.force_recreate;

      let main_albedo_image = render_graph.create_image(main_albedo_image_config);
      let main_emissive_image = render_graph.create_image(main_emissive_image_config);
      let main_smra_image = render_graph.create_image(main_smra_image_config);
      let main_normal_image = render_graph.create_image(main_normal_image_config);
      let main_position_image = render_graph.create_image(main_position_image_config);
      let main_transparency_accum_image = render_graph.create_image(
        main_transparency_accum_image_config
      );
      let main_transparency_reveal_image = render_graph.create_image(
        main_transparency_reveal_image_config
      );
      let main_depth_image = render_graph.create_image(main_depth_image_config);

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌟 Setup Global Illumination Resources                                     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const gi_irradiance_image = render_graph.register_image(
        this.gi_irradiance_volume.config.name
      );
      const gi_depth_image = render_graph.register_image(this.gi_depth_volume.config.name);
      const gi_params_buffer = render_graph.create_buffer({
        name: "gi_params",
        size: 16 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // ═══════════════════════════════════════════════════════════════════════════════
      // 🎨 RENDERING PIPELINE BEGINS
      // ═══════════════════════════════════════════════════════════════════════════════

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
              main_emissive_image,
              main_smra_image,
              main_position_image,
              main_normal_image,
              main_entity_id_image,
              main_transparency_accum_image,
              main_transparency_reveal_image,
              main_depth_image,
            ],
            b_skip_pass_pipeline_setup: true,
            b_skip_pass_bind_group_setup: true,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);

            // Ensure all G-Buffer targets are set to clear on load
            frame_data.g_buffer_data = {
              albedo: graph.get_physical_image(main_albedo_image),
              emissive: graph.get_physical_image(main_emissive_image),
              smra: graph.get_physical_image(main_smra_image),
              position: graph.get_physical_image(main_position_image),
              normal: graph.get_physical_image(main_normal_image),
              entity_id: graph.get_physical_image(main_entity_id_image),
              transparency_accum: graph.get_physical_image(main_transparency_accum_image),
              transparency_reveal: graph.get_physical_image(main_transparency_reveal_image),
              depth: graph.get_physical_image(main_depth_image),
            };
          }
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
            inputs: [lights, light_count, dense_shadow_casting_lights, dense_lights],
            outputs: [light_count, dense_shadow_casting_lights, dense_lights],
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            // Reset light count to zero
            const count_buf = graph.get_physical_buffer(light_count);
            count_buf.write(new Uint32Array([0, 0]));
            // Dispatch compute to compact lights
            const max_light_count = EntityManager.get_max_rows();
            pass.dispatch((max_light_count + 128 - 1) / 128, 1, 1);
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌌 PASS: Skybox Rendering                                                  │
      // │    Render the environment skybox to provide distant lighting context      │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        skybox_output_image_config.width = image_extent.width;
        skybox_output_image_config.height = image_extent.height;
        skybox_output_image_config.force = this.force_recreate;
        skybox_image = render_graph.create_image(skybox_output_image_config);

        const skybox = SharedEnvironmentMapData.get_skybox();
        const skybox_data = SharedEnvironmentMapData.get_skybox_data();

        if (skybox) {
          const skybox_data_buffer = render_graph.register_buffer(skybox_data.config.name);
          const skybox_texture = render_graph.register_image(skybox.config.name);

          render_graph.add_pass(
            skybox_pass_name,
            RenderPassFlags.Graphics,
            {
              inputs: [skybox_texture, skybox_data_buffer],
              outputs: [skybox_image],
              shader_setup: skybox_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              MeshTaskQueue.draw_cube(pass);
            }
          );
        }
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
            if (frame_data.g_buffer_data) {
              if (frame_data.g_buffer_data.albedo) {
                frame_data.g_buffer_data.albedo.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.emissive) {
                frame_data.g_buffer_data.emissive.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.smra) {
                frame_data.g_buffer_data.smra.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.position) {
                frame_data.g_buffer_data.position.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.normal) {
                frame_data.g_buffer_data.normal.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.entity_id) {
                frame_data.g_buffer_data.entity_id.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.transparency_accum) {
                frame_data.g_buffer_data.transparency_accum.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.transparency_reveal) {
                frame_data.g_buffer_data.transparency_reveal.config.load_op = load_op_load;
              }
              if (frame_data.g_buffer_data.depth) {
                frame_data.g_buffer_data.depth.config.load_op = load_op_load;
              }
            }
          }
        );
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🔍 PASS: Reset Visibility Buffers                                          │
      // │    Clear per-view visibility data before frustum culling                  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        for (let view_index = 0; view_index < total_views; ++view_index) {
          if (!SharedViewBuffer.is_render_active(view_index)) continue;

          const visible_buf_no_occlusion = per_view_visible_no_occlusion_buffers[view_index];
          const visible_buf = per_view_visible_buffers[view_index];

          render_graph.add_pass(
            `clear_visibility_data_view_${view_index}`,
            RenderPassFlags.Compute,
            {
              shader_setup: clear_visibility_data_shader_setup,
              inputs: [visible_buf, visible_buf_no_occlusion],
              outputs: [visible_buf, visible_buf_no_occlusion],
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              pass.dispatch((draw_count + 255) / 256, 1, 1);
            }
          );
        }
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🎯 PASS: Frustum Culling (Phase 1 of 2-Pass Occlusion)                    │
      // │    Eliminate objects outside the camera's view frustum                     │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        for (let view_index = 0; view_index < total_views; ++view_index) {
          if (!SharedViewBuffer.is_render_active(view_index)) continue;

          const visible_buf_no_occlusion = per_view_visible_no_occlusion_buffers[view_index];
          const indirect_buf = per_view_indirect_draw_buffers[view_index];
          const draw_cull_data = per_view_draw_cull_data[view_index];

          render_graph.add_pass(
            `${compute_cull_pass_name1}_view_${view_index}`,
            RenderPassFlags.Compute,
            {
              shader_setup: compute_cull_frustum_shader_setup,
              inputs: [
                aabb_bounds,
                object_instances,
                visible_buf_no_occlusion,
                indirect_buf,
                entity_aabb_node_indices,
                draw_cull_data,
              ],
              outputs: [indirect_buf, visible_buf_no_occlusion],
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);

              const hzb = graph.get_physical_image(main_hzb_image);
              const draw_cull = graph.get_physical_buffer(draw_cull_data);

              draw_cull.write([draw_count, hzb.config.width, hzb.config.height]);

              pass.dispatch((draw_count + 255) / 256, 1, 1);
            }
          );
        }
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🏔️  PASS: Depth Pre-Pass                                                   │
      // │    Fill depth buffer early for better GPU efficiency and HZB generation   │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (renderer.is_depth_prepass_enabled()) {
        render_graph.add_pass(
          depth_prepass_name,
          RenderPassFlags.Graphics,
          {
            inputs: [
              entity_transforms,
              entity_flags,
              object_instances,
              per_view_visible_no_occlusion_buffers[current_view],
            ],
            outputs: [main_depth_image],
            shader_setup: depth_only_shader_setup,
            b_skip_pass_pipeline_setup: true,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            // bind_depth = true to use per-material depth-only pipelines
            MeshTaskQueue.submit_indexed_indirect_draws(
              pass,
              current_view,
              false,/* skip_material_bind */
              true, /* opaque_only */
              true, /* depth_only */
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

      // TODO: Meshlet cull pass

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🔄 PASS: Reset Instance Counts                                             │
      // │    Reset instance counts for each view                                   │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        for (let view_index = 0; view_index < total_views; ++view_index) {
          if (!SharedViewBuffer.is_render_active(view_index)) continue;

          render_graph.add_pass(
            `reset_instance_counts_view_${view_index}`,
            RenderPassFlags.Compute,
            {
              shader_setup: clear_indirect_instance_counts_shader_setup,
              inputs: [per_view_indirect_draw_buffers[view_index]],
              outputs: [per_view_indirect_draw_buffers[view_index]],
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              pass.dispatch((draw_count + 255) / 256, 1, 1);
            }
          );
        }
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌫️  PASS: Occlusion Culling (Phase 2 of 2-Pass Occlusion)                 │
      // │    Use HZB to eliminate objects hidden behind other geometry               │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (draw_count > 0) {
        for (let view_index = 0; view_index < total_views; ++view_index) {
          if (!SharedViewBuffer.is_render_active(view_index)) continue;

          const draw_cull_data = per_view_draw_cull_data[view_index];
          const visible_buf_no_occlusion = per_view_visible_no_occlusion_buffers[view_index];
          const visible_buf = per_view_visible_buffers[view_index];
          const indirect_buf = per_view_indirect_draw_buffers[view_index];

          render_graph.add_pass(
            `${compute_cull_pass_name2}_view_${view_index}`,
            RenderPassFlags.Compute,
            {
              shader_setup: compute_cull_occlusion_shader_setup,
              inputs: [
                main_hzb_image,
                aabb_bounds,
                visible_buf_no_occlusion,
                visible_buf,
                object_instances,
                entity_aabb_node_indices,
                draw_cull_data,
                indirect_buf,
              ],
              outputs: [visible_buf, indirect_buf],
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              pass.dispatch((draw_count + 255) / 256, 1, 1);
            }
          );
        }
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
            main_emissive_image,
            main_smra_image,
            main_position_image,
            main_normal_image,
            material.writes_entity_id ? main_entity_id_image : null,
            material.family === MaterialFamilyType.Transparent
              ? main_transparency_reveal_image
              : null,
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
                per_view_visible_buffers[current_view],
              ],
              outputs: outputs,
              shader_setup: g_buffer_shader_setup,
              b_skip_pass_pipeline_setup: true,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);

              MeshTaskQueue.submit_material_indexed_indirect_draws(
                pass,
                material_id,
                current_view
              );
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
          inputs: [main_transparency_accum_image, main_transparency_reveal_image],
          outputs: [main_albedo_image],
          shader_setup: transparency_composite_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);

          MeshTaskQueue.draw_quad(pass);
        }
      );

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 📏 PASS: Line Renderer                                                     │
      // │    Render debug lines and wireframes for visualization                    │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (LineRenderer.enabled && LineRenderer.line_positions.length > 0) {
        const { position_buffer, line_data_buffer, transform_buffer, visible_line_count } =
          LineRenderer.to_gpu_data();
        if (visible_line_count > 0) {
          const line_position_buffer_rg = render_graph.register_buffer(position_buffer.config.name);
          const line_data_buffer_rg = render_graph.register_buffer(line_data_buffer.config.name);
          const line_transform_buffer_rg = render_graph.register_buffer(
            transform_buffer.config.name
          );

          render_graph.add_pass(
            "line_transform_processing",
            RenderPassFlags.Compute,
            {
              inputs: [line_transform_buffer_rg, line_position_buffer_rg],
              outputs: [line_transform_buffer_rg],
              shader_setup: line_transform_processing_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              pass.dispatch((visible_line_count + 63) / 64, 1, 1);
            }
          );

          render_graph.add_pass(
            "line_renderer_pass",
            RenderPassFlags.Graphics,
            {
              inputs: [line_transform_buffer_rg, line_data_buffer_rg],
              outputs: [
                main_albedo_image,
                main_emissive_image,
                main_smra_image,
                main_position_image,
                main_normal_image,
                main_depth_image,
              ],
              shader_setup: line_draw_shader_setup,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);
              MeshTaskQueue.draw_quad(pass, visible_line_count);
            }
          );
        }
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌚 PASS: Adaptive Sparse Virtual Shadow Maps                               │
      // │    High-quality, efficient shadow mapping with virtual memory management  │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (shadows_enabled) {
        // Add AS-VSM passes with mapping
        if (!this.as_vsm) {
          this.as_vsm = new AdaptiveSparseVirtualShadowMaps({
            atlas_size: 2048,
            tile_size: 32,
            virtual_dim: 4096,
            max_lods: 1,
            histogram_bins: 64,
          });
        }
        this.as_vsm.add_passes(render_graph, {
          depth_texture: main_depth_image,
          lights_buffer: dense_lights,
          dense_shadow_casting_lights_buffer: dense_shadow_casting_lights,
          light_count_buffer: light_count,
          transforms_buffer: entity_transforms,
          object_instances: object_instances,
          view_visibility_buffers: per_view_visible_buffers,
          force_recreate: this.force_recreate,
          debug_view: this.debug_view,
        });
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 🌟 PASS: Dynamic Diffuse Global Illumination (DDGI)                       │
      // │    Real-time global illumination using probe-based irradiance volumes    │
      // └─────────────────────────────────────────────────────────────────────────────┘
      if (gi_enabled) {
        const visible_instance_buffer = per_view_visible_buffers[current_view];
        this.gi_probe_volume.update(render_graph, {
          gi_params_buffer,
          gi_irradiance_image,
          gi_depth_image,
          entity_transforms,
          entity_flags,
          visible_instance_buffer,
          lights,
          probe_cubemap: this.probe_cubemap.config.name,
        });
      }

      // ┌─────────────────────────────────────────────────────────────────────────────┐
      // │ 💡 PASS: Deferred Lighting                                                 │
      // │    Combine G-Buffer data with lights to produce final shaded results      │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        post_lighting_image_config.width = image_extent.width;
        post_lighting_image_config.height = image_extent.height;
        post_lighting_image_config.force = this.force_recreate;
        post_lighting_image_desc = render_graph.create_image(post_lighting_image_config);

        const lighting_inputs = [
          skybox_image,
          main_albedo_image,
          main_emissive_image,
          main_smra_image,
          main_normal_image,
          main_position_image,
          main_depth_image,
          dense_lights,
          dense_shadow_casting_lights,
          light_count,
        ];

        if (gi_enabled) {
          deferred_lighting_shader_setup.pipeline_shaders.vertex.defines = ["GI_ENABLED"];
          deferred_lighting_shader_setup.pipeline_shaders.fragment.defines = ["GI_ENABLED"];

          lighting_inputs.push(gi_irradiance_image, gi_depth_image);
        }

        if (shadows_enabled) {
          // Register AS-VSM shadow resources for lighting
          const shadow_atlas = this.as_vsm.shadow_atlas;
          const vsm_page_table = this.as_vsm.page_table;
          const asvsm_settings = this.as_vsm.settings_buf;

          deferred_lighting_shader_setup.pipeline_shaders.vertex.defines = ["SHADOWS_ENABLED"];
          deferred_lighting_shader_setup.pipeline_shaders.fragment.defines = ["SHADOWS_ENABLED"];

          lighting_inputs.push(shadow_atlas, vsm_page_table, asvsm_settings);
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
      // │ ✨ PASS: Bloom Post-Processing                                             │
      // │    Multi-pass gaussian blur to create beautiful light bleeding effects    │
      // └─────────────────────────────────────────────────────────────────────────────┘
      const num_iterations = 4;
      if (num_iterations > 0) {
        const image_extent = renderer.get_canvas_resolution();
        const extent_x = ppot(image_extent.width);
        const extent_y = ppot(image_extent.height);

        let bloom_blur_chain = [];
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
                0.005,
                i,
              ]);

              pass.dispatch((dst_mip_width + 15) / 16, (dst_mip_height + 15) / 16, 1);
            }
          );
        }

        post_bloom_color_image_config.width = image_extent.width;
        post_bloom_color_image_config.height = image_extent.height;
        post_bloom_color_image_config.force = this.force_recreate;
        post_bloom_color_desc = render_graph.create_image(post_bloom_color_image_config);

        let bloom_resolve_params_desc = render_graph.create_buffer(bloom_resolve_params_config);

        render_graph.add_pass(
          bloom_resolve_pass_name,
          RenderPassFlags.Graphics,
          {
            inputs: [post_lighting_image_desc, bloom_blur_chain[0], bloom_resolve_params_desc],
            outputs: [post_bloom_color_desc],
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

      const antialiased_scene_color_desc = post_bloom_color_desc;

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
      // │ 🖼️  PASS: Final Presentation                                               │
      // │    Present the final rendered image to the screen swapchain               │
      // └─────────────────────────────────────────────────────────────────────────────┘
      {
        const swapchain_image = Texture.create_from_texture(
          renderer.context.getCurrentTexture(),
          swapchain_name
        );

        const rg_output_image = render_graph.register_image(swapchain_image.config.name);

        // Choose final output based on debug_view
        let final_output_image = post_processed_image;
        if (shadows_enabled && this.debug_view === DebugView.ShadowAtlas) {
          final_output_image = this.as_vsm.debug_shadow_atlas_image;
        } else if (shadows_enabled && this.debug_view === DebugView.ShadowPageTable) {
          final_output_image = this.as_vsm.debug_page_table_image;
        }

        render_graph.add_pass(
          fullscreen_present_pass_name,
          RenderPassFlags.Graphics | RenderPassFlags.Present,
          {
            inputs: [final_output_image],
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
            if (frame_data.g_buffer_data) {
              if (frame_data.g_buffer_data.albedo) {
                frame_data.g_buffer_data.albedo.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.emissive) {
                frame_data.g_buffer_data.emissive.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.smra) {
                frame_data.g_buffer_data.smra.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.position) {
                frame_data.g_buffer_data.position.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.normal) {
                frame_data.g_buffer_data.normal.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.entity_id) {
                frame_data.g_buffer_data.entity_id.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.transparency_accum) {
                frame_data.g_buffer_data.transparency_accum.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.transparency_reveal) {
                frame_data.g_buffer_data.transparency_reveal.config.load_op = load_op_clear;
              }
              if (frame_data.g_buffer_data.depth) {
                frame_data.g_buffer_data.depth.config.load_op = load_op_clear;
              }
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

    // Use global GIProbeVolume from Renderer
    const gi_dims = this.gi_probe_volume.dims;

    gi_irradiance_config.width = gi_dims[0];
    gi_irradiance_config.height = gi_dims[1];
    gi_irradiance_config.depth = gi_dims[2];
    gi_irradiance_config.force = this.force_recreate;
    this.gi_irradiance_volume = Texture.create(gi_irradiance_config);

    gi_depth_config.width = gi_dims[0];
    gi_depth_config.height = gi_dims[1];
    gi_depth_config.depth = gi_dims[2];
    gi_depth_config.force = this.force_recreate;
    this.gi_depth_volume = Texture.create(gi_depth_config);

    // Create a reusable cubemap-array target for DDGI probes
    probe_cubemap_image_config.force = this.force_recreate;
    this.probe_cubemap = Texture.create(probe_cubemap_image_config);
  }
}
