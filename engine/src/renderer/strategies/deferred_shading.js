import { Renderer } from "../renderer.js";
import { Texture } from "../texture.js";
import { RenderPassFlags, MaterialFamilyType } from "../renderer_types.js";
import { AABB } from "../../acceleration/aabb.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";
import { ComputeRasterTaskQueue } from "../compute_raster_task_queue.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { SceneGraphFragment } from "../../core/ecs/fragments/scene_graph_fragment.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { SharedViewBuffer, SharedEnvironmentMapData } from "../../core/shared_data.js";
import { Material } from "../material.js";
import { PostProcessStack } from "../post_process_stack.js";
import { npot, clamp } from "../../utility/math.js";
import { profile_scope } from "../../utility/performance.js";
import { global_dispatcher } from "../../core/dispatcher.js";
import {
  rgba16float_format,
  r8unorm_format,
  depth32float_format,
  bgra8unorm_format,
  r32float_format,
  rg32uint_format,
  one_one_blend_config,
  src_alpha_one_minus_src_alpha_blend_config,
  load_op_load,
  load_op_clear,
} from "../../utility/config_permutations.js";
import { LineRenderer } from "../line_renderer.js";

const resolution_change_event_name = "resolution_change";
const deferred_shading_profile_scope_name = "DeferredShadingStrategy.draw";

const main_albedo_image_config = {
  name: "main_albedo",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_emissive_image_config = {
  name: "main_emissive",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_smra_image_config = {
  name: "main_smra",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_normal_image_config = {
  name: "main_normal",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_position_image_config = {
  name: "main_position",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  force: false,
};
const main_transparency_accum_image_config = {
  name: "main_transparency_accum",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
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
  format: bgra8unorm_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const compute_cull_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "cull.wgsl",
    },
  },
};
const draw_cull_data_config = {
  name: `draw_cull_data`,
  data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};

const depth_only_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "depth_only.wgsl",
    },
  },
  depth_write_enabled: true,
  depth_compare: "less",
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
  depth_compare: "less",
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
      path: "hzb_reduce.wgsl",
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
  format: rgba16float_format,
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
    vertex: {
      path: "fullscreen.wgsl",
    },
    fragment: {
      path: "fullscreen.wgsl",
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

const swapchain_name = "swapchain";

const clear_g_buffer_pass_name = "clear_g_buffer";
const skybox_pass_name = "skybox_pass";
const depth_prepass_name = "depth_prepass";
const transparency_composite_pass_name = "transparency_composite";
const reset_g_buffer_targets_pass_name = "reset_g_buffer_targets";
const compute_cull_pass_name = "compute_cull";
const lighting_pass_name = "lighting_pass";
const bloom_resolve_pass_name = "bloom_resolve_pass";
const fullscreen_present_pass_name = "fullscreen_present_pass";

export class DeferredShadingStrategy {
  initialized = false;
  hzb_image = null;
  entity_id_image = null;
  force_recreate = false;

  setup(render_graph) {
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

      const renderer = Renderer.get();

      MeshTaskQueue.get().sort_and_batch();
      ComputeTaskQueue.get().compile_rg_passes(render_graph);

      const transform_gpu_data = TransformFragment.to_gpu_data();
      const entity_transforms = render_graph.register_buffer(
        transform_gpu_data.transforms_buffer.config.name
      );
      const entity_aabb_node_indices = render_graph.register_buffer(
        transform_gpu_data.aabb_node_index_buffer.config.name
      );

      const scene_graph_gpu_data = SceneGraphFragment.to_gpu_data();

      const aabb_gpu_data = AABB.to_gpu_data();
      const aabb_bounds = render_graph.register_buffer(
        aabb_gpu_data.node_bounds_buffer.config.name
      );

      const light_gpu_data = LightFragment.to_gpu_data();
      const lights = render_graph.register_buffer(light_gpu_data.light_fragment_buffer.config.name);

      let skybox_image = null;
      let post_lighting_image_desc = null;
      let post_bloom_color_desc = null;

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

      if (this.force_recreate) {
        render_graph.mark_pass_cache_bind_groups_dirty(true /* pass_only */);
      }

      // Clear G-Buffer Pass
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

      // Skybox Pass
      skybox_output_image_config.width = image_extent.width;
      skybox_output_image_config.height = image_extent.height;
      skybox_output_image_config.force = this.force_recreate;
      skybox_image = render_graph.create_image(skybox_output_image_config);

      {
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
              MeshTaskQueue.get().draw_cube(pass);
            }
          );
        }
      }

      const object_instance_buffer = MeshTaskQueue.get().get_object_instance_buffer();
      const object_instances = render_graph.register_buffer(object_instance_buffer.config.name);

      const indirect_draw_buffer = MeshTaskQueue.get().get_indirect_draw_buffer();
      const indirect_draws = render_graph.register_buffer(indirect_draw_buffer.config.name);

      const compacted_object_instances = MeshTaskQueue.get().get_compacted_object_instance_buffer();
      const compacted_object_instance_buffer = render_graph.register_buffer(
        compacted_object_instances.config.name
      );

      // Mesh cull pass
      if (MeshTaskQueue.get().get_total_draw_count() > 0) {
        // Compute cull pass
        draw_cull_data_config.data.fill(0);
        const draw_cull_data = render_graph.create_buffer(draw_cull_data_config);

        render_graph.add_pass(
          compute_cull_pass_name,
          RenderPassFlags.Compute,
          {
            shader_setup: compute_cull_shader_setup,
            inputs: [
              main_hzb_image,
              aabb_bounds,
              object_instances,
              compacted_object_instance_buffer,
              indirect_draws,
              entity_aabb_node_indices,
              draw_cull_data,
            ],
            outputs: [indirect_draws],
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);

            const hzb = graph.get_physical_image(main_hzb_image);
            const draw_cull = graph.get_physical_buffer(draw_cull_data);
            const draw_count = MeshTaskQueue.get().get_total_draw_count();
            const view_data = SharedViewBuffer.get_view_data(0);

            let p00 = view_data.projection_matrix[0];
            let p11 = view_data.projection_matrix[5];
            draw_cull.write([
              draw_count,
              1 /* culling_enabled */,
              1 /* occlusion_enabled */,
              1 /* distance_check */,
              view_data.near,
              view_data.far,
              p00,
              p11,
              hzb.config.width,
              hzb.config.height,
            ]);

            pass.dispatch((draw_count + 255) / 256, 1, 1);
          }
        );
      }

      // TODO: Meshlet cull pass

      // Depth prepass
      {
        render_graph.add_pass(
          depth_prepass_name,
          RenderPassFlags.Graphics,
          {
            inputs: [entity_transforms, compacted_object_instance_buffer],
            outputs: [main_depth_image],
            shader_setup: depth_only_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            MeshTaskQueue.get().submit_indexed_indirect_draws(pass, frame_data);
          }
        );
      }

      // Compute rasterization passes
      // TODO: Automatically run software rasterization over triangle clusters that fall within some maximum screen size
      {
        // Rasterize particle positions into the G-Buffer (albedo & depth)
        ComputeRasterTaskQueue.compile_rg_passes(
          render_graph,
          [main_albedo_image, main_depth_image]
        );
      }

      // GBuffer Base Pass
      {
        const material_buckets = MeshTaskQueue.get().get_material_buckets();
        for (let i = 0; i < material_buckets.length; i++) {
          const material_id = material_buckets[i];
          const material = Material.get(material_id);

          render_graph.add_pass(
            `g_buffer_${material.template.name}_${material_id}`,
            RenderPassFlags.Graphics,
            {
              inputs: [entity_transforms, compacted_object_instance_buffer, lights],
              outputs: [
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
              ],
              shader_setup: g_buffer_shader_setup,
              b_skip_pass_pipeline_setup: true,
            },
            (graph, frame_data, encoder) => {
              const pass = graph.get_physical_pass(frame_data.current_pass);

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

              MeshTaskQueue.get().submit_material_indexed_indirect_draws(
                pass,
                frame_data,
                material_id,
                false /* should_reset */
              );
            }
          );
        }
      }

      // Transparency Composite Pass
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

          MeshTaskQueue.get().draw_quad(pass);
        }
      );

      // Add line renderer pass
      if (LineRenderer.enabled && LineRenderer.line_positions.length > 0) {
        const { position_buffer, line_data_buffer, transform_buffer, visible_line_count } = LineRenderer.to_gpu_data();
        if (visible_line_count > 0) {
          const line_position_buffer_rg = render_graph.register_buffer(position_buffer.config.name);
          const line_data_buffer_rg = render_graph.register_buffer(line_data_buffer.config.name);
          const line_transform_buffer_rg = render_graph.register_buffer(transform_buffer.config.name);
    
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
              MeshTaskQueue.get().draw_quad(pass, visible_line_count);
            }
          );
        }
      }



      // Reset GBuffer targets
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

      // HZB generation pass
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

              pass.dispatch((dst_mip_width + 15) / 16, (dst_mip_height + 15) / 16, 1);
            }
          );
        }
      }

      // Lighting Pass
      {
        post_lighting_image_config.width = image_extent.width;
        post_lighting_image_config.height = image_extent.height;
        post_lighting_image_config.force = this.force_recreate;
        post_lighting_image_desc = render_graph.create_image(post_lighting_image_config);

        render_graph.add_pass(
          lighting_pass_name,
          RenderPassFlags.Graphics,
          {
            inputs: [
              skybox_image,
              main_albedo_image,
              main_emissive_image,
              main_smra_image,
              main_normal_image,
              main_position_image,
              main_depth_image,
              lights,
            ],
            outputs: [post_lighting_image_desc],
            shader_setup: deferred_lighting_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            MeshTaskQueue.get().draw_quad(pass);
          }
        );
      }

      // Bloom pass
      const num_iterations = 4;
      if (num_iterations > 0) {
        const image_extent = renderer.get_canvas_resolution();
        const extent_x = npot(image_extent.width);
        const extent_y = npot(image_extent.height);

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

            MeshTaskQueue.get().draw_quad(pass);
          }
        );
      }

      const antialiased_scene_color_desc = post_bloom_color_desc;

      // Post Process Pass
      const post_processed_image = PostProcessStack.compile_passes(
        0,
        render_graph,
        post_bloom_color_image_config,
        antialiased_scene_color_desc,
        main_depth_image,
        main_normal_image
      );

      // Fullscreen Present Pass
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
            MeshTaskQueue.get().draw_quad(pass);
          }
        );
      }

      this.force_recreate = false;

      ComputeTaskQueue.get().reset();

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
