import { BVH } from "../../acceleration/bvh.js";
import { MeshBLAS } from "../../acceleration/mesh_blas.js";
import { DebugOverlay } from "../debug_overlay.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import { DebugDrawType, RenderPassFlags } from "../renderer_types.js";

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

export class DeferredDebugPipeline {
  debug_overlay = null;

  constructor() {
    this.debug_overlay = new DebugOverlay();
  }

  add_geometry_passes(
    render_graph,
    {
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
      culling_pipeline,
      main_albedo_image,
      main_smra_image,
      main_normal_image,
      main_motion_emissive_image,
      main_depth_image,
    }
  ) {
    if (
      debug_view !== DebugDrawType.EntityBounds &&
      debug_view !== DebugDrawType.BVH &&
      debug_view !== DebugDrawType.BLAS_Bounds
    ) {
      return;
    }

    let max_nodes_debug = BVH.bvh_size;
    if (debug_view === DebugDrawType.BLAS_Bounds) {
      max_nodes_debug = MeshBLAS.bounds_size;
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
      const directory_buffer_size = blas_gpu_data.directory_buffer.config.size;
      const directory_entry_size = 6;
      const mesh_count = Math.floor(directory_buffer_size / (directory_entry_size * 4));

      const closest_entities_per_mesh_buf = render_graph.create_buffer({
        name: "closest_entities_per_mesh",
        size: mesh_count,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const closest_distances_per_mesh_buf = render_graph.create_buffer({
        name: "closest_distances_per_mesh",
        size: mesh_count,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      render_graph.add_pass(
        "debug_init_closest_distances",
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
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
            culling_pipeline.get_frustum_visibility_buffer(current_view, 0),
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

  add_overlay_pass(
    render_graph,
    {
      debug_view,
      image_extent,
      debug_texture_level,
      post_processed_image,
      post_lighting_image,
      prev_lighting_image,
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
      culling_pipeline,
      as_vsm,
      bloom,
      ao,
      gi,
      reflections,
      reflections_enabled,
    }
  ) {
    if (debug_view === DebugDrawType.None) {
      return;
    }

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
          0,
          [0.0, 0.0, 0.0, 1.0],
          1
        );
        break;
      case DebugDrawType.Motion:
        this.debug_overlay.set_properties(
          [main_motion_emissive_image, main_depth_image, post_lighting_image],
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
          [visibility_entity_image, visibility_surface_image],
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.VisibilityMeshletId
        );
        break;
      case DebugDrawType.VisibilityTriangleId:
        this.debug_overlay.set_properties(
          [visibility_entity_image, visibility_surface_image],
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.VisibilityTriangleId
        );
        break;
      case DebugDrawType.HZB: {
        const hzb_max_level = Math.max(0, culling_pipeline.get_hzb_mip_level_count() - 1);
        const hzb_texture_level = Math.min(debug_texture_level, hzb_max_level);
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
      }
      case DebugDrawType.ASVSM_ShadowAtlas:
        this.debug_overlay.set_properties(
          as_vsm.debug_shadow_atlas_image,
          0,
          0,
          Math.min(image_extent.width, image_extent.height) * 0.35,
          Math.min(image_extent.width, image_extent.height) * 0.35,
          DebugDrawType.ASVSM_ShadowAtlas
        );
        break;
      case DebugDrawType.ASVSM_ShadowPageTable:
        this.debug_overlay.set_properties(
          as_vsm.debug_page_table_image,
          0,
          0,
          Math.min(image_extent.width, image_extent.height) * 0.25,
          Math.min(image_extent.width, image_extent.height) * 0.25,
          DebugDrawType.ASVSM_ShadowPageTable
        );
        break;
      case DebugDrawType.ASVSM_TileOverlay:
        this.debug_overlay.set_properties(
          as_vsm.debug_tile_overlay_image,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.ASVSM_TileOverlay
        );
        break;
      case DebugDrawType.ASVSM_TileRenderOutput:
        this.debug_overlay.set_properties(
          as_vsm.debug_tile_render_output_image,
          0,
          0,
          Math.min(image_extent.width, image_extent.height) * 0.3,
          Math.min(image_extent.width, image_extent.height) * 0.3,
          DebugDrawType.ASVSM_TileRenderOutput
        );
        break;
      case DebugDrawType.ASVSM_DirtyTiles:
        this.debug_overlay.set_properties(
          as_vsm.debug_dirty_tiles_image,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.ASVSM_DirtyTiles
        );
        break;
      case DebugDrawType.Bloom:
        this.debug_overlay.set_properties(
          bloom.debug_bloom_image,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.Bloom
        );
        break;
      case DebugDrawType.AO:
        this.debug_overlay.set_properties(
          ao.ao_blur_texture || ao.ao_texture,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.AO
        );
        break;
      case DebugDrawType.BentNormal:
        this.debug_overlay.set_properties(
          ao.bent_normal_texture,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.BentNormal
        );
        break;
      case DebugDrawType.GI_Direct:
        this.debug_overlay.set_properties(
          gi.final_gi_texture_direct,
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
            ? reflections.reflection_texture
            : gi.final_gi_texture_indirect_specular,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.GI_Specular
        );
        break;
      case DebugDrawType.GI_Diffuse:
        this.debug_overlay.set_properties(
          gi.final_gi_texture_indirect_diffuse,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.GI_Diffuse
        );
        break;
      case DebugDrawType.GI_WorldCache:
        this.debug_overlay.set_properties(
          gi.debug_texture,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.GI_WorldCache
        );
        break;
      case DebugDrawType.GI_Probes:
        this.debug_overlay.set_properties(
          gi.debug_texture,
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
            ? reflections.reflection_texture
            : gi.final_gi_texture_indirect_specular,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.GI_Reflections
        );
        break;
      case DebugDrawType.PrevLightingPyramid: {
        const max_level = Math.max(0, prev_lighting_image.config.mip_levels - 1);
        const texture_level = Math.min(debug_texture_level, max_level);
        this.debug_overlay.set_properties(
          prev_lighting,
          0,
          0,
          image_extent.width,
          image_extent.height,
          DebugDrawType.PrevLightingPyramid,
          texture_level + 1
        );
        break;
      }
      default:
        break;
    }

    this.debug_overlay.add_pass(render_graph, post_processed_image);
  }
}
