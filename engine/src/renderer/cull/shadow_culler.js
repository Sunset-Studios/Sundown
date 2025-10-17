import { InstanceCuller } from "../instance_culler.js";
import { RenderPassFlags } from "../renderer_types.js";

const compute_cull_shadows_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "shadow/as_vsm/cull_shadows.wgsl",
      defines: { SHADOWS_ENABLED: true },
    },
  },
};

const compute_dirty_movable_entities_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "shadow/as_vsm/dirty_movable_entities.wgsl",
      defines: { SHADOWS_ENABLED: true },
    },
  },
};

const compute_dirty_slice_reducer_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "shadow/as_vsm/dirty_slice_reducer.wgsl",
      defines: { SHADOWS_ENABLED: true },
    },
  },
};

export class ShadowCuller extends InstanceCuller {
  constructor(prev_culler = null, additional_data = null) {
    super(prev_culler, additional_data);
    this.name = "shadow";
  }

  dispatch_culling(render_graph, draw_count, lights_dirtied) {
    // If no lights were dirtied (moved or explicitly marked dirty),
    // we can check for movable entities to dirty tiles more granularly.
    if (!lights_dirtied) {
      for (let i = 0; i < this.registered_views.length; ++i) {
        const view_index = this.registered_views.get(i);
        const clipmap_index = this.registered_clipmaps.get(i);

        const visible_buf_no_occlusion = this.prev_culler.get_visibility_buffer(
          view_index,
          clipmap_index
        );
        const visible_buf = this.visible_buffers.get(view_index, clipmap_index);
        const draw_cull_data = this.cull_data_buffers.get(view_index, clipmap_index);

        render_graph.add_pass(
          `dirty_movable_entities_${view_index}_clipmap_${clipmap_index}`,
          RenderPassFlags.Compute,
          {
            shader_setup: compute_dirty_movable_entities_shader_setup,
            inputs: [
              this.additional_data.aabb_bounds,
              visible_buf_no_occlusion,
              visible_buf,
              this.additional_data.object_instances,
              draw_cull_data,
              this.additional_data.vsm_settings,
              this.additional_data.entity_flags,
              this.additional_data.bitmask,
              this.additional_data.entity_index_lookup,
              this.additional_data.page_table,
              this.additional_data.page_offset,
            ],
            outputs: [visible_buf],
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch((draw_count + 255) / 256, this.additional_data.light_count, 1);
          }
        );
      }
    }

    // Reduce dirty slices to check if any tiles are dirty within each slice
    render_graph.add_pass(
      "dirty_slice_reducer",
      RenderPassFlags.Compute,
      {
        shader_setup: compute_dirty_slice_reducer_shader_setup,
        inputs: [
          this.additional_data.vsm_settings,
          this.additional_data.page_table,
          this.additional_data.dirty_slices,
        ],
        outputs: [this.additional_data.dirty_slices],
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const pt_image = graph.get_physical_image(this.additional_data.page_table);
        const x_groups = Math.ceil(pt_image.config.width / 8);
        const y_groups = Math.ceil(pt_image.config.height / 8);
        const z_groups = Math.ceil(pt_image.config.depth / 4);
        pass.dispatch(x_groups, y_groups, z_groups);
      }
    );

    // Cull shadow casters for all registered views
    for (let i = 0; i < this.registered_views.length; ++i) {
      const view_index = this.registered_views.get(i);
      const clipmap_index = this.registered_clipmaps.get(i);

      const visible_buf_no_occlusion = this.prev_culler.get_visibility_buffer(
        view_index,
        clipmap_index
      );
      const visible_buf = this.visible_buffers.get(view_index, clipmap_index);
      const indirect_buf = this.indirect_draw_buffers.get(view_index, clipmap_index);
      const draw_cull_data = this.cull_data_buffers.get(view_index, clipmap_index);

      render_graph.add_pass(
        `cull_shadow_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: compute_cull_shadows_shader_setup,
          inputs: [
            visible_buf_no_occlusion,
            visible_buf,
            this.additional_data.object_instances,
            draw_cull_data,
            indirect_buf,
            this.additional_data.vsm_settings,
            this.additional_data.page_table,
            this.additional_data.dirty_slices,
          ],
          outputs: [visible_buf, indirect_buf],
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch((draw_count + 255) / 256, this.additional_data.light_count, 1);
        }
      );
    }
  }
}
