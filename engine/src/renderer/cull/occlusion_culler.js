import { InstanceCuller } from "../instance_culler.js";
import { RenderPassFlags } from "../renderer_types.js";

const compute_cull_occlusion_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/cull_occlusion.wgsl",
    },
  },
};

export class OcclusionCuller extends InstanceCuller {
  constructor(prev_culler = null, additional_data = null) {
    super(prev_culler, additional_data);
    this.name = "occlusion";
  }

  dispatch_culling(render_graph, draw_count) {
    for (let i = 0; i < this.registered_views.length; ++i) {
      const view_index = this.registered_views.get(i);
      const clipmap_index = this.registered_clipmaps.get(i);

      const draw_cull_data = this.cull_data_buffers.get(view_index, clipmap_index);
      const visible_buf = this.visible_buffers.get(view_index, clipmap_index);
      const indirect_buf = this.indirect_draw_buffers.get(view_index, clipmap_index);
      const visible_buf_no_occlusion = this.prev_culler.get_visibility_buffer(view_index, clipmap_index);

      render_graph.add_pass(
        `cull_occlusion_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: compute_cull_occlusion_shader_setup,
          inputs: [
            this.additional_data.main_hzb_image,
            this.additional_data.aabb_bounds,
            visible_buf_no_occlusion,
            visible_buf,
            this.additional_data.object_instances,
            draw_cull_data,
            indirect_buf,
            this.additional_data.main_entity_id_image,
            this.additional_data.entity_occluders,
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
}
