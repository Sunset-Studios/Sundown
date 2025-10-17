import { InstanceCuller } from "../instance_culler.js";
import { RenderPassFlags } from "../renderer_types.js";

const compute_cull_frustum_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/cull_frustum.wgsl",
    },
  },
};

export class FrustumCuller extends InstanceCuller {
  constructor(prev_culler = null, additional_data = null) {
    super(prev_culler, additional_data);
    this.name = "frustum";
  }

  dispatch_culling(render_graph, draw_count) {
    for (let i = 0; i < this.registered_views.length; ++i) {
      const view_index = this.registered_views.get(i);
      const clipmap_index = this.registered_clipmaps.get(i);

      const draw_cull_data = this.cull_data_buffers.get(view_index, clipmap_index);
      const visible_buf = this.visible_buffers.get(view_index, clipmap_index);
      const indirect_buf = this.indirect_draw_buffers.get(view_index, clipmap_index);

      render_graph.add_pass(
        `cull_frustum_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: compute_cull_frustum_shader_setup,
          inputs: [
            this.additional_data.aabb_bounds,
            this.additional_data.object_instances,
            visible_buf,
            indirect_buf,
            draw_cull_data,
            this.additional_data.entity_index_lookup,
          ],
          outputs: [indirect_buf, visible_buf],
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const draw_cull_data_buf = graph.get_physical_buffer(draw_cull_data);

          draw_cull_data_buf.write(
            new Uint32Array([
              draw_count,
              view_index,
              clipmap_index,
            ])
          );
          pass.dispatch((draw_count + 255) / 256, 1, 1);
        }
      );
    }
  }
}
