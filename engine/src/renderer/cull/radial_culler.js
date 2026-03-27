import { InstanceCuller } from "./instance_culler.js";
import { RenderPassFlags } from "../renderer_types.js";

const compute_cull_radius_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/cull_radius.wgsl",
    },
  },
};

const clip0_extent_buf_config = {
  name: `clip0_extent_buf`,
  data: null,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};

export class RadialCuller extends InstanceCuller {
  constructor(prev_culler = null, additional_data = null) {
    super(prev_culler, additional_data);
    this.name = "radius";
    this.clip_extent_data = new Float32Array([0.0, 0.0, 0.0, 0.0]);
  }

  dispatch_culling(render_graph, draw_count) {
    for (let i = 0; i < this.registered_views.length; ++i) {
      const view_index = this.registered_views.get(i);
      const clipmap_index = this.registered_clipmaps.get(i);

      const draw_cull_data = this.cull_data_buffers.get(view_index, clipmap_index);
      const visible_buf = this.visible_buffers.get(view_index, clipmap_index);
      const indirect_buf = this.indirect_draw_buffers.get(view_index, clipmap_index);

      this.clip_extent_data[0] = this.additional_data.clip0_extent ?? 4.0;
      clip0_extent_buf_config.name = `clip0_extent_buf_view_${view_index}_clipmap_${clipmap_index}`;
      clip0_extent_buf_config.data = this.clip_extent_data;
      const clip0_extent_buf = render_graph.create_buffer(clip0_extent_buf_config);

      render_graph.add_pass(
        `cull_radius_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: compute_cull_radius_shader_setup,
          inputs: [
            this.additional_data.aabb_bounds,
            this.additional_data.object_instances,
            visible_buf,
            indirect_buf,
            draw_cull_data,
            clip0_extent_buf,
            this.additional_data.entity_index_lookup,
          ],
          outputs: [indirect_buf, visible_buf],
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const draw_cull_data_buf = graph.get_physical_buffer(draw_cull_data);

          // Update per-dispatch uniform data
          draw_cull_data_buf.write(
            new Uint32Array([
              draw_count,
              view_index,
              clipmap_index,
            ])
          );

          const clip_buf = graph.get_physical_buffer(clip0_extent_buf);
          clip_buf.write(this.clip_extent_data);

          pass.dispatch((draw_count + 255) / 256, 1, 1);
        }
      );
    }
  }
}
