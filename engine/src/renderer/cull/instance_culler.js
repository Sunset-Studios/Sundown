import { Typed2DFrameArray, TypedVector } from "../../memory/container.js";
import { RenderPassFlags } from "../renderer_types.js";
import { RenderTaskQueue } from "../render_task_queue.js";

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

const draw_cull_data_config = {
  name: `draw_cull_data`,
  data: [0, 0, 0],
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};

const visible_buf_config = {
  name: `visible_instances`,
  raw_data: null,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  force: true,
};

export class InstanceCuller {
  constructor(prev_culler = null, additional_data = null) {
    this.name = "";
    this.registered_views = new TypedVector(16, Uint32Array);
    this.registered_clipmaps = new TypedVector(16, Uint32Array);
    this.visible_buffers = new Typed2DFrameArray(16, 4, Uint32Array);
    this.indirect_draw_buffers = new Typed2DFrameArray(16, 4, Uint32Array);
    this.cull_data_buffers = new Typed2DFrameArray(16, 4, Uint32Array);
    this.additional_data = additional_data;
    this.prev_culler = prev_culler;
    this.last_draw_count = 0;
  }

  // Registers the per-view/clipmap buffers
  register_view(render_graph, draw_count, view_index, clipmap_index = 0, force = false) {
    let adjusted_draw_count = Math.max(draw_count, 1);

    this.registered_views.push(view_index);
    this.registered_clipmaps.push(clipmap_index);

    visible_buf_config.name = `visible_instances_${this.name}_view_${view_index}_clipmap_${clipmap_index}`;
    visible_buf_config.force = adjusted_draw_count !== this.last_draw_count || force;
    if (visible_buf_config.force) {
      visible_buf_config.raw_data = new Int32Array(adjusted_draw_count * 2);
    }
    const visible_buf = render_graph.create_buffer(visible_buf_config);

    const indirect_draw_buf = render_graph.register_buffer(
      RenderTaskQueue.get_indirect_draw_buffer(view_index, clipmap_index).config.name
    );

    draw_cull_data_config.name = `draw_cull_data_view_${view_index}_clipmap_${clipmap_index}`;
    const draw_cull_data = render_graph.create_buffer(draw_cull_data_config);

    this.visible_buffers.set(view_index, clipmap_index, visible_buf);
    this.indirect_draw_buffers.set(view_index, clipmap_index, indirect_draw_buf);
    this.cull_data_buffers.set(view_index, clipmap_index, draw_cull_data);
  }

  // Initializes the view buffers
  init_views(render_graph, draw_count) {
    let adjusted_draw_count = Math.max(draw_count, 1);

    render_graph.add_pass(
      `${this.name}_init_views`,
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        for (let i = 0; i < this.registered_views.length; ++i) {
          const view_index = this.registered_views.get(i);
          const clipmap_index = this.registered_clipmaps.get(i);

          const draw_cull_data = this.cull_data_buffers.get(view_index, clipmap_index);
          const draw_cull = graph.get_physical_buffer(draw_cull_data);
          draw_cull.write(new Uint32Array([adjusted_draw_count, view_index, clipmap_index]));
        }
      }
    );
  }

  // Resets the corresponding visibility buffers
  init_visibility(render_graph, draw_count) {
    let adjusted_draw_count = Math.max(draw_count, 1);

    for (let i = 0; i < this.registered_views.length; ++i) {
      const view_index = this.registered_views.get(i);
      const clipmap_index = this.registered_clipmaps.get(i);

      const visible_buf = this.visible_buffers.get(view_index, clipmap_index);

      render_graph.add_pass(
        `${this.name}_clear_visibility_data_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: clear_visibility_data_shader_setup,
          inputs: [visible_buf],
          outputs: [visible_buf],
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch((adjusted_draw_count + 255) / 256, 1, 1);
        }
      );
    }
  }

  // Resets the indirect draw instance counts
  reset_instances(render_graph, draw_count) {
    let adjusted_draw_count = Math.max(draw_count, 1);

    for (let i = 0; i < this.registered_views.length; ++i) {
      const view_index = this.registered_views.get(i);
      const clipmap_index = this.registered_clipmaps.get(i);

      const indirect_draw_buf = this.indirect_draw_buffers.get(view_index, clipmap_index);
      render_graph.add_pass(
        `${this.name}_reset_instance_counts_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: clear_indirect_instance_counts_shader_setup,
          inputs: [indirect_draw_buf],
          outputs: [indirect_draw_buf],
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch((adjusted_draw_count + 255) / 256, 1, 1);
        }
      );
    }
  }

  // Abstract: Subclasses should implement this to submit custom culling dispatches
  dispatch_culling(render_graph, draw_count, ...args) {
    throw new Error("dispatch_culling must be implemented by subclass");
  }

  // Resets the registered views and clipmaps
  reset() {
    this.registered_views.clear();
    this.registered_clipmaps.clear();
  }

  // Calls reset_instances and then dispatch_culling
  submit_cull(render_graph, draw_count, ...args) {
    let adjusted_draw_count = Math.max(draw_count, 1);

    this.reset_instances(render_graph, adjusted_draw_count);
    this.dispatch_culling(render_graph, adjusted_draw_count, ...args);

    this.last_draw_count = adjusted_draw_count;
  }

  // Returns the visibility buffer for a given view and clipmap
  get_visibility_buffer(view_index, clipmap_index) {
    return this.visible_buffers.get(view_index, clipmap_index);
  }

  // Returns the visibility buffers
  get_visibility_buffers() {
    return this.visible_buffers;
  }

  // Returns the indirect draw buffer for a given view and clipmap
  get_indirect_draw_buffer(view_index, clipmap_index) {
    return this.indirect_draw_buffers.get(view_index, clipmap_index);
  }

  // Sets the previous culler for operations that require chained visibility buffers
  set_previous_culler(prev_culler) {
    this.prev_culler = prev_culler;
  }
}
