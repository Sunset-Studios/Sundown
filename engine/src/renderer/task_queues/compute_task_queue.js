import { Buffer } from "../buffer.js";
import { Texture } from "../texture.js";
import { RandomAccessAllocator } from "../../memory/allocator.js";
import { profile_scope } from "../../utility/performance.js";
import { RenderPassFlags } from "../renderer_types.js";

class ComputeTask {
  static init(
    task,
    name,
    shader,
    inputs,
    outputs,
    dispatch_x,
    dispatch_y,
    dispatch_z,
    entry_point = "cs"
  ) {
    task.name = name;
    task.shader = shader;
    task.inputs = inputs;
    task.outputs = outputs;
    task.dispatch_x = dispatch_x;
    task.dispatch_y = dispatch_y;
    task.dispatch_z = dispatch_z;
    task.entry_point = entry_point;
  }
}

export class ComputeTaskQueue {
  static pre_tasks = [];
  static post_tasks = [];
  static tasks_allocator = new RandomAccessAllocator(256, new ComputeTask());

  static new_task(
    name,
    shader,
    inputs,
    outputs,
    dispatch_x,
    dispatch_y = 1,
    dispatch_z = 1,
    entry_point = "cs",
    stage = "pre"
  ) {
    const task = this.tasks_allocator.allocate();

    ComputeTask.init(
      task,
      name,
      shader,
      inputs,
      outputs,
      dispatch_x,
      dispatch_y,
      dispatch_z,
      entry_point
    );

    if (stage === "pre") {
      this.pre_tasks.push(task);
    } else {
      this.post_tasks.push(task);
    }

    return task;
  }

  static compile_pre_rg_passes(render_graph) {
    profile_scope("ComputeTaskQueue.compile_pre_rg_passes", () => {
      for (let i = 0; i < this.pre_tasks.length; i++) {
        const task = this.pre_tasks[i];

        for (let j = 0; j < task.inputs.length; j++) {
          if (task.inputs[j] instanceof Buffer) {
            task.inputs[j] = render_graph.register_buffer(task.inputs[j].config.name);
          } else if (task.inputs[j] instanceof Texture) {
            task.inputs[j] = render_graph.register_image(task.inputs[j].config.name);
          }
        }

        for (let j = 0; j < task.outputs.length; j++) {
          if (task.outputs[j] instanceof Buffer) {
            task.outputs[j] = render_graph.register_buffer(task.outputs[j].config.name);
          } else if (task.outputs[j] instanceof Texture) {
            task.outputs[j] = render_graph.register_image(task.outputs[j].config.name);
          }
        }

        render_graph.add_pass(
          task.name,
          RenderPassFlags.Compute,
          {
            shader_setup: {
              pipeline_shaders: { compute: { path: task.shader, entry_point: task.entry_point } },
            },
            inputs: task.inputs,
            outputs: task.outputs,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(task.dispatch_x, task.dispatch_y, task.dispatch_z);
          }
        );
      }
    });
  }

  static compile_post_rg_passes(render_graph) {
    profile_scope("ComputeTaskQueue.compile_post_rg_passes", () => {
      for (let i = 0; i < this.post_tasks.length; i++) {
        const task = this.post_tasks[i];

        for (let j = 0; j < task.inputs.length; j++) {
          if (task.inputs[j] instanceof Buffer) {
            task.inputs[j] = render_graph.register_buffer(task.inputs[j].config.name);
          } else if (task.inputs[j] instanceof Texture) {
            task.inputs[j] = render_graph.register_image(task.inputs[j].config.name);
          }
        }

        for (let j = 0; j < task.outputs.length; j++) {
          if (task.outputs[j] instanceof Buffer) {
            task.outputs[j] = render_graph.register_buffer(task.outputs[j].config.name);
          } else if (task.outputs[j] instanceof Texture) {
            task.outputs[j] = render_graph.register_image(task.outputs[j].config.name);
          }
        }

        render_graph.add_pass(
          task.name,
          RenderPassFlags.Compute,
          {
            shader_setup: {
              pipeline_shaders: { compute: { path: task.shader, entry_point: task.entry_point } },
            },
            inputs: task.inputs,
            outputs: task.outputs,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(task.dispatch_x, task.dispatch_y, task.dispatch_z);
          }
        );
      }
    });
  }

  static reset() {
    this.tasks_allocator.reset();
    this.pre_tasks.length = 0;
    this.post_tasks.length = 0;
  }
}
