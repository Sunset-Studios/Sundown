import { Buffer } from "./buffer.js";
import { Texture } from "./texture.js";
import { FrameAllocator } from "../memory/allocator.js";
import { RenderPassFlags } from "./render_pass.js";
import { profile_scope } from "../utility/performance.js";

const max_objects = 1024;

class ComputeTask {
  static init(
    task,
    name,
    shader,
    inputs,
    outputs,
    dispatch_x,
    dispatch_y,
    dispatch_z
  ) {
    task.name = name;
    task.shader = shader;
    task.inputs = inputs;
    task.outputs = outputs;
    task.dispatch_x = dispatch_x;
    task.dispatch_y = dispatch_y;
    task.dispatch_z = dispatch_z;
  }
}

export class ComputeTaskQueue {
  constructor() {
    if (ComputeTaskQueue.instance) {
      return ComputeTaskQueue.instance;
    }
    this.tasks = [];
    this.tasks_allocator = new FrameAllocator(max_objects, new ComputeTask());
    ComputeTaskQueue.instance = this;
  }

  static get() {
    if (!ComputeTaskQueue.instance) {
      ComputeTaskQueue.instance = new ComputeTaskQueue();
    }
    return ComputeTaskQueue.instance;
  }

  new_task(
    name,
    shader,
    inputs,
    outputs,
    dispatch_x,
    dispatch_y = 1,
    dispatch_z = 1
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
      dispatch_z
    );

    this.tasks.push(task);

    return task;
  }

  compile_rg_passes(render_graph) {
    profile_scope("ComputeTaskQueue.compile_rg_passes", () => {
      for (let i = 0; i < this.tasks.length; i++) {
        const task = this.tasks[i];

        for (let j = 0; j < task.inputs.length; j++) {
          if (task.inputs[j] instanceof Buffer) {
            task.inputs[j] = render_graph.register_buffer(task.inputs[j].config.name);
          } else if (task.inputs[j] instanceof Texture) {
            task.inputs[j] = render_graph.register_texture(task.inputs[j].config.name);
          }
        }

        for (let j = 0; j < task.outputs.length; j++) {
          if (task.outputs[j] instanceof Buffer) {
            task.outputs[j] = render_graph.register_buffer(task.outputs[j].config.name);
          } else if (task.outputs[j] instanceof Texture) {
            task.outputs[j] = render_graph.register_texture(task.outputs[j].config.name);
          }
        }

        render_graph.add_pass(
          task.name,
          RenderPassFlags.Compute,
          {
            shader_setup: { pipeline_shaders: { compute: { path: task.shader } } },
            inputs: task.inputs,
            outputs: task.outputs,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(
              task.dispatch_x,
              task.dispatch_y,
              task.dispatch_z
            );
          }
        );
      }
    });
  }

  reset() {
    this.tasks_allocator.reset();
    this.tasks.length = 0;
  }
}