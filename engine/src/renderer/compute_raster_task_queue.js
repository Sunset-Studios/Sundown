import { Buffer } from "./buffer.js";
import { Texture } from "./texture.js";
import { RandomAccessAllocator } from "../memory/allocator.js";
import { profile_scope } from "../utility/performance.js";
import { RenderPassFlags } from "./renderer_types.js";
import { Renderer } from "./renderer.js";

const compile_rg_pass_scope_name = "ComputeRasterTaskQueue.compile_rg_passes";
const workgroup_size = 256;

export const ComputeRasterPrimitiveType = {
  Point: "point",
  Line: "line",
  Triangle: "triangle",
  Quad: "quad",
};

const ComputeRasterPrimitiveStride = {
  [ComputeRasterPrimitiveType.Point]: 1,
  [ComputeRasterPrimitiveType.Line]: 2,
  [ComputeRasterPrimitiveType.Triangle]: 3,
  [ComputeRasterPrimitiveType.Quad]: 4,
};

class ComputeRasterTask {
  static init(task, name, shader, points, connections, inputs, primitive_type) {
    if (!points || !(points instanceof Buffer)) {
      throw new Error("ComputeRasterTask requires a valid 'points' buffer.");
    }

    // Validate and determine stride from supported primitive types
    const stride = ComputeRasterPrimitiveStride[primitive_type];
    if (stride === undefined) {
      throw new Error(`Unsupported or undefined primitive type: ${primitive_type}`);
    }

    // Determine the number of primitives (e.g. points) from the connections buffer
    let raw = connections.config.raw_data;
    let num_connections = ArrayBuffer.isView(raw) ? raw.length : 0;

    const num_primitives = Math.floor(num_connections / stride);
    const dispatch_count = Math.ceil(num_primitives / workgroup_size);

    task.name = name;
    task.shader = shader;
    task.points = points;
    task.connections = connections;
    task.inputs = inputs;
    task.primitive_type = primitive_type;
    task.stride = stride;
    task.num_primitives = num_primitives;
    task.dispatch_x = dispatch_count;
    task.dispatch_y = 1;
    task.dispatch_z = 1;
    task.intermediate_buffers = null;
  }
}

export class ComputeRasterTaskQueue {
  static tasks = [];
  static tasks_allocator = new RandomAccessAllocator(256, new ComputeRasterTask());

  /**
   * Creates a new compute raster task.
   * @param {string} name - The name of the task.
   * @param {string} shader - The path to the compute shader.
   * @param {Buffer} points - The required points buffer (vertex buffer).
   * @param {Buffer} connections - The required connections buffer (index buffer).
   * @param {Buffer[]} inputs - The optional input buffers.
   * @param {string} primitiveType - The type of primitive to rasterize (e.g., 'point', 'line').
   * @returns {ComputeRasterTask} The newly created task.
   */
  static new_task(name, shader, points, connections, inputs, primitiveType) {
    const task = this.tasks_allocator.allocate();

    ComputeRasterTask.init(task, name, shader, points, connections, inputs, primitiveType);

    this.tasks.push(task);

    return task;
  }

  static compile_rg_passes(render_graph, pipeline_outputs) {
    profile_scope(compile_rg_pass_scope_name, () => {
      // For each task, bind outputs directly to textures
      for (let i = 0; i < this.tasks.length; i++) {
        const task = this.tasks[i];
        // Register required buffers
        task.points = render_graph.register_buffer(task.points.config.name);
        task.connections = render_graph.register_buffer(task.connections.config.name);
        for (let k = 0; k < task.inputs.length; k++) {
          task.inputs[k] = render_graph.register_buffer(task.inputs[k].config.name);
        }

        // Add compute pass writing directly to G-Buffer textures
        render_graph.add_pass(
          task.name,
          RenderPassFlags.Compute,
          {
            shader_setup: { pipeline_shaders: { compute: { path: task.shader } } },
            inputs: [task.points, task.connections, ...pipeline_outputs, ...task.inputs],
            outputs: pipeline_outputs,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(task.dispatch_x, task.dispatch_y, task.dispatch_z);
          }
        );
      }
      // Clear tasks after dispatch
      this.reset();
    });
  }

  static reset() {
    this.tasks_allocator.reset();
    this.tasks.length = 0;
  }
}
