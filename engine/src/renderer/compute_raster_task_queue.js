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

    if (!ComputeRasterPrimitiveType.hasOwnProperty(primitive_type.toUpperCase())) {
      throw new Error(`Unsupported primitive type: ${primitive_type}`);
    }

    const stride = ComputeRasterPrimitiveStride[primitive_type];
    if (!stride) {
      throw new Error(`Stride not defined for primitive type: ${primitive_type}`);
    }

    const num_connections = connections.size; // Assuming 'size' represents the number of points
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
      const renderer = Renderer.get();
      const canvas_width = renderer.canvas.width;
      const canvas_height = renderer.canvas.height;

      for (let i = 0; i < this.tasks.length; i++) {
        const task = this.tasks[i];

        task.intermediate_buffers = Array(pipeline_outputs.length).fill(null);

        // Register the required points buffer
        task.points = render_graph.register_buffer(task.points.config.name);

        // Register other inputs (if any)
        for (let j = 0; j < task.inputs.length; j++) {
          task.inputs[j] = render_graph.register_buffer(task.inputs[j].config.name);
        }

        // Derive outputs from the rendering strategy's pipeline outputs
        for (let j = 0; j < pipeline_outputs.length; j++) {
          const output = pipeline_outputs[j];
          const output_config = render_graph.get_image_config(output);
          const stride = Texture.stride_from_format(output_config.format);
          task.intermediate_buffers[j] = render_graph.create_buffer({
            name: `${output_config.name}_intermediate`,
            size: canvas_width * canvas_height * stride,
            raw_data: new Uint8Array(canvas_width * canvas_height * stride),
            usage: GPUBufferUsage.Storage | GPUBufferUsage.CopySrc | GPUBufferUsage.CopyDst,
          });
        }

        // Add rasterization pass
        render_graph.add_pass(
          task.name,
          RenderPassFlags.Compute,
          {
            shader_setup: { pipeline_shaders: { compute: { path: task.shader } } },
            inputs: [task.points, task.connections, ...task.inputs], // Include points and connections as input
            outputs: task.intermediate_buffers, // Output to intermediate buffers
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(task.dispatch_x, task.dispatch_y, task.dispatch_z);
          }
        );

        // Add buffer-to-texture copy pass based on pipeline outputs
        for (let j = 0; j < pipeline_outputs.length; j++) {
          const output = pipeline_outputs[j];
          const copy_pass_name = `${task.name}_copy_${j}`;
          const buffer = task.intermediate_buffers[j];
          const texture = output;

          render_graph.add_pass(
            copy_pass_name,
            RenderPassFlags.GraphLocal,
            {},
            (graph, frame_data, encoder) => {
              const texture_object = graph.get_physical_image(texture);
              const buffer_object = graph.get_physical_buffer(buffer);
              texture_object.copy_buffer(encoder, buffer_object);
            }
          );
        }
      }
    });
  }

  static reset() {
    this.tasks_allocator.reset();
    this.tasks.length = 0;
  }
}
