import { Buffer } from "../buffer.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";

// TODO: Make this a base class for anything that needs to raytrace.
// Dispatch function should be overridable. Raytracer base should
// Pull global triangle buffer / TLAS / BLAS / etc. on its own so it's included by default.
// Raytracing backend should split into separate dispatches: one for TLAS, which populates per ray TLAS hits,
// and one for BLAS, which does the actual ray tracing on the BLAS of a mesh.

/**
 * Simple compute shader based raytracer.
 * Expects a flattened BVH and triangle buffer.
 */
export class RayTracer {
  constructor() {
    this.bvh_buffer = null;
    this.triangle_buffer = null;
    this.output_texture = null;
    this.width = 0;
    this.height = 0;
  }

  setup(render_graph, bvh_data, triangle_data, width, height) {
    this.width = width;
    this.height = height;

    this.bvh_buffer = render_graph.create_buffer({
      name: "rt_bvh",
      raw_data: new Float32Array(bvh_data),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.triangle_buffer = render_graph.create_buffer({
      name: "rt_triangles",
      raw_data: new Float32Array(triangle_data),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.output_texture = render_graph.create_image({
      name: "rt_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: true,
    });
  }
}