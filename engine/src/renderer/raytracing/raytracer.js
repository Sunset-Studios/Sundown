import { Buffer } from "../buffer.js";
import { ComputeTaskQueue } from "../compute_task_queue.js";

const output_config = {
  name: "rt_output",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
}

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
    this.output_texture = null;
    this.width = 0;
    this.height = 0;
  }

  setup(render_graph, width, height, force_recreate = false) {
    this.width = width;
    this.height = height;

    output_config.width = width;
    output_config.height = height;
    output_config.force = force_recreate;

    this.output_texture = render_graph.create_image(output_config);
  }
}