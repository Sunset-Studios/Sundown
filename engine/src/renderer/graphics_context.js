import ExecutionQueue from "../utility/execution_queue.js";
import { global_dispatcher } from "../core/dispatcher.js";

const MAX_BUFFERED_FRAMES = 2;

export class GraphicsContext {
  canvas = null;
  adapter = null;
  device = null;
  context = null;
  canvas_format = null;
  frame_number = 0;
  aspect_ratio = 1.0;
  execution_queue = new ExecutionQueue();

  async init(canvas, options = {}) {
    if (!navigator.gpu) {
      throw Error("WebGPU is not supported");
    }

    this.canvas = canvas;
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!this.adapter) {
      throw Error("Unable to request WebGPU adapter");
    }

    try {
      this.device = await this.adapter.requestDevice({
        requiredFeatures: ["indirect-first-instance"],
        requiredLimits: {
          maxColorAttachmentBytesPerSample: 64,
          maxStorageBuffersPerShaderStage: 10,
          maxStorageBufferBindingSize: 256 * 1024 * 1024
        },
      });
    } catch (e) {
      console.log(e);
      console.log("Falling back to default limits");
      this.device = await this.adapter.requestDevice();
    }

    this.context = this.canvas.getContext("webgpu");
    this.canvas_format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.canvas_format,
      alphaMode: "premultiplied",
    });

    this.aspect_ratio = this.canvas.width / this.canvas.height;

    if (options.pointer_lock) {
      this.canvas.addEventListener("click", async () => {
        await this.canvas.requestPointerLock();
      });
    }
  }

  advance_frame() {
    this.frame_number++;
  }

  get_frame_number() {
    return this.frame_number;
  }

  get_buffered_frame_number() {
    return this.frame_number % MAX_BUFFERED_FRAMES;
  }

  get_canvas_resolution() {
    return {
      width: this.canvas.width,
      height: this.canvas.height,
    };
  }

  draw_pass(render_pass, triangles, instance_count = 1) {
    render_pass.pass.draw(triangles, instance_count);
  }

  max_bind_groups() {
    return this.adapter.limits.maxBindGroups;
  }

  on_resize() {
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;
    this.aspect_ratio = this.canvas.width / this.canvas.height;
  }

  static async create(canvas, options = {}) {
    let context = new GraphicsContext();
    await context.init(canvas, options);
    return context;
  }
}
