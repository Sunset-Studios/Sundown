import { RenderGraph } from "./render_graph.js";
import { Texture, TextureSampler } from "./texture.js";
import {
  SharedVertexBuffer,
  SharedViewBuffer,
  SharedFrameInfoBuffer,
  SharedEntityMetadataBuffer,
} from "../core/shared_data.js";
import { MaterialTemplate } from "./material.js";
import { global_dispatcher } from "../core/dispatcher.js";
import { vec2 } from "gl-matrix";
import { profile_scope } from "../utility/performance.js";
import ExecutionQueue from "../utility/execution_queue.js";

const frame_render_event_name = "frame_render";
const MAX_BUFFERED_FRAMES = 2;

export class Renderer {
  canvas = null;
  adapter = null;
  device = null;
  context = null;
  canvas_format = null;
  frame_number = 0;
  aspect_ratio = 1.0;
  has_f16 = false;
  execution_queue = new ExecutionQueue();
  render_strategy = null;
  render_graph = null;
  post_render_callbacks = [];
  pre_render_callbacks = [];

  static renderers = [];

  async setup(canvas, canvas_ui, render_strategy, options = {}) {
    if (!navigator.gpu) {
      throw Error("WebGPU is not supported");
    }

    this.canvas = canvas;
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;

    this.canvas_ui = canvas_ui;
    this.canvas_ui.width = this.canvas_ui.clientWidth;
    this.canvas_ui.height = this.canvas_ui.clientHeight;

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    if (!this.adapter) {
      throw Error("Unable to request WebGPU adapter");
    }

    this.has_f16 = this.adapter.features.has("shader-f16") && !options.use_precision_float;

    let required_features = ["indirect-first-instance"];
    if (this.has_f16) {
      required_features.push("shader-f16");
    }

    try {
      this.device = await this.adapter.requestDevice({
        requiredFeatures: required_features,
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

    if (this.canvas_ui) {
      this.context_ui = this.canvas_ui.getContext("2d", {
        alpha: true,
      });
    }

    this.aspect_ratio = this.canvas.width / this.canvas.height;

    if (options.pointer_lock) {
      this.canvas.addEventListener("click", async () => {
        await this.canvas.requestPointerLock();
      });
    }

    this.render_graph = RenderGraph.create(this.max_bind_groups());

    this.render_strategy = new render_strategy();

    this.refresh_global_shader_bindings();

    this._setup_builtin_material_template();

    this._setup_resize_observer();
  }

  render(delta_time) {
    profile_scope(frame_render_event_name, () => {
      this.advance_frame();

      this.render_graph.begin();

      this.render_strategy.draw(this.render_graph);
    });
  }

  on_pre_render(callback) {
    this.render_graph.on_pre_render(callback);
  }

  enqueue_pre_commands(name, commands_callback) {
    this.render_graph.queue_pre_commands(name, commands_callback);
  }

  enqueue_post_commands(name, commands_callback) {
    this.render_graph.queue_post_commands(name, commands_callback);
  }

  on_post_render(callback) {
    this.render_graph.on_post_render(callback);
  }

  remove_pre_render(callback) {
    this.render_graph.remove_pre_render(callback);
  }

  remove_post_render(callback) {
    this.render_graph.remove_post_render(callback);
  }

  mark_bind_groups_dirty(passes_only = false) {
    this.render_graph.mark_pass_cache_bind_groups_dirty(passes_only);
  }

  refresh_global_shader_bindings() {
    this.render_graph.queue_global_bind_group_write(
      [
        {
          buffer: SharedVertexBuffer.buffer,
          offset: 0,
          size: SharedVertexBuffer.size,
        },
        {
          buffer: SharedViewBuffer.buffer,
          offset: 0,
          size: SharedViewBuffer.buffer_size,
        },
        {
          sampler: Texture.get_default_sampler(),
        },
        {
          sampler: TextureSampler.create({
            name: "non_filtering_sampler",
            mag_filter: "nearest",
            min_filter: "nearest",
            mipmap_filter: "nearest",
          }),
        },
        {
          buffer: SharedFrameInfoBuffer.buffer,
          offset: 0,
          size: SharedFrameInfoBuffer.size,
        },
        {
          buffer: SharedEntityMetadataBuffer.buffer,
          offset: 0,
          size: SharedEntityMetadataBuffer.buffer_size,
        },
      ],
      true /* overwrite */
    );
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

  set_scene_id(scene_id) {
    this.render_graph.set_scene_id(scene_id);
  }

  max_bind_groups() {
    return this.adapter.limits.maxBindGroups;
  }

  on_resize() {
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;
    this.aspect_ratio = this.canvas.width / this.canvas.height;
  }

  _setup_builtin_material_template() {
    // Create a material template for a standard material
    const template = MaterialTemplate.create(
      "StandardMaterial",
      "standard_material.wgsl"
    );
  }

  _setup_resize_observer() {
    const observer = new ResizeObserver((entries) => {
      this.on_resize();
      this._set_shared_frame_resolution();
      global_dispatcher.dispatch("resolution_change", entries[0].contentRect);
    });
    observer.observe(this.canvas);

    this._set_shared_frame_resolution();
  }

  _set_shared_frame_resolution() {
    SharedFrameInfoBuffer.set_resolution(
      vec2.fromValues(this.canvas.width, this.canvas.height)
    );
  }

  static get(index = 0) {
    return this.renderers[index];
  }

  static async create(canvas, canvas_ui, render_strategy, options = {}) {
    const renderer = new Renderer();
    this.renderers.push(renderer);
    await renderer.setup(canvas, canvas_ui, render_strategy, options);
    return this.renderers.length - 1;
  }
}
