import { MAX_BUFFERED_FRAMES } from "../core/minimal.js";
import { DebugDrawType } from "./renderer_types.js";
import { RenderGraph } from "./render_graph.js";
import { Texture, TextureSampler } from "./texture.js";
import { Mesh } from "./mesh.js";
import {
  SharedVertexBuffer,
  SharedViewBuffer,
  SharedFrameInfoBuffer,
} from "../core/shared_data.js";
import { global_dispatcher } from "../core/dispatcher.js";
import { profile_scope } from "../utility/performance.js";
import ExecutionQueue from "../utility/execution_queue.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { log, error } from "../utility/logging.js";
import { vec2 } from "gl-matrix";

const frame_render_event_name = "frame_render";

export class Renderer {
  canvas = null;
  adapter = null;
  device = null;
  context = null;
  canvas_format = null;
  frame_number = 0;
  aspect_ratio = 1.0;
  execution_queue = new ExecutionQueue();
  render_strategy = null;
  render_graph = null;
  post_render_callbacks = [];
  pre_render_callbacks = [];
  
  // Renderer features
  has_f16 = false;
  use_depth_prepass = true;
  shadows_enabled = false;
  gi_enabled = false;
  debug_draw_type = DebugDrawType.None;

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
          maxStorageBufferBindingSize: 256 * 1024 * 1024,
          maxTextureArrayLayers: 2048,
        },
      });
    } catch (e) {
      log(e);
      log("Falling back to default limits");
      this.device = await this.adapter.requestDevice();
    }

    // Use lost to handle lost devices
    this.device.lost.then((info) => {
      error(`WebGPU device was lost: ${info.message}`);
    });

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

    this._setup_resize_observer();

    Mesh.precrete_engine_primitives();
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

  enqueue_pre_commands(name, commands_callback, persistent = false) {
    this.render_graph.queue_pre_commands(name, commands_callback, persistent);
  }

  unqueue_pre_commands(name) {
    this.render_graph.unqueue_pre_commands(name);
  }

  enqueue_post_commands(name, commands_callback, persistent = false) {
    this.render_graph.queue_post_commands(name, commands_callback, persistent);
  }

  unqueue_post_commands(name) {
    this.render_graph.unqueue_post_commands(name);
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
    const global_bindings = [
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
            address_mode_u: "clamp-to-edge",
            address_mode_v: "clamp-to-edge",
            address_mode_w: "clamp-to-edge",
            mag_filter: "nearest",
            min_filter: "nearest",
            mipmap_filter: "nearest",
            type: "non-filtering",
          }),
        },
        {
          sampler: TextureSampler.create({
            name: "clamped_sampler",
            address_mode_u: "clamp-to-edge",
            address_mode_v: "clamp-to-edge",
            address_mode_w: "clamp-to-edge",
            mag_filter: "linear",
            min_filter: "linear",
            mipmap_filter: "linear",
          }),
        },
        {
          sampler: TextureSampler.create({
            name: "comparison_sampler",
            mag_filter: "nearest",
            min_filter: "nearest",
            mipmap_filter: "nearest",
            compare: "less-equal",
            type: "comparison",
          }),
        },
        {
          buffer: SharedFrameInfoBuffer.buffer,
          offset: 0,
          size: SharedFrameInfoBuffer.size,
        },
      ];

    if (FragmentGpuBuffer.entity_index_map_buffer) {
      global_bindings.push({
        buffer: FragmentGpuBuffer.entity_index_map_buffer.buffer,
        offset: 0,
        size: FragmentGpuBuffer.entity_index_map_buffer.buffer.config.size,
      });
    }

    this.render_graph.queue_global_bind_group_write(
      global_bindings,
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

  is_shadows_enabled() {
    return this.shadows_enabled;
  }

  set_shadows_enabled(enabled) {
    this.shadows_enabled = enabled;
  }

  is_gi_enabled() {
    return this.gi_enabled;
  }

  set_gi_enabled(enabled) {
    this.gi_enabled = enabled;
  }

  is_depth_prepass_enabled() {
    return this.use_depth_prepass;
  }

  set_depth_prepass_enabled(enabled) {
    this.use_depth_prepass = enabled;
  }

  draw_pass(render_pass, triangles, instance_count = 1) {
    render_pass.pass.draw(triangles, instance_count);
  }

  set_scene_id(scene_id) {
    this.render_graph.set_scene_id(scene_id);
  }

  get_debug_draw_type() {
    return this.debug_draw_type;
  }
  
  set_debug_draw_type(debug_draw_type) {
    this.debug_draw_type = debug_draw_type;
  }

  max_bind_groups() {
    return this.adapter.limits.maxBindGroups;
  }

  on_resize() {
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;
    this.canvas_ui.width = this.canvas_ui.clientWidth;
    this.canvas_ui.height = this.canvas_ui.clientHeight;
    this.aspect_ratio = this.canvas.width / this.canvas.height;
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
    SharedFrameInfoBuffer.set_resolution(vec2.fromValues(this.canvas.width, this.canvas.height));
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
