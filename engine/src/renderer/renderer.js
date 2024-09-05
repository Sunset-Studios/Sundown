import { GraphicsContext } from "./graphics_context.js";
import { RenderGraph } from "./render_graph.js";
import { Texture, TextureSampler } from "./texture.js";
import {
  SharedVertexBuffer,
  SharedViewBuffer,
  SharedFrameInfoBuffer,
} from "../core/shared_data.js";
import { MaterialTemplate } from "./material.js";
import { global_dispatcher } from "../core/dispatcher.js";
import { vec2 } from "gl-matrix";

export class Renderer {
  graphics_context = null;
  render_strategy = null;
  simple_shader = null;
  simple_vertex_buffer = null;
  render_graph = null;
  post_render_callbacks = [];

  constructor() {
    if (Renderer.instance) {
      return Renderer.instance;
    }
    Renderer.instance = this;
  }

  static get() {
    if (!Renderer.instance) {
      return new Renderer();
    }
    return Renderer.instance;
  }

  async setup(canvas, render_strategy) {
    this.graphics_context = await GraphicsContext.create(canvas, {
      pointer_lock: true,
    });

    this.render_graph = RenderGraph.create();

    this.render_strategy = new render_strategy();

    this.refresh_global_shader_bindings();

    this._setup_builtin_material_template();

    this._setup_resize_observer();
  }

  render(delta_time) {
    performance.mark("frame_render");

    this.graphics_context.advance_frame();

    this.render_graph.begin(this.graphics_context);

    this.render_strategy.draw(this.graphics_context, this.render_graph);

    this._execute_post_render_callbacks();
  }

  enqueue_commands(name, commands_callback) {
    this.render_graph.queue_commands(name, commands_callback);
  }

  enqueue_post_commands(name, commands_callback) {
    this.render_graph.queue_post_commands(name, commands_callback);
  }

  on_post_render(callback) {
    this.post_render_callbacks.push(callback);
  }

  remove_post_render(callback) {
    const index = this.post_render_callbacks.indexOf(callback);
    if (index !== -1) {
      this.post_render_callbacks.splice(index, 1);
    }
  }

  mark_bind_groups_dirty(passes_only = false) {
    this.render_graph.mark_pass_cache_bind_groups_dirty(passes_only);
  }

  refresh_global_shader_bindings() {
    this.render_graph.queue_global_bind_group_write(
      [
        {
          buffer: SharedVertexBuffer.get().buffer,
          offset: 0,
          size: SharedVertexBuffer.get().size,
        },
        {
          buffer: SharedViewBuffer.get().buffer,
          offset: 0,
          size: SharedViewBuffer.get().size,
        },
        {
          sampler: Texture.get_default_sampler(this.graphics_context),
        },
        {
          sampler: TextureSampler.create(this.graphics_context, {
            name: "non_filtering_sampler",
            mag_filter: "nearest",
            min_filter: "nearest",
            mipmap_filter: "nearest",
          }),
        },
        {
          buffer: SharedFrameInfoBuffer.get().buffer,
          offset: 0,
          size: SharedFrameInfoBuffer.get().size,
        },
      ],
      true /* overwrite */
    );
  }

  _setup_builtin_material_template() {
    // Create a material template for a standard material
    const template = MaterialTemplate.create(
      Renderer.get().graphics_context,
      "StandardMaterial",
      "standard_material.wgsl"
    );
  }

  _setup_resize_observer() {
    const observer = new ResizeObserver((entries) => {
      this.graphics_context.on_resize();
      this._set_shared_frame_resolution();
      global_dispatcher.dispatch("resolution_change", entries[0].contentRect);
    });
    observer.observe(this.graphics_context.canvas);

    this._set_shared_frame_resolution();
  }

  _execute_post_render_callbacks() {
    for (let i = 0; i < this.post_render_callbacks.length; i++) {
      this.post_render_callbacks[i]();
    }
  }

  _set_shared_frame_resolution() {
    SharedFrameInfoBuffer.get().set_resolution(
      this.graphics_context,
      vec2.fromValues(
        this.graphics_context.canvas.width,
        this.graphics_context.canvas.height
      )
    );
  }
}
