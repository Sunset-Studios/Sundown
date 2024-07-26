import { GraphicsContext } from "./graphics_context.js";
import { RenderGraph } from "./render_graph.js";
import { DeferredShadingStrategy } from "./strategies/deferred_shading.js";
import { Texture, TextureSampler } from "./texture.js";
import { SharedVertexBuffer, SharedViewBuffer } from "../core/shared_data.js";
import { MaterialTemplate } from "./material.js";

export class Renderer {
  graphics_context = null;
  render_strategy = null;
  simple_shader = null;
  simple_vertex_buffer = null;
  render_graph = null;

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

  async setup(canvas) {
    this.graphics_context = await GraphicsContext.create(canvas, { pointer_lock: true });

    this.render_graph = RenderGraph.create();

    this.render_strategy = new DeferredShadingStrategy();

    this.refresh_global_shader_bindings();
    this.setup_builtin_material_template();
  }

  render(delta_time) {
    performance.mark("frame_render");

    this.graphics_context.advance_frame();

    this.render_graph.begin(this.graphics_context);

    this.render_strategy.draw(this.graphics_context, this.render_graph);
  }

  refresh_global_shader_bindings() {
    this.render_graph.queue_global_bind_group_write([
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
    ], true /* overwrite */);
  }

  setup_builtin_material_template() {
    // Create a material template for a standard material
    const template = MaterialTemplate.create(
      Renderer.get().graphics_context,
      "StandardMaterial",
      "standard_material.wgsl"
    );
  }
}
