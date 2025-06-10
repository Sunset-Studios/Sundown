import { Renderer } from "./renderer.js";
import { MeshTaskQueue } from "./mesh_task_queue.js";
import { DebugDrawType, RenderPassFlags } from "./renderer_types.js";

const overlay_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "" },
  },
  rasterizer_state: {
    cull_mode: "none",
  },
};

// The DebugOverlay class allows you to configure an overlay rectangle
// (with x, y, width, height specified in pixels) and a texture (any debug output)
// to be drawn over the final output.
export class DebugOverlay {
  debug_texture = null; // the texture to be overlaid
  x = 0; // horizontal offset in pixels
  y = 0; // vertical offset in pixels
  width = 0; // overlay width in pixels
  height = 0; // overlay height in pixels
  texture_level = 0; // the mip level of the texture to be overlaid
  enabled = false; // flag to enable/disable the overlay pass
  debug_type = DebugDrawType.None; // the type of debug to draw
  viewport = null; // the viewport to be used for the overlay

  // Allows updating the overlay texture and its rectangle
  set_properties(
    debug_texture,
    x,
    y,
    width,
    height,
    debug_type = DebugDrawType.None,
    texture_level = 0
  ) {
    this.viewport = {
      x: x,
      y: y,
      width: width,
      height: height,
      min_depth: 0,
      max_depth: 1,
    };
    this.debug_texture = debug_texture;
    this.texture_level = texture_level;
    if (this.debug_type !== debug_type) {
      this.enabled = debug_type !== DebugDrawType.None;
      this.debug_type = debug_type;
      Renderer.get().mark_bind_groups_dirty(true);
    }
  }

  // Adds a debug overlay pass to the render graph.
  // The pass takes the overlay texture (from this.debug_texture) and composites it
  // over the provided base_output_image by drawing a quad within the specified viewport.
  add_pass(render_graph, base_output_image) {
    if (!this.enabled || !this.debug_texture) {
      return;
    }

    const shader_path = this._resolve_debug_shader();
    if (!shader_path) {
      return;
    }

    render_graph.add_pass(
      `debug_overlay_setup_${this.debug_type}`,
      RenderPassFlags.GraphLocal,
      { },
      (graph, frame_data, encoder) => {
        const base_output_image_obj = graph.get_physical_image(base_output_image);
        base_output_image_obj.config.load_op = "load";
      }
    );

    overlay_shader_setup.pipeline_shaders.fragment.path = shader_path;
    render_graph.add_pass(
      `debug_overlay_pass_${this.debug_type}`,
      RenderPassFlags.Graphics,
      {
        inputs: [this.debug_texture],
        outputs: [base_output_image],
        input_views: [this.texture_level],
        shader_setup: overlay_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.set_viewport(this.viewport);
        MeshTaskQueue.draw_quad(pass);
      }
    );

    render_graph.add_pass(
      `debug_overlay_cleanup_${this.debug_type}`,
      RenderPassFlags.GraphLocal,
      { },
      (graph, frame_data, encoder) => {
        const base_output_image_obj = graph.get_physical_image(base_output_image);
        base_output_image_obj.config.load_op = "clear";
      }
    );
  }

  _resolve_debug_shader() {
    switch (this.debug_type) {
      case DebugDrawType.Wireframe:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.Depth:
        return "debug/debug_overlay_depth.wgsl";
      case DebugDrawType.Normal:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.EntityId:
        return "debug/debug_overlay_entity.wgsl";
      case DebugDrawType.HZB:
        return "debug/debug_overlay_hzb.wgsl";
      case DebugDrawType.GIProbeVolume:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.Bloom:
        return "debug/debug_overlay_2d.wgsl";
      default:
        return "debug/debug_overlay_2d.wgsl";
    }
  }
}
