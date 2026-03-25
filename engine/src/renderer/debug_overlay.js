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
  debug_textures = null; // array of textures
  x = 0; // horizontal offset in pixels
  y = 0; // vertical offset in pixels
  width = 0; // overlay width in pixels
  height = 0; // overlay height in pixels
  texture_level = 0; // the mip level of the texture to be overlaid
  enabled = false; // flag to enable/disable the overlay pass
  debug_type = DebugDrawType.None; // the type of debug to draw
  viewport = null; // the viewport to be used for the overlay
  channel_mask = [1.0, 1.0, 1.0, 1.0]; // RGBA channel mask
  visualize_mode = 0; // 0 = RGB, 1 = single channel grayscale
  channel_config_buffer = null; // uniform buffer for channel configuration

  // Allows updating the overlay texture and its rectangle
  set_properties(
    debug_textures,
    x,
    y,
    width,
    height,
    debug_type = DebugDrawType.None,
    texture_level = 0,
    channel_mask = [1.0, 1.0, 1.0, 1.0],
    visualize_mode = 0
  ) {
    this.viewport = {
      x: x,
      y: y,
      width: width,
      height: height,
      min_depth: 0,
      max_depth: 1,
    };
    this.debug_textures = debug_textures;
    this.texture_level = texture_level;
    this.channel_mask = channel_mask;
    this.visualize_mode = visualize_mode;
    if (this.debug_type !== debug_type) {
      this.enabled = debug_type !== DebugDrawType.None;
      this.debug_type = debug_type;
      Renderer.get().mark_bind_groups_dirty(true);
    }
    this.channel_config = this.visualize_mode === 1
      ? new Uint32Array(this.channel_mask)
      : null;
  }

  // Adds a debug overlay pass to the render graph.
  // The pass takes the overlay texture (from this.debug_texture) and composites it
  // over the provided base_output_image by drawing a quad within the specified viewport.
  add_pass(render_graph, base_output_image) {
    if (!this.enabled || !this.debug_textures) {
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

    // Create channel config buffer if using channel shader
    let channel_config_buf = null;
    if (this.channel_config) {
      channel_config_buf = render_graph.create_buffer({
        name: `debug_channel_config_${this.debug_type}`,
        raw_data: this.channel_config,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }

    // Build inputs array based on single texture or multiple textures
    let inputs;
    if (Array.isArray(this.debug_textures)) {
      inputs = [...this.debug_textures];
    } else { 
      inputs = [this.debug_textures];
    }

    if (channel_config_buf) {
      inputs.push(channel_config_buf);
    }

    overlay_shader_setup.pipeline_shaders.fragment.path = shader_path;
    render_graph.add_pass(
      `debug_overlay_pass_${this.debug_type}`,
      RenderPassFlags.Graphics,
      {
        inputs: inputs,
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
      case DebugDrawType.Emissive:
        return "debug/debug_overlay_channel.wgsl";
      case DebugDrawType.Motion:
        return "debug/debug_overlay_motion_lines.wgsl";
      case DebugDrawType.EntityId:
        return "debug/debug_overlay_entity.wgsl";
      case DebugDrawType.VisibilityMaterialId:
        return "debug/debug_overlay_visibility_material.wgsl";
      case DebugDrawType.VisibilityEntityId:
        return "debug/debug_overlay_entity.wgsl";
      case DebugDrawType.VisibilityMeshletId:
        return "debug/debug_overlay_visibility_meshlet.wgsl";
      case DebugDrawType.VisibilityTriangleId:
        return "debug/debug_overlay_visibility_triangle.wgsl";
      case DebugDrawType.HZB:
        return "debug/debug_overlay_hzb.wgsl";
      case DebugDrawType.Bloom:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.ASVSM_TileOverlay:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.ASVSM_TileRenderOutput:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.ASVSM_DirtyTiles:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.AO:
        return "debug/debug_overlay_2d_single_comp.wgsl";
      case DebugDrawType.BentNormal:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.GI_Direct:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.GI_Specular:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.GI_Diffuse:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.GI_Probes:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.GI_Reflections:
        return "debug/debug_overlay_2d.wgsl";
      case DebugDrawType.PrevLightingPyramid:
        return "debug/debug_overlay_2d.wgsl";
      default:
        return "debug/debug_overlay_2d.wgsl";
    }
  }
}
