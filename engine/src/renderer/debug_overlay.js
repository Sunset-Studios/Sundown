import { MeshTaskQueue } from "./mesh_task_queue.js";
import { RenderPassFlags } from "./renderer_types.js";
import { src_alpha_one_minus_src_alpha_blend_config } from "../utility/config_permutations.js";

// The DebugOverlay class allows you to configure an overlay rectangle
// (with x, y, width, height specified in pixels) and a texture (any debug output)
// to be drawn over the final output.
export class DebugOverlay {
  debug_texture = null; // the texture to be overlaid
  x = 0; // horizontal offset in pixels
  y = 0; // vertical offset in pixels
  width = 0; // overlay width in pixels
  height = 0; // overlay height in pixels
  enabled = false; // flag to enable/disable the overlay pass

  // Allows updating the overlay texture and its rectangle
  set_properties(debug_texture, x, y, width, height) {
    this.debug_texture = debug_texture;
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.enabled = true;
  }

  // Adds a debug overlay pass to the render graph.
  // The pass takes the overlay texture (from this.debug_texture) and composites it
  // over the provided base_output_image by drawing a quad within the specified viewport.
  add_pass(render_graph, base_output_image) {
    if (!this.enabled) {
      return;
    }

    // Define the shader setup for the overlay pass by selecting the appropriate shader based on the texture type
    let shader_path = "";
    if (this.debug_texture && this.debug_texture.config) {
      const texture_config = this.debug_texture.config;
      if (texture_config.format && texture_config.format.toLowerCase().includes("depth")) {
        shader_path = "debug_overlay_depth.wgsl";
      } else {
        const texture_dimension = texture_config.dimension || "2d";
        if (texture_dimension === "cube") {
          shader_path = "debug_overlay_cube.wgsl";
        } else if (texture_dimension === "3d") {
          shader_path = "debug_overlay_3d.wgsl";
        } else if (texture_dimension === "2d-array") {
          shader_path = "debug_overlay_2d_array.wgsl";
        } else {
          shader_path = "debug_overlay_2d.wgsl";
        }
      }
    } else {
      shader_path = "debug_overlay_2d.wgsl";
    }

    const overlay_shader_setup = {
      pipeline_shaders: {
        vertex: { path: shader_path },
        fragment: { path: shader_path },
      },
      rasterizer_state: {
        cull_mode: "none",
      },
      attachment_blend: src_alpha_one_minus_src_alpha_blend_config,
    };

    render_graph.add_pass(
      "debug_overlay_pass",
      RenderPassFlags.Graphics,
      {
        inputs: [this.debug_texture],
        outputs: [base_output_image],
        shader_setup: overlay_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.set_viewport(this.x, this.y, this.width, this.height, 0, 1);
        MeshTaskQueue.draw_quad(pass);
      }
    );
  }
}
