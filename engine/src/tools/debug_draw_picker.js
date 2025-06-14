import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { DebugDrawType } from "../renderer/renderer_types.js";

/**
 * DebugDrawPicker allows the user to pick a debug draw type from the command line.
 */
export class DebugDrawPicker extends DevConsoleTool {
  execute(args) {
    const debug_view = args[0];
    if (debug_view) {
      switch (debug_view) {
        case "wireframe":
          Renderer.get().set_debug_draw_type(DebugDrawType.Wireframe);
          break;
        case "depth":
          Renderer.get().set_debug_draw_type(DebugDrawType.Depth);
          break;
        case "normal":
          Renderer.get().set_debug_draw_type(DebugDrawType.Normal);
          break;
        case "emissive":
          Renderer.get().set_debug_draw_type(DebugDrawType.Emissive);
          break;
        case "entity":
          Renderer.get().set_debug_draw_type(DebugDrawType.EntityId);
          break;
        case "hzb":
          Renderer.get().set_debug_draw_type(DebugDrawType.HZB);
          break;
        case "gi-probe-volume":
          Renderer.get().set_debug_draw_type(DebugDrawType.GIProbeVolume);
          break;
        case "shadow-atlas":
          Renderer.get().set_debug_draw_type(DebugDrawType.ASVSM_ShadowAtlas);
          break;
        case "shadow-page-table":
          Renderer.get().set_debug_draw_type(DebugDrawType.ASVSM_ShadowPageTable);
          break;
        case "shadow-virtual-tiles":
          Renderer.get().set_debug_draw_type(DebugDrawType.ASVSM_TileOverlay);
          break;
        case "bloom":
          Renderer.get().set_debug_draw_type(DebugDrawType.Bloom);
          break;
        default:
          Renderer.get().set_debug_draw_type(DebugDrawType.None);
          break;
      }
    }
  }
}
