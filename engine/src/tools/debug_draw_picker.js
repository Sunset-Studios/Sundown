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
        case "motion":
          Renderer.get().set_debug_draw_type(DebugDrawType.Motion);
          break;
        case "entity":
          Renderer.get().set_debug_draw_type(DebugDrawType.EntityId);
          break;
        case "hzb":
          Renderer.get().set_debug_draw_type(DebugDrawType.HZB);
          if (args[1] !== undefined) {
            const level = parseInt(args[1], 10);
            if (!Number.isNaN(level)) {
              Renderer.get().set_debug_texture_level(level);
            }
          }
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
        case "shadow-tile-render":
          Renderer.get().set_debug_draw_type(DebugDrawType.ASVSM_TileRenderOutput);
          break;
        case "shadow-dirty-tiles":
          Renderer.get().set_debug_draw_type(DebugDrawType.ASVSM_DirtyTiles);
          break;
        case "bloom":
          Renderer.get().set_debug_draw_type(DebugDrawType.Bloom);
          break;
        case "ao":
          Renderer.get().set_debug_draw_type(DebugDrawType.AO);
          break;
        case "bent-normal":
          Renderer.get().set_debug_draw_type(DebugDrawType.BentNormal);
          break;
        case "gi-direct":
          Renderer.get().set_debug_draw_type(DebugDrawType.GI_Direct);
          break;
        case "gi-specular":
          Renderer.get().set_debug_draw_type(DebugDrawType.GI_Specular);
          break;
        case "gi-diffuse":
          Renderer.get().set_debug_draw_type(DebugDrawType.GI_Diffuse);
          break;
        case "gi-world-cache":
        case "world-cache":
          Renderer.get().set_debug_draw_type(DebugDrawType.GI_WorldCache);
          break;
        case "gi-probes":
          Renderer.get().set_debug_draw_type(DebugDrawType.GI_Probes);
          break;
        case "gi-reflections":
          Renderer.get().set_debug_draw_type(DebugDrawType.GI_Reflections);
          break;
        case "prev-lighting":
          Renderer.get().set_debug_draw_type(DebugDrawType.PrevLightingPyramid);
          if (args[1] !== undefined) {
            const level = parseInt(args[1], 10);
            if (!Number.isNaN(level)) {
              Renderer.get().set_debug_texture_level(level);
            }
          }
          break;
        case "bounds":
          Renderer.get().set_debug_draw_type(DebugDrawType.EntityBounds);
          break;
        case "bvh":
          Renderer.get().set_debug_draw_type(DebugDrawType.BVH);
          break;
        case "blas-bounds":
          Renderer.get().set_debug_draw_type(DebugDrawType.BLAS_Bounds);
          break;
        default:
          Renderer.get().set_debug_draw_type(DebugDrawType.None);
          break;
      }
    }
  }
}
