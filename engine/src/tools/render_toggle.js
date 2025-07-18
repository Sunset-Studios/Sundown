import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";

/**
 * DebugDrawPicker allows the user to pick a debug draw type from the command line.
 */
export class RenderToggle extends DevConsoleTool {
  execute(args) {
    const feature = args[0];
    if (feature) {
      switch (feature) {
        case "gtao":
          const is_gtao_enabled = Renderer.get().is_gtao_enabled();
          Renderer.get().set_gtao_enabled(!is_gtao_enabled);
          break;
        case "shadow":
          const is_shadows_enabled = Renderer.get().is_shadows_enabled();
          Renderer.get().set_shadows_enabled(!is_shadows_enabled);
          break;
        case "gi":
          const is_gi_enabled = Renderer.get().is_gi_enabled();
          Renderer.get().set_gi_enabled(!is_gi_enabled);
          break;
        default:
          break;
      }
    }
  }
}
