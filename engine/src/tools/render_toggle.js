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
        case "ao":
          const is_ao_enabled = Renderer.get().is_ao_enabled();
          Renderer.get().set_ao_enabled(!is_ao_enabled);
          break;
        case "shadow":
          const is_shadows_enabled = Renderer.get().is_shadows_enabled();
          Renderer.get().set_shadows_enabled(!is_shadows_enabled);
          break;
        case "gi":
          const is_gi_enabled = Renderer.get().is_gi_enabled();
          Renderer.get().set_gi_enabled(!is_gi_enabled);
          break;
        case "reflection":
          const is_reflection_enabled = Renderer.get().is_reflection_enabled();
          Renderer.get().set_reflection_enabled(!is_reflection_enabled);
          break;
        default:
          break;
      }
    }
  }
}
