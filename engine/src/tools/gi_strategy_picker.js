import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { GIStrategyType } from "../renderer/renderer_types.js";

const ddgi = "ddgi";
const ptgi = "ptgi";

const help_text = "Available strategies: ddgi, ptgi";

/**
 * GIStrategyPicker allows the user to switch between GI strategies from the command line.
 */
export class GIStrategyPicker extends DevConsoleTool {
  execute(args) {
    const strategy_name = args[0];
    if (strategy_name) {
      switch (strategy_name) {
        case ddgi:
          Renderer.get().set_gi_strategy_type(GIStrategyType.DDGI);
          break;
        case ptgi:
          Renderer.get().set_gi_strategy_type(GIStrategyType.PTGI);
          break;
        default:
          console.log(help_text);
          break;
      }
    }
  }
}

